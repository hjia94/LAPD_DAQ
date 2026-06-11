"""bmotion motor-recovery hardware diagnostics.

Drive ``acquisition.motor_recovery`` directly against real motors to validate
the move-handling fixes. These do NOT run a full acquisition -- they build a
RunManager, command moves on one motion group, and inspect the result.

Three independent checks:
  - LONG MOTION: a slow full-range move must be left to finish (never timed out
    or interrupted mid-travel).
  - ENCODER: read encoder (EP) vs commanded/step (IP) position around a move and
    confirm they track each other. EP is read via the negative-safe raw escape
    hatch (motor_recovery._read_encoder_counts), the same path production uses, so
    a move crossing zero is handled instead of crashing on the library's EP regex.
  - FAILURE: command a known-unreachable index, confirm recovery raises
    MotorError, then confirm a subsequent good move still succeeds (guards the
    original "one failure poisons every later move" bug).
  - SET-ZERO (destructive, off by default): zero the group, then confirm the
    encoder reads back ~0 -- catches a set-zero that the drive ACK'd but did not
    actually apply.

Only ``bmotion_config.toml`` is needed in the current working directory.

Run with:

    pytest tests/test_bmotion_recovery_hw.py -v -s
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from _hardware_check_base import HardwareCheckBase


# ===========================================================================
#  EDIT THESE — everything you need to change for a hardware run is here.
# ===========================================================================

# Path to the motion-config TOML (absolute path recommended).
BMOTION_TOML_PATH = r"E:\Shadow data\Pat\bmotion_config.toml"

# Motion group to drive. Use the group's name (e.g. "Hermes") — recommended,
# it's unambiguous. You may also use its index (0, 1, 2, ...) in TOML order.
# None = use the first group in the TOML.
MOTION_GROUP = 2

# Safety gate: nothing moves a motor until this is True.
BMOTION_ALLOW_MOVE = True

# Turn each check on individually.
RUN_LONG_MOTION_CHECK = False     # slow full-range move must finish, not time out
RUN_ENCODER_CHECK = True         # encoder (EP) vs step (IP) agreement around a move
RUN_FAILURE_CHECK = False         # an unreachable target -> MotorError, then recovers
# DESTRUCTIVE: set-zero redefines the origin for this group. Off by default even
# with BMOTION_ALLOW_MOVE; only enable when you intend to re-zero this rig.
RUN_SET_ZERO_CHECK = False       # zero the group, confirm encoder reads back ~0

# For the FAILURE check only: a motion-list index for MOTION_GROUP that you KNOW
# the probe physically cannot reach on this rig (blocked / past an obstruction).
# Must be set when RUN_FAILURE_CHECK is True.
FAILURE_TARGET_INDEX = None       # e.g. 7

# Encoder agreement tolerance, in motor revolutions.
ENCODER_TOL_REV = 0.01

# ===========================================================================
#  Advanced timing knobs — defaults are fine for most rigs.
# ===========================================================================
STALL_TIMEOUT_S = 10.0    # no-progress window before a move counts as stalled
MAX_MOVE_TIME_S = 600.0   # absolute ceiling for a single move (10 min)
ATTEMPTS = 2              # move attempts before giving up

# ===========================================================================
#  Implementation — no need to edit below this line.
# ===========================================================================


def _have_bmotion_install() -> bool:
    try:
        import bapsf_motion  # noqa: F401
        import xarray  # noqa: F401
        return True
    except ImportError:
        return False


def _ml_index(spec, ml_size: int) -> int:
    """Map 'first'/'last'/int to a concrete motion-list index."""
    if spec == "first":
        return 0
    if spec == "last":
        return ml_size - 1
    idx = int(spec)
    if idx < 0:
        idx += ml_size
    if not 0 <= idx < ml_size:
        raise ValueError(f"motion-list index {spec} out of range for size {ml_size}")
    return idx


class _RecoveryHardwareBase(HardwareCheckBase):
    """Shared RunManager lifecycle + raw EP/IP readers for the recovery checks."""

    label = "bmotion_recovery"

    def gate_checks(self) -> list[tuple[bool, str]]:
        return [
            (not BMOTION_ALLOW_MOVE,
             "BMOTION_ALLOW_MOVE is False — refusing to command motors"),
            (not _have_bmotion_install(),
             "bapsf_motion / xarray not installed on this machine"),
            (not Path(BMOTION_TOML_PATH).is_file(),
             f"Missing {BMOTION_TOML_PATH} in the current working directory"),
        ]

    def setUp(self) -> None:
        super().setUp()
        import bapsf_motion as bmotion

        print("\n[recovery check] loading TOML / starting RunManager...")
        self.rm = bmotion.actors.RunManager(BMOTION_TOML_PATH, auto_run=True)
        self.assertTrue(self.rm.mgs, "no motion groups available in the TOML")

        self.mg_key = self._select_mg_key()
        self.order = {self.mg_key: "forward"}
        name = self.rm.mgs[self.mg_key].config.get("name", self.mg_key)
        print(f"[recovery check] driving motion group: '{self.mg_key}' (name='{name}')")

    def _select_mg_key(self):
        """Resolve MOTION_GROUP (group name or integer index) to an mg key.

        rm.mgs is keyed by integer index (0, 1, 2, ...) in TOML order. Accept a
        display name, an int, or a string like "0"."""
        if MOTION_GROUP is None:
            return next(iter(self.rm.mgs))
        # Match by configured display name (e.g. "Hermes").
        for key, mg in self.rm.mgs.items():
            if mg.config.get("name") == MOTION_GROUP:
                return key
        # Match by index, accepting int 0 or str "0".
        try:
            idx = int(MOTION_GROUP)
            if idx in self.rm.mgs:
                return idx
        except (TypeError, ValueError):
            pass
        self.skipTest(
            f"MOTION_GROUP={MOTION_GROUP!r} not found; "
            f"available indices={list(self.rm.mgs.keys())}, "
            f"names={[mg.config.get('name') for mg in self.rm.mgs.values()]}"
        )

    def tearDown(self) -> None:
        try:
            self.rm.terminate()
        except Exception as exc:  # noqa: BLE001
            print(f"[recovery check] RunManager.terminate failed: {exc}")
        super().tearDown()

    def _wait_until_idle(self, timeout_s=30.0, poll_s=0.2):
        """Block until the driven group is fully stopped.

        EP (encoder_position) is a *buffered* command: the drive NACKs it while
        the motor is moving, so the encoder read must only happen when idle. We
        force a fresh moving-status query (retrieve_motor_status is unbuffered /
        safe mid-move) so we don't trust a stale heartbeat cache, then poll."""
        mg = self.rm.mgs[self.mg_key]
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for ax in mg.drive.axes:
                try:
                    ax.motor.send_command("retrieve_motor_status")
                except Exception:  # noqa: BLE001 - best-effort refresh
                    pass
            if not self.rm.is_moving:
                return True
            time.sleep(poll_s)
        print(f"[recovery check] WARNING: {self.mg_key} still moving after "
              f"{timeout_s:.0f}s; encoder read may NACK")
        return False

    def _print_ep_ip(self, when):
        """Print encoder (EP) vs step (IP) position per axis, in revolutions.

        Must be called when the motor is idle (EP is NACK'd while moving); call
        _wait_until_idle() first. The encoder is read via the same negative-safe
        raw escape hatch production uses (motor_recovery._read_encoder_counts), so
        this exercises the identical path and works at negative positions (a move
        that crosses zero) instead of crashing on the library's buggy EP regex.
        On a read miss this reports *why* (still moving / no encoder, missing
        encoder_resolution, etc.) instead of a generic 'unavailable', so a
        hardware miss is diagnosable."""
        from acquisition.motor_recovery import (
            _read_encoder_counts, _encoder_resolution,
        )

        mg = self.rm.mgs[self.mg_key]
        for idx, ax in enumerate(mg.drive.axes):
            motor = ax.motor
            ip = motor.send_command("get_position")
            ep_counts = _read_encoder_counts(motor)  # negative-safe raw read

            # A NACK / lost-connection / malformed IP read comes back as the
            # motor's own ack_flags enum, not a number.
            ack_flags = getattr(motor, "ack_flags", None)

            def _is_ack(v):
                return ack_flags is not None and isinstance(v, ack_flags)

            # Identify the failure cause before attempting the math.
            spr_q = getattr(motor, "steps_per_rev", None)
            spr = getattr(spr_q, "value", spr_q)
            er = _encoder_resolution(motor)

            reason = None
            if _is_ack(ip):
                reason = (f"drive returned IP={ip.name} "
                          f"(NACK'd while moving — is it still moving?)")
            elif ep_counts is None:
                reason = ("no encoder reading returned (still moving, or drive "
                          "has no encoder)")
            elif spr in (None, 0):
                reason = "steps_per_rev (gearing) not available from the drive"
            elif er in (None, 0):
                reason = "encoder_resolution not available from the drive"

            if reason is not None:
                print(f"[recovery check] {self.mg_key} axis {idx} {when}: "
                      f"EP/IP unavailable — {reason}")
                continue

            ip_rev = float(getattr(ip, "value", ip)) / float(spr)
            ep_rev = float(ep_counts) / float(er)
            print(f"[recovery check] {self.mg_key} axis {idx} {when}: "
                  f"step(IP)={ip_rev:.4f} rev  encoder(EP)={ep_rev:.4f} rev  "
                  f"diff={abs(ip_rev - ep_rev):.4f} rev")

    def _move(self, index):
        from acquisition.motor_recovery import move_with_recovery
        move_with_recovery(
            self.rm, self.order, index,
            attempts=ATTEMPTS, stall_timeout=STALL_TIMEOUT_S,
            max_move_time=MAX_MOVE_TIME_S, encoder_mismatch_tol_rev=ENCODER_TOL_REV,
            log=print,
        )


# --------------------------------------------------------------------------- #
class BmotionLongMotionHardwareCheck(_RecoveryHardwareBase):
    """A long, slow move must be left to finish — never interrupted/timed out.

    Drives a full-range move (first -> last index). Because the move keeps making
    progress, it must finish on its own: no MotorError, the move issued exactly
    once (no soft-stop / re-issue), and the final position matches the target."""

    run_flag = RUN_LONG_MOTION_CHECK
    label = "bmotion_long_motion"

    def test_long_move_completes_without_interruption(self) -> None:
        import numpy as np
        from acquisition.motor_recovery import MotorError

        mg = self.rm.mgs[self.mg_key]
        ml_size = int(mg.mb.motion_list.shape[0])
        self.assertGreaterEqual(ml_size, 2,
                                "motion list needs >= 2 points for a long move")

        from_idx = _ml_index("first", ml_size)
        to_idx = _ml_index("last", ml_size)

        print(f"[recovery check] seeding start position (index {from_idx})")
        self._move(from_idx)

        # Count how many times the move is actually issued: a healthy progressing
        # move must be issued exactly once (no soft-stop + re-issue).
        issued = {"count": 0}
        _orig_move_ml = mg.move_ml

        def _counting_move_ml(index):
            issued["count"] += 1
            return _orig_move_ml(index)

        mg.move_ml = _counting_move_ml
        try:
            t0 = time.time()
            print(f"[recovery check] long move {from_idx} -> {to_idx} starting")
            self._move(to_idx)
            elapsed = time.time() - t0
        except MotorError as exc:
            self.fail(f"long move was wrongly treated as a failure: {exc}")
        finally:
            mg.move_ml = _orig_move_ml

        print(f"[recovery check] long move finished in {elapsed:.1f}s, "
              f"move_ml issued {issued['count']}x")
        self.assertEqual(issued["count"], 1,
                         "a progressing move should be issued exactly once "
                         "(it was soft-stopped + re-issued, i.e. interrupted)")
        target = np.asarray(mg.mb.motion_list.values)[to_idx]
        actual = np.asarray(mg.position.value)
        n = min(len(target), len(actual))
        self.assertTrue(
            np.all(np.abs(actual[:n] - target[:n]) <= 0.5),
            f"did not reach target {tuple(target)} (at {tuple(actual)})",
        )


# --------------------------------------------------------------------------- #
class BmotionEncoderHardwareCheck(_RecoveryHardwareBase):
    """Inspect encoder (EP) vs step (IP) behavior around a real move.

    Reports both before and after a move and asserts they agree within
    ``ENCODER_TOL_REV`` afterward (catches lost steps / encoder slip)."""

    run_flag = RUN_ENCODER_CHECK
    label = "bmotion_encoder"

    def test_encoder_tracks_step_position(self) -> None:
        from acquisition.motor_recovery import encoder_step_mismatch

        mg = self.rm.mgs[self.mg_key]
        ml_size = int(mg.mb.motion_list.shape[0])

        # Read EP only when stopped — EP is NACK'd while the motor is moving.
        self._wait_until_idle()
        self._print_ep_ip("before move")

        self._move(_ml_index("last", ml_size))

        # move_with_recovery returns after settle, but force-confirm idle before
        # the encoder read so a residual settle can't NACK the EP read.
        self._wait_until_idle()
        self._print_ep_ip("after move")

        bad = encoder_step_mismatch(mg, tol_rev=ENCODER_TOL_REV)
        if bad:
            detail = "; ".join(
                f"axis {i}: step={s:.4f} rev vs encoder={e:.4f} rev "
                f"(diff {abs(s - e):.4f} rev)"
                for i, s, e in bad
            )
            self.fail(f"encoder disagrees with motor/step position: {detail}")
        print(f"[recovery check] encoder agrees with step position within "
              f"{ENCODER_TOL_REV} rev")


# --------------------------------------------------------------------------- #
class BmotionSetZeroHardwareCheck(_RecoveryHardwareBase):
    """Set-zero must actually take: after zeroing, the encoder reads back ~0.

    bapsf_motion's set_position(0) writes EP0/SP0 and trusts the drive's ACK
    without reading the encoder back. This check zeroes the group and then
    positively confirms each axis's encoder reads ~0 via the negative-safe raw
    read (verify_encoder_zeroed) -- catching a silently-failed zero.

    DESTRUCTIVE: this redefines the origin for the driven group, so it is gated
    behind RUN_SET_ZERO_CHECK (off by default) in addition to BMOTION_ALLOW_MOVE.
    """

    run_flag = RUN_SET_ZERO_CHECK
    label = "bmotion_set_zero"

    def test_set_zero_is_confirmed_by_encoder(self) -> None:
        from acquisition.motor_recovery import verify_encoder_zeroed

        mg = self.rm.mgs[self.mg_key]

        # Zero only when stopped (EP/SP writes are buffered).
        self._wait_until_idle()
        print(f"[recovery check] zeroing motion group '{self.mg_key}' "
              f"(set_zero -> EP0/SP0)")
        mg.set_zero()

        # Confirm the write took, on a fresh idle read.
        self._wait_until_idle()
        self._print_ep_ip("after set_zero")

        bad = verify_encoder_zeroed(mg, tol_counts=2.0, log=print)
        if bad:
            detail = "; ".join(
                f"axis {i}: encoder={'unreadable' if c is None else c} counts"
                for i, c in bad
            )
            self.fail(f"set-zero not confirmed by encoder read-back: {detail}")
        print("[recovery check] set-zero confirmed: all axes read ~0 counts")


# --------------------------------------------------------------------------- #
class BmotionFailureHandlingHardwareCheck(_RecoveryHardwareBase):
    """A genuinely-unreachable target must raise MotorError, not hang or corrupt.

    Commands a move to FAILURE_TARGET_INDEX (a real index the probe cannot reach)
    and asserts recovery raises ``MotorError``. Then verifies the position is
    still readable and a subsequent good move succeeds — the regression guard for
    the original bug where a failed move poisoned every later move."""

    run_flag = RUN_FAILURE_CHECK
    label = "bmotion_failure"

    def gate_checks(self) -> list[tuple[bool, str]]:
        return super().gate_checks() + [
            (FAILURE_TARGET_INDEX is None,
             "FAILURE_TARGET_INDEX is None — set a motion-list index you know is "
             "unreachable to test failure handling"),
        ]

    def test_unreachable_index_raises_then_next_move_recovers(self) -> None:
        import numpy as np
        from acquisition.motor_recovery import MotorError

        mg = self.rm.mgs[self.mg_key]
        ml_size = int(mg.mb.motion_list.shape[0])
        bad_index = int(FAILURE_TARGET_INDEX)
        self.assertTrue(0 <= bad_index < ml_size,
                        f"FAILURE_TARGET_INDEX {bad_index} out of range [0,{ml_size})")
        base_dim = int(np.asarray(mg.mb.motion_list.values).shape[1])

        # 1) The unreachable index must exhaust recovery and raise MotorError.
        print(f"[recovery check] commanding UNREACHABLE index {bad_index} "
              f"(expecting MotorError after {ATTEMPTS} attempts)")
        with self.assertRaises(MotorError):
            self._move(bad_index)

        # 2) The system is still queryable (skip-and-continue can record the
        #    position and move on rather than aborting the run).
        pos = mg.position.value
        print(f"[recovery check] position still readable after failure: {tuple(pos)}")
        self.assertEqual(len(pos), base_dim)

        # 3) Critically: a subsequent move to a known-good index must succeed.
        good_index = _ml_index("first", ml_size)
        if good_index == bad_index:
            good_index = _ml_index("last", ml_size)
        print(f"[recovery check] verifying recovery: moving to good index {good_index}")
        try:
            self._move(good_index)
        except MotorError as exc:
            self.fail(f"a failed move poisoned the next move (the original bug): {exc}")
        print(f"[recovery check] recovered: reached good index {good_index} "
              f"after the failed move")


if __name__ == "__main__":
    unittest.main()
