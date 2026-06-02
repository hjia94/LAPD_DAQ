"""Tests for acquisition.motor_recovery.move_with_recovery and helpers.

Uses purpose-built fakes that mimic the bapsf_motion shapes the recovery code
reads (mg.mb.motion_list.values, mg.position, mg.stop, mg.drive.axes[i].motor
with a .status dict + alarm_reset/move_off_limit/send_command/retrieve_motor_status,
mg.drive.send_command, rm.is_moving). Each fake injects a specific failure mode
so we can assert the recovery ladder behaves:

  * a slow-but-progressing move is left to finish (never soft-stopped),
  * a genuinely stalled move escalates and (if it never recovers) raises,
  * a transient miss recovers on retry,
  * connection loss waits and recovers,
  * a resettable alarm recovers,
  * verification uses fresh (refreshed) position reads,
  * encoder-vs-step disagreement is detected by encoder_step_mismatch.

No bapsf_motion, hardware, or HDF5 needed.
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from acquisition import motor_recovery
from acquisition.motor_recovery import (
    move_with_recovery, MotorError, safe_stop, encoder_step_mismatch,
    encoder_motion_space_position, _read_encoder_counts, verify_encoder_zeroed,
)


class _Motor:
    def __init__(self):
        self.status = {"connected": True, "alarm": False, "fault": False,
                       "limits": {"CW": False, "CCW": False}, "alarm_message": "",
                       "position": 0}
        self.alarm_reset_calls = 0
        self.refresh_calls = 0

    def alarm_reset(self):
        self.alarm_reset_calls += 1
        self.status["alarm"] = False
        self.status["fault"] = False

    def move_off_limit(self):
        self.status["limits"] = {"CW": False, "CCW": False}

    def retrieve_motor_status(self):
        self.refresh_calls += 1

    def send_command(self, *a, **k):
        pass


class _Axis:
    def __init__(self, motor):
        self.motor = motor

    @property
    def connected(self):
        return self.motor.status.get("connected", True)

    @property
    def is_moving(self):
        return False


class _Drive:
    def __init__(self, n=1):
        self.axes = [_Axis(_Motor()) for _ in range(n)]
        self.enable_calls = 0
        self.disable_calls = 0

    def send_command(self, command, *a, **k):
        if command == "enable":
            self.enable_calls += 1
        elif command == "disable":
            self.disable_calls += 1


class _FakeMG:
    """Configurable fake motion group.

    ``behavior`` controls move_ml: a callable(self, index, attempt) -> None that
    sets self._pos and/or motor status to simulate success/miss/fault.
    """
    def __init__(self, name, ml_values, behavior):
        self.config = {"name": name}
        self.mb = type("MB", (), {"motion_list": type(
            "ML", (), {"values": np.asarray(ml_values, dtype=float),
                       "shape": np.asarray(ml_values).shape})()})()
        self.drive = _Drive(n=1)
        self._pos = (0.0, 0.0)
        self._attempt = 0
        self.behavior = behavior
        self.stop_calls = 0

    # mg.position -> object with .value tuple (like astropy Quantity-ish)
    @property
    def position(self):
        return type("P", (), {"value": self._pos})()

    def stop(self, soft=False):
        self.stop_calls += 1

    def move_ml(self, index):
        self._attempt += 1
        self.behavior(self, index, self._attempt)


class _RM:
    """Run manager whose ``is_moving`` reflects each group's moving countdown.

    A group that wants to model in-flight motion sets ``mg._moving_ticks`` > 0 and
    optionally an ``mg._advance`` callable invoked each poll to step ``_pos``.
    ``is_moving`` (read each settle poll) decrements the countdown and runs the
    advance, so the progress-aware settle exercises real logic.
    """
    def __init__(self, mgs):
        self.mgs = dict(mgs)

    @property
    def is_moving(self):
        any_moving = False
        for mg in self.mgs.values():
            ticks = getattr(mg, "_moving_ticks", 0)
            if ticks > 0:
                advance = getattr(mg, "_advance", None)
                if callable(advance):
                    advance(mg)
                mg._moving_ticks = ticks - 1
                if mg._moving_ticks > 0:
                    any_moving = True
        return any_moving


def _target(mg, index):
    return tuple(float(v) for v in np.asarray(mg.mb.motion_list.values)[index])


# --------------------------------------------------------------------------- #
# Behaviors
# --------------------------------------------------------------------------- #
def always_arrive(mg, index, attempt):
    mg._pos = _target(mg, index)


def miss_then_arrive(n_miss):
    def b(mg, index, attempt):
        if attempt <= n_miss:
            mg._pos = (999.0, 999.0)  # nowhere near target
        else:
            mg._pos = _target(mg, index)
    return b


def disconnect_then_reconnect(reconnect_on_attempt):
    def b(mg, index, attempt):
        motor = mg.drive.axes[0].motor
        if attempt < reconnect_on_attempt:
            motor.status["connected"] = False
            mg._pos = (999.0, 999.0)
        else:
            motor.status["connected"] = True
            mg._pos = _target(mg, index)
    return b


def alarm_then_clear(mg, index, attempt):
    """A limit/transient alarm on the first move; the *re-issue* clears it.

    Models the real library: ``Motor.move_to`` runs ``_moveable`` which resets a
    (limit) alarm and re-enables before moving -- so attempt 2 succeeds without
    the recovery module ever sending its own alarm_reset.
    """
    motor = mg.drive.axes[0].motor
    if attempt == 1:
        motor.status["alarm"] = True
        motor.status["alarm_message"] = "limit"
        mg._pos = (999.0, 999.0)
    else:
        # Re-issue: the library cleared the alarm and the move now lands.
        motor.status["alarm"] = False
        motor.status["alarm_message"] = ""
        mg._pos = _target(mg, index)


def never_arrive(mg, index, attempt):
    mg._pos = (999.0, 999.0)


def slow_progress_then_arrive(ticks, step_frac=0.25):
    """Model a long move: is_moving stays True for ``ticks`` polls, position
    creeps toward the target a fraction each poll, arriving at the end."""
    def b(mg, index, attempt):
        target = np.asarray(_target(mg, index))
        start = np.asarray(mg._pos)
        mg._moving_ticks = ticks + 1  # +1 so the final poll clears is_moving
        state = {"i": 0}

        def advance(m):
            state["i"] += 1
            frac = min(1.0, state["i"] / ticks)
            m._pos = tuple(start + (target - start) * frac)
        mg._advance = advance
    return b


def stuck_while_moving(mg, index, attempt):
    """Model a real stall: is_moving stays True but position never changes."""
    mg._moving_ticks = 10_000  # effectively never settles on its own
    mg._pos = (0.0, 0.0)
    mg._advance = lambda m: None  # no progress -> stall detector should fire


class MoveWithRecoveryTests(unittest.TestCase):
    def setUp(self):
        # Make settle / reconnect polls instant.
        self._orig_sleep = motor_recovery.time.sleep
        motor_recovery.time.sleep = lambda *_a, **_k: None
        # Deterministic, monotonic clock so stall/timeout windows are exact and
        # don't depend on wall-clock speed.
        self._t = {"now": 1000.0}
        self._orig_time = motor_recovery.time.time

        def fake_time():
            self._t["now"] += 0.5  # each call advances 0.5 s
            return self._t["now"]
        motor_recovery.time.time = fake_time

    def tearDown(self):
        motor_recovery.time.sleep = self._orig_sleep
        motor_recovery.time.time = self._orig_time

    def _mg(self, behavior):
        return _FakeMG("A", [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], behavior)

    def _run(self, behavior, **kw):
        mg = self._mg(behavior)
        rm = _RM({"a": mg})
        kw.setdefault("stall_timeout", 5.0)
        kw.setdefault("max_move_time", 1000.0)
        kw.setdefault("tol", 0.5)
        move_with_recovery(rm, {"a": "forward"}, 1, log=lambda *_a: None, **kw)
        return mg, rm

    def test_first_try_success(self):
        mg, _ = self._run(always_arrive)
        self.assertEqual(mg.position.value, (1.0, 1.0))
        self.assertEqual(mg.stop_calls, 0)  # no recovery needed

    def test_slow_progressing_move_is_not_interrupted(self):
        # A move that keeps progressing for many polls must finish on its own
        # without any soft-stop / re-issue, even though it takes "long".
        mg, _ = self._run(slow_progress_then_arrive(ticks=40), attempts=3,
                          stall_timeout=2.0, max_move_time=10_000.0,
                          progress_eps=0.001)
        self.assertEqual(mg.stop_calls, 0)          # never interrupted
        self.assertEqual(mg._attempt, 1)            # issued exactly once
        np.testing.assert_allclose(mg.position.value, (1.0, 1.0), atol=1e-6)

    def test_real_stall_escalates_and_raises(self):
        with self.assertRaises(MotorError):
            self._run(stuck_while_moving, attempts=2, stall_timeout=2.0,
                      max_move_time=1000.0)

    def test_transient_miss_recovers_on_retry(self):
        mg, _ = self._run(miss_then_arrive(n_miss=1), attempts=3)
        self.assertEqual(mg.position.value, (1.0, 1.0))
        self.assertGreaterEqual(mg.stop_calls, 1)  # soft-stop ladder ran

    def test_connection_loss_recovers_on_reissue(self):
        # Link drops during the first move; the re-issue (attempt 2) goes through.
        mg, _ = self._run(disconnect_then_reconnect(reconnect_on_attempt=2), attempts=2)
        self.assertEqual(mg.position.value, (1.0, 1.0))

    def test_resettable_alarm_recovers_via_reissue(self):
        # The library clears a limit alarm on re-issue; the recovery module must
        # NOT send its own alarm_reset (repeated AR is what risked encoder desync).
        mg, _ = self._run(alarm_then_clear, attempts=2)
        self.assertEqual(mg.position.value, (1.0, 1.0))
        self.assertEqual(mg.drive.axes[0].motor.alarm_reset_calls, 0)

    def test_permanent_failure_raises_motorerror(self):
        with self.assertRaises(MotorError):
            self._run(never_arrive, attempts=2)

    def test_not_reached_warns_and_carries_reason_into_motorerror(self):
        # A move that never reaches the target must (1) print a WARNING on the
        # run's log channel so the operator sees it live, and (2) carry the
        # "did not reach target" reason in the MotorError -- that message is what
        # the sink records into the HDF5 shot's skip_reason.
        printed = []
        mg = self._mg(never_arrive)
        rm = _RM({"a": mg})
        with self.assertRaises(MotorError) as ctx:
            move_with_recovery(
                rm, {"a": "forward"}, 1, attempts=2, stall_timeout=5.0,
                max_move_time=1000.0, tol=0.5, log=printed.append)
        # Print channel got a clear "did NOT reach target" warning.
        self.assertTrue(
            any("WARNING" in m and "did NOT reach target" in m for m in printed),
            f"expected a print-channel warning about not reaching target; got {printed}")
        # The reason propagates in the exception (-> HDF5 skip_reason).
        self.assertIn("did not reach target", str(ctx.exception))

    def test_verification_forces_status_refresh(self):
        mg, _ = self._run(always_arrive)
        # _refresh_status must have queried the motor before verifying.
        self.assertGreaterEqual(mg.drive.axes[0].motor.refresh_calls, 1)

    def test_no_alarm_reset_for_plain_miss(self):
        # A miss with no alarm should soft-stop + re-issue but NOT churn
        # alarm_reset (which can desync the encoder on real hardware).
        mg, _ = self._run(miss_then_arrive(n_miss=1), attempts=2)
        self.assertEqual(mg.drive.axes[0].motor.alarm_reset_calls, 0)

    def test_out_of_range_index_is_skipped_not_failed(self):
        # index 5 is beyond the 3-point list -> no motion, treated as success.
        mg = self._mg(never_arrive)
        rm = _RM({"a": mg})
        move_with_recovery(rm, {"a": "forward"}, 5, attempts=2,
                           stall_timeout=5.0, max_move_time=1000.0,
                           log=lambda *_a: None)
        self.assertEqual(mg._attempt, 0)  # move_ml never called


# --------------------------------------------------------------------------- #
# Encoder-vs-step mismatch helper
# --------------------------------------------------------------------------- #
class _EncMotor:
    """Motor exposing get_position (IP) and the raw EP escape hatch plus the
    scaling constants, for encoder_step_mismatch / encoder_motion_space_position.

    EP is read the way production reads it -- via ``_send_raw_command("EP")``,
    which returns the raw drive reply string (e.g. ``"EP=-12345"``) so the
    negative-safe parse is exercised."""
    def __init__(self, ip, ep, steps_per_rev=20000, enc_res=4000, support_ep=True,
                 units_per_rev=1.0, native=False):
        self._ip = ip
        self._ep = ep
        self.steps_per_rev = steps_per_rev
        self.units_per_rev = units_per_rev  # physical units (cm) per rev
        self._motor = {"encoder_resolution": enc_res}
        self._support_ep = support_ep
        self.status = {"connected": True}
        # When native=True, expose the patch_position_regex API (Motor.encoder
        # in counts, Motor.counts_per_rev) so tests can exercise the native path.
        if native:
            self.encoder = ep
            self.counts_per_rev = enc_res

    def send_command(self, command, *a, **k):
        if command == "get_position":
            return self._ip
        if command == "encoder_resolution":
            return self._motor.get("encoder_resolution")
        if command == "encoder_position":
            if a:  # a value was passed -> would be a WRITE; tests must never do this
                raise AssertionError("encoder_position must be read-only here")
            # Production no longer reads EP through send_command; flag if it does.
            raise AssertionError("EP must be read via _send_raw_command, not send_command")
        return None

    def _send_raw_command(self, cmd):
        if cmd == "EP":
            if not self._support_ep:
                return None  # model NACK / unsupported / no encoder
            return f"EP={self._ep}"
        return None


class _EncMG:
    def __init__(self, motors):
        self.config = {"name": "Enc"}
        self.drive = type("D", (), {"axes": [type("Ax", (), {"motor": m})()
                                             for m in motors]})()


class EncoderMismatchTests(unittest.TestCase):
    def test_no_mismatch_when_agree(self):
        # 10000 steps / 20000 = 0.5 rev; 2000 counts / 4000 = 0.5 rev -> agree.
        mg = _EncMG([_EncMotor(ip=10000, ep=2000)])
        self.assertEqual(encoder_step_mismatch(mg, tol_rev=0.01), [])

    def test_mismatch_flagged(self):
        # step = 0.5 rev, encoder = 0.25 rev -> 0.25 rev gap > 0.01 tol.
        mg = _EncMG([_EncMotor(ip=10000, ep=1000)])
        bad = encoder_step_mismatch(mg, tol_rev=0.01)
        self.assertEqual(len(bad), 1)
        idx, step_rev, enc_rev = bad[0]
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(step_rev, 0.5)
        self.assertAlmostEqual(enc_rev, 0.25)

    def test_unsupported_ep_is_skipped_silently(self):
        mg = _EncMG([_EncMotor(ip=10000, ep=0, support_ep=False)])
        self.assertEqual(encoder_step_mismatch(mg, tol_rev=0.01), [])

    def test_missing_constants_skipped(self):
        m = _EncMotor(ip=10000, ep=1000, enc_res=None)
        mg = _EncMG([m])
        self.assertEqual(encoder_step_mismatch(mg, tol_rev=0.01), [])

    def test_negative_encoder_handled(self):
        # The regression: a negative EP (move crossed zero) must be read and
        # compared, not crash or be silently skipped. step = -0.5 rev,
        # encoder = -0.5 rev -> agree.
        mg = _EncMG([_EncMotor(ip=-10000, ep=-2000)])
        self.assertEqual(encoder_step_mismatch(mg, tol_rev=0.01), [])

    def test_negative_encoder_mismatch_flagged(self):
        # step = -0.5 rev, encoder = -0.25 rev -> 0.25 rev gap flagged.
        mg = _EncMG([_EncMotor(ip=-10000, ep=-1000)])
        bad = encoder_step_mismatch(mg, tol_rev=0.01)
        self.assertEqual(len(bad), 1)
        idx, step_rev, enc_rev = bad[0]
        self.assertAlmostEqual(step_rev, -0.5)
        self.assertAlmostEqual(enc_rev, -0.25)

    def test_native_encoder_api_used_when_present(self):
        # With the patch_position_regex API (Motor.encoder + counts_per_rev), the
        # mismatch check reads the native encoder. step=0.5 rev, encoder=0.25 rev.
        mg = _EncMG([_EncMotor(ip=10000, ep=1000, native=True)])
        bad = encoder_step_mismatch(mg, tol_rev=0.01)
        self.assertEqual(len(bad), 1)
        idx, step_rev, enc_rev = bad[0]
        self.assertAlmostEqual(step_rev, 0.5)
        self.assertAlmostEqual(enc_rev, 0.25)

    def test_native_encoder_agrees(self):
        mg = _EncMG([_EncMotor(ip=10000, ep=2000, native=True)])
        self.assertEqual(encoder_step_mismatch(mg, tol_rev=0.01), [])


class ReadEncoderCountsTests(unittest.TestCase):
    def test_parses_positive(self):
        self.assertEqual(_read_encoder_counts(_EncMotor(ip=0, ep=12345)), 12345)

    def test_parses_negative(self):
        # The whole point: the library's EP=[0-9]+ regex can't do this.
        self.assertEqual(_read_encoder_counts(_EncMotor(ip=0, ep=-12345)), -12345)

    def test_unsupported_returns_none(self):
        self.assertIsNone(
            _read_encoder_counts(_EncMotor(ip=0, ep=0, support_ep=False)))

    def test_no_raw_command_returns_none(self):
        self.assertIsNone(_read_encoder_counts(object()))


class _XformMG:
    """Motion group stub with an encoder-bearing drive and a known transform.

    The helper feeds the transform per-axis drive positions in PHYSICAL units
    (cm) -- the same thing MotionGroup.position feeds it -- not motor steps. This
    transform applies a linear motion-space scale (default identity) so a test
    can assert the encoder-derived cm position is what comes back out."""
    def __init__(self, motors, ms_scale=1.0, has_ip=None):
        self.config = {"name": "Xf"}
        self.drive = type("D", (), {"axes": [type("Ax", (), {"motor": m})()
                                             for m in motors]})()
        self._ms_scale = ms_scale
        self._has_ip = has_ip  # tuple used by .position (the IP fallback)

    def transform(self, cm, to_coords="motion_space"):
        assert to_coords == "motion_space"
        # Input is drive position in cm; scale to motion space (identity by
        # default). A real transform would apply the probe geometry here.
        return [c * self._ms_scale for c in cm]

    @property
    def position(self):
        if self._has_ip is None:
            raise AssertionError("position (IP fallback) should not be used here")
        return type("Q", (), {"value": list(self._has_ip)})()


class EncoderMotionSpacePositionTests(unittest.TestCase):
    def test_encoder_derived_position(self):
        # The bug fix: convert encoder COUNTS -> cm before the transform, not to
        # steps. ep=2000 counts, enc_res=4000 -> 0.5 rev; units_per_rev=2.0 cm/rev
        # -> 1.0 cm per axis. Identity motion-space scale -> (1.0, 1.0) cm.
        mg = _XformMG([_EncMotor(ip=0, ep=2000, units_per_rev=2.0),
                       _EncMotor(ip=0, ep=2000, units_per_rev=2.0)])
        pos = encoder_motion_space_position(mg)
        self.assertEqual(len(pos), 2)
        self.assertAlmostEqual(pos[0], 1.0)
        self.assertAlmostEqual(pos[1], 1.0)

    def test_realistic_cm_magnitude(self):
        # Guard against the "weird numbers" regression: a real-ish encoder count
        # must yield a sane cm value, not a raw step count. 0.508 cm/rev,
        # enc_res=4000 counts/rev: 15 cm -> 15/0.508*4000 ~= 118110 counts.
        counts = round(15.0 / 0.508 * 4000)
        mg = _XformMG([_EncMotor(ip=0, ep=counts, enc_res=4000,
                                 units_per_rev=0.508)])
        pos = encoder_motion_space_position(mg)
        self.assertAlmostEqual(pos[0], 15.0, places=2)  # ~15 cm, not ~118110

    def test_negative_encoder_derived_position(self):
        mg = _XformMG([_EncMotor(ip=0, ep=-2000, units_per_rev=2.0),
                       _EncMotor(ip=0, ep=-2000, units_per_rev=2.0)])
        pos = encoder_motion_space_position(mg)
        self.assertAlmostEqual(pos[0], -1.0)
        self.assertAlmostEqual(pos[1], -1.0)

    def test_none_when_units_per_rev_missing(self):
        # Without units_per_rev we can't convert counts->cm -> None (fall back).
        m = _EncMotor(ip=0, ep=2000, units_per_rev=None)
        self.assertIsNone(encoder_motion_space_position(_XformMG([m])))

    def test_none_when_encoder_unavailable(self):
        # No encoder support -> None, so the caller falls back to IP.
        mg = _XformMG([_EncMotor(ip=0, ep=0, support_ep=False, units_per_rev=2.0)])
        self.assertIsNone(encoder_motion_space_position(mg))


class _NativeEncMG:
    """Motion group exposing the native bapsf_motion ``encoder`` property
    (patch_position_regex and later): a motion-space Quantity-like value already
    through the transform. ``manual`` is what the hand-rolled fallback would
    return if the native path is skipped, so a test can prove which path ran."""
    def __init__(self, encoder_value, manual_motors=None, ms_scale=1.0):
        self.config = {"name": "Nat"}
        self._encoder_value = encoder_value  # tuple, None, or "raise"
        # Optional manual-path backing so we can assert fallback behaviour.
        motors = manual_motors or []
        self.drive = type("D", (), {"axes": [type("Ax", (), {"motor": m})()
                                             for m in motors]})()
        self._ms_scale = ms_scale

    @property
    def encoder(self):
        if self._encoder_value == "raise":
            raise RuntimeError("encoder read failed")
        if self._encoder_value is None:
            return None
        return type("Q", (), {"value": list(self._encoder_value)})()

    def transform(self, cm, to_coords="motion_space"):
        assert to_coords == "motion_space"
        return [c * self._ms_scale for c in cm]


class NativeEncoderPreferenceTests(unittest.TestCase):
    def test_uses_native_encoder_when_present(self):
        # Native mg.encoder gives motion-space directly; helper returns it as-is
        # and never touches the manual path (no motors configured).
        mg = _NativeEncMG(encoder_value=(3.0, -4.0))
        self.assertEqual(encoder_motion_space_position(mg), (3.0, -4.0))

    def test_falls_back_to_manual_when_native_none(self):
        # Cold heartbeat cache -> mg.encoder is None; helper falls back to the
        # manual counts->cm path. ep=2000/enc_res=4000=0.5rev * 2.0 cm/rev = 1.0.
        mg = _NativeEncMG(encoder_value=None,
                          manual_motors=[_EncMotor(ip=0, ep=2000, units_per_rev=2.0)])
        self.assertAlmostEqual(encoder_motion_space_position(mg)[0], 1.0)

    def test_falls_back_to_manual_when_native_raises(self):
        mg = _NativeEncMG(encoder_value="raise",
                          manual_motors=[_EncMotor(ip=0, ep=2000, units_per_rev=2.0)])
        self.assertAlmostEqual(encoder_motion_space_position(mg)[0], 1.0)

    def test_falls_back_when_native_non_finite(self):
        # A NaN/inf from the native path is treated as unavailable -> manual.
        mg = _NativeEncMG(encoder_value=(float("nan"), 0.0),
                          manual_motors=[_EncMotor(ip=0, ep=2000, units_per_rev=2.0)])
        self.assertAlmostEqual(encoder_motion_space_position(mg)[0], 1.0)

    def test_returns_none_when_both_paths_fail(self):
        # Native None and no manual backing -> None (caller falls back to IP).
        mg = _NativeEncMG(encoder_value=None)
        self.assertIsNone(encoder_motion_space_position(mg))


class VerifyEncoderZeroedTests(unittest.TestCase):
    def test_all_zero_passes(self):
        mg = _EncMG([_EncMotor(ip=0, ep=0), _EncMotor(ip=0, ep=0)])
        self.assertEqual(verify_encoder_zeroed(mg, log=lambda *_a: None), [])

    def test_small_jitter_within_tol_passes(self):
        # A couple counts of end-of-write jitter is fine.
        mg = _EncMG([_EncMotor(ip=0, ep=1), _EncMotor(ip=0, ep=-2)])
        self.assertEqual(
            verify_encoder_zeroed(mg, tol_counts=2.0, log=lambda *_a: None), [])

    def test_nonzero_encoder_flagged(self):
        # One axis didn't zero (reads 500 counts) -> reported.
        mg = _EncMG([_EncMotor(ip=0, ep=0), _EncMotor(ip=0, ep=500)])
        bad = verify_encoder_zeroed(mg, tol_counts=2.0, log=lambda *_a: None)
        self.assertEqual(bad, [(1, 500)])

    def test_negative_nonzero_encoder_flagged(self):
        # Negative residual must also be caught (the regression-prone case).
        mg = _EncMG([_EncMotor(ip=0, ep=-500)])
        bad = verify_encoder_zeroed(mg, tol_counts=2.0, log=lambda *_a: None)
        self.assertEqual(bad, [(0, -500)])

    def test_unreadable_encoder_flagged_as_unconfirmed(self):
        # Can't read EP -> can't confirm zero -> reported (axis, None).
        mg = _EncMG([_EncMotor(ip=0, ep=0, support_ep=False)])
        bad = verify_encoder_zeroed(mg, log=lambda *_a: None)
        self.assertEqual(bad, [(0, None)])


class SafeStopTests(unittest.TestCase):
    def test_safe_stop_calls_all_and_swallows_errors(self):
        good = _FakeMG("A", [[0.0, 0.0]], always_arrive)

        class _Boom(_FakeMG):
            def stop(self, soft=False):
                raise RuntimeError("stop failed")

        bad = _Boom("B", [[0.0, 0.0]], always_arrive)
        rm = _RM({"a": good, "b": bad})
        safe_stop(rm)  # must not raise
        self.assertEqual(good.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
