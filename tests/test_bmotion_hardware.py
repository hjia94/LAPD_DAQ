"""End-to-end bmotion hardware checks.

Two families of checks, all skipped by default; opt in by flipping the flags at
the top of this file:

  * Full-acquisition runs (``run_acquisition_bmotion`` against real motors):
    interleaved and sequential end-to-end.
  * Motor-recovery diagnostics that drive ``acquisition.motor_recovery`` directly
    against real motors, to validate the move-handling fixes in isolation:
      - LONG MOTION: a slow full-range move must be left to finish (the
        progress-aware settle must not time it out or interrupt it).
      - ENCODER: read the encoder (EP) vs the commanded/step position (IP) around
        a real move and confirm they track each other.
      - FAILURE: command a known-unreachable index, confirm recovery raises
        MotorError, and confirm a subsequent good move still succeeds (the
        regression guard for the original "one failure poisons every later move"
        bug).

Pattern mirrors tests/test_hardware_instruments.py:
  * Module-level RUN_* flag enables the test class.
  * BMOTION_ALLOW_MOVE is a separate destructive-action gate; without it
    the test fails fast rather than touching motors.
  * RECOVERY_MOTION_GROUP picks which motion group the recovery diagnostics drive
    (set it to e.g. "Hermes").

Run with:

    pytest tests/test_bmotion_hardware.py -v -s
    # or
    python -m unittest tests.test_bmotion_hardware -v
"""

from __future__ import annotations

import configparser
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hardware_check_base import HardwareCheckBase


# --------------------------------------------------------------------------- #
# Enable individual hardware checks here. Each is skipped unless True.
# --------------------------------------------------------------------------- #
RUN_BMOTION_INTERLEAVED_CHECK = False
RUN_BMOTION_SEQUENTIAL_CHECK = False

# Recovery-path diagnostics (exercise acquisition.motor_recovery on real motors):
RUN_BMOTION_LONG_MOTION_CHECK = False     # slow full-range move must finish, not time out
RUN_BMOTION_ENCODER_CHECK = False         # encoder (EP) vs step (IP) agreement around a move
RUN_BMOTION_FAILURE_CHECK = False         # a genuinely-unreachable target -> MotorError + skip

# Safety gate — no test will arm motors unless this is also True.
BMOTION_ALLOW_MOVE = False

# --------------------------------------------------------------------------- #
# Connection info. These paths are resolved relative to the current working
# directory; pass absolute paths to avoid surprises.
# --------------------------------------------------------------------------- #
EXPERIMENT_CONFIG_PATH = "experiment_config.txt"
BMOTION_TOML_PATH = "bmotion_config.toml"

# Use 1 shot per position for diagnostics. Increase only after a clean run.
BMOTION_NSHOTS = 1

# --------------------------------------------------------------------------- #
# Recovery-path diagnostic parameters.
# --------------------------------------------------------------------------- #
# Which motion group to drive in the recovery-path checks. Set this to the name
# (e.g. "Hermes") or the TOML key of the group you want to test. Leave as None to
# use the first selected group from experiment_config.txt (or the first group in
# the TOML if no config is present).
RECOVERY_MOTION_GROUP = None       # e.g. "Hermes"

# Long-motion: the move-list indices to travel between (far apart = long move).
# "first"/"last" map to the ends of the configured motion list.
LONG_MOTION_FROM_INDEX = "first"
LONG_MOTION_TO_INDEX = "last"
# Per-move recovery tunables for the long-motion check. A generous ceiling and a
# short stall window prove the progress-aware settle leaves a slow move alone.
LONG_MOTION_MAX_TIME_S = 600.0     # 10-min absolute backstop
LONG_MOTION_STALL_TIMEOUT_S = 10.0
LONG_MOTION_ATTEMPTS = 2

# Encoder check: max |encoder - step| disagreement (motor revolutions) tolerated.
ENCODER_MISMATCH_TOL_REV = 0.01

# Failure check: a motion-list index for the selected group that is known to be
# UNREACHABLE on this rig -- e.g. a position the probe physically cannot get to
# (blocked, past an obstruction, or in a region the drive can't complete). The
# recovery ladder should exhaust its attempts on this index and raise MotorError,
# which the acquisition loop turns into skip-and-continue. MUST be set when
# RUN_BMOTION_FAILURE_CHECK is True. No motion list is faked: this drives a real
# index so the test exercises the real code path end to end.
FAILURE_TARGET_INDEX = None        # e.g. 7  (an index you know is unreachable)
# Keep the failure check fast: few attempts, short windows.
FAILURE_ATTEMPTS = 2
FAILURE_STALL_TIMEOUT_S = 8.0
FAILURE_MAX_TIME_S = 30.0
# --------------------------------------------------------------------------- #


def _have_bmotion_install() -> bool:
    try:
        import bapsf_motion  # noqa: F401
        import xarray  # noqa: F401
        return True
    except ImportError:
        return False


def _have_required_files() -> bool:
    return Path(EXPERIMENT_CONFIG_PATH).is_file() and Path(BMOTION_TOML_PATH).is_file()


def _write_config_variant(src_config: Path, dst_config: Path, execution_order: str,
                          nshots: int) -> None:
    """Copy the source experiment_config.txt and override [bmotion]
    execution_order plus [nshots].num_duplicate_shots so the hardware test
    can pin both without mutating the source-tree config."""
    cp = configparser.ConfigParser(inline_comment_prefixes=None)
    cp.read(src_config)

    if not cp.has_section("nshots"):
        cp.add_section("nshots")
    cp.set("nshots", "num_duplicate_shots", str(nshots))

    if not cp.has_section("bmotion"):
        cp.add_section("bmotion")
    cp.set("bmotion", "execution_order", execution_order)
    # Keep whatever motion_groups / direction the user already configured.

    with open(dst_config, "w") as f:
        cp.write(f)


class _BmotionHardwareBase(HardwareCheckBase):
    """Shared flag-gating + tempdir layout for bmotion end-to-end tests."""

    run_flag: bool = False
    label: str = "bmotion"
    execution_order: str = "interleaved"

    def _run_flag_skip_message(self) -> str:
        return (
            f"{type(self).__name__} disabled "
            f"(set its RUN_BMOTION_*_CHECK flag to True)"
        )

    def gate_checks(self) -> list[tuple[bool, str]]:
        return [
            (not BMOTION_ALLOW_MOVE,
             "BMOTION_ALLOW_MOVE is False — refusing to command motors"),
            (not _have_bmotion_install(),
             "bapsf_motion / xarray not installed on this machine"),
            (not _have_required_files(),
             f"Missing {EXPERIMENT_CONFIG_PATH} or {BMOTION_TOML_PATH} "
             f"in the current working directory"),
        ]

    def _allocate_tempdir(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.output_path = self.tmp_dir / f"{self.label}_{self.execution_order}.hdf5"
        self.config_path = self.tmp_dir / "experiment_config.txt"
        _write_config_variant(
            Path(EXPERIMENT_CONFIG_PATH), self.config_path,
            execution_order=self.execution_order, nshots=BMOTION_NSHOTS,
        )

    def _run(self) -> None:
        # Late import so the module-level skip messages above fire before
        # bapsf_motion is touched.
        from acquisition import run_acquisition_bmotion

        run_acquisition_bmotion(
            str(self.output_path),
            BMOTION_TOML_PATH,
            str(self.config_path),
        )

    def _read_selection_blob(self) -> dict:
        with h5py.File(self.output_path, "r") as f:
            return json.loads(f["Configuration/bmotion_selection"][()])

    def _read_positions(self, mg_name: str) -> np.ndarray:
        with h5py.File(self.output_path, "r") as f:
            return f[f"Control/Positions/{mg_name}/positions_array"][:]

    def _list_mg_names(self) -> list:
        with h5py.File(self.output_path, "r") as f:
            return list(f["Control/Positions"].keys())


# --------------------------------------------------------------------------- #
class BmotionInterleavedHardwareCheck(_BmotionHardwareBase):
    """End-to-end run with execution_order = interleaved against real motors."""

    run_flag = RUN_BMOTION_INTERLEAVED_CHECK
    label = "bmotion_interleaved"
    execution_order = "interleaved"

    def test_interleaved_end_to_end(self) -> None:
        self._run()

        blob = self._read_selection_blob()
        self.assertEqual(blob["execution_order"], "interleaved")
        self.assertTrue(blob["mg_keys"], "no motion groups selected")

        mg_names = self._list_mg_names()
        self.assertTrue(mg_names, "no Control/Positions/<mg> groups created")

        # In interleaved mode every selected MG should have a populated row
        # at every shot index — there are no idle phases.
        first = self._read_positions(mg_names[0])
        self.assertGreater(len(first), 0)
        self.assertTrue(np.all(first["shot_num"] > 0),
                        "interleaved run left zero rows in the first MG")
        for name in mg_names[1:]:
            arr = self._read_positions(name)
            self.assertEqual(len(arr), len(first),
                             "interleaved MGs should share total_shots length")
            self.assertTrue(np.all(arr["shot_num"] > 0),
                            f"interleaved run left zero rows in MG {name}")


# --------------------------------------------------------------------------- #
class BmotionSequentialHardwareCheck(_BmotionHardwareBase):
    """End-to-end run with execution_order = sequential against real motors."""

    run_flag = RUN_BMOTION_SEQUENTIAL_CHECK
    label = "bmotion_sequential"
    execution_order = "sequential"

    def test_sequential_end_to_end(self) -> None:
        self._run()

        blob = self._read_selection_blob()
        self.assertEqual(blob["execution_order"], "sequential")
        self.assertTrue(blob["mg_keys"], "no motion groups selected")

        mg_names = self._list_mg_names()
        self.assertTrue(mg_names, "no Control/Positions/<mg> groups created")
        if len(mg_names) < 2:
            self.skipTest(
                "Sequential-mode assertions need >= 2 motion groups; "
                f"got {mg_names}. Configure at least two in [bmotion]."
            )

        # Each MG's positions_array should have exactly one active (nonzero)
        # contiguous block; the other MGs' rows at those shot indices should
        # be all-zero (idle-group skip behavior).
        per_mg_active = {}
        for name in mg_names:
            arr = self._read_positions(name)
            active_mask = arr["shot_num"] > 0
            per_mg_active[name] = active_mask
            self.assertTrue(active_mask.any(),
                            f"MG {name} has no shots recorded")

        # No two MGs are active at the same shot index.
        for i, name_i in enumerate(mg_names):
            for name_j in mg_names[i + 1:]:
                overlap = per_mg_active[name_i] & per_mg_active[name_j]
                if overlap.any():
                    first_overlap = int(np.where(overlap)[0][0])
                    self.fail(
                        f"MGs {name_i} and {name_j} both wrote to shot index "
                        f"{first_overlap} — sequential mode should record "
                        f"only the active group"
                    )

        # Combined coverage should equal total_shots (i.e. every shot has
        # exactly one active MG).
        total_shots = len(next(iter(per_mg_active.values())))
        combined = np.zeros(total_shots, dtype=bool)
        for mask in per_mg_active.values():
            combined |= mask
        self.assertTrue(combined.all(),
                        "some shot indices have no active MG row")


# --------------------------------------------------------------------------- #
# Recovery-path diagnostics: build a RunManager directly and drive
# acquisition.motor_recovery against real motors. These do NOT run a full
# acquisition; they isolate the motor-move behavior so a slow move, the encoder
# feedback, and a real failure can each be inspected on their own.
# --------------------------------------------------------------------------- #
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
        from acquisition.bmotion import resolve_bmotion_selection
        from acquisition.config import load_experiment_config

        self._bmotion = bmotion
        print("\n[recovery check] loading TOML / starting RunManager...")
        self.rm = bmotion.actors.RunManager(BMOTION_TOML_PATH, auto_run=True)
        # Resolve the configured selection so we drive exactly the groups the
        # operator set up (honoring direction); fall back to all groups if the
        # experiment_config.txt isn't present.
        try:
            config, _ = load_experiment_config(EXPERIMENT_CONFIG_PATH)
            sel = resolve_bmotion_selection(config, self.rm)
            self.mg_keys = list(sel.mg_keys)
            self.ml_order = dict(sel.direction)
        except Exception as exc:  # noqa: BLE001 - diagnostics: be forgiving
            print(f"[recovery check] selection fallback (all groups): {exc}")
            self.mg_keys = list(self.rm.mgs.keys())
            self.ml_order = {k: "forward" for k in self.mg_keys}
        self.assertTrue(self.mg_keys, "no motion groups available")

        # Pick the single group the recovery checks drive. The operator sets
        # RECOVERY_MOTION_GROUP to a group name (e.g. "Hermes") or its TOML key;
        # default is the first selected group.
        self.mg_key = self._select_mg_key()
        self.order = {self.mg_key: self.ml_order.get(self.mg_key, "forward")}
        print(f"[recovery check] available groups: {self.mg_keys}, "
              f"directions: {self.ml_order}")
        print(f"[recovery check] driving motion group: '{self.mg_key}' "
              f"(name='{self.rm.mgs[self.mg_key].config.get('name', self.mg_key)}')")

    def _select_mg_key(self):
        """Resolve RECOVERY_MOTION_GROUP (key or display name) to an mg key."""
        if RECOVERY_MOTION_GROUP is None:
            return self.mg_keys[0]
        # Direct key match.
        if RECOVERY_MOTION_GROUP in self.rm.mgs:
            return RECOVERY_MOTION_GROUP
        # Match by configured display name (e.g. "Hermes").
        for key, mg in self.rm.mgs.items():
            if mg.config.get("name") == RECOVERY_MOTION_GROUP:
                return key
        self.skipTest(
            f"RECOVERY_MOTION_GROUP={RECOVERY_MOTION_GROUP!r} not found; "
            f"available keys={list(self.rm.mgs.keys())}, "
            f"names={[mg.config.get('name') for mg in self.rm.mgs.values()]}"
        )

    def tearDown(self) -> None:
        try:
            self.rm.terminate()
        except Exception as exc:  # noqa: BLE001
            print(f"[recovery check] RunManager.terminate failed: {exc}")
        super().tearDown()

    # --- raw motor reads (bypass the heartbeat cache) ---------------------- #
    def _read_ep_ip_rev(self, mg_key):
        """Return ``[(axis_idx, ip_rev, ep_rev)]`` for one motion group.

        IP (commanded/step) and EP (encoder) are read directly and converted to
        motor revolutions, the unit in which they're comparable. Used to *report*
        encoder behavior, independent of the pass/fail tolerance check."""
        out = []
        mg = self.rm.mgs[mg_key]
        for idx, ax in enumerate(mg.drive.axes):
            motor = ax.motor
            ip = motor.send_command("get_position")
            ep = motor.send_command("encoder_position")  # no arg => read
            ip_v = getattr(ip, "value", ip)
            ep_v = getattr(ep, "value", ep)
            try:
                spr = float(getattr(motor, "steps_per_rev").value)
                er = float(motor._motor["encoder_resolution"])
                out.append((idx, float(ip_v) / spr, float(ep_v) / er))
            except (TypeError, ValueError, AttributeError, KeyError):
                out.append((idx, None, None))
        return out

    def _print_ep_ip(self, mg_key, when):
        for idx, ip_rev, ep_rev in self._read_ep_ip_rev(mg_key):
            if ip_rev is None:
                print(f"[recovery check] {mg_key} axis {idx} {when}: EP/IP unavailable")
            else:
                print(f"[recovery check] {mg_key} axis {idx} {when}: "
                      f"step(IP)={ip_rev:.4f} rev  encoder(EP)={ep_rev:.4f} rev  "
                      f"diff={abs(ip_rev - ep_rev):.4f} rev")


# --------------------------------------------------------------------------- #
class BmotionLongMotionHardwareCheck(_RecoveryHardwareBase):
    """A long, slow move must be left to finish — never interrupted/timed out.

    Drives a full-range move (first -> last motion-list index) through
    ``move_with_recovery`` with a short stall window and a generous absolute
    ceiling. Because the move keeps making progress, the progress-aware settle
    must let it finish: no MotorError, exactly one move issued (no soft-stop /
    re-issue), and the final position matches the target."""

    run_flag = RUN_BMOTION_LONG_MOTION_CHECK
    label = "bmotion_long_motion"

    def test_long_move_completes_without_interruption(self) -> None:
        from acquisition.motor_recovery import move_with_recovery, MotorError

        mg = self.rm.mgs[self.mg_key]
        ml_size = int(mg.mb.motion_list.shape[0])
        self.assertGreaterEqual(ml_size, 2,
                                "motion list needs >= 2 points for a long move")

        from_idx = _resolve_ml_index(LONG_MOTION_FROM_INDEX, ml_size)
        to_idx = _resolve_ml_index(LONG_MOTION_TO_INDEX, ml_size)

        # Move to the start, then time the long move to the far end.
        print(f"[recovery check] seeding start position (index {from_idx})")
        move_with_recovery(self.rm, self.order, from_idx, attempts=LONG_MOTION_ATTEMPTS,
                           stall_timeout=LONG_MOTION_STALL_TIMEOUT_S,
                           max_move_time=LONG_MOTION_MAX_TIME_S,
                           encoder_mismatch_tol_rev=ENCODER_MISMATCH_TOL_REV,
                           log=print)

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
            move_with_recovery(self.rm, self.order, to_idx, attempts=LONG_MOTION_ATTEMPTS,
                               stall_timeout=LONG_MOTION_STALL_TIMEOUT_S,
                               max_move_time=LONG_MOTION_MAX_TIME_S,
                               encoder_mismatch_tol_rev=ENCODER_MISMATCH_TOL_REV,
                               log=print)
            elapsed = time.time() - t0
        except MotorError as exc:
            self.fail(f"long move was wrongly treated as a failure: {exc}")
        finally:
            mg.move_ml = _orig_move_ml

        print(f"[recovery check] long move finished in {elapsed:.1f}s, "
              f"move_ml issued {issued['count']}x")
        # The healthy move must NOT have been interrupted and re-issued.
        self.assertEqual(issued["count"], 1,
                         "a progressing move should be issued exactly once "
                         "(it was soft-stopped + re-issued, i.e. interrupted)")
        # Arrived at the requested target (within the recovery tolerance).
        import numpy as np
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
    ``ENCODER_MISMATCH_TOL_REV`` afterward (catches lost steps / encoder slip),
    using the same read-only check the acquisition loop runs."""

    run_flag = RUN_BMOTION_ENCODER_CHECK
    label = "bmotion_encoder"

    def test_encoder_tracks_step_position(self) -> None:
        from acquisition.motor_recovery import move_with_recovery, encoder_step_mismatch

        mg = self.rm.mgs[self.mg_key]
        ml_size = int(mg.mb.motion_list.shape[0])

        self._print_ep_ip(self.mg_key, "before move")

        # Move to the last index (a meaningful displacement) and re-check.
        to_idx = _resolve_ml_index("last", ml_size)
        move_with_recovery(self.rm, self.order, to_idx, attempts=LONG_MOTION_ATTEMPTS,
                           stall_timeout=LONG_MOTION_STALL_TIMEOUT_S,
                           max_move_time=LONG_MOTION_MAX_TIME_S,
                           encoder_mismatch_tol_rev=ENCODER_MISMATCH_TOL_REV,
                           log=print)

        self._print_ep_ip(self.mg_key, "after move")

        bad = encoder_step_mismatch(mg, tol_rev=ENCODER_MISMATCH_TOL_REV)
        if bad:
            detail = "; ".join(
                f"axis {i}: step={s:.4f} rev vs encoder={e:.4f} rev "
                f"(diff {abs(s - e):.4f} rev)"
                for i, s, e in bad
            )
            self.fail(f"encoder disagrees with motor/step position: {detail}")
        print("[recovery check] encoder agrees with step position within "
              f"{ENCODER_MISMATCH_TOL_REV} rev")


# --------------------------------------------------------------------------- #
class BmotionFailureHandlingHardwareCheck(_RecoveryHardwareBase):
    """A genuinely-unreachable target must raise MotorError, not hang or corrupt.

    Commands a move to FAILURE_TARGET_INDEX -- a *real* motion-list index the
    operator knows the probe cannot reach on this rig -- with a short stall window
    and few attempts, and asserts ``move_with_recovery`` raises ``MotorError``
    after exhausting recovery. Then it verifies the two things the acquisition
    loop relies on for skip-and-continue: the position is still readable, and the
    SAME recovery call succeeds for a known-good index right afterward (the failed
    move did not poison subsequent moves -- the original bug)."""

    run_flag = RUN_BMOTION_FAILURE_CHECK
    label = "bmotion_failure"

    def gate_checks(self) -> list[tuple[bool, str]]:
        return super().gate_checks() + [
            (FAILURE_TARGET_INDEX is None,
             "FAILURE_TARGET_INDEX is None — set a motion-list index you know is "
             "unreachable to test failure handling"),
        ]

    def test_unreachable_index_raises_then_next_move_recovers(self) -> None:
        import numpy as np
        from acquisition.motor_recovery import move_with_recovery, MotorError

        mg = self.rm.mgs[self.mg_key]
        ml_size = int(mg.mb.motion_list.shape[0])
        bad_index = int(FAILURE_TARGET_INDEX)
        self.assertTrue(0 <= bad_index < ml_size,
                        f"FAILURE_TARGET_INDEX {bad_index} out of range [0,{ml_size})")
        base_dim = int(np.asarray(mg.mb.motion_list.values).shape[1])

        # 1) The unreachable index must exhaust recovery and raise MotorError.
        print(f"[recovery check] commanding UNREACHABLE index {bad_index} "
              f"(expecting MotorError after {FAILURE_ATTEMPTS} attempts)")
        with self.assertRaises(MotorError):
            move_with_recovery(
                self.rm, self.order, bad_index,
                attempts=FAILURE_ATTEMPTS,
                stall_timeout=FAILURE_STALL_TIMEOUT_S,
                max_move_time=FAILURE_MAX_TIME_S,
                encoder_mismatch_tol_rev=ENCODER_MISMATCH_TOL_REV,
                log=print,
            )

        # 2) The system is still queryable (skip-and-continue can record the
        #    position and move on rather than aborting the run).
        pos = mg.position.value
        print(f"[recovery check] position still readable after failure: {tuple(pos)}")
        self.assertEqual(len(pos), base_dim)

        # 3) Critically: a subsequent move to a known-good index must succeed.
        #    This is the regression guard for the original bug, where a failed/
        #    interrupted move left the motor in a state that made every later
        #    position read "did not reach target".
        good_index = _resolve_ml_index("first", ml_size)
        if good_index == bad_index:
            good_index = _resolve_ml_index("last", ml_size)
        print(f"[recovery check] verifying recovery: moving to good index {good_index}")
        try:
            move_with_recovery(
                self.rm, self.order, good_index,
                attempts=LONG_MOTION_ATTEMPTS,
                stall_timeout=LONG_MOTION_STALL_TIMEOUT_S,
                max_move_time=LONG_MOTION_MAX_TIME_S,
                encoder_mismatch_tol_rev=ENCODER_MISMATCH_TOL_REV,
                log=print,
            )
        except MotorError as exc:
            self.fail(f"a failed move poisoned the next move (the original bug): {exc}")
        print(f"[recovery check] recovered: reached good index {good_index} "
              f"after the failed move")


# --------------------------------------------------------------------------- #
def _resolve_ml_index(spec, ml_size: int) -> int:
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


if __name__ == "__main__":
    unittest.main()
