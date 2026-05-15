"""Hardware diagnostic tests for individual instruments.

These tests connect to one real instrument at a time (LeCroy scope, motion
controller, Phantom camera) through the lapd_daq adapter code. They are
skipped by default so a normal `pytest` run on a developer machine stays
green; opt in by editing the flags at the top of this file.

Two flag groups gate destructive actions:

  - SCOPE_ALLOW_ACQUIRE    — required before scope arms and writes a shot.
  - MOTION_ALLOW_MOVE      — required before any motor move is commanded.
  - CAMERA_ALLOW_RECORD    — required before the camera waits for trigger.

Run with:

    pytest tests/test_hardware_instruments.py -v -s

There are also `Data_Run`-style end-to-end tests (`DataRunScopeHardware`,
`DataRunMotionHardware`) that exercise the legacy acquisition path with a
real instrument and a fake counterpart.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from lapd_daq.config import load_run_config
from lapd_daq.devices.lab_scopes import LabScopesLeCroyScopeAdapter
from lapd_daq.devices.legacy_motion import LegacyMotorAdapter
from lapd_daq.devices.phantom import PhantomCameraAdapter
from lapd_daq.models import PlannedPosition, ShotPlan, ShotResult
from lapd_daq.storage.hdf5 import HDF5RunWriter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hardware_check_helpers import (
    ensure_fake_scope_config,
    fake_scope_payload,
    fake_time_array,
    restrict_scope_config,
    target_coordinates,
)

# --------------------------------------------------------------------------- #
# Enable individual hardware checks here. Each is skipped unless True.
# --------------------------------------------------------------------------- #
RUN_SCOPE_CHECK = False
RUN_MOTION_CHECK = False
RUN_CAMERA_CHECK = False
RUN_DATA_RUN_SCOPE_CHECK = False
RUN_DATA_RUN_MOTION_CHECK = False

# Safety gates — destructive actions are off by default even when the check
# above is enabled. Flip explicitly to acquire / move / record.
SCOPE_ALLOW_ACQUIRE = False
MOTION_ALLOW_MOVE = False
CAMERA_ALLOW_RECORD = False

# --------------------------------------------------------------------------- #
# Connection info / parameters. EXPERIMENT_CONFIG_PATH is resolved relative to
# the current working directory; pass an absolute path to avoid surprises.
# --------------------------------------------------------------------------- #
EXPERIMENT_CONFIG_PATH = "experiment_config.txt"

# Scope check
SCOPE_NAME = None                # None = first scope in [scope_ips]
SCOPE_CONNECT_TIMEOUT_S = 30.0
SCOPE_SHOT_NUM = 1

# Motion check
MOTION_DIMENSION = "auto"        # "auto" | "xy" | "xyz"
MOTION_TARGET = None             # e.g. (0.0, 0.0) or (0.0, 0.0, 0.0); None = read-only
MOTION_SHOT_NUM = 1

# Camera check
CAMERA_EXPERIMENT_NAME = "hardware_camera_check"
CAMERA_SHOT_NUM = 1

# Data_Run-style scope check
DATA_RUN_SCOPE_NAME = None       # None = use all scopes from [scope_ips]
DATA_RUN_SCOPE_SHOTS = 1

# Data_Run-style motion check (real motors, fake delayed scope)
DATA_RUN_MOTION_MAX_SHOTS = 1
DATA_RUN_MOTION_PAUSE_S = 0.5
DATA_RUN_FAKE_SCOPE = "PauseScope"
DATA_RUN_FAKE_CHANNEL = "C1"
DATA_RUN_FAKE_POINTS = 16
# --------------------------------------------------------------------------- #


class _HardwareCheckBase(unittest.TestCase):
    """Shared tempdir lifecycle and run-flag gating for hardware tests.

    Subclasses set `run_flag` to the boolean controlling whether the test
    runs. setUp skips before allocating resources.
    """

    run_flag: bool = False
    label: str = "check"

    def setUp(self) -> None:
        if not self.run_flag:
            self.skipTest(f"{type(self).__name__} disabled (set its run flag to True)")
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.output_path = self.tmp_dir / f"{self.label}_check.hdf5"

    def tearDown(self) -> None:
        self._tmp.cleanup()


# --------------------------------------------------------------------------- #
class ScopeHardwareCheck(_HardwareCheckBase):
    """Connect to one real LeCroy scope; optionally acquire one shot."""

    run_flag = RUN_SCOPE_CHECK
    label = "scope"

    def test_scope_connects_and_optionally_acquires(self) -> None:
        config = load_run_config(EXPERIMENT_CONFIG_PATH, mode="stationary", output_path=self.output_path)
        scope_config = _select_scope(config, SCOPE_NAME)
        config = replace(config, scopes=[scope_config])

        scope = LabScopesLeCroyScopeAdapter(
            scope_config.name,
            scope_config.ip_address,
            description=scope_config.description,
            timeout=SCOPE_CONNECT_TIMEOUT_S,
        )
        print(f"\n[scope check] connecting to {scope_config.name} at {scope_config.ip_address}")
        try:
            scope.connect()
            scope.initialize()
            time_points = len(scope.time_array())
            print(f"[scope check] initialized; {time_points} displayed time points")
            self.assertGreater(time_points, 0, "scope reported no time points")

            if not SCOPE_ALLOW_ACQUIRE:
                print("[scope check] initialize-only PASS (set SCOPE_ALLOW_ACQUIRE=True to acquire)")
                return

            self._acquire_one_shot(scope, config)
        finally:
            scope.close()

    def _acquire_one_shot(self, scope: LabScopesLeCroyScopeAdapter, config) -> None:
        writer = HDF5RunWriter(config.output_path, config)
        writer.initialize(
            {scope.name: scope.metadata()},
            {scope.name: scope.time_array()},
            {"diagnostic": {"instrument": "scope", "scope_name": scope.name}},
        )
        scope.arm()
        scope_shot = scope.acquire(SCOPE_SHOT_NUM)
        writer.write_scope_shot(scope_shot, SCOPE_SHOT_NUM)
        writer.finalize([ShotResult(plan=ShotPlan(shot_num=SCOPE_SHOT_NUM), scope_shots=[scope_shot])])
        print(f"[scope check] acquisition PASS -> {config.output_path}")


# --------------------------------------------------------------------------- #
class MotionHardwareCheck(_HardwareCheckBase):
    """Connect to the real motion controller; optionally command one move."""

    run_flag = RUN_MOTION_CHECK
    label = "motion"

    def test_motion_reads_position_and_optionally_moves(self) -> None:
        config = load_run_config(EXPERIMENT_CONFIG_PATH, mode="grid", output_path=self.output_path)
        config = replace(config, scopes=[])
        dimension = _motion_dimension(config, MOTION_DIMENSION, MOTION_TARGET)
        adapter = _build_motion_adapter(config, dimension)

        print(f"\n[motion check] connecting {dimension.upper()} controller")
        try:
            adapter.connect()
            current = _read_probe_position(adapter.controller)
            print(f"[motion check] current probe position: {current}")

            writer = HDF5RunWriter(self.output_path, config)
            writer.initialize({}, {}, {"motion": adapter.metadata(), "diagnostic": {"instrument": "motion"}})

            if MOTION_TARGET is None:
                print("[motion check] read-only PASS (set MOTION_TARGET to move)")
                writer.finalize(
                    [ShotResult(plan=ShotPlan(shot_num=MOTION_SHOT_NUM), message="read-only motion check")]
                )
                return

            self.assertTrue(
                MOTION_ALLOW_MOVE,
                "MOTION_TARGET is set but MOTION_ALLOW_MOVE is False — refusing to move motors.",
            )
            self._command_move(adapter, writer)
        finally:
            adapter.close()

    def _command_move(self, adapter: LegacyMotorAdapter, writer: HDF5RunWriter) -> None:
        planned = PlannedPosition(coordinates=target_coordinates(MOTION_TARGET))
        achieved = adapter.move_to(planned)
        writer.write_position(MOTION_SHOT_NUM, achieved)
        writer.finalize([
            ShotResult(plan=ShotPlan(shot_num=MOTION_SHOT_NUM, position=planned), achieved_position=achieved)
        ])
        print(f"[motion check] move PASS achieved={achieved.coordinates} -> {self.output_path}")


# --------------------------------------------------------------------------- #
class CameraHardwareCheck(_HardwareCheckBase):
    """Connect to the Phantom camera; optionally record one .cine."""

    run_flag = RUN_CAMERA_CHECK
    label = "camera"

    def test_camera_configures_and_optionally_records(self) -> None:
        config = load_run_config(EXPERIMENT_CONFIG_PATH, mode="camera", output_path=self.output_path)
        config = replace(config, scopes=[])

        from drivers.phantom_recorder import PhantomRecorder

        adapter = PhantomCameraAdapter(
            PhantomRecorder(_camera_recorder_config(config)),
            experiment_name=CAMERA_EXPERIMENT_NAME,
            save_path=self.output_path.parent,
        )
        try:
            adapter.connect()
            print(f"\n[camera check] configured; metadata={adapter.metadata()}")

            writer = HDF5RunWriter(self.output_path, config)
            writer.initialize({}, {}, {"camera": adapter.metadata(), "diagnostic": {"instrument": "camera"}})

            if not CAMERA_ALLOW_RECORD:
                print("[camera check] configure-only PASS (set CAMERA_ALLOW_RECORD=True to record)")
                writer.finalize(
                    [ShotResult(plan=ShotPlan(shot_num=CAMERA_SHOT_NUM), message="configure-only camera check")]
                )
                return

            self._record_one(adapter, writer)
        finally:
            adapter.close()

    def _record_one(self, adapter: PhantomCameraAdapter, writer: HDF5RunWriter) -> None:
        adapter.arm(CAMERA_SHOT_NUM)
        camera_shot = adapter.complete(CAMERA_SHOT_NUM)
        writer.write_camera_shot(camera_shot)
        writer.finalize([ShotResult(plan=ShotPlan(shot_num=CAMERA_SHOT_NUM), camera_shot=camera_shot)])
        print(f"[camera check] record PASS file={camera_shot.file_name} -> {self.output_path}")


# --------------------------------------------------------------------------- #
class DataRunScopeHardware(_HardwareCheckBase):
    """Legacy Data_Run-style acquisition loop with real scopes, no motors."""

    run_flag = RUN_DATA_RUN_SCOPE_CHECK
    label = "data_run_scope"

    def setUp(self) -> None:
        super().setUp()
        if DATA_RUN_SCOPE_SHOTS < 1:
            self.fail("DATA_RUN_SCOPE_SHOTS must be at least 1")

    def test_data_run_scope_path_writes_hdf5(self) -> None:
        from acquisition.config import load_experiment_config
        from acquisition.scope_runner import MultiScopeAcquisition, single_shot_acquisition
        from acquisition import hdf5_writer

        config, raw_config_text = load_experiment_config(EXPERIMENT_CONFIG_PATH)
        if DATA_RUN_SCOPE_NAME:
            restrict_scope_config(config, DATA_RUN_SCOPE_NAME)

        print(f"\n[data-run-scope] output={self.output_path}")
        with MultiScopeAcquisition(self.output_path, config, raw_config_text) as msa:
            msa.initialize_hdf5_base()
            active_scopes = msa.initialize_scopes()
            self.assertTrue(active_scopes, "No valid data found from any scope")

            for shot_num in range(1, DATA_RUN_SCOPE_SHOTS + 1):
                print(f"[data-run-scope] shot {shot_num}/{DATA_RUN_SCOPE_SHOTS}")
                single_shot_acquisition(msa, active_scopes, shot_num)

            hdf5_writer.record_shot_count(self.output_path, msa.scope_ips, DATA_RUN_SCOPE_SHOTS)

        print(f"[data-run-scope] PASS -> {self.output_path}")


# --------------------------------------------------------------------------- #
class DataRunMotionHardware(_HardwareCheckBase):
    """Legacy Data_Run-style acquisition loop with real motors and fake scope data."""

    run_flag = RUN_DATA_RUN_MOTION_CHECK
    label = "data_run_motion"

    def setUp(self) -> None:
        super().setUp()
        if not MOTION_ALLOW_MOVE:
            self.skipTest("MOTION_ALLOW_MOVE is False; refusing to move motors.")
        if DATA_RUN_MOTION_MAX_SHOTS < 1:
            self.fail("DATA_RUN_MOTION_MAX_SHOTS must be at least 1")
        if DATA_RUN_FAKE_POINTS < 1:
            self.fail("DATA_RUN_FAKE_POINTS must be at least 1")

    def test_data_run_motion_path_writes_hdf5(self) -> None:
        from acquisition.config import load_experiment_config
        from acquisition import hdf5_writer
        from acquisition.scope_runner import handle_movement
        from motion import PositionManager

        config, raw_config_text = load_experiment_config(EXPERIMENT_CONFIG_PATH)
        ensure_fake_scope_config(config, DATA_RUN_FAKE_SCOPE)

        pos_manager = PositionManager(
            self.output_path,
            EXPERIMENT_CONFIG_PATH,
            num_duplicate_shots=int(config.get("nshots", "num_duplicate_shots", fallback=1)),
            num_run_repeats=int(config.get("nshots", "num_run_repeats", fallback=1)),
        )
        positions = pos_manager.initialize_position_hdf5()
        self.assertFalse(pos_manager.is_45deg, "45-degree motion is not supported by this diagnostic")

        total_shots = min(len(positions), DATA_RUN_MOTION_MAX_SHOTS)
        self._write_initial_hdf5(hdf5_writer, config, raw_config_text)

        mc = pos_manager.initialize_motor()
        self.assertIsNotNone(mc, "Motor controller did not initialize (check [motor_ips])")

        last_successful_shot = 0
        try:
            for shot_num in range(1, total_shots + 1):
                pos = positions[shot_num - 1]
                if not handle_movement(pos_manager, mc, shot_num, pos, self.output_path, [DATA_RUN_FAKE_SCOPE]):
                    print(f"[data-run-motion] skipping shot {shot_num} (movement failure)")
                    continue
                self._write_fake_shot(hdf5_writer, shot_num)
                self._record_position(pos_manager, mc, shot_num)
                last_successful_shot = shot_num
        finally:
            hdf5_writer.record_shot_count(self.output_path, [DATA_RUN_FAKE_SCOPE], last_successful_shot)

        self.assertGreater(last_successful_shot, 0, "no shots completed successfully")
        print(f"[data-run-motion] PASS ({last_successful_shot} shots) -> {self.output_path}")

    def _write_initial_hdf5(self, hdf5_writer, config, raw_config_text: str) -> None:
        hdf5_writer.write_experiment_metadata(
            self.output_path,
            description=config.get("experiment", "description", fallback="Data_Run motion hardware mock check"),
            source_code=hdf5_writer.read_source_files(),
            raw_config_text=raw_config_text,
            config=config,
            scope_names=[DATA_RUN_FAKE_SCOPE],
        )
        hdf5_writer.write_scope_metadata(
            self.output_path,
            DATA_RUN_FAKE_SCOPE,
            "Fake pause scope for Data_Run-style motor hardware check",
            "mock://pause",
            "PauseFakeScope",
        )
        hdf5_writer.write_time_array(
            self.output_path, DATA_RUN_FAKE_SCOPE, fake_time_array(DATA_RUN_FAKE_POINTS), 0
        )

    def _write_fake_shot(self, hdf5_writer, shot_num: int) -> None:
        print(f"[data-run-motion] pausing {DATA_RUN_MOTION_PAUSE_S:.3f}s")
        time.sleep(DATA_RUN_MOTION_PAUSE_S)
        payload = fake_scope_payload(DATA_RUN_FAKE_SCOPE, DATA_RUN_FAKE_CHANNEL, DATA_RUN_FAKE_POINTS, shot_num)
        hdf5_writer.write_shot_data(
            self.output_path,
            payload,
            shot_num,
            {(DATA_RUN_FAKE_SCOPE, DATA_RUN_FAKE_CHANNEL): "Fake delayed scope data; motors moved for this shot"},
        )

    def _record_position(self, pos_manager, mc, shot_num: int) -> None:
        if pos_manager.nz is None:
            xpos, ypos = mc.probe_positions
            pos_manager.update_position_hdf5(shot_num, {"x": xpos, "y": ypos, "z": None})
        else:
            xpos, ypos, zpos = mc.probe_positions
            pos_manager.update_position_hdf5(shot_num, {"x": xpos, "y": ypos, "z": zpos})


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
def _select_scope(config, requested):
    if not config.scopes:
        raise RuntimeError("No scopes found in [scope_ips].")
    if requested is None:
        return config.scopes[0]
    needle = requested.lower()
    for scope in config.scopes:
        if scope.name.lower() == needle:
            return scope
    available = ", ".join(scope.name for scope in config.scopes)
    raise RuntimeError(f"Scope {requested!r} not found. Available: {available}")


def _motion_dimension(config, requested: str, target) -> str:
    if requested != "auto":
        return requested
    if target is not None and len(target) == 3:
        return "xyz"
    if "z" in config.motion.motor_ips:
        return "xyz"
    return "xy"


def _build_motion_adapter(config, dimension: str) -> LegacyMotorAdapter:
    motor_ips = config.motion.motor_ips
    if dimension == "xy":
        _require_keys(motor_ips, ("x", "y"), "[motor_ips]")
        from motion.Motor_Control import Motor_Control_2D

        return LegacyMotorAdapter(Motor_Control_2D(motor_ips["x"], motor_ips["y"]))
    _require_keys(motor_ips, ("x", "y", "z"), "[motor_ips]")
    from motion.Motor_Control import Motor_Control_3D

    return LegacyMotorAdapter(Motor_Control_3D(motor_ips["x"], motor_ips["y"], motor_ips["z"]))


def _require_keys(values: dict, keys: tuple[str, ...], section: str) -> None:
    missing = [k for k in keys if k not in values]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)} in {section}.")


def _read_probe_position(controller):
    try:
        return controller.probe_positions
    except Exception as exc:
        print(f"[motion check] could not read probe position: {exc}")
        return None


def _camera_recorder_config(config) -> dict[str, object]:
    params = dict(config.camera.parameters)
    output_path = config.output_path
    return {
        "exposure_us": int(params.get("exposure_us", 30)),
        "fps": int(params.get("fps", 10000)),
        "pre_trigger_frames": int(params.get("pre_trigger_frames", -500)),
        "post_trigger_frames": int(params.get("post_trigger_frames", 1000)),
        "resolution": _resolution(params.get("resolution", (256, 256))),
        "hdf5_file_path": str(output_path),
        "save_path": str(output_path.parent),
    }


def _resolution(value) -> tuple[int, int]:
    if isinstance(value, tuple):
        return (int(value[0]), int(value[1]))
    text = str(value).replace("x", ",")
    first, second = (part.strip() for part in text.split(",", 1))
    return (int(first), int(second))


if __name__ == "__main__":
    unittest.main()
