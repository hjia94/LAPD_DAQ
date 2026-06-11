"""Hardware diagnostic tests for the motion controller.

Connects to the real motion controller through the lapd_daq adapter (and, for
the Data_Run-style check, through the legacy acquisition path with fake scope
data). Skipped by default so a normal run on a developer machine stays green;
opt in via environment variables so an enabled flag can never be committed:

    $env:LAPD_RUN_MOTION_CHECK = "1"           # adapter connect / read check
    $env:LAPD_RUN_DATA_RUN_MOTION_CHECK = "1"  # legacy Data_Run-style check
    $env:LAPD_MOTION_ALLOW_MOVE = "1"          # required before any motor move
    $env:LAPD_MOTION_TARGET = "0, 0"           # optional; omit for read-only
    pytest tests/test_motion_hw.py -v -s
"""

from __future__ import annotations

import time
import unittest
from dataclasses import replace

from lapd_daq.config import load_run_config
from lapd_daq.devices.legacy_motion import LegacyMotorAdapter
from lapd_daq.models import PlannedPosition, ShotPlan, ShotResult
from lapd_daq.storage.hdf5 import HDF5RunWriter

from _hardware_check_base import HardwareCheckBase, env_flag, env_str
from _hardware_check_helpers import (
    ensure_fake_scope_config,
    fake_scope_payload,
    fake_time_array,
    parse_move_to,
    target_coordinates,
)

# --------------------------------------------------------------------------- #
# Run flags — read from the environment; committed defaults are always safe.
# --------------------------------------------------------------------------- #
RUN_MOTION_CHECK = env_flag("LAPD_RUN_MOTION_CHECK")
RUN_DATA_RUN_MOTION_CHECK = env_flag("LAPD_RUN_DATA_RUN_MOTION_CHECK")

# Safety gate — no motor move is commanded unless this is set.
MOTION_ALLOW_MOVE = env_flag("LAPD_MOTION_ALLOW_MOVE")

# --------------------------------------------------------------------------- #
# Connection info / parameters. EXPERIMENT_CONFIG_PATH is resolved relative to
# the current working directory; pass an absolute path to avoid surprises.
# --------------------------------------------------------------------------- #
EXPERIMENT_CONFIG_PATH = env_str("LAPD_EXPERIMENT_CONFIG", "experiment_config.txt")

# Motion check
MOTION_DIMENSION = env_str("LAPD_MOTION_DIMENSION", "auto")  # "auto" | "xy" | "xyz"
# LAPD_MOTION_TARGET is "x, y" or "x, y, z" (e.g. "0, 0"); unset = read-only.
_MOTION_TARGET_RAW = env_str("LAPD_MOTION_TARGET")
MOTION_TARGET = parse_move_to(_MOTION_TARGET_RAW) if _MOTION_TARGET_RAW else None
MOTION_SHOT_NUM = 1

# Data_Run-style motion check (real motors, fake delayed scope)
DATA_RUN_MOTION_MAX_SHOTS = 1
DATA_RUN_MOTION_PAUSE_S = 0.5
DATA_RUN_FAKE_SCOPE = "PauseScope"
DATA_RUN_FAKE_CHANNEL = "C1"
DATA_RUN_FAKE_POINTS = 16
# --------------------------------------------------------------------------- #


class MotionHardwareCheck(HardwareCheckBase):
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
class DataRunMotionHardware(HardwareCheckBase):
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


if __name__ == "__main__":
    unittest.main()
