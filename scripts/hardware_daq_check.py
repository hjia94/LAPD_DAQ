"""Hardware-only DAQ diagnostics for one instrument family at a time.

This script is intentionally outside the automated pytest suite. Run it only on
the hardware PC connected to the instrument you want to check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lapd_daq.config import RunConfig, load_run_config
from lapd_daq.devices.lab_scopes import LabScopesLeCroyScopeAdapter
from lapd_daq.devices.legacy_motion import LegacyMotorAdapter
from lapd_daq.devices.phantom import PhantomCameraAdapter
from lapd_daq.models import PlannedPosition, ShotPlan, ShotResult
from lapd_daq.storage.hdf5 import HDF5RunWriter


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.instrument == "scope":
        return _check_scope(args)
    if args.instrument == "motion":
        return _check_motion(args)
    if args.instrument == "camera":
        return _check_camera(args)
    if args.instrument == "data-run-scope":
        return _check_data_run_scope(args)
    if args.instrument == "data-run-motion":
        return _check_data_run_motion(args)
    parser.error(f"Unknown instrument: {args.instrument}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hardware_daq_check.py",
        description="Run isolated hardware diagnostics through the LAPD_DAQ adapter code.",
    )
    subparsers = parser.add_subparsers(dest="instrument", required=True)

    scope = subparsers.add_parser("scope", help="Connect to one LeCroy scope and optionally acquire one shot")
    scope.add_argument("--config", required=True, help="Path to experiment_config.txt")
    scope.add_argument("--scope", help="Scope name from [scope_ips]. Defaults to the first configured scope.")
    scope.add_argument("--output", help="Output HDF5 path for acquired scope data")
    scope.add_argument("--timeout", type=float, default=30.0, help="Scope connection timeout in seconds")
    scope.add_argument("--shot", type=int, default=1, help="Shot number to write if --acquire is used")
    scope.add_argument("--acquire", action="store_true", help="Acquire and write one shot. Without this, only initialize.")

    motion = subparsers.add_parser("motion", help="Connect to motion control and optionally move to one target")
    motion.add_argument("--config", required=True, help="Path to experiment_config.txt")
    motion.add_argument("--output", help="Output HDF5 path for achieved-position metadata")
    motion.add_argument("--dimension", choices=("auto", "xy", "xyz"), default="auto")
    motion.add_argument(
        "--move-to",
        help="Optional probe target as x,y or x,y,z. If omitted, the script only reads current position.",
    )
    motion.add_argument("--shot", type=int, default=1, help="Shot number for HDF5 position metadata")

    camera = subparsers.add_parser("camera", help="Connect to one Phantom camera and optionally record one shot")
    camera.add_argument("--config", required=True, help="Path to experiment_config.txt")
    camera.add_argument("--output", help="Output HDF5 path for camera metadata")
    camera.add_argument("--experiment-name", default="hardware_camera_check")
    camera.add_argument("--shot", type=int, default=1, help="Shot number for camera metadata")
    camera.add_argument(
        "--record",
        action="store_true",
        help="Arm, wait for trigger, save .cine, and write metadata. Without this, only configure.",
    )

    data_scope = subparsers.add_parser(
        "data-run-scope",
        help="Data_Run-style loop with real scope acquisition and no motor movement",
    )
    data_scope.add_argument("--config", required=True, help="Path to experiment_config.txt")
    data_scope.add_argument("--scope", help="Scope name from [scope_ips]. Defaults to all configured scopes.")
    data_scope.add_argument("--output", help="Output HDF5 path")
    data_scope.add_argument("--shots", type=int, default=1, help="Number of stationary scope shots")

    data_motion = subparsers.add_parser(
        "data-run-motion",
        help="Data_Run-style loop with real motor movement and fake delayed scope data",
    )
    data_motion.add_argument("--config", required=True, help="Path to experiment_config.txt")
    data_motion.add_argument("--output", help="Output HDF5 path")
    data_motion.add_argument("--max-shots", type=int, default=1, help="Maximum positions to move through")
    data_motion.add_argument("--pause", type=float, default=0.5, help="Seconds to pause instead of scope acquisition")
    data_motion.add_argument("--fake-scope", default="PauseScope", help="Fake scope group name written to HDF5")
    data_motion.add_argument("--fake-channel", default="C1", help="Fake channel name written to HDF5")
    data_motion.add_argument("--fake-points", type=int, default=16, help="Number of fake waveform samples")
    data_motion.add_argument(
        "--allow-motion",
        action="store_true",
        help="Required safety flag. Without it, motors are not commanded.",
    )

    return parser


def _check_scope(args) -> int:
    config = load_run_config(args.config, mode="stationary", output_path=_output_path(args.output, "scope"))
    scope_config = _select_scope(config, args.scope)
    config = replace(config, scopes=[scope_config])
    scope = LabScopesLeCroyScopeAdapter(
        scope_config.name,
        scope_config.ip_address,
        description=scope_config.description,
        timeout=args.timeout,
    )

    print(f"Connecting scope {scope_config.name} at {scope_config.ip_address}")
    try:
        scope.connect()
        scope.initialize()
        print(f"Scope initialized. Displayed time points: {len(scope.time_array())}")
        print(f"Metadata: {scope.metadata()}")

        if not args.acquire:
            print("Scope initialize-only check passed. Re-run with --acquire to write one shot.")
            return 0

        writer = HDF5RunWriter(config.output_path, config)
        writer.initialize(
            {scope.name: scope.metadata()},
            {scope.name: scope.time_array()},
            {"diagnostic": {"instrument": "scope", "scope_name": scope.name}},
        )
        scope.arm()
        scope_shot = scope.acquire(args.shot)
        writer.write_scope_shot(scope_shot, args.shot)
        writer.finalize([ShotResult(plan=ShotPlan(shot_num=args.shot), scope_shots=[scope_shot])])
        print(f"Scope acquisition check passed -> {config.output_path}")
        return 0
    finally:
        scope.close()


def _check_motion(args) -> int:
    config = load_run_config(args.config, mode="grid", output_path=_output_path(args.output, "motion"))
    config = replace(config, scopes=[])
    target = _parse_move_to(args.move_to) if args.move_to else None
    dimension = _motion_dimension(config, args.dimension, target)
    adapter = _motion_adapter(config, dimension)

    print(f"Connecting {dimension.upper()} motion controller")
    try:
        adapter.connect()
        current = _read_probe_position(adapter.controller)
        print(f"Current probe position: {current}")

        writer = HDF5RunWriter(config.output_path, config)
        writer.initialize({}, {}, {"motion": adapter.metadata(), "diagnostic": {"instrument": "motion"}})

        if target is None:
            print("Motion read-only check passed. Re-run with --move-to x,y[,z] to command one move.")
            writer.finalize([ShotResult(plan=ShotPlan(shot_num=args.shot), message="read-only motion check")])
            return 0

        planned = PlannedPosition(coordinates=_target_coordinates(target))
        achieved = adapter.move_to(planned)
        writer.write_position(args.shot, achieved)
        writer.finalize([
            ShotResult(plan=ShotPlan(shot_num=args.shot, position=planned), achieved_position=achieved)
        ])
        print(f"Motion move check passed. Achieved: {achieved.coordinates} -> {config.output_path}")
        return 0
    finally:
        adapter.close()


def _check_camera(args) -> int:
    output_path = _output_path(args.output, "camera")
    config = load_run_config(args.config, mode="camera", output_path=output_path)
    config = replace(config, scopes=[])
    camera_config = _camera_config(config)

    from drivers.phantom_recorder import PhantomRecorder

    adapter = PhantomCameraAdapter(
        PhantomRecorder(camera_config),
        experiment_name=args.experiment_name,
        save_path=output_path.parent,
    )
    try:
        adapter.connect()
        print(f"Camera configured. Metadata: {adapter.metadata()}")
        print(f"Cine save directory: {output_path.parent}")

        writer = HDF5RunWriter(output_path, config)
        writer.initialize({}, {}, {"camera": adapter.metadata(), "diagnostic": {"instrument": "camera"}})

        if not args.record:
            print("Camera configure-only check passed. Re-run with --record to wait for trigger and save one cine.")
            writer.finalize([ShotResult(plan=ShotPlan(shot_num=args.shot), message="configure-only camera check")])
            return 0

        adapter.arm(args.shot)
        camera_shot = adapter.complete(args.shot)
        writer.write_camera_shot(camera_shot)
        writer.finalize([ShotResult(plan=ShotPlan(shot_num=args.shot), camera_shot=camera_shot)])
        print(f"Camera record check passed: {camera_shot.file_name} -> {output_path}")
        return 0
    finally:
        adapter.close()


def _check_data_run_scope(args) -> int:
    from acquisition.config import load_experiment_config
    from acquisition.scope_runner import MultiScopeAcquisition, single_shot_acquisition
    from acquisition import hdf5_writer

    output_path = _output_path(args.output, "data_run_scope")
    config, raw_config_text = load_experiment_config(args.config)
    if args.scope:
        _restrict_scope_config(config, args.scope)
    if args.shots < 1:
        raise RuntimeError("--shots must be at least 1")

    print("Running Data_Run-style scope check with real scopes and no motor movement.")
    print(f"Output: {output_path}")
    with MultiScopeAcquisition(output_path, config, raw_config_text) as msa:
        msa.initialize_hdf5_base()
        active_scopes = msa.initialize_scopes()
        if not active_scopes:
            raise RuntimeError("No valid data found from any scope. Aborting diagnostic.")

        for shot_num in range(1, args.shots + 1):
            print(f"\n______Data_Run scope check shot {shot_num}/{args.shots}______")
            single_shot_acquisition(msa, active_scopes, shot_num)

        hdf5_writer.record_shot_count(output_path, msa.scope_ips, args.shots)
    print(f"Data_Run-style scope check passed -> {output_path}")
    return 0


def _check_data_run_motion(args) -> int:
    if not args.allow_motion:
        raise RuntimeError("Refusing to move motors without --allow-motion.")
    if args.max_shots < 1:
        raise RuntimeError("--max-shots must be at least 1")
    if args.fake_points < 1:
        raise RuntimeError("--fake-points must be at least 1")

    from acquisition.config import load_experiment_config
    from acquisition import hdf5_writer
    from acquisition.scope_runner import handle_movement
    from motion import PositionManager

    output_path = _output_path(args.output, "data_run_motion")
    config, raw_config_text = load_experiment_config(args.config)
    _ensure_fake_scope_config(config, args.fake_scope)
    num_duplicate_shots = int(config.get("nshots", "num_duplicate_shots", fallback=1))
    num_run_repeats = int(config.get("nshots", "num_run_repeats", fallback=1))

    print("Running Data_Run-style motion check with real motors and fake delayed scope data.")
    print(f"Output: {output_path}")
    pos_manager = PositionManager(
        output_path,
        args.config,
        num_duplicate_shots=num_duplicate_shots,
        num_run_repeats=num_run_repeats,
    )
    positions = pos_manager.initialize_position_hdf5()
    if pos_manager.is_45deg:
        raise RuntimeError("45-degree motion is not supported by this refactor diagnostic.")
    total_shots = min(len(positions), args.max_shots)

    hdf5_writer.write_experiment_metadata(
        output_path,
        description=config.get("experiment", "description", fallback="Data_Run motion hardware mock check"),
        source_code=hdf5_writer.read_source_files(),
        raw_config_text=raw_config_text,
        config=config,
        scope_names=[args.fake_scope],
    )
    hdf5_writer.write_scope_metadata(
        output_path,
        args.fake_scope,
        "Fake pause scope for Data_Run-style motor hardware check",
        "mock://pause",
        "PauseFakeScope",
    )
    hdf5_writer.write_time_array(output_path, args.fake_scope, _fake_time_array(args.fake_points), 0)

    mc = pos_manager.initialize_motor()
    if mc is None:
        raise RuntimeError("Motor controller did not initialize. Check [motor_ips].")

    shot_num = 0
    try:
        for shot_num in range(1, total_shots + 1):
            pos = positions[shot_num - 1]
            movement_success = handle_movement(pos_manager, mc, shot_num, pos, output_path, [args.fake_scope])
            if not movement_success:
                print(f"Skipping fake scope write for shot {shot_num} due to movement failure.")
                continue

            print(f"Pausing {args.pause:.3f}s instead of taking scope data...")
            time.sleep(args.pause)
            all_data = _fake_scope_payload(args.fake_scope, args.fake_channel, args.fake_points, shot_num)
            hdf5_writer.write_shot_data(
                output_path,
                all_data,
                shot_num,
                {(args.fake_scope, args.fake_channel): "Fake delayed scope data; motors moved for this shot"},
            )
            if pos_manager.nz is None:
                xpos, ypos = mc.probe_positions
                pos_manager.update_position_hdf5(shot_num, {"x": xpos, "y": ypos, "z": None})
            else:
                xpos, ypos, zpos = mc.probe_positions
                pos_manager.update_position_hdf5(shot_num, {"x": xpos, "y": ypos, "z": zpos})
    finally:
        hdf5_writer.record_shot_count(output_path, [args.fake_scope], shot_num)

    print(f"Data_Run-style motion check passed -> {output_path}")
    return 0


def _select_scope(config: RunConfig, requested: str | None):
    if not config.scopes:
        raise RuntimeError("No scopes found in [scope_ips].")
    if requested is None:
        return config.scopes[0]
    requested_lower = requested.lower()
    for scope in config.scopes:
        if scope.name.lower() == requested_lower:
            return scope
    available = ", ".join(scope.name for scope in config.scopes)
    raise RuntimeError(f"Scope {requested!r} not found. Available scopes: {available}")


def _restrict_scope_config(config, scope_name: str) -> None:
    if not config.has_section("scope_ips"):
        raise RuntimeError("No [scope_ips] section found.")
    selected = None
    for name, ip_address in config.items("scope_ips"):
        if name.lower() == scope_name.lower():
            selected = (name, ip_address)
            break
    if selected is None:
        available = ", ".join(name for name, _ in config.items("scope_ips"))
        raise RuntimeError(f"Scope {scope_name!r} not found. Available scopes: {available}")
    for name, _ip_address in list(config.items("scope_ips")):
        config.remove_option("scope_ips", name)
    config.set("scope_ips", selected[0], selected[1])


def _ensure_fake_scope_config(config, scope_name: str) -> None:
    for section in ("scope_ips", "scopes", "channels"):
        if not config.has_section(section):
            config.add_section(section)
    config.set("scope_ips", scope_name, "mock://pause")
    config.set("scopes", scope_name, "Fake pause scope for motor-only hardware check")
    if not config.has_option("channels", f"{scope_name}_C1"):
        config.set("channels", f"{scope_name}_C1", "Fake delayed scope data")


def _motion_adapter(config: RunConfig, dimension: str) -> LegacyMotorAdapter:
    motor_ips = config.motion.motor_ips
    if dimension == "xy":
        _require_keys(motor_ips, ("x", "y"), "[motor_ips]")
        from motion.Motor_Control import Motor_Control_2D

        return LegacyMotorAdapter(Motor_Control_2D(motor_ips["x"], motor_ips["y"]))
    _require_keys(motor_ips, ("x", "y", "z"), "[motor_ips]")
    from motion.Motor_Control import Motor_Control_3D

    return LegacyMotorAdapter(Motor_Control_3D(motor_ips["x"], motor_ips["y"], motor_ips["z"]))


def _motion_dimension(config: RunConfig, requested: str, target: tuple[float, ...] | None) -> str:
    if requested != "auto":
        return requested
    if target is not None and len(target) == 3:
        return "xyz"
    if "z" in config.motion.motor_ips:
        return "xyz"
    return "xy"


def _read_probe_position(controller):
    try:
        return controller.probe_positions
    except Exception as exc:
        print(f"Could not read current probe position: {exc}")
        return None


def _parse_move_to(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) not in (2, 3):
        raise argparse.ArgumentTypeError("--move-to must have two or three comma-separated values")
    return values


def _target_coordinates(target: tuple[float, ...]) -> dict[str, float]:
    coords = {"x": target[0], "y": target[1]}
    if len(target) == 3:
        coords["z"] = target[2]
    return coords


def _fake_scope_payload(scope_name: str, channel: str, points: int, shot_num: int):
    raw = np.arange(points, dtype=np.int16) + np.int16(shot_num)
    return {scope_name: ([channel], {channel: raw}, {channel: _fake_lecroy_header(points)})}


def _fake_time_array(points: int) -> np.ndarray:
    return np.linspace(0.0, 1.0e-6, points, endpoint=False)


def _fake_lecroy_header(points: int) -> bytes:
    try:
        from lab_scopes.lecroy import LeCroyHeader

        header = LeCroyHeader()
        return header.generate_test_data(NTimes=points)
    except Exception:
        return bytes(346)


def _camera_config(config: RunConfig) -> dict[str, object]:
    params = dict(config.camera.parameters)
    output_path = config.output_path or _output_path(None, "camera")
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
    first, second = [part.strip() for part in text.split(",", 1)]
    return (int(first), int(second))


def _output_path(path: str | None, instrument: str) -> Path:
    if path:
        return Path(path)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"hardware_{instrument}_check_{stamp}.hdf5")


def _require_keys(values: dict[str, str], keys: tuple[str, ...], section: str) -> None:
    missing = [key for key in keys if key not in values]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)} in {section}.")


if __name__ == "__main__":
    raise SystemExit(main())
