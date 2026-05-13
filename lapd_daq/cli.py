"""Command line entrypoint for the new LAPD_DAQ framework."""

from __future__ import annotations

import argparse

from lapd_daq.config import load_run_config
from lapd_daq.devices.fakes import FakeCameraDevice, FakeMotionDevice, FakeScopeDevice, FakeTriggerDevice
from lapd_daq.engine import AcquisitionDevices, AcquisitionRun


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lapd-daq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an acquisition")
    run_parser.add_argument("--config", required=True, help="Existing INI experiment config")
    run_parser.add_argument(
        "--mode",
        choices=("stationary", "grid", "bmotion", "camera", "dropper"),
        default="stationary",
    )
    run_parser.add_argument("--output", help="Output HDF5 path")
    run_parser.add_argument("--dry-run", action="store_true", help="Use mock devices only")

    args = parser.parse_args(argv)
    if args.command == "run":
        config = load_run_config(args.config, mode=args.mode, output_path=args.output)
        devices = _fake_devices(config) if args.dry_run else None
        run = AcquisitionRun(config, devices=devices)
        results = run.execute()
        ok = sum(1 for result in results if result.status == "ok")
        print(f"Completed {ok}/{len(results)} shots -> {run.output_path}")
        return 0
    return 1


def _fake_devices(config) -> AcquisitionDevices:
    scopes = [FakeScopeDevice(scope.name) for scope in config.scopes]
    if not scopes:
        scopes = [FakeScopeDevice()]
    motion = FakeMotionDevice() if config.motion.enabled else None
    camera = FakeCameraDevice() if config.camera.enabled else None
    trigger = FakeTriggerDevice() if config.trigger.enabled else None
    return AcquisitionDevices(scopes=scopes, motion=motion, camera=camera, trigger=trigger)


if __name__ == "__main__":
    raise SystemExit(main())
