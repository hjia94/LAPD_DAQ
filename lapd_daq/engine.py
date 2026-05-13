"""Experiment run engine independent of direct hardware or future EPICS adapters."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lapd_daq.config import RunConfig
from lapd_daq.devices.direct import (
    LabScopesLeCroyAdapter,
    LegacyMotorAdapter,
    PhantomCameraAdapter,
    PiTriggerAdapter,
)
from lapd_daq.models import PlannedPosition, ShotPlan, ShotResult
from lapd_daq.storage.hdf5 import RunWriter, planned_positions_to_hdf5


@dataclass
class DeviceSet:
    scopes: list = field(default_factory=list)
    motion: object | None = None
    camera: object | None = None
    trigger: object | None = None


class AcquisitionRun:
    """Coordinates config, devices, shot plans, and storage."""

    def __init__(self, config: RunConfig, devices: DeviceSet | None = None):
        self.config = config
        self.devices = devices or build_direct_devices(config)
        self.output_path = config.output_path or default_output_path(config)
        self.writer = RunWriter(self.output_path, config)

    def build_shot_plan(self) -> list[ShotPlan]:
        if self.config.motion.enabled and self.config.motion.kind in {"xy_grid", "xyz_grid"}:
            return _grid_shot_plan(self.config)
        total = self.config.num_duplicate_shots * self.config.num_run_repeats
        return [
            ShotPlan(shot_num=i + 1, repeat_index=i // self.config.num_duplicate_shots,
                     duplicate_index=i % self.config.num_duplicate_shots)
            for i in range(total)
        ]

    def execute(self) -> list[ShotResult]:
        plans = self.build_shot_plan()
        results: list[ShotResult] = []
        self._connect_devices()
        try:
            self._initialize_scopes()
            scope_metadata = {scope.name: scope.metadata() for scope in self.devices.scopes}
            time_arrays = {scope.name: scope.time_array() for scope in self.devices.scopes}
            self.writer.initialize(scope_metadata, time_arrays, self._non_scope_device_metadata())
            planned_positions_to_hdf5(self.output_path, plans)

            for plan in plans:
                result = self._execute_shot(plan)
                results.append(result)
        finally:
            self.writer.finalize(results)
            self._close_devices()
        return results

    def _connect_devices(self) -> None:
        for scope in self.devices.scopes:
            scope.connect()
        for device in (self.devices.motion, self.devices.camera, self.devices.trigger):
            if device is not None:
                device.connect()

    def _initialize_scopes(self) -> None:
        if not self.devices.scopes:
            raise RuntimeError("No scopes configured for acquisition")
        for scope in self.devices.scopes:
            scope.initialize()

    def _execute_shot(self, plan: ShotPlan) -> ShotResult:
        result = ShotResult(plan=plan)
        try:
            if self.devices.motion is not None:
                result.achieved_position = self.devices.motion.move_to(plan.position)
                self.writer.write_position(plan.shot_num, result.achieved_position)

            for scope in self.devices.scopes:
                scope.arm()

            if self.devices.camera is not None:
                self.devices.camera.arm(plan.shot_num)

            if self.devices.trigger is not None:
                self.devices.trigger.trigger(plan.shot_num)

            for scope in self.devices.scopes:
                scope_shot = scope.acquire(plan.shot_num)
                result.scope_shots.append(scope_shot)
                self.writer.write_scope_shot(scope_shot, plan.shot_num)

            if self.devices.camera is not None:
                result.camera_shot = self.devices.camera.complete(plan.shot_num)
                self.writer.write_camera_shot(result.camera_shot)

        except Exception as exc:
            result.status = "skipped"
            result.message = str(exc)
            self.writer.mark_scopes_skipped(
                [scope.name for scope in self.devices.scopes],
                plan.shot_num,
                str(exc),
            )
        return result

    def _close_devices(self) -> None:
        for device in [*self.devices.scopes, self.devices.motion, self.devices.camera, self.devices.trigger]:
            if device is not None:
                device.close()

    def _non_scope_device_metadata(self) -> dict[str, dict[str, object]]:
        metadata = {}
        for name, device in (
            ("motion", self.devices.motion),
            ("camera", self.devices.camera),
            ("trigger", self.devices.trigger),
        ):
            if device is not None:
                metadata[name] = device.metadata()
        return metadata


def build_direct_devices(config: RunConfig) -> DeviceSet:
    scopes = [
        LabScopesLeCroyAdapter(scope.name, scope.ip_address, description=scope.description)
        for scope in config.scopes
    ]
    return DeviceSet(
        scopes=scopes,
        motion=_build_direct_motion(config),
        camera=_build_direct_camera(config),
        trigger=_build_direct_trigger(config),
    )


def default_output_path(config: RunConfig) -> Path:
    stamp = _dt.date.today().isoformat()
    return Path(f"lapd_daq_{config.mode}_{stamp}.hdf5")


def _build_direct_motion(config: RunConfig):
    if not config.motion.enabled:
        return None
    if config.motion.kind == "bmotion":
        raise NotImplementedError(
            "The new engine has a bmotion adapter boundary, but the existing "
            "interactive bmotion workflow has not been migrated yet. Use "
            "Data_Run_bmotion.py for hardware bmotion runs during this transition."
        )

    motor_ips = config.motion.motor_ips
    if config.motion.kind == "xy_grid":
        from motion.Motor_Control import Motor_Control_2D

        return LegacyMotorAdapter(Motor_Control_2D(motor_ips["x"], motor_ips["y"]))
    if config.motion.kind == "xyz_grid":
        from motion.Motor_Control import Motor_Control_3D

        return LegacyMotorAdapter(Motor_Control_3D(motor_ips["x"], motor_ips["y"], motor_ips["z"]))
    return None


def _build_direct_camera(config: RunConfig):
    if not config.camera.enabled:
        return None
    from drivers.phantom_recorder import PhantomRecorder

    output_path = config.output_path or default_output_path(config)
    params = dict(config.camera.parameters)
    camera_config = {
        "exposure_us": int(params.get("exposure_us", 30)),
        "fps": int(params.get("fps", 10000)),
        "pre_trigger_frames": int(params.get("pre_trigger_frames", -500)),
        "post_trigger_frames": int(params.get("post_trigger_frames", 1000)),
        "resolution": _resolution(params.get("resolution", (256, 256))),
        "hdf5_file_path": str(output_path),
        "save_path": str(output_path.parent),
    }
    return PhantomCameraAdapter(PhantomRecorder(camera_config), experiment_name=output_path.stem)


def _build_direct_trigger(config: RunConfig):
    if not config.trigger.enabled:
        return None
    params = config.trigger.parameters
    if "pi_host" not in params:
        return None
    from pi_gpio.pi_client import TriggerClient

    return PiTriggerAdapter(TriggerClient(str(params["pi_host"]), int(params.get("pi_port", 54321))))


def _resolution(value) -> tuple[int, int]:
    if isinstance(value, tuple):
        return (int(value[0]), int(value[1]))
    text = str(value).replace("x", ",")
    first, second = [part.strip() for part in text.split(",", 1)]
    return (int(first), int(second))


def _grid_shot_plan(config: RunConfig) -> list[ShotPlan]:
    params = config.motion.parameters
    nx = int(params["nx"])
    ny = int(params["ny"])
    xs = np.linspace(float(params["xmin"]), float(params["xmax"]), nx)
    ys = np.linspace(float(params["ymin"]), float(params["ymax"]), ny)
    zs = None
    if config.motion.kind == "xyz_grid":
        nz = int(params["nz"])
        zs = np.linspace(float(params["zmin"]), float(params["zmax"]), nz)

    plans = []
    shot_num = 1
    z_values = zs if zs is not None else [None]
    for repeat_index in range(config.num_run_repeats):
        for z in z_values:
            for y in ys:
                for x in xs:
                    for duplicate_index in range(config.num_duplicate_shots):
                        coords = {"x": float(x), "y": float(y)}
                        if z is not None:
                            coords["z"] = float(z)
                        plans.append(
                            ShotPlan(
                                shot_num=shot_num,
                                position=PlannedPosition(coordinates=coords),
                                repeat_index=repeat_index,
                                duplicate_index=duplicate_index,
                            )
                        )
                        shot_num += 1
    return plans
