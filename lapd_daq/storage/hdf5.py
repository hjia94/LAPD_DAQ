"""Central HDF5 writer for the EPICS-ready acquisition framework."""

from __future__ import annotations

import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np

from lapd_daq.config import RunConfig
from lapd_daq.models import AchievedPosition, CameraShot, ScopeShot, ShotPlan, ShotResult

SCHEMA_VERSION = "0.1"


class RunWriter:
    """Owns all HDF5 group/dataset names for a run."""

    def __init__(self, path: str | Path, config: RunConfig):
        self.path = Path(path)
        self.config = config

    def initialize(self, scope_metadata: dict[str, dict[str, object]],
                   time_arrays: dict[str, np.ndarray],
                   device_metadata: dict[str, dict[str, object]] | None = None) -> None:
        with h5py.File(self.path, "a") as h5:
            h5.attrs["description"] = self.config.experiment_description
            h5.attrs["creation_time"] = time.ctime()
            h5.attrs["schema_version"] = SCHEMA_VERSION
            h5.attrs["run_mode"] = self.config.mode
            h5.attrs["software_versions"] = str(_software_versions())

            config_group = h5.require_group("Configuration")
            _replace_dataset(config_group, "experiment_config", np.string_(self.config.raw_text))

            run_group = h5.require_group("Run")
            run_group.attrs["config_path"] = str(self.config.config_path)
            run_group.attrs["num_duplicate_shots"] = self.config.num_duplicate_shots
            run_group.attrs["num_run_repeats"] = self.config.num_run_repeats

            devices_group = run_group.require_group("Devices")
            for device_name, metadata in (device_metadata or {}).items():
                device_group = devices_group.require_group(device_name)
                for key, value in metadata.items():
                    device_group.attrs[key] = _hdf5_attr(value)
            for scope_name, metadata in scope_metadata.items():
                scope_group = h5.require_group(scope_name)
                device_group = devices_group.require_group(scope_name)
                for key, value in metadata.items():
                    scope_group.attrs[key] = _hdf5_attr(value)
                    device_group.attrs[key] = _hdf5_attr(value)
                time_array = time_arrays.get(scope_name)
                if time_array is not None and "time_array" not in scope_group:
                    ds = scope_group.create_dataset("time_array", data=time_array, dtype="float64")
                    ds.attrs["units"] = "seconds"
                    ds.attrs["description"] = "Time array for all channels"
                    ds.attrs["dtype"] = str(np.asarray(time_array).dtype)

    def write_scope_shot(self, scope_shot: ScopeShot, shot_num: int) -> None:
        with h5py.File(self.path, "a", libver="latest", rdcc_nbytes=0) as h5:
            scope_group = h5.require_group(scope_shot.scope_name)
            shot_group = scope_group.require_group(f"shot_{shot_num}")
            shot_group.attrs["acquisition_time"] = scope_shot.acquisition_time or time.ctime()

            if scope_shot.skipped:
                shot_group.attrs["skipped"] = True
                shot_group.attrs["skip_reason"] = scope_shot.skip_reason
                return

            for trace in scope_shot.traces:
                raw = np.asarray(trace.raw, dtype=np.int16)
                chunks = _chunks_for(raw)
                data_ds = shot_group.create_dataset(
                    f"{trace.channel}_data",
                    data=raw,
                    dtype="int16",
                    chunks=chunks,
                    compression="lzf",
                    shuffle=True,
                    fletcher32=True,
                )
                header_ds = shot_group.create_dataset(
                    f"{trace.channel}_header",
                    data=np.void(trace.header),
                )
                data_ds.attrs["description"] = self.config.channel_descriptions.get(
                    f"{scope_shot.scope_name}_{trace.channel}",
                    f"Channel {trace.channel} - No description available",
                )
                data_ds.attrs["dtype"] = "int16"
                header_ds.attrs["description"] = f"Binary header data for {trace.channel}"

    def write_position(self, shot_num: int, position: AchievedPosition | None) -> None:
        if position is None:
            return
        coords = position.coordinates
        with h5py.File(self.path, "a") as h5:
            pos_group = h5.require_group("Control").require_group("Positions")
            if "positions_array" not in pos_group:
                dtype = _position_dtype(coords)
                pos_group.create_dataset(
                    "positions_array",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dtype,
                    chunks=True,
                )
            ds = pos_group["positions_array"]
            _append_position(ds, shot_num, coords)

    def write_camera_shot(self, camera_shot: CameraShot | None) -> None:
        if camera_shot is None:
            return
        with h5py.File(self.path, "a") as h5:
            group = h5.require_group("Control").require_group("FastCam")
            if "shot number" not in group:
                group.create_dataset("shot number", shape=(0,), maxshape=(None,), dtype="i4", chunks=True)
                group.create_dataset(
                    "cine file name",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    chunks=True,
                )
                group.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype="f8", chunks=True)
            _append_1d(group["shot number"], camera_shot.shot_num)
            _append_1d(group["cine file name"], camera_shot.file_name)
            _append_1d(group["timestamp"], camera_shot.timestamp)

    def mark_scopes_skipped(self, scope_names: list[str], shot_num: int, reason: str) -> None:
        with h5py.File(self.path, "a") as h5:
            for scope_name in scope_names:
                shot_group = h5.require_group(scope_name).require_group(f"shot_{shot_num}")
                shot_group.attrs["skipped"] = True
                shot_group.attrs["skip_reason"] = reason
                shot_group.attrs["acquisition_time"] = time.ctime()

    def finalize(self, results: list[ShotResult]) -> None:
        with h5py.File(self.path, "a") as h5:
            run_group = h5.require_group("Run")
            dtype = np.dtype([
                ("shot_num", ">u4"),
                ("status", h5py.string_dtype(encoding="utf-8")),
                ("message", h5py.string_dtype(encoding="utf-8")),
            ])
            data = np.array(
                [(result.plan.shot_num, result.status, result.message) for result in results],
                dtype=dtype,
            )
            _replace_dataset(run_group, "shot_status", data)
            for scope in self.config.scopes:
                if scope.name in h5:
                    h5[scope.name].attrs["shot_count"] = len(results)


def planned_positions_to_hdf5(path: str | Path, plans: list[ShotPlan]) -> None:
    """Store planned positions when a motion scan is used."""

    rows = [plan for plan in plans if plan.position is not None]
    if not rows:
        return
    first = rows[0].position.coordinates
    dtype = _position_dtype(first)
    data = np.zeros((len(rows),), dtype=dtype)
    for index, plan in enumerate(rows):
        values = [plan.shot_num] + [float(plan.position.coordinates[name]) for name in first]
        data[index] = tuple(values)
    with h5py.File(path, "a") as h5:
        group = h5.require_group("Control").require_group("Positions")
        _replace_dataset(group, "positions_setup_array", data)


def _position_dtype(coords: dict[str, float]) -> np.dtype:
    fields = [("shot_num", ">u4")]
    for axis in coords:
        fields.append((axis, ">f4"))
    return np.dtype(fields)


def _append_position(ds, shot_num: int, coords: dict[str, float]) -> None:
    ds.resize((ds.shape[0] + 1,))
    values = [shot_num] + [float(coords[name]) for name in ds.dtype.names if name != "shot_num"]
    ds[-1] = tuple(values)


def _append_1d(ds, value) -> None:
    ds.resize((ds.shape[0] + 1,))
    ds[-1] = value


def _replace_dataset(group, name: str, data) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def _chunks_for(raw: np.ndarray):
    if raw.ndim > 1:
        return (1, min(raw.shape[1], 8 * 1024 * 1024))
    return (min(len(raw), 8 * 1024 * 1024),)


def _software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "lapd_daq": "0.1.0"}
    for package in ("lab-scopes", "h5py", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _hdf5_attr(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool, np.number)):
        return value
    return str(value)
