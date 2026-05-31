"""End-to-end DAQ framework acquisition check (fake devices only).

Drives ``lapd_daq.engine.AcquisitionRun.execute()`` through the full pipeline
with fake devices and verifies the resulting HDF5 structure. This is the only
test that exercises the engine's fake grid-mode ``execute()`` and its
``Control/Positions/positions_array`` write.

Real-hardware coverage lives elsewhere:
  * real scopes / HDF5 reader compat -> tests/test_lapd_daq_core.py
  * real bmotion runs -> tests/test_bmotion_hardware.py
  * real instruments -> tests/test_hardware_instruments.py

Run:

    pytest tests/test_daq_framework_combined.py -v -s
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import h5py

from lapd_daq.config import load_run_config
from lapd_daq.devices.fakes import (
    FakeCameraDevice,
    FakeMotionDevice,
    FakeScopeDevice,
    FakeTriggerDevice,
)
from lapd_daq.engine import AcquisitionDevices, AcquisitionRun

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hdf5_assertions import (
    assert_channel_datasets,
    assert_positions_array,
    assert_run_status,
    assert_scope_group,
)

# --------------------------------------------------------------------------- #
# Instrument modes: "off" or "fake". The scope must be "fake" — the engine
# requires at least one scope.
# --------------------------------------------------------------------------- #
SCOPE_MODE = "fake"
MOTION_MODE = "fake"
CAMERA_MODE = "off"
RASPBERRY_PI_MODE = "fake"

# Grid / acquisition parameters
NUM_DUPLICATE_SHOTS = 2
GRID_NX = 2
GRID_NY = 2
SCOPE_NAME = "mockscope"
SCOPE_CHANNELS = ("C1", "C2")
SCOPE_POINTS = 12

# --------------------------------------------------------------------------- #
VALID_MODES = ("off", "fake")
SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class RunPlan:
    """Resolved run configuration derived from the top-of-file MODE constants."""

    scope_mode: str
    motion_mode: str
    camera_mode: str
    raspberry_pi_mode: str

    def __post_init__(self) -> None:
        for label, value in (
            ("SCOPE_MODE", self.scope_mode),
            ("MOTION_MODE", self.motion_mode),
            ("CAMERA_MODE", self.camera_mode),
            ("RASPBERRY_PI_MODE", self.raspberry_pi_mode),
        ):
            if value not in VALID_MODES:
                raise ValueError(f"{label}={value!r} must be one of {VALID_MODES}")

    @property
    def run_mode(self) -> str:
        if self.motion_mode != "off":
            return "grid"
        if self.camera_mode != "off":
            return "camera"
        return "stationary"

    @property
    def expected_shots(self) -> int:
        if self.run_mode == "grid":
            return GRID_NX * GRID_NY * NUM_DUPLICATE_SHOTS
        return NUM_DUPLICATE_SHOTS

    def summary(self) -> str:
        return (
            f"scope={self.scope_mode} motion={self.motion_mode} "
            f"camera={self.camera_mode} raspberry_pi={self.raspberry_pi_mode} "
            f"run_mode={self.run_mode}"
        )


# --------------------------------------------------------------------------- #
# Config text generation
# --------------------------------------------------------------------------- #
def _build_config_text(plan: RunPlan) -> str:
    sections: list[str] = [
        _ini_section("nshots", {
            "num_duplicate_shots": str(NUM_DUPLICATE_SHOTS),
            "num_run_repeats": "1",
        }),
        _ini_section("experiment", {"description": "Combined framework check"}),
    ]
    if plan.motion_mode == "fake":
        sections.append(_ini_section("position", {
            "nx": str(GRID_NX),
            "ny": str(GRID_NY),
            "xmin": "-1", "xmax": "1",
            "ymin": "-2", "ymax": "2",
        }))
    if plan.scope_mode != "off":
        sections.append(_ini_section("scopes", {SCOPE_NAME: "LeCroy scope under test"}))
        sections.append(_ini_section("channels", {
            f"{SCOPE_NAME}_{ch}": f"mock channel {ch}" for ch in SCOPE_CHANNELS
        }))
        sections.append(_ini_section("scope_ips", {SCOPE_NAME: "127.0.0.1"}))
    if plan.camera_mode != "off":
        sections.append(_ini_section("camera_config", {
            "exposure_us": "40",
            "fps": "1000",
        }))
    return "\n".join(sections) + "\n"


def _ini_section(name: str, items: dict[str, str]) -> str:
    body = "\n".join(f"{key} = {value}" for key, value in items.items())
    return f"[{name}]\n{body}\n"


# --------------------------------------------------------------------------- #
# Fake device factories
# --------------------------------------------------------------------------- #
def _build_scope(mode: str):
    if mode == "fake":
        return FakeScopeDevice(SCOPE_NAME, channels=SCOPE_CHANNELS, points=SCOPE_POINTS)
    return None


def _build_motion(mode: str):
    if mode == "fake":
        return FakeMotionDevice()
    return None


def _build_camera(mode: str):
    if mode == "fake":
        return FakeCameraDevice()
    return None


def _build_raspberry_pi(mode: str):
    if mode == "fake":
        return FakeTriggerDevice()
    return None


def _build_devices(plan: RunPlan) -> AcquisitionDevices:
    scope = _build_scope(plan.scope_mode)
    return AcquisitionDevices(
        scopes=[scope] if scope is not None else [],
        motion=_build_motion(plan.motion_mode),
        camera=_build_camera(plan.camera_mode),
        trigger=_build_raspberry_pi(plan.raspberry_pi_mode),
    )


# --------------------------------------------------------------------------- #
class CombinedFrameworkAcquisitionTest(unittest.TestCase):
    """End-to-end fake acquisition + HDF5 structural verification."""

    def setUp(self) -> None:
        self.plan = RunPlan(SCOPE_MODE, MOTION_MODE, CAMERA_MODE, RASPBERRY_PI_MODE)
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.config_path = tmp_path / "experiment_config.txt"
        self.output_path = tmp_path / "combined_check.hdf5"
        self.config_path.write_text(_build_config_text(self.plan), encoding="utf-8")

    def tearDown(self) -> None:
        try:
            self._tmp.cleanup()
        except (PermissionError, OSError):
            # HDF5 file on Windows may still be locked briefly; non-fatal.
            pass

    def test_acquisition_runs_and_hdf5_structure_is_correct(self) -> None:
        print(f"\n[combined framework check] {self.plan.summary()}")
        print(f"[combined framework check] expected shots = {self.plan.expected_shots}")

        results = self._run_engine_acquisition()
        self.assertEqual(len(results), self.plan.expected_shots)
        self.assertTrue(self.output_path.exists(), "HDF5 output file was not created")
        self._verify_hdf5_engine()

        print(f"[combined framework check] PASS -> {self.output_path}")

    def _run_engine_acquisition(self):
        config = load_run_config(
            self.config_path, mode=self.plan.run_mode, output_path=self.output_path
        )
        devices = _build_devices(self.plan)
        return AcquisitionRun(config, devices=devices).execute()

    def _verify_hdf5_engine(self) -> None:
        with h5py.File(self.output_path, "r") as h5:
            self.assertEqual(h5.attrs.get("schema_version"), SCHEMA_VERSION)
            self.assertIn("Control/Run", h5, "Control/Run group missing")
            assert_scope_group(self, h5, SCOPE_NAME, self.plan.expected_shots,
                               SCOPE_CHANNELS, points=SCOPE_POINTS)
            if self.plan.motion_mode != "off":
                assert_positions_array(self, h5, self.plan.expected_shots)
            assert_run_status(self, h5, self.plan.expected_shots)


if __name__ == "__main__":
    unittest.main()
