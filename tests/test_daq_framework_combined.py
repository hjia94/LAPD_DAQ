"""End-to-end DAQ framework acquisition check.

Each instrument can be turned OFF, run with a FAKE device, or run against REAL
hardware. Configure the MODE constants below and run:

    pytest tests/test_daq_framework_combined.py -v -s

REAL-hardware modes use the connection info set at the top of the file. If the
hardware is unreachable the test fails with a clear message rather than
silently downgrading. The test runs one acquisition through
`AcquisitionRun.execute()` regardless of the device mix, then re-opens the
resulting HDF5 file and verifies its structure.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from lapd_daq.config import load_run_config
from lapd_daq.devices.fakes import (
    FakeCameraDevice,
    FakeMotionDevice,
    FakeScopeDevice,
    FakeTriggerDevice,
)
from lapd_daq.engine import AcquisitionDevices, AcquisitionRun

# --------------------------------------------------------------------------- #
# Instrument modes: "off", "fake", or "real".
# Scope must be "fake" or "real" — the engine requires at least one scope.
# --------------------------------------------------------------------------- #
SCOPE_MODE = "fake"
MOTION_MODE = "fake"
CAMERA_MODE = "off"
TRIGGER_MODE = "fake"

# Real-hardware connection info (used only when the matching MODE is "real").
REAL_SCOPE_IP = "192.168.1.100"
REAL_SCOPE_TIMEOUT_S = 30.0
REAL_MOTOR_IPS = {"x": "192.168.1.10", "y": "192.168.1.11"}  # add "z" for 3D
REAL_TRIGGER_GPIO_PIN = 17

# Grid / acquisition parameters
NUM_DUPLICATE_SHOTS = 2
GRID_NX = 2
GRID_NY = 2
SCOPE_NAME = "mockscope"
SCOPE_CHANNELS = ("C1", "C2")
SCOPE_POINTS = 12

# --------------------------------------------------------------------------- #
VALID_MODES = ("off", "fake", "real")
SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class RunPlan:
    """Resolved run configuration derived from the top-of-file MODE constants."""

    scope_mode: str
    motion_mode: str
    camera_mode: str
    trigger_mode: str

    def __post_init__(self) -> None:
        for label, value in (
            ("SCOPE_MODE", self.scope_mode),
            ("MOTION_MODE", self.motion_mode),
            ("CAMERA_MODE", self.camera_mode),
            ("TRIGGER_MODE", self.trigger_mode),
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

    @property
    def expected_scope_points(self) -> int | None:
        """Sample count is only predictable for the fake scope."""
        return SCOPE_POINTS if self.scope_mode == "fake" else None

    def summary(self) -> str:
        return (
            f"scope={self.scope_mode} motion={self.motion_mode} "
            f"camera={self.camera_mode} trigger={self.trigger_mode} "
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
    if plan.motion_mode != "off":
        sections.append(_ini_section("position", {
            "nx": str(GRID_NX),
            "ny": str(GRID_NY),
            "xmin": "-1", "xmax": "1",
            "ymin": "-2", "ymax": "2",
        }))
        sections.append(_ini_section("motor_ips", REAL_MOTOR_IPS))
    if plan.scope_mode != "off":
        scope_ip = REAL_SCOPE_IP if plan.scope_mode == "real" else "127.0.0.1"
        sections.append(_ini_section("scopes", {SCOPE_NAME: "LeCroy scope under test"}))
        sections.append(_ini_section("channels", {
            f"{SCOPE_NAME}_{ch}": f"mock channel {ch}" for ch in SCOPE_CHANNELS
        }))
        sections.append(_ini_section("scope_ips", {SCOPE_NAME: scope_ip}))
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
# Device factories (real-hardware imports deferred to keep the file importable
# on machines without the hardware packages installed).
# --------------------------------------------------------------------------- #
def _build_scope(mode: str):
    if mode == "fake":
        return FakeScopeDevice(SCOPE_NAME, channels=SCOPE_CHANNELS, points=SCOPE_POINTS)
    if mode == "real":
        from lapd_daq.devices.lab_scopes import LabScopesLeCroyScopeAdapter

        return LabScopesLeCroyScopeAdapter(
            SCOPE_NAME,
            REAL_SCOPE_IP,
            description="LeCroy scope under test",
            timeout=REAL_SCOPE_TIMEOUT_S,
        )
    return None


def _build_motion(mode: str):
    if mode == "fake":
        return FakeMotionDevice()
    if mode == "real":
        from lapd_daq.devices.legacy_motion import LegacyMotorAdapter

        controller = _build_real_motor_controller()
        return LegacyMotorAdapter(controller)
    return None


def _build_real_motor_controller():
    if "z" in REAL_MOTOR_IPS:
        from motion.Motor_Control import Motor_Control_3D

        return Motor_Control_3D(REAL_MOTOR_IPS["x"], REAL_MOTOR_IPS["y"], REAL_MOTOR_IPS["z"])
    from motion.Motor_Control import Motor_Control_2D

    return Motor_Control_2D(REAL_MOTOR_IPS["x"], REAL_MOTOR_IPS["y"])


def _build_camera(mode: str):
    if mode == "fake":
        return FakeCameraDevice()
    if mode == "real":
        from drivers.phantom_recorder import PhantomRecorder
        from lapd_daq.devices.phantom import PhantomCameraAdapter

        recorder = PhantomRecorder({
            "exposure_us": 30,
            "fps": 10000,
            "pre_trigger_frames": -500,
            "post_trigger_frames": 1000,
            "resolution": (256, 256),
        })
        return PhantomCameraAdapter(recorder, experiment_name="combined_framework_check")
    return None


def _build_trigger(mode: str):
    if mode == "fake":
        return FakeTriggerDevice()
    if mode == "real":
        from lapd_daq.devices.pi_gpio import PiGPIOTriggerAdapter

        return PiGPIOTriggerAdapter(pin=REAL_TRIGGER_GPIO_PIN)
    return None


def _build_devices(plan: RunPlan) -> AcquisitionDevices:
    scope = _build_scope(plan.scope_mode)
    return AcquisitionDevices(
        scopes=[scope] if scope is not None else [],
        motion=_build_motion(plan.motion_mode),
        camera=_build_camera(plan.camera_mode),
        trigger=_build_trigger(plan.trigger_mode),
    )


# --------------------------------------------------------------------------- #
class CombinedFrameworkAcquisitionTest(unittest.TestCase):
    """End-to-end acquisition + HDF5 structural verification."""

    def setUp(self) -> None:
        self.plan = RunPlan(SCOPE_MODE, MOTION_MODE, CAMERA_MODE, TRIGGER_MODE)
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.config_path = tmp_path / "experiment_config.txt"
        self.output_path = tmp_path / "combined_check.hdf5"
        self.config_path.write_text(_build_config_text(self.plan), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_acquisition_runs_and_hdf5_structure_is_correct(self) -> None:
        if self.plan.scope_mode == "off":
            self.skipTest("Engine requires at least one scope; set SCOPE_MODE to 'fake' or 'real'.")

        print(f"\n[combined framework check] {self.plan.summary()}")
        print(f"[combined framework check] expected shots = {self.plan.expected_shots}")

        results = self._run_acquisition()

        self.assertEqual(len(results), self.plan.expected_shots)
        self.assertTrue(self.output_path.exists(), "HDF5 output file was not created")
        self._verify_hdf5()

        print(f"[combined framework check] PASS -> {self.output_path}")

    def _run_acquisition(self):
        config = load_run_config(
            self.config_path, mode=self.plan.run_mode, output_path=self.output_path
        )
        devices = _build_devices(self.plan)
        return AcquisitionRun(config, devices=devices).execute()

    def _verify_hdf5(self) -> None:
        with h5py.File(self.output_path, "r") as h5:
            self._check_top_level(h5)
            self._check_scope_group(h5)
            if self.plan.motion_mode != "off":
                self._check_positions(h5)
            self._check_run_status(h5)

    def _check_top_level(self, h5: h5py.File) -> None:
        self.assertEqual(h5.attrs.get("schema_version"), SCHEMA_VERSION)
        self.assertIn("Control/Run", h5, "Control/Run group missing")

    def _check_scope_group(self, h5: h5py.File) -> None:
        expected_shots = self.plan.expected_shots
        self.assertIn(SCOPE_NAME, h5, f"Scope group {SCOPE_NAME!r} missing")
        scope_group = h5[SCOPE_NAME]
        self.assertEqual(scope_group.attrs.get("shot_count"), expected_shots)

        first_shot = scope_group.get("shot_1")
        self.assertIsNotNone(first_shot, "first shot subgroup missing")
        self._check_scope_shot_channels(first_shot)
        self.assertIn(f"shot_{expected_shots}", scope_group)

    def _check_scope_shot_channels(self, shot_group: h5py.Group) -> None:
        channels = self._channels_to_check(shot_group)
        expected_points = self.plan.expected_scope_points
        for channel in channels:
            dataset = shot_group.get(f"{channel}_data")
            self.assertIsNotNone(dataset, f"{channel}_data missing in {shot_group.name}")
            self.assertGreater(dataset.shape[-1], 0, f"{channel}_data is empty")
            if expected_points is not None:
                self.assertEqual(dataset.shape[-1], expected_points)

    def _channels_to_check(self, shot_group: h5py.Group) -> list[str] | tuple[str, ...]:
        if self.plan.scope_mode == "fake":
            return SCOPE_CHANNELS
        return sorted(
            name.removesuffix("_data") for name in shot_group.keys() if name.endswith("_data")
        )

    def _check_positions(self, h5: h5py.File) -> None:
        positions = h5.get("Control/Positions/positions_array")
        self.assertIsNotNone(positions, "positions_array missing")
        np.testing.assert_array_equal(
            positions["shot_num"], np.arange(1, self.plan.expected_shots + 1)
        )

    def _check_run_status(self, h5: h5py.File) -> None:
        status = h5.get("Control/Run/shot_status")
        self.assertIsNotNone(status, "Control/Run/shot_status missing")
        self.assertEqual(len(status), self.plan.expected_shots)


if __name__ == "__main__":
    unittest.main()
