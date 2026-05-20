"""End-to-end DAQ framework acquisition check.

Each instrument can be turned OFF, run with a FAKE device, or run against REAL
hardware. Configure the MODE constants below and run:

    pytest tests/test_daq_framework_combined.py -v -s

The fake/off path drives ``lapd_daq.engine.AcquisitionRun.execute()`` through
the full pipeline. The real-motion path mirrors a production data run: it
calls ``acquisition.run_acquisition_bmotion`` directly with a bapsf_motion
TOML config, matching ``tests/test_bmotion_hardware.py``. The two paths share
HDF5 structural verification.

REAL-motion runs require:
  * ``BMOTION_TOML_PATH`` to exist on disk
  * ``bapsf_motion`` and ``xarray`` installed
  * ``BMOTION_ALLOW_MOVE = True`` (destructive-action gate)
"""

from __future__ import annotations

import configparser
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
# When MOTION_MODE = "real" the run is routed through the bmotion path
# (acquisition.run_acquisition_bmotion); scope/camera/raspberry-pi flags only
# apply on the engine (fake/off motion) path.
# --------------------------------------------------------------------------- #
SCOPE_MODE = "fake"
MOTION_MODE = "fake"
CAMERA_MODE = "off"
RASPBERRY_PI_MODE = "fake"

# Real-hardware connection info (used only when the matching MODE is "real").
REAL_SCOPE_IP = "192.168.1.100"
REAL_SCOPE_TIMEOUT_S = 30.0
REAL_PI_HOST = "192.168.7.38"
REAL_PI_PORT = 54321

# Real-motion (bmotion) configuration. Mirrors tests/test_bmotion_hardware.py.
BMOTION_TOML_PATH = "bmotion_config.toml"
BMOTION_NSHOTS = 1
BMOTION_MOTION_GROUPS = "all"
BMOTION_DIRECTION = "forward"
BMOTION_EXECUTION_ORDER = "interleaved"
# Destructive-action gate: required to be True before the real-motion path
# will arm motors. Mirrors BMOTION_ALLOW_MOVE in test_bmotion_hardware.py.
BMOTION_ALLOW_MOVE = False

# Grid / acquisition parameters (used by the fake-motion engine path)
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
    def uses_real_bmotion(self) -> bool:
        return self.motion_mode == "real"

    @property
    def run_mode(self) -> str:
        if self.uses_real_bmotion:
            return "bmotion"
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
            f"camera={self.camera_mode} raspberry_pi={self.raspberry_pi_mode} "
            f"run_mode={self.run_mode}"
        )


# --------------------------------------------------------------------------- #
# Config text generation
# --------------------------------------------------------------------------- #
def _build_config_text(plan: RunPlan) -> str:
    sections: list[str] = [
        _ini_section("nshots", {
            "num_duplicate_shots": str(
                BMOTION_NSHOTS if plan.uses_real_bmotion else NUM_DUPLICATE_SHOTS
            ),
            "num_run_repeats": "1",
        }),
        _ini_section("experiment", {"description": "Combined framework check"}),
    ]
    if plan.uses_real_bmotion:
        sections.append(_ini_section("bmotion", {
            "motion_groups": BMOTION_MOTION_GROUPS,
            "direction": BMOTION_DIRECTION,
            "execution_order": BMOTION_EXECUTION_ORDER,
        }))
    elif plan.motion_mode == "fake":
        sections.append(_ini_section("position", {
            "nx": str(GRID_NX),
            "ny": str(GRID_NY),
            "xmin": "-1", "xmax": "1",
            "ymin": "-2", "ymax": "2",
        }))
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
    if plan.raspberry_pi_mode == "real":
        sections.append(_ini_section("raspberry_pi", {
            "pi_host": REAL_PI_HOST,
            "pi_port": str(REAL_PI_PORT),
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
    return None


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


def _build_raspberry_pi(mode: str):
    if mode == "fake":
        return FakeTriggerDevice()
    if mode == "real":
        from lapd_daq.devices.pi_gpio import PiGPIOTriggerAdapter
        from pi_gpio.pi_client import TriggerClient

        return PiGPIOTriggerAdapter(TriggerClient(REAL_PI_HOST, REAL_PI_PORT))
    return None


def _build_devices(plan: RunPlan) -> AcquisitionDevices:
    scope = _build_scope(plan.scope_mode)
    return AcquisitionDevices(
        scopes=[scope] if scope is not None else [],
        motion=_build_motion(plan.motion_mode),
        camera=_build_camera(plan.camera_mode),
        trigger=_build_raspberry_pi(plan.raspberry_pi_mode),
    )


def _have_bmotion_install() -> bool:
    try:
        import bapsf_motion  # noqa: F401
        import xarray  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
class CombinedFrameworkAcquisitionTest(unittest.TestCase):
    """End-to-end acquisition + HDF5 structural verification."""

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
        if self.plan.scope_mode == "off":
            self.skipTest("Engine requires at least one scope; set SCOPE_MODE to 'fake' or 'real'.")

        print(f"\n[combined framework check] {self.plan.summary()}")

        if self.plan.uses_real_bmotion:
            self._maybe_skip_real_bmotion()
            print(f"[combined framework check] driving bmotion path via "
                  f"run_acquisition_bmotion({BMOTION_TOML_PATH!r})")
            self._run_real_bmotion()
            self.assertTrue(self.output_path.exists(), "HDF5 output file was not created")
            self._verify_hdf5_bmotion()
        else:
            print(f"[combined framework check] expected shots = {self.plan.expected_shots}")
            results = self._run_engine_acquisition()
            self.assertEqual(len(results), self.plan.expected_shots)
            self.assertTrue(self.output_path.exists(), "HDF5 output file was not created")
            self._verify_hdf5_engine()

        print(f"[combined framework check] PASS -> {self.output_path}")

    # ----- engine (fake / off motion) path -------------------------------- #
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
            self._check_scope_group(h5)
            if self.plan.motion_mode != "off":
                self._check_positions(h5)
            self._check_run_status(h5)

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

    # ----- real bmotion path ---------------------------------------------- #
    def _maybe_skip_real_bmotion(self) -> None:
        if not BMOTION_ALLOW_MOVE:
            self.skipTest(
                "BMOTION_ALLOW_MOVE is False — refusing to command motors "
                "(set MOTION_MODE != 'real' for fake-motion engine path)"
            )
        if not _have_bmotion_install():
            self.skipTest("bapsf_motion / xarray not installed on this machine")
        if not Path(BMOTION_TOML_PATH).is_file():
            self.skipTest(
                f"Missing {BMOTION_TOML_PATH} in the current working directory"
            )
        # Real bmotion runs need real scopes (the legacy MultiScopeAcquisition
        # path connects directly to scope_ips). Don't pretend a fake scope works.
        if self.plan.scope_mode != "real":
            self.skipTest(
                "MOTION_MODE='real' (bmotion) requires SCOPE_MODE='real' — "
                "acquisition.run_acquisition_bmotion drives the legacy "
                "MultiScopeAcquisition which expects real scope IPs"
            )

    def _run_real_bmotion(self) -> None:
        # Late import so the skip checks above fire before bapsf_motion is touched.
        from acquisition import run_acquisition_bmotion

        # Write the experiment_config to the tempdir but rewrite scope_ips from
        # the on-disk config if the user wants to override. For now, the config
        # we built has scope_ips populated from REAL_SCOPE_IP.
        run_acquisition_bmotion(
            str(self.output_path),
            BMOTION_TOML_PATH,
            str(self.config_path),
        )

    def _verify_hdf5_bmotion(self) -> None:
        with h5py.File(self.output_path, "r") as h5:
            self.assertIn(
                "Configuration/bmotion_selection", h5,
                "Configuration/bmotion_selection missing — bmotion run did not "
                "record its selection blob"
            )
            self.assertIn(
                "Control/Positions", h5,
                "Control/Positions group missing"
            )
            mg_names = list(h5["Control/Positions"].keys())
            self.assertTrue(
                mg_names, "no Control/Positions/<mg> groups created"
            )
            # At least one motion group must have a populated positions_array.
            any_active = False
            for name in mg_names:
                positions = h5.get(f"Control/Positions/{name}/positions_array")
                if positions is not None and len(positions) > 0:
                    if np.any(positions["shot_num"] > 0):
                        any_active = True
                        break
            self.assertTrue(
                any_active,
                "no motion group recorded any active shots"
            )

            # Scope group must exist and have at least one shot.
            self.assertIn(
                SCOPE_NAME, h5,
                f"Scope group {SCOPE_NAME!r} missing from bmotion run"
            )
            scope_group = h5[SCOPE_NAME]
            shot_count = scope_group.attrs.get("shot_count", 0)
            self.assertGreater(
                int(shot_count), 0,
                f"Scope group {SCOPE_NAME!r} has shot_count=0"
            )


if __name__ == "__main__":
    unittest.main()
