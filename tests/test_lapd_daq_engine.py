"""Unit tests for the `lapd_daq.engine.AcquisitionRun` shot planner and the
`acquisition` package import surface.

End-to-end acquisition runs (including all-fake dry-runs) belong in
test_daq_framework_combined.py; this file is restricted to the planner's
contract and to the public-API import hygiene of the legacy `acquisition`
package.
"""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

from lapd_daq.config import load_run_config
from lapd_daq.devices.fakes import FakeScopeDevice
from lapd_daq.engine import AcquisitionDevices, AcquisitionRun

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lapd_daq_fixtures import CONFIG_TEXT


class AcquisitionImportHygieneTests(unittest.TestCase):
    def test_acquisition_import_does_not_import_bmotion_or_scope_hardware(self):
        sys.modules.pop("acquisition", None)
        sys.modules.pop("acquisition.bmotion", None)

        acquisition = importlib.import_module("acquisition")

        self.assertTrue(callable(acquisition.run_acquisition))
        self.assertTrue(callable(acquisition.run_acquisition_bmotion))
        self.assertNotIn("acquisition.bmotion", sys.modules)


class ShotPlanTests(unittest.TestCase):
    def test_grid_shot_plan_uses_duplicates_and_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            output_path = Path(tmp) / "mock.hdf5"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")
            config = load_run_config(config_path, mode="grid", output_path=output_path)
            run = AcquisitionRun(
                config,
                devices=AcquisitionDevices(scopes=[FakeScopeDevice("mockscope")]),
            )

            plans = run.build_shot_plan()

            self.assertEqual(len(plans), 8)
            self.assertEqual(plans[0].position.coordinates, {"x": -1.0, "y": -2.0})
            self.assertEqual(plans[1].duplicate_index, 1)
            self.assertEqual(plans[-1].position.coordinates, {"x": 1.0, "y": 2.0})

    def test_stationary_mode_ignores_position_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            output_path = Path(tmp) / "mock.hdf5"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")
            config = load_run_config(
                config_path, mode="stationary", output_path=output_path,
            )
            run = AcquisitionRun(
                config,
                devices=AcquisitionDevices(scopes=[FakeScopeDevice("mockscope")]),
            )

            plans = run.build_shot_plan()

            self.assertEqual(len(plans), 2)
            self.assertIsNone(plans[0].position)


if __name__ == "__main__":
    unittest.main()
