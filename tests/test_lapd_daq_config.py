"""Unit tests for `lapd_daq.config.load_run_config`.

Covers the experiment-config parser: grid-mode detection, value preservation
across modes, and the gating of [camera_config] outside camera modes.
End-to-end acquisition coverage lives in test_daq_framework_combined.py.
"""

import sys
import tempfile
import unittest
from pathlib import Path

from lapd_daq.config import load_run_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lapd_daq_fixtures import CAMERA_CONFIG_TEXT, CONFIG_TEXT


class LoadRunConfigTests(unittest.TestCase):
    def test_config_loader_preserves_existing_ini_and_detects_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")

            config = load_run_config(config_path, mode="grid")

            self.assertEqual(config.num_duplicate_shots, 2)
            self.assertEqual(config.motion.kind, "xy_grid")
            self.assertEqual(config.scopes[0].name, "mockscope")
            self.assertIn("Mock LAPD run", config.experiment_description)

    def test_camera_config_does_not_enable_camera_outside_camera_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            config_path.write_text(CAMERA_CONFIG_TEXT, encoding="utf-8")

            stationary = load_run_config(config_path, mode="stationary")
            camera = load_run_config(config_path, mode="camera")

            self.assertFalse(stationary.camera.enabled)
            self.assertEqual(stationary.camera.parameters["exposure_us"], 40)
            self.assertTrue(camera.camera.enabled)


if __name__ == "__main__":
    unittest.main()
