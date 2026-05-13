import tempfile
import unittest
from pathlib import Path
import sys

import h5py
import numpy as np

from lapd_daq.config import load_run_config
from lapd_daq.devices.fakes import (
    FakeCameraDevice,
    FakeMotionDevice,
    FakeScopeDevice,
    FakeTriggerDevice,
    TRCReplayScopeDevice,
)
from lapd_daq.engine import AcquisitionDevices, AcquisitionRun


CONFIG_TEXT = """
[nshots]
num_duplicate_shots = 2 # inline comments should be accepted
num_run_repeats = 1

[position]
nx = 2
ny = 2
xmin = -1
xmax = 1
ymin = -2
ymax = 2

[experiment]
description = Mock LAPD run

[scopes]
MockScope = Fake LeCroy scope

[channels]
MockScope_C1 = mock channel one
MockScope_C2 = mock channel two

[scope_ips]
MockScope = 127.0.0.1
"""

TRC_FIXTURE_DIR = Path(r"D:\data\raw data")
TRC_SOURCE_SHOTS = (0, 5)
TRC_CHANNELS = ("C1", "C2", "C3", "C4")


class LapdDaqFrameworkTests(unittest.TestCase):
    def test_config_loader_preserves_existing_ini_and_detects_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")

            config = load_run_config(config_path, mode="grid")

            self.assertEqual(config.num_duplicate_shots, 2)
            self.assertEqual(config.motion.kind, "xy_grid")
            self.assertEqual(config.scopes[0].name, "mockscope")
            self.assertIn("Mock LAPD run", config.experiment_description)

    def test_grid_shot_plan_uses_duplicates_and_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            output_path = Path(tmp) / "mock.hdf5"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")
            config = load_run_config(config_path, mode="grid", output_path=output_path)
            run = AcquisitionRun(config, devices=AcquisitionDevices(scopes=[FakeScopeDevice("mockscope")]))

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
            config = load_run_config(config_path, mode="stationary", output_path=output_path)
            run = AcquisitionRun(config, devices=AcquisitionDevices(scopes=[FakeScopeDevice("mockscope")]))

            plans = run.build_shot_plan()

            self.assertEqual(len(plans), 2)
            self.assertIsNone(plans[0].position)

    def test_dry_run_writes_hdf5_scope_positions_camera_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            output_path = Path(tmp) / "mock.hdf5"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")
            config = load_run_config(config_path, mode="camera", output_path=output_path)
            config = load_run_config(config_path, mode="grid", output_path=output_path)
            devices = AcquisitionDevices(
                scopes=[FakeScopeDevice("mockscope", points=12)],
                motion=FakeMotionDevice(),
                camera=FakeCameraDevice(),
                trigger=FakeTriggerDevice(),
            )

            results = AcquisitionRun(config, devices=devices).execute()

            self.assertEqual(len(results), 8)
            self.assertTrue(output_path.exists())
            with h5py.File(output_path, "r") as h5:
                self.assertEqual(h5.attrs["schema_version"], "0.1")
                self.assertEqual(h5["mockscope"].attrs["shot_count"], 8)
                self.assertIn("shot_1", h5["mockscope"])
                self.assertIn("C1_data", h5["mockscope"]["shot_1"])
                np.testing.assert_array_equal(h5["Control/Positions/positions_array"]["shot_num"], np.arange(1, 9))
                self.assertEqual(len(h5["Control/Run/shot_status"]), 8)

    def test_old_lab_scopes_hdf5_reader_reads_mock_generated_file(self):
        from lab_scopes.io.lecroy_files import read_hdf5_scope_data

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            output_path = Path(tmp) / "mock.hdf5"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")
            config = load_run_config(config_path, mode="stationary", output_path=output_path)
            devices = AcquisitionDevices(scopes=[FakeScopeDevice("mockscope", channels=("C1",), points=10)])

            AcquisitionRun(config, devices=devices).execute()

            with h5py.File(output_path, "r") as h5:
                voltage, dt, t0 = read_hdf5_scope_data(h5, "mockscope", "C1", 1)
                self.assertEqual(len(voltage), 10)
                self.assertAlmostEqual(dt, 0.001)
                self.assertAlmostEqual(t0, 0.002)

    @unittest.skipUnless(TRC_FIXTURE_DIR.exists(), r"TRC fixtures not found at D:\data\raw data")
    def test_trc_replay_scope_writes_hdf5_readable_by_old_pydaq_reader(self):
        missing = [
            TRC_FIXTURE_DIR / f"{channel}-interf-shot{source_shot:05d}.trc"
            for source_shot in TRC_SOURCE_SHOTS
            for channel in TRC_CHANNELS
            if not (TRC_FIXTURE_DIR / f"{channel}-interf-shot{source_shot:05d}.trc").exists()
        ]
        if missing:
            self.skipTest(f"Missing TRC fixture files: {missing[:3]}")

        data_analysis_root = Path(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
        sys.path.insert(0, str(data_analysis_root))
        sys.path.insert(0, str(data_analysis_root / "read"))
        from read.read_scope_data import (
            read_hdf5_all_scopes_channels,
            read_hdf5_scope_data,
            read_scope_channel_descriptions,
            read_trc_data_simplified,
        )

        config_text = """
[nshots]
num_duplicate_shots = 2
num_run_repeats = 1

[experiment]
description = TRC replay compatibility test

[scopes]
mockscope = TRC replay scope

[channels]
mockscope_C1 = interferometer channel 1
mockscope_C2 = interferometer channel 2
mockscope_C3 = interferometer channel 3
mockscope_C4 = interferometer channel 4

[scope_ips]
mockscope = file://D:/data/raw data
"""

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            output_path = Path(tmp) / "trc_replay.hdf5"
            config_path.write_text(config_text, encoding="utf-8")
            config = load_run_config(config_path, mode="stationary", output_path=output_path)
            devices = AcquisitionDevices(
                scopes=[
                    TRCReplayScopeDevice(
                        "mockscope",
                        TRC_FIXTURE_DIR,
                        source_shots=TRC_SOURCE_SHOTS,
                        channels=TRC_CHANNELS,
                    )
                ]
            )

            results = AcquisitionRun(config, devices=devices).execute()

            self.assertEqual([result.status for result in results], ["ok", "ok"])
            with h5py.File(output_path, "r") as h5:
                self.assertNotIn("Run", h5)
                self.assertIn("Control/Run/shot_status", h5)

                hdf_voltage, dt, t0 = read_hdf5_scope_data(h5, "mockscope", "C1", 1)
                trc_voltage, trc_time = read_trc_data_simplified(
                    str(TRC_FIXTURE_DIR / "C1-interf-shot00000.trc")
                )
                np.testing.assert_allclose(hdf_voltage, trc_voltage, rtol=0, atol=1e-12)
                self.assertEqual(len(hdf_voltage), len(trc_time))
                self.assertAlmostEqual(dt, trc_time[1] - trc_time[0])
                self.assertAlmostEqual(t0, trc_time[0])

                result = read_hdf5_all_scopes_channels(h5, 1)
                self.assertEqual(sorted(result.keys()), ["mockscope"])
                self.assertEqual(sorted(result["mockscope"]["channels"].keys()), list(TRC_CHANNELS))
                self.assertEqual(len(result["mockscope"]["time_array"]), len(hdf_voltage))

                descriptions = read_scope_channel_descriptions(h5, "mockscope")
                self.assertEqual(descriptions["C1"], "interferometer channel 1")


if __name__ == "__main__":
    unittest.main()
