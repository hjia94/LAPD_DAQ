"""Compatibility and adapter tests for lapd_daq.

Bundles three orthogonal back-compat / boundary subjects that share no
production code but are all about *interoperating with legacy interfaces*:

  * PhantomCameraAdapter cine-file output naming
  * Data_Run_45deg sunset behavior (must refuse to run)
  * HDF5 files produced by the new engine remain readable by the old
    lab_scopes and pydaq readers, both for synthesised data and for files
    built from real TRC fixtures
"""

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import importlib
import numpy as np

from lapd_daq.config import load_run_config
from lapd_daq.devices.phantom import PhantomCameraAdapter
from lapd_daq.devices.fakes import FakeScopeDevice, TRCReplayScopeDevice
from lapd_daq.engine import AcquisitionDevices, AcquisitionRun

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lapd_daq_fixtures import CONFIG_TEXT


TRC_FIXTURE_DIR = Path(r"D:\data\raw data")
TRC_SOURCE_SHOTS = (0, 5)
TRC_CHANNELS = ("C1", "C2", "C3", "C4")


# --------------------------------------------------------------------------- #
class PhantomAdapterTests(unittest.TestCase):
    def test_phantom_adapter_saves_cine_to_configured_directory(self):
        class Recorder:
            def __init__(self, save_path):
                self.config = {"save_path": str(save_path)}
                self.saved_path = None

            def wait_for_recording_completion(self):
                return 123.0

            def save_cine(self, path):
                self.saved_path = path
                return object()

            def wait_for_save_completion(self, rec_cine):
                return None

            def cleanup(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(Path(tmp))
            adapter = PhantomCameraAdapter(recorder, experiment_name="run")

            shot = adapter.complete(7)

            self.assertEqual(recorder.saved_path, str(Path(tmp) / "run_shot007.cine"))
            self.assertEqual(shot.file_name, "run_shot007.cine")


# --------------------------------------------------------------------------- #
class DataRun45DegSunsetTests(unittest.TestCase):
    def test_45deg_entrypoint_reports_unsupported_without_old_run_call(self):
        module = importlib.import_module("Data_Run_45deg")

        self.assertIn("not migrated", module.UNSUPPORTED_MESSAGE)
        with self.assertRaises(SystemExit) as context:
            module.main()
        self.assertIn("pre-refactor hardware PC workflow", str(context.exception))


# --------------------------------------------------------------------------- #
class HDF5ReaderCompatibilityTests(unittest.TestCase):
    def test_old_lab_scopes_hdf5_reader_reads_mock_generated_file(self):
        from lab_scopes.io.lecroy_files import read_hdf5_scope_data

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            output_path = Path(tmp) / "mock.hdf5"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")
            config = load_run_config(
                config_path, mode="stationary", output_path=output_path,
            )
            devices = AcquisitionDevices(
                scopes=[FakeScopeDevice("mockscope", channels=("C1",), points=10)],
            )

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
            config = load_run_config(
                config_path, mode="stationary", output_path=output_path,
            )
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
                self.assertEqual(
                    sorted(result["mockscope"]["channels"].keys()),
                    list(TRC_CHANNELS),
                )
                self.assertEqual(
                    len(result["mockscope"]["time_array"]), len(hdf_voltage),
                )

                descriptions = read_scope_channel_descriptions(h5, "mockscope")
                self.assertEqual(descriptions["C1"], "interferometer channel 1")


if __name__ == "__main__":
    unittest.main()
