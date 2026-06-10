"""Core lapd_daq unit tests: config parsing, shot planning, and back-compat.

Three orthogonal but cohesive subjects:

  LoadRunConfigTests        -- experiment-config parser (grid detection, value
                               preservation, camera-mode gating)
  ShotPlanTests             -- AcquisitionRun.build_shot_plan() contract
  AcquisitionImportHygieneTests -- acquisition package public-API surface
  PhantomAdapterTests       -- PhantomCameraAdapter cine-file naming
  DataRun45DegSunsetTests   -- legacy entrypoint sunset behavior
  HDF5ReaderCompatibilityTests  -- new-engine HDF5 files remain readable by the
                                   old lab_scopes and pydaq readers

End-to-end acquisition is covered on the hardware PC by the ``*_hw.py`` files
(test_scope_hw, test_motion_hw, test_camera_hw) and by the
routine spooled+parallel DAQ plane run after changes; this module stops at the
config/planning/back-compat units a successful run does not exercise.
"""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lapd_daq.config import load_run_config
from lapd_daq.devices.fakes import FakeScopeDevice, TRCReplayScopeDevice
from lapd_daq.devices.phantom import PhantomCameraAdapter
from lapd_daq.engine import AcquisitionDevices, AcquisitionRun

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bmotion_stubs import guard_sys_modules
from _lapd_daq_fixtures import CAMERA_CONFIG_TEXT, CONFIG_TEXT, DESCRIPTION_TEXT


TRC_FIXTURE_DIR = Path(r"D:\data\raw data")
TRC_SOURCE_SHOTS = (0, 5)
TRC_CHANNELS = ("C1", "C2", "C3", "C4")


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #
class LoadRunConfigTests(unittest.TestCase):
    def test_config_loader_preserves_existing_ini_and_detects_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment_config.txt"
            config_path.write_text(CONFIG_TEXT, encoding="utf-8")
            (Path(tmp) / "description.txt").write_text(DESCRIPTION_TEXT, encoding="utf-8")

            config = load_run_config(config_path, mode="grid")

            self.assertEqual(config.num_duplicate_shots, 2)
            self.assertEqual(config.motion.kind, "xy_grid")
            self.assertEqual(config.scopes[0].name, "mockscope")
            # Description now read live from description.txt next to the config.
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


# --------------------------------------------------------------------------- #
# Shot planner
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Import hygiene
# --------------------------------------------------------------------------- #
class AcquisitionImportHygieneTests(unittest.TestCase):
    def test_acquisition_import_does_not_import_bmotion_or_scope_hardware(self):
        # Popping `acquisition` for a clean re-import rebuilds the package WITHOUT
        # the submodule attributes (spool_adapter, grid_spool_adapter, ...) that
        # other test modules' top-level imports and offload_engine._get_adapter
        # rely on; guard_sys_modules restores the originals so a sibling test
        # later in the same process (e.g. under `unittest discover`) isn't broken.
        with guard_sys_modules("acquisition"):
            sys.modules.pop("acquisition", None)
            sys.modules.pop("acquisition.bmotion", None)

            acquisition = importlib.import_module("acquisition")

            self.assertTrue(callable(acquisition.run_acquisition_spooled))
            # The lazy wrapper must be referenceable without importing bmotion.
            self.assertTrue(callable(acquisition.run_acquisition_bmotion_spooled))
            self.assertNotIn("acquisition.bmotion", sys.modules)


# --------------------------------------------------------------------------- #
# Phantom camera adapter
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
# Legacy entrypoint sunset
# --------------------------------------------------------------------------- #
class DataRun45DegSunsetTests(unittest.TestCase):
    def test_45deg_entrypoint_reports_unsupported_without_old_run_call(self):
        module = importlib.import_module("Data_Run_45deg")

        self.assertIn("not migrated", module.UNSUPPORTED_MESSAGE)
        with self.assertRaises(SystemExit) as context:
            module.main()
        self.assertIn("pre-refactor hardware PC workflow", str(context.exception))


# --------------------------------------------------------------------------- #
# HDF5 reader back-compatibility
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
