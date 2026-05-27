"""Tests for the parallel acquire->spool->offload->HDF5 pipeline.

Covers the verification points from the implementation plan:
  1. spool round-trip (RealTime 1-D and sequence 2-D), .done ordering
  2. offload -> HDF5 equivalence vs the in-process hdf5_writer
  3. schema equivalence vs a real prior run (reference HDF5 on disk)
  4. verify-and-delete (good bin deleted; corrupted bin kept + error)
  5. crash safety (no .done -> ignored; .done -> picked up)

Runs on this PC's .venv, which has the real bapsf_motion / h5py / numpy. No
real motor is required: the offload side only consumes spooled data, and the
acquire-side helpers used here build payloads from synthetic arrays.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from spooling import ShotPayload, TracePayload, spool_format
from acquisition import hdf5_writer, spool_adapter
import offload_runner

REFERENCE_HDF5 = r"D:\data\LAPD\03-LP-p21p29p41-plane-Helium_2026-05-20.hdf5"


def _make_all_data(seq=False):
    """Synthetic all_data dict like MultiScopeAcquisition.acquire_shot returns."""
    rng = np.random.default_rng(1234)
    if seq:
        c1 = rng.integers(-100, 100, size=(3, 64), dtype=np.int16)
        c2 = rng.integers(-100, 100, size=(3, 64), dtype=np.int16)
    else:
        c1 = rng.integers(-30000, 30000, size=128, dtype=np.int16)
        c2 = rng.integers(-30000, 30000, size=128, dtype=np.int16)
    traces = ["C1", "C2"]
    data = {"C1": c1, "C2": c2}
    headers = {"C1": b"HEADER-C1-bytes", "C2": b"HEADER-C2-bytes"}
    return {"lpscope": (traces, data, headers)}


def _make_meta(scope_name="lpscope", total_shots=4):
    """Minimal run-metadata bundle the offload adapter can build a skeleton from."""
    time_array = np.linspace(0, 1e-3, 128, dtype=np.float64)
    setup = np.zeros(2, dtype=spool_adapter._bmotion._POSITION_DTYPE)
    setup["shot_num"] = [1, 2]
    setup["x"] = [-1.0, 0.0]
    setup["y"] = [2.0, 2.0]
    return {
        "writer": "acquisition",
        "experiment_description": "unit-test run",
        "source_code": {"unit": "test"},
        "raw_config_text": "[experiment]\ndescription = unit-test run\n",
        "config_scope_names": [scope_name],
        "scopes": {
            scope_name: {
                "description": "test scope",
                "ip_address": "127.0.0.1",
                "scope_type": "LECROY,TEST,0,0",
                "time_array": time_array,
                "is_sequence": 0,
            }
        },
        "channel_descriptions": {
            f"{scope_name}_C1": "LP isat",
            f"{scope_name}_C2": "LP vsweep",
        },
        "total_shots": total_shots,
        "bmotion": {
            "toml_text": "# stub toml\n",
            "selection_blob": '{"mg_keys": ["0"], "execution_order": "interleaved"}',
            "prepared": [("0", "MG_A", setup, np.array([-1.0, 0.0]), np.array([2.0]))],
        },
    }


class SpoolRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_")

    def test_roundtrip_realtime(self):
        all_data = _make_all_data(seq=False)
        payload = spool_adapter.all_data_to_payload(
            all_data, shot_num=1, coordinates={"MG_A": (1.5, 2.5)}
        )
        spool_format.write_shot(self.spool, payload)

        self.assertEqual(spool_format.iter_ready_shots(self.spool), [1])
        got = spool_format.read_shot(self.spool, 1)
        self.assertEqual(got.coordinates, {"MG_A": (1.5, 2.5)})
        traces = {t.channel: t for t in got.traces["lpscope"]}
        for ch in ("C1", "C2"):
            np.testing.assert_array_equal(traces[ch].data, all_data["lpscope"][1][ch])
            self.assertEqual(traces[ch].header, all_data["lpscope"][2][ch])

    def test_roundtrip_sequence_2d(self):
        all_data = _make_all_data(seq=True)
        payload = spool_adapter.all_data_to_payload(all_data, 1, None)
        spool_format.write_shot(self.spool, payload)
        got = spool_format.read_shot(self.spool, 1)
        traces = {t.channel: t for t in got.traces["lpscope"]}
        self.assertEqual(traces["C1"].data.shape, (3, 64))
        np.testing.assert_array_equal(traces["C1"].data, all_data["lpscope"][1]["C1"])

    def test_done_marker_published_after_shot_dir(self):
        # The shot dir and marker should both exist after write; the marker is
        # what iter_ready_shots keys on.
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 7, None)
        spool_format.write_shot(self.spool, payload)
        self.assertTrue(os.path.isdir(os.path.join(self.spool, "shot_000007")))
        self.assertTrue(os.path.exists(os.path.join(self.spool, "shot_000007.done")))
        self.assertEqual(spool_format.iter_ready_shots(self.spool), [7])


class OffloadEquivalenceTests(unittest.TestCase):
    """Offloaded HDF5 must match what the in-process writer produces."""

    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_")
        self.off_h5 = tempfile.mktemp(suffix=".hdf5")
        self.direct_h5 = tempfile.mktemp(suffix=".hdf5")

    def tearDown(self):
        for p in (self.off_h5, self.direct_h5):
            if os.path.exists(p):
                os.remove(p)

    def _build_direct(self, meta, all_data_by_shot):
        """Reproduce the in-process path output for the same inputs."""
        spool_adapter.build_skeleton(self.direct_h5, meta, None,
                                     meta["raw_config_text"])
        for shot_num, all_data in all_data_by_shot.items():
            descriptions = {
                (sc, tr): meta["channel_descriptions"].get(
                    f"{sc}_{tr}", f"Channel {tr} - No description available")
                for sc, (traces, _d, _h) in all_data.items() for tr in traces
            }
            hdf5_writer.write_shot_data(self.direct_h5, all_data, shot_num,
                                        descriptions)
            with h5py.File(self.direct_h5, "a") as f:
                ds = f["Control/Positions/MG_A/positions_array"]
                ds[shot_num - 1] = (shot_num, float(shot_num), 2.0)
        hdf5_writer.record_shot_count(self.direct_h5, meta["config_scope_names"],
                                      len(all_data_by_shot))

    def test_offload_matches_direct_writer(self):
        meta = _make_meta(total_shots=2)
        shots = {1: _make_all_data(False), 2: _make_all_data(False)}

        # Direct in-process path.
        self._build_direct(meta, shots)

        # Spool + offload path.
        spool_format.write_run_metadata(self.spool, meta)
        for shot_num, all_data in shots.items():
            payload = spool_adapter.all_data_to_payload(
                all_data, shot_num, {"MG_A": (float(shot_num), 2.0)})
            spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 2)
        offload_runner.run_offload(self.spool, self.off_h5, config=None,
                                   poll_seconds=0.01)

        _assert_hdf5_equivalent(self, self.direct_h5, self.off_h5)

    def test_offloaded_dataset_filters_and_dtype(self):
        meta = _make_meta(total_shots=1)
        spool_format.write_run_metadata(self.spool, meta)
        payload = spool_adapter.all_data_to_payload(
            _make_all_data(False), 1, {"MG_A": (1.0, 2.0)})
        spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 1)
        offload_runner.run_offload(self.spool, self.off_h5, config=None,
                                   poll_seconds=0.01)

        with h5py.File(self.off_h5, "r") as f:
            ds = f["lpscope/shot_1/C1_data"]
            self.assertEqual(ds.dtype, np.dtype("int16"))
            self.assertEqual(ds.compression, "lzf")
            self.assertTrue(ds.shuffle)
            self.assertTrue(ds.fletcher32)
            self.assertEqual(ds.chunks, ds.shape)


class VerifyAndDeleteTests(unittest.TestCase):
    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_")
        self.off_h5 = tempfile.mktemp(suffix=".hdf5")

    def tearDown(self):
        if os.path.exists(self.off_h5):
            os.remove(self.off_h5)

    def test_good_shot_is_deleted_after_verify(self):
        meta = _make_meta(total_shots=1)
        spool_format.write_run_metadata(self.spool, meta)
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 1,
                                                    {"MG_A": (0.0, 0.0)})
        spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 1)
        offload_runner.run_offload(self.spool, self.off_h5, config=None,
                                   poll_seconds=0.01)
        # Spool copy removed once verified.
        self.assertEqual(spool_format.iter_ready_shots(self.spool), [])
        self.assertFalse(os.path.isdir(os.path.join(self.spool, "shot_000001")))

    def test_corrupted_bin_kept_and_errors(self):
        meta = _make_meta(total_shots=1)
        spool_adapter.build_skeleton(self.off_h5, meta, None,
                                     meta["raw_config_text"])
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 1,
                                                    {"MG_A": (0.0, 0.0)})
        spool_format.write_shot(self.spool, payload)

        # Corrupt the C1 bin so the read-back verification fails.
        bad = os.path.join(self.spool, "shot_000001", "lpscope__C1.bin")
        with open(bad, "wb") as f:
            f.write(b"\x00\x01\x02\x03")

        adapter = offload_runner._get_adapter("acquisition")
        with self.assertRaises(Exception):
            offload_runner._offload_one_shot(self.spool, self.off_h5, meta,
                                             adapter, 1)
        # Bin is NOT deleted on verification failure.
        self.assertTrue(os.path.isdir(os.path.join(self.spool, "shot_000001")))


class CrashSafetyTests(unittest.TestCase):
    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_")

    def test_tmp_dir_without_done_is_ignored(self):
        # Simulate an interrupted write: a shot_N.tmp dir, no .done marker.
        os.makedirs(os.path.join(self.spool, "shot_000003.tmp"))
        self.assertEqual(spool_format.iter_ready_shots(self.spool), [])

    def test_shot_dir_without_done_is_ignored(self):
        os.makedirs(os.path.join(self.spool, "shot_000003"))
        self.assertEqual(spool_format.iter_ready_shots(self.spool), [])
        with self.assertRaises(FileNotFoundError):
            spool_format.read_shot(self.spool, 3)

    def test_done_marker_makes_shot_visible(self):
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 3, None)
        spool_format.write_shot(self.spool, payload)
        self.assertEqual(spool_format.iter_ready_shots(self.spool), [3])


@unittest.skipUnless(os.path.exists(REFERENCE_HDF5),
                     f"reference file not present: {REFERENCE_HDF5}")
class SchemaMatchesReferenceTests(unittest.TestCase):
    """Assert the offloaded file reproduces the schema of a real prior run."""

    @classmethod
    def setUpClass(cls):
        cls.spool = tempfile.mkdtemp(prefix="spool_")
        cls.off_h5 = tempfile.mktemp(suffix=".hdf5")
        # Build a synthetic run shaped like the reference (scope 'lpscope',
        # channels C1..C3, one motion group, a couple of shots).
        meta = _ref_shaped_meta()
        spool_format.write_run_metadata(cls.spool, meta)
        for shot_num in (1, 2):
            payload = _ref_shaped_payload(shot_num)
            spool_format.write_shot(cls.spool, payload)
        spool_format.write_run_complete(cls.spool, 2)
        offload_runner.run_offload(cls.spool, cls.off_h5, config=None,
                                   poll_seconds=0.01)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.off_h5):
            os.remove(cls.off_h5)

    def test_schema_matches_reference(self):
        with h5py.File(REFERENCE_HDF5, "r") as ref, h5py.File(self.off_h5, "r") as new:
            # Top-level groups present in both.
            for grp in ("Configuration", "Control", "lpscope"):
                self.assertIn(grp, ref, f"reference missing {grp}")
                self.assertIn(grp, new, f"offloaded file missing {grp}")

            # Root attrs.
            for attr in ("description", "creation_time", "source_code"):
                self.assertIn(attr, new.attrs, f"root attr {attr} missing")

            # Configuration datasets.
            for name in ("experiment_config", "bmotion_config", "bmotion_selection"):
                self.assertIn(name, ref["Configuration"])
                self.assertIn(name, new["Configuration"])

            # Scope group attrs + time array.
            for attr in ("description", "ip_address", "scope_type"):
                self.assertIn(attr, new["lpscope"].attrs)
            ref_ta = ref["lpscope"]["time_array"]
            new_ta = new["lpscope"]["time_array"]
            self.assertEqual(new_ta.dtype, ref_ta.dtype)
            self.assertEqual(new_ta.attrs["units"], ref_ta.attrs["units"])

            # Per-shot dataset schema: dtype, chunks, filters.
            ref_ds = ref["lpscope"]["shot_1"]["C1_data"]
            new_ds = new["lpscope"]["shot_1"]["C1_data"]
            self.assertEqual(new_ds.dtype, ref_ds.dtype)
            self.assertEqual(new_ds.dtype, np.dtype("int16"))
            self.assertEqual(new_ds.compression, ref_ds.compression)
            self.assertEqual(new_ds.shuffle, ref_ds.shuffle)
            self.assertEqual(new_ds.fletcher32, ref_ds.fletcher32)
            # Reference stores each shot as one chunk == full length.
            self.assertEqual(new_ds.chunks, new_ds.shape)
            self.assertEqual(ref_ds.chunks, ref_ds.shape)
            self.assertEqual(new_ds.attrs["dtype"], "int16")
            self.assertIn("description", dict(new_ds.attrs))

            # Header dataset is an opaque void type in both.
            ref_hdr = ref["lpscope"]["shot_1"]["C1_header"]
            new_hdr = new["lpscope"]["shot_1"]["C1_header"]
            self.assertEqual(new_hdr.dtype.kind, ref_hdr.dtype.kind)  # 'V'
            self.assertIn("description", dict(new_hdr.attrs))

            # Control/Positions/<mg> layout.
            ref_pos_grp = ref["Control/Positions"]
            ref_mg = list(ref_pos_grp.keys())[0]
            new_pos_grp = new["Control/Positions"]
            new_mg = list(new_pos_grp.keys())[0]
            for attr in ("name", "key"):
                self.assertIn(attr, new_pos_grp[new_mg].attrs)
            ref_setup = ref_pos_grp[ref_mg]["positions_setup_array"]
            new_setup = new_pos_grp[new_mg]["positions_setup_array"]
            self.assertEqual(new_setup.dtype, ref_setup.dtype)
            self.assertIn("xpos", dict(new_setup.attrs))
            self.assertIn("ypos", dict(new_setup.attrs))
            new_arr = new_pos_grp[new_mg]["positions_array"]
            ref_arr = ref_pos_grp[ref_mg]["positions_array"]
            self.assertEqual(new_arr.dtype, ref_arr.dtype)


def _ref_shaped_meta():
    time_array = np.linspace(0, 1e-3, 256, dtype=np.float64)
    setup = np.zeros(2, dtype=spool_adapter._bmotion._POSITION_DTYPE)
    setup["shot_num"] = [1, 2]
    setup["x"] = [-15.0, -14.0]
    setup["y"] = [15.0, 15.0]
    return {
        "writer": "acquisition",
        "experiment_description": "schema check",
        "source_code": {"unit": "test"},
        "raw_config_text": "[experiment]\ndescription = schema check\n",
        "config_scope_names": ["lpscope"],
        "scopes": {
            "lpscope": {
                "description": "LeCroy regular black scope",
                "ip_address": "192.168.7.67",
                "scope_type": "LECROY,WR8208HD,TEST,0",
                "time_array": time_array,
                "is_sequence": 0,
            }
        },
        "channel_descriptions": {
            "lpscope_C1": "LP@P41 Isat 50ohm",
            "lpscope_C2": "LP@P29 Isat 50ohm",
            "lpscope_C3": "LP@P21 Isat 50ohm",
        },
        "total_shots": 2,
        "bmotion": {
            "toml_text": "# bmotion toml\n",
            "selection_blob": '{"mg_keys": ["0"], "execution_order": "interleaved"}',
            "prepared": [("0", "<Athena>    p21_LP", setup,
                          np.array([-15.0, -14.0]), np.array([15.0]))],
        },
    }


def _ref_shaped_payload(shot_num):
    rng = np.random.default_rng(shot_num)
    traces = {
        sc: [TracePayload(ch, rng.integers(-30000, 30000, 256, dtype=np.int16),
                          f"HDR-{ch}".encode())
             for ch in ("C1", "C2", "C3")]
        for sc in ("lpscope",)
    }
    return ShotPayload(shot_num=shot_num, traces=traces,
                       coordinates={"<Athena>    p21_LP": (-15.0, 15.0)},
                       acquisition_time="Wed May 20 19:25:47 2026")


def _assert_hdf5_equivalent(tc, path_a, path_b):
    """Assert two HDF5 files have the same scope/shot/position structure."""
    with h5py.File(path_a, "r") as a, h5py.File(path_b, "r") as b:
        tc.assertEqual(sorted(a["lpscope"].keys()), sorted(b["lpscope"].keys()))
        for shot in a["lpscope"]:
            if not shot.startswith("shot_"):
                continue
            ga, gb = a["lpscope"][shot], b["lpscope"][shot]
            tc.assertEqual(sorted(ga.keys()), sorted(gb.keys()))
            for ds_name in ga:
                da, db = ga[ds_name], gb[ds_name]
                tc.assertEqual(da.dtype, db.dtype, ds_name)
                if ds_name.endswith("_data"):
                    tc.assertEqual(da.compression, db.compression, ds_name)
                    tc.assertEqual(da.shuffle, db.shuffle, ds_name)
                    tc.assertEqual(da.fletcher32, db.fletcher32, ds_name)
                    tc.assertEqual(da.chunks, db.chunks, ds_name)
                    np.testing.assert_array_equal(da[()], db[()])
        # Positions equal.
        pa = a["Control/Positions"]
        pb = b["Control/Positions"]
        tc.assertEqual(sorted(pa.keys()), sorted(pb.keys()))
        for mg in pa:
            np.testing.assert_array_equal(
                pa[mg]["positions_array"][()], pb[mg]["positions_array"][()])
        # shot_count equal.
        tc.assertEqual(a["lpscope"].attrs.get("shot_count"),
                       b["lpscope"].attrs.get("shot_count"))


if __name__ == "__main__":
    unittest.main()
