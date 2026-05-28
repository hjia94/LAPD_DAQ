"""Tests for the parallel acquire->spool->offload->HDF5 pipeline.

Architecture under test (acquire owns the HDF5 skeleton):
  * the ACQUIRE process creates the destination HDF5 and writes its full
    skeleton (experiment/scope metadata, time arrays, Control/Positions) using
    the same `main` writers as the in-process path, then spools only per-shot
    raw traces;
  * the OFFLOAD process reads the destination path verbatim from the slim spool
    run-metadata (`meta["hdf5_path"]`), fills each shot's datasets + the
    per-shot position row into the already-created file, verifies by read-back,
    and deletes the spooled copy.

Covers:
  1. spool round-trip (RealTime 1-D and sequence 2-D), .done ordering
  2. offload -> HDF5 equivalence vs the in-process hdf5_writer (completeness gate)
  3. schema equivalence vs a real prior run (reference HDF5 on disk)
  4. verify-and-delete (good bin deleted; corrupted bin kept + error)
  5. crash safety (no .done -> ignored; .done -> picked up)

Runs on this PC's .venv (real bapsf_motion / h5py / numpy). No motor required.
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
from acquisition import bmotion, hdf5_writer, spool_adapter
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


def _make_meta(scope_name="lpscope", hdf5_path=None):
    """The slim run-info bundle the offload reads (no skeleton material)."""
    return {
        "writer": "acquisition",
        "hdf5_path": hdf5_path,
        "config_scope_names": [scope_name],
        "channel_descriptions": {
            f"{scope_name}_C1": "LP isat",
            f"{scope_name}_C2": "LP vsweep",
        },
    }


def _build_bmotion_skeleton(hdf5_path, scope_name="lpscope", n_samples=128,
                            total_shots=4, mg_name="MG_A"):
    """Create the HDF5 skeleton exactly as the acquire process now does.

    Uses the same `main` writers acquire calls (write_experiment_metadata,
    write_scope_metadata, write_time_array, write_bmotion_position_groups), so a
    test offload fills a file that is byte-for-byte what acquire would create.
    Returns the channel descriptions map used to build the slim meta.
    """
    time_array = np.linspace(0, 1e-3, n_samples, dtype=np.float64)
    hdf5_writer.write_experiment_metadata(
        hdf5_path,
        description="unit-test run",
        source_code={"unit": "test"},
        raw_config_text="[experiment]\ndescription = unit-test run\n",
        config=None,
        scope_names=[scope_name],
    )
    hdf5_writer.write_scope_metadata(
        hdf5_path, scope_name=scope_name, description="test scope",
        ip_address="127.0.0.1", scope_type="LECROY,TEST,0,0",
    )
    hdf5_writer.write_time_array(hdf5_path, scope_name, time_array, 0)

    setup = np.zeros(2, dtype=bmotion._POSITION_DTYPE)
    setup["shot_num"] = [1, 2]
    setup["x"] = [-1.0, 0.0]
    setup["y"] = [2.0, 2.0]
    bmotion.write_bmotion_position_groups(
        hdf5_path,
        total_shots=total_shots,
        toml_text="# stub toml\n",
        selection_blob='{"mg_keys": ["0"], "execution_order": "interleaved"}',
        prepared=[("0", mg_name, setup, np.array([-1.0, 0.0]), np.array([2.0]))],
    )


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
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 7, None)
        spool_format.write_shot(self.spool, payload)
        self.assertTrue(os.path.isdir(os.path.join(self.spool, "shot_000007")))
        self.assertTrue(os.path.exists(os.path.join(self.spool, "shot_000007.done")))
        self.assertEqual(spool_format.iter_ready_shots(self.spool), [7])


class OffloadEquivalenceTests(unittest.TestCase):
    """Completeness gate: offloaded HDF5 must match the in-process writer output."""

    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_")
        self.off_h5 = tempfile.mktemp(suffix=".hdf5")
        self.direct_h5 = tempfile.mktemp(suffix=".hdf5")

    def tearDown(self):
        for p in (self.off_h5, self.direct_h5):
            if os.path.exists(p):
                os.remove(p)

    def _build_direct(self, all_data_by_shot, descriptions):
        """Reproduce the in-process path output for the same inputs."""
        _build_bmotion_skeleton(self.direct_h5, total_shots=len(all_data_by_shot))
        for shot_num, all_data in all_data_by_shot.items():
            hdf5_writer.write_shot_data(self.direct_h5, all_data, shot_num,
                                        descriptions)
            with h5py.File(self.direct_h5, "a") as f:
                ds = f["Control/Positions/MG_A/positions_array"]
                ds[shot_num - 1] = (shot_num, float(shot_num), 2.0)
        hdf5_writer.record_shot_count(self.direct_h5, ["lpscope"],
                                      len(all_data_by_shot))

    def test_offload_matches_direct_writer(self):
        shots = {1: _make_all_data(False), 2: _make_all_data(False)}
        descriptions = {("lpscope", "C1"): "LP isat", ("lpscope", "C2"): "LP vsweep"}

        # Direct in-process path.
        self._build_direct(shots, descriptions)

        # Spool + offload path: acquire creates the skeleton, offload fills it.
        _build_bmotion_skeleton(self.off_h5, total_shots=2)
        meta = _make_meta(hdf5_path=self.off_h5)
        spool_format.write_run_metadata(self.spool, meta)
        for shot_num, all_data in shots.items():
            payload = spool_adapter.all_data_to_payload(
                all_data, shot_num, {"MG_A": (float(shot_num), 2.0)})
            spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 2)
        offload_runner.run_offload(self.spool, poll_seconds=0.01)

        _assert_hdf5_equivalent(self, self.direct_h5, self.off_h5)

    def test_offloaded_dataset_filters_and_dtype(self):
        _build_bmotion_skeleton(self.off_h5, total_shots=1)
        spool_format.write_run_metadata(self.spool, _make_meta(hdf5_path=self.off_h5))
        payload = spool_adapter.all_data_to_payload(
            _make_all_data(False), 1, {"MG_A": (1.0, 2.0)})
        spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 1)
        offload_runner.run_offload(self.spool, poll_seconds=0.01)

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
        _build_bmotion_skeleton(self.off_h5, total_shots=1)
        spool_format.write_run_metadata(self.spool, _make_meta(hdf5_path=self.off_h5))
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 1,
                                                    {"MG_A": (0.0, 0.0)})
        spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 1)
        offload_runner.run_offload(self.spool, poll_seconds=0.01)
        # Spool copy removed once verified.
        self.assertEqual(spool_format.iter_ready_shots(self.spool), [])
        self.assertFalse(os.path.isdir(os.path.join(self.spool, "shot_000001")))

    def test_corrupted_bin_kept_and_errors(self):
        _build_bmotion_skeleton(self.off_h5, total_shots=1)
        meta = _make_meta(hdf5_path=self.off_h5)
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


class OffloadResilienceTests(unittest.TestCase):
    """Bug-1 hardening: idempotent retry + poison-shot quarantine/drain."""

    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_res_")
        self.off_h5 = tempfile.mktemp(suffix=".hdf5")

    def tearDown(self):
        if os.path.exists(self.off_h5):
            os.remove(self.off_h5)

    def test_offload_idempotent_when_shot_already_written(self):
        # Pre-write shot_1's datasets (as an interrupted prior attempt would),
        # then run the full offload: it must NOT trip "shot already exists",
        # must verify the existing data, and must delete the bin.
        _build_bmotion_skeleton(self.off_h5, total_shots=1)
        all_data = _make_all_data(False)
        descriptions = {("lpscope", "C1"): "LP isat", ("lpscope", "C2"): "LP vsweep"}
        hdf5_writer.write_shot_data(self.off_h5, all_data, 1, descriptions)

        spool_format.write_run_metadata(self.spool, _make_meta(hdf5_path=self.off_h5))
        payload = spool_adapter.all_data_to_payload(all_data, 1, {"MG_A": (1.0, 2.0)})
        spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 1)

        offload_runner.run_offload(self.spool, poll_seconds=0.01)
        # Drained cleanly; bin removed.
        self.assertEqual(spool_format.iter_ready_shots(self.spool), [])

    def test_poison_shot_is_quarantined_and_run_drains(self):
        # A shot whose data can never verify must not hang the run: after
        # max_retries it is moved to shot_N.failed and the offload completes.
        _build_bmotion_skeleton(self.off_h5, total_shots=2)
        spool_format.write_run_metadata(self.spool, _make_meta(hdf5_path=self.off_h5))
        # Shot 1 good; shot 2 corrupted on disk after writing.
        for shot in (1, 2):
            payload = spool_adapter.all_data_to_payload(
                _make_all_data(False), shot, {"MG_A": (float(shot), 2.0)})
            spool_format.write_shot(self.spool, payload)
        bad = os.path.join(self.spool, "shot_000002", "lpscope__C1.bin")
        with open(bad, "wb") as f:
            f.write(b"\x00\x01")  # wrong length -> reshape/verify fails
        spool_format.write_run_complete(self.spool, 2)

        offload_runner.run_offload(self.spool, poll_seconds=0.01, max_retries=2)

        # Run drained (no infinite loop); good shot landed, bad shot quarantined.
        self.assertEqual(spool_format.iter_ready_shots(self.spool), [])
        self.assertTrue(os.path.isdir(os.path.join(self.spool, "shot_000002.failed")))
        with h5py.File(self.off_h5, "r") as f:
            self.assertIn("shot_1", f["lpscope"])

    def test_poison_shot_written_then_verify_fails_is_marked_failed(self):
        # The dangerous case: bin has the CORRECT length so it writes to HDF5,
        # but the read-back doesn't match (simulated by tampering the dataset
        # between write and the next verify). The shot_N group must end up marked
        # failed -- not silently kept as if it were good data.
        _build_bmotion_skeleton(self.off_h5, total_shots=1)
        meta = _make_meta(hdf5_path=self.off_h5)
        # Build a payload whose in-HDF5 data we will corrupt so verify fails.
        payload = spool_adapter.all_data_to_payload(_make_all_data(False), 1,
                                                    {"MG_A": (1.0, 2.0)})
        spool_format.write_shot(self.spool, payload)

        adapter = offload_runner._get_adapter("acquisition")
        # Pre-write the shot, then corrupt the stored values so the offload's
        # verify (which compares to the spooled bin) fails on every attempt.
        full = spool_format.read_shot(self.spool, 1)
        adapter.write_shot(self.off_h5, full, meta)
        with h5py.File(self.off_h5, "a") as f:
            f["lpscope/shot_1/C1_data"][...] = 0  # now mismatches the spooled bin

        spool_format.write_run_metadata(self.spool, meta)
        spool_format.write_run_complete(self.spool, 1)
        offload_runner.run_offload(self.spool, poll_seconds=0.01, max_retries=2)

        self.assertTrue(os.path.isdir(os.path.join(self.spool, "shot_000001.failed")))
        with h5py.File(self.off_h5, "r") as f:
            grp = f["lpscope/shot_1"]
            self.assertTrue(grp.attrs.get("failed"))
            self.assertTrue(grp.attrs.get("skipped"))
            # The corrupt data dataset is gone (replaced by the failed marker).
            self.assertNotIn("C1_data", grp)


class BackpressureTests(unittest.TestCase):
    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_bp_")

    def test_over_capacity_on_pending_count(self):
        for shot in (1, 2, 3):
            spool_format.write_shot(
                self.spool, spool_adapter.all_data_to_payload(_make_all_data(), shot, None))
        # 3 pending, limit 2 -> over capacity; limit disabled (0) -> not.
        self.assertIsNotNone(
            spool_format.spool_over_capacity(self.spool, max_pending_shots=2, min_free_gb=0))
        self.assertIsNone(
            spool_format.spool_over_capacity(self.spool, max_pending_shots=0, min_free_gb=0))

    def test_wait_returns_immediately_when_under_limit(self):
        # No pending shots, generous limits -> must not block.
        spool_format.wait_for_capacity(self.spool, max_pending_shots=10,
                                       min_free_gb=0, poll_seconds=0.01)

    def test_pending_count_and_free_space(self):
        self.assertEqual(spool_format.pending_shot_count(self.spool), 0)
        spool_format.write_shot(
            self.spool, spool_adapter.all_data_to_payload(_make_all_data(), 1, None))
        self.assertEqual(spool_format.pending_shot_count(self.spool), 1)
        self.assertGreater(spool_format.free_space_bytes(self.spool), 0)


class SetupFailureTests(unittest.TestCase):
    """A spooled run that aborts during setup (before any shot) must surface the
    real error and NOT leave a misleading RUN_COMPLETE for the offload."""

    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_setupfail_")
        self.off_h5 = tempfile.mktemp(suffix=".hdf5")
        # Config with NO scope_ips and NO position section: initialize_scopes
        # returns {} so the grid runner raises "No valid data" before the loop
        # and before write_run_metadata -- the exact setup-failure path.
        self.cfg = tempfile.mktemp(suffix=".ini")
        with open(self.cfg, "w") as f:
            f.write("[experiment]\nname = setupfail\ndescription = t\n[nshots]\nnum_duplicate_shots = 1\n")

    def tearDown(self):
        for p in (self.off_h5, self.cfg):
            if os.path.exists(p):
                os.remove(p)

    def test_grid_setup_failure_surfaces_real_error_no_run_complete(self):
        from acquisition import run_acquisition_spooled
        # Must raise the genuine RuntimeError, not a NameError from the finally.
        with self.assertRaises(RuntimeError) as ctx:
            run_acquisition_spooled(self.spool, self.off_h5, self.cfg)
        self.assertNotIsInstance(ctx.exception, NameError)
        self.assertIn("No valid data", str(ctx.exception))
        # No RUN_COMPLETE written (metadata was never written either), so an
        # offload would not finalize a false shot_count.
        self.assertFalse(spool_format.run_complete_exists(self.spool))
        self.assertFalse(spool_format.run_metadata_exists(self.spool))


class OffloadMissingTargetTests(unittest.TestCase):
    """The offload must refuse to run if the acquire-created file is absent."""

    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_")

    def test_missing_destination_raises(self):
        missing = tempfile.mktemp(suffix=".hdf5")
        spool_format.write_run_metadata(self.spool, _make_meta(hdf5_path=missing))
        spool_format.write_run_complete(self.spool, 0)
        with self.assertRaises(FileNotFoundError):
            offload_runner.run_offload(self.spool, poll_seconds=0.01)


def _build_grid_skeleton(hdf5_path, scope_name="lpscope", n_samples=128,
                         total_shots=2, nz=None):
    """Create the grid (PositionManager) HDF5 skeleton like acquire now does."""
    time_array = np.linspace(0, 1e-3, n_samples, dtype=np.float64)
    hdf5_writer.write_experiment_metadata(
        hdf5_path, description="grid unit-test", source_code={"unit": "test"},
        raw_config_text="[experiment]\ndescription = grid unit-test\n",
        config=None, scope_names=[scope_name],
    )
    hdf5_writer.write_scope_metadata(
        hdf5_path, scope_name=scope_name, description="test scope",
        ip_address="127.0.0.1", scope_type="LECROY,TEST,0,0",
    )
    hdf5_writer.write_time_array(hdf5_path, scope_name, time_array, 0)

    if nz is None:
        dtype = [('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4')]
        setup = np.zeros(total_shots, dtype=dtype)
        setup['shot_num'] = np.arange(1, total_shots + 1)
        setup['x'] = np.arange(total_shots, dtype=float)
        setup['y'] = 2.0
    else:
        dtype = [('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')]
        setup = np.zeros(total_shots, dtype=dtype)
        setup['shot_num'] = np.arange(1, total_shots + 1)
        setup['x'] = np.arange(total_shots, dtype=float)
        setup['y'] = 2.0
        setup['z'] = 3.0
    with h5py.File(hdf5_path, "a") as f:
        ctl = f.require_group("/Control")
        pos = ctl.create_group("Positions")
        ds = pos.create_dataset("positions_setup_array", data=setup, dtype=dtype)
        ds.attrs["xpos"] = np.unique(setup['x'])
        ds.attrs["ypos"] = np.unique(setup['y'])
        if nz is not None:
            ds.attrs["zpos"] = np.unique(setup['z'])
        pos.create_dataset("positions_array", shape=(total_shots,), dtype=dtype)


class GridOffloadTests(unittest.TestCase):
    """The grid ('grid' writer) offload fills scope data + the single
    positions_array, for both 2-D (x,y) and 3-D (x,y,z) runs."""

    def setUp(self):
        from acquisition import grid_spool_adapter
        self.grid_spool_adapter = grid_spool_adapter
        self.spool = tempfile.mkdtemp(prefix="spool_grid_")
        self.off_h5 = tempfile.mktemp(suffix=".hdf5")

    def tearDown(self):
        if os.path.exists(self.off_h5):
            os.remove(self.off_h5)

    def _run(self, nz, coords_for):
        _build_grid_skeleton(self.off_h5, total_shots=2, nz=nz)
        spool_format.write_run_metadata(self.spool, {
            "writer": "grid",
            "hdf5_path": self.off_h5,
            "config_scope_names": ["lpscope"],
            "channel_descriptions": {"lpscope_C1": "isat", "lpscope_C2": "vsweep"},
            "nz": nz,
        })
        for shot in (1, 2):
            payload = self.grid_spool_adapter.all_data_to_payload(
                _make_all_data(False), shot, coords_for(shot))
            spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 2)
        offload_runner.run_offload(self.spool, poll_seconds=0.01)

    def test_grid_2d(self):
        self._run(nz=None, coords_for=lambda s: {"x": float(s), "y": 2.0, "z": None})
        with h5py.File(self.off_h5, "r") as f:
            arr = f["/Control/Positions/positions_array"][()]
            self.assertEqual(arr.dtype.names, ("shot_num", "x", "y"))
            self.assertEqual(list(arr["shot_num"]), [1, 2])
            self.assertEqual(list(arr["x"]), [1.0, 2.0])
            ds = f["lpscope/shot_1/C1_data"]
            self.assertEqual(ds.dtype, np.dtype("int16"))
            self.assertEqual(f["lpscope"].attrs.get("shot_count"), 2)

    def test_grid_3d(self):
        self._run(nz=11, coords_for=lambda s: {"x": float(s), "y": 2.0, "z": 3.0})
        with h5py.File(self.off_h5, "r") as f:
            arr = f["/Control/Positions/positions_array"][()]
            self.assertEqual(arr.dtype.names, ("shot_num", "x", "y", "z"))
            self.assertEqual(list(arr["z"]), [3.0, 3.0])


class CrashSafetyTests(unittest.TestCase):
    def setUp(self):
        self.spool = tempfile.mkdtemp(prefix="spool_")

    def test_tmp_dir_without_done_is_ignored(self):
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
        # Acquire-side: build the skeleton shaped like the reference.
        _build_ref_shaped_skeleton(cls.off_h5)
        spool_format.write_run_metadata(cls.spool, {
            "writer": "acquisition",
            "hdf5_path": cls.off_h5,
            "config_scope_names": ["lpscope"],
            "channel_descriptions": {
                "lpscope_C1": "LP@P41 Isat 50ohm",
                "lpscope_C2": "LP@P29 Isat 50ohm",
                "lpscope_C3": "LP@P21 Isat 50ohm",
            },
        })
        for shot_num in (1, 2):
            spool_format.write_shot(cls.spool, _ref_shaped_payload(shot_num))
        spool_format.write_run_complete(cls.spool, 2)
        offload_runner.run_offload(cls.spool, poll_seconds=0.01)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.off_h5):
            os.remove(cls.off_h5)

    def test_schema_matches_reference(self):
        with h5py.File(REFERENCE_HDF5, "r") as ref, h5py.File(self.off_h5, "r") as new:
            for grp in ("Configuration", "Control", "lpscope"):
                self.assertIn(grp, ref, f"reference missing {grp}")
                self.assertIn(grp, new, f"offloaded file missing {grp}")

            for attr in ("description", "creation_time", "source_code"):
                self.assertIn(attr, new.attrs, f"root attr {attr} missing")

            for name in ("experiment_config", "bmotion_config", "bmotion_selection"):
                self.assertIn(name, ref["Configuration"])
                self.assertIn(name, new["Configuration"])

            for attr in ("description", "ip_address", "scope_type"):
                self.assertIn(attr, new["lpscope"].attrs)
            ref_ta = ref["lpscope"]["time_array"]
            new_ta = new["lpscope"]["time_array"]
            self.assertEqual(new_ta.dtype, ref_ta.dtype)
            self.assertEqual(new_ta.attrs["units"], ref_ta.attrs["units"])

            ref_ds = ref["lpscope"]["shot_1"]["C1_data"]
            new_ds = new["lpscope"]["shot_1"]["C1_data"]
            self.assertEqual(new_ds.dtype, ref_ds.dtype)
            self.assertEqual(new_ds.dtype, np.dtype("int16"))
            self.assertEqual(new_ds.compression, ref_ds.compression)
            self.assertEqual(new_ds.shuffle, ref_ds.shuffle)
            self.assertEqual(new_ds.fletcher32, ref_ds.fletcher32)
            self.assertEqual(new_ds.chunks, new_ds.shape)
            self.assertEqual(ref_ds.chunks, ref_ds.shape)
            self.assertEqual(new_ds.attrs["dtype"], "int16")
            self.assertIn("description", dict(new_ds.attrs))

            ref_hdr = ref["lpscope"]["shot_1"]["C1_header"]
            new_hdr = new["lpscope"]["shot_1"]["C1_header"]
            self.assertEqual(new_hdr.dtype.kind, ref_hdr.dtype.kind)  # 'V'
            self.assertIn("description", dict(new_hdr.attrs))

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


def _build_ref_shaped_skeleton(hdf5_path, mg_name="<Athena>    p21_LP"):
    time_array = np.linspace(0, 1e-3, 256, dtype=np.float64)
    hdf5_writer.write_experiment_metadata(
        hdf5_path, description="schema check", source_code={"unit": "test"},
        raw_config_text="[experiment]\ndescription = schema check\n",
        config=None, scope_names=["lpscope"],
    )
    hdf5_writer.write_scope_metadata(
        hdf5_path, scope_name="lpscope", description="LeCroy regular black scope",
        ip_address="192.168.7.67", scope_type="LECROY,WR8208HD,TEST,0",
    )
    hdf5_writer.write_time_array(hdf5_path, "lpscope", time_array, 0)
    setup = np.zeros(2, dtype=bmotion._POSITION_DTYPE)
    setup["shot_num"] = [1, 2]
    setup["x"] = [-15.0, -14.0]
    setup["y"] = [15.0, 15.0]
    bmotion.write_bmotion_position_groups(
        hdf5_path, total_shots=2, toml_text="# bmotion toml\n",
        selection_blob='{"mg_keys": ["0"], "execution_order": "interleaved"}',
        prepared=[("0", mg_name, setup, np.array([-15.0, -14.0]), np.array([15.0]))],
    )


def _ref_shaped_payload(shot_num):
    rng = np.random.default_rng(shot_num)
    traces = {
        "lpscope": [TracePayload(ch, rng.integers(-30000, 30000, 256, dtype=np.int16),
                                 f"HDR-{ch}".encode())
                    for ch in ("C1", "C2", "C3")]
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
