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
import errno
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import h5py
import numpy as np

from spooling import ShotPayload, TracePayload, spool_format
from acquisition import bmotion, hdf5_writer, spool_adapter
import offload_engine
from _hdf5_assertions import assert_dataset_filters, assert_hdf5_scope_equivalent

REFERENCE_HDF5 = r"D:\data\LAPD\03-LP-p21p29p41-plane-Helium_2026-05-20.hdf5"


def _temp_spool_dir(tc, prefix="spool_"):
    """A fresh spool dir, removed (with contents) when the test finishes."""
    path = tempfile.mkdtemp(prefix=prefix)
    tc.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


def _temp_path(tc, name):
    """A not-yet-existing pathname inside a fresh auto-cleaned temp dir.

    Replacement for tempfile.mktemp(): the writers under test must create the
    file themselves, so the name has to start absent -- but inside a private
    directory there is no name race, and cleanup removes whatever was created.
    """
    d = tempfile.mkdtemp(prefix="daqspool_")
    tc.addCleanup(shutil.rmtree, d, ignore_errors=True)
    return os.path.join(d, name)


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


def _write_scope_skeleton(hdf5_path, scopes=None, n_samples=128,
                          description="unit-test run",
                          scope_type="LECROY,TEST,0,0"):
    """The skeleton prefix every builder shares: experiment metadata, then
    per-scope metadata + time array, via the same `main` writers acquire calls
    -- so a test offload fills a file shaped exactly like acquire's.

    ``scopes`` maps scope name -> (description, ip_address).
    """
    if scopes is None:
        scopes = {"lpscope": ("test scope", "127.0.0.1")}
    time_array = np.linspace(0, 1e-3, n_samples, dtype=np.float64)
    hdf5_writer.write_experiment_metadata(
        hdf5_path,
        description=description,
        source_code={"unit": "test"},
        raw_config_text=f"[experiment]\ndescription = {description}\n",
        config=None,
        scope_names=list(scopes),
    )
    for scope_name, (scope_desc, ip_address) in scopes.items():
        hdf5_writer.write_scope_metadata(
            hdf5_path, scope_name=scope_name, description=scope_desc,
            ip_address=ip_address, scope_type=scope_type,
        )
        hdf5_writer.write_time_array(hdf5_path, scope_name, time_array, 0)


def _write_two_point_bmotion_positions(hdf5_path, mg_name, total_shots, xs, y):
    """Two-position bmotion Control/Positions groups (setup + preallocated
    positions_array), via the same writer acquire calls."""
    setup = np.zeros(2, dtype=bmotion._POSITION_DTYPE)
    setup["shot_num"] = [1, 2]
    setup["x"] = xs
    setup["y"] = [y, y]
    bmotion.write_bmotion_position_groups(
        hdf5_path,
        total_shots=total_shots,
        toml_text="# stub toml\n",
        selection_blob='{"mg_keys": ["0"], "execution_order": "interleaved"}',
        prepared=[("0", mg_name, setup, np.asarray(xs, dtype=float), np.array([y]))],
    )


def _build_bmotion_skeleton(hdf5_path, scope_name="lpscope", n_samples=128,
                            total_shots=4, mg_name="MG_A"):
    """Create the bmotion HDF5 skeleton exactly as the acquire process does."""
    _write_scope_skeleton(hdf5_path, {scope_name: ("test scope", "127.0.0.1")},
                          n_samples=n_samples)
    _write_two_point_bmotion_positions(hdf5_path, mg_name, total_shots,
                                       xs=[-1.0, 0.0], y=2.0)


class SpoolRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.spool = _temp_spool_dir(self)

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


def _make_two_scope_all_data(seq=False):
    """Synthetic two-scope all_data (parallel-spool-write exercise)."""
    rng = np.random.default_rng(99)
    one = _make_all_data(seq)["lpscope"]
    if seq:
        d = rng.integers(-100, 100, size=(3, 64), dtype=np.int16)
    else:
        d = rng.integers(-30000, 30000, size=128, dtype=np.int16)
    two = (["C1"], {"C1": d}, {"C1": b"HEADER-X-C1"})
    return {"lpscope": one, "xrayscope": two}


class ParallelSpoolWriteTests(unittest.TestCase):
    """`write_shot(parallel=True)` must produce a byte-identical spool + schema.

    Two scopes write disjoint <scope>__* files; parallelizing only reorders the
    bytes hitting separate files, so the on-disk files, the sidecar, and the
    reconstructed schema must all match the serial path exactly.
    """

    def setUp(self):
        self.serial = _temp_spool_dir(self, "spool_ser_")
        self.par = _temp_spool_dir(self, "spool_par_")

    def _files(self, spool_dir, shot_num):
        shot_dir = os.path.join(spool_dir, "shot_%06d" % shot_num)
        out = {}
        for name in sorted(os.listdir(shot_dir)):
            with open(os.path.join(shot_dir, name), "rb") as f:
                out[name] = f.read()
        return out

    def _assert_identical_and_correct(self, all_data, seq):
        payload_s = spool_adapter.all_data_to_payload(all_data, 1, None)
        payload_p = spool_adapter.all_data_to_payload(all_data, 1, None)
        spool_format.write_shot(self.serial, payload_s, parallel=False)
        spool_format.write_shot(self.par, payload_p, parallel=True)

        # (a) identical on-disk files (bin/hdr bytes AND the pickled sidecar).
        self.assertEqual(self._files(self.serial, 1), self._files(self.par, 1))

        # (b) reconstructed schema matches the input arrays/headers per scope.
        got = spool_format.read_shot(self.par, 1)
        self.assertEqual(set(got.traces), {"lpscope", "xrayscope"})
        for scope_name, (traces, data, headers) in all_data.items():
            by_ch = {t.channel: t for t in got.traces[scope_name]}
            self.assertEqual(set(by_ch), set(traces))
            for ch in traces:
                np.testing.assert_array_equal(by_ch[ch].data, data[ch])
                self.assertEqual(by_ch[ch].data.dtype, np.int16)
                self.assertEqual(by_ch[ch].header, headers[ch])

    def test_realtime_parallel_matches_serial(self):
        self._assert_identical_and_correct(_make_two_scope_all_data(False), seq=False)

    def test_sequence_2d_parallel_matches_serial(self):
        self._assert_identical_and_correct(_make_two_scope_all_data(True), seq=True)

    def test_parallel_offloads_to_correct_hdf5_schema(self):
        """End-to-end: parallel-written spool -> offload -> correct HDF5 schema."""
        off_h5 = _temp_path(self, "parallel_offload.hdf5")
        all_data = _make_two_scope_all_data(False)
        _write_scope_skeleton(off_h5, {
            "lpscope": ("lpscope", "127.0.0.1"),
            "xrayscope": ("xrayscope", "127.0.0.2"),
        })

        meta = {
            "writer": "acquisition",
            "hdf5_path": off_h5,
            "config_scope_names": ["lpscope", "xrayscope"],
            "channel_descriptions": {
                "lpscope_C1": "LP isat", "lpscope_C2": "LP vsweep",
                "xrayscope_C1": "xray",
            },
        }
        spool_format.write_run_metadata(self.par, meta)
        payload = spool_adapter.all_data_to_payload(all_data, 1, None)
        spool_format.write_shot(self.par, payload, parallel=True)
        spool_format.write_run_complete(self.par, 1)
        offload_engine.run_offload(self.par, poll_seconds=0.01)

        with h5py.File(off_h5, "r") as f:
            for scope_name, (traces, data, _h) in all_data.items():
                for ch in traces:
                    ds = f[f"{scope_name}/shot_1/{ch}_data"]
                    self.assertEqual(ds.dtype, np.dtype("int16"))
                    np.testing.assert_array_equal(ds[()], data[ch])
                    self.assertIn(f"{ch}_header", f[f"{scope_name}/shot_1"])


class OffloadEquivalenceTests(unittest.TestCase):
    """Completeness gate: offloaded HDF5 must match the in-process writer output."""

    def setUp(self):
        self.spool = _temp_spool_dir(self)
        self.off_h5 = _temp_path(self, "offloaded.hdf5")
        self.direct_h5 = _temp_path(self, "direct.hdf5")

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
        offload_engine.run_offload(self.spool, poll_seconds=0.01)

        assert_hdf5_scope_equivalent(self, self.direct_h5, self.off_h5)

    def test_offloaded_dataset_filters_and_dtype(self):
        _build_bmotion_skeleton(self.off_h5, total_shots=1)
        spool_format.write_run_metadata(self.spool, _make_meta(hdf5_path=self.off_h5))
        payload = spool_adapter.all_data_to_payload(
            _make_all_data(False), 1, {"MG_A": (1.0, 2.0)})
        spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 1)
        offload_engine.run_offload(self.spool, poll_seconds=0.01)

        with h5py.File(self.off_h5, "r") as f:
            ds = f["lpscope/shot_1/C1_data"]
            assert_dataset_filters(self, ds, "int16", compression="lzf",
                                   shuffle=True, fletcher32=True)
            self.assertEqual(ds.chunks, ds.shape)


class VerifyAndDeleteTests(unittest.TestCase):
    def setUp(self):
        self.spool = _temp_spool_dir(self)
        self.off_h5 = _temp_path(self, "verify.hdf5")

    def test_good_shot_is_deleted_after_verify(self):
        _build_bmotion_skeleton(self.off_h5, total_shots=1)
        spool_format.write_run_metadata(self.spool, _make_meta(hdf5_path=self.off_h5))
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 1,
                                                    {"MG_A": (0.0, 0.0)})
        spool_format.write_shot(self.spool, payload)
        spool_format.write_run_complete(self.spool, 1)
        offload_engine.run_offload(self.spool, poll_seconds=0.01)
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

        adapter = offload_engine._get_adapter("acquisition")
        # The truncated bin no longer matches the sidecar's recorded shape, so
        # read_shot's reshape raises ValueError (as would a read-back data
        # mismatch from _verify_shot_in_hdf5) -- not just any Exception.
        with self.assertRaises(ValueError):
            offload_engine._offload_one_shot(self.spool, self.off_h5, meta,
                                             adapter, 1)
        # Bin is NOT deleted on verification failure.
        self.assertTrue(os.path.isdir(os.path.join(self.spool, "shot_000001")))


class OffloadResilienceTests(unittest.TestCase):
    """Bug-1 hardening: idempotent retry + poison-shot quarantine/drain."""

    def setUp(self):
        self.spool = _temp_spool_dir(self, "spool_res_")
        self.off_h5 = _temp_path(self, "resilience.hdf5")

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

        offload_engine.run_offload(self.spool, poll_seconds=0.01)
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

        offload_engine.run_offload(self.spool, poll_seconds=0.01, max_retries=2)

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

        adapter = offload_engine._get_adapter("acquisition")
        # Pre-write the shot, then corrupt the stored values so the offload's
        # verify (which compares to the spooled bin) fails on every attempt.
        full = spool_format.read_shot(self.spool, 1)
        adapter.write_shot(self.off_h5, full, meta)
        with h5py.File(self.off_h5, "a") as f:
            f["lpscope/shot_1/C1_data"][...] = 0  # now mismatches the spooled bin

        spool_format.write_run_metadata(self.spool, meta)
        spool_format.write_run_complete(self.spool, 1)
        offload_engine.run_offload(self.spool, poll_seconds=0.01, max_retries=2)

        self.assertTrue(os.path.isdir(os.path.join(self.spool, "shot_000001.failed")))
        with h5py.File(self.off_h5, "r") as f:
            grp = f["lpscope/shot_1"]
            self.assertTrue(grp.attrs.get("failed"))
            self.assertTrue(grp.attrs.get("skipped"))
            # The corrupt data dataset is gone (replaced by the failed marker).
            self.assertNotIn("C1_data", grp)


class DiskFullRetryTests(unittest.TestCase):
    """The only backpressure now: pause+retry on a real disk-full write error."""

    def setUp(self):
        self.spool = _temp_spool_dir(self, "spool_df_")

    def test_pending_count(self):
        self.assertEqual(spool_format.pending_shot_count(self.spool), 0)
        spool_format.write_shot(
            self.spool, spool_adapter.all_data_to_payload(_make_all_data(), 1, None))
        self.assertEqual(spool_format.pending_shot_count(self.spool), 1)

    def test_is_disk_full_error(self):
        self.assertTrue(spool_format.is_disk_full_error(OSError(errno.ENOSPC, "full")))
        win = OSError("full")
        win.winerror = 112
        self.assertTrue(spool_format.is_disk_full_error(win))
        self.assertFalse(spool_format.is_disk_full_error(OSError(errno.EACCES, "denied")))
        self.assertFalse(spool_format.is_disk_full_error(ValueError("nope")))

    def test_retries_then_succeeds(self):
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 1, None)
        calls = []

        def fake_write_shot(spool_dir, p, parallel=False):
            calls.append(1)
            if len(calls) < 3:  # fail twice, succeed on the third attempt
                raise OSError(errno.ENOSPC, "No space left on device")

        with mock.patch.object(spool_format, "write_shot", side_effect=fake_write_shot), \
                mock.patch.object(spool_format, "_sleep") as sleep:
            spool_format.write_shot_with_disk_full_retry(
                self.spool, payload, pause_seconds=0.0, max_retries=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleep.call_count, 2)  # one sleep per failed attempt

    def test_aborts_after_max_retries(self):
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 1, None)

        def always_full(spool_dir, p, parallel=False):
            raise OSError(errno.ENOSPC, "No space left on device")

        with mock.patch.object(spool_format, "write_shot", side_effect=always_full), \
                mock.patch.object(spool_format, "_sleep"):
            with self.assertRaises(OSError):
                spool_format.write_shot_with_disk_full_retry(
                    self.spool, payload, pause_seconds=0.0, max_retries=2)

    def test_non_disk_full_error_propagates_immediately(self):
        payload = spool_adapter.all_data_to_payload(_make_all_data(), 1, None)
        calls = []

        def other_error(spool_dir, p, parallel=False):
            calls.append(1)
            raise OSError(errno.EACCES, "Permission denied")

        with mock.patch.object(spool_format, "write_shot", side_effect=other_error), \
                mock.patch.object(spool_format, "_sleep") as sleep:
            with self.assertRaises(OSError):
                spool_format.write_shot_with_disk_full_retry(
                    self.spool, payload, pause_seconds=0.0, max_retries=3)
        self.assertEqual(len(calls), 1)  # no retry for a non-disk-full error
        sleep.assert_not_called()


class MetadataWaitTests(unittest.TestCase):
    """The offload is auto-launched before acquire writes meta_run.pkl, so it
    must WAIT for the metadata (not exit), and only time out on a folder that
    will never become a run (e.g. a spool ROOT)."""

    def setUp(self):
        self.spool = _temp_spool_dir(self, "spool_meta_")

    def test_waits_for_late_metadata(self):
        # No metadata yet: a short bounded wait should still pick it up once it
        # appears, then drain to completion.
        off_h5 = _temp_path(self, "late_meta.hdf5")
        _build_bmotion_skeleton(off_h5, total_shots=1)

        def writer():
            spool_format.write_run_metadata(self.spool, _make_meta(hdf5_path=off_h5))
            payload = spool_adapter.all_data_to_payload(
                _make_all_data(False), 1, {"MG_A": (1.0, 2.0)})
            spool_format.write_shot(self.spool, payload)
            spool_format.write_run_complete(self.spool, 1)

        t = threading.Timer(0.05, writer)
        t.start()
        try:
            offload_engine.run_offload(self.spool, poll_seconds=0.01,
                                       metadata_timeout=5.0)
        finally:
            t.join()
        with h5py.File(off_h5, "r") as f:
            self.assertIn("shot_1", f["lpscope"])

    def test_times_out_when_metadata_never_arrives(self):
        # Empty folder, metadata never written -> MetadataTimeout (not a hang).
        with self.assertRaises(offload_engine.MetadataTimeout):
            offload_engine.run_offload(self.spool, poll_seconds=0.01,
                                       metadata_timeout=0.05)


class OffloadErrorPathTests(unittest.TestCase):
    """Error paths give actionable messages instead of bare tracebacks."""

    def setUp(self):
        self.spool = _temp_spool_dir(self, "spool_err_")

    def test_metadata_without_hdf5_path_raises_clear_error(self):
        # An older/truncated bundle lacking hdf5_path must raise a clear
        # ValueError pointing at --list, not a bare KeyError.
        meta = _make_meta(hdf5_path=None)
        meta.pop("hdf5_path", None)
        spool_format.write_run_metadata(self.spool, meta)
        with self.assertRaises(ValueError) as ctx:
            offload_engine.run_offload(self.spool, poll_seconds=0.01,
                                       metadata_timeout=1.0)
        self.assertIn("hdf5_path", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, KeyError)

    def test_unknown_writer_tag_lists_supported(self):
        with self.assertRaises(ValueError) as ctx:
            offload_engine._get_adapter("nope")
        msg = str(ctx.exception)
        self.assertIn("acquisition", msg)
        self.assertIn("grid", msg)

    def test_list_survives_corrupt_folder(self):
        # A spool ROOT with one good run + one corrupt meta_run.pkl: --list must
        # report both, not abort on the corrupt one.
        root = _temp_spool_dir(self, "spool_root_")
        good = os.path.join(root, "good_run")
        os.makedirs(good)
        off_h5 = _temp_path(self, "list_check.hdf5")
        spool_format.write_run_metadata(good, _make_meta(hdf5_path=off_h5))

        bad = os.path.join(root, "bad_run")
        os.makedirs(bad)
        # A meta_run.pkl that exists but is not a valid pickle.
        with open(os.path.join(bad, spool_format._META_RUN), "wb") as f:
            f.write(b"not a pickle")

        import Offload_Run
        # Must not raise despite the corrupt folder.
        Offload_Run._list_spools(root)

    def test_corrupt_metadata_raises_typed_error(self):
        # A present-but-unreadable file raises the shared typed error (so every
        # caller can recognize it), not a raw UnpicklingError.
        with open(os.path.join(self.spool, spool_format._META_RUN), "wb") as f:
            f.write(b"not a pickle")
        with self.assertRaises(spool_format.SpoolMetadataError):
            spool_format.read_run_metadata(self.spool)

    def test_corrupt_run_complete_raises_typed_error(self):
        # Corrupt RUN_COMPLETE must not masquerade as "no completion" (None) --
        # it raises so a finished run isn't silently offered for restart.
        with open(os.path.join(self.spool, spool_format._RUN_COMPLETE), "wb") as f:
            f.write(b"not a pickle")
        with self.assertRaises(spool_format.SpoolMetadataError):
            spool_format.read_run_complete(self.spool)

    def test_absent_files_keep_their_contract(self):
        # Absence is a separate, expected case: metadata -> FileNotFoundError,
        # RUN_COMPLETE -> None (unchanged by the corrupt-file hardening).
        with self.assertRaises(FileNotFoundError):
            spool_format.read_run_metadata(self.spool)
        self.assertIsNone(spool_format.read_run_complete(self.spool))


class SetupFailureTests(unittest.TestCase):
    """A spooled run that aborts during setup (before any shot) must surface the
    real error and NOT leave a misleading RUN_COMPLETE for the offload."""

    def setUp(self):
        self.spool = _temp_spool_dir(self, "spool_setupfail_")
        self.off_h5 = _temp_path(self, "setupfail.hdf5")
        # Config with NO scope_ips and NO position section: initialize_scopes
        # returns {} so the grid runner raises "No valid data" before the loop
        # and before write_run_metadata -- the exact setup-failure path.
        self.cfg = _temp_path(self, "setupfail.ini")
        with open(self.cfg, "w") as f:
            f.write("[experiment]\nname = setupfail\ndescription = t\n[nshots]\nnum_duplicate_shots = 1\n")

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
        self.spool = _temp_spool_dir(self)

    def test_missing_destination_raises(self):
        # Never created -- the offload must refuse to run against it.
        missing = _temp_path(self, "never_created.hdf5")
        spool_format.write_run_metadata(self.spool, _make_meta(hdf5_path=missing))
        spool_format.write_run_complete(self.spool, 0)
        with self.assertRaises(FileNotFoundError):
            offload_engine.run_offload(self.spool, poll_seconds=0.01)


def _build_grid_skeleton(hdf5_path, scope_name="lpscope", n_samples=128,
                         total_shots=2, nz=None):
    """Create the grid (PositionManager) HDF5 skeleton like acquire now does."""
    _write_scope_skeleton(hdf5_path, {scope_name: ("test scope", "127.0.0.1")},
                          n_samples=n_samples, description="grid unit-test")

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
        self.spool = _temp_spool_dir(self, "spool_grid_")
        self.off_h5 = _temp_path(self, "grid.hdf5")

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
        offload_engine.run_offload(self.spool, poll_seconds=0.01)

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
        self.spool = _temp_spool_dir(self)

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
        cls.addClassCleanup(shutil.rmtree, cls.spool, ignore_errors=True)
        out_dir = tempfile.mkdtemp(prefix="daqspool_")
        cls.addClassCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        cls.off_h5 = os.path.join(out_dir, "ref_shaped.hdf5")
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
        offload_engine.run_offload(cls.spool, poll_seconds=0.01)

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
    _write_scope_skeleton(
        hdf5_path, {"lpscope": ("LeCroy regular black scope", "192.168.7.67")},
        n_samples=256, description="schema check",
        scope_type="LECROY,WR8208HD,TEST,0",
    )
    _write_two_point_bmotion_positions(hdf5_path, mg_name, total_shots=2,
                                       xs=[-15.0, -14.0], y=15.0)


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


if __name__ == "__main__":
    unittest.main()
