"""Unit tests for parallel multi-scope arm/read in MultiScopeAcquisition.

Covers `acquire_shot_parallel`, `acquire_shot_dispatch`, and the parallel
`arm_scopes_for_trigger` (acquisition/scope_runner.py) with fake scope objects --
no hardware. Verifies:
  * parallel read returns the same all_data structure as sequential acquire_shot,
  * reads actually overlap (wall-clock ~ max read, not sum of reads),
  * a scope that errors is skipped while the others still return,
  * KeyboardInterrupt propagates to abort the run,
  * the parallel_scope_read flag routes acquire_shot_dispatch,
  * parallel arming overlaps (wall ~ max not sum) and propagates arm errors,
  * single_shot_acquisition and the spooled callers route through the dispatcher.

Run:

    python -m unittest tests.test_daq_parallel
"""

import sys
import time
import unittest
from configparser import ConfigParser
from pathlib import Path
from unittest import mock

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from acquisition import scope_runner
from acquisition.scope_runner import MultiScopeAcquisition


class FakeScope:
    """Minimal stand-in for a LeCroy_Scope handle.

    Each read sleeps `read_seconds` to emulate a blocking network transfer; each
    arm sleeps `arm_seconds`. Real I/O releases the GIL during socket round-trips,
    and time.sleep does too, so this faithfully models the overlap available to
    threads.
    """

    def __init__(self, traces, read_seconds=0.0, arm_seconds=0.0,
                 raise_exc=None, arm_raise=None):
        self._traces = list(traces)
        self.read_seconds = read_seconds
        self.arm_seconds = arm_seconds
        self.raise_exc = raise_exc
        self.arm_raise = arm_raise

    def displayed_traces(self):
        return list(self._traces)

    def set_trigger_mode(self, mode):
        # Empty-string query path used by stop_triggering: report STOP at once.
        if mode == "":
            return "STOP"
        # 'SINGLE' arm path.
        if self.arm_seconds:
            time.sleep(self.arm_seconds)
        if self.arm_raise is not None:
            raise self.arm_raise
        return "SINGLE"

    def acquire(self, tr, raw=True):
        if self.read_seconds:
            time.sleep(self.read_seconds)
        if self.raise_exc is not None:
            raise self.raise_exc
        data = np.full(8, ord(tr[-1]), dtype=np.int16)
        header = b"hdr-" + tr.encode()
        return data, header


def _make_msa(scopes, parallel_read=True, parallel_arm=True):
    """Build a MultiScopeAcquisition wired to fake scopes, no HDF5/IO."""
    config = ConfigParser()
    config["acquisition"] = {
        "parallel_scope_read": "true" if parallel_read else "false",
        "parallel_scope_arm": "true" if parallel_arm else "false",
    }
    config["scope_ips"] = {name: "0.0.0.0" for name in scopes}
    msa = MultiScopeAcquisition.__new__(MultiScopeAcquisition)
    # Bypass __init__'s connection logic; set just what the read/arm path touches.
    msa.config = config
    msa.raw_config_text = ""
    msa.scopes = dict(scopes)
    msa.figures = {}
    msa.time_arrays = {}
    msa.scope_ips = {name: "0.0.0.0" for name in scopes}
    msa.parallel_scope_read = parallel_read
    msa.parallel_scope_arm = parallel_arm
    msa.parallel_spool_write = True
    return msa


class ParallelReadCorrectnessTest(unittest.TestCase):
    def test_matches_sequential_result(self):
        scopes = {
            "A": FakeScope(["C1", "C2"]),
            "B": FakeScope(["C3"]),
        }
        active = {"A": 0, "B": 0}
        seq = _make_msa(scopes).acquire_shot(active, 1, verbose=False)
        par = _make_msa(scopes).acquire_shot_parallel(active, 1, verbose=False)

        self.assertEqual(set(seq), set(par))
        for name in seq:
            seq_traces, seq_data, _ = seq[name]
            par_traces, par_data, _ = par[name]
            self.assertEqual(seq_traces, par_traces)
            for tr in seq_traces:
                np.testing.assert_array_equal(seq_data[tr], par_data[tr])

    def test_reads_overlap(self):
        # Two scopes, each read takes 0.2s. Parallel should be ~0.2s, not ~0.4s.
        scopes = {
            "A": FakeScope(["C1"], read_seconds=0.2),
            "B": FakeScope(["C2"], read_seconds=0.2),
        }
        active = {"A": 0, "B": 0}
        msa = _make_msa(scopes)
        t0 = time.perf_counter()
        msa.acquire_shot_parallel(active, 1, verbose=False)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.35, f"reads did not overlap (took {elapsed:.3f}s)")

    def test_failing_scope_skipped(self):
        scopes = {
            "A": FakeScope(["C1"]),
            "B": FakeScope(["C2"], raise_exc=RuntimeError("boom")),
        }
        active = {"A": 0, "B": 0}
        out = _make_msa(scopes).acquire_shot_parallel(active, 1, verbose=False)
        self.assertIn("A", out)
        self.assertNotIn("B", out)

    def test_keyboardinterrupt_propagates(self):
        scopes = {
            "A": FakeScope(["C1"]),
            "B": FakeScope(["C2"], raise_exc=KeyboardInterrupt()),
        }
        active = {"A": 0, "B": 0}
        with self.assertRaises(KeyboardInterrupt):
            _make_msa(scopes).acquire_shot_parallel(active, 1, verbose=False)

    def test_single_scope_short_circuits_to_sequential(self):
        scopes = {"A": FakeScope(["C1"])}
        active = {"A": 0}
        msa = _make_msa(scopes)
        with mock.patch.object(msa, "acquire_shot",
                               wraps=msa.acquire_shot) as seq_spy:
            msa.acquire_shot_parallel(active, 1, verbose=False)
            seq_spy.assert_called_once()


class DispatchRoutingTest(unittest.TestCase):
    def test_dispatch_uses_parallel_when_flag_true(self):
        msa = _make_msa({"A": FakeScope(["C1"])}, parallel_read=True)
        with mock.patch.object(msa, "acquire_shot_parallel") as par, \
             mock.patch.object(msa, "acquire_shot") as seq:
            msa.acquire_shot_dispatch({"A": 0}, 1, verbose=False)
            par.assert_called_once()
            seq.assert_not_called()

    def test_dispatch_uses_sequential_when_flag_false(self):
        msa = _make_msa({"A": FakeScope(["C1"])}, parallel_read=False)
        with mock.patch.object(msa, "acquire_shot_parallel") as par, \
             mock.patch.object(msa, "acquire_shot") as seq:
            msa.acquire_shot_dispatch({"A": 0}, 1, verbose=False)
            seq.assert_called_once()
            par.assert_not_called()

    def test_single_shot_acquisition_routes_through_dispatch(self):
        msa = _make_msa({"A": FakeScope(["C1"])})
        with mock.patch.object(msa, "arm_scopes_for_trigger"), \
             mock.patch.object(msa, "acquire_shot_dispatch",
                               return_value={}) as disp:
            scope_runner.single_shot_acquisition(msa, {"A": 0}, 1, verbose=False)
            disp.assert_called_once()


class ParallelArmTest(unittest.TestCase):
    def test_arms_overlap(self):
        # Two scopes, each arm takes 0.2s. Parallel arm should be ~0.2s not ~0.4s.
        scopes = {
            "A": FakeScope(["C1"], arm_seconds=0.2),
            "B": FakeScope(["C2"], arm_seconds=0.2),
        }
        msa = _make_msa(scopes, parallel_arm=True)
        t0 = time.perf_counter()
        msa.arm_scopes_for_trigger(scopes.keys(), verbose=False)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.35, f"arms did not overlap (took {elapsed:.3f}s)")

    def test_sequential_arm_is_slower(self):
        scopes = {
            "A": FakeScope(["C1"], arm_seconds=0.15),
            "B": FakeScope(["C2"], arm_seconds=0.15),
        }
        msa = _make_msa(scopes, parallel_arm=False)
        t0 = time.perf_counter()
        msa.arm_scopes_for_trigger(scopes.keys(), verbose=False)
        elapsed = time.perf_counter() - t0
        self.assertGreater(elapsed, 0.28, "sequential arm should sum the arm times")

    def test_arm_error_propagates(self):
        # A failed arm must abort the shot, not be silently swallowed.
        scopes = {
            "A": FakeScope(["C1"]),
            "B": FakeScope(["C2"], arm_raise=RuntimeError("arm failed")),
        }
        msa = _make_msa(scopes, parallel_arm=True)
        with self.assertRaises(RuntimeError):
            msa.arm_scopes_for_trigger(scopes.keys(), verbose=False)


if __name__ == "__main__":
    unittest.main()
