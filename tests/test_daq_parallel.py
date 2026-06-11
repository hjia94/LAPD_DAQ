"""Unit tests for parallel multi-scope arm/read in MultiScopeAcquisition.

Covers `acquire_shot_parallel`, `acquire_shot_dispatch`, and the parallel
`arm_scopes_for_trigger` (acquisition/scope_runner.py) with fake scope objects --
no hardware. Verifies:
  * parallel read returns the same all_data structure as sequential acquire_shot,
  * reads actually overlap (proven by a shared barrier that only releases when
    both reads are in flight together -- no wall-clock thresholds),
  * a scope that errors is skipped while the others still return,
  * KeyboardInterrupt propagates to abort the run,
  * the parallel_scope_read flag routes acquire_shot_dispatch,
  * parallel arming overlaps the slaves (same barrier proof), arms the master
    last, serializes when the flag is off, and propagates arm errors,
  * single_shot_acquisition and the spooled callers route through the dispatcher.

Run:

    python -m unittest tests.test_daq_parallel
"""

import threading
import unittest
from configparser import ConfigParser
from unittest import mock

import numpy as np

from acquisition import scope_runner
from acquisition.scope_runner import MultiScopeAcquisition


class FakeScope:
    """Minimal stand-in for a LeCroy_Scope handle."""

    def __init__(self, traces, raise_exc=None, arm_raise=None, ready=True):
        self._traces = list(traces)
        self.raise_exc = raise_exc
        self.arm_raise = arm_raise
        # When False, the scope never reports trigger-ready (INR bit clear), so
        # arm_single_and_confirm returns ready=False -> the slave arm aborts.
        self.ready = ready

    def displayed_traces(self):
        return list(self._traces)

    def displayed_channels(self):
        # Only the Cn traces are real acquisition channels.
        return tuple(t for t in self._traces if t.startswith("C"))

    def validate_channel(self, ch):
        return ch

    def clear_sweeps(self):
        pass

    def sweeps_per_acq(self, channel):
        # A fresh acquisition has always completed in the fake (counter >= 1).
        return 1

    def wait_for_stop_then_complete(self, channel, timeout=100, poll=0.02):
        # Fake is always STOPped with a fresh sweep available.
        return self.sweeps_per_acq(channel) >= 1

    def arm_single(self, channel=None):
        # Mirrors LeCroyScope.arm_single: clear + SINGLE, return ref channel.
        self.clear_sweeps()
        self.set_trigger_mode("SINGLE")
        chans = self.displayed_channels()
        return channel if channel is not None else (chans[0] if chans else None)

    def read_inr(self):
        return 0x2000 if self.ready else 0

    def wait_for_trigger_ready(self, timeout=5.0, poll=0.01):
        return bool(self.read_inr() & 0x2000)

    def arm_single_and_confirm(self, channel=None, ready_timeout=5.0):
        # Mirrors LeCroyScope.arm_single_and_confirm: arm, then confirm ready.
        ch = self.arm_single(channel=channel)
        return ch, self.wait_for_trigger_ready(timeout=ready_timeout)

    def arm_master_single(self, channel=None):
        # Mirrors LeCroyScope.arm_master_single: arm exactly once.
        return self.arm_single(channel=channel)

    def set_trigger_mode(self, mode):
        # Empty-string query path: report STOP at once.
        if mode == "":
            return "STOP"
        # 'SINGLE' arm path.
        if self.arm_raise is not None:
            raise self.arm_raise
        return "SINGLE"

    def acquire(self, tr, raw=True):
        if self.raise_exc is not None:
            raise self.raise_exc
        data = np.full(8, ord(tr[-1]), dtype=np.int16)
        header = b"hdr-" + tr.encode()
        return data, header


# Generous upper bound for the barrier rendezvous below: a correct (parallel)
# implementation releases the barrier in microseconds; only a regressed
# (sequential) implementation waits this long before failing.
_BARRIER_TIMEOUT = 10.0


class SyncScope(FakeScope):
    """FakeScope whose read/arm rendezvous on barriers and log their order.

    A ``threading.Barrier(n)`` only releases once n participants are blocked in
    ``wait()`` simultaneously, so a test that completes proves the operations
    truly overlapped -- a deterministic concurrency proof with no wall-clock
    thresholds to flake under load. If a regression serializes the calls, the
    barrier times out and raises ``BrokenBarrierError``, failing the test
    loudly (read errors skip the scope; arm errors propagate from the join).

    ``arm_log`` (a shared list) records ``(name, "start"/"end")`` events around
    each arm so tests can assert ordering (master last, sequential arms not
    interleaved). Appends hold the GIL, so the log is thread-safe.
    """

    def __init__(self, traces, name=None, read_barrier=None, arm_barrier=None,
                 arm_log=None):
        super().__init__(traces)
        self.name = name
        self.read_barrier = read_barrier
        self.arm_barrier = arm_barrier
        self.arm_log = arm_log

    def acquire(self, tr, raw=True):
        if self.read_barrier is not None:
            self.read_barrier.wait(timeout=_BARRIER_TIMEOUT)
        return super().acquire(tr, raw)

    def set_trigger_mode(self, mode):
        if mode != "SINGLE":
            return super().set_trigger_mode(mode)
        if self.arm_log is not None:
            self.arm_log.append((self.name, "start"))
        if self.arm_barrier is not None:
            self.arm_barrier.wait(timeout=_BARRIER_TIMEOUT)
        result = super().set_trigger_mode(mode)
        if self.arm_log is not None:
            self.arm_log.append((self.name, "end"))
        return result


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
    msa._arm_channels = {}
    msa.scope_ips = {name: "0.0.0.0" for name in scopes}
    msa.parallel_scope_read = parallel_read
    msa.parallel_scope_arm = parallel_arm
    msa.parallel_spool_write = True
    msa.slave_ready_timeout = 5.0
    msa._sync_warned = False
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
        # Each scope's read blocks on a shared 2-party barrier, so both reads
        # must be in flight at the same time for either to complete. If a
        # regression serialized the reads, the barrier would break and both
        # scopes would be skipped (read errors are swallowed per scope), so
        # asserting both scopes returned data proves the overlap.
        barrier = threading.Barrier(2)
        scopes = {
            "A": SyncScope(["C1"], read_barrier=barrier),
            "B": SyncScope(["C2"], read_barrier=barrier),
        }
        active = {"A": 0, "B": 0}
        out = _make_msa(scopes).acquire_shot_parallel(active, 1, verbose=False)
        self.assertEqual(set(out), {"A", "B"},
                         "reads did not overlap (barrier never released)")

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
    def test_slaves_overlap_master_armed_last(self):
        # The two slaves' arms block on a shared 2-party barrier, so the call
        # only completes if both slave arms were in flight together (a
        # serialized regression breaks the barrier, and arm errors propagate
        # from the join -- the call would raise). The event log then proves the
        # master armed strictly AFTER both slaves finished.
        barrier = threading.Barrier(2)
        events = []
        scopes = {
            "A": SyncScope(["C1"], name="A", arm_barrier=barrier, arm_log=events),
            "B": SyncScope(["C2"], name="B", arm_barrier=barrier, arm_log=events),
            "C": SyncScope(["C3"], name="C", arm_log=events),  # master (last in scope_ips)
        }
        msa = _make_msa(scopes, parallel_arm=True)
        msa.arm_scopes_for_trigger(list(scopes.keys()), verbose=False)
        self.assertEqual(sorted(e[0] for e in events[:4]), ["A", "A", "B", "B"],
                         f"slaves did not arm before the master: {events}")
        self.assertEqual(events[4:], [("C", "start"), ("C", "end")],
                         f"master was not armed last: {events}")

    def test_master_is_last_in_scope_ips(self):
        # The last scope listed in scope_ips is the master regardless of the
        # active_scopes iteration order.
        scopes = {
            "A": FakeScope(["C1"]),
            "B": FakeScope(["C2"]),
            "C": FakeScope(["C3"]),
        }
        msa = _make_msa(scopes)
        self.assertEqual(msa._master_scope({"A": 0, "B": 0, "C": 0}), "C")
        # If the master is inactive, fall back to the last active per scope_ips.
        self.assertEqual(msa._master_scope({"A": 0, "B": 0}), "B")

    def test_sequential_arm_is_serialized(self):
        # With parallel_scope_arm off, arms must happen one at a time in
        # scope_ips order (slaves first, master last): each arm's start/end
        # pair is adjacent in the event log, never interleaved with another.
        events = []
        scopes = {
            "A": SyncScope(["C1"], name="A", arm_log=events),
            "B": SyncScope(["C2"], name="B", arm_log=events),
            "C": SyncScope(["C3"], name="C", arm_log=events),  # master
        }
        msa = _make_msa(scopes, parallel_arm=False)
        msa.arm_scopes_for_trigger(list(scopes.keys()), verbose=False)
        self.assertEqual(events, [("A", "start"), ("A", "end"),
                                  ("B", "start"), ("B", "end"),
                                  ("C", "start"), ("C", "end")])

    def test_arm_error_propagates(self):
        # A failed slave arm must abort the shot, not be silently swallowed.
        # B and C are slaves (A would be master as last? no -- C is last). Make a
        # non-master scope raise so the error surfaces from the slave arm join.
        scopes = {
            "A": FakeScope(["C1"], arm_raise=RuntimeError("arm failed")),
            "B": FakeScope(["C2"]),
            "C": FakeScope(["C3"]),  # master
        }
        msa = _make_msa(scopes, parallel_arm=True)
        with self.assertRaises(RuntimeError):
            msa.arm_scopes_for_trigger(list(scopes.keys()), verbose=False)

    def test_slave_not_ready_aborts_shot(self):
        # A slave whose INR never reports trigger-ready must abort the shot
        # (raise) so the master is never armed against an unready slave.
        scopes = {
            "A": FakeScope(["C1"], ready=False),  # slave never becomes ready
            "B": FakeScope(["C2"]),  # master (last)
        }
        msa = _make_msa(scopes, parallel_arm=False)
        msa.slave_ready_timeout = 0.05  # keep the test fast
        with self.assertRaises(RuntimeError):
            msa.arm_scopes_for_trigger(list(scopes.keys()), verbose=False)

    def test_master_not_gated_by_its_own_readiness(self):
        # The master is armed via arm_master_single (no INR ready wait), so a
        # master that does not report ready does NOT abort the shot.
        scopes = {
            "A": FakeScope(["C1"]),  # slave, ready
            "B": FakeScope(["C2"], ready=False),  # master, not "ready" but fine
        }
        msa = _make_msa(scopes, parallel_arm=False)
        msa.slave_ready_timeout = 0.05
        # Should not raise.
        msa.arm_scopes_for_trigger(list(scopes.keys()), verbose=False)


class SyncTimestampWarningTest(unittest.TestCase):
    def _run_dispatch_with_stamps(self, stamps):
        """Run acquire_shot_dispatch once with patched per-scope trigger stamps.

        ``stamps`` maps scope name -> trigger timestamp returned by the patched
        wavedesc_trigger_timestamp (keyed off the fake header bytes, which encode
        the trace name and thus the scope). Returns captured stdout.
        """
        scopes = {"A": FakeScope(["C1"]), "B": FakeScope(["C2"])}  # B is master
        active = {"A": 0, "B": 0}
        msa = _make_msa(scopes)

        # Fake header -> scope name. FakeScope.acquire returns b"hdr-<trace>",
        # and trace C1 belongs to A, C2 to B.
        trace_to_scope = {"C1": "A", "C2": "B"}

        def fake_translate(self_scope, header_bytes):
            return header_bytes  # pass through; the patched ts reads it

        def fake_ts(hdr):
            tr = hdr.decode().split("-", 1)[1]
            return stamps[trace_to_scope[tr]]

        import io
        from contextlib import redirect_stdout

        for fs in scopes.values():
            fs.translate_header_bytes = (
                lambda hb, _f=fake_translate, _s=fs: _f(_s, hb))

        buf = io.StringIO()
        with mock.patch("lab_scopes.lecroy.wavedesc_trigger_timestamp",
                        side_effect=fake_ts):
            with redirect_stdout(buf):
                msa.acquire_shot_dispatch(active, 1, verbose=False)
                # Second shot must never warn again even if still desynced.
                msa.acquire_shot_dispatch(active, 2, verbose=False)
        return buf.getvalue(), msa

    def test_warns_once_when_desynced(self):
        out, msa = self._run_dispatch_with_stamps({"A": 100.0, "B": 100.9})
        self.assertEqual(out.count("[sync warning]"), 1)
        self.assertTrue(msa._sync_warned)

    def test_no_warning_when_synced(self):
        out, msa = self._run_dispatch_with_stamps({"A": 100.0, "B": 100.001})
        self.assertNotIn("[sync warning]", out)
        self.assertTrue(msa._sync_warned)


if __name__ == "__main__":
    unittest.main()
