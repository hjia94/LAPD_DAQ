"""Direct oscilloscope adapters backed by the lab_scopes package."""

from __future__ import annotations

import time

import numpy as np

from lapd_daq.models import ScopeShot, ScopeTrace


class LabScopesLeCroyScopeAdapter:
    """LeCroy scope adapter backed by the external lab_scopes package."""

    def __init__(self, name: str, ip_address: str, description: str = "", timeout: float = 30.0):
        self.name = name
        self.ip_address = ip_address
        self.description = description
        self.timeout = timeout
        self.scope = None
        self._time_array = None
        # Reference channel chosen at arm time (arm_single), polled for the
        # fresh-sweep completion check during acquire.
        self._arm_channel = None

    def connect(self) -> None:
        from lab_scopes.lecroy import LeCroyScope

        self.scope = LeCroyScope(self.ip_address, verbose=False, timeout=self.timeout)

    def initialize(self) -> None:
        if self.scope is None:
            self.connect()
        self.scope.set_trigger_mode("SINGLE")
        traces = self.scope.displayed_traces()
        if not traces:
            raise RuntimeError(f"No displayed traces found on scope {self.name}")
        self._time_array = self.scope.time_array(traces[0])

    def arm(self) -> None:
        # This adapter drives a single scope, which is therefore the master:
        # arm_master_single clears the sweep counter then sets SINGLE with a
        # strict SIN check (no slaves to gate, so no INR readiness wait needed).
        # It returns the reference channel polled for the fresh-sweep completion
        # check; the channel is cached so later shots skip channel discovery (a
        # per-shot scan of C1..C4 :TRACE? queries).
        self._arm_channel = self._require_scope().arm_master_single(channel=self._arm_channel)

    def acquire(self, shot_num: int) -> ScopeShot:
        scope = self._require_scope()
        # Wait once for a fresh acquisition (sweep-counter based) before reading
        # every trace -- one completed sweep means all channels are ready.
        _stop_triggering(scope, self._arm_channel)
        traces = []
        for trace_name in scope.displayed_traces():
            raw, header = scope.acquire(trace_name, raw=True)
            traces.append(
                ScopeTrace(
                    channel=trace_name,
                    raw=np.asarray(raw, dtype=np.int16),
                    header=bytes(header),
                )
            )
        return ScopeShot(scope_name=self.name, traces=traces, acquisition_time=time.ctime())

    def time_array(self):
        return self._time_array

    def metadata(self) -> dict[str, object]:
        idn = getattr(self.scope, "idn_string", "") if self.scope is not None else ""
        return {
            "description": self.description,
            "ip_address": self.ip_address,
            "scope_type": idn or "LeCroy",
            "adapter": "LabScopesLeCroyScopeAdapter",
        }

    def close(self) -> None:
        if self.scope is not None:
            self.scope.__exit__(None, None, None)
            self.scope = None

    def _require_scope(self):
        if self.scope is None:
            raise RuntimeError(f"Scope {self.name} has not been initialized")
        return self.scope


def _stop_triggering(scope, channel=None, timeout: float = 25.0) -> None:
    """Block until the scope is STOPped AND a *fresh* single acquisition landed.

    Two-stage check (``wait_for_stop_then_complete``): wait for ``TRIG_MODE STOP``
    as a fast hint, then confirm via the monotonic sweep counter (cleared by
    ``arm_master_single``) that a new edge was actually captured. STOP alone is
    ambiguous (read both before any trigger and after a previous one), so the
    counter is the authority -- a leftover STOP from a prior shot reads counter 0
    and is never mistaken for fresh data. ``channel`` is the reference channel
    cleared at arm time; if None the first displayed channel is used.
    """
    if channel is None:
        channels = scope.displayed_channels()
        channel = channels[0] if channels else None
    if channel is None:
        raise RuntimeError("Scope has no displayed channel to poll for completion")
    if not scope.wait_for_stop_then_complete(channel, timeout=timeout):
        raise RuntimeError("Scope did not complete a fresh acquisition")


LabScopesLeCroyAdapter = LabScopesLeCroyScopeAdapter
