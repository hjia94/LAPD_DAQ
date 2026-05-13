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
        self._require_scope().set_trigger_mode("SINGLE")

    def acquire(self, shot_num: int) -> ScopeShot:
        scope = self._require_scope()
        traces = []
        for trace_name in scope.displayed_traces():
            _stop_triggering(scope)
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


def _stop_triggering(scope, retry: int = 500) -> None:
    for _ in range(retry):
        current_mode = scope.set_trigger_mode("")
        if current_mode[0:4] == "STOP":
            return
        time.sleep(0.05)
    raise RuntimeError("Scope did not enter STOP state")


LabScopesLeCroyAdapter = LabScopesLeCroyScopeAdapter
