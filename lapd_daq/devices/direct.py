"""Direct hardware adapters used before EPICS owns device control."""

from __future__ import annotations

import time

import numpy as np

from lapd_daq.models import AchievedPosition, PlannedPosition, ScopeShot, ScopeTrace


class LabScopesLeCroyAdapter:
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
            "adapter": "LabScopesLeCroyAdapter",
        }

    def close(self) -> None:
        if self.scope is not None:
            self.scope.__exit__(None, None, None)
            self.scope = None

    def _require_scope(self):
        if self.scope is None:
            raise RuntimeError(f"Scope {self.name} has not been initialized")
        return self.scope


class LegacyMotorAdapter:
    """Thin adapter around the existing legacy motor-control objects."""

    def __init__(self, controller):
        self.controller = controller

    def connect(self) -> None:
        return None

    def move_to(self, position: PlannedPosition | None) -> AchievedPosition | None:
        if position is None:
            return None
        coords = position.coordinates
        if "z" in coords:
            self.controller.probe_positions = (coords["x"], coords["y"], coords["z"])
        else:
            self.controller.probe_positions = (coords["x"], coords["y"])
        self.controller.wait_for_motion_complete()
        reported = self.controller.probe_positions
        if len(reported) == 2:
            achieved = {"x": float(reported[0]), "y": float(reported[1])}
        else:
            achieved = {"x": float(reported[0]), "y": float(reported[1]), "z": float(reported[2])}
        return AchievedPosition(coordinates=achieved)

    def close(self) -> None:
        return None

    def metadata(self) -> dict[str, object]:
        return {"adapter": "LegacyMotorAdapter"}


class BapsfMotionAdapter:
    """Placeholder boundary for bapsf_motion-backed control."""

    def __init__(self, run_manager, motion_group_keys):
        self.run_manager = run_manager
        self.motion_group_keys = list(motion_group_keys)

    def connect(self) -> None:
        return None

    def move_to(self, position: PlannedPosition | None) -> AchievedPosition | None:
        raise NotImplementedError(
            "BapsfMotionAdapter boundary exists, but the existing acquisition.bmotion "
            "workflow has not yet been migrated into the new run engine."
        )

    def close(self) -> None:
        terminate = getattr(self.run_manager, "terminate", None)
        if terminate is not None:
            terminate()

    def metadata(self) -> dict[str, object]:
        return {"adapter": "BapsfMotionAdapter", "motion_group_keys": self.motion_group_keys}


class PhantomCameraAdapter:
    """Adapter around drivers.phantom_recorder.PhantomRecorder."""

    def __init__(self, recorder, experiment_name: str = ""):
        self.recorder = recorder
        self.experiment_name = experiment_name or "lapd_daq"

    def connect(self) -> None:
        return None

    def arm(self, shot_num: int) -> None:
        self.recorder.start_recording(shot_num)

    def complete(self, shot_num: int):
        from lapd_daq.models import CameraShot

        timestamp = self.recorder.wait_for_recording_completion()
        file_name = f"{self.experiment_name}_shot{shot_num:03d}.cine"
        rec_cine = self.recorder.save_cine(file_name)
        self.recorder.wait_for_save_completion(rec_cine)
        return CameraShot(shot_num=shot_num, file_name=file_name, timestamp=timestamp)

    def close(self) -> None:
        self.recorder.cleanup()

    def metadata(self) -> dict[str, object]:
        return {"adapter": "PhantomCameraAdapter"}


class PiTriggerAdapter:
    """Adapter around the Raspberry Pi trigger client."""

    def __init__(self, trigger_client):
        self.trigger_client = trigger_client

    def connect(self) -> None:
        status = getattr(self.trigger_client, "get_status", None)
        if status is not None:
            status()

    def trigger(self, shot_num: int) -> None:
        self.trigger_client.send_trigger()

    def close(self) -> None:
        close = getattr(self.trigger_client, "close", None)
        if close is not None:
            close()

    def metadata(self) -> dict[str, object]:
        return {"adapter": "PiTriggerAdapter"}


def _stop_triggering(scope, retry: int = 500) -> None:
    for _ in range(retry):
        current_mode = scope.set_trigger_mode("")
        if current_mode[0:4] == "STOP":
            return
        time.sleep(0.05)
    raise RuntimeError("Scope did not enter STOP state")
