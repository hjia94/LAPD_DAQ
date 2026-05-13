"""Mock/fake devices used by automated tests and dry runs."""

from __future__ import annotations

import time

import numpy as np

from lapd_daq.models import (
    AchievedPosition,
    CameraShot,
    PlannedPosition,
    ScopeShot,
    ScopeTrace,
)


class FakeScopeDevice:
    """Deterministic scope for tests and CLI dry runs."""

    def __init__(self, name: str = "FakeScope", channels: tuple[str, ...] = ("C1", "C2"),
                 points: int = 16):
        self.name = name
        self.channels = channels
        self.points = points
        self._time_array = np.linspace(0.0, 1.0e-6, points, endpoint=False)
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def initialize(self) -> None:
        if not self.connected:
            self.connect()

    def arm(self) -> None:
        return None

    def acquire(self, shot_num: int) -> ScopeShot:
        traces = []
        header = _fake_lecroy_header(self.points)
        for index, channel in enumerate(self.channels):
            raw = np.arange(self.points, dtype=np.int16) + shot_num + index
            traces.append(ScopeTrace(channel=channel, raw=raw, header=header))
        return ScopeShot(scope_name=self.name, traces=traces, acquisition_time=time.ctime())

    def time_array(self):
        return self._time_array

    def metadata(self) -> dict[str, object]:
        return {
            "description": "Fake scope for mock-only tests",
            "ip_address": "mock",
            "scope_type": "FakeScopeDevice",
        }

    def close(self) -> None:
        self.connected = False


class FakeMotionDevice:
    def __init__(self):
        self.positions: list[AchievedPosition | None] = []

    def connect(self) -> None:
        return None

    def move_to(self, position: PlannedPosition | None) -> AchievedPosition | None:
        if position is None:
            achieved = None
        else:
            achieved = AchievedPosition(coordinates=dict(position.coordinates))
        self.positions.append(achieved)
        return achieved

    def close(self) -> None:
        return None

    def metadata(self) -> dict[str, object]:
        return {"device_type": "FakeMotionDevice"}


class FakeCameraDevice:
    def connect(self) -> None:
        return None

    def arm(self, shot_num: int) -> None:
        return None

    def complete(self, shot_num: int) -> CameraShot:
        return CameraShot(shot_num=shot_num, file_name=f"mock_shot{shot_num:03d}.cine", timestamp=time.time())

    def close(self) -> None:
        return None

    def metadata(self) -> dict[str, object]:
        return {"device_type": "FakeCameraDevice"}


class FakeTriggerDevice:
    def __init__(self):
        self.triggered_shots: list[int] = []

    def connect(self) -> None:
        return None

    def trigger(self, shot_num: int) -> None:
        self.triggered_shots.append(shot_num)

    def close(self) -> None:
        return None

    def metadata(self) -> dict[str, object]:
        return {"device_type": "FakeTriggerDevice"}


def _fake_lecroy_header(points: int) -> bytes:
    try:
        from lab_scopes.lecroy import LeCroyHeader

        header = LeCroyHeader()
        return header.generate_test_data(NTimes=points)
    except Exception:
        return bytes(346)
