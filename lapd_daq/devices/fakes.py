"""Mock/fake devices used by automated tests and dry runs."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from lapd_daq.models import (
    AchievedPosition,
    CameraShot,
    PlannedPosition,
    ScopeShot,
    ScopeTrace,
)

LECROY_TRC_PREFIX_BYTES = 11
LECROY_WAVEDESC_BYTES = 346
LECROY_TRC_DATA_OFFSET = LECROY_TRC_PREFIX_BYTES + LECROY_WAVEDESC_BYTES


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


class TRCReplayScopeDevice:
    """File-backed fake scope that replays saved LeCroy .trc traces."""

    def __init__(
        self,
        name: str,
        trace_dir: str | Path,
        source_shots: tuple[int, ...] = (0, 5),
        channels: tuple[str, ...] = ("C1", "C2", "C3", "C4"),
        trace_label: str = "interf",
    ):
        self.name = name
        self.trace_dir = Path(trace_dir)
        self.source_shots = tuple(source_shots)
        self.channels = tuple(channels)
        self.trace_label = trace_label
        self.connected = False
        self._time_array = None

    def connect(self) -> None:
        self.connected = True

    def initialize(self) -> None:
        if not self.connected:
            self.connect()
        self._validate_trace_files()
        first = self._read_trace(self.channels[0], self.source_shots[0])
        self._time_array = first["time_array"]

    def arm(self) -> None:
        return None

    def acquire(self, shot_num: int) -> ScopeShot:
        source_shot = self._source_shot_for(shot_num)
        traces = []
        for channel in self.channels:
            trace = self._read_trace(channel, source_shot)
            traces.append(
                ScopeTrace(
                    channel=channel,
                    raw=trace["raw"],
                    header=trace["header"],
                    metadata={"source_file": str(trace["path"]), "source_shot": source_shot},
                )
            )
        return ScopeShot(scope_name=self.name, traces=traces, acquisition_time=time.ctime())

    def time_array(self):
        return self._time_array

    def metadata(self) -> dict[str, object]:
        return {
            "description": "TRC replay fake scope",
            "ip_address": "file://",
            "scope_type": "TRCReplayScopeDevice",
            "trace_dir": str(self.trace_dir),
            "trace_label": self.trace_label,
            "source_shots": str(self.source_shots),
        }

    def close(self) -> None:
        self.connected = False

    def _validate_trace_files(self) -> None:
        missing = [
            self._trace_path(channel, source_shot)
            for source_shot in self.source_shots
            for channel in self.channels
            if not self._trace_path(channel, source_shot).exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing TRC replay fixture(s): {missing[:3]}")

    def _source_shot_for(self, shot_num: int) -> int:
        index = shot_num - 1
        if index < 0 or index >= len(self.source_shots):
            raise IndexError(f"No TRC source shot configured for acquisition shot {shot_num}")
        return self.source_shots[index]

    def _trace_path(self, channel: str, source_shot: int) -> Path:
        return self.trace_dir / f"{channel}-{self.trace_label}-shot{source_shot:05d}.trc"

    def _read_trace(self, channel: str, source_shot: int) -> dict[str, object]:
        path = self._trace_path(channel, source_shot)
        content = path.read_bytes()
        if len(content) <= LECROY_TRC_DATA_OFFSET:
            raise ValueError(f"TRC file is too short: {path}")
        if not content[:2] == b"#9":
            raise ValueError(f"TRC file does not start with a LeCroy block prefix: {path}")

        header = content[LECROY_TRC_PREFIX_BYTES:LECROY_TRC_DATA_OFFSET]
        raw = np.frombuffer(content, dtype=np.dtype("=i2"), offset=LECROY_TRC_DATA_OFFSET).copy()
        time_array = _time_array_from_lecroy_header(header)
        return {"path": path, "header": header, "raw": raw, "time_array": time_array}


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


def _time_array_from_lecroy_header(header: bytes) -> np.ndarray:
    from lab_scopes.lecroy import LeCroyHeader

    return LeCroyHeader(header).time_array


def _fake_lecroy_header(points: int) -> bytes:
    try:
        from lab_scopes.lecroy import LeCroyHeader

        header = LeCroyHeader()
        return header.generate_test_data(NTimes=points)
    except Exception:
        return bytes(LECROY_WAVEDESC_BYTES)
