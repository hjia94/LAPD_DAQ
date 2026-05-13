"""Small data models shared by acquisition, devices, and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlannedPosition:
    """A requested probe position for one shot."""

    coordinates: dict[str, float] = field(default_factory=dict)
    label: str = ""


@dataclass(frozen=True)
class AchievedPosition:
    """The position reported back by the motion system."""

    coordinates: dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    message: str = ""


@dataclass(frozen=True)
class ShotPlan:
    """The experiment intent for one shot."""

    shot_num: int
    position: PlannedPosition | None = None
    repeat_index: int = 0
    duplicate_index: int = 0


@dataclass
class ScopeTrace:
    """One channel/trace acquired from one scope."""

    channel: str
    raw: np.ndarray
    header: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScopeShot:
    """All traces acquired from one scope for one shot."""

    scope_name: str
    traces: list[ScopeTrace]
    acquisition_time: str = ""
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class CameraShot:
    """Camera metadata associated with one shot."""

    shot_num: int
    file_name: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShotResult:
    """What actually happened for one shot."""

    plan: ShotPlan
    scope_shots: list[ScopeShot] = field(default_factory=list)
    achieved_position: AchievedPosition | None = None
    camera_shot: CameraShot | None = None
    status: str = "ok"
    message: str = ""
