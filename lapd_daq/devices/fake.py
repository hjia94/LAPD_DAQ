"""Compatibility imports for the renamed fake-device module."""

from .fakes import (
    FakeCameraDevice,
    FakeMotionDevice,
    FakeScopeDevice,
    FakeTriggerDevice,
    TRCReplayScopeDevice,
)

__all__ = [
    "FakeCameraDevice",
    "FakeMotionDevice",
    "FakeScopeDevice",
    "FakeTriggerDevice",
    "TRCReplayScopeDevice",
]
