"""Compatibility imports for the renamed device protocol module."""

from .protocols import CameraDevice, MotionDevice, ScopeDevice, TriggerDevice

__all__ = ["CameraDevice", "MotionDevice", "ScopeDevice", "TriggerDevice"]
