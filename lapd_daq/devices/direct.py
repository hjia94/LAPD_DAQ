"""Compatibility imports for split direct hardware adapter modules."""

from .bmotion import BMotionAdapter, BapsfMotionAdapter
from .lab_scopes import LabScopesLeCroyAdapter, LabScopesLeCroyScopeAdapter
from .legacy_motion import LegacyMotorAdapter
from .phantom import PhantomCameraAdapter
from .pi_gpio import PiGPIOTriggerAdapter, PiTriggerAdapter

__all__ = [
    "BMotionAdapter",
    "BapsfMotionAdapter",
    "LabScopesLeCroyAdapter",
    "LabScopesLeCroyScopeAdapter",
    "LegacyMotorAdapter",
    "PhantomCameraAdapter",
    "PiGPIOTriggerAdapter",
    "PiTriggerAdapter",
]
