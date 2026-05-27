"""Acquisition engines used by the Data_Run_*.py entry-point scripts.

This package re-exports the public surface that callers previously imported
from the top-level `multi_scope_acquisition` and `acquisition_bmotion` modules,
so callers only need to know about the `acquisition` package.
"""

from .bmotion_config import BmotionSelection, resolve_bmotion_selection
from .config import load_experiment_config
from .scope_runner import (
    MultiScopeAcquisition,
    run_acquisition,
    single_shot_acquisition,
    single_shot_acquisition_45,
)


def run_acquisition_bmotion(*args, **kwargs):
    from .bmotion import run_acquisition_bmotion as _run_acquisition_bmotion

    return _run_acquisition_bmotion(*args, **kwargs)


def run_acquisition_bmotion_spooled(*args, **kwargs):
    from .bmotion import (
        run_acquisition_bmotion_spooled as _run_acquisition_bmotion_spooled,
    )

    return _run_acquisition_bmotion_spooled(*args, **kwargs)


__all__ = [
    "BmotionSelection",
    "MultiScopeAcquisition",
    "load_experiment_config",
    "resolve_bmotion_selection",
    "run_acquisition",
    "run_acquisition_bmotion",
    "run_acquisition_bmotion_spooled",
    "single_shot_acquisition",
    "single_shot_acquisition_45",
]
