"""Acquisition engines used by the Data_Run_*.py entry-point scripts.

This package re-exports the public surface that callers previously imported
from the top-level `multi_scope_acquisition` and `acquisition_bmotion` modules,
so callers only need to know about the `acquisition` package.
"""

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

__all__ = [
    "MultiScopeAcquisition",
    "load_experiment_config",
    "run_acquisition",
    "single_shot_acquisition",
    "single_shot_acquisition_45",
    "run_acquisition_bmotion",
]
