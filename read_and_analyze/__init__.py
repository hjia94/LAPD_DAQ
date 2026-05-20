"""Reading and analysis tools for LAPD_DAQ HDF5 data.

Currently provides inspection/validation utilities for files written by
``Data_Run_bmotion.py``. Future analysis code (per-position profiles,
Isat -> density, etc.) belongs in this package as well.

Public functions live in :mod:`read_and_analyze.read_bmotion_data`::

    from read_and_analyze import validate_file, print_summary, plot_traces, read_positions

Imports are lazy so that ``python -m read_and_analyze.read_bmotion_data`` does
not trigger a re-import of the module while it is executing.
"""

_READER_NAMES = (
    "read_positions",
    "validate_file",
    "print_summary",
    "plot_traces",
)
_FLUCTUATION_NAMES = (
    "find_quiet_window",
    "plot_quiet_window",
)

__all__ = [*_READER_NAMES, *_FLUCTUATION_NAMES]


def __getattr__(name):
    if name in _READER_NAMES:
        from . import read_bmotion_data
        return getattr(read_bmotion_data, name)
    if name in _FLUCTUATION_NAMES:
        from . import analysis_fluctuation
        return getattr(analysis_fluctuation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
