"""Reading and analysis tools for LAPD_DAQ HDF5 data.

Currently provides inspection/validation utilities for files written by
``Data_Run_bmotion.py``. Future analysis code (per-position profiles,
Isat -> density, etc.) belongs in this package as well.

Public functions live in :mod:`read_and_analyze.read_bmotion_data`::

    from read_and_analyze import validate_file, print_summary, plot_traces, read_positions

Imports are lazy so that ``python -m read_and_analyze.read_bmotion_data`` does
not trigger a re-import of the module while it is executing.
"""

__all__ = [
    "read_positions",
    "validate_file",
    "print_summary",
    "plot_traces",
]


def __getattr__(name):
    if name in __all__:
        from . import read_bmotion_data
        return getattr(read_bmotion_data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
