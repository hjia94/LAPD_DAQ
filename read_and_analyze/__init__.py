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
    "find_latest_run",
    "is_run_complete",
    "resolve_data_file",
)
_FLUCTUATION_NAMES = (
    "find_quiet_window",
    "plot_quiet_window",
)
_FILTER_NAMES = (
    "plot_sample_traces",
    "load_filtered_traces",
)
_SMART_NAMES = (
    "analyze_smart_triggers",
    "plot_smart_triggers",
    "detect_glitch",
    "detect_runt",
    "detect_slew",
    "detect_interval",
)

__all__ = [*_READER_NAMES, *_FLUCTUATION_NAMES, *_FILTER_NAMES, *_SMART_NAMES]


def __getattr__(name):
    if name in _READER_NAMES:
        from . import read_bmotion_data
        return getattr(read_bmotion_data, name)
    if name in _FLUCTUATION_NAMES:
        from . import fluctuation_analysis
        return getattr(fluctuation_analysis, name)
    if name in _FILTER_NAMES:
        from . import filter_data
        return getattr(filter_data, name)
    if name in _SMART_NAMES:
        from . import smart_trigger_analysis
        return getattr(smart_trigger_analysis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
