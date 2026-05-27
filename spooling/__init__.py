"""Storage-agnostic spool format shared by acquisition and offload processes.

The acquisition process writes each shot to a fast local disk in this format;
a separate offload process reads it back and writes the final HDF5 file on a
slower/larger disk. Neither side imports the other's HDF5 layout — that lives
behind per-path adapters (e.g. ``acquisition/spool_adapter.py``).
"""

from .spool_format import (
    ShotPayload,
    TracePayload,
    iter_ready_shots,
    read_run_complete,
    read_run_metadata,
    read_shot,
    run_complete_exists,
    write_run_complete,
    write_run_metadata,
    write_shot,
)

__all__ = [
    "ShotPayload",
    "TracePayload",
    "iter_ready_shots",
    "read_run_complete",
    "read_run_metadata",
    "read_shot",
    "run_complete_exists",
    "write_run_complete",
    "write_run_metadata",
    "write_shot",
]
