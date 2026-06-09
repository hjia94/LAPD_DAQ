"""Storage-agnostic spool format shared by acquisition and offload processes.

The acquisition process writes each shot to a fast local disk in this format;
a separate offload process reads it back and writes the final HDF5 file on a
slower/larger disk. Neither side imports the other's HDF5 layout — that lives
behind per-path adapters (e.g. ``acquisition/spool_adapter.py``).
"""

from .spool_format import (
    ShotPayload,
    TracePayload,
    is_disk_full_error,
    iter_ready_shots,
    pending_shot_count,
    quarantine_shot,
    read_run_complete,
    read_run_metadata,
    read_shot,
    run_complete_exists,
    write_run_complete,
    write_run_metadata,
    write_shot,
    write_shot_with_disk_full_retry,
)

__all__ = [
    "ShotPayload",
    "TracePayload",
    "is_disk_full_error",
    "iter_ready_shots",
    "pending_shot_count",
    "quarantine_shot",
    "read_run_complete",
    "read_run_metadata",
    "read_shot",
    "run_complete_exists",
    "write_run_complete",
    "write_run_metadata",
    "write_shot",
    "write_shot_with_disk_full_retry",
]
