"""Glue between the legacy grid (PositionManager) acquisition path and the spool.

Mirrors :mod:`acquisition.spool_adapter` but for the direct-grid path driven by
:class:`motion.position_manager.PositionManager`. As in the bmotion path, the
acquire process builds the full HDF5 skeleton up front (experiment/scope
metadata, time arrays, and the single ``/Control/Positions`` group), so this
adapter only handles the *per-shot* mapping:

* **Acquire side** — reuse :func:`acquisition.spool_adapter.all_data_to_payload`
  / :func:`skipped_payload` (the per-shot scope payload is identical to the
  bmotion path); the per-shot ``coordinates`` here is the grid position dict
  ``{'x': .., 'y': .., 'z': .. | None}``.
* **Offload side** — write each :class:`spooling.ShotPayload` into the
  already-created HDF5 via :func:`acquisition.hdf5_writer.write_shot_data`, plus
  the single ``/Control/Positions/positions_array`` row (2-D or 3-D depending on
  the run's ``nz``).

The offload runner dispatches here on the ``"grid"`` writer tag.
"""

import h5py

from . import hdf5_writer
from . import spool_adapter

WRITER_TAG = "grid"

# Re-export the per-shot payload builders unchanged: the scope-trace mapping is
# identical to the bmotion path; only the coordinates payload differs.
all_data_to_payload = spool_adapter.all_data_to_payload
skipped_payload = spool_adapter.skipped_payload
channel_descriptions = spool_adapter.channel_descriptions


# --------------------------------------------------------------------------- #
# Offload side
# --------------------------------------------------------------------------- #
def write_shot(hdf5_path, payload, meta):
    """Write one ShotPayload's scope data + grid position into the HDF5 file."""
    if payload.skipped:
        spool_adapter._write_skip(hdf5_path, payload, meta)
        _write_positions(hdf5_path, payload, meta)
        return

    all_data = spool_adapter._payload_to_all_data(payload)
    descriptions = spool_adapter._descriptions_for(all_data, meta)
    hdf5_writer.write_shot_data(hdf5_path, all_data, payload.shot_num, descriptions)
    _write_positions(hdf5_path, payload, meta)


def finalize(hdf5_path, meta, final_shot_num):
    """Write the per-scope shot_count attribute (run finalization)."""
    hdf5_writer.record_shot_count(
        hdf5_path, meta["config_scope_names"], final_shot_num
    )


def mark_shot_failed(hdf5_path, meta, shot_num, reason):
    """Replace a poison shot's HDF5 group with a failed marker (quarantine)."""
    hdf5_writer.mark_shot_failed_for_scopes(
        hdf5_path, meta["config_scope_names"], shot_num, reason
    )


def _write_positions(hdf5_path, payload, meta):
    """Write the single grid positions_array row, mirroring update_position_hdf5.

    ``payload.coordinates`` is ``{'x':.., 'y':.., 'z':..|None}``; ``None`` (e.g.
    a stationary run with no motor) writes nothing.
    """
    coords = payload.coordinates
    if not coords:
        return
    with h5py.File(hdf5_path, "a") as f:
        ds_path = "/Control/Positions/positions_array"
        if ds_path not in f:
            return
        pos_arr = f[ds_path]
        shot_num = payload.shot_num
        if meta.get("nz") is None:
            pos_arr[shot_num - 1] = (shot_num, coords["x"], coords["y"])
        else:
            pos_arr[shot_num - 1] = (shot_num, coords["x"], coords["y"], coords["z"])
