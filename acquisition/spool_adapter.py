"""Glue between the bmotion acquisition path and the shared spool format.

The acquire process builds the full HDF5 skeleton (experiment/scope metadata,
time arrays, ``Control/Positions``) directly on the destination file before any
shots, so this adapter is only responsible for the *per-shot* mapping:

* **Acquire side** — turn this path's native ``all_data`` (and per-motion-group
  positions) into a :class:`spooling.ShotPayload`.
* **Offload side** — turn a :class:`spooling.ShotPayload` back into the existing
  :mod:`acquisition.hdf5_writer` shot write + the per-shot bmotion position row,
  filling the already-created HDF5 so the result matches the in-process writer.

Keeping this mapping in one module means :mod:`spooling.spool_format` stays
storage-agnostic and the offload runner can dispatch by the ``"writer"`` tag in
the run metadata.
"""

import time

import h5py
import numpy as np

from . import config as config_module
from . import hdf5_writer

WRITER_TAG = "acquisition"


# --------------------------------------------------------------------------- #
# Acquire side
# --------------------------------------------------------------------------- #
def all_data_to_payload(all_data, shot_num, coordinates):
    """Build a ShotPayload from ``all_data`` and a positions mapping.

    ``all_data`` is ``{scope_name: (traces, data, headers)}`` as produced by
    ``MultiScopeAcquisition.acquire_shot``; ``coordinates`` is the per-shot
    position payload (e.g. ``{mg_name: (x, y)}``) or ``None``.
    """
    from spooling import ShotPayload, TracePayload

    payload = ShotPayload(
        shot_num=shot_num,
        coordinates=coordinates,
        acquisition_time=time.ctime(),
    )
    for scope_name, (traces, data, headers) in all_data.items():
        scope_traces = []
        for tr in traces:
            if tr not in data:
                continue
            scope_traces.append(
                TracePayload(
                    channel=tr,
                    data=np.asarray(data[tr], dtype=np.int16),
                    header=bytes(headers[tr]),
                )
            )
        payload.traces[scope_name] = scope_traces
    return payload


def skipped_payload(shot_num, reason, coordinates=None):
    """Build a ShotPayload marking a shot as skipped."""
    from spooling import ShotPayload

    return ShotPayload(
        shot_num=shot_num,
        coordinates=coordinates,
        acquisition_time=time.ctime(),
        skipped=True,
        skip_reason=str(reason),
    )


# --------------------------------------------------------------------------- #
# Offload side
# --------------------------------------------------------------------------- #
def write_shot(hdf5_path, payload, meta):
    """Write one ShotPayload's scope data + positions into the HDF5 file.

    Scope data (or the skip marker) and the position row are written in a single
    HDF5 open so each shot opens the file once on the offload hot path.
    """
    if payload.skipped:
        _write_skip(hdf5_path, payload, meta)
        with h5py.File(hdf5_path, "a") as f:
            _write_positions(f, payload, meta)
        return

    all_data = _payload_to_all_data(payload)
    with h5py.File(hdf5_path, "a", **hdf5_writer.SHOT_WRITE_OPEN_KWARGS) as f:
        hdf5_writer._write_shot_data_into(f, all_data, payload.shot_num)
        _write_positions(f, payload, meta)


def finalize(hdf5_path, meta, final_shot_num):
    """Write the per-scope shot_count attribute (run finalization).

    Also overwrite the experiment description from ``description.txt`` now that
    all shots are written: this is where the spooled run actually finishes, so a
    description edited before/during the run (and up until the offload drains) is
    captured here. Guarded so a description read can never fail the finalize.
    """
    hdf5_writer.record_shot_count(
        hdf5_path, meta["config_scope_names"], final_shot_num
    )
    description_path = meta.get("description_path")
    if description_path:
        try:
            hdf5_writer.write_description(
                hdf5_path, config_module.read_description_file(description_path))
        except Exception as e:
            print(f"Warning: could not write final description: {e}")


def mark_shot_failed(hdf5_path, meta, shot_num, reason):
    """Replace a poison shot's HDF5 group with a failed marker (quarantine)."""
    hdf5_writer.mark_shot_failed_for_scopes(
        hdf5_path, meta["config_scope_names"], shot_num, reason
    )


def _payload_to_all_data(payload):
    """ShotPayload -> the ``all_data`` dict hdf5_writer.write_shot_data expects."""
    all_data = {}
    for scope_name, traces in payload.traces.items():
        tr_names = [t.channel for t in traces]
        data = {t.channel: t.data for t in traces}
        headers = {t.channel: t.header for t in traces}
        all_data[scope_name] = (tr_names, data, headers)
    return all_data


def _write_skip(hdf5_path, payload, meta):
    hdf5_writer.mark_shot_skipped_for_scopes(
        hdf5_path, meta["config_scope_names"], payload.shot_num,
        payload.skip_reason,
    )


def _write_positions(f, payload, meta):
    """Write per-motion-group positions into open HDF5 ``f``.

    Mirrors record_bmotion_positions. ``f`` is an already-open ``h5py.File`` so
    the position row lands in the same open as the shot's scope data.
    """
    coords = payload.coordinates
    if not coords:
        return
    for mg_name, xy in coords.items():
        ds_path = f"Control/Positions/{mg_name}/positions_array"
        if ds_path not in f:
            continue
        ds = f[ds_path]
        ds[payload.shot_num - 1] = (payload.shot_num, xy[0], xy[1])
