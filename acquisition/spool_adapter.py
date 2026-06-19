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
def all_data_to_payload(all_data, shot_num, coordinates, missing_scopes=None):
    """Build a ShotPayload from ``all_data`` and a positions mapping.

    ``all_data`` is ``{scope_name: (traces, data, headers)}`` as produced by
    ``MultiScopeAcquisition.acquire_shot``; ``coordinates`` is the per-shot
    position payload (e.g. ``{mg_name: (x, y)}``) or ``None``. ``missing_scopes``
    maps a scope name to the reason its data is absent for this shot (an
    arm/read failure); those scopes are recorded as skipped per-scope shot
    groups by the offload so a partial shot is preserved rather than aborted.
    """
    from spooling import ShotPayload, TracePayload

    payload = ShotPayload(
        shot_num=shot_num,
        coordinates=coordinates,
        acquisition_time=time.ctime(),
        missing=dict(missing_scopes or {}),
    )
    for scope_name, (traces, data, headers) in all_data.items():
        scope_traces = []
        for tr in traces:
            if tr not in data:
                # The scope listed this channel but returned no samples for it.
                # Warn rather than absorb silently: this is the channel-data-loss
                # class the lab_scopes v0.3.2 pin addressed, so a recurrence here
                # should be visible instead of quietly dropping the channel.
                print(f"Warning: {scope_name} channel {tr} listed but no data "
                      f"returned for shot {shot_num}; channel dropped.")
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

    # Per-scope partial: scopes that failed to arm/read/spool for this shot get
    # their own skipped shot group, so every config scope always has a shot_N
    # group (data for the good scopes, a skip marker for the missing ones).
    _write_missing_scopes(hdf5_path, payload)


def finalize(hdf5_path, meta, final_shot_num):
    """Write the per-scope shot_count attribute (run finalization).

    Also overwrite the experiment description from ``description.txt`` now that
    all shots are written: this is where the spooled run actually finishes, so a
    description edited before/during the run (and up until the offload drains) is
    captured here. Guarded so a description read can never fail the finalize.

    Finally, pad each ``positions_array`` (grown append-only during offload) back
    to the planned ``total_shots`` with zero-fill, so the finished file matches
    the historical pre-sized layout that the readers expect.
    """
    hdf5_writer.record_shot_count(
        hdf5_path, meta["config_scope_names"], final_shot_num
    )
    total_shots = meta.get("total_shots")
    if total_shots:
        _pad_positions_to_total(hdf5_path, total_shots)
    description_path = meta.get("description_path")
    if description_path:
        try:
            hdf5_writer.write_description(
                hdf5_path, config_module.read_description_file(description_path))
        except Exception as e:
            print(f"Warning: could not write final description: {e}")


def _pad_positions_to_total(hdf5_path, total_shots):
    """Rewrite every ``positions_array`` from append-tight to pre-sized layout.

    During offload ``positions_array`` is append-only: one row per recorded shot,
    in arrival order. This restores the canonical on-disk format every reader
    expects -- a ``(total_shots,)`` array where each recorded row sits at index
    ``shot_num - 1`` and shots that never recorded are left zero-filled. Each row
    carries its own ``shot_num``, so placement needs no external bookkeeping.

    Idempotent: an array already at length ``total_shots`` (a re-run finalize, or
    a legacy pre-sized file) is left untouched. Rows whose ``shot_num`` falls
    outside ``1..total_shots`` are dropped with a warning rather than corrupting
    another shot's slot -- the same out-of-range case the old indexed write could
    hit silently, now surfaced and contained at finalize.
    """
    with h5py.File(hdf5_path, "a") as f:
        pos_grp = f.get("Control/Positions")
        if pos_grp is None:
            return
        for grp in _iter_positions_groups(pos_grp):
            ds_name = "positions_array"
            if ds_name not in grp:
                continue
            tight = grp[ds_name][()]
            if len(tight) == total_shots:
                continue  # already canonical (re-finalize or legacy file)
            padded = np.zeros(total_shots, dtype=tight.dtype)
            idx = tight["shot_num"].astype(np.int64) - 1
            in_range = (idx >= 0) & (idx < total_shots)
            padded[idx[in_range]] = tight[in_range]
            for bad in tight["shot_num"][~in_range]:
                print(f"Warning: positions row shot_num {int(bad)} out of range "
                      f"1..{total_shots} in {grp.name!r}; dropped.")
            del grp[ds_name]
            grp.create_dataset(ds_name, data=padded, dtype=tight.dtype)


def _iter_positions_groups(pos_grp):
    """Yield the HDF5 groups that hold a ``positions_array``.

    Covers both layouts: the grid path keeps a single array directly under
    ``Control/Positions``; the bmotion path nests one per motion-group subgroup.
    """
    if "positions_array" in pos_grp:
        yield pos_grp
    for child in pos_grp.values():
        if isinstance(child, h5py.Group) and "positions_array" in child:
            yield child


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


def _write_missing_scopes(hdf5_path, payload):
    """Mark each scope in ``payload.missing`` as skipped for this shot.

    A per-scope partial skip: the good scopes' real data is already written for
    this shot, so each missing scope gets its own ``skipped`` group with its own
    reason, skipping any that already exist (idempotent across offload retries).
    Delegates to :func:`hdf5_writer.mark_shot_skipped_for_scopes` so the skip
    marker schema lives in one place.
    """
    hdf5_writer.mark_shot_skipped_for_scopes(
        hdf5_path, list(payload.missing), payload.shot_num,
        payload.missing, skip_if_exists=True,
    )


def _warn_missing_positions_ds(ds_path, shot_num):
    """Warn that a coordinate could not be recorded for lack of its dataset.

    Reaching this means the payload carried a coordinate to write but its
    positions dataset does not exist in the HDF5 -- a motion-group name mismatch
    between run setup and the per-shot payload, or a skeleton that was never
    built. Shared by both path adapters so the message can't drift. We warn and
    skip rather than abort: losing one coordinate row loudly beats aborting an
    otherwise-good offload, and the operator can act on the warning.
    """
    print(f"Warning: positions dataset {ds_path!r} missing; "
          f"coordinate for shot {shot_num} not recorded "
          f"(motion-group name mismatch or skeleton not built).")


# positions_array is grown one row per recorded shot, so the resizable dataset
# needs a chunk large enough that a normal run rarely crosses a chunk boundary
# (each crossing is an HDF5 chunk allocation + B-tree update). One row is ~16 B,
# so 1024 rows is ~16 KB -- one allocation covers a typical run; finalize rewrites
# the array to its exact size afterwards, so an oversized chunk costs nothing.
_POSITIONS_CHUNK = (1024,)


def create_positions_array(group, dtype, name="positions_array"):
    """Create an empty, append-only ``positions_array`` under ``group``.

    Shared by every setup path (bmotion + grid) so the resizable+chunked spec
    lives in one place. The offload appends one row per recorded shot (see
    :func:`append_position_row`) and finalize pads it to the planned size.
    """
    return group.create_dataset(
        name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=_POSITIONS_CHUNK)


def append_position_row(ds, row):
    """Append one structured ``row`` tuple to an append-only positions dataset.

    The single place the resize-and-write idiom lives, shared by both offload
    adapters and the in-process fallback writer. The row carries its own
    ``shot_num``, so the array stays a self-describing log of recorded shots; no
    index arithmetic, so a bad shot_num can never overwrite another shot's row.
    """
    ds.resize((len(ds) + 1,))
    ds[-1] = row


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
            _warn_missing_positions_ds(ds_path, payload.shot_num)
            continue
        append_position_row(f[ds_path], (payload.shot_num, xy[0], xy[1]))
