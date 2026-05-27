"""Glue between the bmotion acquisition path and the shared spool format.

Two responsibilities:

* **Acquire side** — turn this path's native ``all_data`` (and per-motion-group
  positions) into a :class:`spooling.ShotPayload`, and gather the run-level
  metadata bundle the offload side needs to rebuild the HDF5 skeleton.
* **Offload side** — turn a :class:`spooling.ShotPayload` back into the existing
  :mod:`acquisition.hdf5_writer` / bmotion position writes, so the resulting
  file is byte-for-byte the layout the in-process writer produces.

Keeping this mapping in one module means :mod:`spooling.spool_format` stays
storage-agnostic and the offload runner can dispatch by the ``"writer"`` tag in
the run metadata.
"""

import time

import h5py
import numpy as np

from . import bmotion as _bmotion
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


def build_run_metadata(msa, active_scopes, run_manager, selected_mg_keys,
                       ml_order, execution_order, toml_path, total_shots):
    """Assemble the run-level bundle written to ``meta_run.pkl``.

    Captures everything the offload side needs to rebuild the HDF5 skeleton with
    the existing writers: experiment/scope/channel metadata, per-scope time
    arrays + is_sequence (from ``active_scopes``, the ``{scope: is_sequence}``
    map returned by ``initialize_scopes``), and the bmotion position-setup
    layout (plain numpy arrays + the toml text and selection blob).
    """
    with open(toml_path, "r") as f:
        toml_text = f.read()

    selection_blob = _bmotion.build_bmotion_selection_blob(
        selected_mg_keys, ml_order, execution_order
    )
    prepared = _bmotion.collect_bmotion_position_setup(
        run_manager, selected_mg_keys
    )

    scopes = {}
    for scope_name in active_scopes:
        scope = msa.scopes.get(scope_name)
        time_array = msa.time_arrays.get(scope_name)
        scopes[scope_name] = {
            "description": msa.get_scope_description(scope_name),
            "ip_address": msa.scope_ips[scope_name],
            "scope_type": getattr(scope, "idn_string", "") if scope else "",
            "time_array": np.asarray(time_array) if time_array is not None else None,
            "is_sequence": active_scopes[scope_name],
        }

    # Channel descriptions come straight from the [channels] config section, so
    # the offload side can label datasets without seeing live trace lists.
    channel_descriptions = {}
    if msa.config.has_section("channels"):
        for key, value in msa.config.items("channels"):
            channel_descriptions[key] = value

    from .config import get_experiment_name, hdf5_filename

    return {
        "writer": WRITER_TAG,
        "experiment_description": msa.get_experiment_description(),
        "source_code": hdf5_writer.read_source_files(),
        "raw_config_text": msa.raw_config_text,
        # Filename the acquire side intends; the offload uses this verbatim so a
        # run that crosses midnight still targets one consistent file.
        "hdf5_filename": hdf5_filename(get_experiment_name(msa.config)),
        "config_scope_names": list(active_scopes.keys()),
        "scopes": scopes,
        "channel_descriptions": channel_descriptions,
        "total_shots": int(total_shots),
        "bmotion": {
            "toml_text": toml_text,
            "selection_blob": selection_blob,
            # prepared holds (mg_key, mg_name, setup_array, xpos, ypos)
            "prepared": [
                (str(k), name, np.asarray(setup), np.asarray(xpos), np.asarray(ypos))
                for (k, name, setup, xpos, ypos) in prepared
            ],
        },
    }


# --------------------------------------------------------------------------- #
# Offload side
# --------------------------------------------------------------------------- #
def build_skeleton(hdf5_path, meta, config, raw_config_text):
    """Create the top-level HDF5 structure from spooled run metadata.

    Reuses the existing hdf5_writer + bmotion writers so the file layout is
    identical to the in-process path.
    """
    scope_names = meta["config_scope_names"]

    hdf5_writer.write_experiment_metadata(
        hdf5_path,
        description=meta["experiment_description"],
        source_code=meta["source_code"],
        raw_config_text=meta["raw_config_text"],
        config=config,
        scope_names=scope_names,
    )

    for scope_name, info in meta["scopes"].items():
        hdf5_writer.write_scope_metadata(
            hdf5_path,
            scope_name=scope_name,
            description=info["description"],
            ip_address=info["ip_address"],
            scope_type=info["scope_type"],
        )
        time_array = info.get("time_array")
        if time_array is not None:
            hdf5_writer.write_time_array(
                hdf5_path, scope_name, np.asarray(time_array),
                info.get("is_sequence"),
            )

    bm = meta.get("bmotion")
    if bm:
        prepared = [
            (k, name, np.asarray(setup), np.asarray(xpos), np.asarray(ypos))
            for (k, name, setup, xpos, ypos) in bm["prepared"]
        ]
        _bmotion.write_bmotion_position_groups(
            hdf5_path,
            total_shots=meta["total_shots"],
            toml_text=bm["toml_text"],
            selection_blob=bm["selection_blob"],
            prepared=prepared,
        )


def write_shot(hdf5_path, payload, meta):
    """Write one ShotPayload's scope data + positions into the HDF5 file."""
    if payload.skipped:
        _write_skip(hdf5_path, payload, meta)
        _write_positions(hdf5_path, payload, meta)
        return

    all_data = _payload_to_all_data(payload)
    descriptions = _descriptions_for(all_data, meta)
    hdf5_writer.write_shot_data(hdf5_path, all_data, payload.shot_num, descriptions)
    _write_positions(hdf5_path, payload, meta)


def finalize(hdf5_path, meta, final_shot_num):
    """Write the per-scope shot_count attribute (run finalization)."""
    hdf5_writer.record_shot_count(
        hdf5_path, meta["config_scope_names"], final_shot_num
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


def _descriptions_for(all_data, meta):
    chan = meta.get("channel_descriptions", {})
    out = {}
    for scope_name, (traces, _d, _h) in all_data.items():
        for tr in traces:
            key = f"{scope_name}_{tr}"
            out[(scope_name, tr)] = chan.get(
                key, f"Channel {tr} - No description available"
            )
    return out


def _write_skip(hdf5_path, payload, meta):
    hdf5_writer.mark_shot_skipped_for_scopes(
        hdf5_path, meta["config_scope_names"], payload.shot_num,
        payload.skip_reason,
    )


def _write_positions(hdf5_path, payload, meta):
    """Write per-motion-group positions, mirroring record_bmotion_positions."""
    coords = payload.coordinates
    if not coords:
        return
    with h5py.File(hdf5_path, "a") as f:
        for mg_name, xy in coords.items():
            ds_path = f"Control/Positions/{mg_name}/positions_array"
            if ds_path not in f:
                continue
            ds = f[ds_path]
            ds[payload.shot_num - 1] = (payload.shot_num, xy[0], xy[1])
