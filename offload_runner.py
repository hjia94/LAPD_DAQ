"""Offload process: turn a fast-disk spool into the final HDF5 on a slow disk.

Runs as a standalone, long-lived companion to the acquisition process. It polls
the spool directory for completed shots (those with a ``.done`` marker), writes
each into the HDF5 file, verifies the write by reading the data back, and only
then deletes the shot's bin files from the fast disk. When the acquisition
process drops a ``RUN_COMPLETE`` sentinel, the offload drains any remaining
shots, finalizes the file (shot_count), and exits.

The loop itself is storage-agnostic: it dispatches to a per-path adapter chosen
by the ``"writer"`` tag in the spooled run metadata. Today the only adapter is
the ``acquisition`` (bmotion) one; a ``lapd_daq`` adapter can be added without
touching this loop.
"""

import time

import h5py
import numpy as np

from spooling import spool_format


# Poll interval while waiting for new shots / the run-complete sentinel.
_POLL_SECONDS = 0.5


def _get_adapter(writer_tag):
    """Return the offload adapter module for a run's ``writer`` tag."""
    if writer_tag == "acquisition":
        from acquisition import spool_adapter
        return spool_adapter
    raise ValueError(
        f"No offload adapter for writer tag {writer_tag!r}. "
        "Only 'acquisition' is supported so far."
    )


def run_offload(spool_dir, hdf5_path, config=None, poll_seconds=_POLL_SECONDS):
    """Drain ``spool_dir`` into ``hdf5_path`` until RUN_COMPLETE, then exit.

    Args:
        spool_dir: fast-disk spool directory written by the acquire process.
        hdf5_path: destination HDF5 file on the slow/large disk.
        config: optional ConfigParser; only used as a fallback by the HDF5
            metadata writer when the spooled raw config text is empty.
        poll_seconds: idle poll interval.
    """
    print(f"Offload: waiting for run metadata in {spool_dir} ...")
    _wait_for(lambda: spool_format.run_metadata_exists(spool_dir), poll_seconds)

    meta = spool_format.read_run_metadata(spool_dir)
    adapter = _get_adapter(meta.get("writer"))
    print(f"Offload: writer={meta.get('writer')}, building HDF5 skeleton -> {hdf5_path}")
    adapter.build_skeleton(hdf5_path, meta, config, meta.get("raw_config_text", ""))

    processed = set()
    final_shot_num = None

    while True:
        ready = [s for s in spool_format.iter_ready_shots(spool_dir) if s not in processed]
        for shot_num in ready:
            try:
                _offload_one_shot(spool_dir, hdf5_path, meta, adapter, shot_num)
                processed.add(shot_num)
            except Exception as e:  # keep the bin, report, move on
                print(f"Offload ERROR on shot {shot_num}: {e} (bin kept for retry)")

        complete = spool_format.read_run_complete(spool_dir)
        if complete is not None:
            final_shot_num = complete.get("final_shot_num")
            # One more drain pass to make sure no late .done slipped in.
            remaining = [s for s in spool_format.iter_ready_shots(spool_dir)
                         if s not in processed]
            if not remaining:
                break

        if not ready and complete is None:
            time.sleep(poll_seconds)

    if final_shot_num is not None:
        adapter.finalize(hdf5_path, meta, final_shot_num)
        print(f"Offload: finalized run (final_shot_num={final_shot_num}).")
    print(f"Offload complete. {len(processed)} shots written to {hdf5_path}")


def _offload_one_shot(spool_dir, hdf5_path, meta, adapter, shot_num):
    """Write one shot, verify it read-back, then delete its spool copy."""
    payload = spool_format.read_shot(spool_dir, shot_num)
    adapter.write_shot(hdf5_path, payload, meta)

    if not payload.skipped:
        _verify_shot_in_hdf5(hdf5_path, payload)

    spool_format.delete_shot(spool_dir, shot_num)


def _verify_shot_in_hdf5(hdf5_path, payload):
    """Read each trace back from the HDF5 and compare to the spooled data.

    Raises on any mismatch so the caller leaves the bin in place. Compares
    dataset shape, full int16 array equality, and the raw header bytes.
    """
    with h5py.File(hdf5_path, "r") as f:
        for scope_name, traces in payload.traces.items():
            shot_group = f[scope_name][f"shot_{payload.shot_num}"]
            for tr in traces:
                data_ds = shot_group[f"{tr.channel}_data"]
                expected = np.asarray(tr.data, dtype=np.int16)
                actual = data_ds[()]
                if actual.shape != expected.shape:
                    raise ValueError(
                        f"{scope_name}/{tr.channel}: shape {actual.shape} != "
                        f"expected {expected.shape}"
                    )
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        f"{scope_name}/{tr.channel}: data mismatch on read-back"
                    )
                header_ds = shot_group[f"{tr.channel}_header"]
                actual_hdr = header_ds[()].tobytes()
                if actual_hdr != bytes(tr.header):
                    raise ValueError(
                        f"{scope_name}/{tr.channel}: header mismatch on read-back"
                    )


def _wait_for(predicate, poll_seconds):
    while not predicate():
        time.sleep(poll_seconds)
