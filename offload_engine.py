"""Offload process: turn a fast-disk spool into the final HDF5 on a slow disk.

Runs as a standalone, long-lived companion to the acquisition process. It polls
the spool directory for completed shots (those with a ``.done`` marker), writes
each into the HDF5 file, verifies the write by reading the data back, and only
then deletes the shot's bin files from the fast disk. When the acquisition
process drops a ``RUN_COMPLETE`` sentinel, the offload drains any remaining
shots, finalizes the file (shot_count), and exits.

The loop itself is storage-agnostic: it dispatches to a per-path adapter chosen
by the ``"writer"`` tag in the spooled run metadata (``acquisition`` for bmotion,
``grid`` for PositionManager grids). A new path's adapter can be added without
touching this loop.
"""

import importlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import h5py
import numpy as np
from tqdm import tqdm

# Register Blosc2 filter so h5py can decompress datasets written with hdf5plugin.
try:
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:
    pass

from spooling import spool_format

_log = logging.getLogger("offload")


# Poll interval while waiting for new shots / the run-complete sentinel.
_POLL_SECONDS = 0.5

# How long to wait for the acquire process to write run metadata (meta_run.pkl)
# before giving up. The offload is auto-launched *before* acquire writes the
# metadata (acquire writes it only after scope init), so a wait is expected and
# normal. The bound exists only so a misdirected manual drain -- e.g. pointed at
# a spool ROOT that will never get metadata -- doesn't hang forever; it must be
# generous enough to cover acquire's scope-init time.
_METADATA_TIMEOUT_SECONDS = 120.0

# How many times to retry a shot that fails to write/verify before quarantining
# it (moving it aside so it can't hang the drain). A transient slow-disk hiccup
# clears on the next pass; a genuinely corrupt shot is set aside after this.
_MAX_RETRIES = 3


# writer tag -> dotted module path of its offload adapter. Imported lazily in
# _get_adapter (only the one a run needs is loaded); adding a path means adding
# one entry here.
_ADAPTERS = {
    "acquisition": "acquisition.spool_adapter",
    "grid": "acquisition.grid_spool_adapter",
}


def _get_adapter(writer_tag: str):
    """Return the offload adapter module for a run's ``writer`` tag.

    Imports the submodule by its full dotted path so the lookup never depends on
    the submodule already being bound as an attribute on the ``acquisition``
    package -- which it only is as a side effect of some earlier import.
    """
    if writer_tag not in _ADAPTERS:
        raise ValueError(
            f"No offload adapter for writer tag {writer_tag!r}. "
            f"Supported: {sorted(_ADAPTERS)}."
        )
    return importlib.import_module(_ADAPTERS[writer_tag])


class MetadataTimeout(Exception):
    """Run metadata never appeared in the spool within the wait window.

    Raised when the spool folder has no ``meta_run.pkl`` after the grace period.
    For an auto-launched offload this should never happen (acquire writes the
    metadata seconds into the run); it signals the offload was pointed at a
    folder that will never become a drainable run -- typically a spool ROOT
    rather than a per-run subfolder. ``Offload_Run.py`` catches this to print the
    ``--list`` hint.
    """


def run_offload(spool_dir: str, config=None, poll_seconds: float = _POLL_SECONDS,
                max_retries: int = _MAX_RETRIES,
                metadata_timeout: float = _METADATA_TIMEOUT_SECONDS) -> None:
    """Drain ``spool_dir`` into the destination HDF5 until RUN_COMPLETE, then exit.

    The acquire process has already created the HDF5 file and written its full
    skeleton (metadata, time arrays, positions groups); this loop only fills the
    per-shot scope datasets and position rows, verifies each by read-back, and
    deletes the spooled copy.

    Args:
        spool_dir: fast-disk spool directory written by the acquire process.
            The destination HDF5 path is read verbatim from the run metadata
            (``meta["hdf5_path"]``) — it is computed exactly once, by the acquire
            entry script, so the offload never recomputes it.
        config: unused placeholder kept for call-site compatibility.
        poll_seconds: idle poll interval.
        max_retries: per-shot write/verify attempts before the shot is moved to
            ``shot_N.failed`` and skipped, so one corrupt shot cannot hang the
            drain (and the spool can still empty at RUN_COMPLETE).
        metadata_timeout: seconds to wait for ``meta_run.pkl`` before raising
            :class:`MetadataTimeout`. The offload is normally auto-launched
            before acquire writes the metadata, so some wait is expected; the
            bound only stops a misdirected drain from hanging forever. ``<= 0``
            waits indefinitely.
    """
    # config is accepted by the public entry point for call-site compatibility
    # but the drain reads everything it needs from the spool metadata.
    # NOTE: no single-instance lock -- one spool is drained by exactly one
    # offload (auto-launched per run, or a manual drain pointed at an explicit
    # --spool-dir). See spool_format for context.
    _run_offload(spool_dir, poll_seconds, max_retries, metadata_timeout)


def _run_offload(spool_dir: str, poll_seconds: float, max_retries: int,
                 metadata_timeout: float) -> None:
    print(f"Offload: waiting for run metadata in {spool_dir} ...")
    if not _wait_for(lambda: spool_format.run_metadata_exists(spool_dir),
                     poll_seconds, timeout=metadata_timeout):
        raise MetadataTimeout(
            f"No run metadata (meta_run.pkl) appeared in {spool_dir} after "
            f"{metadata_timeout:.0f}s."
        )

    meta = spool_format.read_run_metadata(spool_dir)
    hdf5_path = meta.get("hdf5_path")
    if not hdf5_path:
        # Metadata exists but is missing/empty its destination path (e.g. an
        # older-format or truncated bundle). Fail with an actionable message
        # rather than a bare KeyError buried in a traceback.
        raise ValueError(
            f"Run metadata in {spool_dir} has no 'hdf5_path'. This spool may be "
            f"from an incompatible/older run. Inspect it with: "
            f"python Offload_Run.py --list --spool-dir \"{spool_dir}\""
        )

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(
            f"Offload target HDF5 does not exist: {hdf5_path}. The acquire "
            "process is expected to create it (skeleton) before offload runs."
        )

    adapter = _get_adapter(meta.get("writer"))
    print(f"Offload: writer={meta.get('writer')}, filling -> {hdf5_path}")

    processed, quarantined, complete, final_shot_num = _drain_loop(
        spool_dir, hdf5_path, meta, adapter, poll_seconds, max_retries)

    _finalize_and_report(spool_dir, hdf5_path, meta, adapter, processed,
                         quarantined, complete, final_shot_num)


@dataclass
class _DrainState:
    """Mutable accumulators threaded through the drain loop.

    Bundles the three sets the per-shot processing updates in place so the
    helper takes one ``state`` instead of three separate accumulators, keeping
    the read-only run context (paths, meta, adapter) clearly distinct from the
    state being mutated.
    """
    processed: set = field(default_factory=set)       # shot_nums handled
    failures: dict = field(default_factory=dict)       # shot_num -> retry count
    quarantined: list = field(default_factory=list)    # exhausted-retries shots


def _drain_loop(spool_dir: str, hdf5_path: str, meta: dict, adapter,
                poll_seconds: float, max_retries: int):
    """Poll the spool, writing each ready shot, until RUN_COMPLETE drains it.

    Returns ``(processed, quarantined, complete, final_shot_num)``: the set of
    shot numbers handled, the list that exhausted retries (quarantined), the
    RUN_COMPLETE payload (or None), and the run's final shot number (or None).
    """
    total = meta.get("total_shots")  # None for older runs -> indeterminate bar

    state = _DrainState()
    complete = None
    final_shot_num = None

    with tqdm(total=total, desc="Offload", unit="shot", dynamic_ncols=True) as pbar:
        while True:
            ready = [s for s in spool_format.iter_ready_shots(spool_dir)
                     if s not in state.processed]
            _process_ready_shots(ready, spool_dir, hdf5_path, meta, adapter,
                                 max_retries, state, pbar)

            # While shots are still arriving the sentinel can't be there yet, so
            # only pay the read_run_complete + second iter_ready scan once a pass
            # finds nothing ready (the run has caught up or finished). This keeps
            # the busy path to a single directory scan per iteration.
            if not ready:
                complete = spool_format.read_run_complete(spool_dir)
                if complete is not None:
                    final_shot_num = complete.get("final_shot_num")
                    # One more drain pass to make sure no late .done slipped in.
                    # Shots that exhausted their retries are quarantined (not in
                    # iter_ready), so a persistently failing shot can't hang the run.
                    # TODO(drain-race): small window -- a .done published between
                    # this iter_ready_shots() snapshot and the break can be missed,
                    # leaving a shot in the spool after exit (needs a manual
                    # re-drain). Acquire writes RUN_COMPLETE only after its last
                    # write_shot returns, so in practice this is unobserved; tighten
                    # with a re-check loop if a writer is ever allowed to publish a
                    # shot after RUN_COMPLETE.
                    remaining = [s for s in spool_format.iter_ready_shots(spool_dir)
                                 if s not in state.processed]
                    if not remaining:
                        break
                else:
                    time.sleep(poll_seconds)

    return state.processed, state.quarantined, complete, final_shot_num


def _process_ready_shots(ready, spool_dir: str, hdf5_path: str, meta: dict,
                         adapter, max_retries: int, state: "_DrainState",
                         pbar) -> None:
    """Write each ready shot, updating ``state`` in place.

    A transient write/verify error keeps the bin and bumps the shot's failure
    count for a later retry; a shot that exhausts ``max_retries`` is quarantined
    (set aside) so one poison shot can't hang the drain. ``Exception`` is caught
    deliberately broadly: any adapter/IO failure must degrade to retry-then-
    quarantine rather than abort the whole run mid-drain.
    """
    for shot_num in ready:
        try:
            _offload_one_shot(spool_dir, hdf5_path, meta, adapter, shot_num)
            state.processed.add(shot_num)
            state.failures.pop(shot_num, None)
            pbar.update(1)
        except Exception as e:
            attempts = state.failures[shot_num] = state.failures.get(shot_num, 0) + 1
            if attempts >= max_retries:
                _quarantine_failed_shot(spool_dir, hdf5_path, meta, adapter,
                                        shot_num, attempts, e)
                state.processed.add(shot_num)
                state.quarantined.append(shot_num)
                pbar.update(1)
            else:
                tqdm.write(f"Offload ERROR on shot {shot_num}: {e} "
                           f"(attempt {attempts}/{max_retries}, bin kept for retry)")
                _log.warning("shot %s retry %d/%d: %s (bin kept)",
                             shot_num, attempts, max_retries, e)


def _quarantine_failed_shot(spool_dir: str, hdf5_path: str, meta: dict, adapter,
                            shot_num: int, attempts: int,
                            error: Exception) -> None:
    """Set aside a poison shot so the run can drain, marking it failed in HDF5.

    Stops retrying after ``max_retries``: marks the HDF5 group as failed (so
    unverified data isn't silently kept as if good), preserves the bin under
    ``shot_N.failed`` for inspection, and stops counting it as pending.
    """
    try:
        adapter.mark_shot_failed(
            hdf5_path, meta, shot_num,
            f"offload verification failed after {attempts} attempts")
    except Exception as mark_err:
        tqdm.write(f"Offload WARNING: could not mark shot {shot_num} "
                   f"failed in HDF5: {mark_err}")
        _log.warning("shot %s: could not mark failed in HDF5: %s",
                     shot_num, mark_err)
    dest = spool_format.quarantine_shot(spool_dir, shot_num)
    tqdm.write(f"Offload ERROR on shot {shot_num}: {error} "
               f"-- quarantined after {attempts} attempts -> {dest}")
    _log.warning("shot %s QUARANTINED after %d attempts: %s (moved -> %s)",
                 shot_num, attempts, error, dest)


def _finalize_and_report(spool_dir: str, hdf5_path: str, meta: dict, adapter,
                         processed: set, quarantined: list,
                         complete: Optional[dict],
                         final_shot_num: Optional[int]) -> None:
    """Finalize the HDF5 (shot_count), report the outcome, and prune old spools."""
    if final_shot_num is not None:
        adapter.finalize(hdf5_path, meta, final_shot_num)
        tqdm.write(f"Offload: finalized run (final_shot_num={final_shot_num}).")
        _log.warning("finalized run final_shot_num=%s", final_shot_num)
        if complete and complete.get("terminated_early"):
            tqdm.write(f"NOTE: acquisition terminated early "
                       f"({complete.get('abort_reason')}); HDF5 is complete and "
                       f"consistent for the {final_shot_num} shots taken.")
            _log.warning("acquisition terminated early: %s", complete.get('abort_reason'))
    n_ok = len(processed) - len(quarantined)
    tqdm.write(f"Offload complete. {n_ok} shots written to {hdf5_path}")
    _log.warning("offload complete: %d shots written, %d quarantined -> %s",
                 n_ok, len(quarantined), hdf5_path)
    if quarantined:
        tqdm.write(f"WARNING: {len(quarantined)} shot(s) failed and were quarantined "
                   f"(shot_N.failed in {spool_dir}): {sorted(quarantined)}")
        _log.warning("quarantined shots: %s", sorted(quarantined))

    # Housekeeping: drop old restart-rotated spools in this run's parent root.
    pruned = spool_format.prune_superseded(os.path.dirname(spool_dir))
    if pruned:
        _log.warning("pruned %d superseded spool folder(s)", len(pruned))


def _offload_one_shot(spool_dir: str, hdf5_path: str, meta: dict, adapter,
                      shot_num: int) -> None:
    """Write one shot, verify it read-back, then delete its spool copy.

    Idempotent for retries: if ``shot_N`` already exists in the HDF5 from a prior
    interrupted attempt, the write is skipped and the existing data verified
    instead, so a retry never trips ``write_shot_data``'s "already exists" guard.
    """
    payload = spool_format.read_shot(spool_dir, shot_num)

    if not _shot_in_hdf5(hdf5_path, payload):
        adapter.write_shot(hdf5_path, payload, meta)

    # TODO(verify-coverage): the read-back below checks trace data + headers only.
    # adapter.write_shot also writes position rows (Control/Positions/...), which
    # are deleted from the spool here without being verified; skipped shots bypass
    # verification entirely. Consider an adapter.verify_positions() (layout differs
    # per writer: bmotion per-motion-group vs. grid single array) before delete.
    if not payload.skipped:
        _verify_shot_in_hdf5(hdf5_path, payload)

    spool_format.delete_shot(spool_dir, shot_num)


def _shot_in_hdf5(hdf5_path: str, payload) -> bool:
    """True if every scope already has this shot's group written.

    Used to make the offload idempotent across interruptions/retries. A skipped
    shot is reported present once its skip group exists for any scope.
    """
    shot_name = f"shot_{payload.shot_num}"
    with h5py.File(hdf5_path, "r") as f:
        if payload.skipped:
            for sc in f:
                grp = f.get(sc)
                if isinstance(grp, h5py.Group) and shot_name in grp:
                    return True
            return False
        if not payload.traces:
            return False
        for scope_name in payload.traces:
            if scope_name not in f or shot_name not in f[scope_name]:
                return False
    return True


def _verify_shot_in_hdf5(hdf5_path: str, payload) -> None:
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


def _wait_for(predicate: Callable[[], bool], poll_seconds: float,
              timeout: float = 0) -> bool:
    """Block until ``predicate()`` is truthy. Return True if it became true.

    With ``timeout <= 0`` this waits indefinitely (always returns True). With a
    positive timeout it returns False once that many seconds have elapsed without
    the predicate becoming true, so the caller can act on the timeout.
    """
    deadline = (time.monotonic() + timeout) if timeout > 0 else None
    while not predicate():
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)
    return True
