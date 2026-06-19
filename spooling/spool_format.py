"""On-disk spool format: the contract between the acquire and offload processes.

The acquire process writes each shot to a *fast* local disk as raw int16 binary
files plus a small pickled sidecar; the offload process reads them back, writes
the final HDF5 on a *slow/large* disk, verifies the write, and deletes the spool
copy. This module owns every byte of the spool layout and knows nothing about
HDF5 group names or any specific acquisition path — that mapping lives in
per-path adapters.

Spool directory layout::

    <spool_dir>/
      meta_run.pkl              # run-level metadata bundle (written once)
      shot_000001/
        <scope>__<channel>.bin  # raw int16 bytes (ndarray.tofile)
        <scope>__<channel>.hdr  # raw header bytes (e.g. LeCroy WAVEDESC)
        meta.pkl                # per-shot sidecar (shapes, coords, skip info)
      shot_000001.done          # zero-byte marker, written last
      RUN_COMPLETE              # written at end: {"final_shot_num": N}

Crash safety: a shot is written into ``shot_N.tmp/``, atomically renamed to
``shot_N/`` via ``os.replace``, and only then is the ``shot_N.done`` marker
created. The offload side ignores any shot directory that lacks a ``.done``
marker, so a half-written or interrupted shot is never consumed.
"""

import errno
import glob
import os
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_META_RUN = "meta_run.pkl"
_RUN_COMPLETE = "RUN_COMPLETE"
_SHOT_META = "meta.pkl"

# Injectable sleep seam. Tests patch THIS module attribute
# (spool_format._sleep) to skip the disk-full retry pause; patching the stdlib
# ``time`` module's functions would leak into every other module in the process.
_sleep = time.sleep

# Errors from reading a present-but-broken pickle (run metadata, RUN_COMPLETE, or
# a shot sidecar). Wrapped uniformly in SpoolMetadataError so callers see one
# typed "this spool's data is corrupt" failure instead of a raw pickle traceback.
_PICKLE_READ_ERRORS = (OSError, pickle.UnpicklingError, EOFError, ValueError)


class SpoolMetadataError(Exception):
    """A run-metadata / RUN_COMPLETE file exists but could not be read.

    Raised by :func:`read_run_metadata` and :func:`read_run_complete` when the
    file is present but unreadable (truncated/corrupt pickle, or an OS read
    error), with the offending path. Wrapping the low-level ``UnpicklingError``/
    ``OSError`` in one typed exception lets every caller recognize "this spool's
    metadata is broken" instead of leaking a raw pickle traceback. File
    *absence* is a separate, expected case (see each function's contract).
    """

# Separates scope name from channel name in per-trace file names. Double
# underscore avoids colliding with single underscores common in channel ids.
_NAME_SEP = "__"


@dataclass
class TracePayload:
    """One acquired channel: its raw int16 samples and raw header bytes.

    ``data`` is stored/loaded verbatim as int16 (1-D for RealTime mode, 2-D
    ``(segments, samples)`` for sequence mode). ``header`` is opaque bytes.
    """

    channel: str
    data: np.ndarray
    header: bytes


@dataclass
class ShotPayload:
    """A single shot, storage-agnostic.

    ``traces`` maps a scope name to its list of :class:`TracePayload`.
    ``coordinates`` carries probe position info to be written alongside the
    scope data (its structure is interpreted by the path adapter, e.g.
    ``{mg_name: (x, y)}`` for the bmotion path); ``None`` for stationary runs.
    A skipped shot carries no traces and sets ``skipped`` + ``skip_reason``.

    ``missing`` maps a scope name to the reason its data is absent for THIS shot
    (a per-scope arm/read/spool failure). The shot is otherwise a normal data
    shot -- the good scopes are in ``traces`` -- but the offload records each
    missing scope as its own ``skipped`` shot group so a single misbehaving
    scope yields a partial shot rather than aborting the run. Distinct from
    ``skipped``, which marks the WHOLE shot as not taken.
    """

    shot_num: int
    traces: Dict[str, List[TracePayload]] = field(default_factory=dict)
    coordinates: Optional[object] = None
    acquisition_time: Optional[str] = None
    skipped: bool = False
    skip_reason: str = ""
    missing: Dict[str, str] = field(default_factory=dict)


def _shot_dirname(shot_num: int) -> str:
    return f"shot_{shot_num:06d}"


def _trace_basename(scope_name: str, channel: str) -> str:
    return f"{scope_name}{_NAME_SEP}{channel}"


# --------------------------------------------------------------------------- #
# Run-level metadata
# --------------------------------------------------------------------------- #
def write_run_metadata(spool_dir: str, meta: dict) -> None:
    """Pickle the run-level metadata bundle to ``meta_run.pkl`` (atomically).

    ``meta`` must include a ``"writer"`` key naming the offload adapter that
    will build the HDF5 file (e.g. ``"acquisition"``), plus whatever raw values
    that adapter needs to construct the file skeleton.
    """
    os.makedirs(spool_dir, exist_ok=True)
    _atomic_pickle(os.path.join(spool_dir, _META_RUN), meta)


def read_run_metadata(spool_dir: str) -> dict:
    """Load the run-level metadata bundle. Raises if it is absent or corrupt.

    Absence raises ``FileNotFoundError`` (callers normally pre-check via
    :func:`run_metadata_exists`); a present-but-unreadable file raises
    :class:`SpoolMetadataError` so callers see a clear, typed failure instead of
    a raw pickle traceback.
    """
    path = os.path.join(spool_dir, _META_RUN)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise
    except _PICKLE_READ_ERRORS as e:
        raise SpoolMetadataError(f"Cannot read run metadata at {path}: {e}") from e


def run_metadata_exists(spool_dir: str) -> bool:
    return os.path.exists(os.path.join(spool_dir, _META_RUN))


# --------------------------------------------------------------------------- #
# Per-shot write
# --------------------------------------------------------------------------- #
def _write_scope_files(tmp_dir, scope_name, traces):
    """Write one scope's per-trace ``.bin``/``.hdr`` files into ``tmp_dir``.

    Returns the scope's ``scope_meta`` list (one ``{channel, dtype, shape}`` per
    trace, in trace order). Each scope writes only its own ``<scope>__*`` files,
    so distinct scopes touch disjoint paths and this is safe to run in parallel
    threads (``ndarray.tofile`` / file writes are blocking I/O that release the
    GIL, so the writes overlap).
    """
    scope_meta = []
    for tr in traces:
        arr = np.asarray(tr.data, dtype=np.int16)
        base = _trace_basename(scope_name, tr.channel)
        arr.tofile(os.path.join(tmp_dir, base + ".bin"))
        with open(os.path.join(tmp_dir, base + ".hdr"), "wb") as hf:
            hf.write(bytes(tr.header))
        scope_meta.append({
            "channel": tr.channel,
            "dtype": str(arr.dtype),
            "shape": tuple(int(s) for s in arr.shape),
        })
    return scope_meta


def _remove_scope_files(tmp_dir, scope_name):
    """Delete any ``<scope>__*`` files a failed write left half-written.

    A scope whose write raised must leave no partial bytes in the shot dir, so
    the published shot contains only complete scopes plus a ``missing`` record
    for the failed ones.
    """
    prefix = scope_name + _NAME_SEP
    try:
        for name in os.listdir(tmp_dir):
            if name.startswith(prefix):
                try:
                    os.remove(os.path.join(tmp_dir, name))
                except OSError:
                    pass
    except OSError:
        pass


def _collect_scope_write(sidecar, tmp_dir, scope_name, produce_meta):
    """Run/await one scope's write and fold the outcome into ``sidecar``.

    ``produce_meta`` is a zero-arg callable returning the scope_meta list (it
    either does the serial write or reads a finished future). On success the
    scope's metadata is recorded under ``sidecar["scopes"]``. A per-scope
    failure that is NOT a disk-full error is tolerated: the scope's partial
    files are removed and it is recorded under ``sidecar["missing"]`` so the
    offload marks it skipped for this shot. A disk-full error is re-raised so
    the caller's disk-full retry/backpressure handling still fires -- a full
    spool disk is a storage fault for the whole run, not one bad scope.
    """
    try:
        sidecar["scopes"][scope_name] = produce_meta()
    except Exception as exc:  # noqa: BLE001 - any scope fault is tolerated
        # A full spool disk is a storage fault for the whole run, not one bad
        # scope: re-raise so the caller's disk-full retry/backpressure fires.
        if isinstance(exc, OSError) and is_disk_full_error(exc):
            raise
        _remove_scope_files(tmp_dir, scope_name)
        sidecar["missing"][scope_name] = f"spool write failed: {exc}"


def write_shot(spool_dir: str, payload: ShotPayload, parallel: bool = False) -> None:
    """Write one shot to the spool and publish it with a ``.done`` marker.

    Writes into ``shot_N.tmp/``, atomically renames to ``shot_N/``, then creates
    the marker. Safe to call for both data shots and skipped shots.

    When ``parallel`` is true and the shot has 2+ scopes, each scope's files are
    written on its own worker thread so the per-scope writes overlap. The on-disk
    result (file names, bytes, sidecar, and the atomic publish order) is identical
    to the serial path — only the order bytes hit disjoint files changes — so the
    offload reads back a byte-identical shot.
    """
    os.makedirs(spool_dir, exist_ok=True)
    shot_dir = os.path.join(spool_dir, _shot_dirname(payload.shot_num))
    tmp_dir = shot_dir + ".tmp"
    done_path = shot_dir + ".done"

    # Clear any leftovers from a previous interrupted attempt.
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    sidecar = {
        "shot_num": payload.shot_num,
        "acquisition_time": payload.acquisition_time,
        "coordinates": payload.coordinates,
        "skipped": payload.skipped,
        "skip_reason": payload.skip_reason,
        "missing": dict(payload.missing),
        "scopes": {},
    }

    if not payload.skipped:
        scope_items = list(payload.traces.items())
        if parallel and len(scope_items) > 1:
            # One worker per scope, writing disjoint <scope>__* files. Wait for
            # EVERY worker before inspecting any result, so no thread is still
            # touching tmp_dir when we react to a failure; then collect in the
            # original scope order so the sidecar matches the serial path.
            with ThreadPoolExecutor(max_workers=len(scope_items)) as executor:
                future_by_scope = {
                    scope_name: executor.submit(
                        _write_scope_files, tmp_dir, scope_name, traces)
                    for scope_name, traces in scope_items
                }
                wait(future_by_scope.values())
            for scope_name, _traces in scope_items:
                _collect_scope_write(
                    sidecar, tmp_dir, scope_name,
                    future_by_scope[scope_name].result)
        else:
            for scope_name, traces in scope_items:
                _collect_scope_write(
                    sidecar, tmp_dir, scope_name,
                    lambda sn=scope_name, tr=traces: _write_scope_files(
                        tmp_dir, sn, tr))

    with open(os.path.join(tmp_dir, _SHOT_META), "wb") as f:
        pickle.dump(sidecar, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Flush directory contents, then publish atomically.
    if os.path.exists(shot_dir):
        shutil.rmtree(shot_dir)
    os.replace(tmp_dir, shot_dir)

    # Marker last: its existence means the shot dir is complete and readable.
    with open(done_path, "wb"):
        pass


# Default pause/retry behaviour when the spool disk fills up mid-run. Both are
# overridable per call (acquisition reads them from [storage]); the constants are
# the fallbacks so a run with no config still behaves sanely.
DISK_FULL_PAUSE_SECONDS = 30.0
DISK_FULL_MAX_RETRIES = 3

# Windows error code for "There is not enough space on the disk" (the winerror on
# the OSError; the POSIX equivalent is errno.ENOSPC).
_WIN_ERROR_DISK_FULL = 112


def is_disk_full_error(exc: BaseException) -> bool:
    """True if ``exc`` is an out-of-space failure (POSIX ENOSPC / Windows 112).

    The OS surfaces a full disk as ``OSError``; the errno is portable on POSIX
    and ``winerror`` 112 ("There is not enough space on the disk") on Windows.
    """
    if not isinstance(exc, OSError):
        return False
    if exc.errno == errno.ENOSPC:
        return True
    return getattr(exc, "winerror", None) == _WIN_ERROR_DISK_FULL


def write_shot_with_disk_full_retry(
    spool_dir: str, payload: "ShotPayload", parallel: bool = False,
    pause_seconds: float = DISK_FULL_PAUSE_SECONDS,
    max_retries: int = DISK_FULL_MAX_RETRIES, warn=None,
) -> None:
    """Write a shot, pausing and retrying if the spool disk is full.

    This is the only backpressure mechanism: instead of predicting when the
    offload is falling behind, we just write, and only react to a real
    out-of-space failure. On disk-full we sleep ``pause_seconds`` (giving the
    offload time to drain already-written shots into the HDF5 and free their
    bins) and retry, up to ``max_retries`` extra attempts. If it still fails the
    error propagates so the run aborts rather than spinning forever.

    Any error that is not disk-full propagates immediately on the first attempt.
    """
    emit = warn or print
    attempt = 0
    while True:
        try:
            write_shot(spool_dir, payload, parallel=parallel)
            return
        except OSError as exc:
            if not is_disk_full_error(exc) or attempt >= max_retries:
                raise
            attempt += 1
            emit(
                f"Spool disk full writing shot {payload.shot_num}; pausing "
                f"{pause_seconds:.0f}s for offload to drain "
                f"(retry {attempt}/{max_retries})."
            )
            _sleep(pause_seconds)


# --------------------------------------------------------------------------- #
# Per-shot read (offload side)
# --------------------------------------------------------------------------- #
def read_shot(spool_dir: str, shot_num: int) -> ShotPayload:
    """Load a shot previously written with :func:`write_shot`.

    Reconstructs int16 arrays (and 2-D sequence shapes) and raw header bytes.
    Raises if the shot has no ``.done`` marker (i.e. is not safely complete).
    """
    shot_dir = os.path.join(spool_dir, _shot_dirname(shot_num))
    done_path = shot_dir + ".done"
    if not os.path.exists(done_path):
        raise FileNotFoundError(f"Shot {shot_num} is not marked done: {done_path}")

    # A present-but-corrupt sidecar raises the same typed error the run-metadata
    # readers use, rather than a raw UnpicklingError, so a poison shot surfaces
    # consistently (the drain catches it and quarantines either way).
    meta_path = os.path.join(shot_dir, _SHOT_META)
    try:
        with open(meta_path, "rb") as f:
            sidecar = pickle.load(f)
    except _PICKLE_READ_ERRORS as e:
        raise SpoolMetadataError(f"Cannot read shot sidecar at {meta_path}: {e}") from e

    payload = ShotPayload(
        shot_num=sidecar["shot_num"],
        coordinates=sidecar.get("coordinates"),
        acquisition_time=sidecar.get("acquisition_time"),
        skipped=sidecar.get("skipped", False),
        skip_reason=sidecar.get("skip_reason", ""),
        missing=dict(sidecar.get("missing", {})),
    )

    if not payload.skipped:
        for scope_name, scope_meta in sidecar.get("scopes", {}).items():
            traces: List[TracePayload] = []
            for entry in scope_meta:
                base = _trace_basename(scope_name, entry["channel"])
                arr = np.fromfile(
                    os.path.join(shot_dir, base + ".bin"),
                    dtype=np.dtype(entry["dtype"]),
                )
                shape = tuple(entry["shape"])
                if arr.shape != shape:
                    arr = arr.reshape(shape)
                with open(os.path.join(shot_dir, base + ".hdr"), "rb") as hf:
                    header = hf.read()
                traces.append(TracePayload(entry["channel"], arr, header))
            payload.traces[scope_name] = traces

    return payload


def iter_ready_shots(spool_dir: str) -> List[int]:
    """Return shot numbers that have a ``.done`` marker, in ascending order."""
    if not os.path.isdir(spool_dir):
        return []
    shots = []
    for name in os.listdir(spool_dir):
        if name.startswith("shot_") and name.endswith(".done"):
            stem = name[len("shot_"):-len(".done")]
            try:
                shots.append(int(stem))
            except ValueError:
                continue
    return sorted(shots)


def delete_shot(spool_dir: str, shot_num: int) -> None:
    """Remove a shot's directory and its ``.done`` marker after verification."""
    shot_dir = os.path.join(spool_dir, _shot_dirname(shot_num))
    done_path = shot_dir + ".done"
    if os.path.isdir(shot_dir):
        shutil.rmtree(shot_dir)
    if os.path.exists(done_path):
        os.remove(done_path)


def quarantine_shot(spool_dir: str, shot_num: int) -> str:
    """Move a poison shot aside so the offload can stop retrying it and drain.

    Renames ``shot_N/`` to ``shot_N.failed/`` and drops the ``.done`` marker, so
    :func:`iter_ready_shots` no longer returns it (it keys on ``.done``). The
    data is preserved under ``.failed`` for manual inspection/recovery rather
    than deleted. Returns the quarantine directory path.
    """
    shot_dir = os.path.join(spool_dir, _shot_dirname(shot_num))
    failed_dir = shot_dir + ".failed"
    done_path = shot_dir + ".done"
    if os.path.exists(failed_dir):
        shutil.rmtree(failed_dir)
    if os.path.isdir(shot_dir):
        os.replace(shot_dir, failed_dir)
    if os.path.exists(done_path):
        os.remove(done_path)
    return failed_dir


def pending_shot_count(spool_dir: str) -> int:
    """Number of shots written but not yet offloaded (``.done`` markers present).

    Used only for reporting (e.g. ``Offload_Run.py --list``); it is no longer a
    backpressure signal. Acquisition reacts to an actual disk-full write failure
    instead (see :func:`write_shot_with_disk_full_retry`).
    """
    return len(iter_ready_shots(spool_dir))


# --------------------------------------------------------------------------- #
# Run-complete sentinel
# --------------------------------------------------------------------------- #
def write_run_complete(spool_dir: str, final_shot_num: int,
                       terminated_early: bool = False,
                       abort_reason: Optional[str] = None) -> None:
    """Write the RUN_COMPLETE sentinel the offload waits on.

    ``terminated_early`` / ``abort_reason`` record that the run stopped before
    its planned end (e.g. a terminal motor failure or Ctrl-C). The data already
    spooled is still complete and consistent for the shots taken; these fields
    just let the offload/analysis know the scan was cut short.
    """
    os.makedirs(spool_dir, exist_ok=True)
    _atomic_pickle(
        os.path.join(spool_dir, _RUN_COMPLETE),
        {
            "final_shot_num": int(final_shot_num),
            "terminated_early": bool(terminated_early),
            "abort_reason": abort_reason,
        },
    )


def run_complete_exists(spool_dir: str) -> bool:
    return os.path.exists(os.path.join(spool_dir, _RUN_COMPLETE))


def read_run_complete(spool_dir: str) -> Optional[dict]:
    """Load the RUN_COMPLETE sentinel, or ``None`` if it isn't present.

    A present-but-unreadable sentinel raises :class:`SpoolMetadataError` rather
    than returning ``None`` -- treating a corrupt completion record as "no run
    finished" would let a finished run be offered for restart, so the caller is
    told the record is broken instead.
    """
    path = os.path.join(spool_dir, _RUN_COMPLETE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except _PICKLE_READ_ERRORS as e:
        raise SpoolMetadataError(f"Cannot read RUN_COMPLETE at {path}: {e}") from e


# NOTE: a single-instance offload lock (PID + mtime heartbeat) used to live here
# to stop two offloads draining one spool. It was removed: the only path that
# spawned a second offload on the same spool was resume, which is not functional
# on this branch (a separate branch owns the resume fix). Manual `Offload_Run.py`
# runs target an explicit --spool-dir, so they don't collide with the
# auto-launched offload in practice. If concurrent drains become possible again
# (e.g. when resume lands), reintroduce a lock here -- and heartbeat it during
# the metadata wait, not only the drain loop, so it can't go stale mid-wait.


def rotate_spool(spool_dir: str) -> Optional[str]:
    """Move an existing spool subfolder aside so a restart starts from empty.

    On *restart* (redo the same-named run from shot 1) the old subfolder still
    holds the aborted run's ``shot_*``/``RUN_COMPLETE``/``meta_run.pkl``; left in
    place, the next offload would drain those stale shots into the fresh HDF5.
    Renaming to ``<spool_dir>.superseded-<ts>`` preserves the data for inspection
    while guaranteeing the caller can recreate a clean ``spool_dir``. Returns the
    rotated path, or ``None`` if there was nothing to rotate.

    This requires the prior acquire/offload processes to have released their
    handles (log files, HDF5) on exit; they close them in their teardown paths.
    """
    if not os.path.isdir(spool_dir):
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    superseded = f"{spool_dir}.superseded-{ts}"
    os.replace(spool_dir, superseded)
    return superseded


def prune_superseded(spool_root: str, keep_days: float = 7.0) -> List[str]:
    """Delete ``*.superseded-*`` folders older than ``keep_days`` in ``spool_root``.

    Restart rotates old spools aside rather than deleting them (so data survives
    for inspection); this housekeeping pass, run after an offload completes,
    removes the stale rotations once they're past the retention window. Returns
    the paths removed. Best-effort: a folder that can't be removed is skipped.
    """
    removed = []
    cutoff = time.time() - keep_days * 86400
    for path in glob.glob(os.path.join(spool_root, "*.superseded-*")):
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path)
                removed.append(path)
        except OSError as e:
            # Best-effort: keep going, but surface the skip so a folder that
            # repeatedly can't be removed (retention cleanup silently failing,
            # disk slowly filling with stale .superseded-* rotations) is visible.
            print(f"Warning: could not prune superseded spool {path!r}: {e}")
            continue
    return removed


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _atomic_pickle(path: str, obj) -> None:
    """Pickle ``obj`` to ``path`` via a temp file + ``os.replace`` (atomic).

    The temp file is fsync'd before the rename so that after a crash/power loss
    the destination is either the old file or the fully-written new one, never a
    truncated/partially-flushed pickle -- which is exactly the corrupt-metadata
    state the offload's SpoolMetadataError handling otherwise has to absorb.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
