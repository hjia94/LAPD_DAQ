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

import glob
import os
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_META_RUN = "meta_run.pkl"
_RUN_COMPLETE = "RUN_COMPLETE"
_SHOT_META = "meta.pkl"

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
    """

    shot_num: int
    traces: Dict[str, List[TracePayload]] = field(default_factory=dict)
    coordinates: Optional[object] = None
    acquisition_time: Optional[str] = None
    skipped: bool = False
    skip_reason: str = ""


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
    with open(os.path.join(spool_dir, _META_RUN), "rb") as f:
        return pickle.load(f)


def run_metadata_exists(spool_dir: str) -> bool:
    return os.path.exists(os.path.join(spool_dir, _META_RUN))


def update_run_metadata(spool_dir: str, updates: dict) -> dict:
    """Merge ``updates`` into the existing run metadata and rewrite it atomically.

    Used on resume to record a new ``resume_from_shot`` without rebuilding the
    whole bundle (the offload reads it to decide which shots to overwrite).
    """
    meta = read_run_metadata(spool_dir)
    meta.update(updates)
    _atomic_pickle(os.path.join(spool_dir, _META_RUN), meta)
    return meta


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
        "scopes": {},
    }

    if not payload.skipped:
        scope_items = list(payload.traces.items())
        if parallel and len(scope_items) > 1:
            # One worker per scope, writing disjoint <scope>__* files. Collect
            # results after the workers join, then store them in the original
            # scope order so the sidecar matches the serial path exactly.
            with ThreadPoolExecutor(max_workers=len(scope_items)) as executor:
                futures = {
                    executor.submit(_write_scope_files, tmp_dir, scope_name, traces): scope_name
                    for scope_name, traces in scope_items
                }
                meta_by_scope = {futures[f]: f.result() for f in futures}
            for scope_name, _traces in scope_items:
                sidecar["scopes"][scope_name] = meta_by_scope[scope_name]
        else:
            for scope_name, traces in scope_items:
                sidecar["scopes"][scope_name] = _write_scope_files(
                    tmp_dir, scope_name, traces)

    with open(os.path.join(tmp_dir, _SHOT_META), "wb") as f:
        pickle.dump(sidecar, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Flush directory contents, then publish atomically.
    if os.path.exists(shot_dir):
        shutil.rmtree(shot_dir)
    os.replace(tmp_dir, shot_dir)

    # Marker last: its existence means the shot dir is complete and readable.
    with open(done_path, "wb"):
        pass


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

    with open(os.path.join(shot_dir, _SHOT_META), "rb") as f:
        sidecar = pickle.load(f)

    payload = ShotPayload(
        shot_num=sidecar["shot_num"],
        coordinates=sidecar.get("coordinates"),
        acquisition_time=sidecar.get("acquisition_time"),
        skipped=sidecar.get("skipped", False),
        skip_reason=sidecar.get("skip_reason", ""),
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

    Acquisition uses this as a backpressure signal: a growing count means the
    offload process can't keep up (or isn't running), so the spool disk is
    filling.
    """
    return len(iter_ready_shots(spool_dir))


def free_space_bytes(spool_dir: str) -> int:
    """Bytes free on the filesystem holding ``spool_dir`` (0 if unavailable)."""
    try:
        return shutil.disk_usage(spool_dir).free
    except OSError:
        return 0


def spool_over_capacity(spool_dir, max_pending_shots, min_free_gb):
    """Return a reason string if the spool is over a backpressure limit, else None.

    Over capacity means the offload isn't keeping up (or isn't running): either
    too many undrained shots have piled up, or the spool disk is nearly full.
    A limit of ``<= 0`` disables that particular check.
    """
    if max_pending_shots and max_pending_shots > 0:
        pending = pending_shot_count(spool_dir)
        if pending > max_pending_shots:
            return f"{pending} shots pending offload (> {max_pending_shots})"
    if min_free_gb and min_free_gb > 0:
        free_gb = free_space_bytes(spool_dir) / (1024 ** 3)
        if free_gb < min_free_gb:
            return f"{free_gb:.1f} GB free on spool disk (< {min_free_gb} GB)"
    return None


def wait_for_capacity(spool_dir, max_pending_shots, min_free_gb,
                      poll_seconds=2.0, warn=None, check_abort=None):
    """Block until the spool is back under its backpressure limits.

    Returns immediately when there is capacity. Otherwise warns once (via the
    ``warn`` callback, defaulting to ``print``) and polls until the offload
    process drains enough shots / frees enough space. ``check_abort`` (if given)
    is polled each iteration; if it returns truthy the wait raises
    KeyboardInterrupt so the acquire loop's existing Ctrl-C handling runs.

    This is the safety valve for "offload to backup without filling up the PC
    disk": acquisition pauses rather than overrunning the spool disk.
    """
    reason = spool_over_capacity(spool_dir, max_pending_shots, min_free_gb)
    if reason is None:
        return
    (warn or print)(
        f"Spool backpressure: {reason}. Pausing acquisition until the offload "
        f"process catches up (is Offload_Run.py running and its target disk OK?)."
    )
    while spool_over_capacity(spool_dir, max_pending_shots, min_free_gb) is not None:
        if check_abort is not None and check_abort():
            raise KeyboardInterrupt("aborted while waiting for spool capacity")
        time.sleep(poll_seconds)
    (warn or print)("Spool backpressure cleared; resuming acquisition.")


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
    path = os.path.join(spool_dir, _RUN_COMPLETE)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def clear_run_complete(spool_dir: str) -> None:
    """Remove the RUN_COMPLETE sentinel so the spool can accept a resumed run.

    Called when the user chooses to resume from a previous partial run: the
    existing HDF5 and already-spooled shots are kept, but the old sentinel is
    removed so the acquire process can write a new one when the resumed run ends.
    """
    path = os.path.join(spool_dir, _RUN_COMPLETE)
    if os.path.exists(path):
        os.remove(path)


_OFFLOAD_LOCK = "offload.lock"

# A lock whose heartbeat hasn't been touched within this many seconds is treated
# as stale (the offload died without cleaning up), so a new offload may take over.
_LOCK_STALE_SECONDS = 30.0


def offload_lock_is_live(spool_dir: str) -> bool:
    """True if a live offload process currently holds this spool's lock.

    "Live" means the lock file exists, its recorded PID is still running, and its
    heartbeat (file mtime, refreshed by :func:`offload_lock_heartbeat`) is recent.
    A stale lock (process gone or heartbeat old) reports False so a resume/next
    run can take over the spool instead of refusing forever.
    """
    path = os.path.join(spool_dir, _OFFLOAD_LOCK)
    try:
        age = time.time() - os.path.getmtime(path)
    except OSError:
        return False
    if age > _LOCK_STALE_SECONDS:
        return False
    try:
        with open(path, "r") as f:
            pid = int(f.read().strip() or "-1")
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


def acquire_offload_lock(spool_dir: str) -> bool:
    """Take the offload lock for this process. Returns False if one is live.

    Writes our PID; the caller refreshes the heartbeat via
    :func:`offload_lock_heartbeat` and removes it with
    :func:`release_offload_lock` on exit.
    """
    if offload_lock_is_live(spool_dir):
        return False
    os.makedirs(spool_dir, exist_ok=True)
    _atomic_pickle_bytes(os.path.join(spool_dir, _OFFLOAD_LOCK),
                         str(os.getpid()).encode())
    return True


def offload_lock_heartbeat(spool_dir: str) -> None:
    """Refresh the lock's mtime so it doesn't look stale during a long drain."""
    path = os.path.join(spool_dir, _OFFLOAD_LOCK)
    try:
        os.utime(path, None)
    except OSError:
        pass


def release_offload_lock(spool_dir: str) -> None:
    """Remove the offload lock (best-effort) when the offload exits."""
    path = os.path.join(spool_dir, _OFFLOAD_LOCK)
    try:
        os.remove(path)
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)            # POSIX: signal 0 just checks existence
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                # exists but owned by another user
    except OSError:
        # Windows has no os.kill(0) semantics; fall back to a tasklist check.
        return _pid_alive_windows(pid)
    return True


def _pid_alive_windows(pid: int) -> bool:
    import subprocess
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True               # can't tell -> assume alive (don't double-launch)
    return str(pid) in out.stdout


def _atomic_pickle_bytes(path: str, data: bytes) -> None:
    """Atomically write raw bytes (used for the small PID lock file)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


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
        except OSError:
            continue
    return removed


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _atomic_pickle(path: str, obj) -> None:
    """Pickle ``obj`` to ``path`` via a temp file + ``os.replace`` (atomic)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
