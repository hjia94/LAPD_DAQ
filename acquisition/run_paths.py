"""Single source of truth for a run's on-disk identity (HDF5 file + spool dir).

The two-process spooled pipeline must agree on *which* HDF5 file and *which*
spool subfolder a run owns, even across a date rollover (a run started before
midnight and resumed the next day). Identity therefore keys on the experiment
``name`` (which the operator keeps unique), not on ``date.today()``:

* The HDF5 filename keeps its human-friendly ``<name>_<date>.hdf5`` form (the
  date is a safety net for a forgotten name change), but detection globs
  ``<name>_*.hdf5`` so a prior-date file is still found and reused.
* The spool subfolder is paired the same way: ``<name>_<date>`` to create,
  ``<name>_*`` to find.

This replaces the old ``date.today()``-only ``resolve_hdf5_path`` check and the
mtime-based ``_find_latest_spool_subdir`` guess, both of which silently spawned a
fresh run on rollover (or latched onto an unrelated run's spool).
"""

import datetime
import glob
import os
from dataclasses import dataclass

from .config import get_experiment_name, get_storage_paths, hdf5_filename


@dataclass(frozen=True)
class RunPaths:
    """Resolved identity for one run.

    ``hdf5_path`` / ``spool_dir`` are the paths the run should use: an existing
    prior-date match when one is found (resume/restart target), otherwise a fresh
    today-dated path. ``is_existing`` is True when a prior HDF5 was found, so the
    caller knows to inspect/prompt rather than start clean. ``ambiguous`` flags
    that more than one prior match existed (the newest was chosen); the caller
    surfaces it in the prompt.
    """

    name: str
    hdf5_path: str
    spool_dir: str | None
    is_existing: bool
    ambiguous: bool = False


def resolve_run_paths(config, base_path, spool_root=None) -> RunPaths:
    """Resolve the HDF5 file + spool subfolder this run owns.

    ``spool_root`` is the parent spool directory (``[storage] spool_dir``). Pass
    ``None`` when only the HDF5 path is needed (e.g. resolve_hdf5_path,
    update_description); ``spool_dir`` is then ``None``.
    """
    name = get_experiment_name(config)
    _spool_cfg, hdf5_dir = get_storage_paths(config)
    out_dir = hdf5_dir or base_path

    hdf5_path, is_existing, ambiguous = _resolve_hdf5(out_dir, name)
    spool_dir = _resolve_spool(spool_root, name) if spool_root else None

    return RunPaths(
        name=name,
        hdf5_path=hdf5_path,
        spool_dir=spool_dir,
        is_existing=is_existing,
        ambiguous=ambiguous,
    )


def _resolve_hdf5(out_dir, name):
    """Return (hdf5_path, is_existing, ambiguous) for ``name`` in ``out_dir``.

    Globs ``<name>_*.hdf5`` (any date). With matches, reuse the newest existing
    file's exact path; with none, mint a fresh ``<name>_<today>.hdf5``.
    """
    matches = _glob_newest_first(os.path.join(out_dir, f"{name}_*.hdf5"))
    if matches:
        return matches[0], True, len(matches) > 1
    fresh = os.path.join(out_dir, hdf5_filename(name))
    return fresh, False, False


def _resolve_spool(spool_root, name):
    """Return the spool subfolder for ``name``: the newest existing ``<name>_*``,
    else a fresh ``<name>_<today>``."""
    matches = _glob_newest_first(os.path.join(spool_root, f"{name}_*"))
    matches = [m for m in matches if os.path.isdir(m)]
    if matches:
        return matches[0]
    return os.path.join(spool_root, f"{name}_{datetime.date.today()}")


def _glob_newest_first(pattern):
    """Paths matching ``pattern``, most-recently-modified first."""
    return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
