# -*- coding: utf-8 -*-
"""Retrofit channel-description attributes onto an existing run HDF5 file.

Old runs never wrote (or mislabeled) the per-channel descriptions, but every
run stores the verbatim ``experiment_config.ini`` text in
``Configuration/experiment_config``. This tool re-parses that stored config and
writes the canonical ``<CH>_description`` attributes onto each scope group --
the same layout new acquisitions write at scope init -- so any reader (e.g.
``read_bmotion_data.py``) finds the descriptions without touching per-shot data.

Usage (a single file, or every ``*.hdf5`` in a folder):
    python -m read_and_analyze.fix_channel_descriptions <file_or_folder> [--force]
    python -m read_and_analyze.fix_channel_descriptions <folder> --recursive

Idempotent: scope groups that already have ``<CH>_description`` attributes are
left alone unless ``--force`` rewrites them from the stored config.
"""

import argparse
import configparser
import sys
from pathlib import Path

import h5py

try:  # match acquisition.hdf5_writer: register blosc2 so old files open cleanly
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:
    pass

# Allow running directly (IDE "Run" button / from inside this folder) as well as
# ``python -m read_and_analyze.<module>`` from the repo root: the root-level
# ``acquisition``/``read_and_analyze`` packages need the repo root on sys.path,
# which ``-m`` adds but a direct script run does not, so put it there ourselves.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from acquisition.config import get_channel_descriptions
from acquisition.hdf5_writer import scope_channel_descriptions
from scope_io import CHANNEL_DESCRIPTION_SUFFIX, channel_descriptions_from_attrs
from read_and_analyze.read_bmotion_data import _channel_names, _shot_numbers


def _stored_channel_map(f):
    """Parse ``[channels]`` out of the config text stored in the HDF5 file.

    Reuses ``acquisition.config.get_channel_descriptions`` so the parse (inline
    comments, optionxform lowercasing) matches acquisition exactly.
    """
    raw = f["Configuration/experiment_config"][()]
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read_string(text)
    return get_channel_descriptions(config)


def _scope_channels(scope_group):
    """Channel names for a scope, from its first shot that holds data.

    Skipped shots are marker groups with no ``*_data`` datasets, so scan shot
    groups in numeric order until one yields channels.
    """
    for num in _shot_numbers(scope_group):
        channels = _channel_names(scope_group, num)
        if channels:
            return channels
    return []


def fix_file(path, force=False):
    """Write ``<CH>_description`` scope-group attrs from the stored config.

    Returns ``{scope_name: {channel: description}}`` for what was written
    (empty inner dict = scope skipped because attrs already present).
    """
    written = {}
    with h5py.File(path, "r+") as f:
        channel_map = _stored_channel_map(f)
        for scope_name, group in f.items():
            if not isinstance(group, h5py.Group):
                continue
            channels = _scope_channels(group)
            if not channels:
                # Not a scope group, or no data shots to derive channels from.
                continue
            if channel_descriptions_from_attrs(group.attrs) and not force:
                written[scope_name] = {}
                continue
            resolved = scope_channel_descriptions(channel_map, scope_name, channels)
            for channel, text in resolved.items():
                group.attrs[f"{channel}{CHANNEL_DESCRIPTION_SUFFIX}"] = text
            written[scope_name] = resolved
    return written


def find_hdf5_files(folder, recursive=False):
    """List ``*.hdf5`` files in ``folder`` (matching either case), sorted.

    ``Path.glob``/``rglob`` only yield files for a ``*.ext`` pattern, so the
    result never contains directories.
    """
    folder = Path(folder)
    walk = folder.rglob if recursive else folder.glob
    return sorted({p for ext in ("*.hdf5", "*.HDF5") for p in walk(ext)})


def _print_file_result(path, written):
    """Print the per-scope summary for one file's :func:`fix_file` result."""
    print(path)
    if isinstance(written, Exception):
        print(f"  ERROR: {written}")
        return
    if not written:
        print("  No scope groups found (nothing to fix).")
        return
    for scope_name, resolved in written.items():
        if not resolved:
            print(f"  {scope_name}: descriptions already present, skipped "
                  "(use --force to rewrite)")
            continue
        print(f"  {scope_name}:")
        for channel, text in resolved.items():
            print(f"    {channel}_description = {text}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Retrofit <CH>_description scope-group attributes onto old "
                    "run HDF5 file(s), from their stored experiment config. "
                    "Accepts a single .hdf5 file or a folder of them.")
    parser.add_argument("path", help="HDF5 file, or folder of *.hdf5 files, to fix in place")
    parser.add_argument("--force", action="store_true",
                        help="rewrite attributes even if already present")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="when path is a folder, also descend into subfolders")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if path.is_dir():
        files = find_hdf5_files(path, recursive=args.recursive)
        if not files:
            print(f"No *.hdf5 files found in {path}.")
            return 1
    else:
        files = [path]

    failed = 0
    for f in files:
        try:
            written = fix_file(str(f), force=args.force)
        except Exception as exc:  # one bad file must not abort the batch
            written = exc
            failed += 1
        _print_file_result(f, written)

    if len(files) > 1:
        print(f"\nProcessed {len(files)} file(s), {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
