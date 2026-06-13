# -*- coding: utf-8 -*-
"""Retrofit channel-description attributes onto an existing run HDF5 file.

Old runs never wrote (or mislabeled) the per-channel descriptions, but every
run stores the verbatim ``experiment_config.ini`` text in
``Configuration/experiment_config``. This tool re-parses that stored config and
writes the canonical ``<CH>_description`` attributes onto each scope group --
the same layout new acquisitions write at scope init -- so any reader (e.g.
``read_bmotion_data.py``) finds the descriptions without touching per-shot data.

Usage:
    python -m read_and_analyze.fix_channel_descriptions <file.hdf5> [--force]

Idempotent: scope groups that already have ``<CH>_description`` attributes are
left alone unless ``--force`` rewrites them from the stored config.
"""

import argparse
import configparser
import sys

import h5py

try:  # match acquisition.hdf5_writer: register blosc2 so old files open cleanly
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:
    pass

from acquisition.config import get_channel_descriptions
from acquisition.hdf5_writer import (
    CHANNEL_DESCRIPTION_SUFFIX,
    channel_descriptions_from_attrs,
    scope_channel_descriptions,
)
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Retrofit <CH>_description scope-group attributes onto an "
                    "old run HDF5 file, from its stored experiment config.")
    parser.add_argument("path", help="HDF5 file to fix in place")
    parser.add_argument("--force", action="store_true",
                        help="rewrite attributes even if already present")
    args = parser.parse_args(argv)

    written = fix_file(args.path, force=args.force)
    if not written:
        print("No scope groups found (nothing to fix).")
        return 1
    for scope_name, resolved in written.items():
        if not resolved:
            print(f"{scope_name}: descriptions already present, skipped "
                  "(use --force to rewrite)")
            continue
        print(f"{scope_name}:")
        for channel, text in resolved.items():
            print(f"  {channel}_description = {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
