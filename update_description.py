# -*- coding: utf-8 -*-
"""Overwrite the description attribute in an already-written HDF5 file.

Use this when you forgot to set ``description.txt`` before/during a data run.
It finds the run's HDF5 by the ``[experiment] name`` in
``experiment_config.ini`` (the same identity logic the acquisition uses) and
rewrites ``attrs['description']`` from ``description.txt``. No shot data is
touched -- the file is opened in append mode and only the one attribute changes.

Usage:
    1. Edit description.txt in base_path with the correct text.
    2. python update_description.py
"""

import os

from acquisition.config import (
    get_storage_paths,
    load_experiment_config,
    read_description_file,
)
from acquisition.hdf5_writer import write_description
from acquisition.run_paths import resolve_run_paths

# Keep this in sync with base_path in Data_Run_bmotion.py.
base_path = r"E:\Shadow data\Electrode_Biasing\jun2026"
config_path = os.path.join(base_path, 'experiment_config.ini')
description_path = os.path.join(base_path, 'description.txt')


def main():
    config, _ = load_experiment_config(config_path)
    spool_root, _ = get_storage_paths(config)
    # Resolve by experiment name (globs <name>_*.hdf5) so you don't have to type
    # the dated filename; reuses the newest match for that name.
    paths = resolve_run_paths(config, base_path, spool_root=spool_root)

    if not os.path.isfile(paths.hdf5_path):
        print(f'No HDF5 found for run "{paths.name}" '
              f'(looked for {paths.hdf5_path}). Nothing to update.')
        return

    description = read_description_file(description_path)
    write_description(paths.hdf5_path, description)
    print(f'Updated description in "{paths.hdf5_path}":\n')


if __name__ == '__main__':
    main()
