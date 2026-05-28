# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program with probe movement using bapsf_motion library.
Run this program to acquire data from multiple scopes and save it in an HDF5 file.
Result is plotted in real time.

The user should edit this file to:


Created on July 24.2025
@author: Jia Han

Parallel mode: if experiment_config.ini has a [storage] section with a fast
spool_dir, this process creates the HDF5 file + its skeleton (metadata, time
arrays, positions) on the slow/large disk and then spools each shot's raw data
to the fast disk; a separate `Offload_Run.py` process fills those shots into the
same HDF5 file. Otherwise the legacy single-process path writes the HDF5 directly.
"""

import datetime
import os
import numpy as np
import time
import sys
import logging
import subprocess

from acquisition import run_acquisition_bmotion, run_acquisition_bmotion_spooled
from acquisition.config import (
    get_storage_paths,
    load_experiment_config,
    resolve_hdf5_path,
)

logging.basicConfig(
    filename='motor.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s',
)

############################################################################################################################
'''
User sets only the base path below. The experiment name lives in
experiment_config.ini ([experiment] name = ...); the config and bmotion TOML
are found inside base_path, and the HDF5 filename is built from the parsed
experiment name after the config is read.
'''
base_path = r"E:\Shadow data\Pat"
config_path = os.path.join(base_path, 'experiment_config.ini')
toml_path = os.path.join(base_path, 'bmotion_config.toml')

#===============================================================================================================================================
# Main Data Run sequence
#===============================================================================================================================================
def main():
    # Create save directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Parse the config first; the experiment name and the HDF5 filename come
    # from it (not from a hard-coded variable in this script). The output file
    # is <hdf5_dir or base_path>/<exp_name>_<date>.hdf5.
    config, _ = load_experiment_config(config_path)
    hdf5_path = resolve_hdf5_path(config, base_path)

    # Parallel mode is enabled when [storage] provides a fast spool_dir.
    spool_dir, _hdf5_dir = get_storage_paths(config)
    spooled = bool(spool_dir)

    # The acquire process creates the destination HDF5 (and writes its skeleton)
    # in BOTH modes, so guard/overwrite it up front regardless of spooling.
    if os.path.exists(hdf5_path):
        while True:
            response = input(f'File "{hdf5_path}" already exists. Overwrite? (y/n): ').lower()
            if response in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")

        if response == 'n':
            print('Exiting without overwriting existing file')
            sys.exit()
        else:
            print('Overwriting existing file')
            os.remove(hdf5_path)  # Delete the existing file

    if spooled:
        # Acquisition writes the HDF5 skeleton + spools per-shot data; a separate
        # Offload_Run.py process fills the shots into the same HDF5.
        if not os.path.exists(spool_dir):
            os.makedirs(spool_dir)
        print(f'PARALLEL mode: spooling shots to {spool_dir}')
        # Auto-launch the offload in its own console window; it politely waits for
        # run metadata (offload_runner._wait_for), so launching before metadata
        # exists is safe. We detach (no wait) so it keeps draining after this
        # acquire process exits.
        offload_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'Offload_Run.py')
        subprocess.Popen(
            [sys.executable, offload_script,
             '--spool-dir', spool_dir, '--config', config_path],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        print('  Launched Offload_Run.py in a new console window.')

    print('Data run started at', datetime.datetime.now())
    t_start = time.time()

    try:
        if spooled:
            run_acquisition_bmotion_spooled(spool_dir, hdf5_path, toml_path, config_path)
        else:
            run_acquisition_bmotion(hdf5_path, toml_path, config_path)

    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
    except Exception as e:
        import traceback
        print(f'\n______Halted due to error: {str(e)}______', '  at', time.ctime())
        print("Full traceback:")
        traceback.print_exc()
    finally:
        print('Data run finished at', datetime.datetime.now())
        print('Time taken: %.2f hours' % ((time.time()-t_start)/3600))

        if spooled:
            print(f'Shots spooled to "{spool_dir}". The auto-launched '
                  f'Offload_Run.py console keeps draining into the HDF5 file.')
        elif os.path.isfile(hdf5_path):
            size = os.stat(hdf5_path).st_size/(1024*1024)
            print(f'Wrote file "{hdf5_path}", {size:.1f} MB')
        else:
            print(f'File "{hdf5_path}" was not created')


#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
    main()
