# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program with probe movement using bapsf_motion library.
Run this program to acquire data from multiple scopes and save it in an HDF5 file.
Result is plotted in real time.

The user should edit this file to:


Created on July 24.2025
@author: Jia Han

Parallel mode: if experiment_config.ini has a [storage] section with both
spool_dir (fast disk) and hdf5_path (slow/large disk), acquisition writes shots
to the spool and a separate `Offload_Run.py` process offloads them into the
HDF5 file. Otherwise the legacy single-process path writes the HDF5 directly.
"""

import datetime
import os
import numpy as np
import time
import sys
import logging

from acquisition import run_acquisition_bmotion, run_acquisition_bmotion_spooled
from acquisition.config import (
    get_experiment_name,
    get_storage_paths,
    load_experiment_config,
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
    # from it (not from a hard-coded variable in this script).
    config, _ = load_experiment_config(config_path)
    exp_name = get_experiment_name(config)
    date = datetime.date.today()
    hdf5_path = os.path.join(base_path, f"{exp_name}_{date}.hdf5")

    # Parallel mode is enabled when [storage] provides a fast spool_dir.
    spool_dir, storage_hdf5_path = get_storage_paths(config)
    if storage_hdf5_path:
        hdf5_path = storage_hdf5_path
    spooled = bool(spool_dir)

    if spooled:
        # Acquisition writes shots to the spool; Offload_Run.py builds the HDF5.
        if not os.path.exists(spool_dir):
            os.makedirs(spool_dir)
        print(f'PARALLEL mode: spooling shots to {spool_dir}')
        print(f'  Run Offload_Run.py to write the HDF5 file ({hdf5_path}).')
    else:
        # Legacy single-process: guard the destination HDF5 up front.
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

    print('Data run started at', datetime.datetime.now())
    t_start = time.time()

    try:
        if spooled:
            run_acquisition_bmotion_spooled(spool_dir, toml_path, config_path)
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
            print(f'Shots spooled to "{spool_dir}". '
                  f'Offload_Run.py writes the final HDF5 file.')
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
