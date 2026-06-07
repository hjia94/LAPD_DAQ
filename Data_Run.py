# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program (grid / stationary modes).

Acquisition is spooled-only: experiment_config.ini must have a [storage]
section with a fast spool_dir. This process creates the HDF5 file + skeleton and
spools each shot's raw data; a separate Offload_Run.py process fills the shots
into the HDF5 file. (The legacy single-process, non-spooled path was removed;
recover it from git history if ever needed.)

Configuration and metadata:
- Edit experiment_config.ini to set scope/channel descriptions and probe movement/position parameters.
- Put the free-text run description in description.txt next to the config (written into HDF5 at run start, overwritten at run end).
- Use this script to set the base path; scope/motor IPs and run parameters live in experiment_config.ini.

Created on Feb.14.2024
@author: Jia Han

Update July.2025
- Change probe position and movement to read from experiment_config.ini
- Move the run description to description.txt next to the config
"""

import datetime
import os
from acquisition import run_acquisition_spooled
from acquisition.config import (
    get_storage_paths,
    load_experiment_config,
    resolve_hdf5_path,
)
import time
import sys

############################################################################################################################
'''
User sets only the base path below. The experiment name lives in
experiment_config.ini ([experiment] name = ...); the config is found inside
base_path, and the HDF5 filename is built from the parsed experiment name
after the config is read.
'''
base_path = r"E:\Shadow data\Pat"
config_path = os.path.join(base_path, 'experiment_config.ini')

#===============================================================================================================================================
# Main function
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

    # Acquisition is spooled-only: [storage] must provide a fast spool_dir.
    # The legacy single-process (non-spooled) path was removed; without a
    # spool_dir there is nothing to run, so fail loudly instead of silently
    # falling back. (Recover it from git history if ever needed.)
    spool_dir, _hdf5_dir = get_storage_paths(config)
    if not spool_dir:
        print('ERROR: no [storage] spool_dir configured. Non-spooled mode was '
              'removed; set a spool_dir in experiment_config.ini and run '
              'Offload_Run.py to fill the HDF5.')
        sys.exit(1)

    # The acquire process creates the destination HDF5 (and writes its
    # skeleton), so guard/overwrite it up front.
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

    if not os.path.exists(spool_dir):
        os.makedirs(spool_dir)
    print(f'PARALLEL mode: spooling shots to {spool_dir}')
    print(f'  Run Offload_Run.py to fill the HDF5 file ({hdf5_path}).')

    print('Data run started at', datetime.datetime.now())
    t_start = time.time()

    try:
        run_acquisition_spooled(spool_dir, hdf5_path, config_path)

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

        # Print file size if it was created
        if os.path.isfile(hdf5_path):
            size = os.stat(hdf5_path).st_size/(1024*1024)
            print(f'Wrote file "{hdf5_path}", {size:.1f} MB')
        else:
            print(f'File "{hdf5_path}" was not created')

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
    main()