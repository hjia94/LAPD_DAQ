# -*- coding: utf-8 -*-
"""Offload companion to Data_Run_bmotion.py (parallel two-process acquisition).

Run this alongside (or after) the acquisition process. It watches the fast-disk
spool directory, writes each completed shot into the final HDF5 file on the
slow/large disk, verifies every write by reading it back, and deletes the
spooled copy once verified. It exits cleanly when the acquisition process drops
the RUN_COMPLETE sentinel.

Paths come from the [storage] section of experiment_config.ini:

    [storage]
    spool_dir = D:\\spool\\my_run      # fast local disk (e.g. SSD)
    hdf5_path = E:\\Shadow data\\Pat\\my_run.hdf5   # slow/large disk (e.g. USB 3.0)

Usage:
    python Offload_Run.py
"""

import datetime
import os
import sys
import time

from acquisition.config import (
    get_storage_paths,
    load_experiment_config,
    resolve_hdf5_path,
)
from offload_runner import run_offload

############################################################################################################################
'''
User set following
'''
base_path = r"E:\Shadow data\Pat"
config_path = os.path.join(base_path, 'experiment_config.ini')

#===============================================================================================================================================
def main():
    config, _ = load_experiment_config(config_path)
    spool_dir, hdf5_dir = get_storage_paths(config)

    if not spool_dir or not hdf5_dir:
        print("No [storage] section with spool_dir + hdf5_dir found in "
              f"{config_path}. Nothing to offload.")
        sys.exit(1)

    # Derive the same destination file the acquisition process targets.
    hdf5_path = resolve_hdf5_path(config, base_path)

    # Guard the destination file the same way Data_Run does.
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
            os.remove(hdf5_path)

    print('Offload started at', datetime.datetime.now())
    print(f'  spool_dir = {spool_dir}')
    print(f'  hdf5_path = {hdf5_path}')
    t_start = time.time()

    try:
        run_offload(spool_dir, hdf5_path, config=config)
    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
    except Exception as e:
        import traceback
        print(f'\n______Halted due to error: {str(e)}______', '  at', time.ctime())
        traceback.print_exc()
    finally:
        print('Offload finished at', datetime.datetime.now())
        print('Time taken: %.2f hours' % ((time.time()-t_start)/3600))
        if os.path.isfile(hdf5_path):
            size = os.stat(hdf5_path).st_size/(1024*1024)
            print(f'Wrote file "{hdf5_path}", {size:.1f} MB')
        else:
            print(f'File "{hdf5_path}" was not created')

#===============================================================================================================================================
if __name__ == '__main__':
    main()
