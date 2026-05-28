# -*- coding: utf-8 -*-
"""Offload companion to Data_Run_bmotion.py (parallel two-process acquisition).

Run this alongside (or after) the acquisition process. It watches the fast-disk
spool directory and fills each completed shot into the HDF5 file the acquisition
process already created on the slow/large disk, verifies every write by reading
it back, and deletes the spooled copy once verified. It exits cleanly when the
acquisition process drops the RUN_COMPLETE sentinel.

The destination HDF5 path is read verbatim from the spool run metadata
(recorded once by the acquire process); this script only needs spool_dir +
hdf5_dir present in the [storage] section of experiment_config.ini:

    [storage]
    spool_dir = D:\\spool          # fast local disk (e.g. SSD)
    hdf5_dir  = E:\\Shadow data\\Pat   # slow/large disk (e.g. USB 3.0)

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
)
from offload_runner import run_offload
from spooling import spool_format

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

    # The offload neither computes nor creates the destination path: the acquire
    # process records the exact file (and creates it) in the spool metadata.
    print('Offload started at', datetime.datetime.now())
    print(f'  spool_dir = {spool_dir}')
    t_start = time.time()

    hdf5_path = None
    try:
        run_offload(spool_dir, config=config)
    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
    except Exception as e:
        import traceback
        print(f'\n______Halted due to error: {str(e)}______', '  at', time.ctime())
        traceback.print_exc()
    finally:
        # Report the file the acquire side recorded (if metadata was written).
        if spool_format.run_metadata_exists(spool_dir):
            hdf5_path = spool_format.read_run_metadata(spool_dir).get("hdf5_path")
        print('Offload finished at', datetime.datetime.now())
        print('Time taken: %.2f hours' % ((time.time()-t_start)/3600))
        if hdf5_path and os.path.isfile(hdf5_path):
            size = os.stat(hdf5_path).st_size/(1024*1024)
            print(f'Wrote file "{hdf5_path}", {size:.1f} MB')
        else:
            print(f'File "{hdf5_path}" was not created')

#===============================================================================================================================================
if __name__ == '__main__':
    main()
