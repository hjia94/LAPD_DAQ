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

import argparse
import datetime
import logging
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
DEFAULT_BASE_PATH = r"E:\Shadow data\Pat"


def _parse_args():
    p = argparse.ArgumentParser(description="Offload spooled shots into the HDF5.")
    p.add_argument("--spool-dir", default=None,
                   help="Override spool dir (else read from config [storage]).")
    p.add_argument("--config", default=os.path.join(DEFAULT_BASE_PATH,
                                                     "experiment_config.ini"),
                   help="Path to experiment_config.ini.")
    return p.parse_args()


#===============================================================================================================================================
def main():
    args = _parse_args()
    config_path = args.config
    config, _ = load_experiment_config(config_path)
    cfg_spool_dir, hdf5_dir = get_storage_paths(config)
    spool_dir = args.spool_dir or cfg_spool_dir

    if not spool_dir or not hdf5_dir:
        print("No [storage] section with spool_dir + hdf5_dir found in "
              f"{config_path}. Nothing to offload.")
        sys.exit(1)

    # Log failures/issues to a file in the spool folder (mirrors the acquire
    # side's motor.log). A named logger keeps offload logging isolated from root.
    os.makedirs(spool_dir, exist_ok=True)
    log_path = os.path.join(spool_dir, "offload.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    off_logger = logging.getLogger("offload")
    off_logger.setLevel(logging.WARNING)
    off_logger.addHandler(handler)

    # The offload neither computes nor creates the destination path: the acquire
    # process records the exact file (and creates it) in the spool metadata.
    print('Offload started at', datetime.datetime.now())
    print(f'  spool_dir = {spool_dir}')
    print(f'  logging failures to {log_path}')
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
