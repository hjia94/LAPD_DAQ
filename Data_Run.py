# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program with probe movement support.
See multi_scope_acquisition.py for more details.

Configuration and metadata:
- Edit experiment_config.ini to set experiment description, scope/channel descriptions, and probe movement/position parameters.
- Use this script to set file paths, scope and motor IP addresses, and other run-specific parameters.

Created on Feb.14.2024
@author: Jia Han

Update July.2025
- Change experiment description to read from experiment_config.ini
- Change probe position and movement to read from experiment_config.ini


TODO: this script is not optimized for speed. Need to:
- Data_Run_2D.py and Acquire_Scope_Data_2D.py includes saving raw data to disk; this needs to be added here.
- Parallelize the data acquisition
"""

import datetime
import os
from acquisition import run_acquisition
from acquisition.config import get_experiment_name, load_experiment_config
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
    # from it (not from a hard-coded variable in this script).
    config, _ = load_experiment_config(config_path)
    exp_name = get_experiment_name(config)
    date = datetime.date.today()
    hdf5_path = os.path.join(base_path, f"{exp_name}_{date}.hdf5")

    # Check if file already exists
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
        run_acquisition(hdf5_path, config_path)

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