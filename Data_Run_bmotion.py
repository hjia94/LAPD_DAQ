# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program with probe movement using bapsf_motion library.
Run this program to acquire data from multiple scopes and save it in an HDF5 file.
Result is plotted in real time.

The user should edit this file to:


Created on July 24.2025
@author: Jia Han

TODO: this script is not optimized for speed. Need to:
- Data_Run_2D.py and Acquire_Scope_Data_2D.py includes saving raw data to disk; this needs to be added here.
- Parallelize the data acquisition
"""

import datetime
import os
import numpy as np
import time
import sys
import logging

from acquisition import run_acquisition_bmotion

logging.basicConfig(
    filename='motor.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s',
)

############################################################################################################################
'''
User set following
'''
exp_name = '11-Bdot-p20p28-Xline(forward)-He1kG300G-5900A'  # experiment name
date = datetime.date.today()
base_path = r"E:\Shadow data\Energetic_Electron_Ring\AUG2025"
hdf5_path = os.path.join(base_path, f"{exp_name}_{date}.hdf5")
config_path = os.path.join(base_path, 'experiment_config.txt')
toml_path = os.path.join(base_path, 'bmotion_config.toml')

#===============================================================================================================================================
# Main Data Run sequence
#===============================================================================================================================================
def main():
    # Create save directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

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
