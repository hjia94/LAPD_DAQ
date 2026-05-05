# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program with probe movement support.
See multi_scope_acquisition.py for more details.

Configuration and metadata:
- Edit experiment_config.txt to set experiment description, scope/channel descriptions, and probe movement/position parameters.
- Use this script to set file paths, scope and motor IP addresses, and other run-specific parameters.

Created on Feb.14.2024
@author: Jia Han

Update July.2025
- Change experiment description to read from experiment_config.txt
- Change probe position and movement to read from experiment_config.txt


TODO: this script is not optimized for speed. Need to:
- Data_Run_2D.py and Acquire_Scope_Data_2D.py includes saving raw data to disk; this needs to be added here.
- Parallelize the data acquisition
"""

import datetime
import os
from acquisition import run_acquisition
import time
import sys

#===============================================================================================================================================
#===============================================================================================================================================
'''
User: Set experiment name and path
'''
exp_name = '03_speed_test'  # experiment name
date = datetime.date.today()
path = r"C:\data\Energetic_Electron_Ring"
save_path = f"{path}\\{exp_name}_{date}.hdf5"
config_path = r"C:\data\Energetic_Electron_Ring\experiment_config.txt"

#===============================================================================================================================================
# Main function
#===============================================================================================================================================
def main():
    # Create save directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
        
    # Check if file already exists
    if os.path.exists(save_path):
        while True:
            response = input(f'File "{save_path}" already exists. Overwrite? (y/n): ').lower()
            if response in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")
            
        if response == 'n':
            print('Exiting without overwriting existing file')
            sys.exit()
        else:
            print('Overwriting existing file')
            os.remove(save_path)
    
    print('Data run started at', datetime.datetime.now())
    t_start = time.time()
    
    try:
        run_acquisition(save_path, config_path)
    
    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
    finally:
        print('Data run finished at', datetime.datetime.now())
        print('Time taken: %.2f hours' % ((time.time()-t_start)/3600))
        
        # Print file size if it was created
        if os.path.isfile(save_path):
            size = os.stat(save_path).st_size/(1024*1024)
            print(f'Wrote file "{save_path}", {size:.1f} MB')
        else:
            print(f'File "{save_path}" was not created')

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
    main()