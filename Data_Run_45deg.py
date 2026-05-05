# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program with 45degree probe movement.
Run this program to acquire data from multiple scopes and save it in an HDF5 file.

The user should edit this file to:
    1) Set scope IP addresses and motor IP addresses
    2) Set probe position array and movement parameters
    3) Set number of shots and external delays
    4) Set the HDF5 filename and experiment description
    5) Set descriptions for scopes and channels
    6) Configure any other experiment-specific parameters
    7) Set probe movement boundaries

Created on Feb.12.2025
@author: Jia Han
"""

import datetime
import os
import numpy as np
from acquisition import run_acquisition
import time
import sys
import logging
from motion import create_all_positions_45deg

logging.basicConfig(filename='motor.log', level=logging.WARNING, 
                   format='%(asctime)s %(levelname)s %(message)s')

############################################################################################################################
'''
User: Set experiment name and path
'''
exp_name = '17-45deg-p22-1kG-invert-varbias-p30Vmax'  # experiment name
cur_dt = datetime.datetime.now()
path = f"C:\\data"
save_path = f"{path}\\{exp_name}-{cur_dt.month}-{cur_dt.day}-{cur_dt.hour}-{cur_dt.minute}.hdf5"

#-------------------------------------------------------------------------------------------------------------
'''
User: Set probe position array
'''
# Load 45deg probe position parameters from experiment_config.txt
from motion.position_manager import load_config

config = load_config()  # Uses default 'experiment_config.txt'

if config is None:
    print("No position configuration found in experiment_config.txt")
    print("45deg movement requires position configuration with 45deg parameters")
    sys.exit("Please configure [position] section in experiment_config.txt for 45deg movement")
else:
    # Get 45deg specific parameters from config
    xstart = config.get('xstart', {'P16': -38, 'P22': -18, 'P29': -38, 'P34': -38, 'P42': -38})
    xstop = config.get('xstop', {'P16': -38, 'P22': 18, 'P29': -38, 'P34': -38, 'P42': -38})
    nx = config.get('nx', 37)
    nshot = config.get('nshots', 5)


#-------------------------------------------------------------------------------------------------------------
scope_ips = {
    'Scope': '192.168.7.67' # LeCroy WavePro 404HD 4GHz 20GS/s
}

motor_ips = {
    'P16': '192.168.7.141',
    'P22': '192.168.7.142',
    'P29': '192.168.7.143',
    'P34': '192.168.7.144',
    'P42': '192.168.7.145'
}

#===============================================================================================================================================
# Main Data Run sequence
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
            os.remove(save_path)  # Delete the existing file
    
    print('Data run started at', datetime.datetime.now())
    t_start = time.time()
    
    try:
        run_acquisition(save_path, scope_ips, motor_ips, external_delays={}, nz=None, is_45deg=True)
        
    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
    except Exception as e:
        print(f'\n______Halted due to error: {str(e)}______', '  at', time.ctime())
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
    
    # positions, xpos = create_all_positions(['P16', 'P22', 'P29', 'P34', 'P42'], xstart, xstop, nx, nshot)