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
from multi_scope_acquisition import run_acquisition
import time
import sys
import logging

logging.basicConfig(filename='motor.log', level=logging.WARNING, 
                   format='%(asctime)s %(levelname)s %(message)s')

############################################################################################################################
'''
User: Set experiment name and path
'''
exp_name = '03_P22_P34_0'  # experiment name
cur_dt = datetime.datetime.now()
path = f"C:\\data"
save_path = f"{path}\\{exp_name}_{cur_dt.year}-{cur_dt.month}-{cur_dt.day}-{cur_dt.hour}:{cur_dt.minute}.hdf5"

#-------------------------------------------------------------------------------------------------------------
'''
User: Set probe position array
'''
# Set up position array (unit: cm)
xstart = {'P16': 0, 'P22': -30, 'P29': -40, 'P34': -30, 'P42': 0}
xstop  = {'P16': 0, 'P22': 16, 'P29': -40, 'P34': 16, 'P42': 0}
nx = 47       # number of positions

nshot = 8    # number of shots at each position

#-------------------------------------------------------------------------------------------------------------
def get_experiment_description():

    """Return overall experiment description"""
    return f'''
    Test data acquisition program at single location
    
    Experiment: {exp_name}
    Date: {date}
    Operator: Yhoshua W.
    
    Probe Movement:
    - 45deg probe located at P16, P22, P29, P34, P42
    - Probe moves on single radial line    
    - Number of shots at each position: {nshot}
    - Number of positions: {nx}
     

    Purpose: TP 45deg probe drives.  100ks, 50ms window on scope

    LAPD B field:
    ========
    Black magnets at south: 1125 A (PS12, 13) 2000 G
    Yellow magnets: 1000 G
    Purple magnets:  1000 G 

    South LaB6 source:
    ============
    He plasma , discharge PS voltage/current:93V/4460A 
    2/3 Hz rep rate 
    Heater: 2.106 kA
    Fill Pres: 1.3e-4 torr
    Gas Puff (96.5 V on East and West, 19.4 ms) Helium
    40 PSI on tank

    Timing:
    =====
    Discharge Period: 20 ms
    Breakdown time: 5 ms
    DAQ trigger: 0 ms
    RF turn on: 14ms
    Scope trigger: -5 ms

    ANTENA:
    =========
    RF ON: Driving antenna with black box. capacitors in parallel (0.15x2=0.3 micro F) and capacitor in series (2 micro F)
    Using steve's antenna (P26-P28).
    Antenna turn on: 14 ms
    PS Voltage: 430 V
    Driver current: 
    500 cycles (close to 2.5 ms)

    - Probe
        - 4-tip Langmuir with Tunsten wire 1mm long
        - Triple probe boxes are used (see https://drive.google.com/drive/folders/1-9VcMYBBSbKsqKTpIVFkImKsi8OC1s4K?usp=drive_link)
        - P16 -- Not active
        - P22 -- Isat (R knob #5) 
        - P29 -- Not active
        - P34 -- Isat (R knob #5)
        - P42 -- Not active

    - Scope descriptions: See scope_group.attrs['description']
    - Channel descriptions: See channel_group.attrs['description']
    '''
#-------------------------------------------------------------------------------------------------------------
scope_ips = {
    'Scope': '192.168.7.64' # LeCroy WavePro 404HD 4GHz 20GS/s
}

motor_ips = {
    'P16': '192.168.7.141',
    'P22': '192.168.7.142',
    'P29': '192.168.7.143',
    'P34': '192.168.7.144',
    'P42': '192.168.7.145'
}

def get_channel_description(tr):
    """Channel description"""
    descriptions = {
        'Scope_C1': 'Isat_P22',
        'Scope_C2': 'Te_P22',
        'Scope_C3': 'Isat_P34',
        'Scope_C4': 'Te_P34'
    }
    return descriptions.get(tr, f'Channel {tr} - No description available')

def get_scope_description(scope_name):
    """Return description for each scope"""
    descriptions = {
        'Scope': '''LeCroy '''
    }
    return descriptions.get(scope_name, f'Scope {scope_name} - No description available')

def get_positions(xstart, xstop, nx, nshots):
	""" 
	callback function to return the positions array
	"""

	if nx==0:
		sys.exit('Position array is empty.')
        
	xpos = np.linspace(xstart, xstop, nx)

	# allocate positions array, fill it with zeros
	positions = np.zeros((nx*nshots), dtype=[('shot_num', np.int32), ('x', np.float64)])

	#create rectangular shape position array with height z
	index = 0

	for x in xpos:
		for dup_cnt in range(nshots):
			positions[index] = (index+1, x)
			index += 1
                        
	return positions, xpos

def create_all_positions(pr_ls, xstart, xstop, nx, nshots):
	"""
	create position array for all probes
	"""

	positions = {}
	xpos = {}

	for pr in pr_ls:
		positions[pr], xpos[pr] = get_positions(xstart[pr], xstop[pr], nx, nshots)

	return positions, xpos
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
        run_acquisition(save_path, scope_ips, motor_ips, is_45deg=True)
        
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