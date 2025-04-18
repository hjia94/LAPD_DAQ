# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program with probe movement support.
Run this program to acquire data from multiple scopes and save it in an HDF5 file.
Result is plotted in real time.

The user should edit this file to:
    1) Set scope IP addresses and motor IP addresses
    2) Set probe position array and movement parameters
    3) Set number of shots and external delays
    4) Set the HDF5 filename and experiment description
    5) Set descriptions for scopes and channels
    6) Configure any other experiment-specific parameters
    7) Set probe movement boundaries

Created on Feb.14.2024
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
exp_name = 'test01'  # experiment name
date = datetime.date.today()
path = f"C:\\data"
save_path = f"{path}\\{exp_name}_{date}.hdf5"

#-------------------------------------------------------------------------------------------------------------
'''
User: Set probe position array
'''
# Probe position parameters
xmin = 0
xmax = 0
nx = 1

ymin = 20
ymax = 22
ny = 2

# Set z parameters to None if not using XYZ drive
zmin = 4 #-13
zmax = 4  #4
nz = 1

num_duplicate_shots = 5      # number of duplicate shots recorded at each location
num_run_repeats = 1          # number of times to repeat sequentially over all locations
#-------------------------------------------------------------------------------------------------------------
'''
User: Set probe movement boundaries
'''
# Define probe movement limits with 3D drive only
x_limits = (-40, 200)  # (min, max) in cm
y_limits = (-40, 40)
z_limits = (-15, 15)

# Motor limit swtich for 2D or 3D drive
xm_limits = (-84, 38)
ym_limits = (-39, 48)
zm_limits = (-24, 9)

def outer_boundary(x, y, z):
    """Return True if position is within allowed range"""
    return (x_limits[0] <= x <= x_limits[1] and 
            y_limits[0] <= y <= y_limits[1] and 
            z_limits[0] <= z <= z_limits[1])

def obstacle_boundary(x, y, z):
    """Return True if position is NOT in obstacle"""
    # Check large box obstacle (30x6x11 cm box from x=-50 to -20)
    buffer = 0.2  # Small buffer to ensure paths don't get too close
    in_obstacle = ( -60 <= x <= -17 and 
                    -2.5 <= y <= 5 and 
                    -6.5 <= z <= 9)
    
    return not in_obstacle

def motor_boundary(x, y, z):
    """Return True if position is within allowed range"""
    # Check outer boundary
    in_outer_boundary = (xm_limits[0] <= x <= xm_limits[1] and 
                        ym_limits[0] <= y <= ym_limits[1] and 
                        zm_limits[0] <= z <= zm_limits[1])
    return in_outer_boundary

def motor_boundary_2D(x, y, z):
    """Return True if position is within allowed range"""
    # Check outer boundary
    in_outer_boundary = (xm_limits[0] <= x <= xm_limits[1] and 
                        ym_limits[0] <= y <= ym_limits[1] and
                        -999 <= z <= 999)
    return in_outer_boundary

#-------------------------------------------------------------------------------------------------------------
def get_experiment_description():

    """Return overall experiment description"""
    return f'''
    Experiment: {exp_name}
    Date: {date}
    Operator: Jia Han
    
    Probe Movement:
    - X range: {xmin} to {xmax} cm, {nx} points
    - Y range: {ymin} to {ymax} cm, {ny} points
    - Z range: {zmin} to {zmax} cm, {nz} points (part 1 was -12 to -4)
    - {num_duplicate_shots} shots per position
    - {num_run_repeats} full scan repeats
    
    Setup:
    - Plasma condition
        - Heater 2150 A
        - Puff Helium backside pressure 40 Psi
        - Puff voltage 81V for 31ms West+East
        - Hydrogen 200 SCCM MFC is set to "400"
        - Discharge 23 ms; bank charging 73 V; current 4.0 kA
        - Pulsing 1/4.25 Hz; plasma breakdown ~9 ms
        - Pressure ~0.155 mTorr
        - Interferometer density 9e12=>8e12 @P20;  5e12=>4e12 @P29 (assume 40cm)
    - Magnetic field
        - Straight 1.2 kG
        - Black (South) 1.2kG (673 A)
        - Yellow 3120 A
        - Purple 1092 A
        - Black (North) 0 A
    - Antenna (ZZ-#3 six wire mesh paddles wt 1/8inch gap)
        - connected +-+-+- from south to north at 2.48GHz
        - paddles are connected using delay lines to generate pi phase shift at 2.48 GHz
        - tip of mesh is approx 31 cm past wall (x = approx -19)
        - LMX2572 signal generator set to 2.48 GHz, setpoint "30" in TICS software
        - LMX2572 output goes to the RF switch, then to a ($200) DC block then to a -6dB attenuator on the input of the amplifier
        - The amplifier is floating at the plasma potential because there is a direct connection to the mesh launcher
        - The output of the amplifier goes through a directional coupler, then to a 25 foot coax.
        - The coax has a measured attenuation of ?dB at 1.1 GHz  - probably 2.4 dB based on 2.5 GHz measurement
        - The -20dB signal from the directional coupler goes to a -6dB attenuator, then to a ($200) DC block, then a 6 dB attenuator, then to channel 2 of the scope
        - The rf switch is enabled by a Keysight Function generator triggered by Stanford
        - Each RF burst is 30ns long starting at t = T_0 ms;
        - Keysight setting: Freq 6.25kHz, 44 cycles
    - Probe
        - Dipole probe DP-JL-2CEW-1 (2 pairs of tips, Y and Z direction)
        - we are connected to the "+y" whisker
        - then a ($200) dc block and limiter
        - then x100 0.15-2.5GHz amplifier, followed by a second ($200) limiter
        - then to channel 3 of the scope

        note: the ($200) DC blocks described above break both ground and signal connections and introduce only a few degrees of phase at 3 GHz

    - Scope descriptions: See scope_group.attrs['description']
    - Channel descriptions: See channel_group.attrs['description']

    Notes:
    wave is launched with respect to plasma 1kA as T=0
    '''
#-------------------------------------------------------------------------------------------------------------
scope_ips = {
    'FastScope': '192.168.7.63' # LeCroy WavePro 404HD 4GHz 20GS/s
}

motor_ips = { # For 3D X:163, Y:165, Z:164
    'x': '192.168.7.163',  # X-axis motor 163
    'y': '192.168.7.165',   # Y-axis motor 165
    'z': '192.168.7.164'   # Z-axis motor 164
}

def get_channel_description(tr):
    """Channel description"""
    descriptions = {
        'FastScope_C1': 'N/A',
        'FastScope_C2': 'RF signal at amplifier output',
        'FastScope_C3': 'Probe signal',
        'FastScope_C4': 'N/A'
    }
    return descriptions.get(tr, f'Channel {tr} - No description available')

def get_scope_description(scope_name):
    """Return description for each scope"""
    descriptions = {
        'FastScope': '''LeCroy WavePro 404HD 4GHz 20GS/s; triggering on channel 2 (RF signal)'''
    }
    return descriptions.get(scope_name, f'Scope {scope_name} - No description available')

external_delays = { # unit: milliseconds
    'FastScope': 0
}

#-------------------------------------------------------------------------------------------------------------
def get_positions_xy():
    """Generate the positions array for probe movement.
    Returns:
        tuple: (positions, xpos, ypos)
            - positions: Array of tuples (shot_num, x, y)
            - xpos: Array of x positions
            - ypos: Array of y positions
    """
    if nx == 0 or ny == 0:
        sys.exit('Position array is empty.') 
        
    xpos = np.linspace(xmin, xmax, nx)
    ypos = np.linspace(ymin, ymax, ny)

    # Calculate total number of positions including duplicates and repeats
    total_positions = nx * ny * num_duplicate_shots * num_run_repeats

    # Allocate the positions array
    positions = np.zeros(total_positions, 
                        dtype=[('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4')])


    # Create rectangular shape position array
    index = 0
    for repeat_cnt in range(num_run_repeats):
        for y in ypos:
            for x in xpos:
                for dup_cnt in range(num_duplicate_shots):
                    positions[index] = (index + 1, x, y)
                    index += 1
                    
    return positions, xpos, ypos

def get_positions_xyz():
    """Generate the positions array for probe movement in 3D.
    Returns:
        tuple: (positions, xpos, ypos, zpos)
            - positions: Array of tuples (shot_num, x, y, z)
            - xpos: Array of x positions
            - ypos: Array of y positions
            - zpos: Array of z positions
    """
    if nx == 0 or ny == 0 or nz == 0:
        sys.exit('Position array is empty.')
        
    xpos = np.linspace(xmin, xmax, nx)
    ypos = np.linspace(ymin, ymax, ny) 
    zpos = np.linspace(zmin, zmax, nz)

    # Calculate total number of positions including duplicates and repeats
    total_positions = nx * ny * nz * num_duplicate_shots * num_run_repeats

    # Allocate the positions array
    positions = np.zeros(total_positions,
                        dtype=[('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])

    # Create 3D rectangular shape position array
    index = 0
    for repeat_cnt in range(num_run_repeats):
        for z in zpos:
            for y in ypos:
                for x in xpos:
                    for dup_cnt in range(num_duplicate_shots):
                        positions[index] = (index + 1, x, y, z)
                        index += 1
                    
    return positions, xpos, ypos, zpos

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
            run_acquisition(save_path, scope_ips, motor_ips, external_delays, nz)
        
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
    # Run a test acquisition with minimal settings
    # run_test(test_save_path = r"E:\Shadow data\Energetic_Electron_Ring\test.hdf5")
    main()