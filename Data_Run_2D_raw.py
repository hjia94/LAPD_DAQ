# -*- coding: utf-8 -*-

"""
2D data acquisition program modified based on program used in plasma processing lab (Scope_DAQ)

Run this program to acquire data and save the raw data on the disk.

The user should edit this file at lines labeled #user, to
	1) Set up the positions array
	2) Set the Lecroy scope IP address and the IP addresses of the motors.
	4) Set descriptions of the channels being recorded

Created on Oct.20.2024
@author: Jia Han, Donglai Ma

"""
import os
import time

import numpy as np
from Acquire_Scope_Data_2D import Acquire_Scope_Data_to_Disk
from LeCroy_Scope import EXPANDED_TRACE_NAMES

#from tkinter import filedialog
import logging
import sys

logging.basicConfig(filename='motor.log', level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
############################################################################################################################
'''
user: set up simple positions array here (see function get_positions() below)
'''
xmin = 0
xmax = 40
nx   = 41

ymin = 0
ymax = 0
ny   = 1

num_duplicate_shots = 9     # number of duplicate shots recorded at the ith location
num_run_repeats = 1           # number of times to repeat sequentially over all locations
DISK_FIRST = 1 # 1: save data to disk first, then to hdf5 file
exp_name = 'EER_test1'
#-------------------------------------------------------------------------------------------------------------
'''
User: set channel descriptions
'''
def get_channel_description(tr) -> str:
	""" callback function to return a string containing a description of the data in each recorded channel """

	#user: assign channel description text here to override the default:
	if tr == 'C1':
		return 'N/A'
	if tr == 'C2':
		return 'Bx'
	if tr == 'C3':
		return 'By'
	if tr == 'C4':
		return 'N/A'
	if tr == 'C5':
		return 'N/A'
	if tr == 'C6':
		return 'N/A'
	if tr == 'C7':
		return 'N/A'
	if tr == 'C8':
		return 'N/A'
	
	if tr == 'F1':
		return 'Voltage at probe tip (C3 - C4)'#'Antenna power, product of Vant(voltage divider) and C1'
    
	# otherwise, program-generated default description strings follow
	if tr in EXPANDED_TRACE_NAMES.keys():
		return 'no entered description for ' + EXPANDED_TRACE_NAMES[tr]

	return '**** get_channel_description(): unknown trace indicator "'+tr+'". How did we get here?'

#-------------------------------------------------------------------------------------------------------------
'''
user: set known ip addresses:
   scope  - For digitization
   x  - motion in/out. IP address set by dial on motor
   y  - motion transverse. IP address set by dial on motor
   z  - Not Used
   agilent - Not Used
'''
ip_addrs = {'scope':'192.168.7.63', 'x':'192.168.7.161', 'y':'192.168.7.162'}

#-------------------------------------------------------------------------------------------------------------
'''
user: set output file name, or None for prompt (see function get_hdf5_filename() below)
'''
hdf5_filename = r"D:\Data\Energetic_Electron_Ring\test.hdf5"

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================


def get_positions():
    
	"""
	callback function to return the positions array in a legacy format.
	In particular, we assign the positions array as an array of tuples: pos[0] = index, pos[1] = x, pos[2] = y, pos[3] = z
	For eventual convenience, we also store the linear xpos and ypos arrays in the hdf5 file; if these are not relevant set them to None (i.e. the last line should be   return positions,None,None)
	"""
	global xmin, xmax, nx
	global ymin, ymax, ny
 

	if nx==0 or ny==0:
		sys.exit('Position array is empty.') 
        
	xpos = np.linspace(xmin,xmax,nx)
	ypos = np.linspace(ymin,ymax,ny)

	nx = len(xpos)
	ny = len(ypos)

	# allocate the positions array, fill it with zeros
	positions = np.zeros((nx*ny*num_duplicate_shots*num_run_repeats), dtype=[('Line_number', '>u4'), ('x', '>f4'), ('y', '>f4')])

	#create rectangular shape position array with height z
	index = 0

	for repeat_cnt in range(num_run_repeats):

		for y in ypos:
			for x in xpos:
				for dup_cnt in range(num_duplicate_shots):
					positions[index] = (index+1, x, y)
					index += 1
					
	return positions, xpos, ypos

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

# standalone: run the program
if __name__ == '__main__':
    t_start = time.time()
    #position, xpos, ypos = get_positions()
    disk_folder = r"D:\Data\Energetic_Electron_Ring\raw_data"
    if DISK_FIRST:
        shot_filename = Acquire_Scope_Data_to_Disk(disk_folder, get_positions, get_channel_description, ip_addrs, exp_name=exp_name, threading= False)
	
    print('\ndone, %.4f seconds'%((time.time()-t_start)))
    
	# import os
	# import time
	# t_start = time.time()
	# position, xpos, ypos = get_positions()    
	# raw_folder = r"D:\Data\Energetic_Electron_Ring\raw_data"
    # if DISK_FIRST:
    #     shot_filename = Acquire_Scope_Data_to_disk(raw_folder, get_positions, get_channel_description, ip_addrs)
	# print('\ndone, %.4f seconds'%((time.time()-t_start)))