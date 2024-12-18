# -*- coding: utf-8 -*-
"""
 The main data acquisition routine is in this file

 function Acquire_Scope_Data_3D()  - see below
 	Uses user-provided callback functions (given as args) to:
 		specify the output HDF5 filename and whether duplicates should be overwritten
 		specify the positions array, and
 		specify the individual channel descriptions (c = 'C1', 'C2', 'C3', 'C4')
 	the fourth arg is the scope_ip_address
 this function
 	Creates the HDF5 file,
 	Creates the various groups and datasets,
 	Adds metadata
 	Iterates through the positions array (see "MAIN ACQUISITION LOOP"):
 	    calls Motor_Control_3D -> move_to_position(x,y,z)
 	    Waits for the scope to average the data, as per scope settings
 	    Writes the acquired scope data to the HDF5 output file
 	Closes the HDF5 file when done

"""

import numpy
import h5py as h5py
import time
import os.path
import sys
import threading
from LeCroy_Scope import LeCroy_Scope, WAVEDESC_SIZE, Fake_Scope
from Motor_Control_2D import Motor_Control_2D
import pickle

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

def acquire_displayed_traces(scope, datasets, hdr_data, pos_ndx):
	""" worker for below :
		 acquire enough sweeps for the averaging, then read displayed scope trace data into HDF5 datasets
	"""
	# timeout = 2000 # seconds

	# timed_out, N = scope.wait_for_max_sweeps(str(pos_ndx)+': ', timeout)  # wait for averaging to complete; leaves scope not triggering

	# if timed_out:
		# print('**** averaging timed out: got '+str(N)+' at %.6g s' % timeout)
	scope.set_trigger_mode('STOP')
	traces = scope.displayed_traces()
    
	for tr in traces:
		NPos, NTimes = datasets[tr].shape
		
		datasets[tr][pos_ndx, 0:NTimes] = scope.acquire(tr)[0:NTimes]    # sometimes for 10000 the scope hardware returns 10001 samples, so we have to specify [0:NTimes]
		
	for tr in traces:
		hdr_data[tr][pos_ndx] = numpy.void(scope.header_bytes())    # valid after scope.acquire() call
	
	scope.set_trigger_mode('NORM')   # resume triggering

#----------------------------------------------------------------------------------------

def acquire_displayed_traces_to_disk(scope, disk_folder, pos_ndx, exp_name, magnetron = False):
	"""

	"""
	while True:
		try:
			current_mode = scope.set_trigger_mode("")
			
			if current_mode[0:4] == 'STOP':
				break
			time.sleep(0.05)
		except KeyboardInterrupt:
			print('Keyboard interuppted')
			break

	traces = scope.displayed_traces()
	for tr in traces:
		st = time.time()
		data = scope.acquire_raw(tr)
		if not magnetron:
			file_name = disk_folder + os.sep + exp_name + str(pos_ndx) + '_' + tr + '.bin'
			with open(file_name, 'wb') as f:
				f.write(data)
		else:
			file_name = disk_folder + os.sep + exp_name + str(pos_ndx) + '_magnetron_' + tr + '.bin'
			with open(file_name, 'wb') as f:
				f.write(data)
	scope.set_trigger_mode('SINGLE')


def acquire_displayed_traces_to_disk_twoscope(scope, scope2, disk_folder, pos_ndx, exp_name):
	"""

	"""
	while True:
		try:
			current_mode = scope.set_trigger_mode("")
			if current_mode[0:4] == 'STOP':
				break
			time.sleep(0.05)
		except KeyboardInterrupt:
			print('Keyboard interuppted')
			break
			
	traces2 = scope2.displayed_traces()
	for tr in traces2:
		st = time.time()
		data = scope2.acquire_raw(tr)
		file_name = disk_folder + os.sep + exp_name + '_' + str(pos_ndx) + '_magnetron_' + tr + '.bin'
		with open(file_name, 'wb') as f:
			f.write(data)

	traces = scope.displayed_traces()
	for tr in traces:
		st = time.time()
		data = scope.acquire_raw(tr)
		file_name = disk_folder + os.sep + exp_name + '_' + str(pos_ndx) + '_' + tr + '.bin'
		with open(file_name, 'wb') as f:
			f.write(data)
	scope.set_trigger_mode('SINGLE')
	scope2.set_trigger_mode('SINGLE')

#----------------------------------------------------------------------------------------
def acquire_displayed_traces_from_disk(fake_scope, datasets, hdr_data, disk_folder, pos_ndx, exp_name):
	#scope.set_trigger_mode('STOP')
	traces = fake_scope.displayed_traces
	# print('test_DL!!!!!!!!!!!', traces)
	for tr in traces:
		NPos, NTimes = datasets[tr].shape
		st = time.time()
		#data = scope.acquire_raw(tr)
		file_name = disk_folder + os.sep + exp_name + str(pos_ndx) + '_' + tr + '.bin'
		with open(file_name, 'rb') as f:
			data= fake_scope.acquire_from_disk(tr, pos_ndx, exp_name, disk_folder)

		# print(fake_scope.time_array)
		datasets[tr][pos_ndx, 0:NTimes]  = data[0:NTimes] 
		hdr_data[tr][pos_ndx] = numpy.void(fake_scope.header_bytes())
	# elif threading:
	# 	def acquire_sub(scope,disk_folder, pos_ndx, exp_name, tr):
			
		

			#print('ma_test',time.time()-st)
	#scope.set_trigger_mode('NORM')
#----------------------------------------------------------------------------------------

def create_sourcefile_dataset(grp, fn):
	""" worker for below:
		create an HDF5 dataset containing the contents of the specified file
		add attributes file name and modified time
	"""
	fds_name = os.path.basename(fn)
	fds = grp.create_dataset(fds_name, data=open(fn, 'r').read())
	fds.attrs['filename'] = fn
	fds.attrs['modified'] = time.ctime(os.path.getmtime(fn))

#----------------------------------------------------------------------------------------

def clean(agilent, cleaning):
	agilent.function = 'DC' # Goes into DC mode
	agilent.DCoffset = cleaning  # Heat up probe to red hot
	time.sleep(10) # clean for few sec
	agilent.DCoffset = 0.275
	time.sleep(1)
	agilent.function = 'RAMP' # Go back to sweep
	agilent.burst(True, 1, 0)

#----------------------------------------------------------------------------------------
def Acquire_Scope_Data(ifn, get_positions, get_channel_description, ip_addresses):
	# The main data acquisition routine
	#
	# 	Arguments are user-provided callback functions that return the following:
	# 		get_hdf5_filename()          the output HDF5 filename,
	# 		get_positions()              the positions array,
	# 		get_channel_description(c)   the individual channel descriptions (c = 'C1', 'C2', 'C3', 'C4'),
	# 		get_ip_addresses()           a dict of the form {'scope':'10.0.1.122', 'axial':'10.0.0.123', 'trans':'10.0.0.124'}
	# 		                                  if a key is not specified, no motion will be attempted on that axis
	#
	# 	Creates the HDF5 file, creates the various groups and datasets, adds metadata (see "HDF5 OUTPUT FILE SETUP")
	#
	# 	Iterates through the positions array (see "MAIN ACQUISITION LOOP"):
	# 	    calls motor_control.set_position(pos)
	# 	    Waits for the scope to average the data, as per scope settings
	# 	    Writes the acquired scope data to the HDF5 output file
	#
	# 	Closes the HDF5 file when done
	#
	#============================
	# list of files to include in the HDF5 data file
	src_files = [sys.argv[0],
				__file__,           # ASSUME this file is in the same directory as the next two:
				os.path.dirname(__file__)+os.sep+'LeCroy_Scope.py',
				os.path.dirname(__file__)+os.sep+'Motor_Control_2D.py'
			   ]
	#for testing, list these:
	print('Files to record in the hdf5 archive:')
	print('    invoking file      =', src_files[0])
	print('    this file          =', src_files[1])
	print('    LeCroy_Scope file  =', src_files[2])
	print('    motor control file =', src_files[3])

	#============================
	# position array given by Data_Run_3D.py: (ignore_data from Data_Run_3D defines points cannot reach by probe drive)

	positions, xpos, ypos = get_positions()

	# Create empty position arrays
	if xpos is None:
		xpos = numpy.array([])
	if ypos is None:
		ypos = numpy.array([])

	#============================
	######### HDF5 OUTPUT FILE SETUP #########

	# Open hdf5 file for writing (user callback for filename):

	f = h5py.File(ifn,  'w')  # 'w' - overwrite (we should have determined whether we want to overwrite in get_hdf5_filename())
	# f = h5py.File(ofn,  'x')  # 'x' - no overwrite

	#============================
	# create HDF5 groups similar to those in the legacy format:

	acq_grp    = f.create_group('/Acquisition')              # /Acquisition
	acq_grp.attrs['run_time'] = time.ctime()                                       # not legacy
	scope_grp  = acq_grp.create_group('LeCroy_scope')        # /Acquisition/LeCroy_scope
	header_grp = scope_grp.create_group('Headers')                                 # not legacy
#	trace_grp = scope_grp.create_group('Datasets')

	ctl_grp    = f.create_group('/Control')                  # /Control
	pos_grp    = ctl_grp.create_group('Positions')           # /Control/Positions

	meta_grp   = f.create_group('/Meta')                     # /Meta                not legacy
	script_grp = meta_grp.create_group('Python')             # /Meta/Python
	scriptfiles_grp = script_grp.create_group('Files')       # /Meta/Python/Files

	# in the /Meta/Python/Files group:
	for src_file in src_files:
		create_sourcefile_dataset(scriptfiles_grp, src_file)                       # not legacy

	# I don't know how to get this information from the scope:
	scope_grp.create_dataset('LeCroy_scope_Setup_Arrray', data=numpy.array('Sorry, this is not included', dtype='S'))

	pos_ds = pos_grp.create_dataset('positions_setup_array', data=positions)
	pos_ds.attrs['xpos'] = xpos                                                    # not legacy
	pos_ds.attrs['ypos'] = ypos                                                    # not legacy

	pos_ds = pos_grp.create_dataset('positions_array', shape=len(positions), dtype=[('Line_number', '>u4'), ('x', '>f4'), ('y', '>f4')])

	
	# Connect to motor
	mc = Motor_Control_2D(x_ip_addr = ip_addresses['x'], y_ip_addr = ip_addresses['y'])

	# create the scope access object, and iterate over positions
	with LeCroy_Scope(ip_addresses['scope'], verbose=True)   as scope:
		if not scope:
			print('Scope not found at '+ip_addresses['scope'])      # I think we have raised an exception if this is the case, so we never get here
			return

		scope_grp.attrs['ScopeType'] = scope.idn_string

		NPos = len(positions)
		NTimes = scope.max_samples()

		datasets = {}
		hdr_data = {}

		# create 4 default data sets, empty.  These will all be populated for compatibility with legacy format hdf5 files.
		datasets['C1'] = scope_grp.create_dataset('Channel1', shape=(NPos,NTimes), fletcher32=True, compression='gzip', compression_opts=9)
		datasets['C2'] = scope_grp.create_dataset('Channel2', shape=(NPos,NTimes), fletcher32=True, compression='gzip', compression_opts=9)
		datasets['C3'] = scope_grp.create_dataset('Channel3', shape=(NPos,NTimes), fletcher32=True, compression='gzip', compression_opts=9)
		datasets['C4'] = scope_grp.create_dataset('Channel4', shape=(NPos,NTimes), fletcher32=True, compression='gzip', compression_opts=9)

		# create other datasets, one for each displayed trace (but not C1-4, which we just did)
		traces = scope.displayed_traces()
		for tr in traces:
			name = scope.expanded_name(tr)
			if tr not in ('C1','C2','C3','C4'):
				ds = scope_grp.create_dataset(name, (NPos,NTimes), chunks=(1,NTimes), fletcher32=True, compression='gzip', compression_opts=9)
				datasets[tr] = ds

		'''
		For each trace we are storing, we will write one header per position
		(immediately after the data for that position has been acquired); these compress to an insignificant size
		For whatever stupid reason we need to write the header as a binary blob using an "HDF5 opaque" type - here void type 'V346'
		(otherwise I could not manage to avoid invisible string processing and interpretation)
		'''
		for tr in traces:
			name = scope.expanded_name(tr)
			hdr_data[tr] = header_grp.create_dataset(name, shape=(NPos,), dtype="V%i"%(WAVEDESC_SIZE), fletcher32=True, compression='gzip', compression_opts=9)  # V346 = void type, 346 bytes long

		# create "time" dataset
		time_ds = scope_grp.create_dataset('time', shape=(NTimes,), fletcher32=True, compression='gzip', compression_opts=9)
				
		try:

			######### BEGIN MAIN ACQUISITION LOOP #########
			print('starting acquisition loop at', time.ctime())

			for pos in positions:
        
				acquisition_loop_start_time = time.time()

				# move to next position
				print('position index =', pos[0], '  x =', pos[1], '  y =', pos[2], end='')
				try:
					mc.enable
					#mc.probe_positions = (pos[1], pos[2])
					mc.disable
				except KeyboardInterrupt:
					raise KeyboardInterrupt
				except:
					print('Motor fail to move to position index =', pos[0], '  x =', pos[1], '  y =', pos[2], end='\n')
					continue

				print('------------------', scope.gaaak_count, '-------------------- ',pos[0],sep='')

				# Acquire enough sweeps for the averaging, then read displayed scope trace data into HDF5 datasets
				acquire_displayed_traces(scope, datasets, hdr_data, pos[0]-1)   # argh the pos[0] index is 1-based

				if pos[0] == 1:
					time_ds[0:NTimes] = scope.time_array()[0:NTimes] # at least get one time array recorded for swmr functions

				time_per_pos = (time.time()-acquisition_loop_start_time) / 3600
				print ('Remaining time:%6.2f h'%((len(positions) - pos[0]) * time_per_pos))

				x,y = mc.probe_positions
				pos_ds[pos[0]-1] = (pos[0], x, y)

			######### END MAIN ACQUISITION LOOP #########

		except KeyboardInterrupt:
			print('\n______Halted due to Ctrl-C______', '  at', time.ctime())

		except:
			print('\n______Halted due to some error______', '  at', time.ctime())
			pass

		# copy the array of time values, corresponding to the last acquired trace, to the times_dataset
		time_ds[0:NTimes] = scope.time_array()[0:NTimes]      # specify number of points, sometimes scope return extras
		if type(time_ds) == 'stupid':
			print(' this is only included to make the linter happy, otherwise it thinks time_ds is not used')

#		 Set any unused datasets to 0 (e.g. any C1-4 that was not acquired); when compressed they require negligible space
#		 Also add the text descriptions.    Do these together to be able to be able to make a note in the description

		for tr in traces:
			if datasets[tr].len() == 0:
				datasets[tr] = numpy.zeros(shape=(NPos,NTimes))
				datasets[tr].attrs['description'] = 'NOT RECORDED: ' + get_channel_description(tr)           # callback arg to the current function
				datasets[tr].attrs['recorded']    = False
			else:
				datasets[tr].attrs['description'] = get_channel_description(tr)                              # callback arg to the current function
				datasets[tr].attrs['recorded']    = True

	f.close()  # close the HDF5 file
	
	return ifn

def Acquire_Scope_Data_raw(ifn, get_positions, get_channel_description, ip_addresses, disk_folder, exp_name):

	src_files = [sys.argv[0],
				__file__,           # ASSUME this file is in the same directory as the next two:
				os.path.dirname(__file__)+os.sep+'LeCroy_Scope.py',
				os.path.dirname(__file__)+os.sep+'Motor_Control_2D.py'
			   ]
	#for testing, list these:
	print('Files to record in the hdf5 archive:')
	print('    invoking file      =', src_files[0])
	print('    this file          =', src_files[1])
	print('    LeCroy_Scope file  =', src_files[2])
	print('    motor control file =', src_files[3])

	#============================
	# position array given by Data_Run_3D.py: (ignore_data from Data_Run_3D defines points cannot reach by probe drive)

	positions, xpos, ypos = get_positions()

	# Create empty position arrays
	if xpos is None:
		xpos = numpy.array([])
	if ypos is None:
		ypos = numpy.array([])

	pos_log = f"{disk_folder}\\real_positions.bin"
	if not os.path.exists(pos_log):
		with open(pos_log, 'wb') as f:
			statement = b'Log file recording probe actual positions created on ' + time.ctime().encode()
			f.write(statement)
			f.write(b'\n')
			f.write(b'position index, x, y')
			f.write(b'\n')
			print(statement.decode())
	#============================
	
	# Connect to motor
	mc = Motor_Control_2D(x_ip_addr = ip_addresses['x'], y_ip_addr = ip_addresses['y'])
	with LeCroy_Scope(ip_addresses['scope'], verbose=False)   as scope:
		with LeCroy_Scope(ip_addresses['magnetron_scope'], verbose= False) as scope2:
			if not scope:
				print('Scope not found at '+ip_addresses['scope'])      # I think we have raised an exception if this is the case, so we never get here
				return
			if not scope2:
				print('Scope not found at '+ip_addresses['magnetron_scope'])      # I think we have raised an exception if this is the case, so we never get here
				return
			# scope_grp.attrs['ScopeType'] = scope.idn_string

			NPos = len(positions)
			NTimes = scope.max_samples()

			datasets = {}
			hdr_data = {}

			# create other datasets, one for each displayed trace (but not C1-4, which we just did)
			traces = scope.displayed_traces()


			'''
			For each trace we are storing, we will write one header per position
			(immediately after the data for that position has been acquired); these compress to an insignificant size
			For whatever stupid reason we need to write the header as a binary blob using an "HDF5 opaque" type - here void type 'V346'
			(otherwise I could not manage to avoid invisible string processing and interpretation)
			'''
			# for tr in traces:
			# 	name = scope.expanded_name(tr)
			# 	hdr_data[tr] = header_grp.create_dataset(name, shape=(NPos,), dtype="V%i"%(WAVEDESC_SIZE), fletcher32=True, compression='gzip', compression_opts=9)  # V346 = void type, 346 bytes long

			# # create "time" dataset
			# time_ds = scope_grp.create_dataset('time', shape=(NTimes,), fletcher32=True, compression='gzip', compression_opts=9)
					
			try:

				######### BEGIN MAIN ACQUISITION LOOP #########
				print('starting acquisition loop at', time.ctime())

				for pos in positions:
			
					acquisition_loop_start_time = time.time()

					# move to next position
					print('position index =', pos[0], '  x =', pos[1], '  y =', pos[2], end='')
					try:
						mc.enable
						mc.probe_positions = (pos[1], pos[2])
						mc.disable
					except KeyboardInterrupt:
						raise KeyboardInterrupt
					except:
						print('Motor fail to move to position index =', pos[0], '  x =', pos[1], '  y =', pos[2], end='\n')
						continue

					print('------------------', scope.gaaak_count, '-------------------- ',pos[0],sep='')

					# Acquire enough sweeps for the averaging, then read displayed scope trace data into HDF5 datasets
					# acquire_displayed_traces_to_disk(scope, datasets, hdr_data, pos[0]-1)   # argh the pos[0] index is 1-based
					# print('Test begin here')
					# print('motor time', time.time() - acquisition_loop_start_time)
					# fast scope
					scope2.set_trigger_mode('SINGLE')
					# start_time = time.time()
					# result2 = acquire_displayed_traces_to_disk(scope2, disk_folder, pos[0]-1, exp_name=exp_name,magnetron=True,last_time = start_time)
					# # slow scope at behind
					scope.set_trigger_mode('SINGLE')
					acquire_displayed_traces_to_disk_twoscope(scope,scope2, disk_folder, pos[0]-1, exp_name=exp_name)
					# if pos[0] == 1:
					# 	time_ds[0:NTimes] = scope.time_array()[0:NTimes] # at least get one time array recorded for swmr functions

					# time_per_pos = (time.time()-acquisition_loop_start_time) / 3600
					#print ('Remaining time:%6.2f h'%((len(positions) - pos[0]) * time_per_pos))
					x, y = mc.probe_positions
					with open(pos_log, 'ab') as f:
						f.write(pickle.dumps((pos[0], x, y)))

					print('\ndone, %.2f seconds'%((time.time()-acquisition_loop_start_time)))
					
					#real_pos_data = numpy.array([pos[0],x,y])
					#numpy.savetxt(disk_folder + os.sep + 'pos' + pos + '.txt',real_pos_data, fmt = '%.4f', delimiter = ',', header = 'pos, x, y')
					# save the info of the scope
					
					#pos_ds[pos[0]-1] = (pos[0], x, y)

				######### END MAIN ACQUISITION LOOP #########

			except KeyboardInterrupt:
				print('\n______Halted due to Ctrl-C______', '  at', time.ctime())

			except:
				print('\n______Halted due to some error______', '  at', time.ctime())
				pass



			#  save the fake scope

			fake_scope = Fake_Scope(idn_string=scope.idn_string,
									max_samples=scope.max_samples(),
									traces = scope.displayed_traces(),
									displayed_traces= scope.displayed_traces(),
									gaaak_count=scope.gaaak_count)
			# print('Fake scope info:')
			# print(fake_scope.idn_string,'\n')
			# print(fake_scope.max_samples,'\n')
			# print(fake_scope.traces,'\n')
			# print(fake_scope.gaaak_count,'\n')
			with open(disk_folder + os.sep + 'fake_scope.pkl', 'wb') as file:
				pickle.dump(fake_scope, file)

			fake_scope2 = Fake_Scope(idn_string=scope2.idn_string,
									max_samples=scope2.max_samples(),
									traces = scope2.displayed_traces(),
									displayed_traces= scope2.displayed_traces(),
									gaaak_count=scope2.gaaak_count)
			# print('Fake scope2 info:')
			# print(fake_scope2.idn_string,'\n')
			# print(fake_scope2.max_samples,'\n')
			# print(fake_scope2.traces,'\n')
			# print(fake_scope2.gaaak_count,'\n')
			with open(disk_folder + os.sep + 'fake_scope_magnetron.pkl', 'wb') as file:
				pickle.dump(fake_scope2, file)
		
			
		# copy the array of time values, corresponding to the last acquired trace, to the times_dataset
		# time_ds[0:NTimes] = scope.time_array()[0:NTimes]      # specify number of points, sometimes scope return extras
		# if type(time_ds) == 'stupid':
		# 	print(' this is only included to make the linter happy, otherwise it thinks time_ds is not used')

#		 Set any unused datasets to 0 (e.g. any C1-4 that was not acquired); when compressed they require negligible space
#		 Also add the text descriptions.    Do these together to be able to be able to make a note in the description

		# for tr in traces:
		# 	if datasets[tr].len() == 0:
		# 		datasets[tr] = numpy.zeros(shape=(NPos,NTimes))
		# 		datasets[tr].attrs['description'] = 'NOT RECORDED: ' + get_channel_description(tr)           # callback arg to the current function
		# 		datasets[tr].attrs['recorded']    = False
		# 	else:
		# 		datasets[tr].attrs['description'] = get_channel_description(tr)                              # callback arg to the current function
		# 		datasets[tr].attrs['recorded']    = True

	#f.close()  # close the HDF5 file
	
	return 1



def Acquire_hdf5_from_disk(ifn, disk_folder, get_positions, ip_addresses, get_channel_description, exp_name):
	"""
	still need to connect to the scope for simpilicity, #todo remove the required connection to the scope 
	"""

	src_files = [sys.argv[0],
				__file__,           # ASSUME this file is in the same directory as the next two:
				os.path.dirname(__file__)+os.sep+'LeCroy_Scope.py',
				os.path.dirname(__file__)+os.sep+'Motor_Control_2D.py'
			   ]
	#for testing, list these:
	print('Files to record in the hdf5 archive:')
	print('    invoking file      =', src_files[0])
	print('    this file          =', src_files[1])
	print('    LeCroy_Scope file  =', src_files[2])
	print('    motor control file =', src_files[3])

	#============================
	# position array given by Data_Run_3D.py: (ignore_data from Data_Run_3D defines points cannot reach by probe drive)

	positions, xpos, ypos = get_positions()
	if xpos is None:
		xpos = numpy.array([])
	if ypos is None:
		ypos = numpy.array([])


	f = h5py.File(ifn,  'w')  # 'w' - overwrite (we should have determined whether we want to overwrite in get_hdf5_filename())
	# f = h5py.File(ofn,  'x')  # 'x' - no overwrite

	#============================
	# create HDF5 groups similar to those in the legacy format:

	acq_grp    = f.create_group('/Acquisition')              # /Acquisition
	acq_grp.attrs['run_time'] = time.ctime()                                       # not legacy
	scope_grp  = acq_grp.create_group('LeCroy_scope')        # /Acquisition/LeCroy_scope
	header_grp = scope_grp.create_group('Headers')                                 # not legacy
#	trace_grp = scope_grp.create_group('Datasets')

	ctl_grp    = f.create_group('/Control')                  # /Control
	pos_grp    = ctl_grp.create_group('Positions')           # /Control/Positions

	meta_grp   = f.create_group('/Meta')                     # /Meta                not legacy
	script_grp = meta_grp.create_group('Python')             # /Meta/Python
	scriptfiles_grp = script_grp.create_group('Files')       # /Meta/Python/Files

	# in the /Meta/Python/Files group:
	for src_file in src_files:
		create_sourcefile_dataset(scriptfiles_grp, src_file)                       # not legacy

	# I don't know how to get this information from the scope:
	scope_grp.create_dataset('LeCroy_scope_Setup_Arrray', data=numpy.array('Sorry, this is not included', dtype='S'))

	pos_ds = pos_grp.create_dataset('positions_setup_array', data=positions)
	pos_ds.attrs['xpos'] = xpos                                                    # not legacy
	pos_ds.attrs['ypos'] = ypos                                                    # not legacy

	pos_ds = pos_grp.create_dataset('positions_array', shape=len(positions), dtype=[('Line_number', '>u4'), ('x', '>f4'), ('y', '>f4')])



	# with LeCroy_Scope(ip_addresses['scope'], verbose=True)   as scope:
	# 	if not scope:
	# 		print('Scope not found at '+ip_addresses['scope'])      # I think we have raised an exception if this is the case, so we never get here
	# 		return

	with open(disk_folder + os.sep + 'fake_scope.pkl','rb') as file:
		fake_scope_data = pickle.load(file)
	fake_scope = Fake_Scope(idn_string = fake_scope_data.idn_string,
						    max_samples=fake_scope_data.max_samples,
							traces=fake_scope_data.traces,
							gaaak_count=fake_scope_data.gaaak_count,
							displayed_traces=fake_scope_data.displayed_traces)
	# fake_scope.max_samples = fake_scope_data.max_samples
	# fake_scope.traces = fake_scope_data.traces
	# fake_scope.idn_string = fake_scope_data.idn_string
	# fake_scope.gaaak_count = fake_scope_data.gaaak_count
	
	scope_grp.attrs['ScopeType'] = fake_scope.idn_string

	NPos = len(positions)
	NTimes = fake_scope.max_samples

	datasets = {}
	hdr_data = {}

	# create 4 default data sets, empty.  These will all be populated for compatibility with legacy format hdf5 files.
	datasets['C1'] = scope_grp.create_dataset('Channel1', shape=(NPos,NTimes), fletcher32=True, compression='gzip', compression_opts=9)
	datasets['C2'] = scope_grp.create_dataset('Channel2', shape=(NPos,NTimes), fletcher32=True, compression='gzip', compression_opts=9)
	datasets['C3'] = scope_grp.create_dataset('Channel3', shape=(NPos,NTimes), fletcher32=True, compression='gzip', compression_opts=9)
	datasets['C4'] = scope_grp.create_dataset('Channel4', shape=(NPos,NTimes), fletcher32=True, compression='gzip', compression_opts=9)

	# create other datasets, one for each displayed trace (but not C1-4, which we just did)
	traces = fake_scope.traces
	for tr in traces:
		name = fake_scope.expanded_name(tr)
		if tr not in ('C1','C2','C3','C4'):
			ds = scope_grp.create_dataset(name, (NPos,NTimes), chunks=(1,NTimes), fletcher32=True, compression='gzip', compression_opts=9)
			datasets[tr] = ds

	'''
	For each trace we are storing, we will write one header per position
	(immediately after the data for that position has been acquired); these compress to an insignificant size
	For whatever stupid reason we need to write the header as a binary blob using an "HDF5 opaque" type - here void type 'V346'
	(otherwise I could not manage to avoid invisible string processing and interpretation)
	'''
	for tr in traces:
		name = fake_scope.expanded_name(tr)
		hdr_data[tr] = header_grp.create_dataset(name, shape=(NPos,), dtype="V%i"%(WAVEDESC_SIZE), fletcher32=True, compression='gzip', compression_opts=9)  # V346 = void type, 346 bytes long

	# create "time" dataset
	time_ds = scope_grp.create_dataset('time', shape=(NTimes,), fletcher32=True, compression='gzip', compression_opts=9)
			
	

		######### BEGIN MAIN ACQUISITION LOOP #########
	print('starting acquisition loop at', time.ctime())

	for pos in positions:

		acquisition_loop_start_time = time.time()

		# move to next position
		print('position index =', pos[0], '  x =', pos[1], '  y =', pos[2], end='')
		# try:
		# 	mc.enable
		# 	#mc.probe_positions = (pos[1], pos[2])
		# 	mc.disable
		# except KeyboardInterrupt:
		# 	raise KeyboardInterrupt
		# except:
		# 	print('Motor fail to move to position index =', pos[0], '  x =', pos[1], '  y =', pos[2], end='\n')
		# 	continue

		print('------------------', fake_scope.gaaak_count, '-------------------- ',pos[0],sep='')

		# Acquire enough sweeps for the averaging, then read displayed scope trace data into HDF5 datasets
		acquire_displayed_traces_from_disk(fake_scope, datasets, hdr_data, disk_folder, pos[0]-1, exp_name)   # argh the pos[0] index is 1-based

		if pos[0] == 1:
			time_ds[0:NTimes] = fake_scope.time_array[0:NTimes] # at least get one time array recorded for swmr functions

		time_per_pos = (time.time()-acquisition_loop_start_time) / 3600
		print ('Remaining time:%6.2f h'%((len(positions) - pos[0]) * time_per_pos))

		#x,y = mc.probe_positions

		# we don't have probe here so
		pos_ds[pos[0]-1] = (pos[0], pos[1], pos[2])

		######### END MAIN ACQUISITION LOOP #########

	

	# copy the array of time values, corresponding to the last acquired trace, to the times_dataset
	time_ds[0:NTimes] = fake_scope.time_array[0:NTimes]      # specify number of points, sometimes scope return extras
	if type(time_ds) == 'stupid':
		print(' this is only included to make the linter happy, otherwise it thinks time_ds is not used')

#		 Set any unused datasets to 0 (e.g. any C1-4 that was not acquired); when compressed they require negligible space
#		 Also add the text descriptions.    Do these together to be able to be able to make a note in the description

	for tr in traces:
		if datasets[tr].len() == 0:
			datasets[tr] = numpy.zeros(shape=(NPos,NTimes))
			datasets[tr].attrs['description'] = 'NOT RECORDED: ' + get_channel_description(tr)           # callback arg to the current function
			datasets[tr].attrs['recorded']    = False
		else:
			datasets[tr].attrs['description'] = get_channel_description(tr)                              # callback arg to the current function
			datasets[tr].attrs['recorded']    = True

	f.close()  # close the HDF5 file
	
	return ifn