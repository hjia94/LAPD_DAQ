# -*- coding: utf-8 -*-
"""
McPherson Spectrometer DAQ System - Scope Data Acquisition Module

This module handles the coordinated data acquisition from LeCroy oscilloscopes 
and McPherson spectrometer scanning for spectroscopic measurements.

Features:
- LeCroy oscilloscope data acquisition
- HDF5 data file creation and management
- Coordinated spectrometer wavelength scanning
- Multi-trace data recording with headers
- Error handling and timeout management
- Automated file naming and metadata storage

Author: LAPD Team
Created: 2022 (Original version)
Last Modified: June, 2025 (restructured code)
Dependencies: numpy, h5py, tkinter, LeCroy_Scope, spectrometer_controller

Data Format:
- HDF5 files with compressed datasets
- Separate groups for acquisition data, control parameters, and metadata
- Time arrays and channel descriptions included
"""

import time
import sys
import os.path
import tkinter
import tkinter.filedialog
import numpy
import h5py

# Make repo root importable so `drivers/` resolves when this module is launched
# directly via `python McPherson/McPherson_DAQ_Scan_Control.py`.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from lab_scopes.lecroy import LeCroy_Scope, WAVEDESC_SIZE, EXPANDED_TRACE_NAMES
from spectrometer_controller import spectrometer


class scope_data:
	def __init__(self, ip_addr, save_raw=False):

		self.save_raw = save_raw
		self.ip_addr = ip_addr
		print('Scope class at', self.ip_addr)


	def acquire_displayed_traces(self, scope, datasets, hdr_data, pos_ndx):
		""" 
			 acquire enough sweeps for the averaging, then read displayed scope trace data into HDF5 datasets
		"""
		timeout = 2000 # seconds

		# Get number of sweeps for progress monitoring
		NSweeps, ach = scope.max_averaging_count()
		print(f'    Waiting for {NSweeps} sweeps to complete...')
		
		timed_out, N = scope.wait_for_max_sweeps(str(pos_ndx)+': ', timeout)
		if timed_out:
			print(f'    WARNING: Averaging timed out - got {N}/{NSweeps} sweeps after {timeout} seconds')
		else:
			print(f'    Averaging complete: {N}/{NSweeps} sweeps acquired')

		traces = scope.displayed_traces()

		# Acquire data and headers for each trace
		print(f'    Reading data from {len(traces)} traces...')
		for tr in traces:
			NPos, NTimes = datasets[tr].shape
			
			data, header_bytes = scope.acquire(tr, raw=self.save_raw)
			datasets[tr][pos_ndx, 0:NTimes] = data[0:NTimes]     # sometimes for 10000 the scope hardware returns 10001 samples, so we have to specify [0:NTimes]
			hdr_data[tr][pos_ndx] = numpy.void(header_bytes)     # store header bytes
			print(f'      Trace {tr}: {len(data)} samples stored')



	#----------------------------------------------------------------------------------------

	def create_sourcefile_dataset(self, grp, fn):
		""" worker for below:
			create an HDF5 dataset containing the contents of the specified file
			add attributes file name and modified time
		"""
		fds_name = os.path.basename(fn)
		fds = grp.create_dataset(fds_name, data=open(fn, 'r').read())
		fds.attrs['filename'] = fn
		fds.attrs['modified'] = time.ctime(os.path.getmtime(fn))

	#----------------------------------------------------------------------------------------

	def get_hdf5_filename(self, hdf5_filename = None) -> str:
		""" actual callback function to return the output file name """

		avoid_overwrite = True       # <-- setting this to False will allow overwriting an existing file without a prompt

		#user: modify this if desired

		fn = hdf5_filename         # variable assigned at the top of this file

		if fn == None  or len(fn) == 0    or    (avoid_overwrite  and  os.path.isfile(fn)):
			# if we are not allowing possible overwrites as default, and the file already exists, use file open dialog
			tk = tkinter.Tk()
			tk.withdraw()
			fn = tkinter.filedialog.asksaveasfilename(title='Enter name of HDF5 file to write')
			if len(fn) == 0:
				raise SystemExit(0)        # user pressed 'cancel'
			tk.destroy()

		hdf5_filename = fn      # save it for later
		return fn

	def get_channel_description(self, tr) -> str:
		""" callback function to return a string containing a description of the data in each recorded channel """

		#user: assign channel description text here to override the default:
		if tr == 'C1':
			return 'N/A'
		if tr == 'C2':
			return 'N/A'
		if tr == 'C3':
			return 'N/A'
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
			return 'N/A'

		# otherwise, program-generated default description strings follow
		if tr in EXPANDED_TRACE_NAMES.keys():
			return 'no entered description for ' + EXPANDED_TRACE_NAMES[tr]

		return '**** get_channel_description(): unknown trace indicator "'+tr+'". How did we get here?'

	#----------------------------------------------------------------------------------------
	#----------------------------------------------------------------------------------------
	def Acquire_Scope_Data(self, start_wavl, num_wavl, increment, num_shots = 1, speed = None, spec = None):
		'''
		The main data acquisition routine for the McPherson spectrometer
		
		Parameters:
		  start_wavl (float): Starting wavelength in nm
		  num_wavl (int): Number of wavelength points to scan
		  increment (float): Wavelength increment between points in nm
		  num_shots (int): Number of shots to take at each wavelength (default=1)
		  speed (float): Optional scan speed in steps/s
		  spec (spectrometer): Optional spectrometer controller object
		
		This function:
		1. Creates HDF5 file and sets up groups/datasets for data storage
		2. Generates wavelength scan array based on input parameters
		3. For each wavelength point:
		   - Moves spectrometer to position
		   - Acquires scope data (averaging based on scope settings)
		   - Writes data and metadata to HDF5 file
		4. Closes HDF5 file when complete
		
		============================
		list of files to include in the HDF5 data file
		'''
		from lab_scopes.lecroy import scope as _lecroy_scope_mod
		src_files = [sys.argv[0],
					__file__,
					_lecroy_scope_mod.__file__]   # LeCroy driver now lives in the lab_scopes package
		print('Files to record in the hdf5 archive:')
		print('       invoking file      =', src_files[0])
		print('       this file          =', src_files[1])
		print('       LeCroy_Scope file  =', src_files[2])
		

		#============================
		# Create wavelength array:

		num_wavl = int(num_wavl)
		wavelengths = numpy.linspace(start_wavl, start_wavl+increment*(num_wavl-1), num_wavl)

		# Create 'positions' array that corresponds to each shot of the data at each wavelength:
		positions = numpy.zeros((len(wavelengths))*num_shots, dtype=[('Line_number', '>u4'),('Wavelength', '>f4')])

		i = 0
		for w in wavelengths:
			for m in range(num_shots):
				positions[i] = (i, w)
				i += 1

		#============================
		######### HDF5 OUTPUT FILE SETUP #########

		# Open hdf5 file for writing (user callback for filename):

		ofn = self.get_hdf5_filename()        # callback arg to the current function

		f = h5py.File(ofn,    'w')  # 'w' - overwrite (we should have determined whether we want to overwrite in get_hdf5_filename())
		# f = h5py.File(ofn,  'x')    # 'x' - no overwrite

		#============================
		# create HDF5 groups similar to those in the legacy format:

		acq_grp       = f.create_group('/Acquisition')                 # /Acquisition
		acq_grp.attrs['run_time'] = time.ctime()                                       # not legacy
		scope_grp  = acq_grp.create_group('LeCroy_scope')         # /Acquisition/LeCroy_scope
		header_grp = scope_grp.create_group('Headers')                                   # not legacy
		# trace_grp = scope_grp.create_group('Datasets')

		ctl_grp       = f.create_group('/Control')                     # /Control
		pos_grp       = ctl_grp.create_group('Positions')             # /Control/Positions

		meta_grp   = f.create_group('/Meta')                     # /Meta                not legacy
		script_grp = meta_grp.create_group('Python')             # /Meta/Python
		scriptfiles_grp = script_grp.create_group('Files')         # /Meta/Python/Files

		# Source files can be added to metadata if needed
		# for src_file in src_files:
		#     self.create_sourcefile_dataset(scriptfiles_grp, src_file)

		# I don't know how to get this information from the scope:
		scope_grp.create_dataset('LeCroy_scope_Setup_Array', data=numpy.array('Sorry, this is not included', dtype='S'))

		pos_ds = pos_grp.create_dataset('positions_setup_array', data=positions)
		pos_ds.attrs['wavelengths'] = wavelengths                                                  # not legacy


		# Connect to scan motor if no argument is passed
		if spec is not None:
			self.sp = spec
		else:
			self.sp = spectrometer()

		# Set speed if speed is not none
		if speed is not None:
			self.sp.set_speed(speed)

		# create the scope access object, and iterate over positions
		print(f'Connecting to scope at {self.ip_addr}...')
		with LeCroy_Scope(self.ip_addr, verbose=False) as scope:
			if not scope:
				print('Scope not found at '+self.ip_addr)       # I think we have raised an exception if this is the case, so we never get here
				return

			print(f'Successfully connected to scope: {scope.idn_string}')
			scope_grp.attrs['ScopeType'] = scope.idn_string

			NPos = len(positions)
			NTimes = scope.max_samples()
			print(f'DAQ Configuration: {NPos} positions, {NTimes} samples per trace')

			datasets = {}
			hdr_data = {}

			# Create datasets for each displayed trace
			traces = scope.displayed_traces()
			print(f'Found {len(traces)} displayed traces: {traces}')
			for tr in traces:
				name = scope.expanded_name(tr)
				print(f'  Creating dataset for trace {tr} ({name})')
				ds = scope_grp.create_dataset(name, (NPos,NTimes), chunks=(1,NTimes), fletcher32=True, compression='gzip', compression_opts=9)
				datasets[tr] = ds

			# For each trace we are storing, we will write one header per position (immediately after
		   #    the data for that position has been acquired); these compress to an insignificant size
			# For whatever stupid reason we need to write the header as a binary blob using an "HDF5 opaque" type - here void type 'V346'  (otherwise I could not manage to avoid invisible string processing and interpretation)
			for tr in traces:
				name = scope.expanded_name(tr)
				hdr_data[tr] = header_grp.create_dataset(name, shape=(NPos,), dtype="V%i"%(WAVEDESC_SIZE), fletcher32=True, compression='gzip', compression_opts=9)     # V346 = void type, 346 bytes long

			# create "time" dataset
			time_ds = scope_grp.create_dataset('time', shape=(NTimes,), fletcher32=True, compression='gzip', compression_opts=9)



			try:  # try-catch for Ctrl-C keyboard interrupt

				######### BEGIN MAIN ACQUISITION LOOP #########
				print(f'Starting acquisition at {time.ctime()}')
				print(f'Total positions to acquire: {len(positions)}')
				print('=' * 60)

				last_wavl = positions[0][1]

				for pos in positions:

						
					index = pos[0]
					wavl = pos[1]

					print(f'Position {index+1}/{len(positions)}: Wavelength = {wavl:.3f} nm')

					# move to next wavelength after the acquisition at previous wavelength is finished 
					try:
						if wavl == last_wavl:
								print(' McPherson : No movement needed (same wavelength)')
								time.sleep(0.5)
						else:
								print(f' McPherson : Moving by {increment:.3f} nm...')
								self.sp.scan_up(increment)
								time.sleep(0.2)
								self.sp.wait_for_motion_complete()
								print(' McPherson : Movement complete')
						
					except KeyboardInterrupt:
						raise KeyboardInterrupt
					except:
						print(f'McPherson : FAILED to move to wavelength {wavl}')
						continue

					# do averaging, and copy scope data for each trace on the screen to the output HDF5 file
					print('  Scope: Starting data acquisition...')
					self.acquire_displayed_traces(scope, datasets, hdr_data, index)
					print('  Scope: Data acquisition complete')

					last_wavl = wavl
					if index == 0:
						# Get time array from the first displayed trace
						first_trace = traces[0]
						time_ds[0:NTimes] = scope.time_array(first_trace)[0:NTimes] # at least get one time array recorded for swmr functions
					
					# Show completion status
					progress_percent = ((index + 1) / len(positions)) * 100
					print(f'  McPherson Position complete ({progress_percent:.1f}% total progress)')
					print()  # Add blank line for readability

				######### END MAIN ACQUISITION LOOP #########
				print('=' * 60)
				print(f'Acquisition completed successfully at {time.ctime()}')

			except KeyboardInterrupt:
				print('\n' + '=' * 60)
				print(f'______Halted due to Ctrl-C______ at {time.ctime()}')



			# copy the array of time values, corresponding to the last acquired trace, to the times_dataset
			# Get time array from the first displayed trace
			first_trace = traces[0]
			time_ds[0:NTimes] = scope.time_array(first_trace)[0:NTimes]      # specify number of points, sometimes scope return extras


			# Set dataset attributes and descriptions
			for tr in traces:
				if datasets[tr].len() == 0:
					datasets[tr] = numpy.zeros(shape=(NPos,NTimes))
					datasets[tr].attrs['description'] = 'NOT RECORDED: ' + self.get_channel_description(tr)              # callback arg to the current function
					datasets[tr].attrs['recorded']      = False
				else:
					datasets[tr].attrs['description'] = self.get_channel_description(tr)                              # callback arg to the current function
					datasets[tr].attrs['recorded']      = True


		f.close()  # close the HDF5 file
		print(f'HDF5 file closed: {ofn}')
		return ofn 