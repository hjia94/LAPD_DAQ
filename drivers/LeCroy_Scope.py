"""
# -*- coding: utf-8 -*-

This file defines the class that implements communication with a LeCroy scope.

To use this code, follow these setup steps:
1. Install the pyvisa library (required for instrument communication).
2. Choose a backend:
	- Option A: Use pyvisa-py (pure Python, works on any OS)
	- Option B: Use NI-VISA (National Instruments VISA)
3. If using NI-VISA:
	- Install the LeCroy "passport" driver for NI-VISA (see links below)
	- Ensure NI-VISA is properly installed and configured (refer to NI-VISA documentation)
4. Confirm you can communicate with your LeCroy scope over TCP/IP (VICP protocol).
5. Refer to the documentation links below for troubleshooting and advanced setup.

See inline comments for the function of the member functions

Header interpretation is based on PP's previous internet scrapings in C++ program ScopeData, in particular
"P:\W\ScopeData\Scopedata\Lecroy_Binary_Header.h"

PyVisa documentation:                                https://media.readthedocs.org/pdf/pyvisa/1.6/pyvisa.pdf
LeCroy Automation Command Reference Manual:          http://cdn.teledynelecroy.com/files/manuals/automation_command_ref_manual_ws.pdf
LeCroy Remote Control Manual for "X-Stream" scopes:  http://cdn.teledynelecroy.com/files/manuals/wm-rcm-e_rev_d.pdf
pyvisa-py: compatible for any operating system       https://pyvisa.readthedocs.io/projects/pyvisa-py/en/latest/
National Instruments Visa at                         http://www.ni.com/download/ni-visa-16.0/6184/en/     (Aug 2016) NOTE: PyVisa FAQ Points to an old version
	setting up the LeCroy scope "Passport":          http://forums.ni.com/ni/attachments/ni/170/579106/1/VICP-NI-MAX.doc
LeCroy "passport" for NI-Visa:                       http://teledynelecroy.com/support/softwaredownload/home.aspx
NI-VISA User Manual                   http://digital.ni.com/manuals.nsf/websearch/266526277DFF74F786256ADC0065C50C
NI-VISA Programmer Reference Manual   http://digital.ni.com/manuals.nsf/websearch/87E52268CF9ACCEE86256D0F006E860D


Originally created on Wed Aug 31, 2016 to communicate to LeCroy X-Stream scope
@author: Patrick  

Major update Jan.2025
Reformat how header and trace bytes are handled in LeCroy_Scope class
hdr and trace_bytes are no longer updated in the class, but are returned by the function based on input trace
- Added functions
	- get_header_bytes
	- parse_header
	- acquire_bytes
- Modified functions
	- acquire
Add function acquire_sequence_data to acquire data in sequence mode
TODO: Need to modify other functions in class to reflect changes in hdr and trace_bytes

Update June.2025
- Adjusted wait_for_max_sweeps to use "SINGLE" mode when scope internal averaging is not used
- Update wait_for_sweeps with changes in 2021 that checks if scope clears sweeps successfully
- Adjust progress printing during scope internal averaging to show progress without verbose mode
"""

import numpy
import pyvisa as visa
from typing import Tuple
from pyvisa.resources import MessageBasedResource
from pyvisa.errors import VisaIOError
import collections
import struct
import sys
import pylab as plt
import matplotlib.image as mpimg
import time
import os

# the header recorded for each trace
# 63 entries, 346 bytes
WAVEDESC = collections.namedtuple('WAVEDESC',
['descriptor_name', 'template_name', 'comm_type', 'comm_order',
 'wave_descriptor', 'user_text', 'res_desc1', 'trigtime_array', 'ris_time_array',
 'res_array1', 'wave_array_1', 'wave_array_2', 'res_array2', 'res_array3',
 'instrument_name', 'instrument_number', 'trace_label', 'reserved1', 'reserved2',
 'wave_array_count', 'pnts_per_screen', 'first_valid_pnt', 'last_valid_pnt',
 'first_point', 'sparsing_factor', 'segment_index', 'subarray_count', 'sweeps_per_acq',
 'points_per_pair', 'pair_offset', 'vertical_gain', 'vertical_offset', 'max_value',
 'min_value', 'nominal_bits', 'nom_subarray_count', 'horiz_interval', 'horiz_offset',
 'pixel_offset', 'vertunit', 'horunit', 'horiz_uncertainty',
 'tt_second', 'tt_minute', 'tt_hours', 'tt_days', 'tt_months', 'tt_year', 'tt_unused',
 'acq_duration', 'record_type', 'processing_done', 'reserved5', 'ris_sweeps',
 'timebase', 'vert_coupling', 'probe_att', 'fixed_vert_gain', 'bandwidth_limit',
 'vertical_vernier', 'acq_vert_offset', 'wave_source'])

WAVEDESC_SIZE = 346
"""
 The header should be 346 bytes (with correct packing); it is preceded by 15 bytes for the def9header etc.
 note: for simplicity I expanded the leCroy_time struct into explicit fields above, labeled tt_xxx

 To get floating values from the stored raw data: y[i] = vertical_gain * data[i] - vertical_offset

 some entries:
   horiz_offset: Seconds between trigger and first data point (note this is <= 0 if trigger is visible on screen)
   bandwidth_limit: 0 = off, 1 = on
   record_type:  see below
   processing_done: see below
   timebase: see below
   fixed_vert_gain: see below
   vert_coupling: see below
   wave_source:  0=CH1, 1=CH2, 2=CH3, 3=CH4, 9=Unknown
"""
WAVEDESC_FMT = '=16s16shhllllllllll16sl16shhlllllllllhhffffhhfdd48s48sfdBBBBhhfhhhhhhfhhffh'
#    The initial '=' character specifies native byte order, with standard (C) alignment.

RECORD_TYPES = ['single_sweep', 'interleaved', 'histogram', 'graph', 'filter_coefficient',
				'complex', 'extrema', 'sequence_obsolete', 'centered_RIS', 'peak_detect']

PROCESSING_TYPES = ['no_processing', 'fir_filter', 'interpolated', 'sparsed',
					'autoscaled', 'no_result', 'rolling', 'cumulative']

TIMEBASE_IDS = ['1 ps', '2 ps', '5 ps', '10 ps', '20 ps', '50 ps', '100 ps', '200 ps', '500 ps',
				'1 ns', '2 ns', '5 ns', '10 ns', '20 ns', '50 ns', '100 ns', '200 ns', '500 ns',
				'1 us', '2 us', '5 us', '10 us', '20 us', '50 us', '100 us', '200 us', '500 us',
				'1 ms', '2 ms', '5 ms', '10 ms', '20 ms', '50 ms', '100 ms', '200 ms', '500 ms',
				'1 s',  '2 s',  '5 s',  '10 s',  '20 s',  '50 s',  '100 s',  '200 s',  '500 s',
				'1 ks', '2 ks', '5 ks']   # these are per division; ALSO: 100 corresponds to EXTERNAL

VERT_GAIN_IDS = ['1 uV', '2 uV', '5 uV', '10 uV', '20 uV', '50 uV', '100 uV', '200 uV', '500 uV',
				 '1 mV', '2 mV', '5 mV', '10 mV', '20 mV', '50 mV', '100 mV', '200 mV', '500 mV',
				 '1 V',  '2 V',  '5 V',  '10 V',  '20 V',  '50 V',  '100 V',  '200 V',  '500 V',
				 '1 kV', '2 kV', '5 kV', '10 kV']   # these are per division; pp added the last 3

VERT_COUPLINGS = ['DC 50 Ohms', 'ground', 'DC 1 MOhm', 'ground', 'AC 1 MOhm']

EXPANDED_TRACE_NAMES = {'F1': 'Math1'   , 'F2': 'Math2'   , 'F3': 'Math3'   , 'F4': 'Math4'   ,  # documentation indicates these are possible, but some of them result in errors
						'F5': 'Math5'   , 'F6': 'Math6'   , 'F7': 'Math7'   , 'F8': 'Math8'   ,
					  'TA': 'ChannelA', 'TB': 'ChannelB', 'TC': 'ChannelC', 'TD': 'ChannelD',
					  'M1': 'Memory1' , 'M2': 'Memory2' , 'M3': 'Memory3' , 'M4': 'Memory4' ,
					  'C1': 'Channel1', 'C2': 'Channel2', 'C3': 'Channel3', 'C4': 'Channel4',
					  'C5': 'Channel5', 'C6': 'Channel6', 'C7': 'Channel7', 'C8': 'Channel8' }
KNOWN_TRACE_NAMES = sorted(list(EXPANDED_TRACE_NAMES.keys()))

class LeCroy_Scope:
	""" implements communication with a LeCroy X-Stream scope """
	scope     = None        # the common scope instance
	rm        = None        # the common resource manager instance
	rm_status = False
	valid_trace_names = ()  # list of trace names recognized by the scope (filled in on first call)
	gaaak_count = 0         # peculiar error described below (see wait_for_sweeps())
	idn_string = ''         # scope *idn response
	trace_bytes = numpy.zeros(shape=(WAVEDESC_SIZE), dtype='b')   # buffer for trace data, reassigned to the correct size later
	offscale_fraction = .005 # fraction of off-scale samples that results in auto-scale rescaling

	def __init__(self, ipv4_addr, verbose=True, timeout=5000):
		""" Opens the NI-VISA resource manager, then attempts to open the resource 'VICP::'+ipv4_addr+'::INSTR'.
			The resource-manager-open function rm_open() verifies that the scope is communicating.
			Checks with the scope to determine valid trace names (which often causes the scope to beep due to
			queries about invalid names; this is useful to confirm that communication is established, when it
			happens).
		"""
		self.verbose = verbose
		self.rm_status = self.rm_open(ipv4_addr)   # use resource manager to open 'VICP::'+ipv4_addr+'::INSTR'; assign self.scope to this "instrument"
		if not self.rm_status:
			err = '**** program exiting'
			raise(RuntimeError(err))
		self.scope.timeout    = timeout
		self.scope.chunk_size = 1000000
		self.scope.write('COMM_HEADER OFF')
		self.scope.write('COMM_FORMAT DEF9,WORD,BIN')

		if len(self.valid_trace_names) == 0:
			for tr in KNOWN_TRACE_NAMES:
				self.scope.write(tr+':TRACE?')     # this makes a characteristic set of beeps on the scope, as it fails for several of the entries in the list
				self.scope.write('CMR?')           # read (and clear) the Command Status Register to check for errors
				error_code = int(self.scope.read())
				if error_code == 0:
					self.valid_trace_names += (tr,)  # no error, assume ok

	def __repr__(self):
		""" return a printable version: not a useful function """
		return self.scope.__repr__()


	def __str__(self):
		""" return a string representation: slightly less useless than __repr__(), but still not useful """
		txt = self.scope.__repr__() + '\n'
		trs = self.displayed_traces()
		for tr in trs:
			txt += self.scope.query(tr+':VOLT_DIV?')
		txt += self.scope.query('TIME_DIV?')
		txt += self.scope.query('VBS? "return=app.Acquisition.Horizontal.NumPoints"')
		return txt

	def __bool__(self):
		""" boolean test if valid - assumes valid if the resource manager status is True """
		return self.rm_status

	def __enter__(self):
		""" no special processing after __init__() """
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		""" checks for how many times the peculiar error described below was detected (see wait_for_sweeps()),
			then calls __del__() """
		print('LeCroy_Scope:__exit__() called', end='')
		if self.gaaak_count != 0:
			print(' with', self.gaaak_count, '"gaaak" type errors', end='')
		print('  at', time.ctime())
		self.__del__()

	def __del__(self):
		""" cleanup: close the scope resource, close the resource manager """
		if self.scope != None:
			self.scope.close()
			self.scope = None
		if self.rm != None:
			self.rm.close()
			self.rm = None
		self.rm_status = False

	#-------------------------------------------------------------------------

	def rm_list_resources(self):
		""" this is a very slow process --AND-- LeCroy scopes using VISA Passport do not show up in this list, anyway """
		if self.verbose: print('<:> searching for VISA resources')
		t0 = time.time()
		self.rm.list_resources()
		t1 = time.time()
		if self.verbose and (t1-t0 > 1): print('    .............................%6.3g sec' % (t1-t0))

	#-------------------------------------------------------------------------
	#todo: how to find VICP address automatically?

	def rm_open(self, ipv4_addr)  -> bool:
		""" open the NI-VISA resource manager
			then open the scope resource 'VICP::'+ipv4_addr+'::INSTR'
			once open, attempt to communicate with the scope
			throw an exception if any of the above fails
			eventually we need to call rm_close()
		"""
		if self.rm != None:
			return True

		if self.verbose: print('<:> constructing resource manager')
		t0 = time.time()
		self.rm = visa.ResourceManager()
		t1 = time.time()
		if self.verbose and (t1-t0 > 1): print('    .............................%6.3g sec' % (t1-t0), end='')

		if self.verbose: print('<:> attempting to open resource VICP::'+ipv4_addr+'::INSTR')

		# attempt to open a connection to the scope
		try:
			self.scope = self.rm.open_resource('VICP::'+ipv4_addr+'::INSTR', resource_pyclass=MessageBasedResource)
			print('...ok')
		except Exception:
			print('\n**** Scope not found at "', ipv4_addr, '"\n')
			#return False,0,0
			raise

		# send a (standard) *IDN? query as a way of testing whether we have a scope:
		try:
			self.idn_string = self.scope.query('*IDN?')
			if self.verbose: print('<:>', self.idn_string)  # returns scope type, name, version info
		except Exception:
			print('\n**** Scope at "', ipv4_addr,'" did not respond to "*IDN?" query\n')
			self.rm.close()
			return False,0,0
		return True

	#-------------------------------------------------------------------------

	def rm_close(self):
		""" close the resource manager; should eventually be called any time rm_open is called """
		if self.rm != None:
			self.rm.close()
			self.rm = None

	#-------------------------------------------------------------------------

	def screen_dump(self, fig_name, white_background = False, png_fn = 'scope_screen_dump.png', full_screen = True, show_plot=True):
		""" obtain a screen dump from the scope, in the form of a .png file
			write the file with filenam png_fn (=argument)
			read the file and display it on the screen using matplotlib imshow() function
		"""
		if white_background:
			bckg = 'WHITE'
		else:
			bckg = 'BLACK'
		if full_screen:
			area = 'DSOWINDOW'
		else:
			area = 'GRIDAREAONLY'
		# write "hardcopy" setup information:
		self.scope.write('COMM_HEADER OFF')
		self.scope.write('HARDCOPY_SETUP DEV, PNG, BCKG, '+bckg+', DEST, "REMOTE", AREA, '+area)
		# send screen dump command
		self.scope.write('SCREEN_DUMP')
		# read screen dump information: this is exactly the contents of a .png file, typically < 40 kB
		screen_image_png = self.scope.read_raw()
		# write the .png file
		file = open(png_fn, 'wb')      # Can this be achieved without having to go to disk??
		file.write(screen_image_png)    # actually, this is not a bug, it's a feature, since we probably want the image in a file anyway
		file.close()
		x = mpimg.imread(png_fn)
		(h,w,d) = numpy.shape(x)
		if show_plot:
			plt.figure(num=fig_name, figsize=(w/100, h/100), dpi=100, facecolor='w', edgecolor='k')
			plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
			plt.imshow(x)

	#-------------------------------------------------------------------------

	def write_status_msg(self, msg):
		""" send a message to the status line on the scope; nominally this should be < 50 chars, but not checked
		"""
		if len(msg) > 49:   # specs say 49 chars max: todo: is this still the limit?
			self.scope.write('MESSAGE "'+msg[0:46]+'..."')
		else:
			self.scope.write('MESSAGE "'+msg+'"')

	#-------------------------------------------------------------------------

	def validate_channel(self, Cn)  -> str:
		""" convenience function, returns canonical channel label, C1, C2, C3, or C4
			works correctly if Cn is a string or integer
			throws a runtime error if the argument is not a proper channel label
		"""
		# channel should be 'C1','C2','C3','C4' (as opposed to the more general trace labels)
		if type(Cn) == str and (Cn == 'C1' or Cn == 'C2' or Cn == 'C3' or Cn == 'C4' or Cn == 'C5' or Cn == 'C6' or Cn == 'C7' or Cn == 'C8'):
			return Cn
		if type(Cn) == int and (Cn >= 1 and Cn <= 8):
			return 'C'+str(Cn)
		err = '**** validate_channel(): channel = "' + Cn + '" is not allowed, must be C1-4'
		raise(RuntimeError(err)).with_traceback(sys.exc_info()[2])

	#-------------------------------------------------------------------------

	def validate_trace(self, tr)  -> str:
		""" convenience function, returns canonical trace label, which is broader than a channel label
			see valid_trace_names defined at top of file
			if Cn is an integer, assumes we want a channel name
			throws a runtime error if the argument is not a proper trace label
		"""
		if type(tr) == int and (tr >= 1 and tr <= 8):
			return 'C'+str(tr)
		for trn in self.valid_trace_names:
			if tr == trn:
				return trn
		err = '**** validate_trace(): trace name "' + tr + '" is unknown'
		raise(RuntimeError(err)).with_traceback(sys.exc_info()[2])

	#-------------------------------------------------------------------------

	def max_samples(self, N = 0) -> int:
		""" mostly used for determining the number of samples the scope expects to acquire.
			If the argument N is given, this routine can also be used to attempt to set the number of samples
			to one of the following:
				500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, etc.
			except many scopes won't accept all of these
			The return value is the actual number of samples that the scope will acquire
		"""
		if N > 0:
			self.scope.write('VBS "app.Acquisition.Horizontal.MaxSamples='+str(N)+'"')
		# find out what happened:
		return int(self.scope.query('VBS? "return=app.Acquisition.Horizontal.NumPoints"'))

	#-------------------------------------------------------------------------


	def displayed_channels(self)  -> Tuple[str, ...]:    # returns a tuple of channel names, e.g. ('C1', 'C4')
		""" return displayed CHANNELS only, ignoring math, memory, etc """
		channels = ()
		self.scope.write('COMM_HEADER OFF')

		if self.scope.query('C1:TRACE?')[0:2] == 'ON':
			channels += ('C1',)
		if self.scope.query('C2:TRACE?')[0:2] == 'ON':
			channels += ('C2',)
		if self.scope.query('C3:TRACE?')[0:2] == 'ON':
			channels += ('C3',)
		if self.scope.query('C4:TRACE?')[0:2] == 'ON':
			channels += ('C4',)
		
		return channels


	def displayed_traces(self):    # returns a tuple of trace names, e.g. ('C1', 'C4', 'F1')
		""" return displayed TRACES, including math, memory, etc. """
		traces = ()
		self.scope.write('COMM_HEADER OFF')

		for tr in self.valid_trace_names:
			if self.scope.query(tr+':TRACE?')[0:2] == 'ON':
				traces += (tr,)
		return traces

	#-------------------------------------------------------------------------

	def vertical_scale(self, trace) -> float:
		""" get vertical scale setting for the trace
		"""
		Tn = self.validate_trace(trace)
		scale = float(self.scope.query('VBS? "Return=app.Acquisition.'+Tn+'.VerScale"'))
		return scale


	def set_vertical_scale(self, trace, scale) -> float:
		""" set vertical scale setting for the trace
		"""
		Tn = self.validate_trace(trace)
		self.scope.write('VBS "app.Acquisition.'+Tn+'.VerScaleVariable=True"')
		self.scope.write('VBS "app.Acquisition.'+Tn+'.VerScale='+str(scale)+'"')
		return self.vertical_scale(trace)   # it may not be what we asked for

	#-------------------------------------------------------------------------

	def averaging_count(self, channel='C1') -> int:
		""" get count of averages specified for the channel, default = read from channel 'C1'
		"""
		Cn = self.validate_channel(channel)
		NSweeps = int(self.scope.query('VBS? "Return=app.Acquisition.'+Cn+'.AverageSweeps"'))
		#todo: should this deal with traces rather than channels?
		return NSweeps


	def set_averaging_count(self, channel='C1', NSweeps=1):
		""" set count of averages for a given channel (not used)
		"""
		Cn = self.validate_channel(channel)
		if NSweeps < 1:
			NSweeps = 1
		if NSweeps > 1000000:
			NSweeps = 1000000
		self.scope.write('VBS "app.Acquisition.'+Cn+'.AverageSweeps='+str(NSweeps)+'"')


	def max_averaging_count(self) -> Tuple[int, int]:
		""" get maximum averaging count across all displayed channels
			returns #sweeps and corresponding channel. To display progress, use
			self.averaging_count(cc) where cc is the returned channel
		"""
		NSweeps = 0
		ach = None
		for ch in self.displayed_channels():
			n = self.averaging_count(ch)
			if n > NSweeps:
				NSweeps = n
				ach = ch
		if ach == None:
			# throw an exception if no channels had sweeps
			err = '**** max_averaging_count(): no displayed channels'
			raise(RuntimeError(err)).with_traceback(sys.exc_info()[2])
		return NSweeps, ach

	#-------------------------------------------------------------------------

	def wait_for_max_sweeps(self, aux_text='', timeout=100):
		""" determine maximum averaging count across all displayed channels, then wait for that many sweeps
		"""
		NSweeps, ach = self.max_averaging_count()
		if self.verbose:
			print("NSweeps is reading ", NSweeps, " Is something wrong?")
		self.write_status_msg(aux_text + 'Waiting for averaging('+str(NSweeps)+') to complete')

		if NSweeps == 1: # For single sweep, use 'SINGLE' mode on scope
			if self.verbose:
				print(f'"SINGLE" mode acquisition...')
		
			# Clear sweep on scope before acquiring data
			self.scope.write('CLEAR_SWEEPS')     # clear sweeps
			time.sleep(0.05)

			max_wait_per_sweep = timeout  # maximum seconds to wait for the single sweep to complete
			check_interval = 0.01  # seconds between status checks
			
			# Print initial status
			print(f'      Starting single sweep acquisition...', end='', flush=True)
			
			self.set_trigger_mode('SINGLE')
			
			# Wait for scope to enter STOP state with retry mechanism
			sweep_start_time = time.time()
			scope_stopped = False
			
			while time.time() - sweep_start_time < max_wait_per_sweep:
				time.sleep(check_interval)
				current_mode = self.set_trigger_mode("")
				
				if current_mode[0:4] == 'STOP':
					scope_stopped = True
					break
			
			# Complete the progress line
			print(' Complete!' if scope_stopped else ' Timed out!')
			
			# Set return values
			if scope_stopped:
				timed_out = False
				n = 1
			else:
				timed_out = True
				n = 0
		
		else: # For multiple sweeps, use the robust wait_for_sweeps method
			if self.verbose:
				print(f'"NORM" acquisition with scope internal averaging over {NSweeps} sweeps.')
			timed_out, n = self.wait_for_sweeps(ach, NSweeps, timeout)
			
		if timed_out:
			msg = 'averaging timed out at:' + str(n) + '/' + str(NSweeps) + 'after %.1f s' % timeout
		else:
			msg = 'averaging('+str(NSweeps)+'), completed, got '+str(n)

		self.write_status_msg(aux_text + msg)
		return timed_out, n

	def wait_for_sweeps(self, channel, NSweeps, timeout=100, sleep_interval=0.1):
		""" Worker for above: wait for a given channel to trigger NSweeps times
			This polls the scope to determine number of sweeps that have occurred, so may overshoot
			(to get faster polling, set sleep_interval(seconds) to a smaller number)
		"""
		#todo: instead of a timeout, generate a linear fit to sweeps per second, then fail if t_dropout > 6*sigma delayed
		#      In addition, it would be possible to actually project when the process will complete, and
		#      stop at nearly the exact time.  Then take an extra sweep if necessary. If we cared.
		channel = self.validate_channel(channel)

        # read number of sweeps at this time
		self.scope.write(channel+':WAVEFORM?')
		hdr_bytes = self.scope.read_raw()
		initial_sweeps_per_acq = struct.unpack('=l', hdr_bytes[15+148:15+148+4])[0]

		self.set_trigger_mode('AUTO')   # try to make sure it is triggering
		self.scope.write('CLEAR_SWEEPS')     # clear sweeps
		if self.verbose: print('clearing sweeps')
		time.sleep(0.25)
		self.set_trigger_mode('NORM')

		# 2021-03-26 Add the following to check count until the scope finishes clearing the sweeps
		sweeps_per_acq = struct.unpack('=l', hdr_bytes[15+148:15+148+4])[0]
		clear_sweeps_timeout = time.time()+10
		while time.time() < clear_sweeps_timeout and sweeps_per_acq > 1:

            # Some time is apparently required to allow for the scope to propagate the requested
            #   settings through to the hardware. This matters at the beginning of polling after
            #   CLEAR_SWEEPS should set have the number to 0. (Aug 2016 - Tested on LeCroy HDO4104)
            # Eliminating this delay causes intermittent "gaaak" fails as per below
            #time.sleep(1.2)    # 0.1 second still results in a gaaak remediation maybe 0.3% of the time (in verbose mode), and 0.01% (1e-4) in non-verbose mode
            # 2017-07-11 - 0.1 second -> 600 gaaak errors out of 1200 shots; changed to 0.25, much more infrequent (1/1800)

            # see if we managed to clear sweeps:
			self.scope.write(channel+':WAVEFORM?')
			hdr_bytes = self.scope.read_raw()
			sweeps_per_acq = struct.unpack('=l', hdr_bytes[15+148:15+148+4])[0]
			if sweeps_per_acq < initial_sweeps_per_acq  or  sweeps_per_acq == 1:           # added "  or  sweeps_per_acq == 1" - see 21-05-10 comment above UGLY
				break
			
		self.scope.write('COMM_FORMAT DEF9,BYTE,BIN')             # set byte data transfer
		self.scope.write('WAVEFORM_SETUP SP,0,NP,1,FP,1,SN,0')    # read 1 data points
		self.scope.write('COMM_HEADER OFF')
		timeout += time.time()

		print(f'      Waiting for {NSweeps} sweeps: 0/{NSweeps}', end='', flush=True)

		timed_out = True
		gaaak = 0
		while time.time() < timeout:
			#bad idea? pp 4/4/2021
			#if sweeps_per_acq == 1:
			#	timed_out = False
			#	break   # no need to do this if we only want the current trace

			#  -----------  bug 2  ----------------
			# scope goes crazy: says "Processing" for maybe 20 seconds, then comes back to life
			# pyvisa.errors.VisaIOError: VI_ERROR_TMO (-1073807339): Timeout expired before operation completed.
			#   so I added this pyvisa_error_count try-except business:

			pyvisa_error_count = 0
			while pyvisa_error_count < 99:  # 17-07-13 this is failing on 12.5MS (black scope) -> (stupid scope) error; stop-trigger by hand fixes problem somehow
				time.sleep(sleep_interval)
				t0 = time.time()
				try:
					if self.verbose: print("wait_for_sweeps(): attempting to read waveform data")
					self.scope.write(channel+':WAVEFORM?')              ### this is all we really want to do here:
					hdr_bytes = self.scope.read_raw()                   ###    get the scope waveform data
					break                                               ###    and stop trying
				except VisaIOError as err:
					pyvisa_error_count += 1
					if pyvisa_error_count > 98:
						raise
					print('pyvisa.errors.VisaIOError:',err, '  at', time.ctime())   # todo add line# information
					print('(stupid scope)')
					for c in "will try again.":
						print(c,end='',flush=True)
						time.sleep(0.33333)
					print(" now (", pyvisa_error_count, ")",sep='')
					timeout += time.time()-t0

			# self.scope.read_raw()    -----------  bug 1  ----------------
			# NOTE: THERE IS SOMETHING SLIGHTLY OFF ABOUT THE SCOPE PROCESSING HERE
			#       The next value of sweeps_per_acq CAN BE WRONG
			#             -- I discovered this because it occasionally registers as > NSweeps on the first time through
			# NOTE2: delay AND verbose=False reduces the number of errors to ~ 1/10000, so there is possibly a
			#        communications interaction problem, rather than simply a scope issue
			# At any rate, we detect this and tail-recurse to try again if we find the problem occurred

			# desired field is a long int at offset 148 in the header; note: there 15 bytes of non-header at beginning of buffer
			sweeps_per_acq = struct.unpack('=l', hdr_bytes[15+148:15+148+4])[0]   # note: struct.unpack returns a tuple
			
			gaaak = sweeps_per_acq   # for catching error, below
			if sweeps_per_acq >= NSweeps:
				timed_out = False
				break
			
			# Update progress every sweep completion - overwrite same line
			print(f'\r      Waiting for {NSweeps} sweeps: {sweeps_per_acq}/{NSweeps}', end='', flush=True)

		#17-07-11 self.scope.write('TRIG_MODE STOP')  # stop triggering
		self.set_trigger_mode('STOP')

		# get final number after we stop triggering:
		self.scope.write(channel+':WAVEFORM?')
		hdr_bytes = self.scope.read_raw()
		sweeps_per_acq = struct.unpack('=l', hdr_bytes[15+148:15+148+4])[0]

		if gaaak > sweeps_per_acq:			# check for scope error described above
			self.gaaak_count += 1
			print('=o=o=o=o=o=o=o==================================================gaaak, read', gaaak, 'then', sweeps_per_acq)
			return self.wait_for_sweeps(channel, NSweeps, timeout, sleep_interval)  # tail recurse to try again

		print(f'\r      Waiting for {NSweeps} sweeps: {sweeps_per_acq}/{NSweeps} - Complete!')
		return timed_out, sweeps_per_acq

	#-------------------------------------------------------------------------

	def translate_header_bytes(self, header_bytes) -> numpy.array:
		""" return a numpy byte array containing the header """
		return WAVEDESC._make(struct.unpack(WAVEDESC_FMT, header_bytes))
	
	def parse_header(self, hdr):
		"""Parse the header from raw trace data and store it in hdr
		
		Returns:
			tuple: (NSamples, ndx0) for data parsing
			- NSamples: Number of samples in current segment
			- ndx0: Starting index of data in trace_bytes
		"""
		if hdr.comm_type not in [0, 1]:
			err = '**** hdr.comm_type = ' + str(hdr.comm_type) + '; expected value is either 0 or 1'
			raise(RuntimeError(err)).with_traceback(sys.exc_info()[2])

		# Check if in sequence mode
		is_sequence = hdr.subarray_count > 1

		if is_sequence:
			# In sequence mode, wave_array_1 is total points across all segments
			NSamples = int(hdr.wave_array_1 / hdr.subarray_count)
			if hdr.comm_type == 1:  # If WORD data, divide by 2
				NSamples = int(NSamples / 2)
		else:
			# Normal mode - same as before
			if hdr.comm_type == 0:
				NSamples = hdr.wave_array_1
			else:
				NSamples = int(hdr.wave_array_1/2)

		if self.verbose: 
			print('<:> NSamples =', NSamples)
			if is_sequence:
				print('<:> Sequence mode: segment', hdr.segment_index, 'of', hdr.subarray_count)

		if NSamples == 0:
			err = '**** fail because NSamples = 0 (possible cause: trace has no data? scope not triggered?)'
			err = '**** fail because NSamples = 0 (possible cause: trace has no data? scope not triggered?)\nIF SCOPE IS IN 2-CHANNEL MODE BUT CHANNEL 1 or 4 ARE SELECTED, they have no data'
			raise(RuntimeError(err)).with_traceback(sys.exc_info()[2])

		if self.verbose:
			print('<:> record type:      ', RECORD_TYPES[hdr.record_type])
			print('<:> timebase:         ', TIMEBASE_IDS[hdr.timebase], 'per div')
			print('<:> vertical gain:    ', VERT_GAIN_IDS[hdr.fixed_vert_gain], 'per div')
			print('<:> vertical coupling:', VERT_COUPLINGS[hdr.vert_coupling])
			print('<:> processing:       ', PROCESSING_TYPES[hdr.processing_done])
			print('<:> #sweeps:          ', hdr.sweeps_per_acq)
			print('<:> enob:             ', hdr.nominal_bits)
			vert_units = str(hdr.vertunit).split('\\x00')[0][2:]    # for whatever reason this prepends "b'" to string
			horz_units = str(hdr.horunit).split('\\x00')[0][2:]     # so ignore first 2 chars
			print('<:> data scaling      gain = %6.3g, offset = %8.5g' % (hdr.vertical_gain, hdr.vertical_offset), vert_units)
			print('<:> sample timing       dt = %6.3g,   offset = %8.5g' % (hdr.horiz_interval, hdr.horiz_offset), horz_units)

		# Calculate data offset
		ndx0 = (15+WAVEDESC_SIZE) + hdr.user_text + hdr.trigtime_array + hdr.ris_time_array + hdr.res_array1
		
		return NSamples, ndx0
	
	def acquire_bytes(self, trace, seg=0): # segment=0 gets all samples
		trace = self.validate_trace(trace)
		# Set up waveform transfer format
		self.scope.write(f'WAVEFORM_SETUP SP,0,NP,0,FP,1,SN,{seg}')
		
		if self.verbose: print('\n<:> reading',trace,'from scope')
		t0 = time.time()

		self.scope.write(trace+':WAVEFORM?')
		trace_bytes = self.scope.read_raw()
		header_bytes = trace_bytes[15:15+WAVEDESC_SIZE]

		t1 = time.time()
		if self.verbose: print('    .............................%.1f sec' % (t1-t0))
		return trace_bytes, header_bytes

	def acquire(self, trace, seg=0, raw=False):
		"""Acquire scope data for a single trace."""

		trace_bytes, header_bytes = self.acquire_bytes(trace, seg)
		hdr = self.translate_header_bytes(header_bytes)
		NSamples, ndx0 = self.parse_header(hdr)
		
		if self.verbose: print('<:> computing data values')
		t0 = time.time()

		if hdr.comm_type == 1:   # data returned in words (short integers)
			ndx1 = ndx0 + NSamples*2
			wdata = struct.unpack(str(NSamples)+'h', trace_bytes[ndx0:ndx1])
			if raw:
				data = wdata
			else:
				data = numpy.array(wdata) * hdr.vertical_gain - hdr.vertical_offset
		elif hdr.comm_type == 0: # data returned in bytes (signed char)
			ndx1 = ndx0 + NSamples
			cdata = struct.unpack(str(NSamples)+'b', trace_bytes[ndx0:ndx1])
			if raw:
				data = cdata
			else:
				data = numpy.array(cdata) * hdr.vertical_gain - hdr.vertical_offset
				
		t1 = time.time()
		if self.verbose: print('    .............................%.1f sec' % (t1-t0))
		return data, header_bytes

	def acquire_sequence_data(self, trace):
		"""
		Acquire scope data in sequence mode.
		"""

		trace_bytes, header_bytes = self.acquire_bytes(trace)
		hdr = self.translate_header_bytes(header_bytes)

		# Get number of segments
		if hdr.subarray_count < 2:
			raise RuntimeError("Sequence mode requires at least 2 segments.")
			
		if self.verbose:
			print(f'<:> Acquiring {hdr.subarray_count} segments from {trace}')
			
		segment_data = []
		for segment in range(1, hdr.subarray_count + 1):  # LeCroy uses 1-based segment indexing
			if self.verbose:
				print(f'    Reading segment {segment}/{hdr.subarray_count}', end='\r')
			
			data, _ = self.acquire(trace, segment)
			segment_data.append(data)
			
		if self.verbose:
			print('\n<:> Sequence acquisition complete')
				
		return segment_data, header_bytes

	def time_array(self, trace):
		""" Return a numpy array containing sample times.
			In sequence mode, returns time array for a single segment.
			Note: only valid after a call to acquire or acquire_sequence_data
		"""

		trace_bytes, header_bytes = self.acquire_bytes(trace)
		hdr = self.translate_header_bytes(header_bytes)
		
		# In sequence mode, wave_array_1 is total points across all segments
		if hdr.subarray_count > 1:
			NSamples = int(hdr.wave_array_1 / hdr.subarray_count)
			if hdr.comm_type == 1:
				NSamples = int(NSamples / 2)
		else:
			NSamples = hdr.wave_array_1 if hdr.comm_type == 0 else int(hdr.wave_array_1/2)
			
		t0 = float(hdr.horiz_offset)
		horiz_interval = float(hdr.horiz_interval)
		return numpy.linspace(t0, t0 + NSamples * horiz_interval, NSamples, endpoint=False)

	#-------------------------------------------------------------------------
	def get_sequence_trigger_times(self): # DOES NOT WORK
		"""
		Get trigger times for each segment in sequence mode.
		Must be called after acquire_sequence_data.
		
		Returns:
			numpy.array: Array of trigger times in seconds relative to first trigger
		"""
		if not hasattr(self, 'hdr'):
			raise RuntimeError("No trace data available. Call acquire_sequence_data first.")
			
		if self.hdr.subarray_count < 2:
			raise RuntimeError("Not in sequence mode.")
			
		# Calculate offset to trigger time array
		trig_offset = 15 + WAVEDESC_SIZE + self.hdr.user_text
		
		# Each trigger time is stored as a double (8 bytes)
		trigger_times = []
		for i in range(self.hdr.subarray_count):
			trig_time = struct.unpack('d', self.trace_bytes[trig_offset + i*8:trig_offset + (i+1)*8])[0]
			trigger_times.append(trig_time)
			
		return numpy.array(trigger_times)
	#-------------------------------------------------------------------------

	def set_trigger_mode(self, trigger_mode)  -> str:
		""" set the scope trigger mode to: 'AUTO', 'NORM', 'SINGLE', or 'STOP'
			if the argument is not one of these, does not change trigger mode
			returns trigger mode prior to change, or current trigger mode if argument is e.g. ''
		"""
		self.scope.write('COMM_HEADER OFF')
		prev_trigger_mode = self.scope.query('TRIG_MODE?')
		if trigger_mode == 'AUTO':
			self.scope.write('TRIG_MODE AUTO')
		elif trigger_mode == 'NORM':
			self.scope.write('TRIG_MODE NORM')
		elif trigger_mode == 'SINGLE':
			self.scope.write('TRIG_MODE SINGLE')
		elif trigger_mode == 'STOP':
			self.scope.write('TRIG_MODE STOP')
		else:
			return prev_trigger_mode

		for i in range(25):   #17-07-11 added verification
			txt = self.scope.query('TRIG_MODE?')
			if txt[0:3] == trigger_mode[0:3]:   # '\n' stuck on end it seems
				break
			print('set_trigger_mode(',trigger_mode,')    attempt',i,':  TRIG_MODE is',txt)
			time.sleep(0.1)

		return prev_trigger_mode

	#-------------------------------------------------------------------------

	def expanded_name(self, tr) -> str:
		""" Returns a long version of a trace name; e.g. C1 -> Channel1,  F2 -> Math2, etc """
		if tr in EXPANDED_TRACE_NAMES.keys():
			return EXPANDED_TRACE_NAMES[tr]
		return "unknown_trace_name"

	#-------------------------------------------------------------------------

	def dumtest(self):
		r1 = self.scope.query('PANEL_SETUP?')
		self.scope.write('*SAV 1')		# save entire front panel state in nonvolatile #1
		print(len(r1))

		self.scope.write('VBS app.SaveRecall.Setup.PanelFilename="REMOTE"')
		r2 = self.scope.query('app.SaveRecall.Setup.DoSavePanel')
		print(len(r2))

	#-------------------------------------------------------------------------

	def autoscale(self, trace):
		averaging_count = self.averaging_count(trace)
		self.set_averaging_count(trace, 1)
		time.sleep(0.05)

		#print("autoscale() called")

		status = False
		while True:

			#print("wait for sweeps, trace ", trace)
			self.wait_for_sweeps(trace, 1, timeout=100, sleep_interval=.1)

			#print("acquire")
			data = self.acquire(trace, True)     # read raw scope data

			#print("trig mode normal")
			self.scope.write('TRIG_MODE NORM')   # try to make sure it is triggering

			#print("max =",self.hdr.min_value, "min =",self.hdr.max_value)         # edges of the grid

			delta = (self.hdr.max_value - self.hdr.min_value) * self.offscale_fraction
			#print("delta=",delta)

			bins=(-2**15, self.hdr.min_value+delta, self.hdr.max_value-delta, 2**15-1)
			if bins[0] == bins[1]:
				bins[1] = bins[0]+1   # make bottom bin has finite width
			if bins[2] == bins[3]:
				bins[2] = bins[3]-1   # make sure top bin has finite width
			# 3 bins: low, ok, high

			hist,e = numpy.histogram(data, bins=bins)

			if self.verbose:
				print('hist=',hist, end="     ")
				print('edges=',e)

			NSamples = numpy.size(data)
			if hist[1] > (1-self.offscale_fraction)*NSamples:   # e.g. we want 99% of all samples in the OK bin
				status = True
				break

			# if here, more than offscale_fraction of the samples are defined as "saturated" (d >.98*max or d <.98*min)
			scale = self.vertical_scale(trace)
			#print('scale=',scale,end=' -> ')

			# for now limit = 1 V/div as per 50ohm setting, todo: fix
			if scale == 1:
				print("autoscale(): can't go any larger than 1V/div on 50ohm setting")
				status = False
				break

			scale *= 1.41421356237   # sqrt(2)
			if scale > 1:
				scale = 1

			scale = self.set_vertical_scale(trace, scale)
			print("                                 autoscale(",trace,"): setting vertical scale = ",scale, sep='')

			# now loop to try again

		#print("resetting averaging count for", trace, "and returning status =", status)
		self.set_averaging_count(trace, averaging_count)
		return status
	
	def calibrate(self, a=True): # True: calibrate scope  False: turn off auto calibration
		if a == True:
			self.scope.write('*CAL?')
			time.sleep(15)
		else:
			self.scope.write('AUTO_CALIBRATE OFF')




#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================


if __name__ == '__main__':

	import matplotlib.pyplot as plt

	with LeCroy_Scope("192.168.7.91", verbose=True) as scope:
		scope.wait_for_max_sweeps()

