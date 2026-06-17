# -*- coding: utf-8 -*-
"""
McPherson Spectrometer Controller Module

This module provides a Python interface for controlling the McPherson spectrometer 
scan controller 789A-4 via serial communication using PySerial.

Features:
- Serial communication with spectrometer controller
- Motor movement controls (scan up/down, positioning)
- Speed control and status monitoring
- Automated homing procedures
- Error handling and status reporting

Author: LAPD Team
Created: original version 2020 when McPherson was purchased; new modifications 2022?
Last Modified: June, 2025 (restructured code)
Dependencies: pyserial

Hardware Documentation:
- PySerial: https://pyserial.readthedocs.io/
- ASCII commands: https://mcphersoninc.com/pdf/789A-4.pdf
- Manual: https://nstx.pppl.gov/nstxhome/DragNDrop/Operations/Diagnostics_&_Support_Sys/DIMS/789A3%20Manual.pdf
"""

import serial
import serial.tools.list_ports
import time


class spectrometer:
	def __init__(self, comm_port=None, timeout=2, verbose=False):

		self.steps_per_rev = 36000    # Default motor resolution is 36000 steps/rev
		self.nm_per_rev = 4 # 4 nm/Motor rev, this depends on grating

		self.sp = None
		self.verbose = verbose
		if comm_port is not None:
			self.comm_port = comm_port
		else:
			lpis = list(serial.tools.list_ports.comports())        # lpi = ListPortInfo object
			for lpi in lpis:
				self.comm_port = lpi[0]
				print('found', lpi[0], '   description:',lpi[1])
				
		self.comm_port = lpis[0][0]

		self.sp = serial.Serial(self.comm_port, timeout=timeout)        # open serial port

		if self.verbose:
			print('connected as "',self.sp.name,'"',sep='')                # determine and mention which port was really used

		if verbose:
			print('attempting to establish communications with scan controller')
		self.id = self.send_cmd(' ')   # initialize - "After power-up, always send an ASCII [SPACE] before any other command is sent"
		if len(self.id) == 0:
			self.id = b'(unknown due to no response from Scan Controller)'
			if verbose:
				print('Scan Controller did not respond. Initialization already completed?')
		else:
			if self.id[-2:] == "\r\n":
				self.id = self.id[0:-2]
			print('Initialized Scan Controller "',self.id,'"',sep='')

	def __repr__(self):
		return self.id.decode()

	def __str__(self):
		return self.__repr__()

	def __bool__(self):
		"""Valid if the serial port is open."""
		if type(self.sp) != type(None):
			return self.sp.is_open()
		return False

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.__del__()

	def __del__(self):
		"""Close the serial port."""
		if type(self.sp) != type(None):
			self.sp.close()       # close serial port
			self.sp = None

	def send_cmd(self, s):
		""" Send ASCII command to scan controller.
			Add carriage return to every command send except [SPACE] (according to the manual).
			read_until() returns bytes when available and reads until '\n' is found
			The return string contains the command itself at the beginning and '\r\n' at the end. These are removed from the return of the function.
		"""
		c = s
		if c != ' ':
			c += '\r\n'
		cmd = c.encode()
		nw = self.sp.write(cmd)
		if nw != len(cmd):
			self.sp.flush()     # nominally, waits until all data is written
		if self.verbose:
			print('send_cmd("',s,'")', sep='',end='')
		c_r = self.sp.read_until().decode()
		if c_r[-2:] == '\r\n':
			c_r = c_r[:-2]           # truncate cr-lf from end
		if c_r[0:len(s)] == s:
			c_r = c_r[len(s):]       # delete copy of cmd from beginning
		if self.verbose:
			print(' -->', c_r)
		return c_r

	def flush(self):
		self.sp.flush()

	def close(self):
		self.sp.flush()
		self.sp.close()

#=====================================================================
#=====================================================================

	def is_moving(self):
		'''Read moving status to check if motor is still moving'''
		resp = self.send_cmd('^')
		if int(resp) == 0:
			if self.verbose:
				print('Not moving')
			return False

		elif int(resp) == 16:
			if self.verbose:
				print('Slewing')
			return True

		elif int(resp) == 1:
			if self.verbose:
				print('Moving')
			return True

		elif int(resp) == 2:
			if self.verbose:
				print('Moving fast')
			return True
		
		elif int(resp) == 43:
			if self.verbose:
				print('Somehow moving')
			return True

		else:
			print('Unknown moving status, but we set it to moving')
			return True

	def wait_for_motion_complete(self, delay=0.5):
		'''Check motor moving status every 0.1 sec until stopped'''
		
		while self.is_moving() == True:
			time.sleep(delay)
		print('Motor stopped')

#=====================================================================

	def scan_up(self, d):
		steps = int(d / self.nm_per_rev * self.steps_per_rev)
		self.send_cmd('+' + str(steps))

	def scan_down(self, d):
		steps = int(d / self.nm_per_rev * self.steps_per_rev)
		self.send_cmd('-' + str(steps))

	def set_speed(self, d):
		if d < 36:
			print('Speed should be larger than 36 sps')
		if d > 60000:
			print('Speed should be smaller than 60000sps')
		else:

			self.send_cmd('V' + str(d))

	def acquire_speed(self):
		pass

	def stop_motor(self):
		self.send_cmd('@')

#=====================================================================

	def homing(self):
		'''
		Homing should be done prior to scanning.
		Always perform the Homing Procedure every time power is disconnected.
		Refer to the instruction manual for the homing procedure
		'''

		self.send_cmd('A8') # Enable home circuit

		status = self.send_cmd(']') # Check home switch and try to move
		
		if self.verbose: print('The initial home switch status is', status)
		
		# if the current wavelength is below the home wavelength
		if int(status) == 32:
			self.send_cmd('M+23000')
			while True:           
				try:
					resp = self.send_cmd(']')
					if self.verbose: print('status is', status, ', continue scanning...')
					if int(resp) == 2:
						if self.verbose: print('switch is now clear, moving to the next step.')
						self.stop_motor()
						break
					time.sleep(0.8)
				except KeyboardInterrupt:
					self.stop_motor()
					
		# if the current wavelength is above the home wavelength (Home Switch LED doesn't light)                  
		elif int(status) == 0:            
			self.send_cmd('M-23000')
			while True:           
				try:
					resp = self.send_cmd(']')
					if self.verbose: print('status is', resp, ', continue scanning...')
					if int(resp) == 34:
						if self.verbose: print('switch is now blocked, moving to the next step.')
						self.stop_motor()
						break
					time.sleep(0.8)
				except KeyboardInterrupt:
					self.stop_motor()
					print('Motor stopped by keyboard interruption. Homing procedure aborted.')
					return False
				
		
		else:
			print('The starting status is: ', status, ' Cannot perform home switching. Please try again.')
			return False
		time.sleep(1)
		self.send_cmd('-108000')
		self.wait_for_motion_complete()
		self.send_cmd('+72000')
		self.wait_for_motion_complete()
		self.send_cmd('A24')
		time.sleep(0.5)
		self.send_cmd('F1000,0')
		self.wait_for_motion_complete()
		self.send_cmd('A0')
		print('Homing procedure completed.')
		return True


if __name__ == '__main__':
	""" standalone """

	p = spectrometer(verbose=True)

	print('done')
