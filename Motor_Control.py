#
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:29:25 2017

@author: Jia Han

2D and 3D motor control for processing chamber
"""

import math
from Motor_Control_1D import Motor_Control
import time
import numpy

from scipy.optimize import minimize

# TODO: fix calculate_velocity to read from cm_per_turn directly
#############################################################################################
#############################################################################################
"""
Class Motor_Control_2D: controls 2D probe drive with 2-motors (x,y) using Motor_Control_1D

functions: wait_for_motion_complete, test_limit, probe_to_motor, calculate_velocity, set_movement_velocity
properties: motor_velocity, motor_positions, stop_now, reset_motor, set_zero, enable, disable
"""
class Motor_Control_2D:

	def __init__(self, x_ip_addr = None, y_ip_addr = None):
		self.x_mc = Motor_Control(verbose=True, server_ip_addr= x_ip_addr, name='x', cm_per_turn = 0.254)
		self.y_mc = Motor_Control(verbose=True, server_ip_addr= y_ip_addr, name='y', cm_per_turn = 0.508)
		# Velmex model number NN10-0300-E01-21 (short black linear drives)
		
		self.probe_in = 62.948 # LAPD=>58.771 # Distance from chamber wall to chamber center
		self.poi = 125.3624 #112.762 # Length of probe outside the chamber from pivot to end
		self.ph = 20 # Height from probe shaft to where z-motor actually moves

		self.motor_velocity = 4, 4

	#-------------------------------------------------------------------------------------------
	"""
	Set and get the target velocity of motor in units of rev/sec
	"""
	@property
	def motor_velocity(self):
		return self.x_mc.motor_speed, self.y_mc.motor_speed

	@motor_velocity.setter
	def motor_velocity(self, v):
		xv, yv = v
		self.x_mc.motor_speed = xv
		self.y_mc.motor_speed = yv

	#-------------------------------------------------------------------------------------------
	"""
	Set motor velocity according to its current position and target position
	"""
	def set_movement_velocity(self, motor_x, motor_y):
		x_m, y_m = self.motor_positions

		# distance between current motor position to final motor position
		delta_x = abs(motor_x - x_m)
		delta_y = abs(motor_y - y_m)

		# calculate velocity such that probe moves on a straight line
		v_motor_x, v_motor_y = self.calculate_velocity(delta_x, delta_y)

		# Set motor velocity accordingly
		self.motor_velocity = v_motor_x, v_motor_y

	"""
	Convert probe velocity vector to motor velocity vector
	"""
	def calculate_velocity(self, del_x, del_y):
		default_speed = 4

		del_r = math.sqrt(del_x**2 + del_y**2)
		if del_r == 0:
			v_x, v_y = 0, 0
		else:
			v_x = default_speed * del_x / del_r
			v_y = default_speed * del_y / del_r

		v_motor_x = v_x * 2  # factor of 2 due to different cm_per_turn
		v_motor_y = v_y

		v_motor_x = round(v_motor_x, 3)
		v_motor_y = round(v_motor_y, 3)

		return v_motor_x, v_motor_y

	#--------------------------------------------------------------------------------------------------
	"""
	Set and get current motor position
	"""
	@property
	def motor_positions(self):
		return self.x_mc.motor_position, self.y_mc.motor_position
	
	@motor_positions.setter
	def motor_positions(self, mpos):
		x_m, y_m = mpos
		self.x_mc.motor_position = x_m
		self.y_mc.motor_position = y_m

		self.wait_for_motion_complete()
			
	#-------------------------------------------------------------------------------------------
	"""
	Execute after move command send to motor. Wait till motor stops moving.
	"""
	def wait_for_motion_complete(self):
		timeout = time.time() + 300

		while True:
			try:
				x_stat = self.x_mc.motor_status
				y_stat = self.y_mc.motor_status
				
				x_not_moving = x_stat.find('M') == -1
				y_not_moving = y_stat.find('M') == -1

				if x_not_moving and y_not_moving:
					time.sleep(0.2)
					break
				elif time.time() > timeout:
					raise TimeoutError("Motor has been moving for over 5min???")

				time.sleep(0.2)
			except KeyboardInterrupt:
				self.x_mc.stop_now
				self.y_mc.stop_now
				print('\n______Motor stopped and Halted due to Ctrl-C______')
				raise KeyboardInterrupt

	#--------------------------------------------------------------------------------------------------
	@property
	def stop_now(self):  # Stop motor movement immediately
		self.x_mc.stop_now
		self.y_mc.stop_now

	@property
	def set_zero(self):  # Set current position to zero
		self.x_mc.set_zero
		self.y_mc.set_zero

	@property
	def reset_motor(self):  # Similar to restart all motors
		self.x_mc.reset_motor
		self.y_mc.reset_motor

	@property
	def motor_alarm(self):
		x_al = self.x_mc.check_alarm
		y_al = self.y_mc.check_alarm

		return x_al, y_al

	#-------------------------------------------------------------------------------------------
	"""
	Convert probe space dimensions to motor movement in unit of cm 
	"""
	def probe_to_motor_LAPD(self, x, y):
		x = self.probe_in - x
		
		D = math.sqrt(x**2 + y**2)
		d2 = self.ph/math.sqrt((y/x)**2+1)
		Ltc = y/x * d2
		
		motor_x = D - self.probe_in
		motor_y = self.ph - d2 - (self.poi + Ltc)*y/x
		
		return motor_x, motor_y

	"""
	Convert encoder feedback from motor to actual probe position using scipy.minimize
	"""
	def motor_to_probe(self, motor_x, motor_y):
		def distance_fun(r1, r2):
			a = numpy.array(r1)
			b = numpy.array(r2)
			return numpy.linalg.norm(a-b)

		def fun(x, *args):
			mx, my = self.probe_to_motor_LAPD(x[0], x[1])
			return distance_fun((mx, my), args)

		args = [motor_x, motor_y]
		X0 = [motor_x, motor_y]
		res = minimize(fun, X0, (args,), options={'maxiter':15000}, method='BFGS')
		
		return round(res.x[0], 3), round(res.x[1], 3)

	#-------------------------------------------------------------------------------------------------
	"""
	Set probe position by moving two motors through calculation using probe_to_motor
	"""
	@property
	def probe_positions(self):
		x_m, y_m = self.motor_positions
		return self.motor_to_probe(x_m, y_m)

	@probe_positions.setter
	def probe_positions(self, pos):
		xpos, ypos = pos

		# Convert probe distance to motor distance
		motor_x, motor_y = self.probe_to_motor_LAPD(xpos, ypos)

		# Set movement velocity
		self.set_movement_velocity(motor_x, motor_y)

		# Move motor
		self.motor_positions = motor_x, motor_y

	#-------------------------------------------------------------------------------------------------
	@property
	def enable(self):
		self.x_mc.enable
		self.y_mc.enable

		if self.x_mc.check_alarm == True:
			self.x_mc.clear_alarm
		
		if self.y_mc.check_alarm == True:
			self.y_mc.clear_alarm

	@property
	def disable(self):
		self.x_mc.disable
		self.y_mc.disable

#############################################################################################
#############################################################################################
"""
Class Motor_Control_3D: controls 3D probe drive with 3-motors (x,y,z) using Motor_Control_1D

functions: wait_for_motion_complete, test_limit, probe_to_motor, calculate_velocity, set_movement_velocity
properties: motor_velocity, motor_positions, stop_now, reset_motor, set_zero, enable, disable
"""
class Motor_Control_3D:
	def __init__(self, x_ip_addr = None, y_ip_addr = None, z_ip_addr = None):
		self.x_mc = Motor_Control(verbose=True, server_ip_addr= x_ip_addr, name='x', cm_per_turn = 0.254)
		self.y_mc = Motor_Control(verbose=True, server_ip_addr= y_ip_addr, name='y', cm_per_turn = 0.508)
		self.z_mc = Motor_Control(verbose=True, server_ip_addr=z_ip_addr, name='z', cm_per_turn=0.254)
		# Velmex model number NN10-0300-E01-21 (short black linear drives)
		
		self.probe_in = 62.948 # LAPD=>58.771 # Distance from chamber wall to chamber center
		self.poi = 125.3624 #112.762 # Length of probe outside the chamber from pivot to end
		self.ph = 20 # Height from probe shaft to where z-motor actually moves

		self.motor_velocity = 4, 4, 4

	#-------------------------------------------------------------------------------------------
	"""
	Set and get the target velocity of motor in units of rev/sec
	"""
	@property
	def motor_velocity(self):
		return self.x_mc.motor_speed, self.y_mc.motor_speed, self.z_mc.motor_speed

	@motor_velocity.setter
	def motor_velocity(self, v):
		xv, yv, zv = v
		self.x_mc.motor_speed = xv
		self.y_mc.motor_speed = yv
		self.z_mc.motor_speed = zv

	#-------------------------------------------------------------------------------------------
	"""
	Set motor velocity according to its current position and target position
	"""
	def set_movement_velocity(self, motor_x, motor_y, motor_z):
		x_m, y_m, z_m = self.motor_positions

		# distance between current motor position to final motor position
		delta_x = abs(motor_x - x_m)
		delta_y = abs(motor_y - y_m)
		delta_z = abs(motor_z - z_m)

		# calculate velocity such that probe moves on a straight line
		v_motor_x, v_motor_y, v_motor_z = self.calculate_velocity(delta_x, delta_y, delta_z)

		# Set motor velocity accordingly
		self.motor_velocity = v_motor_x, v_motor_y, v_motor_z

	"""
	Convert probe velocity vector to motor velocity vector
	"""
	def calculate_velocity(self, del_x, del_y, del_z):
		default_speed = 4

		del_r = math.sqrt(del_x**2 + del_y**2 + del_z**2)
		if del_r == 0:
			v_x, v_y, v_z = 0, 0, 0
		else:
			v_x = default_speed * del_x / del_r
			v_y = default_speed * del_y / del_r
			v_z = default_speed * del_z / del_r

		v_motor_x = v_x * 2  # factor of 2 due to different cm_per_turn
		v_motor_y = v_y
		v_motor_z = v_z * 2  # factor of 2 due to different cm_per_turn

		v_motor_x = round(v_motor_x, 3)
		v_motor_y = round(v_motor_y, 3)
		v_motor_z = round(v_motor_z, 3)

		return v_motor_x, v_motor_y, v_motor_z

	#--------------------------------------------------------------------------------------------------
	"""
	Set and get current motor position
	"""
	@property
	def motor_positions(self):
		return self.x_mc.motor_position, self.y_mc.motor_position, self.z_mc.motor_position
	
	@motor_positions.setter
	def motor_positions(self, mpos):
		x_m, y_m, z_m = mpos
		self.x_mc.motor_position = x_m
		self.y_mc.motor_position = y_m
		self.z_mc.motor_position = z_m

		self.wait_for_motion_complete()
			
	#-------------------------------------------------------------------------------------------
	"""
	Execute after move command send to motor. Wait till motor stops moving.
	"""
	def wait_for_motion_complete(self):
		timeout = time.time() + 300

		while True:
			try:
				x_stat = self.x_mc.motor_status
				y_stat = self.y_mc.motor_status
				z_stat = self.z_mc.motor_status
				
				x_not_moving = x_stat.find('M') == -1
				y_not_moving = y_stat.find('M') == -1
				z_not_moving = z_stat.find('M') == -1

				if x_not_moving and y_not_moving and z_not_moving:
					time.sleep(0.2)
					break
				elif time.time() > timeout:
					raise TimeoutError("Motor has been moving for over 5min???")

				time.sleep(0.2)
			except KeyboardInterrupt:
				self.x_mc.stop_now
				self.y_mc.stop_now
				self.z_mc.stop_now
				print('\n______Motor stopped and Halted due to Ctrl-C______')
				raise KeyboardInterrupt

	#--------------------------------------------------------------------------------------------------
	@property
	def stop_now(self):  # Stop motor movement immediately
		self.x_mc.stop_now
		self.y_mc.stop_now
		self.z_mc.stop_now

	@property
	def set_zero(self):  # Set current position to zero
		self.x_mc.set_zero
		self.y_mc.set_zero
		self.z_mc.set_zero

	@property
	def reset_motor(self):  # Similar to restart all motors
		self.x_mc.reset_motor
		self.y_mc.reset_motor
		self.z_mc.reset_motor

	@property
	def motor_alarm(self):
		x_al = self.x_mc.check_alarm
		y_al = self.y_mc.check_alarm
		z_al = self.z_mc.check_alarm

		return x_al, y_al, z_al

	#-------------------------------------------------------------------------------------------
	"""
	Convert probe space dimensions to motor movement in unit of cm 
	"""
	def probe_to_motor_LAPD(self, x, y, z):
		x = self.probe_in - x
		
		D = math.sqrt(x**2 + y**2 + z**2)
		d2 = self.ph/math.sqrt((y/x)**2+1)
		Ltc = y/x * d2
		
		motor_x = D - self.probe_in
		motor_y = self.ph - d2 - (self.poi + Ltc)*y/x
		motor_z = z/x * (self.poi+ Ltc)
		
		return motor_x, motor_y, motor_z

	"""
	Convert encoder feedback from motor to actual probe position using scipy.minimize
	"""
	def motor_to_probe(self, motor_x, motor_y, motor_z):
		def distance_fun(r1, r2):
			a = numpy.array(r1)
			b = numpy.array(r2)
			return numpy.linalg.norm(a-b)

		def fun(x, *args):
			mx, my, mz = self.probe_to_motor_LAPD(x[0], x[1], x[2])
			return distance_fun((mx, my, mz), args)

		args = [motor_x, motor_y, motor_z]
		X0 = [motor_x, motor_y, motor_z]
		res = minimize(fun, X0, (args,), options={'maxiter':15000}, method='BFGS')
		
		return round(res.x[0], 3), round(res.x[1], 3), round(res.x[2], 3)

	#-------------------------------------------------------------------------------------------------
	"""
	Set probe position by moving three motors through calculation using probe_to_motor
	"""
	@property
	def probe_positions(self):
		x_m, y_m, z_m = self.motor_positions
		return self.motor_to_probe(x_m, y_m, z_m)

	@probe_positions.setter
	def probe_positions(self, pos):
		xpos, ypos, zpos = pos

		# Convert probe distance to motor distance
		motor_x, motor_y, motor_z = self.probe_to_motor_LAPD(xpos, ypos, zpos)

		# Set movement velocity
		self.set_movement_velocity(motor_x, motor_y, motor_z)

		# Move motor
		self.motor_positions = motor_x, motor_y, motor_z

	#-------------------------------------------------------------------------------------------------
	@property
	def enable(self):
		self.x_mc.enable
		self.y_mc.enable
		self.z_mc.enable

		if self.x_mc.check_alarm == True:
			self.x_mc.clear_alarm
		
		if self.y_mc.check_alarm == True:
			self.y_mc.clear_alarm
			
		if self.z_mc.check_alarm == True:
			self.z_mc.clear_alarm

	@property
	def disable(self):
		self.x_mc.disable
		self.y_mc.disable
		self.z_mc.disable

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================
if __name__ == '__main__': # standalone testing:
	# Test 2D motor control
	print("Testing 2D motor control...")
	mc2d = Motor_Control_2D(x_ip_addr = "192.168.0.40", y_ip_addr = "192.168.0.50")
	mc2d.probe_positions
	
	# Test 3D motor control
	print("\nTesting 3D motor control...")
	mc3d = Motor_Control_3D(x_ip_addr = "192.168.0.40", y_ip_addr = "192.168.0.50", z_ip_addr = "192.168.0.60")
	mc3d.probe_positions