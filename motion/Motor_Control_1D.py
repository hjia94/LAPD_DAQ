#
# -*- coding: utf-8 -*-
"""
Created on Oct.16.2024
@author: Jia Han
Original code from motor control used in process plasma lab

Motor control program for Applied motion motor through Ethernet with the use of Socket
Command reference can be found in: https://www.applied-motion.com/sites/default/files/hardware-manuals/Host-Command-Reference_920-0002R.pdf

Oct.2024 Update
- Change cm_per_turn as input value to init; allow setting for different Velmex drives
- Add function to turn motor by step and read motor current step
- Minor change in init function
"""


import sys
if sys.version_info[0] < 3: raise RuntimeError('This script should be run under Python 3')

import socket
import select
import time
import logging

# TODO: 'DL2' should be sent to motor when limit switch is connected properly
#       Add boolean in init to choose when stop switch is connected or not

#===============================================================================================================================================
#===============================================================================================================================================

class Motor_Control:
    """Talks to a single Applied Motion drive over a TCP socket."""

    MSIPA_CACHE_FN = 'motor_server_ip_address_cache.tmp'
    MOTOR_SERVER_PORT = 7776
    BUF_SIZE = 1024

    def __init__(self, server_ip_addr = None, cm_per_turn = 0.254, stop_switch_mode=3, msipa_cache_fn = None, verbose = False, name='not named'):

        self.cm_per_turn = cm_per_turn
        self.stop_switch_mode = str(stop_switch_mode)
        self.verbose = verbose
        self.name = name
        if msipa_cache_fn == None:
            self.msipa_cache_fn = self.MSIPA_CACHE_FN
        else:
            self.msipa_cache_fn = msipa_cache_fn

        # if we get an ip address argument, set that as the suggest server IP address, otherwise look in cache file
        if server_ip_addr != None:
            self.server_ip_addr = server_ip_addr
        else:
            try:
                # Read the previously saved IP as a first guess for the server address
                self.server_ip_addr = None
                with open(self.msipa_cache_fn, 'r') as f:
                    self.server_ip_addr = f.readline()
            except FileNotFoundError:
                self.server_ip_adddr = None

        # - - - - - - - - - - - - - - - - - - - - - - -
        if self.server_ip_addr != None  and  len(self.server_ip_addr) > 0:
            try:
                print('looking for motor server at', self.server_ip_addr,end=' ',flush=True)
                t = self.send_text('RS')
                print ('status =', t[5:])
                if t != None: #TODO: link different response to corresponding motor status
                    print('...found')
                    self.send_text('IFD') #set response format to decimal
                    cur_stat = self.motor_status

                else:
                    print('motor server returned', t, sep='')
                    print('todo: why not the correct response?')

            except TimeoutError:
                print('...timed out')
            except (KeyboardInterrupt,SystemExit):
                print('...stop finding')
                raise

        with open(self.msipa_cache_fn, 'w') as f:
            f.write(self.server_ip_addr)
        
        # Setup encoder and motor steps
        self.__stepsPerRev = self.steps_per_rev()
        
        # Setup motor stop switch mode depending on electrical hook-up
        # 1: Energized open
        # 2: Energized closed
        # 3: No connection
        self.send_text('DL'+self.stop_switch_mode)
        

        # Check and clear alarm present on motor
        alarm = self.check_alarm
        if alarm == False:
            pass
        elif '0002' in alarm :
            pos = self.motor_position
            print('Drive is hitting a stop switch at ', pos)
        else:
            print('Unknown alarm ', alarm)
            self.clear_alarm
            print(self.name, ' current status ', cur_stat)
        
#         if self.motor_speed == 10:
#             print('Motor has likely been power cycled and lost zero position')
#             print('Motor disabled until futher action')
#             self.disable



########################################################################################################
########################################################################################################
    def __repr__(self):
        """ return a printable version: not a useful function """
        return self.server_ip_addr + '; ' + self.msipa_cache_fn + '; ' + self.verbose


    def __str__(self):
        """ return a string representation: """
        return self.__repr__()

    def __bool__(self):
        """ boolean test if valid - assumes valid if the server IP address is defined """
        return self.server_ip_addr != None

    def __enter__(self):
        """ no special processing after __init__() """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ no special processing after __init__() """

    def __del__(self):
        """ no special processing after __init__() """

########################################################################################################
########################################################################################################

    def send_text(self, text, timeout:int=None, receive=True) -> str:
        """Open a socket, send a command to the motor server, return its reply.

        Note: timeout is currently ignored (needs MS-specific iocontrol).
        """
        RETRIES = 30
        retry_count = 0
        while retry_count < RETRIES:  # Retries added 17-07-11
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                ##if timeout is not None:
                ##	#not on windows: socket.settimeout(timeout)
                ##	s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack('LL', timeout, 0))
                s.connect((self.server_ip_addr, self.MOTOR_SERVER_PORT))
                break
            except ConnectionRefusedError:
                retry_count += 1
                print('...connection refused, at',time.ctime(),' Is motor_server process running on remote machine?',
                           '  Retry', retry_count, '/', RETRIES, "on", str(self.server_ip_addr))
            except TimeoutError:
                retry_count += 1
                print('...connection attempt timed out, at',time.ctime(),
                           '  Retry', retry_count, '/', RETRIES, "on", str(self.server_ip_addr))
            except KeyboardInterrupt:
                sys.exit('_______Halt due to CRTL_C________')


        if retry_count >= RETRIES:
            (" pausing in motor_control.py send_text() function, hit Enter to try again, or ^C: ")
            s.close()
            return self.send_text(text, timeout)  # tail-recurse if retry is requested


        message = bytearray(text, encoding = 'ASCII')
        buf = bytearray(2)
        buf[0] = 0
        buf[1] = 7
        for i in range(len(message)):
            buf.append(message[i])
        buf.append(13)

        s.send(buf)

        if receive:
            BUF_SIZE = 2048
            data = s.recv(BUF_SIZE)
            return_text = data.decode('ASCII')
            return return_text

        s.close()


########################################################################################################
    def steps_per_rev(self):
        """Return encoder/motor resolution (steps/rev), reconciling the two if they differ."""
        resp = self.send_text('ER') # Encoder resolution
        enco_step = int(resp[5:])

        resp = self.send_text('EG') # Motor gearing resolution
        motor_step = int(resp[5:])

        if enco_step == motor_step:
            return enco_step
        else:
            print(self.name + '-motor: Encoder step (%i) does not equal motor step(%i)'%(enco_step, motor_step))

            success = self.set_steps_per_rev(enco_step)
            if success:
                print(self.name + '-motor: set motor and encoder step to ', str(enco_step))
                return enco_step
            else:
                return None

    def set_steps_per_rev(self, steps):

        self.send_text('ER'+str(steps))
        self.send_text('EG'+str(steps))

        res = self.steps_per_rev()
        if res == steps:
            self.__stepsPerRev = steps
            return True
        else:
            print(self.name + "-motor: Fail to set encoder step equal to motor step.")
            return False

    
#-------------------------------------------------------------------------------------------

    @property
    def motor_status(self):
        """Return the drive status string (RS). Letters: A=alarm, D=disabled,
        E=drive fault, F=motor moving, H=homing, J=jogging, M=motion in progress,
        P=in position, R=ready, S=stopping, T=wait time, W=wait input."""
        return self.send_text('RS')

#-------------------------------------------------------------------------------------------

    def cm_to_steps(self, d:float) -> int:
        """ worker: convert distance in cm to number of turns motor rotates"""
        return int(d / self.cm_per_turn * self.__stepsPerRev)

    def steps_to_cm(self, step:int) -> float:
        """opposite conversion from cm_to_steps"""
        return step / self.__stepsPerRev * self.cm_per_turn

#-------------------------------------------------------------------------------------------
    # Get/set motor velocity in rev/sec
    @property
    def motor_speed(self):

        resp = self.send_text('VE')

        return float(resp[5:])

    @motor_speed.setter
    def motor_speed(self, speed):
    
        self.send_text('VE'+str(speed))

    @property
    def instant_velocity(self):
        """ Returns the instantaneous velocity of the motor in units of rev/sec"""

        resp = self.send_text('IV')
        rps = float(resp[5:]) / 60
        return rps

#-------------------------------------------------------------------------------------------
    # Get/set motor position in cm
    @property
    def motor_position(self):
        '''Return current motor position in cm (cross-checks encoder vs internal position).'''

        RETRIES = 100
        retry_count = 0

        while retry_count < RETRIES:

            resp = self.send_text('EP')  # Ask for encoder position
            resp1 = self.send_text('SP') # Ask for motor internal position

            try:
                pos = float(resp[5:])  /self.__stepsPerRev * self.cm_per_turn
                pos1 = float(resp1[5:])  /self.__stepsPerRev * self.cm_per_turn

                if round(pos, 2) == round(pos1, 2):
                    return pos
                else:
                    print(self.name + '-motor: Encoder position(%s) does not equal to Motor position(%s)' %(pos, pos1))
                    return round(pos,4)

            except ValueError:
                time.sleep(0.5)

                if '*' in resp:
                    retry_count += 1
                else:
                    print('Invalid response for %s : motor %s ; encoder %s' %(self.name, resp, resp1) )
                    retry_count += 1
                    if retry_count == 3:
                        print('Halt due to %s motor status = %s' %(self.name, self.motor_status))
                        break

            except KeyboardInterrupt:
                self.stop_now()
                print('\n______Halted due to Ctrl-C______')
                
            time.sleep(0.5)
            
        if retry_count >= RETRIES:
            self.stop_now()
            print('Motor timeout with ', resp, 'Motor status:',stat)

    @motor_position.setter
    def motor_position(self, pos):

        step = self.cm_to_steps(pos)

        self.send_text('DI'+str(step))
        self.send_text('FP')

#-------------------------------------------------------------------------------------------
    def turn_to(self, step):
        '''
        Turn motor by step
        '''
        self.send_text('DI'+str(step))
        self.send_text('FP')

    def current_step(self):
        '''
        Return current motor position in step
        '''
        resp = self.send_text('EP')
        return int(resp[5:])

#-------------------------------------------------------------------------------------------
    def stop_now(self):
        """ Stop motor immediatly
        """

        self.send_text('ST')

#-------------------------------------------------------------------------------------------
    @property
    def set_zero(self):
        """ Set motor position and encoder position to zero
        """
        self.send_text('EP0')  # Set encoder position to zero
        resp = self.send_text('IE')
        if int(resp[5:]) == 0:
            print (self.name + '-motor: Set encoder to zero')
            self.send_text('SP0')  # Set position to zero
            resp = self.send_text('IP')
            if int(resp[5:]) == 0:
                print (self.name + '-motor: Set current position to zero')
            else :
                print (self.name + '-motor: Fail to set current position to zero')
        else :
            print (self.name + '-motor: Fail to set encoder to zero')

#-------------------------------------------------------------------------------------------

    def set_acceleration(self, acceleration):
        self.send_text('AC'+str(acceleration))

    def set_decceleration(self, decceleration):
        self.send_text('DE'+str(decceleration))


#-------------------------------------------------------------------------------------------
    @property
    def reset_motor(self):

        self.send_text('RE',receive=False)
        print(self.name + "-motor: reset motor")

    @property
    def check_alarm(self):

        if 'A' in self.motor_status:
            resp = self.send_text('AL')
            return resp
        else:
            return False

    @property
    def clear_alarm(self):
        self.send_text('AR',receive=False)
        print(self.name + '-motor: Clear alarm on motor')

#-------------------------------------------------------------------------------------------

    def inhibit(self, inh=True):
        """
        inh = True:  Raises the disable line on the PWM controller to disable the output
         False: Lowers the inhibit line
        """
        if inh:
             cmd = 'MD'
#	 		print('inhibit ', sep='', end='', flush=True)
        else:
            cmd = 'ME'
#	 		print('enable ', sep='', end='', flush=True)

        try:
             self.send_text(cmd)  # INHIBIT or ENABLE

        except ConnectionResetError as err:
             print('*** connection to server failed: "'+err.strerror+'"')
             return False
        except ConnectionRefusedError as err:
             print('*** could not connect to server: "'+err.strerror+'"')
             return False
        except KeyboardInterrupt:
             print('\n______Halted due to Ctrl-C______')
             return False

        return True


    @property
    def enable(self):
         return self.inhibit(False)

    @property
    def disable(self):
        return self.inhibit(True)


#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================
# standalone testing:

if __name__ == '__main__':

    mc1 = Motor_Control(verbose=True, server_ip_addr="192.168.7.143")
    print(mc1.motor_status) 



