# -*- coding: utf-8 -*-
"""Serial driver for the McPherson 789A-4 scan controller.

Talks ASCII over a serial port (PySerial). This module is hardware I/O only: it
knows the command set and the nm->steps conversion, but nothing about wavelength
scans, HDF5 files, the scope, or the GUI.

Hardware documentation:
- ASCII commands: https://mcphersoninc.com/pdf/789A-4.pdf
- Manual: https://nstx.pppl.gov/nstxhome/DragNDrop/Operations/Diagnostics_&_Support_Sys/DIMS/789A3%20Manual.pdf

Author: LAPD Team
Created: 2020 (spectrometer purchase); restructured 2026.
"""

import time

import serial
import serial.tools.list_ports

# Motor geometry. Default resolution is 36000 steps/rev; the grating gives
# 4 nm per motor revolution. steps = nm / NM_PER_REV * STEPS_PER_REV.
STEPS_PER_REV = 36000
NM_PER_REV = 4

# Human-readable labels for the moving-status register ('^' reply). Any code not
# listed here is treated as "still moving" (see is_moving).
_MOVING_STATUS_LABELS = {
    0: 'Not moving',
    16: 'Slewing',
    1: 'Moving',
    2: 'Moving fast',
    43: 'Somehow moving',
}


class Spectrometer:
    """Serial interface to the McPherson 789A-4 scan controller."""

    def __init__(self, comm_port=None, timeout=2, verbose=False):
        self.sp = None
        self.verbose = verbose

        if comm_port is not None:
            self.comm_port = comm_port
        else:
            # Auto-detect: use the first available serial port.
            self.comm_port = None
            for lpi in serial.tools.list_ports.comports():  # ListPortInfo
                self.comm_port = lpi.device
                print('found', lpi.device, '   description:', lpi.description)
                break
            if self.comm_port is None:
                raise RuntimeError('No serial ports found for the scan controller')

        self.sp = serial.Serial(self.comm_port, timeout=timeout)  # open serial port

        if self.verbose:
            print('connected as "', self.sp.name, '"', sep='')

        if verbose:
            print('attempting to establish communications with scan controller')

        # "After power-up, always send an ASCII [SPACE] before any other command."
        self.id = self.send_cmd(' ')
        if len(self.id) == 0:
            self.id = '(unknown due to no response from Scan Controller)'
            if verbose:
                print('Scan Controller did not respond. Initialization already completed?')
        else:
            print('Initialized Scan Controller "', self.id, '"', sep='')

    def __repr__(self):
        return self.id

    def __str__(self):
        return self.__repr__()

    @property
    def is_open(self):
        """True if the serial port is open."""
        return self.sp is not None and self.sp.is_open

    def __bool__(self):
        return self.is_open

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def send_cmd(self, s):
        """Send an ASCII command to the scan controller and return its reply.

        A carriage return is appended to every command except a bare [SPACE]
        (per the manual). ``read_until()`` returns bytes up to and including
        '\\n'. The controller echoes the command at the start of the reply and
        terminates it with '\\r\\n'; both are stripped from the returned string.
        """
        c = s
        if c != ' ':
            c += '\r\n'
        cmd = c.encode()
        nw = self.sp.write(cmd)
        if nw != len(cmd):
            self.sp.flush()  # nominally, waits until all data is written
        if self.verbose:
            print('send_cmd("', s, '")', sep='', end='')

        c_r = self.sp.read_until().decode()
        if c_r.endswith('\r\n'):
            c_r = c_r[:-2]            # truncate cr-lf from end
        if c_r[:len(s)] == s:
            c_r = c_r[len(s):]        # delete echoed command from the beginning
        if self.verbose:
            print(' -->', c_r)
        return c_r

    def flush(self):
        if self.sp is not None:
            self.sp.flush()

    def close(self):
        if self.sp is not None:
            self.sp.flush()
            self.sp.close()
            self.sp = None

# =====================================================================

    def is_moving(self):
        """Read the moving-status register to check if the motor is still moving."""
        status = int(self.send_cmd('^'))
        if status in _MOVING_STATUS_LABELS:
            if self.verbose:
                print(_MOVING_STATUS_LABELS[status])
        else:
            print('Unknown moving status', status, '- assuming still moving')
        return status != 0

    def wait_for_motion_complete(self, delay=0.5):
        """Poll the moving status every ``delay`` seconds until the motor stops."""
        while self.is_moving():
            time.sleep(delay)
        print('Motor stopped')

# =====================================================================

    def scan_up(self, nm):
        """Scan up by ``nm`` nanometers (relative move toward longer wavelength)."""
        steps = int(nm / NM_PER_REV * STEPS_PER_REV)
        self.send_cmd('+' + str(steps))

    def scan_down(self, nm):
        """Scan down by ``nm`` nanometers (relative move toward shorter wavelength)."""
        steps = int(nm / NM_PER_REV * STEPS_PER_REV)
        self.send_cmd('-' + str(steps))

    def set_speed(self, sps):
        """Set the scan speed in steps/s (valid range 36..60000)."""
        if sps < 36:
            print('Speed should be larger than 36 sps')
            return
        if sps > 60000:
            print('Speed should be smaller than 60000 sps')
            return
        self.send_cmd('V' + str(sps))

    def scan_to(self, delta, on_speed_change=None):
        """Approach a target with a coarse-to-fine speed ramp.

        ``delta`` is the move expressed in dial-input units (tenths of nm): each
        ``scan_up``/``scan_down`` here divides by 10 to get nm. Larger moves run
        fast and drop to slower speeds as the remaining distance shrinks, so the
        final approach is precise. ``on_speed_change(sps)`` is an optional
        callback invoked whenever the speed changes, letting a GUI mirror the
        value in its display. This is motion policy composed from the serial
        primitives; it lives on the driver so non-GUI callers can reuse it.
        """
        def _speed(sps):
            self.set_speed(sps)
            if on_speed_change is not None:
                on_speed_change(sps)

        if delta <= 0:  # 0 causes a mechanical reset to the current position
            _speed(40000)
            self.scan_down((600 - delta) / 10)
            self.wait_for_motion_complete(0.2)
            delta = 600

        if delta >= 600:
            _speed(40000)
            self.scan_up((delta - 50) / 10)
            self.wait_for_motion_complete(0.2)
            delta = 50

        if delta >= 50:
            _speed(10000)
            self.scan_up((delta - 20) / 10)
            self.wait_for_motion_complete(0.2)
            delta = 20

        if delta >= 20:
            _speed(2500)
            self.scan_up((delta - 10) / 10)
            self.wait_for_motion_complete(0.2)
            delta = 10

        # now delta should be 10 or less
        _speed(2500)
        while delta > 2:
            self.scan_up(1 / 10)
            self.wait_for_motion_complete(0.1)
            delta -= 1

        while delta > 0.1:
            self.scan_up(1 / 100)
            self.wait_for_motion_complete(0.1)
            delta -= 0.1

        self.scan_up(delta / 10)

    def stop_motor(self):
        self.send_cmd('@')

# =====================================================================

    def _home_seek(self, move_cmd, target_status):
        """Start a home-switch move and poll ``]`` until it reaches target_status.

        Returns True when the target status is seen, False if interrupted.
        """
        self.send_cmd(move_cmd)
        while True:
            try:
                resp = self.send_cmd(']')
                if self.verbose:
                    print('status is', resp, ', continue scanning...')
                if int(resp) == target_status:
                    if self.verbose:
                        print('switch reached target status, moving to the next step.')
                    self.stop_motor()
                    return True
                time.sleep(0.8)
            except KeyboardInterrupt:
                self.stop_motor()
                print('Motor stopped by keyboard interruption. Homing procedure aborted.')
                return False

    def homing(self):
        """Run the home-switch procedure.

        Homing should be done prior to scanning and every time power is
        disconnected. See the instruction manual for the full procedure.

        Returns True on success, False if it could not be completed.
        """
        self.send_cmd('A8')  # Enable home circuit

        status = self.send_cmd(']')  # Check home switch and try to move
        if self.verbose:
            print('The initial home switch status is', status)

        if int(status) == 32:
            # Current wavelength is below the home wavelength: move up until clear.
            if not self._home_seek('M+23000', target_status=2):
                return False
        elif int(status) == 0:
            # Current wavelength is above the home wavelength (Home Switch LED
            # off): move down until the switch is blocked.
            if not self._home_seek('M-23000', target_status=34):
                return False
        else:
            print('The starting status is:', status,
                  ' Cannot perform home switching. Please try again.')
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
    """Standalone connection smoke test."""
    p = Spectrometer(verbose=True)
    print('done')
