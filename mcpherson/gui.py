# -*- coding: utf-8 -*-
"""PyQt5 control/acquisition GUI for the McPherson spectrometer.

This module is UI only. It owns the serial :class:`~mcpherson.controller.Spectrometer`
handle, collects scan parameters into a :class:`~mcpherson.acquisition.ScanConfig`,
and calls :func:`~mcpherson.acquisition.run_wavelength_scan`. It contains no HDF5
code, no scope code, and no nm->steps math -- those live in the controller,
acquisition, and writer modules.

Two pages:
- Scan Controller: manual wavelength moves, speed, stop, and homing.
- Data Acquisition: set up and run a wavelength scan.
"""

import os.path
import sys
import time

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .acquisition import ScanConfig, run_wavelength_scan
from .controller import Spectrometer

_FONT = 'Times font'


class Worker(QObject):
    """Runs one blocking hardware operation off the GUI thread.

    The GUI hands a 0-arg callable (a scan, homing, or a scan-to ramp) to the
    worker via the :attr:`submit` signal, which is delivered queued on the worker
    thread so the callable executes there -- the Qt event loop stays live and the
    window never freezes. Results and progress come back as signals the GUI
    connects to; nothing here touches widgets directly.
    """

    submit = pyqtSignal(object)               # GUI -> worker: a 0-arg callable
    started_op = pyqtSignal()                 # op began (GUI: light -> moving)
    progress = pyqtSignal(int, int, float)    # index, total, wavelength
    speed_changed = pyqtSignal(int)           # controller speed changed (steps/s)
    error = pyqtSignal(str)                   # exception text
    done = pyqtSignal(object)                 # op result (or None), on success
    finished = pyqtSignal()                   # op ended (success or failure)

    def __init__(self):
        super().__init__()
        self.submit.connect(self._run_job)

    def _run_job(self, fn):
        """Execute ``fn`` on the worker thread, reporting via signals."""
        self.started_op.emit()
        try:
            result = fn()
            self.done.emit(result)
        except Exception as exc:  # surfaced in the GUI, never crashes the thread
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class Window(QWidget):

    def __init__(self):
        super(Window, self).__init__()

        # --- Scan Controller widgets -------------------------------------
        self.ScanUpLabel = QLabel("Scan Up ")
        self.ScanUpButton = QPushButton("Go", self)
        self.ScanUpInput = QDoubleSpinBox()
        self.ScanUpInput.setRange(0, 100000)

        self.ScanDownLabel = QLabel("Scan Down ")
        self.ScanDownButton = QPushButton("Go", self)
        self.ScanDownInput = QDoubleSpinBox()
        self.ScanDownInput.setRange(0, 100000)

        self.ScanFromLabel = QLabel("Scan from ")
        self.ScanToLabel = QLabel(" to ")
        self.ScanToButton = QPushButton("Go", self)
        self.ScanFromInput = QDoubleSpinBox()
        self.ScanToInput = QDoubleSpinBox()
        self.ScanFromInput.setRange(0, 100000)
        self.ScanToInput.setRange(0, 100000)

        self.SetSpeedLabel = QLabel("Speed (steps/s) ")
        self.SetSpeedInput = QSpinBox()
        self.SetSpeedButton = QPushButton("Confirm")
        self.SetSpeedInput.setRange(36, 60000)
        self.SetSpeedInput.setValue(10000)

        self.StopMotorButton = QPushButton("Stop Motor")

        self.HomeButton = QPushButton("Find Home Switch")
        self.HomeLabel1 = QLabel("(Should be performed each time after connecting to the power)")
        self.HomeLabel2 = QLabel("(1200gr/mm) Home switch at 654.32")
        self.HomeLabel3 = QLabel("(2400gr/mm) Home switch at 654.38")
        self.HomeLabel4 = QLabel()
        self.HomeLabel4.setText('Homing status:..')

        # --- Header widgets (connect + status light) ---------------------
        self.ConnectButton = QPushButton("Connect")
        self.StatusLight = QLabel()
        self.StatusLight.setFixedSize(18, 18)
        self.StatusText = QLabel()

        # --- Data Acquisition widgets ------------------------------------
        self.DAQLabel = QLabel("(Scan Controller is disabled during a data run)")
        self.StartButton = QPushButton("Start Data Run")
        self.StartWavlLabel = QLabel("Starting wavelength:")
        self.NumWavlLabel = QLabel("Number of wavelengths:")
        self.NumShotLabel = QLabel("Number of shots:")
        self.IncrementLabel = QLabel("Increment:")
        self.SpeedLabel = QLabel("Scan speed:")
        self.ScopeIPLabel = QLabel("Scope IP address:")
        self.StatusLabel = QLabel("")

        self.StartWavlInput = QDoubleSpinBox()
        self.NumWavlInput = QSpinBox()
        self.NumShotInput = QSpinBox()
        self.IncrementInput = QDoubleSpinBox()
        self.IncrementInput.setDecimals(3)         # enables e.g. 0.025 increment
        self.SpeedInput = QSpinBox()
        self.ScopeIPInput = QLineEdit()

        self.SpeedInput.setRange(36, 60000)
        self.NumShotInput.setRange(1, 100000)
        self.NumWavlInput.setRange(1, 100000)
        self.IncrementInput.setRange(0, 100000)
        self.StartWavlInput.setRange(0, 100000)
        self.ScopeIPInput.setText('192.168.7.91')

        # --- Wire buttons ------------------------------------------------
        self.ScanUpButton.clicked.connect(self.ScanUp)
        self.ScanDownButton.clicked.connect(self.ScanDown)
        self.SetSpeedButton.clicked.connect(self.SetSpeed)
        self.StopMotorButton.clicked.connect(self.StopMotor)
        self.ScanToButton.clicked.connect(self.ScanTo)
        self.HomeButton.clicked.connect(self.FindHome)
        self.StartButton.clicked.connect(self.StartDataRun)
        self.ConnectButton.clicked.connect(self.Connect)

        # --- Worker thread -----------------------------------------------
        # Blocking hardware ops (scans, homing, scan-to ramps) run here so the
        # Qt event loop -- and thus the whole window -- never freezes.
        self._pending_op = None
        self._thread = QThread()
        self._worker = Worker()
        self._worker.moveToThread(self._thread)
        self._worker.started_op.connect(self._on_op_started)
        self._worker.progress.connect(self._on_progress)
        self._worker.speed_changed.connect(self.SetSpeedInput.setValue)
        self._worker.error.connect(self._on_op_error)
        self._worker.done.connect(self._on_op_done)
        self._worker.finished.connect(self._on_op_finished)
        self._thread.start()

        # --- Panels (side by side, single page) --------------------------
        # Scan Controller panel
        self.page1 = QGroupBox("Scan Controller")
        self.layout1 = QGridLayout()
        self.layout1.addWidget(self.SetSpeedLabel, 0, 0)
        self.layout1.addWidget(self.SetSpeedInput, 0, 1)
        self.layout1.addWidget(self.SetSpeedButton, 0, 4)
        self.layout1.addWidget(self.ScanFromLabel, 1, 0)
        self.layout1.addWidget(self.ScanFromInput, 1, 1)
        self.layout1.addWidget(self.ScanToLabel, 1, 2)
        self.layout1.addWidget(self.ScanToInput, 1, 3)
        self.layout1.addWidget(self.ScanToButton, 1, 4)
        self.layout1.addWidget(self.ScanUpLabel, 2, 0)
        self.layout1.addWidget(self.ScanUpInput, 2, 1)
        self.layout1.addWidget(self.ScanUpButton, 2, 4)
        self.layout1.addWidget(self.ScanDownLabel, 3, 0)
        self.layout1.addWidget(self.ScanDownInput, 3, 1)
        self.layout1.addWidget(self.ScanDownButton, 3, 4)
        self.layout1.addWidget(self.StopMotorButton, 4, 0, 1, 5)
        self.layout1.addWidget(self.HomeButton, 5, 0, 1, 2)
        self.layout1.addWidget(self.HomeLabel2, 6, 0, 1, 2)
        self.layout1.addWidget(self.HomeLabel3, 7, 0, 1, 2)
        self.layout1.addWidget(self.HomeLabel4, 5, 2, 1, 2)
        self.page1.setLayout(self.layout1)

        # Data Acquisition panel
        self.page2 = QGroupBox("Data Acquisition")
        self.layout2 = QGridLayout()
        self.layout2.addWidget(self.DAQLabel, 0, 0, 1, 2)
        self.layout2.addWidget(self.StartWavlLabel, 1, 0)
        self.layout2.addWidget(self.StartWavlInput, 1, 1)
        self.layout2.addWidget(self.NumWavlLabel, 2, 0)
        self.layout2.addWidget(self.NumWavlInput, 2, 1)
        self.layout2.addWidget(self.NumShotLabel, 3, 0)
        self.layout2.addWidget(self.NumShotInput, 3, 1)
        self.layout2.addWidget(self.IncrementLabel, 4, 0)
        self.layout2.addWidget(self.IncrementInput, 4, 1)
        self.layout2.addWidget(self.SpeedLabel, 5, 0)
        self.layout2.addWidget(self.SpeedInput, 5, 1)
        self.layout2.addWidget(self.ScopeIPLabel, 6, 0)
        self.layout2.addWidget(self.ScopeIPInput, 6, 1)
        self.layout2.addWidget(self.StartButton, 7, 0)
        self.layout2.addWidget(self.StatusLabel, 7, 1)
        self.page2.setLayout(self.layout2)

        # Header row: Connect button + status light, spanning both panels.
        header = QHBoxLayout()
        header.addWidget(self.ConnectButton)
        header.addWidget(self.StatusLight)
        header.addWidget(self.StatusText)
        header.addStretch(1)

        # Top-level layout: the two panels side by side in one page. Each panel
        # keeps its top-aligned grid, so unequal row counts don't stretch the
        # shorter panel's rows.
        panels = QHBoxLayout()
        panels.addWidget(self.page1)
        panels.addWidget(self.page2)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addLayout(header)
        layout.addLayout(panels)

        self._apply_fonts()

        # Start disconnected: no blocking connect at construction. The user
        # clicks Connect (see Connect()); controls stay disabled until then.
        self.p = None
        self._set_state('disconnected')

    def _apply_fonts(self):
        """Set the display font on every widget (large for controls, small for notes)."""
        big = [
            self.ScanUpLabel, self.ScanUpButton, self.ScanUpInput,
            self.ScanDownLabel, self.ScanDownButton, self.ScanDownInput,
            self.ScanFromLabel, self.ScanToLabel, self.ScanToButton,
            self.ScanFromInput, self.ScanToInput,
            self.SetSpeedLabel, self.SetSpeedInput, self.SetSpeedButton,
            self.StopMotorButton, self.HomeButton,
            self.StartButton, self.StartWavlLabel, self.NumWavlLabel,
            self.NumShotLabel, self.IncrementLabel, self.SpeedLabel,
            self.ScopeIPLabel, self.StartWavlInput, self.NumWavlInput,
            self.NumShotInput, self.IncrementInput, self.SpeedInput,
            self.ScopeIPInput, self.ConnectButton,
        ]
        small = [
            self.HomeLabel1, self.HomeLabel2, self.HomeLabel3, self.HomeLabel4,
            self.DAQLabel, self.StatusLabel, self.StatusText,
        ]
        for w in big:
            w.setFont(QFont(_FONT, 20))
        for w in small:
            w.setFont(QFont(_FONT, 14))

    # --- Device connection + status light -------------------------------
    def Connect(self):
        """Open (or reopen) the serial connection when the user clicks Connect.

        Replaces the old blocking retry loop: a single attempt that reports
        success or failure via the status light, so the event loop is never
        blocked and the window can never freeze on startup.
        """
        if self.p is not None:
            # Reconnect: drop the old handle first.
            try:
                self.p.close()
            except Exception:
                pass
            self.p = None

        try:
            self.p = Spectrometer(verbose=True)
            self._set_state('ready')
        except Exception as exc:
            self.p = None
            print(f'Serial connection failed: {exc}')
            self._set_state('disconnected')
            self.StatusText.setText(f'Disconnected -- {exc}')

    def _set_state(self, state):
        """Drive the status light and control-enabled state.

        ``state`` is one of 'ready' (green), 'moving' (yellow), or
        'disconnected' (red).
        """
        color, text = {
            'ready':        ('#2ecc40', 'Connected'),
            'moving':       ('#ffdc00', 'Moving...'),
            'disconnected': ('#ff4136', 'Disconnected'),
        }[state]
        self.StatusLight.setStyleSheet(
            f'background-color: {color}; border-radius: 9px; border: 1px solid #555;')
        self.StatusText.setText(text)
        self._apply_enabled(state)

    def _apply_enabled(self, state):
        """Enable/disable controls for the given state.

        Disconnected greys out every control so a click can never reach a
        ``None`` handle; moving leaves only Stop live; ready enables everything
        except Stop (nothing to stop).
        """
        connected = state != 'disconnected'
        moving = state == 'moving'
        controls = [
            self.ScanUpButton, self.ScanDownButton, self.ScanToButton,
            self.SetSpeedButton, self.HomeButton, self.StartButton,
        ]
        for w in controls:
            w.setEnabled(connected and not moving)
        # Stop is only meaningful while something is moving.
        self.StopMotorButton.setEnabled(moving)
        # Don't reconnect mid-move; otherwise Connect is always available.
        self.ConnectButton.setEnabled(not moving)

    # --- Manual scan controls -------------------------------------------
    # A single relative move (scan_up/down) is one non-blocking serial write, so
    # it stays on the GUI thread; only the polling ops (scan_to ramp, homing, the
    # data run) go through the worker.
    def ScanUp(self):
        self.SetSpeed()
        self.p.scan_up(self._dial_to_nm(self.ScanUpInput.value()))

    def ScanDown(self):
        self.SetSpeed()
        self.p.scan_down(self._dial_to_nm(self.ScanDownInput.value()))

    def ScanTo(self):
        """Move to the target wavelength via the driver's coarse-to-fine ramp.

        The motion policy lives on the controller (``Spectrometer.scan_to``);
        the ramp blocks, so it runs on the worker thread. Speed changes are
        mirrored back into the display via a queued signal (``setValue`` from
        the worker thread would otherwise touch a widget off the GUI thread).
        """
        delta = self.ScanToInput.value() - self.ScanFromInput.value()
        # Bind the fields the job needs to locals so the closure handed to the
        # worker doesn't capture the whole Window.
        spec, emit_speed = self.p, self._emit_speed
        self._submit(
            'scan-to',
            lambda: spec.scan_to(delta, on_speed_change=emit_speed))

    @staticmethod
    def _dial_to_nm(value):
        """Convert a spin-box 'dial position' value to nanometers."""
        return value / 10

    def SetSpeed(self):
        self.p.set_speed(self.SetSpeedInput.value())

    def StopMotor(self):
        self.p.stop_motor()

    def FindHome(self):
        """Run the (blocking) homing procedure on the worker thread."""
        self.HomeLabel4.setText('Start homing process...')
        self._submit('homing', self.p.homing)

    # --- Data acquisition ------------------------------------------------
    def StartDataRun(self):
        save_path = self._ask_save_path()
        if not save_path:
            return  # user cancelled

        cfg = self._build_scan_config(save_path)
        # The scan blocks (moves + scope averaging); run it on the worker thread
        # so the event loop stays live. Progress arrives via a queued signal, so
        # the callback here just re-emits it rather than touching widgets. Bind
        # the job's inputs to locals so the closure doesn't capture the Window.
        spec, emit_progress = self.p, self._worker.progress.emit
        self._submit(
            'data-run',
            lambda: run_wavelength_scan(cfg, spec, progress=emit_progress))

    def _build_scan_config(self, save_path):
        """Assemble a ScanConfig from the DAQ-page inputs.

        The ``/10`` dial conversion and the small-increment kluge are preserved
        from the original GUI so users' entry conventions are unchanged.
        """
        increment = self._dial_to_nm(self.IncrementInput.value())
        if increment >= 10:
            # kluge to enter very small numbers, e.g. 250 -> .0025 nm
            increment /= 10000
        return ScanConfig(
            start_nm=self._dial_to_nm(self.StartWavlInput.value()),
            num_wavl=int(self.NumWavlInput.value()),
            increment_nm=increment,
            num_shots=int(self.NumShotInput.value()),
            speed=self.SpeedInput.value(),
            scope_ip=self.ScopeIPInput.text(),
            save_path=save_path,
            save_raw=False,
        )

    def _ask_save_path(self):
        """Prompt for the output HDF5 path (empty string if cancelled)."""
        fn, _ = QFileDialog.getSaveFileName(
            self, 'Enter name of HDF5 file to write', '', 'HDF5 files (*.hdf5);;All files (*)')
        return fn

    # --- Worker plumbing -------------------------------------------------
    def _submit(self, op_name, fn):
        """Hand a blocking 0-arg callable to the worker thread.

        ``op_name`` labels the in-flight operation so the completion handlers
        can report the right status text.
        """
        self._pending_op = op_name
        self._worker.submit.emit(fn)

    def _emit_speed(self, sps):
        """``on_speed_change`` runs on the worker thread; bounce it to the GUI."""
        self._worker.speed_changed.emit(int(sps))

    def _on_op_started(self):
        """A worker op began (GUI thread): show 'moving' (yellow) light."""
        self._set_state('moving')

    def _on_progress(self, index, total, wavelength):
        """Scan progress (GUI thread, delivered queued from the worker)."""
        self.StatusLabel.setText(f'{index + 1}/{total}  ({wavelength:.3f} nm)')

    def _on_op_error(self, message):
        """A worker op raised. Surface it without crashing the thread."""
        print(f'{self._pending_op} failed: {message}')
        if self._pending_op == 'homing':
            self.HomeLabel4.setText('Homing process failed.')
        else:
            self.StatusLabel.setText(f'Failed: {message}')

    def _on_op_done(self, result):
        """A worker op completed successfully (GUI thread)."""
        if self._pending_op == 'homing':
            # homing() returns True on success, False if it could not complete.
            self.HomeLabel4.setText(
                'Homing process done.' if result else 'Homing process failed.')
        elif self._pending_op == 'data-run':
            hdf5_file = result
            if hdf5_file and os.path.isfile(hdf5_file):
                size = os.stat(hdf5_file).st_size / (1024 * 1024)
                print(f'wrote file "{hdf5_file}",  {time.ctime()}, {size:6.1f} MB     ')
                self.StatusLabel.setText('Done: %.1f MB' % size)
            else:
                print('*********** file "', hdf5_file, '" is not found - this seems bad', sep='')
                self.StatusLabel.setText('File not found!')

    def _on_op_finished(self):
        """A worker op ended (success or failure): restore the light.

        Back to 'ready' (green) if still connected, else 'disconnected' (red).
        """
        self._set_state('ready' if self.p is not None else 'disconnected')

    # --- Teardown --------------------------------------------------------
    def fileQuit(self):
        self._thread.quit()
        self._thread.wait()
        if self.p is not None:
            self.p.close()
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
