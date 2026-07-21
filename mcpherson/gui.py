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

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QStackedLayout, QVBoxLayout, QWidget)

from .acquisition import ScanConfig, run_wavelength_scan
from .controller import Spectrometer

_FONT = 'Times font'


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

        # --- Data Acquisition widgets ------------------------------------
        self.DAQLabel = QLabel("(Scan Controller page will be unavailable during data acquisition)")
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

        # --- Pages -------------------------------------------------------
        self.pageCombo = QComboBox()
        self.pageCombo.addItems(["Scan Controller", "Data Acquisition"])
        self.pageCombo.activated.connect(self.SwitchPage)
        self.stackedLayout = QStackedLayout()

        # Scan Controller page
        self.page1 = QWidget()
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
        self.stackedLayout.addWidget(self.page1)

        # Data Acquisition page
        self.page2 = QWidget()
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
        self.stackedLayout.addWidget(self.page2)

        # Top-level layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.pageCombo)
        layout.addLayout(self.stackedLayout)

        self._apply_fonts()

        self.p = None
        self.ConnectToDevice()

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
            self.ScopeIPInput, self.pageCombo,
        ]
        small = [
            self.HomeLabel1, self.HomeLabel2, self.HomeLabel3, self.HomeLabel4,
            self.DAQLabel, self.StatusLabel,
        ]
        for w in big:
            w.setFont(QFont(_FONT, 20))
        for w in small:
            w.setFont(QFont(_FONT, 14))

    # --- Device connection ----------------------------------------------
    def ConnectToDevice(self):
        """Open the serial connection, retrying every 2 s until it succeeds."""
        while True:
            try:
                self.p = Spectrometer(verbose=True)
                return
            except Exception:
                print('Serial connection failed. Trying again in 2s...')
                time.sleep(2)

    def SwitchPage(self):
        self.stackedLayout.setCurrentIndex(self.pageCombo.currentIndex())

    # --- Manual scan controls -------------------------------------------
    def ScanUp(self):
        self.SetSpeed()
        self.p.scan_up(self._dial_to_nm(self.ScanUpInput.value()))

    def ScanDown(self):
        self.SetSpeed()
        self.p.scan_down(self._dial_to_nm(self.ScanDownInput.value()))

    def ScanTo(self):
        """Move to the target wavelength via the driver's coarse-to-fine ramp.

        The motion policy lives on the controller (``Spectrometer.scan_to``);
        this slot just reads the two inputs and mirrors each speed change back
        into the speed display.
        """
        delta = self.ScanToInput.value() - self.ScanFromInput.value()
        self.p.scan_to(delta, on_speed_change=self.SetSpeedInput.setValue)

    @staticmethod
    def _dial_to_nm(value):
        """Convert a spin-box 'dial position' value to nanometers."""
        return value / 10

    def SetSpeed(self):
        self.p.set_speed(self.SetSpeedInput.value())

    def StopMotor(self):
        self.p.stop_motor()

    def FindHome(self):
        self.HomeLabel4.setText('Start homing process...')
        if self.p.homing():
            self.HomeLabel4.setText('Homing process done.')
        else:
            self.HomeLabel4.setText('Homing process failed.')

    # --- Data acquisition ------------------------------------------------
    def StartDataRun(self):
        save_path = self._ask_save_path()
        if not save_path:
            return  # user cancelled

        # Freeze the scan-controller page while acquiring. (Acquisition runs on
        # the GUI thread and blocks the event loop anyway.)
        self.page1.setEnabled(False)
        self.pageCombo.setEnabled(False)

        try:
            cfg = self._build_scan_config(save_path)
            hdf5_file = run_wavelength_scan(cfg, self.p, progress=self._on_progress)

            if os.path.isfile(hdf5_file):
                size = os.stat(hdf5_file).st_size / (1024 * 1024)
                print(f'wrote file "{hdf5_file}",  {time.ctime()}, {size:6.1f} MB     ')
                self.StatusLabel.setText('Done: %.1f MB' % size)
            else:
                print('*********** file "', hdf5_file, '" is not found - this seems bad', sep='')
                self.StatusLabel.setText('File not found!')
        except Exception as exc:
            # A missing scope or serial fault would otherwise escape to the Qt
            # event loop as an uncaught traceback; surface it in the status label.
            print(f'Data run failed: {exc}')
            self.StatusLabel.setText(f'Failed: {exc}')
        finally:
            self.page1.setEnabled(True)
            self.pageCombo.setEnabled(True)

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

    def _on_progress(self, index, total, wavelength):
        """Update the status label and keep the UI responsive during a run."""
        self.StatusLabel.setText(f'{index + 1}/{total}  ({wavelength:.3f} nm)')
        QApplication.processEvents()

    # --- Teardown --------------------------------------------------------
    def fileQuit(self):
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
