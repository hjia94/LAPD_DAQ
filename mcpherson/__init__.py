"""McPherson spectrometer DAQ package.

Controls a McPherson 789A-4 scan controller over serial and coordinates a LeCroy
oscilloscope to record a wavelength scan into an HDF5 file. The package is split
by concern:

- :mod:`mcpherson.controller` -- serial driver for the 789A-4 (class
  :class:`Spectrometer`).
- :mod:`mcpherson.acquisition` -- the wavelength-scan loop (:class:`ScanConfig`,
  :func:`run_wavelength_scan`).
- :mod:`mcpherson.hdf5_writer` -- the only module that writes HDF5 bytes.
- :mod:`mcpherson.gui` -- the PyQt5 control/acquisition GUI.

Launch the GUI with ``python -m mcpherson``.
"""

from .controller import Spectrometer
from .acquisition import ScanConfig, run_wavelength_scan, wavelength_array

__all__ = [
    "ScanConfig",
    "Spectrometer",
    "run_wavelength_scan",
    "wavelength_array",
]
