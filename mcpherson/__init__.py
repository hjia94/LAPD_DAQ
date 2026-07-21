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

The public names below are imported lazily (PEP 562) so ``import mcpherson`` and
the dependency-light :mod:`mcpherson.hdf5_writer` stay usable without pyserial
(needed only by :class:`Spectrometer`) or lab_scopes (needed only by the scan
loop) installed. Accessing a name triggers its module's import.
"""

# name -> submodule that defines it, resolved on first attribute access.
_LAZY_EXPORTS = {
    "Spectrometer": ".controller",
    "ScanConfig": ".acquisition",
    "run_wavelength_scan": ".acquisition",
    "wavelength_array": ".acquisition",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    import importlib

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals()) + __all__)
