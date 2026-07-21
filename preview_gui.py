"""Show the McPherson GUI window WITHOUT any hardware or pyserial.

Run at the end of each reshape phase to eyeball the interface:

    .venv/Scripts/python.exe preview_gui.py

It stubs `mcpherson.controller` and `mcpherson.acquisition` in sys.modules
before importing the GUI, so no pyserial / lab_scopes import runs and nothing
touches a real device. `ConnectToDevice` is neutered so the window just shows;
`self.p` stays None, so do not click the move/scan/home/start controls here.

This file is a temporary dev helper -- not part of the package.
"""

import sys
import types

# --- Stub the hardware modules so gui.py imports UI-only -------------------
_ctrl = types.ModuleType('mcpherson.controller')


class Spectrometer:  # placeholder; the GUI starts with p=None anyway
    pass


_ctrl.Spectrometer = Spectrometer
sys.modules['mcpherson.controller'] = _ctrl

_acq = types.ModuleType('mcpherson.acquisition')


class ScanConfig:  # placeholder dataclass stand-in
    pass


def run_wavelength_scan(*args, **kwargs):  # never called in preview
    pass


_acq.ScanConfig = ScanConfig
_acq.run_wavelength_scan = run_wavelength_scan
sys.modules['mcpherson.acquisition'] = _acq

# --- Show the window -------------------------------------------------------
import mcpherson.gui as g  # noqa: E402

# No hardware attached: neuter Connect so clicking it can't open a real port.
# The window starts in the 'disconnected' state either way (p stays None).
g.Window.Connect = lambda self: None

from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
w = g.Window()
w.setWindowTitle('McPherson GUI - preview (no hardware)')
w.show()
print('window visible:', w.isVisible(),
      '| size:', w.size().width(), 'x', w.size().height(), flush=True)
app.exec_()
