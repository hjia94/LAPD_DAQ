"""Test doubles and sys.modules stubs for the bmotion acquisition loop tests.

Importing this module:
  1. Injects stub modules for the optional deps `bapsf_motion` and `xarray`
     (and for the relative imports `acquisition/bmotion.py` does) into
     sys.modules, so the module can be loaded on a machine that has neither
     bapsf_motion nor the hardware-only `motion` package installed.
  2. Loads `acquisition/bmotion.py` via importlib and exposes it as
     `bmotion_module`. Tests interact with it as a regular module
     (`bmotion_module._run_sequential`, etc.).
  3. Records the pre-existing sys.modules entries so install_stubs() /
     restore_modules() can roundtrip cleanly under unittest's
     setUpModule/tearDownModule pair.

Stub *classes* (motion-group / run-manager / scope test doubles, plus a
real-HDF5 temp-file factory) are defined at the bottom of this file.
Tests use `from _bmotion_stubs import StubMotionGroup, StubRunManager, ...`.
"""

import contextlib
import importlib.util
import pathlib
import sys
import tempfile
import types

import h5py
import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BMOTION_PATH = _REPO_ROOT / "acquisition" / "bmotion.py"

_STUBBED_MODULE_NAMES = (
    "bapsf_motion",
    "bapsf_motion.actors",
    "xarray",
    "acquisition",
    "acquisition.bmotion_config",
    "acquisition.config",
    "acquisition.scope_runner",
    "acquisition.bmotion",
)

# Sentinel distinct from any real module value, used to mark "this key was
# not in sys.modules before install_stubs() ran".
_ABSENT = object()

# Saved sys.modules entries so restore_modules() can put them back. Empty
# while no stubs are installed.
_SAVED_MODULES: dict = {}


# The loaded acquisition.bmotion module — populated by install_stubs().
bmotion_module = None


def _matching(prefix: str):
    """sys.modules keys equal to ``prefix`` or under ``prefix.``."""
    return [n for n in sys.modules if n == prefix or n.startswith(prefix + ".")]


@contextlib.contextmanager
def guard_sys_modules(*prefixes: str):
    """Snapshot the ``prefix``/``prefix.*`` sys.modules entries, restore on exit.

    For a test that must pop/reimport a package (e.g. an import-hygiene check):
    rebuilding a package fresh drops the submodule attributes other tests rely
    on, so this puts the originals back even if the body raises. Purging before
    the reinstate keeps a dangling submodule from leaving the package half-formed.
    """
    saved = {n: sys.modules[n] for p in prefixes for n in _matching(p)}
    try:
        yield
    finally:
        for prefix in prefixes:
            for n in _matching(prefix):
                sys.modules.pop(n, None)
        sys.modules.update(saved)


def install_stubs():
    """Stub the optional deps, load acquisition/bmotion.py once, then restore.

    Loads ``bmotion.py`` under temporary ``bapsf_motion``/``xarray``/stub-
    ``acquisition.scope_runner`` entries so it imports on a machine without that
    hardware, then immediately restores the real ``sys.modules`` (see the tail of
    this function). ``bmotion_module`` keeps the loaded module as a live
    reference. Loads exactly once -- a reload would create a *new* module object
    and desync the ``bmotion_module`` name a consuming test already imported -- so
    repeat calls (e.g. setUpModule) are no-ops.
    """
    global bmotion_module

    if bmotion_module is not None:
        return

    for name in _STUBBED_MODULE_NAMES:
        _SAVED_MODULES[name] = sys.modules.get(name, _ABSENT)

    # bapsf_motion + bapsf_motion.actors
    bmotion_stub = types.ModuleType("bapsf_motion")
    actors_stub = types.ModuleType("bapsf_motion.actors")
    actors_stub.RunManager = object
    bmotion_stub.actors = actors_stub
    sys.modules["bapsf_motion"] = bmotion_stub
    sys.modules["bapsf_motion.actors"] = actors_stub

    # xarray: point DataArray straight at our StubMotionList so the
    # isinstance(motion_list, xr.DataArray) check in get_motion_list_size holds
    # for the doubles the tests build, without StubMotionList having to subclass
    # a stub installed at import time.
    xr_stub = types.ModuleType("xarray")
    xr_stub.DataArray = StubMotionList
    sys.modules["xarray"] = xr_stub

    # acquisition package skeleton — load the two real submodules that
    # bmotion.py imports (bmotion_config, config), stub the
    # hardware-touching one (scope_runner), then load bmotion.py itself.
    acq_pkg = types.ModuleType("acquisition")
    acq_pkg.__path__ = [str(_REPO_ROOT / "acquisition")]
    sys.modules["acquisition"] = acq_pkg

    cfg_spec = importlib.util.spec_from_file_location(
        "acquisition.bmotion_config",
        _REPO_ROOT / "acquisition" / "bmotion_config.py",
    )
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    sys.modules["acquisition.bmotion_config"] = cfg_mod
    cfg_spec.loader.exec_module(cfg_mod)

    exp_spec = importlib.util.spec_from_file_location(
        "acquisition.config",
        _REPO_ROOT / "acquisition" / "config.py",
    )
    exp_mod = importlib.util.module_from_spec(exp_spec)
    sys.modules["acquisition.config"] = exp_mod
    exp_spec.loader.exec_module(exp_mod)

    sr_stub = types.ModuleType("acquisition.scope_runner")
    sr_stub.MultiScopeAcquisition = object
    # Absorb whatever args the real single_shot_acquisition takes (verbose=, etc.)
    # so the stub stays valid as that signature evolves.
    sr_stub.single_shot_acquisition = lambda *a, **k: None
    sys.modules["acquisition.scope_runner"] = sr_stub

    spec = importlib.util.spec_from_file_location(
        "acquisition.bmotion", _BMOTION_PATH,
    )
    bmotion_module = importlib.util.module_from_spec(spec)
    sys.modules["acquisition.bmotion"] = bmotion_module
    spec.loader.exec_module(bmotion_module)

    # Loading is done: bmotion_module holds its own bindings (it did
    # `from .scope_runner import single_shot_acquisition` at load, so the stub is
    # captured in its namespace, not re-read from sys.modules). Restore the real
    # sys.modules NOW -- before any sibling test module is imported under
    # `unittest discover` -- so the stub `acquisition`/`xarray` packages don't
    # leak and break their top-level `from acquisition import spool_adapter` etc.
    # The bmotion tests' own lazy `from . import spool_adapter` / `from spooling
    # import spool_format` then resolve against the real packages, which is fine.
    restore_modules()


def restore_modules():
    """Restore the sys.modules entries install_stubs() replaced.

    Called at the end of install_stubs (so stubs never outlive the bmotion load)
    and again from a consuming module's tearDownModule (idempotent). Leaves
    ``bmotion_module`` intact -- it is a live object reference, not a sys.modules
    lookup. For every name install_stubs() touched, put back exactly what was
    there before (or pop it if it was absent), so the stub ``acquisition`` /
    ``xarray`` / ``bapsf_motion`` packages don't leak into sibling test modules.
    """
    if not _SAVED_MODULES:
        return
    for name in _STUBBED_MODULE_NAMES:
        saved = _SAVED_MODULES.get(name, _ABSENT)
        if saved is _ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved
    _SAVED_MODULES.clear()


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _StubCoord:
    """Mimics the `.values` accessor on an xarray coordinate."""
    def __init__(self, values):
        self.values = np.asarray(values)


class StubMotionList:
    """Quacks like xr.DataArray for the parts of bmotion.py that touch it.

    install_stubs() points the stub ``xarray.DataArray`` at *this* class, so the
    ``isinstance(motion_list, xr.DataArray)`` check in get_motion_list_size holds
    without subclassing a stub installed at import time -- which lets the stub
    install be deferred to setUpModule (no global sys.modules mutation on import).
    """
    def __init__(self, values, space_labels=("x", "y")):
        self._values = np.asarray(values, dtype=float)
        self._coords = {"space": _StubCoord(list(space_labels))}

    @property
    def shape(self):
        return self._values.shape

    @property
    def size(self):
        return int(self._values.size)

    @property
    def values(self):
        return self._values

    @property
    def coords(self):
        return self._coords


class StubMotionBuilder:
    def __init__(self, motion_list):
        self.motion_list = motion_list


class StubPosition:
    """Supports both mg.position[i] (used in _run_*) and mg.position.value[i]
    (used in record_bmotion_positions)."""
    def __init__(self, x=0.0, y=0.0):
        self.value = (float(x), float(y))

    def __getitem__(self, i):
        return self.value[i]


class _StubDrive:
    """Absorbs drive.send_command(...) calls (e.g. the 'disable' move_to_index
    issues after a move) so the call doesn't error. No test asserts on the
    commands, so this is a no-op rather than a recorder."""
    def send_command(self, command):
        pass


class StubMotionGroup:
    def __init__(self, name, ml_values, x=0.0, y=0.0, space_labels=("x", "y")):
        self.config = {"name": name}
        self.mb = StubMotionBuilder(StubMotionList(ml_values, space_labels=space_labels))
        self.position = StubPosition(x, y)
        self.drive = _StubDrive()
        self.move_ml_calls = []

    def move_ml(self, motion_index):
        self.move_ml_calls.append(int(motion_index))
        # Reflect the new requested index in the reported position so
        # record_bmotion_positions captures a nonzero value.
        self.position = StubPosition(float(motion_index) + 1.0,
                                     float(motion_index) + 2.0)


class StubRunManager:
    def __init__(self, mgs):
        self.mgs = dict(mgs)
        self.is_moving = False
        self.terminate_called = False

    def terminate(self):
        self.terminate_called = True


class StubMSA:
    def __init__(self, scope_ips=None):
        self.scope_ips = scope_ips or {"FakeScope": "127.0.0.1"}


# --------------------------------------------------------------------------- #
# Temp-file factories shared by the loop tests
# --------------------------------------------------------------------------- #
def make_temp_hdf5_with_scopes(scope_ips):
    """Create a temp HDF5 with the per-scope top-level group structure that
    `_take_shots_at_position`'s skip path expects."""
    tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
    tmp.close()
    with h5py.File(tmp.name, "w") as f:
        for scope_name in scope_ips:
            f.create_group(scope_name)
    return tmp.name


def make_grid_motion_list(nx, ny, x0=0.0, y0=0.0, dx=1.0, dy=1.0):
    """Build an `(nx*ny, 2)` array of points on a rectangular grid.

    Stubs that feed `StubMotionGroup` need true grids now that the writer
    validates `len(unique_x) * len(unique_y) == N`.
    """
    xs = x0 + dx * np.arange(nx)
    ys = y0 + dy * np.arange(ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def make_toml_file(text="# stub bmotion toml\n"):
    tmp = tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


# Install at import (after the stub classes above exist, since install_stubs
# points xarray.DataArray at StubMotionList) so a consuming test file's
# `from _bmotion_stubs import bmotion_module` binds the loaded module, not None.
# A consuming module's setUpModule re-calls install_stubs() (idempotent) and its
# tearDownModule calls restore_modules() to put the real sys.modules back, so the
# stub `acquisition` package doesn't leak into sibling test modules.
install_stubs()
