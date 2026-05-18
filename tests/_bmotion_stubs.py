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


class _StubDataArray:
    """Permissive xr.DataArray stand-in; StubMotionList inherits from it so
    the `isinstance(..., xr.DataArray)` check in get_motion_list_size
    succeeds."""
    pass


# The loaded acquisition.bmotion module — populated by install_stubs().
bmotion_module = None


def install_stubs():
    """Install sys.modules stubs and load acquisition/bmotion.py.

    Idempotent: subsequent calls before restore_modules() are no-ops, so
    setUpModule can re-invoke without overwriting the original snapshot.
    """
    global bmotion_module

    if _SAVED_MODULES:
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

    # xarray
    xr_stub = types.ModuleType("xarray")
    xr_stub.DataArray = _StubDataArray
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
    sr_stub.single_shot_acquisition = lambda msa, scopes, shot_num: None
    sys.modules["acquisition.scope_runner"] = sr_stub

    spec = importlib.util.spec_from_file_location(
        "acquisition.bmotion", _BMOTION_PATH,
    )
    bmotion_module = importlib.util.module_from_spec(spec)
    sys.modules["acquisition.bmotion"] = bmotion_module
    spec.loader.exec_module(bmotion_module)


def restore_modules():
    """Undo install_stubs(): restore originals and forget the snapshot."""
    for name in _STUBBED_MODULE_NAMES:
        saved = _SAVED_MODULES.get(name, _ABSENT)
        if saved is _ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved
    _SAVED_MODULES.clear()


# Eagerly install so the StubMotionList class body below — which subclasses
# `sys.modules["xarray"].DataArray` — can resolve at import time. The
# setUpModule of a consuming test file should still call install_stubs()
# (it's idempotent and a no-op when stubs are already in place); the
# matching tearDownModule should call restore_modules().
install_stubs()


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class StubMotionList(sys.modules["xarray"].DataArray):
    """Quacks like xr.DataArray for the parts of bmotion.py that touch it."""
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    @property
    def shape(self):
        return self._values.shape

    @property
    def size(self):
        return int(self._values.size)

    @property
    def values(self):
        return self._values


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


class StubMotionGroup:
    def __init__(self, name, ml_values, x=0.0, y=0.0):
        self.config = {"name": name}
        self.mb = StubMotionBuilder(StubMotionList(ml_values))
        self.position = StubPosition(x, y)
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


def make_toml_file(text="# stub bmotion toml\n"):
    tmp = tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name
