# -*- coding: utf-8 -*-
"""Standalone readers for LAPD_DAQ scope HDF5 archives (no lab_scopes needed).

Re-exports the HDF5 reader helpers so callers can do
``from scope_io import read_hdf5_scope_data``. Depends only on numpy and h5py.
"""

from .hdf5 import (
    open_hdf5_readonly,
    read_hdf5_scope_channel_shots,
    read_hdf5_scope_data,
    read_hdf5_scope_tarr,
)
from .wavedesc import WAVEDESC_SIZE

__all__ = [
    "WAVEDESC_SIZE",
    "open_hdf5_readonly",
    "read_hdf5_scope_channel_shots",
    "read_hdf5_scope_data",
    "read_hdf5_scope_tarr",
]
