# -*- coding: utf-8 -*-
"""Standalone readers for LAPD_DAQ scope HDF5 archives (no lab_scopes needed).

Re-exports the HDF5 reader helpers so callers can do
``from scope_io import read_hdf5_scope_data``. Depends only on numpy and h5py.
"""

from .hdf5 import (
    CHANNEL_DESCRIPTION_SUFFIX,
    channel_descriptions_from_attrs,
    open_hdf5_readonly,
    read_hdf5_scope_channel_descriptions,
    read_hdf5_scope_channel_shots,
    read_hdf5_scope_data,
    read_hdf5_scope_tarr,
    scope_shot_numbers,
)
from .wavedesc import WAVEDESC_SIZE

__all__ = [
    "CHANNEL_DESCRIPTION_SUFFIX",
    "WAVEDESC_SIZE",
    "channel_descriptions_from_attrs",
    "open_hdf5_readonly",
    "read_hdf5_scope_channel_descriptions",
    "read_hdf5_scope_channel_shots",
    "read_hdf5_scope_data",
    "read_hdf5_scope_tarr",
    "scope_shot_numbers",
]
