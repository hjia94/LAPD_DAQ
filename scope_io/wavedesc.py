# -*- coding: utf-8 -*-
"""Minimal LeCroy WAVEDESC parser for reading LAPD_DAQ HDF5 archives.

The WAVEDESC is the 346-byte binary descriptor LeCroy X-Stream scopes emit with
every trace; LAPD_DAQ stores it verbatim under ``<channel>_header`` next to the
raw int16 samples. Decoding it is the only way to recover the vertical scaling
(``gain``/``offset``) and time axis (``dt``/``t0``) needed to turn stored counts
into volts.

This is a trimmed copy of ``lab_scopes.lecroy.wavedesc.LeCroyWavedesc``, kept
here so ``read_and_analyze`` can read archives without importing ``lab_scopes``
(which is for live-scope communication). Only the scaling/time members the HDF5
readers use are retained. The full parser still lives in ``lab_scopes`` for the
live-scope and ``.trc`` paths; the HDF5 readers will be removed from there in a
follow-up.
"""

import collections
import struct

import numpy as np

# The WAVEDESC recorded for each trace: 63 fields, 346 bytes.
WAVEDESC = collections.namedtuple('WAVEDESC',
['descriptor_name', 'template_name', 'comm_type', 'comm_order',
 'wave_descriptor', 'user_text', 'res_desc1', 'trigtime_array', 'ris_time_array',
 'res_array1', 'wave_array_1', 'wave_array_2', 'res_array2', 'res_array3',
 'instrument_name', 'instrument_number', 'trace_label', 'reserved1', 'reserved2',
 'wave_array_count', 'pnts_per_screen', 'first_valid_pnt', 'last_valid_pnt',
 'first_point', 'sparsing_factor', 'segment_index', 'subarray_count', 'sweeps_per_acq',
 'points_per_pair', 'pair_offset', 'vertical_gain', 'vertical_offset', 'max_value',
 'min_value', 'nominal_bits', 'nom_subarray_count', 'horiz_interval', 'horiz_offset',
 'pixel_offset', 'vertunit', 'horunit', 'horiz_uncertainty',
 'tt_second', 'tt_minute', 'tt_hours', 'tt_days', 'tt_months', 'tt_year', 'tt_unused',
 'acq_duration', 'record_type', 'processing_done', 'reserved5', 'ris_sweeps',
 'timebase', 'vert_coupling', 'probe_att', 'fixed_vert_gain', 'bandwidth_limit',
 'vertical_vernier', 'acq_vert_offset', 'wave_source'])

WAVEDESC_SIZE = 346

# Native byte order, C alignment. To recover volts: y[i] = vertical_gain*data[i] - vertical_offset
WAVEDESC_FMT = '=16s16shhllllllllll16sl16shhlllllllllhhffffhhfdd48s48sfdBBBBhhfhhhhhhfhhffh'


class LeCroyWavedesc:
    """LeCroy X-Stream scope WAVEDESC interpretation (scaling/time subset)."""

    def __init__(self, wavedesc_bytes=b'\0' * WAVEDESC_SIZE):
        self.wd = WAVEDESC._make(struct.unpack(WAVEDESC_FMT, wavedesc_bytes))

    @property
    def num_samples(self):
        if self.wd.comm_type == 0:        # data returned as signed chars
            return self.wd.wave_array_1
        elif self.wd.comm_type == 1:      # data returned as shorts
            return int(self.wd.wave_array_1 / 2)
        raise RuntimeError(
            f'**** wd.comm_type = {self.wd.comm_type}; expected 0 or 1')

    @property
    def dt(self):
        return self.wd.horiz_interval

    @property
    def t0(self):
        return self.wd.horiz_offset

    @property
    def vertical_offset(self):
        return self.wd.vertical_offset

    @property
    def time_array(self) -> np.ndarray:
        """Return a numpy array of ``num_samples`` sample times."""
        # endpoint=False so e.g. a 2-sample, 10 ms trace lands at 0 and 5 ms,
        # not 0 and 10 ms.
        n = self.num_samples
        t0 = self.wd.horiz_offset
        return np.linspace(t0, t0 + n * self.wd.horiz_interval, n, endpoint=False)

    def generate_test_data(self, NTimes=1000):
        """Return WAVEDESC bytes for a synthetic short-format trace (for tests)."""
        self.wd = self.wd._replace(
            descriptor_name=b"WAVEDESC\0\0\0\0\0\0\0\0",
            comm_type=1,                  # data returned as shorts
            wave_array_1=2 * NTimes,
            vertunit=('\0' * 48).encode('utf8'),   # must be 48 bytes
            horunit=('\0' * 48).encode('utf8'),     # must be 48 bytes
            horiz_interval=0.001,
            horiz_offset=0.002,
            vertical_gain=0.1,
            vertical_offset=0.2)
        return struct.pack(WAVEDESC_FMT, *list(self.wd))
