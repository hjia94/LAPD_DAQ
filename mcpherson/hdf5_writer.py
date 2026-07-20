# -*- coding: utf-8 -*-
"""HDF5 output for a McPherson wavelength scan.

This is the ONLY module that writes HDF5 bytes for a scan. Following the pure
style of :mod:`acquisition.hdf5_writer`, callers pass plain values (numpy arrays,
strings, ints); nothing here reads a live scope, spectrometer, or GUI object, and
no file handle is held across calls.

The file layout is intentionally **unchanged** from the pre-refactor code so that
existing analysis scripts keep reading it:

    /Acquisition                                  attr: run_time
        LeCroy_scope                              attr: ScopeType
            LeCroy_scope_Setup_Array              (placeholder string)
            <trace>                               (NPos, NTimes)   one per trace
            time                                  (NTimes,)
            Headers/
                <trace>                           (NPos,)  V<WAVEDESC_SIZE>
    /Control
        Positions/positions_setup_array          structured (Line_number, Wavelength)
                                                  attr: wavelengths
    /Meta
        Python/Files/<file>                       source-file snapshots (optional)

``<trace>`` is the scope's expanded trace name (e.g. ``Channel1``). ``NPos`` is
``num_wavl * num_shots``.
"""

import os
import time

import h5py
import numpy as np

from lab_scopes.lecroy import WAVEDESC_SIZE

# gzip level 9 matches the pre-refactor McPherson output. The main DAQ uses
# blosc2/lzf, but this file's format is frozen for analysis-script compatibility.
_COMPRESSION_KWARGS = {"fletcher32": True, "compression": "gzip", "compression_opts": 9}

# Structured dtype of /Control/Positions/positions_setup_array. Big-endian to
# match the legacy file exactly.
POSITIONS_DTYPE = np.dtype([('Line_number', '>u4'), ('Wavelength', '>f4')])


def build_positions(wavelengths, num_shots):
    """Return the (Line_number, Wavelength) position array for a scan.

    One row per shot: ``num_shots`` consecutive rows share a wavelength, and
    ``Line_number`` runs 0..NPos-1. This mirrors the legacy ordering exactly.
    """
    positions = np.zeros(len(wavelengths) * num_shots, dtype=POSITIONS_DTYPE)
    i = 0
    for w in wavelengths:
        for _ in range(num_shots):
            positions[i] = (i, w)
            i += 1
    return positions


def initialize_scan_file(save_path, wavelengths, num_shots, traces, ntimes,
                         scope_type, trace_names, source_files=()):
    """Create the HDF5 file and every dataset the scan loop will fill.

    Args:
        save_path: output HDF5 path (overwritten if it exists).
        wavelengths: 1-D array of scan wavelengths (nm).
        num_shots: shots recorded per wavelength.
        traces: ordered trace keys (e.g. ``('C1', 'C2')``).
        ntimes: samples per trace (NTimes).
        scope_type: scope ``*IDN?`` string, stored as the ScopeType attr.
        trace_names: {trace: expanded_name} used for dataset names.
        source_files: iterable of paths to snapshot under /Meta/Python/Files.
    """
    positions = build_positions(wavelengths, num_shots)
    npos = len(positions)

    with h5py.File(save_path, 'w') as f:
        acq_grp = f.create_group('/Acquisition')
        acq_grp.attrs['run_time'] = time.ctime()
        scope_grp = acq_grp.create_group('LeCroy_scope')
        scope_grp.attrs['ScopeType'] = scope_type
        header_grp = scope_grp.create_group('Headers')

        ctl_grp = f.create_group('/Control')
        pos_grp = ctl_grp.create_group('Positions')

        meta_grp = f.create_group('/Meta')
        script_grp = meta_grp.create_group('Python')
        scriptfiles_grp = script_grp.create_group('Files')

        # Placeholder retained for format compatibility.
        scope_grp.create_dataset(
            'LeCroy_scope_Setup_Array',
            data=np.array('Sorry, this is not included', dtype='S'))

        pos_ds = pos_grp.create_dataset('positions_setup_array', data=positions)
        pos_ds.attrs['wavelengths'] = np.asarray(wavelengths)

        # One (NPos, NTimes) dataset + one (NPos,) header dataset per trace.
        for tr in traces:
            name = trace_names[tr]
            scope_grp.create_dataset(
                name, (npos, ntimes), chunks=(1, ntimes), **_COMPRESSION_KWARGS)
            header_grp.create_dataset(
                name, shape=(npos,), dtype="V%i" % WAVEDESC_SIZE, **_COMPRESSION_KWARGS)

        scope_grp.create_dataset('time', shape=(ntimes,), **_COMPRESSION_KWARGS)

        for fn in source_files:
            _snapshot_source_file(scriptfiles_grp, fn)


def write_position(save_path, index, traces, data, headers, trace_names, ntimes):
    """Write one scan position's trace data and headers into the file.

    Args:
        index: 0-based position (row) into the per-trace datasets.
        traces: ordered trace keys acquired at this position.
        data: {trace: 1-D numpy array of samples}.
        headers: {trace: WAVEDESC bytes}.
        trace_names: {trace: expanded_name} (dataset names).
        ntimes: NTimes; guards against the scope occasionally returning extras.
    """
    with h5py.File(save_path, 'a') as f:
        scope_grp = f['/Acquisition/LeCroy_scope']
        header_grp = scope_grp['Headers']
        for tr in traces:
            name = trace_names[tr]
            samples = np.asarray(data[tr])
            # Sometimes the scope returns 10001 samples for a 10000 setting,
            # so slice to NTimes.
            scope_grp[name][index, 0:ntimes] = samples[0:ntimes]
            header_grp[name][index] = np.void(headers[tr])


def write_time_array(save_path, time_array, ntimes):
    """Write the shared time axis (seconds) into /Acquisition/LeCroy_scope/time."""
    with h5py.File(save_path, 'a') as f:
        f['/Acquisition/LeCroy_scope']['time'][0:ntimes] = np.asarray(time_array)[0:ntimes]


def finalize(save_path, traces, trace_names, descriptions):
    """Tag each trace dataset with its description and a recorded=True flag.

    ``descriptions`` maps ``{trace: text}``. Every trace passed here was created
    and filled by the scan loop, so all are marked recorded.
    """
    with h5py.File(save_path, 'a') as f:
        scope_grp = f['/Acquisition/LeCroy_scope']
        for tr in traces:
            ds = scope_grp[trace_names[tr]]
            ds.attrs['description'] = descriptions.get(tr, '')
            ds.attrs['recorded'] = True


def _snapshot_source_file(grp, fn):
    """Store the contents of ``fn`` as a dataset under ``grp`` (best effort)."""
    try:
        with open(fn, 'r') as fh:
            contents = fh.read()
    except OSError:
        return
    fds = grp.create_dataset(os.path.basename(fn), data=contents)
    fds.attrs['filename'] = fn
    fds.attrs['modified'] = time.ctime(os.path.getmtime(fn))
