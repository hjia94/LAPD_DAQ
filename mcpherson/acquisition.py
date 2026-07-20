# -*- coding: utf-8 -*-
"""Wavelength-scan orchestration for the McPherson spectrometer.

This module owns the scope handle for the duration of a run and drives the
spectrometer (moving) and scope (acquiring), delegating every HDF5 byte to
:mod:`mcpherson.hdf5_writer`. It does not open file dialogs, import Qt, or own
the serial-port lifecycle -- the caller supplies a connected
:class:`~mcpherson.controller.Spectrometer` and a destination path.

Data flow:  GUI params -> ScanConfig -> run_wavelength_scan -> hdf5_writer
"""

import sys
import time
from dataclasses import dataclass

import numpy as np

from lab_scopes.lecroy import LeCroy_Scope

from . import hdf5_writer


@dataclass
class ScanConfig:
    """Parameters for one wavelength scan.

    Attributes:
        start_nm: starting wavelength (nm).
        num_wavl: number of wavelength points.
        increment_nm: wavelength step between points (nm).
        num_shots: shots recorded per wavelength.
        speed: scan speed in steps/s (None leaves the controller speed as-is).
        scope_ip: LeCroy scope IP address.
        save_path: destination HDF5 path.
        save_raw: store raw ADC counts (True) vs. scaled volts (False).
    """
    start_nm: float
    num_wavl: int
    increment_nm: float
    num_shots: int = 1
    speed: int = None
    scope_ip: str = ''
    save_path: str = ''
    save_raw: bool = False


def wavelength_array(cfg):
    """Return the 1-D array of scan wavelengths (nm) for ``cfg``."""
    num_wavl = int(cfg.num_wavl)
    return np.linspace(
        cfg.start_nm,
        cfg.start_nm + cfg.increment_nm * (num_wavl - 1),
        num_wavl,
    )


def acquire_traces(scope, save_raw, aux_text='', timeout=2000):
    """Wait for averaging to finish, then read every displayed trace.

    Returns ``(traces, data, headers)``:
        traces: ordered tuple of trace keys currently displayed.
        data: {trace: 1-D numpy array of samples}.
        headers: {trace: WAVEDESC bytes}.
    """
    nsweeps, _ = scope.max_averaging_count()
    print(f'    Waiting for {nsweeps} sweeps to complete...')

    timed_out, n = scope.wait_for_max_sweeps(aux_text, timeout)
    if timed_out:
        print(f'    WARNING: Averaging timed out - got {n}/{nsweeps} sweeps after {timeout} s')
    else:
        print(f'    Averaging complete: {n}/{nsweeps} sweeps acquired')

    traces = scope.displayed_traces()
    print(f'    Reading data from {len(traces)} traces...')

    data = {}
    headers = {}
    for tr in traces:
        samples, header_bytes = scope.acquire(tr, raw=save_raw)
        data[tr] = samples
        headers[tr] = header_bytes
        print(f'      Trace {tr}: {len(samples)} samples read')
    return traces, data, headers


def run_wavelength_scan(cfg, spec, progress=None, descriptions=None):
    """Run a wavelength scan and write the HDF5 file described by ``cfg``.

    Args:
        cfg: the :class:`ScanConfig` for this run.
        spec: a connected :class:`~mcpherson.controller.Spectrometer`. The caller
            owns it; this function never closes the serial port.
        progress: optional callback ``(index, total, wavelength)`` invoked after
            each position completes. Lets a GUI update without acquisition
            importing Qt.
        descriptions: optional {trace: text} channel descriptions for the
            recorded datasets.

    Returns the written HDF5 path (``cfg.save_path``).
    """
    descriptions = descriptions or {}
    wavelengths = wavelength_array(cfg)
    npos = len(wavelengths) * cfg.num_shots

    if cfg.speed is not None:
        spec.set_speed(cfg.speed)

    print(f'Connecting to scope at {cfg.scope_ip}...')
    with LeCroy_Scope(cfg.scope_ip, verbose=False) as scope:
        if not scope:
            raise RuntimeError('Scope not found at ' + cfg.scope_ip)

        print(f'Successfully connected to scope: {scope.idn_string}')

        traces = scope.displayed_traces()
        ntimes = scope.max_samples()
        trace_names = {tr: scope.expanded_name(tr) for tr in traces}
        print(f'DAQ Configuration: {npos} positions, {ntimes} samples per trace')
        print(f'Found {len(traces)} displayed traces: {traces}')

        source_files = [p for p in (sys.argv[0], __file__, hdf5_writer.__file__) if p]
        hdf5_writer.initialize_scan_file(
            cfg.save_path, wavelengths, cfg.num_shots, traces, ntimes,
            scope.idn_string, trace_names, source_files=source_files)

        time_written = False
        try:
            print(f'Starting acquisition at {time.ctime()}')
            print('=' * 60)

            index = 0
            for wi, wavl in enumerate(wavelengths):
                for _ in range(cfg.num_shots):
                    print(f'Position {index + 1}/{npos}: Wavelength = {wavl:.3f} nm')

                    # Move to this wavelength once per new wavelength; the first
                    # shot of the first wavelength stays put (start position).
                    if index == 0:
                        print(' McPherson : At start wavelength (no movement)')
                        time.sleep(0.5)
                    elif index % cfg.num_shots == 0:
                        try:
                            print(f' McPherson : Moving by {cfg.increment_nm:.3f} nm...')
                            spec.scan_up(cfg.increment_nm)
                            time.sleep(0.2)
                            spec.wait_for_motion_complete()
                            print(' McPherson : Movement complete')
                        except KeyboardInterrupt:
                            raise
                        except Exception:
                            print(f'McPherson : FAILED to move to wavelength {wavl}')

                    print('  Scope: Starting data acquisition...')
                    traces, data, headers = acquire_traces(
                        scope, cfg.save_raw, aux_text=str(index) + ': ')
                    hdf5_writer.write_position(
                        cfg.save_path, index, traces, data, headers, trace_names, ntimes)
                    print('  Scope: Data acquisition complete')

                    if not time_written:
                        hdf5_writer.write_time_array(
                            cfg.save_path, scope.time_array(traces[0]), ntimes)
                        time_written = True

                    if progress is not None:
                        progress(index, npos, wavl)

                    percent = ((index + 1) / npos) * 100
                    print(f'  Position complete ({percent:.1f}% total progress)\n')
                    index += 1

            print('=' * 60)
            print(f'Acquisition completed successfully at {time.ctime()}')

        except KeyboardInterrupt:
            print('\n' + '=' * 60)
            print(f'______Halted due to Ctrl-C______ at {time.ctime()}')

        # Ensure a time array exists even if the run was interrupted before the
        # first successful position.
        if not time_written:
            hdf5_writer.write_time_array(cfg.save_path, scope.time_array(traces[0]), ntimes)

        hdf5_writer.finalize(cfg.save_path, traces, trace_names, descriptions)

    print(f'HDF5 file closed: {cfg.save_path}')
    return cfg.save_path
