# -*- coding: utf-8 -*-
"""Read scope data out of LAPD_DAQ HDF5 archives.

LAPD_DAQ stores each channel as raw int16 samples under ``<channel>_data`` plus
the 346-byte LeCroy WAVEDESC under ``<channel>_header``. These helpers read a
shot (or many shots) and scale the counts to volts (``raw*gain - offset``) using
the WAVEDESC decoded by :mod:`scope_io.wavedesc`.

Ported from ``lab_scopes.io.hdf5`` so ``read_and_analyze`` can read archives
without ``lab_scopes`` installed; depends only on numpy and h5py.
"""

import numpy as np

from .wavedesc import LeCroyWavedesc


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install h5py to use the scope_io HDF5 readers.") from exc
    return h5py


def _decode_wavedesc(wavedesc_bytes):
    try:
        return LeCroyWavedesc(wavedesc_bytes)
    except Exception as e:
        print("Error decoding LeCroyWavedesc info:", e)
        return None


def open_hdf5_readonly(path):
    """Open an HDF5 archive read-only without contending for the file lock.

    Returns an open ``h5py.File`` the caller must close (use as a context
    manager). The point is so multiple analysis scripts -- and the live DAQ
    writer -- can touch the same file at once without blocking each other or
    raising "unable to lock file" on Windows, where the HDF5 file lock is
    mandatory.

    Strategy, most-cooperative first:
      1. ``swmr=True`` -- Single-Writer/Multiple-Reader. Readers take no lock that
         conflicts with an appending writer and see a consistent view. Requires
         the file to have been written with ``libver='latest'``.
      2. ``locking='best-effort'`` -- newer h5py/HDF5: open read-only but don't
         fail if the lock can't be taken (older files not written for SWMR).
      3. plain ``'r'`` -- last resort for ancient builds without ``locking``.

    All paths open read-only, so none can modify or corrupt the file; the
    fallbacks only affect *locking* behavior, not the bytes read.
    """
    h5py = _h5py()
    try:
        return h5py.File(path, "r", swmr=True)
    except Exception:
        pass
    try:
        return h5py.File(path, "r", locking="best-effort")
    except (TypeError, ValueError):
        # Old h5py without the `locking` kwarg.
        return h5py.File(path, "r")


def read_hdf5_scope_tarr(f, scope_name):
    """Return the time array for a scope group from an open HDF5 file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file object (not a filename).
    scope_name : str
        Name of the scope group (e.g. 'bdotscope').

    Raises
    ------
    KeyError
        If the scope group or its time array is missing.
    """
    if scope_name not in f:
        raise KeyError(f"Scope group '{scope_name}' not found in HDF5 file")
    scope_group = f[scope_name]
    if 'time_array' not in scope_group:
        raise KeyError(f"Time array not found for scope '{scope_name}'")
    return scope_group['time_array'][:]


def read_hdf5_scope_data(f, scope_name, channel_name, shot_number):
    """Read and scale one shot of one channel to volts.

    Returns ``(voltage_data, dt, t0)``.

    Raises
    ------
    KeyError
        If the group or dataset is missing.
    ValueError
        If the shot is marked skipped or the WAVEDESC cannot be decoded.
    """
    try:
        scope_group = f[scope_name]
        shot_group = scope_group[f'shot_{shot_number}']
    except KeyError as e:
        raise KeyError(f"Missing group: {e}")

    attrs = shot_group.attrs
    if attrs.get('skipped', False):
        raise ValueError(f"Shot {shot_number} was skipped. Reason: {attrs.get('skip_reason', 'Unknown reason')}")

    data_key = f'{channel_name}_data'
    try:
        raw_data = shot_group[data_key][:]
    except KeyError as e:
        raise KeyError(f"Missing dataset: {e}")

    gain, offset, dt, t0 = _scope_channel_scaling(f, scope_name, channel_name, shot_number)

    voltage_data = raw_data.astype(np.float64) * gain - offset
    return voltage_data, dt, t0


def _scope_channel_scaling(f, scope_name, channel_name, shot_number):
    """Decode ``(vertical_gain, vertical_offset, dt, t0)`` from one shot's WAVEDESC.

    These constants are identical across all shots of a channel, so a reader
    walking many shots should decode them ONCE (see
    :func:`read_hdf5_scope_channel_shots`) rather than per shot. The WAVEDESC
    bytes live under ``"<channel>_header"``.
    """
    try:
        shot_group = f[scope_name][f'shot_{shot_number}']
        wavedesc_bytes = shot_group[f'{channel_name}_header'][()]
    except KeyError as e:
        raise KeyError(f"Missing dataset: {e}")
    wavedesc = _decode_wavedesc(wavedesc_bytes)
    if wavedesc is None:
        raise ValueError(f"Could not decode WAVEDESC for {scope_name}/shot_{shot_number}/{channel_name}")
    return wavedesc.wd.vertical_gain, wavedesc.wd.vertical_offset, wavedesc.dt, wavedesc.t0


def _read_shot_raw(f, scope_name, channel_name, shot_number):
    """Return one shot's raw int16 ``_data`` array, or ``None`` if unreadable.

    Unreadable means the shot group is missing, marked ``skipped``, or has no
    ``<channel>_data`` dataset. Never raises -- a bad shot is just ``None`` so
    the caller can emit a NaN row in its place.
    """
    try:
        shot_group = f[scope_name][f'shot_{shot_number}']
    except KeyError:
        return None
    if shot_group.attrs.get('skipped', False):
        return None
    if f'{channel_name}_data' not in shot_group:
        return None
    return shot_group[f'{channel_name}_data'][:]


def read_hdf5_scope_channel_shots(f, scope_name, channel_name, shot_numbers,
                                  expected_len=None):
    """Read many shots of one channel into a ``(nshot, nsamples)`` float64 array.

    Fast path for analysis code scanning many shots of the same channel: the
    WAVEDESC scaling (gain/offset/dt/t0) is identical across a channel's shots,
    so it is decoded **once** here -- avoiding the redundant per-shot header read
    + decode that :func:`read_hdf5_scope_data` would do in a loop.

    Each shot's int16 dataset is read and scaled to volts (``raw*gain - offset``,
    float64). A shot that is missing, skipped, or -- when ``expected_len`` is
    given -- not of that length becomes a row of ``NaN`` so the returned stack
    stays rectangular and row order matches ``shot_numbers``.

    Returns
    -------
    tuple
        ``(stack, dt, t0)`` where ``stack`` is a ``(len(shot_numbers), nsamples)``
        float64 array (NaN rows for unreadable shots), or ``None`` if no shot in
        ``shot_numbers`` could be read; ``dt``/``t0`` are ``None`` when
        ``stack`` is ``None``.
    """
    shot_numbers = list(shot_numbers)

    # One pass: collect raw int16 per shot (None if unreadable) and decode the
    # channel scaling once, on the first shot that actually yields data. That
    # same shot fixes the row width when the caller gave no expected_len.
    raws = []
    gain = offset = dt = t0 = None
    nsamples = expected_len
    for s in shot_numbers:
        raw = _read_shot_raw(f, scope_name, channel_name, s)
        if raw is not None and gain is None:
            try:
                gain, offset, dt, t0 = _scope_channel_scaling(
                    f, scope_name, channel_name, s)
            except (KeyError, ValueError):
                raw = None          # header unreadable -> treat shot as a gap
            else:
                if nsamples is None:
                    nsamples = len(raw)
        raws.append(raw)

    if gain is None:                # nothing readable
        return None, None, None

    nan_row = np.full(nsamples, np.nan, dtype=np.float64)
    stack = np.vstack([
        r.astype(np.float64) * gain - offset if (r is not None and len(r) == nsamples)
        else nan_row
        for r in raws
    ])
    return stack, dt, t0
