"""HDF5 layout for multi-scope acquisition.

Every byte that lands in the output HDF5 file is written from this module.
Callers pass plain values; nothing here reads from `MultiScopeAcquisition`
or any other live object. That keeps the file structure (group names,
dtypes, chunking, compression) easy to audit in one place.
"""

import io
import os
import time

import h5py
import numpy as np


# Files captured into the `source_code` HDF5 attribute for reproducibility.
# Paths are resolved relative to the repository root at write time.
_SOURCE_FILES = (
    'Data_Run.py',
    'acquisition/scope_runner.py',
    'acquisition/hdf5_writer.py',
    'acquisition/config.py',
    'drivers/LeCroy_Scope.py',
)


def _repo_root():
    """Return the absolute path of the LAPD_DAQ repo root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_source_files():
    """Read the contents of the Python scripts used to create an HDF5 file.

    Returns a {relative_path: file_contents_or_error_message} dict suitable
    for stringifying into the top-level `source_code` attribute.
    """
    contents = {}
    root = _repo_root()
    for relpath in _SOURCE_FILES:
        abs_path = os.path.join(root, relpath)
        try:
            with open(abs_path, 'r') as f:
                contents[relpath] = f.read()
        except Exception as e:
            print(f"Warning: Could not read {relpath}: {str(e)}")
            contents[relpath] = f"Error reading file: {str(e)}"
    return contents


def write_experiment_metadata(save_path, description, source_code,
                              raw_config_text, config, scope_names):
    """Initialize the top-level HDF5 structure: experiment attrs, the
    Configuration group, and one empty group per scope.
    """
    with h5py.File(save_path, 'a') as f:
        f.attrs['description'] = description
        f.attrs['creation_time'] = time.ctime()
        f.attrs['source_code'] = str(source_code)

        config_group = f.require_group('Configuration')
        config_group.create_dataset(
            'experiment_config',
            data=np.bytes_(_serialize_config(raw_config_text, config)),
        )

        for scope_name in scope_names:
            if scope_name not in f:
                f.create_group(scope_name)


def _serialize_config(raw_config_text, config):
    """Prefer the verbatim file contents; fall back to ConfigParser.write."""
    if raw_config_text:
        print("Stored full configuration file content from memory")
        return raw_config_text
    try:
        buf = io.StringIO()
        config.write(buf)
        print("Stored configuration using ConfigParser's write method")
        return buf.getvalue()
    except Exception as e:
        print(f"Could not convert config to string: {str(e)}")
        return f"Error saving configuration: {str(e)}"


def write_scope_metadata(save_path, scope_name, description, ip_address, scope_type):
    """Write per-scope metadata attributes onto the scope group."""
    with h5py.File(save_path, 'a') as f:
        scope_group = f[scope_name]
        scope_group.attrs['description'] = description
        scope_group.attrs['ip_address'] = ip_address
        scope_group.attrs['scope_type'] = scope_type


def write_time_array(save_path, scope_name, time_array, is_sequence):
    """Write the time_array dataset for a scope (one per scope, written once)."""
    with h5py.File(save_path, 'a') as f:
        scope_group = f[scope_name]

        if 'time_array' in scope_group:
            raise RuntimeError(
                f"Time array already exists for scope {scope_name}. "
                "This should not happen."
            )

        time_ds = scope_group.create_dataset('time_array', data=time_array, dtype='float64')
        time_ds.attrs['units'] = 'seconds'
        if is_sequence == 1:
            time_ds.attrs['description'] = 'Time array for all channels; data saved in sequence mode'
        else:
            time_ds.attrs['description'] = 'Time array for all channels'
        time_ds.attrs['dtype'] = str(time_array.dtype)


def write_shot_data(save_path, all_data, shot_num, channel_descriptions):
    """Write shot_N for every scope (raw int16, lzf-compressed, fletcher32 on).

    Args:
        save_path: HDF5 file path
        all_data: {scope_name: (traces, data, headers)} as produced by acquire_shot
        shot_num: 1-based shot number
        channel_descriptions: {(scope_name, trace): description_string}
    """
    with h5py.File(save_path, 'a', libver='latest', rdcc_nbytes=0) as f:
        for scope_name, (traces, data, headers) in all_data.items():
            scope_group = f[scope_name]
            shot_name = f'shot_{shot_num}'
            if shot_name in scope_group:
                raise RuntimeError(f"Shot {shot_num} already exists for scope {scope_name}.")
            shot_group = scope_group.create_group(shot_name)
            shot_group.attrs['acquisition_time'] = time.ctime()

            for tr in traces:
                if tr not in data:
                    continue
                trace_data = np.asarray(data[tr], dtype=np.int16)
                is_sequence = len(trace_data.shape) > 1
                if is_sequence:
                    chunk_size = (1, min(trace_data.shape[1], 8 * 1024 * 1024))
                else:
                    chunk_size = (min(len(trace_data), 8 * 1024 * 1024),)

                data_ds = shot_group.create_dataset(
                    f'{tr}_data',
                    data=trace_data,
                    dtype='int16',
                    chunks=chunk_size,
                    compression='lzf',
                    shuffle=True,
                    fletcher32=True,
                )
                header_ds = shot_group.create_dataset(f'{tr}_header', data=np.void(headers[tr]))
                data_ds.attrs['description'] = channel_descriptions.get(
                    (scope_name, tr), f"Channel {tr} - No description available"
                )
                data_ds.attrs['dtype'] = 'int16'
                header_ds.attrs['description'] = f'Binary header data for {tr}'


def mark_shot_skipped_for_scopes(save_path, scope_names, shot_num, reason):
    """Record a skipped shot under each scope group with a human-readable reason."""
    with h5py.File(save_path, 'a') as f:
        for scope_name in scope_names:
            scope_group = f[scope_name]
            shot_group = scope_group.create_group(f'shot_{shot_num}')
            shot_group.attrs['skipped'] = True
            shot_group.attrs['skip_reason'] = str(reason)
            shot_group.attrs['acquisition_time'] = time.ctime()


def mark_shot_skipped_for_probes(save_path, probe_names, shot_num, reason):
    """Record a skipped shot under /Control/Positions/{probe} for each probe."""
    with h5py.File(save_path, 'a') as f:
        for probe in probe_names:
            probe_group = f[f'/Control/Positions/{probe}']
            shot_group = probe_group.create_group(f'shot_{shot_num}')
            shot_group.attrs['skipped'] = True
            shot_group.attrs['skip_reason'] = str(reason)
            shot_group.attrs['acquisition_time'] = time.ctime()


def record_shot_count(save_path, scope_names, shot_count):
    """Write the final shot_count attribute on every scope group.

    Stored as an attribute so consumers can index without filtering keys.
    """
    print(f"Storing shot count ({shot_count}) to HDF5 file...")
    try:
        with h5py.File(save_path, 'a') as f:
            for scope_name in scope_names:
                if scope_name in f:
                    scope_group = f[scope_name]
                    scope_group.attrs['shot_count'] = shot_count
                    print(f"  - {scope_name}: {shot_count} shots recorded")
    except Exception as e:
        print(f"Error storing shot count: {e}")
