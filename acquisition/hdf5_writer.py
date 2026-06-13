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

# Prefer Blosc2 (bitshuffle+lz4) for int16 ADC data: bitshuffle groups bits by
# significance and substantially outperforms byte-shuffle on correlated signals.
# Fall back to lzf if hdf5plugin is not installed.
try:
    import hdf5plugin as _hdf5plugin
    _COMPRESSION = _hdf5plugin.Blosc2(cname='lz4', filters=_hdf5plugin.Blosc2.BITSHUFFLE)
    _COMPRESSION_KWARGS: dict = {"compression": _COMPRESSION, "shuffle": False, "fletcher32": True}
    _COMPRESSION_LABEL = "blosc2/bitshuffle+lz4"
except ImportError:
    _hdf5plugin = None
    _COMPRESSION_KWARGS = {"compression": "lzf", "shuffle": True, "fletcher32": True}
    _COMPRESSION_LABEL = "lzf"

# Open flags for per-shot writes: newest libver (for the chunked/compressed
# datasets) and a disabled chunk cache (each shot is written once, never re-read
# in this open, so caching wastes memory). Shared so the offload adapters, which
# now open the file themselves to batch scope data + position rows in one open,
# use the exact same policy as the in-process writer.
SHOT_WRITE_OPEN_KWARGS = {"libver": "latest", "rdcc_nbytes": 0}


# Files captured into the `source_code` HDF5 attribute for reproducibility.
# Paths are resolved relative to the repository root at write time.
#
# The LeCroy scope driver now lives in the external `lab_scopes` package
# rather than this repo; its source is captured separately by
# `_lab_scopes_source()` so reproducibility is preserved.
_SOURCE_FILES = (
    'Data_Run.py',
    'Data_Run_bmotion.py',
    'Offload_Run.py',
    'acquisition/scope_runner.py',
    'acquisition/hdf5_writer.py',
    'acquisition/config.py',
    'acquisition/bmotion.py',
    'acquisition/spool_adapter.py',
    'acquisition/grid_spool_adapter.py',
    'spooling/spool_format.py',
    'offload_engine.py',
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
    contents.update(_lab_scopes_source())
    return contents


def _lab_scopes_source():
    """Capture the lab_scopes LeCroy driver source for reproducibility.

    The driver is an installed package dependency, so record both its
    version and the on-disk source of the scope module.
    """
    result = {}
    try:
        from importlib.metadata import version

        result['lab_scopes (version)'] = version('lab_scopes')
    except Exception as e:
        result['lab_scopes (version)'] = f"Error reading version: {str(e)}"
    try:
        from lab_scopes.lecroy import scope as _scope_mod

        with open(_scope_mod.__file__, 'r') as f:
            result['lab_scopes/lecroy/scope.py'] = f.read()
    except Exception as e:
        print(f"Warning: Could not read lab_scopes scope source: {str(e)}")
        result['lab_scopes/lecroy/scope.py'] = f"Error reading file: {str(e)}"
    return result


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
        if 'experiment_config' in config_group:
            raise RuntimeError(
                f"{save_path} already holds an initialized run "
                "(Configuration/experiment_config exists). Resume is not "
                "supported; delete or rotate the file to start a fresh run."
            )
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


def write_description(save_path, description):
    """(Over)write the top-level experiment ``description`` attribute.

    Idempotent: the description is written once at run start (whatever
    ``description.txt`` holds then) and overwritten at run finalize (re-read from
    ``description.txt`` after all shots are written), so the on-disk value
    reflects the file as of run completion.
    """
    with h5py.File(save_path, 'a') as f:
        f.attrs['description'] = description


# Suffix for the per-channel description attributes written on a scope group
# (e.g. ``C1_description``). Single source of truth for the layout so the
# writer, the reader, and the old-file retrofit tool agree on one convention;
# the plain ``description`` scope attr (the scope's own free-text label) is NOT
# a channel description and is excluded by :func:`channel_descriptions_from_attrs`.
CHANNEL_DESCRIPTION_SUFFIX = '_description'


def channel_descriptions_from_attrs(attrs):
    """Read ``{channel: text}`` from a scope group's ``<CH>_description`` attrs.

    Inverse of the per-channel writes in :func:`write_scope_metadata`. Skips the
    scope's own ``description`` attr (the bare suffix with no channel prefix).
    """
    return {
        name[:-len(CHANNEL_DESCRIPTION_SUFFIX)]: attrs[name]
        for name in attrs
        if name.endswith(CHANNEL_DESCRIPTION_SUFFIX) and name != 'description'
    }


def write_scope_metadata(save_path, scope_name, description, ip_address, scope_type,
                         channel_descriptions=None):
    """Write per-scope metadata attributes onto the scope group.

    ``channel_descriptions`` is a ``{trace: text}`` mapping (see
    :func:`scope_channel_descriptions`); each entry lands as a
    ``<trace>_description`` attribute on the scope group. This is the canonical
    on-disk location for channel descriptions: written once at scope init by
    the acquire process, so it exists even if the run dies on shot 1 and the
    offload never needs the config.
    """
    with h5py.File(save_path, 'a') as f:
        scope_group = f[scope_name]
        scope_group.attrs['description'] = description
        scope_group.attrs['ip_address'] = ip_address
        scope_group.attrs['scope_type'] = scope_type
        for trace, text in (channel_descriptions or {}).items():
            scope_group.attrs[f'{trace}{CHANNEL_DESCRIPTION_SUFFIX}'] = text


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


def no_description_label(channel):
    """Sentinel description for a channel with no ``[channels]`` config entry.

    Single source of truth: the in-process writer and the offload adapters
    used to build this string independently (with different texts), so the
    same undescribed channel was labeled differently depending on which path
    wrote the shot. The standalone ``lapd_daq.storage.hdf5`` writer keeps its
    own copy of this string (it cannot import this package); keep the two in
    sync if the text ever changes.
    """
    return f"Channel {channel} - No description available"


def scope_channel_descriptions(descriptions, scope_name, traces):
    """Resolve ``{trace: description}`` for one scope's displayed traces.

    ``descriptions`` maps ``"<scope>_<channel>"`` config keys to text (the raw
    ``[channels]`` section). Matching is case-insensitive: ConfigParser
    lowercases the keys (``bdotscope_c1``) while trace names come from the
    scope in uppercase (``C1``), so an exact-key lookup would miss every
    channel and silently label them all with the no-description sentinel.
    """
    chan = {key.lower(): value for key, value in descriptions.items()}
    return {
        tr: chan.get(f"{scope_name}_{tr}".lower(), no_description_label(tr))
        for tr in traces
    }


def write_shot_data(save_path, all_data, shot_num, overwrite=False):
    """Write shot_N for every scope (raw int16, blosc2/lzf-compressed, fletcher32 on).

    Args:
        save_path: HDF5 file path
        all_data: {scope_name: (traces, data, headers)} as produced by acquire_shot
        shot_num: 1-based shot number
        overwrite: when True, replace an existing shot_N group instead of
            raising. A general capability; no caller sets it on this branch.
            When False, an existing shot is treated as a programming error.

    Channel descriptions are NOT written here: they live once per scope as
    ``<trace>_description`` attributes on the scope group (see
    :func:`write_scope_metadata`), not duplicated on every shot's datasets.
    """
    with h5py.File(save_path, 'a', **SHOT_WRITE_OPEN_KWARGS) as f:
        _write_shot_data_into(f, all_data, shot_num, overwrite=overwrite)


def _write_shot_data_into(f, all_data, shot_num, overwrite=False):
    """Write shot_N groups into an already-open HDF5 file handle.

    Split out of :func:`write_shot_data` so a caller that must also write other
    per-shot data (e.g. an offload adapter writing position rows) can do both in
    a single file open instead of reopening the HDF5 for each. ``write_shot_data``
    keeps its public signature and simply opens the file and delegates here.
    """
    for scope_name, (traces, data, headers) in all_data.items():
        scope_group = f[scope_name]
        shot_name = f'shot_{shot_num}'
        if shot_name in scope_group:
            if not overwrite:
                raise RuntimeError(f"Shot {shot_num} already exists for scope {scope_name}.")
            del scope_group[shot_name]
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
                **_COMPRESSION_KWARGS,
            )
            header_ds = shot_group.create_dataset(f'{tr}_header', data=np.void(headers[tr]))
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


def mark_shot_failed_for_scopes(save_path, scope_names, shot_num, reason):
    """Replace a shot's group with a failed marker (offload quarantine path).

    Unlike :func:`mark_shot_skipped_for_scopes`, this first deletes any existing
    ``shot_N`` group, so a shot whose data was written but failed read-back
    verification leaves a clearly-marked failed group instead of silently
    keeping unverified data. ``failed=True`` distinguishes it from an
    intentionally skipped shot.
    """
    with h5py.File(save_path, 'a') as f:
        for scope_name in scope_names:
            scope_group = f[scope_name]
            shot_name = f'shot_{shot_num}'
            if shot_name in scope_group:
                del scope_group[shot_name]
            shot_group = scope_group.create_group(shot_name)
            shot_group.attrs['skipped'] = True
            shot_group.attrs['failed'] = True
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
