"""Experiment-config parsing for the multi-scope acquisition pipeline.

`load_experiment_config` reads `experiment_config.ini` once, returning both a
`ConfigParser` for structured access and the raw text for verbatim storage in
the resulting HDF5 file.
"""

import configparser
import os


DESCRIPTION_FILENAME = "description.txt"

# Written into the HDF5 ``description`` attribute when no usable description.txt
# is found. Kept as a recognizable sentinel so a downstream reader (or the user)
# can tell the prose was never filled in, rather than silently empty.
DESCRIPTION_PLACEHOLDER = (
    "No experiment description provided "
    "(description.txt not found in base_path)"
)


def read_description_file(description_path):
    """Read the free-text experiment description from ``description_path``.

    ``description_path`` is the full path to the ``description.txt`` file (see
    :func:`resolve_description_path`). Mirrors the tolerant style of
    :func:`load_experiment_config`: a missing/empty/unreadable file never raises;
    it returns :data:`DESCRIPTION_PLACEHOLDER` and prints a warning instead, so a
    description problem can never abort an otherwise-good acquisition.
    """
    try:
        with open(description_path, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Warning: description file not found: {description_path}. "
              "Using placeholder description.")
        return DESCRIPTION_PLACEHOLDER
    except Exception as e:
        print(f"Warning: could not read description file {description_path}: {e}. "
              "Using placeholder description.")
        return DESCRIPTION_PLACEHOLDER

    if not text.strip():
        return DESCRIPTION_PLACEHOLDER
    return text


def resolve_description_path(base_path):
    """Return the full path to ``description.txt`` inside ``base_path``."""
    return os.path.abspath(os.path.join(base_path, DESCRIPTION_FILENAME))


def resolve_description_path_from_config(config_path):
    """Return the ``description.txt`` path that sits next to ``config_path``.

    ``description.txt`` lives in the run-inputs directory alongside
    ``experiment_config.ini``; the acquisition entry points derive it from the
    config path so the spooled offload (a separate process) and the acquire
    process agree on one absolute path.
    """
    return resolve_description_path(os.path.dirname(os.path.abspath(config_path)))


def load_experiment_config(config_path='experiment_config.ini'):
    """Load experiment configuration from config file.

    Returns:
        tuple: (config, raw_config_text)
            - config: ConfigParser object with parsed configuration
            - raw_config_text: Raw text content of the configuration file
    """
    # Strip inline comments ("# ..." / "; ..." after a value) so a stray comment
    # on a value line cannot corrupt an IP or a [bmotion] token. Matches
    # lapd_daq.config.load_run_config.
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))

    # Read the raw config text
    raw_config_text = ""
    try:
        with open(config_path, 'r') as f:
            raw_config_text = f.read()
    except Exception as e:
        print(f"Warning: Could not read raw config file: {e}")

    # Parse the config
    config.read(config_path)

    # Set defaults if sections don't exist
    if 'experiment' not in config:
        config.add_section('experiment')
    if 'scopes' not in config:
        config.add_section('scopes')
    if 'channels' not in config:
        config.add_section('channels')

    # NOTE: the run description is NOT read from the config any more. It lives in
    # description.txt (see read_description_file / resolve_description_path); any
    # [experiment] description key in the config is ignored.

    return config, raw_config_text


def get_experiment_name(config):
    """Return the experiment name from [experiment] name (or exp_name).

    The Data_Run entry scripts now take only a base path; the experiment name
    lives in the config file and the HDF5 filename is derived from it after
    parsing. Falls back to 'experiment' if unset.
    """
    name = (config.get('experiment', 'name', fallback=None)
            or config.get('experiment', 'exp_name', fallback=None))
    return (name or 'experiment').strip()


def hdf5_filename(exp_name, date=None):
    """Build the standard HDF5 filename ``<exp_name>_<YYYY-MM-DD>.hdf5``.

    Centralized so every entry point (acquisition and offload) derives the
    exact same name from the same config.
    """
    import datetime as _datetime

    if date is None:
        date = _datetime.date.today()
    return f"{exp_name}_{date}.hdf5"


def get_storage_paths(config):
    """Return parallel-mode storage directories from the optional [storage] section.

    The two-process (spool + offload) pipeline writes shots to a fast local
    ``spool_dir`` and offloads them into an HDF5 file under ``hdf5_dir`` on a
    slower/larger disk. Both values are *directories*; the HDF5 filename is
    derived from the experiment name (see :func:`resolve_hdf5_path`).

    ``hdf5_path`` is accepted as a backward-compatible alias for ``hdf5_dir``
    and is likewise treated as a directory. Returns ``(spool_dir, hdf5_dir)``
    with either possibly ``None`` when not configured, so callers can fall back
    to the legacy single-process path.
    """
    if 'storage' not in config:
        return None, None
    spool_dir = config.get('storage', 'spool_dir', fallback=None) or None
    hdf5_dir = (config.get('storage', 'hdf5_dir', fallback=None)
                or config.get('storage', 'hdf5_path', fallback=None) or None)
    return spool_dir, hdf5_dir


def get_motion_recovery_opts(config):
    """Return motor-move recovery tunables from the optional ``[bmotion]`` keys.

    Used by the bmotion acquisition loop to wait for arrival, verify, and retry a
    failed/silently-missed motor move once before skipping the position. Keys
    (all optional, with defaults so existing configs need no change):

      * ``move_attempts``        - total move attempts incl. the first (default 30).
                                   Each re-issue re-sends the same absolute target;
                                   the library re-clears alarms / re-enables on the
                                   re-issue, so repeated tries give a stuck axis
                                   many chances to recover before the position is
                                   skipped.
      * ``move_retry_wait_s``    - seconds to wait between attempts before
                                   soft-stopping and re-issuing (default 1).
      * ``move_stall_timeout_s`` - seconds a move may make NO position progress
                                   before it is treated as a real stall (default
                                   10). A move that keeps advancing is never timed
                                   out, so a slow-but-healthy long move is left to
                                   finish instead of being killed mid-travel.
      * ``move_max_time_s``      - absolute backstop ceiling for a single move, so
                                   a hung link can't wait forever (default 300).
      * ``position_progress_eps``- min per-axis position change (motion-space
                                   units) that counts as "still progressing"
                                   (default: ``position_tol / 4``).
      * ``position_tol``         - max |achieved - target| per axis, in motion-
                                   space units, to count a move as arrived (0.5)
      * ``encoder_mismatch_tol_rev`` - max |encoder - step| disagreement (in motor
                                   revolutions) before warning that the encoder
                                   and commanded position have drifted apart
                                   (default 0.01). Read-only check; never corrects.

    The legacy ``move_settle_timeout_s`` key is still accepted: if present and
    ``move_max_time_s`` is absent, it is used as the backstop ceiling so existing
    configs don't silently shorten the new ceiling.

    Returns a dict consumed directly as ``move_with_recovery`` keyword args.
    """
    section = 'bmotion'
    if section not in config:
        tol = 0.5
        return {"attempts": 30, "retry_wait": 1.0, "stall_timeout": 10.0,
                "max_move_time": 300.0, "progress_eps": tol / 4, "tol": tol,
                "encoder_mismatch_tol_rev": 0.01}

    tol = config.getfloat(section, 'position_tol', fallback=0.5)

    # Backstop ceiling: prefer the new key, fall back to the legacy settle key
    # (so old configs keep their intent), else the new generous default.
    if config.has_option(section, 'move_max_time_s'):
        max_move_time = config.getfloat(section, 'move_max_time_s')
    elif config.has_option(section, 'move_settle_timeout_s'):
        max_move_time = config.getfloat(section, 'move_settle_timeout_s')
    else:
        max_move_time = 300.0

    return {
        "attempts": config.getint(section, 'move_attempts', fallback=30),
        "retry_wait": config.getfloat(section, 'move_retry_wait_s', fallback=1.0),
        "stall_timeout": config.getfloat(section, 'move_stall_timeout_s', fallback=10.0),
        "max_move_time": max_move_time,
        "progress_eps": config.getfloat(section, 'position_progress_eps', fallback=tol / 4),
        "tol": tol,
        "encoder_mismatch_tol_rev": config.getfloat(
            section, 'encoder_mismatch_tol_rev', fallback=0.01),
    }


def get_backpressure_limits(config):
    """Return ``(max_pending_shots, min_free_gb)`` for spool backpressure.

    Acquisition pauses before a shot when the spool has more than
    ``max_pending_shots`` undrained shots OR less than ``min_free_gb`` free on
    the spool disk, so a slow/stalled offload can't silently fill the disk.
    Both come from the optional ``[storage]`` keys ``max_pending_shots`` /
    ``min_free_gb``; either ``<= 0`` disables that check. Defaults are generous
    (1000 shots, 5 GB) so a healthy run never notices them.
    """
    if 'storage' not in config:
        return 1000, 5.0
    max_pending = config.getint('storage', 'max_pending_shots', fallback=1000)
    min_free_gb = config.getfloat('storage', 'min_free_gb', fallback=5.0)
    return max_pending, min_free_gb


def get_auto_plot_enabled(config):
    """Return whether to auto-plot the line profile after a run finishes.

    Reads the optional ``[analysis] auto_plot`` boolean key; defaults to ``True``
    (enabled) when the section or key is absent, so a line run plots itself
    without extra config. The plotter no-ops on non-line runs, so leaving this on
    is harmless for plane / single-point runs.
    """
    if 'analysis' not in config:
        return True
    return config.getboolean('analysis', 'auto_plot', fallback=True)


def resolve_hdf5_path(config, base_path, date=None):
    """Return the full HDF5 file path for a run.

    Detection keys on the experiment ``name``: an existing ``<name>_*.hdf5`` from
    any date is reused (so a run started before midnight and resumed the next day
    targets the same file), else a fresh ``<name>_<today>.hdf5`` is minted. Both
    the acquisition and offload processes call this so they target the same file.

    ``date`` is accepted for backward compatibility; when given it forces that
    exact dated filename (bypassing detection), which a few callers rely on.
    """
    import os as _os

    _spool_dir, hdf5_dir = get_storage_paths(config)
    out_dir = hdf5_dir or base_path
    if date is not None:
        return _os.path.join(out_dir, hdf5_filename(get_experiment_name(config), date))

    # Delegate to the name-glob resolver (local import avoids a config<->run_paths
    # import cycle: run_paths imports helpers from this module).
    from .run_paths import resolve_run_paths
    return resolve_run_paths(config, base_path, spool_root=None).hdf5_path
