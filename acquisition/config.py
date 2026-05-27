"""Experiment-config parsing for the multi-scope acquisition pipeline.

`load_experiment_config` reads `experiment_config.ini` once, returning both a
`ConfigParser` for structured access and the raw text for verbatim storage in
the resulting HDF5 file.
"""

import configparser


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

    # Set default values if not present
    if not config.get('experiment', 'description', fallback=None):
        config.set('experiment', 'description', 'No experiment description provided')

    return config, raw_config_text


def get_storage_paths(config):
    """Return parallel-mode storage paths from the optional [storage] section.

    The two-process (spool + offload) pipeline writes shots to a fast local
    ``spool_dir`` and offloads them into ``hdf5_path`` on a slower/larger disk.
    Returns ``(spool_dir, hdf5_path)`` with either value possibly ``None`` when
    not configured, so callers can fall back to the legacy single-process path.
    """
    if 'storage' not in config:
        return None, None
    spool_dir = config.get('storage', 'spool_dir', fallback=None) or None
    hdf5_path = config.get('storage', 'hdf5_path', fallback=None) or None
    return spool_dir, hdf5_path
