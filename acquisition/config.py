"""Experiment-config parsing for the multi-scope acquisition pipeline.

`load_experiment_config` reads `experiment_config.txt` once, returning both a
`ConfigParser` for structured access and the raw text for verbatim storage in
the resulting HDF5 file.
"""

import configparser


def load_experiment_config(config_path='experiment_config.txt'):
    """Load experiment configuration from config file.

    Returns:
        tuple: (config, raw_config_text)
            - config: ConfigParser object with parsed configuration
            - raw_config_text: Raw text content of the configuration file
    """
    config = configparser.ConfigParser()

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
