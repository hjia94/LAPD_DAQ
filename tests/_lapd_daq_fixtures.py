"""Shared experiment_config.txt fixtures for the lapd_daq test suite.

Three test modules (test_lapd_daq_config, test_lapd_daq_engine,
test_lapd_daq_compat) need the same representative INI config. Defining
it once here keeps them from drifting.

CONFIG_TEXT carries an inline comment on `num_duplicate_shots` because
one consumer (test_lapd_daq_config) asserts that the loader tolerates
inline comments. Consumers that don't care are unaffected.
"""

CONFIG_TEXT = """
[nshots]
num_duplicate_shots = 2 # inline comments should be accepted
num_run_repeats = 1

[position]
nx = 2
ny = 2
xmin = -1
xmax = 1
ymin = -2
ymax = 2

[experiment]
description = Mock LAPD run

[scopes]
MockScope = Fake LeCroy scope

[channels]
MockScope_C1 = mock channel one
MockScope_C2 = mock channel two

[scope_ips]
MockScope = 127.0.0.1
"""

CAMERA_CONFIG_TEXT = CONFIG_TEXT + """
[camera_config]
exposure_us = 40
fps = 1000
"""
