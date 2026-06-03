"""Shared experiment_config.txt fixtures for the lapd_daq test suite.

Consumed by test_daq_core.py for its config-parser / shot-planning tests
(the formerly-separate lapd_daq config/engine/compat modules were merged
into test_daq_core). Defining the INI once here keeps tests from drifting.

CONFIG_TEXT carries an inline comment on `num_duplicate_shots` because
test_daq_core asserts that the loader tolerates inline comments. Consumers
that don't care are unaffected.
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

# The run description now lives in description.txt next to the config, not in the
# [experiment] section. Tests that need a description write this beside the config.
DESCRIPTION_TEXT = "Mock LAPD run"
