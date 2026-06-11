# Run the bmotion encoder (EP vs IP) hardware check on the DAQ PC.
# Sets the opt-in environment variables for this process only, then runs the
# check. Edit the values here (or set the variables in your shell) per rig.
#
#   .\tests\run_encoder_check.ps1
#
# See tests/test_bmotion_recovery_hw.py for all available knobs
# (LAPD_RUN_LONG_MOTION_CHECK, LAPD_RUN_FAILURE_CHECK, LAPD_RUN_SET_ZERO_CHECK,
# LAPD_BMOTION_FAILURE_INDEX, ...).

$env:LAPD_BMOTION_ALLOW_MOVE = "1"
$env:LAPD_RUN_ENCODER_CHECK = "1"
$env:LAPD_BMOTION_TOML = "E:\Shadow data\Pat\bmotion_config.toml"
$env:LAPD_BMOTION_MOTION_GROUP = "2"   # group name (e.g. "Hermes") or TOML index

python -m unittest tests.test_bmotion_recovery_hw -v
