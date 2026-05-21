# -*- coding: utf-8 -*-
"""
Configuration knobs for :mod:`read_and_analyze.smart_trigger_analysis`.

All settings for the SmartTrigger scan live here, grouped by trigger mode so
each detector can be tuned independently. Edit the values and re-run
``python -m read_and_analyze.smart_trigger_analysis`` -- there is no command
line. Filtering / file-selection knobs (DEFAULT_FILE, SCOPE, CHANNELS,
MED_SIZE, GAUSS_SIGMA) still live in ``filter_data.py`` and are shared.

Levels are fractions of each trace's own (min..max) span (the software analog of
the scope's "Find Level"), so they are dimensionless and work on signals of any
absolute scale. EXCL_DELTA is the exclusion band: a measured value is flagged
when ``|value - nominal| / nominal`` exceeds it.

Created May.2026
@author: Jia Han
"""

# ======================================================================================
# General -- input file, run scope, output, and shared preprocessing
# ======================================================================================
DATA_FILE  = r"D:\data\LAPD\00-LP-p21p29p41-Xline-test_2026-05-19.hdf5"  # HDF5 file to scan
SHOW_PLOT  = False   # display the figure interactively
SAVE_PLOT  = True    # write a PNG to a "plots/" subdir next to the data file
SHOTS      = None    # None = sample shots (first/middle/last per position); or e.g. [12, 57]
HOLDOFF_US = 3000    # ignore the record before this time (us from t=0); mimics trigger holdoff
MATH       = None    # None = filtered trace as-is; or "derivative" / "integral" / "abs"

# ======================================================================================
# Glitch / Width trigger -- flag pulses NARROWER than the nominal width
# ======================================================================================
GLITCH_THRESH_FRAC = 0.5    # pulse-measuring level (fraction of the trace's span)
GLITCH_HYST_FRAC   = 0.05   # hysteresis band (fraction of span) to debounce noisy crossings
GLITCH_EXCL_DELTA  = 0.25   # flag a pulse when its width < nominal*(1 - this)

# ======================================================================================
# Runt trigger -- flag pulses that cross LO but never reach HI
# ======================================================================================
RUNT_LO_FRAC = 0.3    # lower level (fraction of span); a runt crosses this...
RUNT_HI_FRAC = 0.7    # ...but never reaches this upper level before returning

# ======================================================================================
# Slew-rate trigger -- flag edges whose LO<->HI transition time is off-nominal
# ======================================================================================
SLEW_LO_FRAC   = 0.1    # lower level for the transition (fraction of span)
SLEW_HI_FRAC   = 0.9    # upper level for the transition (fraction of span)
SLEW_EXCL_DELTA = 0.25  # flag an edge when its transition time is outside nominal*(1 +/- this)

# ======================================================================================
# Interval trigger -- flag periods between rising edges that are off-nominal
# ======================================================================================
INTERVAL_THRESH_FRAC = 0.5   # rising-edge level (fraction of span)
INTERVAL_HYST_FRAC   = 0.05  # hysteresis band (fraction of span)
INTERVAL_EXCL_DELTA  = 0.25  # flag a period when it is outside nominal*(1 +/- this)
