# -*- coding: utf-8 -*-
"""
Configuration knobs for :mod:`read_and_analyze.smart_trigger_analysis`.

Every user-changeable setting for the SmartTrigger scan lives here -- nothing
needs to be edited in ``smart_trigger_analysis.py``. The knobs are grouped into
clearly separated sections: input/output, preprocessing, then one section per
trigger mode so each detector can be tuned independently. Edit the values and
re-run ``python -m read_and_analyze.smart_trigger_analysis`` -- there is no
command line.

Levels are given in **absolute volts** (the same units as the recorded trace),
matching how you would dial a level on the scope's front panel. Width / slew /
interval limits are given in **nanoseconds** (ns), matching the scope's
SmartTrigger time settings (HOLDOFF stays in microseconds, as before). A
measured value is flagged when it falls OUTSIDE the [min, max] bounds you set
for that detector; leave a bound at ``None`` to disable that side.

The shared knobs (input file, scope/channel, filtering) come from
``analysis_config.py`` so there is a single source of truth; this file adds the
SmartTrigger-specific ones.

Created May.2026
@author: Jia Han
"""
import numpy as np
try:  # works as a package (python -m read_and_analyze.smart_trigger_analysis)
    from read_and_analyze.analysis_config import (
        DATA_FILE, MED_SIZE, GAUSS_SIGMA,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS,
    )
except ImportError:  # fallback when run directly from inside the folder
    from analysis_config import (
        DATA_FILE, MED_SIZE, GAUSS_SIGMA,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS,
    )

# ======================================================================================
# General -- output and preprocessing
# (DATA_FILE / SCOPE / CHANNELS / MED_SIZE / GAUSS_SIGMA come from analysis_config above)
# ======================================================================================
SHOW_PLOT  = False   # display the figure interactively
SAVE_PLOT  = True    # write a PNG to a "plots/" subdir next to the data file
SHOTS      = np.arange(0, 10)    # shots to scan: a list/array e.g. [12, 57] or np.arange(0, 10); None = sample shots (first/middle/last per position)
HOLDOFF_US = 3000    # ignore the record before this time (us from t=0); mimics trigger holdoff
MATH       = None    # None = filtered trace as-is; or "derivative" / "integral" / "abs"

# ======================================================================================
# Glitch / Width trigger -- flag pulses whose width is OUTSIDE [min, max]
#   A pulse is the span between a rising and the next falling crossing of
#   GLITCH_LEVEL (with GLITCH_HYST as the debounce band, both in volts).
#   Set a bound to None to disable that side (e.g. only catch glitches that are
#   too NARROW by setting GLITCH_MAX_WIDTH_NS = None).
# ======================================================================================
GLITCH_LEVEL        = 0.5    # pulse-measuring level in VOLTS
GLITCH_HYST         = 0.05   # hysteresis band in VOLTS to debounce noisy crossings
GLITCH_MIN_WIDTH_NS = 10000  # flag pulses NARROWER than this (ns); None = no lower bound
GLITCH_MAX_WIDTH_NS = 20000  # flag pulses WIDER than this (ns); None = no upper bound

# ======================================================================================
# Runt trigger -- flag pulses that cross LO but never reach HI (both in volts)
# ======================================================================================
RUNT_LO = 0.15    # lower level in VOLTS; a runt crosses this...
RUNT_HI = 0.2    # ...but never reaches this upper level (VOLTS) before returning

# ======================================================================================
# Slew-rate trigger -- flag edges whose LO<->HI transition time is OUTSIDE [min, max]
#   Levels in volts; transition-time bounds in ns. Set a bound to None to
#   disable that side.
# ======================================================================================
SLEW_LO        = 0.2    # lower level in VOLTS for the transition
SLEW_HI        = 0.6    # upper level in VOLTS for the transition
SLEW_MIN_NS    = None   # flag edges FASTER than this LO<->HI time (ns); None = no lower bound
SLEW_MAX_NS    = 500.0   # flag edges SLOWER than this LO<->HI time (ns); None = no upper bound

# ======================================================================================
# Interval trigger -- flag periods between rising edges that are OUTSIDE [min, max]
#   Edge level / hysteresis in volts; period bounds in ns.
# ======================================================================================
INTERVAL_LEVEL   = 0.4    # rising-edge level in VOLTS
INTERVAL_HYST    = 0.05   # hysteresis band in VOLTS
INTERVAL_MIN_NS  = None   # flag periods SHORTER than this (ns); None = no lower bound
INTERVAL_MAX_NS  = None   # flag periods LONGER than this (ns); None = no upper bound
