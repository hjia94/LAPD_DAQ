# -*- coding: utf-8 -*-
"""
Single source of user-changeable knobs for the read_and_analyze modules.

Every setting a user normally edits lives here so there is one place to look --
no need to open the analysis modules themselves. The file is organized into
clearly separated sections:

  * SHARED            -- knobs used by every module: the input file, the scope /
                         channels to analyze (SELECT_SCOPE / SELECT_CHAN), the
                         plot toggles (SHOW_PLOT / SAVE_PLOT), the time-domain
                         filtering pipeline, and the position grid tolerance.
                         (read_bmotion_data.py uses only these shared knobs.)
  * FLUCTUATION       -- fluctuation_analysis.py (quietest-window search)
  * XY_MAP            -- plot_xy_map.py (2D XY-plane maps)

SmartTrigger knobs live in their own ``smart_trigger_config.py`` (which imports
the SHARED knobs from this file), kept separate because there are many of them,
grouped per trigger mode.

Each module reads its values from here. The input file, scope/channel selection,
and plot toggles are shared across every module and live in the SHARED section;
the remaining sections hold only the knobs unique to one module. Levels/times
are in the units noted on each line.

Created May.2026
@author: Jia Han
"""

# ======================================================================================
# SHARED -- used across modules (input file, scope/channel, plot toggles,
#           filtering pipeline, grid tolerance)
# ======================================================================================
DATA_FILE   = r"D:\data\LAPD\03-LP-p21p29p41-plane-Helium_2026-05-20.hdf5"  # HDF5 file to analyze

SELECT_SCOPE = "lpscope"   # scope to analyze; None = all scopes (shared by every module)
SELECT_CHAN  = ["C1"]      # channels to analyze; None = all channels (shared by every module)

SHOW_PLOT   = True  # display figures interactively (shared by every module)
SAVE_PLOT   = False  # write PNGs to a "plots/" subdir next to the data file (shared by every module)

MED_SIZE    = 5    # median-filter window in SAMPLES, applied first (spike/outlier removal); 1 = off
GAUSS_SIGMA = 20   # Gaussian smoothing width in SAMPLES, applied after the median (high-freq noise); 0 = off

POS_TOL     = 0.5  # round (x, y) to this many mm so encoder float-noise groups repeat shots cleanly


# ======================================================================================
# FLUCTUATION -- fluctuation_analysis.py: find the quietest time window per position
# ======================================================================================
FLUCT_WINDOW_US   = 10.0        # analysis window width (microseconds) slid across the record
FLUCT_SIGNAL_FRAC = 0           # window mean must exceed this fraction of the position's peak |mean|


# ======================================================================================
# XY_MAP -- plot_xy_map.py: 2D XY-plane map of a reduced scalar per grid position
# ======================================================================================
XY_MODE         = "step"     # "range" = mean over [T_START_MS, T_END_MS]; "step" = snapshot(s) at XY_T_STEP_MS
XY_T_START_MS   = 10.0         # window start (ms), used when XY_MODE == "range"
XY_T_END_MS     = 22         # window end   (ms), used when XY_MODE == "range"
XY_T_STEP_MS    = [10.0, 12, 14, 16, 18, 20]  # snapshot time(s) in ms for "step" mode; one panel per time.
                                   # A single float (e.g. 4.0) is also accepted -> one panel.

XY_SHOT_INDEX   = 0           # which shot (0-based) per position to map; no shot averaging yet

XY_SHOW_CONTOUR = False       # overlay contour lines on top of the image
XY_N_CONTOURS   = 8           # number of contour levels when XY_SHOW_CONTOUR is True
XY_CMAP         = "rainbow"   # imshow colormap
