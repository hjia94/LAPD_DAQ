# -*- coding: utf-8 -*-
"""
Single source of user-changeable knobs for the read_and_analyze modules.

Knobs shared by every module live here, in two sections:

  * SHARED   -- input file, scope/channel selection, plot toggles, the
                time-domain filtering pipeline, and the grid tolerance
  * XY_MAP   -- plot_xy_map.py / plot_x_line.py (2D XY-plane maps)

Module-private knobs live at the top of the module that owns them, under a
``# ---- knobs ----`` marker (see ``fluctuation_analysis.py``). The exception is
``smart_trigger_analysis.py``, whose ~20 knobs are grouped per trigger mode in
their own ``smart_trigger_config.py``.

Created May.2026
@author: Jia Han
"""

# ======================================================================================
# SHARED -- used across modules (input file, scope/channel, plot toggles,
#           filtering pipeline, grid tolerance)
# ======================================================================================
# Full path to the run HDF5 file to analyze, e.g.
# r"M:\BAPSF_Data\Low_Density_Topo\Jun2026\01-Isat-p21-line-Argon-2kG_2026-06-08.hdf5"
DATA_FILE   = r"D:\data\LAPD\jun2026-jia\32-He-800G-bias40V-Mach-plane_2026-06-13.hdf5"

SELECT_SCOPE = "scope"   # scope to analyze; None = all scopes (shared by every module)
SELECT_CHAN  = 'C3'     # channels to analyze; None = all channels (shared by every module)

SHOW_PLOT   = True  # display figures interactively (shared by every module)
SAVE_PLOT   = False  # write PNGs to a "plots/" subdir next to the data file (shared by every module)

AUTO_PLOT   = True  # fallback default for the post-acquisition auto-plot hook when
                    # called without a config; the run's [analysis] auto_plot key
                    # (experiment_config.ini) overrides this during acquisition

MED_SIZE    = 1    # median-filter window in SAMPLES, applied first (spike/outlier removal); 1 = off
GAUSS_SIGMA = 0    # Gaussian smoothing width in SAMPLES, applied after the median (high-freq noise); 0 = off

POS_TOL     = 0.5  # round (x, y) to this many mm so encoder float-noise groups repeat shots cleanly


# ======================================================================================
# XY_MAP -- plot_xy_map.py: 2D XY-plane map of a reduced scalar per grid position
# ======================================================================================
XY_MODE         = "range"     # "range" = mean over [T_START_MS, T_END_MS]; "step" = snapshot(s) at XY_T_STEP_MS
XY_T_START_MS   = 0         # window start (ms), used when XY_MODE == "range"
XY_T_END_MS     = 2.0         # window end   (ms), used when XY_MODE == "range"
XY_T_STEP_MS    = [10,12,15,19]  # snapshot time(s) in ms for "step" mode; one panel per time.
                                   # A single float (e.g. 4.0) is also accepted -> one panel.

XY_SHOT_INDEX   = 0           # which shot (0-based) per position to map; no shot averaging yet

XY_SHOW_CONTOUR = False       # overlay contour lines on top of the image
XY_N_CONTOURS   = 8           # number of contour levels when XY_SHOW_CONTOUR is True
XY_CMAP         = "rainbow"   # imshow colormap
