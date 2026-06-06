# -*- coding: utf-8 -*-
"""
Defensive post-acquisition auto-plot hook.

After a data run finishes, the acquisition / offload entry scripts call
:func:`maybe_autoplot` to render the 1D line profile for the just-written HDF5
file. The plotter (:func:`read_and_analyze.plot_x_line.plot_line`) already
auto-detects x-line vs y-line and *skips* genuine 2D planes and single-point
runs, so this hook just wires that call into the run-finish path -- gated by a
config flag and wrapped so it can NEVER crash or hang the acquisition process.

Design notes (see the matching plan):
  * In-process, ``show=False`` -- saves PNGs only, never opens a blocking window.
  * The matplotlib / scipy / lab_scopes imports are done lazily *inside* the
    try, so a minimal acquisition interpreter that lacks the analysis deps
    degrades to a printed warning instead of an ImportError crash.
  * ``show``/``save`` are forced (not inherited from analysis_config, whose
    committed defaults SHOW_PLOT=True / SAVE_PLOT=False are wrong for an
    unattended hook).

Created Jun.2026
@author: Jia Han
"""

import os


def _auto_plot_enabled(config):
    """Resolve the on/off flag without importing the acquisition package.

    With a ``config`` (an experiment_config.ini object) read ``[analysis]
    auto_plot`` via the acquisition accessor; without one fall back to the
    analysis-side ``AUTO_PLOT`` default. Errors here propagate to the caller's
    swallow-all handler (so a config/import quirk downgrades to a printed
    warning rather than crashing the run).
    """
    if config is not None:
        try:  # package import; acquisition.config is only needed in this branch
            from acquisition.config import get_auto_plot_enabled
        except ImportError:
            from config import get_auto_plot_enabled
        return get_auto_plot_enabled(config)
    try:
        from read_and_analyze.analysis_config import AUTO_PLOT
    except ImportError:
        from analysis_config import AUTO_PLOT
    return AUTO_PLOT


def maybe_autoplot(hdf5_path, config=None):
    """Render the line profile for ``hdf5_path`` if auto-plotting is enabled.

    Saves PNGs (one per scope/channel) into a ``plots/`` subfolder next to the
    data file and never shows a window. Non-line runs (planes / single points)
    are skipped by the plotter itself. This function is intentionally
    swallow-all: any error -- missing deps, unreadable/partial HDF5, savefig
    failure -- is downgraded to a printed warning so acquisition teardown is
    never interrupted.
    """
    try:
        if not _auto_plot_enabled(config):
            return

        try:  # works as a package (python -m ...) or run directly from the folder
            from read_and_analyze.plot_x_line import plot_line
        except ImportError:
            from plot_x_line import plot_line

        # Force show=False/save=True: this is an unattended hook, so it must
        # save and must not block on a GUI window, regardless of analysis_config.
        saved = plot_line(hdf5_path, show=False, save=True)

        name = os.path.basename(hdf5_path)
        if saved:
            print(f"Auto-plot: saved {len(saved)} line plot(s) for {name}")
        else:
            print(f"Auto-plot: nothing plotted for {name} "
                  f"(non-line run or no usable shots)")
    except Exception as e:  # never let a plotting problem abort the run teardown
        print(f"Warning: auto-plot failed: {e}")
