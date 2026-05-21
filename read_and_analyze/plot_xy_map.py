# -*- coding: utf-8 -*-
"""
2D XY-plane maps of the probe-motion (bmotion) signal.

For a bmotion run the probe steps over an XY grid. This module reduces each grid
position's filtered trace to a single scalar -- either the **mean over a time
range** or the **value at one time step** -- and renders the result as a 2D image
(``imshow``) with the probe grid on the axes, optionally with contour lines on
top. It complements the time-domain views in
:mod:`read_and_analyze.read_bmotion_data` (overlaid traces) and
:mod:`read_and_analyze.filter_data` (raw vs filtered).

Filtering, shot grouping, and trace loading are imported from
:mod:`read_and_analyze.filter_data`; position/shot helpers from
:mod:`read_and_analyze.read_bmotion_data`; reading/decoding from
``lab_scopes.io.hdf5``.

There is NO command line; all knobs are the constants below. Run with:
    python -m read_and_analyze.plot_xy_map
Edit DEFAULT_FILE / the constants to change the file or parameters.

Setup (once):  python -m pip install scipy

Created May.2026
@author: Jia Han
"""

import os

import numpy as np

from lab_scopes.io.hdf5 import read_hdf5_scope_tarr
try:  # works as a package (python -m read_and_analyze.plot_xy_map)
    from read_and_analyze.read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
    )
    from read_and_analyze.filter_data import (
        _as_list, _shots_by_position, load_filtered_traces,
    )
    from read_and_analyze.analysis_config import (
        DATA_FILE as DEFAULT_FILE, MED_SIZE, GAUSS_SIGMA, POS_TOL as _POS_TOL,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
        XY_MODE as MODE, XY_T_START_US as T_START_US, XY_T_END_US as T_END_US,
        XY_T_STEP_US as T_STEP_US, XY_SHOW_CONTOUR as SHOW_CONTOUR,
        XY_N_CONTOURS as N_CONTOURS, XY_CMAP as CMAP,
    )
except ImportError:  # fallback when run directly from inside the folder
    from read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
    )
    from filter_data import (
        _as_list, _shots_by_position, load_filtered_traces,
    )
    from analysis_config import (
        DATA_FILE as DEFAULT_FILE, MED_SIZE, GAUSS_SIGMA, POS_TOL as _POS_TOL,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
        XY_MODE as MODE, XY_T_START_US as T_START_US, XY_T_END_US as T_END_US,
        XY_T_STEP_US as T_STEP_US, XY_SHOW_CONTOUR as SHOW_CONTOUR,
        XY_N_CONTOURS as N_CONTOURS, XY_CMAP as CMAP,
    )


# ======================================================================================
# Trace reduction
# ======================================================================================

def _reduction_indices(tarr, mode, t_start, t_end, t_step):
    """Resolve the requested times to actual ``tarr`` sample indices.

    ``mode == "step"``:  returns ``(idx, idx + 1)`` for the sample nearest
    ``t_step`` (us). ``mode == "range"``: returns the half-open ``[i0, i1)`` slice
    for the closed window [t_start, t_end] (us), using the same searchsorted
    indexing as ``fluctuation_analysis._profile_value``. Both are clamped to the
    record so callers can read the realized bounds straight from ``tarr``.
    """
    if mode == "step":
        idx = int(np.argmin(np.abs(tarr - t_step * 1e-6)))
        return idx, idx + 1

    i0 = int(np.searchsorted(tarr, t_start * 1e-6))
    i1 = int(np.searchsorted(tarr, t_end * 1e-6, side="right"))
    i0 = max(0, min(i0, len(tarr) - 1))
    i1 = max(i0 + 1, min(i1, len(tarr)))
    return i0, i1


def _reduce_trace(vf, tarr, mode, t_start, t_end, t_step):
    """Reduce one filtered trace ``vf`` (vs ``tarr``, seconds) to a scalar.

    ``mode == "range"``: mean over the closed time window [t_start, t_end] (us).
    ``mode == "step"``:  the sample nearest ``t_step`` (us). Times outside the
    record are clamped to the nearest sample. Index resolution is shared with the
    label via :func:`_reduction_indices`.
    """
    i0, i1 = _reduction_indices(tarr, mode, t_start, t_end, t_step)
    if mode == "step":
        return float(vf[i0])
    return float(vf[i0:i1].mean())


# ======================================================================================
# Grid assembly
# ======================================================================================

def _grid_axes(positions):
    """Return (xpos, ypos) regular grid axes from the first motion group that has
    them, or (None, None) if no grid axes are recorded."""
    for info in positions.values():
        xpos, ypos = info.get("xpos"), info.get("ypos")
        if xpos is not None and ypos is not None:
            return np.asarray(xpos, dtype=float), np.asarray(ypos, dtype=float)
    return None, None


def _nearest_index(axis, value):
    """Index of the grid-axis cell nearest ``value`` (within _POS_TOL), else None."""
    j = int(np.argmin(np.abs(axis - value)))
    return j if abs(axis[j] - value) <= _POS_TOL else None


def build_xy_grid(f, scope, ch, positions, mode, t_start, t_end, t_step,
                  med_size, gauss_sigma):
    """Reduce every grid position's repeat-shot traces to one scalar and place it
    on the regular probe grid.

    For each (x, y): load+filter the repeat shots, reduce each to a scalar via
    :func:`_reduce_trace`, average across shots, and write that into the
    ``(len(ypos), len(xpos))`` array ``Z`` at the nearest grid cell. Cells with no
    usable shots stay ``np.nan``. Returns ``(Z, xpos, ypos, t_lo, t_hi)`` where
    ``t_lo``/``t_hi`` are the realized (``tarr``-snapped) window bounds in seconds
    used for the reduction, or ``(None, None, None, None, None)`` if the run has
    no recorded grid axes.
    """
    xpos, ypos = _grid_axes(positions)
    if xpos is None or ypos is None:
        return None, None, None, None, None

    tarr = read_hdf5_scope_tarr(f, scope)
    i0, i1 = _reduction_indices(tarr, mode, t_start, t_end, t_step)
    t_lo, t_hi = float(tarr[i0]), float(tarr[i1 - 1])
    Z = np.full((len(ypos), len(xpos)), np.nan, dtype=float)
    by_pos = _shots_by_position(f, scope, positions)

    for (x, y), shots in by_pos.items():
        ix = _nearest_index(xpos, x)
        iy = _nearest_index(ypos, y)
        if ix is None or iy is None:
            continue
        rows = load_filtered_traces(f, scope, ch, shots, tarr, med_size, gauss_sigma)
        if not rows:
            continue
        vals = [_reduce_trace(vf, tarr, mode, t_start, t_end, t_step) for vf in rows]
        Z[iy, ix] = float(np.mean(vals))

    return Z, xpos, ypos, t_lo, t_hi


def _reduction_label(mode, t_lo, t_hi):
    """Human-readable description of the scalar reduction, for titles/colorbars,
    using the realized (``tarr``-snapped) bounds ``t_lo``/``t_hi`` (seconds).

    For ``range`` it states the averaging *start* in ms and the *width* in us, so
    the reader sees exactly what the figure averaged: e.g.
    "mean V from t=1.234 ms over 10.0 us".
    """
    if mode == "step":
        return f"V @ t={t_lo * 1e3:.4f} ms"
    width_us = (t_hi - t_lo) * 1e6
    return f"mean V from t={t_lo * 1e3:.4f} ms over {width_us:.1f} us"


# ======================================================================================
# Plotting
# ======================================================================================

def plot_xy_map(path, scope=None, channels=None, mode=None,
                t_start=None, t_end=None, t_step=None,
                med_size=None, gauss_sigma=None,
                show_contour=None, cmap=None, show=None, save=None):
    """Render a 2D XY map per (scope, channel): the reduced scalar over the probe grid.

    Uses ``imshow`` on the regular probe grid (origin lower, extent from the grid
    bounds), with an optional contour overlay. Honors SHOW_PLOT/SAVE_PLOT
    (override with show/save). Saves one PNG per (scope, channel) to a ``plots/``
    subdir next to the data file. Returns the saved paths.
    """
    import h5py
    import matplotlib.pyplot as plt

    scope = SCOPE if scope is None else scope
    channels = CHANNELS if channels is None else channels
    mode = MODE if mode is None else mode
    t_start = T_START_US if t_start is None else t_start
    t_end = T_END_US if t_end is None else t_end
    t_step = T_STEP_US if t_step is None else t_step
    med_size = MED_SIZE if med_size is None else med_size
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    show_contour = SHOW_CONTOUR if show_contour is None else show_contour
    cmap = CMAP if cmap is None else cmap
    show = SHOW_PLOT if show is None else show
    save = SAVE_PLOT if save is None else save
    channels = _as_list(channels)

    if mode not in ("range", "step"):
        raise ValueError(f"MODE must be 'range' or 'step', got {mode!r}")

    saved = []
    if save:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "plots")
        os.makedirs(plots_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    with h5py.File(path, "r") as f:
        positions = read_positions(f)
        if not positions:
            print("no /Control/Positions data (not a bmotion file?) — nothing to map")
            return saved
        scopes = [scope] if scope else _scope_groups(f)

        for sc in scopes:
            sg = f[sc]
            shot_nums = _shot_numbers(sg)
            chans = channels if channels else _channel_names(sg, shot_nums[0])

            for ch in chans:
                Z, xpos, ypos, t_lo, t_hi = build_xy_grid(
                    f, sc, ch, positions, mode, t_start, t_end, t_step,
                    med_size, gauss_sigma)
                if Z is None:
                    print(f"scope '{sc}': no recorded grid axes — skipping")
                    break  # same for every channel of this scope
                if np.all(np.isnan(Z)):
                    print(f"scope '{sc}' / {ch}: no usable shots on the grid — skipping")
                    continue

                # Label uses the realized tarr-snapped bounds (start in ms, width in us).
                label = _reduction_label(mode, t_lo, t_hi)
                _render_map(plt, Z, xpos, ypos, sc, ch, label, cmap,
                            show_contour, base, path)

                if save:
                    out_png = os.path.join(plots_dir, f"{base}_{sc}_{ch}_xymap.png")
                    plt.gcf().savefig(out_png, dpi=150)
                    saved.append(out_png)
                    print(f"Saved plot: {out_png}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return saved


def _render_map(plt, Z, xpos, ypos, scope, ch, label, cmap, show_contour,
                base, path):
    """Draw one XY-map figure. Falls back to a 1D line when the grid is degenerate
    (a single row or column, i.e. a 1D scan)."""
    # 1D scan: one of the axes is a single point -> imshow would be a sliver.
    if len(xpos) == 1 or len(ypos) == 1:
        fig, ax = plt.subplots(figsize=(9, 4))
        if len(ypos) == 1:
            xs, vals, xlabel = xpos, Z[0, :], "probe x (mm)"
        else:
            xs, vals, xlabel = ypos, Z[:, 0], "probe y (mm)"
        ax.plot(xs, vals, "o-", lw=1, ms=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        print(f"scope '{scope}' / {ch}: 1D scan (grid is "
              f"{len(xpos)}x{len(ypos)}) — drawing a line plot instead of an image")
    else:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        # imshow expects ascending row/col order; xpos/ypos come ascending from
        # the motion-group attrs, and origin='lower' matches that.
        extent = [xpos.min(), xpos.max(), ypos.min(), ypos.max()]
        im = ax.imshow(Z, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, label=label)
        if show_contour:
            xc = np.linspace(xpos.min(), xpos.max(), Z.shape[1])
            yc = np.linspace(ypos.min(), ypos.max(), Z.shape[0])
            X, Y = np.meshgrid(xc, yc)
            masked = np.ma.masked_invalid(Z)
            cs = ax.contour(X, Y, masked, levels=N_CONTOURS, colors="white",
                            linewidths=0.6, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.3g")
        ax.set_xlabel("probe x (mm)")
        ax.set_ylabel("probe y (mm)")

    ax.set_title(f"scope '{scope}' / {ch}: {label}", fontsize=10, loc="left")
    fig.suptitle(f"{os.path.basename(path)}  —  XY map", fontsize=10)
    fig.tight_layout()


if __name__ == "__main__":
    plot_xy_map(DEFAULT_FILE)
