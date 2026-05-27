# -*- coding: utf-8 -*-
"""
1D line profiles of the probe-motion (bmotion) signal.

For a bmotion *line* run the probe steps along a single axis (one of x or y has
exactly one position, the other varies). This module reduces each grid position
to a single scalar -- either the **mean over a time range** or the **value at
one time step** -- and renders the result as a 1D line plot (value vs probe
position). It is the line-scan counterpart to :mod:`read_and_analyze.plot_xy_map`
(which only handles 2D planes and skips lines).

Reconstruction follows the same proven LAPD analysis pattern as plot_xy_map:
traces are read **in acquisition order**, the per-position shots form a stack, a
reducer turns that stack into one scalar per position, and the per-position
values are simply laid out along the moving axis -- no encoder-position binning.

Only **line scans** are supported; genuine 2D planes are skipped (use
plot_xy_map for those).

There is NO command line; all knobs live in :mod:`read_and_analyze.analysis_config`.
Run with:
    python -m read_and_analyze.plot_x_line

Setup (once):  python -m pip install scipy

Created May.2026
@author: Jia Han
"""

import os

import numpy as np

try:  # progress bar over the per-position loop; optional dependency
    from tqdm import tqdm
except ImportError:  # fall back to a no-op pass-through if tqdm isn't installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

from lab_scopes.io.hdf5 import (
    open_hdf5_readonly, read_hdf5_scope_channel_shots, read_hdf5_scope_tarr,
)
try:  # works as a package (python -m read_and_analyze.plot_x_line)
    from read_and_analyze.read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
    )
    from read_and_analyze.filter_data import _as_list
    from read_and_analyze.plot_xy_map import (
        _reduction_indices, _reduce_trace, _step_indices, _as_step_list,
        _reduction_label, _plane_axes, make_single_shot_reduce,
        _position_shotnums, _load_stack,
    )
    from read_and_analyze.analysis_config import (
        DATA_FILE as DEFAULT_FILE, MED_SIZE, GAUSS_SIGMA,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
        XY_MODE as MODE, XY_T_START_MS as T_START_MS, XY_T_END_MS as T_END_MS,
        XY_T_STEP_MS as T_STEP_MS, XY_CMAP as CMAP, XY_SHOT_INDEX as SHOT_INDEX,
    )
except ImportError:  # fallback when run directly from inside the folder
    from read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
    )
    from filter_data import _as_list
    from plot_xy_map import (
        _reduction_indices, _reduce_trace, _step_indices, _as_step_list,
        _reduction_label, _plane_axes, make_single_shot_reduce,
        _position_shotnums, _load_stack,
    )
    from analysis_config import (
        DATA_FILE as DEFAULT_FILE, MED_SIZE, GAUSS_SIGMA,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
        XY_MODE as MODE, XY_T_START_MS as T_START_MS, XY_T_END_MS as T_END_MS,
        XY_T_STEP_MS as T_STEP_MS, XY_CMAP as CMAP, XY_SHOT_INDEX as SHOT_INDEX,
    )


# ======================================================================================
# Line geometry  (acquisition-order layout -- no encoder-position binning)
# ======================================================================================

def _is_line(xpos, ypos):
    """True only for a genuine 1D line: exactly one axis has more than one position."""
    if xpos is None or ypos is None:
        return False
    nx, ny = len(xpos), len(ypos)
    return (nx > 1 and ny == 1) or (ny > 1 and nx == 1)


def _line_axis(xpos, ypos):
    """Return ``(axis_pos, axis_name, fixed_val)`` for a line scan.

    Picks whichever axis varies (>1 position) as the moving axis; the other axis
    is the fixed coordinate. ``axis_name`` is "x" or "y"; ``fixed_val`` is the
    single value of the stationary axis (for the title).
    """
    if len(xpos) > 1:
        return xpos, "x", float(ypos[0])
    return ypos, "y", float(xpos[0])


# ======================================================================================
# Line assembly
# ======================================================================================

def build_line(f, scope, ch, positions, reduce_fn, med_size, gauss_sigma):
    """Reduce every planned position to one scalar and lay it out along the line.

    Reads each position's repeat-shot stack in acquisition order, applies
    ``reduce_fn(stack, tarr, pos_idx)``, and returns ``(vals, axis_pos,
    axis_name, fixed_val)``; or ``(None, None, None, None)`` if the run has no
    setup array or is not a 1D line.
    """
    xpos, ypos, npos, _name = _plane_axes(positions)
    if not _is_line(xpos, ypos):
        return None, None, None, None
    axis_pos, axis_name, fixed_val = _line_axis(xpos, ypos)

    tarr = read_hdf5_scope_tarr(f, scope)
    total = len(_shot_numbers(f[scope]))
    nshot = total // npos if npos else 0
    mismatch = (nshot == 0) or (npos * nshot != total)
    if mismatch:
        print(f"  warning: scope '{scope}' has {total} shots != npos({npos}) x "
              f"nshot -- not a clean grid; using position-lookup fallback")

    vals = np.full(npos, np.nan, dtype=float)
    for i, shotnums in tqdm(_position_shotnums(positions, npos, nshot, mismatch),
                            total=npos, desc=f"reduce {scope}/{ch}", unit="pos"):
        stack = _load_stack(f, scope, ch, shotnums, tarr, med_size, gauss_sigma)
        vals[i] = reduce_fn(stack, tarr, i)

    return vals, axis_pos, axis_name, fixed_val


def build_lines_step(f, scope, ch, positions, t_steps_ms, shot_index,
                     med_size, gauss_sigma):
    """Build one line profile per snapshot time in a single read pass.

    Loads each position's stack once, picks shot ``shot_index``, and samples it at
    every requested snapshot index. Returns ``(curves, axis_pos, axis_name,
    fixed_val, t_los)`` where ``curves`` is a list of length-``npos`` arrays
    parallel to ``t_los`` (realized tarr-snapped times in seconds); or
    ``(None, None, None, None, None)`` if not a line.
    """
    xpos, ypos, npos, _name = _plane_axes(positions)
    if not _is_line(xpos, ypos):
        return None, None, None, None, None
    axis_pos, axis_name, fixed_val = _line_axis(xpos, ypos)

    tarr = read_hdf5_scope_tarr(f, scope)
    idxs, t_los = _step_indices(tarr, t_steps_ms)
    total = len(_shot_numbers(f[scope]))
    nshot = total // npos if npos else 0
    mismatch = (nshot == 0) or (npos * nshot != total)
    if mismatch:
        print(f"  warning: scope '{scope}' has {total} shots != npos({npos}) x "
              f"nshot -- not a clean grid; using position-lookup fallback")

    curves = [np.full(npos, np.nan, dtype=float) for _ in idxs]
    for i, shotnums in tqdm(_position_shotnums(positions, npos, nshot, mismatch),
                            total=npos, desc=f"reduce {scope}/{ch}", unit="pos"):
        stack = _load_stack(f, scope, ch, shotnums, tarr, med_size, gauss_sigma)
        if stack is None or shot_index >= stack.shape[0]:
            continue
        trace = stack[shot_index]
        for k, idx in enumerate(idxs):
            curves[k][i] = float(trace[idx])

    return curves, axis_pos, axis_name, fixed_val, t_los


# ======================================================================================
# Rendering
# ======================================================================================

def _render_line(plt, vals, axis_pos, axis_name, fixed_val, scope, ch, label,
                 shot_index, path):
    """Draw the single-curve ``range``/``step``-scalar line profile."""
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(axis_pos)
    ax.plot(axis_pos[order], vals[order], "-o", ms=4)
    fixed_name = "y" if axis_name == "x" else "x"
    ax.set_xlabel(f"probe {axis_name} (mm)")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"scope '{scope}' / {ch}: {label}  "
                 f"({fixed_name}={fixed_val:.1f} mm, shot {shot_index})",
                 fontsize=10, loc="left")
    fig.suptitle(f"{os.path.basename(path)}  —  line profile", fontsize=10)
    fig.tight_layout()


def _render_step_overlay(plt, curves, axis_pos, axis_name, fixed_val, t_los,
                         scope, ch, cmap, shot_index, path):
    """Draw the ``step``-mode overlay: one curve per snapshot time on shared axes,
    colored along ``cmap`` from earliest to latest time."""
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(axis_pos)
    n = len(curves)
    colors = cm.get_cmap(cmap)(np.linspace(0, 1, max(n, 1)))
    for k, (vals, t_lo) in enumerate(zip(curves, t_los)):
        ax.plot(axis_pos[order], vals[order], "-o", ms=3, color=colors[k],
                label=f"t={t_lo * 1e3:.4f} ms")
    fixed_name = "y" if axis_name == "x" else "x"
    ax.set_xlabel(f"probe {axis_name} (mm)")
    ax.set_ylabel("V")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"scope '{scope}' / {ch}: line profile (step)  "
                 f"({fixed_name}={fixed_val:.1f} mm, shot {shot_index})",
                 fontsize=10, loc="left")
    fig.suptitle(f"{os.path.basename(path)}  —  line profile (step)", fontsize=10)
    fig.tight_layout()


# ======================================================================================
# Driver
# ======================================================================================

def plot_x_line(path, scope=None, channels=None, mode=None,
                t_start=None, t_end=None, t_step=None, shot_index=None,
                med_size=None, gauss_sigma=None,
                cmap=None, show=None, save=None):
    """Render a line-only profile per (scope, channel).

    For each scope/channel: pick one shot per position by ``shot_index``, reduce it
    in time (``range`` -> mean over [t_start, t_end] ms; ``step`` -> overlay one
    curve per ``t_step`` time), and plot value vs probe position. Genuine 2D planes
    are skipped (use plot_xy_map). Honors SHOW_PLOT/SAVE_PLOT (override with
    show/save); saves one PNG per (scope, channel). Returns the saved paths.
    """
    import matplotlib.pyplot as plt

    scope = SCOPE if scope is None else scope
    channels = CHANNELS if channels is None else channels
    mode = MODE if mode is None else mode
    t_start = T_START_MS if t_start is None else t_start
    t_end = T_END_MS if t_end is None else t_end
    t_step = T_STEP_MS if t_step is None else t_step
    shot_index = SHOT_INDEX if shot_index is None else shot_index
    med_size = MED_SIZE if med_size is None else med_size
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
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

    with open_hdf5_readonly(path) as f:
        positions = read_positions(f)
        if not positions:
            print("no /Control/Positions data (not a bmotion file?) — nothing to plot")
            return saved
        scopes = [scope] if scope else _scope_groups(f)

        for sc in scopes:
            xpos, ypos, _npos, _name = _plane_axes(positions)
            if not _is_line(xpos, ypos):
                nx = 0 if xpos is None else len(xpos)
                ny = 0 if ypos is None else len(ypos)
                print(f"scope '{sc}': grid is {nx}x{ny} (not a line) — "
                      f"plot_x_line only supports line scans; skipping")
                continue

            sg = f[sc]
            shot_nums = _shot_numbers(sg)
            chans = channels if channels else _channel_names(sg, shot_nums[0])

            for ch in chans:
                if mode == "step":
                    t_steps = _as_step_list(t_step)
                    curves, axp, axn, fixed, t_los = build_lines_step(
                        f, sc, ch, positions, t_steps, shot_index,
                        med_size, gauss_sigma)
                    if curves is None or all(np.all(np.isnan(c)) for c in curves):
                        print(f"scope '{sc}' / {ch}: no usable shots — skipping")
                        continue
                    _render_step_overlay(plt, curves, axp, axn, fixed, t_los,
                                         sc, ch, cmap, shot_index, path)
                else:
                    vals, axp, axn, fixed = build_line(
                        f, sc, ch, positions,
                        make_single_shot_reduce(shot_index, mode, t_start, t_end, t_step),
                        med_size, gauss_sigma)
                    if vals is None or np.all(np.isnan(vals)):
                        print(f"scope '{sc}' / {ch}: no usable shots — skipping")
                        continue
                    tarr = read_hdf5_scope_tarr(f, sc)
                    i0, i1 = _reduction_indices(tarr, mode, t_start, t_end, t_step)
                    label = _reduction_label(mode, float(tarr[i0]), float(tarr[i1 - 1]))
                    _render_line(plt, vals, axp, axn, fixed, sc, ch, label,
                                 shot_index, path)

                if save:
                    out_png = os.path.join(plots_dir, f"{base}_{sc}_{ch}_xline.png")
                    plt.gcf().savefig(out_png, dpi=150)
                    saved.append(out_png)
                    print(f"Saved plot: {out_png}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return saved


if __name__ == "__main__":
    plot_x_line(DEFAULT_FILE)
