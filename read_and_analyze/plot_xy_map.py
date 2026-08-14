# -*- coding: utf-8 -*-
"""
2D XY-plane maps of the probe-motion (bmotion) signal.

For a bmotion run the probe steps over an XY *plane*. This module reduces each
grid position to a single scalar -- either the **mean over a time range** or the
**value at one time step** -- and renders the result as a 2D image (``imshow``)
with the probe grid on the axes, optionally with contour lines on top. It
complements the time-domain views in :mod:`read_and_analyze.read_bmotion_data`
(overlaid traces) and :mod:`read_and_analyze.filter_data` (raw vs filtered).

Reconstruction follows the proven LAPD analysis pattern (see
``data-analysis/ucla-lapd/Mar-2026/Mar2026_IV.py``): traces are read **in
acquisition order**, the per-position shots form a stack, a reducer turns that
stack into one scalar per position, and the per-position values are simply
``reshape((ny, nx))`` -- no encoder-position binning. The motion list is acquired
x-fastest with y descending, so row 0 of the reshaped array is max-y; we render
with ``origin='upper'`` and ``contour`` on ``meshgrid(xpos, ypos)`` so the image
and contour align exactly (no half-cell offset).

Only **planes** are supported; line scans (one axis with a single position) are
skipped.

There is NO command line; all knobs live in :mod:`read_and_analyze.analysis_config`.
Run with:
    python -m read_and_analyze.plot_xy_map

Setup (once):  python -m pip install scipy

Created May.2026
@author: Jia Han
"""

import os
from collections import defaultdict

import numpy as np

try:  # progress bar over the per-position loop; optional dependency
    from tqdm import tqdm
except ImportError:  # fall back to a no-op pass-through if tqdm isn't installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Allow running directly (IDE "Run" button / from inside this folder) as well as
# ``python -m read_and_analyze.<module>`` from the repo root: the root-level
# ``scope_io``/``acquisition`` packages need the repo root on sys.path, which ``-m``
# adds but a direct script run does not, so put it there ourselves.
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scope_io import (
    open_hdf5_readonly, read_hdf5_scope_channel_shots, read_hdf5_scope_tarr,
)
try:  # works as a package (python -m read_and_analyze.plot_xy_map)
    from read_and_analyze.read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
        resolve_data_file,
    )
    from read_and_analyze.filter_data import _as_list, _filter_trace
    from read_and_analyze.signals import raw as raw_signal
    from read_and_analyze.analysis_config import (
        MED_SIZE, GAUSS_SIGMA,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
        XY_MODE as MODE, XY_T_START_MS as T_START_MS, XY_T_END_MS as T_END_MS,
        XY_T_STEP_MS as T_STEP_MS, XY_SHOW_CONTOUR as SHOW_CONTOUR,
        XY_N_CONTOURS as N_CONTOURS, XY_CMAP as CMAP, XY_SHOT_INDEX as SHOT_INDEX,
    )
except ImportError:  # fallback when run directly from inside the folder
    from read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
        resolve_data_file,
    )
    from filter_data import _as_list, _filter_trace
    from signals import raw as raw_signal
    from analysis_config import (
        MED_SIZE, GAUSS_SIGMA,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
        XY_MODE as MODE, XY_T_START_MS as T_START_MS, XY_T_END_MS as T_END_MS,
        XY_T_STEP_MS as T_STEP_MS, XY_SHOW_CONTOUR as SHOW_CONTOUR,
        XY_N_CONTOURS as N_CONTOURS, XY_CMAP as CMAP, XY_SHOT_INDEX as SHOT_INDEX,
    )


# ======================================================================================
# Time reduction
# ======================================================================================

def _reduction_indices(tarr, mode, t_start, t_end, t_step):
    """Resolve the requested times to actual ``tarr`` sample indices.

    ``mode == "step"``:  returns ``(idx, idx + 1)`` for the sample nearest
    ``t_step`` (ms). ``mode == "range"``: returns the half-open ``[i0, i1)`` slice
    for the closed window [t_start, t_end] (ms). Both are clamped to the record so
    callers can read the realized bounds straight from ``tarr``.
    """
    if mode == "step":
        idx = int(np.argmin(np.abs(tarr - t_step * 1e-3)))
        return idx, idx + 1

    return window_indices(tarr, t_start * 1e-3, t_end * 1e-3)


def window_indices(tarr, t_lo, t_hi):
    """Half-open ``[i0, i1)`` sample slice for the closed window [t_lo, t_hi].

    Bounds are in **seconds** (``tarr``'s own units, unlike the ms-based callers
    above). Always returns a non-empty slice clamped to the record, so callers
    can read the realized bounds straight from ``tarr``. This is the single
    ms/seconds-to-index contract for the package -- the slider's frame axis and
    the monitor-RMS classifier both resolve their windows through it.
    """
    i0 = int(np.searchsorted(tarr, t_lo))
    i1 = int(np.searchsorted(tarr, t_hi, side="right"))
    i0 = max(0, min(i0, len(tarr) - 1))
    i1 = max(i0 + 1, min(i1, len(tarr)))
    return i0, i1


def _reduce_trace(vf, tarr, mode, t_start, t_end, t_step):
    """Reduce one trace ``vf`` (vs ``tarr``, seconds) to a scalar over time.

    ``mode == "range"``: mean over the closed window [t_start, t_end] (ms).
    ``mode == "step"``:  the sample nearest ``t_step`` (ms). NaN-safe.
    """
    i0, i1 = _reduction_indices(tarr, mode, t_start, t_end, t_step)
    if mode == "step":
        return float(vf[i0])
    return float(np.nanmean(vf[i0:i1]))


def _step_indices(tarr, t_steps_ms):
    """For each snapshot time (ms) return the nearest ``tarr`` sample index and the
    realized (snapped) time in seconds. Returns ``(idxs, t_los)`` parallel lists."""
    idxs, t_los = [], []
    for t in t_steps_ms:
        idx = int(np.argmin(np.abs(tarr - t * 1e-3)))
        idxs.append(idx)
        t_los.append(float(tarr[idx]))
    return idxs, t_los


def _as_step_list(t_step):
    """Accept a single float or a sequence of snapshot times; return a list of floats."""
    if np.isscalar(t_step):
        return [float(t_step)]
    return [float(t) for t in t_step]


def _reduction_label(mode, t_lo, t_hi):
    """Human-readable description of the time reduction, for titles/colorbars,
    using the realized (``tarr``-snapped) bounds ``t_lo``/``t_hi`` (seconds).

    ``range`` states the averaging *start* in ms and the *width* in us; ``step``
    states the snapshot time in ms.
    """
    if mode == "step":
        return f"V @ t={t_lo * 1e3:.4f} ms"
    width_us = (t_hi - t_lo) * 1e6
    return f"mean V from t={t_lo * 1e3:.4f} ms over {width_us:.1f} us"


# ======================================================================================
# Plane geometry  (acquisition-order reshape -- no encoder-position binning)
# ======================================================================================

def _plane_axes(positions):
    """Return ``(xpos, ypos, npos, name)`` from the first motion group.

    Prefers the authoritative ``xpos``/``ypos`` axis vectors the run stored as
    attrs on ``positions_setup_array`` (see ``acquisition/bmotion.py``); falls
    back to deriving them from the setup array's columns with ``np.unique`` for
    older files that lack those attrs. ``npos`` is the number of planned
    positions. Returns ``(None, None, 0, None)`` if no setup array is present.
    """
    for name, info in positions.items():
        setup = info.get("setup_array")
        if setup is None:
            continue
        xpos, ypos = info.get("xpos"), info.get("ypos")
        if xpos is None or ypos is None:  # older file: recover from the columns
            xpos = np.unique(np.asarray(setup["x"], dtype=float))
            ypos = np.unique(np.asarray(setup["y"], dtype=float))
        else:
            xpos = np.asarray(xpos, dtype=float)
            ypos = np.asarray(ypos, dtype=float)
        return xpos, ypos, len(setup), name
    return None, None, 0, None


def _is_plane(xpos, ypos):
    """True only for a genuine 2D plane (both axes have more than one position)."""
    return xpos is not None and ypos is not None and len(xpos) > 1 and len(ypos) > 1


# ======================================================================================
# Reducers   reduce_fn(stack, tarr, pos_idx) -> scalar
# ======================================================================================
# A reducer turns one position's (nshot, nsamples) trace stack into a single
# scalar. Keeping the *whole stack* (rather than pre-selecting a trace) is what
# lets future reducers do per-position time indexing or shot-averaging /
# normalized-std without touching the plane builder or rendering.

def make_single_shot_reduce(shot_index, mode, t_start, t_end, t_step):
    """Reducer: pick one shot per position by ``shot_index`` and reduce it in time.

    No shot averaging -- this is the current default. The full stack is still
    passed in so other reducers (per-position time index, shot-averaged
    normalized-std) can replace this one later with no other code changes.
    """
    def reduce_fn(stack, tarr, pos_idx):
        if stack is None or shot_index >= stack.shape[0]:
            return np.nan
        return _reduce_trace(stack[shot_index], tarr, mode, t_start, t_end, t_step)
    return reduce_fn


# ======================================================================================
# Shot selectors   selector(shotnums) -> {group_label: [shot_nums]}
# ======================================================================================
# A selector answers one question: given the shots recorded at one position,
# which subsets get averaged together, and what is each subset called? Keeping
# it position-local means the selector knows nothing about *why* shots group --
# antenna state, parity, anything -- so a new grouping is a new dict, not a new
# code path through the builder.

def select_all_shots(shotnums):
    """Average every shot at the position into one group (the default)."""
    return {"mean": list(shotnums)}


def select_shot_index(shot_index):
    """Pick one shot per position by index -- the historical single-shot behavior."""
    def selector(shotnums):
        if shot_index >= len(shotnums):
            return {}
        return {f"shot {shot_index}": [shotnums[shot_index]]}
    return selector


def select_by_state(state_by_shot):
    """Group a position's shots by a precomputed ``{shot_num: label}`` map.

    Shots with no entry are dropped; see
    :func:`read_and_analyze.state_grouping.classify_by_channel_rms`, which labels
    the unclassifiable as ``"unknown"`` rather than omitting them, so a dropped
    shot here means the classifier never saw it at all.
    """
    def selector(shotnums):
        groups = {}
        for s in shotnums:
            label = state_by_shot.get(s)
            if label is not None:
                groups.setdefault(label, []).append(s)
        return groups
    return selector


# ======================================================================================
# Frame assembly   (shared by the static map and the HTML slider)
# ======================================================================================

def _grid_layout(f, scope, npos):
    """Time axis, shots-per-position, and whether the run is a clean grid.

    ``mismatch`` True means the recorded shot count is not ``npos x nshot``, so
    callers must map shots to positions by recorded coordinate instead of by
    acquisition order. Shared by both plane builders so the rule (and its
    warning) is stated once.
    """
    tarr = read_hdf5_scope_tarr(f, scope)
    total = len(_shot_numbers(f[scope]))
    nshot = total // npos if npos else 0
    mismatch = (nshot == 0) or (npos * nshot != total)
    if mismatch:
        print(f"  warning: scope '{scope}' has {total} shots != npos({npos}) x "
              f"nshot -- not a clean grid; using position-lookup fallback")
    return tarr, nshot, mismatch


def _reduce_group(stack, rows, idxs):
    """Mean, standard error, and contributing-shot count at each time index.

    ``rows`` are the stack rows belonging to one group. Returns three length-N
    arrays parallel to ``idxs``. The standard error uses ``ddof=1`` because these
    are sample repeats, so a single-shot cell yields NaN -- one measurement
    cannot estimate its own scatter, and NaN says that honestly.
    """
    # Slice the sampled columns *before* the fancy row index: ``stack[rows]``
    # would copy the whole (nshot, nsamples) array first -- 4 MB per position on
    # a 100k-sample record -- only to discard all but a few hundred columns.
    sub = stack[:, idxs][rows]                     # (ngroup_shots, nframes)
    n = np.sum(np.isfinite(sub), axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(sub, axis=0) if sub.size else np.full(len(idxs), np.nan)
        sd = np.nanstd(sub, axis=0, ddof=1) if sub.shape[0] > 1 else np.full(len(idxs), np.nan)
        sem = np.where(n > 1, sd / np.sqrt(np.where(n > 0, n, np.nan)), np.nan)
    return mean, sem, n


def build_frames(f, scope, signal, positions, idxs, selector,
                 med_size, gauss_sigma):
    """Reduce every planned position to per-group (mean, sem, n) at each time index.

    One read pass: each position's channel stacks are loaded once, combined into
    the derived signal per shot, then split into groups and reduced. Returns
    ``({label: (values, sem, n)}, xpos, ypos)`` with each array shaped
    ``(nframes, ny, nx)``; or ``({}, None, None)`` if the run is not a 2D plane.

    Group labels are collected across positions, so a state that appears only at
    some positions still gets a full-size array (NaN where absent).
    """
    xpos, ypos, npos, _name = _plane_axes(positions)
    if not _is_plane(xpos, ypos):
        return {}, None, None
    nx, ny = len(xpos), len(ypos)
    nframes = len(idxs)

    tarr, nshot, mismatch = _grid_layout(f, scope, npos)

    # label -> (values, sem, n), each (npos, nframes). Values/sem default to NaN
    # ("not measured"); n defaults to 0 ("no shots"), which is a real count.
    acc = defaultdict(lambda: (np.full((npos, nframes), np.nan),
                               np.full((npos, nframes), np.nan),
                               np.zeros((npos, nframes))))

    desc = f"reduce {scope}/{signal.key}"
    for i, shotnums in tqdm(_position_shotnums(positions, npos, nshot, mismatch),
                            total=npos, desc=desc, unit="pos"):
        if not shotnums:
            continue
        # Ask the selector first: a position it rejects entirely (an out-of-range
        # shot index, or shots that classified into no state) must not cost a
        # read of every channel the signal needs.
        groups = selector(shotnums)
        if not groups:
            continue
        stack = _load_signal_stack(f, scope, signal, shotnums, tarr,
                                   med_size, gauss_sigma)
        if stack is None:
            continue
        # Map shot number -> row in the stack, so a selector can name shots
        # without knowing how they were ordered on read.
        row_of = {s: r for r, s in enumerate(shotnums)}
        for label, shots in groups.items():
            rows = [row_of[s] for s in shots if s in row_of]
            if not rows:
                continue
            slot = acc[label]
            slot[0][i], slot[1][i], slot[2][i] = _reduce_group(stack, rows, idxs)

    # (npos, nframes) -> (nframes, ny, nx), the same acquisition-order reshape
    # the static map uses.
    def to_plane(a):
        return a.reshape(ny, nx, nframes).transpose(2, 0, 1)

    return ({label: tuple(to_plane(a) for a in arrays)
             for label, arrays in acc.items()}, xpos, ypos)


# ======================================================================================
# Plane assembly
# ======================================================================================

def _position_shotnums(positions, npos, nshot, mismatch):
    """Yield, per planned position index 0..npos-1, the list of global 1-based
    shot_nums for that position.

    Fast path (``mismatch`` False): shots are contiguous in setup order, so
    position ``i`` owns shots ``i*nshot + 1 .. i*nshot + nshot``. Fallback path
    (``mismatch`` True): assign each recorded shot to its nearest planned (x, y)
    by minimum squared distance (vectorized over all shots at once) -- robust to
    skipped shots.
    """
    if not mismatch:
        for i in range(npos):
            base = i * nshot
            yield i, [base + k + 1 for k in range(nshot)]
        return

    # Fallback: build planned (x, y) per position from the setup array, then map
    # each recorded shot to the nearest planned position.
    info = next(iter(positions.values()))
    setup = info["setup_array"]
    planned = np.column_stack((np.asarray(setup["x"], dtype=float),
                               np.asarray(setup["y"], dtype=float)))  # (npos, 2)
    rec = info.get("positions_array")
    buckets = {i: [] for i in range(npos)}
    if rec is not None:
        sn = np.asarray(rec["shot_num"])
        valid = sn != 0  # skip unset/padding rows
        shots = sn[valid].astype(int)
        rx = np.asarray(rec["x"], dtype=float)[valid]
        ry = np.asarray(rec["y"], dtype=float)[valid]
        # Nearest planned position for every recorded shot at once: squared
        # distance from each (rx, ry) to all planned points via broadcasting,
        # then argmin along the planned axis. Replaces the per-shot Python scan.
        d2 = (rx[:, None] - planned[:, 0]) ** 2 + (ry[:, None] - planned[:, 1]) ** 2
        nearest = np.argmin(d2, axis=1)
        for s, i in zip(shots.tolist(), nearest.tolist()):
            buckets[i].append(s)
    for i in range(npos):
        yield i, buckets[i]


def _load_stack(f, scope, ch, shotnums, tarr, med_size, gauss_sigma):
    """Read + filter the given shots into a ``(nshot, nsamples)`` stack.

    The raw shots are read in one pass via ``read_hdf5_scope_channel_shots`` (the
    channel's WAVEDESC is decoded once, not per shot); missing/skipped/length-
    mismatched shots come back as NaN rows. Each non-NaN row is then filtered.
    Reducers are NaN-aware. Returns None if no shot could be read at all.
    """
    raw, _dt, _t0 = read_hdf5_scope_channel_shots(
        f, scope, ch, shotnums, expected_len=len(tarr))
    if raw is None:
        return None
    if not (med_size and med_size > 1) and not (gauss_sigma and gauss_sigma > 0):
        return raw   # both stages disabled: the row scan and vstack would only
                     # rebuild an array identical to the one just read
    rows = [row if np.isnan(row).all() else _filter_trace(row, med_size, gauss_sigma)
            for row in raw]
    return np.vstack(rows)


def _load_signal_stack(f, scope, signal, shotnums, tarr, med_size, gauss_sigma):
    """Read every channel the ``signal`` needs and combine them into one stack.

    Returns a ``(nshot, nsamples)`` array of the *derived* quantity, or None if
    any required channel could not be read. ``signal.combine`` runs here, per
    shot, before any averaging -- see :mod:`read_and_analyze.signals` for why
    that ordering is required for nonlinear signals.
    """
    stacks = {}
    for ch in signal.channels:
        stack = _load_stack(f, scope, ch, shotnums, tarr, med_size, gauss_sigma)
        if stack is None:
            return None
        stacks[ch] = stack
    return np.asarray(signal.combine(stacks), dtype=float)


def build_plane(f, scope, ch, positions, reduce_fn, med_size, gauss_sigma):
    """Reduce every planned position to one scalar and reshape onto the plane.

    Reads each position's repeat-shot stack in acquisition order, applies
    ``reduce_fn(stack, tarr, pos_idx)``, and reshapes the per-position values to
    ``(ny, nx)``. Returns ``(Z, xpos, ypos)``; or ``(None, None, None)`` if the
    run has no setup array or is not a 2D plane.
    """
    xpos, ypos, npos, _name = _plane_axes(positions)
    if not _is_plane(xpos, ypos):
        return None, None, None
    nx, ny = len(xpos), len(ypos)

    tarr, nshot, mismatch = _grid_layout(f, scope, npos)

    vals = np.full(npos, np.nan, dtype=float)
    for i, shotnums in tqdm(_position_shotnums(positions, npos, nshot, mismatch),
                            total=npos, desc=f"reduce {scope}/{ch}", unit="pos"):
        stack = _load_stack(f, scope, ch, shotnums, tarr, med_size, gauss_sigma)
        vals[i] = reduce_fn(stack, tarr, i)

    return vals.reshape((ny, nx)), xpos, ypos


def build_planes_step(f, scope, ch, positions, t_steps_ms, shot_index,
                      med_size, gauss_sigma):
    """Build one plane per snapshot time in a single read pass.

    Thin wrapper over :func:`build_frames` with the single-shot selector, kept
    for the static ``step``-mode montage. Returns ``(Zs, xpos, ypos, t_los)``
    where ``Zs`` is a list of ``(ny, nx)`` arrays parallel to ``t_los`` (realized
    tarr-snapped times in seconds); or ``(None, None, None, None)`` if not a plane.
    """
    tarr = read_hdf5_scope_tarr(f, scope)
    idxs, t_los = _step_indices(tarr, t_steps_ms)
    frames, xpos, ypos = build_frames(
        f, scope, raw_signal(ch), positions, idxs,
        select_shot_index(shot_index), med_size, gauss_sigma)
    if xpos is None:
        return None, None, None, None
    if not frames:   # no position yielded that shot index
        ny, nx = len(ypos), len(xpos)
        return [np.full((ny, nx), np.nan) for _ in idxs], xpos, ypos, t_los
    values = next(iter(frames.values()))[0]      # (nframes, ny, nx)
    return [values[k] for k in range(len(idxs))], xpos, ypos, t_los


# ======================================================================================
# Rendering   (origin='upper': reshape row 0 = max-y; contour on meshgrid aligns)
# ======================================================================================

def _grid_shape(n):
    """Rows x cols for ``n`` panels wrapped into a roughly square grid."""
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _draw_plane(ax, Z, xpos, ypos, cmap, show_contour, vmin=None, vmax=None):
    """Draw one plane onto ``ax`` (imshow + optional aligned contour). Returns the
    image handle for the colorbar."""
    extent = (xpos.min(), xpos.max(), ypos.min(), ypos.max())
    im = ax.imshow(Z, origin="upper", extent=extent, aspect="auto", cmap=cmap,
                   interpolation="nearest", vmin=vmin, vmax=vmax)
    if show_contour:
        X, Y = np.meshgrid(xpos, ypos)
        masked = np.ma.masked_invalid(Z)
        ax.contour(X, Y, masked, levels=N_CONTOURS, colors="white",
                   alpha=0.4, linewidths=0.5)
    ax.set_xlabel("probe x (mm)")
    ax.set_ylabel("probe y (mm)")
    return im


def _render_map(plt, Z, xpos, ypos, scope, ch, label, cmap, show_contour,
                shot_index, path):
    """Draw the single-plane ``range``/``step``-scalar figure."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = _draw_plane(ax, Z, xpos, ypos, cmap, show_contour)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(f"scope '{scope}' / {ch}: {label}  (shot {shot_index})",
                 fontsize=10, loc="left")
    fig.suptitle(f"{os.path.basename(path)}  —  XY map", fontsize=10)
    fig.tight_layout()


def _render_step_montage(plt, Zs, xpos, ypos, t_los, scope, ch, cmap,
                         show_contour, shot_index, path):
    """Draw the ``step``-mode montage: one plane per snapshot time, wrapped into a
    roughly square grid, sharing a common color scale and one colorbar."""
    n = len(Zs)
    finite = np.concatenate([Z[np.isfinite(Z)].ravel() for Z in Zs]) if n else np.array([])
    vmin = float(finite.min()) if finite.size else None
    vmax = float(finite.max()) if finite.size else None

    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    im = None
    for k, (Z, t_lo) in enumerate(zip(Zs, t_los)):
        ax = flat[k]
        im = _draw_plane(ax, Z, xpos, ypos, cmap, show_contour, vmin=vmin, vmax=vmax)
        ax.set_title(f"t={t_lo * 1e3:.4f} ms", fontsize=9, loc="left")

    for ax in flat[n:]:   # hide any unused cells in the wrapped grid
        ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes, label="V", shrink=0.9)
    fig.suptitle(f"{os.path.basename(path)}  —  scope '{scope}' / {ch}: "
                 f"XY map (step, shot {shot_index})", fontsize=10)


# ======================================================================================
# Driver
# ======================================================================================

def plot_xy_map(path, scope=None, channels=None, mode=None,
                t_start=None, t_end=None, t_step=None, shot_index=None,
                med_size=None, gauss_sigma=None,
                show_contour=None, cmap=None, show=None, save=None):
    """Render a plane-only XY map per (scope, channel).

    For each scope/channel: pick one shot per position by ``shot_index``, reduce it
    in time (``range`` -> mean over [t_start, t_end] ms; ``step`` -> snapshot
    montage at the ``t_step`` time(s)), reshape onto the plane, and imshow with an
    optional aligned contour. Line scans are skipped. Honors SHOW_PLOT/SAVE_PLOT
    (override with show/save); saves one PNG per (scope, channel). Returns the
    saved paths.
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

    with open_hdf5_readonly(path) as f:
        positions = read_positions(f)
        if not positions:
            print("no /Control/Positions data (not a bmotion file?) — nothing to map")
            return saved
        scopes = [scope] if scope else _scope_groups(f)

        for sc in scopes:
            xpos, ypos, _npos, _name = _plane_axes(positions)
            if not _is_plane(xpos, ypos):
                nx = 0 if xpos is None else len(xpos)
                ny = 0 if ypos is None else len(ypos)
                print(f"scope '{sc}': grid is {nx}x{ny} (a line) — "
                      f"plot_xy_map only supports planes; skipping")
                continue

            sg = f[sc]
            shot_nums = _shot_numbers(sg)
            chans = channels if channels else _channel_names(sg, shot_nums[0])

            for ch in chans:
                if mode == "step":
                    t_steps = _as_step_list(t_step)
                    Zs, xp, yp, t_los = build_planes_step(
                        f, sc, ch, positions, t_steps, shot_index,
                        med_size, gauss_sigma)
                    if Zs is None or all(np.all(np.isnan(Z)) for Z in Zs):
                        print(f"scope '{sc}' / {ch}: no usable shots — skipping")
                        continue
                    _render_step_montage(plt, Zs, xp, yp, t_los, sc, ch, cmap,
                                         show_contour, shot_index, path)
                else:
                    Z, xp, yp = build_plane(
                        f, sc, ch, positions,
                        make_single_shot_reduce(shot_index, mode, t_start, t_end, t_step),
                        med_size, gauss_sigma)
                    if Z is None or np.all(np.isnan(Z)):
                        print(f"scope '{sc}' / {ch}: no usable shots — skipping")
                        continue
                    tarr = read_hdf5_scope_tarr(f, sc)
                    i0, i1 = _reduction_indices(tarr, mode, t_start, t_end, t_step)
                    label = _reduction_label(mode, float(tarr[i0]), float(tarr[i1 - 1]))
                    _render_map(plt, Z, xp, yp, sc, ch, label, cmap,
                                show_contour, shot_index, path)

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


def main():
    plot_xy_map(resolve_data_file())


if __name__ == "__main__":
    main()
