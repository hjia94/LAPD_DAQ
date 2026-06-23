# -*- coding: utf-8 -*-
"""
Filtering pipeline for fluctuation analysis, plus a sample-trace verification view.

This module owns the time-domain denoising used across the fluctuation analysis:
a median filter (spike/outlier removal) followed by a Gaussian (residual
high-freq smoothing). It also groups repeat shots by grid position and provides
``load_filtered_traces`` -- the in-memory handoff that the analysis module
(``fluctuation_analysis``) imports.

Run it to inspect the effect of each filtering stage on a sample trace:
    python -m read_and_analyze.filter_data
The figure overlays the raw trace, the trace after the median filter, and the
trace after median+gaussian, so the contribution of each stage is visible.

Reading/decoding is delegated to the in-repo ``scope_io`` package; position/shot
helpers are reused from the sibling :mod:`read_and_analyze.read_bmotion_data`.

Setup (once):  python -m pip install scipy

Created May.2026
@author: Jia Han
"""

import os

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

# Allow running directly (IDE "Run" button / from inside this folder) as well as
# ``python -m read_and_analyze.<module>`` from the repo root: the root-level
# ``scope_io``/``acquisition`` packages need the repo root on sys.path, which ``-m``
# adds but a direct script run does not, so put it there ourselves.
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scope_io import (
    read_hdf5_scope_channel_shots, read_hdf5_scope_data, read_hdf5_scope_tarr,
)
try:  # works as a package (python -m read_and_analyze.filter_data)
    from read_and_analyze.read_bmotion_data import (
        read_positions, build_positions_index,
        _scope_groups, _shot_numbers, _channel_names, resolve_data_file,
    )
except ImportError:  # fallback when run directly from inside the folder
    from read_bmotion_data import (
        read_positions, build_positions_index,
        _scope_groups, _shot_numbers, _channel_names, resolve_data_file,
    )

# --------------------------------------------------------------------------------------
# Knobs (no CLI). User-changeable values live in analysis_config.py; they are
# imported here and re-exported under the historical names so the sibling
# analysis modules can keep doing `from filter_data import SCOPE, MED_SIZE, ...`.
# --------------------------------------------------------------------------------------
try:  # works as a package (python -m read_and_analyze.filter_data)
    from read_and_analyze.analysis_config import (
        MED_SIZE, GAUSS_SIGMA, POS_TOL as _POS_TOL,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
    )
except ImportError:  # fallback when run directly from inside the folder
    from analysis_config import (
        MED_SIZE, GAUSS_SIGMA, POS_TOL as _POS_TOL,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
    )


# ======================================================================================
# Filtering
# ======================================================================================

def _filter_trace(volts, med_size, gauss_sigma):
    """Denoise one trace along time: median filter first (removes spikes/outliers),
    then a Gaussian (smooths residual high-freq noise). A med_size of 1 or a
    gauss_sigma of 0 disables that stage."""
    v = np.asarray(volts, dtype=float)
    if med_size and med_size > 1:
        v = median_filter(v, size=int(med_size))
    if gauss_sigma and gauss_sigma > 0:
        v = gaussian_filter1d(v, gauss_sigma)
    return v


def _as_list(channels):
    """Accept None, a single channel string, or a sequence; return None or a list."""
    if channels is None:
        return None
    if isinstance(channels, str):
        return [channels]
    return list(channels)


def load_filtered_traces(f, scope, ch, shots, tarr, med_size, gauss_sigma):
    """Load and denoise the repeat-shot traces for one (scope, channel, position).

    Returns a list of filtered 1-D arrays — one per usable shot — skipping shots
    that cannot be read or whose length does not match ``tarr``. Callers stack
    these and decide for themselves whether there are enough shots to proceed.
    This is the in-memory handoff surface consumed by ``fluctuation_analysis``.

    The raw shots are read in one pass (the channel's WAVEDESC is decoded once,
    not per shot); unreadable/skipped/length-mismatched shots come back as NaN
    rows and are dropped here to preserve the "usable shots only" contract.
    """
    raw, _dt, _t0 = read_hdf5_scope_channel_shots(
        f, scope, ch, shots, expected_len=len(tarr))
    if raw is None:
        return []
    return [_filter_trace(row, med_size, gauss_sigma)
            for row in raw if not np.isnan(row).all()]


class FilteredTraceCache:
    """Memoize :func:`load_filtered_traces` over one analysis pass.

    The fluctuation pipeline reads + filters the *same* (scope, channel,
    position) repeat-shot stack several times per run -- once to pick the quiet
    window, again to build the spatial profile, again to plot -- each call
    re-reading from HDF5 and re-running the median/Gaussian filters. Wrapping the
    open file plus the two filter knobs in this object and caching on
    ``(scope, ch, shots)`` collapses those to a single read+filter per position.

    Holds only ``f``, ``med_size``, ``gauss_sigma`` and its result dict (not any
    enclosing scope), so it can be discarded with the ``with h5py.File(...)``
    block that owns ``f``. Construct one per pass; do not retain it past the file.
    """

    def __init__(self, f, med_size, gauss_sigma):
        self.f = f
        self.med_size = med_size
        self.gauss_sigma = gauss_sigma
        self._cache = {}

    def get(self, scope, ch, shots, tarr):
        """Cached ``load_filtered_traces`` for one (scope, ch, position)."""
        key = (scope, ch, tuple(shots))
        rows = self._cache.get(key)
        if rows is None:
            rows = load_filtered_traces(
                self.f, scope, ch, shots, tarr, self.med_size, self.gauss_sigma)
            self._cache[key] = rows
        return rows


# ======================================================================================
# Grouping
# ======================================================================================

def _shots_by_position(f, scope, positions):
    """Map each grid position to its non-skipped repeat shots.

    Returns ``{(x, y): [shot_num, ...]}`` with (x, y) rounded to ``_POS_TOL`` so
    repeat shots at the same nominal position group together.
    """
    sg = f[scope]
    pos_index = build_positions_index(positions)  # O(1) per-shot lookups below
    groups = {}
    for s in _shot_numbers(sg):
        shot = sg.get(f"shot_{s}")
        if shot is not None and shot.attrs.get("skipped", False):
            continue
        pos = pos_index.get(s)
        if pos is None:
            continue
        key = (round(pos[0] / _POS_TOL) * _POS_TOL, round(pos[1] / _POS_TOL) * _POS_TOL)
        groups.setdefault(key, []).append(s)
    return groups


# ======================================================================================
# Sample-trace plot (filtering verification)
# ======================================================================================

def plot_sample_traces(path, scope=None, channels=None,
                       med_size=None, gauss_sigma=None, show=None, save=None):
    """Show the effect of each filtering stage on a sample trace.

    Shows up to three positions spread across x (first / middle / last usable),
    one per panel. Each panel overlays three traces vs time for one
    representative shot — the raw trace, the trace after the median filter, and
    the trace after median+gaussian — using one color per channel with
    increasing alpha so the final filtered trace reads on top. Honors
    SHOW_PLOT/SAVE_PLOT (override with
    show/save). Saves one PNG per scope to a ``plots/`` subdir next to the data
    file. Returns the saved paths.
    """
    import h5py
    import matplotlib.pyplot as plt

    scope = SCOPE if scope is None else scope
    channels = CHANNELS if channels is None else channels
    med_size = MED_SIZE if med_size is None else med_size
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    show = SHOW_PLOT if show is None else show
    save = SAVE_PLOT if save is None else save
    channels = _as_list(channels)

    saved = []
    if save:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "plots")
        os.makedirs(plots_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    with h5py.File(path, "r") as f:
        positions = read_positions(f)
        scopes = [scope] if scope else _scope_groups(f)

        for sc in scopes:
            sg = f[sc]
            tarr = read_hdf5_scope_tarr(f, sc)
            by_pos = _shots_by_position(f, sc, positions)
            shot_nums = _shot_numbers(sg)
            chans = channels if channels else _channel_names(sg, shot_nums[0])

            # Pick up to 3 positions spread across x (first / middle / last of the
            # usable ones), so the panels span the probe range rather than crowd
            # one spot.
            usable = [((x, y), shots) for (x, y), shots in sorted(by_pos.items()) if shots]
            if not usable:
                print(f"scope '{sc}': no usable shots to sample — skipping")
                continue
            if len(usable) >= 3:
                picks = [usable[0], usable[len(usable) // 2], usable[-1]]
            else:
                picks = usable

            fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3 * len(picks)),
                                     sharex=True, squeeze=False)
            axes = axes[:, 0]
            for ax, ((x, y), shots) in zip(axes, picks):
                for ci, ch in enumerate(chans):
                    # One representative shot per channel: the first that reads cleanly.
                    raw = None
                    for s in shots:
                        try:
                            volts, _dt, _t0 = read_hdf5_scope_data(f, sc, ch, s)
                        except Exception:
                            continue
                        if len(volts) != len(tarr):
                            continue
                        raw = np.asarray(volts, dtype=float)
                        shot_used = s
                        break
                    if raw is None:
                        continue

                    med = median_filter(raw, size=int(med_size)) if med_size and med_size > 1 else raw
                    full = _filter_trace(raw, med_size, gauss_sigma)

                    # Overlay all three stages on the same panel, one color per channel,
                    # increasing alpha so the final filtered trace reads on top.

                    pre = f"{ch} (shot {shot_used})"
                    ax.plot(tarr * 1e3, raw, lw=0.6, color='r', alpha=0.25,
                            label=f"{pre} raw")
                    ax.plot(tarr * 1e3, med, lw=0.7, color='g', alpha=0.5,
                            label=f"{pre} +median")
                    ax.plot(tarr * 1e3, full, lw=1.1, color='b', alpha=1.0,
                            label=f"{pre} +median+gaussian")

                ax.set_title(f"x={x:.1f}, y={y:.1f}", fontsize=10, loc="left")
                ax.set_ylabel("V")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel("time (ms)")

            fig.suptitle(
                f"{os.path.basename(path)}  —  scope '{sc}' raw vs filtered "
                f"(median size={med_size:g}, gaussian sigma={gauss_sigma:g} samples)",
                fontsize=10)
            fig.tight_layout()

            if save:
                out_png = os.path.join(plots_dir, f"{base}_{sc}_filter_samples.png")
                fig.savefig(out_png, dpi=150)
                saved.append(out_png)
                print(f"Saved plot: {out_png}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return saved


if __name__ == "__main__":
    plot_sample_traces(resolve_data_file())
