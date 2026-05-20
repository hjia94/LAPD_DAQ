# -*- coding: utf-8 -*-
"""
Find, per probe position, the time window with the least fluctuation.

For each grid position the repeat shots are grouped, smoothed in time with a
Gaussian filter (high-frequency noise removal), then a fixed-width window is slid
across the record to find where the signal is both:
  * flat in time          -> small (max-min)/|mean| of the per-position mean trace
  * reproducible per shot  -> small std-across-shots / |mean| of the window mean
restricted to windows where there is actual signal (|mean| above a fraction of
the position's peak). The two relative terms are summed into a score; the
lowest-score window wins for each (scope, channel, position).

Reading/decoding is delegated to ``lab_scopes.io.hdf5``; position/shot helpers
are reused from the sibling :mod:`read_and_analyze.read_bmotion_data`.

There is NO command line; all knobs are the constants below. Run with:
    python -m read_and_analyze.analysis_fluctuation
Edit DEFAULT_FILE / the constants to change the file or parameters.

Setup (once):  python -m pip install scipy

Created May.2026
@author: Jia Han
"""

import math
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d

from lab_scopes.io.hdf5 import read_hdf5_scope_data, read_hdf5_scope_tarr
from read_and_analyze.read_bmotion_data import (
    read_positions, _position_for_shot, _scope_groups, _shot_numbers, _channel_names,
)

# --------------------------------------------------------------------------------------
# Knobs (no CLI — edit here)
# --------------------------------------------------------------------------------------
DEFAULT_FILE = r"D:\data\LAPD\00-LP-p21p29p41-Xline-test_2026-05-19.hdf5"
SCOPE        = None    # None = all scopes; or e.g. "lpscope"
CHANNELS     = None    # None = all channels; or e.g. ["C1", "C3"]
WINDOW_US    = 10.0    # analysis window width (microseconds)
GAUSS_SIGMA  = 5.0     # Gaussian time-smoothing width in SAMPLES (high-freq noise removal)
SIGNAL_FRAC  = 0.2     # window mean must exceed this fraction of the position's peak |mean|
SHOW_PLOT    = True    # display the figure interactively
SAVE_PLOT    = True    # write a PNG to a "plots/" subdir next to the data file

_POS_TOL = 0.5         # round (x, y) to this (mm) so encoder float noise groups cleanly


# ======================================================================================
# Grouping
# ======================================================================================

def _shots_by_position(f, scope, positions):
    """Map each grid position to its non-skipped repeat shots.

    Returns ``{(x, y): [shot_num, ...]}`` with (x, y) rounded to ``_POS_TOL`` so
    repeat shots at the same nominal position group together.
    """
    sg = f[scope]
    groups = {}
    for s in _shot_numbers(sg):
        shot = sg.get(f"shot_{s}")
        if shot is not None and shot.attrs.get("skipped", False):
            continue
        pos = _position_for_shot(positions, s)
        if pos is None:
            continue
        key = (round(pos[0] / _POS_TOL) * _POS_TOL, round(pos[1] / _POS_TOL) * _POS_TOL)
        groups.setdefault(key, []).append(s)
    return groups


# ======================================================================================
# Core analysis
# ======================================================================================

def find_quiet_window(path, scope=None, channels=None,
                      window_us=None, gauss_sigma=None, signal_frac=None):
    """Find the least-fluctuating window per (scope, channel, position).

    Parameters default to the module constants. Returns a list of record dicts
    (one per scope/channel/position that has a valid window), sorted by ``score``
    ascending (best first). Each record has keys: ``scope, channel, x, y,
    n_shots, t_center, t_start, t_end, flatness_rel, scatter_rel, score,
    window_mean``.
    """
    import h5py

    scope = SCOPE if scope is None else scope
    channels = CHANNELS if channels is None else channels
    window_us = WINDOW_US if window_us is None else window_us
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    signal_frac = SIGNAL_FRAC if signal_frac is None else signal_frac

    records = []
    with h5py.File(path, "r") as f:
        positions = read_positions(f)
        scopes = [scope] if scope else _scope_groups(f)

        for sc in scopes:
            sg = f[sc]
            tarr = read_hdf5_scope_tarr(f, sc)
            dt = float(tarr[1] - tarr[0])
            w = max(2, math.ceil(window_us * 1e-6 / dt))  # window length in samples

            by_pos = _shots_by_position(f, sc, positions)
            shot_nums = _shot_numbers(sg)
            chans = channels if channels else _channel_names(sg, shot_nums[0])

            for ch in chans:
                for (x, y), shots in sorted(by_pos.items()):
                    rec = _best_window_for_position(
                        f, sc, ch, x, y, shots, tarr, w, gauss_sigma, signal_frac)
                    if rec is not None:
                        records.append(rec)

    records.sort(key=lambda r: r["score"])
    return records


def _best_window_for_position(f, scope, ch, x, y, shots, tarr, w, gauss_sigma, signal_frac):
    """Slide a length-``w`` window over one position's shots; return the best record."""
    # Stack the repeat-shot traces (filtered along time), dropping length mismatches.
    rows = []
    for s in shots:
        try:
            volts, _dt, _t0 = read_hdf5_scope_data(f, scope, ch, s)
        except Exception:
            continue
        if len(volts) != len(tarr):
            continue
        rows.append(gaussian_filter1d(np.asarray(volts, dtype=float), gauss_sigma))
    if len(rows) < 2:  # need >= 2 shots to measure shot-to-shot scatter
        return None

    traces = np.vstack(rows)              # (n_shots, N), filtered
    mean_trace = traces.mean(axis=0)
    peak = float(np.max(np.abs(mean_trace)))
    if peak == 0.0:
        return None
    floor = signal_frac * peak

    n = mean_trace.size
    stride = max(1, w // 2)
    best = None
    for start in range(0, n - w + 1, stride):
        seg = mean_trace[start:start + w]
        window_mean = float(seg.mean())
        if abs(window_mean) < floor:       # signal mask: skip quiet/zero regions
            continue
        flatness_rel = float(seg.max() - seg.min()) / abs(window_mean)
        m_i = traces[:, start:start + w].mean(axis=1)   # per-shot window means
        scatter_rel = float(np.std(m_i)) / abs(window_mean)
        score = flatness_rel + scatter_rel
        if best is None or score < best["score"]:
            best = {
                "scope": scope, "channel": ch, "x": x, "y": y,
                "n_shots": traces.shape[0],
                "t_start": float(tarr[start]),
                "t_end": float(tarr[start + w - 1]),
                "t_center": float(tarr[start + w // 2]),
                "flatness_rel": flatness_rel,
                "scatter_rel": scatter_rel,
                "score": score,
                "window_mean": window_mean,
            }
    return best


def _profile_value(f, scope, ch, shots, tarr, i0, i1, gauss_sigma):
    """Mean filtered signal over the fixed sample window [i0:i1], averaged across
    a position's repeat shots. Returns None if no usable shots."""
    vals = []
    for s in shots:
        try:
            volts, _dt, _t0 = read_hdf5_scope_data(f, scope, ch, s)
        except Exception:
            continue
        if len(volts) != len(tarr):
            continue
        vf = gaussian_filter1d(np.asarray(volts, dtype=float), gauss_sigma)
        vals.append(float(vf[i0:i1].mean()))
    if not vals:
        return None
    return float(np.mean(vals))


# ======================================================================================
# Reporting
# ======================================================================================

def _print_table(records):
    """Print the per-position results, best (lowest score) first."""
    print("=" * 88)
    print("LEAST-FLUCTUATION WINDOW PER POSITION  (lower score = flatter & more reproducible)")
    print(f"window={WINDOW_US:g} us   gauss_sigma={GAUSS_SIGMA:g} samples   "
          f"signal_frac={SIGNAL_FRAC:g}")
    print("-" * 88)
    if not records:
        print("(no valid windows — signal never exceeded the floor at any position)")
        print("=" * 88)
        return
    hdr = f"{'scope':<8} {'ch':<4} {'x':>7} {'y':>6} {'shots':>5} " \
          f"{'t_ctr(ms)':>10} {'flat_rel':>9} {'scat_rel':>9} {'score':>8} {'mean(V)':>9}"
    print(hdr)
    for r in records:
        print(f"{r['scope']:<8} {r['channel']:<4} {r['x']:>7.1f} {r['y']:>6.1f} "
              f"{r['n_shots']:>5d} {r['t_center']*1e3:>10.3f} "
              f"{r['flatness_rel']:>9.4f} {r['scatter_rel']:>9.4f} "
              f"{r['score']:>8.4f} {r['window_mean']:>9.4g}")
    print("-" * 88)
    b = records[0]
    print(f"BEST: {b['scope']}/{b['channel']} @ x={b['x']:.1f}, y={b['y']:.1f}  "
          f"window {b['t_start']*1e3:.3f}..{b['t_end']*1e3:.3f} ms  score={b['score']:.4f}")
    print("=" * 88)


# ======================================================================================
# Plotting
# ======================================================================================

def plot_quiet_window(path, scope=None, channels=None,
                      window_us=None, gauss_sigma=None, signal_frac=None,
                      show=None, save=None):
    """Plot the fluctuation analysis: score-vs-position and the best-window overlay.

    Honors SHOW_PLOT/SAVE_PLOT (override with show/save). Saves one PNG per scope
    to a ``plots/`` subdir next to the data file. Returns the saved paths.
    """
    import h5py
    import matplotlib.pyplot as plt

    scope = SCOPE if scope is None else scope
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    show = SHOW_PLOT if show is None else show
    save = SAVE_PLOT if save is None else save

    records = find_quiet_window(path, scope=scope, channels=channels,
                                window_us=window_us, gauss_sigma=gauss_sigma,
                                signal_frac=signal_frac)

    saved = []
    if save:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "plots")
        os.makedirs(plots_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    # Group records by scope so each scope gets its own figure.
    by_scope = {}
    for r in records:
        by_scope.setdefault(r["scope"], []).append(r)

    with h5py.File(path, "r") as f:
        positions = read_positions(f)
        for sc, recs in by_scope.items():
            fig, (ax_score, ax_profile, ax_trace) = plt.subplots(3, 1, figsize=(10, 10))

            # --- top: score (and components) vs x position, per channel ---
            for ch in sorted({r["channel"] for r in recs}):
                cr = sorted((r for r in recs if r["channel"] == ch), key=lambda r: r["x"])
                xs = [r["x"] for r in cr]
                ax_score.plot(xs, [r["score"] for r in cr], "o-", lw=1, label=f"{ch} score")
                ax_score.plot(xs, [r["flatness_rel"] for r in cr], ".:", lw=0.7, alpha=0.6,
                              label=f"{ch} flat")
                ax_score.plot(xs, [r["scatter_rel"] for r in cr], ".--", lw=0.7, alpha=0.6,
                              label=f"{ch} scatter")
            ax_score.set_xlabel("probe x (mm)")
            ax_score.set_ylabel("relative fluctuation")
            ax_score.set_title(f"scope '{sc}': fluctuation vs position "
                               f"(window={WINDOW_US:g} us)", fontsize=10, loc="left")
            # Score definition (LaTeX via mathtext): flatness-in-time + scatter-across-shots.
            ax_score.text(
                0.5, 0.97,
                r"$\mathrm{score}=\dfrac{\max_t\,\bar V-\min_t\,\bar V}{|\langle \bar V\rangle|}"
                r"+\dfrac{\sigma_{\mathrm{shots}}}{|\langle \bar V\rangle|}$",
                transform=ax_score.transAxes, ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
            ax_score.legend(fontsize=7, ncol=3)
            ax_score.grid(alpha=0.3)

            # The overall best window defines ONE fixed time slice used for the
            # spatial profile, so every position is averaged over the same window
            # (a spatial gradient is only meaningful at a common time).
            best = min(recs, key=lambda r: r["score"])
            by_pos = _shots_by_position(f, sc, positions)
            tarr = read_hdf5_scope_tarr(f, sc)
            i0 = int(np.searchsorted(tarr, best["t_start"]))
            i1 = int(np.searchsorted(tarr, best["t_end"], side="right"))

            # --- middle: spatial profile at the fixed window -> spatial gradient ---
            for ch in sorted({r["channel"] for r in recs}):
                cr = sorted((r for r in recs if r["channel"] == ch), key=lambda r: r["x"])
                xs, vals = [], []
                for r in cr:
                    key = (round(r["x"] / _POS_TOL) * _POS_TOL,
                           round(r["y"] / _POS_TOL) * _POS_TOL)
                    v = _profile_value(f, sc, ch, by_pos.get(key, []), tarr,
                                       i0, i1, gauss_sigma)
                    if v is not None:
                        xs.append(r["x"])
                        vals.append(v)
                ax_profile.plot(xs, vals, "o-", lw=1, ms=3, label=ch)
            ax_profile.set_xlabel("probe x (mm)")
            ax_profile.set_ylabel("window-mean signal (V)")
            ax_profile.set_title(
                f"spatial profile: signal averaged over fixed window "
                f"{best['t_start']*1e3:.3f}..{best['t_end']*1e3:.3f} ms "
                f"(from best {best['channel']})", fontsize=10, loc="left")
            ax_profile.axhline(0, color="0.6", lw=0.6)
            ax_profile.legend(fontsize=8)
            ax_profile.grid(alpha=0.3)

            # --- bottom: best (lowest-score) position, repeat shots + shaded window ---
            key = (round(best["x"] / _POS_TOL) * _POS_TOL,
                   round(best["y"] / _POS_TOL) * _POS_TOL)
            for s in by_pos.get(key, []):
                try:
                    volts, _dt, _t0 = read_hdf5_scope_data(f, sc, best["channel"], s)
                except Exception:
                    continue
                if len(volts) != len(tarr):
                    continue
                vf = gaussian_filter1d(np.asarray(volts, dtype=float), gauss_sigma)
                ax_trace.plot(tarr * 1e3, vf, lw=0.7, alpha=0.7)
            ax_trace.axvspan(best["t_start"] * 1e3, best["t_end"] * 1e3,
                             color="orange", alpha=0.3, label="chosen window")
            ax_trace.set_xlabel("time (ms)")
            ax_trace.set_ylabel("V (filtered)")
            ax_trace.set_title(
                f"best: {best['channel']} @ x={best['x']:.1f}, y={best['y']:.1f}  "
                f"score={best['score']:.4f}  ({best['n_shots']} shots overlaid)",
                fontsize=10, loc="left")
            ax_trace.legend(fontsize=8)
            ax_trace.grid(alpha=0.3)

            fig.suptitle(f"{os.path.basename(path)}  —  fluctuation analysis", fontsize=10)
            fig.tight_layout()

            if save:
                out_png = os.path.join(plots_dir, f"{base}_{sc}_fluctuation.png")
                fig.savefig(out_png, dpi=150)
                saved.append(out_png)
                print(f"Saved plot: {out_png}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return saved


if __name__ == "__main__":
    records = find_quiet_window(DEFAULT_FILE)
    _print_table(records)
    if SHOW_PLOT or SAVE_PLOT:
        plot_quiet_window(DEFAULT_FILE)
