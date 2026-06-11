# -*- coding: utf-8 -*-
"""
Find, per probe position, the time window with the least fluctuation.

For each grid position the repeat shots are grouped and denoised in time (see
:mod:`read_and_analyze.filter_data`), then a fixed-width window is slid across
the record to find where the signal is both:
  * flat in time          -> small (max-min)/|mean| of the per-position mean trace
  * reproducible per shot  -> small std-across-shots / |mean| of the window mean
restricted to windows where there is actual signal (|mean| above a fraction of
the position's peak). The two relative terms are summed into a score; the
lowest-score window wins for each (scope, channel, position).

Filtering, shot grouping, and trace loading are imported from
:mod:`read_and_analyze.filter_data`; reading/decoding is delegated to
``lab_scopes.io.hdf5``.

There is NO command line; all knobs are the constants below (filtering knobs
live in ``filter_data``). Run with:
    python -m read_and_analyze.fluctuation_analysis
Edit DATA_FILE/DATA_DIR and the constants in analysis_config.py to change the
file (or auto-pick the newest completed run) and parameters.

Setup (once):  python -m pip install scipy

Created May.2026
@author: Jia Han
"""

import math
import os

import numpy as np

from lab_scopes.io.hdf5 import read_hdf5_scope_tarr
try:  # works as a package (python -m read_and_analyze.fluctuation_analysis)
    from read_and_analyze.read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
        resolve_data_file,
    )
    from read_and_analyze.filter_data import (
        _filter_trace, _as_list, _shots_by_position, load_filtered_traces,
    )
    from read_and_analyze.analysis_config import (
        MED_SIZE, GAUSS_SIGMA, POS_TOL as _POS_TOL,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
        FLUCT_WINDOW_US as WINDOW_US, FLUCT_SIGNAL_FRAC as SIGNAL_FRAC,
    )
except ImportError:  # fallback when run directly from inside the folder
    from read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
        resolve_data_file,
    )
    from filter_data import (
        _filter_trace, _as_list, _shots_by_position, load_filtered_traces,
    )
    from analysis_config import (
        MED_SIZE, GAUSS_SIGMA, POS_TOL as _POS_TOL,
        SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS, SHOW_PLOT, SAVE_PLOT,
        FLUCT_WINDOW_US as WINDOW_US, FLUCT_SIGNAL_FRAC as SIGNAL_FRAC,
    )


# ======================================================================================
# Core analysis
# ======================================================================================

def find_quiet_window(path, scope=None, channels=None, window_us=None,
                      med_size=None, gauss_sigma=None, signal_frac=None):
    """Find the quiet, steep-gradient window per (scope, channel, position).

    Parameters default to the module constants. Traces are denoised with a
    median filter (``med_size``) then a Gaussian (``gauss_sigma``). The window is
    chosen by lowest shot-to-shot CV; the score is then
    ``cv_shots + 1/|(dV/dx)/V|`` so windows on a steep fractional spatial gradient
    rank better. Returns a list of record dicts (one per scope/channel/position
    that has a valid window), sorted by ``score`` ascending (best first). Each
    record has keys: ``scope, channel, x, y, n_shots, t_center, t_start, t_end,
    flatness_rel, cv_shots, grad_x, score, window_mean``.
    """
    import h5py

    scope = SCOPE if scope is None else scope
    channels = CHANNELS if channels is None else channels
    window_us = WINDOW_US if window_us is None else window_us
    med_size = MED_SIZE if med_size is None else med_size
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    signal_frac = SIGNAL_FRAC if signal_frac is None else signal_frac
    channels = _as_list(channels)

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

            scope_records = []
            for ch in chans:
                for (x, y), shots in sorted(by_pos.items()):
                    rec = _best_window_for_position(
                        f, sc, ch, x, y, shots, tarr, w, med_size, gauss_sigma, signal_frac)
                    if rec is not None:
                        scope_records.append(rec)

            # Add the spatial-gradient term, evaluated at ONE fixed reference window
            # (this scope's quietest), so |(dV/dx)/V| is comparable across positions.
            _add_gradient_term(f, sc, scope_records, by_pos, tarr,
                               med_size, gauss_sigma)
            records.extend(scope_records)

    records.sort(key=lambda r: r["score"])
    return records


def _add_gradient_term(f, scope, recs, by_pos, tarr, med_size, gauss_sigma):
    """Augment each record's score with 1/|(dV/dx)/V|, biasing toward windows on
    a steep *fractional* spatial gradient. Profiles are built at a single fixed
    reference window (the scope's quietest by ``cv_shots``) so the gradient is
    comparable across positions. Records are modified in place; ``grad_x`` stores
    the normalized gradient |(dV/dx)/V| (1/mm)."""
    if not recs:
        return
    ref = min(recs, key=lambda r: r["cv_shots"])
    i0 = int(np.searchsorted(tarr, ref["t_start"]))
    i1 = int(np.searchsorted(tarr, ref["t_end"], side="right"))

    for ch in sorted({r["channel"] for r in recs}):
        cr = sorted((r for r in recs if r["channel"] == ch), key=lambda r: r["x"])
        xs = np.array([r["x"] for r in cr], dtype=float)
        prof = []
        for r in cr:
            key = (round(r["x"] / _POS_TOL) * _POS_TOL, round(r["y"] / _POS_TOL) * _POS_TOL)
            prof.append(_profile_value(f, scope, ch, by_pos.get(key, []), tarr,
                                       i0, i1, med_size, gauss_sigma))
        prof = np.array([np.nan if v is None else v for v in prof], dtype=float)
        # Local dV/dx via finite differences (handles non-uniform x spacing),
        # normalized by the local V -> fractional gradient |(dV/dx)/V| (1/mm).
        dvdx = np.gradient(prof, xs) if len(xs) > 1 else np.zeros_like(xs)
        norm_grad = np.abs(dvdx / np.where(prof == 0, np.nan, prof))
        for r, g in zip(cr, norm_grad):
            r["grad_x"] = float(g)
            # 1/g blows up on flat regions; guard with a tiny floor.
            inv_grad = 1.0 / max(abs(g), 1e-12) if np.isfinite(g) else 1e12
            r["score"] = r["cv_shots"] + inv_grad


def _best_window_for_position(f, scope, ch, x, y, shots, tarr, w, med_size, gauss_sigma, signal_frac):
    """Slide a length-``w`` window over one position's shots; return the best record."""
    # Stack the repeat-shot traces (filtered along time), dropping length mismatches.
    rows = load_filtered_traces(f, scope, ch, shots, tarr, med_size, gauss_sigma)
    if len(rows) < 2:  # need >= 2 shots to measure shot-to-shot scatter
        return None

    traces = np.vstack(rows)              # (n_shots, N), filtered
    mean_trace = traces.mean(axis=0)
    peak = float(np.max(np.abs(mean_trace)))
    if peak == 0.0:
        return None

    starts, t_centers, cv = _cv_curve(traces, mean_trace, tarr, w)
    if cv.size == 0:
        return None

    # Window is chosen by reproducibility alone; the spatial-gradient term is
    # added afterwards (it needs the profile across positions).
    k = int(np.argmin(cv))
    start = starts[k]
    seg = mean_trace[start:start + w]
    window_mean = float(seg.mean())
    return {
        "scope": scope, "channel": ch, "x": x, "y": y,
        "n_shots": traces.shape[0],
        "t_start": float(tarr[start]),
        "t_end": float(tarr[start + w - 1]),
        "t_center": float(tarr[start + w // 2]),
        "flatness_rel": float(seg.max() - seg.min()) / abs(window_mean),
        "cv_shots": float(cv[k]),
        "grad_x": float("nan"),    # filled in after the profile is built
        "score": float(cv[k]),     # base score; gradient term added later
        "window_mean": window_mean,
    }


def _cv_curve(traces, mean_trace, tarr, w):
    """Slide a length-``w`` window across the record; return (window-start indices,
    window-center times, shot-to-shot CV per window). ``traces`` is (n_shots, N)
    filtered; ``mean_trace`` is its across-shot mean."""
    n = mean_trace.size
    stride = max(1, w // 2)
    starts = np.arange(0, n - w + 1, stride)
    t_centers = np.array([tarr[s + w // 2] for s in starts], dtype=float)
    cv = np.empty(starts.size, dtype=float)
    for j, start in enumerate(starts):
        m_i = traces[:, start:start + w].mean(axis=1)        # per-shot window means
        window_mean = float(mean_trace[start:start + w].mean())
        cv[j] = float(np.std(m_i)) / abs(window_mean) if window_mean != 0 else np.nan
    return starts, t_centers, cv


def _cv_curve_for_position(f, scope, ch, shots, tarr, w, med_size, gauss_sigma):
    """Return (window-center times, cv_shots per window) for one position, or
    (None, None) if it has too few usable shots."""
    rows = load_filtered_traces(f, scope, ch, shots, tarr, med_size, gauss_sigma)
    if len(rows) < 2:
        return None, None
    traces = np.vstack(rows)
    _starts, t_centers, cv = _cv_curve(traces, traces.mean(axis=0), tarr, w)
    return t_centers, cv


def _profile_value(f, scope, ch, shots, tarr, i0, i1, med_size, gauss_sigma):
    """Mean filtered signal over the fixed sample window [i0:i1], averaged across
    a position's repeat shots. Returns None if no usable shots."""
    rows = load_filtered_traces(f, scope, ch, shots, tarr, med_size, gauss_sigma)
    if not rows:
        return None
    vals = [float(vf[i0:i1].mean()) for vf in rows]
    return float(np.mean(vals))


# ======================================================================================
# Reporting
# ======================================================================================

def _print_table(records):
    """Print the per-position results, best (lowest score) first."""
    print("=" * 88)
    print("QUIET, STEEP-GRADIENT WINDOW PER POSITION  "
          "(lower score = more reproducible & steeper dV/dx)")
    print(f"window={WINDOW_US:g} us   median={MED_SIZE:g} samples   "
          f"gauss_sigma={GAUSS_SIGMA:g} samples   "
          f"score = cv_shots + 1/|(dV/dx)/V|")
    print("-" * 88)
    if not records:
        print("(no valid windows — signal never exceeded the floor at any position)")
        print("=" * 88)
        return
    hdr = f"{'scope':<8} {'ch':<4} {'x':>7} {'y':>6} {'shots':>5} " \
          f"{'t_ctr(ms)':>10} {'cv_shots':>9} {'|dVdx/V|':>9} {'score':>8} {'mean(V)':>9}"
    print(hdr)
    for r in records:
        print(f"{r['scope']:<8} {r['channel']:<4} {r['x']:>7.1f} {r['y']:>6.1f} "
              f"{r['n_shots']:>5d} {r['t_center']*1e3:>10.3f} "
              f"{r['cv_shots']:>9.4f} {r['grad_x']:>9.4f} "
              f"{r['score']:>8.4f} {r['window_mean']:>9.4g}")
    print("-" * 88)
    b = records[0]
    print(f"BEST: {b['scope']}/{b['channel']} @ x={b['x']:.1f}, y={b['y']:.1f}  "
          f"window {b['t_start']*1e3:.3f}..{b['t_end']*1e3:.3f} ms  score={b['score']:.4f}")
    print("=" * 88)


# ======================================================================================
# Plotting
# ======================================================================================

def plot_quiet_window(path, scope=None, channels=None, window_us=None,
                      med_size=None, gauss_sigma=None, signal_frac=None,
                      show=None, save=None):
    """Plot the fluctuation analysis: score-vs-position and the best-window overlay.

    Honors SHOW_PLOT/SAVE_PLOT (override with show/save). Saves one PNG per scope
    to a ``plots/`` subdir next to the data file. Returns the saved paths.
    """
    import h5py
    import matplotlib.pyplot as plt

    scope = SCOPE if scope is None else scope
    med_size = MED_SIZE if med_size is None else med_size
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    show = SHOW_PLOT if show is None else show
    save = SAVE_PLOT if save is None else save

    records = find_quiet_window(path, scope=scope, channels=channels,
                                window_us=window_us, med_size=med_size,
                                gauss_sigma=gauss_sigma, signal_frac=signal_frac)

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
            fig, (ax_score, ax_cvt, ax_profile, ax_trace) = plt.subplots(
                4, 1, figsize=(10, 13))

            by_pos = _shots_by_position(f, sc, positions)
            tarr = read_hdf5_scope_tarr(f, sc)
            dt = float(tarr[1] - tarr[0])
            w = max(2, math.ceil(WINDOW_US * 1e-6 / dt))
            best = min(recs, key=lambda r: r["score"])
            i0 = int(np.searchsorted(tarr, best["t_start"]))
            i1 = int(np.searchsorted(tarr, best["t_end"], side="right"))

            # --- panel 1: |(dV/dx)/V| vs x position ---
            for ch in sorted({r["channel"] for r in recs}):
                cr = sorted((r for r in recs if r["channel"] == ch), key=lambda r: r["x"])
                xs = [r["x"] for r in cr]
                ax_score.plot(xs, [r["grad_x"] for r in cr],
                              "o-", lw=1, label=f"{ch} |(dV/dx)/V|")
            ax_score.set_xlabel("probe x (mm)")
            ax_score.set_ylabel("|(dV/dx)/V|  (1/mm)")
            ax_score.set_title(f"scope '{sc}': normalized spatial gradient vs position "
                               f"(window={WINDOW_US:g} us)", fontsize=10, loc="left")
            ax_score.legend(fontsize=8)
            ax_score.grid(alpha=0.3)

            # --- panel 2: cv_shots vs TIME at the worst-case (highest-score) position ---
            worst = max(recs, key=lambda r: r["score"])
            wkey = (round(worst["x"] / _POS_TOL) * _POS_TOL,
                    round(worst["y"] / _POS_TOL) * _POS_TOL)
            t_centers, cv_t = _cv_curve_for_position(
                f, sc, worst["channel"], by_pos.get(wkey, []), tarr, w, med_size, gauss_sigma)
            if cv_t is not None:
                ax_cvt.plot(t_centers * 1e3, cv_t, lw=0.8)
                kmin = int(np.nanargmin(cv_t))
                ax_cvt.plot(t_centers[kmin] * 1e3, cv_t[kmin], "rv",
                            label=f"min cv @ {t_centers[kmin]*1e3:.2f} ms")
                ax_cvt.legend(fontsize=8)
            ax_cvt.set_xlabel("time (ms)")
            ax_cvt.set_ylabel(r"$\sigma_{\mathrm{shots}}/|\langle \bar V\rangle|$")
            ax_cvt.set_title(
                f"shot-to-shot CV vs time at worst position: "
                f"{worst['channel']} @ x={worst['x']:.1f}, y={worst['y']:.1f}",
                fontsize=10, loc="left")
            ax_cvt.grid(alpha=0.3)

            # --- middle: spatial profile at the fixed window -> spatial gradient ---
            for ch in sorted({r["channel"] for r in recs}):
                cr = sorted((r for r in recs if r["channel"] == ch), key=lambda r: r["x"])
                xs, vals = [], []
                for r in cr:
                    key = (round(r["x"] / _POS_TOL) * _POS_TOL,
                           round(r["y"] / _POS_TOL) * _POS_TOL)
                    v = _profile_value(f, sc, ch, by_pos.get(key, []), tarr,
                                       i0, i1, med_size, gauss_sigma)
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
            for vf in load_filtered_traces(f, sc, best["channel"], by_pos.get(key, []),
                                           tarr, med_size, gauss_sigma):
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
    data_file = resolve_data_file()
    records = find_quiet_window(data_file)
    _print_table(records)
    if SHOW_PLOT or SAVE_PLOT:
        plot_quiet_window(data_file)
