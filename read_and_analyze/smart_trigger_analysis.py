# -*- coding: utf-8 -*-
"""
Simulate LeCroy oscilloscope SmartTrigger detection on recorded traces.

A LeCroy scope's SmartTriggers fire live on anomalies in a signal's timing and
amplitude parameters -- glitch/width, runt, slew rate, and interval (see
``ten_minute_tutorial_smart_triggers.pdf``). Here the traces are already
recorded in the bmotion HDF5 file, so this module runs a post-hoc "what would
have triggered" pass: it scans each trace and reports the events a SmartTrigger
would have caught.

Crossing levels are derived **per trace** from that trace's own (min..max) span
(the software analog of the scope's "Find Level"), so the detectors work on
Isat/Vfloat signals of any absolute scale. Nominal width/period come from the
**median** of the measured population on the trace (robust against the rare
outliers we are hunting), mirroring the tutorial's "accumulate measurements,
read the mean".

Each SmartTrigger type is a separate, pure function of ``(volts, tarr)`` so they
are unit-testable without HDF5 and reusable on their own:
    detect_glitch, detect_runt, detect_slew, detect_interval

Two scope-like preprocessing knobs apply before detection:
  * MATH       -- run a waveform-math op (derivative / integral / abs) first,
                  mimicking triggering off a scope Math trace.
  * HOLDOFF_US -- ignore the record before this time, mimicking trigger holdoff.

Filtering, shot grouping, and trace loading are imported from
:mod:`read_and_analyze.filter_data`; reading/decoding is delegated to
``lab_scopes.io.hdf5``.

There is NO command line; all SmartTrigger knobs live in
``smart_trigger_config.py`` (grouped per trigger mode); filtering knobs live in
``filter_data``. Run with:
    python -m read_and_analyze.smart_trigger_analysis

Setup (once):  python -m pip install scipy

Created May.2026
@author: Jia Han
"""

import os

import numpy as np

from lab_scopes.io.hdf5 import read_hdf5_scope_data, read_hdf5_scope_tarr
try:  # works as a package (python -m read_and_analyze.smart_trigger_analysis)
    from read_and_analyze.read_bmotion_data import (
        read_positions, _position_for_shot, _scope_groups, _shot_numbers,
        _channel_names, _sample_shots,
    )
    from read_and_analyze.filter_data import (
        _as_list, _filter_trace,
    )
except ImportError:  # fallback when run directly from inside the folder
    from read_bmotion_data import (
        read_positions, _position_for_shot, _scope_groups, _shot_numbers,
        _channel_names, _sample_shots,
    )
    from filter_data import (
        _as_list, _filter_trace,
    )

# All SmartTrigger knobs live in smart_trigger_config.py, grouped per trigger
# mode; import the module so edits there take effect without touching this file.
try:  # works as a package (python -m read_and_analyze.smart_trigger_analysis)
    from read_and_analyze import smart_trigger_config as cfg
except ImportError:  # fallback when run directly from inside the folder
    import smart_trigger_config as cfg

# General knobs hoisted to module level for convenience / backwards compat.
DEFAULT_FILE = cfg.DATA_FILE   # input HDF5 file (shared via analysis_config)
SCOPE       = cfg.SCOPE
CHANNELS    = cfg.CHANNELS
MED_SIZE    = cfg.MED_SIZE
GAUSS_SIGMA = cfg.GAUSS_SIGMA
SHOW_PLOT  = cfg.SHOW_PLOT
SAVE_PLOT  = cfg.SAVE_PLOT
SHOTS      = cfg.SHOTS
HOLDOFF_US = cfg.HOLDOFF_US
MATH       = cfg.MATH

# Per-kind colors for the plot's shaded event spans.
_KIND_COLORS = {"glitch": "red", "runt": "purple", "slew": "green", "interval": "orange"}


# ======================================================================================
# Preprocessing (math + holdoff), applied before detection
# ======================================================================================

def _apply_math(volts, tarr, math):
    """Optional scope-like waveform math applied to the filtered trace BEFORE
    level/edge detection. Returns an array the same length as ``tarr``.

    ``None`` -> pass-through; ``"derivative"`` -> dV/dt (V/s) via ``np.gradient``;
    ``"integral"`` -> running integral (V*s) via cumulative trapezoid;
    ``"abs"`` -> ``|V|``. Unknown values raise ``ValueError``.
    """
    if math is None:
        return np.asarray(volts, dtype=float)
    v = np.asarray(volts, dtype=float)
    t = np.asarray(tarr, dtype=float)
    if math == "derivative":
        return np.gradient(v, t)
    if math == "integral":
        from scipy.integrate import cumulative_trapezoid
        return cumulative_trapezoid(v, t, initial=0.0)
    if math == "abs":
        return np.abs(v)
    raise ValueError(f"unknown MATH {math!r} (expected None, 'derivative', 'integral', or 'abs')")


def _holdoff_slice(volts, tarr, holdoff_us):
    """Restrict the trace to ``tarr >= holdoff_us*1e-6`` (relative to t=0), the
    digital analog of trigger holdoff. Times stay in the original time base.
    A holdoff of 0 (or before the record start) is a no-op."""
    if not holdoff_us or holdoff_us <= 0:
        return np.asarray(volts, dtype=float), np.asarray(tarr, dtype=float)
    mask = np.asarray(tarr) >= holdoff_us * 1e-6
    return np.asarray(volts, dtype=float)[mask], np.asarray(tarr, dtype=float)[mask]


# ======================================================================================
# Level / edge primitives shared by the detectors
# ======================================================================================

def _levels(volts, *fracs):
    """Map fractional levels onto a trace's (min..max) span; return absolute
    levels. ``_levels(v, 0.1, 0.9)`` -> (lo, hi) at 10% / 90% of the span."""
    v = np.asarray(volts, dtype=float)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    span = vmax - vmin
    return tuple(vmin + frac * span for frac in fracs)


def _interp_cross(tarr, volts, i, level):
    """Linear-interpolated time at which the segment [i, i+1] crosses ``level``."""
    v0, v1 = volts[i], volts[i + 1]
    if v1 == v0:
        return float(tarr[i])
    frac = (level - v0) / (v1 - v0)
    return float(tarr[i] + frac * (tarr[i + 1] - tarr[i]))


def _edges(volts, tarr, lo, hi):
    """Hysteresis crossing detector. A rising edge is registered when the signal,
    having last been at/below ``lo``, reaches ``hi``; a falling edge when, having
    last been at/above ``hi``, it drops to ``lo``. Crossing times are linearly
    interpolated. Returns ``(rising_times, falling_times)`` as float arrays.

    The lo/hi band debounces noise near a single level so a wiggle doesn't
    register multiple edges; pass ``lo == hi`` for a plain threshold.
    """
    v = np.asarray(volts, dtype=float)
    rising, falling = [], []
    state = None  # "low" once we've been <= lo, "high" once we've been >= hi
    if v[0] >= hi:
        state = "high"
    elif v[0] <= lo:
        state = "low"
    for i in range(len(v) - 1):
        if state != "high" and v[i] < hi <= v[i + 1]:
            rising.append(_interp_cross(tarr, v, i, hi))
            state = "high"
        elif state != "low" and v[i] > lo >= v[i + 1]:
            falling.append(_interp_cross(tarr, v, i, lo))
            state = "low"
    return np.array(rising, dtype=float), np.array(falling, dtype=float)


def _pulses(rising, falling):
    """Pair each rising edge with the next falling edge after it. Returns a list
    of ``(t_start, t_end, width)`` for completed positive pulses."""
    out = []
    fi = 0
    for tr in rising:
        while fi < len(falling) and falling[fi] <= tr:
            fi += 1
        if fi >= len(falling):
            break
        tf = falling[fi]
        out.append((float(tr), float(tf), float(tf - tr)))
        fi += 1
    return out


# ======================================================================================
# SmartTrigger detectors -- one pure function per type, uniform return shape
# ======================================================================================

def _result(events, nominal):
    """Uniform detector return: ``{events, nominal, n}``."""
    return {"events": events, "nominal": float(nominal) if nominal is not None else float("nan"),
            "n": len(events)}


def detect_glitch(volts, tarr, thresh_frac=None, hyst_frac=None, excl_delta=None):
    """Glitch hunt: flag positive pulses whose width is BELOW the nominal width.

    Pulses are measured at the main level (``thresh_frac`` of the span, with a
    ``hyst_frac`` hysteresis band). Nominal = median pulse width; a pulse is a
    glitch when its width < ``nominal*(1-excl_delta)`` -- the tutorial's
    Glitch ``<`` condition. Returns the uniform detector dict.
    """
    thresh_frac = cfg.GLITCH_THRESH_FRAC if thresh_frac is None else thresh_frac
    hyst_frac = cfg.GLITCH_HYST_FRAC if hyst_frac is None else hyst_frac
    excl_delta = cfg.GLITCH_EXCL_DELTA if excl_delta is None else excl_delta

    mid, = _levels(volts, thresh_frac)
    half = (_levels(volts, hyst_frac)[0] - _levels(volts, 0.0)[0])  # hyst band in volts
    lo, hi = mid - half, mid + half
    rising, falling = _edges(volts, tarr, lo, hi)
    pulses = _pulses(rising, falling)
    if len(pulses) < 2:
        return _result([], None)
    widths = np.array([p[2] for p in pulses], dtype=float)
    nominal = float(np.median(widths))
    floor = nominal * (1.0 - excl_delta)
    events = [{"t_start": ts, "t_end": te, "value": w, "kind": "glitch"}
              for (ts, te, w) in pulses if w < floor]
    return _result(events, nominal)


def detect_runt(volts, tarr, lo_frac=None, hi_frac=None):
    """Runt: flag excursions that cross the LO level but never reach HI before
    returning below LO. ``lo_frac``/``hi_frac`` set the two levels as fractions
    of the span. Event spans the LO-up to the matching LO-down crossing.
    """
    lo_frac = cfg.RUNT_LO_FRAC if lo_frac is None else lo_frac
    hi_frac = cfg.RUNT_HI_FRAC if hi_frac is None else hi_frac
    lo, hi = _levels(volts, lo_frac, hi_frac)

    v = np.asarray(volts, dtype=float)
    events = []
    i, n = 0, len(v)
    while i < n - 1:
        # find a LO upward crossing
        if v[i] < lo <= v[i + 1]:
            t_up = _interp_cross(tarr, v, i, lo)
            reached_hi = False
            j = i + 1
            while j < n - 1 and not (v[j] > lo >= v[j + 1]):
                if v[j] >= hi:
                    reached_hi = True
                j += 1
            if j < n - 1:
                t_dn = _interp_cross(tarr, v, j, lo)
            else:
                t_dn = float(tarr[-1])
            if not reached_hi:
                events.append({"t_start": t_up, "t_end": t_dn,
                               "value": float(t_dn - t_up), "kind": "runt"})
            i = j + 1
        else:
            i += 1
    return _result(events, None)


def detect_slew(volts, tarr, lo_frac=None, hi_frac=None, excl_delta=None):
    """Slew rate: measure each edge's LO<->HI transition time; flag edges whose
    transition time is OUTSIDE ``nominal*(1 +/- excl_delta)`` (slow or fast edge).
    Nominal = median transition time over all rising and falling edges.
    """
    lo_frac = cfg.SLEW_LO_FRAC if lo_frac is None else lo_frac
    hi_frac = cfg.SLEW_HI_FRAC if hi_frac is None else hi_frac
    excl_delta = cfg.SLEW_EXCL_DELTA if excl_delta is None else excl_delta
    lo, hi = _levels(volts, lo_frac, hi_frac)

    v = np.asarray(volts, dtype=float)
    transitions = []  # (t_start, t_end, dt)
    last_lo = last_hi = None
    for i in range(len(v) - 1):
        if v[i] < lo <= v[i + 1]:
            last_lo = _interp_cross(tarr, v, i, lo)
        if v[i] < hi <= v[i + 1] and last_lo is not None:  # rising edge LO->HI
            t_hi = _interp_cross(tarr, v, i, hi)
            transitions.append((last_lo, t_hi, t_hi - last_lo))
            last_lo = None
        if v[i] > hi >= v[i + 1]:
            last_hi = _interp_cross(tarr, v, i, hi)
        if v[i] > lo >= v[i + 1] and last_hi is not None:  # falling edge HI->LO
            t_lo = _interp_cross(tarr, v, i, lo)
            transitions.append((last_hi, t_lo, t_lo - last_hi))
            last_hi = None
    if len(transitions) < 2:
        return _result([], None)
    dts = np.array([t[2] for t in transitions], dtype=float)
    nominal = float(np.median(dts))
    lo_b, hi_b = nominal * (1.0 - excl_delta), nominal * (1.0 + excl_delta)
    events = [{"t_start": float(ts), "t_end": float(te), "value": float(d), "kind": "slew"}
              for (ts, te, d) in transitions if d < lo_b or d > hi_b]
    return _result(events, nominal)


def detect_interval(volts, tarr, thresh_frac=None, hyst_frac=None, excl_delta=None):
    """Interval: measure the period between successive rising edges at the main
    level; flag periods OUTSIDE ``nominal*(1 +/- excl_delta)`` (e.g. the long
    interval after a missed/runt cycle). Nominal = median period.
    """
    thresh_frac = cfg.INTERVAL_THRESH_FRAC if thresh_frac is None else thresh_frac
    hyst_frac = cfg.INTERVAL_HYST_FRAC if hyst_frac is None else hyst_frac
    excl_delta = cfg.INTERVAL_EXCL_DELTA if excl_delta is None else excl_delta

    mid, = _levels(volts, thresh_frac)
    half = (_levels(volts, hyst_frac)[0] - _levels(volts, 0.0)[0])
    lo, hi = mid - half, mid + half
    rising, _falling = _edges(volts, tarr, lo, hi)
    if len(rising) < 3:  # need >= 2 periods to have a meaningful median
        return _result([], None)
    periods = np.diff(rising)
    nominal = float(np.median(periods))
    lo_b, hi_b = nominal * (1.0 - excl_delta), nominal * (1.0 + excl_delta)
    events = []
    for k, p in enumerate(periods):
        if p < lo_b or p > hi_b:
            events.append({"t_start": float(rising[k]), "t_end": float(rising[k + 1]),
                           "value": float(p), "kind": "interval"})
    return _result(events, nominal)


DETECTORS = {
    "glitch": detect_glitch,
    "runt": detect_runt,
    "slew": detect_slew,
    "interval": detect_interval,
}


# ======================================================================================
# Driver
# ======================================================================================

def _resolve_shots(sg, shots):
    """Shot list to scan: explicit ``shots`` (skipping skipped ones) or the
    first/middle/last sample. Returns a list of shot numbers."""
    shot_nums = _shot_numbers(sg)
    use = list(shots) if shots else _sample_shots(shot_nums)
    return [s for s in use
            if sg.get(f"shot_{s}") is not None
            and not sg[f"shot_{s}"].attrs.get("skipped", False)]


def analyze_smart_triggers(path, scope=None, channels=None, shots=None, kinds=None,
                           holdoff_us=None, math=None, med_size=None, gauss_sigma=None):
    """Scan recorded traces for the events each SmartTrigger type would catch.

    Parameters default to the module constants. For every (scope, channel,
    selected shot) the trace is denoised (median ``med_size`` then Gaussian
    ``gauss_sigma``), optionally transformed by ``math`` (derivative / integral /
    abs), gated by ``holdoff_us``, then run through each detector in ``kinds``
    (default all four). Returns a list of record dicts -- one per
    (scope, channel, shot, kind) -- each with keys: ``scope, channel, shot, x,
    y, kind, math, holdoff_us, n_events, nominal, events`` (``events`` is the
    detector's per-event list).
    """
    import h5py

    scope = SCOPE if scope is None else scope
    channels = CHANNELS if channels is None else channels
    shots = SHOTS if shots is None else shots
    holdoff_us = HOLDOFF_US if holdoff_us is None else holdoff_us
    math = MATH if math is None else math
    med_size = MED_SIZE if med_size is None else med_size
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    channels = _as_list(channels)
    kinds = list(DETECTORS) if kinds is None else list(kinds)

    records = []
    with h5py.File(path, "r") as f:
        positions = read_positions(f)
        scopes = [scope] if scope else _scope_groups(f)

        for sc in scopes:
            sg = f[sc]
            tarr = read_hdf5_scope_tarr(f, sc)
            shot_list = _resolve_shots(sg, shots)
            if not shot_list:
                print(f"scope '{sc}': no usable shots to scan -- skipping")
                continue
            chans = channels if channels else _channel_names(sg, shot_list[0])

            for ch in chans:
                for s in shot_list:
                    try:
                        volts, _dt, _t0 = read_hdf5_scope_data(f, sc, ch, s)
                    except Exception:
                        continue
                    if len(volts) != len(tarr):
                        continue
                    filt = _filter_trace(volts, med_size, gauss_sigma)
                    sig = _apply_math(filt, tarr, math)
                    sig, t = _holdoff_slice(sig, tarr, holdoff_us)
                    if len(sig) < 4:
                        continue
                    pos = _position_for_shot(positions, s)
                    x, y = (pos if pos is not None else (float("nan"), float("nan")))
                    for kind in kinds:
                        res = DETECTORS[kind](sig, t)
                        records.append({
                            "scope": sc, "channel": ch, "shot": s,
                            "x": float(x), "y": float(y), "kind": kind,
                            "math": math, "holdoff_us": float(holdoff_us),
                            "n_events": res["n"], "nominal": res["nominal"],
                            "events": res["events"],
                        })
    return records


# ======================================================================================
# Reporting
# ======================================================================================

def _print_table(records):
    """Print per (scope, channel, shot, kind) anomaly counts; totals per kind."""
    print("=" * 92)
    print("SMART-TRIGGER SCAN  (events a LeCroy SmartTrigger would have caught)")
    math = next((r["math"] for r in records), MATH)
    holdoff = next((r["holdoff_us"] for r in records), HOLDOFF_US)
    print(f"math={math}   holdoff={holdoff:g} us   "
          f"median={MED_SIZE:g} samples   gauss_sigma={GAUSS_SIGMA:g} samples")
    print(f"excl_delta: glitch={cfg.GLITCH_EXCL_DELTA:g}  slew={cfg.SLEW_EXCL_DELTA:g}  "
          f"interval={cfg.INTERVAL_EXCL_DELTA:g}")
    print("-" * 92)
    if not records:
        print("(no traces scanned)")
        print("=" * 92)
        return
    hdr = (f"{'scope':<8} {'ch':<4} {'shot':>5} {'x':>7} {'y':>6} "
           f"{'kind':<9} {'nominal':>11} {'#events':>7}")
    print(hdr)
    for r in records:
        nom = r["nominal"]
        nom_s = "  n/a" if (nom is None or not np.isfinite(nom)) else f"{nom:>11.4g}"
        print(f"{r['scope']:<8} {r['channel']:<4} {r['shot']:>5d} "
              f"{r['x']:>7.1f} {r['y']:>6.1f} {r['kind']:<9} {nom_s} {r['n_events']:>7d}")
    print("-" * 92)
    totals = {}
    for r in records:
        totals[r["kind"]] = totals.get(r["kind"], 0) + r["n_events"]
    summary = "   ".join(f"{k}={totals.get(k, 0)}" for k in DETECTORS)
    print(f"TOTAL flagged events:   {summary}")
    print("=" * 92)


# ======================================================================================
# Plotting
# ======================================================================================

def plot_smart_triggers(path, scope=None, channels=None, shots=None, kinds=None,
                        holdoff_us=None, math=None, med_size=None, gauss_sigma=None,
                        show=None, save=None):
    """Plot the scanned signal per shot with detected SmartTrigger events shaded.

    One figure per scope, one panel per scanned shot (shared x-axis). Each panel
    shows the signal that was actually scanned (filtered, or the math trace when
    ``MATH`` is set), the derived crossing levels, the holdoff band, and a shaded
    span per detected event colored by kind. Honors SHOW_PLOT/SAVE_PLOT (override
    with show/save). Saves one PNG per scope to a ``plots/`` subdir next to the
    data file. Returns the saved paths.
    """
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    scope = SCOPE if scope is None else scope
    channels = CHANNELS if channels is None else channels
    shots = SHOTS if shots is None else shots
    holdoff_us = HOLDOFF_US if holdoff_us is None else holdoff_us
    math = MATH if math is None else math
    med_size = MED_SIZE if med_size is None else med_size
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    show = SHOW_PLOT if show is None else show
    save = SAVE_PLOT if save is None else save
    channels = _as_list(channels)
    kinds = list(DETECTORS) if kinds is None else list(kinds)

    ylabel = {"derivative": "dV/dt (V/s)", "integral": "integral (V*s)",
              "abs": "|V|"}.get(math, "V (filtered)")

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
            shot_list = _resolve_shots(sg, shots)
            if not shot_list:
                print(f"scope '{sc}': no usable shots to plot -- skipping")
                continue
            chans = channels if channels else _channel_names(sg, shot_list[0])
            ch = chans[0]  # one channel per figure for clarity

            fig, axes = plt.subplots(len(shot_list), 1,
                                     figsize=(11, 2.8 * len(shot_list)),
                                     sharex=True, squeeze=False)
            axes = axes[:, 0]
            for ax, s in zip(axes, shot_list):
                try:
                    volts, _dt, _t0 = read_hdf5_scope_data(f, sc, ch, s)
                except Exception:
                    continue
                if len(volts) != len(tarr):
                    continue
                filt = _filter_trace(volts, med_size, gauss_sigma)
                sig_full = _apply_math(filt, tarr, math)
                sig, t = _holdoff_slice(sig_full, tarr, holdoff_us)
                if len(sig) < 4:
                    continue

                # Full math trace for context; detection happens on the gated slice.
                ax.plot(tarr * 1e3, sig_full, lw=0.8, color="0.4", alpha=0.9)

                # Derived crossing levels (computed on the scanned slice).
                mid, = _levels(sig, cfg.GLITCH_THRESH_FRAC)
                r_lo, r_hi = _levels(sig, cfg.RUNT_LO_FRAC, cfg.RUNT_HI_FRAC)
                for lvl, name in ((mid, "thresh"), (r_lo, "runt lo"), (r_hi, "runt hi")):
                    ax.axhline(lvl, color="0.7", lw=0.6, ls="--")

                if holdoff_us and holdoff_us > 0:
                    ax.axvspan(tarr[0] * 1e3, holdoff_us * 1e-3,
                               color="0.85", alpha=0.6, zorder=0)

                counts = {}
                for kind in kinds:
                    res = DETECTORS[kind](sig, t)
                    counts[kind] = res["n"]
                    for ev in res["events"]:
                        ax.axvspan(ev["t_start"] * 1e3, ev["t_end"] * 1e3,
                                   color=_KIND_COLORS[kind], alpha=0.3)

                pos = _position_for_shot(positions, s)
                pos_s = f" @ x={pos[0]:.1f}, y={pos[1]:.1f}" if pos is not None else ""
                cnt_s = ", ".join(f"{k}:{counts[k]}" for k in kinds)
                ax.set_title(f"shot {s}{pos_s}  ({cnt_s})", fontsize=9, loc="left")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel("time (ms)")

            legend_handles = [Patch(color=_KIND_COLORS[k], alpha=0.3, label=k) for k in kinds]
            axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right",
                           ncol=len(kinds))

            math_s = f"math={math}, " if math else ""
            fig.suptitle(
                f"{os.path.basename(path)}  --  scope '{sc}' / {ch} SmartTrigger scan "
                f"({math_s}holdoff={holdoff_us:g} us)",
                fontsize=10)
            fig.tight_layout()

            if save:
                out_png = os.path.join(plots_dir, f"{base}_{sc}_smart_triggers.png")
                fig.savefig(out_png, dpi=150)
                saved.append(out_png)
                print(f"Saved plot: {out_png}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return saved


if __name__ == "__main__":
    records = analyze_smart_triggers(DEFAULT_FILE)
    _print_table(records)
    if SHOW_PLOT or SAVE_PLOT:
        plot_smart_triggers(DEFAULT_FILE)
