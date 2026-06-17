# -*- coding: utf-8 -*-
"""
Simulate LeCroy oscilloscope SmartTrigger detection on recorded traces.

A LeCroy scope's SmartTriggers fire live on anomalies in a signal's timing and
amplitude parameters -- glitch/width, runt, slew rate, and interval (see
``ten_minute_tutorial_smart_triggers.pdf``). Here the traces are already
recorded in the bmotion HDF5 file, so this module runs a post-hoc "what would
have triggered" pass: it scans each trace and reports the events a SmartTrigger
would have caught.

Crossing levels are given in **absolute volts** (the same units as the recorded
trace), matching how you would dial a level on the scope's front panel. Width /
slew / interval limits are given in **nanoseconds** and a measured value is
flagged when it falls OUTSIDE the [min, max] bounds set for that detector
(either bound may be ``None`` to disable that side) -- the digital analog of the
scope's SmartTrigger time settings.

Each SmartTrigger type is a separate, pure function of ``(volts, tarr)`` so they
are unit-testable without HDF5 and reusable on their own:
    detect_glitch, detect_runt, detect_slew, detect_interval

Two scope-like preprocessing knobs apply before detection:
  * MATH       -- run a waveform-math op (derivative / integral / abs) first,
                  mimicking triggering off a scope Math trace.
  * HOLDOFF_US -- ignore the record before this time, mimicking trigger holdoff.

Filtering, shot grouping, and trace loading are imported from
:mod:`read_and_analyze.filter_data`; reading/decoding is delegated to the
in-repo ``scope_io`` package.

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

from scope_io import (
    open_hdf5_readonly, read_hdf5_scope_channel_shots, read_hdf5_scope_tarr,
)
try:  # works as a package (python -m read_and_analyze.smart_trigger_analysis)
    from read_and_analyze.read_bmotion_data import (
        read_positions, _position_for_shot, _scope_groups, _shot_numbers,
        _channel_names, _sample_shots, resolve_data_file,
    )
    from read_and_analyze.filter_data import (
        _as_list, _filter_trace,
    )
except ImportError:  # fallback when run directly from inside the folder
    from read_bmotion_data import (
        read_positions, _position_for_shot, _scope_groups, _shot_numbers,
        _channel_names, _sample_shots, resolve_data_file,
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
# (The input file is resolved at run time via resolve_data_file, not a knob here.)
SCOPE       = cfg.SCOPE
CHANNELS    = cfg.CHANNELS
MED_SIZE    = cfg.MED_SIZE
GAUSS_SIGMA = cfg.GAUSS_SIGMA
SHOW_PLOT  = cfg.SHOW_PLOT
SAVE_PLOT  = cfg.SAVE_PLOT
SHOTS      = cfg.SHOTS
HOLDOFF_US = cfg.HOLDOFF_US
MATH       = cfg.MATH

# Per-kind colors and marker shapes for the plot's detected-event scatter points.
_KIND_COLORS = {"glitch": "red", "runt": "purple", "slew": "green", "interval": "orange"}
_KIND_MARKERS = {"glitch": "o", "runt": "s", "slew": "^", "interval": "D"}


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
# Edge primitives shared by the detectors
# ======================================================================================

def _ns_to_s(value_ns):
    """Convert a nanosecond bound to seconds; pass ``None`` through unchanged."""
    return None if value_ns is None else value_ns * 1e-9


def _outside(value_s, min_s, max_s):
    """True when ``value_s`` falls outside the [min_s, max_s] band; a ``None``
    bound disables that side."""
    if min_s is not None and value_s < min_s:
        return True
    if max_s is not None and value_s > max_s:
        return True
    return False


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


def detect_glitch(volts, tarr, level=None, hyst=None, min_width_ns=None, max_width_ns=None):
    """Glitch / width hunt: flag positive pulses whose width is OUTSIDE the
    [min, max] bounds.

    Pulses are measured at ``level`` volts (with a ``hyst`` volt hysteresis band).
    A pulse is flagged when its width is below ``min_width_ns`` or above
    ``max_width_ns`` (either may be ``None`` to disable that side). ``nominal`` in
    the result is the median measured width, for reference only. Returns the
    uniform detector dict.
    """
    level = cfg.GLITCH_LEVEL if level is None else level
    hyst = cfg.GLITCH_HYST if hyst is None else hyst
    if min_width_ns is None:
        min_width_ns = cfg.GLITCH_MIN_WIDTH_NS
    if max_width_ns is None:
        max_width_ns = cfg.GLITCH_MAX_WIDTH_NS

    lo, hi = level - hyst, level + hyst
    rising, falling = _edges(volts, tarr, lo, hi)
    pulses = _pulses(rising, falling)
    if not pulses:
        return _result([], None)
    widths = np.array([p[2] for p in pulses], dtype=float)
    nominal = float(np.median(widths))
    min_s, max_s = _ns_to_s(min_width_ns), _ns_to_s(max_width_ns)
    events = [{"t_start": ts, "t_end": te, "value": w, "kind": "glitch"}
              for (ts, te, w) in pulses if _outside(w, min_s, max_s)]
    return _result(events, nominal)


def detect_runt(volts, tarr, lo=None, hi=None):
    """Runt: flag excursions that cross the LO level but never reach HI before
    returning below LO. ``lo``/``hi`` are the two levels in VOLTS. Event spans
    the LO-up to the matching LO-down crossing.
    """
    lo = cfg.RUNT_LO if lo is None else lo
    hi = cfg.RUNT_HI if hi is None else hi

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


def detect_slew(volts, tarr, lo=None, hi=None, min_ns=None, max_ns=None):
    """Slew rate: measure each edge's LO<->HI transition time; flag edges whose
    transition time is OUTSIDE the [min, max] bounds (faster than ``min_ns`` or
    slower than ``max_ns``; either may be ``None``). ``lo``/``hi`` are levels in
    VOLTS. ``nominal`` in the result is the median transition time, for reference.
    """
    lo = cfg.SLEW_LO if lo is None else lo
    hi = cfg.SLEW_HI if hi is None else hi
    if min_ns is None:
        min_ns = cfg.SLEW_MIN_NS
    if max_ns is None:
        max_ns = cfg.SLEW_MAX_NS

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
    if not transitions:
        return _result([], None)
    dts = np.array([t[2] for t in transitions], dtype=float)
    nominal = float(np.median(dts))
    min_s, max_s = _ns_to_s(min_ns), _ns_to_s(max_ns)
    events = [{"t_start": float(ts), "t_end": float(te), "value": float(d), "kind": "slew"}
              for (ts, te, d) in transitions if _outside(d, min_s, max_s)]
    return _result(events, nominal)


def detect_interval(volts, tarr, level=None, hyst=None, min_ns=None, max_ns=None):
    """Interval: measure the period between successive rising edges at ``level``
    volts; flag periods OUTSIDE the [min, max] bounds (shorter than ``min_ns`` or
    longer than ``max_ns``; either may be ``None``). ``nominal`` in the result is
    the median period, for reference.
    """
    level = cfg.INTERVAL_LEVEL if level is None else level
    hyst = cfg.INTERVAL_HYST if hyst is None else hyst
    if min_ns is None:
        min_ns = cfg.INTERVAL_MIN_NS
    if max_ns is None:
        max_ns = cfg.INTERVAL_MAX_NS

    lo, hi = level - hyst, level + hyst
    rising, _falling = _edges(volts, tarr, lo, hi)
    if len(rising) < 2:  # need >= 1 period to measure
        return _result([], None)
    periods = np.diff(rising)
    nominal = float(np.median(periods))
    min_s, max_s = _ns_to_s(min_ns), _ns_to_s(max_ns)
    events = []
    for k, p in enumerate(periods):
        if _outside(p, min_s, max_s):
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
    first/middle/last sample. Returns a list of shot numbers.

    ``shots`` may be any sequence -- a list, tuple, or numpy array (e.g.
    ``np.arange(10, 60)``); ``None`` or empty falls back to the sample shots.
    """
    shot_nums = _shot_numbers(sg)
    use = [int(s) for s in shots] if shots is not None and len(shots) else _sample_shots(shot_nums)
    return [s for s in use
            if sg.get(f"shot_{s}") is not None
            and not sg[f"shot_{s}"].attrs.get("skipped", False)]


def analyze_smart_triggers(path, scope=None, channels=None, shots=None, kinds=None,
                           holdoff_us=None, math=None, med_size=None, gauss_sigma=None):
    """Scan recorded traces for the events each SmartTrigger type would catch.

    Parameters default to the module constants. ``shots`` may be a list, tuple,
    or numpy array of shot numbers (e.g. ``np.arange(0, 10)``); ``None`` uses the
    sample shots (first/middle/last per position). For every (scope, channel,
    selected shot) the trace is denoised (median ``med_size`` then Gaussian
    ``gauss_sigma``), optionally transformed by ``math`` (derivative / integral /
    abs), gated by ``holdoff_us``, then run through each detector in ``kinds``
    (default all four). Returns a list of record dicts -- one per
    (scope, channel, shot, kind) -- each with keys: ``scope, channel, shot, x,
    y, kind, math, holdoff_us, n_events, nominal, events`` (``events`` is the
    detector's per-event list).
    """
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
    with open_hdf5_readonly(path) as f:
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
                # Read all shots of this channel in one pass (WAVEDESC decoded
                # once); NaN rows mark unreadable/skipped/length-mismatched shots.
                stack, _dt, _t0 = read_hdf5_scope_channel_shots(
                    f, sc, ch, shot_list, expected_len=len(tarr))
                if stack is None:
                    continue
                for s, volts in zip(shot_list, stack):
                    if np.isnan(volts).all():
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

    def _bnd(lo, hi):  # format an [min, max] ns band, "-" for an open side
        return f"[{'-' if lo is None else f'{lo:g}'}, {'-' if hi is None else f'{hi:g}'}] ns"
    print(f"bounds: glitch width={_bnd(cfg.GLITCH_MIN_WIDTH_NS, cfg.GLITCH_MAX_WIDTH_NS)}  "
          f"slew={_bnd(cfg.SLEW_MIN_NS, cfg.SLEW_MAX_NS)}  "
          f"interval={_bnd(cfg.INTERVAL_MIN_NS, cfg.INTERVAL_MAX_NS)}")
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
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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

    with open_hdf5_readonly(path) as f:
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

            # Read all plotted shots of this channel in one pass (WAVEDESC
            # decoded once); NaN rows mark unreadable/skipped/short shots.
            stack, _dt, _t0 = read_hdf5_scope_channel_shots(
                f, sc, ch, shot_list, expected_len=len(tarr))
            if stack is None:
                print(f"scope '{sc}': no usable shots to plot -- skipping")
                continue

            fig, axes = plt.subplots(len(shot_list), 1,
                                     figsize=(11, 2.8 * len(shot_list)),
                                     sharex=True, squeeze=False)
            axes = axes[:, 0]
            for ax, s, volts in zip(axes, shot_list, stack):
                if np.isnan(volts).all():
                    continue
                filt = _filter_trace(volts, med_size, gauss_sigma)
                sig_full = _apply_math(filt, tarr, math)
                sig, t = _holdoff_slice(sig_full, tarr, holdoff_us)
                if len(sig) < 4:
                    continue

                # Full math trace for context; detection happens on the gated slice.
                ax.plot(tarr * 1e3, sig_full, lw=0.8, color="0.4", alpha=0.9)

                # Configured crossing levels (absolute volts). Only meaningful
                # when no math transform is active, since the levels are in the
                # original V scale -- skip the guide lines for math traces.
                if not math:
                    for lvl in (cfg.GLITCH_LEVEL, cfg.RUNT_LO, cfg.RUNT_HI):
                        ax.axhline(lvl, color="0.7", lw=0.6, ls="--")

                if holdoff_us and holdoff_us > 0:
                    ax.axvspan(tarr[0] * 1e3, holdoff_us * 1e-3,
                               color="0.85", alpha=0.6, zorder=0)

                counts = {}
                for kind in kinds:
                    res = DETECTORS[kind](sig, t)
                    counts[kind] = res["n"]
                    if not res["events"]:
                        continue
                    # Mark each detected event as a point on the scanned signal at
                    # its start time, one color/marker per trigger kind.
                    ev_t = np.array([ev["t_start"] for ev in res["events"]])
                    ev_y = np.interp(ev_t, t, sig)
                    ax.scatter(ev_t * 1e3, ev_y, s=40, color=_KIND_COLORS[kind],
                               marker=_KIND_MARKERS[kind], edgecolors="black",
                               linewidths=0.5, zorder=5)

                pos = _position_for_shot(positions, s)
                pos_s = f" @ x={pos[0]:.1f}, y={pos[1]:.1f}" if pos is not None else ""
                cnt_s = ", ".join(f"{k}:{counts[k]}" for k in kinds)
                ax.set_title(f"shot {s}{pos_s}  ({cnt_s})", fontsize=9, loc="left")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel("time (ms)")

            legend_handles = [
                Line2D([], [], color=_KIND_COLORS[k], marker=_KIND_MARKERS[k],
                       linestyle="none", markeredgecolor="black", markeredgewidth=0.5,
                       label=k)
                for k in kinds
            ]
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
    data_file = resolve_data_file()
    records = analyze_smart_triggers(data_file)
    _print_table(records)
    if SHOW_PLOT or SAVE_PLOT:
        plot_smart_triggers(data_file)
