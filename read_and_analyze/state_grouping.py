# -*- coding: utf-8 -*-
"""
Classify shots into groups by monitor-channel amplitude.

When a run interleaves conditions -- two antennas each independently on or off,
say -- the shots at one probe position belong to different physical states and
must be averaged *within* state, not across it. Nothing in the HDF5 records the
state, but the antenna-current monitors do: a driven channel has a large RMS in
the drive window, an undriven one has only noise.

:func:`classify_by_channel_rms` measures that per shot and turns it into a
``{shot_num: label}`` map, which
:func:`read_and_analyze.plot_xy_map.select_by_state` consumes as a shot selector.

The on/off threshold is **derived from the data**, not configured: sort each
channel's per-shot RMS, take logs, and cut at the largest gap. A genuinely
bimodal on/off population separates by a large factor, so the largest
logarithmic gap lands between the two clusters wherever they happen to sit. This
avoids hardcoding a volt level that silently misclassifies when the drive
amplitude or the gain changes. If the separation is weak the classifier says so
rather than guessing -- see ``min_ratio``.

The approach is adapted from ``LAPD_analysis_TB/three_condition_plane_sequence.py``,
generalized here to N channels with a caller-supplied label map.

Created Aug.2026
@author: Jia Han
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scope_io import read_hdf5_scope_channel_shots, read_hdf5_scope_tarr

try:  # works as a package (python -m read_and_analyze.state_grouping)
    from read_and_analyze.plot_xy_map import window_indices
except ImportError:  # fallback when run directly from inside the folder
    from plot_xy_map import window_indices

UNKNOWN = "unknown"


class ClassificationError(ValueError):
    """The monitor channels do not separate cleanly into on/off populations."""


def largest_log_gap_threshold(values, min_ratio=2.0):
    """Split ``values`` into an off and an on population; return the threshold.

    Sorts the positive values, finds the largest gap between consecutive values
    in log space, and returns the geometric midpoint of that gap. Also returns
    the gap's ratio, which is the evidence that the split is real: a clean
    on/off population separates by a large factor, so a small ratio means the
    values form one continuous cluster and no honest threshold exists.

    Raises ClassificationError when there are too few values to judge, or when
    the best ratio falls under ``min_ratio``.
    """
    v = np.asarray(values, dtype=float)
    positive = np.sort(v[np.isfinite(v) & (v > 0)])
    if positive.size < 4:
        raise ClassificationError(
            f"need at least 4 positive RMS values to classify, got {positive.size}")

    gaps = np.diff(np.log(positive))
    i = int(np.argmax(gaps))
    ratio = float(positive[i + 1] / positive[i])
    if ratio < min_ratio:
        raise ClassificationError(
            f"no clean off/on separation: largest RMS ratio is {ratio:.3g}, "
            f"below min_ratio={min_ratio}. The shots may all be in one state, or "
            "the monitor window may miss the drive.")
    return float(np.sqrt(positive[i] * positive[i + 1])), ratio


def channel_rms(f, scope, channel, shots, t_window_ms=None):
    """Mean-subtracted RMS of one channel in a time window, per shot.

    Returns a float array parallel to ``shots`` (NaN for unreadable shots). The
    mean is removed first so a DC offset does not masquerade as drive amplitude.
    ``t_window_ms`` is ``(start, end)`` in ms; None uses the whole record.
    """
    tarr = read_hdf5_scope_tarr(f, scope)
    raw, _dt, _t0 = read_hdf5_scope_channel_shots(
        f, scope, channel, list(shots), expected_len=len(tarr))
    if raw is None:
        return np.full(len(shots), np.nan)

    if t_window_ms is not None:
        i0, i1 = window_indices(tarr, t_window_ms[0] * 1e-3, t_window_ms[1] * 1e-3)
        raw = raw[:, i0:i1]

    with np.errstate(invalid="ignore"):
        centered = raw - np.nanmean(raw, axis=1, keepdims=True)
        return np.sqrt(np.nanmean(centered ** 2, axis=1))


def classify_by_channel_rms(f, scope, channels, shots, labels,
                            t_window_ms=None, min_ratio=2.0):
    """Label every shot by which monitor channels were active.

    ``channels`` is the ordered monitor list (e.g. ``("C7", "C8")``); ``labels``
    maps an on/off tuple to a name, e.g.::

        {(True, False): "south_only",
         (False, True): "north_only",
         (True, True):  "both_on",
         (False, False): "background"}

    Returns ``(state_by_shot, metadata)``. Shots whose on/off tuple has no entry
    in ``labels`` are labeled ``"unknown"`` rather than raising: this feeds a
    plotting tool, and an ``unknown`` group visible in the viewer is more useful
    for exploration than an aborted run. The count is reported in ``metadata``
    and printed.

    Reads only the RMS window of the monitor channels, but does read every shot,
    so this is a second pass over the file -- call it only when state grouping is
    actually wanted.
    """
    shots = list(shots)
    rms, thresholds, ratios = {}, {}, {}
    for ch in channels:
        rms[ch] = channel_rms(f, scope, ch, shots, t_window_ms)
        thresholds[ch], ratios[ch] = largest_log_gap_threshold(rms[ch], min_ratio)
        n_on = int(np.sum(rms[ch] > thresholds[ch]))
        print(f"  {scope}/{ch}: threshold {thresholds[ch]:.4g} V rms "
              f"(gap ratio {ratios[ch]:.3g}) -> {n_on}/{len(shots)} shots on")

    state_by_shot, counts, unknown = {}, {}, 0
    for i, shot in enumerate(shots):
        active = tuple(bool(rms[ch][i] > thresholds[ch]) for ch in channels)
        label = labels.get(active, UNKNOWN)
        if label == UNKNOWN:
            unknown += 1
        state_by_shot[shot] = label
        counts[label] = counts.get(label, 0) + 1

    if unknown:
        print(f"  warning: {unknown}/{len(shots)} shots matched no configured "
              f"state and are grouped as '{UNKNOWN}'")
    for label in sorted(counts):
        print(f"  state '{label}': {counts[label]} shots")

    metadata = {
        "scope": scope,
        "channels": list(channels),
        "rms_window_ms": list(t_window_ms) if t_window_ms else None,
        "thresholds_v_rms": {ch: float(thresholds[ch]) for ch in channels},
        "gap_ratios": {ch: float(ratios[ch]) for ch in channels},
        "state_counts": counts,
        "unknown_shots": unknown,
        "method": ("Mean-subtracted RMS in the monitor window; each channel's "
                   "off/on threshold is the geometric midpoint of its largest "
                   "logarithmic gap."),
    }
    return state_by_shot, metadata


# Ready-made label map for the two-antenna case these runs use.
TWO_ANTENNA_LABELS = {
    (False, False): "background",
    (True, False): "south_only",
    (False, True): "north_only",
    (True, True): "both_on",
}
