# -*- coding: utf-8 -*-
"""
Interactive position-slider FFT page comparing shot groups, as standalone HTML.

Where :mod:`read_and_analyze.plot_xy_slider` sweeps *time* across a plane, this
module sweeps *position*: at every probe position it computes one power spectrum
per shot group and draws them on a shared log axis, so the question "does the
spectrum at this position differ between antenna states?" is answered by moving
one slider instead of by opening several files.

The design follows ``LAPD_analysis_TB/three_condition_plane_position_spectra.py``
-- a per-shot periodogram averaged within group, shot SEM as shading, a position
locator beside the spectrum -- with three deliberate departures:

* **Geometry-agnostic.** Positions are carried as a flat list with their (x, y)
  coordinates, so a vertical line, a horizontal line, and a full plane all render
  through the same code. Nothing here asks whether the run is a plane.
* **One page per scope.** Every channel of a scope lands in one HTML file behind
  a dropdown, rather than one file per channel: the comparison being made is
  across groups *and* channels at the same position.
* **Grouping is optional.** With ``SHOT_MODE = "mean"`` the page shows a single
  all-shot average, so this module is useful on a run with no antenna states.

The written page needs neither Python nor the HDF5 file. Re-running is the only
way to change the window, the frequency limit, or the grouping.

There is NO command line; knobs live below and in analysis_config.py. Run with:
    python -m read_and_analyze.plot_fft_slider

Created Aug.2026
@author: Jia Han
"""

import base64
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:  # progress bar over the per-position loop; optional dependency
    from tqdm import tqdm
except ImportError:  # fall back to a no-op pass-through if tqdm isn't installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scope_io import open_hdf5_readonly, read_hdf5_scope_tarr

try:  # works as a package (python -m read_and_analyze.plot_fft_slider)
    from read_and_analyze.read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
        resolve_data_file,
    )
    from read_and_analyze.plot_xy_map import (
        _grid_layout, _load_signal_stack, _position_shotnums,
        select_all_shots, select_shot_index, select_by_state, window_indices,
    )
    from read_and_analyze.filter_data import _as_list
    from read_and_analyze.signals import as_signal_list, raw as raw_signal
    from read_and_analyze.state_grouping import (
        classify_by_channel_rms, default_labels,
    )
    from read_and_analyze.analysis_config import (
        MED_SIZE, GAUSS_SIGMA, SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS,
    )
except ImportError:  # fallback when run directly from inside the folder
    from read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
        resolve_data_file,
    )
    from plot_xy_map import (
        _grid_layout, _load_signal_stack, _position_shotnums,
        select_all_shots, select_shot_index, select_by_state, window_indices,
    )
    from filter_data import _as_list
    from signals import as_signal_list, raw as raw_signal
    from state_grouping import classify_by_channel_rms, default_labels
    from analysis_config import (
        MED_SIZE, GAUSS_SIGMA, SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS,
    )

# ---- knobs ---- (module-private; shared ones come from analysis_config.py)
# Which scope to spectrum-analyze. None = every scope in the file, one page each.
FFT_SCOPE = None          # None = fall back to SELECT_SCOPE in analysis_config.py

# Which channels go into the page's dropdown. None = fall back to SELECT_CHAN,
# and None there means every channel the scope recorded.
FFT_CHANNELS = None

# Time window the spectrum is computed over, in ms. None = the whole digitized
# record. This is the single most important knob here: a window that straddles
# the antenna turn-on mixes driven and quiet plasma into one spectrum.
FFT_T_START_MS = None
FFT_T_END_MS   = None

# Upper frequency shown, in kHz. None = the full band up to Nyquist. Trimming
# here is what keeps the page small: the payload scales with the bin count.
FFT_F_MAX_KHZ = 200.0

# Welch segment length in ms. None = a plain periodogram over the whole window
# (finest frequency resolution, noisiest estimate). A number splits the window
# into half-overlapping Hann segments and averages them -- smoother, at a
# frequency resolution of roughly 1/segment.
FFT_SEGMENT_MS = 1.5

SHOT_MODE  = "state"   # "mean" = one all-shot average | "index" | "state"
SHOT_INDEX = 0         # which shot, when SHOT_MODE == "index"

# Add an extra curve that averages every shot at a position together, ignoring
# the grouping entirely -- the ungrouped baseline the grouped curves are read
# against. Only meaningful alongside a mode that actually splits the shots, so
# it is ignored under SHOT_MODE "mean" (already that average) and "index" (a
# single shot). Set to a string to relabel the curve.
INCLUDE_ALL_SHOTS = True
ALL_SHOTS_LABEL = None   # None = auto-label from the window, e.g. "all shots 5-7 ms"

# Window for that baseline curve alone, in ms. None = the same window as the
# grouped curves. Pointing it at the quiet interval before the antenna fires is
# what makes it a background reference rather than a blend of the conditions --
# the same split analysis_TB draws between its "pre_antenna" and "driven" gates.
#
# Under Welch the frequency axis is set by the segment length, not the window,
# so a different window here still lands on the shared axis; build_spectra
# verifies that rather than assuming it. A window shorter than one segment is
# still valid but gets no segment averaging, so it will look noisier than the
# grouped curves -- shorten FFT_SEGMENT_MS if that matters more than resolution.
ALL_SHOTS_T_START_MS = 5.0
ALL_SHOTS_T_END_MS   = 7.0

# Group shots based on the following channels (same contract as plot_xy_slider)
STATE_GROUPS = {"channels": ("C7", "C8"), "scope": "bdot_scope", "window_ms": (0.0, 20.0)}

# Where the HTML lands. None = a "plots/" subdir beside the data file; a
# directory = auto-named files inside it; a path ending in .html = that exact
# file (only valid when a single scope is being written).
OUTPUT_PATH = None

# The two knobs above that shadow a shared config value, resolved once here so
# the call site has a single uniform "knob or argument" rule like every other.
_SCOPE = FFT_SCOPE if FFT_SCOPE is not None else SCOPE
_CHANNELS = FFT_CHANNELS if FFT_CHANNELS is not None else CHANNELS


# ======================================================================================
# Spectrum estimation
# ======================================================================================

def fft_friendly_length(n):
    """Largest length <= ``n`` whose only prime factors are 2, 3, 5, and 7.

    A scope record is very often a round decimal count plus one -- 1000001
    samples, which factors as 101 x 9901. Both are large primes, so numpy cannot
    use a radix FFT and falls back to Bluestein's algorithm: measured ~10x slower
    than the adjacent even length, and it dominates this module's runtime.
    Dropping the odd sample or two costs a few parts per thousand in the estimate
    and buys back that factor of ten. The search is capped at 1% of the record so
    the analysis window is never meaningfully shortened to chase a nicer length;
    if no smooth length exists within that budget the original ``n`` is returned
    and the transform is simply slower.
    """
    if n < 8:
        return n
    floor = max(n - max(8, n // 100), 7)
    for candidate in range(n, floor, -1):
        m = candidate
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return candidate
    return n


def spectrum_grid(nsamples, dt, segment_ms, f_max_khz):
    """Resolve the transform length, frequency axis, and segment length up front.

    Returns ``(freq_hz, keep, nperseg, nfft)`` where ``keep`` slices the axis to
    ``f_max_khz``, ``nperseg`` is None for a plain periodogram, and ``nfft`` is
    the (possibly trimmed) sample count actually transformed. Computing this
    before the read means every position writes into an array of known width, and
    a segment longer than the window is caught here rather than after a long read.
    """
    if segment_ms is None:
        nperseg = None
        nfft = fft_friendly_length(nsamples)
    else:
        nperseg = int(round(segment_ms * 1e-3 / dt))
        if nperseg < 8:
            raise ValueError(
                f"FFT_SEGMENT_MS={segment_ms} is only {nperseg} samples at "
                f"dt={dt:.3g} s; use a longer segment or leave it None")
        if nperseg > nsamples:
            raise ValueError(
                f"FFT_SEGMENT_MS={segment_ms} ms exceeds the {nsamples * dt * 1e3:.4g} ms "
                "analysis window; shorten it or widen FFT_T_START_MS/FFT_T_END_MS")
        nperseg = fft_friendly_length(nperseg)
        nfft = nperseg

    freq = np.fft.rfftfreq(nfft, d=dt)
    # The axis is monotonic, so the band limit is always a prefix: a slice keeps
    # ``psd[:, keep]`` a view instead of the copy a boolean mask would force.
    keep = (slice(None) if f_max_khz is None
            else slice(0, int(np.searchsorted(freq, f_max_khz * 1e3 + 1e-6))))
    if freq[keep].size < 2:
        raise ValueError(
            f"FFT_F_MAX_KHZ={f_max_khz} is below the first nonzero bin "
            f"({freq[1] / 1e3:.4g} kHz); nothing would be plotted")
    return freq[keep], keep, nperseg, nfft


def hann_window(n, dt):
    """Hann window of length ``n`` and its density normalization, memoized.

    Both depend only on ``(n, dt)``, which are fixed for a whole run, but
    :func:`shot_psd` is called once per position per signal -- rebuilding a
    million-sample window each time cost measurable seconds across a run.
    Dividing by ``fs * sum(w^2)`` makes the result a PSD in V^2/Hz independent of
    segment length, so a Welch page and a periodogram page sit on the same scale.
    """
    key = (n, dt)
    cached = _WINDOWS.get(key)
    if cached is None:
        window = np.hanning(n)
        cached = (window, (1.0 / dt) * np.sum(window ** 2))
        _WINDOWS[key] = cached
    return cached


_WINDOWS = {}


def shot_psd(traces, dt, keep, nperseg, nfft):
    """One-sided PSD (V^2/Hz) of every row of ``traces``, trimmed to ``keep``.

    ``nperseg`` None means a Hann periodogram over the first ``nfft`` samples;
    otherwise the row is split into half-overlapping Hann segments whose
    periodograms are averaged (Welch). Rows that are entirely NaN come back as
    NaN rather than poisoning the average, which is what lets an unreadable shot
    sit in the stack without special-casing upstream.

    Deliberately not ``scipy.signal.welch``: welch propagates a single NaN sample
    across a shot's entire spectrum, and unreadable shots arrive here as NaN rows
    as a matter of course (see ``plot_xy_map._load_stack``).
    """
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    n = nfft if nperseg is None else nperseg
    starts = ([0] if nperseg is None
              else range(0, traces.shape[1] - n + 1, max(1, n // 2)))
    window, norm = hann_window(n, dt)
    # An all-NaN row has a NaN mean in every segment, so one column of the first
    # detrended segment identifies it -- no second full-record scan needed.
    bad = None

    total, count = None, 0
    for s in starts:
        # Detrend into a fresh array, then window and NaN-fill in place: each of
        # these is a full-size temporary on a million-sample record, so the
        # copies are worth avoiding rather than chaining.
        seg = traces[:, s:s + n] - np.nanmean(traces[:, s:s + n], axis=1,
                                              keepdims=True)
        if bad is None:
            bad = ~np.isfinite(seg[:, 0])
        np.nan_to_num(seg, copy=False)
        seg *= window
        spec = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        spec /= norm
        spec[:, 1:-1] *= 2.0    # fold the negative frequencies onto the positive
        total = spec if total is None else total + spec
        count += 1

    psd = total if count == 1 else total / count
    psd[bad] = np.nan
    return psd[:, keep]


def group_statistics(psd, rows):
    """Mean PSD, SEM, and contributing-shot count over ``rows`` of ``psd``.

    ddof=1 for the same reason :func:`plot_xy_map._reduce_group` uses it: these
    are sample repeats, and one shot cannot estimate its own scatter, so a
    single-shot group yields a NaN band rather than a misleading zero-width one.
    """
    sub = psd[rows]
    n = float(np.sum(np.isfinite(sub).all(axis=1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0, ddof=1) if sub.shape[0] > 1 else np.full(sub.shape[1], np.nan)
        sem = sd / np.sqrt(n) if n > 1 else np.full(sub.shape[1], np.nan)
    return mean, sem, n


# ======================================================================================
# Assembly
# ======================================================================================

def planned_coords(positions):
    """Flat ``((npos, 2) coords, npos)`` of planned (x, y) in mm.

    This module is geometry-agnostic on purpose: a vertical line, a horizontal
    line, and a full plane are all just a list of positions here, so unlike
    :func:`plot_xy_map._plane_axes` nothing is reshaped to ``(ny, nx)`` and no
    run is refused for being one-dimensional.

    Picks the same motion group ``_plane_axes`` would -- the first one that
    actually carries a setup array -- so ``coords`` and ``npos`` always describe
    one group. Taking the first group unconditionally would silently pair one
    group's count with another's coordinates on a file whose first motion group
    has no setup array.
    """
    for _name, info in positions.items():
        setup = info.get("setup_array")
        if setup is None:
            continue
        coords = np.column_stack((np.asarray(setup["x"], dtype=float),
                                  np.asarray(setup["y"], dtype=float)))
        return coords, len(setup)
    return None, 0


def build_spectra(f, scope, signals, positions, selector, t_start_ms, t_end_ms,
                  f_max_khz, segment_ms, med_size, gauss_sigma,
                  alt_label=None, alt_t_start_ms=None, alt_t_end_ms=None):
    """Compute per-position, per-group, per-signal spectra in one read pass.

    Returns ``(result, freq_khz, coords, meta)``:

    * ``result`` maps ``"signal_key group_label"`` to ``(mean, sem, n)``, each
      shaped ``(npos, nfreq)`` for mean/sem and ``(npos,)`` for n
    * ``coords`` is the ``(npos, 2)`` array of planned (x, y) in mm

    Every planned position gets a row whether or not it yielded data, so the
    slider index and the position index are the same number -- a NaN row reads as
    "nothing usable here", which is the honest answer.

    ``alt_label``, when given, is computed over ``alt_t_start_ms..alt_t_end_ms``
    instead of the shared window -- the pre-antenna baseline case. Both windows
    must land on the same frequency axis for the page to plot them together;
    that is checked here rather than assumed.
    """
    coords, npos = planned_coords(positions)
    if npos == 0:
        return {}, None, None, None

    tarr = read_hdf5_scope_tarr(f, scope)
    dt = float(np.median(np.diff(tarr)))
    t_lo = tarr[0] if t_start_ms is None else t_start_ms * 1e-3
    t_hi = tarr[-1] if t_end_ms is None else t_end_ms * 1e-3
    i0, i1 = window_indices(tarr, t_lo, t_hi)
    nrequested = i1 - i0
    freq_hz, keep, nperseg, nfft = spectrum_grid(nrequested, dt, segment_ms, f_max_khz)
    if nperseg is None:
        i1 = i0 + nfft   # transform the FFT-friendly prefix, not the odd tail

    # The baseline curve's own window, resolved the same way and then required to
    # produce the identical axis. Under Welch it will, because the axis follows
    # the segment length -- but a mismatched knob pair (say a window shorter than
    # one segment, or segment_ms=None) has to fail loudly rather than write a
    # curve whose bins silently mean different frequencies than its neighbours'.
    alt = None
    if alt_label is not None and (alt_t_start_ms is not None or alt_t_end_ms is not None):
        a_lo = tarr[0] if alt_t_start_ms is None else alt_t_start_ms * 1e-3
        a_hi = tarr[-1] if alt_t_end_ms is None else alt_t_end_ms * 1e-3
        a0, a1 = window_indices(tarr, a_lo, a_hi)
        a_freq, a_keep, a_nperseg, a_nfft = spectrum_grid(a1 - a0, dt, segment_ms,
                                                          f_max_khz)
        if a_nperseg is None:
            a1 = a0 + a_nfft
        if a_freq.shape != freq_hz.shape or not np.allclose(a_freq, freq_hz,
                                                            rtol=1e-7, atol=1e-6):
            raise ValueError(
                f"'{alt_label}' window {alt_t_start_ms}-{alt_t_end_ms} ms gives a "
                f"{a_freq.size}-bin axis at {a_freq[1] - a_freq[0]:.4g} Hz, but the "
                f"grouped windows give {freq_hz.size} bins at "
                f"{freq_hz[1] - freq_hz[0]:.4g} Hz. Set FFT_SEGMENT_MS to a value "
                "that fits inside both windows so they share one frequency axis.")
        alt = (a0, a1, a_keep, a_nperseg, a_nfft)

    _tarr, nshot, mismatch = _grid_layout(f, scope, npos)

    nfreq = freq_hz.size
    acc = defaultdict(lambda: (np.full((npos, nfreq), np.nan),
                               np.full((npos, nfreq), np.nan),
                               np.zeros(npos)))

    desc = f"spectra {scope}"
    for i, shotnums in tqdm(_position_shotnums(positions, npos, nshot, mismatch),
                            total=npos, desc=desc, unit="pos"):
        if not shotnums:
            continue
        groups = selector(shotnums)
        if not groups:
            continue
        row_of = {s: r for r, s in enumerate(shotnums)}
        for sig in signals:
            stack = _load_signal_stack(f, scope, sig, shotnums, tarr,
                                       med_size, gauss_sigma)
            if stack is None:
                continue
            psd = shot_psd(stack[:, i0:i1], dt, keep, nperseg, nfft)
            # Second transform only when a baseline window was asked for. The
            # read above already pulled the whole record, so this is one more
            # slice and FFT, not another pass over the file.
            alt_psd = (None if alt is None else
                       shot_psd(stack[:, alt[0]:alt[1]], dt, *alt[2:]))
            for label, shots in groups.items():
                rows = [row_of[s] for s in shots if s in row_of]
                if not rows:
                    continue
                src = alt_psd if (alt_psd is not None and label == alt_label) else psd
                # Space-joined to match key() in the page. Unambiguous because
                # signals._safe_key collapses whitespace out of a signal key, so
                # the first space is always the separator.
                mean_a, sem_a, n_a = acc[f"{sig.key} {label}"]
                mean_a[i], sem_a[i], n_a[i] = group_statistics(src, rows)

    meta = {
        "window_ms": (float(tarr[i0] * 1e3), float(tarr[i1 - 1] * 1e3)),
        "nsamples": int(i1 - i0),
        "df_hz": float(freq_hz[1] - freq_hz[0]) if freq_hz.size > 1 else 0.0,
        "nperseg": nperseg,
        "ntrimmed": int(nrequested - nfft) if nperseg is None else 0,
        "nseg": (None if nperseg is None else
                 max(1, (i1 - i0 - nperseg) // max(1, nperseg // 2) + 1)),
        "alt_window_ms": (None if alt is None else
                          (float(tarr[alt[0]] * 1e3), float(tarr[alt[1] - 1] * 1e3))),
        "alt_nseg": (None if alt is None or alt[3] is None else
                     max(1, (alt[1] - alt[0] - alt[3]) // max(1, alt[3] // 2) + 1)),
    }
    return dict(acc), freq_hz / 1e3, coords, meta


# ======================================================================================
# Payload encoding
# ======================================================================================

def encode_array(a):
    """Encode a float array as little-endian float32 bytes in base64.

    Same rationale as :func:`read_and_analyze.plot_xy_slider.encode_array`: about
    4x smaller than a JSON number list and decoded with one ``atob`` instead of
    tokenizing hundreds of thousands of numbers. float32 is far more precision
    than a PSD estimate carries; the .npz sidecar is the bit-exact record.
    """
    arr = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    return {"b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            "shape": list(arr.shape)}


def resolve_output(path, output_path, base, scope, n_expected):
    """Decide where one page is written.

    Mirrors :func:`read_and_analyze.plot_xy_slider.resolve_output`: None means a
    ``plots/`` dir beside the data file, a directory means auto-named files
    inside it, and an explicit ``.html`` is refused when the run would write more
    than one page rather than quietly leaving one file behind.
    """
    auto = f"{base}_{scope}_fftslider.html"
    if output_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "plots")
    elif str(output_path).lower().endswith(".html"):
        if n_expected > 1:
            raise ValueError(
                f"output_path={output_path!r} names a single .html file but this "
                f"run would write {n_expected} pages. Pass a directory instead, "
                "or set FFT_SCOPE to one scope.")
        out_file = os.path.abspath(str(output_path))
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        return out_file
    else:
        out_dir = os.path.abspath(str(output_path))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, auto)


# ======================================================================================
# HTML
# ======================================================================================

_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark light;font-family:Inter,ui-sans-serif,system-ui,sans-serif}
body{margin:0;background:#11151c;color:#e8eef6}
main{max-width:1220px;margin:auto;padding:22px}
h1{margin:0 0 4px;font-size:1.3rem}
.sub{color:#9fb0c4;font-size:.85rem;margin:0 0 16px}
.panel{display:grid;grid-template-columns:minmax(420px,1fr) 330px;gap:18px;align-items:start}
canvas{width:100%;background:#fff;border-radius:8px}
.controls{background:#182030;border:1px solid #2a3548;border-radius:8px;padding:14px;margin-top:12px}
button,select{color:#e8eef6;background:#25334a;border:1px solid #3e4d67;border-radius:5px;padding:6px 10px;font-size:.9rem}
input[type=range]{width:100%;margin:12px 0}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
label{font-size:.85rem;color:#9fb0c4}
dl{display:grid;grid-template-columns:auto 1fr;gap:6px 12px;margin:14px 0 0;font-variant-numeric:tabular-nums;font-size:.88rem}
dt{color:#8fa2b8}dd{margin:0}
.note{color:#8fa2b8;font-size:.78rem;line-height:1.45;margin-top:14px;border-top:1px solid #2a3548;padding-top:10px}
@media(max-width:900px){.panel{grid-template-columns:1fr}}
</style></head><body><main>
<h1>__TITLE__</h1>
<p class="sub">__SUBTITLE__</p>
<div class="panel">
  <div>
    <canvas id="spec" width="820" height="620"></canvas>
    <section class="controls">
      <input id="pos" type="range" min="0" step="1" value="0">
      <div class="row">
        <button id="prev">&minus;</button><button id="next">+</button>
        <label>Signal <select id="sig"></select></label>
        <label>Scale <select id="scale"><option value="global">global</option><option value="pos">per position</option></select></label>
        <label>Band <select id="band"><option value="full">full</option><option value="low">low 10%</option></select></label>
      </div>
      <div class="row" id="toggles"></div>
      <dl>
        <dt>Position</dt><dd id="pTxt">&mdash;</dd>
        <dt>Index</dt><dd id="iTxt">&mdash;</dd>
        <dt>Shots</dt><dd id="nTxt">&mdash;</dd>
        <dt>Cursor</dt><dd id="hTxt">&mdash;</dd>
      </dl>
      <p class="note">__NOTE__</p>
    </section>
  </div>
  <canvas id="loc" width="330" height="400"></canvas>
</div>
<script>
const D=__PAYLOAD__;
function decode(s){const b=atob(s.b64),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);
  return new Float32Array(u.buffer);}
const M={},E={},N={};
for(const k in D.mean){M[k]=decode(D.mean[k]);E[k]=decode(D.sem[k]);N[k]=decode(D.n[k]);}
const nf=D.freq_khz.length,np_=D.x.length;
const cv=document.getElementById('spec'),g=cv.getContext('2d');
const lc=document.getElementById('loc'),lg=lc.getContext('2d');
const sl=document.getElementById('pos');sl.max=np_-1;
const sig=document.getElementById('sig');
D.signals.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;sig.appendChild(o);});
sig.value=D.signals[0];
if(D.signals.length<2)sig.disabled=true;
// One checkbox per group, on by default: comparing states is the point, but a
// crowded page needs a way to drop one without regenerating the file.
const on={};
D.groups.forEach(grp=>{on[grp]=true;
  const w=document.createElement('label');w.style.color=D.colors[grp];
  const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;
  cb.onchange=()=>{on[grp]=cb.checked;draw();};
  w.appendChild(cb);w.appendChild(document.createTextNode(' '+grp));
  document.getElementById('toggles').appendChild(w);});
function key(grp){return sig.value+' '+grp;}
function val(A,grp,p,f){const a=A[key(grp)];return a?a[p*nf+f]:NaN;}
function shots(grp,p){const a=N[key(grp)];return a?a[p]:NaN;}
function fmt(v){if(!Number.isFinite(v))return '—';
  const m=Math.abs(v);return (m!==0&&(m<1e-3||m>=1e5))?v.toExponential(3):v.toFixed(4);}
function active(){return D.groups.filter(grp=>on[grp]&&M[key(grp)]);}
function nfMax(){return document.getElementById('band').value==='low'
  ?Math.max(2,Math.ceil(nf/10)):nf;}
// One extractor feeds both the axis limits and the drawing, so the bins the
// curve is built from and the bins the range is computed over cannot drift.
// Returns finite, positive bins only, each with its shading band.
function bins(grp,p){const out=[],fm=nfMax();
  for(let f=1;f<fm;f++){const m=val(M,grp,p,f),e=val(E,grp,p,f);
    if(!Number.isFinite(m)||m<=0)continue;
    const d=Number.isFinite(e)?e:0;
    out.push({f:D.freq_khz[f],mid:m,lo:Math.max(m-d,m*1e-3),hi:m+d});}
  return out;}
// Keyed by everything the global range depends on, so a stale entry cannot be
// reused after the signal, band, or group selection changes.
let GL={};
function limits(p){let lo=Infinity,hi=-Infinity;
  const ps=(p===null)?[...Array(np_).keys()]:[p];
  for(const q of ps)for(const grp of active())for(const b of bins(grp,q)){
    if(b.lo<lo)lo=b.lo;if(b.hi>hi)hi=b.hi;}
  if(!(hi>lo))return [1e-12,1e-9];
  return [lo,hi];}
const L=88,T=30,W=680,H=470;
function draw(){
  const p=+sl.value,fm=nfMax(),F=D.freq_khz;
  let lo,hi;
  if(document.getElementById('scale').value==='global'){
    const gk=sig.value+'|'+document.getElementById('band').value+'|'+active().join(',');
    if(!GL[gk])GL[gk]=limits(null);
    [lo,hi]=GL[gk];}
  else [lo,hi]=limits(p);
  const y0=Math.floor(Math.log10(lo)),y1=Math.ceil(Math.log10(hi));
  const fmax=F[fm-1]||1;
  const X=v=>L+v/fmax*W,Y=v=>T+H-(Math.log10(v)-y0)*H/((y1-y0)||1);
  g.fillStyle='#fff';g.fillRect(0,0,cv.width,cv.height);
  g.font='12px system-ui';
  const step=Math.max(1,Math.round(fmax/8));
  for(let t=0;t<=fmax;t+=step){const x=X(t);
    g.strokeStyle='#e2e2e2';g.beginPath();g.moveTo(x,T);g.lineTo(x,T+H);g.stroke();
    g.fillStyle='#222';g.textAlign='center';g.fillText(t,x,T+H+18);}
  for(let q=y0;q<=y1;q++){const y=Y(10**q);
    g.strokeStyle='#e2e2e2';g.beginPath();g.moveTo(L,y);g.lineTo(L+W,y);g.stroke();
    g.fillStyle='#222';g.textAlign='right';g.fillText('10^'+q,L-8,y+4);}
  const floor=10**y0;
  active().forEach(grp=>{
    const b=bins(grp,p);
    if(!b.length)return;
    const C=(v)=>Y(Math.max(floor,v));
    g.beginPath();
    b.forEach((q,j)=>{j?g.lineTo(X(q.f),C(q.hi)):g.moveTo(X(q.f),C(q.hi));});
    for(let j=b.length-1;j>=0;j--)g.lineTo(X(b[j].f),C(b[j].lo));
    g.closePath();g.fillStyle=D.fills[grp];g.fill();
    g.strokeStyle=D.colors[grp];g.lineWidth=1.9;g.beginPath();
    b.forEach((q,j)=>{j?g.lineTo(X(q.f),C(q.mid)):g.moveTo(X(q.f),C(q.mid));});
    g.stroke();});
  g.strokeStyle='#222';g.lineWidth=1;g.strokeRect(L,T,W,H);
  g.fillStyle='#111';g.font='14px system-ui';g.textAlign='center';
  g.fillText('frequency (kHz)',L+W/2,T+H+48);
  g.save();g.translate(20,T+H/2);g.rotate(-Math.PI/2);
  g.fillText('PSD (V²/Hz)',0,0);g.restore();
  g.font='12px system-ui';g.textAlign='left';
  let lx=L+12,ly=T+16;
  active().forEach(grp=>{
    g.strokeStyle=D.colors[grp];g.lineWidth=3;
    g.beginPath();g.moveTo(lx,ly);g.lineTo(lx+22,ly);g.stroke();
    g.fillStyle='#111';g.fillText(grp+'  (n='+fmt(shots(grp,p))+')',lx+28,ly+4);
    ly+=18;});
  document.getElementById('pTxt').textContent='('+D.x[p]+', '+D.y[p]+') mm';
  document.getElementById('iTxt').textContent=(p+1)+' / '+np_;
  document.getElementById('nTxt').textContent=
    active().map(grp=>grp+': '+fmt(shots(grp,p))).join(', ')||'—';
  locator(p);
}
function locator(p){
  const xs=D.x,ys=D.y,pad=26,W2=lc.width-2*pad,H2=lc.height-2*pad-22;
  let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  // A line scan has zero extent on one axis; pad it so the points do not collapse
  // onto the border and the same drawing code serves lines and planes alike.
  if(xmax-xmin<1e-9){xmin-=1;xmax+=1;}
  if(ymax-ymin<1e-9){ymin-=1;ymax+=1;}
  const X=v=>pad+(v-xmin)/(xmax-xmin)*W2,Y=v=>pad+H2-(v-ymin)/(ymax-ymin)*H2;
  lg.fillStyle='#fff';lg.fillRect(0,0,lc.width,lc.height);
  lg.strokeStyle='#222';lg.strokeRect(pad,pad,W2,H2);
  for(let i=0;i<np_;i++){lg.fillStyle=i===p?'#e31a1c':'rgba(45,90,140,.45)';
    lg.beginPath();lg.arc(X(xs[i]),Y(ys[i]),i===p?5:2.2,0,2*Math.PI);lg.fill();}
  lg.fillStyle='#111';lg.font='13px system-ui';lg.textAlign='center';
  lg.fillText('probe x (mm)',lc.width/2,lc.height-6);
  lg.save();lg.translate(11,pad+H2/2);lg.rotate(-Math.PI/2);
  lg.fillText('probe y (mm)',0,0);lg.restore();
  lg.font='bold 13px system-ui';
  lg.fillText('('+xs[p]+', '+ys[p]+') mm',lc.width/2,17);
}
sl.addEventListener('input',draw);
sig.addEventListener('change',draw);
document.getElementById('scale').addEventListener('change',draw);
document.getElementById('band').addEventListener('change',draw);
document.getElementById('prev').onclick=()=>{sl.value=Math.max(0,+sl.value-1);draw();};
document.getElementById('next').onclick=()=>{sl.value=Math.min(np_-1,+sl.value+1);draw();};
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),
  mx=(e.clientX-r.left)*cv.width/r.width,my=(e.clientY-r.top)*cv.height/r.height;
  if(mx<L||mx>=L+W||my<T||my>=T+H){document.getElementById('hTxt').textContent='—';return;}
  const fm=nfMax(),fmax=D.freq_khz[fm-1]||1,fq=(mx-L)/W*fmax;
  let best=1,bd=Infinity;
  for(let f=1;f<fm;f++){const d=Math.abs(D.freq_khz[f]-fq);if(d<bd){bd=d;best=f;}}
  const p=+sl.value;
  document.getElementById('hTxt').textContent=D.freq_khz[best].toFixed(3)+' kHz: '+
    active().map(grp=>grp+'='+fmt(val(M,grp,p,best))).join(', ');});
draw();
</script></main></body></html>
"""

# Distinct at a glance and colorblind-tolerable; recycled if a run somehow
# classifies into more groups than this. Kept as RGB triples so the stroke and
# the translucent fill are both built from one source.
_COLORS = [(214, 39, 40), (31, 119, 180), (148, 103, 189),
           (85, 85, 85), (44, 160, 44), (255, 127, 14)]

# The pooled all-shot curve is a reference line rather than another condition,
# so it takes a fixed neutral gray instead of a slot in the rotation. That also
# keeps the condition colors identical to runs made without it: the labels are
# sorted, and "all shots (avg)" would otherwise sort to the front and shift
# every real group one position along the palette.
_POOLED_COLOR = (110, 110, 110)


def write_html(out_file, acc, freq_khz, coords, signals, title, subtitle, note,
               pooled_label=None):
    """Embed every (signal, group) spectrum in one standalone page.

    Returns the payload size in bytes. Groups are sorted so the legend order is
    stable between runs; signals keep the order the caller asked for. ``acc`` is
    keyed ``"signal_key group_label"``, the same key the page rebuilds.

    ``pooled_label``, when present, is held out of that sort and appended last so
    the ungrouped baseline reads as a reference curve under the conditions it is
    being compared against, rather than as one more peer in the legend.
    """
    mean, sem, n = {}, {}, {}
    labels = {k.split(" ", 1)[1] for k in acc}
    pooled = pooled_label if pooled_label in labels else None
    groups = sorted(labels - {pooled}) + ([pooled] if pooled else [])
    for k, (m, e, c) in acc.items():
        mean[k] = encode_array(m)
        sem[k] = encode_array(e)
        n[k] = encode_array(c)

    # Resolve the palette here, once per group, so the page indexes by label and
    # no modulo/index coupling (or rgb->rgba string surgery) survives in the JS.
    colors, fills = {}, {}
    for i, label in enumerate(groups):
        r, g, b = _POOLED_COLOR if label == pooled else _COLORS[i % len(_COLORS)]
        colors[label] = f"rgb({r},{g},{b})"
        fills[label] = f"rgba({r},{g},{b},.14)"

    payload = {
        "freq_khz": [round(float(v), 6) for v in freq_khz],
        "x": [round(float(v), 4) for v in coords[:, 0]],
        "y": [round(float(v), 4) for v in coords[:, 1]],
        "signals": [s.key for s in signals],
        "groups": groups,
        "colors": colors, "fills": fills,
        "mean": mean, "sem": sem, "n": n,
    }
    blob = json.dumps(payload, separators=(",", ":"))
    html = (_HTML.replace("__TITLE__", title)
                 .replace("__SUBTITLE__", subtitle)
                 .replace("__NOTE__", note)
                 .replace("__PAYLOAD__", blob))
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(html)
    return len(blob)


# ======================================================================================
# Driver
# ======================================================================================

def _ms(value, fallback):
    """Format a ms knob for a label, falling back when it is None.

    Both being None means the window runs to the edge of the record, whose
    numeric value is not known until the file is open; the label says so rather
    than guessing, and the page subtitle carries the resolved numbers anyway.
    """
    if value is None:
        value = fallback
    return "?" if value is None else f"{value:g}"


def _with_pooled_group(selector, label):
    """Wrap a selector so every position also yields one all-shot group.

    The pooled group is built from the position's full shot list, not from the
    union of the base selector's groups, so it stays a true "every shot here"
    average even when the classifier drops shots it could not label. That makes
    it the honest ungrouped baseline: if it diverges from a shot-count-weighted
    blend of the grouped curves, the difference is exactly the unclassified
    shots, which is worth seeing rather than hiding.

    A position whose base selector returns nothing still returns nothing, so an
    excluded position does not reappear carrying only this curve.
    """
    def wrapped(shotnums):
        groups = selector(shotnums)
        if not groups:
            return groups
        return {**groups, label: list(shotnums)}
    return wrapped


def _resolve_selector(f, scope, shot_mode, shot_index, state_groups,
                      include_all_shots=False, all_shots_label=ALL_SHOTS_LABEL):
    """Build the shot selector, running the RMS classifier only if asked.

    Identical contract to :func:`read_and_analyze.plot_xy_slider._resolve_selector`
    -- the grouping question is the same one, and answering it differently here
    would mean the two pages could disagree about what a state is -- except for
    the optional pooled group, which only adds a curve and never changes how a
    shot is labelled.
    """
    if shot_mode == "index":
        return select_shot_index(shot_index), f"shot {shot_index}, no averaging"
    if shot_mode == "state":
        if not state_groups:
            raise ValueError("SHOT_MODE='state' requires STATE_GROUPS to be set")
        cls_scope = state_groups.get("scope", scope)
        channels = state_groups["channels"]
        labels = state_groups.get("labels") or default_labels(channels)
        shots = _shot_numbers(f[cls_scope])
        print(f"classifying {len(shots)} shots on {cls_scope}/"
              f"{','.join(channels)} ...")
        state_by_shot, meta = classify_by_channel_rms(
            f, cls_scope, channels, shots, labels,
            state_groups.get("window_ms"),
            state_groups.get("min_ratio", 2.0))
        counts = ", ".join(f"{k}: {v}" for k, v in sorted(meta["state_counts"].items()))
        selector = select_by_state(state_by_shot)
        how = f"grouped by antenna state ({counts})"
        if include_all_shots:
            selector = _with_pooled_group(selector, all_shots_label)
            how += f", plus '{all_shots_label}'"
        return selector, how
    return select_all_shots, "mean over all shots at each position, with SEM"


def plot_fft_slider(path, scope=None, channels=None, signals=None,
                    shot_mode=None, shot_index=None,
                    t_start=None, t_end=None, f_max_khz=None, segment_ms=None,
                    med_size=None, gauss_sigma=None, state_groups=None,
                    include_all_shots=None, all_shots_label=None,
                    alt_start=None, alt_end=None,
                    output_path=None, save_npz=True):
    """Write one standalone HTML position-slider FFT page per scope.

    Every channel of a scope goes into the same page behind a dropdown. Returns
    the list of files written; every argument defaults to the corresponding knob
    when left as None.
    """
    scope = _SCOPE if scope is None else scope
    channels = _CHANNELS if channels is None else channels
    shot_mode = SHOT_MODE if shot_mode is None else shot_mode
    shot_index = SHOT_INDEX if shot_index is None else shot_index
    t_start = FFT_T_START_MS if t_start is None else t_start
    t_end = FFT_T_END_MS if t_end is None else t_end
    f_max_khz = FFT_F_MAX_KHZ if f_max_khz is None else f_max_khz
    segment_ms = FFT_SEGMENT_MS if segment_ms is None else segment_ms
    med_size = MED_SIZE if med_size is None else med_size
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    state_groups = STATE_GROUPS if state_groups is None else state_groups
    include_all_shots = (INCLUDE_ALL_SHOTS if include_all_shots is None
                         else include_all_shots)
    alt_start = ALL_SHOTS_T_START_MS if alt_start is None else alt_start
    alt_end = ALL_SHOTS_T_END_MS if alt_end is None else alt_end
    if all_shots_label is None:
        all_shots_label = ALL_SHOTS_LABEL
    # The label names the window it was computed over, so the legend cannot claim
    # a baseline curve shares the grouped curves' window when it does not.
    if all_shots_label is None:
        all_shots_label = ("all shots (avg)" if alt_start is None and alt_end is None
                           else f"all shots {_ms(alt_start, t_start)}"
                                f"–{_ms(alt_end, t_end)} ms")
    output_path = OUTPUT_PATH if output_path is None else output_path

    sig_list = as_signal_list(signals)
    base = os.path.splitext(os.path.basename(path))[0]
    written = []

    with open_hdf5_readonly(path) as f:
        positions = read_positions(f)
        if not positions:
            print("no /Control/Positions data (not a bmotion file?) — nothing to do")
            return written

        scopes = [scope] if scope else _scope_groups(f)
        for sc in scopes:
            sigs = sig_list
            if sigs is None:
                chans = (_as_list(channels) if channels is not None
                         else _channel_names(f[sc], _shot_numbers(f[sc])[0]))
                sigs = [raw_signal(c) for c in chans]
            if not sigs:
                print(f"scope '{sc}': no channel to analyze — skipping")
                continue

            out_file = resolve_output(path, output_path, base, sc, len(scopes))
            selector, how = _resolve_selector(f, sc, shot_mode, shot_index,
                                              state_groups, include_all_shots,
                                              all_shots_label)
            print(f"scope '{sc}': {len(sigs)} signal(s) — {how}")

            acc, freq_khz, coords, meta = build_spectra(
                f, sc, sigs, positions, selector, t_start, t_end,
                f_max_khz, segment_ms, med_size, gauss_sigma,
                alt_label=all_shots_label if include_all_shots else None,
                alt_t_start_ms=alt_start, alt_t_end_ms=alt_end)
            if not acc:
                print(f"  no usable shots for {sc} — skipping")
                continue
            if meta["ntrimmed"]:
                print(f"  note: dropped {meta['ntrimmed']} trailing sample(s) so the "
                      f"transform length ({meta['nsamples'] - meta['ntrimmed']}) "
                      "factors into small primes")

            est = ("periodogram over the whole window" if meta["nperseg"] is None
                   else f"Welch, {meta['nperseg']} samples/segment, 50% overlap")
            filt = (f"median {med_size} / gaussian {gauss_sigma} samples"
                    if (med_size > 1 or gauss_sigma > 0) else "unfiltered")
            # Spelled out because the subtitle's window applies only to the
            # grouped curves once a baseline window is in play, and a reader who
            # missed that would compare two different intervals as if they were
            # one. The segment count is the honest caveat: a short baseline
            # window averages fewer segments and so looks noisier.
            alt_note = ""
            if meta["alt_window_ms"]:
                a_lo, a_hi = meta["alt_window_ms"]
                nseg = meta["alt_nseg"]
                alt_note = (f"The '{all_shots_label}' curve pools every shot at the "
                            f"position over {a_lo:.4g}–{a_hi:.4g} ms instead, on the "
                            "same frequency axis"
                            + (f" but averaging {nseg} segment(s) rather than "
                               f"{meta['nseg']}, so it carries more scatter"
                               if nseg and meta.get("nseg") and nseg < meta["nseg"]
                               else "")
                            + ". ")
            nbytes = write_html(
                out_file, acc, freq_khz, coords, sigs,
                title=f"{base} — {sc} FFT vs position",
                subtitle=(f"{len(coords)} positions · {freq_khz[-1]:.4g} kHz band · "
                          f"{meta['window_ms'][0]:.4g}–{meta['window_ms'][1]:.4g} ms · {how}"),
                note=(f"Estimator: Hann {est}; df = {meta['df_hz']:.4g} Hz. Each curve "
                      "is the mean of the per-shot PSDs in that group at that position, "
                      "with the shot standard error shaded (absent when a group has one "
                      f"shot). {alt_note}Filtering: {filt}. Values are scope V²/Hz with "
                      "no normalization. Spectra are embedded in this file — it needs "
                      "neither the HDF5 nor Python to open."),
                pooled_label=all_shots_label if include_all_shots else None)
            written.append(out_file)
            print(f"  wrote {out_file}  ({nbytes/1e6:.2f} MB payload)")

            if save_npz:
                npz = os.path.splitext(out_file)[0] + ".npz"
                arrays = {"freq_khz": freq_khz, "x": coords[:, 0], "y": coords[:, 1]}
                for k, (m, e, c) in acc.items():
                    stem = k.replace(" ", "__")
                    arrays[f"{stem}__mean"] = m
                    arrays[f"{stem}__sem"] = e
                    arrays[f"{stem}__n"] = c
                # Uncompressed for the same reason plot_xy_slider is: deflate
                # costs far more time than the megabytes it saves on data the
                # HTML already carries.
                np.savez(npz, **arrays)
                written.append(npz)
                print(f"  wrote {npz}")

    return written


def main():
    plot_fft_slider(resolve_data_file())


if __name__ == "__main__":
    main()
