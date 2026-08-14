# -*- coding: utf-8 -*-
"""
Interactive XY-plane map with a time slider, written as a standalone HTML page.

Where :mod:`read_and_analyze.plot_xy_map` renders one plane (or a fixed montage
of a few snapshot times), this module renders *many* time steps into a single
self-contained HTML file with a slider and a Play button, so a structure moving
across the probe plane can be watched instead of inferred from four panels.

Three things are configurable beyond the shared knobs:

* **What is mapped** -- a raw channel, or arithmetic across channels of the same
  scope (``"C3 - C4"``, ``"sqrt(C2*C2 + C3*C3)"``). See
  :mod:`read_and_analyze.signals`.
* **Which shots are averaged** -- all shots at each position (the default, with
  the repeat standard error alongside), one shot by index, or grouped by antenna
  state classified from monitor-channel RMS (see
  :mod:`read_and_analyze.state_grouping`).
* **Where the file lands** -- ``output_path``; see :func:`resolve_output`.

The written page needs neither Python nor the HDF5 file: the reduced frames are
embedded in it. The expensive read happens once, here; the page is its frozen
result, and changing the window, frame count, filtering, or grouping means
re-running this module.

There is NO command line; knobs live below and in analysis_config.py. Run with:
    python -m read_and_analyze.plot_xy_slider

Created Aug.2026
@author: Jia Han
"""

import base64
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scope_io import open_hdf5_readonly, read_hdf5_scope_tarr

try:  # works as a package (python -m read_and_analyze.plot_xy_slider)
    from read_and_analyze.read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
        resolve_data_file,
    )
    from read_and_analyze.plot_xy_map import (
        build_frames, select_all_shots, select_shot_index, select_by_state,
        _plane_axes, _is_plane, window_indices,
    )
    from read_and_analyze.signals import as_signal_list, raw as raw_signal
    from read_and_analyze.state_grouping import (
        classify_by_channel_rms, default_labels,
    )
    from read_and_analyze.analysis_config import (
        MED_SIZE, GAUSS_SIGMA, SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS,
        XY_CMAP as CMAP,
    )
except ImportError:  # fallback when run directly from inside the folder
    from read_bmotion_data import (
        read_positions, _scope_groups, _shot_numbers, _channel_names,
        resolve_data_file,
    )
    from plot_xy_map import (
        build_frames, select_all_shots, select_shot_index, select_by_state,
        _plane_axes, _is_plane, window_indices,
    )
    from signals import as_signal_list, raw as raw_signal
    from state_grouping import classify_by_channel_rms, default_labels
    from analysis_config import (
        MED_SIZE, GAUSS_SIGMA, SELECT_SCOPE as SCOPE, SELECT_CHAN as CHANNELS,
        XY_CMAP as CMAP,
    )

# ---- knobs ---- (module-private; shared ones come from analysis_config.py)
# None means "use the digitized record": tarr already *is* the digitized window,
# so this covers a full discharge for a Langmuir run and confines itself to the
# digitized segment for a B-dot run, with no per-run editing.
T_START_MS = None     # window start (ms); None = start of the record
T_END_MS   = None    # window end   (ms); None = end of the record
N_FRAMES   = 1000     # time steps in the slider; file size scales with this

SHOT_MODE  = "state"  # "mean" = average all shots (+ SEM) | "index" | "state"
SHOT_INDEX = 0       # which shot, when SHOT_MODE == "index"

# SIGNALS: None = one raw signal per channel in SELECT_CHAN. Otherwise a list of
# expressions or Signal objects, e.g. ["C2", "C3 - C4", "sqrt(C2*C2+C3*C3)"].
SIGNALS = ["C1","C2","C3"]

# Group shots based on the following channels
STATE_GROUPS = {"channels": ("C7", "C8"), "scope": "bdot_scope", "window_ms": (0.0, 20.0)}

# Where the HTML lands. None = a "plots/" subdir beside the data file (the
# convention the other modules follow); a directory = auto-named files inside
# it; a path ending in .html = that exact file (only when one page is written).
OUTPUT_PATH = r"E:\Shadow data\Alfven_zonal_flow\aug2026\results"


# ======================================================================================
# Time axis
# ======================================================================================

def frame_indices(tarr, t_start_ms, t_end_ms, n_frames):
    """Evenly spaced sample indices spanning the requested window.

    ``None`` bounds fall back to the ends of ``tarr``. Requests outside the
    record are clamped to it (with a notice) rather than silently producing
    empty frames. Returns ``(idxs, times_ms)`` with the realized, tarr-snapped
    times -- never the requested ones, so the page reports what was actually
    sampled.
    """
    t_lo = tarr[0] if t_start_ms is None else t_start_ms * 1e-3
    t_hi = tarr[-1] if t_end_ms is None else t_end_ms * 1e-3

    lo, hi = float(tarr[0]), float(tarr[-1])
    if t_lo < lo or t_hi > hi:
        print(f"  note: requested window [{t_lo*1e3:.4g}, {t_hi*1e3:.4g}] ms is "
              f"outside the digitized record [{lo*1e3:.4g}, {hi*1e3:.4g}] ms; clamping")
    t_lo = max(lo, min(t_lo, hi))
    t_hi = max(lo, min(t_hi, hi))
    if t_hi <= t_lo:
        t_lo, t_hi = lo, hi

    i0, i1 = window_indices(tarr, t_lo, t_hi)

    n = max(1, min(int(n_frames), i1 - i0))
    idxs = np.unique(np.linspace(i0, i1 - 1, n).astype(int)).tolist()
    return idxs, [float(tarr[i] * 1e3) for i in idxs]


# ======================================================================================
# Payload encoding
# ======================================================================================

def encode_array(a):
    """Encode a float array as little-endian float32 bytes in base64.

    Chosen over a JSON number list for size and parse speed: JSON writes each
    float at full repr precision (~20 chars for a value whose useful content is
    5), while this is a fixed 4 bytes regardless of magnitude -- about 4x smaller
    overall, and the browser decodes it with one ``atob`` into a Float32Array
    instead of tokenizing hundreds of thousands of numbers.

    float32 carries ~7 significant digits, far more than a scope voltage needs;
    the .npz sidecar is the bit-exact record. NaN survives the round trip and is
    caught by ``Number.isFinite`` in the page, so empty cells stay grey.
    """
    arr = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    return {"b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            "shape": list(arr.shape)}


# ======================================================================================
# Output location
# ======================================================================================

def resolve_output(path, output_path, base, scope, key, n_expected):
    """Decide where one page is written.

    ``output_path`` may be None (a ``plots/`` dir beside the data file, matching
    every other module), a directory (auto-named files inside it), or a path
    ending in ``.html`` (used verbatim).

    An explicit filename is only unambiguous when the scope x signal loop yields
    exactly one page, so ``n_expected > 1`` raises instead of quietly leaving one
    file where several were expected.
    """
    auto = f"{base}_{scope}_{key}_xyslider.html"
    if output_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "plots")
    elif str(output_path).lower().endswith(".html"):
        if n_expected > 1:
            raise ValueError(
                f"output_path={output_path!r} names a single .html file but this "
                f"run would write {n_expected} pages. Pass a directory instead, "
                "or narrow SELECT_SCOPE / SELECT_CHAN / SIGNALS to one signal.")
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
main{max-width:1140px;margin:auto;padding:22px}
h1{margin:0 0 4px;font-size:1.3rem}
.sub{color:#9fb0c4;font-size:.85rem;margin:0 0 16px}
.panel{display:grid;grid-template-columns:minmax(400px,720px) 1fr;gap:18px;align-items:start}
canvas{width:100%;aspect-ratio:1/1;background:#fff;border-radius:8px}
.controls{background:#182030;border:1px solid #2a3548;border-radius:8px;padding:14px}
button,select{color:#e8eef6;background:#25334a;border:1px solid #3e4d67;border-radius:5px;padding:6px 10px;font-size:.9rem}
input[type=range]{width:100%;margin:12px 0}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
label{font-size:.85rem;color:#9fb0c4}
dl{display:grid;grid-template-columns:auto 1fr;gap:6px 12px;margin:14px 0 0;font-variant-numeric:tabular-nums;font-size:.88rem}
dt{color:#8fa2b8}dd{margin:0}
.note{color:#8fa2b8;font-size:.78rem;line-height:1.45;margin-top:14px;border-top:1px solid #2a3548;padding-top:10px}
@media(max-width:860px){.panel{grid-template-columns:1fr}}
</style></head><body><main>
<h1>__TITLE__</h1>
<p class="sub">__SUBTITLE__</p>
<div class="panel">
  <canvas id="map" width="760" height="760"></canvas>
  <section class="controls">
    <div class="row"><button id="play">&#9654; Play</button><button id="prev">&minus;</button><button id="next">+</button></div>
    <input id="time" type="range" min="0" step="1" value="0">
    <div class="row">
      <label>Group <select id="group"></select></label>
      <label>Field <select id="field"><option value="value">mean</option><option value="sem">standard error</option><option value="n">shots (n)</option></select></label>
    </div>
    <div class="row">
      <label>Scale <select id="scale"><option value="global">global</option><option value="frame">per frame</option></select></label>
      <label>Speed <select id="speed"><option value="120">slow</option><option value="60" selected>normal</option><option value="20">fast</option></select></label>
    </div>
    <dl>
      <dt>Time</dt><dd id="tTxt">&mdash;</dd>
      <dt>Frame</dt><dd id="fTxt">&mdash;</dd>
      <dt>Mean</dt><dd id="mTxt">&mdash;</dd>
      <dt>Range</dt><dd id="rTxt">&mdash;</dd>
      <dt>Peak</dt><dd id="pTxt">&mdash;</dd>
      <dt>Cursor</dt><dd id="hTxt">&mdash;</dd>
    </dl>
    <p class="note">__NOTE__</p>
  </section>
</div>
<script>
const D=__PAYLOAD__;
function decode(s){const b=atob(s.b64),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);
  return {a:new Float32Array(u.buffer),shape:s.shape};}
const F={};for(const k in D.fields)F[k]=decode(D.fields[k]);
const nx=D.x.length,ny=D.y.length,nt=D.times_ms.length;
const cv=document.getElementById('map'),cx=cv.getContext('2d');
const sl=document.getElementById('time');sl.max=nt-1;
const gsel=document.getElementById('group');
D.groups.forEach(g=>{const o=document.createElement('option');o.value=g;o.textContent=g;gsel.appendChild(o);});
gsel.value=D.groups[0];   // explicit: never rely on implicit first-option selection
if(D.groups.length<2)gsel.disabled=true;
let playing=false,timer=null;
// Canvas geometry, defined once: draw() and the hover handler must agree on the
// plot rectangle or the readout reports the wrong cell.
const L=80,T=32,W=560,H=560,CB=672,CW=22;
const AN=[[0,[68,1,84]],[.13,[71,44,122]],[.25,[59,81,139]],[.38,[44,113,142]],[.5,[33,144,141]],[.63,[39,173,129]],[.75,[92,200,99]],[.88,[170,220,50]],[1,[253,231,37]]];
function col(q){q=Math.max(0,Math.min(1,q));let a=AN[0],b=AN[AN.length-1];
  for(let i=1;i<AN.length;i++)if(q<=AN[i][0]){a=AN[i-1];b=AN[i];break;}
  const f=(q-a[0])/((b[0]-a[0])||1);
  return 'rgb('+a[1].map((v,j)=>Math.round(v+f*(b[1][j]-v))).join(',')+')';}
function key(){return gsel.value+'/'+document.getElementById('field').value;}
function fld(){return F[key()]||F[D.groups[0]+'/value'];}
function at(fr,iy,ix){return fld().a[fr*ny*nx+iy*nx+ix];}
// Same flat indexing as at(), but for an explicit group/field (the hover readout
// shows value, SEM and n together regardless of which field is displayed).
function cell(g,f,fr,iy,ix){const s=F[g+'/'+f];return s?s.a[fr*ny*nx+iy*nx+ix]:NaN;}
function fmt(v){if(!Number.isFinite(v))return '—';
  const m=Math.abs(v);return (m!==0&&(m<1e-3||m>=1e5))?v.toExponential(3):v.toFixed(4);}
function lims(){const g=fld().a;let lo=Infinity,hi=-Infinity;
  for(let i=0;i<g.length;i++){const v=g[i];if(Number.isFinite(v)){if(v<lo)lo=v;if(v>hi)hi=v;}}
  return [lo,hi];}
let GL=null;
function draw(){
  const fr=+sl.value;
  let lo,hi;
  if(document.getElementById('scale').value==='global'){if(!GL)GL=lims();[lo,hi]=GL;}
  else{lo=Infinity;hi=-Infinity;for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const v=at(fr,y,x);
    if(Number.isFinite(v)){if(v<lo)lo=v;if(v>hi)hi=v;}}}
  cx.clearRect(0,0,cv.width,cv.height);cx.fillStyle='#fff';cx.fillRect(0,0,cv.width,cv.height);
  // row 0 of the plane is max-y (acquisition order), so draw it at the top.
  let sum=0,cnt=0,pk=-Infinity,pxi=0,pyi=0;
  for(let sy=0;sy<ny;sy++)for(let ix=0;ix<nx;ix++){
    const v=at(fr,sy,ix);
    cx.fillStyle=Number.isFinite(v)?col((v-lo)/((hi-lo)||1)):'#b9b9b9';
    cx.fillRect(L+ix*W/nx,T+sy*H/ny,W/nx+.5,H/ny+.5);
    if(Number.isFinite(v)){sum+=v;cnt++;if(v>pk){pk=v;pxi=ix;pyi=sy;}}}
  cx.strokeStyle='#222';cx.strokeRect(L,T,W,H);
  cx.fillStyle='#111';cx.font='16px system-ui';cx.textAlign='center';
  const xstep=Math.max(1,Math.ceil(nx/8)),ystep=Math.max(1,Math.ceil(ny/8));
  for(let k=0;k<nx;k+=xstep)cx.fillText(D.x[k],L+(k+.5)*W/nx,T+H+26);
  cx.fillText('probe x (mm)',L+W/2,T+H+54);
  cx.save();cx.translate(20,T+H/2);cx.rotate(-Math.PI/2);cx.fillText('probe y (mm)',0,0);cx.restore();
  cx.textAlign='right';
  for(let k=0;k<ny;k+=ystep)cx.fillText(D.y[k],L-9,T+(k+.5)*H/ny+6);
  for(let p=0;p<180;p++){cx.fillStyle=col(1-p/179);cx.fillRect(CB,T+p*H/180,CW,H/180+1);}
  cx.strokeRect(CB,T,CW,H);cx.fillStyle='#111';cx.textAlign='left';
  cx.fillText(fmt(hi),CB+29,T+8);cx.fillText(fmt(lo),CB+29,T+H);
  cx.save();cx.translate(750,T+H/2);cx.rotate(-Math.PI/2);cx.textAlign='center';
  cx.fillText(document.getElementById('field').value==='sem'?('SEM of '+D.label):D.label,0,0);cx.restore();
  document.getElementById('tTxt').textContent=D.times_ms[fr].toFixed(4)+' ms';
  document.getElementById('fTxt').textContent=(fr+1)+' / '+nt;
  document.getElementById('mTxt').textContent=cnt?fmt(sum/cnt):'—';
  document.getElementById('rTxt').textContent=fmt(lo)+'  …  '+fmt(hi);
  document.getElementById('pTxt').textContent=cnt?fmt(pk)+' at ('+D.x[pxi]+', '+D.y[pyi]+') mm':'—';
}
function reset(){GL=null;draw();}
sl.addEventListener('input',draw);
document.getElementById('field').addEventListener('change',reset);
gsel.addEventListener('change',reset);
document.getElementById('scale').addEventListener('change',draw);
document.getElementById('prev').onclick=()=>{sl.value=Math.max(0,+sl.value-1);draw();};
document.getElementById('next').onclick=()=>{sl.value=Math.min(nt-1,+sl.value+1);draw();};
function run(){clearInterval(timer);
  if(playing)timer=setInterval(()=>{sl.value=(+sl.value+1)%nt;draw();},+document.getElementById('speed').value);}
document.getElementById('play').onclick=function(){playing=!playing;
  this.innerHTML=playing?'&#10074;&#10074; Pause':'&#9654; Play';run();};
document.getElementById('speed').addEventListener('change',run);
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),
  mx=(e.clientX-r.left)*cv.width/r.width,my=(e.clientY-r.top)*cv.height/r.height;
  if(mx<L||mx>=L+W||my<T||my>=T+H){document.getElementById('hTxt').textContent='—';return;}
  const ix=Math.floor((mx-L)/W*nx),iy=Math.floor((my-T)/H*ny),fr=+sl.value,g=gsel.value;
  const v=cell(g,'value',fr,iy,ix),s=cell(g,'sem',fr,iy,ix),n=cell(g,'n',fr,iy,ix);
  document.getElementById('hTxt').textContent='('+D.x[ix]+', '+D.y[iy]+') mm: '+fmt(v)+
    ' ± '+fmt(s)+'  n='+(Number.isFinite(n)?n:'—');});
draw();
</script></main></body></html>
"""


def write_html(out_file, frames, xpos, ypos, times_ms, label, title, subtitle, note):
    """Embed the frames in a standalone page. Returns the payload size in bytes."""
    fields = {}
    for group, (v, e, n) in frames.items():
        fields[f"{group}/value"] = encode_array(v)
        fields[f"{group}/sem"] = encode_array(e)
        fields[f"{group}/n"] = encode_array(n)

    payload = {
        "x": [round(float(x), 4) for x in xpos],
        "y": [round(float(y), 4) for y in ypos],
        "times_ms": [round(float(t), 6) for t in times_ms],
        "groups": sorted(frames),
        "label": label,
        "fields": fields,
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

def _resolve_selector(f, scope, shot_mode, shot_index, state_groups):
    """Build the shot selector, running the RMS classifier only if asked."""
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
        return select_by_state(state_by_shot), f"grouped by antenna state ({counts})"
    return select_all_shots, "mean over all shots at each position, with SEM"


def plot_xy_slider(path, scope=None, signals=None, shot_mode=None, shot_index=None,
                   t_start=None, t_end=None, n_frames=None,
                   med_size=None, gauss_sigma=None, state_groups=None,
                   output_path=None, save_npz=True):
    """Write one standalone HTML time-slider page per (scope, signal).

    Returns the list of files written. See the module docstring for the knobs;
    every argument defaults to the corresponding knob when left as None.
    """
    scope = SCOPE if scope is None else scope
    shot_mode = SHOT_MODE if shot_mode is None else shot_mode
    shot_index = SHOT_INDEX if shot_index is None else shot_index
    t_start = T_START_MS if t_start is None else t_start
    t_end = T_END_MS if t_end is None else t_end
    n_frames = N_FRAMES if n_frames is None else n_frames
    med_size = MED_SIZE if med_size is None else med_size
    gauss_sigma = GAUSS_SIGMA if gauss_sigma is None else gauss_sigma
    state_groups = STATE_GROUPS if state_groups is None else state_groups
    output_path = OUTPUT_PATH if output_path is None else output_path

    sig_list = as_signal_list(signals if signals is not None else SIGNALS)
    base = os.path.splitext(os.path.basename(path))[0]
    written = []

    with open_hdf5_readonly(path) as f:
        positions = read_positions(f)
        if not positions:
            print("no /Control/Positions data (not a bmotion file?) — nothing to map")
            return written

        xpos, ypos, _npos, _name = _plane_axes(positions)
        if not _is_plane(xpos, ypos):
            nx = 0 if xpos is None else len(xpos)
            ny = 0 if ypos is None else len(ypos)
            print(f"grid is {nx}x{ny} (a line) — plot_xy_slider only supports "
                  "planes; nothing to do")
            return written

        scopes = [scope] if scope else _scope_groups(f)

        # Resolve every (scope, signal) pair up front so an explicit .html
        # output_path can be validated before any expensive reading happens.
        jobs = []
        for sc in scopes:
            sigs = sig_list
            if sigs is None:
                chans = CHANNELS
                if chans is None:
                    chans = _channel_names(f[sc], _shot_numbers(f[sc])[0])
                elif isinstance(chans, str):
                    chans = [chans]
                sigs = [raw_signal(c) for c in chans]
            for sig in sigs:
                jobs.append((sc, sig))
        if not jobs:
            print("no scope/channel to plot")
            return written

        for sc, sig in jobs:
            out_file = resolve_output(path, output_path, base, sc, sig.key, len(jobs))
            tarr = read_hdf5_scope_tarr(f, sc)
            idxs, times_ms = frame_indices(tarr, t_start, t_end, n_frames)
            selector, how = _resolve_selector(f, sc, shot_mode, shot_index, state_groups)

            print(f"scope '{sc}' / {sig.label}: {len(idxs)} frames over "
                  f"[{times_ms[0]:.4g}, {times_ms[-1]:.4g}] ms — {how}")
            frames, xp, yp = build_frames(f, sc, sig, positions, idxs, selector,
                                          med_size, gauss_sigma)
            if not frames:
                print(f"  no usable shots for {sc}/{sig.key} — skipping")
                continue

            filt = (f"median {med_size} / gaussian {gauss_sigma} samples"
                    if (med_size > 1 or gauss_sigma > 0) else "unfiltered")
            nbytes = write_html(
                out_file, frames, xp, yp, times_ms, sig.label,
                title=f"{base} — {sc} / {sig.label}",
                subtitle=(f"{len(idxs)} frames, {times_ms[0]:.4g}–{times_ms[-1]:.4g} ms "
                          f"· {len(xp)}×{len(yp)} grid · {how}"),
                note=(f"Filtering: {filt}. Values are scope volts unless the channel "
                      "carries a calibration applied elsewhere. SEM is the repeat "
                      "standard error (NaN where a cell has one shot); grey cells have "
                      "no usable data. Frames are embedded in this file — it needs "
                      "neither the HDF5 nor Python to open."))
            written.append(out_file)
            print(f"  wrote {out_file}  ({nbytes/1e6:.2f} MB payload)")

            if save_npz:
                npz = os.path.splitext(out_file)[0] + ".npz"
                arrays = {"x": xp, "y": yp, "times_ms": np.asarray(times_ms)}
                for g, (v, e, n) in frames.items():
                    arrays[f"{g}__value"] = v
                    arrays[f"{g}__sem"] = e
                    arrays[f"{g}__n"] = n
                # Uncompressed: deflate costs ~26x the write time to save a
                # couple of MB, on data the HTML already carries anyway.
                np.savez(npz, **arrays)
                written.append(npz)
                print(f"  wrote {npz}")

    return written


def main():
    plot_xy_slider(resolve_data_file())


if __name__ == "__main__":
    main()
