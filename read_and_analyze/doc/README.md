# `read_and_analyze` — inspect & analyze bmotion HDF5 data

Tools for inspecting and analyzing the HDF5 files produced by
**`Data_Run_bmotion.py`** (a probe stepped over an XY grid, repeat shots per
position). All low-level reading/decoding is delegated to the in-repo
**`scope_io`** package (`scope_io.hdf5`); this package adds validation,
filtering, and several analysis/plot views on top. No `lab_scopes` install is
required.

---

## Read data in Python

**The three things you usually want, and the helper that returns each:**

| What | `scope_io` helper (open `f` with `h5py.File(path, "r")`) | Returns |
|---|---|---|
| Channel voltage for one shot | `read_hdf5_scope_data(f, scope, "C1", shot)` | `(volts, dt, t0)` |
| Many shots of one channel | `read_hdf5_scope_channel_shots(f, scope, "C1", shots)` | 2D volts (header decoded once) |
| Time base for a scope | `read_hdf5_scope_tarr(f, scope)` | `(samples,)` seconds |
| Probe `(x, y)` per shot | `f["Control/Positions/<motion_group>/positions_array"]` | structured `(shot_num, x, y)` |

The helpers decode the per-trace WAVEDESC and apply
`volts = raw_int16 × vertical_gain − vertical_offset` for you, so you never touch
the header directly:

```python
import h5py
from scope_io import (
    read_hdf5_scope_data, read_hdf5_scope_tarr, read_hdf5_scope_channel_shots,
)

with h5py.File(r"D:\data\LAPD\my_run.hdf5", "r") as f:
    tarr = read_hdf5_scope_tarr(f, "bdotscope")          # seconds
    volts, dt, t0 = read_hdf5_scope_data(f, "bdotscope", "C1", shot_number=1)
    pos = f["Control/Positions/probe1/positions_array"][:]   # (shot_num, x, y)
```

`read_hdf5_scope_data` raises on a *skipped* shot — catch `ValueError`, or use
`read_hdf5_scope_channel_shots`, which fills missing/skipped shots with `NaN`
rows. Source: [`scope_io/hdf5.py`](../../scope_io/hdf5.py); the raw 346-byte
WAVEDESC parser is [`scope_io/wavedesc.py`](../../scope_io/wavedesc.py)
(`LeCroyWavedesc`). For the full on-disk layout, see the
[HDF5 Output section of the main README](../../README.md#hdf5-output).

---

## Setup (once)

These tools need only standard scientific packages (no `lab_scopes`):

```bash
python -m pip install numpy h5py scipy matplotlib
```

`h5py` reads the data, `numpy` decodes/scales it, and `scipy` + `matplotlib`
drive the filtering and plot views. Verify with
`python -c "import h5py, numpy, matplotlib, scipy; print('ok')"`.

Run every module **from the LAPD_DAQ repo root** as `python -m read_and_analyze.<module>`.

---

## The modules

| Module | What it does | Run |
|---|---|---|
| [`read_bmotion_data.py`](../read_bmotion_data.py) | Validates the file (groups, dtypes, positions, sampled traces) and overlays raw voltage traces per scope. Prints a summary + `PASS`/`WARN`/`FAIL` report (exit code `1` on any `FAIL`). | `python -m read_and_analyze.read_bmotion_data [file.hdf5]` |
| [`filter_data.py`](../filter_data.py) | Shows the denoising pipeline — raw vs median vs median+Gaussian — on a sample trace, so you can tune the filter. Owns the shared filtering helpers other modules reuse. | `python -m read_and_analyze.filter_data` |
| [`fluctuation_analysis.py`](../fluctuation_analysis.py) | Finds, per position, the **flattest / most reproducible** time window (see [below](#fluctuation-analysis)). Prints a best-first table + a score/overlay figure. | `python -m read_and_analyze.fluctuation_analysis` |
| [`plot_xy_map.py`](../plot_xy_map.py) | Reduces each grid position's trace to one scalar (mean over a time range, or value at one instant) and renders a **2D XY map** (`imshow`, optional contours). Genuine 2D planes only; line scans are skipped (use `plot_x_line`). | `python -m read_and_analyze.plot_xy_map` |
| [`plot_x_line.py`](../plot_x_line.py) | The 1D **line-scan** counterpart to `plot_xy_map`: reduces each position to a scalar and plots value vs probe position. Auto-detects the moving axis (x or y). Genuine 2D planes are skipped. | `python -m read_and_analyze.plot_x_line` |
| [`smart_trigger_analysis.py`](../smart_trigger_analysis.py) | Replays a LeCroy scope's **SmartTriggers** post-hoc (see [below](#smarttrigger-scan)) and reports which events would have triggered. Prints a per-shot table + a per-shot scan figure. | `python -m read_and_analyze.smart_trigger_analysis` |
| [`fix_channel_descriptions.py`](../fix_channel_descriptions.py) | Maintenance CLI: retrofits per-channel `<CH>_description` attributes onto an **existing** run HDF5 by re-parsing the stored `experiment_config`. Idempotent (skips groups already labeled unless `--force`); accepts a file or folder. | `python -m read_and_analyze.fix_channel_descriptions <file_or_folder> [--force] [--recursive]` |

Two modules have a CLI: `read_bmotion_data` and `fix_channel_descriptions`.
`read_bmotion_data` options: `--no-show` / `--no-save` (override plot toggles),
`--scope NAME`, `--channels C1 C2`, `--shots 1 250 510`. The remaining analysis
modules (`filter_data`, `fluctuation_analysis`, `plot_xy_map`, `plot_x_line`,
`smart_trigger_analysis`) are configured by editing constants (below) — there is
no command line.

> A separate internal hook, [`auto_plot.py`](../auto_plot.py), is not run by hand:
> the acquisition/offload scripts call it after a run to render the 1D line
> profile (via `plot_x_line`), gated by the `[analysis] auto_plot` config key. It
> forces `show=False`/save-only and can never crash the run.

---

## Where the constants live

Every user-changeable knob is a constant in one of **two files**. Edit the file,
re-run the module.

### `analysis_config.py` — shared across all modules

[`analysis_config.py`](../analysis_config.py) holds the knobs every module reads,
plus the module-specific analysis knobs. Editing one value here changes it
everywhere it applies.

```python
# SHARED — used by every module
DATA_DIR     = r"E:\Shadow data\..."        # Data_Run_bmotion.py's output folder (its base_path)
DATA_FILE    = None                         # None = auto-pick the newest COMPLETED run in
                                            # DATA_DIR (in-progress runs are skipped);
                                            # or pin one file: r"D:\data\LAPD\my_run.hdf5"
SELECT_SCOPE = None        # scope to analyze; None = all scopes
SELECT_CHAN  = None        # channels to analyze; None = all channels
SHOW_PLOT    = True        # display figures interactively
SAVE_PLOT    = False       # write PNGs to a "plots/" subdir next to the data file
AUTO_PLOT    = True        # fallback default for the auto_plot.py post-run hook when
                           # called without a config; the run's [analysis] auto_plot
                           # key (experiment_config.ini) overrides this in acquisition
MED_SIZE     = 5           # median-filter width in SAMPLES (spike removal); 1 = off
GAUSS_SIGMA  = 20          # Gaussian smoothing width in SAMPLES; 0 = off
POS_TOL      = 0.5         # group repeat shots within this many mm

# FLUCTUATION — fluctuation_analysis.py only
FLUCT_WINDOW_US   = 10.0   # window width (us) slid across the record
FLUCT_SIGNAL_FRAC = 0      # window |mean| must exceed this fraction of the position's peak

# XY_MAP — plot_xy_map.py and plot_x_line.py
XY_MODE         = "range"  # "range" = mean over [T_START_MS, T_END_MS]; "step" = snapshot(s) at XY_T_STEP_MS
XY_T_START_MS   = 0        # range start (ms), used when XY_MODE == "range"
XY_T_END_MS     = 2.0      # range end (ms), used when XY_MODE == "range"
XY_T_STEP_MS    = [10,12,15,19]  # snapshot time(s) in ms for "step" mode; one panel per time
                                 # (a single float, e.g. 4.0, is also accepted -> one panel)
XY_SHOT_INDEX   = 0        # which shot (0-based) per position to map; no shot averaging yet
XY_SHOW_CONTOUR = False    # overlay contour lines
XY_N_CONTOURS   = 8        # contour count when XY_SHOW_CONTOUR is True
XY_CMAP         = "rainbow"
```

> The SmartTrigger scan keeps its **own** plot toggles in `smart_trigger_config.py`
> (`SHOW_PLOT`/`SAVE_PLOT`); the shared toggles above drive the other analysis
> modules (`read_bmotion_data`, `filter_data`, `fluctuation_analysis`,
> `plot_xy_map`, `plot_x_line`). `read_bmotion_data` can also override its toggles
> per-run with `--no-show`/`--no-save`.

### `smart_trigger_config.py` — SmartTrigger scan only

[`smart_trigger_config.py`](../smart_trigger_config.py) imports
`DATA_FILE`/`SELECT_SCOPE`/`SELECT_CHAN`/`MED_SIZE`/`GAUSS_SIGMA` from
`analysis_config.py`, then adds the scan-specific knobs, grouped per trigger mode:

```python
SHOW_PLOT  = False; SAVE_PLOT = True   # this module's own plot toggles
SHOTS      = None    # None = sample shots (first/mid/last per position); or e.g. [12, 57]
HOLDOFF_US = 3000    # ignore the record before this time (us); mimics trigger holdoff
MATH       = None    # None, or "derivative" / "integral" / "abs" (preprocess before detection)

# one block per trigger mode. Levels are ABSOLUTE VOLTS; width/slew/interval
# limits are NANOSECONDS. A value OUTSIDE [min, max] is flagged; a None bound
# disables that side:
GLITCH_LEVEL = 0.5; GLITCH_HYST = 0.05; GLITCH_MIN_WIDTH_NS = None; GLITCH_MAX_WIDTH_NS = 100.0
RUNT_LO = 0.3;      RUNT_HI = 0.7
SLEW_LO = 0.1;      SLEW_HI = 0.9;      SLEW_MIN_NS = None; SLEW_MAX_NS = 50.0
INTERVAL_LEVEL = 0.5; INTERVAL_HYST = 0.05; INTERVAL_MIN_NS = None; INTERVAL_MAX_NS = None
```

---

## Plot output

`SHOW_PLOT` / `SAVE_PLOT` are independent. When saving is on, PNGs go in a
`plots/` subdir next to the data file (created automatically), one per scope (or
per scope/channel for the XY map), at 150 dpi — e.g.
`D:\data\LAPD\plots\my_run_<scope>.png`.

---

## Fluctuation analysis

For each position, scores every candidate window by two terms, both relative to
the window mean (so large steady signal beats small noisy signal):

- **temporal flatness** — `(max − min) / |mean|` of the position's mean trace;
- **shot-to-shot reproducibility** — `std-across-shots / |mean|` of the per-shot
  window means.

Their sum is the **score**; the lowest wins per (scope, channel, position). Only
windows with `|mean| > FLUCT_SIGNAL_FRAC × peak` qualify, so the quiet pre-plasma
region can't trivially win. Traces are denoised (median `MED_SIZE` → Gaussian
`GAUSS_SIGMA`) first. Output: a best-first table (position, window-center time,
`flat_rel`, `scat_rel`, `score`, mean V) and
`plots/<base>_<scope>_fluctuation.png`.

> On a short window (e.g. 10 µs ≈ 13 samples at an 800 ns base) the signal is
> nearly flat, so reproducibility usually dominates the ranking.

---

## SmartTrigger scan

Replays a LeCroy scope's **SmartTriggers** over recorded traces, reporting the
events each *would* have caught. Crossing levels are given in **absolute volts**
and width/slew/interval limits in **nanoseconds** (matching the scope's front
panel); a measured value is flagged when it falls **outside** the `[min, max]`
band for that detector (a `None` bound disables that side). The four detectors
are pure functions of `(volts, tarr)`:

- **Glitch/Width** — flags pulses whose width is outside `[GLITCH_MIN_WIDTH_NS, GLITCH_MAX_WIDTH_NS]`.
- **Runt** — flags excursions that cross `RUNT_LO` (V) but never reach `RUNT_HI` (V).
- **Slew rate** — flags edges whose `SLEW_LO`↔`SLEW_HI` transition time is outside `[SLEW_MIN_NS, SLEW_MAX_NS]`.
- **Interval** — flags periods between rising edges outside `[INTERVAL_MIN_NS, INTERVAL_MAX_NS]`.

Two scope-like preprocessing knobs apply first: **`MATH`** (run derivative /
integral / abs, like triggering off a Math trace) and **`HOLDOFF_US`** (ignore
the record before a given time). Traces are denoised before detection. Output: a
table per (scope, channel, shot, kind) with the nominal and flagged-event count,
and `plots/<base>_<scope>_smart_triggers.png` (one panel per shot showing the
scanned signal, derived levels, holdoff band, and a shaded span per event colored
by kind).

---

## What the validator checks

`read_bmotion_data` tags each check `PASS` / `WARN` / `FAIL`:

- **Root attrs** — `description`, `creation_time`, `source_code`.
- **`/Configuration`** — `experiment_config`, `bmotion_config`, `bmotion_selection`
  present; `bmotion_selection` parses as JSON.
- **`/Control/Positions/<motion_group>`** — setup/positions arrays with dtype
  `(shot_num, x, y)`; `xpos`/`ypos` grid attrs; array fully populated, `shot_num`
  monotonic, positions within the grid.
- **Each scope** — `time_array` and `shot_*` groups present.
- **Sampled traces** (first/middle/last + first skipped) — each `C*_data` has a
  346-byte `C*_header`, dtype `int16`, the WAVEDESC decodes, lengths match, and
  decoded voltage is finite.

`WARN` = unexpected but readable; `FAIL` = malformed in a way that breaks reading.

---

## Expected HDF5 layout (reference)

The [main README's HDF5 Output section](../../README.md#hdf5-output) is the
canonical layout reference. Quick recap of what the validator and analysis
modules rely on:

<details>
<summary>File layout</summary>

```
/                                 attrs: description, creation_time, source_code
├── Configuration/
│   ├── experiment_config         (bytes)
│   ├── bmotion_config            (bytes, TOML)
│   └── bmotion_selection         (bytes, JSON)
├── Control/
│   └── Positions/
│       └── <motion_group_name>/  attrs: name, key
│           ├── positions_setup_array   structured (shot_num,x,y); attrs xpos, ypos
│           └── positions_array         structured (shot_num,x,y); actual encoder values
└── <scope_name>/                 attrs: description, ip_address, scope_type
    ├── time_array                float64 seconds, shape (N,)
    └── shot_<n>/                 attrs: acquisition_time [, skipped, skip_reason]
        ├── C<k>_data             int16, shape (N,) or (N_seg, N) in sequence mode
        └── C<k>_header           void, 346-byte LeCroy WAVEDESC
```

Voltage = `raw_int16 × vertical_gain − vertical_offset`, gain/offset from the
per-channel WAVEDESC header (decoded by `scope_io`).

</details>

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No module named 'h5py'` / `'scipy'` / `'matplotlib'` | Not installed in the interpreter you ran with. Re-run setup with `python -m pip install numpy h5py scipy matplotlib`. |
| `No module named 'read_and_analyze'` / `'scope_io'` | Run from the **LAPD_DAQ repo root** as `python -m read_and_analyze.<module>`. |
| `Install h5py to use the scope_io HDF5 readers.` | `h5py` is missing — `python -m pip install h5py`. |
| Plot window never appears | `SHOW_PLOT=False`, `--no-show`, or a headless machine — use `SAVE_PLOT=True` instead. |
| A shot reports as *skipped* | The acquisition marked it; the validator counts it `PASS` and plotting omits it. |
| Fluctuation table says *no valid windows* | Signal never exceeded `FLUCT_SIGNAL_FRAC × peak` — lower it, or check the file has plasma signal. |

---

*Documentation for the `read_and_analyze` package. Keep in sync with
[`analysis_config.py`](../analysis_config.py) and
[`smart_trigger_config.py`](../smart_trigger_config.py) (all user knobs),
[`read_bmotion_data.py`](../read_bmotion_data.py),
[`filter_data.py`](../filter_data.py),
[`fluctuation_analysis.py`](../fluctuation_analysis.py),
[`plot_xy_map.py`](../plot_xy_map.py),
[`plot_x_line.py`](../plot_x_line.py),
[`smart_trigger_analysis.py`](../smart_trigger_analysis.py),
[`fix_channel_descriptions.py`](../fix_channel_descriptions.py), and
[`auto_plot.py`](../auto_plot.py). The
[`test_read_analyze_doc_sync`](../../tests/test_read_analyze_doc_sync.py) test
enforces that every module is listed here and that the config constants shown
above still exist.*
