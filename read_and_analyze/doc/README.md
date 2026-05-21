# `read_and_analyze` — inspect & analyze bmotion HDF5 data

Tools for inspecting and analyzing the HDF5 files produced by
**`Data_Run_bmotion.py`** (a probe stepped over an XY grid, repeat shots per
position). All low-level reading/decoding is delegated to the **`lab_scopes`**
library (`lab_scopes.io.hdf5`); this package adds validation, filtering, and
several analysis/plot views on top.

---

## Setup (once)

Install `lab_scopes` into the **same interpreter** you run with, plus `scipy`
(used by the filtering/analysis modules):

```bash
python -m pip install -e "C:/Users/hjia9/Documents/GitHub/lab_scopes[hdf5,plot]"
python -m pip install scipy
```

`lab_scopes` is also LAPD_DAQ's optional `scope` dependency, so
`pip install -e ".[scope]"` from the repo root works too. Verify with
`python -c "import lab_scopes, h5py, matplotlib, scipy; print('ok')"`.

Run every module **from the LAPD_DAQ repo root** as `python -m read_and_analyze.<module>`.

---

## The modules

| Module | What it does | Run |
|---|---|---|
| [`read_bmotion_data.py`](../read_bmotion_data.py) | Validates the file (groups, dtypes, positions, sampled traces) and overlays raw voltage traces per scope. Prints a summary + `PASS`/`WARN`/`FAIL` report (exit code `1` on any `FAIL`). | `python -m read_and_analyze.read_bmotion_data [file.hdf5]` |
| [`filter_data.py`](../filter_data.py) | Shows the denoising pipeline — raw vs median vs median+Gaussian — on a sample trace, so you can tune the filter. Owns the shared filtering helpers other modules reuse. | `python -m read_and_analyze.filter_data` |
| [`fluctuation_analysis.py`](../fluctuation_analysis.py) | Finds, per position, the **flattest / most reproducible** time window (see [below](#fluctuation-analysis)). Prints a best-first table + a score/overlay figure. | `python -m read_and_analyze.fluctuation_analysis` |
| [`plot_xy_map.py`](../plot_xy_map.py) | Reduces each grid position's trace to one scalar (mean over a time range, or value at one instant) and renders a **2D XY map** (`imshow`, optional contours). Falls back to a line plot for 1D scans. | `python -m read_and_analyze.plot_xy_map` |
| [`smart_trigger_analysis.py`](../smart_trigger_analysis.py) | Replays a LeCroy scope's **SmartTriggers** post-hoc (see [below](#smarttrigger-scan)) and reports which events would have triggered. Prints a per-shot table + a per-shot scan figure. | `python -m read_and_analyze.smart_trigger_analysis` |

`read_bmotion_data` is the only one with a CLI. Its options:
`--no-show` / `--no-save` (override plot toggles), `--scope NAME`,
`--channels C1 C2`, `--shots 1 250 510`. All other modules are configured by
editing constants (below) — there is no command line.

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
DATA_FILE    = r"D:\data\LAPD\my_run.hdf5"  # the HDF5 file to analyze
SELECT_SCOPE = "lpscope"   # scope to analyze; None = all scopes
SELECT_CHAN  = ["C1"]      # channels to analyze; None = all channels
SHOW_PLOT    = True        # display figures interactively
SAVE_PLOT    = False       # write PNGs to a "plots/" subdir next to the data file
MED_SIZE     = 5           # median-filter width in SAMPLES (spike removal); 1 = off
GAUSS_SIGMA  = 20          # Gaussian smoothing width in SAMPLES; 0 = off
POS_TOL      = 0.5         # group repeat shots within this many mm

# FLUCTUATION — fluctuation_analysis.py only
FLUCT_WINDOW_US   = 10.0   # window width (us) slid across the record
FLUCT_SIGNAL_FRAC = 0      # window |mean| must exceed this fraction of the position's peak

# XY_MAP — plot_xy_map.py only
XY_MODE         = "range"  # "range" = mean over [T_START_US, T_END_US]; "step" = value at T_STEP_US
XY_T_START_US   = 4000.0   # range start (us)
XY_T_END_US     = 4500.0   # range end (us)
XY_T_STEP_US    = 100.0    # snapshot time (us), used when XY_MODE == "step"
XY_SHOW_CONTOUR = False    # overlay contour lines
XY_N_CONTOURS   = 8        # contour count when XY_SHOW_CONTOUR is True
XY_CMAP         = "viridis"
```

> The SmartTrigger scan keeps its **own** plot toggles in `smart_trigger_config.py`
> (`SHOW_PLOT`/`SAVE_PLOT`); the shared toggles above drive the other four modules.
> `read_bmotion_data` can also override its toggles per-run with `--no-show`/`--no-save`.

### `smart_trigger_config.py` — SmartTrigger scan only

[`smart_trigger_config.py`](../smart_trigger_config.py) imports
`DATA_FILE`/`SELECT_SCOPE`/`SELECT_CHAN`/`MED_SIZE`/`GAUSS_SIGMA` from
`analysis_config.py`, then adds the scan-specific knobs, grouped per trigger mode:

```python
SHOW_PLOT  = False; SAVE_PLOT = True   # this module's own plot toggles
SHOTS      = None    # None = sample shots (first/mid/last per position); or e.g. [12, 57]
HOLDOFF_US = 3000    # ignore the record before this time (us); mimics trigger holdoff
MATH       = None    # None, or "derivative" / "integral" / "abs" (preprocess before detection)

# one block per trigger mode (levels are fractions of each trace's min..max span;
# EXCL_DELTA = the +/- band beyond which a measured value is flagged):
GLITCH_THRESH_FRAC = 0.5;  GLITCH_HYST_FRAC = 0.05; GLITCH_EXCL_DELTA = 0.25
RUNT_LO_FRAC = 0.3;        RUNT_HI_FRAC = 0.7
SLEW_LO_FRAC = 0.1;        SLEW_HI_FRAC = 0.9;      SLEW_EXCL_DELTA = 0.25
INTERVAL_THRESH_FRAC = 0.5; INTERVAL_HYST_FRAC = 0.05; INTERVAL_EXCL_DELTA = 0.25
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
events each *would* have caught. Crossing levels are derived **per trace** from
its own min..max span (the software analog of *Find Level*), so the detectors
work at any absolute scale; nominal widths/periods use the **median** of the
measured population (robust against the outliers being hunted). The four
detectors are pure functions of `(volts, tarr)`:

- **Glitch/Width** — flags pulses narrower than `nominal × (1 − GLITCH_EXCL_DELTA)`.
- **Runt** — flags excursions that cross `RUNT_LO_FRAC` but never reach `RUNT_HI_FRAC`.
- **Slew rate** — flags edges whose lo↔hi transition time is outside `nominal × (1 ± SLEW_EXCL_DELTA)`.
- **Interval** — flags periods between rising edges outside `nominal × (1 ± INTERVAL_EXCL_DELTA)`.

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
per-channel WAVEDESC header (handled by `lab_scopes`).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No module named 'lab_scopes'` / `'scipy'` | Not installed in the interpreter you ran with. Re-run setup with `python -m pip install ...`. |
| `No module named 'read_and_analyze'` | Run from the **LAPD_DAQ repo root** as `python -m read_and_analyze.<module>`. |
| `Install lab-scopes[hdf5] ...` | The `hdf5` extra (h5py) is missing — reinstall with `[hdf5,plot]`. |
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
[`plot_xy_map.py`](../plot_xy_map.py), and
[`smart_trigger_analysis.py`](../smart_trigger_analysis.py).*
