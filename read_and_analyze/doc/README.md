# `read_and_analyze` — reading & validating bmotion HDF5 data

Tools for inspecting the HDF5 files produced by **`Data_Run_bmotion.py`**. After a
run you can confirm the file is well-formed (groups, datasets, dtypes, probe
positions all present and consistent) and visually check that the stored traces,
decoded back to voltage, match what the oscilloscope displayed.

All low-level reading/decoding is delegated to the **`lab_scopes`** library
(`lab_scopes.io.hdf5`). This package only adds a *validator*, a console *summary*,
and *trace plotting* on top of it.

> **Audience:** anyone who just acquired data with `Data_Run_bmotion.py` and wants
> a quick "is my file OK and does it look right?" check before analysis.

---

## 1. One-time setup

The reader needs `lab_scopes` installed **into the same Python interpreter** you run
the script with, including its `hdf5` and `plot` extras (h5py + matplotlib):

```bash
pip install -e "C:/Users/hjia9/Documents/GitHub/lab_scopes[hdf5,plot]"
```

> ⚠️ Use `python -m pip install ...` if `pip` on your PATH points at a different
> interpreter than the `python` you run the script with. Verify with:
>
> ```bash
> python -c "import lab_scopes, h5py, matplotlib; print('ok')"
> ```

`lab_scopes` is also declared as LAPD_DAQ's optional `scope` dependency
(`pyproject.toml`), so `pip install -e ".[scope]"` from the repo root works too.

The fluctuation analysis ([§8](#8-fluctuation-analysis-flattestmost-reproducible-window))
additionally needs **`scipy`** (Gaussian smoothing). It is part of the `scope`
extra; install standalone if needed:

```bash
python -m pip install scipy
```

---

## 2. Quick start (command line)

Run as a module **from the LAPD_DAQ repository root**:

```bash
# Summary + validation report + plots (uses the built-in default file)
python -m read_and_analyze.read_bmotion_data

# Point it at your own file
python -m read_and_analyze.read_bmotion_data "D:\data\LAPD\my_run_2026-05-19.hdf5"

# Report only, no figures
python -m read_and_analyze.read_bmotion_data <file.hdf5> --no-show --no-save

# Plot specific scope / channels / shots
python -m read_and_analyze.read_bmotion_data <file.hdf5> \
    --scope lpscope --channels C1 C2 --shots 1 250 510
```

A run prints three things in order:

1. **Summary** — file size, scopes, channel descriptions, shot count, motion grid.
2. **Validation report** — one `[PASS]` / `[WARN]` / `[FAIL]` line per check.
3. **Plots** — one figure per scope (saved and/or shown, see below).

The process exit code is `0` when no checks failed and `1` when any `[FAIL]` is
present, so it can be used in scripts.

### Command-line options

| Option | Default | Effect |
|---|---|---|
| `path` (positional) | built-in `DEFAULT_FILE` | HDF5 file to inspect |
| `--no-show` | off | Do not display plots (overrides `SHOW_PLOT`) |
| `--no-save` | off | Do not save plots (overrides `SAVE_PLOT`) |
| `--scope NAME` | all scopes | Restrict plotting to one scope group |
| `--channels C1 C2 …` | all channels | Channels to plot |
| `--shots 1 250 510` | first / middle / last | Shot numbers to overlay |

---

## 3. Plot output toggles

Two switches at the top of
[`read_bmotion_data.py`](../read_bmotion_data.py) control plotting globally:

```python
SHOW_PLOT = True   # display figures interactively
SAVE_PLOT = True   # write PNGs to a "plots/" subdirectory next to the data file
```

They are **independent** — turn either on or off. Flip a constant to change the
default for every run, or override per run with `--no-show` / `--no-save`.

When saving is on, figures are written next to the data file:

```
D:\data\LAPD\my_run_2026-05-19.hdf5
D:\data\LAPD\plots\my_run_2026-05-19_<scope>.png   ← one PNG per scope, 150 dpi
```

The `plots/` directory is created automatically if it does not exist.

---

## 4. Using it as a library

The same functions are importable for use in notebooks or other scripts:

```python
from read_and_analyze import (
    print_summary, validate_file, plot_traces, read_positions,
)

path = r"D:\data\LAPD\my_run_2026-05-19.hdf5"

print_summary(path)                       # console overview

ok, report = validate_file(path)          # programmatic validation
for line in report:
    print(line)
assert ok, "file failed validation"

# Save plots without opening windows (e.g. on a headless machine / batch job)
saved = plot_traces(path, show=False, save=True)
print("wrote:", saved)
```

### Public API

| Function | Returns | Purpose |
|---|---|---|
| `print_summary(path)` | `None` (prints) | File size, scopes, channel descriptions, shot count, motion grid. |
| `validate_file(path)` | `(ok: bool, report: list[str])` | Structural/format checks; `ok` is `False` if any `[FAIL]`. |
| `plot_traces(path, scope=None, channels=None, shots=None, show=None, save=None)` | `list[str]` (saved PNG paths) | Overlay traces per scope; `show`/`save` default to the module toggles. |
| `read_positions(f, mg_name=None)` | `dict` | Probe-position metadata from `/Control/Positions`. **Takes an open `h5py.File`**, not a path. |
| `find_quiet_window(path, scope=None, channels=None, window_us=None, gauss_sigma=None, signal_frac=None)` | `list[dict]` | Per (scope, channel, position): the least-fluctuating window. See [§8](#8-fluctuation-analysis-flattestmost-reproducible-window). |
| `plot_quiet_window(path, ..., show=None, save=None)` | `list[str]` (saved PNG paths) | Score-vs-position + best-window overlay; toggles like `plot_traces`. |

`read_positions` returns, per motion group, a dict with keys
`name, key, xpos, ypos, setup_array, positions_array`:

```python
import h5py
from read_and_analyze import read_positions

with h5py.File(path, "r") as f:
    pos = read_positions(f)                 # {motion_group_name: info}
    for name, info in pos.items():
        print(name, "x:", info["xpos"])     # unique grid X positions
        print("recorded:", info["positions_array"]["x"][:5])  # actual encoder X
```

For reading the actual voltage traces, use `lab_scopes` directly — the reader
builds on these:

```python
from lab_scopes.io.hdf5 import read_hdf5_scope_data, read_hdf5_scope_tarr

with h5py.File(path, "r") as f:
    volts, dt, t0 = read_hdf5_scope_data(f, "lpscope", "C1", shot_number=1)
    tarr = read_hdf5_scope_tarr(f, "lpscope")   # seconds
```

---

## 5. What the validator checks

`validate_file` walks the file and tags each check `PASS` / `WARN` / `FAIL`:

- **Root attributes** — `description`, `creation_time`, `source_code` present.
- **`/Configuration`** — `experiment_config`, `bmotion_config`, `bmotion_selection`
  datasets present; `bmotion_selection` parses as JSON.
- **`/Control/Positions/<motion_group>`** — `positions_setup_array` and
  `positions_array` present with the expected structured dtype
  `(shot_num, x, y)`; `xpos` / `ypos` grid attributes present; `positions_array`
  fully populated, `shot_num` monotonic, recorded positions within the grid.
- **Each scope** — `time_array` present; `shot_*` groups present.
- **Sampled traces** (first / middle / last shot, plus the first skipped shot) —
  each `C*_data` has a matching 346-byte `C*_header`, data dtype is `int16`,
  the WAVEDESC decodes, the time array length matches the trace length, and the
  decoded voltage is finite.

`WARN` means "unexpected but readable" (e.g. a reconstructed time base); `FAIL`
means the file is malformed in a way that breaks downstream reading.

---

## 6. Expected HDF5 layout (reference)

Files written by `Data_Run_bmotion.py` look like:

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

Voltage is recovered as `raw_int16 * vertical_gain - vertical_offset`, with the
gain/offset taken from the per-channel WAVEDESC header (handled by `lab_scopes`).

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'lab_scopes'` | Not installed in the interpreter you ran with. Re-run the setup step using `python -m pip install ...`. |
| `ModuleNotFoundError: No module named 'read_and_analyze'` | Run from the **LAPD_DAQ repo root** (so the package is importable), e.g. `python -m read_and_analyze.read_bmotion_data`. |
| `Install lab-scopes[hdf5] to use HDF5 reader helpers` | The `hdf5` extra (h5py) is missing — reinstall with `[hdf5,plot]`. |
| Plot window never appears | `SHOW_PLOT` is `False` or `--no-show` was passed; or you are on a headless machine (use `--no-show --no-save` off → just `save`). |
| A shot reports as *skipped* | The acquisition marked it (`skipped=True`); the validator counts this as a `PASS` and plotting omits it. |
| `ModuleNotFoundError: No module named 'scipy'` | The fluctuation analysis needs scipy. Run `python -m pip install scipy` in the same interpreter. |
| Fluctuation table says *no valid windows* | The signal never exceeded `SIGNAL_FRAC × peak` at any position — lower `SIGNAL_FRAC` or check the file actually has plasma signal. |

---

## 8. Fluctuation analysis — flattest/most-reproducible window

[`analysis_fluctuation.py`](../analysis_fluctuation.py) answers a different
question than the reader: **for each probe position, which short time window has
the least fluctuation?** "Least fluctuation" combines two things, both relative
to the window mean so large steady signal beats small noisy signal:

- **temporal flatness** — `(max − min) / |mean|` of the position's mean trace
  over the window (signal doesn't change much *across* the window in time);
- **shot-to-shot reproducibility** — `std-across-shots / |mean|` of the per-shot
  window means (repeat shots agree).

Their sum is the **score**; the lowest-score window wins per (scope, channel,
position). Only windows where `|mean| > SIGNAL_FRAC × peak` are considered, so
the search can't trivially pick the quiet pre-plasma region.

Each raw trace is first **smoothed in time** with `scipy.ndimage.gaussian_filter1d`
(width `GAUSS_SIGMA` samples) to strip high-frequency noise before the metrics
are computed.

### Run it

There is **no command line** — all knobs are constants at the top of the file:

```python
DEFAULT_FILE = r"D:\data\LAPD\my_run.hdf5"
SCOPE        = None    # None = all scopes; or e.g. "lpscope"
CHANNELS     = None    # None = all channels; or e.g. ["C1", "C3"]
WINDOW_US    = 10.0    # window width (microseconds); rounds up to whole samples
GAUSS_SIGMA  = 5.0     # Gaussian time-smoothing width in SAMPLES
SIGNAL_FRAC  = 0.2     # window mean must exceed this fraction of the position's peak |mean|
SHOW_PLOT    = True
SAVE_PLOT    = True
```

Edit those, then run from the LAPD_DAQ repo root:

```bash
python -m read_and_analyze.analysis_fluctuation
```

It prints a per-position table sorted best-first (position, window-center time in
ms, `flat_rel`, `scat_rel`, `score`, window mean voltage) and writes
`plots/<base>_<scope>_fluctuation.png` (score-vs-position on top, the best
position's repeat shots with the chosen window shaded on the bottom).

> A 10 µs window on the test file's 800 ns timebase is 13 samples. At that
> sampling the signal is nearly flat over the window, so `scat_rel`
> (reproducibility) usually dominates the score — i.e. the ranking is effectively
> "which position is most shot-to-shot reproducible."

### As a library

```python
from read_and_analyze import find_quiet_window, plot_quiet_window

recs = find_quiet_window(path, scope="lpscope", channels=["C3"])
best = recs[0]   # sorted best (lowest score) first
print(best["x"], best["t_center"], best["score"])

plot_quiet_window(path, show=False, save=True)   # headless
```

---

*Generated as documentation for the `read_and_analyze` package on the
`feature/read-analyze-data` branch. Keep in sync with
[`read_bmotion_data.py`](../read_bmotion_data.py) and
[`analysis_fluctuation.py`](../analysis_fluctuation.py).*
