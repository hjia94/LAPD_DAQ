# LAPD_DAQ

Python data-acquisition tools for LAPD experiments using LeCroy oscilloscopes,
Ethernet motor controllers, Phantom cameras, Raspberry Pi triggers, and
`bapsf_motion` workflows.

## Current Status

`Data_Run_*.py` is the production path for routine runs. It uses the **spooled
pipeline**: the acquire process spools each shot to a fast disk and a separate
`Offload_Run.py` drains the spool into the HDF5 file. This is the path the
hardware PC runs and the main integration target.

```text
  Data_Run_*.py  ──spool each shot──▶  fast disk  ──drain──▶  run.hdf5
   (acquire)                                       (Offload_Run.py)
```

| Area | Status |
|---|---|
| `Data_Run_bmotion.py` | Uses the `bapsf_motion` library for motor communication and control |
| 45-degree probe | Not migrated; `Data_Run_45deg.py` exits early with an unsupported message |
| Testing | Automated tests run on mock devices; real-hardware tests also exist |
| Control backend | EPICS is the planned future backend (see [migration path](#epics-migration-path)) |

## Install

Use Python 3.10 or newer.

Command Prompt:

```cmd
git clone https://github.com/hjia94/LAPD_DAQ.git
cd LAPD_DAQ
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -e .
```

PowerShell:

```powershell
git clone https://github.com/hjia94/LAPD_DAQ.git
cd LAPD_DAQ
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e .
```

Optional extras (install only what you need):

| Extra | Adds | Command |
|---|---|---|
| `scope` | LeCroy scope control (`lab_scopes` + PyVISA) | `pip install -e ".[scope]"` |
| `bmotion` | `bapsf_motion` workflows (`bapsf-motion` + xarray) | `pip install -e ".[bmotion]"` |
| `dev` | Jupyter for the `notebooks/` examples | `pip install -e ".[dev]"` |
| both | scope + bmotion at once | `pip install -e ".[scope,bmotion]"` |

The `camera` extra is intentionally empty because the Phantom camera SDK and
Python bindings are hardware-PC specific. Install those according to the camera
PC setup notes before using camera mode.


## Configuration Reference

Below are sections to be included in experiment_config.ini

### 🔴 Required (validated at startup and the run aborts if missing)

| Section | Key | Purpose |
|---|---|---|
| `[storage]` | `spool_dir` | Fast local disk for the per-shot spool — the spooled pipeline cannot run without it |
| `[experiment]` | `name` | Used to build the HDF5 filename; the run aborts if absent |

> A scope run won't capture anything useful without scope sections
> (`[scopes]` / `[scope_ips]` and usually `[channels]`), but these aren't
> validated at startup — they're "optional" only in the sense that the run will
> launch without them.

### 🟢 Optional (mode-specific or defaulted)

| Section | Purpose / key keys |
|---|---|
| `[storage]` | `hdf5_dir`, plus `disk_full_pause_seconds` / `disk_full_max_retries` to tune the pause+retry when the spool disk fills |
| `[acquisition]` | Per-shot tuning for the spooled path |
| `[nshots]` | `num_duplicate_shots`, `num_run_repeats` |
| `[experiment]` | Run description lives in a separate `description.txt` next to the config (written to the HDF5 `description` attr at run start, overwritten at run end) |
| `[scopes]` | Scope display names and descriptions |
| `[channels]` | Channel descriptions, keyed `ScopeName_C1` |
| `[scope_ips]` | Direct scope IPs |
| `[scope_modes]` | Per-scope acquisition mode: `single` (default) or `sequence` — see [Acquisition modes](#acquisition-modes) |
| `[analysis]` | `auto_plot` — post-run line-profile PNG plotting (default on) |
| `[position]` / `[motor_ips]` | XY/XYZ grid parameters and motor IPs (grid mode) |
| `[camera_config]` | Phantom camera settings (camera/dropper modes only) |
| `[raspberry_pi]` | Raspberry Pi trigger settings (dropper mode) |
| `[bmotion]` | Motion-group / direction / recovery settings for the bmotion script |

See example_description.txt for full description of configurations.

### Acquisition modes

Each scope runs in one acquisition mode for the whole run, declared in the
optional `[scope_modes]` section and keyed by scope name. Any scope not listed
defaults to `single`, so existing configs are unchanged.

```ini
[scope_ips]
BdotScope = 192.168.7.63
XrayScope = 192.168.7.64

[scope_modes]
BdotScope = sequence
# XrayScope omitted -> single (the default)
```

| Mode | What it captures | Stored shape (`/<Scope>/shot_N/C1_data`) |
|---|---|---|
| `single` (default) | One trace per shot — synchronized single capture (master/slave) | `(samples,)` int16 |
| `sequence` | Segmented (sequence) capture — one acquisition fills all segments | `(n_segments, samples)` int16, one row per segment |

A `sequence` scope is armed, triggered, and waited on **exactly like a `single`
scope** (SINGLE trigger, master/slave arming, STOP-on-complete). The only
difference is the returned data structure and how it is stored: sequence data is
written as a 2-D dataset (one row per segment) instead of a 1-D trace. Because
the two modes are otherwise identical, a sequence scope can coexist with single
scopes in the same run.

The scope must be placed into sequence (segmented) acquisition on its front
panel; the config setting only tells the DAQ how to read and store the result.
If the declared mode and the scope's actual acquisition setup disagree at
startup, a warning is printed (the config value is still used).

## HDF5 Output

`Data_Run_bmotion.py`  writes one HDF5 file per run.

**If you want:**

| What | HDF5 path | Shape / dtype |
| --- | --- | --- |
| Channel waveform (shot `N`, trace `C1`) | `/<ScopeName>/shot_<N>/C1_data` | `(samples,)` raw `int16` (single mode); `(n_segments, samples)` in sequence mode |
| Per-trace WAVEDESC header (gain/offset) | `/<ScopeName>/shot_<N>/C1_header` | 346-byte opaque (`np.void`) |
| Time base for all that scope's traces | `/<ScopeName>/time_array` | `(samples,)` `float64`, seconds |
| Probe position per shot | `/Control/Positions/<motion_group>/positions_array` | structured `(shot_num, x, y)` |

A trace's voltage is `vertical_gain * C1_data - vertical_offset`, where gain and
offset come from the `C1_header` (LeCroy WAVEDESC) sibling dataset; sample `i`
occurs at `time_array[i]`. The `positions_array` row whose `shot_num == N` gives
the `(x, y)` the probe was at for `/<ScopeName>/shot_N/`.

### Reading voltage data

Reader scripts that decode the WAVEDESC, apply the gain/offset, and returning volts directly:

- In-repo: [scope_io/hdf5.py](scope_io/hdf5.py) — `read_hdf5_scope_data(f,
  scope, channel, shot)` → `(volts, dt, t0)`, `read_hdf5_scope_tarr(f, scope)`
  → time array, and `read_hdf5_scope_channel_shots(...)` to read many shots
  while decoding the header once. These are re-exported from `scope_io` and
  used by the `read_and_analyze/` plotting tools (e.g.
  [read_and_analyze/read_bmotion_data.py](read_and_analyze/read_bmotion_data.py)).
- `lab_scopes`: `lab_scopes.io.lecroy_files` (and the `read_scope_data` legacy
  shim) for `.trc`/HDF5 traces.

Datasets are Blosc2 (bitshuffle+lz4, falling back to lzf) compressed, so HDFView
needs the matching plugin to view them; the Python readers above do not (they
register the filter automatically).

### Parsing the WAVEDESC header

Each `C*_header` is the raw 346-byte LeCroy WAVEDESC struct, stored opaquely
(`np.void`). Decode it with the in-repo parser rather than reading bytes by hand:

```python
from scope_io import WAVEDESC_SIZE          # == 346
from scope_io.wavedesc import LeCroyWavedesc

with h5py.File(path, "r") as f:
    raw  = f["bdotscope/shot_1/C1_data"][:]            # int16
    hdr  = f["bdotscope/shot_1/C1_header"][()]          # 346 bytes
    wd   = LeCroyWavedesc(hdr)
    volts = raw.astype("float64") * wd.wd.vertical_gain - wd.wd.vertical_offset
    # wd.dt, wd.t0, wd.num_samples, wd.time_array are also available
```

- Parser: [scope_io/wavedesc.py](scope_io/wavedesc.py) — `LeCroyWavedesc` unpacks
  all 63 fields; `vertical_gain`, `vertical_offset`, `dt`, `t0` are the ones you
  usually need.
- In `lab_scopes`, the equivalent is `lab_scopes.lecroy.wavedesc.LeCroyWavedesc`.

<details>
<summary>Full file layout</summary>

```text
run.hdf5
├─ attrs: description, creation_time, source_code
├─ Configuration/
│    ├─ experiment_config    # verbatim experiment_config.ini  (bytes)
│    ├─ bmotion_config       # verbatim bmotion_config.toml    (bytes)
│    └─ bmotion_selection    # JSON: motion-group keys, direction, order
├─ Control/
│    └─ Positions/
│         └─ <motion_group>/                 # one per selected motion group
│              ├─ attrs: name, key
│              ├─ positions_setup_array       # planned grid (shot_num, x, y)
│              │     └─ attrs: xpos, ypos      #   unique X / Y axis vectors
│              └─ positions_array             # ACTUAL recorded (shot_num, x, y)
└─ <ScopeName>/                              # one per scope (e.g. bdotscope)
     ├─ attrs: description, ip_address, scope_type, shot_count,
     │         <C1>_description, <C2>_description, …   # channel labels
     ├─ time_array                           # float64 seconds
     ├─ shot_1/
     │    ├─ attrs: acquisition_time
     │    ├─ C1_data     # raw int16 samples (compressed)
     │    ├─ C1_header   # LeCroy WAVEDESC binary header (gain/offset live here)
     │    ├─ C2_data
     │    └─ C2_header
     └─ shot_2/  …
```

- `positions_setup_array` / `positions_array` dtype:
  `[('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4')]`.
- Skipped/failed shots are a `shot_N` group with `skipped=True`
  (and `failed=True` if quarantined) + a `skip_reason` attr, and no datasets.

</details>



## Repository Layout

**Packages**

| Path | Contents |
|---|---|
| `lapd_daq/` | New framework: CLI, config model, run engine, devices, HDF5 writer |
| `acquisition/` | Acquisition package used by `Data_Run*.py` (spool, offload, scope/bmotion loops) |
| `spooling/` | Per-shot spool format + disk-full pause/retry helper |
| `drivers/` | LeCroy and Phantom hardware driver wrappers |
| `motion/` | Motor control and position management helpers |
| `pi_gpio/` | Raspberry Pi trigger/dropper client package |
| `legacy/` | Superseded scripts kept for reference |
| `notebooks/` | Scratch notebooks for scope and motor testing |
| `tests/` | Automated tests (mock by default, gated hardware checks) — see [docs/tests.md](docs/tests.md) |
| `docs/` | Long-form documentation pages |

**Entry-point scripts**

| Script | Role |
|---|---|
| `Data_Run.py` | Standard spooled acquisition (stationary / grid) |
| `Data_Run_bmotion.py` | Spooled bmotion acquisition |
| `Data_Run_MultiScope_Camera.py` | Multi-scope + camera acquisition |
| `Data_Run_45deg.py` | Unsupported (exits early) |
| `Offload_Run.py` | Drains the spool into the HDF5 file (run alongside `Data_Run*.py`) |

Plus `example_experiment_config.ini` and `pyproject.toml` at the root.

## Tests

See [docs/tests.md](docs/tests.md) for the full test suite documentation:
what each module covers, the recommended run sequence on a development
machine vs. the hardware PC, and the gating flags for hardware-touching
tests.

## Under development

- **`lapd-daq` CLI** — the newer framework (`lapd_daq/` package): typed config,
  centralized HDF5 writing, mock-device dry runs, and adapter boundaries that can
  later be backed by EPICS PVs. Covers stationary, grid, camera, and dropper modes.


### EPICS Migration Path

1. Clean Python framework using direct hardware adapters.
2. Add EPICS PV naming fields while direct adapters still run hardware.
3. Move devices behind EPICS IOCs and replace direct adapters with EPICS PV
   adapters.
4. Let EPICS own device state, interlocks, autosave, and control logic.
5. Keep `LAPD_DAQ` focused on experiment orchestration, shot sequencing, and
   HDF5 data products.

  Future EPICS-related PV fields may be added to the INI config during migration,
but EPICS-native `.db`, `.dbd`, `.template`, `.substitutions`, and `st.cmd`
files should eventually own hardware control configuration.
