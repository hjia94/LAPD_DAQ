# LAPD_DAQ

Python data-acquisition tools for LAPD experiments using LeCroy oscilloscopes,
Ethernet motor controllers, Phantom cameras, Raspberry Pi triggers, and
`bapsf_motion` workflows.

## Current Status

```text
  `Data_Run_**.py` — the production path for routine runs. They use the
  spooled pipeline: the acquire process spools each shot to a fast disk and a
  separate `Offload_Run.py` drains the spool into the HDF5 file. This is the path
  the hardware PC runs and the main integration target.
```

- `Data_Run_bmotion.py` uses bapsf_motion library for motor communication and control.
- 45-degree probe acquisition is not migrated; `Data_Run_45deg.py` exits early
  with an unsupported message.
- Automated tests run on mock devices and real hardware tests exist.
- EPICS is the planned future control backend.

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

Optional hardware extras
```
REM LeCroy scope control (lab_scopes + PyVISA)
python -m pip install -e ".[scope]"

REM bapsf_motion workflows (bapsf-motion + xarray)
python -m pip install -e ".[bmotion]"

REM both at once
python -m pip install -e ".[scope,bmotion]"

REM Jupyter for the notebooks/ examples
python -m pip install -e ".[dev]"
```

The `camera` extra is intentionally empty because the Phantom camera SDK and
Python bindings are hardware-PC specific. Install those according to the camera
PC setup notes before using camera mode.


## Configuration Reference

The repo still uses INI-style experiment configs for immediate usability.
Internally, `lapd_daq.config.load_run_config()` converts the INI file to typed
Python config objects.

Important sections:

- `[storage]`: `spool_dir` (fast disk for per-shot spool) and `hdf5_dir` —
  required by the spooled `Data_Run*.py` path. Optional `disk_full_pause_seconds`
  / `disk_full_max_retries` tune the pause+retry when the spool disk fills.
- `[acquisition]`: per-shot tuning for the spooled path.
- `[nshots]`: `num_duplicate_shots`, `num_run_repeats`
- `[experiment]`: experiment `name` (used to build the HDF5 filename). The
  run description lives in a separate `description.txt` next to the config,
  written into the HDF5 `description` attribute at run start and overwritten at
  run end.
- `[scopes]`: scope display names and descriptions
- `[channels]`: channel descriptions using `ScopeName_C1` keys
- `[scope_ips]`: direct scope IPs
- `[analysis]`: `auto_plot` — post-run line-profile PNG plotting (default on).
- `[position]` / `[motor_ips]`: XY/XYZ grid parameters and motor IPs for grid mode
- `[camera_config]`: Phantom camera settings, used only in camera/dropper modes
- `[raspberry_pi]`: Raspberry Pi trigger settings for dropper mode
- `[bmotion]`: motion-group/direction/recovery settings for the bmotion script


## bmotion TOML Reference

`Data_Run_bmotion.py` reads a **second** file, `bmotion_config.toml`, that
describes the probe-drive *hardware* (the INI's `[bmotion]` section only selects
which of these groups to run today). This TOML is parsed by `bapsf_motion`'s
`RunManager`. If it is missing a required table or field, the run aborts at
startup with a `DATA RUN DID NOT START -- configuration error` message naming
`bmotion_config.toml`; the notes below say what must be present so that does not
happen.

A motion group only loads if **every** part below is present and valid. A
missing/empty `drive` or `transform` is the usual cause of the
`'NoneType' object has no attribute 'terminated'` failure: the library silently
discards the half-built component and then dereferences it.

Required structure (one `[run]`, at least one motion group under it):

```toml
[run]
name = "my data run"            # required: a name for the run

# One motion group. The table name under [run] (here "mg") is arbitrary; add
# more (e.g. [run.mg2]) for multiple drives. The INI [bmotion] motion_groups
# key selects groups by the drive name below.
[run.mg]
name = "P32"                    # required: motion-group name

# --- Drive: the physical stage and its motor axes -----------------------------
[run.mg.drive]
name = "XY-drive"              # required: drive name (used by [bmotion] selection)

# One [...axes.N] table per motor axis. Each axis requires ALL of these keys:
[run.mg.drive.axes.0]
name = "x"                     # axis label (the run expects axes named x and y)
ip = "192.168.0.70"            # motor controller IP
units = "cm"                   # motion-space units
units_per_rev = 0.508          # distance per motor revolution

[run.mg.drive.axes.1]
name = "y"
ip = "192.168.0.80"
units = "cm"
units_per_rev = 0.508

# --- Transform: motion-space <-> drive-space mapping --------------------------
[run.mg.transform]
type = "identity"              # required: transform type (e.g. "identity")

# --- Motion builder: the grid of positions to visit --------------------------
[run.mg.motion_builder]

# One [...space.N] per axis, in the same order as the drive axes. The DAQ
# requires exactly two axes labelled x and y (a rectangular grid).
[run.mg.motion_builder.space.0]
label = "x"
range = [-30, 30]              # min, max in motion-space units
num = 13                       # number of points along this axis

[run.mg.motion_builder.space.1]
label = "y"
range = [-30, 30]
num = 13
```

Checklist for "what must be included":

- **`[run]`** with a `name`, and **at least one motion-group table** under it
  (no motion groups → `no valid motion groups were defined`).
- **`[...drive]`** with a `name` and an **`axes`** table; each
  **`[...drive.axes.N]`** must have `name`, `ip`, `units`, and `units_per_rev`
  (any missing axis key → the drive is discarded → `NoneType ... terminated`).
- **`[...transform]`** with a `type`.
- **`[...motion_builder]`** with one **`[...space.N]`** per axis, each having
  `label`, `range`, and `num`.
- The DAQ's HDF5 layout expects exactly **two axes labelled `x` and `y`** on a
  rectangular grid; other layouts are rejected later with a clear message.
- Each drive `name` must be **unique** across motion groups (the INI `[bmotion]`
  selection resolves groups by drive name).

## HDF5 Output

The new writer keeps the root-level scope layout compatible with old analysis
readers:

```text
run.hdf5
  attrs:
    description
    creation_time
    schema_version
    run_mode
    software_versions
  Configuration/
    experiment_config
  Control/
    Run/
      attrs: config_path, num_duplicate_shots, num_run_repeats
      shot_status
    Devices/
    Positions/
      positions_setup_array
      positions_array
    FastCam/
      shot number
      cine file name
      timestamp
  ScopeName/
    attrs: description, ip_address, scope_type, shot_count
    time_array
    shot_1/
      C1_data
      C1_header
      C2_data
      C2_header
```

Scope waveform datasets store raw `int16` samples and LeCroy binary headers.
Use the `lab_scopes` readers (e.g. `read_scope_data`) or the in-repo
`read_and_analyze/` package to convert raw traces to voltage.

Channel datasets are compressed (Blosc2, falling back to lzf), so HDFView may
not decode them without the matching plugin; read them via the Python readers
above.

## Repository Layout

```text
LAPD_DAQ/
  lapd_daq/        New framework: CLI, config model, run engine, devices, HDF5 writer
  acquisition/     Acquisition package used by Data_Run*.py (spool, offload, scope/bmotion loops)
  spooling/        Per-shot spool format + disk-full pause/retry helper
  drivers/         LeCroy and Phantom hardware driver wrappers
  motion/          Motor control and position management helpers
  pi_gpio/         Raspberry Pi trigger/dropper client package
  legacy/          Superseded scripts kept for reference
  notebooks/       Scratch notebooks for scope and motor testing
  tests/           Automated tests (mock by default, gated hardware checks) — see docs/tests.md
  docs/            Long-form documentation pages
  Data_Run.py      Standard spooled acquisition (stationary / grid)
  Data_Run_bmotion.py            Spooled bmotion acquisition
  Data_Run_MultiScope_Camera.py  Multi-scope + camera acquisition
  Data_Run_45deg.py              Unsupported (exits early)
  Offload_Run.py   Drains the spool into the HDF5 file (run alongside Data_Run*.py)
  example_experiment_config.ini
  pyproject.toml
```

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
