# LAPD_DAQ

Python data-acquisition tools for LAPD experiments using LeCroy oscilloscopes,
Ethernet motor controllers, Phantom cameras, Raspberry Pi triggers, and
`bapsf_motion` workflows.

Two acquisition front-ends coexist during the transition:

- **`lapd-daq` CLI** — the newer framework (`lapd_daq/` package): typed config,
  centralized HDF5 writing, mock-device dry runs, and adapter boundaries that can
  later be backed by EPICS PVs. Covers stationary, grid, camera, and dropper modes.
- **`Data_Run_*.py` scripts** — the production path for routine runs. They use the
  spooled pipeline: the acquire process spools each shot to a fast disk and a
  separate `Offload_Run.py` drains the spool into the HDF5 file. This is the path
  the hardware PC runs and the main integration target.

## Current Status

- bmotion runs are script-driven (`Data_Run_bmotion.py`); the CLI's bmotion
  adapter is not yet wired and points you to that script.
- 45-degree probe acquisition is not migrated; `Data_Run_45deg.py` exits early
  with an unsupported message.
- Automated tests run on mock devices; real hardware tests run separately on the
  hardware PC. EPICS is the planned future control backend.

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

## Install

Use Python 3.10 or newer.

> **Use a standalone Python, not Anaconda/conda.** Mixing `pip` and `conda`
> causes "installed but not importable" failures: a bare `pip install` can land
> packages in conda's base environment while your acquisition runs under a
> different interpreter, so `lab_scopes` (and other deps) appear missing.
>
> On each PC:
> 1. Install Python from [python.org](https://www.python.org/downloads/windows/)
>    (3.11 or 3.12 recommended), 64-bit.
> 2. If Anaconda is installed, stop it from hijacking every shell:
>    `conda config --set auto_activate_base false`, then reopen the terminal.
> 3. Build the venv from the standalone Python (e.g.
>    `C:\Python312\python.exe -m venv .venv`), activate it, and **always use
>    `python -m pip ...`** (never bare `pip`) so installs go into the venv.
>
> Verify the environment is self-consistent before running — all three must
> point inside `.venv`:
> ```powershell
> python -c "import sys; print(sys.executable)"
> python -c "import lab_scopes; print(lab_scopes.__file__)"
> pip -V
> ```

Replace `C:\Python312\python.exe` below with the path to your standalone
Python (see the note above); the rest is the same on every PC.

Command Prompt:

```cmd
git clone https://github.com/hjia94/LAPD_DAQ.git
cd LAPD_DAQ
C:\Python312\python.exe -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -e .
```

PowerShell:

```powershell
git clone https://github.com/hjia94/LAPD_DAQ.git
cd LAPD_DAQ
C:\Python312\python.exe -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e .
```

Install optional hardware extras only on machines that need them. Each name in
the brackets is a separate group; combine them comma-separated (no spaces) to
install several at once, e.g. a PC running both scope acquisition and bapsf
motion:

Command Prompt:

```cmd
REM LeCroy scope control (lab_scopes + PyVISA)
python -m pip install -e ".[scope]"

REM bapsf_motion workflows (bapsf-motion + xarray)
python -m pip install -e ".[bmotion]"

REM both at once
python -m pip install -e ".[scope,bmotion]"

REM Jupyter for the notebooks/ examples
python -m pip install -e ".[dev]"
```

PowerShell:

```powershell
# LeCroy scope control (lab_scopes + PyVISA)
python -m pip install -e ".[scope]"

# bapsf_motion workflows (bapsf-motion + xarray)
python -m pip install -e ".[bmotion]"

# both at once
python -m pip install -e ".[scope,bmotion]"

# Jupyter for the notebooks/ examples
python -m pip install -e ".[dev]"
```

The `camera` extra is intentionally empty because the Phantom camera SDK and
Python bindings are hardware-PC specific. Install those according to the camera
PC setup notes before using camera mode.

## Step-by-Step: Run a Mock Acquisition

Use this first after any install or code change. It writes an HDF5 file without
touching hardware.

1. Create a local config:

   Command Prompt:

   ```cmd
   copy example_experiment_config.ini experiment_config.ini
   ```

   PowerShell:

   ```powershell
   Copy-Item example_experiment_config.ini experiment_config.ini
   ```

2. Edit `experiment_config.ini` for the run. For a simple stationary mock run,
   keep `[scope_ips]` populated and either remove or comment out active
   `[position]` values.

3. Run the CLI in dry-run mode:

   Command Prompt:

   ```cmd
   lapd-daq run --config experiment_config.ini --mode stationary --dry-run --output mock_stationary.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.ini --mode stationary --dry-run --output mock_stationary.hdf5
   ```

4. Confirm the output by running the lapd_daq unit tests. See
   [docs/tests.md](docs/tests.md#on-a-development-machine-no-hardware)
   for the full sequence.

Expected HDF5 structure includes a root scope group such as `bdotscope`, a
`time_array`, `shot_N/C1_data`, `shot_N/C1_header`, and run metadata under
`/Control/Run`.

## Step-by-Step: Standard Scope Acquisition

Use this on the hardware PC after the mock dry run succeeds.

1. Activate the environment:

   Command Prompt:

   ```cmd
   .venv\Scripts\activate.bat
   ```

   PowerShell:

   ```powershell
   .\.venv\Scripts\activate
   ```

2. Install scope dependencies if needed:

   Command Prompt:

   ```cmd
   python -m pip install -e ".[scope]"
   ```

   PowerShell:

   ```powershell
   python -m pip install -e ".[scope]"
   ```

3. Edit `experiment_config.ini`:

   ```ini
   [nshots]
   num_duplicate_shots = 5
   num_run_repeats = 1

   [experiment]
   name = my_experiment
   # The run description is no longer set here. Put it in a separate
   # description.txt next to this config (see description.txt below).

   [scopes]
   BdotScope = LeCroy HDO4104 - 4GHz 20GS/s

   [channels]
   BdotScope_C1 = Bdot probe signal
   BdotScope_C2 = Trigger monitor

   [scope_ips]
   BdotScope = 192.168.7.63
   ```

   Optionally create a `description.txt` next to the config with the free-text
   run description (plasma conditions, probe setup, timing, operator notes). It
   is written into the HDF5 `description` attribute at run start and overwritten
   at run end, so you can write or edit it before or during the run. See
   `example_description.txt`.

4. For stationary acquisition, leave `[position]` empty or remove it:

   Command Prompt:

   ```cmd
   lapd-daq run --config experiment_config.ini --mode stationary --output run_stationary.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.ini --mode stationary --output run_stationary.hdf5
   ```

5. For XY grid acquisition, fill `[position]` and `[motor_ips]`:

   ```ini
   [position]
   nx = 31
   ny = 41
   xmin = -15
   xmax = 15
   ymin = -20
   ymax = 20

   [motor_ips]
   x = 192.168.7.101
   y = 192.168.7.102
   # z = 192.168.7.103
   ```

   Then run:

   Command Prompt:

   ```cmd
   lapd-daq run --config experiment_config.ini --mode grid --output run_grid.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.ini --mode grid --output run_grid.hdf5
   ```

6. Check the terminal summary and inspect the HDF5 file before moving it into
   long-term storage.

## Step-by-Step: Camera and Dropper Modes

Camera mode uses the new CLI and `drivers.phantom_recorder.PhantomRecorder`.
The camera is enabled only when `--mode camera` or `--mode dropper` is selected;
having `[camera_config]` in the config does not affect stationary or grid runs.

1. Confirm the Phantom SDK and Python camera bindings are installed on the
   hardware PC.

2. Add camera settings:

   ```ini
   [camera_config]
   exposure_us = 30
   fps = 10000
   pre_trigger_frames = -500
   post_trigger_frames = 1000
   resolution = 256,256
   ```

3. Run camera mode:

   Command Prompt:

   ```cmd
   lapd-daq run --config experiment_config.ini --mode camera --output run_camera.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.ini --mode camera --output run_camera.hdf5
   ```

4. For dropper mode, add trigger settings:

   ```ini
   [raspberry_pi]
   pi_host = 192.168.7.38
   pi_port = 54321
   ```

   Then run:

   Command Prompt:

   ```cmd
   lapd-daq run --config experiment_config.ini --mode dropper --output run_dropper.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.ini --mode dropper --output run_dropper.hdf5
   ```

Camera `.cine` files are saved next to the HDF5 output. HDF5 camera metadata
stores the basename for compatibility.

## Step-by-Step: bmotion Acquisition

The bmotion workflow remains script-driven during this transition.

1. Install bmotion dependencies on the hardware PC:

   Command Prompt:

   ```cmd
   python -m pip install -e ".[bmotion]"
   ```

   PowerShell:

   ```powershell
   python -m pip install -e ".[bmotion]"
   ```

2. Edit `Data_Run_bmotion.py` and set `base_path` (the only user setting). The
   config, bmotion TOML, and `description.txt` are found inside it; the
   experiment name and HDF5 filename come from `experiment_config.ini`.

3. Prepare, inside `base_path`:
   - `experiment_config.ini` — scopes, shots, and a `[storage] spool_dir`.
   - `bmotion_config.toml` — motion groups, drives, transforms, builders, and
     exclusion zones.

4. Run (it auto-launches `Offload_Run.py` in a new console to drain the spool):

   Command Prompt:

   ```cmd
   python Data_Run_bmotion.py
   ```

   PowerShell:

   ```powershell
   python Data_Run_bmotion.py
   ```

5. Follow the interactive prompts to select motion groups and motion direction.

The output stores scope data in the standard root-level scope groups and bmotion
configuration/position data under `/Configuration` and `/Control/Positions`.

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

Future EPICS-related PV fields may be added to the INI config during migration,
but EPICS-native `.db`, `.dbd`, `.template`, `.substitutions`, and `st.cmd`
files should eventually own hardware control configuration.

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

## Tests

See [docs/tests.md](docs/tests.md) for the full test suite documentation:
what each module covers, the recommended run sequence on a development
machine vs. the hardware PC, and the gating flags for hardware-touching
tests.

## EPICS Migration Path

1. Clean Python framework using direct hardware adapters.
2. Add EPICS PV naming fields while direct adapters still run hardware.
3. Move devices behind EPICS IOCs and replace direct adapters with EPICS PV
   adapters.
4. Let EPICS own device state, interlocks, autosave, and control logic.
5. Keep `LAPD_DAQ` focused on experiment orchestration, shot sequencing, and
   HDF5 data products.
