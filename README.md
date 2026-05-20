# LAPD_DAQ

Python data-acquisition tools for LAPD experiments using LeCroy oscilloscopes,
Ethernet motor controllers, Phantom cameras, Raspberry Pi triggers, and
`bapsf_motion` workflows.

The current branch introduces a cleaner `lapd_daq` acquisition framework with a
single CLI, mock-device dry runs, centralized HDF5 writing, and adapter
boundaries that can later be replaced by EPICS PV-backed devices. The older
`Data_Run_*.py` scripts are still present for transition workflows.

## Current Status

- Supported now: stationary scope acquisition, XY/XYZ grid acquisition, camera
  mode, dropper trigger mode, bmotion script workflow, and mock dry runs.
- Mock tests are the automated test target in this repo. Real hardware tests
  should be performed separately on the dedicated hardware PC.
- 45-degree probe acquisition is not migrated in this refactor branch.
  `Data_Run_45deg.py` exits early with a clear unsupported message.
- EPICS is the future control backend. For now, Python directly controls
  devices through adapter classes.

## Repository Layout

```text
LAPD_DAQ/
  lapd_daq/                  New package: CLI, config model, run engine, devices, HDF5 writer
  acquisition/               Transitional acquisition package used by Data_Run*.py scripts
  drivers/                   LeCroy and Phantom hardware driver wrappers
  motion/                    Motor control and position management helpers
  pi_gpio/                   Raspberry Pi trigger/dropper client package
  legacy/                    Superseded scripts kept for reference
  notebooks/                 Scratch notebooks for scope and motor testing
  tests/                     Automated tests (mock by default, gated hardware checks) — see docs/tests.md
  docs/                      Long-form documentation pages
  Data_Run.py                Transitional legacy-style standard acquisition script
  Data_Run_MultiScope_Camera.py
  Data_Run_bmotion.py
  Data_Run_45deg.py          Explicitly unsupported in this refactor branch
  example_experiment_config.txt
  pyproject.toml
```

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

Install optional hardware extras only on machines that need them:

Command Prompt:

```cmd
REM LeCroy scope control through lab_scopes and PyVISA
python -m pip install -e ".[scope]"

REM bapsf_motion workflows
python -m pip install -e ".[bmotion]"
```

PowerShell:

```powershell
# LeCroy scope control through lab_scopes and PyVISA
python -m pip install -e ".[scope]"

# bapsf_motion workflows
python -m pip install -e ".[bmotion]"
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
   copy example_experiment_config.txt experiment_config.txt
   ```

   PowerShell:

   ```powershell
   Copy-Item example_experiment_config.txt experiment_config.txt
   ```

2. Edit `experiment_config.txt` for the run. For a simple stationary mock run,
   keep `[scope_ips]` populated and either remove or comment out active
   `[position]` values.

3. Run the CLI in dry-run mode:

   Command Prompt:

   ```cmd
   lapd-daq run --config experiment_config.txt --mode stationary --dry-run --output mock_stationary.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.txt --mode stationary --dry-run --output mock_stationary.hdf5
   ```

4. Confirm the output by running the lapd_daq unit tests. See
   [docs/tests.md](docs/tests.md#on-a-development-machine-no-hardware-connected)
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

3. Edit `experiment_config.txt`:

   ```ini
   [nshots]
   num_duplicate_shots = 5
   num_run_repeats = 1

   [experiment]
   description = Describe plasma conditions, probe setup, timing, and operator notes.

   [scopes]
   BdotScope = LeCroy HDO4104 - 4GHz 20GS/s

   [channels]
   BdotScope_C1 = Bdot probe signal
   BdotScope_C2 = Trigger monitor

   [scope_ips]
   BdotScope = 192.168.7.63
   ```

4. For stationary acquisition, leave `[position]` empty or remove it:

   Command Prompt:

   ```cmd
   lapd-daq run --config experiment_config.txt --mode stationary --output run_stationary.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.txt --mode stationary --output run_stationary.hdf5
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
   lapd-daq run --config experiment_config.txt --mode grid --output run_grid.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.txt --mode grid --output run_grid.hdf5
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
   lapd-daq run --config experiment_config.txt --mode camera --output run_camera.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.txt --mode camera --output run_camera.hdf5
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
   lapd-daq run --config experiment_config.txt --mode dropper --output run_dropper.hdf5
   ```

   PowerShell:

   ```powershell
   lapd-daq run --config experiment_config.txt --mode dropper --output run_dropper.hdf5
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

2. Edit `Data_Run_bmotion.py` and set:

   - `exp_name`
   - `base_path`
   - `config_path`
   - `toml_path`

3. Prepare `experiment_config.txt` for scopes and shots.

4. Prepare the separate bmotion TOML file for motion groups, drives, transforms,
   motion builders, and exclusion zones.

5. Run:

   Command Prompt:

   ```cmd
   python Data_Run_bmotion.py
   ```

   PowerShell:

   ```powershell
   python Data_Run_bmotion.py
   ```

6. Follow the interactive prompts to select motion groups and motion direction.

The output stores scope data in the standard root-level scope groups and bmotion
configuration/position data under `/Configuration` and `/Control/Positions`.

## Configuration Reference

The repo still uses INI-style experiment configs for immediate usability.
Internally, `lapd_daq.config.load_run_config()` converts the INI file to typed
Python config objects.

Important sections:

- `[nshots]`: `num_duplicate_shots`, `num_run_repeats`
- `[experiment]`: human-readable run description stored in HDF5
- `[scopes]`: scope display names and descriptions
- `[channels]`: channel descriptions using `ScopeName_C1` keys
- `[scope_ips]`: direct scope IPs
- `[position]`: XY/XYZ grid parameters for `--mode grid`
- `[motor_ips]`: motor controller IPs for direct grid motion
- `[camera_config]`: Phantom camera settings, used only in camera/dropper modes
- `[raspberry_pi]`: Raspberry Pi trigger settings for dropper mode

Future EPICS-related PV fields may be added to the INI config during migration,
but EPICS-native `.db`, `.dbd`, `.template`, `.substitutions`, and `st.cmd`
files should eventually own hardware control configuration.

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
Use the existing `data-analysis/read/read_scope_data.py` helpers or
`lab_scopes` readers to convert raw traces to voltage.

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
