# Motion Package

The `motion` package contains position generation, legacy Ethernet motor
control, boundary checks, and HDF5 position helpers used by LAPD acquisition
runs.

## Current Status

- Supported in the refactored runner: stationary, XY grid, and XYZ grid motion.
- `bapsf_motion` runs are handled by `Data_Run_bmotion.py` and
  `acquisition/bmotion.py`.
- 45-degree probe acquisition is not migrated in this branch.
  `Data_Run_45deg.py` exits early with an unsupported message.

## XY/XYZ Grid Configuration

Add a `[position]` section to `experiment_config.ini` and run the CLI with
`--mode grid`.

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

For 3D motion, also add:

```ini
nz = 11
zmin = -5
zmax = 5
```

Run:

Command Prompt:

```cmd
lapd-daq run --config experiment_config.ini --mode grid --output run_grid.hdf5
```

PowerShell:

```powershell
lapd-daq run --config experiment_config.ini --mode grid --output run_grid.hdf5
```

## Stationary Runs

For stationary acquisition, remove the `[position]` section or leave it empty:

Command Prompt:

```cmd
lapd-daq run --config experiment_config.ini --mode stationary --output run_stationary.hdf5
```

PowerShell:

```powershell
lapd-daq run --config experiment_config.ini --mode stationary --output run_stationary.hdf5
```

## PositionManager

`PositionManager` is used by the transitional `acquisition` runner to create
planned positions, initialize HDF5 position groups, initialize direct motor
control, and record achieved positions.

```python
from motion import PositionManager

pos_manager = PositionManager("run.hdf5", config_path="experiment_config.ini")
positions = pos_manager.initialize_position_hdf5()
mc = pos_manager.initialize_motor()
```

During acquisition, achieved positions are written with:

```python
pos_manager.update_position_hdf5(shot_num, {"x": xpos, "y": ypos, "z": None})
```

The new `lapd_daq` framework stores planned positions under
`/Control/Positions/positions_setup_array` and achieved positions under
`/Control/Positions/positions_array`.

## Utility Functions

Common helpers remain available for legacy scripts and exploratory checks:

```python
from motion import get_positions_xy, get_positions_xyz
from motion import outer_boundary, obstacle_boundary, motor_boundary

positions, xpos, ypos = get_positions_xy(config)
positions, xpos, ypos, zpos = get_positions_xyz(config)
is_valid = outer_boundary(x, y, z, config)
```

## 45-Degree Probe Notes

Older configs may contain 45-degree fields such as:

```ini
[position]
probe_list = P16,P22,P29,P34,P42
nx = 37
xstart = {"P16": -38, "P22": -18, "P29": -38, "P34": -38, "P42": -38}
xstop = {"P16": -38, "P22": 18, "P29": -38, "P34": -38, "P42": -38}
```

Those settings are preserved for reference, but this refactor branch does not
run 45-degree acquisition. Use the known pre-refactor hardware PC workflow until
45-degree support is explicitly migrated and mock-tested.

## bmotion

`bapsf_motion` support uses a separate TOML file and the script workflow:

Command Prompt:

```cmd
python Data_Run_bmotion.py
```

PowerShell:

```powershell
python Data_Run_bmotion.py
```

The script stores selected motion lists and achieved positions under
`/Control/Positions/{motion_group_name}/` and preserves the TOML text under
`/Configuration/bmotion_config`.
