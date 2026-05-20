# Test Suite

This page documents every test module under [`tests/`](../tests), what it
covers, what (if anything) it needs to run, and the recommended sequence
for running them on a development machine versus on the hardware PC.

The format is Markdown with light MyST admonitions
(`:::{note}`, `:::{warning}`) so the file renders cleanly on GitHub today
and via [myst-parser](https://myst-parser.readthedocs.io/) when this
project is built on ReadTheDocs.

## At a glance

| Module | Tests | Runs on | Needs hardware? | Optional deps |
|---|---|---|---|---|
| [`test_bmotion_config.py`](#test_bmotion_configpy) | 14 | any PC | no | — |
| [`test_bmotion_loops.py`](#test_bmotion_loopspy) | 11 | any PC | no | — (stubs in [`_bmotion_stubs.py`](../tests/_bmotion_stubs.py)) |
| [`test_bmotion_hardware.py`](#test_bmotion_hardwarepy) | 2 | hardware PC | **yes** (motors) | `bapsf_motion`, `xarray` |
| [`test_lapd_daq_config.py`](#test_lapd_daq_configpy) | 2 | any PC | no | `lapd_daq` installed |
| [`test_lapd_daq_engine.py`](#test_lapd_daq_enginepy) | 3 | any PC | no | `lapd_daq` installed |
| [`test_lapd_daq_compat.py`](#test_lapd_daq_compatpy) | 4 | any PC* | no | `lab_scopes` for 1 test, TRC fixtures for 1 test |
| [`test_daq_framework_combined.py`](#test_daq_framework_combinedpy) | 1 | any PC or hardware PC | configurable | `bapsf_motion` + `xarray` when `MOTION_MODE = "real"`; otherwise none |
| [`test_hardware_instruments.py`](#test_hardware_instrumentspy) | 5 | hardware PC | **yes** (scope / motors / camera) | per-instrument |
| [`test_hardware_daq_check.py`](#test_hardware_daq_checkpy) | 5 | any PC | no | — |

*`test_lapd_daq_compat.py` self-skips when `lab_scopes` or the TRC
fixture directory aren't present.*

There are also three private helper modules — they're not tests but the
test files import from them:

| Helper | Purpose |
|---|---|
| [`_bmotion_stubs.py`](../tests/_bmotion_stubs.py) | `sys.modules` stubs for `bapsf_motion`/`xarray`, plus the `StubRunManager` / `StubMotionGroup` / `StubMSA` test doubles and HDF5 temp-file factories used by `test_bmotion_loops.py` |
| [`_hardware_check_helpers.py`](../tests/_hardware_check_helpers.py) | Fake scope payloads, parsing utilities, and config-restriction helpers used by `test_hardware_instruments.py` |
| [`_lapd_daq_fixtures.py`](../tests/_lapd_daq_fixtures.py) | Shared `CONFIG_TEXT` / `CAMERA_CONFIG_TEXT` INI strings used by all three `test_lapd_daq_*` files |

## Recommended run sequence

### On a development machine (no hardware connected)

Run in this order. Each step takes seconds; later steps assume earlier
steps pass.

1. **Pure unit tests** — fastest, no optional dependencies.
   ```cmd
   python -m unittest tests.test_bmotion_config tests.test_bmotion_loops tests.test_hardware_daq_check
   ```
   Expected: **30 passed**.

2. **lapd_daq unit tests** — requires the `lapd_daq` package installed
   (`pip install -e .`).
   ```cmd
   python -m unittest tests.test_lapd_daq_config tests.test_lapd_daq_engine tests.test_lapd_daq_compat
   ```
   Expected: **6 passed, 3 may error** if `lab_scopes` /
   `matplotlib` aren't installed on the dev box. These three errors
   are environment-only and identical to running the same tests at
   any prior commit.

3. **Hardware-gated files (should all skip)** — these confirm the gating
   is wired up. Nothing should actually run.
   ```cmd
   python -m unittest tests.test_bmotion_hardware
   ```
   Expected: **2 skipped** with messages pointing to `RUN_BMOTION_*_CHECK`.

4. **End-to-end framework run with fakes** — drives `AcquisitionRun`
   through the full pipeline using only fake devices. Slowest of the
   no-hardware tests (still under a second).
   ```cmd
   python -m unittest tests.test_daq_framework_combined
   ```
   Expected: **1 passed** with the top-of-file mode flags at their
   defaults (`SCOPE_MODE="fake"`, `MOTION_MODE="fake"`,
   `CAMERA_MODE="off"`, `RASPBERRY_PI_MODE="fake"`).

### On the hardware PC

Always run the dev-machine sequence first to catch regressions before
touching instruments. Then run the gated hardware checks one at a time,
enabling only the flag you need.

5. **bmotion end-to-end with real motors** — see
   [`test_bmotion_hardware.py`](#test_bmotion_hardwarepy) below for the
   flags. Run **after** confirming `test_bmotion_loops` is green on the
   same machine.

6. **Per-instrument hardware diagnostics** — see
   [`test_hardware_instruments.py`](#test_hardware_instrumentspy) below.
   Each subsystem (scope / motion / camera) has its own gate.

7. **Combined framework against real hardware** — flip the mode flags at
   the top of `test_daq_framework_combined.py` to `"real"` for the
   instruments you want to exercise. Treat this as a final integration
   check.

:::{warning}
Every hardware-gated test has both a `RUN_*_CHECK` flag and an
`ALLOW_*` (or `BMOTION_ALLOW_MOVE`) destructive-action gate. Both must
be `True` before any test will arm an instrument or command motion.
This is intentional belt-and-braces — don't remove either gate.
:::

## Pytest equivalents

Every command above also works under pytest, which gives richer output:

```cmd
python -m pytest tests/test_bmotion_config.py tests/test_bmotion_loops.py -v
python -m pytest tests/ -v -k "not hardware"
```

To select a single class or test:

```cmd
python -m pytest tests/test_hardware_instruments.py::ScopeHardwareCheck -v -s
```

The `-s` flag is important for hardware tests — they print live status
that gets swallowed by pytest's default output capture.

## Module reference

### `test_bmotion_config.py`

**Subject:** the `[bmotion]` section parser at
[`acquisition/bmotion_config.py`](../acquisition/bmotion_config.py).

**14 tests** covering:
- `motion_groups` selector: `"all"`, comma-separated keys, whitespace
  keys, string keys (e.g. `"P22 P29"`)
- `direction`: bare word (`"forward"` / `"backward"`) broadcast,
  per-key mapping (`"0=forward, 2=backward"`)
- `execution_order`: default `"interleaved"`, explicit
  `"sequential"`, invalid value rejection
- Error paths: invalid motion-group keys, invalid direction values,
  unknown keys in a direction mapping, empty `RunManager`

**Dependencies:** stub-based — uses an in-test `_StubRunManager` with a
`.mgs` dict. Importable on any machine; no `bapsf_motion` install
required.

**Run:**
```cmd
python -m unittest tests.test_bmotion_config -v
```

---

### `test_bmotion_loops.py`

**Subject:** the acquisition loop in
[`acquisition/bmotion.py`](../acquisition/bmotion.py) — both the
interleaved and the sequential-per-group execution paths.

**11 tests** verifying the invariants documented in commit `3e9c8a2`:
- `configure_bmotion_hdf5_group` writes `execution_order` and the
  per-MG `positions_array` shape into the HDF5 selection blob
- `get_motion_list_size` / `get_max_motion_list_size` handle multi-group
  managers and reject empty motion lists
- `move_to_index` handles forward / backward direction and out-of-range
  indices (warning + skip)
- `_run_interleaved` calls `move_to_index` once per motion-list index,
  records every group every shot, with `shot_num` incrementing by
  `nshots`
- `_run_sequential` completes group A's full motion list before group
  B starts; records **only** the active group's `positions_array` row;
  `shot_num` is a single global counter
- A real temp-HDF5 end-to-end of `_run_sequential` proves idle rows
  stay zero and active rows contain `1..total_shots`
- `_take_shots_at_position` skip-on-`ValueError` path creates a
  `shot_N` group with `attrs['skipped'] = True` and still records
  positions

**Dependencies:** none on the host machine. The module's
[`setUpModule` / `tearDownModule`](../tests/test_bmotion_loops.py)
install stubs from [`_bmotion_stubs.py`](../tests/_bmotion_stubs.py)
into `sys.modules` for `bapsf_motion`, `xarray`, and
`acquisition.scope_runner`, then roundtrip the state on teardown so the
stubs don't leak into sibling tests.

**Run:**
```cmd
python -m unittest tests.test_bmotion_loops -v
```

---

### `test_bmotion_hardware.py`

**Subject:** end-to-end `run_acquisition_bmotion(...)` against real
motors and a real `bmotion_config.toml`.

**2 tests** (both classes inherit `_BmotionHardwareBase`):
- `BmotionInterleavedHardwareCheck.test_interleaved_end_to_end` — runs a
  small motion list with `execution_order = interleaved`; asserts every
  selected motion group has a populated row at every shot index
- `BmotionSequentialHardwareCheck.test_sequential_end_to_end` — same
  with `execution_order = sequential`; asserts each motion group's
  active rows form a disjoint contiguous block and combined coverage
  equals `total_shots`

**Gating flags** at the top of the file:
```python
RUN_BMOTION_INTERLEAVED_CHECK = False
RUN_BMOTION_SEQUENTIAL_CHECK   = False
BMOTION_ALLOW_MOVE             = False   # destructive-action gate
EXPERIMENT_CONFIG_PATH         = "experiment_config.txt"
BMOTION_TOML_PATH              = "bmotion_config.toml"
BMOTION_NSHOTS                 = 1
```

Tests skip automatically when:
1. The corresponding `RUN_*` flag is `False`
2. `BMOTION_ALLOW_MOVE` is `False`
3. `bapsf_motion` or `xarray` aren't installed
4. `experiment_config.txt` or `bmotion_config.toml` aren't found

:::{warning}
Setting `BMOTION_ALLOW_MOVE = True` will command real motors. Confirm
the configured motion list is safe for the installed probe before
flipping this flag.
:::

**Run a single mode:**
```cmd
python -m unittest tests.test_bmotion_hardware.BmotionSequentialHardwareCheck -v
```

---

### `test_lapd_daq_config.py`

**Subject:** `lapd_daq.config.load_run_config` — the experiment-config
parser used by the new CLI / engine path.

**2 tests:**
- `test_config_loader_preserves_existing_ini_and_detects_grid` — round-trip
  of a representative INI, asserts `num_duplicate_shots`,
  `motion.kind == "xy_grid"`, scope name, and `experiment.description`
  survive parsing. Also exercises the inline-comment-in-INI path.
- `test_camera_config_does_not_enable_camera_outside_camera_modes` —
  loading the same config with `mode="stationary"` vs `mode="camera"`
  flips `config.camera.enabled` even though the `[camera_config]`
  section is identical.

Uses the shared [`CONFIG_TEXT`](../tests/_lapd_daq_fixtures.py) /
`CAMERA_CONFIG_TEXT` fixture.

**Run:**
```cmd
python -m unittest tests.test_lapd_daq_config -v
```

---

### `test_lapd_daq_engine.py`

**Subject:** the `lapd_daq.engine.AcquisitionRun` shot planner plus the
public import surface of the legacy `acquisition` package.

**3 tests:**
- `test_acquisition_import_does_not_import_bmotion_or_scope_hardware` —
  asserts `import acquisition` does **not** eagerly pull in
  `acquisition.bmotion` (lazy-import guard).
- `test_grid_shot_plan_uses_duplicates_and_positions` — `nx=2, ny=2,
  nshots=2` produces 8 shot plans with correct coordinates and
  duplicate indices.
- `test_stationary_mode_ignores_position_section` — same config with
  `mode="stationary"` yields 2 shot plans, both with `position is
  None`.

End-to-end acquisition runs belong to
[`test_daq_framework_combined.py`](#test_daq_framework_combinedpy);
this file deliberately stops at the planner.

**Run:**
```cmd
python -m unittest tests.test_lapd_daq_engine -v
```

---

### `test_lapd_daq_compat.py`

**Subject:** back-compat / boundary tests that don't share production
code but all concern *interoperating with legacy or external
interfaces*.

**4 tests** grouped into three classes:

- `PhantomAdapterTests.test_phantom_adapter_saves_cine_to_configured_directory`
  — `PhantomCameraAdapter` writes `<experiment>_shot007.cine` into the
  recorder's configured `save_path`.

- `DataRun45DegSunsetTests.test_45deg_entrypoint_reports_unsupported_without_old_run_call`
  — `Data_Run_45deg.main()` raises `SystemExit` with the expected
  "not migrated" message.

- `HDF5ReaderCompatibilityTests.test_old_lab_scopes_hdf5_reader_reads_mock_generated_file`
  — files written by the new engine are still readable by
  `lab_scopes.io.lecroy_files.read_hdf5_scope_data`.
  **Self-skips when `lab_scopes` isn't installed.**

- `HDF5ReaderCompatibilityTests.test_trc_replay_scope_writes_hdf5_readable_by_old_pydaq_reader`
  — `TRCReplayScopeDevice` produces files readable by the pydaq
  reader at `data-analysis/read/read_scope_data.py`.
  **Self-skips when `D:\data\raw data` fixtures or `data-analysis`
  aren't present.**

**Run:**
```cmd
python -m unittest tests.test_lapd_daq_compat -v
```

---

### `test_daq_framework_combined.py`

**Subject:** end-to-end acquisition exercising the full pipeline — scope
→ motion → camera → raspberry-pi trigger → HDF5 writer — with each
device independently switchable between `"off"`, `"fake"`, and
`"real"`. The fake/off-motion path drives `AcquisitionRun.execute()`;
the real-motion path routes through
[`acquisition.run_acquisition_bmotion`](../acquisition/bmotion.py) to
match how a production data run actually looks.

**1 test** (`CombinedFrameworkAcquisitionTest.test_acquisition_runs_and_hdf5_structure_is_correct`)
parameterised by the module-level mode flags:
```python
SCOPE_MODE        = "fake"
MOTION_MODE       = "fake"
CAMERA_MODE       = "off"
RASPBERRY_PI_MODE = "fake"
```

The test:
1. Builds an experiment_config.txt on the fly from the mode flags. When
   `MOTION_MODE = "real"` it writes a `[bmotion]` section
   (motion_groups / direction / execution_order) instead of `[position]`.
2. Engine path (fake/off motion): builds an `AcquisitionDevices` bundle
   with the chosen fake/real mix and runs `AcquisitionRun.execute()`.
3. Bmotion path (real motion): calls `run_acquisition_bmotion(hdf5_path,
   toml_path, config_path)` directly with `BMOTION_TOML_PATH`, exactly
   like [`test_bmotion_hardware.py`](#test_bmotion_hardwarepy).
4. Re-opens the HDF5 file and verifies the structure. Engine path:
   schema version, scope group + `shot_count`, per-shot channel
   datasets, `Control/Positions/positions_array` shot numbering, and
   `Control/Run/shot_status` length. Bmotion path:
   `Configuration/bmotion_selection`, at least one populated
   `Control/Positions/<mg>/positions_array`, and a non-empty scope
   group.

**Default behavior:** all-fake — runs on any PC, takes <1 second,
covers the engine's happy path.

**Real-hardware mode:** flip `SCOPE_MODE` / `CAMERA_MODE` /
`RASPBERRY_PI_MODE` to `"real"` to exercise individual instruments on
the engine path. To exercise real motion, set `MOTION_MODE = "real"` —
this requires a `bmotion_config.toml` at `BMOTION_TOML_PATH`,
`bapsf_motion` + `xarray` installed, and the destructive-action gate
`BMOTION_ALLOW_MOVE = True`. Real-motion runs also require
`SCOPE_MODE = "real"` because `run_acquisition_bmotion` drives the
legacy `MultiScopeAcquisition` which connects directly to scope IPs.
Connection info for real devices is hard-coded in `REAL_SCOPE_IP`,
`REAL_PI_HOST` / `REAL_PI_PORT`, and `BMOTION_TOML_PATH` at the top of
the file.

:::{warning}
Setting `BMOTION_ALLOW_MOVE = True` commands real motors. Confirm the
configured motion list is safe for the installed probe before flipping
this flag. Same gate semantics as
[`test_bmotion_hardware.py`](#test_bmotion_hardwarepy).
:::

**Run:**
```cmd
python -m unittest tests.test_daq_framework_combined -v
```

---

### `test_hardware_instruments.py`

**Subject:** per-instrument hardware diagnostics for the legacy
`Data_Run.py` path. Each test connects to **one** real instrument.

**5 tests**, each in its own class, all inheriting `_HardwareCheckBase`:

| Class | Run flag | Destructive gate | What it does |
|---|---|---|---|
| `ScopeHardwareCheck` | `RUN_SCOPE_CHECK` | `SCOPE_ALLOW_ACQUIRE` | Connect to one LeCroy scope, read time array, optionally arm + write one shot |
| `MotionHardwareCheck` | `RUN_MOTION_CHECK` | `MOTION_ALLOW_MOVE` (+ `MOTION_TARGET`) | Connect to motion controller, read probe position, optionally move to `MOTION_TARGET` |
| `CameraHardwareCheck` | `RUN_CAMERA_CHECK` | `CAMERA_ALLOW_RECORD` | Configure Phantom camera, optionally wait for trigger + save `.cine` |
| `DataRunScopeHardware` | `RUN_DATA_RUN_SCOPE_CHECK` | — | End-to-end `Data_Run.py` scope path (no motion) against real scopes |
| `DataRunMotionHardware` | `RUN_DATA_RUN_MOTION_CHECK` | `MOTION_ALLOW_MOVE` | End-to-end `Data_Run.py` motion path with real motors + fake delayed scope |

All flags default to `False`. All checks read connection info from the
`experiment_config.txt` at `EXPERIMENT_CONFIG_PATH`.

:::{note}
This file is the largest in the suite (465 lines, 5 test classes). It
predates the cleanup that split the bmotion tests by subsystem. A
future refactor could split it into `test_hardware_scope.py`,
`test_hardware_motion.py`, `test_hardware_camera.py` and hoist
`_HardwareCheckBase` into `_hardware_check_helpers.py`.
:::

**Run one check:**
```cmd
python -m pytest tests/test_hardware_instruments.py::ScopeHardwareCheck -v -s
```

**Run all enabled checks:**
```cmd
python -m pytest tests/test_hardware_instruments.py -v -s
```

---

### `test_hardware_daq_check.py`

**Subject:** pure unit tests for the helpers in
[`_hardware_check_helpers.py`](../tests/_hardware_check_helpers.py).

**5 tests:**
- `parse_move_to` accepts `"x, y"` and `"x, y, z"` strings, rejects
  wrong dimensions
- `target_coordinates` maps a tuple to `{"x": ..., "y": ..., "z": ...}`
- `fake_scope_payload` produces the expected `(traces, data, headers)`
  shape
- `restrict_scope_config` strips other scopes from a ConfigParser

No hardware. Runs in milliseconds.

**Run:**
```cmd
python -m unittest tests.test_hardware_daq_check -v
```

## CI / automation notes

The recommended pre-merge command for a dev machine:

```cmd
python -m unittest ^
    tests.test_bmotion_config ^
    tests.test_bmotion_loops ^
    tests.test_bmotion_hardware ^
    tests.test_hardware_daq_check ^
    tests.test_lapd_daq_config ^
    tests.test_lapd_daq_engine ^
    tests.test_lapd_daq_compat ^
    tests.test_daq_framework_combined
```

This runs every test that doesn't require real hardware. On a clean
checkout with the package installed in editable mode, expect:

- 42 tests run
- 36 passed
- 2 skipped (bmotion hardware checks, by design)
- 3 errors only if `matplotlib` and/or `lab_scopes` aren't installed
  (environment-only — same errors at every commit)

## ReadTheDocs setup (future)

This file is written so it can be dropped into a Sphinx project with
`myst-parser` enabled. A minimal `docs/conf.py` would look like:

```python
project = "LAPD_DAQ"
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
myst_enable_extensions = ["colon_fence"]   # for ::: admonitions
```

And a top-level `index.md` would link to this page:

```markdown
# LAPD_DAQ Documentation

```{toctree}
:maxdepth: 2

tests
```
```

No `.readthedocs.yaml` exists in the repo yet — when you're ready to
publish, see the
[ReadTheDocs configuration reference](https://docs.readthedocs.io/en/stable/config-file/v2.html).
