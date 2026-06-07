# Test Suite

This page documents every test module under [`tests/`](../tests): what it
covers, what (if anything) it needs to run, and the recommended sequence for
running them on a development machine versus the hardware PC.

## Naming convention

Test modules are named **`test_<subsystem>[_<aspect>][_hw].py`**:

- **subsystem first** — `bmotion`, `daq` (the acquisition engine), `motor`,
  `scope`, `motion`, `camera` — so related tests sort together.
- **`_hw` suffix iff the test needs real hardware.** A `_hw` file connects to a
  real instrument (scope / motors / camera) and **self-skips** on a machine
  without it. A file with no `_hw` suffix runs on **any PC** (unit / fake-device).

So at a glance: `*_hw.py` → run on the hardware PC; everything else → run
anywhere.

The routine **spooled + parallel DAQ plane** run on the hardware PC after a
change is the integration layer over these units — it exercises the real
acquisition pipeline, spool→offload, parallel scope read, and bmotion iteration
end to end. The unit files below deliberately cover what a *successful* plane run
does **not**: failure/recovery paths, config-parser error paths, and old-reader
back-compat.

## At a glance

| Module | Tests | Runs on | Needs hardware? |
|---|---|---|---|
| [`test_bmotion_config.py`](#test_bmotion_configpy) | 22 | any PC | no |
| [`test_bmotion_loops.py`](#test_bmotion_loopspy) | 14 | any PC | no |
| [`test_bmotion_recovery_hw.py`](#test_bmotion_recovery_hwpy) | 4 | hardware PC | **yes** (motors) |
| [`test_daq_core.py`](#test_daq_corepy) | 9 | any PC | no |
| [`test_daq_parallel.py`](#test_daq_parallelpy) | 11 | any PC | no |
| [`test_daq_spool.py`](#test_daq_spoolpy) | 24 | any PC | no |
| [`test_daq_check_helpers.py`](#test_daq_check_helperspy) | 5 | any PC | no |
| [`test_motor_recovery.py`](#test_motor_recoverypy) | 32 | any PC | no |
| [`test_scope_hw.py`](#test_scope_hwpy) | 2 | hardware PC | **yes** (scope) |
| [`test_motion_hw.py`](#test_motion_hwpy) | 2 | hardware PC | **yes** (motors) |
| [`test_camera_hw.py`](#test_camera_hwpy) | 1 | hardware PC | **yes** (camera) |

Private helper modules (imported by the tests, never collected themselves):

| Helper | Purpose |
|---|---|
| [`_bmotion_stubs.py`](../tests/_bmotion_stubs.py) | `sys.modules` stubs for `bapsf_motion`/`xarray` + `StubRunManager`/`StubMotionGroup`/`StubMSA` doubles and HDF5 temp-file factories used by `test_bmotion_loops.py` |
| [`_hardware_check_base.py`](../tests/_hardware_check_base.py) | `HardwareCheckBase`: tempdir lifecycle + run-flag / gate skip mechanism for the `*_hw.py` files |
| [`_hardware_check_helpers.py`](../tests/_hardware_check_helpers.py) | Fake scope payloads, parsing, and config-restriction helpers used by `test_scope_hw.py` / `test_motion_hw.py` (and unit-tested by `test_daq_check_helpers.py`) |
| [`_hdf5_assertions.py`](../tests/_hdf5_assertions.py) | Shared HDF5 structural assertions used by `test_daq_spool.py` |
| [`_lapd_daq_fixtures.py`](../tests/_lapd_daq_fixtures.py) | Shared `CONFIG_TEXT` / `CAMERA_CONFIG_TEXT` INI fixtures used by `test_daq_core.py` |

## Recommended run sequence

### On a development machine (no hardware)

Run the unit files — all of them run anywhere and finish in seconds:

```cmd
python -m pytest tests/ -k "not _hw" -q
```

The `*_hw.py` files self-skip here; to confirm the gating is wired up you can run
the whole suite and see them reported as skipped:

```cmd
python -m pytest tests/ -q
```

### On the hardware PC

1. Run the dev-machine unit sequence first to catch regressions before touching
   instruments.
2. Run the routine **spooled + parallel DAQ plane** — the primary integration
   check.
3. Run the gated hardware diagnostics one at a time, enabling only the flag you
   need (each `*_hw.py` has both a `RUN_*` flag and a destructive-action gate;
   both must be `True` before anything moves or arms).

```cmd
python -m pytest tests/test_scope_hw.py::ScopeHardwareCheck -v -s
```

The `-s` flag matters for `*_hw.py` files — they print live status that pytest's
default capture would swallow.

:::{warning}
Every `*_hw.py` test has both a `RUN_*_CHECK` flag and an `ALLOW_*` (or
`BMOTION_ALLOW_MOVE`) destructive-action gate. Both must be `True` before any
test will arm an instrument or command motion. Intentional belt-and-braces — keep
both gates.
:::

## Module reference

### `test_bmotion_config.py`

**Subject:** the `[bmotion]` section parser at
[`acquisition/bmotion_config.py`](../acquisition/bmotion_config.py).
**Needs hardware:** no (stub `RunManager` with a `.mgs` dict).
Covers the `motion_groups` selector, `direction` (bare + per-key mapping),
`execution_order`, and all the parser error paths (invalid keys/values, unknown
keys, empty manager).

### `test_bmotion_loops.py`

**Subject:** the acquisition loop edge/error paths in
[`acquisition/bmotion.py`](../acquisition/bmotion.py).
**Needs hardware:** no (stubs from [`_bmotion_stubs.py`](../tests/_bmotion_stubs.py)).
Covers `configure_bmotion_hdf5_group` validation (non-grid / 3-D / bad axis
labels), `move_to_index` out-of-range skip, `_take_shots_at_position`
skip-on-error, the spool sink, and terminal-motor-failure skip-and-continue.
The happy-path iteration order / active-group-only HDF5 rows are intentionally
**not** unit-tested here — they're covered by the routine spooled DAQ plane run.

### `test_bmotion_recovery_hw.py`

**Subject:** `acquisition.motor_recovery` driven directly against real motors.
**Needs hardware:** **yes** (motors). Four checks (flags at top of file): LONG
MOTION (a slow move must finish, not time out), ENCODER (EP vs IP agreement
around a move, incl. negative/zero-crossing), FAILURE (an unreachable index
raises `MotorError`, then a good move still succeeds), and SET-ZERO (destructive,
off by default — zero the group and confirm the encoder reads back ~0).

### `test_daq_core.py`

**Subject:** core `lapd_daq` units — `load_run_config` parsing, grid detection,
`AcquisitionRun.build_shot_plan()`, acquisition import hygiene, the Phantom
adapter cine naming, the `Data_Run_45deg` sunset, and **old lab_scopes/pydaq
HDF5 reader back-compat** (self-skips when `lab_scopes` / fixtures absent).
**Needs hardware:** no.

### `test_daq_parallel.py`

**Subject:** parallel multi-scope arm/read in
[`acquisition/scope_runner.py`](../acquisition/scope_runner.py) —
`acquire_shot_parallel`, `acquire_shot_dispatch`, parallel `arm_scopes_for_trigger`.
**Needs hardware:** no (fake scopes). Covers result-equivalence with sequential,
read/arm overlap, scope-error skip, KeyboardInterrupt abort, and dispatch routing.
:::{note}
A couple of the timing-overlap assertions are sensitive to load and can flake
under a busy full-suite run; they pass reliably when the file is run on its own.
:::

### `test_daq_spool.py`

**Subject:** the acquire→spool→offload→HDF5 pipeline.
**Needs hardware:** no. Covers the spool round-trip (1-D and 2-D), `.done`
ordering, offload fill + read-back verify + delete, resume / partial-run, and
corrupt-record handling — the offload edge cases a happy plane run won't trigger.

### `test_daq_check_helpers.py`

**Subject:** pure unit tests for the helpers in
[`_hardware_check_helpers.py`](../tests/_hardware_check_helpers.py) (`parse_move_to`,
`target_coordinates`, `fake_scope_payload`, `restrict_scope_config`).
**Needs hardware:** no. (Formerly mis-named `test_hardware_daq_check.py`.)

### `test_motor_recovery.py`

**Subject:** `acquisition.motor_recovery.move_with_recovery` and helpers, with
purpose-built fakes injecting each failure mode.
**Needs hardware:** no. Covers the recovery ladder (slow-but-progressing move
left alone, genuine stall escalates/raises, transient miss recovers, connection
loss recovers, resettable alarm recovers, fresh-read verification), the
encoder-vs-step mismatch + negative-position reads, set-zero confirmation, and
not-reached-position recording. The highest-value unit file — none of this is hit
by a *successful* run.

### `test_scope_hw.py`

**Subject:** per-instrument LeCroy scope diagnostics (inherits `HardwareCheckBase`).
**Needs hardware:** **yes** (scope). `ScopeHardwareCheck` (connect, read time
array, optionally arm+write one shot) and `DataRunScopeHardware` (end-to-end
`Data_Run.py` scope path). Flags default to `False`.

### `test_motion_hw.py`

**Subject:** per-instrument motion-controller diagnostics (inherits
`HardwareCheckBase`).
**Needs hardware:** **yes** (motors). `MotionHardwareCheck` (read position,
optionally move to `MOTION_TARGET`) and `DataRunMotionHardware` (end-to-end
`Data_Run.py` motion path with fake delayed scope). `MOTION_ALLOW_MOVE` gates both.

### `test_camera_hw.py`

**Subject:** per-instrument Phantom camera diagnostic (inherits `HardwareCheckBase`).
**Needs hardware:** **yes** (camera). `CameraHardwareCheck` — configure the
camera, optionally wait for trigger + save `.cine` (gated by `CAMERA_ALLOW_RECORD`).
