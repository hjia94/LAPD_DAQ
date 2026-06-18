import bapsf_motion as bmotion
import h5py
import json
import numpy as np
import time
import traceback
import warnings
import xarray as xr

from collections import deque
from typing import Dict

from tqdm import tqdm

from . import config as config_module
from . import hdf5_writer
from .bmotion_config import resolve_bmotion_selection
from .config import load_experiment_config
from .scope_runner import MultiScopeAcquisition, single_shot_acquisition


_POSITION_DTYPE = [('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4')]

# Injectable clock seams. Tests patch THESE module attributes
# (bmotion._sleep / bmotion._now) to make motion-wait polls instant and timing
# deterministic; patching the stdlib ``time`` module's functions would leak
# fake time into every other module in the process.
_sleep = time.sleep
_now = time.time

# Names of motion groups for which we've already warned that the encoder position
# was unavailable and we fell back to IP. Keeps read_bmotion_positions from
# emitting that warning on every shot of a long scan (warn once per group).
_ENCODER_FALLBACK_WARNED = set()


class _LineTimeEstimator:
    """Line-based remaining-time estimate for a raster bmotion run.

    Within a line (fixed y, sweeping x) the moves are short hops; the
    expensive part is the long traverse when y changes to start the next
    line. Per-shot timing therefore swings depending on whether a shot
    followed a within-line hop or a line-start traverse, which makes any
    per-shot extrapolation jump around. tqdm already shows that per-shot
    rate on the bar, so we leave it alone.

    For the *total* estimate we instead measure how long each completed line
    takes end to end (the line-start traverse plus all of its positions and
    shots) and project: remaining_seconds = avg_line_time * remaining_lines,
    plus a partial-line correction for the line in progress. This is steadier
    and matches how the run actually spends time.
    """

    def __init__(self, total_lines, window=5):
        self.total_lines = max(int(total_lines), 0)
        self._line_times = deque(maxlen=window)
        self.lines_done = 0
        self._current_line_start = None

    def start_line(self):
        """Call when beginning a new line (just before the line-start move)."""
        self._current_line_start = _now()

    def finish_line(self):
        """Call after the last shot of a line has been acquired."""
        if self._current_line_start is not None:
            self._line_times.append(_now() - self._current_line_start)
            self.lines_done += 1
            self._current_line_start = None

    def remaining_seconds(self):
        """Best estimate of seconds left, or None until one line is complete."""
        if not self._line_times:
            return None
        avg_line = sum(self._line_times) / len(self._line_times)
        remaining_lines = max(self.total_lines - self.lines_done, 0)
        total = avg_line * remaining_lines
        # Credit time already spent on the line currently in progress so the
        # estimate counts down smoothly within a line instead of only stepping
        # at line boundaries.
        if self._current_line_start is not None and remaining_lines > 0:
            spent = _now() - self._current_line_start
            total -= min(spent, avg_line)
        return max(total, 0.0)

    def postfix(self):
        """Short string for tqdm.set_postfix_str, or '' if not enough data."""
        secs = self.remaining_seconds()
        if secs is None:
            return ""
        return f"~{secs / 3600:.2f}h left ({self.lines_done}/{self.total_lines} lines)"


def _build_setup_array(mg):
    """Validate `mg`'s motion list and return (setup_array, xpos, ypos).

    Mirrors the planned-positions layout from PositionManager's XY path:
    structured array of (shot_num,x,y) with `xpos`/`ypos` axis vectors.
    Rejects motion groups bmotion can't honestly represent in that layout
    (non-2D, non-(x,y) axes, non-rectangular grids).
    """
    name = mg.config['name']
    ml = mg.mb.motion_list
    arr = ml.values
    N, M = arr.shape
    axis_labels = tuple(str(s).lower() for s in ml.coords['space'].values)
    if axis_labels != ('x', 'y'):
        raise RuntimeError(
            f"bmotion expects axis labels ('x','y'); motion group '{name}' has {axis_labels}"
        )
    xpos = np.unique(arr[:, 0])
    ypos = np.unique(arr[:, 1])

    setup = np.zeros(N, dtype=_POSITION_DTYPE)
    setup['shot_num'] = np.arange(1, N + 1)
    setup['x'] = arr[:, 0]
    setup['y'] = arr[:, 1]
    return setup, xpos, ypos


def collect_bmotion_position_setup(run_manager, selected_mg_keys):
    """Validate selected motion groups and gather their planned-position layout.

    Returns a list of ``(mg_key, mg_name, setup_array, xpos, ypos)`` tuples,
    pre-validated so callers can abort before creating any HDF5 datasets if a
    motion group is unsupported. Split out of ``configure_bmotion_hdf5_group``
    so the spool/offload path can capture this layout (plain numpy arrays) and
    rebuild the HDF5 positions groups later without a live ``RunManager``.
    """
    prepared = []
    for mg_key in selected_mg_keys:
        mg = run_manager.mgs[mg_key]
        setup, xpos, ypos = _build_setup_array(mg)
        prepared.append((mg_key, mg.config['name'], setup, xpos, ypos))
    return prepared


def write_bmotion_position_groups(
    hdf5_path: str,
    total_shots: int,
    toml_text: str,
    selection_blob: str,
    prepared,
):
    """Create the Control/Positions groups and bmotion Configuration datasets.

    Takes plain values (``toml_text``, ``selection_blob`` JSON string, and the
    ``prepared`` list from :func:`collect_bmotion_position_setup`) so it is
    usable both in-process (with a live RunManager) and from the offload process
    (reconstructing from spooled metadata).
    """
    with h5py.File(hdf5_path, 'a') as f:
        ctl_grp = f.require_group('Control')
        pos_grp = ctl_grp.require_group('Positions')

        config_grp = f.require_group('Configuration')
        if 'bmotion_config' not in config_grp:
            config_grp.create_dataset('bmotion_config', data=np.bytes_(toml_text))
        if 'bmotion_selection' not in config_grp:
            config_grp.create_dataset('bmotion_selection', data=np.bytes_(selection_blob))

        for mg_key, mg_name, setup, xpos, ypos in prepared:
            if mg_name in pos_grp:
                continue
            mg_group = pos_grp.create_group(mg_name)
            mg_group.attrs['name'] = mg_name
            mg_group.attrs['key'] = str(mg_key)

            setup_ds = mg_group.create_dataset(
                'positions_setup_array', data=setup, dtype=_POSITION_DTYPE,
            )
            setup_ds.attrs['xpos'] = xpos
            setup_ds.attrs['ypos'] = ypos

            mg_group.create_dataset(
                'positions_array', shape=(total_shots,), dtype=_POSITION_DTYPE,
            )


def build_bmotion_selection_blob(selected_mg_keys, ml_order, execution_order):
    """JSON string recording which motion-group subset/direction a run used."""
    return json.dumps({
        "mg_keys": [str(k) for k in selected_mg_keys],
        "direction": {str(k): v for k, v in (ml_order or {}).items()},
        "execution_order": execution_order,
    })


def configure_bmotion_hdf5_group(
    hdf5_path: str,
    total_shots: int,
    n_motion_groups: int,
    toml_path: str,
    run_manager: bmotion.actors.RunManager,
    selected_mg_keys,
    ml_order: Dict = None,
    execution_order: str = "interleaved",
):
    # Validate every selected motion group up front so we abort before
    # creating any HDF5 datasets if one of them is unsupported.
    prepared = collect_bmotion_position_setup(run_manager, selected_mg_keys)

    with open(toml_path, 'r') as toml_file:
        toml_text = toml_file.read()
    selection_blob = build_bmotion_selection_blob(
        selected_mg_keys, ml_order, execution_order
    )

    write_bmotion_position_groups(
        hdf5_path, total_shots, toml_text, selection_blob, prepared,
    )


def get_motion_list_size(rm: bmotion.actors.RunManager, mg_key) -> int:
    mg = rm.mgs[mg_key]
    if not isinstance(mg.mb.motion_list, xr.DataArray):
        raise RuntimeError(
            f"Selected motion group '{mg.config['name']}' motion list is invalid."
        )
    if mg.mb.motion_list.size == 0:
        raise RuntimeError(
            f"Selected motion group '{mg.config['name']}' has an empty motion list"
        )
    return int(mg.mb.motion_list.shape[0])


def get_max_motion_list_size(rm: bmotion.actors.RunManager, mg_keys) -> int:

    sizes = []
    for key in mg_keys:
        mg = rm.mgs[key]

        if not isinstance(mg.mb.motion_list, xr.DataArray):
            raise RuntimeError(
                f"Selected motion group '{mg.config['name']}' motion "
                f"list is invalid."
            )

        if mg.mb.motion_list.size == 0:
            raise RuntimeError(
                f"Selected motion group '{mg.config['name']}' has an "
                f"empty motion list"
            )

        sizes.append(mg.mb.motion_list.shape[0])

    return int(np.max(sizes))


def move_to_index(
    index: int,
    rm: bmotion.actors.RunManager,
    ml_order_dict: Dict,
) -> None:

    for mg_key, order in ml_order_dict.items():
        mg = rm.mgs[mg_key]
        ml_size = int(mg.mb.motion_list.shape[0])

        # Use a local variable to avoid modifying the passed index
        motion_index = index
        if order == "backward":
            motion_index = ml_size - index - 1

        if motion_index not in range(ml_size):
            warnings.warn(
                f"Motion list index {motion_index} is out of range for motion "
                f"group '{mg.config['name']}'.  NO MOTION DONE."
            )
            continue

        # Use move_ml to move to the specified index in the motion list
        mg.move_ml(motion_index)

    # wait for motion to stop
    _sleep(.5)
    while rm.is_moving:
        _sleep(.5)

    # disable all motors
    for mg in rm.mgs.values():
        mg.drive.send_command('disable')


def read_bmotion_positions(rm, mg_keys):
    """Read each selected motion group's current position into a coords dict.

    The recorded position is the **encoder** (EP) position pushed through the
    motion group's transform -- the real physical probe position -- rather than
    ``mg.position`` (which sources IP, the *calculated trajectory* position the
    Applied Motion manual notes "is not always equal to actual position"). If the
    encoder can't be read for a group (no encoder / missing constants / read
    miss), it falls back to ``mg.position`` (IP) and warns, so a drive without an
    encoder still records something rather than crashing.

    Returns ``{mg_name: (x, y)}`` so callers can either write it straight to
    HDF5 (in-process) or bundle it into a spooled shot payload (parallel mode).
    """
    from .motor_recovery import encoder_motion_space_position, _refresh_status

    coords = {}
    for key in mg_keys:
        mg = rm.mgs[key]
        name = mg.config['name']
        # Force a fresh, idle-state status read so the encoder isn't NACK'd and
        # the IP fallback isn't a stale heartbeat value.
        _refresh_status(mg)
        pos = encoder_motion_space_position(mg)
        if pos is None:
            pos = tuple(mg.position.value)
            # Warn once per group per run, not once per shot, so a drive without
            # an encoder doesn't flood the log on a long scan.
            if name not in _ENCODER_FALLBACK_WARNED:
                _ENCODER_FALLBACK_WARNED.add(name)
                warnings.warn(
                    f"motion group '{name}': encoder position unavailable; "
                    f"recording IP (calculated trajectory) position instead, "
                    f"which may not match the physical probe. (Warned once; "
                    f"applies to all shots for this group.)")
        coords[name] = (pos[0], pos[1])
    return coords


def record_bmotion_positions(
    hdf5_path: str,
    shotnum: int,
    rm: bmotion.actors.RunManager,
    mg_keys,
) -> None:

    coords = read_bmotion_positions(rm, mg_keys)
    with h5py.File(hdf5_path, 'a') as f:
        for mg_name, (x, y) in coords.items():
            # Access the positions_array for this specific motion group directly under Control/Positions
            dataset = f[f"Control/Positions/{mg_name}/positions_array"]

            # Record position for this shot using structured array format (shot_num is 1-based, array is 0-based)
            dataset[shotnum - 1] = (shotnum, x, y)


class _Hdf5ShotSink:
    """Per-shot sink that writes straight into the final HDF5 file.

    Production always spools (``_SpoolShotSink``); this direct-write sink is the
    default fallback for the loop helpers and is what the unit tests drive.
    """

    def __init__(self, msa, active_scopes, hdf5_path, run_manager):
        self.msa = msa
        self.active_scopes = active_scopes
        self.hdf5_path = hdf5_path
        self.run_manager = run_manager

    def take_shot(self, shot_num, record_keys):
        # arm + acquire + write scope data to HDF5 (as single_shot_acquisition).
        single_shot_acquisition(self.msa, self.active_scopes, shot_num,
                                verbose=False)
        record_bmotion_positions(
            hdf5_path=self.hdf5_path, shotnum=shot_num,
            rm=self.run_manager, mg_keys=record_keys,
        )

    def mark_skipped(self, shot_num, reason, record_keys):
        with h5py.File(self.hdf5_path, 'a') as f:
            for scope_name in self.msa.scope_ips:
                scope_group = f[scope_name]
                if f'shot_{shot_num}' not in scope_group:
                    shot_group = scope_group.create_group(f'shot_{shot_num}')
                    shot_group.attrs['skipped'] = True
                    shot_group.attrs['skip_reason'] = str(reason)
                    shot_group.attrs['acquisition_time'] = time.ctime()
        record_bmotion_positions(
            hdf5_path=self.hdf5_path, shotnum=shot_num,
            rm=self.run_manager, mg_keys=record_keys,
        )


class _SpoolShotSink:
    """Per-shot sink that writes to the fast-disk spool instead of the HDF5.

    The probe is already at position when this runs; per the parallel design
    the order per shot is arm -> acquire -> read positions -> write bin+done.
    A separate offload process turns the spool into the HDF5 file.
    """

    def __init__(self, msa, active_scopes, spool_dir, run_manager,
                 pause_seconds=None, max_retries=None):
        from spooling import spool_format

        self.msa = msa
        self.active_scopes = active_scopes
        self.spool_dir = spool_dir
        self.run_manager = run_manager
        self.pause_seconds = (spool_format.DISK_FULL_PAUSE_SECONDS
                              if pause_seconds is None else pause_seconds)
        self.max_retries = (spool_format.DISK_FULL_MAX_RETRIES
                            if max_retries is None else max_retries)

    def take_shot(self, shot_num, record_keys):
        from spooling import spool_format
        from . import spool_adapter

        self.msa.arm_scopes_for_trigger(self.active_scopes, verbose=False)
        all_data = self.msa.acquire_shot_dispatch(self.active_scopes, shot_num, verbose=False)
        missing = self.msa.last_missing_scopes
        if not all_data:
            # Every scope failed to arm/read -> a fully-missing shot. Raise so the
            # caller records it as skipped and the circuit-breaker can count it;
            # the run continues rather than aborting on one bad shot.
            raise RuntimeError(
                _format_missing_reason(missing)
                or f"No valid data acquired at shot {shot_num}")
        coords = read_bmotion_positions(self.run_manager, record_keys)
        payload = spool_adapter.all_data_to_payload(
            all_data, shot_num, coords, missing_scopes=missing)
        spool_format.write_shot_with_disk_full_retry(
            self.spool_dir, payload, parallel=self.msa.parallel_spool_write,
            pause_seconds=self.pause_seconds, max_retries=self.max_retries,
            warn=tqdm.write)

    def mark_skipped(self, shot_num, reason, record_keys):
        from spooling import spool_format
        from . import spool_adapter

        coords = read_bmotion_positions(self.run_manager, record_keys)
        payload = spool_adapter.skipped_payload(shot_num, reason, coords)
        spool_format.write_shot(self.spool_dir, payload)


def _format_missing_reason(missing):
    """Human-readable summary of a {scope: reason} missing map, or ''."""
    if not missing:
        return ""
    return "all scopes missing: " + "; ".join(
        f"{name} ({reason})" for name, reason in missing.items())


def _safe_mark_skipped(sink, shot_num, reason, record_keys):
    """Record a skipped shot, but never let skip-recording itself abort the run.

    ``mark_skipped`` reads the motor position (and, for the HDF5 sink, reopens
    the file); if the underlying cause is a broken motor link, that read can
    raise. A failure here must degrade to a logged warning -- losing the skip
    record for one shot -- rather than killing a long run.
    """
    try:
        sink.mark_skipped(shot_num, reason, record_keys)
    except Exception as e:  # noqa: BLE001
        tqdm.write(f"Warning: could not record skip for shot {shot_num}: {e}")


class _RunAborted(RuntimeError):
    """The circuit-breaker tripped: too many fully-skipped shots in a row.

    Carries the abort reason so the driver can finalize the HDF5 with
    ``terminated_early=True`` instead of leaving the run looking complete.
    """


def _note_shot_outcome(run_state, full_skip):
    """Update the consecutive-full-skip counter and trip the breaker if exceeded.

    A *fully-skipped* shot (no scope produced data) increments the counter; any
    shot with data resets it. When the counter reaches
    ``run_state["max_consecutive_skips"]`` (>0) a :class:`_RunAborted` is raised
    so the run stops cleanly rather than spooling thousands of empty shots from a
    dead master/trigger. A None/absent run_state (legacy callers) is a no-op.
    """
    if run_state is None:
        return
    if not full_skip:
        run_state["consecutive_skips"] = 0
        return
    run_state["consecutive_skips"] = run_state.get("consecutive_skips", 0) + 1
    count = run_state["consecutive_skips"]
    limit = run_state.get("max_consecutive_skips", 0)
    if config_module.consecutive_skip_breaker_tripped(count, limit):
        raise _RunAborted(
            config_module.consecutive_skip_abort_message(count, limit))


def _take_shots_at_position(
    msa,
    active_scopes,
    hdf5_path: str,
    run_manager: bmotion.actors.RunManager,
    record_keys,
    shot_num: int,
    nshots: int,
    pbar,
    estimator=None,
    sink=None,
    run_state=None,
):
    """Acquire nshots at the current position, recording positions only for
    the motion groups in record_keys. Advances `pbar` once per shot and
    returns the next shot_num.

    The actual per-shot write is delegated to `sink`; when None, a legacy
    HDF5 sink is used (writes straight to `hdf5_path`). A shot whose every scope
    failed (raised) is counted by the circuit-breaker via ``run_state``; a shot
    with any data resets it. The breaker raising :class:`_RunAborted` unwinds to
    the driver, which finalizes the run early."""
    if sink is None:
        sink = _Hdf5ShotSink(msa, active_scopes, hdf5_path, run_manager)

    for n in range(nshots):
        full_skip = False
        try:
            sink.take_shot(shot_num, record_keys)
        except (ValueError, RuntimeError) as e:
            tqdm.write(f'Skipping shot {shot_num} - {str(e)}')
            _safe_mark_skipped(sink, shot_num, str(e), record_keys)
            full_skip = True
        except Exception as e:
            tqdm.write(f'Motion failed for shot {shot_num} - {str(e)}')
            _safe_mark_skipped(sink, shot_num, f"Motion failed: {str(e)}", record_keys)
            full_skip = True
        finally:
            shot_num += 1
            # tqdm tracks per-shot rate on the bar itself; we only overlay the
            # line-based total-time estimate as a postfix, refreshed each shot
            # so it counts down smoothly within a line.
            if estimator is not None:
                postfix = estimator.postfix()
                if postfix:
                    pbar.set_postfix_str(postfix)
            pbar.update(1)
            # Track the next-shot number so the driver can report the count
            # actually emitted even if the breaker aborts the run below.
            if run_state is not None:
                run_state["last_shot_num"] = shot_num
        # Outside the finally so a breaker trip propagates (not masked by it).
        _note_shot_outcome(run_state, full_skip)
    return shot_num


def _format_position_line(rm, mg_keys, motion_index, total_positions):
    """Build one compact status line for a position, e.g.

        -> Hermes x=3.00 y=-5.00 | pos 12/400 | line 3/20 | shots 56-60/2000

    Leads with the motion group(s) that just moved (named, e.g. 'Hermes') so
    it's clear which probe is positioned, followed by per-position progress,
    which raster line it belongs to, and the shot-number range this position
    will fill. Designed for tqdm.write so it sits cleanly above the bar.

    The printed ``x=``/``y=`` are the *measured* motion-space position (to 2
    decimals), not the requested target. A direct status re-query is forced first
    so the value reflects where the probe actually is, not a stale heartbeat
    cache (which only refreshes every ~0.2-1.5 s and can lag a just-finished move).
    """
    # Refresh so the reported position is current, not the cached pre-move value.
    try:
        from .motor_recovery import _refresh_status
        for k in mg_keys:
            _refresh_status(rm.mgs[k])
    except Exception:  # noqa: BLE001 - reporting must never break the run
        pass

    def _xy(mg):
        """Encoder (real) motion-space x/y, falling back to IP if unavailable.

        Matches what read_bmotion_positions records, so the printed line reflects
        the value that goes into the HDF5 file."""
        try:
            from .motor_recovery import encoder_motion_space_position
            pos = encoder_motion_space_position(mg)
            if pos is not None:
                return pos[0], pos[1]
        except Exception:  # noqa: BLE001 - reporting must never break the run
            pass
        return mg.position[0], mg.position[1]

    movers = " ".join(
        "{} x={:.2f} y={:.2f}".format(rm.mgs[k].config['name'], *_xy(rm.mgs[k]))
        for k in mg_keys
    )
    parts = [
        f"-> {movers}",
        f"pos {motion_index + 1}/{total_positions}"
    ]
    return " | ".join(parts)


def _positions_per_line(rm, mg_key):
    """Number of positions in one raster line (fixed y, sweeping x).

    A line is the set of unique x values at a given y, so positions_per_line
    equals the count of distinct x coordinates in the motion list. Falls back
    to 1 (every position is its own line) if the layout can't be determined,
    which keeps the estimator conservative rather than wrong.
    """
    try:
        arr = rm.mgs[mg_key].mb.motion_list.values
        return max(int(np.unique(arr[:, 0]).size), 1)
    except Exception:
        return 1


def _do_move(run_manager, ml_order_dict, motion_index, move_opts, run_state):
    """Move to ``motion_index``. Returns ``"ok"`` or ``"skip"``.

    When ``move_opts`` is None (legacy in-process path), uses the original
    ``move_to_index`` unchanged and returns ``"ok"``. When provided (spooled
    path), uses ``move_with_recovery``; if recovery is exhausted
    (:class:`MotorError`) the position is **skipped** -- the failure is recorded
    in ``run_state`` and ``"skip"`` is returned so the caller can record/flag the
    bad position and continue the scan rather than aborting the whole run. (A
    motor that is merely slow is never treated as a failure -- recovery only
    raises when the motor is genuinely stuck after retries.)
    """
    if move_opts is None:
        move_to_index(index=motion_index, rm=run_manager, ml_order_dict=ml_order_dict)
        return "ok"
    from .motor_recovery import move_with_recovery, MotorError
    try:
        move_with_recovery(run_manager, ml_order_dict, motion_index,
                           log=tqdm.write, **move_opts)
        return "ok"
    except MotorError as e:
        tqdm.write(f"\n______Skipping position (motor failure)______\n{e}")
        if run_state is not None:
            run_state.setdefault("skipped_positions", []).append(
                {"motion_index": motion_index, "reason": str(e)})
        return "skip"


def _last_skip_reason(run_state, motion_index):
    """Reason for the position just skipped, from run_state.

    ``_do_move`` records each terminal motor failure in
    ``run_state["skipped_positions"]`` as ``{"motion_index", "reason"}``; the most
    recent entry is the position we are recording now. Falls back to a generic
    message if (defensively) no entry is present."""
    skipped = (run_state or {}).get("skipped_positions") or []
    if skipped:
        return skipped[-1]["reason"]
    return f"motor failed to reach motion index {motion_index}"


def _record_skipped_shots(msa, active_scopes, hdf5_path, run_manager, record_keys,
                          shot_num, nshots, reason, sink):
    """Record every shot of a skipped position into the HDF5 as skipped.

    When a motor move fails (recovery exhausted) the position's shots are not
    acquired, but they must still be written to the HDF5 with ``skipped=True`` and
    the failure ``skip_reason`` -- otherwise the file is left with silent
    empty/zero rows and no record that the probe never reached that position.
    Falls back to the legacy in-process HDF5 sink when no sink is provided."""
    if sink is None:
        sink = _Hdf5ShotSink(msa, active_scopes, hdf5_path, run_manager)
    for n in range(nshots):
        _safe_mark_skipped(sink, shot_num + n, reason, record_keys)


def _run_interleaved(msa, active_scopes, hdf5_path, run_manager, ml_order, nshots,
                     total_shots, sink=None, move_opts=None, run_state=None):
    max_ml_size = get_max_motion_list_size(run_manager, list(ml_order))
    shot_num = 1
    record_keys = list(ml_order.keys())

    # Lines are defined by the first selected group's grid; interleaved runs
    # share a common rectangular layout across groups.
    per_line = _positions_per_line(run_manager, record_keys[0])
    total_lines = int(np.ceil(max_ml_size / per_line))
    estimator = _LineTimeEstimator(total_lines)

    with tqdm(total=total_shots, desc="Shots", unit="shot", dynamic_ncols=True) as pbar:
        line_idx = 0
        for motion_index in range(max_ml_size):
            # A new line begins whenever we cross a positions-per-line boundary.
            if motion_index % per_line == 0:
                estimator.finish_line()   # close the previous line (no-op on first)
                estimator.start_line()
                line_idx += 1

            if _do_move(run_manager, ml_order, motion_index, move_opts, run_state) == "skip":
                # Recovery exhausted for this position -> record its shots as
                # skipped (with the not-reached reason) in the HDF5, then move on,
                # keeping shot numbering / progress bar consistent.
                reason = _last_skip_reason(run_state, motion_index)
                _record_skipped_shots(msa, active_scopes, hdf5_path, run_manager,
                                      record_keys, shot_num, nshots, reason, sink)
                shot_num += nshots
                pbar.update(nshots)
                continue
            tqdm.write(_format_position_line(run_manager, list(ml_order), motion_index, max_ml_size))

            shot_num = _take_shots_at_position(
                msa, active_scopes, hdf5_path, run_manager, record_keys, shot_num, nshots, pbar,
                estimator=estimator, sink=sink, run_state=run_state,
            )
        estimator.finish_line()  # close the final line
    return shot_num


def _run_sequential(msa, active_scopes, hdf5_path, run_manager, ml_order, nshots,
                    total_shots, sink=None, move_opts=None, run_state=None):
    shot_num = 1

    # Total lines summed across all groups (each group is a self-contained
    # raster run before the next group starts).
    total_lines = 0
    for mg_key in ml_order:
        size = get_motion_list_size(run_manager, mg_key)
        total_lines += int(np.ceil(size / _positions_per_line(run_manager, mg_key)))
    estimator = _LineTimeEstimator(total_lines)

    with tqdm(total=total_shots, desc="Shots", unit="shot", dynamic_ncols=True) as pbar:
        line_idx = 0
        for mg_key, direction in ml_order.items():
            mg = run_manager.mgs[mg_key]
            ml_size = get_motion_list_size(run_manager, mg_key)
            per_line = _positions_per_line(run_manager, mg_key)
            tqdm.write(f"=== Motion group '{mg.config['name']}' "
                       f"(key={mg_key}, {ml_size} positions, {direction}) ===")

            single_group_order = {mg_key: direction}
            for motion_index in range(ml_size):
                if motion_index % per_line == 0:
                    estimator.finish_line()   # close the previous line (no-op on first)
                    estimator.start_line()
                    line_idx += 1

                if _do_move(run_manager, single_group_order, motion_index,
                            move_opts, run_state) == "skip":
                    # Recovery exhausted for this position -> record its shots as
                    # skipped (with the not-reached reason) in the HDF5, then
                    # continue the scan, keeping shot numbering consistent.
                    reason = _last_skip_reason(run_state, motion_index)
                    _record_skipped_shots(msa, active_scopes, hdf5_path, run_manager,
                                          [mg_key], shot_num, nshots, reason, sink)
                    shot_num += nshots
                    pbar.update(nshots)
                    continue
                tqdm.write(_format_position_line(run_manager, [mg_key], motion_index, ml_size))

                shot_num = _take_shots_at_position(
                    msa, active_scopes, hdf5_path, run_manager, [mg_key], shot_num, nshots, pbar,
                    estimator=estimator, sink=sink, run_state=run_state,
                )
        estimator.finish_line()  # close the final line
    return shot_num


def _prepare_bmotion_run(toml_path, config_path):
    """Shared setup for both the in-process and spooled bmotion runs.

    Loads config, starts the RunManager, resolves the motion-group selection,
    and computes total shot counts. Returns everything the run loops need.
    """
    config, raw_config_text = load_experiment_config(config_path)
    nshots = config.getint('nshots', 'num_duplicate_shots', fallback=1)

    print("Loading TOML configuration...", end='')
    run_manager = bmotion.actors.RunManager(toml_path, auto_run=True)
    print("done")

    try:
        sel = resolve_bmotion_selection(config, run_manager)
    except (ValueError, RuntimeError):
        run_manager.terminate()
        raise
    selection = sel.mg_keys
    ml_order = sel.direction
    execution_order = sel.execution_order
    print(f"Selected motion groups: {selection}")
    print(f"Directions: {ml_order}")
    print(f"Execution order: {execution_order}")

    if execution_order == "sequential":
        per_group_sizes = [get_motion_list_size(run_manager, k) for k in ml_order]
        total_positions = sum(per_group_sizes)
        print(f"Per-group motion list sizes: {dict(zip(list(ml_order), per_group_sizes))}")
        print(f"Total positions across all groups: {total_positions}")
    else:
        max_ml_size = get_max_motion_list_size(run_manager, list(ml_order))
        total_positions = max_ml_size
        print(f"Maximum motion list size is {max_ml_size}")

    print(f"Number of shots per position: {nshots}")
    total_shots = total_positions * nshots
    print(f"Total shots: {total_shots}")

    return {
        "config": config,
        "raw_config_text": raw_config_text,
        "nshots": nshots,
        "run_manager": run_manager,
        "ml_order": ml_order,
        "execution_order": execution_order,
        "total_shots": total_shots,
    }


def run_acquisition_bmotion_spooled(spool_dir, hdf5_path, toml_path, config_path,
                                    description_path=None):
    """Parallel-mode acquisition: build the HDF5 skeleton, spool each shot.

    The acquire process owns the *initial* HDF5 file: it creates ``hdf5_path``
    and writes everything known up front (experiment metadata, source code, raw
    config, scope metadata, time arrays, and the empty ``Control/Positions``
    skeleton) using the shared bmotion HDF5 writers, then
    closes the file and never reopens it. Only per-shot scope traces (+ their
    headers/positions) go to the fast-disk ``spool_dir``; a separate offload
    process fills those into the already-created HDF5.

    Per-shot order is unchanged: move + settle + disable (per position), then
    arm -> acquire -> read position -> write bin+done.

    Resume is not supported: every call is a fresh run from shot 1 (the entry
    script restarts an existing run by deleting its HDF5 and rotating its spool
    aside first).
    """
    from spooling import spool_format
    from . import spool_adapter

    print('Starting spooled acquisition at', time.ctime())

    ctx = _prepare_bmotion_run(toml_path, config_path)
    config = ctx["config"]
    raw_config_text = ctx["raw_config_text"]
    nshots = ctx["nshots"]
    run_manager = ctx["run_manager"]
    ml_order = ctx["ml_order"]
    execution_order = ctx["execution_order"]
    total_shots = ctx["total_shots"]

    if description_path is None:
        description_path = config_module.resolve_description_path_from_config(config_path)

    from .config import get_max_consecutive_skips
    last_shot_num = 0
    run_state = {
        "terminated_early": False,
        "abort_reason": None,
        "consecutive_skips": 0,
        "max_consecutive_skips": get_max_consecutive_skips(config),
    }
    with MultiScopeAcquisition(hdf5_path, config, raw_config_text,
                               description_path=description_path) as msa:
        try:
            print("Initializing HDF5 file...", end='')
            msa.initialize_hdf5_base()
            print("done")

            print("\nStarting initial acquisition...")
            active_scopes = msa.initialize_scopes()
            if msa.scope_ips and not active_scopes:
                raise RuntimeError(
                    "No valid data found from any scope. Aborting acquisition."
                )

            configure_bmotion_hdf5_group(
                hdf5_path, total_shots, len(ml_order), toml_path, run_manager,
                list(ml_order.keys()), ml_order=ml_order,
                execution_order=execution_order,
            )

            # Slim run-info: the offload only needs which adapter to use and the
            # exact file to fill (computed once by the caller). Everything else
            # -- including the per-channel descriptions, which live as scope-group
            # attributes -- is already written into the HDF5 above.
            spool_format.write_run_metadata(spool_dir, {
                "writer": spool_adapter.WRITER_TAG,
                "hdf5_path": hdf5_path,
                "config_scope_names": list(active_scopes.keys()),
                "description_path": description_path,
                "total_shots": total_shots,
            })
            print(f"Wrote run metadata to spool: {spool_dir}")

            from .config import get_disk_full_pause_opts, get_motion_recovery_opts
            pause_seconds, max_retries = get_disk_full_pause_opts(config)
            sink = _SpoolShotSink(msa, active_scopes, spool_dir, run_manager,
                                  pause_seconds=pause_seconds,
                                  max_retries=max_retries)
            move_opts = get_motion_recovery_opts(config)

            if execution_order == "sequential":
                last_shot_num = _run_sequential(
                    msa, active_scopes, spool_dir, run_manager,
                    ml_order, nshots, total_shots, sink=sink,
                    move_opts=move_opts, run_state=run_state,
                )
            else:
                last_shot_num = _run_interleaved(
                    msa, active_scopes, spool_dir, run_manager,
                    ml_order, nshots, total_shots, sink=sink,
                    move_opts=move_opts, run_state=run_state,
                )

        except _RunAborted as err:
            # Circuit-breaker tripped (dead master/trigger): stop cleanly,
            # finalize with the shots already spooled rather than aborting hard.
            print(f'\n______Run stopped: {err}______', '  at', time.ctime())
            run_state["terminated_early"] = True
            run_state["abort_reason"] = str(err)
            last_shot_num = run_state.get("last_shot_num", last_shot_num)
        except KeyboardInterrupt as err:
            print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
            run_state["terminated_early"] = True
            run_state["abort_reason"] = "KeyboardInterrupt (Ctrl-C)"
            last_shot_num = run_state.get("last_shot_num", last_shot_num)
            raise RuntimeError() from err
        finally:
            run_manager.terminate()
            # `_run_*` return the next (unused) shot number, so the count
            # actually emitted is last_shot_num - 1. If the run aborted during
            # setup (before any shot), last_shot_num is still 0 -> report 0.
            final = max(last_shot_num - 1, 0)
            # Only signal completion if the run actually started (metadata
            # written). If setup failed before that, there is nothing for the
            # offload to finalize, and a RUN_COMPLETE with no metadata would
            # just leave the offload waiting on metadata that never comes.
            if spool_format.run_metadata_exists(spool_dir):
                spool_format.write_run_complete(
                    spool_dir, final,
                    terminated_early=run_state["terminated_early"],
                    abort_reason=run_state["abort_reason"],
                )
                if run_state["terminated_early"]:
                    print(f"Run terminated early ({run_state['abort_reason']}); "
                          f"{final} shots safely spooled. Wrote RUN_COMPLETE.")
                else:
                    print(f"Wrote RUN_COMPLETE (final_shot_num={final}) to spool")
            else:
                print("Run aborted before metadata was written; "
                      "no RUN_COMPLETE emitted.")
