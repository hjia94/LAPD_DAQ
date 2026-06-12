"""Scope coordination and run-loop orchestration.

Layered like this:

    helper functions (init_acquire_from_scope, acquire_from_scope,
        acquire_from_scope_sequence; the shared completion wait lives in
        lapd_daq.devices.lab_scopes.wait_for_fresh_acquisition)
        - talk to a single LeCroy_Scope instance

    MultiScopeAcquisition class
        - owns the live scope handles and the active config
        - hides the file-write details behind hdf5_writer

    single_shot_acquisition / handle_movement
        - one acquisition step, with/without motion. single_shot_acquisition is
          shared by the spooled bmotion loop and the hardware diagnostics;
          handle_movement is now only used by the motion hardware diagnostic.

    run_acquisition_spooled
        - top-level grid/stationary loop driven by experiment_config.ini that
          Data_Run.py calls into (spools each shot for Offload_Run.py to fill).
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from motion import PositionManager

from . import hdf5_writer
from . import config as config_module
from .config import load_experiment_config


# =============================================================================
# Single-scope helpers
# =============================================================================
def init_acquire_from_scope(scope, scope_name):
    """Initialize acquisition from a single scope and get initial data and time arrays
    Args:
        scope: LeCroy_Scope instance
        scope_name: Name of the scope
    Returns:
        tuple: (is_sequence, time_array, traces)
            - is_sequence: 0 for RealTime mode, 1 for sequence mode
            - time_array: Time array for the scope
            - traces: displayed-trace tuple captured here. The caller caches it
              and passes it to every per-shot acquire so the whole run reads
              one fixed channel set: a per-shot re-scan costs round-trips and
              silently drops any channel whose :TRACE? reply times out.
    """
    time_array = None
    is_sequence = None

    traces = scope.displayed_traces()

    for tr in traces:
        try:
            trace_bytes, header_bytes = scope.acquire_bytes(tr)
            hdr = scope.translate_header_bytes(header_bytes)

            if hdr.subarray_count < 2:  # Get number of segments
                is_sequence = 0  # in RealTime mode
            else:
                is_sequence = 1  # in sequence mode

            # Get time array from first valid trace
            if time_array is None:
                time_array = scope.time_array(tr)

            # Successfully got data from at least one trace, break out of loop
            break

        except Exception as e:
            print(f"Error initializing {tr} from {scope_name}: {e}")
            continue

    # Check if we got valid data
    if is_sequence is None or time_array is None:
        print(f"Warning: Could not get valid data from any trace on {scope_name}")
        return None, None, ()

    return is_sequence, time_array, traces


def acquire_from_scope(scope, scope_name, traces, ref_channel=None):
    """Acquire data from a single scope with optimized speed (int16/raw).

    Per-scope steps of one shot:
      4. check the scope is STOPped and a fresh sweep landed
         (wait_for_fresh_acquisition: TRIG_MODE STOP hint -> sweep-counter
         confirm on ``ref_channel`` cleared at arm time);
      5. if so, data exists -> read every displayed trace (one completed sweep
         means all channels of that sweep are ready, so no per-trace polling);
      otherwise raise so the shot is recorded as having no valid data.

    ``traces`` is the displayed-trace tuple captured at init time
    (see init_acquire_from_scope).
    """
    from lapd_daq.devices.lab_scopes import wait_for_fresh_acquisition

    data = {}
    headers = {}
    active_traces = []

    # Step 4: check STOP + fresh-sweep. Step 5: raises if no fresh data exists.
    wait_for_fresh_acquisition(scope, ref_channel)

    for tr in traces:
        data[tr], headers[tr] = scope.acquire(tr, raw=True)
        active_traces.append(tr)

    return active_traces, data, headers


def acquire_from_scope_sequence(scope, scope_name, traces, ref_channel=None):
    """Acquire sequence mode data from a single scope (int16/raw).

    ``traces`` as in acquire_from_scope.
    """
    from lapd_daq.devices.lab_scopes import wait_for_fresh_acquisition

    data = {}
    headers = {}
    active_traces = []

    wait_for_fresh_acquisition(scope, ref_channel)

    for tr in traces:
        segment_data, header = scope.acquire_sequence_data(tr)
        segment_data = [np.asarray(seg, dtype=np.int16) for seg in segment_data]
        data[tr] = np.stack(segment_data)
        headers[tr] = header
        active_traces.append(tr)

    return active_traces, data, headers


# =============================================================================
# Multi-scope coordinator
# =============================================================================
class MultiScopeAcquisition:
    """Owns scope connections, performs acquisition, and forwards to hdf5_writer."""

    def __init__(self, save_path, config, raw_config_text="", description_path=None):
        """
        Args:
            save_path: path to save HDF5 file
            config: ConfigParser object with experiment configuration
            raw_config_text: Raw text content of the configuration file (optional)
            description_path: full path to the run's ``description.txt`` (optional).
                The free-text experiment description is read from this file rather
                than from ``[experiment] description`` in the config, so the user
                can edit it before or during a run. ``None`` falls back to the
                placeholder description.
        """
        self.save_path = save_path
        self.scopes = {}
        self.figures = {}
        self.time_arrays = {}
        # Per-scope reference channel chosen at arm time (arm_single), polled by
        # the completion check so a fresh-sweep transition is read on the same
        # channel that was cleared. {scope_name: "C1"}.
        self._arm_channels = {}
        # Per-scope displayed-trace tuple captured once at initialize_scopes and
        # reused every shot. {scope_name: ("C1", "C2", ...)}.
        self._displayed_traces = {}
        self.config = config
        self.raw_config_text = raw_config_text
        self.description_path = description_path

        # Parallelism flags (all default true). Each gates an independent layer
        # so it can be A/B-tested or disabled alone:
        #   parallel_scope_arm   -- arm all scopes concurrently so they go live
        #       within a tight window (avoids a slow-arming scope missing a
        #       trigger edge at high rep rates -> scope desync).
        #   parallel_scope_read  -- overlap the per-scope waveform transfers.
        #   parallel_spool_write -- write each scope's spool files concurrently.
        # The read/arm layers are pure network-I/O overlap and unconditionally
        # good; spool-write parallelism helps on SSD/NVMe but can thrash a
        # single spinning disk, hence the separate switch.
        self.parallel_scope_arm = config.getboolean(
            'acquisition', 'parallel_scope_arm', fallback=True)
        self.parallel_scope_read = config.getboolean(
            'acquisition', 'parallel_scope_read', fallback=True)
        self.parallel_spool_write = config.getboolean(
            'acquisition', 'parallel_spool_write', fallback=True)

        # Seconds to wait for a slave scope to report (via its INR register) that
        # its trigger is armed and ready before the master is armed. The master's
        # trigger-out drives the slaves, so a slave that is not yet ready would
        # miss the master's edge and desync the run; we abort the shot instead.
        self.slave_ready_timeout = config.getfloat(
            'acquisition', 'slave_ready_timeout', fallback=5.0)

        # One-time cross-scope trigger-timestamp sanity warning (advisory only;
        # see _warn_if_trigger_timestamps_desynced). Flipped True after it fires
        # so it prints at most once per run.
        self._sync_warned = False

        if 'scope_ips' in config:
            self.scope_ips = dict(config.items('scope_ips'))
        else:
            self.scope_ips = {}

    def cleanup(self):
        """Close every open scope handle."""
        print("Cleaning up scope resources...")
        for name, scope in self.scopes.items():
            try:
                print(f"Closing scope {name}...")
                scope.__exit__(None, None, None)
            except Exception as e:
                print(f"Error closing scope {name}: {e}")
        self.scopes.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    # -- description lookups -------------------------------------------------
    def get_scope_description(self, scope_name):
        return self.config.get('scopes', scope_name,
                               fallback=f'Scope {scope_name} - No description available')

    def get_channel_description(self, channel_name):
        return self.config.get('channels', channel_name,
                               fallback=f'Channel {channel_name} - No description available')

    def get_experiment_description(self):
        """Return the run description, read from ``description.txt``.

        The description now lives in its own file (``description_path``) rather
        than in the config. If no path was supplied, fall back to the placeholder
        so the HDF5 ``description`` attribute is always present.
        """
        if not self.description_path:
            return config_module.DESCRIPTION_PLACEHOLDER
        return config_module.read_description_file(self.description_path)

    # -- HDF5 lifecycle (delegates to hdf5_writer) ---------------------------
    def initialize_hdf5_base(self):
        """Initialize HDF5 file structure for scopes and experiment metadata."""
        print(f"HDF5 compression: {hdf5_writer._COMPRESSION_LABEL}")
        hdf5_writer.write_experiment_metadata(
            self.save_path,
            description=self.get_experiment_description(),
            source_code=hdf5_writer.read_source_files(),
            raw_config_text=self.raw_config_text,
            config=self.config,
            scope_names=self.scope_ips.keys(),
        )

    def _save_scope_metadata(self, scope_name):
        hdf5_writer.write_scope_metadata(
            self.save_path,
            scope_name=scope_name,
            description=self.get_scope_description(scope_name),
            ip_address=self.scope_ips[scope_name],
            scope_type=self.scopes[scope_name].idn_string,
        )

    def save_time_arrays(self, scope_name, time_array, is_sequence):
        """Save time array for a scope to HDF5 file."""
        self.time_arrays[scope_name] = time_array
        hdf5_writer.write_time_array(self.save_path, scope_name, time_array, is_sequence)

    def update_scope_hdf5(self, all_data, shot_num, overwrite=False):
        """Append a shot of scope data to the HDF5 file (raw int16).

        ``overwrite`` replaces an existing shot group instead of raising; left
        in as a general capability (no caller sets it on this branch).
        """
        descriptions = {
            (scope_name, tr): self.get_channel_description(f"{scope_name}_{tr}")
            for scope_name, (traces, _data, _headers) in all_data.items()
            for tr in traces
        }
        hdf5_writer.write_shot_data(self.save_path, all_data, shot_num, descriptions,
                                    overwrite=overwrite)

    # -- scope lifecycle -----------------------------------------------------
    def initialize_scopes(self):
        """Connect to every scope, capture its time array, and save metadata.

        Returns: {scope_name: is_sequence_flag} for every scope that came up.
        """
        active_scopes = {}
        for name, ip in self.scope_ips.items():
            print(f"\nInitializing {name}...", end='')

            try:
                LeCroy_Scope = _lecroy_scope_class()
                # 30 s timeout. Set via the constructor (which normalizes units),
                # never via scope.scope.timeout -- the VICP transport takes seconds.
                self.scopes[name] = LeCroy_Scope(ip, verbose=False, timeout=30.0)
                scope = self.scopes[name]

                # Optimize scope settings for faster acquisition
                scope.scope.chunk_size = 4 * 1024 * 1024  # 4MB transfer chunk

                scope.set_trigger_mode('SINGLE')

                is_sequence, time_array, traces = init_acquire_from_scope(scope, name)

                if is_sequence is not None and time_array is not None:
                    self._displayed_traces[name] = traces
                    self.save_time_arrays(name, time_array, is_sequence)
                    self._save_scope_metadata(name)

                    active_scopes[name] = is_sequence
                    print(f"Successfully initialized {name}")
                else:
                    print(f"Warning: Could not initialize {name} - no valid data returned")
                    self.cleanup_scope(name)

            except Exception as e:
                print(f"Error initializing {name}: {str(e)}")
                self.cleanup_scope(name)
                continue
        return active_scopes

    def cleanup_scope(self, name):
        """Clean up resources for a specific scope."""
        if name in self.scopes:
            try:
                self.scopes[name].__exit__(None, None, None)
                del self.scopes[name]
            except Exception as e:
                print(f"Error closing scope {name}: {e}")

    # -- per-shot operations -------------------------------------------------
    def _read_one_scope(self, name, mode):
        """Read all traces from a single scope. Returns (traces, data, headers).

        Pure per-scope work with no shared state, so it is safe to call from a
        worker thread (one scope == one independent TCP/VICP transport).
        """
        scope = self.scopes[name]
        ref_channel = self._arm_channels.get(name)
        traces = self._displayed_traces[name]
        if mode == 0:
            return acquire_from_scope(scope, name, traces, ref_channel)
        elif mode == 1:
            return acquire_from_scope_sequence(scope, name, traces, ref_channel)
        else:
            raise ValueError(f"Invalid active_scopes value for {name}: {mode}")

    def acquire_shot(self, active_scopes, shot_num, verbose=True):
        """Acquire data from all active scopes for one shot (sequential)."""
        all_data = {}
        failed_scopes = []

        for name in active_scopes:
            try:
                if verbose:
                    print(f"Acquiring data from {name}...", end='')

                traces, data, headers = self._read_one_scope(name, active_scopes[name])

                if traces:
                    all_data[name] = (traces, data, headers)
                else:
                    print(f"Warning: No valid data from {name} for shot {shot_num}")
                    failed_scopes.append(name)

            except KeyboardInterrupt:
                print(f"\nScope acquisition interrupted for {name}")
                raise
            except Exception as e:
                print(f"Error acquiring from {name}: {e}")
                failed_scopes.append(name)
        if verbose:
            print("done")
        return all_data

    def acquire_shot_parallel(self, active_scopes, shot_num, verbose=True):
        """Acquire data from all active scopes for one shot, reading scopes in
        parallel threads.

        Each scope owns its own TCP/VICP socket, and the waveform read is
        blocking socket I/O that releases the GIL, so threads overlap the
        transfers. Contract matches `acquire_shot`: returns the same
        {name: (traces, data, headers)} dict, skips scopes that error or return
        no data, and lets KeyboardInterrupt propagate to abort the run.
        """
        if len(active_scopes) <= 1:
            # Nothing to overlap; avoid thread/executor overhead.
            return self.acquire_shot(active_scopes, shot_num, verbose=verbose)

        if verbose:
            print(f"Acquiring data from {len(active_scopes)} scopes in parallel...", end='')

        all_data = {}
        failed_scopes = []

        with ThreadPoolExecutor(max_workers=len(active_scopes)) as executor:
            futures = {
                executor.submit(self._read_one_scope, name, mode): name
                for name, mode in active_scopes.items()
            }
            for future in futures:
                name = futures[future]
                try:
                    traces, data, headers = future.result()
                    if traces:
                        all_data[name] = (traces, data, headers)
                    else:
                        print(f"Warning: No valid data from {name} for shot {shot_num}")
                        failed_scopes.append(name)
                except KeyboardInterrupt:
                    print(f"\nScope acquisition interrupted for {name}")
                    raise
                except Exception as e:
                    print(f"Error acquiring from {name}: {e}")
                    failed_scopes.append(name)
        if verbose:
            print("done")
        return all_data

    def acquire_shot_dispatch(self, active_scopes, shot_num, verbose=True):
        """Read all scopes for one shot, parallel or sequential per config."""
        if self.parallel_scope_read:
            all_data = self.acquire_shot_parallel(active_scopes, shot_num, verbose=verbose)
        else:
            all_data = self.acquire_shot(active_scopes, shot_num, verbose=verbose)
        self._warn_if_trigger_timestamps_desynced(all_data, active_scopes)
        return all_data

    def _warn_if_trigger_timestamps_desynced(self, all_data, active_scopes):
        """Advisory-only: warn ONCE if scopes' trigger timestamps disagree.

        Reads the trigger time from each scope's WAVEDESC (first trace this shot)
        and, if the master and any slave differ by more than ~one inter-edge
        interval, prints a single warning. This is a sanity hint that the
        master/slave arming may be capturing different edges -- it is NOT
        authoritative because each scope's real-time clock may not be
        synchronized, so it never raises and never writes to the log file.
        Fires at most once per run and is fully swallowed on any error.
        """
        if self._sync_warned or len(all_data) < 2:
            return
        try:
            from lab_scopes.lecroy import wavedesc_trigger_timestamp

            master = self._master_scope(active_scopes)
            stamps = {}
            for name, (traces, _data, headers) in all_data.items():
                if not traces:
                    continue
                hdr = self.scopes[name].translate_header_bytes(headers[traces[0]])
                ts = wavedesc_trigger_timestamp(hdr)
                if ts is not None:
                    stamps[name] = ts
            if master not in stamps or len(stamps) < 2:
                # Not enough parseable timestamps to compare yet; try again next
                # shot rather than permanently disabling the check.
                return

            # We have a real comparison -- this is the one-and-only attempt.
            self._sync_warned = True

            # Tolerance: a generous fraction of a second. We cannot know the
            # timer's rep rate here, so use a fixed small window -- a same-edge
            # capture differs by ~ the trigger-out propagation (sub-ms); a
            # one-shot desync differs by a full inter-edge interval.
            tol = 0.5
            master_ts = stamps[master]
            desynced = {n: (ts - master_ts) for n, ts in stamps.items()
                        if n != master and abs(ts - master_ts) > tol}
            if desynced:
                deltas = ", ".join(f"{n} {d:+.3f}s" for n, d in desynced.items())
                print(f"\n[sync warning] scope trigger timestamps disagree with "
                      f"master '{master}': {deltas}. This MAY indicate the scopes "
                      f"captured different edges, but can also be a clock-sync "
                      f"artifact (scope RTCs are not synchronized). Advisory only.")
        except Exception:
            pass

    def _master_scope(self, active_scopes):
        """Return the master scope name: the last scope listed in [scope_ips]
        that is currently active. By convention the master's hardware
        trigger-out is wired to the other scopes' EXT trigger input, so the
        master must be armed *last* (see arm_scopes_for_trigger)."""
        for name in reversed(list(self.scope_ips)):
            if name in active_scopes:
                return name
        # Fallback: no scope_ips ordering available, use last active scope.
        return list(active_scopes)[-1] if active_scopes else None

    def _arm_slave(self, name):
        """Arm a slave scope and confirm via INR that it is trigger-ready.

        Uses arm_single_and_confirm (CLEAR_SWEEPS + TRIG_MODE SINGLE + poll the
        scope's INR trigger-ready bit). The master's trigger-out drives the
        slaves, so a slave that is not yet listening on EXT would miss the
        master's edge and desync the run; if the slave does not report ready
        within slave_ready_timeout we RAISE so the shot aborts rather than
        silently desyncing. The reference channel is cached after the first shot
        and passed back in, so later shots skip channel discovery (a per-shot
        scan of the scope's detected input channels (C1-C4 or C1-C8) via
        :TRACE? queries).
        """
        cached = self._arm_channels.get(name)
        channel, ready = self.scopes[name].arm_single_and_confirm(
            channel=cached, ready_timeout=self.slave_ready_timeout)
        self._arm_channels[name] = channel
        if not ready:
            raise RuntimeError(
                f"Slave scope {name} did not report trigger-ready within "
                f"{self.slave_ready_timeout}s; aborting shot to avoid desync")

    def _arm_master(self, name):
        """Arm the master scope last, requiring a real SIN (not an instant STOP).

        arm_master_single does not accept an immediate STOP as armed, so the
        master never silently fires (driving the slaves) before this returns.
        Reference channel is cached/reused like the slave path.
        """
        cached = self._arm_channels.get(name)
        self._arm_channels[name] = self.scopes[name].arm_master_single(channel=cached)

    def arm_scopes_for_trigger(self, active_scopes, verbose=True):
        """Arm all scopes for trigger, master armed LAST.

        With a free-running external timer the software cannot gate trigger
        edges, so scopes that re-arm at slightly different moments can latch
        different edges -- scope A captures edge N while scope B captures N+1,
        silently desyncing the shot. To guarantee every scope captures the SAME
        edge, one scope is the *master*: its hardware trigger-out is wired to
        the other scopes' EXT trigger input (set up in hardware; trigger sources
        configured manually on each scope). Slaves therefore cannot fire until
        the master fires.

        The fix for the observed desync: each slave is confirmed trigger-READY
        (via its INR status register, _arm_slave) before the master is armed, and
        the master is armed with a strict SIN check (_arm_master) so it cannot
        fire before that confirmation. Slaves may still be armed concurrently
        (parallel_scope_arm) to minimise skew; we join all slave arms -- which
        also surfaces a not-ready slave as a raised error -- before arming the
        master. Each scope owns its own transport with no shared state, so the
        threaded arm is safe; arm errors propagate (we do NOT swallow them) so a
        scope that fails to arm or confirm ready aborts the shot.
        """
        if verbose:
            print("Arming scopes for trigger... ", end='')

        master = self._master_scope(active_scopes)
        slaves = [name for name in active_scopes if name != master]

        # Arm all slaves first and confirm each is trigger-ready before the
        # master goes. A slave that fails to confirm raises here and aborts.
        if slaves:
            if self.parallel_scope_arm and len(slaves) > 1:
                with ThreadPoolExecutor(max_workers=len(slaves)) as executor:
                    futures = {
                        executor.submit(self._arm_slave, name): name for name in slaves
                    }
                    for future in futures:
                        future.result()  # surface (re-raise) any per-scope arm failure
            else:
                for name in slaves:
                    self._arm_slave(name)

        # Master last: its trigger-out drives the slaves, so no slave can fire
        # until every slave is already armed+ready and the master goes live.
        if master is not None:
            self._arm_master(master)

        if verbose:
            print(f"armed (master={master})")


def _lecroy_scope_class():
    from lab_scopes.lecroy import LeCroy_Scope

    return LeCroy_Scope


# =============================================================================
# Per-shot orchestration (used by the spooled loops and external callers)
# =============================================================================
def single_shot_acquisition(msa, active_scopes, shot_num, verbose=True,
                            overwrite=False):
    """Run one shot end to end.

    Per-shot sequence:
      1. New shot: the caller's loop is sequential, so the previous shot's reads
         are already complete before we get here (no extra guard needed); the
         arm-time CLEAR_SWEEPS resets each scope's counter so this shot is
         distinguishable.
      2-3. Arm scopes (arm_scopes_for_trigger): slaves first, master last.
      4-6. For each scope (in parallel): check STOP + fresh sweep, flag that data
         exists, then acquire its traces (acquire_shot_dispatch ->
         acquire_from_scope). 7. Write, then return so the loop advances.
    """
    # Steps 2-3: arm slaves then master.
    msa.arm_scopes_for_trigger(active_scopes, verbose=verbose)
    # Steps 4-6: per scope -- check STOP+fresh, flag data exists, acquire (parallel).
    all_data = msa.acquire_shot_dispatch(active_scopes, shot_num, verbose=verbose)

    if all_data:
        if verbose:
            print('Updating scope data to HDF5...')
        msa.update_scope_hdf5(all_data, shot_num, overwrite=overwrite)
    else:
        tqdm.write(f"Warning: No valid data acquired at shot {shot_num}")


def handle_movement(pos_manager, mc, shot_num, pos, save_path, scope_ips):
    """Move the probe and record skip metadata if movement fails.

    Returns True on a successful move, False if the shot was logged as skipped.
    """
    if pos_manager.nz is None:
        tqdm.write(f'Shot = {shot_num}, x = {pos["x"]}, y = {pos["y"]}')
    else:
        tqdm.write(f'Shot = {shot_num}, x = {pos["x"]}, y = {pos["y"]}, z = {pos["z"]}')

    try:
        mc.enable
        if pos_manager.nz is None:
            mc.probe_positions = (pos['x'], pos['y'])
        else:
            mc.probe_positions = (pos['x'], pos['y'], pos['z'])

        mc.wait_for_motion_complete()
        mc.disable
        return True

    except KeyboardInterrupt:
        mc.stop_now
        raise KeyboardInterrupt
    except ValueError as e:
        tqdm.write(f'Skipping position - {str(e)}')
        hdf5_writer.mark_shot_skipped_for_scopes(save_path, scope_ips, shot_num, e)
        return False
    except Exception as e:
        tqdm.write(f'Motor failed to move with {str(e)}')
        hdf5_writer.mark_shot_skipped_for_scopes(
            save_path, scope_ips, shot_num, f"Motor movement failed: {str(e)}"
        )
        return False


# =============================================================================
# Main acquisition loop
# =============================================================================
def run_acquisition_spooled(spool_dir, hdf5_path, config_path):
    """Parallel-mode grid acquisition: build the HDF5 skeleton, spool each shot.

    The grid/stationary direct-to-HDF5 (non-spooled) path was removed; this is
    now the only grid acquisition entry point. Like the bmotion spooled path,
    the acquire process creates ``hdf5_path`` and writes its full skeleton
    (experiment/scope metadata, time arrays, and the ``/Control/Positions``
    group) up front, then spools each shot's raw traces to the fast-disk
    ``spool_dir`` for a separate offload process to fill in.
    """
    from spooling import spool_format
    from . import grid_spool_adapter

    print('Starting spooled grid acquisition loop at', time.ctime())
    config, raw_config_text = load_experiment_config(config_path)
    num_duplicate_shots = int(config.get('nshots', 'num_duplicate_shots', fallback=1))
    num_run_repeats = int(config.get('nshots', 'num_run_repeats', fallback=1))
    shot_num = 0

    has_position_config = 'position' in config and config.items('position')
    if has_position_config:
        pos_manager = PositionManager(
            hdf5_path,
            config_path,
            num_duplicate_shots=num_duplicate_shots,
            num_run_repeats=num_run_repeats,
        )
        if pos_manager.is_45deg:
            raise RuntimeError("45-degree acquisition is not supported in spooled mode")
    else:
        pos_manager = None

    # description.txt lives next to the config (both in base_path).
    description_path = config_module.resolve_description_path_from_config(config_path)

    # Defined before the try so the finally can always report a correct count,
    # even if setup fails before the shot loop (0 shots emitted).
    shot_num = 0
    with MultiScopeAcquisition(hdf5_path, config, raw_config_text,
                               description_path=description_path) as msa:
        try:
            print("Initializing HDF5 file...", end='')
            msa.initialize_hdf5_base()
            print("done")

            if pos_manager is not None:
                pos_manager.initialize_position_hdf5()
                mc = pos_manager.initialize_motor()
                if mc is None:
                    print("\n[!] Warning: Failed to initialize motor controller; "
                          "continuing stationary (motors disabled)")
                total_shots = len(pos_manager.positions)
            else:
                mc = None
                print("\nStationary acquisition - No position configuration found")
                total_shots = num_duplicate_shots * num_run_repeats
            print(f"Total shots: {total_shots}")

            print("\nStarting initial acquisition...")
            active_scopes = msa.initialize_scopes()
            if not active_scopes:
                raise RuntimeError("No valid data found from any scope. Aborting acquisition.")

            spool_format.write_run_metadata(spool_dir, {
                "writer": grid_spool_adapter.WRITER_TAG,
                "hdf5_path": hdf5_path,
                "config_scope_names": list(active_scopes.keys()),
                "channel_descriptions": grid_spool_adapter.channel_descriptions(msa),
                "description_path": description_path,
                "nz": pos_manager.nz if pos_manager is not None else None,
            })
            print(f"Wrote run metadata to spool: {spool_dir}")

            from .config import get_disk_full_pause_opts
            pause_seconds, max_retries = get_disk_full_pause_opts(config)

            with tqdm(total=total_shots, desc="Shots", unit="shot") as pbar:
                for n in range(total_shots):
                    shot_num = n + 1
                    coords = None

                    if pos_manager is not None and mc is not None:
                        positions = pos_manager.positions[n]
                        # positions is a structured record (shot_num, x, y[, z]).
                        target = {'x': float(positions['x']), 'y': float(positions['y'])}
                        if pos_manager.nz is not None:
                            target['z'] = float(positions['z'])
                        if not _spooled_grid_move(mc, pos_manager, target):
                            tqdm.write(f"Skipping shot {shot_num} due to movement failure.")
                            spool_format.write_shot(
                                spool_dir,
                                grid_spool_adapter.skipped_payload(
                                    shot_num, "Motor movement failed", target),
                            )
                            pbar.update(1)
                            continue

                    msa.arm_scopes_for_trigger(active_scopes, verbose=False)
                    all_data = msa.acquire_shot_dispatch(active_scopes, shot_num, verbose=False)
                    if not all_data:
                        tqdm.write(f"Skipping shot {shot_num} - no valid data")
                        spool_format.write_shot(
                            spool_dir,
                            grid_spool_adapter.skipped_payload(
                                shot_num, "No valid data acquired", coords),
                        )
                        pbar.update(1)
                        continue

                    if pos_manager is not None and mc is not None:
                        if pos_manager.nz is None:
                            xpos, ypos = mc.probe_positions
                            coords = {'x': xpos, 'y': ypos, 'z': None}
                        else:
                            xpos, ypos, zpos = mc.probe_positions
                            coords = {'x': xpos, 'y': ypos, 'z': zpos}

                    payload = grid_spool_adapter.all_data_to_payload(all_data, shot_num, coords)
                    spool_format.write_shot_with_disk_full_retry(
                        spool_dir, payload, parallel=msa.parallel_spool_write,
                        pause_seconds=pause_seconds, max_retries=max_retries,
                        warn=tqdm.write)
                    pbar.update(1)

        except KeyboardInterrupt as err:
            print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
            raise RuntimeError() from err
        finally:
            # Only signal completion if the run actually started (metadata
            # written). If setup failed before that, there is nothing for the
            # offload to finalize. shot_num is 0 here when no shot was emitted.
            if spool_format.run_metadata_exists(spool_dir):
                spool_format.write_run_complete(spool_dir, shot_num)
                print(f"Wrote RUN_COMPLETE (final_shot_num={shot_num}) to spool")
            else:
                print("Run aborted before metadata was written; "
                      "no RUN_COMPLETE emitted.")


def _spooled_grid_move(mc, pos_manager, target):
    """Move the probe to ``target`` (a dict), returning True on success.

    Mirrors the motion part of :func:`handle_movement` but, since the offload
    owns the HDF5, it does NOT write skip metadata here — the caller spools a
    skipped shot instead.
    """
    try:
        mc.enable
        if pos_manager.nz is None:
            mc.probe_positions = (target['x'], target['y'])
        else:
            mc.probe_positions = (target['x'], target['y'], target['z'])
        mc.wait_for_motion_complete()
        mc.disable
        return True
    except KeyboardInterrupt:
        mc.stop_now
        raise
    except Exception as e:
        tqdm.write(f'Motor failed to move with {str(e)}')
        return False


# =============================================================================
if __name__ == '__main__':
    save_path = 'test_multiscope.h5'
    config_path = 'experiment_config.ini'
    config, _ = load_experiment_config(config_path)

    with MultiScopeAcquisition(save_path, config) as msa:
        active_scopes = msa.initialize_scopes()
        print('Active scopes:', active_scopes)
        msa.arm_scopes_for_trigger(active_scopes)
        all_data = msa.acquire_shot_dispatch(active_scopes, 1)
        print('Acquired data from scopes:', all_data.keys())
