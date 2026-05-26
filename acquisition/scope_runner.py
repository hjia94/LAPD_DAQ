"""Scope coordination and run-loop orchestration.

Layered like this:

    helper functions (stop_triggering, init_acquire_from_scope,
        acquire_from_scope, acquire_from_scope_sequence)
        - talk to a single LeCroy_Scope instance

    MultiScopeAcquisition class
        - owns the live scope handles and the active config
        - hides the file-write details behind hdf5_writer

    single_shot_acquisition / single_shot_acquisition_45 / handle_movement
        - one acquisition step (with or without motion)

    run_acquisition
        - top-level loop driven by experiment_config.txt; the function
          Data_Run.py and Data_Run_45deg.py call into.
"""

import time

import numpy as np
from tqdm import tqdm

from motion import PositionManager

from . import hdf5_writer
from .config import load_experiment_config


# =============================================================================
# Single-scope helpers
# =============================================================================
def stop_triggering(scope, retry=500):
    retry_count = 0
    while retry_count < retry:
        try:
            current_mode = scope.set_trigger_mode("")
            if current_mode[0:4] == 'STOP':
                return True
            time.sleep(0.05)
        except KeyboardInterrupt:
            print('Keyboard interrupted in stop_triggering')
            raise
        retry_count += 1

    print('Scope did not enter STOP state')
    return False


def init_acquire_from_scope(scope, scope_name):
    """Initialize acquisition from a single scope and get initial data and time arrays
    Args:
        scope: LeCroy_Scope instance
        scope_name: Name of the scope
    Returns:
        tuple: (is_sequence, time_array)
            - is_sequence: 0 for RealTime mode, 1 for sequence mode
            - time_array: Time array for the scope
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
        return None, None

    return is_sequence, time_array


def acquire_from_scope(scope, scope_name):
    """Acquire data from a single scope with optimized speed (int16/raw)."""
    data = {}
    headers = {}
    active_traces = []

    traces = scope.displayed_traces()

    for tr in traces:
        if stop_triggering(scope) is True:
            data[tr], headers[tr] = scope.acquire(tr, raw=True)
            active_traces.append(tr)
        else:
            raise Exception('Scope did not enter STOP state')

    return active_traces, data, headers


def acquire_from_scope_sequence(scope, scope_name):
    """Acquire sequence mode data from a single scope (int16/raw)."""
    data = {}
    headers = {}
    active_traces = []

    traces = scope.displayed_traces()

    for tr in traces:
        if stop_triggering(scope) is True:
            segment_data, header = scope.acquire_sequence_data(tr)
            segment_data = [np.asarray(seg, dtype=np.int16) for seg in segment_data]
            data[tr] = np.stack(segment_data)
            headers[tr] = header
            active_traces.append(tr)
        else:
            raise Exception('Scope did not enter STOP state')

    return active_traces, data, headers


# =============================================================================
# Multi-scope coordinator
# =============================================================================
class MultiScopeAcquisition:
    """Owns scope connections, performs acquisition, and forwards to hdf5_writer."""

    def __init__(self, save_path, config, raw_config_text=""):
        """
        Args:
            save_path: path to save HDF5 file
            config: ConfigParser object with experiment configuration
            raw_config_text: Raw text content of the configuration file (optional)
        """
        self.save_path = save_path
        self.scopes = {}
        self.figures = {}
        self.time_arrays = {}
        self.config = config
        self.raw_config_text = raw_config_text

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
        return self.config.get('experiment', 'description',
                               fallback='No experiment description provided')

    # -- HDF5 lifecycle (delegates to hdf5_writer) ---------------------------
    def initialize_hdf5_base(self):
        """Initialize HDF5 file structure for scopes and experiment metadata."""
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

    def update_scope_hdf5(self, all_data, shot_num):
        """Append a shot of scope data to the HDF5 file (raw int16)."""
        descriptions = {
            (scope_name, tr): self.get_channel_description(f"{scope_name}_{tr}")
            for scope_name, (traces, _data, _headers) in all_data.items()
            for tr in traces
        }
        hdf5_writer.write_shot_data(self.save_path, all_data, shot_num, descriptions)

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
                self.scopes[name] = LeCroy_Scope(ip, verbose=False)
                scope = self.scopes[name]

                # Optimize scope settings for faster acquisition
                scope.scope.chunk_size = 4 * 1024 * 1024  # 4MB transfer chunk
                scope.scope.timeout = 30000  # 30 second timeout

                scope.set_trigger_mode('SINGLE')

                is_sequence, time_array = init_acquire_from_scope(scope, name)

                if is_sequence is not None and time_array is not None:
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
    def acquire_shot(self, active_scopes, shot_num, verbose=True):
        """Acquire data from all active scopes for one shot."""
        all_data = {}
        failed_scopes = []

        for name in active_scopes:
            try:
                if verbose:
                    print(f"Acquiring data from {name}...", end='')
                scope = self.scopes[name]

                if active_scopes[name] == 0:
                    traces, data, headers = acquire_from_scope(scope, name)
                elif active_scopes[name] == 1:
                    traces, data, headers = acquire_from_scope_sequence(scope, name)
                else:
                    raise ValueError(f"Invalid active_scopes value for {name}: {active_scopes[name]}")

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
            print("✓")
        return all_data

    def arm_scopes_for_trigger(self, active_scopes, verbose=True):
        """Arm all scopes for trigger without waiting for completion (for parallel operation)."""
        if verbose:
            print("Arming scopes for trigger... ", end='')
        for name in active_scopes:
            scope = self.scopes[name]
            scope.set_trigger_mode('SINGLE')
        if verbose:
            print("armed")


def _lecroy_scope_class():
    from lab_scopes.lecroy import LeCroy_Scope

    return LeCroy_Scope


# =============================================================================
# Per-shot orchestration (used by run_acquisition and external callers)
# =============================================================================
def single_shot_acquisition(msa, active_scopes, shot_num, verbose=True):
    msa.arm_scopes_for_trigger(active_scopes, verbose=verbose)
    all_data = msa.acquire_shot(active_scopes, shot_num, verbose=verbose)

    if all_data:
        if verbose:
            print('Updating scope data to HDF5...')
        msa.update_scope_hdf5(all_data, shot_num)
    else:
        print(f"Warning: No valid data acquired at shot {shot_num}")


def single_shot_acquisition_45(pos, motors, msa, pos_manager, save_path, scope_ips, active_scopes):
    """Acquire a single shot for 45-degree probe setup.

    Args:
        pos: Dictionary {probe_name: numpy record} where each record has
            (shot_num, x) accessible by index.
        motors: Dictionary of motor controllers for each probe (None entries skipped).
        msa: MultiScopeAcquisition instance
        pos_manager: PositionManager instance
        save_path: Path to save HDF5 file
        scope_ips: Dictionary of scope IPs
        active_scopes: Dictionary of active scopes
    """
    # Shot number is read from the first probe's record (index 0 = shot_num).
    shot_num = int(pos['P16'][0])
    positions = {}

    print(f'Shot = {shot_num}')

    active_motors = []
    target_positions = []
    probe_order = []  # parallel to active_motors / target_positions

    for probe, motor in motors.items():
        if motor is not None:
            x_position = float(pos[probe][1])  # index 1 = x
            print(f', {probe}: {x_position}', end='')
            active_motors.append(motor)
            target_positions.append(x_position)
            probe_order.append(probe)
            positions[probe] = None
        else:
            positions[probe] = None

    if active_motors:
        try:
            achieved_positions = move_45deg_probes(active_motors, target_positions)

            for i, probe in enumerate(probe_order):
                positions[probe] = achieved_positions[i]

        except Exception as e:
            print(f'\nError moving probes: {str(e)}')
            hdf5_writer.mark_shot_skipped_for_probes(save_path, probe_order, shot_num, e)
            return

    msa.arm_scopes_for_trigger(active_scopes)
    all_data = msa.acquire_shot(active_scopes, shot_num)

    if all_data:
        msa.update_scope_hdf5(all_data, shot_num)
        pos_manager.update_position_hdf5(shot_num, positions)
    else:
        print(f"Warning: No valid data acquired at shot {shot_num}")


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
def run_acquisition(save_path, config_path):
    print('Starting acquisition loop at', time.ctime())
    config, raw_config_text = load_experiment_config(config_path)
    num_duplicate_shots = int(config.get('nshots', 'num_duplicate_shots', fallback=1))
    num_run_repeats = int(config.get('nshots', 'num_run_repeats', fallback=1))
    shot_num = 0

    has_position_config = 'position' in config and config.items('position')

    if has_position_config:
        pos_manager = PositionManager(
            save_path,
            config_path,
            num_duplicate_shots=num_duplicate_shots,
            num_run_repeats=num_run_repeats,
        )
    else:
        pos_manager = None

    with MultiScopeAcquisition(save_path, config, raw_config_text) as msa:
        try:
            print("Initializing HDF5 file...", end='')
            msa.initialize_hdf5_base()
            print("✓")

            if pos_manager is not None:
                positions = pos_manager.initialize_position_hdf5()

                if pos_manager.is_45deg:
                    motors = pos_manager.initialize_motor_45deg()
                    print("45-degree acquisition not implemented yet")
                    return
                else:
                    mc = pos_manager.initialize_motor()
                    if mc is None:
                        print("\n× Warning: Failed to initialize motor controller")
                        print("  - Check [motor_ips] section in your config file")
                        print("  - Continuing with stationary acquisition (motors disabled)")
                    else:
                        print("\n✓ Motor controller initialized and ready for movement")
                total_shots = len(positions)
                print(f"Number of positions: {len(positions)}")
                print(f"Number of shots per position: {num_duplicate_shots}")
                print(f"Total shots: {total_shots}")

            else:
                positions = None
                mc = None
                print("\nStationary acquisition - No position configuration found")
                total_shots = num_duplicate_shots * num_run_repeats
                print(f"Number of shots: {total_shots}")

            print("\nStarting initial acquisition...")
            active_scopes = msa.initialize_scopes()
            if not active_scopes:
                raise RuntimeError("No valid data found from any scope. Aborting acquisition.")

            with tqdm(total=total_shots, desc="Shots", unit="shot") as pbar:
                for n in range(total_shots):
                    shot_num = n + 1

                    if pos_manager is not None:
                        movement_success = handle_movement(
                            pos_manager, mc, shot_num, positions[n], save_path, msa.scope_ips
                        )
                        if not movement_success:
                            tqdm.write(f"Skipping shot {shot_num} due to movement failure.")
                            pbar.update(1)
                            continue

                    single_shot_acquisition(msa, active_scopes, shot_num, verbose=False)

                    if pos_manager is not None and mc is not None:
                        if pos_manager.nz is None:
                            xpos, ypos = mc.probe_positions
                            current_positions = {'x': xpos, 'y': ypos, 'z': None}
                        else:
                            xpos, ypos, zpos = mc.probe_positions
                            current_positions = {'x': xpos, 'y': ypos, 'z': zpos}

                        pos_manager.update_position_hdf5(shot_num, current_positions)

                    pbar.update(1)

        except KeyboardInterrupt:
            print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
            raise

        finally:
            hdf5_writer.record_shot_count(save_path, msa.scope_ips, shot_num)


# =============================================================================
if __name__ == '__main__':
    save_path = 'test_multiscope.h5'
    config_path = 'experiment_config.txt'
    config, _ = load_experiment_config(config_path)

    with MultiScopeAcquisition(save_path, config) as msa:
        active_scopes = msa.initialize_scopes()
        print('Active scopes:', active_scopes)
        msa.arm_scopes_for_trigger(active_scopes)
        all_data = msa.acquire_shot(active_scopes, 1)
        print('Acquired data from scopes:', all_data.keys())
