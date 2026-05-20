'''
Multi-scope data acquisition called by Data_Run.py

This module handles scope configuration, data acquisition, and HDF5 file management.
Motion control and probe positioning functionality has been moved to the motion package.

Key components:
- MultiScopeAcquisition class: Manages multiple oscilloscopes, data acquisition and storage
- Scope configuration and metadata loaded from experiment_config.txt
- Save experiment info and data in HDF5 file
- Parallel scope arming support for synchronized acquisition

Dependencies:
- LeCroy_Scope: Interface to LeCroy oscilloscopes
- h5py: HDF5 file handling
- numpy: Numerical operations
- configparser: Configuration file parsing
- motion package: All probe movement and positioning functionality

Created on Feb.14.2024
@author: Jia Han

Updated July.2025
- Separated motion control into dedicated motion package
- Separate scope arming into a separate function
- Change experiment description to read from experiment_config.txt

TODO: MultiscopeAquisition class getting initial time array needs to be optimized.
'''

import numpy as np
from lab_scopes.lecroy import LeCroy_Scope, WAVEDESC_SIZE
import h5py
import time
import os
import configparser
import traceback
import warnings
import xarray as xr

from motion import PositionManager
#===============================================================================================================================================
#===============================================================================================================================================
def load_experiment_config(config_path='experiment_config.txt'):
    """Load experiment configuration from config file.
    
    Returns:
        tuple: (config, raw_config_text)
            - config: ConfigParser object with parsed configuration
            - raw_config_text: Raw text content of the configuration file
    """
    config = configparser.ConfigParser()
    
    # Read the raw config text
    raw_config_text = ""
    try:
        with open(config_path, 'r') as f:
            raw_config_text = f.read()
    except Exception as e:
        print(f"Warning: Could not read raw config file: {e}")
    
    # Parse the config
    config.read(config_path)
    
    # Set defaults if sections don't exist
    if 'experiment' not in config:
        config.add_section('experiment')
    if 'scopes' not in config:
        config.add_section('scopes')
    if 'channels' not in config:
        config.add_section('channels')
    
    # Set default values if not present
    if not config.get('experiment', 'description', fallback=None):
        config.set('experiment', 'description', 'No experiment description provided')
    
    return config, raw_config_text

#===============================================================================================================================================
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
            # if stop_triggering(scope) == True:
            trace_bytes, header_bytes = scope.acquire_bytes(tr)
            hdr = scope.translate_header_bytes(header_bytes)
            # else:
                # raise Exception('Scope did not enter STOP state')

            if hdr.subarray_count < 2: # Get number of segments
                is_sequence = 0 # in RealTime mode
            else:
                is_sequence = 1 # in sequence mode

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
    TIMEOUT = 10  # Timeout in seconds for acquisition

    traces = scope.displayed_traces()
    
    for tr in traces:
        if stop_triggering(scope) == True:
            # Acquire raw int16 data
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
        if stop_triggering(scope) == True:
            # Acquire raw int16 data for all segments
            segment_data, header = scope.acquire_sequence_data(tr)
            # Convert each segment to np.int16 array if not already
            segment_data = [np.asarray(seg, dtype=np.int16) for seg in segment_data]
            data[tr] = np.stack(segment_data)
            headers[tr] = header
            active_traces.append(tr)
        else:
            raise Exception('Scope did not enter STOP state')
    
    return active_traces, data, headers

class MultiScopeAcquisition:
    """Handles scope connections, data acquisition, and scope data storage"""
    
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
        self.time_arrays = {}  # Store time arrays for each scope
        self.config = config
        self.raw_config_text = raw_config_text  # Store the raw config text for later use
        
        # Load scope IPs from config
        if 'scope_ips' not in config:
            raise RuntimeError("No [scope_ips] section found in config. Please check experiment_config.txt")
        self.scope_ips = dict(config.items('scope_ips'))
        if not self.scope_ips:
            raise RuntimeError("No scope IPs found in [scope_ips] section. Please uncomment and configure scope IP addresses in experiment_config.txt")
        

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up scope resources...")
        
        # Close all scope connections
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

    def get_scope_description(self, scope_name):
        """Get scope description from experiment config"""
        return self.config.get('scopes', scope_name, fallback=f'Scope {scope_name} - No description available')
    
    def get_channel_description(self, channel_name):
        """Get channel description from experiment config"""
        return self.config.get('channels', channel_name, fallback=f'Channel {channel_name} - No description available')
    
    def get_experiment_description(self):
        """Get experiment description from experiment config"""
        description = self.config.get('experiment', 'description', fallback='No experiment description provided')
        return description

    def get_script_contents(self):
        """Read the contents of the Python scripts used to create the HDF5 file"""
        script_contents = {}
        
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # List of scripts to include
        # LeCroy_Scope now lives in the external lab_scopes package, not this repo.
        scripts = ['Data_Run.py', 'multi_scope_acquisition.py']
        
        for script in scripts:
            script_path = os.path.join(current_dir, script)
            try:
                with open(script_path, 'r') as f:
                    script_contents[script] = f.read()
            except Exception as e:
                print(f"Warning: Could not read {script}: {str(e)}")
                script_contents[script] = f"Error reading file: {str(e)}"
        
        return script_contents
    
    def initialize_hdf5_base(self):
        """Initialize HDF5 file structure for scopes and experiment metadata"""
        with h5py.File(self.save_path, 'a') as f:
            # Add experiment description and creation time
            f.attrs['description'] = self.get_experiment_description()
            f.attrs['creation_time'] = time.ctime()
            
            # Add Python scripts used to create the file
            script_contents = self.get_script_contents()
            f.attrs['source_code'] = str(script_contents)
            
            # Store configuration files
            config_group = f.require_group('Configuration')
            
            # Store experiment_config.txt from the raw_config_text
            if self.raw_config_text:
                config_group.create_dataset('experiment_config', data=np.bytes_(self.raw_config_text))
                print("Stored full configuration file content from memory")
            else:
                # As a fallback, try to convert config to string using configparser's write method
                try:
                    import io
                    config_buffer = io.StringIO()
                    self.config.write(config_buffer)
                    config_text = config_buffer.getvalue()
                    config_group.create_dataset('experiment_config', data=np.bytes_(config_text))
                    print("Stored configuration using ConfigParser's write method")
                except Exception as e:
                    print(f"Could not convert config to string: {str(e)}")
                    config_group.create_dataset('experiment_config', data=np.bytes_(f"Error saving configuration: {str(e)}"))
            
            # Create scope groups with their descriptions
            for scope_name in self.scope_ips:
                if scope_name not in f:
                    f.create_group(scope_name)

    def initialize_scopes(self):
        """Initialize scopes and get time arrays on first acquisition"""
        active_scopes = {}
        for name, ip in self.scope_ips.items():
            print(f"\nInitializing {name}...", end='')
            
            try:
                # Create scope instance
                self.scopes[name] = LeCroy_Scope(ip, verbose=False)
                scope = self.scopes[name]
                
                # Optimize scope settings for faster acquisition
                scope.scope.chunk_size = 4*1024*1024  # Increase chunk size to 4MB for faster transfer
                scope.scope.timeout = 30000  # 30 second timeout
                
                scope.set_trigger_mode('SINGLE') # Set trigger mode
                
                # Get initial data and time arrays
                is_sequence, time_array = init_acquire_from_scope(scope, name)

                if is_sequence is not None and time_array is not None:  # Only save if we got valid data
                    self.save_time_arrays(name, time_array, is_sequence)
                    
                    # Save scope metadata immediately after successful initialization
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

    def _save_scope_metadata(self, scope_name):
        """Save scope metadata to HDF5 immediately after initialization"""
        with h5py.File(self.save_path, 'a') as f:
            scope_group = f[scope_name]
            scope_group.attrs['description'] = self.get_scope_description(scope_name)
            scope_group.attrs['ip_address'] = self.scope_ips[scope_name]
            scope_group.attrs['scope_type'] = self.scopes[scope_name].idn_string

    def cleanup_scope(self, name):
        """Clean up resources for a specific scope"""
        if name in self.scopes:
            try:
                self.scopes[name].__exit__(None, None, None)
                del self.scopes[name]
            except Exception as e:
                print(f"Error closing scope {name}: {e}")

    def acquire_shot(self, active_scopes, shot_num):
        """Acquire data from all active scopes for one shot"""
        all_data = {}
        failed_scopes = []
        
        for name in active_scopes:
            try:
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
                raise  # Re-raise to propagate the interrupt
            except Exception as e:
                print(f"Error acquiring from {name}: {e}")
                failed_scopes.append(name)
        print("✓")
        return all_data
    
    def arm_scopes_for_trigger(self, active_scopes):
        """Arm all scopes for trigger without waiting for completion (for parallel operation)"""
        print("Arming scopes for trigger... ", end='')
        for name in active_scopes:
            scope = self.scopes[name]
            scope.set_trigger_mode('SINGLE')
        print("armed")
    
    def save_time_arrays(self, scope_name, time_array, is_sequence):
        """Save time array for a scope to HDF5 file
        
        Args:
            scope_name: Name of the scope
            time_array: Time array to save
            is_sequence: Whether this is sequence mode data
        """
        with h5py.File(self.save_path, 'a') as f:
            scope_group = f[scope_name]
            # Store the time array for this scope
            self.time_arrays[scope_name] = time_array
            
            # Check if time_array already exists
            if 'time_array' in scope_group:
                raise RuntimeError(f"Time array already exists for scope {scope_name}. This should not happen.")
            
            # Save to HDF5
            time_ds = scope_group.create_dataset('time_array', data=time_array, dtype='float64')
            time_ds.attrs['units'] = 'seconds'
            if is_sequence == 1:
                time_ds.attrs['description'] = 'Time array for all channels; data saved in sequence mode'
            else:
                time_ds.attrs['description'] = 'Time array for all channels'
            time_ds.attrs['dtype'] = str(time_array.dtype)

    def update_scope_hdf5(self, all_data, shot_num):
        """Update HDF5 file with scope data only (save as int16)."""
        with h5py.File(self.save_path, 'a', libver='latest', rdcc_nbytes=0) as f:
            for scope_name, (traces, data, headers) in all_data.items():
                scope_group = f[scope_name]
                shot_name = f'shot_{shot_num}'
                if shot_name in scope_group:
                    raise RuntimeError(f"Shot {shot_num} already exists for scope {scope_name}.")
                shot_group = scope_group.create_group(shot_name)
                shot_group.attrs['acquisition_time'] = time.ctime()
                for tr in traces:
                    if tr not in data:
                        continue
                    trace_data = np.asarray(data[tr], dtype=np.int16)
                    is_sequence = len(trace_data.shape) > 1
                    if is_sequence:
                        chunk_size = (1, min(trace_data.shape[1], 8*1024*1024))
                    else:
                        chunk_size = (min(len(trace_data), 8*1024*1024),)
                    data_ds = shot_group.create_dataset(
                        f'{tr}_data',
                        data=trace_data,
                        dtype='int16',
                        chunks=chunk_size,
                        compression='lzf', # compression_opts=9,
                        shuffle=True, # true if compression is enabled
                        fletcher32=True
                    )
                    header_ds = shot_group.create_dataset(f'{tr}_header', data=np.void(headers[tr]))
                    # Use full key for channel description lookup
                    full_channel_key = f"{scope_name}_{tr}"
                    data_ds.attrs['description'] = self.get_channel_description(full_channel_key)
                    data_ds.attrs['dtype'] = 'int16'
                    header_ds.attrs['description'] = f'Binary header data for {tr}'

#===============================================================================================================================================
# Data Acquisition Functions
#===============================================================================================================================================
def single_shot_acquisition(msa, active_scopes, shot_num):
    
    msa.arm_scopes_for_trigger(active_scopes)
    all_data = msa.acquire_shot(active_scopes, shot_num)
    
    if all_data:
        print('Updating scope data to HDF5...')
        msa.update_scope_hdf5(all_data, shot_num)
    else:
        print(f"Warning: No valid data acquired at shot {shot_num}")

def single_shot_acquisition_45(pos, motors, msa, pos_manager, save_path, scope_ips, active_scopes):
    """Acquire a single shot for 45-degree probe setup
    Args:
        pos: Dictionary containing positions for each probe, where each position is a numpy record with 'shot_num' and 'x' fields
        motors: Dictionary of motor controllers for each probe
        msa: MultiScopeAcquisition instance
        pos_manager: PositionManager instance
        save_path: Path to save HDF5 file
        scope_ips: Dictionary of scope IPs
        active_scopes: Dictionary of active scopes
    """
    # Extract the shot number from the first probe's position data
    # Access by index since it's a numpy record array (0 is shot_num, 1 is x)
    shot_num = int(pos['P16'][0])
    positions = {}
    
    print(f'Shot = {shot_num}')
    
    # Collect active motors and their target positions
    active_motors = []
    target_positions = []
    probe_order = []  # Keep track of probe order for mapping returned positions
    
    for probe, motor in motors.items():
        if motor is not None:
            # Access by index since it's a numpy record array (0 is shotnum, 1 is x)
            x_position = float(pos[probe][1])  # Get the x field
            print(f', {probe}: {x_position}', end='')
            active_motors.append(motor)
            target_positions.append(x_position)
            probe_order.append(probe)
            positions[probe] = None  # Initialize all positions to None
        else:
            positions[probe] = None
    
    # Move all active probes simultaneously if we have any
    if active_motors:
        try:
            achieved_positions = move_45deg_probes(active_motors, target_positions)
            
            # Update positions dictionary with achieved positions
            for i, probe in enumerate(probe_order):
                positions[probe] = achieved_positions[i]
                
        except Exception as e:
            print(f'\nError moving probes: {str(e)}')
            # Create empty shot groups with explanation for failed probes
            with h5py.File(save_path, 'a') as f:
                for probe in probe_order:
                    probe_group = f[f'/Control/Positions/{probe}']
                    shot_group = probe_group.create_group(f'shot_{shot_num}')
                    shot_group.attrs['skipped'] = True
                    shot_group.attrs['skip_reason'] = str(e)
                    shot_group.attrs['acquisition_time'] = time.ctime()
            return
    
    # Arm scopes for trigger as soon as probes are in position
    msa.arm_scopes_for_trigger(active_scopes)

    # Acquire data from all scopes at this position
    all_data = msa.acquire_shot(active_scopes, shot_num)
    
    if all_data:
        # Update scope data in HDF5
        msa.update_scope_hdf5(all_data, shot_num)
        
        # Update position data in HDF5
        pos_manager.update_position_hdf5(shot_num, positions)
    else:
        print(f"Warning: No valid data acquired at shot {shot_num}")
        
#===============================================================================================================================================
# Main Acquisition Loop  
#===============================================================================================================================================
def handle_movement(pos_manager, mc, shot_num, pos, save_path, scope_ips):
    if pos_manager.nz is None:
        print(f'Shot = {shot_num}, x = {pos["x"]}, y = {pos["y"]}')
    else:
        print(f'Shot = {shot_num}, x = {pos["x"]}, y = {pos["y"]}, z = {pos["z"]}')

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
        print(f'\nSkipping position - {str(e)}')
        # Create empty shot group with explanation
        with h5py.File(save_path, 'a') as f:
            for scope_name in scope_ips:
                scope_group = f[scope_name]
                shot_group = scope_group.create_group(f'shot_{shot_num}')
                shot_group.attrs['skipped'] = True
                shot_group.attrs['skip_reason'] = str(e)
                shot_group.attrs['acquisition_time'] = time.ctime()
        return False
    except Exception as e:
        print(f'\nMotor failed to move with {str(e)}')
        # Create empty shot group with explanation
        with h5py.File(save_path, 'a') as f:
            for scope_name in scope_ips:
                scope_group = f[scope_name]
                shot_group = scope_group.create_group(f'shot_{shot_num}')
                shot_group.attrs['skipped'] = True
                shot_group.attrs['skip_reason'] = f"Motor movement failed: {str(e)}"
                shot_group.attrs['acquisition_time'] = time.ctime()
        return False



def run_acquisition(save_path, config_path):

    print('Starting acquisition loop at', time.ctime())
    config, raw_config_text = load_experiment_config(config_path)
    num_duplicate_shots = int(config.get('nshots', 'num_duplicate_shots', fallback=1))
    num_run_repeats = int(config.get('nshots', 'num_run_repeats', fallback=1))

    # Check if position configuration exists
    has_position_config = 'position' in config and config.items('position')
    
    if has_position_config:
        pos_manager = PositionManager(
            save_path, 
            config_path,
            num_duplicate_shots=num_duplicate_shots,
            num_run_repeats=num_run_repeats
        )
    else:
        pos_manager = None

    # Initialize multi-scope acquisition
    with MultiScopeAcquisition(save_path, config, raw_config_text) as msa:
        try:
            # Initialize HDF5 file structure
            print("Initializing HDF5 file...", end='')
            msa.initialize_hdf5_base()  # Initialize scope structure
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

            else:
                # Stationary acquisition - no positions or motors
                positions = None
                mc = None
                print("\nStationary acquisition - No position configuration found")
                total_shots = num_duplicate_shots * num_run_repeats

            # First shot: Initialize scopes and save time arrays
            print("\nStarting initial acquisition...")
            active_scopes = msa.initialize_scopes()
            if not active_scopes:
                raise RuntimeError("No valid data found from any scope. Aborting acquisition.")
            
            # Main acquisition loop
            n = 0  # 1-based shot numbering
            for n in range(total_shots):
                shot_num = n + 1
                acquisition_loop_start_time = time.time()

                if pos_manager is not None:
                    movement_success = handle_movement(pos_manager, mc, shot_num, positions[n], save_path, msa.scope_ips)
                    if not movement_success:
                        print(f"Skipping shot {shot_num} due to movement failure.")
                        continue
                else:
                    print(f'Shot = {shot_num}') # Stationary acquisition

                single_shot_acquisition(msa, active_scopes, shot_num)

                if pos_manager is not None and mc is not None: # Update position data in HDF5
                    if pos_manager.nz is None:
                        # For 2D movement, get current probe position
                        xpos, ypos = mc.probe_positions
                        current_positions = {'x': xpos, 'y': ypos, 'z': None}
                    else:
                        # For 3D movement, get current probe position
                        xpos, ypos, zpos = mc.probe_positions
                        current_positions = {'x': xpos, 'y': ypos, 'z': zpos}
                    
                    # Update the positions in HDF5 file
                    pos_manager.update_position_hdf5(shot_num, current_positions)

                # Calculate and display remaining time
                time_per_shot = (time.time() - acquisition_loop_start_time)
                remaining_shots = total_shots - shot_num
                remaining_time = remaining_shots * time_per_shot
                print(f' | Remaining time: {remaining_time/3600:.2f}h')
                
                n += 1

        except KeyboardInterrupt:
            print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
            raise

        finally:
            # Store the final shot count for each scope group
            # This makes shot existence checks much faster for large datasets
            print(f"Storing shot count ({shot_num}) to HDF5 file...")
            try:
                with h5py.File(save_path, 'a') as f:
                    for scope_name in msa.scope_ips:
                        if scope_name in f:
                            scope_group = f[scope_name]
                            scope_group.attrs['shot_count'] = shot_num
                            print(f"  - {scope_name}: {shot_num} shots recorded")
            except Exception as e:
                print(f"Error storing shot count: {e}")

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
    # test
    save_path = 'test_multiscope.h5'
    config_path = 'experiment_config.txt'
    config, _ = load_experiment_config(config_path)

    with MultiScopeAcquisition(save_path, config) as msa:
        active_scopes = msa.initialize_scopes()
        print('Active scopes:', active_scopes)
        msa.arm_scopes_for_trigger(active_scopes)
        all_data = msa.acquire_shot(active_scopes, 1)
        print('Acquired data from scopes:', all_data.keys())
        