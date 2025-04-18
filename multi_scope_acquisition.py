import numpy as np
import matplotlib.pyplot as plt
from LeCroy_Scope import LeCroy_Scope, WAVEDESC_SIZE
import h5py
import time
import os
from Motor_Control import Motor_Control_2D, Motor_Control_3D
from Motor_Control_1D import Motor_Control as MC

#===============================================================================================================================================
def stop_triggering(scope, retry=500):
    retry_count = 0
    while retry_count < retry:
        try:
            current_mode = scope.set_trigger_mode("")
            if current_mode[0:4] == 'STOP':
                return True
            time.sleep(0.01)
        except KeyboardInterrupt:
            print('Keyboard interrupted')
            break
        retry_count += 1

    print('Scope did not enter STOP state')
    return False

def init_acquire_from_scope(scope, scope_name):
    """Initialize acquisition from a single scope and get initial data and time arrays
    Args:
        scope: LeCroy_Scope instance
        scope_name: Name of the scope
    Returns:
        tuple: (traces, data, headers, time_array)
            - traces: List of trace names that have valid data
            - data: Dict of trace data
            - headers: Dict of trace headers
            - time_array: Time array for the scope
    """
    time_array = None
    is_sequence = None

    traces = scope.displayed_traces()
    
    for tr in traces:
        if stop_triggering(scope) == True:
            trace_bytes, header_bytes = scope.acquire_bytes(tr)
            hdr = scope.translate_header_bytes(header_bytes)
        else:
            raise Exception('Scope did not enter STOP state')

        if hdr.subarray_count < 2: # Get number of segments
            is_sequence = 0 # in RealTime mode
        else:
            is_sequence = 1 # in sequence mode


        # Get time array from first valid trace
        if time_array is None:
            time_array = scope.time_array(tr)

        # except Exception as e:
        #     if "timeout" in str(e).lower():
        #         print(f"Timeout acquiring {tr} from {scope_name} after {TIMEOUT}s")
        #     elif "NSamples = 0" in str(e):
        #         print(f"Skipping {tr} from {scope_name}: Channel is displayed but not active")
        #     else:
        #         print(f"Error acquiring {tr} from {scope_name}: {e}")
        #     continue
    
    return is_sequence, time_array

def acquire_from_scope(scope, scope_name):
    """Acquire data from a single scope with optimized speed
    Args:
        scope: LeCroy_Scope instance
        scope_name: Name of the scope
    Returns:
        tuple: (traces, data, headers)
            - traces: List of trace names that have valid data
            - data: Dict of trace data
            - headers: Dict of trace headers
    """
    data = {}
    headers = {}
    active_traces = []
    TIMEOUT = 10  # Timeout in seconds for acquisition

    traces = scope.displayed_traces()
    
    for tr in traces:
        if stop_triggering(scope) == True:
            data[tr], headers[tr] = scope.acquire(tr)
            active_traces.append(tr)
        else:
            raise Exception('Scope did not enter STOP state')
    
    return active_traces, data, headers

def acquire_from_scope_sequence(scope, scope_name):
    """Acquire sequence mode data from a single scope
    Args:
        scope: LeCroy_Scope instance
        scope_name: Name of the scope
    Returns:
        tuple: (traces, data, headers)
            - traces: List of trace names that have valid data
            - data: Dict of trace data arrays for each segment
            - headers: Dict of trace headers with sequence info
    """
    data = {}
    headers = {}
    active_traces = []

    traces = scope.displayed_traces()
    
    for tr in traces:

        if stop_triggering(scope) == True:
            data[tr], headers[tr] = scope.acquire_sequence_data(tr)
            active_traces.append(tr)
        else:
            raise Exception('Scope did not enter STOP state')
    
    return active_traces, data, headers

class MultiScopeAcquisition:
    def __init__(self, scope_ips, save_path, external_delays, nz=None, is_45deg=False):
        """
        Args:
            scope_ips: dict of scope names and IP addresses
            save_path: path to save HDF5 file
            external_delays: dict of scope names and their external delays in seconds
            nz: number of z positions (None for 2D, int for 3D)
            is_45deg: whether this is a 45-degree probe acquisition
        """
        self.scope_ips = scope_ips
        self.save_path = save_path
        self.external_delays = external_delays if external_delays else {}
        self.nz = nz
        self.is_45deg = is_45deg
        
        self.scopes = {}
        self.figures = {}
        self.time_arrays = {}  # Store time arrays for each scope
        # self.plot_data = {}    # Store plot data for each scope/trace/shot
        

        # # Just create figures
        # for name in self.scope_ips:
        #     try:
        #         self.figures[name] = plt.figure(figsize=(12, 8))
        #         self.figures[name].canvas.manager.set_window_title(f'Scope: {name}')
        #         self.plot_data[name] = {}  # Initialize plot data storage for this scope
        #     except Exception as e:
        #         print(f"Error creating figure for {name}: {e}")
        #         self.cleanup()
        #         raise

    def cleanup(self):
        """Clean up resources"""
        # Close all scope connections
        for scope in self.scopes.values():
            try:
                scope.__exit__(None, None, None)
            except Exception as e:
                print(f"Error closing scope: {e}")
        
        # # Close all figures
        # for fig in self.figures.values():
        #     try:
        #         plt.close(fig)
        #     except Exception as e:
        #         print(f"Error closing figure: {e}")
        
        self.scopes.clear()
        # self.figures.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def get_scope_description(self, scope_name):
        from Data_Run import get_scope_description
        return get_scope_description(scope_name)
    
    def get_channel_description(self, channel_name):
        from Data_Run import get_channel_description
        return get_channel_description(channel_name)
    
    def get_experiment_description(self):
        from Data_Run import get_experiment_description
        return get_experiment_description()
    
    def get_positions(self):
        """Get position arrays based on acquisition type"""
        if self.is_45deg:
            from Data_Run_45deg import create_all_positions, xstart, xstop, nx, nshot
            return create_all_positions(['P16', 'P22', 'P29', 'P34', 'P42'], xstart, xstop, nx, nshot)
        else:
            from Data_Run import get_positions_xy, get_positions_xyz
            if self.nz is None:
                return get_positions_xy()
            else:
                return get_positions_xyz()
    

    def get_script_contents(self):
        """Read the contents of the Python scripts used to create the HDF5 file"""

        script_contents = {}
        
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # List of scripts to include
        scripts = ['Data_Run.py', 'multi_scope_acquisition.py', 'LeCroy_Scope.py']
        
        for script in scripts:
            script_path = os.path.join(current_dir, script)
            try:
                with open(script_path, 'r') as f:
                    script_contents[script] = f.read()
            except Exception as e:
                print(f"Warning: Could not read {script}: {str(e)}")
                script_contents[script] = f"Error reading file: {str(e)}"
        
        return script_contents
    
    def initialize_hdf5(self):
        """Initialize HDF5 file with scope and position information"""
        if self.is_45deg:
            positions, xpos = self.get_positions()
            dtype = {'P16': [('shot_num', '>u4'), ('x', '>f4')],
                    'P22': [('shot_num', '>u4'), ('x', '>f4')],
                    'P29': [('shot_num', '>u4'), ('x', '>f4')],
                    'P34': [('shot_num', '>u4'), ('x', '>f4')],
                    'P42': [('shot_num', '>u4'), ('x', '>f4')]}
        else:
            if self.nz is None:
                positions, xpos, ypos = self.get_positions()
                dtype = [('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4')]
            else:
                positions, xpos, ypos, zpos = self.get_positions()
                dtype = [('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')]

        with h5py.File(self.save_path, 'a') as f:
            # Add experiment description and creation time
            f.attrs['description'] = self.get_experiment_description()
            f.attrs['creation_time'] = time.ctime()
            
            # Add Python scripts used to create the file
            script_contents = self.get_script_contents()
            f.attrs['source_code'] = str(script_contents)
            
            # Create scope groups with their descriptions
            for scope_name in self.scope_ips:
                if scope_name not in f:
                    f.create_group(scope_name)
            
            # Create Control/Positions group and datasets
            if '/Control' not in f:
                ctl_grp = f.create_group('/Control')
                pos_grp = ctl_grp.create_group('Positions')
                
                if self.is_45deg:
                    # Create separate position arrays for each probe
                    for probe in positions:
                        probe_grp = pos_grp.create_group(probe)
                        # Create setup array with metadata
                        setup_ds = probe_grp.create_dataset('positions_setup_array', 
                                                          data=positions[probe], 
                                                          dtype=dtype[probe])
                        setup_ds.attrs['xpos'] = xpos[probe]
                        
                        # Create array for actual positions
                        probe_grp.create_dataset('positions_array', 
                                               shape=(len(positions[probe]),), 
                                               dtype=dtype[probe])
                else:
                    # Create positions setup array with metadata
                    pos_ds = pos_grp.create_dataset('positions_setup_array', data=positions, dtype=dtype)
                    pos_ds.attrs['xpos'] = xpos
                    if not self.is_45deg:
                        pos_ds.attrs['ypos'] = ypos
                        if self.nz is not None:
                            pos_ds.attrs['zpos'] = zpos
                    
                    # Create positions array for actual positions
                    pos_grp.create_dataset('positions_array', shape=(len(positions),), dtype=dtype)

        return positions

    def initialize_scopes(self):
        """Initialize scopes and get time arrays on first acquisition"""
        active_scopes = {}
        for name, ip in self.scope_ips.items():
            print(f"\nInitializing {name}...")
            
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

                if is_sequence != None:  # Only save if we got valid data
                    self.save_time_arrays(name, time_array, is_sequence)
                    active_scopes[name] = is_sequence
                    print(f"Successfully initialized {name}")
                else:
                    print(f"Warning: Could not initialize {name} - no traces returned")
                    self.cleanup_scope(name)
                    
            except Exception as e:
                print(f"Error initializing {name}: {str(e)}")
                self.cleanup_scope(name)
                continue
        return active_scopes

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
            print(f"Acquiring data from {name}...")
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
        
        # Remove failed scopes from active list
        for name in failed_scopes:
            active_scopes.remove(name)
        
        return all_data

    def save_time_arrays(self, scope_name, time_array, is_sequence):
        """Save time array for a scope to HDF5 file
        
        Args:
            scope_name: Name of the scope
            time_array: Time array to save
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


    def update_hdf5(self, all_data, shot_num, positions):
        """Update HDF5 file with acquired data using optimized settings"""
        with h5py.File(self.save_path, 'a') as f:
            # Save data for each scope
            for scope_name, (traces, data, headers) in all_data.items():
                scope_group = f[scope_name]
                
                # Check if shot group already exists
                shot_name = f'shot_{shot_num}'
                if shot_name in scope_group:
                    raise RuntimeError(f"Shot {shot_num} already exists for scope {scope_name}.")
                
                # Create shot group with optimized settings
                shot_group = scope_group.create_group(shot_name)
                shot_group.attrs['acquisition_time'] = time.ctime()
                
                # Save trace data and headers with optimized chunk size and compression
                for tr in traces:
                    if tr not in data:
                        continue
                        
                    # Convert data to appropriate dtype
                    trace_data = np.asarray(data[tr])
                    if trace_data.dtype != np.float64:
                        trace_data = trace_data.astype(np.float64)
                    
                    # Check if this is sequence data (will be 2D array)
                    is_sequence = len(trace_data.shape) > 1
                    
                    # Calculate optimal chunk size
                    if is_sequence:
                        chunk_size = (1, min(trace_data.shape[1], 512*1024))
                    else:
                        chunk_size = (min(len(trace_data), 512*1024),)
                    
                    # Create dataset
                    data_ds = shot_group.create_dataset(f'{tr}_data', 
                                                      data=trace_data,
                                                      chunks=chunk_size,
                                                      compression='gzip',
                                                      compression_opts=9,
                                                      shuffle=True,
                                                      fletcher32=True)
                    
                    # Store header
                    header_ds = shot_group.create_dataset(f'{tr}_header', data=np.void(headers[tr]))
                    
                    # Add metadata
                    data_ds.attrs['description'] = self.get_channel_description(tr)
                    data_ds.attrs['dtype'] = str(trace_data.dtype)
                    header_ds.attrs['description'] = f'Binary header data for {tr}'
            
            # Update position array if we have valid position data
            if self.is_45deg:
                # For 45-degree probes, update each probe's position separately
                for probe, pos in positions.items():
                    if pos is not None:  # Only update if we have valid position data
                        pos_arr = f[f'/Control/Positions/{probe}/positions_array']
                        pos_arr[shot_num-1] = (shot_num, pos)
            else:
                # For regular XY/XYZ acquisition
                pos_arr = f['/Control/Positions/positions_array']
                if all(p is not None for p in positions.values()):
                    if self.nz is None:
                        pos_arr[shot_num-1] = (shot_num, positions['x'], positions['y'])
                    else:
                        pos_arr[shot_num-1] = (shot_num, positions['x'], positions['y'], positions['z'])


    def update_plots(self, all_data, shot_num):
        """Update plots for all scopes with optimized data handling"""
        MAX_PLOT_POINTS = 10000  # Maximum number of points to plot
        
        for scope_name, (traces, data, _) in all_data.items():
            if not traces or scope_name not in self.time_arrays:
                continue
            
            fig = self.figures[scope_name]
            time_array = self.time_arrays[scope_name]
            
            try:
                # Clear the figure but maintain subplot structure
                fig.clear()
                
                # Calculate optimal downsample factor
                n_points = len(time_array)
                if n_points == 0:
                    print(f"Warning: Empty time array for {scope_name}")
                    continue
                
                downsample = max(1, n_points // MAX_PLOT_POINTS)
                plot_time = time_array[::downsample]
                
                # Store the current shot's data
                for tr in traces:
                    if tr not in data:
                        continue
                    if tr not in self.plot_data[scope_name]:
                        self.plot_data[scope_name][tr] = []
                        self.plot_data[scope_name][tr].append({
                                'time': plot_time,
                                'data': data[tr][::downsample],
                                'shot': shot_num
                        })
                
                # Create subplots for each trace
                for i, tr in enumerate(traces):
                    if tr not in data:
                        print(f"Warning: No data for trace {tr}")
                        continue
                    
                    ax = fig.add_subplot(len(traces), 1, i + 1)
                    
                    # Plot all stored shots for this trace
                    for shot_data in self.plot_data[scope_name][tr]:
                        ax.plot(shot_data['time'], shot_data['data'],
                               label=f'Shot {shot_data["shot"]}',
                               alpha=0.7)
                    
                    ax.legend()
                    ax.set_title(f'Trace {tr}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Voltage (V)')
                    ax.grid(True, alpha=0.3)
                
                # Adjust layout to prevent overlapping
                fig.tight_layout()
                
            except Exception as e:
                print(f"Error updating plot for {scope_name}: {e}")
                continue
            
        plt.pause(0.01)  # Single pause after all plots are updated

#===============================================================================================================================================
# Acquisition Function for XY or XYZ probe drive
#===============================================================================================================================================
def initialize_motor(positions, motor_ips, nz):
    # Check if motor movement is needed
    needs_movement = False
    if len(positions) > 1:
        first_pos = positions[0]
        last_pos = positions[-1]
        if not (first_pos['x'] == last_pos['x'] and first_pos['y'] == last_pos['y'] and 
                (nz is None or first_pos['z'] == last_pos['z'])):
            needs_movement = True
    
    # Initialize motor control if needed
    mc = None
    if needs_movement:
        print("Initializing motor...", end='')
        if nz is None:
            from Data_Run import motor_boundary_2D
            print("XY drive in use")
            mc = Motor_Control_2D(motor_ips['x'], motor_ips['y'])
            # Add motor boundary for 2D drive
            mc.boundary_checker.add_motor_boundary(motor_boundary_2D)
        else:
            from Data_Run import outer_boundary, obstacle_boundary, motor_boundary
            print("XYZ drive in use")
            mc = Motor_Control_3D(motor_ips['x'], motor_ips['y'], motor_ips['z'])
            # Add boundaries only for 3D drive
            mc.boundary_checker.add_probe_boundary(outer_boundary, is_outer_boundary=True)
            # mc.boundary_checker.add_probe_boundary(obstacle_boundary)
            mc.boundary_checker.add_motor_boundary(motor_boundary)
    else:
        print("No motor movement required")
    return mc, needs_movement
#===============================================================================================================================================
def single_shot_acquisition(pos, needs_movement, nz, msa, mc, save_path, scope_ips, active_scopes):

    shot_num = pos['shot_num']  # shot_num is 1-based
    # Move to next position if motor control is active
    if needs_movement:
        if nz is None:
            print(f'Shot = {shot_num}, x = {pos["x"]}, y = {pos["y"]}', end='')
        else:
            print(f'Shot = {shot_num}, x = {pos["x"]}, y = {pos["y"]}, z = {pos["z"]}', end='')
            
        try:
            mc.enable
            if nz is None:
                mc.probe_positions = (pos['x'], pos['y'])
            else:
                mc.probe_positions = (pos['x'], pos['y'], pos['z'])
            mc.disable
        except KeyboardInterrupt:
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
            return
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
            return
    else:
        print(f'Shot = {shot_num}', end='')
    
    # Start triggering all scopes as soon as probe is in position
    for name in active_scopes:
        scope = msa.scopes[name]
        scope.set_trigger_mode('SINGLE')

    # Acquire data from all scopes at this position
    all_data = msa.acquire_shot(active_scopes, shot_num)
    
    if all_data:
        if not needs_movement:
            xpos, ypos, zpos = None, None, None
        elif nz is None:
            xpos, ypos = mc.probe_positions
            zpos = None
        else:
            xpos, ypos, zpos = mc.probe_positions
        
        positions = {'x': xpos, 'y': ypos, 'z': zpos}
        msa.update_hdf5(all_data, shot_num, positions)
    else:
        print(f"Warning: No valid data acquired at shot {shot_num}")
#===============================================================================================================================================

#===============================================================================================================================================
# Acquisition Function for 45 degree probe drive
#===============================================================================================================================================
def initialize_motor_45deg(positions, motor_ips):
    # Initialize dictionary to store motor controllers
    motors = {}
    
    # Try to connect to each probe's motor
    for probe in ['P16', 'P22', 'P29', 'P34', 'P42']:
        try:
            # Special case for P29 which has different cm_per_turn
            if probe == 'P29':
                motors[probe] = MC(server_ip_addr=motor_ips[probe], name=f"probe_{probe}", 
                                 stop_switch_mode=2, cm_per_turn=0.127)
            else:
                motors[probe] = MC(server_ip_addr=motor_ips[probe], name=f"probe_{probe}", 
                                 stop_switch_mode=2)
                
            # Set motor speed if connection successful
            motors[probe].motor_speed = 4
            print(f"Connected to {probe} motor")
            
        except Exception as e:
            print(f"Could not connect to {probe} motor: {str(e)}")
            motors[probe] = None
            
    return motors

def move_45deg_probes(mc_list, move_to_list):

	pos_list = []

	for mc in mc_list:
		mc.enable

	for i, mc in enumerate(mc_list):
		mc.motor_position = move_to_list[i]

	for i, mc in enumerate(mc_list): 

		pos = mc.motor_position
		
		if round(pos,2) != move_to_list[i]:
			retry = 0
			while retry < 3:
				try:
					mc.motor_position = move_to_list[i]
					pos = mc.motor_position
					if round(pos,2) == move_to_list[i]:
						break
					else:
						retry += 1
				except:
					print("Failed to move to position %.2f" %(move_to_list[i]))

		pos_list.append(round(pos,2))
		mc.disable
	
	return pos_list

def single_shot_acquisition_45(pos, motors, msa, save_path, scope_ips, active_scopes):
    """Acquire a single shot for 45-degree probe setup
    Args:
        pos: Dictionary containing positions for each probe, where each position is a numpy record with 'shot_num' and 'x' fields
        motors: Dictionary of motor controllers for each probe
        msa: MultiScopeAcquisition instance
        save_path: Path to save HDF5 file
        scope_ips: Dictionary of scope IPs
        active_scopes: Dictionary of active scopes
    """
    # Extract the shot number from the first probe's position data
    # Access by index since it's a numpy record array (0 is shot_num, 1 is x)
    shot_num = int(pos['P16'][0])
    positions = {}
    
    print(f'Shot = {shot_num}', end='')
    
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
    
    # Start triggering all scopes as soon as probes are in position
    for name in active_scopes:
        scope = msa.scopes[name]
        scope.set_trigger_mode('SINGLE')

    # Acquire data from all scopes at this position
    all_data = msa.acquire_shot(active_scopes, shot_num)
    
    if all_data:
        msa.update_hdf5(all_data, shot_num, positions)
    else:
        print(f"Warning: No valid data acquired at shot {shot_num}")

#===============================================================================================================================================
# Main Acquisition Loop
#===============================================================================================================================================
def run_acquisition(save_path, scope_ips, motor_ips, external_delays=None, nz=None, is_45deg=False):
    """Run the main acquisition sequence
    Args:
        save_path: Path to save HDF5 file
        scope_ips: Dictionary of scope IPs
        motor_ips: Dictionary of motor IPs
        external_delays: Dictionary of external delays for scopes
        nz: Number of z positions (None for 2D, int for 3D)
        is_45deg: Whether this is a 45-degree probe acquisition
    """
    print('Starting acquisition loop at', time.ctime())
    
    # Initialize multi-scope acquisition
    with MultiScopeAcquisition(scope_ips, save_path, external_delays, nz, is_45deg) as msa:
        try:
            # Initialize HDF5 file structure
            print("Initializing HDF5 file...")
            positions = msa.initialize_hdf5()
            
            # Initialize motors based on acquisition type
            if is_45deg:
                motors = initialize_motor_45deg(positions, motor_ips)
                needs_movement = any(motor is not None for motor in motors.values())
            else:
                motors, needs_movement = initialize_motor(positions, motor_ips, nz)

            # First shot: Initialize scopes and save time arrays
            print("\nStarting initial acquisition...")
            active_scopes = msa.initialize_scopes()
            if not active_scopes:
                raise RuntimeError("No valid data found from any scope. Aborting acquisition.")
            
            # Main acquisition loop
            if is_45deg:
                # For 45-degree probes, we need to extract corresponding positions for each shot
                shots_count = len(positions['P16'])  # Use P16 as reference for number of shots
                for i in range(shots_count):
                    shot_pos = {}
                    for probe in positions:
                        shot_pos[probe] = positions[probe][i]  # Get ith position for each probe
                    
                    acquisition_loop_start_time = time.time()
                    
                    single_shot_acquisition_45(shot_pos, motors, msa, save_path, scope_ips, active_scopes)
                    
                    # Calculate and display remaining time
                    time_per_pos = (time.time() - acquisition_loop_start_time)
                    remaining_positions = shots_count - (i+1)
                    remaining_time = remaining_positions * time_per_pos
                    print(f'Remaining time: {remaining_time/3600:.2f}h')
            else:
                # Original loop for XY/XYZ acquisition
                for pos in positions:
                    acquisition_loop_start_time = time.time()

                    single_shot_acquisition(pos, needs_movement, nz, msa, motors, save_path, scope_ips, active_scopes)

                    # Calculate and display remaining time
                    time_per_pos = (time.time() - acquisition_loop_start_time)
                    remaining_positions = len(positions) - pos['shot_num']
                    remaining_time = remaining_positions * time_per_pos
                    print(f'Remaining time: {remaining_time/3600:.2f}h')

        except KeyboardInterrupt:
            print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
            raise

        finally:
            # Save final metadata
            with h5py.File(save_path, 'a') as f:
                for scope_name in scope_ips:
                    scope_group = f[scope_name]
                    scope_group.attrs['description'] = msa.get_scope_description(scope_name)
                    scope_group.attrs['ip_address'] = scope_ips[scope_name]
                    scope_group.attrs['scope_type'] = msa.scopes[scope_name].idn_string
            plt.close('all')  # Ensure all figures are closed
