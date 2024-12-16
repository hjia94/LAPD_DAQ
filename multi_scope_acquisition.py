import numpy as np
import matplotlib.pyplot as plt
from LeCroy_Scope import LeCroy_Scope, WAVEDESC_SIZE
import h5py
import time
import os
import struct

#===============================================================================================================================================
def acquire_from_scope(scope, scope_name, first_acquisition=False):
    """Acquire data from a single scope with optimized speed
    Args:
        scope: LeCroy_Scope instance
        scope_name: Name of the scope
        first_acquisition: If True, return time array as well
    Returns:
        traces: List of trace names that have valid data
        data: Dict of trace data
        headers: Dict of trace headers
        time_array: Time array (only if first_acquisition=True)
    """
    st_time = time.time() # for checking acquisition time

    # Check if scope is in STOP mode before acquiring
    MAX_RETRIES = 100  # Maximum number of retries
    RETRY_DELAY = 0.1  # Delay between retries in seconds
    
    for retry in range(MAX_RETRIES):
        current_mode = scope.set_trigger_mode('')  # Get current mode without changing it
        if current_mode.strip() == 'STOP':
            break
        if retry == 0:  # Only print first time
            print(f"Waiting for {scope_name} trigger mode to become STOP (currently {current_mode})")
        time.sleep(RETRY_DELAY)
    else:  # Loop completed without finding STOP mode
        print(f"Warning: Timeout waiting for {scope_name} trigger mode to become STOP after {MAX_RETRIES * RETRY_DELAY:.1f}s")
        if first_acquisition:
            return [], {}, {}, None
        return [], {}, {}

    data = {}
    headers = {}
    time_array = None
    active_traces = []  # List to store only traces that have data
    TIMEOUT = 10  # Timeout in seconds for acquisition

    traces = scope.displayed_traces()
    
    for tr in traces:
        try:
            # Set timeout for acquisition
            scope.timeout = TIMEOUT * 1000  # Convert to ms
            
            # Get raw data using acquire_raw (which now also parses header)
            trace_bytes = scope.acquire_raw(tr)
            
            # Get header bytes for storage
            headers[tr] = np.void(trace_bytes[15:15+WAVEDESC_SIZE])
            
            # Get data indices from already parsed header
            NSamples, ndx0, ndx1 = scope.parse_header(trace_bytes)
            
            # Parse the actual waveform data
            if scope.hdr.comm_type == 1:  # data returned in words
                wdata = struct.unpack(str(NSamples)+'h', trace_bytes[ndx0:ndx1])
                trace_data = np.array(wdata) * scope.hdr.vertical_gain - scope.hdr.vertical_offset
            else:  # data returned in bytes
                cdata = struct.unpack(str(NSamples)+'b', trace_bytes[ndx0:ndx1])
                trace_data = np.array(cdata) * scope.hdr.vertical_gain - scope.hdr.vertical_offset
            
            # Store the data
            data[tr] = trace_data
            active_traces.append(tr)
            
            # Get time array from first valid trace if needed
            if first_acquisition and time_array is None:
                time_array = scope.time_array()

        except Exception as e:
            if "timeout" in str(e).lower():
                print(f"Timeout acquiring {tr} from {scope_name} after {TIMEOUT}s")
            elif "NSamples = 0" in str(e):
                print(f"Skipping {tr} from {scope_name}: Channel is displayed but not active")
            else:
                print(f"Error acquiring {tr} from {scope_name}: {e}")
            continue
    
    # Print acquisition time
    print(f"Acquisition from {scope_name} completed in {time.time() - st_time:.2f} seconds")

    if first_acquisition:
        return active_traces, data, headers, time_array
    return active_traces, data, headers

class MultiScopeAcquisition:
    def __init__(self, scope_ips, num_loops=10, save_path='multi_scope_data.hdf5', 
                 external_delays=None):
        """
        Args:
            scope_ips: dict of scope names and IP addresses
            num_loops: number of shots to acquire
            save_path: path to save HDF5 file
            external_delays: dict of scope names and their external delays in seconds
        """
        self.scope_ips = scope_ips
        self.num_loops = num_loops
        self.save_path = save_path
        self.external_delays = external_delays if external_delays else {}
        self.scopes = {}
        self.figures = {}
        self.time_arrays = {}  # Store time arrays for each scope
        
        # Initialize scopes and figures
        for name, ip in self.scope_ips.items():
            try:
                self.scopes[name] = LeCroy_Scope(ip, verbose=False)
                self.scopes[name].set_trigger_mode('SINGLE')
                self.figures[name] = plt.figure(figsize=(12, 8))
                self.figures[name].canvas.manager.set_window_title(f'Scope: {name}')
            except Exception as e:
                print(f"Error initializing scope {name}: {e}")
                # Clean up any scopes that were successfully initialized
                self.cleanup()
                raise

    def cleanup(self):
        """Clean up resources"""
        # Close all scope connections
        for scope in self.scopes.values():
            try:
                scope.__exit__(None, None, None)
            except Exception as e:
                print(f"Error closing scope: {e}")
        
        # Close all figures
        for fig in self.figures.values():
            try:
                plt.close(fig)
            except Exception as e:
                print(f"Error closing figure: {e}")
        
        self.scopes.clear()
        self.figures.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def get_scope_description(self, scope_name):
        from Data_Run_0D import get_scope_description
        return get_scope_description(scope_name)
    
    def get_channel_description(self, channel_name):
        from Data_Run_0D import get_channel_description
        return get_channel_description(channel_name)
    
    def get_experiment_description(self):
        from Data_Run_0D import get_experiment_description
        return get_experiment_description()
    
    def get_script_contents(self):
        """Read the contents of the Python scripts used to create the HDF5 file"""
        script_contents = {}
        
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # List of scripts to include
        scripts = ['Data_Run_0D.py', 'multi_scope_acquisition.py']
        
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
        """Initialize HDF5 file with scope information and time arrays"""
        with h5py.File(self.save_path, 'a') as f:
            # Add experiment description and creation time
            f.attrs['description'] = self.get_experiment_description()
            f.attrs['creation_time'] = time.ctime()
            
            # Add Python scripts used to create the file
            script_contents = self.get_script_contents()
            f.attrs['source_code'] = str(script_contents)
            
            # Create scope groups with their descriptions
            for scope_name in self.scope_ips:  # Fix: iterate over keys only
                if scope_name not in f:
                    scope_group = f.create_group(scope_name)
                    scope_group.attrs['description'] = self.get_scope_description(scope_name)
                    scope_group.attrs['ip_address'] = self.scope_ips[scope_name]
                    scope_group.attrs['scope_type'] = self.scopes[scope_name].idn_string
                    scope_group.attrs['external_delay(ms)'] = self.external_delays.get(scope_name, '')
        return scope_group

    def save_time_arrays(self, scope_group, time_array):
        """Save time arrays to HDF5 file"""
        with h5py.File(self.save_path, 'a') as f:
                time_ds = scope_group.create_dataset('time_array', data=time_array, dtype='float64')
                
                time_ds.attrs['units'] = 'seconds'
                time_ds.attrs['description'] = 'Time array for all channels'
                time_ds.attrs['dtype'] = str(time_array.dtype)  # Store dtype information

    def update_hdf5(self, all_data, shot_num):
        """Update HDF5 file with acquired data using optimized settings"""
        with h5py.File(self.save_path, 'a') as f:
            # Save data for each scope
            for scope_name, (traces, data, headers) in all_data.items():
                scope_group = f[scope_name]
                
                # Create shot group with optimized settings
                shot_group = scope_group.create_group(f'shot_{shot_num}')
                shot_group.attrs['acquisition_time'] = time.ctime()
                
                # Save trace data and headers with optimized chunk size and compression
                for tr in traces:
                    if tr in data:
                        # Convert data to appropriate dtype if needed (e.g., uint16 for 12-bit data)
                        trace_data = np.asarray(data[tr])
                        if trace_data.dtype != np.uint16:
                            trace_data = trace_data.astype(np.uint16)
                        
                        # Calculate optimal chunk size (aim for ~1MB chunks)
                        chunk_size = min(len(trace_data), 512*1024)  # 512K samples per chunk
                        
                        # Create dataset with optimized settings
                        data_ds = shot_group.create_dataset(
                            f'{tr}_data', 
                            data=trace_data,
                            chunks=(chunk_size,),
                            compression='gzip',
                            compression_opts=1,  # Light compression for speed
                            shuffle=True,  # Helps with compression of binary data
                            fletcher32=True  # Add checksum for data integrity
                        )
                        
                        # Store header as binary data
                        header_ds = shot_group.create_dataset(
                            f'{tr}_header', 
                            data=headers[tr],
                            compression='gzip',
                            compression_opts=1
                        )
                        
                        # Add channel descriptions and metadata
                        data_ds.attrs['description'] = self.get_channel_description(tr)
                        data_ds.attrs['dtype'] = str(trace_data.dtype)
                        data_ds.attrs['original_size'] = len(trace_data)
                        data_ds.attrs['voltage_scale'] = '12-bit centered at 0V, ±10V range'
                        header_ds.attrs['description'] = f'Binary header data for {tr}'

    def update_plots(self, all_data, shot_num):
        """Update plots for all scopes with optimized data handling"""
        MAX_PLOT_POINTS = 10000  # Maximum number of points to plot
        
        for scope_name, (traces, data, _) in all_data.items():
            if not traces:  # Skip if no valid traces for this scope
                continue
                
            fig = self.figures[scope_name]
            time_array = self.time_arrays[scope_name]
            
            try:
                # Calculate optimal downsample factor
                n_points = len(time_array)
                downsample = max(1, n_points // MAX_PLOT_POINTS)
                
                # Pre-calculate downsampled time array
                plot_time = time_array[::downsample]
                
                # Clear the entire figure and create new subplot
                fig.clear()
                ax = fig.add_subplot(self.num_loops, 1, shot_num + 1)
                
                # Plot each trace with optimized downsampling
                for tr in traces:
                    if tr in data:
                        # Convert binary data to voltage values if needed
                        trace_data = np.asarray(data[tr])
                        if trace_data.dtype == np.uint16:
                            # Convert 12-bit data to voltage (-10V to +10V range)
                            trace_data = (trace_data.astype(float) - 2048) * (20.0/4096)
                        
                        # Efficient downsampling using array slicing
                        plot_data = trace_data[::downsample]
                        
                        # Plot in milliseconds
                        ax.plot(plot_time * 1000, plot_data, label=tr)
                
                ax.set_title(f'Shot {shot_num+1} ({len(plot_time)} points)')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Voltage (V)')
                ax.grid(True)
                if len(traces) > 1:
                    ax.legend()
                
                # Optimize figure updates
                fig.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            except Exception as e:
                print(f"Error updating plot for {scope_name}: {e}")
                continue
            
        plt.pause(0.01)  # Single pause after all plots are updated

    def run_acquisition(self):
        """Main acquisition loop"""
        try:
            # Initialize plots and HDF5 file
            plt.ion()  # Interactive mode on
            scope_group = self.initialize_hdf5()

            active_scopes = []  # Keep track of scopes that have valid data
            
            # Main acquisition loop
            for shot in range(self.num_loops):
                print(f"Starting acquisition shot {shot+1}/{self.num_loops}")
                start_time = time.time()
                
                all_data = {}
                
                # Acquire data from each scope sequentially
                for name, scope in self.scopes.items():
                    print(f"\nAcquiring data from {name}...")
                    
                    # First shot: get time arrays and validate scopes
                    if shot == 0:
                        traces, data, headers, time_array = acquire_from_scope(scope, name, first_acquisition=True)
                        self.save_time_arrays(scope_group, time_array=time_array)
                        active_scopes.append(name)

                    else:
                        # Subsequent shots: only acquire from active scopes
                        if name in active_scopes:
                            traces, data, headers = acquire_from_scope(scope, name, first_acquisition=False)
                        else:
                            continue
                    
                    if traces:  # Only add to all_data if we got valid traces
                        all_data[name] = (traces, data, headers)
                
                if not all_data:
                    print(f"Warning: No valid data acquired for shot {shot+1}")
                    continue
                
                # Save data to HDF5
                self.update_hdf5(all_data, shot)
                
                # Update plots with downsampled data
                self.update_plots(all_data, shot)
                
                print(f"Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
                
                if shot == 0 and not active_scopes:
                    raise RuntimeError("No valid data found from any scope. Aborting acquisition.")
            
            # Keep figures open after acquisition
            plt.show(block=False)
            input("Press Enter to close figures and exit...")
            
        finally:
            plt.close('all')  # Ensure all figures are closed
            # Cleanup will be handled by __exit__ when using context manager


