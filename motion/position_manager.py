"""
Position management module for LAPD DAQ system.
Created by JH, 2025 for centralizing position management and motor control.

Contains the PositionManager class for handling position arrays,
HDF5 position data storage, and position-related metadata.

TODO: boundary checking functions
"""

import os
import sys
import json
import socket
import time
import configparser

import numpy as np
import h5py
from .Motor_Control import Motor_Control_2D, Motor_Control_3D
from .Motor_Control_1D import Motor_Control

# Chunk for the append-only positions_array (see acquisition.spool_adapter for
# the matching offload writer); 1024 rows ~16 KB keeps per-shot appends in-chunk.
_POSITIONS_CHUNK = (1024,)


def _create_positions_array(group, dtype, planned_len, append_mode):
    """Create ``positions_array`` under ``group`` in the right mode.

    ``append_mode`` True (spooled path): empty + resizable, grown one row per
    recorded shot by the offload, padded to the planned size at finalize.
    False (legacy in-process path): pre-sized to ``planned_len`` and written by
    index via :meth:`PositionManager.update_position_hdf5`. One helper so the two
    create sites (45-degree and regular) don't each carry the branch.
    """
    if append_mode:
        group.create_dataset('positions_array', shape=(0,), maxshape=(None,),
                             dtype=dtype, chunks=_POSITIONS_CHUNK)
    else:
        group.create_dataset('positions_array', shape=(planned_len,), dtype=dtype)


# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def load_position_config(config_path):
    """
    Load position configuration from config file.
    
    Returns:
        tuple: (pos_config, is_45deg) where:
            - pos_config: dict containing parsed position parameters, or None if no config
            - is_45deg: bool indicating if this is a 45-degree probe acquisition
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'position' not in config or not dict(config.items('position')):
        return None, False  # No position config, stationary
    
    pos_config = {}
    for key, value in config.items('position'):
        try:
            if value is None or value.strip() == '':
                continue
            if ',' in value and not value.startswith('{'):
                # Comma-separated tuple, e.g. x_limits = -40,200
                pos_config[key] = tuple(float(x) if '.' in x else int(x) for x in value.split(','))
            elif value.lower() == 'none' or value == 'None':
                pos_config[key] = None
            elif value.startswith('{') and value.endswith('}'):
                # JSON dict, e.g. xstart = {"P16": -38, "P22": -18}
                pos_config[key] = json.loads(value)
            elif '.' in value:
                pos_config[key] = float(value)
            else:
                pos_config[key] = int(value)
        except Exception:
            # String values, e.g. probe_list = P16,P22,P29,P34,P42
            if ',' in value:
                pos_config[key] = [item.strip() for item in value.split(',')]
            else:
                pos_config[key] = value

    # probe_list is only present for 45-degree acquisitions
    is_45deg = 'probe_list' in pos_config
    return pos_config, is_45deg

# ============================================================================
# POSITION GENERATION FUNCTIONS
# ============================================================================

def get_positions_xy(config):
    """Generate the positions array for probe movement using config dict.
    Returns:
        tuple: (positions, xpos, ypos)
            - positions: Array of tuples (shot_num, x, y)
            - xpos: Array of x positions
            - ypos: Array of y positions
    """
    # Check required position keys are in config
    required_keys = ['nx', 'ny', 'xmin', 'xmax', 'ymin', 'ymax']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        sys.exit(f'Error: Missing required position keys in config: {missing_keys}')
        
    # Check shot parameters are in config
    if 'num_duplicate_shots' not in config or 'num_run_repeats' not in config:
        print("Warning: Shot parameters not found in config. Using defaults of 1.")
        config['num_duplicate_shots'] = 1
        config['num_run_repeats'] = 1
        
    nx = config['nx']
    ny = config['ny']
    xmin = config['xmin']
    xmax = config['xmax']
    ymin = config['ymin']
    ymax = config['ymax']
    num_duplicate_shots = config['num_duplicate_shots']
    num_run_repeats = config['num_run_repeats']
    if nx == 0 or ny == 0:
        sys.exit('Position array is empty.') 
        
    xpos = np.linspace(xmin, xmax, nx)
    ypos = np.linspace(ymin, ymax, ny)

    # Calculate total number of positions including duplicates and repeats
    total_positions = nx * ny * num_duplicate_shots * num_run_repeats

    # Allocate the positions array
    positions = np.zeros(total_positions, 
                        dtype=[('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4')])

    # Create rectangular shape position array
    index = 0
    for repeat_cnt in range(num_run_repeats):
        for y in ypos:
            for x in xpos:
                for dup_cnt in range(num_duplicate_shots):
                    positions[index] = (index + 1, x, y)
                    index += 1
                    
    return positions, xpos, ypos

def get_positions_xyz(config):
    """Generate the positions array for probe movement in 3D using config dict.
    Returns:
        tuple: (positions, xpos, ypos, zpos)
            - positions: Array of tuples (shot_num, x, y, z)
            - xpos: Array of x positions
            - ypos: Array of y positions
            - zpos: Array of z positions
    """
    # Check required position keys are in config
    required_keys = ['nx', 'ny', 'nz', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        sys.exit(f'Error: Missing required position keys in config: {missing_keys}')
        
    # Check shot parameters are in config
    if 'num_duplicate_shots' not in config or 'num_run_repeats' not in config:
        print("Warning: Shot parameters not found in config. Using defaults of 1.")
        config['num_duplicate_shots'] = 1
        config['num_run_repeats'] = 1
    
    nx = config['nx']
    ny = config['ny']
    nz = config['nz']
    xmin = config['xmin']
    xmax = config['xmax']
    ymin = config['ymin']
    ymax = config['ymax']
    zmin = config['zmin']
    zmax = config['zmax']
    num_duplicate_shots = config['num_duplicate_shots']
    num_run_repeats = config['num_run_repeats']
    if nx == 0 or ny == 0 or nz == 0:
        sys.exit('Position array is empty.')
        
    xpos = np.linspace(xmin, xmax, nx)
    ypos = np.linspace(ymin, ymax, ny) 
    zpos = np.linspace(zmin, zmax, nz)

    # Calculate total number of positions including duplicates and repeats
    total_positions = nx * ny * nz * num_duplicate_shots * num_run_repeats

    # Allocate the positions array
    positions = np.zeros(total_positions,
                        dtype=[('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])

    # Create 3D rectangular shape position array
    index = 0
    for repeat_cnt in range(num_run_repeats):
        for z in zpos:
            for y in ypos:
                for x in xpos:
                    for dup_cnt in range(num_duplicate_shots):
                        positions[index] = (index + 1, x, y, z)
                        index += 1
                    
    return positions, xpos, ypos, zpos

def get_positions_45deg(xstart, xstop, nx, nshots):
    """Generate the positions array for 45deg probe movement.
    Args:
        xstart: Starting x position
        xstop: Ending x position  
        nx: Number of x positions
        nshots: Number of shots per position
    Returns:
        tuple: (positions, xpos)
            - positions: Array of tuples (shot_num, x)
            - xpos: Array of x positions
    """
    import sys
    
    if nx == 0:
        sys.exit('Position array is empty.')
        
    xpos = np.linspace(xstart, xstop, nx)

    # allocate positions array, fill it with zeros
    positions = np.zeros((nx*nshots), dtype=[('shot_num', np.int32), ('x', np.float64)])

    #create rectangular shape position array with height z
    index = 0

    for x in xpos:
        for dup_cnt in range(nshots):
            positions[index] = (index+1, x)
            index += 1
                        
    return positions, xpos

def create_all_positions_45deg(pr_ls, xstart, xstop, nx, nshots):
    """Create position array for all 45deg probes.
    Args:
        pr_ls: List of probe names (e.g., ['P16', 'P22', 'P29', 'P34', 'P42'])
        xstart: Dict of starting x positions for each probe
        xstop: Dict of ending x positions for each probe
        nx: Number of x positions
        nshots: Number of shots per position
    Returns:
        tuple: (positions, xpos)
            - positions: Dict of position arrays for each probe
            - xpos: Dict of x position arrays for each probe
    """
    positions = {}
    xpos = {}

    for pr in pr_ls:
        positions[pr], xpos[pr] = get_positions_45deg(xstart[pr], xstop[pr], nx, nshots)

    return positions, xpos

# ============================================================================
# BOUNDARY CHECKING FUNCTIONS
# ============================================================================

def outer_boundary(x, y, z, config):
    """Return True if (x, y, z) is within the probe limits (x/y/z_limits)."""
    x_limits = config['x_limits']
    y_limits = config['y_limits']
    z_limits = config['z_limits']
    return (x_limits[0] <= x <= x_limits[1] and
            y_limits[0] <= y <= y_limits[1] and
            z_limits[0] <= z <= z_limits[1])

def obstacle_boundary(x, y, z, config):
    """Return True if (x, y, z) is clear of the hard-coded box obstacle."""
    in_obstacle = ( -60 <= x <= -17 and
                    -2.5 <= y <= 5 and
                    -6.5 <= z <= 9)
    return not in_obstacle

def motor_boundary(x, y, z, config):
    """Return True if (x, y, z) is within the motor limits (xm/ym/zm_limits)."""
    xm_limits = config['xm_limits']
    ym_limits = config['ym_limits']
    zm_limits = config['zm_limits']
    in_outer_boundary = (xm_limits[0] <= x <= xm_limits[1] and
                        ym_limits[0] <= y <= ym_limits[1] and
                        zm_limits[0] <= z <= zm_limits[1])
    return in_outer_boundary

def motor_boundary_2D(x, y, z, config):
    """Return True if (x, y) is within the motor limits (z unconstrained)."""
    xm_limits = config['xm_limits']
    ym_limits = config['ym_limits']
    in_outer_boundary = (xm_limits[0] <= x <= xm_limits[1] and 
                        ym_limits[0] <= y <= ym_limits[1] and
                        -999 <= z <= 999)
    return in_outer_boundary



# ============================================================================
# MAIN POSITION MANAGER CLASS
# ============================================================================

class PositionManager:
    """Handles position arrays, HDF5 position data, and position-related metadata"""
    
    def __init__(self, save_path, config_path, num_duplicate_shots=1, num_run_repeats=1):
        """
        Args:
            save_path: Path to HDF5 file
            config_path: Path to experiment_config.ini file
            num_duplicate_shots: Number of shots per position
            num_run_repeats: Number of times to repeat the entire run
        """
        self.save_path = save_path
        self.config_path = config_path
        self.num_duplicate_shots = num_duplicate_shots
        self.num_run_repeats = num_run_repeats
        
        self.pos_config, self.is_45deg = load_position_config(config_path)

        # Full config is kept around for the motor IPs
        self.full_config = configparser.ConfigParser()
        self.full_config.read(config_path)

        # nz drives the 2D-vs-3D branch throughout the class; normalize 'None' to None
        if self.pos_config:
            nz_value = self.pos_config.get('nz', None)
            self.nz = None if (nz_value is None or nz_value == 'None') else nz_value
        else:
            self.nz = None
            
        # Get positions (or create stationary shots)
        positions_result = self.get_positions()
        if self.is_45deg:
            self.positions, self.xpos = positions_result
            self.ypos = None
            self.zpos = None
        elif self.nz is None:
            # For 2D movement, get_positions_xy returns (positions, xpos, ypos)
            self.positions, self.xpos, self.ypos = positions_result
            self.zpos = None
        else:
            # For 3D movement, get_positions_xyz returns (positions, xpos, ypos, zpos)
            self.positions, self.xpos, self.ypos, self.zpos = positions_result
        
    def get_positions(self):
        """Get position arrays based on acquisition type"""
        # Create a dictionary with just the parameters needed for position generation
        pos_params = {}
        if self.pos_config:
            # Copy the position parameters
            pos_params.update(self.pos_config)
            
            # Add shot parameters
            pos_params['num_duplicate_shots'] = self.num_duplicate_shots
            pos_params['num_run_repeats'] = self.num_run_repeats
            
        if self.is_45deg:
            # Get 45deg specific parameters from config
            pr_ls = self.pos_config.get('probe_list', ['P16', 'P22', 'P29', 'P34', 'P42'])
            xstart = self.pos_config.get('xstart', {})
            xstop = self.pos_config.get('xstop', {})
            nx = self.pos_config.get('nx', 0)
            
            # Use the num_duplicate_shots passed from the acquisition package
            return create_all_positions_45deg(pr_ls, xstart, xstop, nx, self.num_duplicate_shots)
        else:
            if self.nz is None:
                return get_positions_xy(pos_params)
            else:
                return get_positions_xyz(pos_params)
    
    def initialize_position_hdf5(self, append_mode=False):
        """Initialize HDF5 position structure and return positions.

        ``append_mode`` controls how ``positions_array`` (the RECORDED per-shot
        positions, distinct from the pre-filled ``positions_setup_array``) is
        created:

        * ``False`` (default, legacy in-process path): pre-sized to the planned
          position count and written by index via :meth:`update_position_hdf5`.
        * ``True`` (spooled grid path): created empty and resizable, grown one
          row per recorded shot by the offload; the offload's finalize pads it
          back to the planned size with zero-fill, so the final layout matches
          the legacy one. Use this whenever the offload owns the per-shot write.
        """
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
            # Create Control/Positions group and datasets
            if '/Control' not in f:
                ctl_grp = f.create_group('/Control')
            else:
                ctl_grp = f['/Control']
                
            if 'Positions' not in ctl_grp:
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
                        
                        # Create array for actual recorded positions
                        _create_positions_array(
                            probe_grp, dtype[probe], len(positions[probe]), append_mode)
                else:
                    # Create positions setup array with metadata
                    pos_ds = pos_grp.create_dataset('positions_setup_array', data=positions, dtype=dtype)
                    pos_ds.attrs['xpos'] = xpos
                    if not self.is_45deg:
                        pos_ds.attrs['ypos'] = ypos
                        if self.nz is not None:
                            pos_ds.attrs['zpos'] = zpos
                    
                    # Create positions array for actual recorded positions
                    _create_positions_array(pos_grp, dtype, len(positions), append_mode)

        return positions
    
    def update_position_hdf5(self, shot_num, positions):
        """Update HDF5 position arrays with current shot position data"""
        if positions is None:
            return
            
        with h5py.File(self.save_path, 'a') as f:
            if self.is_45deg:
                # For 45-degree probes, update each probe's position separately
                for probe, pos in positions.items():
                    if pos is not None:  # Only update if we have valid position data
                        pos_arr = f[f'/Control/Positions/{probe}/positions_array']
                        pos_arr[shot_num-1] = (shot_num, pos)
            else:
                # For regular XY/XYZ acquisition
                pos_arr = f['/Control/Positions/positions_array']
                
                if self.nz is None:
                    # 2D case - need x and y
                    pos_arr[shot_num-1] = (shot_num, positions['x'], positions['y'])
                else:
                    # 3D case - need x, y, and z
                    pos_arr[shot_num-1] = (shot_num, positions['x'], positions['y'], positions['z'])

    # ============================================================================
    # MOTOR CONTROL FUNCTIONS
    # ============================================================================
    def initialize_motor(self):
        """Initialize motor control based on config dict."""
        nz = self.nz

        # Get motor IPs directly from the full config
        if 'motor_ips' not in self.full_config:
            print("\nError: No [motor_ips] section found in the configuration file")
            print("  - Please add a [motor_ips] section to your configuration")
            return None
            
        # Parse and clean motor IPs
        motor_ips = {}
        for key, value in self.full_config.items('motor_ips'):
            # Skip empty values or comments
            if not value or value.strip() == '' or value.strip().startswith('#'):
                continue
            motor_ips[key] = value.strip()  # Remove leading/trailing whitespace
            
        if not motor_ips:
            print("\nError: No motor IPs found in the configuration")
            print("  - Please uncomment or add motor IP addresses in the [motor_ips] section")
            return None
        
        print("Initializing motor...")

        try:
            if nz is None:
                print("XY drive in use")
                # Ensure we have the required IPs for x and y motors
                if 'x' not in motor_ips or 'y' not in motor_ips:
                    print(f"Error: Missing required motor IPs for XY drive")
                    return None
                
                x_ip = motor_ips['x']
                y_ip = motor_ips['y']
                
                # Create motor controller
                mc = Motor_Control_2D(x_ip, y_ip)
                print(f"  - Connected to X motor at {x_ip}")
                print(f"  - Connected to Y motor at {y_ip}")
            else:
                print("XYZ drive in use")
                # Ensure we have the required IPs for x, y, and z motors
                if 'x' not in motor_ips or 'y' not in motor_ips or 'z' not in motor_ips:
                    print(f"Error: Missing required motor IPs for XYZ drive")
                    return None
                
                x_ip = motor_ips['x']
                y_ip = motor_ips['y']
                z_ip = motor_ips['z']
                
                # Create motor controller
                mc = Motor_Control_3D(x_ip, y_ip, z_ip)
                print(f"  - Connected to X motor at {x_ip}")
                print(f"  - Connected to Y motor at {y_ip}")
                print(f"  - Connected to Z motor at {z_ip}")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except ConnectionRefusedError as e:
            print(f"Connection refused to motor controller: {str(e)}")
            print("  - Please check that the motor controllers are powered on")
            print("  - Verify the IP addresses are correct and can be reached")
            print("  - Check network connectivity to the motor controller devices")
            return None
        except socket.error as e:
            print(f"Socket error connecting to motor controller: {str(e)}")
            print("  - Please check network configuration")
            print("  - Verify the IP addresses are correct and can be reached")
            return None

        return mc

    def initialize_motor_45deg(self):
        """Initialize motor control for 45deg probes based on config dict."""
        # Get motor IPs directly from the full config
        if 'motor_ips' not in self.full_config:
            print("\nError: No [motor_ips] section found in the configuration file")
            return None
            
        # Parse and clean motor IPs
        motor_ips = {}
        for key, value in self.full_config.items('motor_ips'):
            motor_ips[key] = value.strip()  # Remove leading/trailing whitespace
            
        if not motor_ips:
            print("\nError: No motor IPs found in the configuration")
            return None
        
        print("Initializing motors...")
        motors = {}

        
        for probe in ['P16', 'P22', 'P29', 'P34', 'P42']: # Try to connect to each probe's motor
            try:
                if probe == 'P29': # Special case for P29 which has different cm_per_turn
                    motors[probe] = Motor_Control(server_ip_addr=motor_ips[probe], name=f"probe_{probe}", 
                                    stop_switch_mode=2, cm_per_turn=0.127)
                else:
                    motors[probe] = Motor_Control(server_ip_addr=motor_ips[probe], name=f"probe_{probe}", 
                                    stop_switch_mode=2)
                    
                # Set motor speed if connection successful
                motors[probe].motor_speed = 4
                print(f"Connected to {probe} motor")
                
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(f"Could not connect to {probe} motor: {str(e)}")
                motors[probe] = None
                
        return motors

    def move_45deg_probes(self, mc_list, move_to_list):
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