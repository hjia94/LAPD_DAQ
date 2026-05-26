# -*- coding: utf-8 -*-
"""
Multi-scope and high-speed camera data acquisition program with parallel arming support.
Run this program to acquire data from multiple scopes and Phantom camera, saving everything in a unified HDF5 file.

This combines the functionality of:
- multi_scope_acquisition.py for multiple scope data
- phantom_recorder.py for high-speed camera data

Main functions:
- run_acquisition_with_camera(): Basic multi-scope and camera acquisition
- run_acquisition_with_WDropper(): Multi-scope and camera acquisition with tungsten dropper

Configuration and metadata:
- Scope and channel descriptions, as well as experiment metadata, are now loaded from experiment_config.ini.
- Edit experiment_config.ini to set experiment description, scope descriptions, and channel descriptions.
- Use this script to set scope IP addresses, camera configuration, number of shots, delays, and file paths.

Created July.2025
@author: AI assistant based on Jia Han's Data_Run.py
"""

import datetime
import os
import numpy as np
from acquisition import MultiScopeAcquisition, load_experiment_config
from drivers.phantom_recorder import PhantomRecorder
import time
import sys
import h5py

from pi_gpio.pi_client import TungstenDropper, TriggerClient

MOTOR_IP = '192.168.7.99'
PI_HOST = '192.168.7.38'
PI_PORT = 54321

############################################################################################################################
'''
User: Set experiment name and path
'''
exp_name = '00_scope_cam_test'  # experiment name
date = datetime.date.today()
base_path = r"E:\Shadow data\Energetic_Electron_Ring\test"
hdf5_path = os.path.join(base_path, f"{exp_name}_{date}.hdf5")

config_path = os.path.join(base_path, 'experiment_config.ini')


#-------------------------------------------------------------------------------------------------------------
def get_camera_config(config):
    """
    Extract camera configuration from configparser.ConfigParser object.
    Returns a dictionary suitable for PhantomRecorder.
    Returns None if [camera_config] section is missing or disabled.
    """
    if not config.has_section('camera_config'):
        print("No [camera_config] section found in config. Camera will be disabled.")
        return None

    cam_config = {
        'exposure_us': config.getint('camera_config', 'exposure_us', fallback=30),
        'fps': config.getint('camera_config', 'fps', fallback=10000),
        'pre_trigger_frames': config.getint('camera_config', 'pre_trigger_frames', fallback=-500),
        'post_trigger_frames': config.getint('camera_config', 'post_trigger_frames', fallback=1000),
    }
    # Parse resolution as tuple
    res_str = config.get('camera_config', 'resolution', fallback='256,256')
    try:
        cam_config['resolution'] = tuple(int(x) for x in res_str.replace('x', ',').split(','))
    except Exception:
        print(f"Warning: Could not parse camera resolution '{res_str}', using (256, 256)")
        cam_config['resolution'] = (256, 256)

    cam_config['hdf5_file_path'] = hdf5_path
    cam_config['save_path'] = base_path
    return cam_config

#===============================================================================================================================================
# Enhanced acquisition function with camera integration
#===============================================================================================================================================
# Used for both run_acquisition_with_camera() and run_acquisition_with_WDropper()
config, raw_config_text = load_experiment_config(config_path)
scope_ips = dict(config.items('scope_ips')) if config.has_section('scope_ips') else {}
cam_config = get_camera_config(config)
num_shots = config.getint('nshots', 'num_duplicate_shots', fallback=1)  # Get from [nshots] section

def run_acquisition_with_camera(hdf5_path):
    """Run the main acquisition sequence with integrated camera recording
    
    Args:
        hdf5_path: Path to save HDF5 file

    """
    print('Starting multi-scope and camera acquisition at', time.ctime())
    
    camera_recorder = None  # Initialize to avoid NameError in finally block
    
    try:
        with MultiScopeAcquisition(hdf5_path, config, raw_config_text) as msa:
            print("Initializing HDF5 file structure...", end='')
            msa.initialize_hdf5_base()
            print("✓")

            if cam_config is not None:
                try:
                    print("Initializing Phantom camera...", end='')
                    camera_recorder = PhantomRecorder(cam_config)
                    print("✓")
                except Exception as e:
                    print(f"⚠ Camera initialization failed: {e}")
                    print("Continuing with scope-only acquisition...")
                    camera_recorder = None
            else:
                print("Camera recording disabled")
                camera_recorder = None

            for shot_num in range(1, num_shots + 1):
                try:
                    acquisition_loop_start_time = time.time()
                    print(f"\n______Acquiring shot {shot_num}/{num_shots}______")

                    if shot_num == 1:
                        print("\nStarting initial scope acquisition...")
                        active_scopes = msa.initialize_scopes()
                        if not active_scopes:
                            raise RuntimeError("No valid data found from any scope. Aborting acquisition.")
                        print(f"Active scopes: {list(active_scopes.keys())}")
                    else:
                        msa.arm_scopes_for_trigger(active_scopes)

                    if camera_recorder:
                        camera_recorder.start_recording(shot_num)
                        timestamp = camera_recorder.wait_for_recording_completion()

                    all_data = msa.acquire_shot(active_scopes, shot_num)

                    if camera_recorder:
                        filename = f"{exp_name}_shot{shot_num:03d}.cine"
                        ifn = os.path.join(base_path, filename)
                        rec_cine = camera_recorder.save_cine(ifn)

                    print("Saving scope data to HDF5")
                    msa.update_scope_hdf5(all_data, shot_num)

                    if camera_recorder:
                        camera_recorder.wait_for_save_completion(rec_cine)
                        camera_recorder._update_hdf5_metadata(shot_num, filename, timestamp)
                        print("Camera metadata saved to HDF5")

                    # Calculate remaining run time
                    time_per_shot = (time.time() - acquisition_loop_start_time)
                    remaining_shots = num_shots - shot_num
                    remaining_time = remaining_shots * time_per_shot
                    print(f' | Remaining: {remaining_time/60:.1f}min ({remaining_time:.1f}s)')

                except KeyboardInterrupt:
                    print(f'\n______Shot {shot_num} interrupted by Ctrl-C______')
                    raise

    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
        raise

    finally:
        # Cleanup camera
        if camera_recorder:
            try:
                camera_recorder.cleanup()
                print("Camera resources cleaned up successfully")
            except Exception as e:
                print(f"Error cleaning up camera: {e}")


def run_acquisition_with_WDropper(hdf5_path):
    """
    Main acquisition function with tungsten dropper and optional camera recording
    
    Args:
        save_path: Path to save HDF5 file
        scope_ips: Dictionary of scope IPs {scope_name: ip_address}
        cam_config: Camera configuration dictionary (None to disable camera)
    """
    print('Starting acquisition at', time.ctime())
    
    dropper = None
    trigger_client = None
    camera_recorder = None
    
    try:
        config, raw_config_text = load_experiment_config(config_path)
        # Initialize multi-scope acquisition (no motor control)
        with MultiScopeAcquisition(hdf5_path, config, raw_config_text) as msa:

            # Initialize HDF5 file structure (append mode since file already exists)
            print("Initializing HDF5 file structure...", end='')
            msa.initialize_hdf5_base()
            print("✓")

            # Initialize tungsten dropper
            print("Initializing tungsten dropper...")
            dropper = TungstenDropper(motor_ip=MOTOR_IP, timeout=15)
            trigger_client = TriggerClient(PI_HOST, PI_PORT)
            trigger_client.get_status()  # Test connection
            print("✓ Trigger client initialized")

            if cam_config is not None: # Initialize camera recorder if configured
                try:
                    print("Initializing Phantom camera...", end='')
                    camera_recorder = PhantomRecorder(cam_config)
                    print("✓")
                except Exception as e:
                    print(f"⚠ Camera initialization failed: {e}")
                    print("Continuing with scope-only acquisition...")
                    camera_recorder = None
            else:
                print("Camera recording disabled")
                camera_recorder = None
            
            for shot_num in range(1, num_shots + 1): 
                try:
                    acquisition_loop_start_time = time.time()
                    print(f"\n______Acquiring shot {shot_num}/{num_shots}______")

                    if shot_num == 1: # First shot: Initialize scopes and save time arrays
                        print("\nStarting initial scope acquisition...")
                        trigger_client.send_trigger()
                        print("Trigger sent for getting scope time array")
                        active_scopes = msa.initialize_scopes()
                        if not active_scopes:
                            raise RuntimeError("No valid data found from any scope. Aborting acquisition.")
                        print(f"Active scopes: {list(active_scopes.keys())}")
                    else:
                        msa.arm_scopes_for_trigger(active_scopes) # Arm scopes for trigger

                    # Load tungsten ball and send trigger
                    print("Loading tungsten ball...")
                    dropper.load_ball()
                    print("Sending trigger signal...")
                    trigger_client.send_trigger()

                    if camera_recorder:
                        camera_recorder.start_recording(shot_num)
                        timestamp = camera_recorder.wait_for_recording_completion()
                    
                    all_data = msa.acquire_shot(active_scopes, shot_num)

                    if camera_recorder:
                        filename = f"{exp_name}_shot{shot_num:03d}.cine"
                        ifn = os.path.join(base_path, filename)
                        rec_cine = camera_recorder.save_cine(ifn)

                    print("Saving scope data to HDF5")
                    msa.update_scope_hdf5(all_data, shot_num)

                    if camera_recorder:
                        camera_recorder.wait_for_save_completion(rec_cine)
                        camera_recorder._update_hdf5_metadata(shot_num, filename, timestamp)
                        print("Camera metadata saved to HDF5")

                    # Calculate remaining run time
                    time_per_shot = (time.time() - acquisition_loop_start_time)
                    remaining_shots = num_shots - shot_num
                    remaining_time = remaining_shots * time_per_shot
                    print(f' | Remaining: {remaining_time/60:.1f}min ({remaining_time:.1f}s)')

                except KeyboardInterrupt:
                    print(f'\n______Shot {shot_num} interrupted by Ctrl-C______')
                    raise
    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
        raise

    finally:
        # Cleanup all resources
        print("\n=== Cleaning up resources ===")
        
        # Cleanup camera
        if camera_recorder:
            try:
                camera_recorder.cleanup()
                print("✓ Camera resources cleaned up successfully")
            except Exception as e:
                print(f"⚠ Error cleaning up camera: {e}")

#===============================================================================================================================================
# Main Data Run sequence
#===============================================================================================================================================
def main():
    # Create save directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    # Check if file already exists
    if os.path.exists(hdf5_path):
        while True:
            response = input(f'File "{hdf5_path}" already exists. Overwrite? (y/n): ').lower()
            if response in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")
            
        if response == 'n':
            print('Exiting without overwriting existing file')
            sys.exit()
        else:
            print('Overwriting existing file')
            os.remove(hdf5_path)  # Delete the existing file
    
    print('=== Multi-Scope and Camera Data Acquisition ===')
    print(f'Experiment: {exp_name}')
    print(f'Base path: {base_path}')
    print(f'HDF5 file: {hdf5_path}')
    print(f'Scopes: {list(scope_ips.keys())}')

    if cam_config:
        print(f'Camera settings:')
        print(f'  Resolution: {cam_config["resolution"]}')
        print(f'  Frame rate: {cam_config["fps"]} fps')
        print(f'  Frames: {cam_config["pre_trigger_frames"]} to +{cam_config["post_trigger_frames"]}')
    else:
        print('Camera recording disabled')

    print(f'Total shots: {num_shots}')
    
    t_start = time.time()
    
    try:
        run_acquisition_with_camera(hdf5_path)
        
    except KeyboardInterrupt:
        print('\n' + '='*60)
        print('______Acquisition INTERRUPTED by Ctrl-C______', '  at', time.ctime())
        print('='*60)
    except Exception as e:
        print('\n' + '='*60)
        print(f'______Acquisition FAILED due to error: {str(e)}______', '  at', time.ctime())
        print('='*60)
        import traceback
        traceback.print_exc()
    finally:
        print('\n' + '='*50)
        print('Data run finished at', datetime.datetime.now())
        print('Time taken: %.2f minutes' % ((time.time()-t_start)/60))
        
        # Print file size if it was created
        if os.path.isfile(hdf5_path):
            size = os.stat(hdf5_path).st_size/(1024*1024)
            print(f'Wrote file "{hdf5_path}", {size:.1f} MB')
        else:
            print(f'File "{hdf5_path}" was not created')
        print('='*50)

#===============================================================================================================================================
if __name__ == '__main__':
    main()
