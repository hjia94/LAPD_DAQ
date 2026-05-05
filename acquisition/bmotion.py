import bapsf_motion as bmotion
import h5py
import numpy as np
import time
import traceback
import warnings
import xarray as xr

from typing import Dict

from .multi_scope_acquisition import (
    load_experiment_config,
    MultiScopeAcquisition,
    single_shot_acquisition,
)


def configure_bmotion_hdf5_group(
    hdf5_path: str,
    total_shots: int,
    n_motion_groups: int,
    toml_path: str,
    run_manager: bmotion.actors.RunManager,
    selected_mg_keys,
):
    with h5py.File(hdf5_path, 'a') as f:
        ctl_grp = f.require_group('Control')
        pos_grp = ctl_grp.require_group('Positions')
        
        # Store TOML configuration file
        config_grp = f.require_group('Configuration')
        with open(toml_path, 'r') as toml_file:
            bmotion_config_text = toml_file.read()
            config_grp.create_dataset('bmotion_config', data=np.string_(bmotion_config_text))
        
        # Save motion_list from each selected motion group directly under Control/Positions
        for mg_key in selected_mg_keys:
            mg = run_manager.mgs[mg_key]
            mg_name = mg.config['name']
            motion_list = mg.mb.motion_list
            
            # Create group for this motion group using the name as key directly under Positions
            mg_group = pos_grp.create_group(mg_name)
            mg_group.attrs['name'] = mg_name
            mg_group.attrs['key'] = str(mg_key)
            
            # Store motion list values
            ds = mg_group.create_dataset('motion_list', data=motion_list.values)
            
            # Create structured array to save actual achieved positions (like reference implementation)
            dtype = [('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4')]
            mg_group.create_dataset('positions_array', shape=(total_shots,), dtype=dtype)

        # No longer need the combined positions_array since each motion group has its own


def select_motion_groups(rm: bmotion.actors.RunManager):
    # Print and select from all available motion lists
    if not rm.mgs:
        raise RuntimeError("No motion groups found in TOML configuration")

    print(f"\nAvailable motion groups ({len(rm.mgs)}):")
    print(f"  Key : {'Name':<25} -- {'Size':<16}")
    for mg_key, mg in rm.mgs.items():
        motion_list_size = 0 if mg.mb.motion_list is None else mg.mb.motion_list.shape[0]
        print(
            f"{mg_key:>5} : {mg.config['name']:.25} -- {motion_list_size:6d} positions"
        )

    # Prompt user to select a motion list
    while True:
        selection = input(
            f"Select motion group(s) [^Key Column]:\n"
            f"  A - for all\n"
            f"  Space Separated Keys for each motion group (e.g. '1 3 5'): "
        )

        if selection.startswith("'") or selection.startswith('"'):
            selection = selection[1:]

        if selection.endswith("'") or selection.endswith('"'):
            selection = selection[:-1]

        if selection == "A":
            selection = list(rm.mgs.keys())
            break

        selection_items = selection.split()
        initial_selection = selection_items
        validated_selection = []
        
        for item in selection_items:
            if item == "":
                continue

            # First try the item as-is (string key)
            if item in rm.mgs:
                validated_selection.append(item)
                continue

            # Then try converting to integer
            try:
                item_int = int(item)
                if item_int in rm.mgs:
                    validated_selection.append(item_int)
                    continue
                else:
                    print(f"Motion Group key '{item}' not found in available groups.")
                    validated_selection = None
                    break
            except ValueError:
                print(f"Invalid motion group key '{item}'. Must be a valid key from the list above.")
                validated_selection = None
                break

        if validated_selection is None or len(validated_selection) == 0:
            print(f"Motion Group selection was invalid. Please SELECT AGAIN.\n")
            continue
        
        selection = validated_selection
        break

    # Ensure selection is always a list, even for single selections
    if not isinstance(selection, list):
        selection = [selection]
        
    return selection


def select_motion_list_order(rm: bmotion.actors.RunManager, order: Dict):
    for mg_key in list(order.keys()):  # Convert keys to list to avoid changing dict during iteration
        mg = rm.mgs[mg_key]

        while True:
            direction = input(
                f"Motion list direction is forward for '{mg.config['name']}'.\n"
                f"Press Enter to continue, or R + Enter to reverse: "
            ).strip().upper()

            if direction == "":
                # Forward direction (default)
                order[mg_key] = "forward"
                print("Motion list direction is forward.\n")
                break
            elif direction == "R":
                # Reverse direction
                order[mg_key] = "backward"
                print("Motion list direction is reversed.\n")
                break
            else:
                print(
                    f"Motion list direction selection was invalid '{direction}'.  "
                    f"Please press Enter for forward or 'R' + Enter for reverse. TRY AGAIN...\n"
                )
                continue

    return order


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
    time.sleep(.5)
    while rm.is_moving:
        time.sleep(.5)


def record_bmotion_positions(
    hdf5_path: str,
    shotnum: int,
    rm: bmotion.actors.RunManager,
    mg_keys,
) -> None:

    with h5py.File(hdf5_path, 'a') as f:
        for key in mg_keys:
            mg = rm.mgs[key]
            mg_name = mg.config['name']
            positions = mg.position.value

            # Access the positions_array for this specific motion group directly under Control/Positions
            dataset = f[f"Control/Positions/{mg_name}/positions_array"]
            
            # Record position for this shot using structured array format (shot_num is 1-based, array is 0-based)
            dataset[shotnum - 1] = (shotnum, positions[0], positions[1])


def run_acquisition_bmotion(hdf5_path, toml_path, config_path):
    print('Starting acquisition at', time.ctime())

    config = load_experiment_config(config_path)
    nshots = config.getint('nshots', 'num_duplicate_shots', fallback=1)

    print("Loading TOML configuration...", end='')
    run_manager = bmotion.actors.RunManager(toml_path, auto_run=True)
    print("✓")

    try:
        selection = select_motion_groups(run_manager)

        ml_order = dict(zip(selection, ["forward"] * len(selection)))
        ml_order = select_motion_list_order(run_manager, ml_order)
    except KeyboardInterrupt as err:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
        run_manager.terminate()
        raise KeyboardInterrupt from err

    max_ml_size = get_max_motion_list_size(run_manager, list(ml_order))

    print(f"Maximum motion list size is {max_ml_size}")
    print(f"Number of shots per position: {nshots}")
    total_shots = max_ml_size * nshots
    print(f"Total shots: {total_shots}")

    # Start acquisition loop
    with MultiScopeAcquisition(hdf5_path, config) as msa:
        try:
            print("Initializing HDF5 file...", end='')
            msa.initialize_hdf5_base()
            print("✓")

            print("\nStarting initial acquisition...")
            active_scopes = msa.initialize_scopes()
            if not active_scopes:
                raise RuntimeError(
                    "No valid data found from any scope. Aborting acquisition."
                )

            # create position group in hdf5
            configure_bmotion_hdf5_group(
                hdf5_path, total_shots, len(ml_order), toml_path, run_manager, list(ml_order.keys())
            )

            # Main acquisition loop
            shot_num = 1  # 1-based shot numbering
            for motion_index in range(max_ml_size):
                print(
                    f"\nMoving to position {motion_index + 1}/{max_ml_size}..."
                )

                move_to_index(
                    index=motion_index,
                    rm=run_manager,
                    ml_order_dict=ml_order,
                )

                # print motion group positions
                print("Current positions:")
                for mg_key in ml_order:
                    mg = run_manager.mgs[mg_key]
                    print(f"  '{mg.config['name']}'  : x={mg.position[0]:.2f}, y={mg.position[1]:.2f}")

                # Record data and positions
                for n in range(nshots):
                    acquisition_loop_start_time = time.time()
                    print(f"{n}.", end=' ')
                    try:
                        single_shot_acquisition(msa, active_scopes, shot_num)

                        # Update positions_array with actual achieved position
                        record_bmotion_positions(
                            hdf5_path=hdf5_path,
                            shotnum=shot_num,
                            rm=run_manager,
                            mg_keys=list(ml_order.keys()),
                        )

                    except (ValueError, RuntimeError) as e:
                        print(f'\nSkipping shot {shot_num} - {str(e)}')

                        # Create empty shot group with explanation
                        with h5py.File(hdf5_path, 'a') as f:
                            for scope_name in msa.scope_ips:
                                scope_group = f[scope_name]
                                # Check if shot group already exists
                                if f'shot_{shot_num}' not in scope_group:
                                    shot_group = scope_group.create_group(f'shot_{shot_num}')
                                    shot_group.attrs['skipped'] = True
                                    shot_group.attrs['skip_reason'] = str(e)
                                    shot_group.attrs['acquisition_time'] = time.ctime()

                            # Still update positions_array for skipped shots
                            record_bmotion_positions(
                                hdf5_path=hdf5_path,
                                shotnum=shot_num,
                                rm=run_manager,
                                mg_keys=list(ml_order.keys()),
                            )

                    except Exception as e:
                        print(f'\nMotion failed for shot {shot_num} - {str(e)}')

                        # Create empty shot group with explanation
                        with h5py.File(hdf5_path, 'a') as f:
                            for scope_name in msa.scope_ips:
                                scope_group = f[scope_name]
                                # Check if shot group already exists
                                if f'shot_{shot_num}' not in scope_group:
                                    shot_group = scope_group.create_group(f'shot_{shot_num}')
                                    shot_group.attrs['skipped'] = True
                                    shot_group.attrs['skip_reason'] = f"Motion failed: {str(e)}"
                                    shot_group.attrs['acquisition_time'] = time.ctime()

                            # Still update positions_array for failed shots
                            record_bmotion_positions(
                                hdf5_path=hdf5_path,
                                shotnum=shot_num,
                                rm=run_manager,
                                mg_keys=list(ml_order.keys()),
                            )
                    finally:
                        shot_num += 1  # Always increment shot number

                # Calculate and display remaining time
                if shot_num > 1:
                    time_per_shot = (time.time() - acquisition_loop_start_time)
                    remaining_shots = total_shots - shot_num
                    remaining_time = remaining_shots * time_per_shot
                    print(
                        f' | Remaining: {remaining_time / 3600:.2f}h '
                        f'({remaining_shots} shots)'
                    )
                else:
                    print()

        except KeyboardInterrupt as err:
            print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
            raise RuntimeError() from err
        finally:
            run_manager.terminate()
