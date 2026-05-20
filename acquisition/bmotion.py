import bapsf_motion as bmotion
import h5py
import json
import numpy as np
import time
import traceback
import warnings
import xarray as xr

from typing import Dict

from tqdm import tqdm

from .bmotion_config import resolve_bmotion_selection
from .config import load_experiment_config
from .scope_runner import MultiScopeAcquisition, single_shot_acquisition


_POSITION_DTYPE = [('shot_num', '>u4'), ('x', '>f4'), ('y', '>f4')]


def _build_setup_array(mg):
    """Validate `mg`'s motion list and return (setup_array, xpos, ypos).

    Mirrors the planned-positions layout from PositionManager's XY path:
    structured array of (shot_num,x,y) with `xpos`/`ypos` axis vectors.
    Rejects motion groups bmotion can't honestly represent in that layout
    (non-2D, non-(x,y) axes, non-rectangular grids).
    """
    name = mg.config['name']
    ml = mg.mb.motion_list
    arr = np.asarray(ml.values, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(
            f"bmotion motion group '{name}' has motion_list of ndim={arr.ndim}; expected 2."
        )
    N, M = arr.shape
    if M != 2:
        raise RuntimeError(
            f"bmotion currently supports 2D motion only; motion group '{name}' has M={M}"
        )
    axis_labels = tuple(str(s).lower() for s in ml.coords['space'].values)
    if axis_labels != ('x', 'y'):
        raise RuntimeError(
            f"bmotion expects axis labels ('x','y'); motion group '{name}' has {axis_labels}"
        )
    xpos = np.unique(arr[:, 0])
    ypos = np.unique(arr[:, 1])
    if len(xpos) * len(ypos) != N:
        raise RuntimeError(
            f"bmotion requires a rectangular grid; motion group '{name}' "
            f"has N={N} but len(xpos)*len(ypos)={len(xpos) * len(ypos)}"
        )
    setup = np.zeros(N, dtype=_POSITION_DTYPE)
    setup['shot_num'] = np.arange(1, N + 1)
    setup['x'] = arr[:, 0]
    setup['y'] = arr[:, 1]
    return setup, xpos, ypos


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
    prepared = []
    for mg_key in selected_mg_keys:
        mg = run_manager.mgs[mg_key]
        setup, xpos, ypos = _build_setup_array(mg)
        prepared.append((mg_key, mg.config['name'], setup, xpos, ypos))

    with h5py.File(hdf5_path, 'a') as f:
        ctl_grp = f.require_group('Control')
        pos_grp = ctl_grp.require_group('Positions')

        # Store TOML configuration file
        config_grp = f.require_group('Configuration')
        with open(toml_path, 'r') as toml_file:
            bmotion_config_text = toml_file.read()
            config_grp.create_dataset('bmotion_config', data=np.bytes_(bmotion_config_text))

        # Record exactly which subset/direction was used for this run.
        selection_blob = json.dumps({
            "mg_keys": [str(k) for k in selected_mg_keys],
            "direction": {str(k): v for k, v in (ml_order or {}).items()},
            "execution_order": execution_order,
        })
        config_grp.create_dataset('bmotion_selection', data=np.bytes_(selection_blob))

        for mg_key, mg_name, setup, xpos, ypos in prepared:
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


def _take_shots_at_position(
    msa,
    active_scopes,
    hdf5_path: str,
    run_manager: bmotion.actors.RunManager,
    record_keys,
    shot_num: int,
    nshots: int,
    pbar,
):
    """Acquire nshots at the current position, recording positions only for
    the motion groups in record_keys. Advances `pbar` once per shot and
    returns the next shot_num."""
    for n in range(nshots):
        try:
            single_shot_acquisition(msa, active_scopes, shot_num, verbose=False)
            record_bmotion_positions(
                hdf5_path=hdf5_path,
                shotnum=shot_num,
                rm=run_manager,
                mg_keys=record_keys,
            )
        except (ValueError, RuntimeError) as e:
            tqdm.write(f'Skipping shot {shot_num} - {str(e)}')
            with h5py.File(hdf5_path, 'a') as f:
                for scope_name in msa.scope_ips:
                    scope_group = f[scope_name]
                    if f'shot_{shot_num}' not in scope_group:
                        shot_group = scope_group.create_group(f'shot_{shot_num}')
                        shot_group.attrs['skipped'] = True
                        shot_group.attrs['skip_reason'] = str(e)
                        shot_group.attrs['acquisition_time'] = time.ctime()
                record_bmotion_positions(
                    hdf5_path=hdf5_path,
                    shotnum=shot_num,
                    rm=run_manager,
                    mg_keys=record_keys,
                )
        except Exception as e:
            tqdm.write(f'Motion failed for shot {shot_num} - {str(e)}')
            with h5py.File(hdf5_path, 'a') as f:
                for scope_name in msa.scope_ips:
                    scope_group = f[scope_name]
                    if f'shot_{shot_num}' not in scope_group:
                        shot_group = scope_group.create_group(f'shot_{shot_num}')
                        shot_group.attrs['skipped'] = True
                        shot_group.attrs['skip_reason'] = f"Motion failed: {str(e)}"
                        shot_group.attrs['acquisition_time'] = time.ctime()
                record_bmotion_positions(
                    hdf5_path=hdf5_path,
                    shotnum=shot_num,
                    rm=run_manager,
                    mg_keys=record_keys,
                )
        finally:
            shot_num += 1
            pbar.update(1)
    return shot_num


def _run_interleaved(msa, active_scopes, hdf5_path, run_manager, ml_order, nshots, total_shots):
    max_ml_size = get_max_motion_list_size(run_manager, list(ml_order))
    shot_num = 1
    record_keys = list(ml_order.keys())
    with tqdm(total=total_shots, desc="Shots", unit="shot") as pbar:
        for motion_index in range(max_ml_size):
            tqdm.write(f"\nMoving to position {motion_index + 1}/{max_ml_size}...")
            move_to_index(index=motion_index, rm=run_manager, ml_order_dict=ml_order)

            tqdm.write("Current positions:")
            for mg_key in ml_order:
                mg = run_manager.mgs[mg_key]
                tqdm.write(f"  '{mg.config['name']}'  : x={mg.position[0]:.2f}, y={mg.position[1]:.2f}")

            shot_num = _take_shots_at_position(
                msa, active_scopes, hdf5_path, run_manager, record_keys, shot_num, nshots, pbar,
            )


def _run_sequential(msa, active_scopes, hdf5_path, run_manager, ml_order, nshots, total_shots):
    shot_num = 1
    with tqdm(total=total_shots, desc="Shots", unit="shot") as pbar:
        for mg_key, direction in ml_order.items():
            mg = run_manager.mgs[mg_key]
            ml_size = get_motion_list_size(run_manager, mg_key)
            tqdm.write(f"\n=== Starting motion group '{mg.config['name']}' "
                       f"(key={mg_key}, {ml_size} positions, {direction}) ===")

            single_group_order = {mg_key: direction}
            for motion_index in range(ml_size):
                tqdm.write(f"\n[{mg.config['name']}] Moving to position "
                           f"{motion_index + 1}/{ml_size}...")
                move_to_index(index=motion_index, rm=run_manager, ml_order_dict=single_group_order)

                tqdm.write(f"  '{mg.config['name']}'  : x={mg.position[0]:.2f}, y={mg.position[1]:.2f}")

                shot_num = _take_shots_at_position(
                    msa, active_scopes, hdf5_path, run_manager, [mg_key], shot_num, nshots, pbar,
                )


def run_acquisition_bmotion(hdf5_path, toml_path, config_path):
    print('Starting acquisition at', time.ctime())

    config, raw_config_text = load_experiment_config(config_path)
    nshots = config.getint('nshots', 'num_duplicate_shots', fallback=1)

    print("Loading TOML configuration...", end='')
    run_manager = bmotion.actors.RunManager(toml_path, auto_run=True)
    print("✓")

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

    with MultiScopeAcquisition(hdf5_path, config, raw_config_text) as msa:
        try:
            print("Initializing HDF5 file...", end='')
            msa.initialize_hdf5_base()
            print("✓")

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

            if execution_order == "sequential":
                _run_sequential(msa, active_scopes, hdf5_path, run_manager,
                                ml_order, nshots, total_shots)
            else:
                _run_interleaved(msa, active_scopes, hdf5_path, run_manager,
                                 ml_order, nshots, total_shots)

        except KeyboardInterrupt as err:
            print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
            raise RuntimeError() from err
        finally:
            run_manager.terminate()
