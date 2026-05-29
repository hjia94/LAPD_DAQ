# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program with probe movement using bapsf_motion library.
Run this program to acquire data from multiple scopes and save it in an HDF5 file.
Result is plotted in real time.

The user should edit this file to:


Created on July 24.2025
@author: Jia Han

Parallel mode: if experiment_config.ini has a [storage] section with a fast
spool_dir, this process creates the HDF5 file + its skeleton (metadata, time
arrays, positions) on the slow/large disk and then spools each shot's raw data
to the fast disk; a separate `Offload_Run.py` process fills those shots into the
same HDF5 file. Otherwise the legacy single-process path writes the HDF5 directly.
"""

import datetime
import os
import time
import sys
import logging
import subprocess

from acquisition import run_acquisition_bmotion, run_acquisition_bmotion_spooled
from acquisition.config import (
    get_experiment_name,
    get_storage_paths,
    load_experiment_config,
    resolve_hdf5_path,
)

logging.basicConfig(
    filename='motor.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s',
)

############################################################################################################################
'''
User sets only the base path below. The experiment name lives in
experiment_config.ini ([experiment] name = ...); the config and bmotion TOML
are found inside base_path, and the HDF5 filename is built from the parsed
experiment name after the config is read.
'''
base_path = r"E:\Shadow data\Pat"
config_path = os.path.join(base_path, 'experiment_config.ini')
toml_path = os.path.join(base_path, 'bmotion_config.toml')

#===============================================================================================================================================
# Partial-run detection and user prompt helpers
#===============================================================================================================================================

def _find_latest_spool_subdir(spool_root):
    """Return the most recently modified timestamped subfolder in spool_root, or None.

    Each run creates a subfolder named ``YYYYMMDD_HHMMSS`` inside the spool root.
    On restart/resume detection we look at the most recent one to find prior run
    artifacts, rather than the new (not-yet-created) timestamped folder.
    """
    if not os.path.isdir(spool_root):
        return None
    subdirs = [
        os.path.join(spool_root, d)
        for d in os.listdir(spool_root)
        if os.path.isdir(os.path.join(spool_root, d))
    ]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def _check_partial_run(hdf5_path, spool_subdir):
    """Inspect an existing HDF5 (and optional spool subfolder) for a partial run.

    ``spool_subdir`` should be the most recent run's subfolder (from
    ``_find_latest_spool_subdir``), not the spool root or the new timestamped dir.

    Returns a dict with:
      - 'is_partial': True if the previous run was terminated early
      - 'resume_from_shot': shot number to continue from (last completed + 1)
      - 'completed_shots': number of shots written so far
      - 'abort_reason': why it stopped, or None
      - 'spool_subdir': the subfolder that contains the prior run's artifacts
    """
    info = {
        'is_partial': False,
        'resume_from_shot': 1,
        'completed_shots': 0,
        'abort_reason': None,
        'spool_subdir': spool_subdir,
    }

    if spool_subdir is not None:
        from spooling import spool_format as _sf
        complete = _sf.read_run_complete(spool_subdir)
        if complete is not None:
            if complete.get('terminated_early'):
                final = int(complete.get('final_shot_num', 0))
                info['is_partial'] = True
                info['completed_shots'] = final
                info['resume_from_shot'] = final + 1
                info['abort_reason'] = complete.get('abort_reason')
            # else: clean RUN_COMPLETE means the run finished normally;
            # treat as completed so we only offer restart/quit.
            return info

        # No RUN_COMPLETE but spool metadata exists with leftover shots →
        # process was killed before writing the sentinel.
        if _sf.run_metadata_exists(spool_subdir):
            pending = _sf.iter_ready_shots(spool_subdir)
            if pending:
                last = max(pending)
                info['is_partial'] = True
                info['completed_shots'] = last
                info['resume_from_shot'] = last + 1
                info['abort_reason'] = 'process killed before RUN_COMPLETE was written'
            return info

    # Non-spooled or no spool artifacts: flag as partial without shot count.
    info['is_partial'] = True
    info['abort_reason'] = 'previous run was interrupted'
    return info


def _prompt_existing_file(hdf5_path, partial_info):
    """Ask the user what to do with an existing HDF5 file.

    Returns one of 'resume', 'restart', or 'exit'.
    For non-partial (completed) runs only 'restart'/'exit' are offered.
    """
    if partial_info['is_partial']:
        shots_done = partial_info['completed_shots']
        resume_from = partial_info['resume_from_shot']
        reason = partial_info['abort_reason'] or 'terminated early'
        print(f'\nFile "{hdf5_path}" exists from a partially completed run.')
        print(f'  Reason stopped : {reason}')
        print(f'  Shots completed: {shots_done}  (next shot would be {resume_from})')
        print()
        print('  r = Resume  – continue acquisition from shot', resume_from)
        print('  n = Restart – delete existing file and start fresh')
        print('  q = Quit    – exit without changes')
        valid = {'r': 'resume', 'n': 'restart', 'q': 'exit'}
        prompt = 'Choice (r/n/q): '
    else:
        print(f'\nFile "{hdf5_path}" already exists (previous run completed normally).')
        print('  n = Overwrite – delete existing file and start fresh')
        print('  q = Quit      – exit without changes')
        valid = {'n': 'restart', 'q': 'exit'}
        prompt = 'Choice (n/q): '

    while True:
        response = input(prompt).strip().lower()
        if response in valid:
            return valid[response]
        print(f"Please enter one of: {', '.join(valid)}")


#===============================================================================================================================================
# Main Data Run sequence
#===============================================================================================================================================
def main():
    # Create save directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Parse the config first; the experiment name and the HDF5 filename come
    # from it (not from a hard-coded variable in this script). The output file
    # is <hdf5_dir or base_path>/<exp_name>_<date>.hdf5.
    config, _ = load_experiment_config(config_path)
    hdf5_path = resolve_hdf5_path(config, base_path)

    # Parallel mode is enabled when [storage] provides a fast spool_dir.
    # Each run gets its own timestamped subfolder inside the spool root so runs
    # never share artifacts. The root comes from config; the subfolder is new.
    _spool_root, _ = get_storage_paths(config)
    spooled = bool(_spool_root)
    new_spool_dir = (
        os.path.join(_spool_root, f"{get_experiment_name(config)}_{datetime.date.today()}")
        if spooled else None
    )

    # Detect the state of any existing partial run before deciding what to do.
    # Look at the most recent existing subfolder, not the new (not-yet-created) one.
    start_shot = 1
    spool_dir = new_spool_dir  # default: fresh run uses the new timestamped dir
    if os.path.exists(hdf5_path):
        latest_subdir = _find_latest_spool_subdir(_spool_root) if spooled else None
        partial_info = _check_partial_run(hdf5_path, latest_subdir)
        action = _prompt_existing_file(hdf5_path, partial_info)

        if action == 'exit':
            print('Exiting.')
            sys.exit()
        elif action == 'resume':
            start_shot = partial_info['resume_from_shot']
            spool_dir = partial_info['spool_subdir']  # reuse the existing subfolder
            print(f'Resuming from shot {start_shot} in spool {spool_dir}.')
            if spooled:
                from spooling import spool_format as _sf
                _sf.clear_run_complete(spool_dir)
        else:  # 'restart'
            print('Restarting fresh.')
            os.remove(hdf5_path)
            # Leave old spool subfolders in place; they are self-contained and
            # harmless. The new run will write into a fresh timestamped subfolder.

    if spooled:
        # Acquisition writes the HDF5 skeleton + spools per-shot data; a separate
        # Offload_Run.py process fills the shots into the same HDF5.
        if not os.path.exists(spool_dir):
            os.makedirs(spool_dir)
        print(f'PARALLEL mode: spooling shots to {spool_dir}')
        # Auto-launch the offload in its own console window; it politely waits for
        # run metadata (offload_runner._wait_for), so launching before metadata
        # exists is safe. We detach (no wait) so it keeps draining after this
        # acquire process exits.
        offload_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'Offload_Run.py')
        subprocess.Popen(
            [sys.executable, offload_script,
             '--spool-dir', spool_dir, '--config', config_path],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        print('  Launched Offload_Run.py in a new console window.')

    print('Data run started at', datetime.datetime.now())
    t_start = time.time()

    try:
        if spooled:
            run_acquisition_bmotion_spooled(spool_dir, hdf5_path, toml_path, config_path,
                                            start_shot=start_shot)
        else:
            run_acquisition_bmotion(hdf5_path, toml_path, config_path,
                                    start_shot=start_shot)

    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
    except Exception as e:
        import traceback
        print(f'\n______Halted due to error: {str(e)}______', '  at', time.ctime())
        print("Full traceback:")
        traceback.print_exc()
    finally:
        print('Data run finished at', datetime.datetime.now())
        print('Time taken: %.2f hours' % ((time.time()-t_start)/3600))

        if spooled:
            print(f'Shots spooled to "{spool_dir}". The auto-launched '
                  f'Offload_Run.py console keeps draining into the HDF5 file.')
        elif os.path.isfile(hdf5_path):
            size = os.stat(hdf5_path).st_size/(1024*1024)
            print(f'Wrote file "{hdf5_path}", {size:.1f} MB')
        else:
            print(f'File "{hdf5_path}" was not created')


#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
    main()
