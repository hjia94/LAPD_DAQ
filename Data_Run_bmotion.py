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
    get_storage_paths,
    load_experiment_config,
)
from acquisition.logging_utils import close_log_file_handlers
from acquisition.run_paths import resolve_run_paths
from acquisition.run_resume import (
    QUIT,
    apply_action,
    inspect_run,
    prompt_action,
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
# Free-text run description lives in its own file (not in experiment_config.ini),
# so it can be written before or during the run. It is written into the HDF5
# `description` attribute once at run start and overwritten at run end.
description_path = os.path.join(base_path, 'description.txt')

#===============================================================================================================================================
# Offload launch helper
#===============================================================================================================================================

def launch_offload(spool_dir, config_path):
    """Auto-launch Offload_Run.py in its own console, unless one already runs.

    The offload politely waits for run metadata (offload_runner._wait_for), so
    launching before metadata exists is safe; we detach (no wait) so it keeps
    draining after this acquire process exits. A live ``offload.lock`` in the
    spool means a previous run's offload is still attached to this subfolder (a
    resume reuses it), so we don't start a second one that would race it.
    """
    from spooling import spool_format

    if spool_format.offload_lock_is_live(spool_dir):
        print('  An offload process is already attached to this spool; '
              'not launching a second one.')
        return

    offload_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'Offload_Run.py')
    subprocess.Popen(
        [sys.executable, offload_script,
         '--spool-dir', spool_dir, '--config', config_path],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    print('  Launched Offload_Run.py in a new console window.')


#===============================================================================================================================================
# Main Data Run sequence
#===============================================================================================================================================
def main():
    # Create save directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # The experiment name and HDF5 filename come from the config, not a hard-coded
    # variable. resolve_run_paths keys identity on the name (globbing <name>_*),
    # so a run started before midnight and resumed the next day targets the same
    # HDF5 file + spool subfolder instead of silently starting a new pair.
    config, _ = load_experiment_config(config_path)
    nshots = config.getint('nshots', 'num_duplicate_shots', fallback=1)
    spool_root, _ = get_storage_paths(config)
    spooled = bool(spool_root)
    paths = resolve_run_paths(config, base_path, spool_root=spool_root)
    hdf5_path = paths.hdf5_path

    # Decide (inspect + prompt) and do (apply) are separated so this stays flat:
    # a fresh run skips straight through; an existing one resumes or restarts.
    start_shot = 1
    spool_dir = paths.spool_dir
    if paths.is_existing:
        state = inspect_run(paths)
        action = prompt_action(paths, state, nshots=nshots)
        if action == QUIT:
            print('Exiting.')
            sys.exit()
        plan = apply_action(action, paths, state, nshots=nshots)
        spool_dir, start_shot = plan.spool_dir, plan.start_shot
        print(f'{action.capitalize()}: starting at shot {start_shot}.')

    if spooled:
        # Acquisition writes the HDF5 skeleton + spools per-shot data; a separate
        # Offload_Run.py process fills the shots into the same HDF5.
        os.makedirs(spool_dir, exist_ok=True)
        print(f'PARALLEL mode: spooling shots to {spool_dir}')
        launch_offload(spool_dir, config_path)

    print('Data run started at', datetime.datetime.now())
    t_start = time.time()

    try:
        if spooled:
            run_acquisition_bmotion_spooled(spool_dir, hdf5_path, toml_path, config_path,
                                            start_shot=start_shot,
                                            description_path=description_path)
        else:
            run_acquisition_bmotion(hdf5_path, toml_path, config_path,
                                    start_shot=start_shot,
                                    description_path=description_path)

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
            # Non-spooled: the HDF5 is complete here, so auto-plot the line
            # profile (no-op on plane/single-point runs). maybe_autoplot is
            # itself swallow-all; the try here only guards the import so a
            # missing analysis package can't break run teardown.
            try:
                from read_and_analyze.auto_plot import maybe_autoplot
                maybe_autoplot(hdf5_path, config)
            except Exception as e:
                print(f"Warning: auto-plot hook failed to load: {e}")
        else:
            print(f'File "{hdf5_path}" was not created')

        # Release motor.log so a later restart isn't blocked by an open handle.
        close_log_file_handlers(logging.getLogger())


#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
    main()
