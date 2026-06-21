# -*- coding: utf-8 -*-
"""
Multi-scope data acquisition program with probe movement using bapsf_motion library.
Run this program to acquire data from multiple scopes and save it in an HDF5 file.
When the run finishes, the offload process auto-plots the 1D line profile (saved
as PNGs next to the data file) for line runs; disable via [analysis] auto_plot.

The user edits the base_path below; everything else lives in experiment_config.ini.

Created on July 24.2025
@author: Jia Han

Spooled (parallel) mode is the only mode: experiment_config.ini must have a
[storage] section with a fast spool_dir. This process creates the HDF5 file +
its skeleton (metadata, time arrays, positions) on the slow/large disk and then
spools each shot's raw data to the fast disk; a separate `Offload_Run.py`
process fills those shots into the same HDF5 file. (The legacy single-process,
non-spooled path was removed; recover it from git history if ever needed.)
"""

import datetime
import os
import time
import sys
import logging
import subprocess

from acquisition import run_acquisition_bmotion_spooled
from acquisition.config import (
    get_storage_paths,
    load_experiment_config,
    validate_bmotion_ini,
)
from acquisition.config_errors import ConfigError
from acquisition.logging_utils import close_log_file_handlers
from acquisition.run_paths import resolve_run_paths
from acquisition.run_resume import (
    QUIT,
    apply_action,
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
base_path = r"E:\Shadow data\Electrode_Biasing\jun2026"
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
    """Auto-launch Offload_Run.py in its own console.

    The offload politely waits for run metadata (offload_engine._wait_for), so
    launching before metadata exists is safe; we detach (no wait) so it keeps
    draining after this acquire process exits.
    """
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
# Config-error reporting
#===============================================================================================================================================

def report_config_error(err):
    """Print a clear, boxed message for an INI/TOML configuration error.

    The boxed message already names the file and location; the exception's own
    ``edit_instruction`` adds the file-specific "what to do next" line, so this
    reporter doesn't need to know which subclass it was handed.
    """
    print()
    print(err.format_for_terminal())
    if err.edit_instruction:
        print(err.edit_instruction)


#===============================================================================================================================================
# Main Data Run sequence
#===============================================================================================================================================
def main():
    # Create save directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # The experiment name and HDF5 filename come from the config, not a hard-coded
    # variable. resolve_run_paths keys identity on the name (globbing <name>_*),
    # so a run started before midnight and continued the next day still targets
    # the same HDF5 file + spool subfolder instead of silently starting a new pair.
    #
    # Load + validate the INI and prepare paths *before* launching the offload or
    # touching hardware. A mistake in experiment_config.ini is caught here and
    # reported below as an IniConfigError, so nothing is started against a broken
    # config and the user sees exactly which file/key is wrong.
    try:
        config, _ = load_experiment_config(config_path, required=True)
        validate_bmotion_ini(config, config_path)
        spool_root, _ = get_storage_paths(config)
        paths = resolve_run_paths(config, base_path, spool_root=spool_root)
    except ConfigError as e:
        report_config_error(e)
        sys.exit(1)

    hdf5_path = paths.hdf5_path

    # Prompt (ask) and apply (do) are separated so this stays flat: a fresh run
    # skips straight through; an existing one can only be restarted (delete +
    # redo from shot 1) or quit -- resume is not supported.
    spool_dir = paths.spool_dir
    if paths.is_existing:
        action = prompt_action(paths)
        if action == QUIT:
            print('Exiting.')
            sys.exit()
        spool_dir = apply_action(action, paths)
        print('Restart: starting fresh from shot 1.')

    # Acquisition writes the HDF5 skeleton + spools per-shot data; a separate
    # Offload_Run.py process fills the shots into the same HDF5.
    os.makedirs(spool_dir, exist_ok=True)
    print(f'PARALLEL mode: spooling shots to {spool_dir}')
    launch_offload(spool_dir, config_path)

    print('Data run started at', datetime.datetime.now())
    t_start = time.time()

    try:
        run_acquisition_bmotion_spooled(spool_dir, hdf5_path, toml_path, config_path,
                                        description_path=description_path)

    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
    except ConfigError as e:
        # A bad bmotion_config.toml surfaces here (the INI was already validated
        # above). Report it the same clean way -- the run never actually started.
        report_config_error(e)
    except Exception as e:
        import traceback
        print(f'\n______Halted due to error: {str(e)}______', '  at', time.ctime())
        print("Full traceback:")
        traceback.print_exc()
    finally:
        print('Data run finished at', datetime.datetime.now())
        print('Time taken: %.2f hours' % ((time.time()-t_start)/3600))

        print(f'Shots spooled to "{spool_dir}". The auto-launched '
              f'Offload_Run.py console keeps draining into the HDF5 file.')

        # Release motor.log so a later restart isn't blocked by an open handle.
        close_log_file_handlers(logging.getLogger())


#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
    main()
