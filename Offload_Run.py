# -*- coding: utf-8 -*-
"""Offload companion to the Data_Run_*.py acquisition scripts (two-process mode).

Run this alongside (or after) the acquisition process. It watches the fast-disk
spool directory and fills each completed shot into the HDF5 file the acquisition
process already created on the slow/large disk, verifies every write by reading
it back, and deletes the spooled copy once verified. It exits cleanly when the
acquisition process drops the RUN_COMPLETE sentinel.

The destination HDF5 path is read verbatim from the spool run metadata
(recorded once by the acquire process); this script only needs spool_dir +
hdf5_dir present in the [storage] section of experiment_config.ini:

    [storage]
    spool_dir = D:\\spool          # fast local disk (e.g. SSD)
    hdf5_dir  = E:\\Shadow data\\Pat   # slow/large disk (e.g. USB 3.0)

Both Data_Run.py and Data_Run_bmotion.py now AUTO-LAUNCH this script in its own
console, so for a normal run you never start it by hand.

You only run it manually to DRAIN A LEFTOVER SPOOL into HDF5 -- e.g. bin files
left by an older Data_Run.py (before auto-launch existed), or a run whose offload
console was closed early. The offload is fully decoupled from acquisition: it
just drains whatever shots are in the spool and exits at RUN_COMPLETE, so it does
not matter that acquisition finished long ago. Point it at the spool subfolder:

    python Offload_Run.py --spool-dir "D:\\spool\\<exp-name>_<date>"

Requirements to drain a leftover spool:
  * the skeleton HDF5 recorded in the spool metadata must still exist at its
    original path (the acquire run created it; do not move/delete it), and
  * a RUN_COMPLETE sentinel must be present so the drain knows when to stop
    (the acquire run writes one when it ends, even on Ctrl-C).

Usage:
    python Offload_Run.py                              # spool from config [storage]
    python Offload_Run.py --spool-dir <leftover spool> # drain a specific spool
"""

import argparse
import datetime
import logging
import os
import sys
import time

from acquisition.config import (
    get_storage_paths,
    load_experiment_config,
)
from acquisition.logging_utils import close_log_file_handlers
from offload_engine import MetadataTimeout, run_offload
from spooling import spool_format

############################################################################################################################
'''
User set following
'''
DEFAULT_BASE_PATH = r"E:\Shadow data\Pat"


def _pause_before_exit():
    """Keep the auto-launched console window open so its output stays readable.

    Data_Run_bmotion.py spawns this script with CREATE_NEW_CONSOLE; that window
    closes the instant the process exits, hiding any error/summary. Block on
    input so the user can read it. Harmless when run manually in an existing
    terminal; skipped if stdin isn't interactive (e.g. piped/CI).
    """
    try:
        if sys.stdin and sys.stdin.isatty():
            input("\nPress Enter to close this window...")
    except (EOFError, OSError):
        pass


def _parse_args():
    p = argparse.ArgumentParser(
        description="Offload (drain) spooled shots into the HDF5. The Data_Run_*.py "
                    "scripts auto-launch this; run it by hand only to drain a "
                    "leftover spool (e.g. old bin files not yet in HDF5).")
    p.add_argument("--spool-dir", default=None,
                   help="Spool subfolder to drain (e.g. an old <exp>_<date> spool). "
                        "Defaults to the config [storage] spool_dir.")
    p.add_argument("--config", default=os.path.join(DEFAULT_BASE_PATH,
                                                     "experiment_config.ini"),
                   help="Path to experiment_config.ini.")
    p.add_argument("--list", action="store_true",
                   help="Don't drain: just list every spool folder under the "
                        "spool dir (or root) with the HDF5 it targets and its "
                        "state, so you can confirm which spool fills a missing "
                        "HDF5 before offloading.")
    return p.parse_args()


def _spool_subdirs(path):
    """Spool folders to inspect: ``path`` itself if it holds run metadata,
    else its immediate subdirectories that do (so a spool *root* lists each run).

    Returns a list of (spool_dir, has_metadata) so even a folder missing its
    meta_run.pkl is surfaced (it just can't be drained), rather than silently
    dropped — which is exactly the "why isn't my run listed?" footgun to avoid.
    """
    if spool_format.run_metadata_exists(path):
        return [(path, True)]
    out = []
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name)
    except OSError:
        return out
    for entry in entries:
        if entry.is_dir():
            out.append((entry.path, spool_format.run_metadata_exists(entry.path)))
    return out


def _list_spools(path):
    """Print, for each spool folder under ``path``, the HDF5 it targets + state.

    Read-only: opens no scope, writes nothing, drains nothing. The HDF5 path
    shown is read verbatim from each spool's own metadata (recorded once by the
    acquire run), so it is the authoritative answer to "which spool fills this
    file?" — matched by content, never guessed from names.
    """
    folders = _spool_subdirs(path)
    if not folders:
        print(f"No spool folders found under: {path}")
        return

    print(f"Spool folders under {path}:\n")
    for spool_dir, has_meta in folders:
        print(f"  {os.path.basename(spool_dir.rstrip(os.sep)) or spool_dir}")
        if not has_meta:
            print("    (no run metadata — not a drainable spool; skip)\n")
            continue
        meta = spool_format.read_run_metadata(spool_dir)
        hdf5_path = meta.get("hdf5_path")
        exists = bool(hdf5_path) and os.path.isfile(hdf5_path)
        pending = spool_format.pending_shot_count(spool_dir)
        complete = spool_format.read_run_complete(spool_dir)
        print(f"    -> HDF5:   {hdf5_path}  [{'EXISTS' if exists else 'MISSING'}]")
        print(f"       writer: {meta.get('writer')}   pending shots in spool: {pending}")
        if complete is None:
            print("       RUN_COMPLETE: no (acquire still running or was killed "
                  "before writing it)")
        else:
            early = " (terminated early)" if complete.get("terminated_early") else ""
            print(f"       RUN_COMPLETE: yes, final_shot_num="
                  f"{complete.get('final_shot_num')}{early}")
        # The actionable line: this is the exact command to fill that HDF5.
        print(f"       drain with: python Offload_Run.py --spool-dir \"{spool_dir}\"\n")


#===============================================================================================================================================
def main():
    args = _parse_args()
    config_path = args.config
    config, _ = load_experiment_config(config_path)
    cfg_spool_dir, _hdf5_dir = get_storage_paths(config)
    spool_dir = args.spool_dir or cfg_spool_dir

    # --list short-circuits: inspect spools and print their HDF5 targets, then
    # exit without opening/draining anything. Run this first when an HDF5 is
    # missing to confirm which spool fills it before committing to a drain.
    if args.list:
        if not spool_dir:
            print("No spool dir to list (give --spool-dir or set config [storage]).")
            sys.exit(1)
        _list_spools(spool_dir)
        sys.exit(0)

    # Only spool_dir is required: the offload reads the destination HDF5 path
    # from the spool metadata (meta["hdf5_path"]), never from the config's
    # hdf5_dir. When launched with --spool-dir we proceed even if the config
    # has no [storage] section.
    if not spool_dir:
        print("No spool_dir given (via --spool-dir or config [storage]). "
              f"Nothing to offload. (config: {config_path})")
        _pause_before_exit()
        sys.exit(1)

    # Log failures/issues to a file in the spool folder (mirrors the acquire
    # side's motor.log). A named logger keeps offload logging isolated from root.
    os.makedirs(spool_dir, exist_ok=True)
    log_path = os.path.join(spool_dir, "offload.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    off_logger = logging.getLogger("offload")
    off_logger.setLevel(logging.WARNING)
    off_logger.addHandler(handler)

    # The offload neither computes nor creates the destination path: the acquire
    # process records the exact file (and creates it) in the spool metadata.
    print('Offload started at', datetime.datetime.now())
    print(f'  spool_dir = {spool_dir}')
    print(f'  logging failures to {log_path}')

    # Echo the resolved HDF5 target (from THIS spool's own metadata) before
    # draining, so a mismatched/missing file is caught now rather than after a
    # long fill. The metadata being ABSENT here is normal and expected for the
    # auto-launched case: the offload is launched before acquire writes
    # meta_run.pkl (acquire writes it only after scope init), so run_offload
    # waits for it (bounded by MetadataTimeout). We therefore do NOT fail fast on
    # absence -- only the timeout (a spool ROOT or never-started run) is fatal,
    # handled below with the --list hint.
    if spool_format.run_metadata_exists(spool_dir):
        meta = spool_format.read_run_metadata(spool_dir)
        target = meta.get("hdf5_path")
        state = 'exists' if (target and os.path.isfile(target)) else 'MISSING - will be filled if skeleton present'
        print(f'  target HDF5 = {target}  [{state}]')
    else:
        print('  No run metadata yet; waiting for the acquire process to write it '
              '(normal when auto-launched).')

    t_start = time.time()

    hdf5_path = None
    try:
        run_offload(spool_dir, config=config)
    except MetadataTimeout as e:
        print(f'\n  ERROR: {e}')
        print('  This is not a drainable spool folder (no acquire process wrote '
              'metadata). If you pointed at a spool ROOT, list the runs under it '
              'and pick one:')
        print(f'    python Offload_Run.py --list --spool-dir "{spool_dir}"')
        _pause_before_exit()
        sys.exit(1)
    except KeyboardInterrupt:
        print('\n______Halted due to Ctrl-C______', '  at', time.ctime())
    except Exception as e:
        import traceback
        print(f'\n______Halted due to error: {str(e)}______', '  at', time.ctime())
        traceback.print_exc()
    finally:
        # Report the file the acquire side recorded (if metadata was written).
        if spool_format.run_metadata_exists(spool_dir):
            hdf5_path = spool_format.read_run_metadata(spool_dir).get("hdf5_path")
        print('Offload finished at', datetime.datetime.now())
        print('Time taken: %.2f hours' % ((time.time()-t_start)/3600))
        if hdf5_path and os.path.isfile(hdf5_path):
            size = os.stat(hdf5_path).st_size/(1024*1024)
            print(f'Wrote file "{hdf5_path}", {size:.1f} MB')
            # Spooled: the HDF5 is only complete once offload finishes, so this
            # is the right place to auto-plot the line profile (no-op on
            # plane/single-point runs). maybe_autoplot is itself swallow-all;
            # the try here only guards the import so a missing analysis package
            # can't break offload teardown.
            try:
                from read_and_analyze.auto_plot import maybe_autoplot
                maybe_autoplot(hdf5_path, config)
            except Exception as e:
                print(f"Warning: auto-plot hook failed to load: {e}")
        else:
            print(f'File "{hdf5_path}" was not created')
        # Close the log file BEFORE pausing: this process sits at the pause prompt
        # until the user closes the window, so an open offload.log handle would
        # otherwise block a later "restart" from rotating/removing the spool
        # folder on Windows (the dir rename fails while a child file is open).
        close_log_file_handlers(off_logger)
        _pause_before_exit()

#===============================================================================================================================================
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        _pause_before_exit()
        raise
