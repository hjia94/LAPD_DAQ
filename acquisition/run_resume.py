"""Restart / quit decision logic for the spooled acquisition pipeline.

Resume was removed: a correct resume needs more care than the previous
position-boundary / provisional-sentinel attempt gave it, so this branch ships
the safe baseline only. An existing run can be **restarted** (delete the HDF5,
rotate its spool aside, re-run from shot 1) or left alone; there is no partial
continuation.

The "ask the operator / do the file surgery" split is kept as two functions so
the entry script (`Data_Run_bmotion.py`) stays a thin orchestrator -- and so a
future resume feature has a clean seam to slot a third "resume" action into:

    prompt_action(run_paths)  -> "restart" | "quit"   (ask)
    apply_action(action, …)   -> spool_dir             (do; side effects here)

Only the spool layer (``spooling.spool_format``) is imported, so this module is
light and does not pull in bmotion/hardware (preserving import hygiene).
"""

import os

from spooling import spool_format

RESTART = "restart"
QUIT = "quit"


def prompt_action(run_paths, *, input_fn=input) -> str:
    """Ask the operator what to do with an existing run. Returns an action.

    An existing run offers only Restart / Quit -- there is no resume.
    ``input_fn`` is injectable for testing.
    """
    name = run_paths.name
    fname = os.path.basename(run_paths.hdf5_path)

    print(f'\nRun "{name}" exists (file: {fname}).')
    if run_paths.ambiguous:
        print("  NOTE: multiple files match this name; using the most recent.")
    print()
    print("  [X] Restart - DELETE the file and PURGE its spool, then re-run "
          "from shot 1.")
    print("  [Q] Quit    - change nothing and exit.")

    choices = {"x": RESTART, "q": QUIT}
    prompt = "Choice (X/Q): "
    while True:
        response = input_fn(prompt).strip().lower()
        if response in choices:
            return choices[response]
        print(f"Please enter one of: {', '.join(choices)}")


def apply_action(action: str, run_paths) -> str | None:
    """Perform the side effects for ``action`` and return the spool dir to use.

    RESTART deletes the HDF5 and rotates its spool aside so the run starts from
    a clean slate at shot 1; the (unchanged) spool dir is returned for the caller
    to recreate.
    """
    if action != RESTART:
        raise ValueError(f"apply_action: unhandled action {action!r}")

    spool_dir = run_paths.spool_dir
    # An offload still actively draining holds the HDF5 (mid-write) open; on
    # Windows that blocks the delete/rotate below. The os.remove / rotate failure
    # paths tell the operator to close any offload window first.
    if os.path.exists(run_paths.hdf5_path):
        try:
            os.remove(run_paths.hdf5_path)
        except OSError as e:
            raise RuntimeError(
                f"Could not delete {run_paths.hdf5_path}: {e}. Close any "
                "offload window still writing to it, then restart.") from e
    if spool_dir:
        try:
            rotated = spool_format.rotate_spool(spool_dir)
        except OSError as e:
            raise RuntimeError(
                f"Could not rotate the old spool {spool_dir}: {e}. Close any "
                "offload window still attached to it, then restart.") from e
        if rotated:
            print(f"Rotated old spool aside -> {rotated}")
    return spool_dir
