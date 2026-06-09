"""Resume / restart decision logic for the spooled acquisition pipeline.

Splits the "what state is the existing run in / what does the operator want / do
it" flow into three pure-ish steps so the entry script (`Data_Run_bmotion.py`)
stays a thin orchestrator instead of nesting detection, prompting, and file
surgery inside ``main()``:

    inspect_run(run_paths)        -> RunState     (read-only)
    prompt_action(run_paths, st)  -> "resume"|"restart"|"quit"
    apply_action(action, paths,…) -> StartPlan    (side effects in one place)

Only the spool layer (``spooling.spool_format``) is imported, so this module is
light and does not pull in bmotion/hardware (preserving import hygiene).
"""

import os
from dataclasses import dataclass

from spooling import spool_format


# Run lifecycle states inferred from the HDF5 + spool artifacts.
FRESH = "fresh"        # no prior file -> start clean
PARTIAL = "partial"    # terminated early -> resume or restart
COMPLETE = "complete"  # finished normally -> restart or quit only

RESUME = "resume"
RESTART = "restart"
QUIT = "quit"


@dataclass(frozen=True)
class RunState:
    """Read-only summary of an existing run, used to drive the prompt."""

    status: str                 # FRESH | PARTIAL | COMPLETE
    completed_shots: int = 0    # highest shot known written
    abort_reason: str | None = None


@dataclass(frozen=True)
class StartPlan:
    """Result of applying the operator's choice: how to start the run."""

    spool_dir: str | None
    start_shot: int


def position_start_shot(highest_present_shot: int, nshots: int) -> int:
    """First shot of the position that ``highest_present_shot`` belongs to.

    Resume re-takes the probe's last *position* from its first shot (not just the
    missing shots within it), so the resume point is rounded **down** to a
    position boundary. With ``nshots`` shots per position and 1-based shot
    numbers, the position containing shot ``H`` starts at
    ``nshots*floor((H-1)/nshots) + 1``. The acquire loop's existing
    "skip whole position below start_shot" test then replays that whole position.
    """
    if highest_present_shot < 1 or nshots < 1:
        return 1
    return nshots * ((highest_present_shot - 1) // nshots) + 1


def inspect_run(run_paths) -> RunState:
    """Classify an existing run from its spool artifacts (read-only).

    Mirrors the prior ``_check_partial_run`` but flattened with early returns and
    keyed off the single resolved spool dir (no ``None``-spelunking across a
    separate "latest subdir" lookup).
    """
    if not run_paths.is_existing:
        return RunState(status=FRESH)

    spool_dir = run_paths.spool_dir
    if spool_dir and spool_format.run_complete_exists(spool_dir):
        try:
            complete = spool_format.read_run_complete(spool_dir)
        except spool_format.SpoolMetadataError:
            # A corrupt RUN_COMPLETE can't tell us the final shot count; rather
            # than crash the resume prompt, treat it like an interrupted run and
            # let the operator decide (resume from the highest .done / restart).
            complete = None
        if complete is not None:
            if not complete.get("terminated_early"):
                # Clean RUN_COMPLETE: the run finished normally.
                return RunState(status=COMPLETE,
                                completed_shots=int(complete.get("final_shot_num", 0)))
            return RunState(
                status=PARTIAL,
                completed_shots=int(complete.get("final_shot_num", 0)),
                abort_reason=complete.get("abort_reason"),
            )

    if spool_dir and spool_format.run_metadata_exists(spool_dir):
        # Metadata but no RUN_COMPLETE: the acquire process was killed before it
        # could write the sentinel; the highest .done shot is what survived.
        ready = spool_format.iter_ready_shots(spool_dir)
        return RunState(
            status=PARTIAL,
            completed_shots=max(ready) if ready else 0,
            abort_reason="process killed before RUN_COMPLETE was written",
        )

    # An HDF5 exists but no usable spool artifacts (e.g. the spool was wiped):
    # treat as partial without a known shot count.
    return RunState(status=PARTIAL, abort_reason="previous run was interrupted")


def prompt_action(run_paths, state: RunState, *, nshots: int = 1,
                  input_fn=input) -> str:
    """Ask the operator what to do with an existing run. Returns an action.

    A PARTIAL run offers Resume / Restart / Quit; a COMPLETE run offers only
    Restart / Quit (nothing to resume). ``input_fn`` is injectable for testing.
    """
    name = run_paths.name
    fname = os.path.basename(run_paths.hdf5_path)

    print(f'\nRun "{name}" exists (file: {fname}).')
    if run_paths.ambiguous:
        print("  NOTE: multiple files match this name; using the most recent.")
    if state.completed_shots:
        resume_at = position_start_shot(state.completed_shots, nshots)
        print(f"  Shots completed : {state.completed_shots}")
        print(f"  On resume       : re-take the position at shot {resume_at} "
              f"(overwrites from there), then continue")
    if state.abort_reason:
        print(f"  Stopped because : {state.abort_reason}")
    print()

    if state.status == PARTIAL:
        print("  [R] Resume  - keep file + spool; return the probe to its last "
              "position and RE-TAKE it (overwrites those shots), then continue.")
        print("  [X] Restart - DELETE the file and PURGE its spool, then re-run "
              "from shot 1.")
        print("  [Q] Quit    - change nothing and exit.")
        choices = {"r": RESUME, "x": RESTART, "q": QUIT}
        prompt = "Choice (R/X/Q): "
    else:  # COMPLETE
        print(f'  (Run "{name}" finished normally.)')
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


def apply_action(action: str, run_paths, state: RunState, *,
                 nshots: int = 1) -> StartPlan:
    """Perform the side effects for ``action`` and return how to start the run.

    All file/spool surgery lives here so the caller is branch-free:

    * RESUME  - keep the HDF5 + spool, drop the old RUN_COMPLETE so the resumed
      run can write a fresh one, and compute the position-boundary start shot.
    * RESTART - delete the HDF5 and rotate its spool aside, then hand back a
      clean spool dir with ``start_shot=1``.
    """
    spool_dir = run_paths.spool_dir

    if action == RESUME:
        if spool_dir:
            spool_format.clear_run_complete(spool_dir)
        start_shot = position_start_shot(state.completed_shots, nshots)
        return StartPlan(spool_dir=spool_dir, start_shot=max(start_shot, 1))

    if action == RESTART:
        # An offload still actively draining holds the HDF5 (mid-write) open; on
        # Windows that blocks the delete/rotate below. The offload single-instance
        # lock was removed on this branch, so we no longer pre-warn here; the
        # os.remove failure path below still tells the operator to close any
        # offload window still writing to the file.
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
        return StartPlan(spool_dir=spool_dir, start_shot=1)

    raise ValueError(f"apply_action: unhandled action {action!r}")
