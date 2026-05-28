"""Motor-error recovery for the bmotion acquisition loop.

`bapsf_motion` surfaces motor trouble in two ways, and *neither* is a plain
exception we can rely on:

* A move that cannot be performed (active alarm, excluded position, limit hit)
  is **logged and silently skipped** by ``MotionGroup.move_to`` / ``move_ml`` /
  ``Motor.move_to`` -- no raise. So a "failed move" is detected by checking the
  achieved position against the requested one.
* A lost TCP connection shows up as ``motor.status["connected"] == False`` (and
  the motor actor's heartbeat auto-reconnects in the background); a raw socket
  error may also bubble out of ``send_command``.

Because scan positions are sequential, a motor that stays broken makes every
later position unreachable -- so the policy is: retry a move through an
escalating recovery ladder, and if it still fails, raise :class:`MotorError` so
the caller can stop the run cleanly and finalize whatever data was acquired.

The recovery ladder (per :func:`move_with_recovery`) mirrors the library's own
``Motor._moveable`` logic:

  1. soft-stop + re-issue the move (transient miss / stale queue state),
  2. clear faults: ``alarm_reset`` (+ ``move_off_limit`` if on a limit) then
     re-issue,
  3. if any axis lost connection, wait for the heartbeat to restore it, then
     re-issue.

All status reads go through defensive accessors so this module works against
both the real actors and lightweight test stubs, and never itself raises an
AttributeError on a partially-implemented object.
"""

import time

import numpy as np


class MotorError(RuntimeError):
    """Raised when motor-move recovery is exhausted (terminal failure).

    Carries a human-readable diagnostic (per-group alarm messages / last
    position vs target) so the run can log why it stopped.
    """


# --------------------------------------------------------------------------- #
# Defensive accessors (tolerate real actors and stubs alike)
# --------------------------------------------------------------------------- #
def _position_tuple(mg):
    """Return the motion-group position as a plain (x, y[, ...]) float tuple."""
    pos = getattr(mg, "position", None)
    # Real MotionGroup.position is an astropy Quantity; stubs expose .value or
    # are directly indexable.
    val = getattr(pos, "value", pos)
    try:
        return tuple(float(v) for v in val)
    except TypeError:
        # Indexable but not iterable as floats (e.g. StubPosition) -> probe 0/1.
        return (float(pos[0]), float(pos[1]))


def _target_point(mg, motion_index):
    """The requested (x, y[, ...]) for ``motion_index`` from the motion list."""
    arr = np.asarray(mg.mb.motion_list.values)
    return tuple(float(v) for v in arr[motion_index])


def _axes(mg):
    drive = getattr(mg, "drive", None)
    if drive is None:
        return []
    return list(getattr(drive, "axes", []) or [])


def _axis_status(ax):
    """Best-effort status dict for an axis's motor; {} if unavailable."""
    motor = getattr(ax, "motor", ax)
    status = getattr(motor, "status", None)
    return status if isinstance(status, dict) else {}


def _all_connected(mg):
    """True if every axis reports a live connection (or can't report -> assume ok)."""
    axes = _axes(mg)
    if not axes:
        return True
    for ax in axes:
        st = _axis_status(ax)
        if "connected" in st and st["connected"] is False:
            return False
        if getattr(ax, "connected", True) is False:
            return False
    return True


def _has_alarm(mg):
    for ax in _axes(mg):
        st = _axis_status(ax)
        if st.get("alarm") or st.get("fault"):
            return True
    return False


def _alarm_messages(mg):
    msgs = []
    for ax in _axes(mg):
        st = _axis_status(ax)
        m = st.get("alarm_message")
        if m:
            msgs.append(str(m))
    return msgs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def safe_stop(rm):
    """Best-effort stop of every motion group; never raises.

    Called on terminal failure before teardown so the probe is commanded to
    halt even if one group is misbehaving.
    """
    for key, mg in getattr(rm, "mgs", {}).items():
        try:
            mg.stop()
        except Exception as e:  # noqa: BLE001 - best effort
            print(f"safe_stop: motion group {key} stop failed: {e}")


def _settle(rm, settle_timeout, poll=0.5):
    """Wait until ``rm.is_moving`` clears or the timeout elapses.

    Returns True if motion stopped within the timeout, False if it timed out
    (which is treated as a move failure by the caller).
    """
    time.sleep(poll)
    deadline = time.time() + settle_timeout
    while rm.is_moving:
        if time.time() > deadline:
            return False
        time.sleep(poll)
    return True


def _disable_all(rm):
    for mg in rm.mgs.values():
        try:
            mg.drive.send_command("disable")
        except Exception as e:  # noqa: BLE001 - disable is best-effort
            print(f"disable failed for '{mg.config.get('name', '?')}': {e}")


def _arrived(mg, motion_index, tol):
    """True if the group's achieved position matches the target within tol."""
    target = _target_point(mg, motion_index)
    actual = _position_tuple(mg)
    n = min(len(target), len(actual))
    return all(abs(actual[i] - target[i]) <= tol for i in range(n))


def _resolve_indices(rm, ml_order_dict, index):
    """Map the raster index to each group's (motion_index) honoring direction."""
    out = {}
    for mg_key, order in ml_order_dict.items():
        mg = rm.mgs[mg_key]
        ml_size = int(mg.mb.motion_list.shape[0])
        motion_index = index
        if order == "backward":
            motion_index = ml_size - index - 1
        out[mg_key] = (motion_index, ml_size)
    return out


def move_with_recovery(rm, ml_order_dict, index, *, attempts=3,
                       settle_timeout=30.0, reconnect_timeout=60.0, tol=0.5,
                       log=print):
    """Move selected groups to ``index`` with escalating recovery on failure.

    Mirrors :func:`acquisition.bmotion.move_to_index` (move each group, wait for
    motion to settle, disable) but adds: a settle timeout, position
    verification, and a recovery ladder. Raises :class:`MotorError` if a group
    cannot be moved/verified after ``attempts`` tries (after clearing faults and
    waiting out any connection loss).

    Out-of-range indices for a group are skipped (as the legacy mover did): a
    group whose motion list is shorter than the current raster index simply
    does not move this position.
    """
    indices = _resolve_indices(rm, ml_order_dict, index)
    detail = "no successful move and no diagnostic captured"

    for attempt in range(1, attempts + 1):
        # --- escalating recovery BETWEEN attempts (not before the first) ---
        if attempt >= 2:
            log(f"Move retry {attempt}/{attempts} at index {index}: soft-stopping and re-issuing.")
            for mg_key in ml_order_dict:
                _try(rm.mgs[mg_key].stop, soft=True)
            _reenable(rm, ml_order_dict, log)
        if attempt >= 3:
            log(f"Move retry {attempt}/{attempts}: clearing alarms / backing off limits.")
            _clear_faults(rm, ml_order_dict, log)

        # If any selected group lost its connection, wait for the actor's
        # heartbeat to restore it before issuing (don't burn an attempt racing
        # a known-down link).
        _await_reconnect(rm, ml_order_dict, reconnect_timeout, log)

        # --- issue the move ---
        try:
            for mg_key, (motion_index, ml_size) in indices.items():
                if motion_index not in range(ml_size):
                    continue  # group shorter than this raster line; no motion
                rm.mgs[mg_key].move_ml(motion_index)
        except Exception as e:  # comms/socket error during the move command
            detail = f"move command raised: {e}"
            log(f"Move command raised at index {index} (attempt {attempt}): {e}")
            continue

        settled = _settle(rm, settle_timeout)
        _disable_all(rm)
        if not settled:
            detail = f"motion did not settle within {settle_timeout}s"
            log(f"{detail} (attempt {attempt}).")
            continue

        # --- verify arrival + no active fault ---
        bad = []
        for mg_key, (motion_index, ml_size) in indices.items():
            if motion_index not in range(ml_size):
                continue
            mg = rm.mgs[mg_key]
            if not _all_connected(mg):
                bad.append((mg_key, "connection lost"))
            elif _has_alarm(mg):
                bad.append((mg_key, "; ".join(_alarm_messages(mg)) or "alarm/fault active"))
            elif not _arrived(mg, motion_index, tol):
                bad.append((mg_key, f"did not reach target "
                                    f"{_target_point(mg, motion_index)} "
                                    f"(at {_position_tuple(mg)})"))
        if not bad:
            return  # success

        detail = "; ".join(f"{rm.mgs[k].config.get('name', k)}: {why}" for k, why in bad)
        log(f"Move verification failed at index {index} (attempt {attempt}): {detail}")

    # Exhausted all attempts.
    raise MotorError(
        f"Motor move to index {index} failed after {attempts} attempts: {detail}"
    )


# --------------------------------------------------------------------------- #
# Ladder steps
# --------------------------------------------------------------------------- #
def _try(fn, **kw):
    try:
        fn(**kw)
    except TypeError:
        # Stub stop() may not accept soft=...; retry without kwargs.
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"recovery step failed: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"recovery step failed: {e}")


def _reenable(rm, ml_order_dict, log):
    for mg_key in ml_order_dict:
        mg = rm.mgs[mg_key]
        try:
            mg.drive.send_command("enable")
        except Exception as e:  # noqa: BLE001
            log(f"re-enable failed for '{mg.config.get('name', mg_key)}': {e}")


def _clear_faults(rm, ml_order_dict, log):
    for mg_key in ml_order_dict:
        mg = rm.mgs[mg_key]
        for ax in _axes(mg):
            motor = getattr(ax, "motor", ax)
            st = _axis_status(ax)
            if st.get("alarm") or st.get("fault"):
                # Prefer the motor's explicit alarm_reset when present.
                fn = getattr(motor, "alarm_reset", None)
                if callable(fn):
                    _try(fn)
                else:
                    send = getattr(motor, "send_command", None)
                    if callable(send):
                        _try(lambda: send("alarm_reset"))
                limits = st.get("limits") or {}
                if limits.get("CW") or limits.get("CCW"):
                    fn = getattr(motor, "move_off_limit", None)
                    if callable(fn):
                        _try(fn)


def _await_reconnect(rm, ml_order_dict, reconnect_timeout, log):
    """If any selected group is disconnected, wait for the heartbeat to restore it."""
    def all_up():
        return all(_all_connected(rm.mgs[k]) for k in ml_order_dict)

    if all_up():
        return
    log(f"Motor connection lost; waiting up to {reconnect_timeout}s for reconnect "
        f"(bapsf_motion heartbeat auto-reconnects).")
    deadline = time.time() + reconnect_timeout
    while not all_up():
        if time.time() > deadline:
            log("Reconnect wait timed out.")
            return
        time.sleep(1.0)
    log("Motor connection restored.")
