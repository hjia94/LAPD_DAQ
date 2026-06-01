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

This module is deliberately thin: ``bapsf_motion`` already handles most recovery
internally, so we only add what the library does *not* do. Specifically, the
library's ``Motor.move_to`` is fire-and-forget (sends ``DI``+``FP`` and returns;
no wait, no arrival check) and ``Motor._moveable`` already runs ``alarm_reset``
and re-``enable`` before every move and refuses a move on a real (non-limit)
alarm; the heartbeat already auto-reconnects a dropped link in the background.
So re-issuing a move is, by itself, the recovery -- the library re-clears alarms
and re-enables for us. We therefore do **not** duplicate alarm-reset / enable /
reconnect logic here (that churn -- repeated ``AR`` commands in particular -- is
what risked desyncing the encoder).

What we add, and why each is needed:

* **Wait for the move to actually finish, by progress not by clock.** This is the
  core fix. A move that keeps advancing is *never* interrupted, so a slow-but-
  healthy long move is left to finish. Killing it mid-travel (the old fixed-
  timeout behavior) stranded the probe at a partial position and made every later
  position read "did not reach target." See :func:`_settle`.
* **Verify arrival.** Because targets are *absolute* and ``move_to`` is silent on
  failure, the only way to know a move worked is to compare the achieved position
  (on a fresh, non-cached read -- see :func:`_refresh_status`) to the target.
* **One retry: soft-stop + re-issue.** If a move stalls or misses, soft-stop and
  re-issue the same (absolute) target once. The library handles alarm-clear /
  re-enable on that re-issue. No separate fault-clearing ladder.
* **Encoder cross-check.** After arrival, warn (read-only) if the encoder (EP)
  disagrees with the commanded/step position (IP) -- the independent check that
  catches silent step loss / encoder slip.

If the retry still fails, raise :class:`MotorError` so the caller can record the
bad position and continue (skip-and-continue) or stop.

All status reads go through defensive accessors so this module works against
both the real actors and lightweight test stubs, and never itself raises an
AttributeError on a partially-implemented object.
"""

import re
import time
import warnings

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


def _scalar(val):
    """Coerce a possibly-Quantity/AckFlags command return to a float, or None."""
    if val is None:
        return None
    v = getattr(val, "value", val)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None  # AckFlags / NACK / non-numeric -> "unavailable"


# Negative-aware parse of the drive's raw "EP=<counts>" reply. The library's own
# encoder_position recv regex is "EP=[0-9]+" (no '-?'), which fails to match a
# negative encoder value and raises an uncaught AttributeError out of
# send_command. We read the encoder through the documented low-level escape hatch
# (Motor._send_raw_command) and parse it here instead, so negative encoder
# positions (any move that crosses zero) are handled correctly without touching
# the bapsf_motion library.
_EP_RE = re.compile(r"EP=(-?\d+)")


def _read_encoder_counts(motor):
    """Read the encoder position (EP) in counts via the raw escape hatch.

    Returns the signed integer encoder count, or None if the motor can't supply
    it (no ``_send_raw_command``, comms hiccup, NACK / non-numeric reply). This
    bypasses the library's buggy ``encoder_position`` recv regex which cannot
    parse negative values; the raw reply (e.g. ``"EP=-12345"``) is parsed here.
    """
    raw = getattr(motor, "_send_raw_command", None)
    if not callable(raw):
        return None
    try:
        reply = raw("EP")
    except Exception:  # noqa: BLE001 - comms hiccup / lost connection -> unavailable
        return None
    m = _EP_RE.search(str(reply))
    return int(m.group(1)) if m else None


def _encoder_counts(motor):
    """Encoder reading in counts (signed), preferring the native API.

    bapsf_motion ``patch_position_regex`` exposes ``Motor.encoder`` (counts, read
    via the immediate ``IE`` command, negative-safe). Use it when available; fall
    back to the raw ``_send_raw_command("EP")`` parse for an older library. Returns
    a float count or None."""
    native = _scalar(getattr(motor, "encoder", None))
    if native is not None:
        return native
    counts = _read_encoder_counts(motor)
    return float(counts) if counts is not None else None


def _counts_per_rev(motor):
    """Encoder counts per motor revolution, preferring native ``counts_per_rev``.

    bapsf_motion ``patch_position_regex`` adds ``Motor.counts_per_rev``; older
    libraries only expose it via the cached ``encoder_resolution``. Returns a
    float or None."""
    native = _scalar(getattr(motor, "counts_per_rev", None))
    if native:
        return native
    return _encoder_resolution(motor)


def _encoder_resolution(motor):
    """Encoder resolution (counts/rev) for ``motor``, or None.

    Read from the motor's cached ``_motor`` params when present (matching
    :func:`encoder_step_mismatch`), else via the unbuffered ``encoder_resolution``
    command. Returns None if neither is available."""
    motor_params = getattr(motor, "_motor", None)
    if isinstance(motor_params, dict):
        er = _scalar(motor_params.get("encoder_resolution"))
        if er:
            return er
    send = getattr(motor, "send_command", None)
    if callable(send):
        try:
            return _scalar(send("encoder_resolution"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _units_per_rev(ax, motor):
    """Physical units translated per motor revolution (e.g. cm/rev), or None.

    This is the axis-level scale that converts motor revolutions to the physical
    drive units the transform expects. It lives on the |Axis| (``ax.units_per_rev``,
    an astropy Quantity in units/rev); ``Axis.position`` uses it to report cm. We
    read it from the axis first, falling back to the motor object for stubs that
    expose it there. Returns a plain float (the unit magnitude), or None."""
    for obj in (ax, motor):
        upr = getattr(obj, "units_per_rev", None)
        val = _scalar(upr)
        if val:
            return val
    return None


def _coerce_position_tuple(value):
    """Coerce a (possibly Quantity / array / None) motion-space value to a float
    tuple of finite numbers, or None if it can't be (None, NaN, AckFlag, etc.)."""
    if value is None:
        return None
    arr = np.asarray(getattr(value, "value", value)).squeeze()
    try:
        out = tuple(float(v) for v in np.atleast_1d(arr))
    except (TypeError, ValueError):
        return None
    if not out or not all(np.isfinite(v) for v in out):
        return None
    return out


def _native_encoder_motion_space(mg):
    """Motion-space encoder position from the library's native ``mg.encoder``.

    ``MotionGroup.encoder`` (bapsf_motion ``patch_position_regex`` and later)
    returns the encoder reading already pushed through the coordinate transform,
    i.e. the real probe position in motion-space coordinates -- the EP-sourced
    twin of ``mg.position``. Returns a float tuple, or None when the property is
    absent (older library), errors, or yields a non-finite/None value (e.g. a
    cold heartbeat cache), so the caller can fall back to the manual path."""
    # Access inside the try: ``mg.encoder`` is a property that may raise (e.g. on
    # a None cache or comms error), and a raising property is not caught by
    # hasattr() -- only AttributeError is. A missing attribute lands here as
    # AttributeError and is treated the same as "unavailable".
    try:
        enc = mg.encoder
    except AttributeError:
        return None  # older library without the native property
    except Exception:  # noqa: BLE001 - property raised (None cache / comms)
        return None
    return _coerce_position_tuple(enc)


def _manual_encoder_motion_space(mg):
    """Motion-space encoder position computed by hand (compat fallback).

    Used when the library lacks a working native ``mg.encoder`` (older
    bapsf_motion on ``main``). Per axis: read EP counts (negative-safe, via
    :func:`_read_encoder_counts`) and convert to the axis's physical units (cm)
    the transform expects::

        cm = ep_counts / encoder_resolution * units_per_rev

    then feed the cm vector through the same ``mg.transform(..., to_coords=
    "motion_space")`` that ``MotionGroup.position`` uses. Returns a float tuple,
    or None if any axis can't supply EP / the scaling constants / the transform.
    (Passing raw step counts here instead of cm is what produced nonsense like
    ``41.73`` / ``14833`` where ``15`` / ``0`` cm was expected.)"""
    axes = _axes(mg)
    if not axes:
        return None

    cm = []
    for ax in axes:
        motor = getattr(ax, "motor", ax)
        counts = _read_encoder_counts(motor)
        enc_res = _encoder_resolution(motor)
        upr = _units_per_rev(ax, motor)
        if counts is None or not enc_res or not upr:
            return None
        # counts -> rev (encoder_resolution) -> physical units (units_per_rev)
        cm.append(counts / enc_res * upr)

    transform = getattr(mg, "transform", None)
    if not callable(transform):
        return None
    try:
        ms = transform(cm, to_coords="motion_space")
    except Exception:  # noqa: BLE001 - transform math failure -> fall back to IP
        return None
    return _coerce_position_tuple(ms)


def encoder_motion_space_position(mg):
    """Probe position in motion space, derived from the ENCODER -- the *real*
    physical probe position (the Applied Motion manual notes IP, the calculated
    trajectory position behind ``mg.position``, "is not always equal to actual
    position"; the encoder is the physical feedback).

    Prefers the library's native :attr:`MotionGroup.encoder` (bapsf_motion
    ``patch_position_regex`` and later), which already applies the coordinate
    transform. Falls back to computing it by hand (:func:`_manual_encoder_motion_space`)
    when running against an older library without that property, so this works on
    both versions.

    Returns a ``(x, y[, ...])`` float tuple in the same motion-space coordinates
    as ``mg.position``, or ``None`` if neither path can supply a finite value --
    so callers can fall back to ``mg.position`` (IP). Read-only; the encoder read
    must happen with motion settled (callers refresh status first)."""
    return _native_encoder_motion_space(mg) or _manual_encoder_motion_space(mg)


def encoder_step_mismatch(mg, tol_rev=0.01):
    """Per-axis encoder-vs-step disagreement, as ``[(axis_idx, step_rev, enc_rev)]``.

    The motor tracks two independent position counters: the commanded *step*
    position (IP, what ``status["position"]`` reflects) and the *encoder* position
    (EP, the physical feedback). They should track each other; a growing gap means
    the motor lost steps or the encoder slipped -- exactly the "encoder doesn't
    match motor position" condition worth flagging.

    Both are read **directly** (bypassing the heartbeat cache) and converted to
    motor revolutions using ``steps_per_rev`` (gearing) and ``encoder_resolution``
    so they're comparable. An axis is reported only when both reads succeed and
    ``|step_rev - enc_rev| > tol_rev``.

    Read-only and best-effort: callers must have settled motion first (EP is a
    buffered command and is refused while the motor is moving). Any axis whose
    motor can't supply the values (stub, NACK, missing constants) is skipped
    silently -- never warns spuriously, never writes/corrects anything.

    The encoder is read via :func:`_encoder_counts`, which prefers the library's
    native negative-safe ``Motor.encoder`` (immediate ``IE``) and falls back to the
    raw ``EP`` escape hatch on older libraries -- either way an axis past zero is
    checked correctly instead of being silently skipped by the library's old buggy
    ``encoder_position`` read regex.
    """
    bad = []
    for idx, ax in enumerate(_axes(mg)):
        motor = getattr(ax, "motor", ax)
        send = getattr(motor, "send_command", None)
        if not callable(send):
            continue

        # Read step (IP) directly; read encoder counts via the native API
        # (negative-safe) with a raw-EP fallback.
        try:
            step = _scalar(send("get_position"))
        except Exception:  # noqa: BLE001 - comms hiccup -> just skip this axis
            continue
        enc = _encoder_counts(motor)
        if step is None or enc is None:
            continue

        steps_per_rev = _scalar(getattr(motor, "steps_per_rev", None))
        enc_res = _counts_per_rev(motor)
        if not steps_per_rev or not enc_res:
            continue  # can't convert to a common unit -> skip

        step_rev = step / steps_per_rev
        enc_rev = enc / enc_res
        if abs(step_rev - enc_rev) > tol_rev:
            bad.append((idx, step_rev, enc_rev))
    return bad


def warn_on_encoder_mismatch(mg, tol_rev=0.01, log=print):
    """Emit a clear warning per axis whose encoder/step positions disagree.

    Thin wrapper over :func:`encoder_step_mismatch` used by the acquisition loop
    after each settled move. Returns the mismatch list (empty if all agree).
    """
    bad = encoder_step_mismatch(mg, tol_rev=tol_rev)
    name = mg.config.get("name", "?") if hasattr(mg, "config") else "?"
    for idx, step_rev, enc_rev in bad:
        log(f"WARNING: motion group '{name}' axis {idx}: encoder position "
            f"({enc_rev:.4f} rev) disagrees with motor/step position "
            f"({step_rev:.4f} rev) by {abs(step_rev - enc_rev):.4f} rev "
            f"(> {tol_rev} rev tol). Motor may have lost steps or the encoder "
            f"slipped; physical position may differ from the recorded value.")
    return bad


def verify_encoder_zeroed(mg, tol_counts=2.0, log=print):
    """Confirm a set-zero actually took, by reading the encoder back.

    ``MotionGroup.set_zero`` / ``Motor.set_position(0)`` writes ``EP0`` then
    ``SP0`` and relies on the drive's ``%`` ACK -- it does **not** read the
    encoder back to confirm. The write is correct (the buggy ``EP=[0-9]+`` read
    regex is short-circuited by the ``%`` ACK, so even negative writes are safe),
    but a silently-failed zero (lost link mid-write, drive in a state that
    rejected it) would go unnoticed.

    This is the positive read-back check: after zeroing, read each axis's encoder
    via :func:`_encoder_counts` (native negative-safe ``Motor.encoder`` with a raw
    ``EP`` fallback) and confirm it reads ~0 (within ``tol_counts`` encoder counts
    -- a couple of counts of end-of-write jitter is normal). Returns the list of
    ``(axis_idx, counts)`` for axes that did NOT zero (empty == all good), warning
    on each.

    Call this immediately after zeroing and while the motor is idle. Read-only:
    never writes or re-zeros -- it only reports.
    """
    name = mg.config.get("name", "?") if hasattr(mg, "config") else "?"
    bad = []
    for idx, ax in enumerate(_axes(mg)):
        motor = getattr(ax, "motor", ax)
        counts = _encoder_counts(motor)
        if counts is None:
            # Can't read the encoder -> can't confirm; surface it rather than
            # silently claiming success.
            log(f"WARNING: motion group '{name}' axis {idx}: could not read "
                f"encoder to confirm set-zero (no encoder / read miss).")
            bad.append((idx, None))
            continue
        if abs(counts) > tol_counts:
            log(f"WARNING: motion group '{name}' axis {idx}: set-zero did not "
                f"take -- encoder reads {counts} counts (> {tol_counts} tol). "
                f"The EP0/SP0 write may have been rejected; re-zero before "
                f"trusting recorded positions.")
            bad.append((idx, counts))
    return bad


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


def _progress_position(rm, ml_order_dict):
    """Concatenated (x, y, ...) of every selected group, for progress tracking."""
    pos = []
    for mg_key in ml_order_dict:
        pos.extend(_position_tuple(rm.mgs[mg_key]))
    return tuple(pos)


def _moved(prev, cur, eps):
    """True if any coordinate changed by more than ``eps`` (i.e. still moving)."""
    if prev is None or len(prev) != len(cur):
        return True
    return any(abs(c - p) > eps for p, c in zip(prev, cur))


def _settle(rm, ml_order_dict, *, stall_timeout, max_move_time, progress_eps,
            poll=0.5):
    """Wait for motion to finish, tolerating slow-but-healthy long moves.

    Unlike a fixed wall-clock timeout, this only gives up when the move stops
    making progress: as long as any selected group's position keeps changing by
    more than ``progress_eps`` it is considered healthy and we keep waiting --
    so a legitimately long move is left to finish instead of being killed.

    Returns one of:
      * ``"settled"``  - ``rm.is_moving`` cleared (motion finished),
      * ``"stalled"``  - still moving but no position progress for
                         ``stall_timeout`` seconds (genuinely stuck),
      * ``"timeout"``  - hit the absolute ``max_move_time`` backstop.

    A fresh status read is forced before sampling position so progress is judged
    on current data, not the heartbeat cache.
    """
    time.sleep(poll)
    overall_deadline = time.time() + max_move_time
    last_pos = None
    last_progress = time.time()

    while rm.is_moving:
        now = time.time()
        if now > overall_deadline:
            return "timeout"

        for mg_key in ml_order_dict:
            _refresh_status(rm.mgs[mg_key])
        cur = _progress_position(rm, ml_order_dict)
        if _moved(last_pos, cur, progress_eps):
            last_pos = cur
            last_progress = now
        elif now - last_progress > stall_timeout:
            return "stalled"

        time.sleep(poll)

    return "settled"


def _disable_all(rm):
    for mg in rm.mgs.values():
        try:
            mg.drive.send_command("disable")
        except Exception as e:  # noqa: BLE001 - disable is best-effort
            print(f"disable failed for '{mg.config.get('name', '?')}': {e}")


def _refresh_status(mg):
    """Force a direct motor-status re-query so reads aren't fooled by the cache.

    ``MotionGroup.position`` returns the heartbeat-cached ``status["position"]``
    while the loop is live (updated only every ~0.2-1.5 s), so a read taken right
    after a move/recovery step can be stale. Calling ``Motor.retrieve_motor_status``
    issues only un-buffered commands (RS/IP/AL) and updates the cache, so a
    subsequent ``mg.position`` reflects the motor's current state.

    Best-effort and stub-tolerant: any axis/motor lacking the method is skipped.
    """
    for ax in _axes(mg):
        motor = getattr(ax, "motor", ax)
        fn = getattr(motor, "retrieve_motor_status", None)
        if callable(fn):
            _try(fn)


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


def move_with_recovery(rm, ml_order_dict, index, *, attempts=30, retry_wait=1.0,
                       stall_timeout=10.0, max_move_time=300.0, progress_eps=0.125,
                       tol=0.5, encoder_mismatch_tol_rev=0.01, log=print):
    """Move selected groups to ``index``, wait for arrival, verify, retry.

    Mirrors :func:`acquisition.bmotion.move_to_index` (move each group, wait for
    motion to settle, disable) but adds a *progress-aware* settle, position
    verification, an encoder-vs-step sanity warning, and one retry. Raises
    :class:`MotorError` if a group cannot be moved/verified after ``attempts``
    tries.

    Each attempt after the first waits ``retry_wait`` seconds, then soft-stops and
    re-issues the same (absolute) target -- the library's ``Motor.move_to``
    re-clears any alarm and re-enables the motor on its own, so there is no
    separate fault-clearing or reconnect ladder here. The progress-aware settle is the key behavior: a move
    that keeps making position progress is *never* interrupted, so a slow-but-
    healthy long move is left to finish rather than being soft-stopped mid-travel
    (which would strand the probe at a partial position and cascade "did not reach
    target" into every later position).

    Out-of-range indices for a group are skipped (as the legacy mover did): a
    group whose motion list is shorter than the current raster index simply
    does not move this position.
    """
    indices = _resolve_indices(rm, ml_order_dict, index)
    detail = "no successful move and no diagnostic captured"

    for attempt in range(1, attempts + 1):
        # Retry = wait, soft-stop the previous (failed) move, then re-issue. The
        # library handles alarm-reset + re-enable when the move is re-sent, so we
        # don't duplicate that here.
        if attempt >= 2:
            log(f"Move retry {attempt}/{attempts} at index {index}: "
                f"waiting {retry_wait}s, soft-stopping and re-issuing.")
            if retry_wait > 0:
                time.sleep(retry_wait)
            for mg_key in ml_order_dict:
                _try(rm.mgs[mg_key].stop, soft=True)

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

        outcome = _settle(rm, ml_order_dict, stall_timeout=stall_timeout,
                          max_move_time=max_move_time, progress_eps=progress_eps)
        _disable_all(rm)
        if outcome != "settled":
            detail = (f"motion stalled (no progress for {stall_timeout}s)"
                      if outcome == "stalled"
                      else f"motion exceeded {max_move_time}s ceiling")
            log(f"{detail} at index {index} (attempt {attempt}).")
            continue

        # --- verify arrival (on fresh, non-cached reads) ---
        bad = []
        for mg_key, (motion_index, ml_size) in indices.items():
            if motion_index not in range(ml_size):
                continue
            mg = rm.mgs[mg_key]
            _refresh_status(mg)  # defeat stale heartbeat cache before judging
            name = mg.config.get("name", mg_key) if hasattr(mg, "config") else mg_key
            if not _all_connected(mg):
                bad.append((mg_key, "connection lost"))
            elif _has_alarm(mg):
                bad.append((mg_key, "; ".join(_alarm_messages(mg)) or "alarm/fault active"))
            elif not _arrived(mg, motion_index, tol):
                target = _target_point(mg, motion_index)
                actual = _position_tuple(mg)
                # Surface a failure to reach the requested position immediately on
                # the run's print channel so a probe that didn't make it is visible
                # live. The reason is also carried into the HDF5 skip_reason via the
                # MotorError raised below (recorded by the sink's mark_skipped).
                log(f"WARNING: motion group '{name}' did NOT reach target "
                    f"{target} (at {actual}, tol {tol}) at index {index} "
                    f"(attempt {attempt}/{attempts})")
                bad.append((mg_key, f"did not reach target {target} (at {actual})"))
        if not bad:
            # Arrived. Sanity-check that the encoder agrees with the commanded
            # position before declaring success (read-only; warns, never aborts).
            for mg_key in indices:
                motion_index, ml_size = indices[mg_key]
                if motion_index in range(ml_size):
                    warn_on_encoder_mismatch(rm.mgs[mg_key],
                                             tol_rev=encoder_mismatch_tol_rev, log=log)
            return  # success

        detail = "; ".join(f"{rm.mgs[k].config.get('name', k)}: {why}" for k, why in bad)
        log(f"Move verification failed at index {index} (attempt {attempt}): {detail}")

    # Exhausted all attempts. The message carries the not-reached detail; the
    # caller records it into the HDF5 shot's skip_reason via mark_skipped.
    raise MotorError(
        f"Motor move to index {index} failed after {attempts} attempts: {detail}"
    )


# --------------------------------------------------------------------------- #
# Misc
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
