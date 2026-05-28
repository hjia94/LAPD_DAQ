"""Tests for acquisition.motor_recovery.move_with_recovery.

Uses purpose-built fakes that mimic the bapsf_motion shapes the recovery code
reads (mg.mb.motion_list.values, mg.position, mg.stop, mg.drive.axes[i].motor
with a .status dict + alarm_reset/move_off_limit/send_command, mg.drive.
send_command). Each fake injects a specific failure mode so we can assert the
recovery ladder behaves: transient miss recovers, connection loss waits and
recovers, a resettable alarm recovers, and a permanent fault raises MotorError.

No bapsf_motion, hardware, or HDF5 needed.
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from acquisition import motor_recovery
from acquisition.motor_recovery import move_with_recovery, MotorError, safe_stop


class _Motor:
    def __init__(self):
        self.status = {"connected": True, "alarm": False, "fault": False,
                       "limits": {"CW": False, "CCW": False}, "alarm_message": ""}
        self.alarm_reset_calls = 0

    def alarm_reset(self):
        self.alarm_reset_calls += 1
        self.status["alarm"] = False
        self.status["fault"] = False

    def move_off_limit(self):
        self.status["limits"] = {"CW": False, "CCW": False}

    def send_command(self, *a, **k):
        pass


class _Axis:
    def __init__(self, motor):
        self.motor = motor

    @property
    def connected(self):
        return self.motor.status.get("connected", True)

    @property
    def is_moving(self):
        return False


class _Drive:
    def __init__(self, n=1):
        self.axes = [_Axis(_Motor()) for _ in range(n)]
        self.enable_calls = 0
        self.disable_calls = 0

    def send_command(self, command, *a, **k):
        if command == "enable":
            self.enable_calls += 1
        elif command == "disable":
            self.disable_calls += 1


class _FakeMG:
    """Configurable fake motion group.

    ``behavior`` controls move_ml: a callable(self, index, attempt) -> None that
    sets self._pos and/or motor status to simulate success/miss/fault.
    """
    def __init__(self, name, ml_values, behavior):
        self.config = {"name": name}
        self.mb = type("MB", (), {"motion_list": type(
            "ML", (), {"values": np.asarray(ml_values, dtype=float),
                       "shape": np.asarray(ml_values).shape})()})()
        self.drive = _Drive(n=1)
        self._pos = (0.0, 0.0)
        self._attempt = 0
        self.behavior = behavior
        self.stop_calls = 0

    # mg.position -> object with .value tuple (like astropy Quantity-ish)
    @property
    def position(self):
        return type("P", (), {"value": self._pos})()

    def stop(self, soft=False):
        self.stop_calls += 1

    def move_ml(self, index):
        self._attempt += 1
        self.behavior(self, index, self._attempt)


class _RM:
    def __init__(self, mgs):
        self.mgs = dict(mgs)
        self.is_moving = False


def _target(mg, index):
    return tuple(float(v) for v in np.asarray(mg.mb.motion_list.values)[index])


# --------------------------------------------------------------------------- #
# Behaviors
# --------------------------------------------------------------------------- #
def always_arrive(mg, index, attempt):
    mg._pos = _target(mg, index)


def miss_then_arrive(n_miss):
    def b(mg, index, attempt):
        if attempt <= n_miss:
            mg._pos = (999.0, 999.0)  # nowhere near target
        else:
            mg._pos = _target(mg, index)
    return b


def disconnect_then_reconnect(reconnect_on_attempt):
    def b(mg, index, attempt):
        motor = mg.drive.axes[0].motor
        if attempt < reconnect_on_attempt:
            motor.status["connected"] = False
            mg._pos = (999.0, 999.0)
        else:
            motor.status["connected"] = True
            mg._pos = _target(mg, index)
    return b


def alarm_then_clear(mg, index, attempt):
    motor = mg.drive.axes[0].motor
    if attempt == 1:
        motor.status["alarm"] = True
        motor.status["alarm_message"] = "drive fault"
        mg._pos = (999.0, 999.0)
    else:
        # _clear_faults() will have called alarm_reset before this attempt.
        mg._pos = _target(mg, index)


def never_arrive(mg, index, attempt):
    mg._pos = (999.0, 999.0)


class MoveWithRecoveryTests(unittest.TestCase):
    def setUp(self):
        # Make _settle / _await_reconnect fast.
        self._orig_sleep = motor_recovery.time.sleep
        motor_recovery.time.sleep = lambda *_a, **_k: None

    def tearDown(self):
        motor_recovery.time.sleep = self._orig_sleep

    def _run(self, behavior, **kw):
        mg = _FakeMG("A", [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], behavior)
        rm = _RM({"a": mg})
        move_with_recovery(rm, {"a": "forward"}, 1,
                           settle_timeout=1, reconnect_timeout=1, tol=0.5,
                           log=lambda *_a: None, **kw)
        return mg, rm

    def test_first_try_success(self):
        mg, _ = self._run(always_arrive)
        self.assertEqual(mg.position.value, (1.0, 1.0))
        self.assertEqual(mg.stop_calls, 0)  # no recovery needed

    def test_transient_miss_recovers_on_retry(self):
        mg, _ = self._run(miss_then_arrive(n_miss=1), attempts=3)
        self.assertEqual(mg.position.value, (1.0, 1.0))
        self.assertGreaterEqual(mg.stop_calls, 1)  # soft-stop ladder ran

    def test_connection_loss_waits_then_recovers(self):
        mg, _ = self._run(disconnect_then_reconnect(reconnect_on_attempt=2), attempts=3)
        self.assertEqual(mg.position.value, (1.0, 1.0))

    def test_resettable_alarm_recovers(self):
        mg, _ = self._run(alarm_then_clear, attempts=3)
        self.assertEqual(mg.position.value, (1.0, 1.0))
        self.assertGreaterEqual(mg.drive.axes[0].motor.alarm_reset_calls, 1)

    def test_permanent_failure_raises_motorerror(self):
        with self.assertRaises(MotorError):
            self._run(never_arrive, attempts=3)

    def test_out_of_range_index_is_skipped_not_failed(self):
        # index 5 is beyond the 3-point list -> no motion, treated as success.
        mg = _FakeMG("A", [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], never_arrive)
        rm = _RM({"a": mg})
        move_with_recovery(rm, {"a": "forward"}, 5, attempts=2,
                           settle_timeout=1, reconnect_timeout=1, log=lambda *_a: None)
        self.assertEqual(mg._attempt, 0)  # move_ml never called


class SafeStopTests(unittest.TestCase):
    def test_safe_stop_calls_all_and_swallows_errors(self):
        good = _FakeMG("A", [[0.0, 0.0]], always_arrive)

        class _Boom(_FakeMG):
            def stop(self, soft=False):
                raise RuntimeError("stop failed")

        bad = _Boom("B", [[0.0, 0.0]], always_arrive)
        rm = _RM({"a": good, "b": bad})
        safe_stop(rm)  # must not raise
        self.assertEqual(good.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
