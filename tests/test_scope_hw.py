"""Hardware diagnostic tests for the LeCroy scope.

Connects to one real scope through the lapd_daq adapter (and, for the
Data_Run-style check, through the legacy acquisition path). Skipped by default
so a normal run on a developer machine stays green; opt in via environment
variables so an enabled flag can never be committed:

    $env:LAPD_RUN_SCOPE_CHECK = "1"            # adapter connect check
    $env:LAPD_RUN_DATA_RUN_SCOPE_CHECK = "1"   # legacy Data_Run-style check
    $env:LAPD_SCOPE_ALLOW_ACQUIRE = "1"        # arm + write a shot (destructive)
    $env:LAPD_SCOPE_NAME = "lpscope"           # optional; default first scope
    pytest tests/test_scope_hw.py -v -s
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from lapd_daq.config import load_run_config
from lapd_daq.devices.lab_scopes import LabScopesLeCroyScopeAdapter
from lapd_daq.models import ShotPlan, ShotResult
from lapd_daq.storage.hdf5 import HDF5RunWriter

from _hardware_check_base import HardwareCheckBase, env_flag, env_str
from _hardware_check_helpers import restrict_scope_config

# --------------------------------------------------------------------------- #
# Run flags — read from the environment; committed defaults are always safe.
# --------------------------------------------------------------------------- #
RUN_SCOPE_CHECK = env_flag("LAPD_RUN_SCOPE_CHECK")
RUN_DATA_RUN_SCOPE_CHECK = env_flag("LAPD_RUN_DATA_RUN_SCOPE_CHECK")

# Safety gate — destructive acquisition is off by default even when the check
# above is enabled. Set explicitly to arm and write a shot.
SCOPE_ALLOW_ACQUIRE = env_flag("LAPD_SCOPE_ALLOW_ACQUIRE")

# --------------------------------------------------------------------------- #
# Connection info / parameters. EXPERIMENT_CONFIG_PATH is resolved relative to
# the current working directory; pass an absolute path to avoid surprises.
# --------------------------------------------------------------------------- #
EXPERIMENT_CONFIG_PATH = env_str("LAPD_EXPERIMENT_CONFIG", "experiment_config.txt")

# Scope check
SCOPE_NAME = env_str("LAPD_SCOPE_NAME")          # None = first scope in [scope_ips]
SCOPE_CONNECT_TIMEOUT_S = 30.0
SCOPE_SHOT_NUM = 1

# Data_Run-style scope check
DATA_RUN_SCOPE_NAME = env_str("LAPD_DATA_RUN_SCOPE_NAME")  # None = all scopes
DATA_RUN_SCOPE_SHOTS = 1
# --------------------------------------------------------------------------- #


class ScopeHardwareCheck(HardwareCheckBase):
    """Connect to one real LeCroy scope; optionally acquire one shot."""

    run_flag = RUN_SCOPE_CHECK
    label = "scope"

    def test_scope_connects_and_optionally_acquires(self) -> None:
        config = load_run_config(EXPERIMENT_CONFIG_PATH, mode="stationary", output_path=self.output_path)
        scope_config = _select_scope(config, SCOPE_NAME)
        config = replace(config, scopes=[scope_config])

        scope = LabScopesLeCroyScopeAdapter(
            scope_config.name,
            scope_config.ip_address,
            description=scope_config.description,
            timeout=SCOPE_CONNECT_TIMEOUT_S,
        )
        print(f"\n[scope check] connecting to {scope_config.name} at {scope_config.ip_address}")
        try:
            scope.connect()
            scope.initialize()
            time_points = len(scope.time_array())
            print(f"[scope check] initialized; {time_points} displayed time points")
            self.assertGreater(time_points, 0, "scope reported no time points")

            if not SCOPE_ALLOW_ACQUIRE:
                print("[scope check] initialize-only PASS (set SCOPE_ALLOW_ACQUIRE=True to acquire)")
                return

            self._acquire_one_shot(scope, config)
        finally:
            scope.close()

    def _acquire_one_shot(self, scope: LabScopesLeCroyScopeAdapter, config) -> None:
        writer = HDF5RunWriter(config.output_path, config)
        writer.initialize(
            {scope.name: scope.metadata()},
            {scope.name: scope.time_array()},
            {"diagnostic": {"instrument": "scope", "scope_name": scope.name}},
        )
        scope.arm()
        scope_shot = scope.acquire(SCOPE_SHOT_NUM)
        writer.write_scope_shot(scope_shot, SCOPE_SHOT_NUM)
        writer.finalize([ShotResult(plan=ShotPlan(shot_num=SCOPE_SHOT_NUM), scope_shots=[scope_shot])])
        print(f"[scope check] acquisition PASS -> {config.output_path}")


# --------------------------------------------------------------------------- #
class DataRunScopeHardware(HardwareCheckBase):
    """Legacy Data_Run-style acquisition loop with real scopes, no motors."""

    run_flag = RUN_DATA_RUN_SCOPE_CHECK
    label = "data_run_scope"

    def setUp(self) -> None:
        super().setUp()
        if DATA_RUN_SCOPE_SHOTS < 1:
            self.fail("DATA_RUN_SCOPE_SHOTS must be at least 1")

    def test_data_run_scope_path_writes_hdf5(self) -> None:
        from acquisition.config import load_experiment_config
        from acquisition.scope_runner import MultiScopeAcquisition, single_shot_acquisition
        from acquisition import hdf5_writer

        config, raw_config_text = load_experiment_config(EXPERIMENT_CONFIG_PATH)
        if DATA_RUN_SCOPE_NAME:
            restrict_scope_config(config, DATA_RUN_SCOPE_NAME)

        print(f"\n[data-run-scope] output={self.output_path}")
        with MultiScopeAcquisition(self.output_path, config, raw_config_text) as msa:
            msa.initialize_hdf5_base()
            active_scopes = msa.initialize_scopes()
            self.assertTrue(active_scopes, "No valid data found from any scope")

            for shot_num in range(1, DATA_RUN_SCOPE_SHOTS + 1):
                print(f"[data-run-scope] shot {shot_num}/{DATA_RUN_SCOPE_SHOTS}")
                single_shot_acquisition(msa, active_scopes, shot_num)

            hdf5_writer.record_shot_count(self.output_path, msa.scope_ips, DATA_RUN_SCOPE_SHOTS)

        print(f"[data-run-scope] PASS -> {self.output_path}")


# --------------------------------------------------------------------------- #
def _select_scope(config, requested):
    if not config.scopes:
        raise RuntimeError("No scopes found in [scope_ips].")
    if requested is None:
        return config.scopes[0]
    needle = requested.lower()
    for scope in config.scopes:
        if scope.name.lower() == needle:
            return scope
    available = ", ".join(scope.name for scope in config.scopes)
    raise RuntimeError(f"Scope {requested!r} not found. Available: {available}")


if __name__ == "__main__":
    unittest.main()
