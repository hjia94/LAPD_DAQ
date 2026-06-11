"""Hardware diagnostic test for the Phantom camera.

Connects to the real Phantom camera through the lapd_daq adapter. Skipped by
default so a normal run on a developer machine stays green; opt in via
environment variables so an enabled flag can never be committed:

    $env:LAPD_RUN_CAMERA_CHECK = "1"       # configure-only check
    $env:LAPD_CAMERA_ALLOW_RECORD = "1"    # wait for trigger + record a .cine
    pytest tests/test_camera_hw.py -v -s
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from lapd_daq.config import load_run_config
from lapd_daq.devices.phantom import PhantomCameraAdapter
from lapd_daq.models import ShotPlan, ShotResult
from lapd_daq.storage.hdf5 import HDF5RunWriter

from _hardware_check_base import HardwareCheckBase
from _hardware_check_helpers import EXPERIMENT_CONFIG_PATH, env_flag

# --------------------------------------------------------------------------- #
# Run flags — read from the environment; committed defaults are always safe.
# --------------------------------------------------------------------------- #
RUN_CAMERA_CHECK = env_flag("LAPD_RUN_CAMERA_CHECK")

# Safety gate — the camera does not wait for a trigger / record unless this is set.
CAMERA_ALLOW_RECORD = env_flag("LAPD_CAMERA_ALLOW_RECORD")

# --------------------------------------------------------------------------- #
# Connection info / parameters. EXPERIMENT_CONFIG_PATH (imported above) comes
# from LAPD_EXPERIMENT_CONFIG; pass an absolute path to avoid surprises.
# --------------------------------------------------------------------------- #
CAMERA_EXPERIMENT_NAME = "hardware_camera_check"
CAMERA_SHOT_NUM = 1
# --------------------------------------------------------------------------- #


class CameraHardwareCheck(HardwareCheckBase):
    """Connect to the Phantom camera; optionally record one .cine."""

    run_flag = RUN_CAMERA_CHECK
    label = "camera"

    def test_camera_configures_and_optionally_records(self) -> None:
        config = load_run_config(EXPERIMENT_CONFIG_PATH, mode="camera", output_path=self.output_path)
        config = replace(config, scopes=[])

        from drivers.phantom_recorder import PhantomRecorder

        adapter = PhantomCameraAdapter(
            PhantomRecorder(_camera_recorder_config(config)),
            experiment_name=CAMERA_EXPERIMENT_NAME,
            save_path=self.output_path.parent,
        )
        try:
            adapter.connect()
            print(f"\n[camera check] configured; metadata={adapter.metadata()}")

            writer = HDF5RunWriter(self.output_path, config)
            writer.initialize({}, {}, {"camera": adapter.metadata(), "diagnostic": {"instrument": "camera"}})

            if not CAMERA_ALLOW_RECORD:
                print("[camera check] configure-only PASS (set CAMERA_ALLOW_RECORD=True to record)")
                writer.finalize(
                    [ShotResult(plan=ShotPlan(shot_num=CAMERA_SHOT_NUM), message="configure-only camera check")]
                )
                return

            self._record_one(adapter, writer)
        finally:
            adapter.close()

    def _record_one(self, adapter: PhantomCameraAdapter, writer: HDF5RunWriter) -> None:
        adapter.arm(CAMERA_SHOT_NUM)
        camera_shot = adapter.complete(CAMERA_SHOT_NUM)
        writer.write_camera_shot(camera_shot)
        writer.finalize([ShotResult(plan=ShotPlan(shot_num=CAMERA_SHOT_NUM), camera_shot=camera_shot)])
        print(f"[camera check] record PASS file={camera_shot.file_name} -> {self.output_path}")


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
def _camera_recorder_config(config) -> dict[str, object]:
    params = dict(config.camera.parameters)
    output_path = config.output_path
    return {
        "exposure_us": int(params.get("exposure_us", 30)),
        "fps": int(params.get("fps", 10000)),
        "pre_trigger_frames": int(params.get("pre_trigger_frames", -500)),
        "post_trigger_frames": int(params.get("post_trigger_frames", 1000)),
        "resolution": _resolution(params.get("resolution", (256, 256))),
        "hdf5_file_path": str(output_path),
        "save_path": str(output_path.parent),
    }


def _resolution(value) -> tuple[int, int]:
    # Config loaders may hand back a tuple, a list (e.g. parsed JSON/TOML), or
    # a "WxH" / "W,H" string; accept all three.
    if isinstance(value, (tuple, list)):
        return (int(value[0]), int(value[1]))
    text = str(value).replace("x", ",")
    first, second = (part.strip() for part in text.split(",", 1))
    return (int(first), int(second))


if __name__ == "__main__":
    unittest.main()
