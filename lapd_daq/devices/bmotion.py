"""Direct adapter boundary for bapsf_motion-backed control."""

from __future__ import annotations

from lapd_daq.models import AchievedPosition, PlannedPosition


class BMotionAdapter:
    """Placeholder boundary for bapsf_motion-backed control."""

    def __init__(self, run_manager, motion_group_keys):
        self.run_manager = run_manager
        self.motion_group_keys = list(motion_group_keys)

    def connect(self) -> None:
        return None

    def move_to(self, position: PlannedPosition | None) -> AchievedPosition | None:
        raise NotImplementedError(
            "BMotionAdapter boundary exists, but the existing acquisition.bmotion "
            "workflow has not yet been migrated into the new run engine."
        )

    def close(self) -> None:
        terminate = getattr(self.run_manager, "terminate", None)
        if terminate is not None:
            terminate()

    def metadata(self) -> dict[str, object]:
        return {"adapter": "BMotionAdapter", "motion_group_keys": self.motion_group_keys}


BapsfMotionAdapter = BMotionAdapter
