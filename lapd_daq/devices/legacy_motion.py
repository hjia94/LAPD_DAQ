"""Direct adapters for existing LAPD motor-control objects."""

from __future__ import annotations

from lapd_daq.models import AchievedPosition, PlannedPosition


class LegacyMotorAdapter:
    """Thin adapter around the existing legacy motor-control objects."""

    def __init__(self, controller):
        self.controller = controller

    def connect(self) -> None:
        return None

    def move_to(self, position: PlannedPosition | None) -> AchievedPosition | None:
        if position is None:
            return None
        coords = position.coordinates
        if "z" in coords:
            self.controller.probe_positions = (coords["x"], coords["y"], coords["z"])
        else:
            self.controller.probe_positions = (coords["x"], coords["y"])
        self.controller.wait_for_motion_complete()
        reported = self.controller.probe_positions
        if len(reported) == 2:
            achieved = {"x": float(reported[0]), "y": float(reported[1])}
        else:
            achieved = {"x": float(reported[0]), "y": float(reported[1]), "z": float(reported[2])}
        return AchievedPosition(coordinates=achieved)

    def close(self) -> None:
        return None

    def metadata(self) -> dict[str, object]:
        return {"adapter": "LegacyMotorAdapter"}
