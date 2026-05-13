"""Direct Phantom camera adapter."""

from __future__ import annotations

from pathlib import Path


class PhantomCameraAdapter:
    """Adapter around drivers.phantom_recorder.PhantomRecorder."""

    def __init__(self, recorder, experiment_name: str = "", save_path: str | Path | None = None):
        self.recorder = recorder
        self.experiment_name = experiment_name or "lapd_daq"
        self.save_path = Path(save_path) if save_path is not None else None

    def connect(self) -> None:
        return None

    def arm(self, shot_num: int) -> None:
        self.recorder.start_recording(shot_num)

    def complete(self, shot_num: int):
        from lapd_daq.models import CameraShot

        timestamp = self.recorder.wait_for_recording_completion()
        file_name = f"{self.experiment_name}_shot{shot_num:03d}.cine"
        save_dir = self.save_path
        if save_dir is None:
            save_dir_text = getattr(self.recorder, "config", {}).get("save_path")
            save_dir = Path(save_dir_text) if save_dir_text else Path(".")
        rec_cine = self.recorder.save_cine(str(save_dir / file_name))
        self.recorder.wait_for_save_completion(rec_cine)
        return CameraShot(shot_num=shot_num, file_name=file_name, timestamp=timestamp)

    def close(self) -> None:
        self.recorder.cleanup()

    def metadata(self) -> dict[str, object]:
        return {"adapter": "PhantomCameraAdapter"}
