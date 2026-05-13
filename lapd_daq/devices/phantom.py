"""Direct Phantom camera adapter."""

from __future__ import annotations


class PhantomCameraAdapter:
    """Adapter around drivers.phantom_recorder.PhantomRecorder."""

    def __init__(self, recorder, experiment_name: str = ""):
        self.recorder = recorder
        self.experiment_name = experiment_name or "lapd_daq"

    def connect(self) -> None:
        return None

    def arm(self, shot_num: int) -> None:
        self.recorder.start_recording(shot_num)

    def complete(self, shot_num: int):
        from lapd_daq.models import CameraShot

        timestamp = self.recorder.wait_for_recording_completion()
        file_name = f"{self.experiment_name}_shot{shot_num:03d}.cine"
        rec_cine = self.recorder.save_cine(file_name)
        self.recorder.wait_for_save_completion(rec_cine)
        return CameraShot(shot_num=shot_num, file_name=file_name, timestamp=timestamp)

    def close(self) -> None:
        self.recorder.cleanup()

    def metadata(self) -> dict[str, object]:
        return {"adapter": "PhantomCameraAdapter"}
