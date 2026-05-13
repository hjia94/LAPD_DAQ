"""Direct Raspberry Pi GPIO trigger adapter."""

from __future__ import annotations


class PiGPIOTriggerAdapter:
    """Adapter around the Raspberry Pi trigger client."""

    def __init__(self, trigger_client):
        self.trigger_client = trigger_client

    def connect(self) -> None:
        status = getattr(self.trigger_client, "get_status", None)
        if status is not None:
            status()

    def trigger(self, shot_num: int) -> None:
        self.trigger_client.send_trigger()

    def close(self) -> None:
        close = getattr(self.trigger_client, "close", None)
        if close is not None:
            close()

    def metadata(self) -> dict[str, object]:
        return {"adapter": "PiGPIOTriggerAdapter"}


PiTriggerAdapter = PiGPIOTriggerAdapter
