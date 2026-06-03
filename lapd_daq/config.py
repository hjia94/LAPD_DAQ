"""INI compatibility loader and typed internal run configuration."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path


DESCRIPTION_FILENAME = "description.txt"

# Written into the HDF5 ``description`` attribute when no usable description.txt
# is found, so a downstream reader can tell the prose was never filled in.
DESCRIPTION_PLACEHOLDER = (
    "No experiment description provided "
    "(description.txt not found next to the config)"
)


def read_description_file(description_path: str | Path | None) -> str:
    """Read the free-text run description from ``description.txt``.

    The description lives in its own file next to the config (it is no longer a
    config value). A missing/empty/unreadable file never raises; it returns
    :data:`DESCRIPTION_PLACEHOLDER` instead, so a description problem can never
    abort an acquisition.
    """
    if not description_path:
        return DESCRIPTION_PLACEHOLDER
    try:
        text = Path(description_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return DESCRIPTION_PLACEHOLDER
    except Exception:
        return DESCRIPTION_PLACEHOLDER
    return text if text.strip() else DESCRIPTION_PLACEHOLDER


@dataclass(frozen=True)
class ScopeConfig:
    name: str
    ip_address: str
    description: str = ""
    pv_prefix: str | None = None


@dataclass(frozen=True)
class MotionConfig:
    enabled: bool = False
    kind: str = "stationary"
    parameters: dict[str, object] = field(default_factory=dict)
    motor_ips: dict[str, str] = field(default_factory=dict)
    pv_prefix: str | None = None


@dataclass(frozen=True)
class CameraConfig:
    enabled: bool = False
    parameters: dict[str, object] = field(default_factory=dict)
    pv_prefix: str | None = None


@dataclass(frozen=True)
class TriggerConfig:
    enabled: bool = False
    parameters: dict[str, object] = field(default_factory=dict)
    pv_prefix: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """Typed configuration used by the new acquisition framework."""

    config_path: Path
    raw_text: str
    mode: str
    description_path: Path
    scopes: list[ScopeConfig]
    channel_descriptions: dict[str, str]
    num_duplicate_shots: int = 1
    num_run_repeats: int = 1
    motion: MotionConfig = field(default_factory=MotionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    output_path: Path | None = None

    @property
    def experiment_description(self) -> str:
        """Current run description, read live from ``description.txt``.

        Read on access (not cached) so the value written at run start and the
        value overwritten at run finalize each reflect the file's contents at
        that moment.
        """
        return read_description_file(self.description_path)


def load_run_config(config_path: str | Path, mode: str = "stationary",
                    output_path: str | Path | None = None) -> RunConfig:
    """Load the existing INI config into a typed internal model."""

    path = Path(config_path)
    raw_text = path.read_text(encoding="utf-8") if path.exists() else ""
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read_string(raw_text or "")

    scopes = []
    for name, ip_address in _items(parser, "scope_ips").items():
        scopes.append(
            ScopeConfig(
                name=name,
                ip_address=ip_address,
                description=parser.get("scopes", name, fallback=""),
                pv_prefix=parser.get("epics_scope_pvs", name, fallback=None),
            )
        )

    position = _coerce_section(_items(parser, "position"))
    motor_ips = _items(parser, "motor_ips")
    motion_kind = _motion_kind(position) if mode == "grid" else "stationary"
    motion = MotionConfig(
        enabled=mode in {"grid", "bmotion"},
        kind="bmotion" if mode == "bmotion" else motion_kind,
        parameters=position,
        motor_ips=motor_ips,
        pv_prefix=parser.get("epics", "motion_pv_prefix", fallback=None),
    )

    camera_params = _coerce_section(_items(parser, "camera_config"))
    camera = CameraConfig(
        enabled=mode in {"camera", "dropper"},
        parameters=camera_params,
        pv_prefix=parser.get("epics", "camera_pv_prefix", fallback=None),
    )

    trigger = TriggerConfig(
        enabled=mode == "dropper" or parser.has_section("raspberry_pi"),
        parameters=_coerce_section(_items(parser, "raspberry_pi")),
        pv_prefix=parser.get("epics", "trigger_pv_prefix", fallback=None),
    )

    return RunConfig(
        config_path=path,
        raw_text=raw_text,
        mode=mode,
        # The run description lives in description.txt next to the config, not in
        # the [experiment] section. Read live at run start/finalize.
        description_path=(path.parent / DESCRIPTION_FILENAME).resolve(),
        scopes=scopes,
        channel_descriptions=_items(parser, "channels"),
        num_duplicate_shots=parser.getint("nshots", "num_duplicate_shots", fallback=1),
        num_run_repeats=parser.getint("nshots", "num_run_repeats", fallback=1),
        motion=motion,
        camera=camera,
        trigger=trigger,
        output_path=Path(output_path) if output_path is not None else None,
    )


def _items(parser: configparser.ConfigParser, section: str) -> dict[str, str]:
    if not parser.has_section(section):
        return {}
    return {key: value.strip() for key, value in parser.items(section) if value.strip()}


def _coerce_section(values: dict[str, str]) -> dict[str, object]:
    return {key: _coerce_value(value) for key, value in values.items()}


def _coerce_value(value: str) -> object:
    text = value.strip()
    if text.lower() == "none":
        return None
    if "," in text and not text.startswith("{"):
        parts = [part.strip() for part in text.split(",") if part.strip()]
        return tuple(_coerce_value(part) for part in parts)
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _motion_kind(position: dict[str, object]) -> str:
    if not position:
        return "stationary"
    if "probe_list" in position:
        return "45deg"
    if position.get("nz") is not None:
        return "xyz_grid"
    return "xy_grid"
