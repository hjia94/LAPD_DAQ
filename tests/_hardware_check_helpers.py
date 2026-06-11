"""Shared helpers for hardware diagnostic tests.

Pure functions extracted from the former `scripts/hardware_daq_check.py`. They
have no side effects so they can be unit-tested directly and reused by the
per-instrument hardware tests.

Also home to the env_flag/env_str/env_int readers the hardware tests use for
their run flags and rig-specific values, so an armed configuration lives in
the environment, never in committed source.
"""

from __future__ import annotations

import argparse
import configparser
import os

import numpy as np

LECROY_HEADER_BYTES = 346

_TRUTHY = ("1", "true", "yes", "on")


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean opt-in from the environment ('1'/'true'/'yes'/'on')."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def env_str(name: str, default: str | None = None) -> str | None:
    """Read a string setting from the environment, falling back to default."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_int(name: str, default: int | None = None) -> int | None:
    """Read an integer setting from the environment, falling back to default."""
    value = env_str(name)
    return default if value is None else int(value)


# Shared by the scope/motion/camera hardware checks; resolved relative to the
# current working directory unless LAPD_EXPERIMENT_CONFIG is an absolute path.
EXPERIMENT_CONFIG_PATH = env_str("LAPD_EXPERIMENT_CONFIG", "experiment_config.txt")


def parse_move_to(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) not in (2, 3):
        raise argparse.ArgumentTypeError("move-to must have two or three comma-separated values")
    return values


def target_coordinates(target: tuple[float, ...]) -> dict[str, float]:
    coords = {"x": target[0], "y": target[1]}
    if len(target) == 3:
        coords["z"] = target[2]
    return coords


def fake_scope_payload(scope_name: str, channel: str, points: int, shot_num: int):
    raw = np.arange(points, dtype=np.int16) + np.int16(shot_num)
    return {scope_name: ([channel], {channel: raw}, {channel: fake_lecroy_header(points)})}


def fake_time_array(points: int) -> np.ndarray:
    return np.linspace(0.0, 1.0e-6, points, endpoint=False)


def fake_lecroy_header(points: int) -> bytes:
    try:
        from lab_scopes.lecroy import LeCroyHeader

        header = LeCroyHeader()
        return header.generate_test_data(NTimes=points)
    except Exception:
        return bytes(LECROY_HEADER_BYTES)


def restrict_scope_config(config: configparser.ConfigParser, scope_name: str) -> None:
    if not config.has_section("scope_ips"):
        raise RuntimeError("No [scope_ips] section found.")
    selected = None
    for name, ip_address in config.items("scope_ips"):
        if name.lower() == scope_name.lower():
            selected = (name, ip_address)
            break
    if selected is None:
        available = ", ".join(name for name, _ in config.items("scope_ips"))
        raise RuntimeError(f"Scope {scope_name!r} not found. Available scopes: {available}")
    for name, _ in list(config.items("scope_ips")):
        config.remove_option("scope_ips", name)
    config.set("scope_ips", selected[0], selected[1])


def ensure_fake_scope_config(config: configparser.ConfigParser, scope_name: str) -> None:
    for section in ("scope_ips", "scopes", "channels"):
        if not config.has_section(section):
            config.add_section(section)
    config.set("scope_ips", scope_name, "mock://pause")
    config.set("scopes", scope_name, "Fake pause scope for motor-only hardware check")
    if not config.has_option("channels", f"{scope_name}_C1"):
        config.set("channels", f"{scope_name}_C1", "Fake delayed scope data")
