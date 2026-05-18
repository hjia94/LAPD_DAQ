"""Parse the [bmotion] section of experiment_config.txt into a validated
selection of motion groups and per-group traversal directions.

The bmotion TOML describes hardware (drives, transforms, motion builders).
This module owns the *run-time* layer on top of it: which subset of motion
groups to actually run today, and which direction to traverse each motion
list.
"""

import configparser
from dataclasses import dataclass
from typing import Any, Dict, List

_VALID_DIRECTIONS = ("forward", "backward")


@dataclass(frozen=True)
class BmotionSelection:
    mg_keys: List[Any]
    direction: Dict[Any, str]


def _resolve_key(token: str, available: Dict[Any, Any]):
    """Mirror the string-then-int lookup used in the original prompt code."""
    if token in available:
        return token
    try:
        token_int = int(token)
    except ValueError:
        return None
    if token_int in available:
        return token_int
    return None


def _parse_motion_groups(raw: str, available: Dict[Any, Any]) -> List[Any]:
    raw = (raw or "").strip()
    if raw == "" or raw.lower() == "all":
        return list(available.keys())

    tokens = [t for t in raw.replace(",", " ").split() if t]
    resolved: List[Any] = []
    for tok in tokens:
        key = _resolve_key(tok, available)
        if key is None:
            raise ValueError(
                f"motion_groups: '{tok}' is not a valid motion-group key. "
                f"Valid keys from the TOML: {list(available.keys())}"
            )
        if key not in resolved:
            resolved.append(key)

    if not resolved:
        raise ValueError("motion_groups resolved to an empty list")
    return resolved


def _parse_direction(raw: str, selected: List[Any], available: Dict[Any, Any]) -> Dict[Any, str]:
    raw = (raw or "").strip()
    if raw == "":
        return {k: "forward" for k in selected}

    if "=" not in raw:
        # Bare word form -> broadcast to all selected keys.
        word = raw.lower()
        if word not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction: '{raw}' is not valid. "
                f"Expected one of {_VALID_DIRECTIONS} or a per-key mapping."
            )
        return {k: word for k in selected}

    # Per-key mapping form: "0=forward, 2=backward"
    mapping: Dict[Any, str] = {k: "forward" for k in selected}
    for pair in raw.replace(",", " ").split():
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"direction: entry '{pair}' is missing '='. "
                f"Use 'key=forward' or 'key=backward'."
            )
        tok, value = pair.split("=", 1)
        tok = tok.strip()
        value = value.strip().lower()
        if value not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction: value '{value}' for key '{tok}' is not valid. "
                f"Expected one of {_VALID_DIRECTIONS}."
            )
        key = _resolve_key(tok, available)
        if key is None:
            raise ValueError(
                f"direction: '{tok}' is not a valid motion-group key. "
                f"Valid keys from the TOML: {list(available.keys())}"
            )
        if key not in selected:
            raise ValueError(
                f"direction: key '{tok}' is not in the selected motion_groups "
                f"({selected}). Add it to motion_groups or remove it here."
            )
        mapping[key] = value
    return mapping


def resolve_bmotion_selection(
    config: configparser.ConfigParser,
    run_manager,
) -> BmotionSelection:
    """Parse [bmotion] from the experiment config.

    [bmotion] absent or empty -> all motion groups, all forward.
    Raises ValueError with an actionable message on any invalid entry.
    """
    available = dict(run_manager.mgs)
    if not available:
        raise RuntimeError("No motion groups found in TOML configuration")

    if config.has_section("bmotion"):
        mg_raw = config.get("bmotion", "motion_groups", fallback="all")
        dir_raw = config.get("bmotion", "direction", fallback="forward")
    else:
        mg_raw = "all"
        dir_raw = "forward"

    selected = _parse_motion_groups(mg_raw, available)
    direction = _parse_direction(dir_raw, selected, available)
    return BmotionSelection(mg_keys=selected, direction=direction)
