"""Parse the [bmotion] section of experiment_config.ini into a validated
selection of motion groups and per-group traversal directions.

The bmotion TOML describes hardware (drives, transforms, motion builders).
This module owns the *run-time* layer on top of it: which subset of motion
groups to actually run today, and which direction to traverse each motion
list.

Motion groups are selected by **physical drive name** (e.g. "Hades", "Athena",
"Hermes", "Apollo"), matched case-insensitively against each motion group's
``config["drive"]["name"]``. The legacy TOML motion-group key (integer index or
group name) is still accepted as a fallback for backward compatibility.
"""

import configparser
from dataclasses import dataclass
from typing import Any, Dict, List

_VALID_DIRECTIONS = ("forward", "backward")
_VALID_EXECUTION_ORDERS = ("interleaved", "sequential")


@dataclass(frozen=True)
class BmotionSelection:
    mg_keys: List[Any]
    direction: Dict[Any, str]
    execution_order: str = "interleaved"


def _drive_name(mg: Any):
    """Return the physical drive name of a motion group, or None if unavailable.

    The drive name (e.g. "Hades", "Athena") lives at
    ``mg.config["drive"]["name"]`` for a bapsf_motion MotionGroup. Stubs or
    partially-built groups may not expose it, so failures degrade to None.
    """
    try:
        return mg.config["drive"]["name"]
    except (AttributeError, KeyError, TypeError):
        return None


def _build_drive_index(available: Dict[Any, Any]) -> Dict[str, Any]:
    """Map lowercased drive name -> motion-group key.

    Raises ValueError if two motion groups share a drive name, since a name
    would then resolve ambiguously.
    """
    index: Dict[str, Any] = {}
    for key, mg in available.items():
        name = _drive_name(mg)
        if name is None:
            continue
        lname = str(name).lower()
        if lname in index and index[lname] != key:
            raise ValueError(
                f"motion_groups: drive name '{name}' is shared by motion groups "
                f"{index[lname]!r} and {key!r}; cannot select by drive name. "
                f"Fix the TOML so each drive name is unique."
            )
        index[lname] = key
    return index


def _valid_token_hint(available: Dict[Any, Any], drive_index: Dict[str, Any]) -> str:
    """Human-readable hint listing the tokens a user may type."""
    if drive_index:
        drives = sorted({str(_drive_name(available[k])) for k in available
                         if _drive_name(available[k]) is not None})
        return f"Valid drive names: {drives} (or TOML keys: {list(available.keys())})"
    return f"Valid keys from the TOML: {list(available.keys())}"


def _resolve_key(token: str, available: Dict[Any, Any], drive_index: Dict[str, Any]):
    """Resolve a token to a motion-group key.

    Order: (1) drive name, case-insensitive; (2) fallback to the legacy
    string-then-int match against the TOML motion-group keys.
    """
    key = drive_index.get(token.strip().lower())
    if key is not None:
        return key
    if token in available:
        return token
    try:
        token_int = int(token)
    except ValueError:
        return None
    if token_int in available:
        return token_int
    return None


def _parse_motion_groups(raw: str, available: Dict[Any, Any],
                         drive_index: Dict[str, Any]) -> List[Any]:
    raw = (raw or "").strip()
    if raw == "" or raw.lower() == "all":
        return list(available.keys())

    tokens = [t for t in raw.replace(",", " ").split() if t]
    resolved: List[Any] = []
    for tok in tokens:
        key = _resolve_key(tok, available, drive_index)
        if key is None:
            raise ValueError(
                f"motion_groups: '{tok}' is not a valid drive name or "
                f"motion-group key. {_valid_token_hint(available, drive_index)}"
            )
        if key not in resolved:
            resolved.append(key)

    if not resolved:
        raise ValueError("motion_groups resolved to an empty list")
    return resolved


def _parse_direction(raw: str, selected: List[Any], available: Dict[Any, Any],
                     drive_index: Dict[str, Any]) -> Dict[Any, str]:
    raw = (raw or "").strip()
    if raw == "":
        return {k: "forward" for k in selected}

    if "=" not in raw:
        # Bare word form -> broadcast to all selected keys.
        word = raw.lower()
        if word not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction: '{raw}' is not valid. "
                f"Expected one of {_VALID_DIRECTIONS} or a per-drive mapping."
            )
        return {k: word for k in selected}

    # Per-drive mapping form: "Hades=forward, Athena=backward"
    mapping: Dict[Any, str] = {k: "forward" for k in selected}
    for pair in raw.replace(",", " ").split():
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"direction: entry '{pair}' is missing '='. "
                f"Use 'drive=forward' or 'drive=backward'."
            )
        tok, value = pair.split("=", 1)
        tok = tok.strip()
        value = value.strip().lower()
        if value not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction: value '{value}' for '{tok}' is not valid. "
                f"Expected one of {_VALID_DIRECTIONS}."
            )
        key = _resolve_key(tok, available, drive_index)
        if key is None:
            raise ValueError(
                f"direction: '{tok}' is not a valid drive name or "
                f"motion-group key. {_valid_token_hint(available, drive_index)}"
            )
        if key not in selected:
            raise ValueError(
                f"direction: '{tok}' is not in the selected motion_groups "
                f"({selected}). Add it to motion_groups or remove it here."
            )
        mapping[key] = value
    return mapping


def _parse_execution_order(raw: str) -> str:
    raw = (raw or "").strip().lower()
    if raw == "":
        return "interleaved"
    if raw not in _VALID_EXECUTION_ORDERS:
        raise ValueError(
            f"execution_order: '{raw}' is not valid. "
            f"Expected one of {_VALID_EXECUTION_ORDERS}."
        )
    return raw


def resolve_bmotion_selection(
    config: configparser.ConfigParser,
    run_manager,
) -> BmotionSelection:
    """Parse [bmotion] from the experiment config.

    [bmotion] absent or empty -> all motion groups, all forward, interleaved.
    Raises ValueError with an actionable message on any invalid entry.
    """
    available = dict(run_manager.mgs)
    if not available:
        raise RuntimeError("No motion groups found in TOML configuration")

    if config.has_section("bmotion"):
        mg_raw = config.get("bmotion", "motion_groups", fallback="all")
        dir_raw = config.get("bmotion", "direction", fallback="forward")
        order_raw = config.get("bmotion", "execution_order", fallback="interleaved")
    else:
        mg_raw = "all"
        dir_raw = "forward"
        order_raw = "interleaved"

    drive_index = _build_drive_index(available)
    selected = _parse_motion_groups(mg_raw, available, drive_index)
    direction = _parse_direction(dir_raw, selected, available, drive_index)
    execution_order = _parse_execution_order(order_raw)
    return BmotionSelection(
        mg_keys=selected,
        direction=direction,
        execution_order=execution_order,
    )
