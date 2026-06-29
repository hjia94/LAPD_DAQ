"""Per-scope acquisition mode constants.

Every scope runs in exactly one mode for the whole run, chosen at init. The
integer values are the on-the-wire contract used throughout acquisition (the
dispatch in ``scope_runner._read_one_scope`` and ``write_time_array``'s
``is_sequence`` flag both key off them), so the values must not change:

    MODE_SINGLE   = 0   synchronized SINGLE (master/slave). today's default.
    MODE_SEQUENCE = 1   segmented capture, single scope.

``MODE_AVERAGE`` (NORM + on-scope AverageSweeps) is reserved by the unified plan
but not implemented yet; it will take value 2. Sequence completes on the same
STOP + new-signal contract as SINGLE (confirmed against the MAUI manual), so a
single sequence scope rides the existing SINGLE arm/wait path unchanged; a
self-arming set will be added alongside ``arm_scopes_for_trigger`` when a mode
actually needs to skip master/slave arming.
"""

MODE_SINGLE = 0
MODE_SEQUENCE = 1

# Human-readable names accepted in the [scope_modes] config section.
_NAME_TO_MODE = {
    "single": MODE_SINGLE,
    "sequence": MODE_SEQUENCE,
}
_MODE_TO_NAME = {mode: name for name, mode in _NAME_TO_MODE.items()}

VALID_MODE_NAMES = tuple(_NAME_TO_MODE)


def mode_from_name(name):
    """Map a config mode name (case-insensitive) to its integer constant.

    Raises ValueError on an unrecognized name so a typo in [scope_modes] fails
    loudly at config time rather than silently defaulting to SINGLE.
    """
    try:
        return _NAME_TO_MODE[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown scope mode {name!r}; valid modes are "
            f"{', '.join(VALID_MODE_NAMES)}"
        ) from None


def name_from_mode(mode):
    """Inverse of mode_from_name: the config name for a mode constant.

    Used for human-readable log/warning messages so each new mode is labeled
    correctly from the single _NAME_TO_MODE source of truth.
    """
    return _MODE_TO_NAME.get(mode, str(mode))
