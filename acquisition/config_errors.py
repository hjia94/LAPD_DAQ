"""Typed, user-facing configuration errors for the bmotion data run.

`Data_Run_bmotion.py` reads two separate files before a run can start:

  * ``experiment_config.ini`` -- the run-time layer (storage paths, experiment
    name, [bmotion] selection). Parsed by :mod:`acquisition.config`.
  * ``bmotion_config.toml``   -- the hardware layer (drives, transforms, motion
    groups). Parsed by ``bapsf_motion``'s ``RunManager``.

When either file is missing a required piece or has a typo, the failure used to
surface as a raw traceback (INI) or, worse, as a cryptic
``AttributeError: 'NoneType' object has no attribute 'terminated'`` raised deep
inside ``bapsf_motion`` during teardown (TOML). In both cases the user could not
tell *which* file was at fault or *where* the mistake was.

This module gives those failures a type (:class:`IniConfigError` /
:class:`TomlConfigError`) and a single ``format_for_terminal`` renderer so the
entry script can print one clear, boxed message that names the file, the
location, and how to fix it -- making it immediately obvious that the run did
not start because of an INI/TOML mistake.
"""

_BOX_WIDTH = 64
_RULE = "=" * _BOX_WIDTH

#: Shared "the file the run expected wasn't there" hint (INI and TOML alike live
#: under base_path), so the wording stays in one place.
MISSING_FILE_HINT = (
    "Check base_path in Data_Run_bmotion.py and that the file exists there."
)


class ConfigError(Exception):
    """A configuration file is missing a required piece or has a typo.

    Carries enough structured context to render an explicit terminal message:
    which file, where in it (line / section / key, any subset), what is wrong
    (the exception message), and how to fix it (``hint``).
    """

    #: Short label for the kind of file, used when no ``file_path`` is given.
    file_kind = "configuration file"

    #: One-line "what to do next" shown after the boxed message. Subclasses set
    #: a file-specific instruction so the reporter doesn't need to know types.
    edit_instruction = ""

    def __init__(self, message, *, file_path=None, line=None,
                 section=None, key=None, hint=None):
        super().__init__(message)
        self.file_path = file_path
        self.line = line
        self.section = section
        self.key = key
        self.hint = hint

    def _where(self):
        """Human-readable location string, or None if nothing is known."""
        parts = []
        if self.line is not None:
            parts.append(f"line {self.line}")
        if self.section is not None:
            parts.append(f"section [{self.section}]")
        if self.key is not None:
            parts.append(f"key '{self.key}'")
        return ", ".join(parts) if parts else None

    def format_for_terminal(self):
        """Render a boxed, multi-line message for printing to the terminal.

        The first line makes it unambiguous that the run did not start and that
        the cause is a config mistake; the body names the file, the location,
        the problem, and the fix.
        """
        lines = [
            _RULE,
            "  DATA RUN DID NOT START -- configuration error",
            _RULE,
            f"  File:    {self.file_path or self.file_kind}",
        ]
        where = self._where()
        if where is not None:
            lines.append(f"  Where:   {where}")
        lines.append(f"  Problem: {self}")
        if self.hint:
            lines.append(f"  Fix:     {self.hint}")
        lines.append(_RULE)
        return "\n".join(lines)


class IniConfigError(ConfigError):
    """A problem in ``experiment_config.ini`` (missing, unparsable, or incomplete)."""

    file_kind = "experiment_config.ini"
    edit_instruction = "  -> Edit experiment_config.ini and re-run."


class TomlConfigError(ConfigError):
    """A problem in ``bmotion_config.toml`` (missing, unparsable, or no motion groups)."""

    file_kind = "bmotion_config.toml"
    edit_instruction = "  -> Edit bmotion_config.toml and re-run."
