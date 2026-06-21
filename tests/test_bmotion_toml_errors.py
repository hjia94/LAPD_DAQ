"""Tests for explicit TOML errors when loading the bmotion RunManager.

Focus: `_load_bmotion_run_manager` must turn the three startup failure modes of
bmotion_config.toml into a typed TomlConfigError -- and, crucially, must NOT let
the old cryptic ``AttributeError: 'NoneType' object has no attribute
'terminated'`` escape when the TOML defines no motion groups.

Requires bapsf_motion (the acquisition package imports it). Skipped if absent.
"""

import pathlib
import tempfile
import unittest
from unittest import mock

try:
    import acquisition.bmotion as bmotion_mod
    from acquisition.bmotion import _load_bmotion_run_manager
    from acquisition.config_errors import TomlConfigError
    _HAVE_DEPS = True
except Exception:  # pragma: no cover - environment without bapsf_motion
    _HAVE_DEPS = False


def _write_toml(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    f.write(text)
    f.close()
    return f.name


@unittest.skipUnless(_HAVE_DEPS, "bapsf_motion not available")
class LoadRunManagerErrorTests(unittest.TestCase):
    def test_missing_file_raises_toml_error(self):
        missing = str(pathlib.Path(tempfile.gettempdir()) / "absent_bmotion_xyz.toml")
        with self.assertRaises(TomlConfigError) as cm:
            _load_bmotion_run_manager(missing)
        self.assertIn("not found", str(cm.exception))

    def test_syntax_error_raises_toml_error(self):
        path = _write_toml('[run]\nname = "x\nbroken\n')  # unterminated string
        with self.assertRaises(TomlConfigError) as cm:
            _load_bmotion_run_manager(path)
        self.assertIn("not valid TOML", str(cm.exception))

    def test_no_motion_groups_raises_toml_error_not_attributeerror(self):
        # Valid TOML, but no [motion_group] tables. The library logs-and-swallows
        # this, leaving an empty manager that later blew up with
        # "'NoneType' object has no attribute 'terminated'". We must convert it
        # to a clear TomlConfigError instead.
        path = _write_toml('[run]\nname = "x"\n')
        with self.assertRaises(TomlConfigError) as cm:
            _load_bmotion_run_manager(path)
        self.assertIn("motion group", str(cm.exception).lower())
        self.assertEqual(cm.exception.section, "motion_group")

    def test_bad_drive_config_raises_toml_error(self):
        # A motion group whose drive table is malformed: the library raises while
        # building the drive (here a KeyError on the missing axes config). It
        # must be converted, not allowed to escape.
        path = _write_toml(
            '[run]\nname = "x"\n'
            '[run.mg]\nname = "P"\n'
            '[run.mg.drive]\nname = "BadDrive"\n'
            '[run.mg.drive.axes.0]\nname = "x"\n'
        )
        with self.assertRaises(TomlConfigError) as cm:
            _load_bmotion_run_manager(path)
        self.assertIn("not a valid run configuration", str(cm.exception))

    def test_nonetype_terminated_signature_is_translated(self):
        # Regression for the exact hardware failure: RunManager construction
        # raised ``AttributeError: 'NoneType' object has no attribute
        # 'terminated'`` (a drive/transform that silently failed to build). The
        # wrapper must catch it and emit a typed error with a drive-specific
        # hint -- never let the raw AttributeError reach the user.
        path = _write_toml('[run]\nname = "x"\n[run.mg]\nname = "P"\n')

        def _boom(*a, **k):
            raise AttributeError("'NoneType' object has no attribute 'terminated'")

        with mock.patch.object(bmotion_mod.bmotion.actors, "RunManager", _boom):
            with self.assertRaises(TomlConfigError) as cm:
                _load_bmotion_run_manager(path)
        self.assertIn("drive", cm.exception.hint.lower())


if __name__ == "__main__":
    unittest.main()
