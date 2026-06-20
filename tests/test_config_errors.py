"""Unit tests for typed INI/TOML configuration errors (Data_Run_bmotion path).

These cover the startup error reporting added so a user can immediately tell a
data run did not start because of a mistake in experiment_config.ini or
bmotion_config.toml -- and where that mistake is.

The modules under test (`acquisition.config_errors`, `acquisition.config`) are
loaded directly, bypassing `acquisition/__init__.py` (which pulls in matplotlib
via the motion package) the same way test_bmotion_config.py does.
"""

import pathlib
import tempfile
import unittest

# config.py depends only on configparser/os + config_errors, so importing the
# submodules directly is cheap and -- importantly -- uses the *real* acquisition
# package, so this test never replaces ``acquisition`` in sys.modules and can't
# leak a stubbed package into other tests that import the full acquisition API.
from acquisition import config
from acquisition import config_errors
from acquisition.config_errors import ConfigError, IniConfigError, TomlConfigError


def _write(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False)
    f.write(text)
    f.close()
    return f.name


class FormatForTerminalTests(unittest.TestCase):
    def test_message_names_file_and_says_run_did_not_start(self):
        err = IniConfigError("oops", file_path="C:/runs/experiment_config.ini")
        out = err.format_for_terminal()
        self.assertIn("DATA RUN DID NOT START", out)
        self.assertIn("experiment_config.ini", out)
        self.assertIn("oops", out)

    def test_where_includes_line_section_key_when_present(self):
        err = TomlConfigError("bad", file_path="x.toml", line=7,
                              section="motion_group", key="drive")
        out = err.format_for_terminal()
        self.assertIn("line 7", out)
        self.assertIn("[motion_group]", out)
        self.assertIn("key 'drive'", out)

    def test_hint_rendered_only_when_set(self):
        with_hint = ConfigError("p", hint="do this").format_for_terminal()
        without = ConfigError("p").format_for_terminal()
        self.assertIn("do this", with_hint)
        self.assertNotIn("Fix:", without)

    def test_edit_instruction_is_file_specific(self):
        # report_config_error relies on this per-subclass attribute instead of
        # isinstance dispatch, so each must name its own file.
        self.assertIn("experiment_config.ini", IniConfigError("p").edit_instruction)
        self.assertIn("bmotion_config.toml", TomlConfigError("p").edit_instruction)
        self.assertEqual(ConfigError("p").edit_instruction, "")


class IniLoadErrorTests(unittest.TestCase):
    def test_missing_file_raises_only_when_required(self):
        missing = str(pathlib.Path(tempfile.gettempdir()) / "definitely_absent_xyz.ini")
        # Default (tolerant) contract: no raise, empty config -- so the many
        # existing callers that relied on this keep working.
        cfg, raw = config.load_experiment_config(missing)
        self.assertEqual(raw, "")
        # required=True (the bmotion entry point) turns it into a typed error.
        with self.assertRaises(IniConfigError) as cm:
            config.load_experiment_config(missing, required=True)
        self.assertIn("not found", str(cm.exception))
        self.assertEqual(cm.exception.file_path, missing)

    def test_no_section_header_raises_ini_error_with_line(self):
        path = _write("spool_dir = D:/spool\n")  # value before any [section]
        with self.assertRaises(IniConfigError) as cm:
            config.load_experiment_config(path)
        self.assertEqual(cm.exception.line, 1)

    def test_duplicate_section_raises_ini_error(self):
        path = _write("[storage]\nspool_dir = a\n[storage]\nspool_dir = b\n")
        with self.assertRaises(IniConfigError):
            config.load_experiment_config(path)

    def test_valid_ini_loads(self):
        path = _write("[storage]\nspool_dir = D:/spool\n[experiment]\nname = run1\n")
        cfg, raw = config.load_experiment_config(path)
        self.assertIn("storage", cfg)
        self.assertIn("spool_dir", raw)


class IniValidateTests(unittest.TestCase):
    def _cfg(self, text):
        cfg, _ = config.load_experiment_config(_write(text))
        return cfg

    def test_missing_spool_dir_raises(self):
        cfg = self._cfg("[experiment]\nname = run1\n")
        with self.assertRaises(IniConfigError) as cm:
            config.validate_bmotion_ini(cfg)
        self.assertEqual(cm.exception.key, "spool_dir")

    def test_missing_experiment_name_raises(self):
        cfg = self._cfg("[storage]\nspool_dir = D:/spool\n")
        with self.assertRaises(IniConfigError) as cm:
            config.validate_bmotion_ini(cfg)
        self.assertEqual(cm.exception.key, "name")

    def test_complete_ini_passes(self):
        cfg = self._cfg("[storage]\nspool_dir = D:/spool\n[experiment]\nname = run1\n")
        config.validate_bmotion_ini(cfg)  # must not raise


if __name__ == "__main__":
    unittest.main()
