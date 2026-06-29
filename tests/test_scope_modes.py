"""Unit tests for per-scope acquisition mode parsing (acquisition/scope_modes.py
and acquisition.config.get_scope_modes). No hardware.

Run:

    python -m unittest tests.test_scope_modes
"""

import io
import unittest
from configparser import ConfigParser

from acquisition.config import get_scope_modes
from acquisition.scope_modes import (
    MODE_SINGLE,
    MODE_SEQUENCE,
    mode_from_name,
    name_from_mode,
    VALID_MODE_NAMES,
)


def _msa_with_modes(config):
    """Resolve modes from a config the way MultiScopeAcquisition.__init__ does."""
    scope_names = (dict(config.items("scope_ips"))
                   if config.has_section("scope_ips") else {})
    return get_scope_modes(config, scope_names)


def _config(text):
    cp = ConfigParser()
    cp.read_file(io.StringIO(text))
    return cp


class ModeFromNameTests(unittest.TestCase):
    def test_known_names_map_to_constants(self):
        self.assertEqual(mode_from_name("single"), MODE_SINGLE)
        self.assertEqual(mode_from_name("sequence"), MODE_SEQUENCE)

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(mode_from_name("  SeQuEnCe \n"), MODE_SEQUENCE)

    def test_unknown_name_raises_listing_valid_modes(self):
        with self.assertRaises(ValueError) as ctx:
            mode_from_name("average")
        for valid in VALID_MODE_NAMES:
            self.assertIn(valid, str(ctx.exception))

    def test_name_from_mode_round_trips(self):
        for name in VALID_MODE_NAMES:
            self.assertEqual(name_from_mode(mode_from_name(name)), name)


class LoadScopeModesTests(unittest.TestCase):
    def test_no_section_defaults_all_single(self):
        modes = _msa_with_modes(_config(
            "[scope_ips]\nLeCroy_a = 1.1.1.1\nLeCroy_b = 2.2.2.2\n"))
        self.assertEqual(modes, {"lecroy_a": MODE_SINGLE, "lecroy_b": MODE_SINGLE})

    def test_declared_mode_overrides_default(self):
        modes = _msa_with_modes(_config(
            "[scope_ips]\nLeCroy_a = 1.1.1.1\nLeCroy_b = 2.2.2.2\n"
            "[scope_modes]\nLeCroy_b = sequence\n"))
        self.assertEqual(modes["lecroy_b"], MODE_SEQUENCE)
        # Scopes not listed still default to SINGLE.
        self.assertEqual(modes["lecroy_a"], MODE_SINGLE)

    def test_bad_mode_name_raises(self):
        with self.assertRaises(ValueError):
            _msa_with_modes(_config(
                "[scope_ips]\nLeCroy_a = 1.1.1.1\n"
                "[scope_modes]\nLeCroy_a = bogus\n"))

    def test_unknown_scope_in_modes_is_ignored_not_fatal(self):
        # A [scope_modes] key naming no configured scope warns but does not raise
        # or leak into the returned map.
        modes = _msa_with_modes(_config(
            "[scope_ips]\nLeCroy_a = 1.1.1.1\n"
            "[scope_modes]\ntypo_scope = sequence\n"))
        self.assertEqual(modes, {"lecroy_a": MODE_SINGLE})


if __name__ == "__main__":
    unittest.main()
