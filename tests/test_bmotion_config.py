"""Unit tests for the [bmotion] section parser.

No bapsf_motion or hardware imports — a stub RunManager with a `.mgs` dict is
sufficient for the parser's contract.
"""

import configparser
import importlib.util
import pathlib
import unittest

# Import the parser module directly, bypassing `acquisition/__init__.py` which
# pulls in matplotlib via the motion package — none of which the parser needs.
_PARSER_PATH = pathlib.Path(__file__).resolve().parents[1] / "acquisition" / "bmotion_config.py"
_spec = importlib.util.spec_from_file_location("bmotion_config", _PARSER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BmotionSelection = _mod.BmotionSelection
resolve_bmotion_selection = _mod.resolve_bmotion_selection


class _StubRunManager:
    def __init__(self, mg_keys):
        self.mgs = {k: object() for k in mg_keys}


def _make_config(section_text=None):
    cp = configparser.ConfigParser()
    if section_text is not None:
        cp.read_string(section_text)
    return cp


class BmotionConfigTests(unittest.TestCase):
    def test_no_section_defaults_to_all_forward(self):
        rm = _StubRunManager([0, 1, 2])
        sel = resolve_bmotion_selection(_make_config(), rm)
        self.assertIsInstance(sel, BmotionSelection)
        self.assertEqual(sel.mg_keys, [0, 1, 2])
        self.assertEqual(sel.direction, {0: "forward", 1: "forward", 2: "forward"})
        self.assertEqual(sel.execution_order, "interleaved")

    def test_execution_order_defaults_to_interleaved(self):
        rm = _StubRunManager([0, 1])
        cfg = _make_config("[bmotion]\nmotion_groups = all\n")
        self.assertEqual(resolve_bmotion_selection(cfg, rm).execution_order, "interleaved")

    def test_execution_order_sequential(self):
        rm = _StubRunManager([0, 1])
        cfg = _make_config("[bmotion]\nmotion_groups = all\nexecution_order = sequential\n")
        self.assertEqual(resolve_bmotion_selection(cfg, rm).execution_order, "sequential")

    def test_execution_order_invalid_raises(self):
        rm = _StubRunManager([0, 1])
        cfg = _make_config("[bmotion]\nmotion_groups = all\nexecution_order = parallel\n")
        with self.assertRaises(ValueError) as ctx:
            resolve_bmotion_selection(cfg, rm)
        self.assertIn("parallel", str(ctx.exception))

    def test_motion_groups_all_keyword(self):
        rm = _StubRunManager([0, 1, 2])
        cfg = _make_config("[bmotion]\nmotion_groups = all\n")
        self.assertEqual(resolve_bmotion_selection(cfg, rm).mg_keys, [0, 1, 2])

    def test_motion_groups_comma_subset(self):
        rm = _StubRunManager([0, 1, 2])
        cfg = _make_config("[bmotion]\nmotion_groups = 0, 2\n")
        self.assertEqual(resolve_bmotion_selection(cfg, rm).mg_keys, [0, 2])

    def test_motion_groups_whitespace_subset(self):
        rm = _StubRunManager([0, 1, 2])
        cfg = _make_config("[bmotion]\nmotion_groups = 0 2\n")
        self.assertEqual(resolve_bmotion_selection(cfg, rm).mg_keys, [0, 2])

    def test_motion_groups_string_keys(self):
        rm = _StubRunManager(["P22", "P29"])
        cfg = _make_config("[bmotion]\nmotion_groups = P22 P29\n")
        self.assertEqual(resolve_bmotion_selection(cfg, rm).mg_keys, ["P22", "P29"])

    def test_direction_bare_word_broadcasts(self):
        rm = _StubRunManager([0, 1, 2])
        cfg = _make_config("[bmotion]\nmotion_groups = all\ndirection = backward\n")
        sel = resolve_bmotion_selection(cfg, rm)
        self.assertEqual(sel.direction, {0: "backward", 1: "backward", 2: "backward"})

    def test_direction_per_key_mapping(self):
        rm = _StubRunManager([0, 1, 2])
        cfg = _make_config(
            "[bmotion]\nmotion_groups = 0, 1, 2\ndirection = 0=forward, 2=backward\n"
        )
        sel = resolve_bmotion_selection(cfg, rm)
        self.assertEqual(sel.direction, {0: "forward", 1: "forward", 2: "backward"})

    def test_invalid_motion_group_key_raises(self):
        rm = _StubRunManager([0, 1, 2])
        cfg = _make_config("[bmotion]\nmotion_groups = 0, 9\n")
        with self.assertRaises(ValueError) as ctx:
            resolve_bmotion_selection(cfg, rm)
        self.assertIn("9", str(ctx.exception))
        self.assertIn("Valid keys", str(ctx.exception))

    def test_invalid_direction_value_raises(self):
        rm = _StubRunManager([0, 1])
        cfg = _make_config("[bmotion]\nmotion_groups = all\ndirection = sideways\n")
        with self.assertRaises(ValueError):
            resolve_bmotion_selection(cfg, rm)

    def test_direction_mapping_unknown_key_raises(self):
        rm = _StubRunManager([0, 1, 2])
        cfg = _make_config(
            "[bmotion]\nmotion_groups = 0, 1\ndirection = 0=forward, 2=backward\n"
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_bmotion_selection(cfg, rm)
        self.assertIn("2", str(ctx.exception))

    def test_empty_run_manager_raises(self):
        rm = _StubRunManager([])
        with self.assertRaises(RuntimeError):
            resolve_bmotion_selection(_make_config(), rm)


if __name__ == "__main__":
    unittest.main()
