"""Unit tests for the pure helpers used by hardware diagnostic tests."""

from __future__ import annotations

import argparse
import configparser
import unittest

from _hardware_check_helpers import (
    fake_scope_payload,
    parse_move_to,
    restrict_scope_config,
    target_coordinates,
)


class HardwareCheckHelperTests(unittest.TestCase):
    def test_parse_move_to_accepts_xy_and_xyz(self):
        self.assertEqual(parse_move_to("1, 2"), (1.0, 2.0))
        self.assertEqual(parse_move_to("1,2,3"), (1.0, 2.0, 3.0))

    def test_parse_move_to_rejects_wrong_dimension(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_move_to("1")

    def test_target_coordinates(self):
        self.assertEqual(target_coordinates((1.0, 2.0)), {"x": 1.0, "y": 2.0})
        self.assertEqual(
            target_coordinates((1.0, 2.0, 3.0)),
            {"x": 1.0, "y": 2.0, "z": 3.0},
        )

    def test_fake_scope_payload_shape(self):
        payload = fake_scope_payload("PauseScope", "C2", 4, 3)
        traces, data, headers = payload["PauseScope"]
        self.assertEqual(traces, ["C2"])
        self.assertEqual(data["C2"].tolist(), [3, 4, 5, 6])
        self.assertEqual(len(headers["C2"]), 346)

    def test_restrict_scope_config_keeps_one_scope(self):
        config = configparser.ConfigParser()
        config.add_section("scope_ips")
        config.set("scope_ips", "scope_a", "1.1.1.1")
        config.set("scope_ips", "scope_b", "2.2.2.2")

        restrict_scope_config(config, "scope_b")

        self.assertEqual(dict(config.items("scope_ips")), {"scope_b": "2.2.2.2"})


if __name__ == "__main__":
    unittest.main()
