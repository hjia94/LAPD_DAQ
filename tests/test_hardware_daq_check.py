import argparse
import configparser
import unittest

from scripts import hardware_daq_check


class HardwareDaqCheckTests(unittest.TestCase):
    def test_parse_move_to_accepts_xy_and_xyz(self):
        self.assertEqual(hardware_daq_check._parse_move_to("1, 2"), (1.0, 2.0))
        self.assertEqual(hardware_daq_check._parse_move_to("1,2,3"), (1.0, 2.0, 3.0))

    def test_parse_move_to_rejects_wrong_dimension(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            hardware_daq_check._parse_move_to("1")

    def test_target_coordinates(self):
        self.assertEqual(hardware_daq_check._target_coordinates((1.0, 2.0)), {"x": 1.0, "y": 2.0})
        self.assertEqual(
            hardware_daq_check._target_coordinates((1.0, 2.0, 3.0)),
            {"x": 1.0, "y": 2.0, "z": 3.0},
        )

    def test_fake_scope_payload_shape(self):
        payload = hardware_daq_check._fake_scope_payload("PauseScope", "C2", 4, 3)

        traces, data, headers = payload["PauseScope"]
        self.assertEqual(traces, ["C2"])
        self.assertEqual(data["C2"].tolist(), [3, 4, 5, 6])
        self.assertEqual(len(headers["C2"]), 346)

    def test_restrict_scope_config_keeps_one_scope(self):
        config = configparser.ConfigParser()
        config.add_section("scope_ips")
        config.set("scope_ips", "scope_a", "1.1.1.1")
        config.set("scope_ips", "scope_b", "2.2.2.2")

        hardware_daq_check._restrict_scope_config(config, "scope_b")

        self.assertEqual(dict(config.items("scope_ips")), {"scope_b": "2.2.2.2"})


if __name__ == "__main__":
    unittest.main()
