"""Tests for the old-file retrofit tool and the layout-aware description reader.

Old runs carry channel descriptions only as per-shot dataset attrs (or not at
all), but always store the verbatim config in Configuration/experiment_config.
The retrofit tool re-parses that stored config and writes the canonical
``<CH>_description`` scope-group attrs; the reader prefers those attrs and
falls back to the first data-carrying shot for unfixed old files.
"""

import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from read_and_analyze.fix_channel_descriptions import fix_file
from read_and_analyze.read_bmotion_data import read_channel_descriptions

_OLD_CONFIG_TEXT = """\
[scopes]
LPscope = LeCroy test scope

[channels]
LPscope_C1 = LP isat
LPscope_C2 = LP vsweep
"""


def _build_old_format_file(path, with_per_shot_attrs=True):
    """An old-layout file: stored config, skipped shot_1, data in shot_2."""
    with h5py.File(path, "w") as f:
        f.create_group("Control")
        cfg = f.create_group("Configuration")
        cfg.create_dataset("experiment_config", data=np.bytes_(_OLD_CONFIG_TEXT))

        scope = f.create_group("lpscope")
        scope.create_dataset("time_array", data=np.linspace(0, 1e-3, 8))

        # First shot skipped: marker group only, no *_data datasets. The old
        # lab_scopes reader hardcoded shot_1 and found nothing for such runs.
        skip = scope.create_group("shot_1")
        skip.attrs["skipped"] = True

        shot = scope.create_group("shot_2")
        for ch in ("C1", "C2"):
            ds = shot.create_dataset(f"{ch}_data", data=np.zeros(8, dtype=np.int16))
            if with_per_shot_attrs:
                ds.attrs["description"] = f"old per-shot {ch}"


class FixChannelDescriptionsTests(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp(prefix="fixdesc_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        self.path = os.path.join(d, "old_run.hdf5")
        _build_old_format_file(self.path)

    def _scope_attrs(self):
        with h5py.File(self.path, "r") as f:
            return dict(f["lpscope"].attrs)

    def test_writes_attrs_from_stored_config(self):
        written = fix_file(self.path)
        self.assertEqual(written["lpscope"],
                         {"C1": "LP isat", "C2": "LP vsweep"})
        attrs = self._scope_attrs()
        self.assertEqual(attrs["C1_description"], "LP isat")
        self.assertEqual(attrs["C2_description"], "LP vsweep")

    def test_skips_when_already_fixed_unless_forced(self):
        fix_file(self.path)
        with h5py.File(self.path, "a") as f:
            f["lpscope"].attrs["C1_description"] = "hand-edited"

        # Second pass without --force must not clobber.
        written = fix_file(self.path)
        self.assertEqual(written["lpscope"], {})
        self.assertEqual(self._scope_attrs()["C1_description"], "hand-edited")

        # --force rewrites from the stored config.
        fix_file(self.path, force=True)
        self.assertEqual(self._scope_attrs()["C1_description"], "LP isat")

    def test_channel_without_config_entry_gets_sentinel(self):
        with h5py.File(self.path, "a") as f:
            f["lpscope/shot_2"].create_dataset(
                "C3_data", data=np.zeros(8, dtype=np.int16))
        written = fix_file(self.path)
        self.assertEqual(written["lpscope"]["C3"],
                         "Channel C3 - No description available")


class ReadChannelDescriptionsTests(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp(prefix="readdesc_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        self.path = os.path.join(d, "run.hdf5")

    def test_prefers_scope_group_attrs(self):
        _build_old_format_file(self.path)
        fix_file(self.path)
        with h5py.File(self.path, "r") as f:
            descs = read_channel_descriptions(f, "lpscope")
        self.assertEqual(descs, {"C1": "LP isat", "C2": "LP vsweep"})

    def test_falls_back_to_first_data_shot_for_old_files(self):
        # No scope attrs, shot_1 skipped: must read shot_2's per-shot attrs
        # (the old lab_scopes reader returned {} here).
        _build_old_format_file(self.path)
        with h5py.File(self.path, "r") as f:
            descs = read_channel_descriptions(f, "lpscope")
        self.assertEqual(descs, {"C1": "old per-shot C1", "C2": "old per-shot C2"})

    def test_missing_scope_returns_empty(self):
        _build_old_format_file(self.path)
        with h5py.File(self.path, "r") as f:
            self.assertEqual(read_channel_descriptions(f, "nope"), {})


if __name__ == "__main__":
    unittest.main()
