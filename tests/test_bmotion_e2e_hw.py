"""bmotion full-acquisition end-to-end hardware checks (RARELY used).

These run the complete ``run_acquisition_bmotion`` pipeline against real motors
in both interleaved and sequential execution order and verify the resulting
HDF5 layout. They are slow and exercise the whole stack, so they are kept
separate from the fast motor-recovery diagnostics in
``tests/test_bmotion_recovery_hw.py`` (run those routinely; run these only
when validating the full acquisition path).

Gating pattern (shared with the other ``*_hw`` files):
  * Module-level RUN_* flag enables the test class.
  * BMOTION_ALLOW_MOVE is a separate destructive-action gate; without it
    the test fails fast rather than touching motors.

Both ``experiment_config.txt`` and ``bmotion_config.toml`` must be present in
the current working directory.

Run with:

    pytest tests/test_bmotion_e2e_hw.py -v -s
    # or
    python -m unittest tests.test_bmotion_e2e_hw -v
"""

from __future__ import annotations

import configparser
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hardware_check_base import HardwareCheckBase


# --------------------------------------------------------------------------- #
# Enable individual end-to-end checks here. Each is skipped unless True.
# --------------------------------------------------------------------------- #
RUN_BMOTION_INTERLEAVED_CHECK = False
RUN_BMOTION_SEQUENTIAL_CHECK = False

# Safety gate — no test will arm motors unless this is also True.
BMOTION_ALLOW_MOVE = False

# --------------------------------------------------------------------------- #
# Connection info. These paths are resolved relative to the current working
# directory; pass absolute paths to avoid surprises.
# --------------------------------------------------------------------------- #
EXPERIMENT_CONFIG_PATH = "experiment_config.txt"
BMOTION_TOML_PATH = "bmotion_config.toml"

# Use 1 shot per position for diagnostics. Increase only after a clean run.
BMOTION_NSHOTS = 1
# --------------------------------------------------------------------------- #


def _have_bmotion_install() -> bool:
    try:
        import bapsf_motion  # noqa: F401
        import xarray  # noqa: F401
        return True
    except ImportError:
        return False


def _have_required_files() -> bool:
    return Path(EXPERIMENT_CONFIG_PATH).is_file() and Path(BMOTION_TOML_PATH).is_file()


def _write_config_variant(src_config: Path, dst_config: Path, execution_order: str,
                          nshots: int) -> None:
    """Copy the source experiment_config.txt and override [bmotion]
    execution_order plus [nshots].num_duplicate_shots so the hardware test
    can pin both without mutating the source-tree config."""
    cp = configparser.ConfigParser(inline_comment_prefixes=None)
    cp.read(src_config)

    if not cp.has_section("nshots"):
        cp.add_section("nshots")
    cp.set("nshots", "num_duplicate_shots", str(nshots))

    if not cp.has_section("bmotion"):
        cp.add_section("bmotion")
    cp.set("bmotion", "execution_order", execution_order)
    # Keep whatever motion_groups / direction the user already configured.

    with open(dst_config, "w") as f:
        cp.write(f)


class _BmotionHardwareBase(HardwareCheckBase):
    """Shared flag-gating + tempdir layout for bmotion end-to-end tests."""

    run_flag: bool = False
    label: str = "bmotion"
    execution_order: str = "interleaved"

    def _run_flag_skip_message(self) -> str:
        return (
            f"{type(self).__name__} disabled "
            f"(set its RUN_BMOTION_*_CHECK flag to True)"
        )

    def gate_checks(self) -> list[tuple[bool, str]]:
        return [
            (not BMOTION_ALLOW_MOVE,
             "BMOTION_ALLOW_MOVE is False — refusing to command motors"),
            (not _have_bmotion_install(),
             "bapsf_motion / xarray not installed on this machine"),
            (not _have_required_files(),
             f"Missing {EXPERIMENT_CONFIG_PATH} or {BMOTION_TOML_PATH} "
             f"in the current working directory"),
        ]

    def _allocate_tempdir(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.output_path = self.tmp_dir / f"{self.label}_{self.execution_order}.hdf5"
        self.config_path = self.tmp_dir / "experiment_config.txt"
        _write_config_variant(
            Path(EXPERIMENT_CONFIG_PATH), self.config_path,
            execution_order=self.execution_order, nshots=BMOTION_NSHOTS,
        )

    def _run(self) -> None:
        # Late import so the module-level skip messages above fire before
        # bapsf_motion is touched.
        from acquisition import run_acquisition_bmotion

        run_acquisition_bmotion(
            str(self.output_path),
            BMOTION_TOML_PATH,
            str(self.config_path),
        )

    def _read_selection_blob(self) -> dict:
        with h5py.File(self.output_path, "r") as f:
            return json.loads(f["Configuration/bmotion_selection"][()])

    def _read_positions(self, mg_name: str) -> np.ndarray:
        with h5py.File(self.output_path, "r") as f:
            return f[f"Control/Positions/{mg_name}/positions_array"][:]

    def _list_mg_names(self) -> list:
        with h5py.File(self.output_path, "r") as f:
            return list(f["Control/Positions"].keys())


# --------------------------------------------------------------------------- #
class BmotionInterleavedHardwareCheck(_BmotionHardwareBase):
    """End-to-end run with execution_order = interleaved against real motors."""

    run_flag = RUN_BMOTION_INTERLEAVED_CHECK
    label = "bmotion_interleaved"
    execution_order = "interleaved"

    def test_interleaved_end_to_end(self) -> None:
        self._run()

        blob = self._read_selection_blob()
        self.assertEqual(blob["execution_order"], "interleaved")
        self.assertTrue(blob["mg_keys"], "no motion groups selected")

        mg_names = self._list_mg_names()
        self.assertTrue(mg_names, "no Control/Positions/<mg> groups created")

        # In interleaved mode every selected MG should have a populated row
        # at every shot index — there are no idle phases.
        first = self._read_positions(mg_names[0])
        self.assertGreater(len(first), 0)
        self.assertTrue(np.all(first["shot_num"] > 0),
                        "interleaved run left zero rows in the first MG")
        for name in mg_names[1:]:
            arr = self._read_positions(name)
            self.assertEqual(len(arr), len(first),
                             "interleaved MGs should share total_shots length")
            self.assertTrue(np.all(arr["shot_num"] > 0),
                            f"interleaved run left zero rows in MG {name}")


# --------------------------------------------------------------------------- #
class BmotionSequentialHardwareCheck(_BmotionHardwareBase):
    """End-to-end run with execution_order = sequential against real motors."""

    run_flag = RUN_BMOTION_SEQUENTIAL_CHECK
    label = "bmotion_sequential"
    execution_order = "sequential"

    def test_sequential_end_to_end(self) -> None:
        self._run()

        blob = self._read_selection_blob()
        self.assertEqual(blob["execution_order"], "sequential")
        self.assertTrue(blob["mg_keys"], "no motion groups selected")

        mg_names = self._list_mg_names()
        self.assertTrue(mg_names, "no Control/Positions/<mg> groups created")
        if len(mg_names) < 2:
            self.skipTest(
                "Sequential-mode assertions need >= 2 motion groups; "
                f"got {mg_names}. Configure at least two in [bmotion]."
            )

        # Each MG's positions_array should have exactly one active (nonzero)
        # contiguous block; the other MGs' rows at those shot indices should
        # be all-zero (idle-group skip behavior).
        per_mg_active = {}
        for name in mg_names:
            arr = self._read_positions(name)
            active_mask = arr["shot_num"] > 0
            per_mg_active[name] = active_mask
            self.assertTrue(active_mask.any(),
                            f"MG {name} has no shots recorded")

        # No two MGs are active at the same shot index.
        for i, name_i in enumerate(mg_names):
            for name_j in mg_names[i + 1:]:
                overlap = per_mg_active[name_i] & per_mg_active[name_j]
                if overlap.any():
                    first_overlap = int(np.where(overlap)[0][0])
                    self.fail(
                        f"MGs {name_i} and {name_j} both wrote to shot index "
                        f"{first_overlap} — sequential mode should record "
                        f"only the active group"
                    )

        # Combined coverage should equal total_shots (i.e. every shot has
        # exactly one active MG).
        total_shots = len(next(iter(per_mg_active.values())))
        combined = np.zeros(total_shots, dtype=bool)
        for mask in per_mg_active.values():
            combined |= mask
        self.assertTrue(combined.all(),
                        "some shot indices have no active MG row")


if __name__ == "__main__":
    unittest.main()
