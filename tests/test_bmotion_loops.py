"""Unit tests for the bmotion acquisition loop functions.

Runs on any PC: stubs for `bapsf_motion`, `xarray`, and the parts of
`acquisition.scope_runner` that pull in real hardware live in
[_bmotion_stubs.py](_bmotion_stubs.py), which loads `acquisition/bmotion.py`
via importlib and exposes it as `bmotion_module`. setUpModule /
tearDownModule below roundtrip the sys.modules state so the stubs don't
leak into sibling tests in the same Python process.

Verifies the loop invariants documented in commit 3e9c8a2:
  * sequential mode completes group A's full motion list before group B
  * shot_num is a single global counter across groups
  * only the active motion group's positions_array row is written in
    sequential mode
  * total_shots = sum(per-group sizes) * nshots
  * Configuration/bmotion_selection blob records execution_order
"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bmotion_stubs import (
    StubMSA,
    StubMotionGroup,
    StubRunManager,
    bmotion_module,
    install_stubs,
    make_grid_motion_list,
    make_temp_hdf5_with_scopes,
    make_toml_file,
    restore_modules,
)


def setUpModule():
    install_stubs()


def tearDownModule():
    restore_modules()


class BmotionLoopTests(unittest.TestCase):
    def setUp(self):
        # Suppress the half-second sleeps in move_to_index — irrelevant for
        # unit tests and would slow the suite to a crawl.
        self._orig_sleep = bmotion_module.time.sleep
        bmotion_module.time.sleep = lambda *_a, **_kw: None

        # Silence the loop's print statements so the unittest output stays
        # clean. (Per-test rather than module-level so failures still surface
        # via the assertion machinery.)
        self._stdout_ctx = contextlib.redirect_stdout(io.StringIO())
        self._stdout_ctx.__enter__()

        # Two motion groups: A has a 3x1 grid (3 points), B has a 5x1 grid (5 points).
        # Real rectangular grids are required by the writer's validation.
        ml_a = make_grid_motion_list(nx=3, ny=1)
        ml_b = make_grid_motion_list(nx=5, ny=1)
        self.mg_a = StubMotionGroup("A", ml_a)
        self.mg_b = StubMotionGroup("B", ml_b)
        self.rm = StubRunManager({"a": self.mg_a, "b": self.mg_b})

    def tearDown(self):
        bmotion_module.time.sleep = self._orig_sleep
        self._stdout_ctx.__exit__(None, None, None)

    # ----- configure_bmotion_hdf5_group ----------------------------------- #
    def test_configure_hdf5_writes_execution_order(self):
        hdf5_path = make_temp_hdf5_with_scopes(["FakeScope"])
        toml_path = make_toml_file()
        ml_order = {"a": "forward", "b": "backward"}
        total_shots = (3 + 5) * 2  # sequential nshots=2

        bmotion_module.configure_bmotion_hdf5_group(
            hdf5_path, total_shots, len(ml_order), toml_path, self.rm,
            list(ml_order.keys()), ml_order=ml_order,
            execution_order="sequential",
        )

        with h5py.File(hdf5_path, "r") as f:
            blob = json.loads(f["Configuration/bmotion_selection"][()])
            self.assertEqual(blob["execution_order"], "sequential")
            self.assertEqual(blob["mg_keys"], ["a", "b"])
            self.assertEqual(blob["direction"], {"a": "forward", "b": "backward"})

            for name, ml_size in (("A", 3), ("B", 5)):
                self.assertIn(f"Control/Positions/{name}/positions_setup_array", f)
                self.assertIn(f"Control/Positions/{name}/positions_array", f)
                # Raw motion_list dataset is gone now (single source of truth).
                self.assertNotIn(f"Control/Positions/{name}/motion_list", f)

                setup = f[f"Control/Positions/{name}/positions_setup_array"]
                self.assertEqual(setup.shape, (ml_size,))
                self.assertEqual(setup.dtype.names, ("shot_num", "x", "y"))
                self.assertEqual(list(setup["shot_num"]), list(range(1, ml_size + 1)))
                self.assertIn("xpos", setup.attrs)
                self.assertIn("ypos", setup.attrs)
                # Grids built by make_grid_motion_list are nx*1, so ypos has one unique value.
                self.assertEqual(len(setup.attrs["xpos"]), ml_size)
                self.assertEqual(len(setup.attrs["ypos"]), 1)

                ds = f[f"Control/Positions/{name}/positions_array"]
                self.assertEqual(ds.shape, (total_shots,))
                self.assertEqual(ds.dtype.names, ("shot_num", "x", "y"))

    def test_configure_hdf5_defaults_execution_order_to_interleaved(self):
        hdf5_path = make_temp_hdf5_with_scopes(["FakeScope"])
        toml_path = make_toml_file()
        bmotion_module.configure_bmotion_hdf5_group(
            hdf5_path, 5, 1, toml_path, self.rm,
            ["a"], ml_order={"a": "forward"},
        )
        with h5py.File(hdf5_path, "r") as f:
            blob = json.loads(f["Configuration/bmotion_selection"][()])
            self.assertEqual(blob["execution_order"], "interleaved")

    # ----- get_motion_list_size / get_max_motion_list_size ---------------- #
    def test_get_motion_list_size_per_group(self):
        self.assertEqual(bmotion_module.get_motion_list_size(self.rm, "a"), 3)
        self.assertEqual(bmotion_module.get_motion_list_size(self.rm, "b"), 5)

    def test_get_max_motion_list_size(self):
        self.assertEqual(bmotion_module.get_max_motion_list_size(self.rm, ["a", "b"]), 5)
        self.assertEqual(bmotion_module.get_max_motion_list_size(self.rm, ["a"]), 3)

    def test_empty_motion_list_raises(self):
        empty_mg = StubMotionGroup("E", np.zeros((0, 2)))
        rm = StubRunManager({"e": empty_mg})
        with self.assertRaises(RuntimeError):
            bmotion_module.get_motion_list_size(rm, "e")

    # ----- move_to_index -------------------------------------------------- #
    def test_move_to_index_forward_and_backward(self):
        bmotion_module.move_to_index(2, self.rm, {"a": "forward", "b": "backward"})
        self.assertEqual(self.mg_a.move_ml_calls, [2])
        # B has ml_size=5, backward => 5 - 2 - 1 = 2
        self.assertEqual(self.mg_b.move_ml_calls, [2])

        bmotion_module.move_to_index(0, self.rm, {"a": "forward", "b": "backward"})
        self.assertEqual(self.mg_a.move_ml_calls, [2, 0])
        self.assertEqual(self.mg_b.move_ml_calls, [2, 4])  # 5-0-1

    def test_move_to_index_out_of_range_skips_with_warning(self):
        # mg_a has ml_size=3 so index=10 is out of range
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bmotion_module.move_to_index(10, self.rm, {"a": "forward"})
        self.assertEqual(self.mg_a.move_ml_calls, [])
        self.assertTrue(any("out of range" in str(wi.message) for wi in w))

    # ----- _run_interleaved ----------------------------------------------- #
    def test_run_interleaved_iteration_order(self):
        nshots = 2
        max_size = 5  # max(3, 5)
        total_shots = max_size * nshots

        move_calls = []
        shot_calls = []

        def spy_move(index, rm, ml_order_dict):
            move_calls.append((index, dict(ml_order_dict)))

        def spy_take(msa, active_scopes, hdf5_path, run_manager,
                     record_keys, shot_num, nshots_, pbar, estimator=None):
            shot_calls.append((shot_num, list(record_keys)))
            return shot_num + nshots_

        _orig_move = bmotion_module.move_to_index
        _orig_take = bmotion_module._take_shots_at_position
        bmotion_module.move_to_index = spy_move
        bmotion_module._take_shots_at_position = spy_take
        try:
            bmotion_module._run_interleaved(
                StubMSA(), {}, "/dev/null", self.rm,
                {"a": "forward", "b": "forward"}, nshots, total_shots,
            )
        finally:
            bmotion_module.move_to_index = _orig_move
            bmotion_module._take_shots_at_position = _orig_take

        # 5 motion indices, each moves BOTH groups
        self.assertEqual(len(move_calls), max_size)
        for i, (idx, order_dict) in enumerate(move_calls):
            self.assertEqual(idx, i)
            self.assertEqual(set(order_dict.keys()), {"a", "b"})

        # 5 _take_shots calls; each records both groups
        self.assertEqual(len(shot_calls), max_size)
        for i, (sn, rec_keys) in enumerate(shot_calls):
            self.assertEqual(sn, 1 + i * nshots)
            self.assertEqual(rec_keys, ["a", "b"])

    # ----- _run_sequential ------------------------------------------------ #
    def test_run_sequential_iteration_order(self):
        nshots = 2
        total_shots = (3 + 5) * nshots  # 16

        move_calls = []
        shot_calls = []

        def spy_move(index, rm, ml_order_dict):
            move_calls.append((index, dict(ml_order_dict)))

        def spy_take(msa, active_scopes, hdf5_path, run_manager,
                     record_keys, shot_num, nshots_, pbar, estimator=None):
            shot_calls.append((shot_num, list(record_keys)))
            return shot_num + nshots_

        _orig_move = bmotion_module.move_to_index
        _orig_take = bmotion_module._take_shots_at_position
        bmotion_module.move_to_index = spy_move
        bmotion_module._take_shots_at_position = spy_take
        try:
            bmotion_module._run_sequential(
                StubMSA(), {}, "/dev/null", self.rm,
                {"a": "forward", "b": "forward"}, nshots, total_shots,
            )
        finally:
            bmotion_module.move_to_index = _orig_move
            bmotion_module._take_shots_at_position = _orig_take

        # 3 moves for A, then 5 for B; each move targets exactly one group
        self.assertEqual(len(move_calls), 3 + 5)
        for i in range(3):
            idx, order_dict = move_calls[i]
            self.assertEqual(idx, i)
            self.assertEqual(list(order_dict.keys()), ["a"])
        for j in range(5):
            idx, order_dict = move_calls[3 + j]
            self.assertEqual(idx, j)
            self.assertEqual(list(order_dict.keys()), ["b"])

        # Shot count: 8 calls (3 + 5), each records exactly one group
        self.assertEqual(len(shot_calls), 8)
        expected_shotnums = [1 + k * nshots for k in range(8)]
        self.assertEqual([sn for sn, _ in shot_calls], expected_shotnums)
        self.assertEqual(
            [rec for _, rec in shot_calls],
            [["a"]] * 3 + [["b"]] * 5,
        )

    def test_run_sequential_writes_only_active_group(self):
        """End-to-end of _run_sequential against a real temp HDF5 file."""
        nshots = 2
        ml_order = {"a": "forward", "b": "forward"}
        size_a, size_b = 3, 5
        total_shots = (size_a + size_b) * nshots

        hdf5_path = make_temp_hdf5_with_scopes(["FakeScope"])
        toml_path = make_toml_file()
        bmotion_module.configure_bmotion_hdf5_group(
            hdf5_path, total_shots, len(ml_order), toml_path, self.rm,
            list(ml_order.keys()), ml_order=ml_order,
            execution_order="sequential",
        )

        # Stub single_shot_acquisition (the module-local reference imported
        # at top of bmotion.py) to a no-op so we never touch real scopes.
        _orig_single = bmotion_module.single_shot_acquisition
        bmotion_module.single_shot_acquisition = lambda msa, scopes, shot_num, verbose=True: None
        try:
            bmotion_module._run_sequential(
                StubMSA({"FakeScope": "127.0.0.1"}), {}, hdf5_path, self.rm,
                ml_order, nshots, total_shots,
            )
        finally:
            bmotion_module.single_shot_acquisition = _orig_single

        with h5py.File(hdf5_path, "r") as f:
            arr_a = f["Control/Positions/A/positions_array"][:]
            arr_b = f["Control/Positions/B/positions_array"][:]

        # A wrote into rows [0, 3*2); rows [3*2, 16) untouched.
        active_a = size_a * nshots
        self.assertTrue(np.all(arr_a["shot_num"][:active_a] > 0))
        self.assertTrue(np.all(arr_a["shot_num"][active_a:] == 0))

        # B wrote into rows [3*2, 16); rows [0, 3*2) untouched.
        self.assertTrue(np.all(arr_b["shot_num"][:active_a] == 0))
        self.assertTrue(np.all(arr_b["shot_num"][active_a:] > 0))

        # Shot numbers across the run are 1..total_shots, no gaps.
        all_shots = np.concatenate(
            [arr_a["shot_num"][:active_a], arr_b["shot_num"][active_a:]]
        )
        self.assertEqual(list(all_shots), list(range(1, total_shots + 1)))

    # ----- _take_shots_at_position error handling ------------------------- #
    def test_take_shots_skips_on_value_error(self):
        nshots = 2
        ml_order = {"a": "forward"}
        hdf5_path = make_temp_hdf5_with_scopes(["FakeScope"])
        toml_path = make_toml_file()
        bmotion_module.configure_bmotion_hdf5_group(
            hdf5_path, nshots, 1, toml_path, self.rm, ["a"],
            ml_order=ml_order, execution_order="interleaved",
        )

        call_count = {"n": 0}

        def flaky_acquire(msa, scopes, shot_num, verbose=True):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("simulated scope timeout")

        _orig_single = bmotion_module.single_shot_acquisition
        bmotion_module.single_shot_acquisition = flaky_acquire
        try:
            with bmotion_module.tqdm(total=nshots) as pbar:
                new_shot_num = bmotion_module._take_shots_at_position(
                    StubMSA({"FakeScope": "127.0.0.1"}), {}, hdf5_path,
                    self.rm, ["a"], shot_num=1, nshots=nshots, pbar=pbar,
                )
        finally:
            bmotion_module.single_shot_acquisition = _orig_single

        # shot_num advanced through both shots even with one failure
        self.assertEqual(new_shot_num, 1 + nshots)

        with h5py.File(hdf5_path, "r") as f:
            # Failed shot recorded as a skipped group under the scope
            self.assertIn("shot_1", f["FakeScope"])
            self.assertTrue(f["FakeScope/shot_1"].attrs["skipped"])
            self.assertIn("simulated scope timeout",
                          f["FakeScope/shot_1"].attrs["skip_reason"])

            # positions_array row was still written for the failed shot
            arr_a = f["Control/Positions/A/positions_array"][:]
            self.assertEqual(int(arr_a["shot_num"][0]), 1)
            self.assertEqual(int(arr_a["shot_num"][1]), 2)

    # ----- _build_setup_array validation --------------------------------- #
    def test_configure_hdf5_rejects_non_grid_motion_list(self):
        # Points (0,0),(1,2),(3,4): 3 unique x * 3 unique y = 9 != 3 points.
        non_grid = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        mg = StubMotionGroup("NG", non_grid)
        rm = StubRunManager({"ng": mg})
        hdf5_path = make_temp_hdf5_with_scopes(["FakeScope"])
        toml_path = make_toml_file()
        with self.assertRaisesRegex(RuntimeError, "rectangular grid"):
            bmotion_module.configure_bmotion_hdf5_group(
                hdf5_path, 3, 1, toml_path, rm, ["ng"],
                ml_order={"ng": "forward"}, execution_order="interleaved",
            )

    def test_configure_hdf5_rejects_3d_motion_list(self):
        # 2x1x1 grid in 3D = 2 rows of (x,y,z): valid 3D but not supported yet.
        ml_3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mg = StubMotionGroup("M3", ml_3d, space_labels=("x", "y", "z"))
        rm = StubRunManager({"m3": mg})
        hdf5_path = make_temp_hdf5_with_scopes(["FakeScope"])
        toml_path = make_toml_file()
        with self.assertRaisesRegex(RuntimeError, "2D motion only"):
            bmotion_module.configure_bmotion_hdf5_group(
                hdf5_path, 2, 1, toml_path, rm, ["m3"],
                ml_order={"m3": "forward"}, execution_order="interleaved",
            )

    def test_configure_hdf5_rejects_unexpected_axis_labels(self):
        ml = make_grid_motion_list(nx=2, ny=2)
        mg = StubMotionGroup("RT", ml, space_labels=("r", "theta"))
        rm = StubRunManager({"rt": mg})
        hdf5_path = make_temp_hdf5_with_scopes(["FakeScope"])
        toml_path = make_toml_file()
        with self.assertRaisesRegex(RuntimeError, r"axis labels"):
            bmotion_module.configure_bmotion_hdf5_group(
                hdf5_path, 4, 1, toml_path, rm, ["rt"],
                ml_order={"rt": "forward"}, execution_order="interleaved",
            )


if __name__ == "__main__":
    unittest.main()
