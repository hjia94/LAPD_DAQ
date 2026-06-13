"""Shared HDF5 structural assertion helpers for the lapd_daq test suite.

Used by:
  test_daq_spool.py  -- offload equivalence and schema checks
"""

import numpy as np


def assert_scope_group(tc, h5, scope_name, expected_shot_count, channels, points=None):
    """Assert scope group exists with correct shot_count and channel datasets."""
    tc.assertIn(scope_name, h5, f"Scope group {scope_name!r} missing")
    scope_group = h5[scope_name]
    tc.assertEqual(scope_group.attrs.get("shot_count"), expected_shot_count,
                   f"{scope_name} shot_count mismatch")

    first_shot = scope_group.get("shot_1")
    tc.assertIsNotNone(first_shot, f"{scope_name}/shot_1 missing")
    assert_channel_datasets(tc, first_shot, channels, points=points)
    tc.assertIn(f"shot_{expected_shot_count}", scope_group,
                f"{scope_name}/shot_{expected_shot_count} missing")


def assert_channel_datasets(tc, shot_group, channels, points=None):
    """Assert each channel has a non-empty _data dataset in shot_group."""
    for channel in channels:
        dataset = shot_group.get(f"{channel}_data")
        tc.assertIsNotNone(dataset, f"{channel}_data missing in {shot_group.name}")
        tc.assertGreater(dataset.shape[-1], 0, f"{channel}_data is empty")
        if points is not None:
            tc.assertEqual(dataset.shape[-1], points,
                           f"{channel}_data length mismatch")


def assert_channel_description_attrs(tc, h5, scope_name, expected):
    """Assert the scope group carries the canonical ``<CH>_description`` attrs.

    ``expected`` maps channel name -> description text. This is the coverage
    the original case bug slipped through: nothing asserted the description
    actually landed in the file.
    """
    scope_group = h5[scope_name]
    for channel, text in expected.items():
        tc.assertEqual(scope_group.attrs.get(f"{channel}_description"), text,
                       f"{scope_name} attr {channel}_description mismatch")


def assert_positions_array(tc, h5, expected_shot_count):
    """Assert Control/Positions/positions_array has the correct shot_num column."""
    positions = h5.get("Control/Positions/positions_array")
    tc.assertIsNotNone(positions, "positions_array missing")
    np.testing.assert_array_equal(
        positions["shot_num"], np.arange(1, expected_shot_count + 1)
    )


def assert_run_status(tc, h5, expected_shot_count):
    """Assert Control/Run/shot_status exists with the correct length."""
    status = h5.get("Control/Run/shot_status")
    tc.assertIsNotNone(status, "Control/Run/shot_status missing")
    tc.assertEqual(len(status), expected_shot_count)


def assert_dataset_filters(tc, dataset, expected_dtype, compression,
                           shuffle, fletcher32):
    """Assert a dataset's dtype and filter settings match expected values."""
    tc.assertEqual(dataset.dtype, np.dtype(expected_dtype),
                   f"{dataset.name} dtype mismatch")
    tc.assertEqual(dataset.compression, compression,
                   f"{dataset.name} compression mismatch")
    tc.assertEqual(dataset.shuffle, shuffle,
                   f"{dataset.name} shuffle mismatch")
    tc.assertEqual(dataset.fletcher32, fletcher32,
                   f"{dataset.name} fletcher32 mismatch")


def assert_hdf5_scope_equivalent(tc, path_a, path_b, scope_name="lpscope"):
    """Assert two HDF5 files have the same scope/shot/position structure.

    Checks: shot groups present in both, per-channel dtype/compression/data
    equality, positions_array equality, and shot_count attribute.
    """
    import h5py
    with h5py.File(path_a, "r") as a, h5py.File(path_b, "r") as b:
        tc.assertEqual(sorted(a[scope_name].keys()), sorted(b[scope_name].keys()))
        # Canonical channel descriptions are scope-group attrs (written once at
        # init); both paths share the skeleton writer, so they must agree.
        for attr in set(a[scope_name].attrs) | set(b[scope_name].attrs):
            if attr.endswith("_description"):
                tc.assertEqual(a[scope_name].attrs.get(attr),
                               b[scope_name].attrs.get(attr), attr)
        for shot in a[scope_name]:
            if not shot.startswith("shot_"):
                continue
            ga, gb = a[scope_name][shot], b[scope_name][shot]
            tc.assertEqual(sorted(ga.keys()), sorted(gb.keys()))
            for ds_name in ga:
                da, db = ga[ds_name], gb[ds_name]
                tc.assertEqual(da.dtype, db.dtype, ds_name)
                if ds_name.endswith("_data"):
                    tc.assertEqual(da.compression, db.compression, ds_name)
                    tc.assertEqual(da.shuffle, db.shuffle, ds_name)
                    tc.assertEqual(da.fletcher32, db.fletcher32, ds_name)
                    tc.assertEqual(da.chunks, db.chunks, ds_name)
                    np.testing.assert_array_equal(da[()], db[()])
        pa = a["Control/Positions"]
        pb = b["Control/Positions"]
        tc.assertEqual(sorted(pa.keys()), sorted(pb.keys()))
        for mg in pa:
            np.testing.assert_array_equal(
                pa[mg]["positions_array"][()], pb[mg]["positions_array"][()])
        tc.assertEqual(a[scope_name].attrs.get("shot_count"),
                       b[scope_name].attrs.get("shot_count"))
