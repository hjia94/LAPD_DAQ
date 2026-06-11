"""Shared base class for hardware diagnostic tests.

Provides the tempdir lifecycle and the run-flag/gate skip mechanism reused by
the per-instrument hardware tests (test_scope_hw/test_motion_hw/test_camera_hw)
and the bmotion hardware checks (test_bmotion_recovery_hw).
Lives in a leading-underscore module so unittest discovery never collects the
base class itself.

Hardware-run flags and rig-specific values are read from environment variables
(env_flag/env_str/env_int in _hardware_check_helpers.py) so an enabled flag can
never be committed: the source defaults are always safe, and the DAQ PC opts in
per run, e.g.

    $env:LAPD_RUN_ENCODER_CHECK = "1"
    $env:LAPD_BMOTION_ALLOW_MOVE = "1"
    python -m unittest tests.test_bmotion_recovery_hw

Pure functions for those tests live in _hardware_check_helpers.py; this module
is the one place that carries a unittest.TestCase subclass.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class HardwareCheckBase(unittest.TestCase):
    """Run-flag gating + tempdir lifecycle for hardware diagnostic tests.

    Subclasses set ``run_flag`` (the boolean that enables the test) and
    ``label`` (used in the temp HDF5 filename). Override ``gate_checks`` to add
    extra skip conditions and ``_allocate_tempdir`` for a custom output layout.
    """

    run_flag: bool = False
    label: str = "check"

    def gate_checks(self) -> list[tuple[bool, str]]:
        """Extra (should_skip, message) pairs, evaluated in order after the
        run_flag check and before tempdir allocation."""
        return []

    def setUp(self) -> None:
        if not self.run_flag:
            self.skipTest(self._run_flag_skip_message())
        for should_skip, message in self.gate_checks():
            if should_skip:
                self.skipTest(message)
        self._allocate_tempdir()

    def _run_flag_skip_message(self) -> str:
        return f"{type(self).__name__} disabled (set its run flag to True)"

    def _allocate_tempdir(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._cleanup_tempdir)
        self.tmp_dir = Path(self._tmp.name)
        self.output_path = self.tmp_dir / f"{self.label}_check.hdf5"

    def _cleanup_tempdir(self) -> None:
        try:
            self._tmp.cleanup()
        except OSError:
            # HDF5 file on Windows may still be locked briefly; non-fatal.
            pass
