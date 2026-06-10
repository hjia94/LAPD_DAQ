"""Test package bootstrap.

The suite runs under the stdlib ``unittest`` runner (no pytest), both as
``python -m unittest discover -s tests -p "test_*.py"`` and as
``python -m unittest tests.<module>``. Neither form puts the repo root or this
``tests/`` directory on ``sys.path`` automatically, yet the test modules import
both production packages (``import offload_engine``, ``from spooling import
...``) and leading-underscore sibling helpers (``from _bmotion_stubs import
...``). This package init adds both once, so individual modules don't each
carry their own ``sys.path.insert`` preamble.

``unittest`` imports this ``__init__`` before collecting any test module under
``tests`` (discover imports the start-directory package; ``tests.<module>``
imports the parent package first), so the paths are in place by the time a test
module's top-level imports run.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

for _path in (_REPO_ROOT, _TESTS_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
