# -*- coding: utf-8 -*-
"""
45-degree probe acquisition entry point.

This workflow has not been migrated into the refactored acquisition package yet.
Keep using the pre-refactor hardware PC workflow for 45-degree runs until this
mode is explicitly ported and mock-tested.
"""

UNSUPPORTED_MESSAGE = (
    "45-degree acquisition is not migrated in refactor/reorganize-daq-folders. "
    "Use the known pre-refactor hardware PC workflow until 45-degree support is "
    "implemented in the config-driven acquisition runner."
)


def main():
    raise SystemExit(UNSUPPORTED_MESSAGE)


if __name__ == "__main__":
    main()
