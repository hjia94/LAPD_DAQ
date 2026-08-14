"""Keep read_and_analyze/doc/README.md in sync with the code it documents.

The doc carries an explicit "Keep in sync" footer but had drifted: a new CLI
tool (fix_channel_descriptions) and the plot_x_line module were undocumented,
and several config example values were stale (renamed/retyped knobs). These two
checks pin the kinds of drift that actually happened:

  1. Every user-facing module in read_and_analyze/ is mentioned in the doc.
  2. Every CONSTANT shown in the doc's ``python`` example blocks still exists in
     the config module it claims to mirror (analysis_config / smart_trigger_config).

Pure stdlib (ast + re) so it runs on any machine with no analysis deps. It reads
source text and never imports the analysis modules, so matplotlib / lab_scopes /
scipy are not required.
"""

import ast
import pathlib
import re
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PKG_DIR = _REPO_ROOT / "read_and_analyze"
_DOC = _PKG_DIR / "doc" / "README.md"

# Modules that are internal plumbing, not user-facing analysis tools. They are
# intentionally not required in the module table (auto_plot IS mentioned in the
# doc as a note, but we don't force it). Keep this list tight: a genuinely new
# user tool should fail the test until it's documented.
_NOT_REQUIRED = {
    "__init__",
    "analysis_config",        # documented as a config file, not in the module table
    "smart_trigger_config",   # documented as a config file, not in the module table
    "auto_plot",              # internal post-run hook; mentioned as a note, not required
}


def _module_names():
    """User-facing .py modules in read_and_analyze/ (excluding plumbing)."""
    names = []
    for path in sorted(_PKG_DIR.glob("*.py")):
        stem = path.stem
        if stem in _NOT_REQUIRED:
            continue
        names.append(stem)
    return names


def _doc_text():
    return _DOC.read_text(encoding="utf-8")


def _module_constants(path):
    """Module-level UPPER_CASE assignment names in a .py file, via ast."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:  # module level only, not nested scopes
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and re.fullmatch(r"[A-Z][A-Z0-9_]+", target.id):
                    names.add(target.id)
    return names


def _all_knob_names():
    """Every module-level CONSTANT in the package.

    Knobs live in analysis_config.py (shared), smart_trigger_config.py, or at
    the top of the module that owns them (the ``# ---- knobs ----`` convention,
    e.g. WINDOW_US in fluctuation_analysis.py). Scanning every module keeps
    in-module knobs covered by this check instead of silently exempt.
    """
    names = set()
    for path in sorted(_PKG_DIR.glob("*.py")):
        names |= _module_constants(path)
    return names


def _python_block_assignments(text):
    """CONSTANT names assigned in every ```python fenced block of the doc."""
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    names = set()
    for block in blocks:
        for m in re.finditer(r"^([A-Z][A-Z0-9_]+)\s*=", block, flags=re.MULTILINE):
            names.add(m.group(1))
    return names


class TestReadAnalyzeDocSync(unittest.TestCase):
    def test_every_module_is_documented(self):
        text = _doc_text()
        missing = []
        for name in _module_names():
            # The doc links modules as `name.py` (in tables and the footer);
            # require the filename to appear somewhere in the doc.
            if f"{name}.py" not in text:
                missing.append(name)
        self.assertFalse(
            missing,
            "read_and_analyze modules not mentioned in doc/README.md: "
            f"{missing}. Add them to the module table (or, if intentionally "
            "internal, to _NOT_REQUIRED in this test).",
        )

    def test_doc_config_constants_exist(self):
        """Every CONSTANT shown in a doc python block must still exist in the code.

        Guards against stale/renamed knobs in the example blocks, wherever the
        knob lives -- a config module or the owning module's own knob block.
        """
        documented = _python_block_assignments(_doc_text())
        stale = sorted(documented - _all_knob_names())
        self.assertFalse(
            stale,
            "Constants in doc/README.md python blocks no longer exist in any "
            f"read_and_analyze module: {stale}. Update the doc example to match "
            "the renamed/removed knobs.",
        )

    def test_documented_inline_knobs_exist(self):
        """Knobs the doc names in prose (backticked) must exist in the code.

        The doc references module-private knobs such as `WINDOW_US` outside the
        python blocks; without this they could be renamed or deleted and the doc
        would rot silently, which the block-only check above would not catch.
        """
        known = _all_knob_names()
        # Backticked ALL_CAPS names in prose that look like knobs (>=2 chars,
        # underscore or long) -- ignore prose acronyms like HDF5/PASS/FAIL.
        prose = set(re.findall(r"`([A-Z][A-Z0-9]*_[A-Z0-9_]+)`", _doc_text()))
        stale = sorted(prose - known)
        self.assertFalse(
            stale,
            "Knobs named in doc/README.md prose no longer exist in any "
            f"read_and_analyze module: {stale}.",
        )


if __name__ == "__main__":
    unittest.main()
