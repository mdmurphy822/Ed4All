"""Regenerate docs/wcag_coverage.md from the canonical coverage table.

Usage:
    .venv/bin/python scripts/gen_wcag_coverage_doc.py

The doc is GENERATED — edit
``semantik_structure/gates/wcag_coverage.py:COVERAGE`` and re-run this.
``tests/test_wcag_coverage.py`` fails when the two drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from semantik_structure.gates.wcag_coverage import render_markdown  # noqa: E402

OUT = REPO_ROOT / "docs" / "wcag_coverage.md"


def main() -> int:
    OUT.write_text(render_markdown(), encoding="utf-8")
    print(f"[wcag-coverage] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
