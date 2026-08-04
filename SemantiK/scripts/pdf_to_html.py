"""Compatibility entry point for :mod:`scripts.ops.pdf_to_html`.

New callers should use ``scripts/ops/pdf_to_html.py``.
"""

from __future__ import annotations

if __package__:
    from .ops.pdf_to_html import main
else:
    import sys
    from pathlib import Path

    _SEMANTIK_ROOT = Path(__file__).resolve().parent.parent
    if str(_SEMANTIK_ROOT) not in sys.path:
        sys.path.insert(0, str(_SEMANTIK_ROOT))

    from scripts.ops.pdf_to_html import main


if __name__ == "__main__":
    raise SystemExit(main())
