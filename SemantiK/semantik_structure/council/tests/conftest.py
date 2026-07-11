"""Test bootstrap — put the vendored SemantiK package root on sys.path.

These tests import ``semantik_structure.council.*``. The package root is
``SemantiK/`` (three levels above this file's ``semantik_structure/`` grandparent),
so we prepend it when not already importable. Mirrors
``semantik_structure/qwen_specialists/tests/conftest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../SemantiK/semantik_structure/council/tests/conftest.py
#   parents[0]=tests parents[1]=council parents[2]=semantik_structure
#   parents[3]=SemantiK  -> the importable package root for ``semantik_structure``.
_SEMANTIK_ROOT = Path(__file__).resolve().parents[3]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))
