"""Test bootstrap — put the vendored SemantiK package root on sys.path.

These tests import ``dart_semantic.council.*``. The package root is
``SemantiK/`` (three levels above this file's ``dart_semantic/`` grandparent),
so we prepend it when not already importable. Mirrors
``dart_semantic/qwen_specialists/tests/conftest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../SemantiK/dart_semantic/council/tests/conftest.py
#   parents[0]=tests parents[1]=council parents[2]=dart_semantic
#   parents[3]=SemantiK  -> the importable package root for ``dart_semantic``.
_SEMANTIK_ROOT = Path(__file__).resolve().parents[3]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))
