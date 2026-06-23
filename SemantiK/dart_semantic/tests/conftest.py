"""Test bootstrap — put the vendored SemantiK package root on sys.path.

These tests import ``dart_semantic.*``. The package root is ``SemantiK/``
(two levels above this file's ``dart_semantic/`` parent), so we prepend it
when not already importable. Mirrors
``dart_semantic/council/tests/conftest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../SemantiK/dart_semantic/tests/conftest.py
#   parents[0]=tests parents[1]=dart_semantic parents[2]=SemantiK
_SEMANTIK_ROOT = Path(__file__).resolve().parents[2]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))
