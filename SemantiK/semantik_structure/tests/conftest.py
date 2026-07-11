"""Test bootstrap — put the vendored SemantiK package root on sys.path.

These tests import ``semantik_structure.*``. The package root is ``SemantiK/``
(two levels above this file's ``semantik_structure/`` parent), so we prepend it
when not already importable. Mirrors
``semantik_structure/council/tests/conftest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../SemantiK/semantik_structure/tests/conftest.py
#   parents[0]=tests parents[1]=semantik_structure parents[2]=SemantiK
_SEMANTIK_ROOT = Path(__file__).resolve().parents[2]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))
