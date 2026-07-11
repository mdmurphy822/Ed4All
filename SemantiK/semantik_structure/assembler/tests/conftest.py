"""Test bootstrap — put the vendored SemantiK package root on sys.path.

These tests import ``semantik_structure.assembler.*``. The package root is
``SemantiK/`` (three levels above this file's ``semantik_structure/`` parent), so
we prepend it when not already importable. Mirrors the qwen_specialists/tests
conftest so the SemantiK tree stays self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../SemantiK/semantik_structure/assembler/tests/conftest.py
#   parents[0]=tests parents[1]=assembler parents[2]=semantik_structure
#   parents[3]=SemantiK  -> the importable package root for ``semantik_structure``.
_SEMANTIK_ROOT = Path(__file__).resolve().parents[3]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))
