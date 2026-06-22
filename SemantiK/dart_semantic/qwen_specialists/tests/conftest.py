"""Test bootstrap — put the vendored SemantiK package root on sys.path.

These tests import ``dart_semantic.qwen_specialists.*``. The package root
is ``SemantiK/`` (two levels above this file's ``dart_semantic/`` parent),
so we prepend it when not already importable. Mirrors how the lib/semantik
adapter tests reach the vendored tree, but kept local to the vendored
package so the SemantiK tree stays self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../SemantiK/dart_semantic/qwen_specialists/tests/conftest.py
#   parents[0]=tests parents[1]=qwen_specialists parents[2]=dart_semantic
#   parents[3]=SemantiK  -> the importable package root for ``dart_semantic``.
_SEMANTIK_ROOT = Path(__file__).resolve().parents[3]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))
