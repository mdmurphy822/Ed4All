"""Test configuration for ``lib/semantik/tests``.

``test_reading_order_provenance.py`` drives the REAL cascade internals
(``semantik_structure.cascade`` / ``.structure_graph`` / ``.council``), which
live under ``SemantiK/`` and are imported by their TOP-LEVEL package name. Its
module docstring documents the invocation ``PYTHONPATH=SemantiK:. python -m
pytest …``; without that prefix the module fails at COLLECTION with
``ModuleNotFoundError: No module named 'semantik_structure'`` and pytest aborts
the whole directory, so a bare ``pytest lib/semantik/tests/`` could never be
green.

This conftest puts ``SemantiK/`` on ``sys.path`` the same way the root
``conftest.py`` puts the repo root there, so the documented environment is the
default one. Scoped to this directory: ``lib/semantik/tests`` is not in
``pytest.ini::testpaths``, so the path insert only happens when someone
explicitly targets these tests.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEMANTIK_ROOT = _REPO_ROOT / "SemantiK"

for _p in (str(_REPO_ROOT), str(_SEMANTIK_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
