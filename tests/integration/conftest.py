"""Skip ``tests/integration/`` tests unless explicitly requested.

The end-to-end pipeline test is heavy (it shells out to ``ed4all run`` and
runs the full 10-phase textbook-to-course workflow). It's **intentionally
failing** right now — it gates the landing of workers α / β / γ — so it
must not pollute the default test suite.

Opt-in policy: the tests in this directory run only when pytest is invoked
with one of:

    pytest -m slow tests/integration/
    pytest -m integration tests/integration/
    pytest tests/integration/test_pipeline_end_to_end.py  # explicit file

Any invocation that relies on default test-path discovery (``pytest`` from
repo root, or a CI lane that hasn't been updated) skips the whole dir.

Detection is best-effort — we look at ``config.option.markexpr`` and the
``pytest.ini`` invocation args for the explicit opt-in signals.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_THIS_DIR = Path(__file__).resolve().parent
_OPT_IN_MARKER_EXPRS = ("slow", "integration")


def _opted_in_via_markexpr(markexpr: str) -> bool:
    if not markexpr:
        return False
    # Crude but sufficient: any occurrence of an opt-in marker name in the
    # -m expression counts. Deliberately permissive — ``slow and not
    # flaky`` should still run these.
    return any(name in markexpr for name in _OPT_IN_MARKER_EXPRS)


def _opted_in_via_path(invocation_paths: list[str]) -> bool:
    """Did the user explicitly target ``tests/integration/`` or a file within?"""
    for raw in invocation_paths:
        # pytest accepts plain paths and path::test-id specs.
        path_part = raw.split("::", 1)[0]
        candidate = Path(path_part).resolve()
        try:
            candidate.relative_to(_THIS_DIR)
            return True
        except ValueError:
            continue
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    markexpr = getattr(config.option, "markexpr", "") or ""
    if _opted_in_via_markexpr(markexpr):
        return

    invocation_paths = list(config.args or [])
    if _opted_in_via_path(invocation_paths):
        return

    skip_reason = pytest.mark.skip(
        reason=(
            "tests/integration/ skipped by default — run with "
            "`pytest -m slow tests/integration/` or target the file directly."
        )
    )
    for item in items:
        if Path(str(item.fspath)).is_relative_to(_THIS_DIR):
            item.add_marker(skip_reason)
