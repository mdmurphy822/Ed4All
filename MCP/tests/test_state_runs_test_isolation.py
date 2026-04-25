"""Audit guard: unit tests must not pollute project ``state/runs/``.

History (Wave 74): an audit found 18 ephemeral
``run_20260424_HHMMSS`` directories left behind in the project's
``state/runs/`` by unit tests writing to project state instead of
``tmp_path``. This test asserts that no such leftover directories
exist after a test run.

The pattern matched is the timestamped scratch dir produced by
``TaskExecutor`` when ``run_id`` is auto-generated and no
``ED4ALL_STATE_RUNS_DIR`` override is in place. Any test that needs
to exercise that codepath must opt into the ``state_runs_isolated``
fixture defined in the repo-root ``conftest.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

from lib.paths import STATE_PATH


# Pattern of the leaked scratch dirs: ``run_YYYYMMDD_HHMMSS``.
_LEAK_PATTERN = re.compile(r"^run_\d{8}_\d{6}$")


def test_no_test_creates_state_runs_dirs() -> None:
    """The project ``state/runs/`` must contain no run_<date>_<time> leftovers."""
    runs_dir = STATE_PATH / "runs"
    if not runs_dir.exists():
        return  # No state/runs at all → no leakage.

    leaked = sorted(
        entry.name
        for entry in runs_dir.iterdir()
        if entry.is_dir() and _LEAK_PATTERN.match(entry.name)
    )
    assert not leaked, (
        f"Found {len(leaked)} run_<date>_<time> dirs in project state/runs/ "
        f"— a test is leaking state. Leaked dirs: {leaked}. "
        f"Use the ``state_runs_isolated`` fixture from the repo-root "
        f"conftest.py so tests redirect into tmp_path."
    )
