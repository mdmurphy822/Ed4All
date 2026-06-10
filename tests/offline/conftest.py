"""Fixtures local to the offline-verification suite.

The ``offline_guard`` fixture wraps a test body in the process-global
``no_network`` socket guard so any network attempt inside the lexical
retrieval path raises loudly. Scoped to this directory only (this conftest is
NOT at the ``tests/`` root) so no other suite is affected. No autouse: the
guard is opt-in per-test, per the plan — a test requests it by naming the
fixture in its signature.

The ``mini_course`` fixture materialises Executor A's shared read-only
``tests/fixtures/retrieval/mini_course/`` corpus into a per-test
``tmp_path`` LibV2 layout (``LibV2/courses/mini-retrieval-101/``) and points
``ED4ALL_LIBV2_ROOT`` at it, so the retriever + citation-anchor + gold-set
loaders resolve the course exactly as they would a real archived course —
without writing into the committed fixture tree.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

import pytest

from lib.testing.no_network import no_network

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "retrieval"
    / "mini_course"
)
COURSE_SLUG = "mini-retrieval-101"


@pytest.fixture()
def offline_guard() -> Iterator[None]:
    """Block all INET network I/O for the duration of the test."""
    with no_network():
        yield


@pytest.fixture()
def mini_course(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy the shared mini-course fixture into a tmp LibV2 layout.

    Returns the ``LibV2/courses/<slug>/`` course dir; sets ``ED4ALL_LIBV2_ROOT``
    to the synthesised LibV2 root so resolution helpers find the course.
    """
    if not FIXTURE_ROOT.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"mini-course fixture missing at {FIXTURE_ROOT}")
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / COURSE_SLUG
    shutil.copytree(FIXTURE_ROOT, course_dir)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return course_dir
