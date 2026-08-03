"""Plan C — LibV2 root isolation regression tests.

Covers the env-overridable resolvers added to ``lib/paths.py`` and the
``LibV2Storage`` / ``DecisionCapture`` construction-time root derivation
that stops a pytest run from littering the real in-tree ``LibV2/`` tree
with empty skeleton dirs (or growing ``runtime/training-captures/`` by thousands
of files).

Root cause (pre-fix):
  * ``lib/paths.py`` hardcoded the LibV2 root; ``ED4ALL_LIBV2_ROOT`` was
    honored by the MCP/GUI/CLI layers but NEVER by ``lib/libv2_storage.py``
    or ``lib/decision_capture.py``.
  * ``DecisionCapture.__init__`` does ``LibV2Storage(course_code,
    auto_create=True)`` → ``ensure_directories()`` which mkdir'd
    ``LibV2/courses/<slug>/...`` + ``LibV2/catalog/<COURSE>/...`` in the
    REAL tree for any test exercising an LLM call path, plus mirrored
    decision JSONL into the real ``runtime/training-captures/`` tree.

Precedence contract: explicit kwarg > ``ED4ALL_LIBV2_ROOT`` >
``lib.paths.LIBV2_PATH`` default. ``ED4ALL_LIBV2_ROOT`` does NOT govern
``runtime/training-captures/`` — that has its own ``ED4ALL_TRAINING_CAPTURES_DIR``
override (default ``lib.paths.TRAINING_DIR``).

Test-suite isolation design: the repo-root conftest's session-scoped
autouse fixture ``_ed4all_default_roots_isolated`` monkeypatches the
DEFAULT constants (``lib.paths.LIBV2_PATH`` / ``lib.paths.TRAINING_DIR``)
into a session tmp tree — NOT the env vars — so tests that deliberately
exercise the env/default legs of the resolution chain see unchanged
semantics, while every default-leg WRITE lands in tmp. Session scope
covers module-/session-scoped fixtures that instantiate before any
function-scoped autouse fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lib import paths
from lib.decision_capture import DecisionCapture
from lib.libv2_storage import LibV2Storage

# The REAL in-tree roots (literals, immune to the session-scope patch).
REAL_LIBV2 = paths.PROJECT_ROOT / "LibV2"
REAL_CAPTURES = paths.PROJECT_ROOT / "runtime/training-captures"


def _real_courses_snapshot() -> set:
    courses = REAL_LIBV2 / "courses"
    return set(p.name for p in courses.iterdir()) if courses.exists() else set()


# ---------------------------------------------------------------------------
# Resolver precedence — lib.paths.libv2_path / get_training_captures_dir.
# ---------------------------------------------------------------------------


class TestLibV2PathResolver:
    def test_env_override_wins_over_default(self, tmp_path, monkeypatch):
        custom = tmp_path / "mounted_libv2"
        monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(custom))
        assert paths.libv2_path() == custom

    def test_falls_through_to_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("ED4ALL_LIBV2_ROOT", raising=False)
        # The default leg resolves to the CURRENT module global (patched
        # to session tmp by the autouse isolation fixture) — read at call
        # time, not import time.
        assert paths.libv2_path() == paths.LIBV2_PATH

    def test_blank_env_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("ED4ALL_LIBV2_ROOT", "   ")
        assert paths.libv2_path() == paths.LIBV2_PATH


class TestTrainingCapturesResolver:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        custom = tmp_path / "captures"
        monkeypatch.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(custom))
        assert paths.get_training_captures_dir() == custom

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("ED4ALL_TRAINING_CAPTURES_DIR", raising=False)
        assert paths.get_training_captures_dir() == paths.TRAINING_DIR

    def test_libv2_root_does_not_govern_captures(self, tmp_path, monkeypatch):
        """``ED4ALL_LIBV2_ROOT`` must NOT redirect training-captures."""
        monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path / "lib"))
        monkeypatch.delenv("ED4ALL_TRAINING_CAPTURES_DIR", raising=False)
        assert paths.get_training_captures_dir() == paths.TRAINING_DIR


# ---------------------------------------------------------------------------
# LibV2Storage construction-time root derivation + precedence.
# ---------------------------------------------------------------------------


class TestLibV2StoragePrecedence:
    def test_explicit_kwarg_wins_over_env(self, tmp_path, monkeypatch):
        env_root = tmp_path / "env"
        explicit = tmp_path / "explicit"
        monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(env_root))
        s = LibV2Storage("TST_908", libv2_root=str(explicit))
        assert s.catalog_path == explicit / "catalog" / "TST_908"
        assert s.course_path == explicit / "courses" / "tst-908"

    def test_env_root_used_when_no_kwarg(self, tmp_path, monkeypatch):
        env_root = tmp_path / "env"
        monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(env_root))
        s = LibV2Storage("TST_908")
        assert s.catalog_path == env_root / "catalog" / "TST_908"
        assert s.course_path == env_root / "courses" / "tst-908"

    def test_default_root_when_unset(self, monkeypatch):
        monkeypatch.delenv("ED4ALL_LIBV2_ROOT", raising=False)
        s = LibV2Storage("TST_908")
        # Resolves to the call-time lib.paths.LIBV2_PATH (session tmp
        # under the autouse isolation) — NOT the real in-tree root.
        assert s.catalog_path == paths.LIBV2_PATH / "catalog" / "TST_908"
        assert not str(s.catalog_path).startswith(str(REAL_LIBV2))

    def test_auto_create_default_leg_never_touches_real_tree(
        self, monkeypatch
    ):
        """``auto_create=True`` with NO env and NO kwarg (the exact shape
        of the original leak: ``DecisionCapture`` →
        ``LibV2Storage(course_code, auto_create=True)``) must land in the
        session-isolated default, leaving the real ``LibV2/courses/``
        untouched."""
        monkeypatch.delenv("ED4ALL_LIBV2_ROOT", raising=False)
        before = _real_courses_snapshot()
        s = LibV2Storage("LEAKTEST_999", auto_create=True)
        assert s.course_path.exists()
        assert not str(s.course_path).startswith(str(REAL_LIBV2))
        assert _real_courses_snapshot() == before, (
            "LibV2Storage(auto_create=True) leaked a skeleton dir into the "
            "real in-tree LibV2/courses/ on the default resolution leg."
        )


# ---------------------------------------------------------------------------
# DecisionCapture honors both redirects — no leak into the real trees.
# ---------------------------------------------------------------------------


class TestDecisionCaptureIsolation:
    def test_capture_default_leg_writes_under_isolated_roots(
        self, monkeypatch
    ):
        """A bare DecisionCapture (no env overrides — the default leg)
        must write its primary capture under the isolated LibV2 root and
        its legacy mirror under the isolated training-captures root."""
        monkeypatch.delenv("ED4ALL_LIBV2_ROOT", raising=False)
        monkeypatch.delenv("ED4ALL_TRAINING_CAPTURES_DIR", raising=False)

        before = _real_courses_snapshot()

        cap = DecisionCapture("CAP_777", phase="content-generator", tool="trainforge")
        cap.log_decision(
            decision_type="content_structure",
            decision="x",
            rationale="rationale long enough to clear the minimum length gate",
        )
        cap.close()

        assert not str(cap.output_dir).startswith(str(REAL_LIBV2))
        assert not str(cap.legacy_output_dir).startswith(str(REAL_CAPTURES))
        assert Path(cap.legacy_output_dir).exists()
        assert _real_courses_snapshot() == before

    def test_capture_env_overrides_still_win(self, tmp_path, monkeypatch):
        libv2_root = tmp_path / "lib"
        captures_root = tmp_path / "caps"
        monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
        monkeypatch.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(captures_root))

        cap = DecisionCapture("CAP_778", phase="content-generator", tool="trainforge")
        cap.close()

        assert str(cap.output_dir).startswith(str(libv2_root))
        assert str(cap.legacy_output_dir).startswith(str(captures_root))


# ---------------------------------------------------------------------------
# Autouse session isolation — every test is hermetic by default.
# ---------------------------------------------------------------------------


def test_session_isolation_repoints_default_constants():
    """The session-scoped autouse fixture must have repointed the
    lib-layer default constants away from the real in-tree roots."""
    import os

    assert paths.LIBV2_PATH != REAL_LIBV2, (
        "lib.paths.LIBV2_PATH should be session-patched to tmp"
    )
    assert paths.TRAINING_DIR != REAL_CAPTURES, (
        "lib.paths.TRAINING_DIR should be session-patched to tmp"
    )
    assert os.environ.get("ED4ALL_STATE_RUNS_DIR"), (
        "ED4ALL_STATE_RUNS_DIR should be set by the session isolation fixture"
    )
