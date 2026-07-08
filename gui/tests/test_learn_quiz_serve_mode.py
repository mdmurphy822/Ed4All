"""Serve-mode + wiring for the learner quiz player (Q6 + L1).

The quiz.js front-end module is service-layer tested (``test_quiz_service`` +
``test_learn_quiz_router``), but nothing loaded it into the learner page. This
lane mounts it: ``index.html`` gains a Quizzes section and loads a thin
``quiz-panel.js`` integration that imports the tested ``quiz.js`` functions.

These tests assert (a) the learner page references the quiz module, (b) the
integration imports the tested quiz module + calls its list/mount entry points,
and (c) the new static assets serve over the ``/learn/`` mount. The A11y of the
new markup is gated in ``test_learner_a11y_gate`` (extended in the same lane).

Skipped on a default install (no ``fastapi``/``uvicorn``), mirroring the rest of
``gui/tests``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parents[1] / "static" / "learn"


# --------------------------------------------------------------------------- #
# 1) Static shape — the page references the quiz module + carries the section.
# --------------------------------------------------------------------------- #


def test_index_loads_quiz_panel_module():
    """The learner page loads the quiz-panel integration as an ES module."""
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    assert 'type="module"' in html
    assert 'src="/learn/quiz-panel.js"' in html, (
        "learner index.html must load the quiz-panel module"
    )
    # It still loads the ask surface (this is additive, not a replacement).
    assert 'src="/learn/learn.js"' in html


def test_index_carries_quiz_section_markup():
    """The Quizzes section shell (heading, load button, list + host regions)."""
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="quizzes"' in html
    assert 'id="quizzes-h"' in html and "aria-labelledby=\"quizzes-h\"" in html
    assert 'id="quiz-load-btn"' in html
    assert 'id="quiz-list-region"' in html
    assert 'id="quiz-host"' in html
    # The list region announces load/empty/error copy politely but is NOT a
    # second role=status region (the ask #status stays the single one).
    assert "aria-live=\"polite\"" in html
    assert 'id="quiz-list-region" aria-live="polite"' in html


def test_quiz_panel_imports_and_calls_tested_module():
    """quiz-panel.js reuses the tested quiz.js (list + one-call mount), not a
    re-implementation, and defends its own list markup with escapeHtml."""
    js = (_STATIC / "quiz-panel.js").read_text(encoding="utf-8")
    assert 'from "/learn/quiz.js"' in js, "must import the tested quiz module"
    assert "listQuizzes" in js and "mountQuiz" in js
    # Empty-list + error states are explicit copy, not accidental blanks.
    assert "No quizzes available for this course yet." in js
    assert "Could not load quizzes" in js
    assert "Could not open this quiz" in js
    # Dynamic list text is escaped before it touches innerHTML.
    assert "escapeHtml" in js


# --------------------------------------------------------------------------- #
# 2) Static JS validity (node --check when available).
# --------------------------------------------------------------------------- #


def test_quiz_panel_js_node_check():
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    panel = _STATIC / "quiz-panel.js"
    proc = subprocess.run(
        [node, "--check", str(panel)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------- #
# 3) Serve mode — the quiz assets serve over the /learn/ mount (full mode) and
#    at the learner-mode root. Skipped without the gui extra.
# --------------------------------------------------------------------------- #


def _client_full(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from gui.app import create_app  # noqa: PLC0415

    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_TOKEN", raising=False)
    monkeypatch.setattr("gui.auth._settings_token", lambda: None)
    return TestClient(create_app())


def test_quiz_assets_served_in_full_mode(state_dir, libv2_root, monkeypatch):
    client = _client_full(monkeypatch)
    # The learner page + both JS modules ride the open /learn/ mount.
    assert client.get("/learn/index.html").status_code == 200
    assert client.get("/learn/quiz.js").status_code == 200
    assert client.get("/learn/quiz-panel.js").status_code == 200
    # The tested quiz module's own import target (learner identity) also serves.
    assert client.get("/learn/learner_id.js").status_code == 200


def test_quiz_assets_served_in_learner_mode(state_dir, libv2_root, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from gui.app import create_app  # noqa: PLC0415

    monkeypatch.delenv("ED4ALL_GUI_MODE", raising=False)
    monkeypatch.delenv("ED4ALL_GUI_LEARNER", raising=False)
    client = TestClient(create_app(learner_only=True))
    # In learner-only mode the learn subtree is served at the root ``/``.
    assert client.get("/quiz.js").status_code == 200
    assert client.get("/quiz-panel.js").status_code == 200
