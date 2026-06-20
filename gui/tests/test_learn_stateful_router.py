"""Router contract tests for the W10 §10 STATEFUL learner endpoints.

Covers:
  * POST /api/learn/quiz/{slug}/{assessment_id}/grade  — persists an attempt.
  * GET  /api/learn/quiz/{slug}/{assessment_id}/attempts — caller's own attempts.
  * POST/GET /api/learn/discussion/{slug}/{topic_id}/posts — threading + replies.
  * POST /api/learn/assignment/{slug}/{assignment_id}/submit — text + file upload.
  * GET  /api/learn/assignment/{slug}/{assignment_id}/submission — caller's own.
  * X-Learner-Id OWNERSHIP isolation (learner A can't read learner B's records).
  * Stored-XSS rejection (a <script>/<form action> in a post is stripped).
  * File-upload validation (oversized / disallowed-ext rejected over HTTP).

State is isolated via the ``state_dir`` (ED4ALL_STATE_RUNS_DIR) + ``libv2_root``
redirects. The quiz fixture writes loose QTI under the temp LibV2 course so the
existing grade route can locate + grade it. No hardcoded production slug.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("bs4")
from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402

_SLUG = "stateful-test-course"

_QUIZ_XML = """<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">
  <assessment ident="quiz_01" title="Quiz 1">
    <section ident="root">
      <item ident="q1">
        <presentation>
          <material><mattext texttype="text/html">2 + 2?</mattext></material>
          <response_lid ident="r1" rcardinality="Single">
            <render_choice>
              <response_label ident="A"><material><mattext>3</mattext></material></response_label>
              <response_label ident="B"><material><mattext>4</mattext></material></response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes><decvar varname="SCORE" vartype="Integer"/></outcomes>
          <respcondition>
            <conditionvar><varequal respident="r1">B</varequal></conditionvar>
            <setvar action="Set" varname="SCORE">1</setvar>
          </respcondition>
        </resprocessing>
      </item>
    </section>
  </assessment>
</questestinterop>
"""


@pytest.fixture
def client(state_dir: Path, libv2_root: Path):
    adir = libv2_root / "courses" / _SLUG / "06_assessments"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "quiz_01.xml").write_text(_QUIZ_XML, encoding="utf-8")
    return TestClient(create_app())


# --------------------------------------------------------- attempt persistence


def test_grade_persists_attempt_and_lists_it(client):
    hdr = {"X-Learner-Id": "learner-A"}
    g = client.post(
        f"/api/learn/quiz/{_SLUG}/quiz_01/grade",
        json={"answers": {"q1": "B"}},
        headers=hdr,
    )
    assert g.status_code == 200
    assert g.json()["score"] == 1

    a = client.get(f"/api/learn/quiz/{_SLUG}/quiz_01/attempts", headers=hdr)
    assert a.status_code == 200
    attempts = a.json()
    assert len(attempts) == 1
    assert attempts[0]["score"] == 1
    assert attempts[0]["answers"] == {"q1": "B"}
    assert attempts[0]["ts"]  # server-injected


def test_attempt_ownership_isolation(client):
    client.post(
        f"/api/learn/quiz/{_SLUG}/quiz_01/grade",
        json={"answers": {"q1": "B"}},
        headers={"X-Learner-Id": "learner-A"},
    )
    # Learner B reads their OWN (empty) attempts — never learner A's.
    b = client.get(
        f"/api/learn/quiz/{_SLUG}/quiz_01/attempts",
        headers={"X-Learner-Id": "learner-B"},
    )
    assert b.status_code == 200
    assert b.json() == []
    # Learner A sees their attempt.
    a = client.get(
        f"/api/learn/quiz/{_SLUG}/quiz_01/attempts",
        headers={"X-Learner-Id": "learner-A"},
    )
    assert len(a.json()) == 1


def test_attempts_without_learner_id_is_empty(client):
    # No header → no owner → no history (anonymous caller).
    a = client.get(f"/api/learn/quiz/{_SLUG}/quiz_01/attempts")
    assert a.status_code == 200
    assert a.json() == []


def test_grade_without_learner_id_still_scores(client):
    # Persistence is best-effort; grading returns the score regardless.
    g = client.post(
        f"/api/learn/quiz/{_SLUG}/quiz_01/grade", json={"answers": {"q1": "A"}}
    )
    assert g.status_code == 200
    assert g.json()["score"] == 0


# ------------------------------------------------------------------ discussion


def test_discussion_post_and_read(client):
    p = client.post(
        f"/api/learn/discussion/{_SLUG}/topic_1/posts",
        json={"body": "First post!"},
        headers={"X-Learner-Id": "learner-A"},
    )
    assert p.status_code == 200
    root_id = p.json()["post_id"]

    r = client.post(
        f"/api/learn/discussion/{_SLUG}/topic_1/posts",
        json={"body": "A reply", "parent_post_id": root_id},
        headers={"X-Learner-Id": "learner-B"},
    )
    assert r.status_code == 200

    thread = client.get(f"/api/learn/discussion/{_SLUG}/topic_1/posts")
    assert thread.status_code == 200
    posts = thread.json()
    assert len(posts) == 2
    # World-readable within the course (no header needed to read).
    assert {p["learner_id"] for p in posts} == {"learner-A", "learner-B"}
    reply = next(p for p in posts if p["learner_id"] == "learner-B")
    assert reply["parent_post_id"] == root_id


def test_discussion_post_strips_stored_xss(client):
    p = client.post(
        f"/api/learn/discussion/{_SLUG}/topic_x/posts",
        json={
            "body": 'Hi <script>alert(1)</script> '
            '<form action="https://evil.example/c"><input name="x"></form>'
        },
        headers={"X-Learner-Id": "learner-A"},
    )
    assert p.status_code == 200
    body = p.json()["body_sanitized"]
    assert "<script" not in body.lower()
    assert "alert(1)" not in body
    assert "https://evil.example" not in body

    # And it stays scrubbed on read.
    thread = client.get(f"/api/learn/discussion/{_SLUG}/topic_x/posts").json()
    assert "<script" not in thread[0]["body_sanitized"].lower()


def test_discussion_post_requires_learner_id(client):
    p = client.post(
        f"/api/learn/discussion/{_SLUG}/topic_1/posts", json={"body": "hi"}
    )
    assert p.status_code == 422
    assert p.json()["error"] == "missing_learner_id"


def test_discussion_empty_body_rejected(client):
    p = client.post(
        f"/api/learn/discussion/{_SLUG}/topic_1/posts",
        json={"body": "   "},
        headers={"X-Learner-Id": "learner-A"},
    )
    assert p.status_code == 422


# ------------------------------------------------------------- submission (text)


def test_assignment_text_submit_and_get(client):
    hdr = {"X-Learner-Id": "learner-A"}
    s = client.post(
        f"/api/learn/assignment/{_SLUG}/asn_01/submit",
        json={"text": "My answer is 42."},
        headers=hdr,
    )
    assert s.status_code == 200
    assert s.json()["text_sanitized"] == "My answer is 42."

    g = client.get(f"/api/learn/assignment/{_SLUG}/asn_01/submission", headers=hdr)
    assert g.status_code == 200
    assert g.json()["text_sanitized"] == "My answer is 42."


def test_submission_ownership_isolation(client):
    client.post(
        f"/api/learn/assignment/{_SLUG}/asn_01/submit",
        json={"text": "A's secret work"},
        headers={"X-Learner-Id": "learner-A"},
    )
    # Learner B has no submission → 404 (cannot read A's).
    g = client.get(
        f"/api/learn/assignment/{_SLUG}/asn_01/submission",
        headers={"X-Learner-Id": "learner-B"},
    )
    assert g.status_code == 404


def test_submission_requires_learner_id(client):
    s = client.post(
        f"/api/learn/assignment/{_SLUG}/asn_01/submit", json={"text": "x"}
    )
    assert s.status_code == 422
    assert s.json()["error"] == "missing_learner_id"


def test_submission_strips_stored_xss(client):
    s = client.post(
        f"/api/learn/assignment/{_SLUG}/asn_01/submit",
        json={"text": "Answer <script>steal()</script> here"},
        headers={"X-Learner-Id": "learner-A"},
    )
    assert s.status_code == 200
    assert "<script" not in s.json()["text_sanitized"].lower()


# ------------------------------------------------------------- submission (file)


def test_submission_with_valid_file_upload(client):
    hdr = {"X-Learner-Id": "learner-A"}
    s = client.post(
        f"/api/learn/assignment/{_SLUG}/asn_01/submit",
        data={"text": "see attached"},
        files={"file": ("homework.pdf", b"%PDF-1.4 bytes", "application/pdf")},
        headers=hdr,
    )
    assert s.status_code == 200
    body = s.json()
    assert body["file"]["stored_name"] == "upload.pdf"
    assert body["file"]["original_filename"] == "homework.pdf"


def test_submission_rejects_disallowed_extension(client):
    s = client.post(
        f"/api/learn/assignment/{_SLUG}/asn_01/submit",
        data={"text": "x"},
        files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
        headers={"X-Learner-Id": "learner-A"},
    )
    assert s.status_code == 415
    assert s.json()["error"] == "unsupported_file_type"


def test_submission_rejects_oversized_file(client):
    from gui.services import learner_state

    big = b"x" * (learner_state.MAX_UPLOAD_BYTES + 100)
    s = client.post(
        f"/api/learn/assignment/{_SLUG}/asn_01/submit",
        data={"text": "x"},
        files={"file": ("big.pdf", big, "application/pdf")},
        headers={"X-Learner-Id": "learner-A"},
    )
    assert s.status_code == 413
    assert s.json()["error"] == "file_too_large"


def test_submission_empty_rejected(client):
    s = client.post(
        f"/api/learn/assignment/{_SLUG}/asn_01/submit",
        json={"text": "   "},
        headers={"X-Learner-Id": "learner-A"},
    )
    assert s.status_code == 422
    assert s.json()["error"] == "empty_submission"
