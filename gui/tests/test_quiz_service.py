"""Unit tests for ``gui.services.quiz_service`` (W10 Phase 3 service layer).

Covers:
  * the KEY anti-leak assertion — the learner-facing JSON omits ``correct_answer``
    (and every other answer-revealing field) for every question;
  * grading: all-correct → full score, a wrong selection → marked incorrect;
  * numeric Fraction-equivalence (``1/2`` submitted against key ``0.5`` → correct);
  * multiple_response all-or-nothing (no partial credit, spec §9.5);
  * locate fail-soft (empty for a course with no quizzes) + a dynamic
    archive-discovery test that SKIPS when no real course exists.

Fixtures are hand-authored QTI 1.2 (Pattern-15 shape) IN-TEST and fed directly
via path/bytes/str injection — no real archived course, no hardcoded slug, no
dependency on the concurrent qti_emitter module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui.services import quiz_service as qs


# --------------------------------------------------------------------------- #
# In-test QTI 1.2 fixture (Pattern-15 shape): one MC item + one numeric FIB item
# + one multiple_response item. No course slug, no emitter dependency.
# --------------------------------------------------------------------------- #

_QTI_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <assessment ident="quiz_week_01" title="Week 1 Quiz">
    <qtimetadata>
      <qtimetadatafield>
        <fieldlabel>cc_profile</fieldlabel>
        <fieldentry>cc.exam.v0p1</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    <section ident="root_section">
      <item ident="question_1">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.multiple_choice.v0p1</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">What is 2 + 2?</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
              <response_label ident="A">
                <material><mattext texttype="text/html">3</mattext></material>
              </response_label>
              <response_label ident="B">
                <material><mattext texttype="text/html">4</mattext></material>
              </response_label>
              <response_label ident="C">
                <material><mattext texttype="text/html">5</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar varname="SCORE" vartype="Integer" minvalue="0" maxvalue="1"/>
          </outcomes>
          <respcondition>
            <conditionvar>
              <varequal respident="response1">B</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">1</setvar>
          </respcondition>
        </resprocessing>
      </item>
      <item ident="question_2">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.fib.v0p1</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">What fraction is one half?</mattext></material>
          <response_str ident="response2" rcardinality="Single">
            <render_fib><response_label ident="fib2"/></render_fib>
          </response_str>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar varname="SCORE" vartype="Integer" minvalue="0" maxvalue="1"/>
          </outcomes>
          <respcondition>
            <conditionvar>
              <varequal respident="response2">0.5</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">1</setvar>
          </respcondition>
        </resprocessing>
      </item>
      <item ident="question_3">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.multiple_response.v0p1</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">Which are even numbers?</mattext></material>
          <response_lid ident="response3" rcardinality="Multiple">
            <render_choice>
              <response_label ident="P">
                <material><mattext texttype="text/html">2</mattext></material>
              </response_label>
              <response_label ident="Q">
                <material><mattext texttype="text/html">3</mattext></material>
              </response_label>
              <response_label ident="R">
                <material><mattext texttype="text/html">4</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar varname="SCORE" vartype="Integer" minvalue="0" maxvalue="1"/>
          </outcomes>
          <respcondition>
            <conditionvar>
              <varequal respident="response3">P</varequal>
              <varequal respident="response3">R</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">1</setvar>
          </respcondition>
        </resprocessing>
      </item>
    </section>
  </assessment>
</questestinterop>
"""


@pytest.fixture
def assessment():
    return qs.parse_quiz(_QTI_FIXTURE)


# --------------------------------------------------------------------------- #
# Parse + learner JSON
# --------------------------------------------------------------------------- #


def test_parse_quiz_reads_all_items(assessment):
    assert assessment.id == "quiz_week_01"
    assert assessment.title == "Week 1 Quiz"
    ids = {q.id for q in assessment.questions}
    assert ids == {"question_1", "question_2", "question_3"}


def test_parse_quiz_accepts_bytes_and_path(tmp_path: Path):
    by_bytes = qs.parse_quiz(_QTI_FIXTURE.encode("utf-8"))
    assert by_bytes.id == "quiz_week_01"

    xml_path = tmp_path / "quiz.xml"
    xml_path.write_text(_QTI_FIXTURE, encoding="utf-8")
    by_path = qs.parse_quiz(xml_path)
    assert by_path.id == "quiz_week_01"
    # Bare path string injection also works.
    by_path_str = qs.parse_quiz(str(xml_path))
    assert by_path_str.id == "quiz_week_01"


def test_parse_quiz_malformed_raises():
    with pytest.raises(qs.QuizServiceError) as exc:
        qs.parse_quiz("<questestinterop><unclosed></questestinterop>")
    assert exc.value.status == 422


def test_learner_json_shape(assessment):
    payload = qs.to_learner_json(assessment)
    assert payload["assessment_id"] == "quiz_week_01"
    assert payload["title"] == "Week 1 Quiz"
    assert len(payload["questions"]) == 3
    q1 = next(q for q in payload["questions"] if q["question_id"] == "question_1")
    assert q1["stem"] == "What is 2 + 2?"
    assert q1["type"] == "multiple_choice"
    assert {c["id"] for c in q1["choices"]} == {"A", "B", "C"}
    assert {c["text"] for c in q1["choices"]} == {"3", "4", "5"}


def test_learner_json_omits_answer_key(assessment):
    """KEY ANTI-LEAK assertion: no answer-revealing field anywhere in learner JSON."""
    payload = qs.to_learner_json(assessment)
    blob = json.dumps(payload).lower()
    # No top-level / per-question answer-key field of any kind.
    for forbidden in ("correct_answer", "correct_response", "is_correct", "answer_key"):
        assert forbidden not in blob, f"learner JSON leaks {forbidden!r}"
    for q in payload["questions"]:
        assert "correct_answer" not in q
        assert "correct_response" not in q
        assert "feedback" not in q
        for choice in q["choices"]:
            # A choice carries ONLY id + text — never the is_correct flag.
            assert set(choice.keys()) == {"id", "text"}


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def test_grade_all_correct_full_score(assessment):
    submission = {
        "question_1": "B",
        "question_2": "0.5",
        "question_3": ["P", "R"],
    }
    result = qs.grade_submission(assessment, submission, quiz_xml=_QTI_FIXTURE)
    assert result["max_score"] == 3
    assert result["score"] == 3
    assert all(item["correct"] for item in result["per_item"])


def test_grade_wrong_selection_marked_incorrect(assessment):
    submission = {
        "question_1": "A",  # wrong (correct is B)
        "question_2": "0.5",
        "question_3": ["P", "R"],
    }
    result = qs.grade_submission(assessment, submission, quiz_xml=_QTI_FIXTURE)
    assert result["score"] == 2
    by_id = {item["question_id"]: item for item in result["per_item"]}
    assert by_id["question_1"]["correct"] is False
    assert "incorrect" in by_id["question_1"]["feedback"].lower()
    assert by_id["question_2"]["correct"] is True


def test_grade_numeric_fraction_equivalence(assessment):
    """`1/2` submitted against key `0.5` → CORRECT (Fraction equivalence, §9.1)."""
    for submitted in ("1/2", ".50", "0.50", "0.5"):
        result = qs.grade_submission(
            assessment,
            {"question_1": "B", "question_2": submitted, "question_3": ["P", "R"]},
            quiz_xml=_QTI_FIXTURE,
        )
        by_id = {i["question_id"]: i for i in result["per_item"]}
        assert by_id["question_2"]["correct"] is True, f"{submitted!r} should equal 0.5"


def test_grade_numeric_wrong_value_incorrect(assessment):
    result = qs.grade_submission(
        assessment,
        {"question_1": "B", "question_2": "0.6", "question_3": ["P", "R"]},
        quiz_xml=_QTI_FIXTURE,
    )
    by_id = {i["question_id"]: i for i in result["per_item"]}
    assert by_id["question_2"]["correct"] is False


def test_multiple_response_all_or_nothing(assessment):
    """Multi-select is all-or-nothing — partial credit is out of scope (§9.5)."""
    base = {"question_1": "B", "question_2": "0.5"}

    # Exactly the correct set → correct.
    full = qs.grade_submission(
        assessment, {**base, "question_3": ["P", "R"]}, quiz_xml=_QTI_FIXTURE
    )
    assert {i["question_id"]: i for i in full["per_item"]}["question_3"]["correct"] is True

    # Missing one correct id → incorrect (no partial credit).
    partial = qs.grade_submission(
        assessment, {**base, "question_3": ["P"]}, quiz_xml=_QTI_FIXTURE
    )
    assert {i["question_id"]: i for i in partial["per_item"]}["question_3"]["correct"] is False

    # Correct set PLUS a wrong id → incorrect.
    superset = qs.grade_submission(
        assessment, {**base, "question_3": ["P", "Q", "R"]}, quiz_xml=_QTI_FIXTURE
    )
    assert {i["question_id"]: i for i in superset["per_item"]}["question_3"]["correct"] is False


def test_grade_falls_back_to_parser_key_without_xml(assessment):
    """Single-correct grading still works when quiz_xml isn't threaded through."""
    result = qs.grade_submission(
        assessment, {"question_1": "B", "question_2": "0.5"}
    )
    by_id = {i["question_id"]: i for i in result["per_item"]}
    assert by_id["question_1"]["correct"] is True  # parser is_correct flag
    assert by_id["question_2"]["correct"] is True  # parser correct_response


def test_grade_missing_selection_is_incorrect(assessment):
    result = qs.grade_submission(assessment, {}, quiz_xml=_QTI_FIXTURE)
    assert result["score"] == 0
    assert all(item["correct"] is False for item in result["per_item"])


# --------------------------------------------------------------------------- #
# Locate (fail-soft + dynamic discovery)
# --------------------------------------------------------------------------- #


def test_locate_fail_soft_no_quizzes(libv2_root: Path):
    """A course dir with no archived quizzes returns [] — never raises."""
    slug = "no-quiz-course"
    (libv2_root / "courses" / slug).mkdir(parents=True, exist_ok=True)
    assert qs.locate_quiz_xml(slug, libv2_root=libv2_root) == []


def test_locate_unknown_course_fail_soft(libv2_root: Path):
    """An unknown course is 'no quizzes', not an error, on the locate surface."""
    assert qs.locate_quiz_xml("never-imported", libv2_root=libv2_root) == []


def test_locate_invalid_slug_raises(libv2_root: Path):
    with pytest.raises(qs.QuizServiceError) as exc:
        qs.locate_quiz_xml("../etc/passwd", libv2_root=libv2_root)
    assert exc.value.status == 422


def test_locate_finds_loose_assessment_xml(libv2_root: Path):
    """A loose 06_assessments/*.xml is discovered + round-trips the parser."""
    slug = "alg-quiz-course"
    adir = libv2_root / "courses" / slug / "06_assessments"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "quiz_week_01.xml").write_text(_QTI_FIXTURE, encoding="utf-8")

    blobs = qs.locate_quiz_xml(slug, libv2_root=libv2_root)
    assert len(blobs) == 1
    parsed = qs.parse_quiz(blobs[0])
    assert parsed.id == "quiz_week_01"
    # Learner JSON off the located blob still omits the key.
    assert "correct" not in json.dumps(qs.to_learner_json(parsed)).lower()


def test_locate_from_real_archive_discovers_or_skips():
    """Dynamic archive-discovery: locate a real course's quiz, or SKIP if none.

    No hardcoded slug — discovers a course under the real LibV2 root and only
    asserts the locate surface doesn't crash. Skips cleanly when no archived
    course carries a quiz (a clean checkout / CI).
    """
    from lib.paths import LIBV2_PATH  # noqa: PLC0415

    courses = Path(LIBV2_PATH) / "courses"
    if not courses.is_dir():
        pytest.skip("no real LibV2 courses dir on this checkout")

    found_any = False
    for course_dir in sorted(courses.iterdir()):
        if not course_dir.is_dir():
            continue
        slug = course_dir.name
        if not qs._SLUG_RE.match(slug):
            continue
        blobs = qs.locate_quiz_xml(slug)  # default real root
        if blobs:
            found_any = True
            parsed = qs.parse_quiz(blobs[0])
            assert parsed.id  # parses without raising
            assert "correct" not in json.dumps(qs.to_learner_json(parsed)).lower()
            break

    if not found_any:
        pytest.skip("no archived course carries a quiz on this checkout")
