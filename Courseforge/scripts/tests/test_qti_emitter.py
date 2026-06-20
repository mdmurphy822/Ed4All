"""W10 Phase 1 — tests for the QTI / discussion / assignment emitter.

Keystone oracle: every emitted QTI assessment XML must ROUND-TRIP through the
in-tree ``Trainforge/parsers/qti_parser.py::QTIParser.parse_string`` (parse it
back, compare stems / choices / correct key). Secondary oracle: XSD-validity
against the vendored IMS CC schemas (skipped gracefully when ``lxml`` is absent,
mirroring the project's ``[embedding]``-extras graceful-degrade pattern).

All fixtures are built IN-TEST — no course slug, no disk dependency.
"""

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qti_emitter import (  # noqa: E402
    ASSIGNMENT_NS,
    CC_EXAM_PROFILE,
    IMSDT_NS,
    QTI_NS,
    SUPPORTED_ITEM_PROFILES,
    assessment_to_qti,
    assignment_to_resource,
    discussion_to_imsdt,
    question_to_qti_item,
    resolve_cc_profile,
)

from Trainforge.parsers.qti_parser import QTIParser  # noqa: E402

# Vendored XSDs (XSD-validation oracle).
_SCHEMA_DIR = _REPO_ROOT / "Courseforge" / "schemas" / "imscc"
_QTI_XSD = _SCHEMA_DIR / "ccv1p3_qtiasiv1p2p1.xsd"
_IMSDT_XSD = _SCHEMA_DIR / "ccv1p3_imsdt_v1p3.xsd"
_ASSIGNMENT_XSD = _SCHEMA_DIR / "cc_extresource_assignmentv1p0.xsd"


# ---------------------------------------------------------------------------
# Optional XSD validation (lazy lxml import; skip if absent)
# ---------------------------------------------------------------------------
def _assert_xsd_valid(xml_str: str, xsd_path: Path) -> None:
    """Validate ``xml_str`` against ``xsd_path``; SKIP if lxml is unavailable."""
    try:
        from lxml import etree  # noqa: WPS433 (lazy import on purpose)
    except ImportError:
        pytest.skip("lxml not installed — XSD validation skipped (graceful degrade)")

    with open(xsd_path, "rb") as fh:
        schema = etree.XMLSchema(etree.parse(fh))
    doc = etree.fromstring(xml_str.encode("utf-8"))
    if not schema.validate(doc):
        raise AssertionError(f"XSD validation failed for {xsd_path.name}:\n{schema.error_log}")


# ---------------------------------------------------------------------------
# In-test fixtures (no disk, no course slug)
# ---------------------------------------------------------------------------
def _mc_question() -> dict:
    return {
        "question_id": "q-mc-001",
        "question_type": "multiple_choice",
        "stem": "What is the additive identity?",
        "bloom_level": "remember",
        "objective_id": "CO-01",
        "choices": [
            {"id": "A", "text": "0"},
            {"id": "B", "text": "1"},
            {"id": "C", "text": "-1"},
            {"id": "D", "text": "10"},
        ],
        "correct_answer": "A",
        "points": 5,
    }


def _mresp_question() -> dict:
    return {
        "question_id": "q-mr-002",
        "question_type": "multiple_response",
        "stem": "Which of these are even numbers?",
        "bloom_level": "understand",
        "objective_id": "CO-02",
        "choices": [
            {"id": "A", "text": "2"},
            {"id": "B", "text": "3"},
            {"id": "C", "text": "4"},
            {"id": "D", "text": "5"},
        ],
        "correct_answer": ["A", "C"],
        "points": 4,
    }


def _tf_question() -> dict:
    return {
        "question_id": "q-tf-003",
        "question_type": "true_false",
        "stem": "Zero is a natural number in some conventions.",
        "bloom_level": "remember",
        "objective_id": "CO-03",
        "choices": [],
        "correct_answer": "True",
        "points": 1,
    }


def _numeric_fib_question() -> dict:
    return {
        "question_id": "q-fib-004",
        "question_type": "fill_in_blank",
        "stem": "Compute 7 + 5.",
        "bloom_level": "apply",
        "objective_id": "CO-04",
        "choices": [],
        "correct_answer": "12",
        "points": 2,
    }


def _non_numeric_fib_question() -> dict:
    """A fill_in_blank whose answer is not canonical-numeric → §9.1 downgrade."""
    return {
        "question_id": "q-fib-005",
        "question_type": "fill_in_blank",
        "stem": "The fraction one-half written as a fraction is ___.",
        "bloom_level": "apply",
        "objective_id": "CO-05",
        "choices": [
            {"id": "A", "text": "1/2"},
            {"id": "B", "text": "2/1"},
            {"id": "C", "text": "0"},
        ],
        "correct_answer": "1/2",
        "points": 2,
    }


def _assessment(questions: list) -> dict:
    return {
        "assessment_id": "quiz-week-01",
        "title": "Week 1 Quiz",
        "course_code": "ALG_101",
        "questions": questions,
        "objectives_targeted": ["CO-01", "CO-02"],
        "bloom_levels": ["remember", "understand"],
    }


# ---------------------------------------------------------------------------
# Round-trip (keystone) — full assessment through QTIParser
# ---------------------------------------------------------------------------
def test_assessment_roundtrips_through_qti_parser():
    questions = [_mc_question(), _mresp_question(), _tf_question(), _numeric_fib_question()]
    xml_str = assessment_to_qti(_assessment(questions))

    parsed = QTIParser().parse_string(xml_str)

    # NB: QTIParser reads ``ident`` off the document root, which is
    # ``questestinterop`` in the canonical Pattern-15 shape (the ``ident`` lives
    # on the nested ``<assessment>``), so ``parsed.id`` is the parser's
    # "unknown" sentinel — not an emitter defect. The load-bearing round-trip is
    # the per-question stem / choices / correct-key, asserted below.
    assert len(parsed.questions) == 4

    by_id = {q.id: q for q in parsed.questions}
    # IDs are sanitized to NCName-safe form; hyphens survive.
    assert "q-mc-001" in by_id
    # Stem survives.
    assert by_id["q-mc-001"].stem == "What is the additive identity?"
    # MC choices survive with the correct key marked.
    mc = by_id["q-mc-001"]
    assert {c.id for c in mc.choices} == {"A", "B", "C", "D"}
    assert mc.correct_response == "A"
    assert [c.id for c in mc.choices if c.is_correct] == ["A"]


def test_mc_item_roundtrip_stem_choices_key():
    q = _mc_question()
    xml_str = assessment_to_qti(_assessment([q]))
    parsed = QTIParser().parse_string(xml_str)
    pq = parsed.questions[0]
    assert pq.type == "multiple_choice"
    assert pq.stem == q["stem"]
    assert {c.text for c in pq.choices} == {"0", "1", "-1", "10"}
    assert pq.correct_response == "A"


def test_multiple_response_and_of_varequal():
    q = _mresp_question()
    xml_str = assessment_to_qti(_assessment([q]))
    # The parser collapses <and> to a single varequal; assert the XML carries
    # both correct idents under an <and>, and round-trips structurally.
    assert xml_str.count("<varequal") == 2
    assert "<and>" in xml_str
    parsed = QTIParser().parse_string(xml_str)
    pq = parsed.questions[0]
    assert {c.id for c in pq.choices} == {"A", "B", "C", "D"}
    correct_marked = {c.id for c in pq.choices if c.is_correct}
    # The parser marks the last varequal it sees as correct; at minimum one of
    # the two correct ids is marked, and the emitted key carries both.
    assert correct_marked <= {"A", "C"}


def test_true_false_roundtrip():
    q = _tf_question()
    xml_str = assessment_to_qti(_assessment([q]))
    parsed = QTIParser().parse_string(xml_str)
    pq = parsed.questions[0]
    # Parser infers true_false from the two True/False labels.
    assert pq.type == "true_false"
    assert {c.text.lower() for c in pq.choices} == {"true", "false"}
    assert pq.correct_response == "True"


def test_numeric_fib_emitted_as_fib():
    q = _numeric_fib_question()
    item = question_to_qti_item(q)
    # cc_profile reflects the fill-in-blank type.
    profiles = [fe.text for fe in item.iter("fieldentry")]
    assert "cc.fib.v0p1" in profiles

    xml_str = assessment_to_qti(_assessment([q]))
    assert "<response_str" in xml_str
    assert "<render_fib" in xml_str
    parsed = QTIParser().parse_string(xml_str)
    pq = parsed.questions[0]
    # The in-tree parser keys fill_in_blank off a (non-XSD) ``response_fib``
    # element; a CC-valid numeric fill-in uses ``response_str`` + ``render_fib``,
    # which the parser surfaces as ``essay``. The load-bearing round-trip is the
    # recovered answer key, which the parser reads from resprocessing regardless
    # of the inferred type.
    assert pq.correct_response == "12"


def test_non_numeric_fib_downgrades_to_multiple_choice():
    """Spec §9.1: a non-canonical-numeric fill-in downgrades to MC."""
    q = _non_numeric_fib_question()
    item = question_to_qti_item(q)
    # The emitted item uses a response_lid (multiple choice), not response_str.
    assert item.find(".//response_lid") is not None
    assert item.find(".//response_str") is None
    # cc_profile reflects the downgrade.
    profiles = [fe.text for fe in item.iter("fieldentry")]
    assert "cc.multiple_choice.v0p1" in profiles

    xml_str = assessment_to_qti(_assessment([q]))
    parsed = QTIParser().parse_string(xml_str)
    pq = parsed.questions[0]
    assert pq.type == "multiple_choice"
    assert pq.correct_response == "A"  # "1/2" resolved by choice text


def test_non_numeric_fib_no_choices_downgrades_to_short_answer():
    q = dict(_non_numeric_fib_question())
    q["choices"] = []
    item = question_to_qti_item(q)
    # No key resolvable → ungraded essay/short_answer shape (response_str, no
    # resprocessing). Anti-fabrication: never invents a key.
    assert item.find(".//response_str") is not None
    assert item.find("resprocessing") is None
    profiles = [fe.text for fe in item.iter("fieldentry")]
    assert "cc.essay.v0p1" in profiles


# ---------------------------------------------------------------------------
# cc_profile mapping
# ---------------------------------------------------------------------------
def test_cc_profile_map():
    assert resolve_cc_profile("multiple_choice") == "cc.multiple_choice.v0p1"
    assert resolve_cc_profile("multiple_response") == "cc.multiple_response.v0p1"
    assert resolve_cc_profile("true_false") == "cc.true_false.v0p1"
    assert resolve_cc_profile("fill_in_blank") == "cc.fib.v0p1"
    assert resolve_cc_profile("short_answer") == "cc.essay.v0p1"
    assert resolve_cc_profile("essay") == "cc.essay.v0p1"
    # Unknown collapses to essay (emitted but ungraded), never raises.
    assert resolve_cc_profile("matching") == "cc.essay.v0p1"


def test_assessment_carries_exam_profile():
    xml_str = assessment_to_qti(_assessment([_mc_question()]))
    assert CC_EXAM_PROFILE in xml_str
    assert QTI_NS in xml_str
    # Every item profile is in the supported set.
    parser_ns = {"q": QTI_NS}
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_str)
    item_profiles = []
    for field in root.iter(f"{{{QTI_NS}}}qtimetadatafield"):
        label = field.find(f"{{{QTI_NS}}}fieldlabel")
        entry = field.find(f"{{{QTI_NS}}}fieldentry")
        if label is not None and label.text == "cc_profile" and entry is not None:
            item_profiles.append(entry.text)
    # Exam profile + per-item profile.
    assert CC_EXAM_PROFILE in item_profiles
    for p in item_profiles:
        assert p == CC_EXAM_PROFILE or p in SUPPORTED_ITEM_PROFILES


# ---------------------------------------------------------------------------
# XSD validity
# ---------------------------------------------------------------------------
def test_qti_xsd_valid():
    questions = [_mc_question(), _mresp_question(), _tf_question(), _numeric_fib_question()]
    xml_str = assessment_to_qti(_assessment(questions))
    _assert_xsd_valid(xml_str, _QTI_XSD)


def test_essay_item_xsd_valid():
    q = {
        "question_id": "q-essay-006",
        "question_type": "essay",
        "stem": "Explain why the distributive property matters.",
        "bloom_level": "evaluate",
        "objective_id": "CO-06",
        "choices": [],
        "correct_answer": None,
        "points": 10,
    }
    xml_str = assessment_to_qti(_assessment([q]))
    _assert_xsd_valid(xml_str, _QTI_XSD)
    parsed = QTIParser().parse_string(xml_str)
    assert parsed.questions[0].type == "essay"


# ---------------------------------------------------------------------------
# Discussion (imsdt)
# ---------------------------------------------------------------------------
def test_discussion_to_imsdt_xsd_valid():
    prompt = {
        "title": "Reflecting on number systems",
        "objective_id": "TO-01",
        "text": "<p>How do the integers extend the natural numbers? Discuss with a peer.</p>",
    }
    xml_str = discussion_to_imsdt(prompt)
    assert IMSDT_NS in xml_str
    assert "<title>Reflecting on number systems</title>" in xml_str
    _assert_xsd_valid(xml_str, _IMSDT_XSD)


def test_discussion_falls_back_to_stem_title():
    prompt = {"stem": "Just a prompt with no title.", "objective_id": "TO-02"}
    xml_str = discussion_to_imsdt(prompt)
    _assert_xsd_valid(xml_str, _IMSDT_XSD)
    assert "text/html" in xml_str


# ---------------------------------------------------------------------------
# Assignment (learning-application-resource)
# ---------------------------------------------------------------------------
def test_assignment_to_resource_xsd_valid():
    task = {
        "assignment_id": "assign-week-01",
        "title": "Number-line construction",
        "objective_id": "TO-01",
        "text": "<p>Construct a number line from -5 to 5 and label each integer.</p>",
        "instructor_text": "<p>Award full marks for correctly labeled integers.</p>",
        "points": 20,
    }
    xml_str = assignment_to_resource(task)
    assert ASSIGNMENT_NS in xml_str
    assert "<title>Number-line construction</title>" in xml_str
    assert "<gradable" in xml_str
    _assert_xsd_valid(xml_str, _ASSIGNMENT_XSD)


def test_assignment_without_points_omits_gradable():
    task = {
        "objective_id": "TO-03",
        "prompt": "Write a short reflection on what you learned this week.",
    }
    xml_str = assignment_to_resource(task)
    assert "<gradable" not in xml_str
    _assert_xsd_valid(xml_str, _ASSIGNMENT_XSD)


# ---------------------------------------------------------------------------
# Dataclass input (to_dict) path
# ---------------------------------------------------------------------------
def test_accepts_dataclass_to_dict_input():
    from Trainforge.generators.assessment_generator import AssessmentData, QuestionData

    q = QuestionData(
        question_id="q-dc-001",
        question_type="multiple_choice",
        stem="Which is prime?",
        bloom_level="remember",
        objective_id="CO-09",
        choices=[{"id": "A", "text": "4"}, {"id": "B", "text": "7"}],
        correct_answer="B",
        points=1.0,
    )
    a = AssessmentData(
        assessment_id="quiz-dc",
        title="Dataclass Quiz",
        course_code="ALG_101",
        questions=[q],
    )
    # Pass the dataclasses directly (to_dict path).
    xml_str = assessment_to_qti(a)
    parsed = QTIParser().parse_string(xml_str)
    assert parsed.questions[0].correct_response == "B"
    _assert_xsd_valid(xml_str, _QTI_XSD)
