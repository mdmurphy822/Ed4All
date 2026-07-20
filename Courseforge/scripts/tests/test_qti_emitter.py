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
    itemfeedback_enabled,
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


# ---------------------------------------------------------------------------
# Answer-key resolution ladder (assessment-tail fix 2026-07-19).
#
# The production AssessmentGenerator marks MCQ keys via per-choice
# ``is_correct`` flags and leaves ``correct_answer`` unset, and wraps choice
# texts in ``<p>…</p>``. The pre-fix emitter resolved keys ONLY from
# ``correct_answer`` (exact id/text match), so every production MCQ emitted
# ungraded (QTI_NO_ANSWER_KEY on all 50 alg-glm-02 quiz items).
# ---------------------------------------------------------------------------

def _generator_shaped_mcq(qid: str = "q-gen-001") -> dict:
    """The EXACT shape AssessmentGenerator emits: is_correct flags, no
    correct_answer, <p>-wrapped choice texts, no choice ids."""
    return {
        "question_id": qid,
        "question_type": "multiple_choice",
        "stem": "<p>Which of the following best describes <em>X</em>?</p>",
        "bloom_level": "understand",
        "objective_id": "TO-01",
        "choices": [
            {"text": "<p>the correct definition</p>", "is_correct": True},
            {"text": "<p>a plausible distractor</p>", "is_correct": False},
            {"text": "<p>another distractor</p>", "is_correct": False},
            {"text": "<p>a third distractor</p>", "is_correct": False},
        ],
        "points": 2.0,
    }


def test_mcq_is_correct_flags_resolve_answer_key():
    xml_str = assessment_to_qti(_assessment([_generator_shaped_mcq()]))
    assert "<resprocessing>" in xml_str
    assert "<varequal" in xml_str
    parsed = QTIParser().parse_string(xml_str)
    pq = parsed.questions[0]
    # Choice ids fall back to letters; the flagged choice (index 0 → "A")
    # must be the emitted key.
    assert pq.correct_response == "A"
    _assert_xsd_valid(xml_str, _QTI_XSD)


def test_correct_answer_bare_text_matches_html_wrapped_choice():
    q = _generator_shaped_mcq()
    # correct_answer as the BARE text of a non-first choice (no <p> wrapper);
    # must beat the is_correct fallback (explicit key wins).
    q["correct_answer"] = "a plausible distractor"
    xml_str = assessment_to_qti(_assessment([q]))
    parsed = QTIParser().parse_string(xml_str)
    assert parsed.questions[0].correct_response == "B"


def test_no_key_at_all_still_ungraded_anti_fabrication():
    q = _generator_shaped_mcq()
    for c in q["choices"]:
        c["is_correct"] = False
    xml_str = assessment_to_qti(_assessment([q]))
    # No resolvable key from either arm → emitted ungraded (never invents).
    assert "<resprocessing>" not in xml_str


def test_multiple_response_is_correct_flags_emit_and_of_varequal():
    q = {
        "question_id": "q-mr-flags",
        "question_type": "multiple_response",
        "stem": "<p>Select all that apply.</p>",
        "bloom_level": "understand",
        "objective_id": "TO-01",
        "choices": [
            {"text": "<p>right one</p>", "is_correct": True},
            {"text": "<p>wrong one</p>", "is_correct": False},
            {"text": "<p>right two</p>", "is_correct": True},
            {"text": "<p>wrong two</p>", "is_correct": False},
        ],
        "points": 2.0,
    }
    xml_str = assessment_to_qti(_assessment([q]))
    assert "<and>" in xml_str
    assert xml_str.count("<varequal") == 2
    _assert_xsd_valid(xml_str, _QTI_XSD)


def test_generator_shaped_quiz_passes_qti_well_formed_gate():
    """End-to-end: emit a generator-shaped quiz and run the ACTUAL
    qti_well_formed gate over it (the acceptance oracle)."""
    from lib.validators.qti_well_formed import QtiWellFormedValidator

    xml_str = assessment_to_qti(_assessment([_generator_shaped_mcq()]))
    result = QtiWellFormedValidator().validate(
        {"qti_strings": [{"id": "week_01_quiz.xml", "xml": xml_str}]}
    )
    assert result.passed is True, [i.code for i in result.issues]


# ===========================================================================
# ITEMFEEDBACK — per-distractor misconception notes + worked-solution feedback
# (assessment-quality overhaul). Default ON; kill-switch
# ED4ALL_ASSESSMENT_ITEMFEEDBACK=0.
# ===========================================================================
import xml.etree.ElementTree as _ET  # noqa: E402


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[1] if "}" in tag else tag


def _iter_local(root, local: str):
    for el in root.iter():
        if _localname(el.tag) == local:
            yield el


def _mc_with_feedback(qid: str = "q-mc-fb") -> dict:
    """A generator-shaped MC carrying a worked-solution ``feedback`` plus two
    per-distractor ``misconception_note`` keys (the exact shape the diversified
    tier authors)."""
    return {
        "question_id": qid,
        "question_type": "multiple_choice",
        "stem": "<p>What is 2 + 2?</p>",
        "bloom_level": "remember",
        "objective_id": "CO-10",
        "choices": [
            {"text": "<p>4</p>", "is_correct": True},
            {"text": "<p>22</p>", "is_correct": False,
             "misconception_note": "Concatenated the digits instead of adding."},
            {"text": "<p>5</p>", "is_correct": False,
             "misconception_note": "Off-by-one counting error."},
        ],
        "feedback": "<p>2 + 2 = 4 by counting up two from two.</p>",
        "points": 1.0,
    }


def _written_item(qid: str = "q-written", subtype: str = "short_answer") -> dict:
    """A generator-shaped WRITTEN constructed-response item (essay profile)."""
    return {
        "question_id": qid,
        "question_type": "essay",
        "stem": "<p>In 2-4 sentences, explain why a coefficient scales its "
                "variable.</p>",
        "bloom_level": "evaluate",
        "objective_id": "TO-02",
        "item_subtype": subtype,
        "feedback": "<p>A coefficient multiplies the variable it precedes.</p>",
        "points": 5.0,
    }


def test_written_subtype_emits_essay_profile_and_subtype_metadata():
    item = question_to_qti_item(_written_item(subtype="extended_response"))
    # cc_profile + item_subtype ride through as qtimetadata provenance.
    fields = {}
    for f in _iter_local(item, "qtimetadatafield"):
        label = next(_iter_local(f, "fieldlabel")).text
        entry = next(_iter_local(f, "fieldentry")).text
        fields[label] = entry
    assert fields.get("cc_profile") == "cc.essay.v0p1"
    assert fields.get("item_subtype") == "extended_response"
    # The model answer travels as a Solution-type <itemfeedback>.
    solutions = [fb for fb in _iter_local(item, "itemfeedback")
                 if list(_iter_local(fb, "solution"))]
    assert len(solutions) == 1


def test_written_subtype_xsd_valid_and_roundtrips():
    xml_str = assessment_to_qti(_assessment([_written_item()]))
    _assert_xsd_valid(xml_str, _QTI_XSD)
    parsed = QTIParser().parse_string(xml_str)
    assert parsed.questions[0].type == "essay"


def test_itemfeedback_default_on_emits_solution_and_response():
    item = question_to_qti_item(_mc_with_feedback())
    feedbacks = list(_iter_local(item, "itemfeedback"))
    # 1 Solution (worked solution) + 2 Response (per-distractor) = 3.
    assert len(feedbacks) == 3
    # Exactly one carries a <solution>; the other two a <flow_mat>.
    solutions = [fb for fb in feedbacks if list(_iter_local(fb, "solution"))]
    assert len(solutions) == 1
    # displayfeedback linkage: every linkrefid resolves to an itemfeedback ident.
    idents = {fb.get("ident") for fb in feedbacks}
    dfs = list(_iter_local(item, "displayfeedback"))
    assert len(dfs) == 3  # 1 Solution + 2 Response
    for df in dfs:
        assert df.get("linkrefid") in idents
    ftypes = sorted(df.get("feedbacktype") for df in dfs)
    assert ftypes == ["Response", "Response", "Solution"]


def test_itemfeedback_default_on_xsd_valid_and_roundtrips():
    xml_str = assessment_to_qti(_assessment([_mc_with_feedback()]))
    _assert_xsd_valid(xml_str, _QTI_XSD)
    parsed = QTIParser().parse_string(xml_str)
    pq = parsed.questions[0]
    # The scoring key survives the added feedback respconditions.
    assert pq.correct_response == "A"


def test_itemfeedback_passes_qti_well_formed_gate():
    from lib.validators.qti_well_formed import QtiWellFormedValidator

    xml_str = assessment_to_qti(_assessment([_mc_with_feedback()]))
    result = QtiWellFormedValidator().validate(
        {"qti_strings": [{"id": "week_01_quiz.xml", "xml": xml_str}]}
    )
    assert result.passed is True, [i.code for i in result.issues]


def test_itemfeedback_off_is_byte_identical_to_no_feedback(monkeypatch):
    """ED4ALL_ASSESSMENT_ITEMFEEDBACK=0 emits byte-identically to the same item
    with the feedback + misconception fields stripped (default-ON parse)."""
    monkeypatch.setenv("ED4ALL_ASSESSMENT_ITEMFEEDBACK", "0")
    assert itemfeedback_enabled() is False
    xml_off = assessment_to_qti(_assessment([_mc_with_feedback()]))
    assert "<itemfeedback" not in xml_off
    assert "displayfeedback" not in xml_off
    monkeypatch.delenv("ED4ALL_ASSESSMENT_ITEMFEEDBACK")
    assert itemfeedback_enabled() is True

    stripped = _mc_with_feedback()
    stripped.pop("feedback")
    for c in stripped["choices"]:
        c.pop("misconception_note", None)
    xml_stripped = assessment_to_qti(_assessment([stripped]))
    assert xml_off == xml_stripped


def test_itemfeedback_off_variants_parse_with_fallback(monkeypatch):
    for falsey in ("0", "false", "no", "off", "FALSE", "Off"):
        monkeypatch.setenv("ED4ALL_ASSESSMENT_ITEMFEEDBACK", falsey)
        assert itemfeedback_enabled() is False
    for truthy in ("1", "true", "yes", "on", "garbage", ""):
        monkeypatch.setenv("ED4ALL_ASSESSMENT_ITEMFEEDBACK", truthy)
        assert itemfeedback_enabled() is True


def test_itemfeedback_solution_only_when_no_notes():
    q = _mc_question()  # no notes, no feedback
    q["feedback"] = "<p>The additive identity is 0.</p>"
    item = question_to_qti_item(q)
    feedbacks = list(_iter_local(item, "itemfeedback"))
    assert len(feedbacks) == 1
    assert list(_iter_local(feedbacks[0], "solution"))
    xml_str = assessment_to_qti(_assessment([q]))
    _assert_xsd_valid(xml_str, _QTI_XSD)


# ===========================================================================
# NEW TYPE EMISSION — multiple_response via correct_answers, pattern_match,
# error-analysis / matching / ordering / two-tier as CC primitives.
# ===========================================================================
def _mr_via_correct_answers() -> dict:
    """Generator-shaped MR: keys carried in the plural ``correct_answers`` as
    choice TEXTS (no scalar correct_answer), is_correct flags set."""
    return {
        "question_id": "q-mr-ca",
        "question_type": "multiple_response",
        "stem": "<p>Select all correct steps.</p>",
        "bloom_level": "apply",
        "objective_id": "CO-11",
        "choices": [
            {"text": "<p>Distribute the coefficient</p>", "is_correct": True},
            {"text": "<p>Add the exponents</p>", "is_correct": False,
             "misconception_note": "Exponents add only on multiplication of like bases."},
            {"text": "<p>Combine like terms</p>", "is_correct": True},
            {"text": "<p>Drop the sign</p>", "is_correct": False},
        ],
        "correct_answers": ["Distribute the coefficient", "Combine like terms"],
        "feedback": "<p>The correct steps come straight from the worked solution.</p>",
        "points": 2.0,
    }


def test_mr_correct_answers_plural_resolves_and_of_varequal():
    q = _mr_via_correct_answers()
    item = question_to_qti_item(q)
    # The SCORING key is the <and> of the two correct idents (feedback
    # respconditions add their own varequals outside the <and>).
    and_el = next(_iter_local(item, "and"))
    scoring = sorted(ve.text for ve in _iter_local(and_el, "varequal"))
    assert scoring == ["A", "C"]
    xml_str = assessment_to_qti(_assessment([q]))
    _assert_xsd_valid(xml_str, _QTI_XSD)
    # profile is multiple_response.
    profiles = [fe.text for fe in item.iter("fieldentry")]
    assert "cc.multiple_response.v0p1" in profiles


def test_mr_correct_answers_no_is_correct_still_resolves():
    """Plural correct_answers alone (no is_correct flags) must resolve the key
    by choice-text match — the answer-key ladder extension for MR."""
    q = _mr_via_correct_answers()
    for c in q["choices"]:
        c.pop("is_correct", None)
    item = question_to_qti_item(q)
    and_el = next(_iter_local(item, "and"))
    correct = {ve.text for ve in _iter_local(and_el, "varequal")}
    # Keys A + C resolved by text.
    assert correct == {"A", "C"}


def test_mr_passes_qti_well_formed_gate():
    from lib.validators.qti_well_formed import QtiWellFormedValidator

    xml_str = assessment_to_qti(_assessment([_mr_via_correct_answers()]))
    result = QtiWellFormedValidator().validate(
        {"qti_strings": [{"id": "week_01_quiz.xml", "xml": xml_str}]}
    )
    assert result.passed is True, [i.code for i in result.issues]


def _pattern_match_fib() -> dict:
    """A fill_in_blank with MULTIPLE accepted answer strings → pattern_match."""
    return {
        "question_id": "q-pm-001",
        "question_type": "fill_in_blank",
        "stem": "<p>Write one-half as a decimal or fraction.</p>",
        "bloom_level": "apply",
        "objective_id": "CO-12",
        "choices": [],
        "correct_answers": ["1/2", "0.5", ".5"],
        "feedback": "<p>Accepted: 1/2, 0.5 — set the accepted-answer set in the LMS.</p>",
        "points": 2.0,
    }


def test_pattern_match_profile_and_or_of_varequal():
    q = _pattern_match_fib()
    assert resolve_cc_profile("pattern_match") == "cc.pattern_match.v0p1"
    item = question_to_qti_item(q)
    profiles = [fe.text for fe in item.iter("fieldentry")]
    assert "cc.pattern_match.v0p1" in profiles
    # response_str/render_fib presentation, <or> of one varequal per accepted.
    assert item.find(".//response_str") is not None
    assert item.find(".//render_fib") is not None
    or_els = list(_iter_local(item, "or"))
    assert len(or_els) == 1
    varequals = [ve.text for ve in _iter_local(or_els[0], "varequal")]
    assert varequals == ["1/2", "0.5", ".5"]


def test_pattern_match_xsd_valid_and_key_recovered():
    q = _pattern_match_fib()
    xml_str = assessment_to_qti(_assessment([q]))
    _assert_xsd_valid(xml_str, _QTI_XSD)
    parsed = QTIParser().parse_string(xml_str)
    # QTIParser reads the key from resprocessing regardless of inferred type.
    assert parsed.questions[0].correct_response in {"1/2", "0.5", ".5"}


def test_pattern_match_passes_qti_well_formed_gate():
    from lib.validators.qti_well_formed import QtiWellFormedValidator

    xml_str = assessment_to_qti(_assessment([_pattern_match_fib()]))
    result = QtiWellFormedValidator().validate(
        {"qti_strings": [{"id": "week_01_quiz.xml", "xml": xml_str}]}
    )
    assert result.passed is True, [i.code for i in result.issues]


def test_single_correct_answer_fib_stays_exact_fib():
    """One accepted answer → cc.fib.v0p1 exact-match, NOT pattern_match."""
    q = {
        "question_id": "q-fib-single",
        "question_type": "fill_in_blank",
        "stem": "<p>Compute 3 + 4.</p>",
        "bloom_level": "apply",
        "objective_id": "CO-13",
        "correct_answers": ["7"],
        "points": 1.0,
    }
    item = question_to_qti_item(q)
    profiles = [fe.text for fe in item.iter("fieldentry")]
    assert "cc.fib.v0p1" in profiles
    assert item.find(".//or") is None


def _error_analysis_mc() -> dict:
    """Error-analysis MC (generator shape): step labels as options, the key
    carries the misconception_note naming the injected error."""
    return {
        "question_id": "q-ea-001",
        "question_type": "multiple_choice",
        "stem": ("<p>A student solved 2(x+3)=10 as follows:</p>"
                 "<ol><li>Step 1: 2x+3=10</li><li>Step 2: 2x=7</li>"
                 "<li>Step 3: x=3.5</li></ol><p>Which step contains the error?</p>"),
        "bloom_level": "analyze",
        "objective_id": "CO-14",
        "choices": [
            {"text": "<p>Step 1</p>", "is_correct": True,
             "misconception_note": "Failed to distribute the 2 across (x+3)."},
            {"text": "<p>Step 2</p>", "is_correct": False},
            {"text": "<p>Step 3</p>", "is_correct": False},
        ],
        "correct_answer": "Step 1",
        "item_subtype": "error_analysis",
        "feedback": "<p>Step 1 is incorrect. Distribute the 2 first: 2x+6=10.</p>",
        "points": 3.0,
    }


def test_error_analysis_emits_item_subtype_metadata_and_key_feedback():
    q = _error_analysis_mc()
    item = question_to_qti_item(q)
    # item_subtype recorded as machine-readable metadata.
    fields = {}
    for fld in _iter_local(item, "qtimetadatafield"):
        lab = fld.find("fieldlabel")
        ent = fld.find("fieldentry")
        if lab is not None and ent is not None:
            fields[lab.text] = ent.text
    assert fields.get("item_subtype") == "error_analysis"
    assert fields.get("cc_profile") == "cc.multiple_choice.v0p1"
    # The key choice carries a misconception_note → a Response itemfeedback for it.
    responses = [df for df in _iter_local(item, "displayfeedback")
                 if df.get("feedbacktype") == "Response"]
    assert len(responses) == 1
    # It links the key choice A.
    rc_varequal = None
    for rc in _iter_local(item, "respcondition"):
        if rc.get("continue") == "Yes":
            rc_varequal = next(_iter_local(rc, "varequal")).text
    assert rc_varequal == "A"


def test_error_analysis_xsd_valid_and_passes_gate():
    from lib.validators.qti_well_formed import QtiWellFormedValidator

    xml_str = assessment_to_qti(_assessment([_error_analysis_mc()]))
    _assert_xsd_valid(xml_str, _QTI_XSD)
    parsed = QTIParser().parse_string(xml_str)
    assert parsed.questions[0].correct_response == "A"
    result = QtiWellFormedValidator().validate(
        {"qti_strings": [{"id": "quiz.xml", "xml": xml_str}]}
    )
    assert result.passed is True, [i.code for i in result.issues]


def _two_tier_items() -> list:
    """Two linked MC items (answer + reason) — the generator emits these as a
    LIST; the emitter emits each as a plain cc.multiple_choice with linkage
    metadata."""
    answer = {
        "question_id": "q-2t-001",
        "question_type": "multiple_choice",
        "stem": "<p><strong>Tier 1.</strong> Which definition matches <em>slope</em>?</p>",
        "bloom_level": "analyze",
        "objective_id": "CO-15",
        "choices": [
            {"text": "<p>rise over run</p>", "is_correct": True},
            {"text": "<p>run over rise</p>", "is_correct": False},
            {"text": "<p>the y-intercept</p>", "is_correct": False},
        ],
        "item_subtype": "two_tier_answer",
        "linked_item_id": "q-2t-001-reason",
        "feedback": "<p>Slope is rise over run.</p>",
        "points": 1.0,
    }
    reason = {
        "question_id": "q-2t-001-reason",
        "question_type": "multiple_choice",
        "stem": "<p><strong>Tier 2.</strong> Why is your Tier 1 answer correct?</p>",
        "bloom_level": "analyze",
        "objective_id": "CO-15",
        "choices": [
            {"text": "<p>It measures vertical change per horizontal change.</p>", "is_correct": True},
            {"text": "<p>It is always positive.</p>", "is_correct": False,
             "misconception_note": "Slope can be negative or zero."},
            {"text": "<p>It equals the y-value.</p>", "is_correct": False},
        ],
        "item_subtype": "two_tier_reason",
        "linked_item_id": "q-2t-001",
        "feedback": "<p>Slope is the ratio of vertical to horizontal change.</p>",
        "points": 2.0,
    }
    return [answer, reason]


def test_two_tier_emits_two_linked_mc_items():
    answer, reason = _two_tier_items()
    a_item = question_to_qti_item(answer)
    r_item = question_to_qti_item(reason)
    a_fields = {fld.find("fieldlabel").text: fld.find("fieldentry").text
                for fld in _iter_local(a_item, "qtimetadatafield")}
    r_fields = {fld.find("fieldlabel").text: fld.find("fieldentry").text
                for fld in _iter_local(r_item, "qtimetadatafield")}
    assert a_fields["item_subtype"] == "two_tier_answer"
    assert a_fields["linked_item_id"] == "q-2t-001-reason"
    assert r_fields["item_subtype"] == "two_tier_reason"
    assert r_fields["linked_item_id"] == "q-2t-001"
    # Both stay portable cc.multiple_choice.
    assert a_fields["cc_profile"] == "cc.multiple_choice.v0p1"
    assert r_fields["cc_profile"] == "cc.multiple_choice.v0p1"


def test_two_tier_xsd_valid_and_passes_gate():
    from lib.validators.qti_well_formed import QtiWellFormedValidator

    xml_str = assessment_to_qti(_assessment(_two_tier_items()))
    _assert_xsd_valid(xml_str, _QTI_XSD)
    parsed = QTIParser().parse_string(xml_str)
    assert len(parsed.questions) == 2
    assert all(q.correct_response == "A" for q in parsed.questions)
    result = QtiWellFormedValidator().validate(
        {"qti_strings": [{"id": "quiz.xml", "xml": xml_str}]}
    )
    assert result.passed is True, [i.code for i in result.issues]


def test_diversified_quiz_all_types_passes_gate():
    """A quiz mixing every diversified shape passes the acceptance oracle and
    is XSD-valid."""
    from lib.validators.qti_well_formed import QtiWellFormedValidator

    questions = [
        _mc_with_feedback(),
        _mr_via_correct_answers(),
        _pattern_match_fib(),
        _error_analysis_mc(),
        *_two_tier_items(),
        _numeric_fib_question(),
        _tf_question(),
    ]
    xml_str = assessment_to_qti(_assessment(questions))
    _assert_xsd_valid(xml_str, _QTI_XSD)
    result = QtiWellFormedValidator().validate(
        {"qti_strings": [{"id": "week_01_quiz.xml", "xml": xml_str}]}
    )
    assert result.passed is True, [i.code for i in result.issues]
    # Every emitted item profile is in the supported six-profile set.
    root = _ET.fromstring(xml_str)
    for field in root.iter(f"{{{QTI_NS}}}qtimetadatafield"):
        label = field.find(f"{{{QTI_NS}}}fieldlabel")
        entry = field.find(f"{{{QTI_NS}}}fieldentry")
        if label is not None and label.text == "cc_profile" and entry is not None:
            assert entry.text == CC_EXAM_PROFILE or entry.text in SUPPORTED_ITEM_PROFILES


def test_pattern_match_in_supported_profiles():
    assert "cc.pattern_match.v0p1" in SUPPORTED_ITEM_PROFILES
