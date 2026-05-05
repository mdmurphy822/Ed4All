"""Wave 26 — AssessmentQualityValidator real-failure-mode tests.

The legacy validator's placeholder regex list missed the actual
real-world failure modes observed on a production RAG training run:

1. Distinct-stem ratio across an assessment (30 questions, 2 unique stems).
2. TOC-fragment correct answers ("1.1 Structural ... 14 1.7 ...").
3. Cross-question distractor duplication (single template string on every Q).
4. Verb-less stems ("Which best describes Structural?") bypassing Bloom.

Each test here locks in one failure mode. See the Wave 26 spec in the
worktree task description.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.assessment import AssessmentQualityValidator  # noqa: E402


def _q(qid: str, stem: str, correct: str, distractors=None, qtype="multiple_choice"):
    """Build a minimal MCQ question dict."""
    if distractors is None:
        distractors = [f"Distractor {qid}-{i}" for i in range(3)]
    choices = [{"text": correct, "is_correct": True}]
    for d in distractors:
        choices.append({"text": d, "is_correct": False})
    return {
        "question_id": qid,
        "question_type": qtype,
        "stem": stem,
        "bloom_level": "understand",
        "objective_id": "LO-01",
        "choices": choices,
    }


def _q_tf(qid: str, stem: str, correct_answer: str = "True"):
    return {
        "question_id": qid,
        "question_type": "true_false",
        "stem": stem,
        "bloom_level": "remember",
        "objective_id": "LO-01",
        "correct_answer": correct_answer,
        "choices": [
            {"text": "True", "is_correct": correct_answer == "True"},
            {"text": "False", "is_correct": correct_answer == "False"},
        ],
    }


def test_30_questions_2_stems_fails_low_stem_diversity():
    """Real-run smoking gun: 30 questions, 2 distinct stems. Must fail
    LOW_STEM_DIVERSITY as critical."""
    stems = [
        "<p>Explain the structural changes in the economy</p>",
        "<p>Describe how the knowledge society emerged</p>",
    ]
    # 30 questions alternating 2 stems → 2/30 = 0.067 distinct ratio.
    questions = [
        _q(
            qid=f"q-{i:03d}",
            stem=stems[i % 2],
            correct=f"Unique answer {i}",
        )
        for i in range(30)
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    codes = [i.code for i in result.issues]
    assert "LOW_STEM_DIVERSITY" in codes, f"codes: {codes}"
    # A critical-severity issue should flip passed=False regardless of
    # score threshold.
    assert not result.passed
    crit = [i for i in result.issues if i.severity == "critical"]
    assert any(i.code == "LOW_STEM_DIVERSITY" for i in crit)


def test_identical_distractor_template_fails_templated_distractors():
    """10 questions, all share the same canned distractor text. Must fail
    TEMPLATED_DISTRACTORS (>= 30% repetition)."""
    template = (
        "Students often assume this is a single idea with a single definition"
    )
    questions = []
    for i in range(10):
        questions.append(_q(
            qid=f"q-{i:03d}",
            stem=f"<p>Explain concept number {i}</p>",
            correct=f"Correct-{i}",
            distractors=[template, f"Unique-{i}-a", f"Unique-{i}-b"],
        ))
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    codes = [i.code for i in result.issues]
    assert "TEMPLATED_DISTRACTORS" in codes, f"codes: {codes}"
    assert not result.passed


def test_toc_fragment_correct_answer_fails_toc_check():
    """The exact real-run failure: correct answer is raw TOC text
    with page numbers. Must fail TOC_FRAGMENT_ANSWER."""
    toc_answer = (
        "1.1 Structural changes in the economy: the growth of a "
        "knowledge society 14 1.7 From the periphery to the core 22"
    )
    questions = [
        _q(
            qid="q-001",
            stem="<p>Which of the following best describes Structural?</p>",
            correct=toc_answer,
        )
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    codes = [i.code for i in result.issues]
    assert "TOC_FRAGMENT_ANSWER" in codes, f"codes: {codes}"
    crit = [i for i in result.issues if i.severity == "critical"]
    assert any(i.code == "TOC_FRAGMENT_ANSWER" for i in crit)


def test_toc_fragment_in_fill_in_blank_correct_answer():
    """correct_answer field (fill-in-blank shape) with TOC text."""
    toc_answer = "3 The Calvin Cycle 42 4 Photosynthesis 56 5 Carbon 78"
    q = {
        "question_id": "q-001",
        "question_type": "fill_in_blank",
        "stem": "<p>Explain the photosynthesis process</p>",
        "bloom_level": "understand",
        "objective_id": "LO-01",
        "correct_answer": toc_answer,
    }
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": [q]},
    })
    codes = [i.code for i in result.issues]
    assert "TOC_FRAGMENT_ANSWER" in codes


def test_legitimate_short_correct_answer_passes():
    """A clean short correct answer ("photosynthesis") must NOT fire the
    TOC check."""
    questions = []
    for i in range(5):
        questions.append({
            "question_id": f"q-{i:03d}",
            "question_type": "fill_in_blank",
            "stem": f"<p>Explain the role of organelle number {i}</p>",
            "bloom_level": "understand",
            "objective_id": "LO-01",
            "correct_answer": f"photosynthesis_{i}",
        })
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    codes = [i.code for i in result.issues]
    assert "TOC_FRAGMENT_ANSWER" not in codes
    # No critical issues → gate passes.
    assert not any(i.severity == "critical" for i in result.issues)
    assert result.passed


def test_single_verbless_tf_stem_warns_but_passes():
    """A single T/F stem without a Bloom verb is allowed (one exception
    per assessment). All other stems have Bloom verbs."""
    questions = [
        _q("q-001", "<p>Explain the cell cycle</p>", "cell division occurs here 1"),
        _q("q-002", "<p>Describe mitosis phases</p>", "prophase 2"),
        _q("q-003", "<p>Compare and contrast mitosis vs meiosis</p>", "chromosome number 3"),
        _q("q-004", "<p>Analyze the role of spindle fibers</p>", "separate chromatids 4"),
        _q_tf("q-005", "<p>Mitochondria have their own DNA.</p>"),  # verbless TF
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
        "min_score": 0.5,
    })
    # VERB_LESS_STEM appears as a warning for q-005 but there are no
    # critical issues.
    codes = [i.code for i in result.issues]
    assert "VERB_LESS_STEM" in codes
    assert not any(i.severity == "critical" for i in result.issues)


def test_60pct_stem_diversity_fails_below_70pct_threshold():
    """6 unique stems in 10 questions = 60% → fails (threshold 70%)."""
    stems = [
        "<p>Explain concept A</p>",
        "<p>Describe mechanism B</p>",
        "<p>Analyze function C</p>",
        "<p>Compare items D and E</p>",
        "<p>Evaluate outcome F</p>",
        "<p>Identify pattern G</p>",
    ]
    questions = []
    # 6 unique + 4 duplicates of the first stem → 6/10 = 0.6
    for i, s in enumerate(stems):
        questions.append(_q(f"q-{i:03d}", s, f"answer-{i}"))
    for i in range(4):
        questions.append(_q(f"q-dup-{i}", stems[0], f"answer-dup-{i}"))
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    codes = [i.code for i in result.issues]
    assert "LOW_STEM_DIVERSITY" in codes
    assert not result.passed


def test_5_unique_questions_passes_clean():
    """5 genuinely distinct questions with good Bloom verbs and distinct
    answers. No critical issues → passes."""
    questions = [
        _q("q-001", "<p>Explain what mitochondria are</p>", "organelle producing ATP 1"),
        _q("q-002", "<p>Describe the Calvin cycle</p>", "carbon fixation pathway 2"),
        _q("q-003", "<p>Compare glycolysis to fermentation</p>", "glycolysis uses oxygen 3"),
        _q("q-004", "<p>Analyze photosystem II function</p>", "splits water molecules 4"),
        _q("q-005", "<p>Identify the major cellular respiration stages</p>", "three stages 5"),
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    assert not any(i.severity == "critical" for i in result.issues), [
        i.to_dict() for i in result.issues
    ]
    assert result.passed


def test_low_answer_diversity_fails():
    """10 unique stems, all share the same correct answer. Must fail
    LOW_ANSWER_DIVERSITY (1/10 = 0.1 < 0.6 threshold)."""
    single_answer = "The only answer"
    questions = []
    for i in range(10):
        questions.append(_q(
            qid=f"q-{i:03d}",
            stem=f"<p>Explain unique concept number {i}</p>",
            correct=single_answer,
        ))
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    codes = [i.code for i in result.issues]
    assert "LOW_ANSWER_DIVERSITY" in codes


def test_long_toc_passage_with_headings_fails():
    """A >500-char answer with >= 3 integers and >= 2 dotted-numeric
    headings is rejected as TOC prose."""
    long_toc = (
        "Chapter overview 1.1 Photosynthesis basics 14 introduces the "
        "two-stage model. 1.2 The light reactions 22 describes the "
        "thylakoid membranes. Students should review pages 14, 22, "
        "and 36 before attempting this assessment. 2.1 Calvin cycle "
        "36 elaborates further." * 3
    )
    assert len(long_toc) > 500
    q = {
        "question_id": "q-001",
        "question_type": "fill_in_blank",
        "stem": "<p>Explain the photosynthesis model</p>",
        "bloom_level": "understand",
        "correct_answer": long_toc,
    }
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": [q]},
    })
    codes = [i.code for i in result.issues]
    assert "TOC_FRAGMENT_ANSWER" in codes


def test_pervasive_verbless_stems_escalates_critical():
    """Multiple non-TF verb-less stems exhaust the one-exception budget
    and should escalate to critical."""
    questions = [
        _q("q-001", "<p>Widget</p>", "widget answer 1"),
        _q("q-002", "<p>Gadget</p>", "gadget answer 2"),
        _q("q-003", "<p>Sprocket</p>", "sprocket answer 3"),
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    codes = [i.code for i in result.issues]
    assert "PERVASIVE_VERBLESS_STEMS" in codes


def test_mix_verbful_and_one_tf_verbless_passes():
    """4 verbful + 1 TF-verbless → passes (one-exception rule)."""
    questions = [
        _q("q-001", "<p>Explain the role of the mitochondrion</p>", "organelle ATP 1"),
        _q("q-002", "<p>Describe the Calvin cycle stages</p>", "carbon fixation 2"),
        _q("q-003", "<p>Compare glycolysis to fermentation</p>", "oxygen presence 3"),
        _q("q-004", "<p>Analyze the photosystem II function</p>", "splits water 4"),
        _q_tf("q-005", "<p>Mitochondria carry their own DNA separately from nuclear DNA.</p>"),
    ]
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": questions},
    })
    assert not any(i.severity == "critical" for i in result.issues)
    assert result.passed


def test_assessment_placeholder_emits_critical_severity():
    """Worker W4: the four placeholder GateIssue codes
    (PLACEHOLDER_QUESTION, PLACEHOLDER_CHOICE, PLACEHOLDER_ANSWER,
    PLACEHOLDER_FEEDBACK) are fail-closed defense-in-depth — any leak
    of placeholder text into a published assessment payload MUST flip
    the gate to passed=False at critical severity, not merely degrade
    the score-based pass threshold.

    Synthesizes one question that hits all four sites simultaneously:
    placeholder regex match in the stem, in a choice, in correct_answer,
    and in feedback. Asserts each code is emitted at severity="critical"
    and that the overall gate fails.
    """
    placeholder_question = {
        "question_id": "q-placeholder-all-sites",
        "question_type": "multiple_choice",
        "stem": "<p>What is the concept from LO-001?</p>",  # PLACEHOLDER_QUESTION
        "bloom_level": "remember",
        "objective_id": "LO-001",
        "choices": [
            # PLACEHOLDER_CHOICE — matches "Correct answer based on content"
            {"text": "<p>Correct answer based on content</p>", "is_correct": True},
            {"text": "<p>Plausible distractor A</p>", "is_correct": False},
            {"text": "<p>Plausible distractor B</p>", "is_correct": False},
            {"text": "<p>Plausible distractor C</p>", "is_correct": False},
        ],
        # PLACEHOLDER_ANSWER — fill-in-blank shape
        "correct_answer": "Review content for objective LO-001",
        # PLACEHOLDER_FEEDBACK — same regex family
        "feedback": "<p>Review content for objective LO-001.</p>",
    }
    result = AssessmentQualityValidator().validate({
        "assessment_data": {"questions": [placeholder_question]},
    })
    by_code = {i.code: i for i in result.issues}
    expected_codes = {
        "PLACEHOLDER_QUESTION",
        "PLACEHOLDER_CHOICE",
        "PLACEHOLDER_ANSWER",
        "PLACEHOLDER_FEEDBACK",
    }
    missing = expected_codes - by_code.keys()
    assert not missing, (
        f"Expected all four placeholder codes, missing: {missing}; "
        f"emitted codes: {list(by_code.keys())}"
    )
    for code in expected_codes:
        assert by_code[code].severity == "critical", (
            f"{code} severity must be 'critical' (fail-closed defense-in-depth), "
            f"got {by_code[code].severity!r}"
        )
    # Critical severity flips passed=False regardless of score.
    assert not result.passed, (
        f"placeholder leak must fail the gate; got passed=True with issues: "
        f"{[(i.code, i.severity) for i in result.issues]}"
    )
