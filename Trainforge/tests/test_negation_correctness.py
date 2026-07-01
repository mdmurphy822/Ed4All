"""W7.4 — negation-correctness check for T/F + negated-MCQ distractors.

``AssessmentGenerator._negate_statement`` regex-flips a claim's polarity with
ZERO correctness check, so a regex-negated statement can be accidentally TRUE
(a double-negation of an already-negative source, or a vacuous no-op). These
tests pin the new ``_negation_is_verifiably_false`` guardrail + the
``_safe_negate_statement`` wrapper + the two call-site behaviours (MCQ drops a
bad negated distractor; T/F falls back to the TRUE variant), plus the
decision-capture emit on the reject path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from Trainforge.generators.assessment_generator import (
    AssessmentGenerator,
    QuestionData,
    SkippedItem,
)


@dataclass
class _Stmt:
    statement: str
    source_chunk_id: str = "chunk_1"
    key_subject: str = "theory"


class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


# --------------------------------------------------------------------- #
# _negation_is_verifiably_false — the deterministic guardrail.
# --------------------------------------------------------------------- #

def test_good_negation_of_positive_claim_accepted():
    assert AssessmentGenerator._negation_is_verifiably_false(
        "The theory is based on evidence.",
        "The theory is not based on evidence.",
    )


def test_polarity_swap_cue_free_accepted():
    assert AssessmentGenerator._negation_is_verifiably_false(
        "Pressure increases the reaction rate.",
        "Pressure decreases the reaction rate.",
    )


def test_qualifier_swap_accepted():
    assert AssessmentGenerator._negation_is_verifiably_false(
        "Extraneous load always increases difficulty.",
        "Extraneous load never increases difficulty.",
    )


def test_vacuous_negation_rejected():
    # No change happened → not a checkable falsification.
    assert not AssessmentGenerator._negation_is_verifiably_false(
        "The theory holds.", "The theory holds."
    )


def test_already_negative_source_rejected():
    # Source already carries a negation cue → double-negative risk.
    assert not AssessmentGenerator._negation_is_verifiably_false(
        "The theory is not based on evidence.",
        "The theory is not not based on evidence.",
    )


def test_double_negation_rejected():
    assert not AssessmentGenerator._negation_is_verifiably_false(
        "Enzymes speed up reactions.",
        "Enzymes do not never speed up reactions.",
    )


def test_safe_negate_returns_none_for_negative_source():
    gen = AssessmentGenerator(capture=None)
    assert gen._safe_negate_statement("Water never boils below 100C.") is None


def test_safe_negate_returns_text_for_positive_source():
    gen = AssessmentGenerator(capture=None)
    out = gen._safe_negate_statement("Water boils at 100C.")
    assert out is not None
    assert "not" in out.lower() or out.lower() != "water boils at 100c."


# --------------------------------------------------------------------- #
# Call-site integration — MCQ statement distractors.
# --------------------------------------------------------------------- #

class _StubExtractor:
    def __init__(self, statements: List[_Stmt]) -> None:
        self._statements = statements

    def extract_key_terms(self, chunks: Any) -> List[Any]:
        return []

    def extract_factual_statements(self, chunks: Any) -> List[_Stmt]:
        return list(self._statements)


def _gen_with_statements(statements: List[_Stmt], capture: Any = None):
    gen = AssessmentGenerator(capture=capture)
    gen._content_extractor = _StubExtractor(statements)
    return gen


def test_mcq_drops_unverifiable_negated_distractors():
    # First statement is the correct key; the rest are already-negative, so
    # their regex negations are NOT verifiably false → all dropped → <2 valid
    # distractors → skip-emit rather than ship true "wrong" options.
    capture = _RecordingCapture()
    statements = [
        _Stmt("Photosynthesis converts light to chemical energy."),
        _Stmt("Photosynthesis does not occur in the dark."),
        _Stmt("Respiration is not an anabolic process."),
        _Stmt("Enzymes are never consumed in a reaction."),
    ]
    gen = _gen_with_statements(statements, capture=capture)
    result = gen._generate_multiple_choice(
        question_id="q1",
        objective_id="LO-01",
        bloom_level="understand",
        level_config={},
        source_chunks=[{"id": "chunk_1", "content": "x"}],
    )
    assert isinstance(result, SkippedItem)
    assert result.reason == "negation_unverifiable"
    # Capture fired on the drop path.
    assert any(
        e.get("decision_type") == "distractor_generation" for e in capture.events
    )


def test_mcq_keeps_verifiable_negated_distractors():
    statements = [
        _Stmt("Mitochondria produce ATP."),
        _Stmt("Chloroplasts absorb sunlight."),
        _Stmt("Ribosomes synthesize proteins."),
        _Stmt("The nucleus stores DNA."),
    ]
    gen = _gen_with_statements(statements)
    result = gen._generate_multiple_choice(
        question_id="q2",
        objective_id="LO-01",
        bloom_level="understand",
        level_config={},
        source_chunks=[{"id": "chunk_1", "content": "x"}],
    )
    assert isinstance(result, QuestionData)
    correct = [c for c in result.choices if c["is_correct"]]
    distractors = [c for c in result.choices if not c["is_correct"]]
    assert len(correct) == 1
    assert len(distractors) >= 2
    # Every distractor differs from the correct key text.
    key = correct[0]["text"]
    assert all(d["text"] != key for d in distractors)


# --------------------------------------------------------------------- #
# Call-site integration — T/F never ships an accidentally-true "false" item.
# --------------------------------------------------------------------- #

def test_true_false_never_emits_unverifiable_false():
    # Already-negative source: if make_false fires, the negation is rejected
    # and the item falls back to the TRUE variant. So a returned "False" item
    # must ALWAYS carry a verifiably-false stem.
    statements = [_Stmt("Catalysts are not consumed during a reaction.")]
    gen = _gen_with_statements(statements)
    result = gen._generate_true_false(
        question_id="tf-negative",
        objective_id="LO-01",
        bloom_level="remember",
        level_config={},
        source_chunks=[{"id": "chunk_1", "content": "x"}],
    )
    assert isinstance(result, QuestionData)
    if result.correct_answer == "False":
        stem = result.stem.replace("<p>", "").replace("</p>", "")
        assert AssessmentGenerator._negation_is_verifiably_false(
            statements[0].statement, stem
        )
    else:
        # Fell back to the TRUE variant — the source statement verbatim.
        assert result.correct_answer == "True"
