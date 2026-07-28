"""Three evidenced defects in the pair-claim-support NLI verifier.

Each defect below was measured on a real synthesis run before it was fixed, so
each test pins the specific behaviour the measurement showed to be wrong rather
than a general property. The verifier is broadly correct — an audit put its
false-negative rate at ~11% — so none of these loosens a threshold: the
entailment floor stays 0.70, the unsupported ceiling 0.20, the contradicted
ceiling 0.05.

1. Math notation was not normalized, and the normalizer that DID exist was
   applied to the hypothesis only. A one-sided normalizer cannot align two
   strings; it can only move one away from the other.
2. The contradiction head fires spuriously on long textbook premises, so
   `contradiction >= 0.50` alone was routing meaningfully-entailed sentences
   into the trapdoor `contradicted` bucket whose ceiling is 0.05.
3. Rejected pairs discarded their per-sentence scores, which is why an audit of
   150 rejections could only adjudicate 14.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.classifiers.nli_classifier import NliScore  # noqa: E402
from lib.validators.pair.claim_support import (  # noqa: E402
    _DEFAULT_CONTRADICTION_FLOOR,
    _DEFAULT_CONTRADICTION_MAX_ENTAILMENT,
    _DEFAULT_ENTAILMENT_FLOOR,
    _MAX_PERSISTED_CLAIM_CHARS,
    _MAX_PERSISTED_FAILING_CLAIMS,
    PairClaimSupportValidator,
    _normalize_nli_text,
    summarize_claim_support_rejection,
)


# --------------------------------------------------------------------- #
# DEFECT 1 — math notation, and the premise/hypothesis asymmetry.
#
# Measured casualty: this completion sentence scored entailment 0.583 /
# contradiction 0.348 -> unsupported against a source carrying the same
# sentence in unicode-superscript form. The tell that it was notation rather
# than judgment: a LOOSER paraphrase of the same chunk ("create" for "form")
# scored 0.823 and passed.
# --------------------------------------------------------------------- #

_SOURCE_MATH = (
    "Key Idea: To form a perfect square trinomial from x² + bx, "
    "add (b/2)² to the expression."
)
_CLAIM_MATH = (
    "To form a perfect square trinomial from x^2 + bx, "
    "add (b/2)^2 to the expression."
)


def test_unicode_superscripts_and_carets_normalize_onto_each_other() -> None:
    """The two spellings of the SAME sentence must reach NLI identically."""
    normalized_source = _normalize_nli_text(_SOURCE_MATH)
    normalized_claim = _normalize_nli_text(_CLAIM_MATH)

    assert "x^2" in normalized_source
    assert "(b/2)^2" in normalized_source
    # The source line differs from the claim only by its "Key Idea: " lead-in,
    # so after normalization the claim is a literal substring of it.
    assert normalized_claim in normalized_source


def test_subscripts_and_braced_scripts_normalize_without_losing_the_relation() -> None:
    """Notation is unified; the exponent/index relation is preserved.

    This is the reason `fold_math` is not reused wholesale: it folds `x**2` to
    the bare tokens `x 2`, which no longer states an exponent for an entailment
    check to verify.
    """
    assert _normalize_nli_text("x₁ and x₂") == "x_1 and x_2"
    assert _normalize_nli_text("x^{2} and a_{ij}") == "x^2 and a_ij"
    assert _normalize_nli_text("5 − 3") == "5 - 3"
    # Strict no-op on prose.
    prose = "When a is greater than zero, the parabola opens upward."
    assert _normalize_nli_text(prose) == prose


class _RecordingNli:
    """Returns a fixed score and records exactly what it was asked to score."""

    def __init__(self, score: NliScore) -> None:
        self.score = score
        self.pairs: list = []

    def score_pair(self, *, premise, hypothesis):
        self.pairs.append((premise, hypothesis))
        return self.score

    def score_batch(self, *, pairs):
        return [
            self.score_pair(premise=p, hypothesis=h) for p, h in pairs
        ]


def test_premise_is_normalized_too_not_only_the_hypothesis() -> None:
    """The asymmetry that defeated the pre-existing typography folding.

    Before the fix the hypothesis went through `_normalize_nli_text` and the
    premise went through `str()`, so a source carrying unicode superscripts (or
    a smart quote) never met the normalized claim.
    """
    nli = _RecordingNli(NliScore(entailment=0.95, neutral=0.05, contradiction=0.0))
    PairClaimSupportValidator(nli=nli).validate_pair(
        {"chunk_id": "c1", "completion": _CLAIM_MATH},
        kind="instruction",
        chunk={"id": "c1", "text": _SOURCE_MATH},
    )

    assert nli.pairs, "fixture is inert — nothing was scored"
    for premise, hypothesis in nli.pairs:
        assert "²" not in premise, (
            "premise reached NLI un-normalized: "
            "a one-sided normalizer cannot fix a mismatch"
        )
        assert "²" not in hypothesis
        # Both sides now speak caret form, so the claim aligns with its source.
        assert hypothesis in premise


def test_premise_typography_is_normalized_too() -> None:
    """Same asymmetry, in the alphabet the translation table already covered."""
    nli = _RecordingNli(NliScore(entailment=0.95, neutral=0.05, contradiction=0.0))
    PairClaimSupportValidator(nli=nli).validate_pair(
        {"chunk_id": "c1", "completion": "The student's model is complete."},
        kind="instruction",
        chunk={"id": "c1", "text": "The student’s model is complete."},
    )

    premise, hypothesis = nli.pairs[0]
    assert "’" not in premise
    assert hypothesis in premise


# --------------------------------------------------------------------- #
# DEFECT 2 — the contradiction head fires spuriously on this premise shape.
#
# Real scores from sentences that PASSED, and only because `entailed` is
# bucketed before `contradicted`: (0.989, 0.825), (0.984, 0.720),
# (0.975, 0.516), (0.823, 0.502). Measured casualty: a summary sentence
# restating two 0.998-entailed sentences scored (0.406, 0.549) and was marked
# `contradicted`, whose 0.05 ceiling rejected the whole pair.
# --------------------------------------------------------------------- #

_PARABOLA_SOURCE = (
    "The graph of a quadratic function is a parabola. When the leading "
    "coefficient a is greater than zero, the parabola opens upward. When a is "
    "less than zero, the parabola opens downward. The vertex is the turning "
    "point of the graph. The axis of symmetry passes through the vertex. The "
    "direction in which the parabola opens is determined solely by the sign of "
    "a; the values of b and c shift and translate the graph but never reverse "
    "its opening direction."
)
_UPWARD = "When a is greater than zero, the parabola opens upward."
_DOWNWARD = "When a is less than zero, the parabola opens downward."
_SUMMARY = "The direction of opening depends only on the sign of a."
#: Filler entailed sentences so the ONE mis-bucketed sentence lands where the
#: trapdoor actually is: below the 0.20 unsupported ceiling (1/8 = 0.125) but
#: above the 0.05 contradicted ceiling. That gap is the whole defect — before
#: the fix the sentence was routed to the bucket with the 4x tighter ceiling.
_FILLER = [
    "A quadratic function graphs as a parabola.",
    "The vertex is the turning point of the parabola.",
    "The axis of symmetry passes through the vertex.",
    "The coefficient b translates the graph horizontally.",
    "The coefficient c translates the graph vertically.",
]
_PARABOLA_COMPLETION = " ".join([_UPWARD, _DOWNWARD, *_FILLER, _SUMMARY])


class _MeasuredNli:
    """Replays the exact scores the audit recorded for this completion."""

    _SCORES = {
        _UPWARD: NliScore(entailment=0.998, neutral=0.001, contradiction=0.001),
        _DOWNWARD: NliScore(entailment=0.998, neutral=0.001, contradiction=0.001),
        _SUMMARY: NliScore(entailment=0.406, neutral=0.045, contradiction=0.549),
    }

    def score_pair(self, *, premise, hypothesis):
        for sentence, score in self._SCORES.items():
            if _normalize_nli_text(sentence) == hypothesis:
                return score
        return NliScore(entailment=0.90, neutral=0.10, contradiction=0.0)

    def score_batch(self, *, pairs):
        return [
            self.score_pair(premise=p, hypothesis=h) for p, h in pairs
        ]


def _validate_parabola():
    return PairClaimSupportValidator(nli=_MeasuredNli()).validate_pair(
        {"chunk_id": "chunk-parabola", "completion": _PARABOLA_COMPLETION},
        kind="instruction",
        chunk={"id": "chunk-parabola", "text": _PARABOLA_SOURCE},
    )


def test_meaningful_entailment_is_never_bucketed_as_contradicted() -> None:
    """The adjudicated error: a restatement of two entailed sentences.

    0.406 entailment is not a contradiction verdict — it is the model split
    between two mutually exclusive relations on a long premise. At worst the
    sentence is `unsupported`.
    """
    _, _, fields = _validate_parabola()
    summary = [
        entry for entry in fields["per_claim_support"]
        if entry["sentence"] == _SUMMARY
    ]
    assert len(summary) == 1
    assert summary[0]["contradiction"] >= _DEFAULT_CONTRADICTION_FLOOR, (
        "fixture is inert — the contradiction head is not above the floor"
    )
    assert summary[0]["outcome"] == "unsupported"
    assert fields["claim_contradicted_rate"] == 0.0


def test_the_spurious_contradiction_no_longer_rejects_the_pair() -> None:
    """The trapdoor: one sentence in eight is 0.125.

    Under the 0.20 unsupported ceiling, so the pair survives — but four times
    the 0.05 contradicted ceiling, so mis-bucketing that ONE sentence as
    contradicted rejected the whole pair.
    """
    status, reason, fields = _validate_parabola()
    assert fields["claim_contradicted_rate"] == 0.0
    unsupported_rate = sum(
        1 for entry in fields["per_claim_support"]
        if entry["outcome"] != "entailed"
    ) / len(fields["per_claim_support"])
    assert 0.05 < unsupported_rate <= 0.20, (
        "fixture is inert — the sentence must land in the band where the two "
        "ceilings disagree, or the test proves nothing about the bucketing"
    )
    assert (status, reason) == ("validated", None)


def test_a_genuine_contradiction_is_still_contradicted() -> None:
    """The fix TIGHTENS what counts as a contradiction; it must not empty it.

    A decisive contradiction puts near-zero mass on entailment, which is the
    regime the cutoff was placed below.
    """

    class _FlatContradiction:
        def score_pair(self, *, premise, hypothesis):
            return NliScore(entailment=0.02, neutral=0.08, contradiction=0.90)

        def score_batch(self, *, pairs):
            return [self.score_pair(premise=p, hypothesis=h) for p, h in pairs]

    status, reason, fields = PairClaimSupportValidator(
        nli=_FlatContradiction(),
    ).validate_pair(
        {
            "chunk_id": "c1",
            "completion": "When a is greater than zero the parabola opens downward.",
        },
        kind="instruction",
        chunk={"id": "c1", "text": _PARABOLA_SOURCE},
    )
    assert fields["per_claim_support"][0]["outcome"] == "contradicted"
    assert (status, reason) == ("rejected", "contradicted_claim")


def test_the_cutoff_sits_below_the_measured_casualty_and_under_the_floor() -> None:
    """Pins the two bounds the chosen value has to satisfy.

    Above: the adjudicated casualty scored 0.406 entailment, so a cutoff at or
    above it does not fix the evidenced defect. Below: the value must stay far
    under the 0.70 entailment floor so `entailed` bucketing is untouched.
    """
    assert _DEFAULT_CONTRADICTION_MAX_ENTAILMENT < 0.406
    assert _DEFAULT_CONTRADICTION_MAX_ENTAILMENT < _DEFAULT_ENTAILMENT_FLOOR
    # Margin against the low-order variation batched NLI scoring introduces.
    assert 0.406 - _DEFAULT_CONTRADICTION_MAX_ENTAILMENT >= 0.10


def test_thresholds_that_must_not_move() -> None:
    assert _DEFAULT_ENTAILMENT_FLOOR == 0.70
    assert _DEFAULT_CONTRADICTION_FLOOR == 0.50


def test_added_premises_still_cannot_worsen_a_sentence_outcome() -> None:
    """The monotonicity invariant, re-checked against the new bucketing.

    Adding premises moves entailment monotonically UP (max over a superset) and
    contradiction monotonically DOWN (the existing clamp). The new condition is
    `entailment < cutoff`, which those two movements can only make HARDER to
    satisfy — so an added premise can move a sentence OUT of `contradicted` but
    never INTO it. This test pins that direction on a sentence sitting right at
    the boundary.
    """
    quote = "Row reduction terminates in finitely many pivot operations."
    chunk_text = f"Chapter 4 opens with elimination. {quote} Pivoting order matters."
    sentence = "The pivot ordering guarantees a unique reduced echelon form."

    class _AdversarialNli:
        """The added quote out-entails AND out-contradicts the base chunk."""

        def score_pair(self, *, premise, hypothesis):
            if premise == quote:
                return NliScore(entailment=0.20, neutral=0.0, contradiction=0.80)
            return NliScore(entailment=0.10, neutral=0.85, contradiction=0.05)

        def score_batch(self, *, pairs):
            return [self.score_pair(premise=p, hypothesis=h) for p, h in pairs]

    _, _, fields = PairClaimSupportValidator(
        nli=_AdversarialNli(),
    ).validate_pair(
        {
            "chunk_id": "c1",
            "completion": sentence,
            "provenance": {"claim_evidence": [
                {"claim": "Row reduction terminates.", "evidence_quote": quote},
            ]},
        },
        kind="instruction",
        chunk={"id": "c1", "text": chunk_text},
    )
    entry = fields["per_claim_support"][0]
    assert entry["outcome"] == "unsupported"
    assert entry["contradiction"] <= 0.05


# --------------------------------------------------------------------- #
# DEFECT 3 — rejected pairs discarded their evidence.
# --------------------------------------------------------------------- #


def _rejected_fields(claim_count: int) -> dict:
    per_claim = [
        {
            "sentence": f"Unsupported assertion number {idx}.",
            "entailment": 0.05 + idx * 0.01,
            "contradiction": 0.10,
            "outcome": "unsupported",
            "source_chunk_ids": None,
        }
        for idx in range(claim_count)
    ]
    per_claim.append({
        "sentence": "A sentence the source entails.",
        "entailment": 0.95,
        "contradiction": 0.01,
        "outcome": "entailed",
        "source_chunk_ids": None,
    })
    return {
        "per_claim_support": per_claim,
        "claim_support_rate": 1.0 / len(per_claim),
        "claim_contradicted_rate": 0.0,
        "deps_missing": False,
    }


def test_rejection_evidence_carries_the_failing_sentences_and_their_scores() -> None:
    evidence = summarize_claim_support_rejection(
        _rejected_fields(3), rejection_reason="unsupported_claim",
    )
    assert evidence is not None
    assert evidence["stage"] == "claim_support"
    assert evidence["rejection_reason"] == "unsupported_claim"
    assert evidence["total_claims"] == 4
    assert evidence["outcome_counts"] == {"unsupported": 3, "entailed": 1}
    assert len(evidence["failing_claims"]) == 3
    first = evidence["failing_claims"][0]
    assert set(first) == {"sentence", "entailment", "contradiction", "outcome"}
    assert first["outcome"] == "unsupported"
    # Worst-entailment first, so the most damning sentence is never the one cut.
    entailments = [claim["entailment"] for claim in evidence["failing_claims"]]
    assert entailments == sorted(entailments)


def test_only_failing_sentences_are_persisted() -> None:
    """An entailed sentence explains nothing about a rejection."""
    evidence = summarize_claim_support_rejection(
        _rejected_fields(2), rejection_reason="unsupported_claim",
    )
    assert all(
        claim["outcome"] != "entailed" for claim in evidence["failing_claims"]
    )


def test_persisted_detail_is_capped_and_the_remainder_is_counted() -> None:
    """Bounded on purpose — the dispositions file is an operator artifact."""
    over_cap = _MAX_PERSISTED_FAILING_CLAIMS + 5
    evidence = summarize_claim_support_rejection(
        _rejected_fields(over_cap), rejection_reason="unsupported_claim",
    )
    assert len(evidence["failing_claims"]) == _MAX_PERSISTED_FAILING_CLAIMS
    assert evidence["failing_claims_truncated"] == 5


def test_a_long_sentence_is_clipped_and_marked_as_clipped() -> None:
    long_sentence = "word " * 400
    fields = {
        "per_claim_support": [{
            "sentence": long_sentence,
            "entailment": 0.1,
            "contradiction": 0.2,
            "outcome": "unsupported",
        }],
        "claim_support_rate": 0.0,
        "claim_contradicted_rate": 0.0,
    }
    evidence = summarize_claim_support_rejection(
        fields, rejection_reason="unsupported_claim",
    )
    persisted = evidence["failing_claims"][0]["sentence"]
    assert len(persisted) <= _MAX_PERSISTED_CLAIM_CHARS + len("…[truncated]")
    assert persisted.endswith("…[truncated]")


def test_no_source_chunk_text_is_copied_into_the_evidence() -> None:
    """Only the generated pair text is persisted; chunk_id resolves the source."""
    source_marker = "PROPRIETARY SOURCE PARAGRAPH THAT MUST NOT BE COPIED"
    _, _, fields = PairClaimSupportValidator(
        nli=_MeasuredNli(),
    ).validate_pair(
        {"chunk_id": "c1", "completion": _SUMMARY},
        kind="instruction",
        chunk={"id": "c1", "text": f"{source_marker} {_PARABOLA_SOURCE}"},
    )
    evidence = summarize_claim_support_rejection(
        fields, rejection_reason="unsupported_claim",
    )
    assert source_marker not in repr(evidence)


def test_graceful_degrade_and_empty_arms_record_nothing() -> None:
    assert summarize_claim_support_rejection(
        {"per_claim_support": None}, rejection_reason="unsupported_claim",
    ) is None
    assert summarize_claim_support_rejection(
        {"per_claim_support": []}, rejection_reason="unsupported_claim",
    ) is None
    assert summarize_claim_support_rejection(
        {}, rejection_reason=None,
    ) is None
