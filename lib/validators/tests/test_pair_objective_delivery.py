"""Wave W-D1 T1.4 — `PairObjectiveDeliveryValidator.validate()` walk tests.

Audit-walk only: asserts every pair on disk carries
`pair_objective_alignment` + `pair_objective_alignment_pass_rate` keys.
Per the validator docstring at `pair_objective_delivery.py:1038-1068`,
this is NOT a re-run of the per-pair tri-axis NLI / Bloom-gap / verb
fan-out; that path requires the synthesized-objectives map and is the
call-site's responsibility.

Mirror of `lib/validators/tests/test_assessment_objective_alignment.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.pair_objective_delivery import (  # noqa: E402
    PairObjectiveDeliveryValidator,
)
from lib.validators.pair.objective_delivery import (  # noqa: E402
    recompute_complete_objective_bloom_authority,
)
from lib.classifiers.nli_classifier import NliScore  # noqa: E402


# T1.4 stub — used in lib/validators/tests/test_pair_*.py (see plan §2
# "Shared test-stub helpers"; inline by design).
class _StubCapture:
    def __init__(self) -> None:
        self.calls = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.calls.append({
            "decision_type": decision_type,
            "decision": decision,
            "rationale": rationale,
            "kwargs": kw,
        })


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _pair_with_audit_fields() -> dict:
    return {
        "prompt": "x" * 50,
        "completion": "x" * 60,
        "chunk_id": "c1",
        "lo_refs": ["TO-01"],
        "bloom_level": "remember",
        "content_type": "explanation",
        "seed": 17,
        "decision_capture_id": "evt_test",
        "pair_objective_alignment": None,
        "pair_objective_alignment_pass_rate": None,
    }


def test_validator_instantiates() -> None:
    """Smoke: validator can be instantiated with no args."""
    validator = PairObjectiveDeliveryValidator()
    assert validator is not None
    assert callable(getattr(validator, "validate", None))


def test_empty_inputs_returns_gate_result_no_crash() -> None:
    """Empty-inputs case returns a GateResult (does not raise)."""
    result = PairObjectiveDeliveryValidator().validate({})
    assert result is not None
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert "MISSING_INPUTS" in codes


def test_pairs_with_audit_fields_pass(tmp_path: Path) -> None:
    capture = _StubCapture()
    inst = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst, [_pair_with_audit_fields() for _ in range(3)])

    result = PairObjectiveDeliveryValidator().validate({
        "instruction_pairs_path": str(inst),
        "decision_capture": capture,
    })
    assert result.passed is True
    assert len(capture.calls) == 1
    event = capture.calls[0]
    assert event["decision_type"] == "pair_objective_delivery_check"
    assert event["decision"].startswith("audit:passed")
    rationale = event["rationale"]
    assert len(rationale) >= 30
    assert "audited" in rationale


def test_missing_pair_objective_alignment_fires_critical(tmp_path: Path) -> None:
    capture = _StubCapture()
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_audit_fields()
    pair.pop("pair_objective_alignment")
    _write_jsonl(inst, [pair])

    result = PairObjectiveDeliveryValidator().validate({
        "instruction_pairs_path": str(inst),
        "decision_capture": capture,
    })
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert "MISSING_PAIR_OBJECTIVE_ALIGNMENT" in codes
    assert any(i.severity == "critical" for i in result.issues)
    assert capture.calls[0]["decision"].startswith("audit:failed")


def test_missing_pass_rate_fires_critical(tmp_path: Path) -> None:
    capture = _StubCapture()
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_audit_fields()
    pair.pop("pair_objective_alignment_pass_rate")
    _write_jsonl(inst, [pair])

    result = PairObjectiveDeliveryValidator().validate({
        "instruction_pairs_path": str(inst),
        "decision_capture": capture,
    })
    assert result.passed is False
    codes = [i.code for i in result.issues]
    assert "MISSING_PAIR_OBJECTIVE_ALIGNMENT_PASS_RATE" in codes


def test_no_capture_no_crash(tmp_path: Path) -> None:
    inst = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst, [_pair_with_audit_fields()])
    result = PairObjectiveDeliveryValidator().validate({
        "instruction_pairs_path": str(inst),
    })
    assert result.passed is True


class _Nli:
    def __init__(self, entailment: float, contradiction: float) -> None:
        self.score = NliScore(
            entailment=entailment,
            neutral=max(0.0, 1.0 - entailment - contradiction),
            contradiction=contradiction,
        )

    def score_pair(self, *, premise: str, hypothesis: str) -> NliScore:
        return self.score


class _DirectionalNli:
    """Return separately adjudicated pair→objective and objective→pair scores."""

    def __init__(self, forward: NliScore, reverse: NliScore) -> None:
        self._scores = [forward, reverse]
        self.calls = []

    def score_pair(self, *, premise: str, hypothesis: str) -> NliScore:
        self.calls.append((premise, hypothesis))
        return self._scores[len(self.calls) - 1]


def _score(entailment: float, contradiction: float = 0.02) -> NliScore:
    return NliScore(
        entailment=entailment,
        neutral=max(0.0, 1.0 - entailment - contradiction),
        contradiction=contradiction,
    )


def _objective_pair(observed_bloom="apply") -> dict:
    return {
        "id": "pair-1",
        "chunk_id": "chunk-1",
        "lo_refs": ["co-1"],
        "prompt": "Apply the least common denominator procedure.",
        "completion": "Apply the LCD before adding the numerators.",
        "observed_bloom": observed_bloom,
    }


def _objectives() -> dict:
    return {
        "co-1": {
            "statement": "Apply the least common denominator procedure.",
            "bloom_level": "apply",
            "bloom_verb": "apply",
        }
    }


def test_unverifiable_bloom_rejects_required_training_pair() -> None:
    status, reason, fields = PairObjectiveDeliveryValidator(
        nli=_Nli(0.91, 0.02)
    ).validate_pair(
        _objective_pair(observed_bloom=None),
        kind="instruction",
        objectives=_objectives(),
    )

    assert status == "rejected"
    assert reason == "objective_validation_unavailable"
    assert fields["pair_objective_alignment"][0]["status"] == "unverifiable"


def test_near_zero_objective_entailment_cannot_pass_as_neutral() -> None:
    status, reason, fields = PairObjectiveDeliveryValidator(
        nli=_Nli(0.000947, 0.02)
    ).validate_pair(
        _objective_pair(),
        kind="instruction",
        objectives=_objectives(),
    )

    assert status == "rejected"
    assert reason == "objective_statement_undersupported"
    assert fields["pair_objective_alignment"][0][
        "statement_entailment_score"
    ] == 0.000947


def test_verifiable_objective_delivery_records_scores_and_passes() -> None:
    pair = _objective_pair()
    pair["bloom_level"] = "apply"
    status, reason, fields = PairObjectiveDeliveryValidator(
        nli=_Nli(0.91, 0.02)
    ).validate_pair(
        pair,
        kind="instruction",
        objectives=_objectives(),
    )

    assert status == "validated"
    assert reason is None
    alignment = fields["pair_objective_alignment"][0]
    assert alignment["status"] == "delivered"
    assert alignment["statement_entailment_score"] == 0.91
    assert alignment["observed_bloom"] == "apply"
    assert fields["pair_objective_alignment_pass_rate"] == 1.0
    assert fields["observed_bloom"] == "apply"


def test_prompt_is_zero_evidence_for_objective_delivery() -> None:
    pair = _objective_pair()
    pair["prompt"] = _objectives()["co-1"]["statement"]
    pair["completion"] = "A response that omits the requested procedure."
    nli = _DirectionalNli(_score(0.05), _score(0.05))
    status, reason, _ = PairObjectiveDeliveryValidator(nli=nli).validate_pair(
        pair, kind="instruction", objectives=_objectives(),
    )
    assert status == "rejected"
    assert reason == "objective_statement_undersupported"
    assert nli.calls[0][0] == pair["completion"]
    assert pair["prompt"] not in nli.calls[0][0]


def test_high_contradiction_cannot_be_authoritative_delivered() -> None:
    pair = _objective_pair()
    status, reason, fields = PairObjectiveDeliveryValidator(
        nli=_Nli(0.67333984375, 0.92041015625)
    ).validate_pair(pair, kind="instruction", objectives=_objectives())
    assert status == "rejected"
    assert reason == "objective_statement_undersupported"
    entry = fields["pair_objective_alignment"][0]
    assert entry["status"] == "underdelivered"
    assert entry["contradiction_score"] == 0.92041015625
    assert recompute_complete_objective_bloom_authority(
        {**pair, **fields, "bloom_level": "apply"}
    ) is None


def _complete_objective_pair() -> dict:
    return {
        "chosen": "A complete objective-grounded answer.",
        "lo_refs": ["co-1"],
        "bloom_level": "analyze",
        "bloom_alignment": None,
        "pair_objective_alignment_pass_rate": 1.0,
        "pair_objective_alignment": [{
            "objective_id": "co-1",
            "status": "delivered",
            "statement_entailment_score": 0.78,
            "contradiction_score": 0.10,
            "bloom_gap": 0,
            "verb_match_count": 1,
            "declared_bloom": "analyze",
            "observed_bloom": "analyze",
            "entailment_threshold": 0.45,
        }],
    }


def test_complete_objective_bloom_authority_recomputes_without_null_weakening():
    pair = _complete_objective_pair()
    assert recompute_complete_objective_bloom_authority(pair) == {
        "observed_bloom": "analyze",
        "objective_ids": ["co-1"],
        "entry_count": 1,
    }


def test_objective_bloom_authority_rejects_null_false_threshold_id_and_tamper():
    import copy

    mutations = (
        ("pair_objective_alignment_pass_rate", None),
        ("bloom_alignment", False),
    )
    # Explicit false is a release-verifier failure and is never converted to
    # true by this recomputation surface.
    for field, value in mutations:
        pair = _complete_objective_pair()
        pair[field] = value
        authority = recompute_complete_objective_bloom_authority(pair)
        if field == "bloom_alignment":
            assert authority is not None
            assert pair["bloom_alignment"] is False
        else:
            assert authority is None

    for key, value in (
        ("objective_id", "co-other"),
        ("status", "underdelivered"),
        ("statement_entailment_score", 0.44),
        ("contradiction_score", 0.50),
        ("observed_bloom", "apply"),
        ("declared_bloom", "apply"),
        ("bloom_gap", 1),
    ):
        pair = copy.deepcopy(_complete_objective_pair())
        pair["pair_objective_alignment"][0][key] = value
        assert recompute_complete_objective_bloom_authority(pair) is None


def test_cross_pair_objective_substitution_fails_identity_binding():
    pair = _complete_objective_pair()
    pair["lo_refs"] = ["co-2"]
    assert recompute_complete_objective_bloom_authority(pair) is None


def test_concrete_skill_instance_passes_reverse_semantic_direction() -> None:
    """A concrete worked method need not entail the whole abstract competency."""
    capture = _StubCapture()
    nli = _DirectionalNli(_score(0.18), _score(0.78))
    pair = {
        "id": "pair-field-method",
        "chunk_id": "chunk-field-method",
        "lo_refs": ["objective-field-model"],
        "prompt": (
            "Explain how to convert a rectangular-field problem with given "
            "length and diagonal relationships into a quadratic equation "
            "using the Pythagorean theorem."
        ),
        "completion": (
            "Let width be w, express the stated length and diagonal in w, "
            "then apply the Pythagorean theorem to form the quadratic equation."
        ),
        "observed_bloom": "understand",
    }
    objectives = {
        "objective-field-model": {
            "statement": (
                "Translate a word problem involving a rectangular field with "
                "diagonal and dimensional relationships into a quadratic equation."
            ),
            "bloom_level": "understand",
            "bloom_verb": "translate",
        }
    }

    status, reason, fields = PairObjectiveDeliveryValidator(
        nli=nli, decision_capture=capture
    ).validate_pair(pair, kind="instruction", objectives=objectives)

    assert status == "validated"
    assert reason is None
    assert fields["pair_objective_alignment"][0][
        "statement_entailment_score"
    ] == 0.78
    assert len(nli.calls) == 2
    rationale = capture.calls[0]["rationale"]
    assert "reverse_entailment_score=0.7800" in rationale
    assert "pair-field-method" in rationale
    assert "chunk-field-method" in rationale
    assert "bidirectional_instantiated_skill" in rationale


def test_delivered_skill_does_not_require_literal_objective_verb() -> None:
    """Cognitive delivery is behavior, not literal Bloom-verb repetition."""
    pair = {
        "id": "pair-parabola-direction",
        "chunk_id": "chunk-parabola-direction",
        "lo_refs": ["objective-parabola-direction"],
        "prompt": "Describe how coefficient a indicates the direction a parabola opens.",
        "completion": (
            "For coefficient a greater than zero it opens upward; for "
            "coefficient a less than zero it opens downward."
        ),
        "observed_bloom": "understand",
    }
    objectives = {
        "objective-parabola-direction": {
            "statement": (
                "Identify whether a parabola opens upward or downward by "
                "examining coefficient a in y = ax^2 + bx + c."
            ),
            "bloom_level": "understand",
            "bloom_verb": "identify",
        }
    }

    status, reason, fields = PairObjectiveDeliveryValidator(
        nli=_Nli(0.99, 0.0)
    ).validate_pair(pair, kind="instruction", objectives=objectives)

    assert status == "validated"
    assert reason is None
    alignment = fields["pair_objective_alignment"][0]
    assert alignment["status"] == "delivered"
    assert alignment["verb_match_count"] == 0


def test_semantically_wrong_coefficient_is_rejected() -> None:
    """Topical word overlap cannot rescue an adjudicated semantic mismatch."""
    pair = {
        "id": "pair-wrong-coefficient",
        "chunk_id": "chunk-wrong-coefficient",
        "lo_refs": ["objective-parabola-direction"],
        "prompt": "Describe how coefficient c indicates the direction a parabola opens.",
        "completion": (
            "A positive c makes the parabola open upward and a negative c "
            "makes it open downward."
        ),
        "observed_bloom": "understand",
    }
    objectives = {
        "objective-parabola-direction": {
            "statement": (
                "Identify whether a parabola opens upward or downward by "
                "examining coefficient a in y = ax^2 + bx + c."
            ),
            "bloom_level": "understand",
            "bloom_verb": "identify",
        }
    }

    status, reason, fields = PairObjectiveDeliveryValidator(
        # The deployed NLI model can rate these topically similar surfaces
        # above the entailment floor; the named-symbol constraint must still
        # catch the substantive a→c substitution.
        nli=_DirectionalNli(_score(0.52), _score(0.74))
    ).validate_pair(pair, kind="instruction", objectives=objectives)

    assert status == "rejected"
    assert reason == "objective_statement_undersupported"
    assert fields["pair_objective_alignment"][0]["status"] == "underdelivered"


def test_generic_advice_does_not_instantiate_objective_skill() -> None:
    pair = {
        "id": "pair-generic-advice",
        "chunk_id": "chunk-generic-advice",
        "lo_refs": ["objective-field-model"],
        "prompt": "How should the problem be solved?",
        "chosen": "Use a formula carefully.",
        "observed_bloom": "understand",
    }
    objectives = {
        "objective-field-model": {
            "statement": (
                "Translate a word problem involving a rectangular field with "
                "diagonal and dimensional relationships into a quadratic equation."
            ),
            "bloom_level": "understand",
            "bloom_verb": "translate",
        }
    }

    status, reason, _ = PairObjectiveDeliveryValidator(
        nli=_DirectionalNli(_score(0.04), _score(0.06))
    ).validate_pair(pair, kind="preference", objectives=objectives)

    assert status == "rejected"
    assert reason == "objective_statement_undersupported"
