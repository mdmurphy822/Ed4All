"""Worker W2.E — TrainingPairPromotionValidator unit tests.

Pins the per-pair filter contract: every pair returns
``(promotion_status, rejection_reason, new_fields)`` with the rejection
matrix below, and every return path emits exactly one
``training_pair_promotion_check`` decision-capture event.

Coverage:

- One test per rejection_reason (7 hard-rejection criteria).
- Accept-path test (clean instruction + clean preference pair).
- Embedding-deps-missing graceful-degrade fallback.
- Strict-mode (TRAINFORGE_REQUIRE_EMBEDDINGS=true) re-raises.
- Decision-capture-emit regression assertion (per W2 plan Test 1):
  every return path fires exactly one event.
- On-disk gate path (validate()) walks training_specs/*.jsonl and
  fails closed when promotion_status missing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from lib.validators.training_pair_promotion import (
    DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS,
    DEFAULT_MIN_ANSWER_SUPPORT_SCORE,
    DEFAULT_MIN_PROMPT_CHUNK_JACCARD,
    DEFAULT_MIN_RATIONALE_RICHNESS_SCORE,
    TrainingPairPromotionValidator,
)


# --------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------- #


class _RecordingCapture:
    """Minimal DecisionCapture stand-in recording log_decision kwargs."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


class _StubEmbedder:
    """Toy embedder that maps text → bag-of-tokens → unit vector.

    Cosine similarity between two texts is the cosine of the angle
    between their token bags. Deterministic, no ML dependency, lets us
    drive the per-criterion threshold tests without [embedding] extras.
    """

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}

    def _tokenise(self, text: str) -> List[str]:
        return [t.lower() for t in text.split() if t.strip()]

    def _ensure_idx(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def encode(self, text: str) -> List[float]:
        # Build a sparse-but-fixed-size vector. We don't know the
        # final vocab up-front, so resize lazily on every encode.
        tokens = self._tokenise(text)
        for tok in tokens:
            self._ensure_idx(tok)
        vec = [0.0] * len(self._vocab)
        for tok in tokens:
            vec[self._vocab[tok]] += 1.0
        # Normalise.
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class _FixedVocabEmbedder(_StubEmbedder):
    """``_StubEmbedder`` with the vocabulary pre-built from a corpus.

    ``_StubEmbedder`` grows its vocab lazily, so two ``encode`` calls
    return different-length vectors and the validator's cosine path
    fails over to the (sparser) Jaccard signal. Pre-seeding the vocab
    makes every vector the same length, so cosine similarity actually
    computes — required by tests that pin cosine-calibrated floors.
    """

    def __init__(self, corpus: List[str]) -> None:
        super().__init__()
        for text in corpus:
            for tok in self._tokenise(text):
                self._ensure_idx(tok)


class _StubBloomEnsemble:
    """Toy ensemble that returns a configured (level, score) tuple."""

    def __init__(
        self,
        level: str = "remember",
        score: float = 0.85,
    ) -> None:
        self._level = level
        self._score = score

    def classify(self, text: str) -> Dict[str, Any]:
        return {
            "winner_level": self._level,
            "winner_score": self._score,
            "dispersion": 0.0,
            "per_member": [],
        }


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


_CHUNK_TEXT = (
    "RDF is a declarative graph data model. "
    "An RDF triple has the form subject predicate object. "
    "RDF is the W3C standard for representing knowledge as graphs."
)


def _instruction_pair(
    *,
    chunk_id: str = "rdf_shacl_chunk_00001",
    prompt: str = "What is the structure of an RDF triple in the source?",
    completion: str = (
        "An RDF triple is composed of a subject, a predicate, "
        "and an object — the canonical declarative graph form."
    ),
    bloom_level: str = "remember",
    rationale: str = (
        "Definition recall: the source explicitly states the subject "
        "predicate object structure as the canonical RDF triple form."
    ),
    source_chunk_id: Optional[str] = "rdf_shacl_chunk_00001",
) -> Dict[str, Any]:
    pair: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "prompt": prompt,
        "completion": completion,
        "bloom_level": bloom_level,
        "content_type": "definition",
        "lo_refs": ["TO-01"],
        "seed": 42,
        "decision_capture_id": "evt_0001",
        "rationale": rationale,
    }
    if source_chunk_id is not None:
        pair["source_chunk_id"] = source_chunk_id
    return pair


def _preference_pair(
    *,
    chunk_id: str = "rdf_shacl_chunk_00001",
    prompt: str = "Which best describes the structure of an RDF triple?",
    chosen: str = (
        "An RDF triple is composed of a subject, a predicate, and an "
        "object — the declarative graph data form per the W3C standard."
    ),
    rejected: str = (
        "An RDF triple is a procedural function-call signature with "
        "named arguments evaluated left-to-right at runtime."
    ),
    bloom_level: str = "understand",
    rationale: str = (
        "Comparative reasoning: the chosen completion preserves the "
        "declarative subject predicate object form while the rejected "
        "completion encodes a procedural-language misconception."
    ),
    source_chunk_id: Optional[str] = "rdf_shacl_chunk_00001",
) -> Dict[str, Any]:
    pair: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "bloom_level": bloom_level,
        "content_type": "definition",
        "lo_refs": ["TO-01"],
        "seed": 42,
        "decision_capture_id": "evt_0001",
        "rationale": rationale,
    }
    if source_chunk_id is not None:
        pair["source_chunk_id"] = source_chunk_id
    return pair


def _chunk(text: str = _CHUNK_TEXT) -> Dict[str, Any]:
    return {"id": "rdf_shacl_chunk_00001", "text": text}


def _validator(
    *,
    bloom_level: str = "remember",
    bloom_score: float = 0.85,
    embedder: Optional[Any] = None,
) -> TrainingPairPromotionValidator:
    """Build a validator with stub classifier + (optionally) stub embedder.

    Default keeps Bloom classifier returning the declared level so the
    Bloom-alignment criterion does NOT fire by default — most tests are
    isolating ONE rejection axis at a time.
    """
    return TrainingPairPromotionValidator(
        embedder=embedder if embedder is not None else _StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(
            level=bloom_level, score=bloom_score
        ),
    )


# --------------------------------------------------------------------- #
# Accept path
# --------------------------------------------------------------------- #


def test_accept_path_instruction_clean_pair():
    """Clean instruction pair sails through every criterion."""
    capture = _RecordingCapture()
    v = _validator()
    pair = _instruction_pair()
    chunk = _chunk()
    status, reason, new_fields = v.validate_pair(
        pair, kind="instruction", chunk=chunk, decision_capture=capture
    )
    assert status == "validated"
    assert reason is None
    assert new_fields["promotion_status"] == "validated"
    assert "answer_support_score" in new_fields
    assert new_fields["observed_bloom"] == "remember"
    assert new_fields["bloom_alignment"] is True
    assert isinstance(new_fields.get("rationale_richness_score"), float)
    # Exactly one decision-capture event per validate_pair call.
    assert len(capture.events) == 1
    assert capture.events[0]["decision_type"] == (
        "training_pair_promotion_check"
    )
    assert capture.events[0]["decision"] == "validated"


def test_accept_path_preference_clean_pair():
    """Clean preference pair sails through and stamps distractor_quality."""
    capture = _RecordingCapture()
    v = _validator(bloom_level="understand")
    pair = _preference_pair()
    chunk = _chunk()
    status, reason, new_fields = v.validate_pair(
        pair, kind="preference", chunk=chunk, decision_capture=capture
    )
    assert status == "validated"
    assert reason is None
    assert "distractor_quality" in new_fields
    assert "semantic_distinctness" in new_fields["distractor_quality"]
    assert (
        new_fields["distractor_quality"]["semantic_distinctness"]
        > DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS
    )
    assert len(capture.events) == 1


# --------------------------------------------------------------------- #
# Rejection criterion tests
# --------------------------------------------------------------------- #


def test_reject_placeholder_residue():
    """Criterion 1 — the 13 placeholder regexes from
    lib/validators/assessment.py fire on instruction/preference text."""
    capture = _RecordingCapture()
    v = _validator()
    pair = _instruction_pair(
        completion="Correct answer based on content of TO-01.",
    )
    status, reason, _ = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )
    assert status == "rejected"
    assert reason == "placeholder_residue"
    assert len(capture.events) == 1
    assert capture.events[0]["decision"] == (
        "rejected:placeholder_residue"
    )


def test_reject_source_free_generation():
    """Criterion 5 — empty / missing source_chunk_id rejects."""
    capture = _RecordingCapture()
    v = _validator()
    pair = _instruction_pair(source_chunk_id=None)
    status, reason, _ = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )
    assert status == "rejected"
    assert reason == "source_free_generation"
    assert len(capture.events) == 1


def test_reject_unanswerable_stem():
    """Criterion 4 — prompt has near-zero Jaccard overlap with chunk.

    The default ``min_prompt_chunk_jaccard`` floor was recalibrated to
    0.0 on 2026-06-09 (retiring the reject arm), so we construct the
    validator with an explicit positive floor to keep the *mechanism*
    pinned independent of the default.
    """
    capture = _RecordingCapture()
    v = TrainingPairPromotionValidator(
        embedder=_StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(),
        min_prompt_chunk_jaccard=0.05,
        min_answer_support_score=0.40,
        dpo_min_distractor_distinctness=0.40,
    )
    # Prompt about a topic alien to the RDF chunk.
    pair = _instruction_pair(
        prompt="zzz qqq xxx yyy unrelated foreign vocabulary tokens here.",
    )
    status, reason, _ = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )
    assert status == "rejected"
    assert reason == "unanswerable_stem"
    assert len(capture.events) == 1


def test_reject_unsupported_answer():
    """Criterion 2 — answer cosine sim with chunk below floor.

    With the stub embedder, an answer composed of tokens disjoint from
    the chunk yields cosine 0.0 < 0.40 floor. The stem still has SOME
    overlap with the chunk so unanswerable_stem doesn't trip first.
    """
    capture = _RecordingCapture()
    # Explicit floors pin the mechanism independent of the 2026-06-09
    # default recalibration (min_answer_support_score 0.40 → 0.10).
    v = TrainingPairPromotionValidator(
        embedder=_StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(),
        min_prompt_chunk_jaccard=0.05,
        min_answer_support_score=0.40,
        dpo_min_distractor_distinctness=0.40,
    )
    pair = _instruction_pair(
        prompt="Describe RDF triple subject predicate object.",
        completion=(
            "Foo bar baz qux quux corge grault garply waldo "
            "fred plugh xyzzy thud blarg fizzle hooble snarf."
        ),
    )
    status, reason, _ = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )
    assert status == "rejected"
    assert reason == "unsupported_answer"
    assert len(capture.events) == 1


def _authoritative_claim_pair(
    realizations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Canary-027-shaped complete staged proof with generic content."""
    values = realizations or [
        (
            "When objects travel in opposite directions, the distance "
            "between them increases by adding how far each has moved."
        )
    ]
    ids = [f"claim-{index}" for index in range(len(values))]
    chosen = " ".join(values)
    private_sha = "a" * 64
    pair = _preference_pair(
        prompt="Analyze the source-grounded relationship and explain it.",
        chosen=chosen,
        rejected="The quantities must always be subtracted.",
    )
    pair.update(
        {
            "projection_contract": "ed4all-dpo-preference.v2",
            "deps_missing": False,
            "claim_support_rate": 1.0,
            "claim_contradicted_rate": 0.0,
            "synthesis_plan_sha256": private_sha,
            "per_claim_support": [
                {
                    "sentence": value,
                    "outcome": "entailed",
                    "entailment": 0.79,
                    "contradiction": 0.15,
                }
                for value in values
            ],
            "provenance": {
                "source_chunk_id": pair["source_chunk_id"],
                "source_refs": ["generic-source#section"],
                "synthesis_plan_sha256": private_sha,
                "claim_realizations": dict(zip(ids, values)),
                "assembled_realization": {
                    "contract_version": (
                        "ed4all.micro-stage-d-claim-realizations.v3"
                    ),
                    "ordered_claim_ids": ids,
                    "item_sha256": [
                        hashlib.sha256(value.encode()).hexdigest()
                        for value in values
                    ],
                    "assembled_sha256": hashlib.sha256(
                        chosen.encode()
                    ).hexdigest(),
                    "private_artifact_sha256": private_sha,
                },
            },
        }
    )
    assembly = pair["provenance"]["assembled_realization"]
    assembly["provenance_sha256"] = hashlib.sha256(
        json.dumps(
            assembly, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    provenance = pair["provenance"]
    provenance["provenance_sha256"] = hashlib.sha256(
        json.dumps(
            provenance, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return pair


def test_canary027_complete_claim_proof_supersedes_only_low_lexical_support():
    capture = _RecordingCapture()
    pair = _authoritative_claim_pair()
    validator = TrainingPairPromotionValidator(
        embedder=None,
        bloom_classifier=_StubBloomEnsemble(level="understand", score=0.1),
        min_answer_support_score=0.40,
    )
    status, reason, fields = validator.validate_pair(
        pair, kind="preference", chunk=_chunk(), decision_capture=capture,
    )
    assert fields["answer_support_score"] < 0.40
    assert status == "validated"
    assert reason is None
    metrics = capture.events[0]["metrics"]
    assert metrics["answer_support_outcome"] == "below_floor_superseded"
    assert metrics["answer_support_authority"]["claim_count"] == 1
    assert len(
        metrics["answer_support_authority"]["proof_sha256"]
    ) == 64
    assert metrics["answer_support_authority"]["ordered_claim_ids"] == pair[
        "provenance"
    ]["assembled_realization"]["ordered_claim_ids"]
    assert metrics["answer_support_authority"]["assembled_sha256"] == pair[
        "provenance"
    ]["assembled_realization"]["assembled_sha256"]
    assert set(fields).isdisjoint(
        {"answer_support_outcome", "answer_support_authority"}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_proof",
        "unsupported_claim",
        "contradiction",
        "operator_polarity",
        "numeric_drift",
        "stale_item_hash",
        "stale_assembled_hash",
        "stale_private_hash",
        "source_mismatch",
        "missing_source_ref",
        "substituted_source_ref",
        "assembly_contract_drift",
        "nonfinite_entailment",
        "boolean_contradiction",
        "deps_missing",
    ],
)
def test_incomplete_or_tampered_claim_proof_cannot_supersede(
    mutation: str,
):
    pair = _authoritative_claim_pair()
    proof = pair["per_claim_support"][0]
    assembly = pair["provenance"]["assembled_realization"]
    if mutation == "missing_proof":
        pair.pop("per_claim_support")
    elif mutation == "unsupported_claim":
        proof["outcome"] = "unsupported"
        proof["entailment"] = 0.1
    elif mutation == "contradiction":
        proof["outcome"] = "contradicted"
        proof["contradiction"] = 0.8
        pair["claim_contradicted_rate"] = 1.0
    elif mutation in {"operator_polarity", "numeric_drift"}:
        suffix = (
            " The values must be subtracted."
            if mutation == "operator_polarity"
            else " The result is 47."
        )
        pair["chosen"] += suffix
    elif mutation == "stale_item_hash":
        assembly["item_sha256"][0] = "b" * 64
    elif mutation == "stale_assembled_hash":
        assembly["assembled_sha256"] = "b" * 64
    elif mutation == "stale_private_hash":
        pair["synthesis_plan_sha256"] = "b" * 64
    elif mutation == "source_mismatch":
        pair["provenance"]["source_chunk_id"] = "other-source"
    elif mutation == "missing_source_ref":
        pair["provenance"]["source_refs"] = []
    elif mutation == "substituted_source_ref":
        pair["provenance"]["source_refs"] = ["other-source#section"]
    elif mutation == "assembly_contract_drift":
        assembly["contract_version"] = "ed4all.micro-stage-d-claim-realizations.v2"
    elif mutation == "nonfinite_entailment":
        proof["entailment"] = float("nan")
    elif mutation == "boolean_contradiction":
        proof["contradiction"] = False
    elif mutation == "deps_missing":
        pair["deps_missing"] = True
    validator = TrainingPairPromotionValidator(
        embedder=None,
        bloom_classifier=_StubBloomEnsemble(level="understand", score=0.1),
        min_answer_support_score=0.40,
    )
    status, reason, fields = validator.validate_pair(
        pair, kind="preference", chunk=_chunk(),
    )
    assert fields["answer_support_score"] < 0.40
    assert (status, reason) == ("rejected", "unsupported_answer")
    assert fields["promotion_status"] == "rejected"


def test_complete_multi_sentence_claim_mapping_is_required_in_order():
    pair = _authoritative_claim_pair(
        [
            "A grounded first relation is established.",
            "A grounded second relation follows from it.",
        ]
    )
    validator = TrainingPairPromotionValidator(
        embedder=None,
        bloom_classifier=_StubBloomEnsemble(level="understand", score=0.1),
        min_answer_support_score=0.99,
    )
    capture = _RecordingCapture()
    status, reason, fields = validator.validate_pair(
        pair, kind="preference", chunk=_chunk(), decision_capture=capture,
    )
    assert status == "validated"
    assert reason is None
    assert capture.events[0]["metrics"]["answer_support_authority"][
        "claim_count"
    ] == 2

    pair["per_claim_support"].reverse()
    capture = _RecordingCapture()
    status, reason, fields = validator.validate_pair(
        pair, kind="preference", chunk=_chunk(), decision_capture=capture,
    )
    assert capture.events[0]["metrics"]["answer_support_authority"] is None
    assert (status, reason) == ("rejected", "unsupported_answer")


def test_post_write_audit_recomputes_superseding_authority(tmp_path: Path):
    pair = _authoritative_claim_pair()
    validator = TrainingPairPromotionValidator(
        embedder=None,
        bloom_classifier=_StubBloomEnsemble(level="understand", score=0.1),
        min_answer_support_score=0.40,
    )
    status, reason, fields = validator.validate_pair(
        pair, kind="preference", chunk=_chunk(),
    )
    assert (status, reason) == ("validated", None)
    pair.update(fields)
    pair["promotion_status"] = status
    path = tmp_path / "preference_pairs.jsonl"
    path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    result = validator.validate({"preference_pairs_path": str(path)})
    assert result.passed is True

    pair["provenance"]["assembled_realization"]["item_sha256"][0] = "f" * 64
    path.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    result = validator.validate({"preference_pairs_path": str(path)})
    assert result.passed is False
    assert result.issues[0].code == "INVALID_ANSWER_SUPPORT_AUTHORITY"


def test_promotion_preserves_complete_objective_bloom_projection():
    pair = _authoritative_claim_pair()
    pair["bloom_level"] = "analyze"
    pair["lo_refs"] = ["co-generic"]
    pair["bloom_alignment"] = None
    pair["pair_objective_alignment_pass_rate"] = 1.0
    pair["pair_objective_alignment"] = [{
        "objective_id": "co-generic",
        "status": "delivered",
        "statement_entailment_score": 0.78,
        "contradiction_score": 0.10,
        "bloom_gap": 0,
        "verb_match_count": 1,
        "declared_bloom": "analyze",
        "observed_bloom": "analyze",
        "entailment_threshold": 0.45,
    }]
    validator = TrainingPairPromotionValidator(
        embedder=None,
        bloom_classifier=_StubBloomEnsemble(level="apply", score=0.1),
    )
    status, reason, fields = validator.validate_pair(
        pair, kind="preference", chunk=_chunk(),
    )
    assert (status, reason) == ("validated", None)
    assert fields["promotion_status"] == "validated"
    assert fields["observed_bloom"] == "analyze"
    assert fields["bloom_alignment"] is None


def test_reject_weak_distractor():
    """Criterion 3 — chosen / rejected too close (preference only).

    Explicit floor (0.40) pins the mechanism independent of the
    2026-06-09 default recalibration (0.40 → 0.05).
    """
    capture = _RecordingCapture()
    _explicit_floor = 0.40
    v = TrainingPairPromotionValidator(
        embedder=_StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(level="understand", score=0.85),
        min_prompt_chunk_jaccard=0.05,
        min_answer_support_score=0.40,
        dpo_min_distractor_distinctness=_explicit_floor,
    )
    # Chosen and rejected are paraphrases — same tokens, same intent.
    pair = _preference_pair(
        chosen=(
            "An RDF triple has subject predicate object as its "
            "declarative graph data structure W3C standard form."
        ),
        rejected=(
            "An RDF triple has subject predicate object as its "
            "declarative graph data structure W3C standard form."
        ),
    )
    status, reason, new_fields = v.validate_pair(
        pair, kind="preference", chunk=_chunk(),
        decision_capture=capture,
    )
    assert status == "rejected"
    assert reason == "weak_distractor"
    assert (
        new_fields["distractor_quality"]["semantic_distinctness"]
        < _explicit_floor
    )
    assert len(capture.events) == 1


def test_low_bloom_alignment_reject_arm_is_retired_by_default():
    """Criterion 6 — clear disagreement, high confidence, still NO reject.

    The backing Bloom head is not trustworthy enough to discard training
    data (the BERT ensemble degrades to the ``unknown`` sentinel, so
    classification falls through to the DeBERTa zero-shot head, which is
    itself pending replacement). The default confidence floor sits above
    1.0 so the strict ``>`` can never fire — but the SIGNAL is still
    stamped, which is the whole point of retiring rather than deleting.
    """
    capture = _RecordingCapture()
    # Stub classifier observes "remember", far below the declared "create",
    # at a confidence that would have rejected under the old 0.40 floor.
    v = TrainingPairPromotionValidator(
        embedder=_StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(level="remember", score=0.80),
    )
    pair = _instruction_pair(bloom_level="create")
    status, reason, new_fields = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )
    assert status == "validated"
    assert reason is None
    # Audit stamps survive the retirement.
    assert new_fields["observed_bloom"] == "remember"
    assert new_fields["bloom_alignment"] is False
    assert len(capture.events) == 1


def test_low_bloom_alignment_reject_arm_is_re_armable():
    """Retired, not deleted: an explicit floor in [0, 1] restores the reject.

    Guards the path back — once a purpose-trained Bloom classifier lands,
    re-arming must be a threshold change, not a code change.
    """
    capture = _RecordingCapture()
    v = TrainingPairPromotionValidator(
        embedder=_StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(level="remember", score=0.80),
        bloom_alignment_min_confidence=0.40,
    )
    pair = _instruction_pair(bloom_level="create")
    status, reason, new_fields = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )
    assert status == "rejected"
    assert reason == "low_bloom_alignment"
    assert new_fields["observed_bloom"] == "remember"
    assert new_fields["bloom_alignment"] is False


def test_low_bloom_below_confidence_floor_does_not_reject():
    """Criterion 6 — disagreement BUT low winner score ⇒ no reject.

    Pins the contract that low-confidence ensemble votes are noise and
    don't override the declared bloom level.
    """
    capture = _RecordingCapture()
    v = TrainingPairPromotionValidator(
        embedder=_StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(level="remember", score=0.30),
    )
    pair = _instruction_pair(bloom_level="create")
    status, reason, new_fields = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )
    assert status == "validated"
    assert reason is None
    # Stamps land regardless of pass/fail.
    assert new_fields["observed_bloom"] == "remember"
    assert new_fields["bloom_alignment"] is None
    assert len(capture.events) == 1


def test_higher_observed_bloom_satisfies_declared_minimum():
    """Evaluate-level delivery satisfies an apply-level objective."""
    capture = _RecordingCapture()
    v = TrainingPairPromotionValidator(
        embedder=_StubEmbedder(),
        bloom_classifier=_StubBloomEnsemble(level="evaluate", score=0.95),
    )
    pair = _instruction_pair(bloom_level="apply")
    status, reason, new_fields = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )

    assert status == "validated"
    assert reason is None
    assert new_fields["observed_bloom"] == "evaluate"
    assert new_fields["bloom_alignment"] is True


def test_reject_generic_rationale():
    """Criterion 7 — repetitive rationale below richness floor."""
    capture = _RecordingCapture()
    v = _validator()
    # Same single content token repeated; richness ≈ 1/N where N = N
    # repeats. With 20 repeats, richness ≈ 0.05 (well below 0.30 floor).
    pair = _instruction_pair(
        rationale=(
            "because because because because because because because "
            "because because because because because because because "
            "because because because because because because"
        ),
    )
    status, reason, new_fields = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )
    assert status == "rejected"
    assert reason == "generic_rationale"
    assert new_fields["rationale_richness_score"] < (
        DEFAULT_MIN_RATIONALE_RICHNESS_SCORE
    )
    assert len(capture.events) == 1


# --------------------------------------------------------------------- #
# Decision-capture-emit regression: every return path fires exactly one
# event (Wave 2 plan Test 1).
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "scenario",
    [
        "accept_instruction",
        "accept_preference",
        "placeholder_residue",
        "source_free_generation",
        "unanswerable_stem",
        "unsupported_answer",
        "weak_distractor",
        "low_bloom_alignment",
        "generic_rationale",
    ],
)
def test_every_return_path_emits_exactly_one_decision(scenario: str):
    """Per W2 plan Test 1 — capture-wiring regression."""
    capture = _RecordingCapture()

    # Build per-scenario inputs.
    if scenario == "accept_instruction":
        v = _validator()
        v.validate_pair(
            _instruction_pair(),
            kind="instruction", chunk=_chunk(),
            decision_capture=capture,
        )
    elif scenario == "accept_preference":
        v = _validator(bloom_level="understand")
        v.validate_pair(
            _preference_pair(),
            kind="preference", chunk=_chunk(),
            decision_capture=capture,
        )
    elif scenario == "placeholder_residue":
        v = _validator()
        v.validate_pair(
            _instruction_pair(
                completion="Correct answer based on content of TO-01."
            ),
            kind="instruction", chunk=_chunk(),
            decision_capture=capture,
        )
    elif scenario == "source_free_generation":
        v = _validator()
        v.validate_pair(
            _instruction_pair(source_chunk_id=None),
            kind="instruction", chunk=_chunk(),
            decision_capture=capture,
        )
    elif scenario == "unanswerable_stem":
        # Explicit positive floor keeps the reject arm armed despite the
        # 2026-06-09 default recalibration to 0.0.
        v = TrainingPairPromotionValidator(
            embedder=_StubEmbedder(),
            bloom_classifier=_StubBloomEnsemble(),
            min_prompt_chunk_jaccard=0.05,
            min_answer_support_score=0.40,
            dpo_min_distractor_distinctness=0.40,
        )
        v.validate_pair(
            _instruction_pair(prompt="zzz qqq xxx yyy alien tokens."),
            kind="instruction", chunk=_chunk(),
            decision_capture=capture,
        )
    elif scenario == "unsupported_answer":
        v = _validator()
        v.validate_pair(
            _instruction_pair(
                prompt="Describe RDF triple subject predicate object.",
                completion=(
                    "Foo bar baz qux quux corge grault garply waldo "
                    "fred plugh xyzzy thud blarg fizzle hooble snarf."
                ),
            ),
            kind="instruction", chunk=_chunk(),
            decision_capture=capture,
        )
    elif scenario == "weak_distractor":
        v = _validator(bloom_level="understand")
        identical = (
            "An RDF triple has subject predicate object as its "
            "declarative graph data structure W3C standard form."
        )
        v.validate_pair(
            _preference_pair(chosen=identical, rejected=identical),
            kind="preference", chunk=_chunk(),
            decision_capture=capture,
        )
    elif scenario == "low_bloom_alignment":
        v = TrainingPairPromotionValidator(
            embedder=_StubEmbedder(),
            bloom_classifier=_StubBloomEnsemble(
                level="create", score=0.80,
            ),
        )
        v.validate_pair(
            _instruction_pair(bloom_level="remember"),
            kind="instruction", chunk=_chunk(),
            decision_capture=capture,
        )
    elif scenario == "generic_rationale":
        v = _validator()
        v.validate_pair(
            _instruction_pair(
                rationale=(
                    "because " * 20
                ),
            ),
            kind="instruction", chunk=_chunk(),
            decision_capture=capture,
        )
    else:
        raise AssertionError(f"unknown scenario {scenario!r}")

    # Hard contract: ONE event per validate_pair invocation.
    assert len(capture.events) == 1, (
        f"scenario={scenario}: expected 1 event, got "
        f"{len(capture.events)}: {capture.events}"
    )
    assert capture.events[0]["decision_type"] == (
        "training_pair_promotion_check"
    )


# --------------------------------------------------------------------- #
# Embedding-deps-missing fallback + strict-mode
# --------------------------------------------------------------------- #


def test_embedding_extras_absent_falls_back_to_jaccard(monkeypatch):
    """When try_load_embedder returns None and strict mode is OFF, the
    validator runs the cosine criteria via the Jaccard surrogate. A
    pair that paraphrases the chunk passes; a pair that doesn't, fails.
    The audit metrics carry fallback_to_jaccard=True so the silent
    degrade lands in the capture stream.
    """
    # Force the loader to None.
    import lib.validators.training_pair_promotion as mod

    def _stub_loader():
        return None

    monkeypatch.setattr(
        "lib.embedding.sentence_embedder.try_load_embedder",
        _stub_loader,
        raising=True,
    )
    # Ensure strict mode is OFF.
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)

    capture = _RecordingCapture()
    # No embedder injection — validator must lazy-load and get None.
    v = TrainingPairPromotionValidator(
        embedder=None,
        bloom_classifier=_StubBloomEnsemble(),
    )
    pair = _instruction_pair()
    status, reason, _ = v.validate_pair(
        pair, kind="instruction", chunk=_chunk(),
        decision_capture=capture,
    )
    # Pair shares enough tokens with chunk to clear the Jaccard fallback
    # against the same 0.40 floor as the cosine path.
    assert status == "validated"
    assert reason is None
    assert len(capture.events) == 1
    metrics = capture.events[0].get("metrics") or {}
    assert metrics.get("fallback_to_jaccard") is True


def test_strict_mode_propagates_loader_error(monkeypatch):
    """TRAINFORGE_REQUIRE_EMBEDDINGS=true makes the loader's
    EmbeddingDepsMissing exception propagate, mirroring the
    ObjectiveAssessmentSimilarityValidator strict-mode contract."""
    from lib.embedding.sentence_embedder import EmbeddingDepsMissing

    def _strict_loader():
        raise EmbeddingDepsMissing(
            "sentence-transformers extras not installed (test stub)"
        )

    monkeypatch.setattr(
        "lib.embedding.sentence_embedder.try_load_embedder",
        _strict_loader,
        raising=True,
    )
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    v = TrainingPairPromotionValidator(
        embedder=None,
        bloom_classifier=_StubBloomEnsemble(),
    )
    with pytest.raises(EmbeddingDepsMissing):
        v.validate_pair(
            _instruction_pair(),
            kind="instruction", chunk=_chunk(),
            decision_capture=None,
        )


# --------------------------------------------------------------------- #
# On-disk gate path
# --------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def test_validate_passes_on_stamped_pairs(tmp_path: Path):
    """Every pair on disk carries promotion_status ⇒ passed=True."""
    course_dir = tmp_path / "course"
    ts = course_dir / "training_specs"
    _write_jsonl(
        ts / "instruction_pairs.jsonl",
        [
            {**_instruction_pair(), "promotion_status": "validated"},
            {**_instruction_pair(), "promotion_status": "validated"},
        ],
    )
    _write_jsonl(
        ts / "preference_pairs.jsonl",
        [
            {**_preference_pair(), "promotion_status": "validated"},
        ],
    )
    v = TrainingPairPromotionValidator()
    result = v.validate({"course_dir": str(course_dir)})
    assert result.passed is True
    assert result.action is None


def test_validate_fails_on_unstamped_pairs(tmp_path: Path):
    """A pair on disk WITHOUT promotion_status ⇒ passed=False."""
    course_dir = tmp_path / "course"
    ts = course_dir / "training_specs"
    rows: List[Dict[str, Any]] = [
        {**_instruction_pair(), "promotion_status": "validated"},
        _instruction_pair(),  # no promotion_status
    ]
    _write_jsonl(ts / "instruction_pairs.jsonl", rows)
    v = TrainingPairPromotionValidator()
    result = v.validate({"course_dir": str(course_dir)})
    assert result.passed is False
    assert result.action == "block"
    codes = {issue.code for issue in result.issues}
    assert "MISSING_PROMOTION_STATUS" in codes


def test_validate_missing_inputs():
    """No course_dir / training_specs_dir / explicit paths ⇒ MISSING_INPUTS."""
    v = TrainingPairPromotionValidator()
    result = v.validate({})
    assert result.passed is False
    codes = {issue.code for issue in result.issues}
    assert "MISSING_INPUTS" in codes


# --------------------------------------------------------------------- #
# Constructor smoke — defaults preserved
# --------------------------------------------------------------------- #


def test_constructor_default_thresholds():
    """Pin the four default thresholds so a re-tune is a deliberate
    edit."""
    v = TrainingPairPromotionValidator()
    assert v._min_answer_support_score == DEFAULT_MIN_ANSWER_SUPPORT_SCORE
    assert v._dpo_min_distractor_distinctness == (
        DEFAULT_DPO_MIN_DISTRACTOR_DISTINCTNESS
    )
    assert v._min_prompt_chunk_jaccard == DEFAULT_MIN_PROMPT_CHUNK_JACCARD
    assert v._min_rationale_richness_score == (
        DEFAULT_MIN_RATIONALE_RICHNESS_SCORE
    )


# --------------------------------------------------------------------- #
# Real-corpus recalibration regression (2026-06-09, RDF/SHACL calibration corpus)
# --------------------------------------------------------------------- #


def test_real_shaped_pair_passes_under_recalibrated_defaults():
    """A real-shaped preference pair PASSES under the 2026-06-09
    recalibrated defaults but would have been rejected under the old
    (0.05 / 0.40 / 0.40) floors.

    Real corpus signal (measured on the RDF/SHACL calibration corpus
    training_specs, n=417 instruction + 262 preference pairs):

    - prompt is an on-topic paraphrase deliberately reworded away from
      the source → ``prompt_chunk_jaccard`` ≈ 0.0 (old 0.05 floor
      rejected it as ``unanswerable_stem``; new 0.0 floor retires the
      reject arm so this is audit-stamp-only).
    - the answer is grounded but reworded → ``answer_support_score`` ≈
      0.16 cosine (old 0.40 floor → ``unsupported_answer``; new 0.10
      floor accepts).
    - the rejected distractor is deliberately plausible /
      semantically close → ``distractor_distinctness`` ≈ 0.07 (old 0.40
      floor → ``weak_distractor``; new 0.05 floor accepts).

    With a fixed-vocab stub embedder injected (no [embedding] extras
    needed), the cosine path actually computes (the lazily-growing
    ``_StubEmbedder`` would fail over to Jaccard and judge the sparser
    signal against the cosine floor) and the pair clears all three
    recalibrated criteria: support cosine ≈ 0.16, distinctness
    ``1 - cos(chosen, rejected)`` ≈ 0.07.
    """
    capture = _RecordingCapture()
    prompt = (
        "Could you explain how the smallest standalone statement "
        "unit is assembled?"
    )
    chosen = (
        "Each standalone statement links one entity to another "
        "through a labelled relationship, asserting a single fact "
        "within a declarative graph structure."
    )
    rejected = (
        "Each standalone statement links one entity to another via "
        "a labelled relationship, asserting a single fact inside a "
        "declarative graph structure."
    )
    v = TrainingPairPromotionValidator(
        embedder=_FixedVocabEmbedder(
            [prompt, chosen, rejected, _CHUNK_TEXT]
        ),
        bloom_classifier=_StubBloomEnsemble(level="understand", score=0.85),
    )
    pair = _preference_pair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        bloom_level="understand",
    )
    chunk = _chunk()
    status, reason, new_fields = v.validate_pair(
        pair, kind="preference", chunk=chunk, decision_capture=capture,
    )
    assert status == "validated", (
        f"expected pass under recalibrated defaults, got "
        f"rejected:{reason}; new_fields={new_fields}"
    )
    assert reason is None
    # The three recalibrated signals land in the band [old_floor, new_floor):
    # each is below the pre-2026-06-09 floor yet at/above the new one.
    assert new_fields["answer_support_score"] < 0.40
    assert new_fields["answer_support_score"] >= 0.10
    distinctness = new_fields["distractor_quality"]["semantic_distinctness"]
    assert distinctness < 0.40
    assert distinctness >= 0.05
    assert len(capture.events) == 1
    metrics = capture.events[0].get("metrics") or {}
    # Audit stamp survives even though criterion 4's reject arm is retired.
    assert metrics.get("prompt_chunk_jaccard") is not None
