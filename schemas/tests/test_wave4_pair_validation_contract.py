"""GPT Feedback v2 — Wave 4 end-of-wave pair-validation contract gate.

Authored 2026-05-07 against the closing 6-test gate enumerated in
``plans/trainforge-symmetric-validation-investigation-2026-05-06.md`` § 6.

Predecessors landed:

* W4.A — :class:`lib.validators.pair_claim_support.PairClaimSupportValidator`
  (per-pair-per-claim NLI entailment audit on the cited chunk text).
* W4.B — :class:`lib.validators.pair_lo_refs.PairLearningOutcomeRefsValidator`
  (per-pair LO-refs subset gate against the cited chunk's
  ``learning_outcome_refs[]``).
* W4.D (this gate's sibling) — pair-schema migrations on
  ``schemas/knowledge/instruction_pair{,.strict}.schema.json`` and
  ``schemas/knowledge/preference_pair.schema.json`` adding the four
  new optional fields ``per_claim_support`` / ``claim_support_rate``
  / ``claim_contradicted_rate`` / ``pair_lo_resolution``. Mirrors the
  W2.E ``promotion_status`` migration shape (optional day-1, may
  upgrade to required-conditional once the validator has rolled
  across every active corpus).

This file is a SELF-CONTAINED contract test mirroring the Wave 1.5,
1.6, 1.7 predecessor gates (``test_wave15_per_claim_attribution_contract.py``,
``test_wave16_per_objective_attribution_contract.py``,
``test_wave17_block_against_objective_contract.py``). Every fixture is
inlined locally so a future rename of a per-worker test file cannot
silently disable this wave-end gate.

Tests:

* ``test_pair_claim_support_validator_entails_passes`` — feed a
  3-sentence completion where every sentence is verbatim-supported
  by the chunk. Stub NLI returning entailment=0.85 / contradiction=0.05
  for every pair. Assert ``(status, reason) == ("validated", None)``.
  Assert ``new_fields["claim_support_rate"] == 1.0``.
* ``test_pair_claim_support_validator_rejects_unsupported_claim`` —
  feed a 4-sentence completion where 1 sentence is hallucinated.
  Stub NLI returns entailment=0.20 / contradiction=0.10 for the
  hallucinated one, entailment=0.85 / contradiction=0.05 for the
  others. With ``max_unsupported_claim_rate=0.20`` (default), assert
  ``(status, reason) == ("rejected", "unsupported_claim")``
  (1/4 = 0.25 > 0.20).
* ``test_pair_claim_support_validator_rejects_contradicted_claim`` —
  feed 3 sentences with 1 contradicted (entailment=0.10 /
  contradiction=0.80). Assert ``(status, reason) ==
  ("rejected", "contradicted_claim")`` — contradicted-rate ceiling
  fires before unsupported-rate ceiling.
* ``test_pair_lo_refs_validator_subset_passes`` — pair carries
  ``learning_outcome_refs=["TO-02"]``, chunk carries
  ``["TO-02", "TO-07"]``. Assert ``(status, reason) ==
  ("validated", None)``. Assert
  ``new_fields["pair_lo_resolution"]["phantom_los"] == []``.
* ``test_pair_lo_refs_validator_phantom_rejects`` — pair carries
  ``["TO-05"]``, chunk carries ``["TO-02", "TO-07"]``. Assert
  ``(status, reason) == ("rejected", "phantom_pair_lo_refs")``.
  Assert
  ``new_fields["pair_lo_resolution"]["phantom_los"] == ["TO-05"]``
  (preserving emit case).
* ``test_wave4_decision_capture_wiring`` — instantiate both
  validators with a stub :class:`DecisionCapture`, run one pass +
  one fail per validator, assert exactly 4 events (2 per validator),
  assert each event's ``decision_type`` is one of
  ``{"pair_claim_support_check", "pair_lo_refs_check"}``, assert each
  rationale is ≥20 chars + interpolates ``chunk_id`` (W4.A) /
  ``pair_id`` (W4.B; falls back to chunk_id when no pair_id minted)
  AND ``pair_kind`` / ``kind`` AND the dynamic per-call values that
  drove the verdict.

Optional Test 7 (recommended, not part of the 6-test contract):

* ``test_wave4_strict_pair_schema_round_trip`` — emit a pair through
  ``PairClaimSupportValidator.validate_pair`` + the strict-schema
  validator, assert no schema error on the resulting pair.

Drift notes vs plan §6:

* Plan §6 suggests writing the test file at
  ``lib/validators/tests/test_wave4_pair_validation_contract.py`` but
  this file lives at ``schemas/tests/`` for consistency with the
  Wave 1.5 / 1.6 / 1.7 predecessor gates. Same import topology and
  ratchet semantics.
* The W4.A validator's rationale string interpolates ``pair_kind``;
  the W4.B validator's rationale string interpolates ``kind``. The
  decision-capture wiring test (Test 6) accepts either token because
  both surface the dynamic per-call kind value as the plan §6 contract
  requires.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Helpers — stub NLI classifier + DecisionCapture stub
# --------------------------------------------------------------------------- #


class _StubNliScore:
    """Minimal NliScore-shaped object exposing the three probability
    attributes the W4.A consumer reads.

    Inlined so the test file doesn't import the production
    :class:`NliScore` (which would couple the gate to that class's
    evolution). Mirrors the Wave 2 W2.F precedent of stubbing the NLI
    surface for tests.
    """

    def __init__(
        self,
        *,
        entailment: float,
        neutral: float,
        contradiction: float,
    ) -> None:
        self.entailment = float(entailment)
        self.neutral = float(neutral)
        self.contradiction = float(contradiction)


class _StubNliClassifier:
    """Stub :class:`NliClassifier` that returns canned per-pair scores.

    Constructor accepts a callable ``score_fn(premise, hypothesis) ->
    (entailment, neutral, contradiction)`` so every test can shape the
    NLI verdict without instantiating the real DeBERTa-v3 model.

    The W4.A validator calls ``nli.score_batch(pairs=[...])`` exactly
    once per :meth:`validate_pair` invocation; the stub honours that
    surface contract.
    """

    def __init__(self, *, score_fn) -> None:
        self._score_fn = score_fn

    def score_batch(self, *, pairs, batch_size=None):
        out: List[_StubNliScore] = []
        for premise, hypothesis in pairs:
            ent, neu, con = self._score_fn(premise, hypothesis)
            out.append(
                _StubNliScore(
                    entailment=ent, neutral=neu, contradiction=con,
                )
            )
        return out

    def score_pair(self, *, premise, hypothesis):
        return self.score_batch(pairs=[(premise, hypothesis)])[0]


class _RecordingCapture:
    """Minimal capture stub recording every ``log_decision`` payload.

    Inlined so a rename of any per-worker capture helper can't silently
    break this gate. Mirrors the
    ``test_wave15_per_claim_attribution_contract.py::_RecordingCapture``
    pattern (attribute name preserved for cross-wave grep symmetry).
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #


# Three-sentence chunk text — premise body for NLI scoring. The
# completion sentences in the support-passes test are verbatim-derived
# from this text so the canned NLI stub (which doesn't actually look at
# content) returns its canned entailment scores against a real chunk.
_CHUNK_TEXT_THREE_FACT = (
    "RDF triples are the smallest unit of structured knowledge. "
    "Each triple links a subject to an object via a predicate. "
    "Subjects, predicates, and objects are all IRIs."
)

# Four-sentence chunk-text body for the "1-of-4 hallucinated" fixture.
_CHUNK_TEXT_FOUR_FACT = (
    "RDF triples are the smallest unit of structured knowledge. "
    "Each triple links a subject to an object via a predicate. "
    "Subjects, predicates, and objects are all IRIs. "
    "RDF graphs compose triples into larger knowledge bases."
)

# Three-sentence chunk-text body for the "1-of-3 contradicted" fixture.
_CHUNK_TEXT_FOR_CONTRADICTION = (
    "RDF triples are the smallest unit of structured knowledge. "
    "Each triple links a subject to an object via a predicate. "
    "Subjects, predicates, and objects are all IRIs."
)


def _instruction_pair(
    *,
    completion: str,
    chunk_id: str = "ch1#sec1",
    learning_outcome_refs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a minimally-valid instruction pair dict.

    Includes only the fields the W4.A / W4.B per-pair filter actually
    reads (``completion``, ``chunk_id``, ``learning_outcome_refs``).
    Other schema-required fields like ``prompt`` / ``seed`` /
    ``decision_capture_id`` are not under test here — the per-pair
    filter doesn't read them.
    """
    pair: Dict[str, Any] = {
        "completion": completion,
        "chunk_id": chunk_id,
    }
    if learning_outcome_refs is not None:
        pair["learning_outcome_refs"] = list(learning_outcome_refs)
    return pair


def _chunk_payload(
    *,
    text: str,
    learning_outcome_refs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a minimally-valid chunk dict (the W4.A / W4.B chunk arg)."""
    chunk: Dict[str, Any] = {"text": text}
    if learning_outcome_refs is not None:
        chunk["learning_outcome_refs"] = list(learning_outcome_refs)
    return chunk


# --------------------------------------------------------------------------- #
# Test 1 — pair_claim_support: every-sentence-entailed passes
# --------------------------------------------------------------------------- #


def test_pair_claim_support_validator_entails_passes() -> None:
    """Plan §6 Test 1 — 3 sentences, every one entailed (≥0.70)
    against the chunk, MUST pass with claim_support_rate == 1.0.
    """
    from lib.validators.pair_claim_support import PairClaimSupportValidator

    completion = (
        "RDF triples are the smallest unit of structured knowledge. "
        "Each triple links a subject to an object via a predicate. "
        "Subjects, predicates, and objects are all IRIs."
    )
    pair = _instruction_pair(completion=completion)
    chunk = _chunk_payload(text=_CHUNK_TEXT_THREE_FACT)

    # Stub NLI returns canned (0.85, 0.10, 0.05) for every pair —
    # entailment ≥ 0.70 → "entailed"; contradiction = 0.05 < 0.50.
    stub = _StubNliClassifier(
        score_fn=lambda premise, hypothesis: (0.85, 0.10, 0.05),
    )
    validator = PairClaimSupportValidator(nli=stub)

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk,
    )

    assert (status, reason) == ("validated", None), (
        f"expected (validated, None); got ({status!r}, {reason!r})"
    )
    assert new_fields["claim_support_rate"] == 1.0, (
        "every-sentence-entailed fixture MUST yield "
        f"claim_support_rate=1.0; got {new_fields['claim_support_rate']!r}"
    )
    # Defence-in-depth: per_claim_support carries 3 entries, all "entailed".
    pcs = new_fields["per_claim_support"]
    assert isinstance(pcs, list) and len(pcs) == 3, (
        f"expected 3 per_claim_support entries; got {pcs!r}"
    )
    outcomes = [entry["outcome"] for entry in pcs]
    assert outcomes == ["entailed", "entailed", "entailed"], (
        f"expected every outcome=entailed; got {outcomes!r}"
    )
    # Contradicted-rate is the canonical 0.0.
    assert new_fields["claim_contradicted_rate"] == 0.0, (
        "all-entailed fixture MUST yield claim_contradicted_rate=0.0; "
        f"got {new_fields['claim_contradicted_rate']!r}"
    )


# --------------------------------------------------------------------------- #
# Test 2 — pair_claim_support: 1-of-4 hallucinated rejects (unsupported_claim)
# --------------------------------------------------------------------------- #


def test_pair_claim_support_validator_rejects_unsupported_claim() -> None:
    """Plan §6 Test 2 — 4 sentences, 1 hallucinated → unsupported_rate
    = 1/4 = 0.25 > 0.20 ceiling → reject with 'unsupported_claim'.
    """
    from lib.validators.pair_claim_support import PairClaimSupportValidator

    # Marker-string for the hallucinated sentence — the stub keys off
    # this substring so the canned NLI returns the unsupported scores
    # for exactly one of the four sentences.
    hallucinated = (
        "Hallucinated content claiming RDF graphs are mutable databases."
    )
    completion = (
        "RDF triples are the smallest unit of structured knowledge. "
        "Each triple links a subject to an object via a predicate. "
        f"{hallucinated} "
        "RDF graphs compose triples into larger knowledge bases."
    )
    pair = _instruction_pair(completion=completion)
    chunk = _chunk_payload(text=_CHUNK_TEXT_FOUR_FACT)

    def _score(premise, hypothesis):
        if "Hallucinated" in hypothesis:
            return (0.20, 0.70, 0.10)  # unsupported (ent<0.70, con<0.50)
        return (0.85, 0.10, 0.05)  # entailed

    stub = _StubNliClassifier(score_fn=_score)
    validator = PairClaimSupportValidator(nli=stub)

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk,
    )

    assert (status, reason) == ("rejected", "unsupported_claim"), (
        f"expected (rejected, unsupported_claim); "
        f"got ({status!r}, {reason!r}); new_fields={new_fields!r}"
    )
    # Defence-in-depth: 1 unsupported, 3 entailed, 0 contradicted.
    pcs = new_fields["per_claim_support"]
    assert isinstance(pcs, list) and len(pcs) == 4, (
        f"expected 4 per_claim_support entries; got {pcs!r}"
    )
    counts = {"entailed": 0, "unsupported": 0, "contradicted": 0}
    for entry in pcs:
        counts[entry["outcome"]] += 1
    assert counts == {"entailed": 3, "unsupported": 1, "contradicted": 0}, (
        f"expected 3 entailed / 1 unsupported / 0 contradicted; "
        f"got {counts!r}"
    )
    # claim_support_rate = entailed/total = 0.75; contradicted_rate = 0.
    assert new_fields["claim_support_rate"] == pytest.approx(0.75), (
        f"expected claim_support_rate≈0.75; "
        f"got {new_fields['claim_support_rate']!r}"
    )
    assert new_fields["claim_contradicted_rate"] == pytest.approx(0.0), (
        f"expected claim_contradicted_rate=0.0; "
        f"got {new_fields['claim_contradicted_rate']!r}"
    )


# --------------------------------------------------------------------------- #
# Test 3 — pair_claim_support: 1-of-3 contradicted rejects (contradicted_claim)
# --------------------------------------------------------------------------- #


def test_pair_claim_support_validator_rejects_contradicted_claim() -> None:
    """Plan §6 Test 3 — 3 sentences, 1 contradicted (con=0.80 ≥ 0.50)
    → contradicted_rate = 1/3 = 0.33 > 0.05 ceiling → reject with
    'contradicted_claim'. Contradicted-rate ceiling fires BEFORE
    unsupported-rate ceiling per the W4.A reject precedence.
    """
    from lib.validators.pair_claim_support import PairClaimSupportValidator

    contradicted_marker = (
        "RDF triples are mutable, six-element data structures."
    )
    completion = (
        "RDF triples are the smallest unit of structured knowledge. "
        f"{contradicted_marker} "
        "Subjects, predicates, and objects are all IRIs."
    )
    pair = _instruction_pair(completion=completion)
    chunk = _chunk_payload(text=_CHUNK_TEXT_FOR_CONTRADICTION)

    def _score(premise, hypothesis):
        if "mutable" in hypothesis:
            return (0.10, 0.10, 0.80)  # contradicted (con ≥ 0.50)
        return (0.85, 0.10, 0.05)

    stub = _StubNliClassifier(score_fn=_score)
    validator = PairClaimSupportValidator(nli=stub)

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk,
    )

    assert (status, reason) == ("rejected", "contradicted_claim"), (
        f"expected (rejected, contradicted_claim) — contradicted-rate "
        f"ceiling MUST fire before unsupported-rate ceiling; "
        f"got ({status!r}, {reason!r}); new_fields={new_fields!r}"
    )
    pcs = new_fields["per_claim_support"]
    assert isinstance(pcs, list) and len(pcs) == 3, (
        f"expected 3 per_claim_support entries; got {pcs!r}"
    )
    counts = {"entailed": 0, "unsupported": 0, "contradicted": 0}
    for entry in pcs:
        counts[entry["outcome"]] += 1
    assert counts == {"entailed": 2, "unsupported": 0, "contradicted": 1}, (
        f"expected 2 entailed / 0 unsupported / 1 contradicted; "
        f"got {counts!r}"
    )
    assert new_fields["claim_contradicted_rate"] == pytest.approx(1 / 3), (
        f"expected claim_contradicted_rate≈0.333; "
        f"got {new_fields['claim_contradicted_rate']!r}"
    )


# --------------------------------------------------------------------------- #
# Test 4 — pair_lo_refs: subset passes
# --------------------------------------------------------------------------- #


def test_pair_lo_refs_validator_subset_passes() -> None:
    """Plan §6 Test 4 — pair LO ⊂ chunk LO under case-insensitive
    comparison MUST pass with empty phantom_los[]."""
    from lib.validators.pair_lo_refs import PairLearningOutcomeRefsValidator

    pair = _instruction_pair(
        completion=(
            "Filler completion long enough to satisfy any sanity bound."
        ),
        learning_outcome_refs=["TO-02"],
    )
    chunk = _chunk_payload(
        text="Filler chunk body.",
        learning_outcome_refs=["TO-02", "TO-07"],
    )

    validator = PairLearningOutcomeRefsValidator()
    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk,
    )

    assert (status, reason) == ("validated", None), (
        f"expected (validated, None); got ({status!r}, {reason!r})"
    )
    resolution = new_fields["pair_lo_resolution"]
    assert resolution["phantom_los"] == [], (
        "pair LO ⊂ chunk LO MUST yield empty phantom_los[]; "
        f"got {resolution['phantom_los']!r}"
    )
    # Defence-in-depth — declared/chunk lists carry the original emit
    # case (preservation contract from W4.B brief).
    assert resolution["declared_los"] == ["TO-02"], (
        f"expected declared_los=['TO-02']; got {resolution['declared_los']!r}"
    )
    assert resolution["chunk_los"] == ["TO-02", "TO-07"], (
        f"expected chunk_los=['TO-02', 'TO-07']; "
        f"got {resolution['chunk_los']!r}"
    )


# --------------------------------------------------------------------------- #
# Test 5 — pair_lo_refs: phantom rejects, preserves emit case
# --------------------------------------------------------------------------- #


def test_pair_lo_refs_validator_phantom_rejects() -> None:
    """Plan §6 Test 5 — pair carries an LO the chunk doesn't carry →
    reject with 'phantom_pair_lo_refs'; phantom_los preserves emit case.
    """
    from lib.validators.pair_lo_refs import PairLearningOutcomeRefsValidator

    pair = _instruction_pair(
        completion=(
            "Filler completion long enough to satisfy any sanity bound."
        ),
        learning_outcome_refs=["TO-05"],
    )
    chunk = _chunk_payload(
        text="Filler chunk body.",
        learning_outcome_refs=["TO-02", "TO-07"],
    )

    validator = PairLearningOutcomeRefsValidator()
    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk,
    )

    assert (status, reason) == ("rejected", "phantom_pair_lo_refs"), (
        f"expected (rejected, phantom_pair_lo_refs); "
        f"got ({status!r}, {reason!r})"
    )
    resolution = new_fields["pair_lo_resolution"]
    assert resolution["phantom_los"] == ["TO-05"], (
        "phantom_los MUST preserve emit case (W4.B brief contract); "
        f"got {resolution['phantom_los']!r}"
    )
    # Defence-in-depth — declared / chunk lists also preserve emit case.
    assert resolution["declared_los"] == ["TO-05"], (
        f"expected declared_los=['TO-05']; got {resolution['declared_los']!r}"
    )
    assert resolution["chunk_los"] == ["TO-02", "TO-07"], (
        f"expected chunk_los=['TO-02', 'TO-07']; "
        f"got {resolution['chunk_los']!r}"
    )


# --------------------------------------------------------------------------- #
# Test 6 — both validators wire decision capture; rationale is dynamic
# --------------------------------------------------------------------------- #


def test_wave4_decision_capture_wiring() -> None:
    """Plan §6 Test 6 — decision-capture wiring contract.

    Run one pass + one fail per validator with a stub
    :class:`DecisionCapture`. Assert:

    1. Exactly 4 events total (2 per validator).
    2. Every event's ``decision_type`` is one of
       ``{"pair_claim_support_check", "pair_lo_refs_check"}``.
    3. Every rationale ≥20 chars.
    4. Every rationale interpolates the per-call dynamic signals — for
       W4.A the chunk_id + pair_kind + per-pair rate values; for W4.B
       the pair_id (falls through to chunk_id when no pair_id minted)
       + kind + the declared / chunk / phantom LO sets.
    """
    from lib.validators.pair_claim_support import PairClaimSupportValidator
    from lib.validators.pair_lo_refs import PairLearningOutcomeRefsValidator

    capture = _RecordingCapture()

    # ---- W4.A pass + fail ---- #
    pass_pair = _instruction_pair(
        completion=(
            "RDF triples are the smallest unit of structured knowledge. "
            "Each triple links a subject to an object via a predicate."
        ),
        chunk_id="ch1#pass",
    )
    pass_chunk = _chunk_payload(text=_CHUNK_TEXT_THREE_FACT)
    pass_stub = _StubNliClassifier(
        score_fn=lambda p, h: (0.85, 0.10, 0.05),
    )
    PairClaimSupportValidator(nli=pass_stub).validate_pair(
        pass_pair, kind="instruction",
        chunk=pass_chunk, decision_capture=capture,
    )

    fail_pair = _instruction_pair(
        completion=(
            "RDF triples are mutable, six-element data structures. "
            "Filler sentence to extend the completion past minimum length."
        ),
        chunk_id="ch1#fail",
    )
    fail_chunk = _chunk_payload(text=_CHUNK_TEXT_FOR_CONTRADICTION)
    fail_stub = _StubNliClassifier(
        score_fn=lambda p, h: (
            (0.10, 0.10, 0.80) if "mutable" in h else (0.85, 0.10, 0.05)
        ),
    )
    PairClaimSupportValidator(nli=fail_stub).validate_pair(
        fail_pair, kind="instruction",
        chunk=fail_chunk, decision_capture=capture,
    )

    # ---- W4.B pass + fail ---- #
    PairLearningOutcomeRefsValidator().validate_pair(
        _instruction_pair(
            completion=(
                "Filler completion long enough to satisfy any sanity bound."
            ),
            chunk_id="ch1#lo_pass",
            learning_outcome_refs=["TO-02"],
        ),
        kind="instruction",
        chunk=_chunk_payload(
            text="Filler chunk body.",
            learning_outcome_refs=["TO-02", "TO-07"],
        ),
        decision_capture=capture,
    )
    PairLearningOutcomeRefsValidator().validate_pair(
        _instruction_pair(
            completion=(
                "Filler completion long enough to satisfy any sanity bound."
            ),
            chunk_id="ch1#lo_fail",
            learning_outcome_refs=["TO-05"],
        ),
        kind="preference",
        chunk=_chunk_payload(
            text="Filler chunk body.",
            learning_outcome_refs=["TO-02", "TO-07"],
        ),
        decision_capture=capture,
    )

    # ---- Sub-clause 1 — exactly 4 events. ---- #
    assert len(capture.events) == 4, (
        f"expected exactly 4 decision events (2 per validator × 2 "
        f"validators); got {len(capture.events)}: "
        f"{[e.get('decision_type') for e in capture.events]!r}"
    )

    # ---- Sub-clause 2 — decision_type membership. ---- #
    valid_types = {"pair_claim_support_check", "pair_lo_refs_check"}
    seen_types = {e.get("decision_type") for e in capture.events}
    assert seen_types == valid_types, (
        f"expected decision_types == {valid_types!r}; got {seen_types!r}"
    )

    # Type counts: 2 of each.
    type_counts = {t: 0 for t in valid_types}
    for e in capture.events:
        type_counts[e["decision_type"]] += 1
    assert type_counts == {
        "pair_claim_support_check": 2,
        "pair_lo_refs_check": 2,
    }, f"expected 2 of each decision_type; got {type_counts!r}"

    # ---- Sub-clause 3 — every rationale ≥20 chars. ---- #
    for e in capture.events:
        rationale = str(e.get("rationale", ""))
        assert len(rationale) >= 20, (
            f"every rationale MUST be ≥20 chars per the canonical "
            f"DecisionCapture contract; got {len(rationale)} chars on "
            f"event {e!r}"
        )

    # ---- Sub-clause 4 — rationale interpolates dynamic signals. ---- #
    # W4.A events: rationale interpolates chunk_id + pair_kind + the
    # per-call rate values that drove the verdict.
    claim_events = [
        e for e in capture.events
        if e["decision_type"] == "pair_claim_support_check"
    ]
    chunk_ids_seen = set()
    for e in claim_events:
        rationale = str(e["rationale"])
        # pair_kind interpolated verbatim.
        assert "instruction" in rationale, (
            "W4.A rationale MUST interpolate pair_kind verbatim "
            f"(instruction); got rationale={rationale!r}"
        )
        # chunk_id interpolated verbatim — extract from the rationale and
        # assert it matches one of the two chunk ids we drove.
        for token in ("ch1#pass", "ch1#fail"):
            if token in rationale:
                chunk_ids_seen.add(token)
        # Dynamic per-call values (counts + rates) interpolated.
        assert "total_claims=" in rationale, (
            "W4.A rationale MUST interpolate total_claims=<n>; "
            f"got rationale={rationale!r}"
        )
        assert "claim_support_rate=" in rationale, (
            "W4.A rationale MUST interpolate claim_support_rate=<v>; "
            f"got rationale={rationale!r}"
        )
        assert "claim_contradicted_rate=" in rationale, (
            "W4.A rationale MUST interpolate claim_contradicted_rate=<v>; "
            f"got rationale={rationale!r}"
        )
        assert "promotion_status=" in rationale, (
            "W4.A rationale MUST interpolate promotion_status=<v>; "
            f"got rationale={rationale!r}"
        )
    assert chunk_ids_seen == {"ch1#pass", "ch1#fail"}, (
        "W4.A rationales MUST surface BOTH chunk_ids verbatim across "
        f"the pass + fail invocations; got {chunk_ids_seen!r}"
    )

    # W4.B events: rationale interpolates pair_id (falls through to
    # chunk_id when pair_id is unminted) + kind + the declared / chunk
    # / phantom LO sets that drove the verdict.
    lo_events = [
        e for e in capture.events
        if e["decision_type"] == "pair_lo_refs_check"
    ]
    lo_chunk_ids_seen = set()
    kinds_seen = set()
    for e in lo_events:
        rationale = str(e["rationale"])
        # pair_id falls back to chunk_id — both expected ids appear.
        for token in ("ch1#lo_pass", "ch1#lo_fail"):
            if token in rationale:
                lo_chunk_ids_seen.add(token)
        # The kind value — instruction (pass) or preference (fail).
        for token in ("instruction", "preference"):
            if token in rationale:
                kinds_seen.add(token)
        # Per-call dynamic signals — declared/chunk/phantom LO sets.
        assert "declared_los=" in rationale, (
            "W4.B rationale MUST interpolate declared_los=<list>; "
            f"got rationale={rationale!r}"
        )
        assert "chunk_los=" in rationale, (
            "W4.B rationale MUST interpolate chunk_los=<list>; "
            f"got rationale={rationale!r}"
        )
        assert "phantom_los=" in rationale, (
            "W4.B rationale MUST interpolate phantom_los=<list>; "
            f"got rationale={rationale!r}"
        )
        assert "resolved=" in rationale, (
            "W4.B rationale MUST interpolate resolved=<bool>; "
            f"got rationale={rationale!r}"
        )
    assert lo_chunk_ids_seen == {"ch1#lo_pass", "ch1#lo_fail"}, (
        "W4.B rationales MUST surface BOTH pair_id (chunk_id fallback) "
        f"values across the pass + fail invocations; got {lo_chunk_ids_seen!r}"
    )
    assert kinds_seen == {"instruction", "preference"}, (
        "W4.B rationales MUST surface BOTH kind values across the "
        f"pass + fail invocations; got {kinds_seen!r}"
    )


# --------------------------------------------------------------------------- #
# Optional Test 7 — strict-pair-schema round trip (recommended)
# --------------------------------------------------------------------------- #


def test_wave4_strict_pair_schema_round_trip() -> None:
    """Plan §6 optional Test 7 — emit a pair through
    :meth:`PairClaimSupportValidator.validate_pair` then validate the
    resulting pair against ``instruction_pair.strict.schema.json``.
    Asserts no schema error on the populated Wave 4 fields.

    Defence-in-depth on the W4.D pair-schema migration: catches the
    regression class where a future schema author tightens the
    Wave 4 field shapes without re-validating against a real
    validator emit.
    """
    jsonschema = pytest.importorskip("jsonschema")
    from lib.validators.pair_claim_support import PairClaimSupportValidator
    from lib.validators.pair_lo_refs import PairLearningOutcomeRefsValidator

    # Build a schema-valid base pair (minimum required fields per
    # ``instruction_pair.strict.schema.json::required``).
    base_pair: Dict[str, Any] = {
        "prompt": "X" * 50,
        "completion": (
            "RDF triples are the smallest unit of structured knowledge. "
            "Each triple links a subject to an object via a predicate. "
            "Subjects, predicates, and objects are all IRIs."
        ),
        "chunk_id": "ch1#sec1",
        "lo_refs": ["TO-02"],
        "learning_outcome_refs": ["TO-02"],  # W4.B reads this
        "bloom_level": "understand",
        "content_type": "overview",
        "seed": 42,
        "decision_capture_id": "evt_w4_round_trip",
    }
    chunk = _chunk_payload(
        text=_CHUNK_TEXT_THREE_FACT,
        learning_outcome_refs=["TO-02", "TO-07"],
    )

    # W4.A populates per_claim_support / claim_support_rate /
    # claim_contradicted_rate.
    stub = _StubNliClassifier(
        score_fn=lambda p, h: (0.85, 0.10, 0.05),
    )
    _, _, claim_fields = PairClaimSupportValidator(nli=stub).validate_pair(
        base_pair, kind="instruction", chunk=chunk,
    )
    base_pair.update(claim_fields)

    # W4.B populates pair_lo_resolution.
    _, _, lo_fields = PairLearningOutcomeRefsValidator().validate_pair(
        base_pair, kind="instruction", chunk=chunk,
    )
    base_pair.update(lo_fields)

    # The W4.A new_fields carry a ``deps_missing`` debug field that is
    # NOT part of the canonical pair schema (it's per-call metadata
    # the caller can drop before write). Strip it for the round-trip
    # assertion.
    base_pair.pop("deps_missing", None)

    # Round-trip against both schemas.
    schema_paths = [
        PROJECT_ROOT / "schemas" / "knowledge" / "instruction_pair.schema.json",
        PROJECT_ROOT
        / "schemas"
        / "knowledge"
        / "instruction_pair.strict.schema.json",
    ]
    for schema_path in schema_paths:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        # Should not raise.
        jsonschema.Draft202012Validator(schema).validate(base_pair)


# --------------------------------------------------------------------------- #
# W4.C.2 schema-migration tests — pair_objective_alignment shape
# --------------------------------------------------------------------------- #


_W4C_PAIR_SCHEMA_PATHS = [
    "schemas/knowledge/instruction_pair.schema.json",
    "schemas/knowledge/instruction_pair.strict.schema.json",
    "schemas/knowledge/preference_pair.schema.json",
]


def _w4c_minimal_instruction_pair() -> Dict[str, Any]:
    """Minimum-required-field instruction pair valid against both
    instruction-pair schemas (lenient + strict)."""
    return {
        "prompt": "X" * 50,
        "completion": "Y" * 100,
        "chunk_id": "ch1#sec1",
        "lo_refs": ["TO-02"],
        "bloom_level": "understand",
        "content_type": "overview",
        "seed": 42,
        "decision_capture_id": "evt_w4c_round_trip",
    }


def _w4c_minimal_preference_pair() -> Dict[str, Any]:
    """Minimum-required-field preference pair valid against
    preference_pair.schema.json."""
    return {
        "prompt": "X" * 50,
        "chosen": "Y" * 100,
        "rejected": "Z" * 100,
        "chunk_id": "ch1#sec1",
        "lo_refs": ["TO-02"],
        "seed": 42,
        "decision_capture_id": "evt_w4c_round_trip_pref",
    }


def _w4c_validate(pair: Dict[str, Any], schema_rel_path: str) -> None:
    """Helper — Draft-07 validate the pair against the named schema."""
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = PROJECT_ROOT / schema_rel_path
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(pair)


def _w4c_assert_invalid(pair: Dict[str, Any], schema_rel_path: str) -> None:
    """Helper — assert the pair FAILS Draft-07 validation against the
    named schema (negative test)."""
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = PROJECT_ROOT / schema_rel_path
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(pair)


def test_pair_objective_alignment_populated_passes_all_three_schemas() -> None:
    """W4.C.2 positive — a fully-populated pair_objective_alignment[]
    array passes Draft-07 validation against all three pair schemas
    (instruction, strict instruction, preference).

    Mirrors the W4.A / W4.B day-1 contract: the audit field is
    optional, but a well-formed populated array MUST validate so
    downstream emitters can ship the surface without schema drift.
    """
    populated_alignment = [
        {
            "objective_id": "TO-02",
            "status": "delivered",
            "statement_entailment_score": 0.91,
            "contradiction_score": 0.03,
            "bloom_gap": 0,
            "verb_match_count": 2,
            "declared_bloom": "understand",
            "observed_bloom": "understand",
            "entailment_threshold": 0.70,
        },
        {
            # Demonstrates nullable-field tolerance — every optional
            # field is null and the entry still validates.
            "objective_id": "TO-07",
            "status": "unverifiable",
            "statement_entailment_score": None,
            "contradiction_score": None,
            "bloom_gap": None,
            "verb_match_count": None,
            "declared_bloom": None,
            "observed_bloom": None,
            "entailment_threshold": None,
        },
    ]
    pass_rate = 0.5  # 1 of 2 entries delivered

    # Validate against the two instruction-pair schemas.
    for schema_rel_path in _W4C_PAIR_SCHEMA_PATHS[:2]:
        pair = _w4c_minimal_instruction_pair()
        pair["pair_objective_alignment"] = [dict(e) for e in populated_alignment]
        pair["pair_objective_alignment_pass_rate"] = pass_rate
        _w4c_validate(pair, schema_rel_path)

    # Validate against preference_pair.schema.json.
    pref_pair = _w4c_minimal_preference_pair()
    pref_pair["pair_objective_alignment"] = [
        dict(e) for e in populated_alignment
    ]
    pref_pair["pair_objective_alignment_pass_rate"] = pass_rate
    _w4c_validate(pref_pair, _W4C_PAIR_SCHEMA_PATHS[2])


def test_pair_objective_alignment_rejects_malformed_status_enum() -> None:
    """W4.C.2 negative #1 — an entry with an unknown ``status`` enum
    value MUST fail Draft-07 validation against all three pair schemas.
    Closes the regression class where a future emitter ships a
    typo'd / out-of-spec status string.
    """
    bad_alignment = [
        {
            "objective_id": "TO-02",
            # NOT in the canonical 4-value enum.
            "status": "partially_delivered",
            "statement_entailment_score": 0.55,
            "contradiction_score": 0.10,
            "bloom_gap": 1,
            "verb_match_count": 1,
            "declared_bloom": "apply",
            "observed_bloom": "understand",
            "entailment_threshold": 0.70,
        }
    ]

    for schema_rel_path in _W4C_PAIR_SCHEMA_PATHS[:2]:
        pair = _w4c_minimal_instruction_pair()
        pair["pair_objective_alignment"] = [dict(e) for e in bad_alignment]
        _w4c_assert_invalid(pair, schema_rel_path)

    pref_pair = _w4c_minimal_preference_pair()
    pref_pair["pair_objective_alignment"] = [dict(e) for e in bad_alignment]
    _w4c_assert_invalid(pref_pair, _W4C_PAIR_SCHEMA_PATHS[2])


def test_pair_objective_alignment_rejects_missing_objective_id() -> None:
    """W4.C.2 negative #2 — an entry missing the required
    ``objective_id`` field MUST fail Draft-07 validation against all
    three pair schemas. Closes the regression class where a future
    emitter ships an alignment entry without the canonical LO id.

    Defence-in-depth on the ``additionalProperties: false`` discipline
    by also asserting that an entry with an unminted extra property
    (mistyped key) fails alongside the missing-required-field case.
    """
    missing_objective_id = [
        {
            # objective_id deliberately omitted.
            "status": "delivered",
            "statement_entailment_score": 0.91,
            "contradiction_score": 0.03,
            "bloom_gap": 0,
            "verb_match_count": 2,
            "declared_bloom": "understand",
            "observed_bloom": "understand",
            "entailment_threshold": 0.70,
        }
    ]

    for schema_rel_path in _W4C_PAIR_SCHEMA_PATHS[:2]:
        pair = _w4c_minimal_instruction_pair()
        pair["pair_objective_alignment"] = [
            dict(e) for e in missing_objective_id
        ]
        _w4c_assert_invalid(pair, schema_rel_path)

    pref_pair = _w4c_minimal_preference_pair()
    pref_pair["pair_objective_alignment"] = [
        dict(e) for e in missing_objective_id
    ]
    _w4c_assert_invalid(pref_pair, _W4C_PAIR_SCHEMA_PATHS[2])

    # Defence-in-depth — additionalProperties: false catches an
    # unminted key (e.g. emitter-side typo).
    typo_alignment = [
        {
            "objective_id": "TO-02",
            "status": "delivered",
            # Typo'd field (should be statement_entailment_score).
            "statement_entail_score": 0.91,
        }
    ]
    for schema_rel_path in _W4C_PAIR_SCHEMA_PATHS[:2]:
        pair = _w4c_minimal_instruction_pair()
        pair["pair_objective_alignment"] = [dict(e) for e in typo_alignment]
        _w4c_assert_invalid(pair, schema_rel_path)

    pref_pair = _w4c_minimal_preference_pair()
    pref_pair["pair_objective_alignment"] = [dict(e) for e in typo_alignment]
    _w4c_assert_invalid(pref_pair, _W4C_PAIR_SCHEMA_PATHS[2])


# --------------------------------------------------------------------------- #
# Wave 5 W5.D — pair-side per-claim attribution scoping
# --------------------------------------------------------------------------- #


class _StubEmbedder:
    """Stub :class:`SentenceEmbedder` returning canned vectors keyed
    by a per-test text → vector dict.

    Mirrors the in-tree precedent at
    ``lib/validators/tests/test_concept_example_similarity.py``: tests
    inject vectors keyed by text so cosine similarity is fully
    deterministic and independent of the actual embedding model.
    """

    def __init__(self, *, text_to_vec: Dict[str, List[float]]) -> None:
        self._table = dict(text_to_vec)

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        # Return a unit-normalised vector so cosine-similarity is a
        # plain dot product downstream.
        vec = self._table.get(text)
        if vec is None:
            # Default to a constant orthogonal-to-everything vector
            # so an unmapped text never matches a claim above the
            # 0.50 floor.
            return [0.0, 0.0, 0.0, 1.0]
        # Normalise.
        norm = sum(x * x for x in vec) ** 0.5
        if norm == 0:
            return list(vec)
        return [x / norm for x in vec]


def _patch_embedder(monkeypatch, embedder) -> None:
    """Patch :func:`lib.embedding.sentence_embedder.try_load_embedder`
    so the per-claim attribution path resolves to ``embedder``.

    Mirrors the test idiom used elsewhere in the project (see
    ``lib/validators/tests/test_objective_assessment_similarity.py``)
    — patch the loader rather than the validator import so the
    validator's lazy import inside
    ``_try_load_embedder_safe`` resolves to the stub.
    """
    import lib.embedding.sentence_embedder as sem_mod

    monkeypatch.setattr(
        sem_mod, "try_load_embedder", lambda *_a, **_kw: embedder,
    )


def test_pair_claim_support_uses_per_claim_attribution_when_chunk_carries_key_claims(
    monkeypatch,
) -> None:
    """Wave 5 W5.D — when the cited chunk carries the W1.5 / W5.A
    structured ``key_claims=[{claim, source_chunk_ids[]}]`` shape AND
    a ``chunk_id_to_text_map`` is threaded, the validator MUST scope
    its NLI premise per-claim.

    Assertions:

    1. ``per_claim_support[].source_chunk_ids`` is populated for every
       sentence that cosine-matched a structured claim (drawn from
       the matched entry's ``source_chunk_ids[]``).
    2. The decision-capture rationale interpolates
       ``structured_attribution_used=True`` and
       ``per_claim_match_count=N`` so an audit can replay whether the
       per-claim attribution scoping path fired.
    3. The NLI premise(s) for each matched sentence are the chunk
       texts named in the matched entry's ``source_chunk_ids[]`` —
       NOT the cited chunk's whole-text body. We verify by stamping
       distinct sentinel strings into the lookup-map chunks and
       asserting the stub NLI received those sentinels as the
       premise.
    """
    from lib.validators.pair_claim_support import (
        PairClaimSupportValidator,
    )

    # Two sentences in the completion. Each lines up 1:1 with one of
    # the structured key_claims on the chunk.
    completion = (
        "Sentence aligning with the first structured claim verbatim. "
        "Sentence aligning with the second structured claim verbatim."
    )
    pair = _instruction_pair(completion=completion, chunk_id="ch1#sec1")

    # Cited chunk carries:
    #   - whole-chunk text body (would be the legacy NLI premise).
    #   - structured key_claims with attribution to two OTHER chunks
    #     (ch:src_a, ch:src_b) — distinct from the cited chunk_id so
    #     the test can prove the validator resolved attribution via
    #     the lookup map rather than falling back to chunk["text"].
    chunk = _chunk_payload(
        text="Whole-chunk-text body that MUST NOT be the NLI premise.",
    )
    chunk["key_claims"] = [
        {
            "claim": "First structured claim text.",
            "source_chunk_ids": ["ch:src_a"],
        },
        {
            "claim": "Second structured claim text.",
            "source_chunk_ids": ["ch:src_b"],
        },
    ]

    # Sentinel strings — the stub NLI recognises these as the per-
    # claim attribution premise. If the validator falls back to
    # whole-chunk-text scoring, the stub will see the chunk["text"]
    # body instead and the assertion below will fail.
    chunk_id_to_text_map = {
        "ch:src_a": "PREMISE_FROM_SRC_A_SENTINEL_STRING",
        "ch:src_b": "PREMISE_FROM_SRC_B_SENTINEL_STRING",
    }

    # Embedder stub: each sentence vector aligns with its
    # corresponding claim vector (cosine 1.0); cross-pairs have low
    # cosine.
    text_to_vec = {
        "Sentence aligning with the first structured claim verbatim.":
            [1.0, 0.0, 0.0, 0.0],
        "Sentence aligning with the second structured claim verbatim.":
            [0.0, 1.0, 0.0, 0.0],
        "First structured claim text.": [1.0, 0.0, 0.0, 0.0],
        "Second structured claim text.": [0.0, 1.0, 0.0, 0.0],
    }
    _patch_embedder(monkeypatch, _StubEmbedder(text_to_vec=text_to_vec))

    # NLI stub: every (premise, hypothesis) pair entails. The premise
    # text is recorded so we can assert the per-claim attribution
    # path actually fed sentinel chunks (not whole-chunk-text).
    seen_premises: List[str] = []

    def _score(premise, hypothesis):
        seen_premises.append(premise)
        return (0.85, 0.10, 0.05)

    stub_nli = _StubNliClassifier(score_fn=_score)
    validator = PairClaimSupportValidator(nli=stub_nli)
    capture = _RecordingCapture()

    status, reason, new_fields = validator.validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
        chunk_id_to_text_map=chunk_id_to_text_map,
        decision_capture=capture,
    )

    # Both sentences entail → validated.
    assert (status, reason) == ("validated", None), (
        f"expected (validated, None); got ({status!r}, {reason!r}); "
        f"new_fields={new_fields!r}"
    )

    # ---- Sub-clause 1: source_chunk_ids populated per-sentence. ---- #
    pcs = new_fields["per_claim_support"]
    assert isinstance(pcs, list) and len(pcs) == 2, (
        f"expected 2 per_claim_support entries; got {pcs!r}"
    )
    attribution = [entry.get("source_chunk_ids") for entry in pcs]
    assert attribution == [["ch:src_a"], ["ch:src_b"]], (
        f"expected per-sentence source_chunk_ids attribution to "
        f"mirror the matched key_claims entries; got {attribution!r}"
    )

    # ---- Sub-clause 2: decision-capture interpolates W5.D signals. ---- #
    assert len(capture.events) == 1, (
        f"expected exactly 1 pair_claim_support_check event; "
        f"got {len(capture.events)}"
    )
    rationale = str(capture.events[0]["rationale"])
    assert "structured_attribution_used=True" in rationale, (
        "W5.D rationale MUST interpolate structured_attribution_used=True; "
        f"got rationale={rationale!r}"
    )
    assert "per_claim_match_count=2" in rationale, (
        "W5.D rationale MUST interpolate per_claim_match_count=<n>; "
        f"got rationale={rationale!r}"
    )

    # ---- Sub-clause 3: NLI premise was the sentinel, not chunk text. ---- #
    # The whole-chunk-text body MUST NOT have been used as a premise.
    assert all(
        "Whole-chunk-text body" not in p for p in seen_premises
    ), (
        "Per-claim attribution scoping MUST replace the whole-chunk-text "
        "premise with the per-claim source_chunk_ids[] texts; got "
        f"premises={seen_premises!r}"
    )
    # Both sentinels appear in the recorded premises (one per sentence).
    assert "PREMISE_FROM_SRC_A_SENTINEL_STRING" in seen_premises, (
        f"expected ch:src_a sentinel in NLI premises; got {seen_premises!r}"
    )
    assert "PREMISE_FROM_SRC_B_SENTINEL_STRING" in seen_premises, (
        f"expected ch:src_b sentinel in NLI premises; got {seen_premises!r}"
    )


def test_pair_claim_support_falls_back_to_whole_chunk_text_without_structured_key_claims(
    monkeypatch,
) -> None:
    """Wave 5 W5.D negative — same pair + chunk WITHOUT a structured
    ``key_claims`` field MUST fall back to whole-chunk-text NLI
    scoring (byte-identical to the W4.A pre-W5.D path).

    Assertions:

    1. ``per_claim_support[].source_chunk_ids`` is ``None`` for every
       sentence (no per-claim attribution applies).
    2. The decision-capture rationale interpolates
       ``structured_attribution_used=False`` AND
       ``per_claim_match_count=0``.
    3. The NLI premise was the chunk's whole-text body — NOT any
       per-claim sentinel.
    """
    from lib.validators.pair_claim_support import (
        PairClaimSupportValidator,
    )

    completion = (
        "Sentence aligning with the first structured claim verbatim. "
        "Sentence aligning with the second structured claim verbatim."
    )
    pair = _instruction_pair(completion=completion, chunk_id="ch1#sec1")

    # Chunk does NOT carry structured key_claims.
    whole_text = (
        "Whole-chunk-text body that IS the NLI premise in this fallback path."
    )
    chunk = _chunk_payload(text=whole_text)
    # No chunk["key_claims"] at all — the legacy / pre-W1.5 corpus arm.

    # Lookup map is irrelevant in this arm — but provide one anyway
    # to demonstrate that the validator's structured_flag check
    # short-circuits the per-claim path BEFORE attempting any
    # lookups.
    chunk_id_to_text_map = {
        "ch:src_a": "SHOULD_NEVER_APPEAR_AS_PREMISE",
    }

    # Embedder stub — should never be consulted in the fallback arm.
    embedder_called = []

    class _FailIfCalledEmbedder:
        def encode(self, text, normalize=True):
            embedder_called.append(text)
            return [1.0, 0.0, 0.0, 0.0]

    _patch_embedder(monkeypatch, _FailIfCalledEmbedder())

    seen_premises: List[str] = []

    def _score(premise, hypothesis):
        seen_premises.append(premise)
        return (0.85, 0.10, 0.05)

    stub_nli = _StubNliClassifier(score_fn=_score)
    validator = PairClaimSupportValidator(nli=stub_nli)
    capture = _RecordingCapture()

    status, reason, new_fields = validator.validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
        chunk_id_to_text_map=chunk_id_to_text_map,
        decision_capture=capture,
    )

    assert (status, reason) == ("validated", None), (
        f"expected (validated, None); got ({status!r}, {reason!r})"
    )

    # ---- Sub-clause 1: every per_claim_support entry has source_chunk_ids=None. ---- #
    pcs = new_fields["per_claim_support"]
    assert isinstance(pcs, list) and len(pcs) == 2, (
        f"expected 2 per_claim_support entries; got {pcs!r}"
    )
    attribution = [entry.get("source_chunk_ids") for entry in pcs]
    assert attribution == [None, None], (
        f"expected source_chunk_ids=None on every entry in fallback "
        f"path; got {attribution!r}"
    )

    # ---- Sub-clause 2: rationale interpolates the fallback-path signals. ---- #
    assert len(capture.events) == 1
    rationale = str(capture.events[0]["rationale"])
    assert "structured_attribution_used=False" in rationale, (
        "W5.D rationale MUST interpolate structured_attribution_used=False "
        f"in the legacy / fallback arm; got rationale={rationale!r}"
    )
    assert "per_claim_match_count=0" in rationale, (
        "W5.D rationale MUST interpolate per_claim_match_count=0 in the "
        f"legacy / fallback arm; got rationale={rationale!r}"
    )

    # ---- Sub-clause 3: NLI premise == whole chunk text. ---- #
    assert all(p == whole_text for p in seen_premises), (
        "Fallback path MUST use whole chunk text as NLI premise for "
        f"every sentence; got premises={seen_premises!r}"
    )
    assert all(
        "SHOULD_NEVER_APPEAR_AS_PREMISE" not in p for p in seen_premises
    ), (
        f"lookup-map text leaked into fallback-path NLI premise; "
        f"premises={seen_premises!r}"
    )
    # Embedder MUST NOT have been consulted (structured_flag was False).
    assert embedder_called == [], (
        "Fallback path MUST short-circuit the embedder load; "
        f"got encoder calls={embedder_called!r}"
    )


def test_pair_claim_support_falls_back_to_whole_chunk_text_when_no_chunk_id_to_text_map(
    monkeypatch,
) -> None:
    """Wave 5 W5.D second negative — chunk DOES carry structured
    ``key_claims`` BUT the call site did not thread a
    ``chunk_id_to_text_map`` lookup. Validator MUST graceful-degrade
    to whole-chunk-text scoring (back-compat day-1 — the
    ``chunks_by_id`` map can be empty in legacy synthesize_training
    code paths).

    Assertions: identical shape to the no-structured-key_claims case
    (source_chunk_ids=None, structured_attribution_used=False,
    per_claim_match_count=0).
    """
    from lib.validators.pair_claim_support import (
        PairClaimSupportValidator,
    )

    completion = (
        "First sentence in the completion body for testing. "
        "Second sentence in the completion body for testing."
    )
    pair = _instruction_pair(completion=completion, chunk_id="ch1#sec1")

    whole_text = "Whole chunk body used as the fallback NLI premise."
    chunk = _chunk_payload(text=whole_text)
    chunk["key_claims"] = [
        {
            "claim": "First structured claim text.",
            "source_chunk_ids": ["ch:src_a"],
        },
    ]

    seen_premises: List[str] = []

    def _score(premise, hypothesis):
        seen_premises.append(premise)
        return (0.85, 0.10, 0.05)

    stub_nli = _StubNliClassifier(score_fn=_score)
    validator = PairClaimSupportValidator(nli=stub_nli)
    capture = _RecordingCapture()

    status, _reason, new_fields = validator.validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
        # Note: chunk_id_to_text_map intentionally omitted (defaults
        # to None) — the back-compat-no-map arm.
        decision_capture=capture,
    )

    assert status == "validated"
    pcs = new_fields["per_claim_support"]
    attribution = [entry.get("source_chunk_ids") for entry in pcs]
    assert attribution == [None, None], (
        f"expected source_chunk_ids=None when chunk_id_to_text_map "
        f"is absent; got {attribution!r}"
    )
    rationale = str(capture.events[0]["rationale"])
    assert "structured_attribution_used=False" in rationale
    assert "per_claim_match_count=0" in rationale
    assert all(p == whole_text for p in seen_premises)


def test_resolve_chunk_key_claims_with_attribution_helper() -> None:
    """Wave 5 W5.D unit — the resolution helper distinguishes the
    structured / legacy / malformed shapes per the W5.A migration
    contract. Closes the regression class where a future schema
    author tightens or loosens the chunk-side key_claims shape
    without re-validating against the validator's resolution helper.
    """
    from lib.validators.pair_claim_support import (
        _resolve_chunk_key_claims_with_attribution,
    )

    # ---- Case 1: structured shape with usable attribution. ---- #
    structured_chunk = {
        "key_claims": [
            {"claim": "X", "source_chunk_ids": ["ch:src_a"]},
            {"claim": "Y", "source_chunk_ids": ["ch:src_b", "ch:src_c"]},
        ],
    }
    entries, structured = _resolve_chunk_key_claims_with_attribution(
        structured_chunk
    )
    assert structured is True, (
        "structured key_claims MUST set structured_flag=True"
    )
    assert len(entries) == 2
    assert entries[0]["source_chunk_ids"] == ["ch:src_a"]
    assert entries[1]["source_chunk_ids"] == ["ch:src_b", "ch:src_c"]

    # ---- Case 2: legacy List[str] shape. ---- #
    legacy_chunk = {"key_claims": ["bare claim 1", "bare claim 2"]}
    entries, structured = _resolve_chunk_key_claims_with_attribution(
        legacy_chunk
    )
    assert structured is False, (
        "legacy List[str] key_claims MUST yield structured_flag=False"
    )
    # Bare strings are dropped (per-claim attribution scoping is a
    # no-op on legacy corpora — block-level fallback applies elsewhere).
    assert entries == []

    # ---- Case 3: absent key_claims. ---- #
    bare_chunk = {"text": "some chunk body"}
    assert _resolve_chunk_key_claims_with_attribution(bare_chunk) == (
        [],
        False,
    )

    # ---- Case 4: malformed entries. ---- #
    malformed_chunk = {
        "key_claims": [
            "bare string in a structured-mixed list",  # dropped
            {"claim": ""},  # empty claim → dropped
            {"claim": "no attribution"},  # no source_chunk_ids → dropped
            {"claim": "empty list", "source_chunk_ids": []},  # empty → dropped
            {
                "claim": "valid",
                "source_chunk_ids": ["ch:src_a", 42, "", "ch:src_b"],
            },
            42,  # not a dict → dropped
        ],
    }
    entries, structured = _resolve_chunk_key_claims_with_attribution(
        malformed_chunk
    )
    assert structured is True, (
        "the one valid structured entry MUST set structured_flag=True"
    )
    assert len(entries) == 1
    # Non-string IDs filtered.
    assert entries[0]["source_chunk_ids"] == ["ch:src_a", "ch:src_b"]

    # ---- Case 5: chunk is not a dict. ---- #
    assert _resolve_chunk_key_claims_with_attribution(None) == ([], False)
    assert _resolve_chunk_key_claims_with_attribution("not a dict") == (
        [],
        False,
    )


# --------------------------------------------------------------------------- #
# Wave 9 TIGHT — dual-source DART cross-check (chunker-drift slice)
# --------------------------------------------------------------------------- #


def _w9_chunk_with_source_refs(
    *,
    text: str,
    source_ids: List[str],
    role: str = "primary",
) -> Dict[str, Any]:
    """Build a chunk dict carrying ``source.source_references[]`` so
    the Wave 9 dual-source check resolves DART block texts via the
    canonical ``sourceId`` chain.
    """
    return {
        "text": text,
        "source": {
            "source_references": [
                {"sourceId": sid, "role": role} for sid in source_ids
            ],
        },
    }


def test_pair_claim_support_dual_source_off_by_default() -> None:
    """Wave 9 TIGHT — legacy call site without the new kwargs MUST
    skip the DART cross-check. Validates back-compat: the W4.A / W5.D
    surface contract is preserved byte-for-byte for callers that
    haven't yet wired ``dart_block_text_map`` / ``dual_source_severity``
    through.
    """
    from lib.validators.pair_claim_support import PairClaimSupportValidator

    completion = (
        "RDF triples are the smallest unit of structured knowledge. "
        "Each triple links a subject to an object via a predicate."
    )
    pair = _instruction_pair(completion=completion, chunk_id="ch1#legacy")
    chunk = _w9_chunk_with_source_refs(
        text=_CHUNK_TEXT_THREE_FACT,
        source_ids=["dart:doc1#blk_001"],
    )

    seen_premises: List[str] = []

    def _score(premise, hypothesis):
        seen_premises.append(premise)
        return (0.85, 0.10, 0.05)

    stub = _StubNliClassifier(score_fn=_score)
    validator = PairClaimSupportValidator(nli=stub)

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk,
        # No dart_block_text_map, no dual_source_severity — defaults.
    )

    assert (status, reason) == ("validated", None)
    pcs = new_fields["per_claim_support"]
    assert isinstance(pcs, list) and len(pcs) == 2
    # Wave 9 audit field MUST NOT be stamped on legacy callers.
    for entry in pcs:
        assert "dart_source_check" not in entry, (
            "Legacy call without dual-source kwargs MUST NOT stamp "
            f"dart_source_check; got entry={entry!r}"
        )
    # Outcome stays in the canonical 3-value set.
    outcomes = {entry["outcome"] for entry in pcs}
    assert outcomes <= {"entailed", "unsupported", "contradicted"}, (
        f"legacy outcome set MUST stay 3-value; got {outcomes!r}"
    )
    # No NLI premise was the DART block text — the only premise was
    # the IMSCC chunk text (W4.A path).
    assert all(p == _CHUNK_TEXT_THREE_FACT for p in seen_premises), (
        f"legacy path MUST score against the IMSCC chunk text only; "
        f"got premises={seen_premises!r}"
    )


def test_pair_claim_support_dual_source_passes_when_dart_entails() -> None:
    """Wave 9 TIGHT positive — IMSCC entails AND DART entails; the
    audit field is stamped, outcome stays ``"entailed"``.

    Asserts:

    1. ``dart_source_check`` is stamped on every per-claim entry that
       was ``entailed`` after the IMSCC pass.
    2. The DART NLI premise was the resolved DART block text (NOT the
       IMSCC chunk text).
    3. ``per_claim_support[*].outcome`` stays ``"entailed"`` because
       the DART check did not contradict.
    4. Decision-capture rationale interpolates
       ``dart_check_fired_count=2`` and ``dart_disagreement_count=0``.
    """
    from lib.validators.pair_claim_support import PairClaimSupportValidator

    completion = (
        "RDF triples are the smallest unit of structured knowledge. "
        "Each triple links a subject to an object via a predicate."
    )
    pair = _instruction_pair(completion=completion, chunk_id="ch1#w9_pass")
    chunk = _w9_chunk_with_source_refs(
        text=_CHUNK_TEXT_THREE_FACT,
        source_ids=["dart:doc1#blk_001"],
    )
    dart_premise_sentinel = "DART_W9_PASS_BLOCK_TEXT_SENTINEL_001"
    dart_block_text_map = {"dart:doc1#blk_001": dart_premise_sentinel}

    seen_premises: List[str] = []

    def _score(premise, hypothesis):
        seen_premises.append(premise)
        return (0.85, 0.10, 0.05)  # entailed everywhere

    stub = _StubNliClassifier(score_fn=_score)
    validator = PairClaimSupportValidator(nli=stub)
    capture = _RecordingCapture()

    status, reason, new_fields = validator.validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
        decision_capture=capture,
        dart_block_text_map=dart_block_text_map,
        dual_source_severity="warning",
    )

    assert (status, reason) == ("validated", None)
    pcs = new_fields["per_claim_support"]
    assert isinstance(pcs, list) and len(pcs) == 2

    # ---- Sub-clause 1: dart_source_check stamped on each entry. ---- #
    for entry in pcs:
        assert "dart_source_check" in entry, (
            f"DART check MUST stamp dart_source_check on entailed "
            f"entries; got entry={entry!r}"
        )
        ds = entry["dart_source_check"]
        assert ds["entailment"] == pytest.approx(0.85)
        assert ds["contradiction"] == pytest.approx(0.05)
        assert ds["outcome"] == "entailed"
        # IMSCC outcome stays "entailed" because DART agreed.
        assert entry["outcome"] == "entailed"

    # ---- Sub-clause 2: DART premise was the sentinel. ---- #
    assert dart_premise_sentinel in seen_premises, (
        f"dual-source check MUST score against the DART block text; "
        f"got premises={seen_premises!r}"
    )

    # ---- Sub-clause 3: rationale interpolates W9 signals. ---- #
    assert len(capture.events) == 1
    rationale = str(capture.events[0]["rationale"])
    assert "dart_check_fired_count=2" in rationale, (
        "W9 rationale MUST interpolate dart_check_fired_count=<n>; "
        f"got rationale={rationale!r}"
    )
    assert "dart_disagreement_count=0" in rationale, (
        "W9 rationale MUST interpolate dart_disagreement_count=<n>; "
        f"got rationale={rationale!r}"
    )


def test_pair_claim_support_dual_source_stamps_dart_disagreement_on_contradiction() -> None:
    """Wave 9 TIGHT negative — IMSCC entails BUT DART contradicts at
    >= 0.50; outcome bumps to ``"dart_disagreement"`` and the pair is
    NOT rejected (warning severity).

    Asserts:

    1. ``per_claim_support[*].outcome`` is ``"dart_disagreement"`` for
       the entry the DART pass contradicted.
    2. ``dart_source_check.outcome`` is ``"dart_disagreement"``.
    3. The pair is NOT rejected — ``(status, reason) == ("validated",
       None)`` because dual-source severity is warning-only.
    4. ``dart_disagreement_count=1`` interpolated in rationale.
    """
    from lib.validators.pair_claim_support import (
        PairClaimSupportValidator,
        _DEFAULT_DART_CONTRADICTION_FLOOR,
    )

    completion = (
        "RDF triples are the smallest unit of structured knowledge. "
        "Each triple links a subject to an object via a predicate."
    )
    pair = _instruction_pair(
        completion=completion, chunk_id="ch1#w9_disagree",
    )
    chunk = _w9_chunk_with_source_refs(
        text=_CHUNK_TEXT_THREE_FACT,
        source_ids=["dart:doc1#blk_002"],
    )
    dart_premise = "DART_W9_DISAGREE_BLOCK_TEXT"
    dart_block_text_map = {"dart:doc1#blk_002": dart_premise}

    def _score(premise, hypothesis):
        # IMSCC pass: every sentence entailed.
        # DART pass: first sentence contradicted (>= 0.50).
        if premise == dart_premise:
            if "smallest unit" in hypothesis:
                return (0.05, 0.10, 0.85)  # contradicted
            return (0.85, 0.10, 0.05)  # entailed (DART)
        return (0.85, 0.10, 0.05)  # IMSCC entailed

    stub = _StubNliClassifier(score_fn=_score)
    validator = PairClaimSupportValidator(nli=stub)
    capture = _RecordingCapture()

    status, reason, new_fields = validator.validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
        decision_capture=capture,
        dart_block_text_map=dart_block_text_map,
        dual_source_severity="warning",
    )

    # ---- Sub-clause 3 (eager): warning-only — pair NOT rejected. ---- #
    assert (status, reason) == ("validated", None), (
        f"Wave 9 dual-source disagreement MUST NOT reject the pair "
        f"(warning severity); got ({status!r}, {reason!r})"
    )

    pcs = new_fields["per_claim_support"]
    assert len(pcs) == 2
    # First entry: DART contradicted → outcome bumped + dart_source_check
    # carries the disagreement bucket.
    first = pcs[0]
    assert first["outcome"] == "dart_disagreement", (
        f"DART contradiction MUST bump outcome to dart_disagreement; "
        f"got entry={first!r}"
    )
    assert "dart_source_check" in first
    ds = first["dart_source_check"]
    assert ds["outcome"] == "dart_disagreement"
    assert ds["contradiction"] >= _DEFAULT_DART_CONTRADICTION_FLOOR
    # Second entry: DART agreed → outcome stays entailed.
    second = pcs[1]
    assert second["outcome"] == "entailed"
    assert second["dart_source_check"]["outcome"] == "entailed"

    # ---- Sub-clause 4: rationale carries dart_disagreement_count=1. ---- #
    rationale = str(capture.events[0]["rationale"])
    assert "dart_disagreement_count=1" in rationale, (
        "W9 rationale MUST interpolate dart_disagreement_count on the "
        f"disagreement path; got rationale={rationale!r}"
    )
    assert "dart_check_fired_count=2" in rationale


def test_pair_claim_support_dual_source_skips_deterministic_pairs() -> None:
    """Wave 9 TIGHT × W8 precedence — pairs carrying
    ``pair_lo_resolution.skipped == "deterministic_template"`` (W8
    Worker A audit-stamp helper) MUST NOT trigger the dual-source
    check.

    Deterministic pairs don't go through the LLM paraphrase path so a
    DART / IMSCC drift signal is meaningless on them. Skipping cleanly
    (no ``dart_source_check`` field, no ``dart_disagreement`` outcome
    bump) preserves the W8 audit stamp as the only resolution stamped
    on the pair.
    """
    from lib.validators.pair_claim_support import PairClaimSupportValidator

    completion = (
        "RDF triples are the smallest unit of structured knowledge. "
        "Each triple links a subject to an object via a predicate."
    )
    pair = _instruction_pair(
        completion=completion, chunk_id="ch1#w9_w8_skip",
    )
    # W8 deterministic-template marker.
    pair["pair_lo_resolution"] = {
        "declared_los": ["kg-metadata"],
        "chunk_los": ["kg-metadata"],
        "phantom_los": [],
        "resolved": True,
        "skipped": "deterministic_template",
    }
    chunk = _w9_chunk_with_source_refs(
        text=_CHUNK_TEXT_THREE_FACT,
        source_ids=["dart:doc1#blk_w8"],
    )
    # Even a contradicting DART text would be irrelevant — the dual-
    # source check MUST short-circuit BEFORE the DART NLI fan-out.
    dart_premise = "DART_W8_SKIP_BLOCK_TEXT_CONTRADICTS"
    dart_block_text_map = {"dart:doc1#blk_w8": dart_premise}

    seen_premises: List[str] = []

    def _score(premise, hypothesis):
        seen_premises.append(premise)
        if premise == dart_premise:
            return (0.05, 0.05, 0.90)  # would contradict if scored
        return (0.85, 0.10, 0.05)  # IMSCC entailed

    stub = _StubNliClassifier(score_fn=_score)
    validator = PairClaimSupportValidator(nli=stub)
    capture = _RecordingCapture()

    status, reason, new_fields = validator.validate_pair(
        pair,
        kind="instruction",
        chunk=chunk,
        decision_capture=capture,
        dart_block_text_map=dart_block_text_map,
        dual_source_severity="warning",
    )

    assert (status, reason) == ("validated", None)
    pcs = new_fields["per_claim_support"]
    assert isinstance(pcs, list) and len(pcs) == 2
    # No entry MUST carry a dart_source_check — deterministic pairs are
    # skipped before the DART NLI pass fires.
    for entry in pcs:
        assert "dart_source_check" not in entry, (
            "Deterministic-template pair MUST NOT receive "
            f"dart_source_check stamp; got entry={entry!r}"
        )
        # Outcome stays in the legacy 3-value set.
        assert entry["outcome"] in {
            "entailed", "unsupported", "contradicted",
        }, f"deterministic pair outcome stayed legacy; got {entry!r}"
    # No DART premise leaked into NLI scoring.
    assert dart_premise not in seen_premises, (
        f"deterministic-pair short-circuit MUST skip DART NLI; got "
        f"premises={seen_premises!r}"
    )
    # Rationale records dart_check_fired_count=0 — the audit trail
    # surfaces the skip.
    rationale = str(capture.events[0]["rationale"])
    assert "dart_check_fired_count=0" in rationale, (
        "deterministic-pair rationale MUST surface "
        f"dart_check_fired_count=0; got rationale={rationale!r}"
    )


def test_validate_walk_emits_dart_disagreement_rate_high_warning(
    tmp_path,
) -> None:
    """Wave 9 TIGHT gate-runner walk — when > 5% of pairs on disk
    carry a per-claim ``"dart_disagreement"`` outcome, the validator
    MUST emit a warning-severity ``DART_DISAGREEMENT_RATE_HIGH``
    GateIssue with the rate + total count interpolated into the
    message.

    Day-1 contract: warning severity does not block the gate; the
    overall ``passed`` flag stays True so promotion is unaffected.
    """
    from lib.validators.pair_claim_support import (
        PairClaimSupportValidator,
        _CODE_DART_DISAGREEMENT_RATE_HIGH,
        _DART_DISAGREEMENT_RATE_WARN_CEILING,
    )

    # Build 10 pairs: 2 carry dart_disagreement (20%) — well above the
    # 5% ceiling so the warning MUST fire.
    pairs_path = tmp_path / "instruction_pairs.jsonl"
    rows: List[Dict[str, Any]] = []
    for i in range(8):
        rows.append({
            "prompt": "x" * 50,
            "completion": "y" * 100,
            "chunk_id": f"ch{i}",
            "per_claim_support": [
                {
                    "sentence": "Filler sentence content.",
                    "entailment": 0.85,
                    "contradiction": 0.05,
                    "outcome": "entailed",
                },
            ],
            "claim_support_rate": 1.0,
        })
    for i in range(2):
        rows.append({
            "prompt": "x" * 50,
            "completion": "y" * 100,
            "chunk_id": f"ch_w9_{i}",
            "per_claim_support": [
                {
                    "sentence": "Filler with dart disagreement.",
                    "entailment": 0.85,
                    "contradiction": 0.05,
                    "outcome": "dart_disagreement",
                    "dart_source_check": {
                        "entailment": 0.05,
                        "contradiction": 0.85,
                        "outcome": "dart_disagreement",
                    },
                },
            ],
            "claim_support_rate": 0.0,
        })
    with pairs_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    validator = PairClaimSupportValidator()
    result = validator.validate({
        "instruction_pairs_path": str(pairs_path),
    })

    # Day-1 contract: warning severity → passed stays True (no
    # critical issue raised).
    assert result.passed, (
        "Wave 9 day-1 contract: aggregate-rate warning MUST NOT block "
        f"the gate; got result={result!r}"
    )
    # Find the warning issue.
    warn_issues = [
        i for i in result.issues
        if i.code == _CODE_DART_DISAGREEMENT_RATE_HIGH
    ]
    assert len(warn_issues) == 1, (
        f"expected exactly one DART_DISAGREEMENT_RATE_HIGH issue; "
        f"got issues={result.issues!r}"
    )
    issue = warn_issues[0]
    assert issue.severity == "warning"
    # Message interpolates the rate + count.
    msg = issue.message
    assert "2 of 10 pair(s)" in msg, (
        f"warning MUST interpolate count; got msg={msg!r}"
    )
    assert "20.0%" in msg, (
        f"warning MUST interpolate rate; got msg={msg!r}"
    )
    # Sanity — the ceiling default really is 5%.
    assert _DART_DISAGREEMENT_RATE_WARN_CEILING == 0.05
