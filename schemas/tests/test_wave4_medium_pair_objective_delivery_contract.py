"""GPT Feedback v2 — Wave 4 MEDIUM end-of-wave pair-objective-delivery
contract gate.

Authored 2026-05-07 against the closing 6-test gate enumerated in
``plans/gpt-feedback-2-wave4-medium-pair-objective-delivery-2026-05.md``
§ 4.

Predecessors landing in parallel:

* W4.C.1 — :class:`lib.validators.pair_objective_delivery.
  PairObjectiveDeliveryValidator` (per-pair-per-objective tri-axis
  NLI / Bloom / verb delivery audit). Pair-side mirror of
  :class:`lib.validators.block_objective_delivery.
  BlockObjectiveDeliveryValidator` (Wave 1.7 W1.7.C). Inherits the
  same NLI-singleton, Bloom-gap, verb-cooccurrence helpers.
* W4.C.2 — schema migration adding ``pair_objective_alignment[]`` +
  ``pair_objective_alignment_pass_rate`` to
  ``schemas/knowledge/instruction_pair{,.strict}.schema.json`` and
  ``schemas/knowledge/preference_pair.schema.json``. Optional in
  strict mode day-1 (mirrors W4.A/W4.B migration shape).

Predecessor wave-end gates this file mirrors:

* ``test_wave4_pair_validation_contract.py`` (Wave 4 TIGHT) — THE
  TEMPLATE for the stub-NLI / stub-DecisionCapture wiring patterns
  and overall file structure.
* ``test_wave17_block_against_objective_contract.py`` (Wave 1.7) —
  the closest precedent for the per-axis tri-axis test patterns
  (status assertions, per-axis-only test cases, verb-absent dispatch).

This file is a SELF-CONTAINED contract test mirroring the Wave 1.5,
1.6, 1.7, Wave 4 TIGHT predecessor gates. Every fixture is inlined
locally so a future rename of a per-worker test file cannot silently
disable this wave-end gate.

Tests:

1. ``test_pair_objective_delivery_entails_passes`` — feed an instruction
   pair whose ``prompt + completion`` text covers the objective
   statement; pair ``bloom_level=apply``, ``observed_bloom=apply``,
   ``lo_refs=["TO-05"]``. Stub NLI returns ``entailment=0.85,
   contradiction=0.05``. Assert
   ``(status, reason) == ("validated", None)`` and
   ``new_fields["pair_objective_alignment"][0]["status"] ==
   "delivered"`` and ``new_fields["pair_objective_alignment_pass_rate"]
   == 1.0``.
2. ``test_pair_objective_delivery_rejects_undersupported_statement`` —
   same fixture, NLI returns ``entailment=0.10, contradiction=0.70``
   (below ``instruction_entailment_floor=0.40`` AND above
   ``contradiction_floor=0.50``). Assert
   ``(status, reason) == ("rejected", "objective_statement_undersupported")``.
3. ``test_pair_objective_delivery_rejects_bloom_gap`` — pair
   ``bloom_level=apply`` (declared via objectives map),
   ``observed_bloom=remember`` (BERT-classified pair attribute).
   ``BLOOM_LEVELS.index("apply") - BLOOM_LEVELS.index("remember") = 2``,
   gap=2 fires. NLI passes. Assert
   ``(status, reason) == ("rejected", "objective_bloom_undermet")``.
4. ``test_pair_objective_delivery_rejects_verb_absent`` — pair text
   contains zero ``apply``-level synonyms. NLI passes, Bloom passes,
   verb axis fires. Assert
   ``(status, reason) == ("rejected", "objective_verb_absent")``.
5. ``test_pair_objective_delivery_per_pair_kind_threshold_table`` —
   instantiate validator with default thresholds; for instruction
   (floor 0.40), preference (floor 0.45), and a deterministic-template
   pair (``template_id="kg_metadata.is_a"``, floor 0.30), feed an NLI
   score of exactly 0.35. Assert instruction + preference reject;
   deterministic-template accepts. Confirms the
   ``_PER_PAIR_KIND_ENTAILMENT_FLOOR`` table is consulted via the
   ``template_id``-prefix dispatcher.
6. ``test_wave4_medium_decision_capture_wiring`` — instantiate
   validator with a stub :class:`DecisionCapture`, run one pass + one
   NLI-fail + one Bloom-fail + one verb-fail, assert exactly 4 events,
   assert each event's ``decision_type ==
   "pair_objective_delivery_check"``, assert each rationale ≥20 chars
   AND interpolates ``pair_id`` + ``objective_id`` + ``declared_bloom``
   + ``entailment_threshold`` + ``status``.

Optional Test 7 (recommended, not part of the 6-test contract):

* ``test_wave4_medium_strict_pair_schema_round_trip`` — emit a pair
  through ``PairObjectiveDeliveryValidator.validate_pair`` then
  validate the resulting pair against
  ``instruction_pair.strict.schema.json``. Asserts no schema error
  on the populated W4.C fields. Defence-in-depth on the W4.C.2 schema
  migration. Mirrors the Wave 4 TIGHT round-trip test.

Drift notes vs plan §4:

* The W4.C.1 validator's per-pair-kind dispatch resolves
  ``deterministic_template`` via the ``template_id``-prefix matcher
  (per plan §3 ``_resolve_pair_kind`` lambda). Test 5 sets
  ``template_id="kg_metadata.is_a"`` which matches the
  ``"kg_metadata."`` prefix per plan §3 — accepted alternatives include
  ``"violation_detection."``, ``"abstention."``, and
  ``"schema_translation."``.
* Test 4 accepts EITHER ``"verb_only"`` OR ``"underdelivered"`` for
  the alignment-entry status, per the W1.7.C ``_resolve_status``
  semantics: when the verb axis is the ONLY passing axis the status
  is ``verb_only``; otherwise (entailment + Bloom passed, verb missed)
  the status is ``underdelivered``. The plan §4 text explicitly
  permits either.
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
    """Minimal ``NliScore``-shaped object exposing the three probability
    attributes the W4.C consumer reads.

    Inlined so the test file doesn't import the production
    :class:`NliScore` (which would couple the gate to that class's
    evolution). Mirrors the Wave 4 TIGHT precedent of stubbing the
    NLI surface for tests.
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

    The W4.C validator may call either ``nli.score_pair`` (single pair
    per (pair, lo_id) tuple — the W1.7.C surface) or
    ``nli.score_batch`` (the W4.A batched surface). The stub honours
    both contracts so the test is robust to either implementation
    choice.
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
    ``test_wave4_pair_validation_contract.py::_RecordingCapture``
    pattern (attribute name preserved for cross-wave grep symmetry).
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #


# Default objective for the bulk of the tests — apply-level objective
# whose statement carries the canonical apply-level verb (``apply``)
# so the verb axis passes when the pair text echoes the verb.
_OBJECTIVE_APPLY_TO_05 = {
    "id": "TO-05",
    "statement": "Apply RDF triples to model real-world scenarios.",
    "bloom_level": "apply",
    "bloom_verb": "apply",
}


# Verbatim-supported instruction pair text (passes the verb axis when
# objective is apply-level) — contains "apply" as a free-standing token.
_PAIR_TEXT_APPLY_VERBATIM = (
    "Apply RDF triples to model the relationship between subjects "
    "and objects in a real-world knowledge graph scenario."
)


# Pair text containing zero apply-level synonyms — for Test 4.
# apply-level Bloom verbs per ``schemas/taxonomies/bloom_verbs.json``
# include: ``apply``, ``demonstrate``, ``use``, ``solve``, ``implement``,
# ``execute``. This string deliberately avoids all of them while
# preserving topical relevance so the NLI axis still passes.
_PAIR_TEXT_NO_APPLY_VERB = (
    "The diagram describes the relationship between subjects and "
    "objects clearly."
)


def _instruction_pair(
    *,
    text: str = _PAIR_TEXT_APPLY_VERBATIM,
    pair_id: str = "pair_w4c_001",
    chunk_id: str = "ch1#sec1",
    lo_refs: Optional[List[str]] = None,
    bloom_level: str = "apply",
    observed_bloom: Optional[str] = "apply",
    template_id: Optional[str] = None,
    kind: str = "instruction",
) -> Dict[str, Any]:
    """Build a minimally-valid instruction-shaped pair dict.

    Includes only the fields the W4.C per-pair filter actually reads
    plus a ``pair_id`` for the decision-capture rationale. Keeps
    ``prompt`` short and ``completion`` carrying the bulk of the text
    so ``text = prompt + " " + completion`` (per plan §3) yields the
    intended hypothesis surface.

    Args:
        text: The full hypothesis text — split between ``prompt`` and
            ``completion``. The validator concatenates them per plan §3
            so the test doesn't care which surface carries each token.
        pair_id: Stable id surfaced into the decision-capture rationale.
        chunk_id: Chunk-id for the chunk-by-id lookup; only matters
            when the chunk text feeds the NLI premise (which the W4.C
            contract reads from ``objectives[lo_id]["statement"]``,
            NOT the chunk; see plan §3 axis 1).
        lo_refs: Pair's declared LO refs. Defaults to ``["TO-05"]``.
        bloom_level: Pair's declared Bloom level. Defaults to ``apply``.
        observed_bloom: BERT-ensemble-classified observed Bloom level.
            Defaults to ``apply`` (matches declared → no Bloom-axis
            miss). Set to ``"remember"`` for Test 3.
        template_id: Synthesis template id — set to a
            ``kg_metadata.*`` / ``violation_detection.*`` / etc. prefix
            to dispatch the deterministic-template entailment floor.
        kind: ``"instruction"`` or ``"preference"`` — preserved so the
            validator's ``_resolve_pair_kind`` dispatcher can read it.
    """
    if lo_refs is None:
        lo_refs = ["TO-05"]
    # Split text between prompt and completion so the concatenation
    # surfaces the full text. Prompt gets the leading sentence.
    parts = text.split(" ", 4)
    prompt = " ".join(parts[:4]) if len(parts) >= 4 else text
    completion = " ".join(parts[4:]) if len(parts) > 4 else text
    pair: Dict[str, Any] = {
        "pair_id": pair_id,
        "prompt": prompt,
        "completion": completion,
        "chosen": completion,  # for kind=preference symmetry
        "chunk_id": chunk_id,
        "lo_refs": list(lo_refs),
        "bloom_level": bloom_level,
        "observed_bloom": observed_bloom,
        "kind": kind,
    }
    if template_id is not None:
        pair["template_id"] = template_id
    return pair


def _objectives_map(
    *,
    objective: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build the per-pair objectives lookup map (per plan §3 axis 1).

    Defaults to the canonical apply-level TO-05 objective; pass an
    override to swap in a different objective for the per-axis tests.
    """
    obj = objective if objective is not None else _OBJECTIVE_APPLY_TO_05
    return {obj["id"]: dict(obj)}


def _chunk_payload(*, text: str = "Filler chunk body.") -> Dict[str, Any]:
    """Build a minimally-valid chunk dict.

    The W4.C validator reads ``objectives[lo_id]["statement"]`` as the
    NLI hypothesis and the pair's ``prompt + completion`` text as the
    premise; the chunk surface is consulted only for chunk-by-id
    lookup symmetry with the W4.A/W4.B precedent.
    """
    return {"text": text}


# --------------------------------------------------------------------------- #
# Test 1 — verbatim-entailed pair passes with all three axes green
# --------------------------------------------------------------------------- #


def test_pair_objective_delivery_entails_passes() -> None:
    """Plan §4 Test 1 — entailment-passes path.

    Build an instruction pair whose ``prompt + completion`` text
    verbatim covers the apply-level objective statement; ``bloom_level``
    + ``observed_bloom`` both ``apply`` (no Bloom gap); pair text
    contains ``apply`` (canonical apply-level verb → verb axis passes).
    Stub NLI returns ``entailment=0.85, contradiction=0.05`` — comfortably
    above the ``instruction`` floor 0.40 and below the contradiction
    floor 0.50, so the entailment axis passes.

    Assert ``(status, reason) == ("validated", None)`` and the per-LO
    alignment entry carries ``status="delivered"`` and the pair-level
    pass-rate is 1.0.
    """
    from lib.validators.pair_objective_delivery import (
        PairObjectiveDeliveryValidator,
    )

    pair = _instruction_pair()
    chunk = _chunk_payload()
    objectives = _objectives_map()

    stub = _StubNliClassifier(
        score_fn=lambda premise, hypothesis: (0.85, 0.10, 0.05),
    )
    validator = PairObjectiveDeliveryValidator(nli=stub)

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk, objectives=objectives,
    )

    assert (status, reason) == ("validated", None), (
        f"expected (validated, None); got ({status!r}, {reason!r}); "
        f"new_fields={new_fields!r}"
    )
    alignment = new_fields["pair_objective_alignment"]
    assert isinstance(alignment, list) and len(alignment) == 1, (
        f"expected exactly 1 pair_objective_alignment entry "
        f"(one objective); got {alignment!r}"
    )
    entry = alignment[0]
    assert entry["objective_id"] == "TO-05", (
        f"expected objective_id='TO-05'; got {entry['objective_id']!r}"
    )
    assert entry["status"] == "delivered", (
        f"expected status='delivered' on the all-axes-pass fixture; "
        f"got status={entry['status']!r}, full_entry={entry!r}"
    )
    assert new_fields["pair_objective_alignment_pass_rate"] == 1.0, (
        f"expected pair_objective_alignment_pass_rate=1.0 on "
        f"all-axes-pass; got "
        f"{new_fields['pair_objective_alignment_pass_rate']!r}"
    )


# --------------------------------------------------------------------------- #
# Test 2 — NLI entailment below floor + contradiction above floor rejects
# --------------------------------------------------------------------------- #


def test_pair_objective_delivery_rejects_undersupported_statement() -> None:
    """Plan §4 Test 2 — NLI-axis miss path.

    Same fixture as Test 1 but stub NLI returns
    ``entailment=0.10, contradiction=0.70`` — below the ``instruction``
    entailment floor 0.40 AND above the contradiction floor 0.50.
    Per plan §3 axis 1, a miss requires BOTH conditions; this fixture
    satisfies both.

    Assert ``(status, reason) == ("rejected",
    "objective_statement_undersupported")``. Assert the alignment entry
    carries ``status="underdelivered"`` (the validator's
    ``_resolve_status`` returns ``_STATUS_UNDERDELIVERED`` when the
    entailment axis genuinely missed and the verb / Bloom axes still
    passed). Assert the entailment score is preserved verbatim.
    """
    from lib.validators.pair_objective_delivery import (
        PairObjectiveDeliveryValidator,
    )

    pair = _instruction_pair()
    chunk = _chunk_payload()
    objectives = _objectives_map()

    stub = _StubNliClassifier(
        score_fn=lambda premise, hypothesis: (0.10, 0.20, 0.70),
    )
    validator = PairObjectiveDeliveryValidator(nli=stub)

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk, objectives=objectives,
    )

    assert (status, reason) == (
        "rejected", "objective_statement_undersupported",
    ), (
        f"expected (rejected, objective_statement_undersupported); "
        f"got ({status!r}, {reason!r}); new_fields={new_fields!r}"
    )
    alignment = new_fields["pair_objective_alignment"]
    assert isinstance(alignment, list) and len(alignment) == 1, (
        f"expected exactly 1 pair_objective_alignment entry; "
        f"got {alignment!r}"
    )
    entry = alignment[0]
    assert entry["status"] == "underdelivered", (
        f"expected status='underdelivered' on the NLI-axis-miss "
        f"fixture; got status={entry['status']!r}, "
        f"full_entry={entry!r}"
    )
    assert entry["statement_entailment_score"] == pytest.approx(0.10), (
        f"expected statement_entailment_score≈0.10 (preserved verbatim "
        f"from the stub); got "
        f"{entry['statement_entailment_score']!r}"
    )


# --------------------------------------------------------------------------- #
# Test 3 — Bloom-gap-≥2 fires (declared=apply, observed=remember; gap=2)
# --------------------------------------------------------------------------- #


def test_pair_objective_delivery_rejects_bloom_gap() -> None:
    """Plan §4 Test 3 — Bloom-axis miss path.

    Pair declared via objectives map at ``bloom_level=apply``;
    ``observed_bloom=remember`` (BERT-ensemble classification). Per the
    canonical Bloom enum
    (``lib/ontology/bloom.BLOOM_LEVELS``), ``apply`` is index 2 and
    ``remember`` is index 0, so ``bloom_gap = 2``. Per plan §3 axis 2,
    ``bloom_gap >= 2`` is a miss.

    NLI passes (``entailment=0.85, contradiction=0.05`` — well above /
    below the floors). The verb axis also passes — the pair text
    contains ``apply``. Pair text deliberately does NOT contain
    ``remember``-level synonyms so the verb axis lines up against the
    DECLARED ``apply`` level (per plan §3 axis 3, the synonym set is
    drawn from the declared bloom_level + bloom_verb).

    Assert ``(status, reason) == ("rejected", "objective_bloom_undermet")``,
    bloom_gap=2 in the alignment entry, status='underdelivered'.
    """
    from lib.validators.pair_objective_delivery import (
        PairObjectiveDeliveryValidator,
    )

    pair = _instruction_pair(observed_bloom="remember")
    chunk = _chunk_payload()
    objectives = _objectives_map()

    stub = _StubNliClassifier(
        score_fn=lambda premise, hypothesis: (0.85, 0.10, 0.05),
    )
    validator = PairObjectiveDeliveryValidator(nli=stub)

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk, objectives=objectives,
    )

    assert (status, reason) == ("rejected", "objective_bloom_undermet"), (
        f"expected (rejected, objective_bloom_undermet); "
        f"got ({status!r}, {reason!r}); new_fields={new_fields!r}"
    )
    alignment = new_fields["pair_objective_alignment"]
    assert isinstance(alignment, list) and len(alignment) == 1, (
        f"expected exactly 1 pair_objective_alignment entry; "
        f"got {alignment!r}"
    )
    entry = alignment[0]
    assert entry["bloom_gap"] == 2, (
        f"expected bloom_gap=2 (apply idx 2 - remember idx 0); "
        f"got bloom_gap={entry['bloom_gap']!r}"
    )
    assert entry["status"] == "underdelivered", (
        f"expected status='underdelivered' on the Bloom-axis-miss "
        f"fixture; got status={entry['status']!r}, "
        f"full_entry={entry!r}"
    )


# --------------------------------------------------------------------------- #
# Test 4 — verb-cooccurrence axis fires (zero apply-level synonyms)
# --------------------------------------------------------------------------- #


def test_pair_objective_delivery_rejects_verb_absent() -> None:
    """Plan §4 Test 4 — verb-axis miss path.

    Pair text contains zero ``apply``-level synonyms (no ``apply``,
    ``demonstrate``, ``use``, ``solve``, ``implement``, ``execute``,
    etc. per ``schemas/taxonomies/bloom_verbs.json``). NLI passes
    (entailment=0.85), Bloom passes (declared=apply, observed=apply),
    so ONLY the verb axis fires.

    Per the W1.7.C ``_resolve_status`` semantics this can resolve to
    EITHER ``"verb_only"`` (if the validator interprets the verb axis
    as the sole survivor — but it isn't, the other two passed) or
    ``"underdelivered"`` (the more typical case where verb is the only
    axis that missed while the other two passed). The plan §4 text
    explicitly accepts either status; the BLOCKING assertions are the
    rejection reason and the verb_match_count.

    Assert ``(status, reason) == ("rejected", "objective_verb_absent")``,
    ``verb_match_count == 0``, and ``status in {"verb_only",
    "underdelivered"}``.
    """
    from lib.validators.pair_objective_delivery import (
        PairObjectiveDeliveryValidator,
    )

    pair = _instruction_pair(text=_PAIR_TEXT_NO_APPLY_VERB)
    chunk = _chunk_payload()
    objectives = _objectives_map()

    stub = _StubNliClassifier(
        score_fn=lambda premise, hypothesis: (0.85, 0.10, 0.05),
    )
    validator = PairObjectiveDeliveryValidator(nli=stub)

    status, reason, new_fields = validator.validate_pair(
        pair, kind="instruction", chunk=chunk, objectives=objectives,
    )

    assert (status, reason) == ("rejected", "objective_verb_absent"), (
        f"expected (rejected, objective_verb_absent); "
        f"got ({status!r}, {reason!r}); new_fields={new_fields!r}"
    )
    alignment = new_fields["pair_objective_alignment"]
    assert isinstance(alignment, list) and len(alignment) == 1, (
        f"expected exactly 1 pair_objective_alignment entry; "
        f"got {alignment!r}"
    )
    entry = alignment[0]
    assert entry["verb_match_count"] == 0, (
        f"expected verb_match_count=0 on a fixture with zero "
        f"apply-level synonyms; got "
        f"verb_match_count={entry['verb_match_count']!r}"
    )
    # Per the W1.7.C ``_resolve_status`` semantics — accept either
    # "verb_only" (verb is the sole survivor) or "underdelivered"
    # (verb is the only miss). The plan §4 text explicitly permits
    # either.
    assert entry["status"] in {"verb_only", "underdelivered"}, (
        f"expected status in {{'verb_only', 'underdelivered'}} per "
        f"the W1.7.C _resolve_status semantics; got "
        f"status={entry['status']!r}, full_entry={entry!r}"
    )


# --------------------------------------------------------------------------- #
# Test 5 — per-pair-kind threshold-table dispatch
# --------------------------------------------------------------------------- #


def test_pair_objective_delivery_per_pair_kind_threshold_table() -> None:
    """Plan §4 Test 5 — ``_PER_PAIR_KIND_ENTAILMENT_FLOOR`` dispatch.

    Three pairs scored at NLI ``entailment=0.35`` (sits between the
    deterministic-template floor 0.30 and the instruction floor 0.40,
    and below the preference floor 0.45):

    * ``instruction`` (no ``template_id``) → floor 0.40 → 0.35 < 0.40
      → REJECT with ``objective_statement_undersupported``.
    * ``preference`` (no ``template_id``) → floor 0.45 → 0.35 < 0.45
      → REJECT with ``objective_statement_undersupported``.
    * ``instruction`` with ``template_id="kg_metadata.is_a"`` →
      ``_resolve_pair_kind`` returns ``"deterministic_template"`` →
      floor 0.30 → 0.35 ≥ 0.30 → ACCEPT (status="validated").

    Confirms the per-pair-kind dispatch table is consulted via the
    ``template_id``-prefix matcher per plan §3
    (``_resolve_pair_kind`` lambda). The ``kg_metadata.`` prefix is
    one of four canonical deterministic-template prefixes per plan §3
    axis 3 — the others (``violation_detection.``, ``abstention.``,
    ``schema_translation.``) follow the same dispatch path.

    Drift note: the contradiction score is held at 0.55 (above the
    contradiction floor 0.50) so the NLI-miss two-condition guard
    (per plan §3 axis 1: ``entailment < threshold AND contradiction >
    contradiction_floor``) fires for the failing pairs. If the
    contradiction were below 0.50 the entailment axis would
    silently pass per the W1.7.C two-condition guard, masking the
    threshold-dispatch test.
    """
    from lib.validators.pair_objective_delivery import (
        PairObjectiveDeliveryValidator,
    )

    chunk = _chunk_payload()
    objectives = _objectives_map()
    # Score: entailment=0.35 (between floors), contradiction=0.55
    # (above contradiction floor so the two-condition NLI miss fires).
    stub = _StubNliClassifier(
        score_fn=lambda premise, hypothesis: (0.35, 0.10, 0.55),
    )
    validator = PairObjectiveDeliveryValidator(nli=stub)

    # ---- Sub-test 5a: instruction (floor 0.40, 0.35 < 0.40 → reject) ---- #
    instr_pair = _instruction_pair(
        pair_id="pair_w4c_5a", kind="instruction",
    )
    instr_status, instr_reason, instr_new_fields = validator.validate_pair(
        instr_pair, kind="instruction", chunk=chunk, objectives=objectives,
    )
    assert (instr_status, instr_reason) == (
        "rejected", "objective_statement_undersupported",
    ), (
        f"instruction pair (entailment_floor=0.40) MUST reject at "
        f"NLI=0.35; got ({instr_status!r}, {instr_reason!r}); "
        f"new_fields={instr_new_fields!r}"
    )

    # ---- Sub-test 5b: preference (floor 0.45, 0.35 < 0.45 → reject) ---- #
    pref_pair = _instruction_pair(
        pair_id="pair_w4c_5b", kind="preference",
    )
    pref_status, pref_reason, pref_new_fields = validator.validate_pair(
        pref_pair, kind="preference", chunk=chunk, objectives=objectives,
    )
    assert (pref_status, pref_reason) == (
        "rejected", "objective_statement_undersupported",
    ), (
        f"preference pair (entailment_floor=0.45) MUST reject at "
        f"NLI=0.35; got ({pref_status!r}, {pref_reason!r}); "
        f"new_fields={pref_new_fields!r}"
    )

    # ---- Sub-test 5c: deterministic_template (floor 0.30, 0.35 ≥ 0.30
    # → accept) ---- #
    det_pair = _instruction_pair(
        pair_id="pair_w4c_5c",
        kind="instruction",
        template_id="kg_metadata.is_a",
    )
    det_status, det_reason, det_new_fields = validator.validate_pair(
        det_pair, kind="instruction", chunk=chunk, objectives=objectives,
    )
    assert det_status == "validated", (
        f"deterministic_template pair (template_id='kg_metadata.is_a' "
        f"→ entailment_floor=0.30) MUST accept at NLI=0.35; "
        f"got ({det_status!r}, {det_reason!r}); "
        f"new_fields={det_new_fields!r}"
    )
    # Defence-in-depth — the alignment entry surfaces the lower
    # entailment_threshold so a postmortem reader can confirm which
    # floor was applied.
    det_alignment = det_new_fields["pair_objective_alignment"]
    assert isinstance(det_alignment, list) and len(det_alignment) == 1, (
        f"expected exactly 1 pair_objective_alignment entry on the "
        f"deterministic-template pair; got {det_alignment!r}"
    )
    det_entry = det_alignment[0]
    assert det_entry["entailment_threshold"] == pytest.approx(0.30), (
        f"deterministic_template pair MUST surface "
        f"entailment_threshold=0.30 in the alignment entry "
        f"(confirms the threshold-table dispatch consulted the "
        f"deterministic_template floor, not the instruction default); "
        f"got entailment_threshold={det_entry['entailment_threshold']!r}"
    )


# --------------------------------------------------------------------------- #
# Test 6 — decision-capture wiring; rationale interpolates dynamic signals
# --------------------------------------------------------------------------- #


def test_wave4_medium_decision_capture_wiring() -> None:
    """Plan §4 Test 6 — decision-capture wiring contract.

    Run one pass + one NLI-fail + one Bloom-fail + one verb-fail with
    a stub :class:`DecisionCapture`. Assert:

    1. Exactly 4 events total (one per (pair, lo_id) tuple — each
       fixture carries exactly one LO ref).
    2. Every event's ``decision_type ==
       "pair_objective_delivery_check"``.
    3. Every rationale ≥20 chars per the canonical DecisionCapture
       contract.
    4. Every rationale interpolates the per-call dynamic signals
       enumerated in plan §3: ``pair_id`` + ``objective_id`` +
       ``declared_bloom`` + ``entailment_threshold`` + ``status``.

    Mirrors the W4.A/W4.B wiring test (Wave 4 TIGHT Test 6) verbatim.
    """
    from lib.validators.pair_objective_delivery import (
        PairObjectiveDeliveryValidator,
    )

    capture = _RecordingCapture()
    chunk = _chunk_payload()
    objectives = _objectives_map()

    # ---- Pass: all axes green. ---- #
    pass_stub = _StubNliClassifier(
        score_fn=lambda p, h: (0.85, 0.10, 0.05),
    )
    PairObjectiveDeliveryValidator(nli=pass_stub).validate_pair(
        _instruction_pair(pair_id="pair_w4c_pass"),
        kind="instruction", chunk=chunk, objectives=objectives,
        decision_capture=capture,
    )

    # ---- NLI-fail: entailment below floor + contradiction above floor. ---- #
    nli_fail_stub = _StubNliClassifier(
        score_fn=lambda p, h: (0.10, 0.20, 0.70),
    )
    PairObjectiveDeliveryValidator(nli=nli_fail_stub).validate_pair(
        _instruction_pair(pair_id="pair_w4c_nli_fail"),
        kind="instruction", chunk=chunk, objectives=objectives,
        decision_capture=capture,
    )

    # ---- Bloom-fail: declared=apply, observed=remember (gap=2). ---- #
    bloom_pass_stub = _StubNliClassifier(
        score_fn=lambda p, h: (0.85, 0.10, 0.05),
    )
    PairObjectiveDeliveryValidator(nli=bloom_pass_stub).validate_pair(
        _instruction_pair(
            pair_id="pair_w4c_bloom_fail", observed_bloom="remember",
        ),
        kind="instruction", chunk=chunk, objectives=objectives,
        decision_capture=capture,
    )

    # ---- Verb-fail: pair text contains zero apply-level synonyms. ---- #
    verb_pass_stub = _StubNliClassifier(
        score_fn=lambda p, h: (0.85, 0.10, 0.05),
    )
    PairObjectiveDeliveryValidator(nli=verb_pass_stub).validate_pair(
        _instruction_pair(
            pair_id="pair_w4c_verb_fail", text=_PAIR_TEXT_NO_APPLY_VERB,
        ),
        kind="instruction", chunk=chunk, objectives=objectives,
        decision_capture=capture,
    )

    # ---- Sub-clause 1 — exactly 4 events. ---- #
    assert len(capture.events) == 4, (
        f"expected exactly 4 decision events (1 per (pair, lo_id) "
        f"tuple × 4 invocations × 1 LO ref each); "
        f"got {len(capture.events)}: "
        f"events={[e.get('decision_type') for e in capture.events]!r}"
    )

    # ---- Sub-clause 2 — decision_type membership. ---- #
    for e in capture.events:
        assert e.get("decision_type") == "pair_objective_delivery_check", (
            f"expected decision_type='pair_objective_delivery_check' "
            f"on every event; got decision_type="
            f"{e.get('decision_type')!r}, full_event={e!r}"
        )

    # ---- Sub-clause 3 — every rationale ≥20 chars. ---- #
    for e in capture.events:
        rationale = str(e.get("rationale", ""))
        assert len(rationale) >= 20, (
            f"every rationale MUST be ≥20 chars per the canonical "
            f"DecisionCapture contract; got {len(rationale)} chars on "
            f"event {e!r}"
        )

    # ---- Sub-clause 4 — rationale interpolates dynamic signals. ---- #
    # Each rationale MUST surface pair_id + objective_id +
    # declared_bloom + entailment_threshold + status (per plan §3
    # decision-capture brief).
    pair_ids_seen: set = set()
    statuses_seen: set = set()
    for e in capture.events:
        rationale = str(e.get("rationale", ""))
        # objective_id verbatim — every fixture references TO-05.
        assert "TO-05" in rationale, (
            f"rationale MUST interpolate objective_id ('TO-05') "
            f"verbatim; got rationale={rationale!r}"
        )
        # declared_bloom interpolated — every fixture's objective is
        # apply-level.
        assert "apply" in rationale, (
            f"rationale MUST interpolate declared_bloom ('apply') "
            f"verbatim; got rationale={rationale!r}"
        )
        # entailment_threshold key surfaced — the validator stamps the
        # per-pair-kind floor used for the NLI-axis check.
        assert "entailment_threshold=" in rationale, (
            f"rationale MUST interpolate entailment_threshold=<v> "
            f"per plan §3 decision-capture brief; got "
            f"rationale={rationale!r}"
        )
        # status key surfaced — the resolved per-pair status enum
        # value (delivered / underdelivered / verb_only / unverifiable).
        assert "status=" in rationale, (
            f"rationale MUST interpolate status=<v> per plan §3 "
            f"decision-capture brief; got rationale={rationale!r}"
        )
        # pair_id verbatim — extract whichever of the four ids appears.
        for token in (
            "pair_w4c_pass",
            "pair_w4c_nli_fail",
            "pair_w4c_bloom_fail",
            "pair_w4c_verb_fail",
        ):
            if token in rationale:
                pair_ids_seen.add(token)
        # status enum value verbatim — record which appears for the
        # cross-rationale uniqueness assertion below.
        for token in (
            "delivered", "underdelivered", "verb_only", "unverifiable",
        ):
            if f"status={token}" in rationale:
                statuses_seen.add(token)

    # All four pair_ids MUST appear exactly once across the four
    # rationales — this is the dynamic-signal-interpolation contract:
    # a hard-coded rationale would not surface all four ids.
    assert pair_ids_seen == {
        "pair_w4c_pass",
        "pair_w4c_nli_fail",
        "pair_w4c_bloom_fail",
        "pair_w4c_verb_fail",
    }, (
        f"every pair_id MUST surface verbatim in its rationale "
        f"(dynamic-signal contract — a static rationale would not "
        f"interpolate the per-call pair_id); got pair_ids_seen="
        f"{pair_ids_seen!r}"
    )

    # At least 2 distinct status values across the 4 rationales — the
    # pass fixture stamps "delivered"; the three fail fixtures stamp
    # one of {"underdelivered", "verb_only"} per the W1.7.C
    # ``_resolve_status`` semantics. A static rationale would flatten
    # to a single status value.
    assert len(statuses_seen) >= 2, (
        f"expected at least 2 distinct status values across 4 "
        f"rationales (pass='delivered'; fails='underdelivered' or "
        f"'verb_only'); got statuses_seen={statuses_seen!r}"
    )


# --------------------------------------------------------------------------- #
# Optional Test 7 — strict-pair-schema round trip (recommended)
# --------------------------------------------------------------------------- #


def test_wave4_medium_strict_pair_schema_round_trip() -> None:
    """Plan §4 optional Test 7 — emit a pair through
    :meth:`PairObjectiveDeliveryValidator.validate_pair` then validate
    the resulting pair against
    ``instruction_pair{,.strict}.schema.json``. Asserts no schema
    error on the populated W4.C fields.

    Defence-in-depth on the W4.C.2 pair-schema migration: catches the
    regression class where a future schema author tightens the W4.C
    field shapes without re-validating against a real validator emit.

    Mirrors the Wave 4 TIGHT round-trip test
    (``test_wave4_strict_pair_schema_round_trip``) approach: build a
    schema-valid base pair carrying the strict-mode required fields,
    run W4.C validate_pair, merge the new_fields into the pair, strip
    any debug fields the W4.A precedent stripped (e.g., ``deps_missing``),
    and round-trip against both the lenient and strict pair schemas.
    """
    jsonschema = pytest.importorskip("jsonschema")
    from lib.validators.pair_objective_delivery import (
        PairObjectiveDeliveryValidator,
    )

    # Build a schema-valid base pair (minimum required fields per
    # ``instruction_pair.strict.schema.json::required``).
    base_pair: Dict[str, Any] = {
        "prompt": (
            "Apply the RDF triple model to a real-world scenario "
            "demonstrating subject-predicate-object structure."
        ),
        "completion": (
            "Apply RDF triples to model the relationship between "
            "subjects and objects in a real-world knowledge graph "
            "scenario. Each triple captures one assertion."
        ),
        "chunk_id": "ch1#sec1",
        "lo_refs": ["TO-05"],
        "bloom_level": "apply",
        "content_type": "explanation",
        "seed": 42,
        "decision_capture_id": "evt_w4c_round_trip",
        "observed_bloom": "apply",
    }
    chunk = _chunk_payload()
    objectives = _objectives_map()

    stub = _StubNliClassifier(
        score_fn=lambda p, h: (0.85, 0.10, 0.05),
    )
    _, _, new_fields = PairObjectiveDeliveryValidator(nli=stub).validate_pair(
        base_pair, kind="instruction", chunk=chunk, objectives=objectives,
    )
    base_pair.update(new_fields)

    # Strip any per-call debug fields the W4.A precedent stripped
    # before round-tripping (deps_missing is the canonical example —
    # it's per-call metadata the caller can drop before write).
    base_pair.pop("deps_missing", None)

    # Round-trip against both the lenient and strict pair schemas.
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
