"""GPT Feedback v2 Wave 1.7 W1.7.C — `BlockObjectiveDeliveryValidator`.

Tri-axis per-block-per-objective check that fires across the
``inter_tier_validation`` and ``post_rewrite_validation`` seams.

For every audited Block (block_type in :data:`_AUDITED_BLOCK_TYPES`)
and every objective_id declared on that Block, the validator runs three
axes:

1. **Statement entailment** (NLI): ``entail(block_text, objective_statement)``
   via :class:`lib.classifiers.nli_classifier.NliClassifier`. Threshold
   per-block-type via :data:`_PER_BLOCK_TYPE_ENTAILMENT_FLOOR`; falls
   through to :data:`_DEFAULT_ENTAILMENT_FLOOR` (0.40). A miss requires
   BOTH ``entailment < threshold`` AND ``contradiction > _CONTRADICTION_FLOOR``
   (0.50) — the contradiction guard filters surface-form variance from
   genuine pedagogical misses.

2. **Bloom alignment**: ``bloom_gap = BLOOM_LEVELS.index(declared) -
   BLOOM_LEVELS.index(observed_block_bloom)``. ``gap >= 2`` triggers
   :data:`_CODE_BLOOM_UNDERMET`. ``gap == 1`` is tolerated scaffolding
   (concept-block at ``understand`` for an ``apply`` objective is the
   prerequisite-scaffolding pattern). When ``block.observed_bloom_level``
   is None (legacy blocks pre-W1, or blocks outside the BERT ensemble's
   audit set per Drift C), the axis silently skips for that block —
   captured as ``unverifiable`` in the persisted alignment field.

3. **Action-verb cooccurrence**: search the block prose for any
   whole-word match against the FULL Bloom-level synonym set from
   :func:`lib.ontology.bloom.get_verbs`. Zero matches → emit
   :data:`_CODE_VERB_ABSENT`. The synonym-set check (vs literal verb
   match) avoids false positives where the block legitimately uses
   ``judge`` / ``critique`` / ``assess`` for an ``evaluate``-level
   objective.

All three axes emit warning-severity GateIssues; ``action="regenerate"``
fires when ANY axis misses. Day-1 the validator does NOT promote to
critical — calibration of per-block-type entailment thresholds against
the RDF/SHACL calibration corpus is a Wave 3 follow-up before promotion.

Graceful-degrade contract (mirrors
:mod:`lib.validators.bloom_classifier_disagreement` and
:mod:`lib.validators.objective_assessment_similarity`):

* NLI loader missing → emit :data:`_CODE_NLI_DEPS_MISSING` warning,
  ``passed=True, action=None``. Skip entailment axis only; Bloom + verb
  axes still run.
* ``inputs["objectives"]`` AND ``inputs["objective_statements"]``
  missing → emit :data:`_CODE_STATEMENT_UNRESOLVED` warning, skip the
  block, ``passed=True, action=None``.
* ``objectives`` missing but ``objective_statements`` present →
  entailment axis can run on the statement map; verb axis cannot
  (no ``bloom_verb`` / ``bloom_level`` to look up the synonym set).
  Emit :data:`_CODE_VERB_AXIS_UNAVAILABLE` warning, skip the verb axis
  for that pair only.
* ``block.observed_bloom_level`` missing → Bloom axis silently skipped,
  status captured as ``unverifiable``. No issue emitted unless every
  audited block has it None and a future strict-mode opt-in is set.

When ``inputs["return_touched_blocks"]=True`` the ``GateResult`` is
returned alongside an ``inputs["_touched_blocks"]`` list of Block
instances carrying the populated :attr:`Courseforge.scripts.blocks.Block.
objective_alignment` tuple — one entry per audited (block, objective_id)
pair so a postmortem reader sees per-objective delivery per block.

Cross-references:

* W1.7.A landed the ``Block.objective_alignment`` audit field +
  JSON-LD emit (``schemas/knowledge/courseforge_jsonld_v1.schema.json::
  $defs.Block``).
* W1.7.B landed the outline / rewrite prompt directives surfacing the
  Bloom triple ``[Bloom: <level>, verb: <verb>]`` and the behavioral
  outcome statement.
* W1.7.D lands the rewrite-tier remediation suffix (per-axis branch on
  the GateIssue codes this validator emits).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.nli_classifier import NliClassifier, NliScore
from lib.ontology.bloom import BLOOM_LEVELS, get_verbs

# Reuse the canonical text-extraction helper from the BERT-disagreement
# validator. The plan calls this out explicitly: "REUSE, don't fork".
# Underscore prefix is leading but Python doesn't enforce it; the
# contract is documented in
# ``bloom_classifier_disagreement.py:182-211``'s docstring and shared
# across the two validator surfaces by design.
from lib.validators.bloom.classifier_disagreement import (
    _extract_text_for_classification,
)

# W-D7 T7.8: route per-axis scoring through the shared
# ``lib.utils.objective_delivery_axes`` package. Bodies of the three axes
# now live in one module each, deduplicating the W4.C MEDIUM mirror in
# ``lib/validators/pair/objective_delivery.py``. The validator below
# stamps surface-specific GateIssues + alignment_entries from the shared
# axes' tuple outputs.
from lib.utils.objective_delivery_axes import (
    BLOOM_INDEX as _SHARED_BLOOM_INDEX,
    resolve_status as _shared_resolve_status,
    score_bloom_gap_axis,
    score_nli_axis,
    score_verb_cooccurrence_axis,
)

# IB3.2 — verb-triple EQUALITY axis (the framework keystone). Reuses the
# single-source-of-truth verb-band resolver so the objective/interaction/
# assessment triple is computed ONE way across every IB3 gate.
from lib.validators.alignment.verb_triple import (
    alignment_verb_triple_enabled,
    bloom_below,
    resolve_block_verb_level,
    resolve_objective_verb_level,
    verb_triple_misaligned,
)

logger = logging.getLogger(__name__)


#: Block types whose declared ``objective_ids`` are pedagogically
#: load-bearing AND whose surface text is rich enough to audit
#: against the objective statement. 6 of 16 block_types — see plan
#: §2 Fix 1.7.C "Per-block-type audited set" table for the rationale
#: per-type. ``objective`` is intentionally excluded (cyclic — the
#: objective IS the source of truth); ``misconception`` is intentionally
#: excluded (antagonistic to the objective by design).
_AUDITED_BLOCK_TYPES: frozenset = frozenset(
    {
        "concept",
        "example",
        "explanation",
        "assessment_item",
        "activity",
        "self_check_question",
    }
)


#: Default entailment floor — used when the per-block-type table is
#: missing the block_type (defensive against future block_type
#: additions before the threshold table is updated).
_DEFAULT_ENTAILMENT_FLOOR: float = 0.40


#: Per-block-type entailment thresholds (plan §6.1 calibration).
#: Day-1 starting points; calibrate against the RDF/SHACL calibration corpus
#: rebuild before promoting severity to critical (Wave 3 follow-up).
#:
#: Rationale per type:
#:   - concept: definitional surface; the concept-block legitimately
#:     scaffolds a downstream apply/analyze objective without entailing
#:     it. Lowest floor.
#:   - example, activity: worked examples / practice; moderate.
#:   - explanation: prose-rich; expected to entail.
#:   - self_check_question: question-shape; matches assessment_item
#:     calibration but slightly looser to admit informal stems.
#:   - assessment_item: tightest floor — matches the existing
#:     `objective_assessment_similarity` gate's 0.55 cosine floor for
#:     symmetry with the calibration that already shipped.
_PER_BLOCK_TYPE_ENTAILMENT_FLOOR: Dict[str, float] = {
    "concept": 0.30,
    "example": 0.40,
    "explanation": 0.45,
    "activity": 0.40,
    "self_check_question": 0.50,
    "assessment_item": 0.55,
}


#: Contradiction floor — entailment misses only fire when contradiction
#: is ALSO above this threshold. Filters surface-form variance from
#: genuine misses; see plan §2 Fix 1.7.C step 1.
_CONTRADICTION_FLOOR: float = 0.50


#: Cap the number of per-block issues emitted so a uniformly-broken
#: outline batch doesn't drown the gate report. Mirrors
#: ``Courseforge/router/inter_tier_gates.py::_ISSUE_LIST_CAP``.
_ISSUE_LIST_CAP: int = 50


# --------------------------------------------------------------------- #
# Canonical GateIssue codes
# --------------------------------------------------------------------- #

_CODE_STATEMENT_UNDERSUPPORTED: str = "BLOCK_OBJECTIVE_STATEMENT_UNDERSUPPORTED"
_CODE_BLOOM_UNDERMET: str = "BLOCK_OBJECTIVE_BLOOM_UNDERMET"
_CODE_VERB_ABSENT: str = "BLOCK_OBJECTIVE_VERB_ABSENT"
_CODE_NLI_DEPS_MISSING: str = "BLOCK_OBJECTIVE_NLI_DEPS_MISSING"
_CODE_OBSERVED_BLOOM_UNAVAILABLE: str = "BLOCK_OBJECTIVE_OBSERVED_BLOOM_UNAVAILABLE"
_CODE_STATEMENT_UNRESOLVED: str = "BLOCK_OBJECTIVE_STATEMENT_UNRESOLVED"
_CODE_VERB_AXIS_UNAVAILABLE: str = "BLOCK_OBJECTIVE_VERB_AXIS_UNAVAILABLE"
# IB3.2 — verb-triple EQUALITY mismatch (the framework keystone). Distinct
# from _CODE_VERB_ABSENT (axis 3: any band-synonym PRESENT in prose); this
# fires when the block's RESOLVED verb does not share the OBJECTIVE's Bloom
# band — verb EQUALITY across the triple, not mere presence. Gated behind
# ED4ALL_ALIGNMENT_VERB_TRIPLE (default OFF). Warning day-1.
_CODE_VERB_TRIPLE_MISMATCH: str = "BLOCK_OBJECTIVE_VERB_TRIPLE_MISMATCH"

# IB3.2 — block types whose RESOLVED verb the framework's verb-triple
# compares against the objective. Interaction/activity types (what the
# learner practices) + the assessment type (what certifies). assessment_item
# additionally carries the Alignment-cap-at-1 signal when its Bloom is below
# the objective's.
_VERB_TRIPLE_INTERACTION_TYPES: frozenset = frozenset(
    {"activity", "self_check_question", "scenario", "problem"}
)
_VERB_TRIPLE_ASSESSMENT_TYPES: frozenset = frozenset({"assessment_item"})


#: Canonical alignment-status enum that lands in
#: ``Block.objective_alignment[i].status`` per audited pair. Mirrors
#: ``schemas/knowledge/courseforge_jsonld_v1.schema.json::
#: $defs.ObjectiveAlignmentEntry.properties.status.enum``.
_STATUS_DELIVERED: str = "delivered"
_STATUS_UNDERDELIVERED: str = "underdelivered"
_STATUS_VERB_ONLY: str = "verb_only"
_STATUS_UNVERIFIABLE: str = "unverifiable"


# Re-export the canonical Bloom-level → index map from the shared axes
# package so existing intra-module references (and any
# back-compat-private callers) keep resolving.
_BLOOM_INDEX: Dict[str, int] = _SHARED_BLOOM_INDEX


def _block_attr(block: Any, key: str) -> Any:
    """Get ``block.<key>`` for dataclass blocks OR ``block[<key>]`` for dicts.

    Mirror of the helper in
    :mod:`lib.validators.bloom_classifier_disagreement`. Lets the
    validator stay shape-agnostic over the dataclass / dict round-trip.
    """
    if hasattr(block, key):
        return getattr(block, key)
    if isinstance(block, dict):
        return block.get(key)
    return None


def _resolve_objective_ids(block: Any) -> List[str]:
    """Pull the canonical objective_id list off a block.

    Same shape-tolerance as
    :func:`lib.validators.objective_assessment_similarity._resolve_objective_ids`
    — prefers the structural ``Block.objective_ids`` tuple, falls back
    to ``content["objective_ids"]`` / ``content["objective_refs"]`` on
    outline-tier dicts.
    """
    structural = list(_block_attr(block, "objective_ids") or ())
    if structural:
        return [o for o in structural if isinstance(o, str) and o]
    content = _block_attr(block, "content")
    if isinstance(content, dict):
        raw = (
            content.get("objective_ids")
            or content.get("objective_refs")
            or []
        )
        return [o for o in raw if isinstance(o, str) and o]
    return []


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Any], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs["blocks"]``.

    Mirrors :func:`lib.validators.bloom_classifier_disagreement._coerce_blocks`.
    """
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances or block-shaped dicts."
            ),
        )
    if not isinstance(raw, list):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=(
                f"inputs['blocks'] must be a list; got {type(raw).__name__}."
            ),
        )
    return list(raw), None


def _resolve_threshold_table(
    inputs: Dict[str, Any],
) -> Dict[str, float]:
    """Resolve the per-block-type entailment threshold table.

    Operators can override per-gate via
    ``gate.config.thresholds.per_block_type_entailment_floor`` in
    ``config/workflows.yaml``; the executor merges ``gate.config`` into
    the inputs dict at ``executor.py:1442`` per the existing pattern.
    """
    raw = inputs.get("per_block_type_entailment_floor")
    if isinstance(raw, dict):
        merged = dict(_PER_BLOCK_TYPE_ENTAILMENT_FLOOR)
        for k, v in raw.items():
            try:
                merged[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return merged
    return dict(_PER_BLOCK_TYPE_ENTAILMENT_FLOOR)


def _emit_decision(
    capture: Any,
    *,
    block_id: Optional[str],
    objective_id: Optional[str],
    block_type: Optional[str],
    declared_bloom: Optional[str],
    observed_bloom: Optional[str],
    bloom_gap: Optional[int],
    statement_entailment_score: Optional[float],
    contradiction_score: Optional[float],
    verb_match_count: Optional[int],
    entailment_threshold: float,
    nli_loaded: bool,
    status: str,
    failure_codes: List[str],
    verb_triple_status: Optional[str] = None,
    objective_verb: Optional[str] = None,
    block_verb: Optional[str] = None,
    alignment_cap_at_1: bool = False,
) -> None:
    """Emit one ``block_objective_delivery_check`` decision per audited
    (block, objective_id) pair.

    Rationale interpolates dynamic signals enumerated in plan
    §2 Fix 1.7.C "Decision capture": block_id, objective_id,
    declared_bloom, observed_bloom, bloom_gap,
    statement_entailment_score, contradiction_score, verb_match_count,
    the per-block-type entailment threshold used, whether NLI deps
    were loaded, and the resolved status enum value.
    """
    if capture is None:
        return
    decision = "passed" if not failure_codes else f"failed:{','.join(failure_codes)}"
    ent_str = (
        f"{statement_entailment_score:.4f}"
        if statement_entailment_score is not None
        else "n/a"
    )
    con_str = (
        f"{contradiction_score:.4f}"
        if contradiction_score is not None
        else "n/a"
    )
    gap_str = str(bloom_gap) if bloom_gap is not None else "n/a"
    verb_str = str(verb_match_count) if verb_match_count is not None else "n/a"
    rationale = (
        f"Block-objective delivery check on Block {block_id or 'n/a'!r} "
        f"(block_type={block_type or 'n/a'!r}) for objective "
        f"{objective_id or 'n/a'!r}: declared_bloom={declared_bloom or 'n/a'!r}, "
        f"observed_bloom={observed_bloom or 'n/a'!r}, "
        f"bloom_gap={gap_str}, "
        f"statement_entailment_score={ent_str}, "
        f"contradiction_score={con_str}, "
        f"entailment_threshold={entailment_threshold:.4f}, "
        f"verb_match_count={verb_str}, "
        f"nli_loaded={nli_loaded}, "
        f"status={status}, "
        f"failure_codes={','.join(failure_codes) or 'none'}."
    )
    # IB3.2 — append the verb-triple signals to the rationale only when the
    # axis ran (flag on). Keeps the flag-OFF rationale byte-identical.
    if verb_triple_status is not None:
        rationale += (
            f" verb_triple_status={verb_triple_status}, "
            f"objective_verb={objective_verb or 'n/a'!r}, "
            f"block_verb={block_verb or 'n/a'!r}, "
            f"alignment_cap_at_1={alignment_cap_at_1}."
        )
    try:
        capture.log_decision(
            decision_type="block_objective_delivery_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "block_objective_delivery_check: %s",
            exc,
        )


def _resolve_status(
    *,
    entailment_passed: Optional[bool],
    bloom_passed: Optional[bool],
    verb_passed: Optional[bool],
) -> str:
    """Map per-axis pass/skip flags to the canonical status enum.

    W-D7 T7.8: thin wrapper around
    :func:`lib.utils.objective_delivery_axes.resolve_status`. Kept as a
    back-compat shim for the cross-file import in
    ``lib/validators/pair/objective_delivery.py:140``
    (``from lib.validators.block_objective_delivery import _resolve_status``)
    AND any test that imports the private symbol directly.
    """
    return _shared_resolve_status(
        entailment_passed=entailment_passed,
        bloom_passed=bloom_passed,
        verb_passed=verb_passed,
    )


class BlockObjectiveDeliveryValidator:
    """Wave 1.7 W1.7.C — tri-axis per-block-per-objective delivery gate.

    Validator-protocol-compatible class wired into both
    ``inter_tier_validation::outline_block_objective_delivery`` and
    ``post_rewrite_validation::rewrite_block_objective_delivery``.
    """

    name = "block_objective_delivery"
    version = "1.0.0"

    def __init__(
        self,
        *,
        nli: Optional[NliClassifier] = None,
        contradiction_floor: float = _CONTRADICTION_FLOOR,
    ) -> None:
        # Test injection point. When None, the validator calls
        # ``NliClassifier.get_or_load()`` — which W1.7.C ships as a
        # STUB returning None (graceful-degrade exercised on every run
        # until W2.F lands the real loader).
        self._nli_override = nli
        self._contradiction_floor = float(contradiction_floor)

    def _get_nli(self) -> Optional[NliClassifier]:
        if self._nli_override is not None:
            return self._nli_override
        return NliClassifier.get_or_load()

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        return_touched = bool(inputs.get("return_touched_blocks", False))

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Empty input is a no-op pass.
        if not blocks:
            if return_touched:
                inputs["_touched_blocks"] = []
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        objective_statements: Dict[str, str] = (
            inputs.get("objective_statements") or {}
        )
        if not isinstance(objective_statements, dict):
            objective_statements = {}
        objectives_map: Dict[str, Dict[str, Any]] = (
            inputs.get("objectives") or {}
        )
        if not isinstance(objectives_map, dict):
            objectives_map = {}

        threshold_table = _resolve_threshold_table(inputs)
        nli = self._get_nli()
        nli_loaded = nli is not None

        verbs_by_level = get_verbs()  # Dict[str, Set[str]]

        # IB3.2 — the verb-triple equality axis is gated behind
        # ED4ALL_ALIGNMENT_VERB_TRIPLE (default OFF). When OFF the new axis is
        # skipped entirely AND scenario/problem stay outside the audited set,
        # so the three existing axes + the objective_alignment entry shape are
        # BYTE-IDENTICAL to the pre-IB3 baseline. When ON, scenario/problem
        # join the audited set so their RESOLVED verb is compared against the
        # objective's Bloom band.
        verb_triple_on = alignment_verb_triple_enabled()
        if verb_triple_on:
            audited_types = _AUDITED_BLOCK_TYPES | _VERB_TRIPLE_INTERACTION_TYPES
        else:
            audited_types = _AUDITED_BLOCK_TYPES

        issues: List[GateIssue] = []
        emitted_codes: set[str] = set()
        touched_blocks: List[Any] = []
        audited_pairs = 0
        passed_pairs = 0
        any_regenerate = False

        # Emit ONE NLI-deps-missing warning aggregate when the loader
        # is None — informational, ``passed=True`` per the
        # graceful-degrade contract. Per-pair we still fan out the
        # decision event so the audit trail records that the validator
        # ran on each block.
        if not nli_loaded:
            issues.append(
                GateIssue(
                    severity="warning",
                    code=_CODE_NLI_DEPS_MISSING,
                    message=(
                        "NliClassifier.get_or_load() returned None; "
                        "Wave-1.7 statement-entailment axis skipped. "
                        "Bloom + verb axes still run. W2.F lands the "
                        "real NLI loader."
                    ),
                    suggestion=(
                        "Wait for W2.F to land "
                        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli "
                        "loader, or set TRAINFORGE_REQUIRE_NLI=true (W2.F "
                        "follow-up) to fail-closed."
                    ),
                )
            )
            emitted_codes.add(_CODE_NLI_DEPS_MISSING)

        for block in blocks:
            block_type = _block_attr(block, "block_type")
            if block_type not in audited_types:
                # Non-audited block_type → silent skip (no event, no
                # issue). Mirrors the bloom-classifier-disagreement
                # validator's per-type filter.
                touched_blocks.append(block)
                continue

            block_id = _block_attr(block, "block_id")
            obj_ids = _resolve_objective_ids(block)
            if not obj_ids:
                # Block is in the audited type set but declares no
                # objective_refs. The page_objectives gate already covers
                # required-ref shape; this validator silently no-ops on
                # the per-block missing-refs path.
                touched_blocks.append(block)
                continue

            # Both inputs surfaces missing → emit one
            # STATEMENT_UNRESOLVED warning and skip the block. We dedupe
            # the code so a uniform-empty-inputs run doesn't drown the
            # report.
            if not objective_statements and not objectives_map:
                if _CODE_STATEMENT_UNRESOLVED not in emitted_codes:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=_CODE_STATEMENT_UNRESOLVED,
                            message=(
                                "Neither inputs['objective_statements'] "
                                "nor inputs['objectives'] is populated; "
                                "Wave-1.7 block-objective delivery gate "
                                "cannot resolve statements / Bloom levels. "
                                "Skipping all audited blocks."
                            ),
                            location=block_id,
                            suggestion=(
                                "Confirm gate-input routing populates "
                                "objective_statements + objectives from "
                                "synthesized_objectives.json (Drift A/B "
                                "fix lives in MCP/hardening/"
                                "gate_input_routing.py and "
                                "Courseforge/router/router.py)."
                            ),
                        )
                    )
                    emitted_codes.add(_CODE_STATEMENT_UNRESOLVED)
                _emit_decision(
                    capture,
                    block_id=block_id, objective_id=None,
                    block_type=block_type,
                    declared_bloom=None, observed_bloom=None,
                    bloom_gap=None,
                    statement_entailment_score=None,
                    contradiction_score=None,
                    verb_match_count=None,
                    entailment_threshold=threshold_table.get(
                        block_type, _DEFAULT_ENTAILMENT_FLOOR
                    ),
                    nli_loaded=nli_loaded,
                    status=_STATUS_UNVERIFIABLE,
                    failure_codes=[_CODE_STATEMENT_UNRESOLVED],
                )
                touched_blocks.append(block)
                continue

            # Resolve a textual surface for entailment + verb axes.
            block_text = _extract_text_for_classification(block)

            observed_bloom = _block_attr(block, "observed_bloom_level")
            entailment_threshold = threshold_table.get(
                block_type, _DEFAULT_ENTAILMENT_FLOOR
            )

            alignment_entries: List[Dict[str, Any]] = list(
                _block_attr(block, "objective_alignment") or ()
            )

            for obj_id in obj_ids:
                audited_pairs += 1
                obj_dict = objectives_map.get(obj_id) if objectives_map else None
                if not isinstance(obj_dict, dict):
                    obj_dict = {}

                statement = objective_statements.get(obj_id)
                if not isinstance(statement, str) or not statement.strip():
                    raw_stmt = obj_dict.get("statement")
                    if isinstance(raw_stmt, str) and raw_stmt.strip():
                        statement = raw_stmt
                    else:
                        statement = None

                declared_bloom = obj_dict.get("bloom_level")
                if isinstance(declared_bloom, str):
                    declared_bloom = declared_bloom.strip().lower() or None
                else:
                    declared_bloom = None

                bloom_verb = obj_dict.get("bloom_verb")
                if isinstance(bloom_verb, str):
                    bloom_verb = bloom_verb.strip().lower() or None
                else:
                    bloom_verb = None

                # ---------- Axis 1: NLI entailment --------------- #
                entailment_score: Optional[float] = None
                contradiction_score: Optional[float] = None
                entailment_passed: Optional[bool] = None  # None == skipped
                if nli_loaded and statement and block_text:
                    try:
                        score: NliScore = nli.score_pair(
                            premise=block_text, hypothesis=statement,
                        )
                        entailment_score = float(score.entailment)
                        contradiction_score = float(score.contradiction)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "NliClassifier.score_pair raised on "
                            "(%s, %s): %s",
                            block_id, obj_id, exc,
                        )
                        entailment_score = None
                        contradiction_score = None

                    if (
                        entailment_score is not None
                        and contradiction_score is not None
                    ):
                        if (
                            entailment_score < entailment_threshold
                            and contradiction_score > self._contradiction_floor
                        ):
                            entailment_passed = False
                            if (
                                _CODE_STATEMENT_UNDERSUPPORTED not in emitted_codes
                                or len(issues) < _ISSUE_LIST_CAP
                            ):
                                issues.append(
                                    GateIssue(
                                        severity="warning",
                                        code=_CODE_STATEMENT_UNDERSUPPORTED,
                                        message=(
                                            f"Block {block_id!r} "
                                            f"(block_type={block_type!r}) "
                                            f"prose did not entail objective "
                                            f"{obj_id!r}. NLI entailment="
                                            f"{entailment_score:.4f} "
                                            f"(floor "
                                            f"{entailment_threshold:.4f}); "
                                            f"contradiction="
                                            f"{contradiction_score:.4f} "
                                            f"(floor "
                                            f"{self._contradiction_floor:.4f})."
                                        ),
                                        location=block_id,
                                        suggestion=(
                                            "Re-emit prose that explicitly "
                                            "covers the objective's "
                                            "behavioral outcome."
                                        ),
                                    )
                                )
                                emitted_codes.add(
                                    _CODE_STATEMENT_UNDERSUPPORTED
                                )
                            any_regenerate = True
                        else:
                            entailment_passed = True

                # ---------- Axis 2: Bloom alignment --------------- #
                bloom_passed: Optional[bool] = None
                bloom_gap: Optional[int] = None
                if (
                    declared_bloom is not None
                    and declared_bloom in _BLOOM_INDEX
                    and isinstance(observed_bloom, str)
                    and observed_bloom.lower() in _BLOOM_INDEX
                ):
                    obs_lower = observed_bloom.lower()
                    bloom_gap = (
                        _BLOOM_INDEX[declared_bloom] - _BLOOM_INDEX[obs_lower]
                    )
                    if bloom_gap >= 2:
                        bloom_passed = False
                        if (
                            _CODE_BLOOM_UNDERMET not in emitted_codes
                            or len(issues) < _ISSUE_LIST_CAP
                        ):
                            issues.append(
                                GateIssue(
                                    severity="warning",
                                    code=_CODE_BLOOM_UNDERMET,
                                    message=(
                                        f"Block {block_id!r} "
                                        f"(block_type={block_type!r}) "
                                        f"bloom_level={obs_lower!r} is "
                                        f"{bloom_gap} levels below the "
                                        f"declared objective {obj_id!r}'s "
                                        f"bloom_level={declared_bloom!r}. "
                                        f"Re-emit at or above the declared "
                                        f"level — scaffold up to the "
                                        f"declared cognitive demand."
                                    ),
                                    location=block_id,
                                    suggestion=(
                                        f"Author block prose that demands "
                                        f"{declared_bloom!r}-level cognition "
                                        f"(or accept the scaffolding gap "
                                        f"and update the objective's "
                                        f"bloom_level)."
                                    ),
                                )
                            )
                            emitted_codes.add(_CODE_BLOOM_UNDERMET)
                        any_regenerate = True
                    else:
                        bloom_passed = True

                # ---------- Axis 3: Action-verb cooccurrence ------ #
                verb_passed: Optional[bool] = None
                verb_match_count: Optional[int] = None
                synonym_set: set = set()
                if declared_bloom is not None and declared_bloom in verbs_by_level:
                    synonym_set = set(verbs_by_level[declared_bloom])
                    # Canonical verb on the objective stays in the set
                    # even when not in the level's default canonical
                    # subset (defensive against synonym-set drift).
                    if bloom_verb:
                        synonym_set.add(bloom_verb)

                if synonym_set and block_text:
                    lowered = block_text.lower()
                    matches = 0
                    for v in synonym_set:
                        if not v:
                            continue
                        if re.search(rf"\b{re.escape(v)}\b", lowered):
                            matches += 1
                    verb_match_count = matches
                    if matches == 0:
                        verb_passed = False
                        if (
                            _CODE_VERB_ABSENT not in emitted_codes
                            or len(issues) < _ISSUE_LIST_CAP
                        ):
                            preview = ", ".join(
                                sorted(synonym_set)[:8]
                            )
                            issues.append(
                                GateIssue(
                                    severity="warning",
                                    code=_CODE_VERB_ABSENT,
                                    message=(
                                        f"Block {block_id!r} "
                                        f"(block_type={block_type!r}) "
                                        f"prose contains no synonym of the "
                                        f"objective {obj_id!r}'s "
                                        f"bloom_verb={bloom_verb!r}. "
                                        f"Re-emit prose using "
                                        f"{bloom_verb!r} or one of the "
                                        f"{declared_bloom!r}-level synonyms "
                                        f"(e.g. {preview})."
                                    ),
                                    location=block_id,
                                    suggestion=(
                                        f"Use a verb from "
                                        f"BLOOMS_VERBS[{declared_bloom!r}] "
                                        f"in the block's prose."
                                    ),
                                )
                            )
                            emitted_codes.add(_CODE_VERB_ABSENT)
                        any_regenerate = True
                    else:
                        verb_passed = True
                elif declared_bloom is None and bloom_verb is None:
                    # No declared bloom data at all — verb axis can't
                    # run. Single warning per gate run for visibility.
                    if _CODE_VERB_AXIS_UNAVAILABLE not in emitted_codes:
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code=_CODE_VERB_AXIS_UNAVAILABLE,
                                message=(
                                    "inputs['objectives'] missing "
                                    "bloom_verb / bloom_level for at "
                                    "least one referenced objective; "
                                    "verb-axis check skipped for those "
                                    "pairs."
                                ),
                                location=block_id,
                            )
                        )
                        emitted_codes.add(_CODE_VERB_AXIS_UNAVAILABLE)

                # ---------- Axis 4 (IB3.2): verb-triple EQUALITY ---- #
                # Gated behind ED4ALL_ALIGNMENT_VERB_TRIPLE. Distinct from
                # axis 3 (presence of ANY band-synonym): this asserts the
                # block's RESOLVED verb shares the OBJECTIVE's Bloom band —
                # objective-verb = activity/assessment-verb. Only fires for
                # interaction/activity + assessment block types.
                verb_triple_status: Optional[str] = None
                obj_verb_for_entry: Optional[str] = None
                blk_verb_for_entry: Optional[str] = None
                obj_level_for_entry: Optional[str] = None
                blk_level_for_entry: Optional[str] = None
                alignment_cap_at_1 = False
                if verb_triple_on and (
                    block_type in _VERB_TRIPLE_INTERACTION_TYPES
                    or block_type in _VERB_TRIPLE_ASSESSMENT_TYPES
                ):
                    obj_verb, obj_level = resolve_objective_verb_level(obj_dict)
                    blk_verb, blk_level = resolve_block_verb_level(block)
                    obj_verb_for_entry = obj_verb
                    blk_verb_for_entry = blk_verb
                    obj_level_for_entry = obj_level
                    blk_level_for_entry = blk_level
                    # IB3 recalibration (2-corpus gap-map): the strict
                    # band-EQUALITY fire (verbs_share_band-negation) tripped
                    # ~32% mostly as FALSE POSITIVES — blocks resolving ABOVE
                    # the objective band (the math-idiom prose inflation +
                    # genuinely higher-order blocks that still serve the
                    # objective). The mismatch fire is now ASYMMETRIC: ONLY a
                    # block whose resolved Bloom is strictly BELOW the
                    # objective band fires (genuine UNDER-delivery — the real
                    # constructive-alignment defect: you cannot certify a
                    # cognitive demand with evidence collected at a LOWER
                    # demand). Over-shoot (block band >= objective band, equal
                    # OR above) is ALIGNED — a higher-order block still serves
                    # the objective and over-shoot never fraudulently
                    # certifies. See verb_triple_misaligned() for the full
                    # rationale.
                    if obj_level is None or blk_level is None:
                        verb_triple_status = "unverifiable"
                    elif not verb_triple_misaligned(obj_level, blk_level):
                        # At or ABOVE the objective band → aligned (equal band
                        # OR pedagogically-acceptable over-shoot).
                        verb_triple_status = "aligned"
                    else:
                        verb_triple_status = "mismatch"
                        if (
                            _CODE_VERB_TRIPLE_MISMATCH not in emitted_codes
                            or len(issues) < _ISSUE_LIST_CAP
                        ):
                            issues.append(
                                GateIssue(
                                    severity="warning",
                                    code=_CODE_VERB_TRIPLE_MISMATCH,
                                    message=(
                                        f"Block {block_id!r} "
                                        f"(block_type={block_type!r}) "
                                        f"exercises verb {blk_verb!r} "
                                        f"(bloom={blk_level!r}) but objective "
                                        f"{obj_id!r} declares verb {obj_verb!r} "
                                        f"(bloom={obj_level!r}). The "
                                        f"constructive-alignment verb-triple "
                                        f"requires the practiced/certified "
                                        f"verb to reach the objective's Bloom "
                                        f"band — a "
                                        f"{obj_level!r} objective UNDER-served "
                                        f"by a lower-order {blk_level!r} block "
                                        f"cannot certify the objective "
                                        f"(over-shoot is tolerated; "
                                        f"under-delivery is not)."
                                    ),
                                    location=block_id,
                                    suggestion=(
                                        f"Re-author the block to exercise a "
                                        f"verb at-or-above the {obj_level!r} "
                                        f"band (matching the objective's "
                                        f"{obj_verb!r}), or re-route to a "
                                        f"block type that can."
                                    ),
                                )
                            )
                            emitted_codes.add(_CODE_VERB_TRIPLE_MISMATCH)
                        any_regenerate = True
                    # Alignment-cap-at-1 signal — assessment Bloom below the
                    # objective's invalidates certification (IB6 applies the
                    # cap; IB3 records the signal). Only meaningful for the
                    # assessment type.
                    if (
                        block_type in _VERB_TRIPLE_ASSESSMENT_TYPES
                        and bloom_below(blk_level, obj_level)
                    ):
                        alignment_cap_at_1 = True

                status = _resolve_status(
                    entailment_passed=entailment_passed,
                    bloom_passed=bloom_passed,
                    verb_passed=verb_passed,
                )

                pair_failure_codes: List[str] = []
                if entailment_passed is False:
                    pair_failure_codes.append(_CODE_STATEMENT_UNDERSUPPORTED)
                if bloom_passed is False:
                    pair_failure_codes.append(_CODE_BLOOM_UNDERMET)
                if verb_passed is False:
                    pair_failure_codes.append(_CODE_VERB_ABSENT)
                if verb_triple_status == "mismatch":
                    pair_failure_codes.append(_CODE_VERB_TRIPLE_MISMATCH)

                if not pair_failure_codes:
                    passed_pairs += 1

                # Persist the per-pair audit entry on the block.
                entry: Dict[str, Any] = {
                    "objective_id": obj_id,
                    "status": status,
                    "statement_entailment_score": (
                        entailment_score
                        if entailment_score is not None
                        else None
                    ),
                    "contradiction_score": (
                        contradiction_score
                        if contradiction_score is not None
                        else None
                    ),
                    "bloom_gap": bloom_gap,
                    "verb_match_count": verb_match_count,
                    "declared_bloom": declared_bloom,
                    "observed_bloom": (
                        observed_bloom.lower()
                        if isinstance(observed_bloom, str)
                        else None
                    ),
                    "entailment_threshold": entailment_threshold,
                }
                # IB3.2 — stamp the verb-triple signals ONLY when the flag is
                # on (and the axis ran). Flag-OFF → no new keys → entry shape
                # byte-identical to the pre-IB3 baseline. ``alignment_cap_at_1``
                # is added only when True (the IB6 rollup reads its presence).
                if verb_triple_status is not None:
                    entry["verb_triple_status"] = verb_triple_status
                    entry["objective_verb"] = obj_verb_for_entry
                    entry["block_verb"] = blk_verb_for_entry
                    entry["objective_bloom"] = obj_level_for_entry
                    entry["block_bloom"] = blk_level_for_entry
                    if alignment_cap_at_1:
                        entry["alignment_cap_at_1"] = True
                alignment_entries.append(entry)

                _emit_decision(
                    capture,
                    block_id=block_id,
                    objective_id=obj_id,
                    block_type=block_type,
                    declared_bloom=declared_bloom,
                    observed_bloom=(
                        observed_bloom.lower()
                        if isinstance(observed_bloom, str)
                        else None
                    ),
                    bloom_gap=bloom_gap,
                    statement_entailment_score=entailment_score,
                    contradiction_score=contradiction_score,
                    verb_match_count=verb_match_count,
                    entailment_threshold=entailment_threshold,
                    nli_loaded=nli_loaded,
                    status=status,
                    failure_codes=pair_failure_codes,
                    verb_triple_status=verb_triple_status,
                    objective_verb=obj_verb_for_entry,
                    block_verb=blk_verb_for_entry,
                    alignment_cap_at_1=alignment_cap_at_1,
                )

            # Persist the populated alignment field back onto the block
            # (for the return_touched_blocks=True opt-in path).
            if return_touched and alignment_entries:
                try:
                    import dataclasses as _dc
                    if _dc.is_dataclass(block):
                        touched_blocks.append(
                            _dc.replace(
                                block,
                                objective_alignment=tuple(alignment_entries),
                            )
                        )
                        continue
                except Exception:  # noqa: BLE001 — best-effort persist
                    pass
                # Dict-shaped block — mutate in place under a copy.
                if isinstance(block, dict):
                    new_block = dict(block)
                    new_block["objective_alignment"] = list(alignment_entries)
                    touched_blocks.append(new_block)
                    continue
            touched_blocks.append(block)

        # Score = pass rate over audited (block, objective_id) pairs.
        score = (
            1.0
            if audited_pairs == 0
            else round(passed_pairs / audited_pairs, 4)
        )

        # Action: regenerate when any axis fired a real miss; else None.
        # ``passed`` stays True because every issue is warning-severity
        # by construction (Day-1 contract — promotion to critical
        # deferred per plan §6.1).
        action: Optional[str] = "regenerate" if any_regenerate else None

        if return_touched:
            inputs["_touched_blocks"] = touched_blocks

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=score,
            issues=issues,
            action=action,
        )


__all__ = [
    "BlockObjectiveDeliveryValidator",
    "_AUDITED_BLOCK_TYPES",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_PER_BLOCK_TYPE_ENTAILMENT_FLOOR",
    "_CONTRADICTION_FLOOR",
    "_CODE_STATEMENT_UNDERSUPPORTED",
    "_CODE_BLOOM_UNDERMET",
    "_CODE_VERB_ABSENT",
    "_CODE_NLI_DEPS_MISSING",
    "_CODE_OBSERVED_BLOOM_UNAVAILABLE",
    "_CODE_STATEMENT_UNRESOLVED",
    "_CODE_VERB_AXIS_UNAVAILABLE",
    "_CODE_VERB_TRIPLE_MISMATCH",
    "_VERB_TRIPLE_INTERACTION_TYPES",
    "_VERB_TRIPLE_ASSESSMENT_TYPES",
]
