"""GPT Feedback v2 Wave 4 MEDIUM W4.C.1 — :class:`PairObjectiveDeliveryValidator`.

Per-pair, per-objective semantic check that fans out every (pair, lo_id)
tuple emitted by the synthesizer. Closes Gap C from the Wave 4 TIGHT
investigation §3.3 ("pair-against-objective tri-axis (NLI + Bloom +
verb)") on the Trainforge pair surface — the symmetric mirror of the
Courseforge-side
:class:`lib.validators.block_objective_delivery.BlockObjectiveDeliveryValidator`
(Wave 1.7 W1.7.C). The Bloom + NLI + verb triple is the same algorithm;
only the surface (training pair vs Courseforge Block) differs.

Rationale: W2.E (`TrainingPairPromotionValidator`) covers the Bloom
axis partially via BERT-ensemble exact-match disagreement, but uses
exact-match semantics rather than the gap-≥-2-with-scaffolding-tolerance
from W1.7.C; does not run NLI entailment of ``pair_text →
objective_statement``; and does not run verb cooccurrence against
:func:`lib.ontology.bloom.get_verbs`. The eval-side
``bloom_alignment_rate`` (W3.F F4) catches under-delivery only POST-
training; W4.C is the symmetric pre-write gate so a pair tagged
``TO-05 (apply)`` whose completion never exemplifies (``understand``-
level prose) is dropped before it can poison the training corpus toward
``understand``-level responses on ``apply``-tier prompts.

Surface contract (mirrors W4.A / W4.B):

1. :meth:`validate_pair` — per-pair pre-write filter, called from
   :mod:`Trainforge.synthesize_training` immediately AFTER W4.B
   (:meth:`PairLearningOutcomeRefsValidator.validate_pair`) returns
   ``"validated"``. Returns
   ``(promotion_status, rejection_reason, new_fields)``. The caller
   ``pair.update(new_fields)`` so the audit signals
   (``pair_objective_alignment`` + ``pair_objective_alignment_pass_rate``)
   land regardless of pass/fail.
2. :meth:`validate` — gate-runner surface, walks
   ``training_specs/{instruction,preference}_pairs.jsonl`` post-write
   and asserts every pair carries ``pair_objective_alignment`` (list-or
   -None) + ``pair_objective_alignment_pass_rate``. Same shape as
   :meth:`PairClaimSupportValidator.validate` (presence audit, not a
   re-run of the per-pair NLI fan-out).

Algorithm (mirrors
:meth:`BlockObjectiveDeliveryValidator.validate` lines 391-907):

For each pair, for each ``lo_id`` in ``pair["lo_refs"]`` (with
``learning_outcome_refs`` fallback per Wave 4 TIGHT lesson 1, mirrored
in :class:`PairLearningOutcomeRefsValidator`):

1. Resolve objective metadata from ``objectives[lo_id]``. Missing entry
   → emit :data:`_CODE_OBJECTIVE_NOT_RESOLVED` warning, status
   ``unverifiable``, skip all axes for this (pair, lo_id) tuple.
2. Resolve pair text. ``kind="instruction"`` →
   ``text = prompt + " " + completion``. ``kind="preference"`` →
   ``text = prompt + " " + chosen``. Empty text → status
   ``unverifiable``, skip.
3. **Axis 1 (instantiated-skill semantics)**: score NLI in both
   directions between the pair and the competency statement, and use
   the stronger entailment signal. A training pair is an *instance* of
   a competency, not a proposition that must logically entail the
   complete competency statement. The reverse direction captures the
   specialization relation without weakening the existing threshold.
   Threshold from
   :data:`_PER_PAIR_KIND_ENTAILMENT_FLOOR` table (instruction=0.40,
   preference=0.45, deterministic_template=0.30). Miss =
   ``entailment < threshold``.
4. **Axis 2 (Bloom gap)**: ``bloom_gap = BLOOM_LEVELS.index(declared)
   - BLOOM_LEVELS.index(observed_bloom)``. ``observed_bloom =
   pair.get("observed_bloom")``. When ``observed_bloom is None``, axis
   silently skipped. Miss = ``gap >= _BLOOM_GAP_THRESHOLD`` (2).
5. **Action realization evidence**: literal Bloom-verb matches are
   recorded for audit only. They are not a validity condition. The
   observed-Bloom classifier evaluates the cognitive work elicited by
   the pair; requiring the objective's literal verb confuses wording
   with skill delivery (for example, a response can correctly explain
   a relationship without containing the word ``identify``).
6. Resolve status via the shared
   :func:`lib.validators.block_objective_delivery._resolve_status`
   helper (imported as :data:`_resolve_block_objective_status` to
   disambiguate from any local helper).

Reject precedence (mirrors W1.7.C ``_resolve_status`` axis ordering):
NLI > Bloom > verb. Day-1 stance: ALL axes are warning-severity at the
per-axis level; :meth:`validate_pair` only rejects when EVERY (pair,
lo_id) tuple resolves to ``underdelivered`` — i.e. the pair has no
pedagogically-aligned objective at all. Calibration debt: per-axis
severity may promote to critical after a corpus rebuild lands a stable
per-corpus pass rate.

Decision capture: emit one ``pair_objective_delivery_check`` decision
per (pair, lo_id) audited tuple. Rationale interpolates dynamic signals:
``pair_id``, ``pair_kind``, ``chunk_id``, ``objective_id``,
``declared_bloom``, ``observed_bloom``, ``bloom_gap``,
``statement_entailment_score``, ``contradiction_score``,
``entailment_threshold``, ``verb_match_count``, ``nli_loaded``,
``status``, ``failure_codes``. Mirrors the W1.7.C
``block_objective_delivery_check`` rationale shape verbatim so a
downstream replay can use the same parser.

GateIssue codes (mirror W1.7.C's seven codes; namespace-prefix swap
from ``BLOCK_OBJECTIVE_*`` to ``PAIR_OBJECTIVE_*``):

* :data:`_CODE_STATEMENT_UNDERSUPPORTED` — NLI-axis miss.
* :data:`_CODE_BLOOM_UNDERMET` — Bloom-gap-≥2 miss.
* :data:`_CODE_VERB_ABSENT` — verb-cooccurrence miss.
* :data:`_CODE_NLI_DEPS_MISSING` — :meth:`NliClassifier.get_or_load`
  returned None.
* :data:`_CODE_OBSERVED_BLOOM_UNAVAILABLE` — pair's ``observed_bloom``
  is None.
* :data:`_CODE_OBJECTIVE_NOT_RESOLVED` — ``objectives`` map is missing
  this lo_id.
* :data:`_CODE_VERB_AXIS_UNAVAILABLE` — ``bloom_level`` + ``bloom_verb``
  both missing for the cited objective.

Cross-references:

* W4.A `pair_claim_support.py` — established the per-pair
  ``validate_pair`` shape + NLI singleton-loader pattern.
* W4.B `pair_lo_refs.py` — established the per-pair-side two-surface
  validator pattern + ``lo_refs`` field-name resolution chain (Wave 4
  TIGHT lesson 1).
* W1.7.C `block_objective_delivery.py` — THE TEMPLATE. Mirrors the
  algorithm, the status-resolution helper, the decision-emit shape,
  and the seven GateIssue codes (with namespace swap).
* W4.C.2 (parallel worker) — schema migration adding
  ``pair_objective_alignment`` + ``pair_objective_alignment_pass_rate``
  optional fields to the pair schemas, plus the
  ``pair_objective_delivery_check`` decision-event enum addition.
* W4.C.3 (parallel worker) — call-site wiring + ``SynthesisStats``
  counter (``objective_delivery_rejected``).
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.classifiers.nli_classifier import NliClassifier, NliScore

# W-D7 T7.3: route through the shared
# ``lib.utils.objective_delivery_axes`` package. The status mapper
# previously imported from ``lib.validators.block_objective_delivery``
# (back-compat shim retained at the new canonical via the same module).
from lib.utils.objective_delivery_axes import (
    BLOOM_INDEX as _SHARED_BLOOM_INDEX,
    resolve_status as _resolve_block_objective_status,
)
# Reuse the shared objective-flattener (Courseforge-form +
# LibV2-archive-form tolerant). The W4.C.3 call-site loads the
# synthesized objectives map once per run via
# :func:`_load_synthesized_objectives_for_w4c` (defined below) which
# wraps this flattener and keys by ``lo.id``.
from lib.validators.abcd_objective import _flatten_objectives

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Calibration knobs
# --------------------------------------------------------------------- #


#: Per-pair-kind entailment thresholds (plan §3 W4.C.1 calibration).
#: Day-1 starting points; calibrate against the first post-W4.C corpus
#: rebuild before promoting severity to critical (W4.C.4 follow-up).
#:
#: Rationale per kind:
#:   - ``instruction``: SFT pair, completion paraphrases the chunk;
#:     entailment of the abstract objective is moderate. Mirrors the
#:     W1.7.C ``explanation`` block tier (0.45) but slightly looser.
#:   - ``preference``: DPO chosen surface, tighter than SFT — the
#:     preference signal must match objective semantics. Mirrors the
#:     W1.7.C ``explanation`` block tier.
#:   - ``deterministic_template``: deterministic literal-fact templates
#:     (``kg_metadata.*``, ``violation_detection.*``, ``abstention.*``,
#:     ``schema_translation.*``) whose entailment vs an abstract
#:     objective statement is weaker by construction. Mirrors the
#:     W1.7.C ``concept`` block tier (0.30) on the same reasoning —
#:     definitional surface scaffolds the apply/analyze objective
#:     without strict entailment.
_PER_PAIR_KIND_ENTAILMENT_FLOOR: Dict[str, float] = {
    "instruction": 0.40,
    "preference": 0.45,
    "deterministic_template": 0.30,
}


#: Default entailment floor — used when the per-pair-kind table is
#: missing the kind (defensive against future kind additions before
#: the threshold table is updated).
_DEFAULT_ENTAILMENT_FLOOR: float = 0.40


#: Retained for constructor/API and audit compatibility. Contradiction
#: is recorded conservatively across both NLI directions; the existing
#: entailment floor remains the semantic decision threshold.
_CONTRADICTION_FLOOR: float = 0.50


#: Bloom-gap threshold. ``declared - observed >= 2`` triggers a Bloom
#: miss. ``gap == 1`` is tolerated scaffolding (a pair drafted at
#: ``understand`` for an ``apply`` objective is the prerequisite-
#: scaffolding pattern, same as W1.7.C). Mirrors W1.7.C's
#: hard-coded ``2`` threshold; surfaced as a constant so the
#: constructor can override.
_BLOOM_GAP_THRESHOLD: int = 2


#: Cap the number of issues emitted so a uniformly-broken pair batch
#: doesn't drown the gate report. Mirrors
#: :data:`lib.validators.block_objective_delivery._ISSUE_LIST_CAP`.
_ISSUE_LIST_CAP: int = 50


#: Cap on number of audit-trail issues for the gate-runner walk path
#: (per file). Keeps the GateResult JSON bounded on a 1000-pair corpus.
_GATE_ISSUE_CAP: int = 50


#: Deterministic-template id prefixes. A pair whose ``template_id``
#: starts with one of these prefixes routes to the
#: ``deterministic_template`` entailment-floor bucket regardless of its
#: declared ``kind``. Matches the four post-Wave-100 deterministic
#: generators per the W4.C.1 brief.
_DETERMINISTIC_TEMPLATE_PREFIXES: Tuple[str, ...] = (
    "kg_metadata.",
    "violation_detection.",
    "abstention.",
    "schema_translation.",
)


# --------------------------------------------------------------------- #
# Canonical GateIssue codes (PAIR_OBJECTIVE_* mirrors of W1.7.C's
# BLOCK_OBJECTIVE_* codes — same algorithm, pair surface)
# --------------------------------------------------------------------- #


_CODE_STATEMENT_UNDERSUPPORTED: str = "PAIR_OBJECTIVE_STATEMENT_UNDERSUPPORTED"
_CODE_BLOOM_UNDERMET: str = "PAIR_OBJECTIVE_BLOOM_UNDERMET"
_CODE_VERB_ABSENT: str = "PAIR_OBJECTIVE_VERB_ABSENT"
_CODE_NLI_DEPS_MISSING: str = "PAIR_OBJECTIVE_NLI_DEPS_MISSING"
_CODE_OBSERVED_BLOOM_UNAVAILABLE: str = "PAIR_OBJECTIVE_OBSERVED_BLOOM_UNAVAILABLE"
_CODE_OBJECTIVE_NOT_RESOLVED: str = "PAIR_OBJECTIVE_OBJECTIVE_NOT_RESOLVED"
_CODE_VERB_AXIS_UNAVAILABLE: str = "PAIR_OBJECTIVE_VERB_AXIS_UNAVAILABLE"

#: Gate-runner audit-only codes — fired by the post-write walk in
#: :meth:`PairObjectiveDeliveryValidator.validate` when a pair on disk
#: is missing the audit fields W4.C.1 stamps.
_CODE_MISSING_PAIR_OBJECTIVE_ALIGNMENT: str = "MISSING_PAIR_OBJECTIVE_ALIGNMENT"
_CODE_MISSING_PAIR_OBJECTIVE_ALIGNMENT_PASS_RATE: str = (
    "MISSING_PAIR_OBJECTIVE_ALIGNMENT_PASS_RATE"
)
_CODE_PAIRS_FILE_READ_ERROR: str = "PAIRS_FILE_READ_ERROR"
_CODE_MISSING_INPUTS: str = "MISSING_INPUTS"


# --------------------------------------------------------------------- #
# Canonical rejection reasons (consumed by call-site for stats /
# rejected-reason bookkeeping)
# --------------------------------------------------------------------- #


_REASON_OBJECTIVE_STATEMENT_UNDERSUPPORTED: str = "objective_statement_undersupported"
_REASON_OBJECTIVE_BLOOM_UNDERMET: str = "objective_bloom_undermet"
_REASON_OBJECTIVE_VERB_ABSENT: str = "objective_verb_absent"
_REASON_OBJECTIVE_VALIDATION_UNAVAILABLE: str = (
    "objective_validation_unavailable"
)


# --------------------------------------------------------------------- #
# Canonical alignment-status enum (mirrors W1.7.C)
# --------------------------------------------------------------------- #


_STATUS_DELIVERED: str = "delivered"
_STATUS_UNDERDELIVERED: str = "underdelivered"
_STATUS_VERB_ONLY: str = "verb_only"
_STATUS_UNVERIFIABLE: str = "unverifiable"


# W-D7 T7.3: reuse the canonical Bloom-level → index map from the
# shared axes package so both validators index the same way.
_BLOOM_INDEX: Dict[str, int] = _SHARED_BLOOM_INDEX


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _resolve_pair_kind(pair: Mapping[str, Any], *, kind: str) -> str:
    """Return the pair-kind bucket for entailment-floor dispatch.

    Deterministic-template pairs (the four post-Wave-100 generators)
    route to the ``deterministic_template`` bucket regardless of the
    declared ``kind`` — their literal-fact templates carry weaker
    entailment vs an abstract objective and need a looser floor. Every
    other pair routes to the declared kind (``instruction`` /
    ``preference``).
    """
    template_id = pair.get("template_id")
    if isinstance(template_id, str) and template_id.startswith(
        _DETERMINISTIC_TEMPLATE_PREFIXES
    ):
        return "deterministic_template"
    if kind in _PER_PAIR_KIND_ENTAILMENT_FLOOR:
        return kind
    return "instruction"


def _resolve_pair_text(pair: Mapping[str, Any], *, kind: str) -> str:
    """Return completion-only evidence for objective delivery.

    A prompt states the requested objective and therefore contributes zero
    evidence that the answer delivered it.
    """
    if kind == "instruction":
        return str(pair.get("completion") or "").strip()
    if kind == "preference":
        return str(pair.get("chosen") or "").strip()
    # Unknown kind — best-effort fall-through. Try chosen then
    # completion; the caller is responsible for the kind contract.
    return str(
        pair.get("chosen") or pair.get("completion") or ""
    ).strip()


_NAMED_SYMBOL_PATTERN = re.compile(
    r"\b(coefficient|variable|parameter|constant|term)\s+"
    r"(?:named\s+|called\s+)?([a-zA-Z])\b",
    flags=re.IGNORECASE,
)


def _named_symbol_constraints(text: str) -> set[str]:
    """Extract discriminative named-symbol constraints from math prose.

    NLI models often treat ``coefficient a`` and ``coefficient c`` as
    topically equivalent even though the substitution changes the taught
    relationship. These compact constraints are semantic operands, not
    lexical style requirements, so an objective's constraints must be
    preserved by its concrete skill instance.
    """
    return {
        f"{kind.lower()}:{symbol.lower()}"
        for kind, symbol in _NAMED_SYMBOL_PATTERN.findall(text)
    }


def _resolve_pair_id(pair: Mapping[str, Any]) -> str:
    """Return a printable pair_id for decision-capture rationale.

    Falls back through ``pair_id`` → ``id`` → ``chunk_id`` → ``"?"``.
    Mirrors the resolution chain in
    :meth:`PairLearningOutcomeRefsValidator.validate_pair`.
    """
    return str(
        pair.get("pair_id")
        or pair.get("id")
        or pair.get("chunk_id")
        or "?"
    )


def _resolve_lo_refs(pair: Mapping[str, Any]) -> List[str]:
    """Return the pair's declared LO refs.

    Reads ``lo_refs`` (canonical pair-side field per the Wave 4 TIGHT
    lesson 1 audit) with ``learning_outcome_refs`` as fallback.
    Mirrors :meth:`PairLearningOutcomeRefsValidator.validate_pair`'s
    field-name resolution chain.
    """
    raw = pair.get("lo_refs") or pair.get("learning_outcome_refs")
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: set = set()
    for entry in raw:
        if not isinstance(entry, str):
            continue
        s = entry.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def recompute_complete_objective_bloom_authority(
    pair: Mapping[str, Any],
    *,
    alignment_entries: Any = None,
    pass_rate: Any = None,
) -> Optional[Dict[str, Any]]:
    """Recompute strict Bloom authority from canonical pair audit fields.

    This release-facing proof is deliberately stricter than the ordinary
    objective-delivery tolerance: every declared LO must be present in order,
    delivered, above its existing entailment threshold, below the existing
    contradiction ceiling, and at or above the declared Bloom level.
    """
    entries = (
        pair.get("pair_objective_alignment")
        if alignment_entries is None
        else alignment_entries
    )
    rate = (
        pair.get("pair_objective_alignment_pass_rate")
        if pass_rate is None
        else pass_rate
    )
    lo_refs = _resolve_lo_refs(pair)
    declared = pair.get("bloom_level")
    kind = "preference" if isinstance(pair.get("chosen"), str) else "instruction"
    expected_threshold = _PER_PAIR_KIND_ENTAILMENT_FLOOR[
        _resolve_pair_kind(pair, kind=kind)
    ]
    if (
        not lo_refs
        or rate != 1.0
        or declared not in _SHARED_BLOOM_INDEX
        or not isinstance(entries, list)
        or len(entries) != len(lo_refs)
    ):
        return None
    observed: List[str] = []
    for lo_id, entry in zip(lo_refs, entries):
        if not isinstance(entry, Mapping):
            return None
        seen = entry.get("observed_bloom")
        entailment = entry.get("statement_entailment_score")
        contradiction = entry.get("contradiction_score")
        threshold = entry.get("entailment_threshold")
        gap = entry.get("bloom_gap")
        if (
            entry.get("objective_id") != lo_id
            or entry.get("status") != _STATUS_DELIVERED
            or entry.get("declared_bloom") != declared
            or seen not in _SHARED_BLOOM_INDEX
            or isinstance(entailment, bool)
            or isinstance(contradiction, bool)
            or isinstance(threshold, bool)
            or isinstance(gap, bool)
            or not isinstance(entailment, (int, float))
            or not isinstance(contradiction, (int, float))
            or not isinstance(threshold, (int, float))
            or not isinstance(gap, int)
            or not math.isfinite(float(entailment))
            or not math.isfinite(float(contradiction))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(entailment) <= 1.0
            or not 0.0 <= float(contradiction) <= 1.0
            or float(threshold) != expected_threshold
            or entailment < expected_threshold
            or contradiction >= _CONTRADICTION_FLOOR
            or gap > 0
            or _SHARED_BLOOM_INDEX[seen] < _SHARED_BLOOM_INDEX[declared]
        ):
            return None
        observed.append(str(seen))
    if len(set(observed)) != 1:
        return None
    return {
        "observed_bloom": observed[0],
        "objective_ids": list(lo_refs),
        "entry_count": len(entries),
    }


def _emit_decision(
    capture: Any,
    *,
    pair_id: Optional[str],
    pair_kind: str,
    chunk_id: Optional[str],
    objective_id: Optional[str],
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
    reverse_entailment_score: Optional[float] = None,
    semantic_constraint_matches: int = 0,
    semantic_constraint_misses: Optional[List[str]] = None,
) -> None:
    """Emit one ``pair_objective_delivery_check`` decision per audited
    (pair, lo_id) tuple.

    Rationale interpolates dynamic signals enumerated in W4.C.1:
    ``pair_id``, ``pair_kind``, ``chunk_id``, ``objective_id``,
    ``declared_bloom``, ``observed_bloom``, ``bloom_gap``,
    ``statement_entailment_score``, ``contradiction_score``,
    ``verb_match_count``, the per-pair-kind entailment threshold used,
    whether NLI deps were loaded, and the resolved status enum value.
    Mirrors W1.7.C's ``block_objective_delivery_check`` rationale shape
    verbatim so a downstream replay can use the same parser.
    """
    if capture is None:
        return
    decision = (
        "passed" if not failure_codes else f"failed:{','.join(failure_codes)}"
    )
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
    reverse_ent_str = (
        f"{reverse_entailment_score:.4f}"
        if reverse_entailment_score is not None
        else "n/a"
    )
    constraint_misses = semantic_constraint_misses or []
    rationale = (
        f"Pair-objective instantiated-skill delivery check on {pair_kind!r} pair "
        f"{pair_id or 'n/a'!r} (chunk_id={chunk_id or 'n/a'!r}) for "
        f"objective {objective_id or 'n/a'!r}: "
        f"declared_bloom={declared_bloom or 'n/a'!r}, "
        f"observed_bloom={observed_bloom or 'n/a'!r}, "
        f"bloom_gap={gap_str}, "
        f"statement_entailment_score={ent_str}, "
        f"reverse_entailment_score={reverse_ent_str}, "
        f"contradiction_score={con_str}, "
        f"entailment_threshold={entailment_threshold:.4f}, "
        f"verb_match_count={verb_str}, "
        f"nli_loaded={nli_loaded}, "
        f"status={status}, "
        f"semantic_contract='bidirectional_instantiated_skill', "
        f"semantic_constraint_matches={semantic_constraint_matches}, "
        f"semantic_constraint_misses={','.join(constraint_misses) or 'none'}, "
        f"failure_codes={','.join(failure_codes) or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="pair_objective_delivery_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "pair_objective_delivery_check: %s",
            exc,
        )


def _load_synthesized_objectives_for_w4c(
    path: Any,
) -> Dict[str, Dict[str, Any]]:
    """Wrap :func:`lib.validators.abcd_objective._flatten_objectives` and
    return the per-LO map W4.C.1 needs.

    Reads the synthesized-objectives JSON at ``path`` (Courseforge-form
    OR LibV2-archive form — the flattener is tolerant of both), then
    projects each LO down to the three fields the per-pair tri-axis
    check consumes:

    * ``statement`` — the behavioral outcome (NLI hypothesis).
    * ``bloom_level`` — declared cognitive demand (Bloom-gap axis).
    * ``bloom_verb`` — canonical verb (verb-cooccurrence axis).

    Returns ``{lo_id: {"statement", "bloom_level", "bloom_verb"}}``.
    Returns ``{}`` when the path is missing / unreadable / empty. Real
    providers fail closed on the resulting unverifiable objective
    status; the mock plumbing provider is explicitly exempt.

    Cross-link: W4.C.3 imports this helper at the synthesizer's
    validator-instantiation block so the synthesized-objectives map
    loads exactly once per run.
    """
    if path is None:
        return {}
    try:
        p = Path(path)
    except TypeError:
        return {}
    if not p.exists():
        logger.debug(
            "synthesized objectives path %r does not exist; "
            "W4.C.1 will run with an empty objectives map.",
            str(p),
        )
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "Failed to load synthesized objectives at %r for W4.C.1: %s",
            str(p),
            exc,
        )
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for lo in _flatten_objectives(data):
        lo_id_raw = lo.get("id") or lo.get("objective_id")
        if not isinstance(lo_id_raw, str):
            continue
        lo_id = lo_id_raw.strip()
        if not lo_id:
            continue

        statement_raw = lo.get("statement") or lo.get("text")
        statement = (
            statement_raw.strip()
            if isinstance(statement_raw, str) and statement_raw.strip()
            else None
        )

        bloom_level_raw = lo.get("bloom_level") or lo.get("bloomLevel")
        bloom_level = (
            bloom_level_raw.strip().lower()
            if isinstance(bloom_level_raw, str) and bloom_level_raw.strip()
            else None
        )

        bloom_verb_raw = lo.get("bloom_verb") or lo.get("bloomVerb")
        bloom_verb = (
            bloom_verb_raw.strip().lower()
            if isinstance(bloom_verb_raw, str) and bloom_verb_raw.strip()
            else None
        )

        out[lo_id.lower()] = {
            "statement": statement,
            "bloom_level": bloom_level,
            "bloom_verb": bloom_verb,
        }
    return out


# --------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------- #


class PairObjectiveDeliveryValidator:
    """Per-pair-per-objective tri-axis (NLI + Bloom + verb) delivery
    gate.

    Two surfaces (mirrors W4.A / W4.B):

    1. :meth:`validate_pair` — per-pair pre-write filter called from
       :mod:`Trainforge.synthesize_training` immediately AFTER
       :meth:`PairLearningOutcomeRefsValidator.validate_pair` returns
       ``"validated"``. Returns
       ``(promotion_status, rejection_reason, new_fields)``.
    2. :meth:`validate` — gate-runner surface, walks
       ``training_specs/{instruction,preference}_pairs.jsonl``
       post-write, asserts every pair carries
       ``pair_objective_alignment`` + ``pair_objective_alignment_pass_rate``.

    Constructor:

    Args:
        contradiction_floor: Per-axis NLI contradiction-score floor.
            Above this score the contradiction guard fires (entailment
            miss only triggers when both ``entailment < threshold`` AND
            ``contradiction > contradiction_floor``). Default 0.50.
        bloom_gap_threshold: Bloom-gap rejection threshold. Gap >=
            this value triggers ``_CODE_BLOOM_UNDERMET``. Default 2 —
            ``gap == 1`` is tolerated scaffolding.
        nli: Optional pre-loaded NLI classifier. Test-injection seam
            and singleton-cache override. When ``None``, calls
            :meth:`NliClassifier.get_or_load` lazily.
        decision_capture: Optional :class:`DecisionCapture` instance.
            Per-call captures override this constructor-level default.

    Availability contract:

    * NLI loader returns ``None`` (extras absent / load failed) →
      stamp ``pair_objective_alignment=None`` +
      ``pair_objective_alignment_pass_rate=None``, emit
      :data:`_CODE_NLI_DEPS_MISSING` warning, return
      ``("rejected", "objective_validation_unavailable", ...)`` for
      real providers. The non-shippable mock provider remains exempt.
    * ``objectives`` map missing the cited ``lo_id`` → emit
      :data:`_CODE_OBJECTIVE_NOT_RESOLVED` warning per (pair, lo_id),
      status ``unverifiable``, skip all axes for that tuple.
    * ``observed_bloom`` is None → Bloom axis silently skipped
      (``bloom_passed = None``); status captured as ``unverifiable``
      via :func:`_resolve_block_objective_status`. No issue emitted
      unless EVERY (pair, lo_id) tuple has it None
      (:data:`_CODE_OBSERVED_BLOOM_UNAVAILABLE` aggregate fires once).
    * ``declared_bloom`` + ``bloom_verb`` both missing → verb axis
      silently skipped, single :data:`_CODE_VERB_AXIS_UNAVAILABLE`
      warning per gate run for visibility.

    Day-1 reject contract:

    :meth:`validate_pair` only rejects when EVERY (pair, lo_id) entry
    resolves to ``underdelivered`` — i.e. the pair has no
    pedagogically-aligned objective at all. Per-axis severity is
    warning at the issue level. Calibration debt: promote per-axis
    severity to critical after a corpus rebuild lands a stable
    per-corpus pass rate.

    Reject precedence (mirrors W1.7.C ``_resolve_status``): when the
    pair's per-(lo_id) failure modes are non-empty across all LOs,
    NLI > Bloom > verb determines ``rejection_reason`` (the first
    miss across the pair's LOs in axis order).
    """

    name = "pair_objective_delivery"
    version = "1.0.0"

    def __init__(
        self,
        *,
        contradiction_floor: float = _CONTRADICTION_FLOOR,
        bloom_gap_threshold: int = _BLOOM_GAP_THRESHOLD,
        nli: Optional[NliClassifier] = None,
        decision_capture: Any = None,
        require_verifiable: Optional[bool] = None,
    ) -> None:
        self._contradiction_floor = float(contradiction_floor)
        self._bloom_gap_threshold = int(bloom_gap_threshold)
        self._nli_override = nli
        self._decision_capture = decision_capture
        self._require_verifiable = require_verifiable

    def _get_nli(self) -> Optional[NliClassifier]:
        """Resolve the NLI classifier — explicit override wins, else
        the singleton-cached :meth:`NliClassifier.get_or_load`."""
        if self._nli_override is not None:
            return self._nli_override
        return NliClassifier.get_or_load()

    # ------------------------------------------------------------------ #
    # Per-pair filter — call site in synthesize_training.py
    # ------------------------------------------------------------------ #

    def validate_pair(
        self,
        pair: Dict[str, Any],
        *,
        kind: str,
        chunk: Optional[Dict[str, Any]] = None,
        objectives: Optional[Mapping[str, Mapping[str, Any]]] = None,
        decision_capture: Any = None,
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Audit one pair against the per-objective tri-axis criteria.

        Args:
            pair: The training pair dict. Reads ``prompt`` +
                ``completion`` (when ``kind="instruction"``) or
                ``prompt`` + ``chosen`` (when ``kind="preference"``)
                for the entailment + verb axes; ``observed_bloom`` for
                the Bloom axis; ``lo_refs`` (canonical) or
                ``learning_outcome_refs`` (legacy fallback) for the LO
                fan-out; ``template_id`` for the per-pair-kind floor
                dispatch; ``pair_id`` / ``id`` / ``chunk_id`` for
                decision-capture identifier.
            kind: ``"instruction"`` or ``"preference"``. Selects which
                surface text to assemble. ``template_id`` prefix
                matching can re-route to ``"deterministic_template"``
                for the floor table dispatch (see
                :func:`_resolve_pair_kind`).
            chunk: Optional cited source chunk dict. Currently unused
                by the tri-axis algorithm — accepted for surface
                symmetry with W4.A / W4.B and for future calibration
                that may want to score the pair text against the
                chunk's LO context.
            objectives: ``{lo_id: {"statement", "bloom_level",
                "bloom_verb"}}`` map. The W4.C.3 call-site loads this
                once per run via
                :func:`_load_synthesized_objectives_for_w4c` and threads
                it in here. ``None`` is tolerated — every (pair, lo_id)
                audit emits :data:`_CODE_OBJECTIVE_NOT_RESOLVED` and
                stamps ``unverifiable``; real providers reject it.
            decision_capture: Optional :class:`DecisionCapture` instance.
                Per-call wins over the constructor default; when wired,
                one ``pair_objective_delivery_check`` event per
                (pair, lo_id) tuple.

        Returns:
            ``(promotion_status, rejection_reason, new_fields)``:

            - ``promotion_status``: ``"validated"`` on a verifiable
              pass; real-provider pairs are rejected when objective
              validation is unavailable or every objective is
              underdelivered.
            - ``rejection_reason``: one of
              ``"objective_statement_undersupported"``,
              ``"objective_bloom_undermet"``,
              ``"objective_verb_absent"``, or ``None``. NLI > Bloom >
              verb precedence (the first axis miss across the pair's
              LOs in axis order).
            - ``new_fields``: ``{"pair_objective_alignment": [...] |
              None, "pair_objective_alignment_pass_rate": float | None}``.
              The list carries one entry per (pair, lo_id) audited
              tuple with shape ``{"objective_id", "status",
              "statement_entailment_score", "contradiction_score",
              "bloom_gap", "verb_match_count", "declared_bloom",
              "observed_bloom", "entailment_threshold"}``. Stamped on
              the pair regardless of pass/fail; the caller MUST
              ``pair.update(new_fields)`` so the audit signals land on
              disk.
        """
        capture = decision_capture or self._decision_capture
        objectives_map: Mapping[str, Mapping[str, Any]] = (
            objectives if isinstance(objectives, Mapping) else {}
        )

        pair_id = _resolve_pair_id(pair)
        chunk_id_raw = pair.get("chunk_id")
        chunk_id = (
            str(chunk_id_raw).strip() if isinstance(chunk_id_raw, str) else None
        )
        pair_kind_bucket = _resolve_pair_kind(pair, kind=kind)
        # Staged synthesis explicitly requires every objective axis to be
        # verifiable. The optional constructor override lets its call site make
        # that contract fail closed while the legacy flag-off path preserves
        # the historical provider-based behavior.
        requires_verifiable = (
            bool(self._require_verifiable)
            if self._require_verifiable is not None
            else str(pair.get("provider") or "").lower() != "mock"
        )
        entailment_threshold = _PER_PAIR_KIND_ENTAILMENT_FLOOR.get(
            pair_kind_bucket, _DEFAULT_ENTAILMENT_FLOOR
        )

        # NLI loader — singleton-cached via NliClassifier.get_or_load.
        # Graceful-degrade arm 1: deps missing → stamp None and pass.
        nli = self._get_nli()
        nli_loaded = nli is not None

        # Resolve the pair's declared LO refs.
        lo_refs = _resolve_lo_refs(pair)

        # Arm 1 (deps missing): stamp None alignment + None pass rate,
        # emit one decision per (pair, lo_id) so the audit trail
        # records the validator ran on each, return validated.
        if not nli_loaded:
            new_fields: Dict[str, Any] = {
                "pair_objective_alignment": None,
                "pair_objective_alignment_pass_rate": None,
            }
            # Emit one decision per LO ref so postmortem readers can
            # see the validator ran (rationale records nli_loaded=False).
            for lo_id in lo_refs or [None]:
                _emit_decision(
                    capture,
                    pair_id=pair_id,
                    pair_kind=pair_kind_bucket,
                    chunk_id=chunk_id,
                    objective_id=lo_id,
                    declared_bloom=None,
                    observed_bloom=None,
                    bloom_gap=None,
                    statement_entailment_score=None,
                    contradiction_score=None,
                    verb_match_count=None,
                    entailment_threshold=entailment_threshold,
                    nli_loaded=False,
                    status=_STATUS_UNVERIFIABLE,
                    failure_codes=[_CODE_NLI_DEPS_MISSING],
                )
            return (
                (
                    "rejected",
                    _REASON_OBJECTIVE_VALIDATION_UNAVAILABLE,
                    new_fields,
                )
                if requires_verifiable
                else ("validated", None, new_fields)
            )

        # Resolve the pair surface text (entailment + verb axes).
        pair_text = _resolve_pair_text(pair, kind=kind)

        # Resolve observed_bloom for the Bloom axis (W2.E stamps this).
        observed_bloom_raw = pair.get("observed_bloom")
        observed_bloom: Optional[str] = None
        if isinstance(observed_bloom_raw, str):
            observed_bloom_normalized = observed_bloom_raw.strip().lower()
            if observed_bloom_normalized in _BLOOM_INDEX:
                observed_bloom = observed_bloom_normalized

        alignment_entries: List[Dict[str, Any]] = []
        # Track per-axis "first miss" for reject precedence (NLI >
        # Bloom > verb). We resolve rejection_reason post-loop based on
        # whether EVERY (pair, lo_id) resolved to underdelivered.
        per_lo_failure_axes: List[List[str]] = []
        per_lo_status: List[str] = []

        # Empty / no-LO pair → vacuous pass with empty alignment list.
        # Day-1 contract: pair_lo_refs gate (W4.B) is the canonical
        # surface that catches missing-LO; W4.C.1 stamps an empty list
        # so the audit gate's "carries field" check still passes.
        if not lo_refs:
            new_fields = {
                "pair_objective_alignment": [],
                "pair_objective_alignment_pass_rate": None,
            }
            _emit_decision(
                capture,
                pair_id=pair_id,
                pair_kind=pair_kind_bucket,
                chunk_id=chunk_id,
                objective_id=None,
                declared_bloom=None,
                observed_bloom=observed_bloom,
                bloom_gap=None,
                statement_entailment_score=None,
                contradiction_score=None,
                verb_match_count=None,
                entailment_threshold=entailment_threshold,
                nli_loaded=nli_loaded,
                status=_STATUS_UNVERIFIABLE,
                failure_codes=[],
            )
            return (
                (
                    "rejected",
                    _REASON_OBJECTIVE_VALIDATION_UNAVAILABLE,
                    new_fields,
                )
                if requires_verifiable
                else ("validated", None, new_fields)
            )

        for lo_id in lo_refs:
            obj_dict = objectives_map.get(lo_id) if objectives_map else None
            if not isinstance(obj_dict, Mapping):
                # Objective not resolvable — emit one warning per
                # (pair, lo_id), status unverifiable, skip all axes.
                alignment_entries.append({
                    "objective_id": lo_id,
                    "status": _STATUS_UNVERIFIABLE,
                    "statement_entailment_score": None,
                    "contradiction_score": None,
                    "bloom_gap": None,
                    "verb_match_count": None,
                    "declared_bloom": None,
                    "observed_bloom": observed_bloom,
                    "entailment_threshold": entailment_threshold,
                })
                per_lo_failure_axes.append([_CODE_OBJECTIVE_NOT_RESOLVED])
                per_lo_status.append(_STATUS_UNVERIFIABLE)
                _emit_decision(
                    capture,
                    pair_id=pair_id,
                    pair_kind=pair_kind_bucket,
                    chunk_id=chunk_id,
                    objective_id=lo_id,
                    declared_bloom=None,
                    observed_bloom=observed_bloom,
                    bloom_gap=None,
                    statement_entailment_score=None,
                    contradiction_score=None,
                    verb_match_count=None,
                    entailment_threshold=entailment_threshold,
                    nli_loaded=nli_loaded,
                    status=_STATUS_UNVERIFIABLE,
                    failure_codes=[_CODE_OBJECTIVE_NOT_RESOLVED],
                )
                continue

            statement_raw = obj_dict.get("statement")
            statement = (
                statement_raw.strip()
                if isinstance(statement_raw, str) and statement_raw.strip()
                else None
            )

            declared_bloom_raw = obj_dict.get("bloom_level")
            declared_bloom: Optional[str] = None
            if isinstance(declared_bloom_raw, str):
                normalized = declared_bloom_raw.strip().lower()
                if normalized in _BLOOM_INDEX:
                    declared_bloom = normalized

            bloom_verb_raw = obj_dict.get("bloom_verb")
            bloom_verb: Optional[str] = None
            if isinstance(bloom_verb_raw, str):
                v = bloom_verb_raw.strip().lower()
                if v:
                    bloom_verb = v

            # ---------- Axis 1: NLI entailment --------------- #
            entailment_score: Optional[float] = None
            reverse_entailment_score: Optional[float] = None
            contradiction_score: Optional[float] = None
            entailment_passed: Optional[bool] = None  # None = skipped
            objective_constraints = (
                _named_symbol_constraints(statement) if statement else set()
            )
            pair_constraints = _named_symbol_constraints(pair_text)
            semantic_constraint_matches = len(
                objective_constraints.intersection(pair_constraints)
            )
            semantic_constraint_misses = sorted(
                objective_constraints.difference(pair_constraints)
            )
            if statement and pair_text:
                forward_score: Optional[NliScore] = None
                reverse_score: Optional[NliScore] = None
                try:
                    forward_score = nli.score_pair(
                        premise=pair_text, hypothesis=statement,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "NliClassifier.score_pair raised on pair-objective "
                        "delivery forward direction "
                        "(pair_id=%r, objective_id=%r): %s",
                        pair_id, lo_id, exc,
                    )
                    forward_score = None

                if forward_score is not None:
                    forward_entailment = float(forward_score.entailment)
                    entailment_score = forward_entailment
                    contradiction_score = float(forward_score.contradiction)
                    # Only score the reverse direction when the traditional
                    # pair→competency direction misses. This preserves the
                    # legacy fast path while recognizing a concrete instance
                    # of a broader skill.
                    if forward_entailment < entailment_threshold:
                        try:
                            reverse_score = nli.score_pair(
                                premise=statement, hypothesis=pair_text,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "NliClassifier.score_pair raised on "
                                "pair-objective delivery reverse direction "
                                "(pair_id=%r, objective_id=%r): %s",
                                pair_id, lo_id, exc,
                            )
                        if reverse_score is not None:
                            reverse_entailment_score = float(
                                reverse_score.entailment
                            )
                            entailment_score = max(
                                forward_entailment, reverse_entailment_score
                            )
                            contradiction_score = max(
                                contradiction_score,
                                float(reverse_score.contradiction),
                            )
                    # All three are required. Entailment must clear the
                    # per-kind floor in its own right: the earlier contract
                    # failed a pair only when low entailment AND high
                    # contradiction coincided, which let a NEUTRAL answer —
                    # near-zero entailment, near-zero contradiction, i.e. text
                    # the source neither supports nor denies — pass as
                    # delivered. That is the shape unsupported generated
                    # content takes, so it is the one case this axis most needs
                    # to catch.
                    entailment_passed = (
                        entailment_score >= entailment_threshold
                        and contradiction_score < self._contradiction_floor
                        and not semantic_constraint_misses
                    )

            # ---------- Axis 2: Bloom alignment --------------- #
            bloom_passed: Optional[bool] = None
            bloom_gap: Optional[int] = None
            if (
                declared_bloom is not None
                and observed_bloom is not None
            ):
                bloom_gap = (
                    _BLOOM_INDEX[declared_bloom] - _BLOOM_INDEX[observed_bloom]
                )
                if bloom_gap >= self._bloom_gap_threshold:
                    bloom_passed = False
                else:
                    bloom_passed = True

            # ---------- Axis 3: Action-verb cooccurrence ----- #
            verb_passed: Optional[bool] = None
            verb_match_count: Optional[int] = None
            if bloom_verb and pair_text:
                lowered = pair_text.lower()
                verb_match_count = int(
                    bool(re.search(rf"\b{re.escape(bloom_verb)}\b", lowered))
                )
            # Literal action-verb occurrence is useful diagnostic evidence, but
            # it is not evidence of cognitive demand. The observed-Bloom axis
            # assesses whether the pair actually instantiates the requested
            # skill; wording such as "How does..." can deliver an "identify"
            # objective without repeating the token "identify".
            if bloom_passed is not None:
                verb_passed = True

            status = _resolve_block_objective_status(
                entailment_passed=entailment_passed,
                bloom_passed=bloom_passed,
                verb_passed=verb_passed,
            )

            pair_failure_codes: List[str] = []
            if entailment_passed is False:
                pair_failure_codes.append(_CODE_STATEMENT_UNDERSUPPORTED)
            if bloom_passed is False:
                pair_failure_codes.append(_CODE_BLOOM_UNDERMET)

            alignment_entries.append({
                "objective_id": lo_id,
                "status": status,
                "statement_entailment_score": entailment_score,
                "contradiction_score": contradiction_score,
                "bloom_gap": bloom_gap,
                "verb_match_count": verb_match_count,
                "declared_bloom": declared_bloom,
                "observed_bloom": observed_bloom,
                "entailment_threshold": entailment_threshold,
            })
            per_lo_failure_axes.append(pair_failure_codes)
            per_lo_status.append(status)

            _emit_decision(
                capture,
                pair_id=pair_id,
                pair_kind=pair_kind_bucket,
                chunk_id=chunk_id,
                objective_id=lo_id,
                declared_bloom=declared_bloom,
                observed_bloom=observed_bloom,
                bloom_gap=bloom_gap,
                statement_entailment_score=entailment_score,
                contradiction_score=contradiction_score,
                verb_match_count=verb_match_count,
                entailment_threshold=entailment_threshold,
                nli_loaded=nli_loaded,
                status=status,
                failure_codes=pair_failure_codes,
                reverse_entailment_score=reverse_entailment_score,
                semantic_constraint_matches=semantic_constraint_matches,
                semantic_constraint_misses=semantic_constraint_misses,
            )

        # Compute pass rate over audited (pair, lo_id) tuples.
        total = len(alignment_entries)
        passed_count = sum(
            1 for s in per_lo_status if s == _STATUS_DELIVERED
        )
        pass_rate: Optional[float] = (
            round(passed_count / total, 4) if total > 0 else None
        )

        # Day-1 reject contract: only reject when EVERY (pair, lo_id)
        # entry resolved to underdelivered. Rejection reason follows
        # NLI > Bloom > verb precedence on the FIRST per-LO failure
        # axis seen.
        all_underdelivered = (
            total > 0
            and all(s == _STATUS_UNDERDELIVERED for s in per_lo_status)
        )
        any_unverifiable = any(
            s == _STATUS_UNVERIFIABLE for s in per_lo_status
        )

        rejection_reason: Optional[str] = None
        promotion_status = "validated"
        if any_unverifiable and requires_verifiable:
            rejection_reason = _REASON_OBJECTIVE_VALIDATION_UNAVAILABLE
            promotion_status = "rejected"
        elif requires_verifiable and (
            all_underdelivered or (total > 0 and passed_count == 0)
        ):
            # Walk per-LO failure axes in pair-order; pick the first
            # axis hit per the NLI > Bloom > verb precedence.
            for axes in per_lo_failure_axes:
                if _CODE_STATEMENT_UNDERSUPPORTED in axes:
                    rejection_reason = _REASON_OBJECTIVE_STATEMENT_UNDERSUPPORTED
                    break
            if rejection_reason is None:
                for axes in per_lo_failure_axes:
                    if _CODE_BLOOM_UNDERMET in axes:
                        rejection_reason = _REASON_OBJECTIVE_BLOOM_UNDERMET
                        break
            if rejection_reason is None:
                for axes in per_lo_failure_axes:
                    if _CODE_VERB_ABSENT in axes:
                        rejection_reason = _REASON_OBJECTIVE_VERB_ABSENT
                        break
            if rejection_reason is not None:
                promotion_status = "rejected"

        new_fields = {
            "pair_objective_alignment": alignment_entries,
            "pair_objective_alignment_pass_rate": pass_rate,
        }
        authority = recompute_complete_objective_bloom_authority(
            pair,
            alignment_entries=alignment_entries,
            pass_rate=pass_rate,
        )
        if authority is not None:
            new_fields["observed_bloom"] = authority["observed_bloom"]
        return promotion_status, rejection_reason, new_fields

    # ------------------------------------------------------------------ #
    # Gate-runner surface — walks training_specs/*.jsonl
    # ------------------------------------------------------------------ #

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Post-hoc audit that every pair on disk carries
        ``pair_objective_alignment`` + ``pair_objective_alignment_pass_rate``.

        NOT a re-run of the per-pair tri-axis fan-out (that would
        require the synthesized-objectives map and the NLI loader; both
        are the call-site's responsibility — same contract as
        :meth:`PairClaimSupportValidator.validate`).

        Inputs:

        - ``course_dir`` (str) — preferred. Resolves to
          ``<course_dir>/training_specs/{instruction,preference}_pairs.jsonl``.
        - ``training_specs_dir`` (str) — sibling alternative.
        - ``instruction_pairs_path`` (str) +
          ``preference_pairs_path`` (str) — explicit overrides.

        Outputs:

        - ``passed=True`` when every pair on disk has
          ``pair_objective_alignment`` (list-or-None) AND
          ``pair_objective_alignment_pass_rate`` (number-or-None) keys
          present.
        - ``passed=False`` with capped
          :data:`_CODE_MISSING_PAIR_OBJECTIVE_ALIGNMENT` /
          :data:`_CODE_MISSING_PAIR_OBJECTIVE_ALIGNMENT_PASS_RATE`
          critical issues otherwise.
        - Decision-capture: emits one
          ``pair_objective_delivery_check`` event with
          ``decision="audit:passed"`` /
          ``decision="audit:failed:N_missing"``.
        """
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture

        # Resolve paths.
        inst_path: Optional[Path] = None
        pref_path: Optional[Path] = None
        course_dir_raw = inputs.get("course_dir")
        training_specs_dir_raw = inputs.get("training_specs_dir")
        explicit_inst = inputs.get("instruction_pairs_path")
        explicit_pref = inputs.get("preference_pairs_path")

        if isinstance(explicit_inst, str) and explicit_inst:
            inst_path = Path(explicit_inst)
        if isinstance(explicit_pref, str) and explicit_pref:
            pref_path = Path(explicit_pref)
        if course_dir_raw and (inst_path is None or pref_path is None):
            cd = Path(course_dir_raw)
            if inst_path is None:
                inst_path = cd / "training_specs" / "instruction_pairs.jsonl"
            if pref_path is None:
                pref_path = cd / "training_specs" / "preference_pairs.jsonl"
        if (
            training_specs_dir_raw
            and (inst_path is None or pref_path is None)
        ):
            ts = Path(training_specs_dir_raw)
            if inst_path is None:
                inst_path = ts / "instruction_pairs.jsonl"
            if pref_path is None:
                pref_path = ts / "preference_pairs.jsonl"

        if inst_path is None and pref_path is None:
            issue = GateIssue(
                severity="critical",
                code=_CODE_MISSING_INPUTS,
                message=(
                    "PairObjectiveDeliveryValidator requires one of: "
                    "course_dir, training_specs_dir, or explicit "
                    "{instruction,preference}_pairs_path."
                ),
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[issue],
                action="block",
            )

        # Walk both files; missing files are tolerated (a course may
        # have only instruction pairs or only preference pairs).
        issues: List[GateIssue] = []
        audited = 0
        missing_field_count = 0
        for path in (inst_path, pref_path):
            if path is None or not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line_num, line in enumerate(fh, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        if not isinstance(row, dict):
                            continue
                        audited += 1
                        # ``pair_objective_alignment`` may legitimately
                        # be ``None`` (deps-missing arm) or a list (the
                        # scored arm); the audit checks key presence,
                        # not value type.
                        has_align = "pair_objective_alignment" in row
                        has_rate = (
                            "pair_objective_alignment_pass_rate" in row
                        )
                        if not has_align:
                            missing_field_count += 1
                            if len(issues) < _GATE_ISSUE_CAP:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code=_CODE_MISSING_PAIR_OBJECTIVE_ALIGNMENT,
                                    message=(
                                        f"{path.name}:{line_num} pair "
                                        f"missing pair_objective_alignment; "
                                        f"the per-pair objective-delivery "
                                        f"validator did not stamp the pair "
                                        f"before emit. Re-run "
                                        f"training_synthesis with W4.C.1's "
                                        f"per-pair filter wired in."
                                    ),
                                    location=str(path),
                                ))
                        if not has_rate:
                            missing_field_count += 1
                            if len(issues) < _GATE_ISSUE_CAP:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code=(
                                        _CODE_MISSING_PAIR_OBJECTIVE_ALIGNMENT_PASS_RATE
                                    ),
                                    message=(
                                        f"{path.name}:{line_num} pair "
                                        f"missing "
                                        f"pair_objective_alignment_pass_rate; "
                                        f"the per-pair objective-delivery "
                                        f"validator did not stamp the pair "
                                        f"before emit."
                                    ),
                                    location=str(path),
                                ))
            except OSError as exc:
                issues.append(GateIssue(
                    severity="critical",
                    code=_CODE_PAIRS_FILE_READ_ERROR,
                    message=(
                        f"Failed to read {path}: {exc}"
                    ),
                    location=str(path),
                ))

        passed = missing_field_count == 0 and not any(
            i.severity == "critical" for i in issues
        )
        action: Optional[str] = None if passed else "block"

        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="pair_objective_delivery_check",
                    decision=(
                        "audit:passed"
                        if passed
                        else f"audit:failed:{missing_field_count}_missing"
                    ),
                    rationale=(
                        f"PairObjectiveDeliveryValidator on-disk audit: "
                        f"{audited} pair(s) audited across "
                        f"instruction + preference; "
                        f"{missing_field_count} missing "
                        f"pair_objective_alignment / "
                        f"pair_objective_alignment_pass_rate fields. "
                        f"inst_path="
                        f"{inst_path.name if inst_path else 'n/a'}, "
                        f"pref_path="
                        f"{pref_path.name if pref_path else 'n/a'}."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "DecisionCapture.log_decision raised on "
                    "pair_objective_delivery_check (gate path): %s",
                    exc,
                )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=(
                1.0
                if audited == 0
                else round((audited - missing_field_count) / audited, 4)
            ),
            issues=issues,
            action=action,
        )


__all__ = [
    "PairObjectiveDeliveryValidator",
    "recompute_complete_objective_bloom_authority",
    "_load_synthesized_objectives_for_w4c",
    "_PER_PAIR_KIND_ENTAILMENT_FLOOR",
    "_DEFAULT_ENTAILMENT_FLOOR",
    "_CONTRADICTION_FLOOR",
    "_BLOOM_GAP_THRESHOLD",
    "_DETERMINISTIC_TEMPLATE_PREFIXES",
    "_CODE_STATEMENT_UNDERSUPPORTED",
    "_CODE_BLOOM_UNDERMET",
    "_CODE_VERB_ABSENT",
    "_CODE_NLI_DEPS_MISSING",
    "_CODE_OBSERVED_BLOOM_UNAVAILABLE",
    "_CODE_OBJECTIVE_NOT_RESOLVED",
    "_CODE_VERB_AXIS_UNAVAILABLE",
    "_REASON_OBJECTIVE_STATEMENT_UNDERSUPPORTED",
    "_REASON_OBJECTIVE_BLOOM_UNDERMET",
    "_REASON_OBJECTIVE_VERB_ABSENT",
]
