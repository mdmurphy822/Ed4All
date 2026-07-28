"""Deterministic eligibility and objective focus for training synthesis.

The source chunk's ``bloom_level`` and ordering of
``learning_outcome_refs`` are retrieval metadata, not authoritative
pedagogical declarations.  This module resolves both the objective statement
and Bloom level from the canonical objectives artifact before any provider
call, then decides SFT and DPO eligibility independently.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from Trainforge.generators.synthesis_window_contract import (
    build_evidence_window,
    objective_card,
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
_STOP_WORDS = frozenset({
    "and", "are", "for", "from", "given", "into", "its", "that", "the",
    "their", "then", "this", "through", "using", "with", "will",
})
_OBJECTIVE_LINE_RE = re.compile(
    r"\b(?P<id>(?:TO|CO)-\d+)\s*[—–]\s*"
    r"(?P<statement>.+?)"
    r"(?:\s+Bloom:\s*(?P<bloom>"
    r"remember|understand|apply|analyze|evaluate|create)\b"
    r"|(?=\s+(?:TO|CO)-\d+\s*[—–])|$)",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_OBJECTIVE_RE = re.compile(
    r"^\s*(?P<id>(?:TO|CO)-\d+)\s*:\s*(?P<statement>[^\n]+)",
    re.IGNORECASE,
)
_CANONICAL_OBJECTIVE_RELATIONS = frozenset({
    "descendant", "descendant-of",
    "equivalent", "equivalent-to",
    "broader", "broader-than",
    "narrower", "narrower-than",
})

_MISCONCEPTION_AFFORDANCE_RE = re.compile(
    r"\b(?:common error|incorrect|misconception|rather than|instead of|"
    r"compare|contrast|difference|if|unless|because|rule|property|steps?|"
    r"equation|formula|solution|example)\b",
    re.IGNORECASE,
)
_MALFORMED_ASSESSMENT_RE = re.compile(
    r"(?:\s\?\s|^\s*solve\s+process\b|"
    r"\bcompare and contrast\s+(?:shown below|how|line y)\b|"
    r"\bwhich definition best matches\s+(?:the term\s+)?(?:this|because)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PairEligibility:
    eligible: bool
    reason: Optional[str] = None


def _normalise_objectives(
    objectives: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(key).strip().lower(): value
        for key, value in objectives.items()
        if isinstance(value, Mapping)
    }


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token.lower()
        for token in _TOKEN_RE.findall(str(text or ""))
        if token.lower() not in _STOP_WORDS
    )


def _alignment_score(statement: str, text: str) -> Tuple[int, float]:
    objective_tokens = _tokens(statement)
    source_tokens = _tokens(text)
    if not objective_tokens or not source_tokens:
        return 0, 0.0
    overlap = len(objective_tokens & source_tokens)
    return overlap, overlap / len(objective_tokens)


def _evidences_every_content_obligation(
    objective: Mapping[str, Any], text: str,
) -> bool:
    """Require the source to support every independently requested content facet."""
    source_tokens = _tokens(text)
    obligations = objective_card(objective).get("content_obligations") or []
    for obligation in obligations:
        required = _tokens(str(obligation))
        if not required:
            continue
        overlap = len(required & source_tokens)
        minimum = 1 if len(required) <= 2 else 2
        if overlap < minimum or overlap / len(required) < 0.25:
            return False
    return True


def _canonical_objective(
    objective_id: str,
    objectives: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    raw = objectives.get(objective_id.lower())
    if not isinstance(raw, Mapping):
        return None
    statement = " ".join(str(raw.get("statement") or "").split()).strip()
    bloom = str(raw.get("bloom_level") or "").strip().lower()
    if not statement or bloom not in {
        "remember", "understand", "apply", "analyze", "evaluate", "create",
    }:
        return None
    canonical: Dict[str, Any] = {
        "id": objective_id.lower(),
        "statement": statement,
        "bloom_level": bloom,
    }
    behavior = raw.get("behavior")
    behavior_map = behavior if isinstance(behavior, Mapping) else {}
    bloom_verb = " ".join(
        str(raw.get("bloom_verb") or behavior_map.get("verb") or "").split()
    ).strip().lower()
    if bloom_verb:
        canonical["bloom_verb"] = bloom_verb

    abcd: Dict[str, str] = {}
    for field in ("action_object", "condition", "degree"):
        value = " ".join(
            str(raw.get(field) or behavior_map.get(field) or "").split()
        ).strip()
        if value:
            abcd[field] = value
            # Retain the established flattened aliases while also exposing the
            # canonical nested behavior card consumed by prompt windows.
            canonical[field] = value
    if bloom_verb or abcd:
        canonical["behavior"] = {
            **({"verb": bloom_verb} if bloom_verb else {}),
            **abcd,
        }
    # These fields are authoritative audit data, not prompt-authored metadata.
    # Retain them when present so an ontology-related focus remains traceable
    # to the canonical objectives artifact.
    for field in ("provenance", "source_chunk_ids", "source_refs"):
        value = raw.get(field)
        if value:
            canonical[field] = value
    return canonical


def _relation_records(objective: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield only structured relations explicitly attached to an objective."""
    for field in ("ontology_relations", "canonical_relations"):
        records = objective.get(field) or []
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            for record in records:
                if isinstance(record, Mapping):
                    yield record


def _related_objective_candidates(
    declared_ids: Sequence[str],
    objectives: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    """Return provenance-bearing ontology neighbours of declared objectives.

    This deliberately performs no lexical search over the objective catalog.
    A neighbour is admitted only by a structured canonical relation whose
    endpoint is one of the chunk's own learning-outcome refs.
    """
    declared = set(declared_ids)
    related: Dict[str, Mapping[str, Any]] = {}
    for owner_id, owner in objectives.items():
        for relation in _relation_records(owner):
            relation_type = str(
                relation.get("relation")
                or relation.get("type")
                or relation.get("predicate")
                or ""
            ).strip().lower()
            target_id = str(
                relation.get("objective_id")
                or relation.get("target_id")
                or relation.get("target")
                or ""
            ).strip().lower()
            provenance = relation.get("provenance")
            if (
                relation_type not in _CANONICAL_OBJECTIVE_RELATIONS
                or not target_id
                or not isinstance(provenance, Mapping)
                or not provenance
            ):
                continue
            if owner_id in declared and target_id not in declared:
                candidate_id = target_id
            elif target_id in declared and owner_id not in declared:
                candidate_id = owner_id
            else:
                continue
            if _canonical_objective(candidate_id, objectives) is not None:
                related[candidate_id] = {
                    "relation": relation_type,
                    "source_objective_id": owner_id,
                    "target_objective_id": target_id,
                    "provenance": dict(provenance),
                }
    return related


def focus_chunk_on_canonical_objective(
    chunk: Mapping[str, Any],
    *,
    seed: int,
    objectives: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return a provider view focused on one authoritative objective.

    Selection is deterministic.  Inline objective IDs may establish which
    referenced objectives are locally relevant, but their statement and Bloom
    metadata never override the canonical objectives artifact.
    """

    focused = dict(chunk)
    declared = [
        str(ref).strip().lower()
        for ref in (chunk.get("learning_outcome_refs") or [])
        if str(ref).strip()
    ]
    if len(declared) > 16:
        focused["learning_outcome_refs"] = []
        focused["synthesis_focus_skip_reason"] = (
            "broad_objective_index_not_instructional_content"
        )
        return focused

    canonical = _normalise_objectives(objectives)
    if not canonical:
        focused["learning_outcome_refs"] = []
        focused["synthesis_focus_skip_reason"] = (
            "authoritative_objectives_unavailable"
        )
        return focused

    missing = [ref for ref in declared if _canonical_objective(ref, canonical) is None]
    if not declared or len(missing) == len(declared):
        focused["learning_outcome_refs"] = []
        focused["synthesis_focus_skip_reason"] = (
            "authoritative_objective_not_resolved"
        )
        return focused

    text = str(chunk.get("text") or "")
    inline_ids = []
    for match in _OBJECTIVE_LINE_RE.finditer(text):
        objective_id = match.group("id").lower()
        if objective_id in declared and objective_id not in inline_ids:
            inline_ids.append(objective_id)
    leading = _LEADING_OBJECTIVE_RE.search(text)
    if leading is not None:
        objective_id = leading.group("id").lower()
        if objective_id in declared and objective_id not in inline_ids:
            inline_ids.append(objective_id)

    candidate_ids: Sequence[str] = inline_ids or declared
    ranked = []
    for objective_id in candidate_ids:
        objective = _canonical_objective(objective_id, canonical)
        if objective is None:
            continue
        overlap, score = _alignment_score(objective["statement"], text)
        ranked.append((overlap, score, objective_id, objective))
    if not ranked:
        focused["learning_outcome_refs"] = []
        focused["synthesis_focus_skip_reason"] = (
            "authoritative_objective_not_resolved"
        )
        return focused

    # Inline objective statements are explicit local provenance.  For ordinary
    # multi-ref chunks, require at least two objective content tokens and 20%
    # objective coverage so unrelated prerequisite refs cannot win by order.
    if inline_ids:
        eligible = [
            row for row in ranked
            if _evidences_every_content_obligation(row[3], text)
        ]
    else:
        eligible = [
            row for row in ranked
            if row[0] >= 2 and row[1] >= 0.20
            and _evidences_every_content_obligation(row[3], text)
        ]
    if not eligible:
        # A broad referenced objective must never be sampled from partial
        # evidence.  A narrower objective is permissible only when it already
        # exists in the authoritative artifact and independently aligns with
        # this source; no objective text is synthesized here.
        related = _related_objective_candidates(candidate_ids, canonical)
        narrower = []
        for objective_id in sorted(related):
            objective = _canonical_objective(objective_id, canonical)
            if objective is None:
                continue
            objective["ontology_relation"] = dict(related[objective_id])
            overlap, score = _alignment_score(objective["statement"], text)
            if (
                overlap >= 2 and score >= 0.20
                and _evidences_every_content_obligation(objective, text)
            ):
                narrower.append((overlap, score, objective_id, objective))
        eligible = narrower
    if not eligible:
        focused["learning_outcome_refs"] = []
        focused["synthesis_focus_skip_reason"] = (
            "objective_content_obligations_not_evidenced"
        )
        return focused

    best_overlap = max(row[0] for row in eligible)
    best_score = max(row[1] for row in eligible if row[0] == best_overlap)
    tied = [
        row for row in eligible
        if row[0] == best_overlap and row[1] == best_score
    ]
    digest = hashlib.sha256(
        f"{chunk.get('id') or chunk.get('chunk_id')}|{int(seed)}".encode()
    ).digest()
    selected = tied[int.from_bytes(digest[:8], "big") % len(tied)][3]
    focused["learning_outcome_refs"] = [selected["id"]]
    focused["bloom_level"] = selected["bloom_level"]
    focused["synthesis_focus_objective"] = dict(selected)
    focused["synthesis_original_bloom_level"] = chunk.get("bloom_level")
    return focused


def pair_eligibility(
    focused_chunk: Mapping[str, Any],
    *,
    kind: str,
) -> PairEligibility:
    """Return deterministic SFT/DPO eligibility before provider dispatch."""

    skip_reason = focused_chunk.get("synthesis_focus_skip_reason")
    if skip_reason:
        return PairEligibility(False, str(skip_reason))
    if not focused_chunk.get("learning_outcome_refs"):
        return PairEligibility(False, "missing_aligned_objective")

    focus = focused_chunk.get("synthesis_focus_objective")
    if not isinstance(focus, Mapping):
        return PairEligibility(False, "missing_canonical_objective_focus")
    try:
        build_evidence_window(focused_chunk, focus)
    except ValueError:
        return PairEligibility(False, "objective_evidence_window_not_viable")

    text = " ".join(str(focused_chunk.get("text") or "").split())
    if len(text) < 80 or len(_tokens(text)) < 8:
        return PairEligibility(False, "insufficient_standalone_evidence")

    content_type = str(focused_chunk.get("chunk_type") or "").lower()
    if "assessment" in content_type:
        if _MALFORMED_ASSESSMENT_RE.search(text):
            return PairEligibility(False, "malformed_assessment_item")
        structured_answer = any(
            focused_chunk.get(key)
            for key in (
                "correct_answer", "answer_key", "reference_answer",
                "assessment_answer",
            )
        )
        if not structured_answer:
            return PairEligibility(False, "assessment_answer_key_missing")

    if kind == "instruction":
        return PairEligibility(True)
    if kind != "preference":
        return PairEligibility(False, "unsupported_pair_kind")

    # Preference admission must use the exact dependency resolver consumed by
    # micro Stage E. A lexical affordance can suggest a misconception while
    # still providing no source-backed incorrect/correction candidate.
    from Trainforge.generators.staged_synthesis_micro import (
        micro_preference_eligibility,
    )
    result = micro_preference_eligibility(focused_chunk, focus=focus)
    return PairEligibility(bool(result["eligible"]), str(result["reason"])
                           if result.get("reason") else None)
