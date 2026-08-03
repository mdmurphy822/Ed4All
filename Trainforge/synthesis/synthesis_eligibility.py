"""Deterministic eligibility and objective focus for training synthesis.

The source chunk's ``bloom_level`` and ordering of
``learning_outcome_refs`` are retrieval metadata, not authoritative
pedagogical declarations.  This module resolves both the objective statement
and Bloom level from the canonical objectives artifact before any provider
call, then decides SFT and DPO eligibility independently.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from Trainforge.generators.synthesis_window_contract import (
    build_evidence_window,
    objective_card,
)
from lib.ontology.lexical_concept_seeds import FUNCTION_WORDS

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


# ---------------------------------------------------------------------------
# Chunk text-field contract
# ---------------------------------------------------------------------------
#
# ``text`` is the canonical chunk_v4 prose field and the ONLY one any emitter
# in this tree writes (measured: 5,012/5,012 archived chunks carry ``text``;
# zero carry ``content`` or ``body``).  ``content`` / ``body`` are READ-side
# aliases that sibling readers already tolerate —
# ``lib/validators/chunk_wcag_status.py`` reads ``text or content or html``,
# ``MCP/hardening/gate_input_routing.py`` and ``lib/objectives/*`` read
# ``text or body`` — so a mapping that reaches this module in one of those
# shapes must resolve here the same way it resolves there.  Accepting them on
# READ is the whole widening: nothing in this module ever WRITES a chunk, and
# the alias set is closed (add a producer, not an alias).
#
# The load-bearing half is the FAILURE mode.  ``chunk.get("text") or ""``
# collapses two different facts into one empty string:
#
#   * "this chunk declares itself empty"     -> a real, gate-able disposition
#     (``chunk_carries_no_groundable_content`` / ``degenerate_source_stem``),
#   * "this mapping has no prose field AT ALL" -> shape drift, which the gate
#     then silently scored as slot-filler residue and excluded from synthesis.
#
# The second case is what produced the false ``degenerate_source_stem``
# verdict on a chunk whose prose was sitting in ``content``.  Per the project's
# no-design-intent-fallbacks rule it now raises instead of being defaulted to
# empty prose.
CHUNK_TEXT_FIELDS: Tuple[str, ...] = ("text", "content", "body")


class ChunkTextContractError(ValueError):
    """A chunk mapping carries no prose field under any accepted alias.

    Raised — never defaulted to ``""`` — because a missing field and a
    declared-empty field are different facts, and only the second one is a
    verdict this module is entitled to reach.  Silently conflating them
    excluded real chunks from synthesis under a fabricated
    ``degenerate_source_stem`` reason.
    """


def resolve_chunk_text(chunk: Mapping[str, Any]) -> str:
    """Return a chunk's prose, resolved across :data:`CHUNK_TEXT_FIELDS`.

    Precedence is declaration order: the canonical ``text`` wins whenever it
    carries prose, and an alias is consulted only when the higher-precedence
    fields are absent or blank.

    An explicitly blank value under ANY accepted alias returns ``""`` — that
    is an honest empty chunk and the content gate owns the verdict.  A mapping
    carrying NONE of the accepted keys raises :class:`ChunkTextContractError`,
    naming the keys it did carry so the drifting producer is identifiable.
    """
    if not isinstance(chunk, Mapping):
        raise ChunkTextContractError(
            f"chunk text is unresolvable on {type(chunk).__name__}; "
            f"expected a mapping carrying one of {list(CHUNK_TEXT_FIELDS)}"
        )
    declared = False
    for field in CHUNK_TEXT_FIELDS:
        if field not in chunk:
            continue
        declared = True
        value = chunk[field]
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    if declared:
        # Every alias present was blank/None: a genuinely empty chunk.
        return ""
    raise ChunkTextContractError(
        "chunk carries no prose field under any accepted alias "
        f"{list(CHUNK_TEXT_FIELDS)}; chunk_id="
        f"{chunk.get('id') or chunk.get('chunk_id') or '<unidentified>'!r}, "
        f"keys={sorted(str(key) for key in chunk)}"
    )


@dataclass(frozen=True)
class PairEligibility:
    eligible: bool
    reason: Optional[str] = None
    #: Optional free-text signal detail interpolated into the caller's
    #: DecisionCapture rationale. Deliberately NOT folded into ``reason``
    #: so the per-reason stats/checkpoint key stays low-cardinality while
    #: the audit trail still names the concrete per-chunk signals.
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Pre-generation content gate (parse-with-fallback env resolvers)
# ---------------------------------------------------------------------------
#
# A chunk that carries no groundable content produces a syntactically valid
# but semantically empty training pair (the deterministic factory falls
# through to boilerplate). Per the project's no-design-intent-fallbacks rule
# the unit is excluded BEFORE a model call is spent, as a deterministic
# ``ineligible`` disposition rather than a post-hoc quality rejection.

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off", ""})

CONTENT_GATE_ENV = "TRAINFORGE_SYNTHESIS_CONTENT_GATE"
MIN_PROSE_WORDS_ENV = "TRAINFORGE_SYNTHESIS_MIN_PROSE_WORDS"
MIN_STEM_CONTENT_WORDS_ENV = "TRAINFORGE_SYNTHESIS_MIN_STEM_CONTENT_WORDS"

#: Word floor for the prose arm of the content gate. A chunk with neither
#: ``concept_tags`` nor ``key_terms`` must carry at least this much prose for
#: a generator to have anything to ground a completion in.
DEFAULT_MIN_PROSE_WORDS = 40
#: Content-word (non-function-word) floor for a chunk's leading stem.
DEFAULT_MIN_STEM_CONTENT_WORDS = 3

# A slot-filled template whose filler was empty or a bare fragment leaves a
# whitespace gap immediately before the sentence-terminal punctuation, e.g.
# ``"Solve process ."`` / ``"Compare and contrast Shown below and How ."`` /
# ``"... the term Because of this, it ?"``. Real prose never emits that gap.
#
# The preceding token must end in >=2 LETTERS. Flattened math notation
# routinely emits a spaced terminal period after a numeral or a single-letter
# variable (``"79 ."``, ``"... + x ."``, ``"2 x 2 + 3 x + 5 ."``); those are
# legitimate content chunks, not slot-filler residue, and the letter anchor is
# what keeps a math-heavy scan corpus out of this arm.
_STEM_SLOT_GAP_RE = re.compile(r"[A-Za-z]{2}\s[.?!]\s*$")
_STEM_SPLIT_RE = re.compile(r"(?<=[.?!])\s")
#: Multiple of the prose floor bounding an "item-sized" chunk. Only chunks at
#: or below this size get the degenerate-stem check: an auto-generated
#: question item is a stem plus at most a sentence of generic instruction,
#: while a 300-500-word instructional chunk merely OPENS with a sentence and
#: must never be judged on it. Measured against the real corpora, dropping
#: this bound false-excluded ~2500 legitimate chunks whose flattened OCR/math
#: text emits a spaced terminal period mid-prose ("...are called integers .").
_STEM_ITEM_MAX_WORD_MULTIPLE = 2


def resolve_content_gate_enabled() -> bool:
    """Return True unless the content gate is explicitly disabled (default ON).

    Read at call time (not import) so operators / tests can toggle per-run.
    Unset / garbage → ON; only an explicit falsey token turns it off.
    """
    raw = os.environ.get(CONTENT_GATE_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSEY


def _resolve_int(env_name: str, default: int) -> int:
    """Parse a positive-int threshold env; unset / garbage / <=0 → ``default``."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return value


def resolve_min_prose_words() -> int:
    return _resolve_int(MIN_PROSE_WORDS_ENV, DEFAULT_MIN_PROSE_WORDS)


def resolve_min_stem_content_words() -> int:
    return _resolve_int(
        MIN_STEM_CONTENT_WORDS_ENV, DEFAULT_MIN_STEM_CONTENT_WORDS,
    )


def _key_term_count(chunk: Mapping[str, Any]) -> int:
    """Count key_terms entries that actually carry a term surface form."""
    count = 0
    for entry in chunk.get("key_terms") or []:
        if isinstance(entry, Mapping):
            if str(entry.get("term") or "").strip():
                count += 1
        elif str(entry or "").strip():
            count += 1
    return count


def _concept_tag_count(chunk: Mapping[str, Any]) -> int:
    return sum(1 for tag in (chunk.get("concept_tags") or []) if str(tag).strip())


def describe_content_sources(chunk: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the per-chunk content-source census the gate + captures read.

    Exposed (rather than inlined) so the DecisionCapture rationale on a
    skipped unit can name exactly which sources were empty.
    """
    text = " ".join(resolve_chunk_text(chunk).split())
    return {
        "concept_tags": _concept_tag_count(chunk),
        "key_terms": _key_term_count(chunk),
        "summary_chars": len(str(chunk.get("summary") or "").strip()),
        "prose_words": len(text.split()),
    }


def leading_stem(text: str) -> str:
    """Return the chunk's leading sentence — the surface a template fills."""
    flat = " ".join(str(text or "").split())
    if not flat:
        return ""
    return _STEM_SPLIT_RE.split(flat, maxsplit=1)[0].strip()


def is_degenerate_stem(stem: str, *, min_content_words: Optional[int] = None) -> bool:
    """True when ``stem`` is auto-generated slot-filler residue, not a question.

    Two domain-agnostic signals, both observed on OCR'd-scan MCQ residue:

    1. A whitespace gap immediately before terminal punctuation — the mark of
       a template slot filled with an empty string or a trailing-space
       fragment (``"Solve process ."``).
    2. Fewer than ``min_content_words`` distinct non-function words, i.e. the
       stem is structurally a template with nothing substantive in its slots.

    ``min_content_words=0`` disables signal 2. The gate does that whenever the
    stem is only the OPENING of a longer chunk: a 400-word worked-example
    chunk that happens to open with ``"Simplify."`` is not degenerate, and
    only signal 1 is safe to apply there.
    """
    flat = " ".join(str(stem or "").split())
    if not flat:
        return True
    if _STEM_SLOT_GAP_RE.search(flat):
        return True
    floor = (
        resolve_min_stem_content_words()
        if min_content_words is None
        else min_content_words
    )
    if floor <= 0:
        return False
    return len(distinct_content_words(flat)) < floor


def distinct_content_words(text: str) -> frozenset[str]:
    """Distinct lower-cased non-function words in ``text``."""
    return frozenset(
        token.lower()
        for token in _TOKEN_RE.findall(str(text or ""))
        if token.lower() not in FUNCTION_WORDS
    )


def content_gate_eligibility(chunk: Mapping[str, Any]) -> PairEligibility:
    """Deterministic pre-dispatch content gate.

    Admits a chunk only when it carries something a generator can ground a
    completion in — ``concept_tags`` OR ``key_terms`` OR at least
    ``resolve_min_prose_words()`` words of prose — AND its leading stem is not
    auto-generated slot-filler residue. Runs before any provider dispatch, so
    an excluded unit costs no model call.

    Returns ``PairEligibility(True)`` unchanged when the gate is switched off.
    """
    if not resolve_content_gate_enabled():
        return PairEligibility(True)

    sources = describe_content_sources(chunk)
    min_prose = resolve_min_prose_words()
    if not (
        sources["concept_tags"]
        or sources["key_terms"]
        or sources["prose_words"] >= min_prose
    ):
        return PairEligibility(
            False,
            "chunk_carries_no_groundable_content",
            detail=(
                f"concept_tags={sources['concept_tags']}, "
                f"key_terms={sources['key_terms']}, "
                f"summary_chars={sources['summary_chars']}, "
                f"prose_words={sources['prose_words']} "
                f"(min_prose_words={min_prose})"
            ),
        )

    # The degenerate-stem check targets AUTO-GENERATED QUESTION ITEMS, which
    # are item-sized. It is deliberately NOT applied to the opening sentence
    # of a long instructional chunk (see _STEM_ITEM_MAX_WORD_MULTIPLE).
    item_word_ceiling = min_prose * _STEM_ITEM_MAX_WORD_MULTIPLE
    if sources["prose_words"] <= item_word_ceiling:
        text = resolve_chunk_text(chunk)
        stem = leading_stem(text)
        min_content = resolve_min_stem_content_words()
        # Signal 1 (slot gap) reads the STEM — that is where the template
        # slots live. Signal 2 (content-word floor) reads the WHOLE item, so
        # a real item that merely opens with a thin sentence
        # ("He bought 7 textbooks.") is not judged on that sentence alone.
        content_words = distinct_content_words(text)
        if (
            is_degenerate_stem(stem, min_content_words=0)
            or len(content_words) < min_content
        ):
            return PairEligibility(
                False,
                "degenerate_source_stem",
                detail=(
                    f"leading stem {stem[:120]!r} is slot-filler residue "
                    f"(prose_words={sources['prose_words']} <= item ceiling "
                    f"{item_word_ceiling}, distinct_content_words="
                    f"{len(content_words)}, min_stem_content_words="
                    f"{min_content})"
                ),
            )
    return PairEligibility(True)


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

    text = resolve_chunk_text(chunk)
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
    content_gate = content_gate_eligibility(focused_chunk)
    if not content_gate.eligible:
        return content_gate
    if not focused_chunk.get("learning_outcome_refs"):
        return PairEligibility(False, "missing_aligned_objective")

    focus = focused_chunk.get("synthesis_focus_objective")
    if not isinstance(focus, Mapping):
        return PairEligibility(False, "missing_canonical_objective_focus")
    try:
        build_evidence_window(focused_chunk, focus)
    except ValueError:
        return PairEligibility(False, "objective_evidence_window_not_viable")

    text = " ".join(resolve_chunk_text(focused_chunk).split())
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
