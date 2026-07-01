"""Pre-retrieval query / retrieval augmentation arms for the grounded-answer path.

Four opt-in, default-OFF, fail-open arms that run BEFORE / AT the step-1 retrieve
inside :func:`lib.retrieval.grounded_answer.answer_course_question`. Each is gated
by its own ``ED4ALL_ANSWER_*`` env flag; unset ⇒ the arm is a strict no-op and the
retrieved passage set + native scores are byte-identical to the legacy path.

Arms
----
- **W5.1 intent-route bias** (``ED4ALL_ANSWER_INTENT_ROUTE``) — classify the query
  via ``LibV2/tools/intent_router.py::classify_intent`` and BIAS (stable reorder,
  NOT a hard filter, NO set change) the retrieved passages so facet-matching
  chunks float to the front. Never drops a passage → cannot change the refusal
  verdict (the confidence gate reads set-membership, not order). Fail-open to the
  flat order.
- **W5.2 multi-turn antecedent rewrite** (``ED4ALL_ANSWER_MULTITURN``) — with an
  optional ``prior_turns`` transcript, deterministically LOCAL-rewrite (append the
  prior turn's salient terms) a follow-up query BEFORE retrieve. The rewrite feeds
  RETRIEVAL ONLY; the original query drives compose / the grounding gate. No model
  call. Verdict-safe: when the rewrite changes the query, the hook retrieves with
  BOTH the rewritten AND the original query and UNIONs them via
  :func:`merge_passage_candidates` (original passages as the ``flat`` base), so an
  off-topic prior-turn rewrite can never evict the original's best hits and turn a
  would-be answer into a refusal — same guarantee as decompose / HyDE below.
- **W5.6 decompose** (``ED4ALL_ANSWER_DECOMPOSE``) — split a multi-part question
  via the pure-lexical ``answer_completeness.split_question_parts`` splitter,
  retrieve per sub-query, and MERGE (union + dedup by chunk id, keep the
  higher-score instance, rank by native score, keep top ``limit``). The merged set
  is a superset-then-top-limit of the flat set, so the flat top-scorer is always
  retained → the verdict can only improve, never worsen.
- **W5.7 HyDE** (``ED4ALL_ANSWER_HYDE``) — generate a hypothetical answer via the
  LOCAL composer client, retrieve with it (the retriever embeds the hypothetical
  text for the semantic engine), and UNION with the literal-query retrieval. The
  hypothetical DOCUMENT never becomes a citation — only the real chunks it surfaces
  enter the candidate set. Same verdict-safe merge as decompose.

Anti-fabrication contract (all arms): every passage in the returned set is a REAL
chunk from a REAL retrieval call; no id is invented; the merge helpers keep ≥1
passage and only ever add / reorder real retrieved chunks. Each arm emits one
``DecisionCapture`` event with a dynamic, replayable rationale.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from dataclasses import replace as _dc_replace

# --------------------------------------------------------------------------- #
# Env flags (single source of truth for the ED4ALL_ANSWER_* augmentation knobs)
# --------------------------------------------------------------------------- #
ENV_INTENT_ROUTE = "ED4ALL_ANSWER_INTENT_ROUTE"
ENV_MULTITURN = "ED4ALL_ANSWER_MULTITURN"
ENV_DECOMPOSE = "ED4ALL_ANSWER_DECOMPOSE"
ENV_HYDE = "ED4ALL_ANSWER_HYDE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Decision-capture enum members (registered in
# schemas/events/decision_event.schema.json by the Integrate agent — this module
# only EMITS them; DECISION_VALIDATION_STRICT off ⇒ an unknown type is accepted).
DECISION_TYPE_INTENT_ROUTE = "grounded_answer_intent_route"
DECISION_TYPE_MULTITURN = "grounded_answer_multiturn_rewrite"
DECISION_TYPE_DECOMPOSE = "grounded_answer_decompose"
DECISION_TYPE_HYDE = "grounded_answer_hyde"

# HyDE generation bounds (local composer; deterministic temperature).
_HYDE_MAX_TOKENS = 256
_HYDE_MAX_CHARS = 2000
_HYDE_SYSTEM = (
    "You write a short, factual, single-paragraph passage that would plausibly "
    "answer the user's question, as if excerpted from a course textbook. Do not "
    "hedge, do not mention that it is hypothetical, do not add citations."
)

# Deterministic follow-up detection: a standalone antecedent-bearing pronoun.
_FOLLOWUP_PRONOUNS = frozenset({
    "it", "its", "it's", "that", "this", "they", "them", "those", "these",
    "he", "she", "him", "her", "one", "ones", "such",
})
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
_WEEK_RE = re.compile(r"(?i)week[\s_\-]*0*(\d{1,2})")

_STOPWORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "have", "has", "had", "what", "when", "where", "which", "who", "how", "why",
    "does", "did", "can", "could", "would", "should", "will", "your", "you",
    "about", "into", "than", "then", "them", "they", "there", "here", "its",
    "his", "her", "our", "out", "not", "but", "all", "any", "one", "two",
})


# --------------------------------------------------------------------------- #
# Flag resolvers (parse-with-fallback; default OFF)
# --------------------------------------------------------------------------- #
def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def intent_route_enabled() -> bool:
    """``ED4ALL_ANSWER_INTENT_ROUTE`` truthy? Default OFF."""
    return _flag(ENV_INTENT_ROUTE)


def multiturn_enabled() -> bool:
    """``ED4ALL_ANSWER_MULTITURN`` truthy? Default OFF."""
    return _flag(ENV_MULTITURN)


def decompose_enabled() -> bool:
    """``ED4ALL_ANSWER_DECOMPOSE`` truthy? Default OFF."""
    return _flag(ENV_DECOMPOSE)


def hyde_enabled() -> bool:
    """``ED4ALL_ANSWER_HYDE`` truthy? Default OFF."""
    return _flag(ENV_HYDE)


# --------------------------------------------------------------------------- #
# Decision capture (never breaks the answer path)
# --------------------------------------------------------------------------- #
def _safe_log(capture: Optional[Any], **kwargs: Any) -> None:
    if capture is None:
        return
    try:
        capture.log_decision(**kwargs)
    except Exception:  # pragma: no cover - capture must never break the path
        pass


# --------------------------------------------------------------------------- #
# Facet enrichment — stamp the RetrievalResult's facet fields into a passage's
# ``source`` dict so the intent-bias reorder can read them (only on the opt-in
# intent-route path; ``source`` is a free-form dict, extra keys are ignored by
# the citation gate).
# --------------------------------------------------------------------------- #
_FACET_FIELDS = ("chunk_type", "concept_tags", "learning_outcome_refs", "bloom_level")


def enrich_passage_facets(passage: Any, result: Any) -> Any:
    """Return a copy of ``passage`` whose ``source`` carries the result's facets.

    Reads ``chunk_type`` / ``concept_tags`` / ``learning_outcome_refs`` /
    ``bloom_level`` off the raw ``RetrievalResult`` (they live at the top level of
    the result, NOT inside ``chunk.source``) and folds any present value into a
    COPY of the passage source. A field already present on the source is not
    overwritten. Pure: the frozen passage is replaced, never mutated.
    """
    src = dict(getattr(passage, "source", {}) or {})
    changed = False
    for name in _FACET_FIELDS:
        if name in src and src.get(name) not in (None, [], ""):
            continue
        val = getattr(result, name, None)
        if val in (None, [], ""):
            continue
        src[name] = val
        changed = True
    if not changed:
        return passage
    try:
        return _dc_replace(passage, source=src)
    except Exception:  # pragma: no cover - non-dataclass duck type
        return passage


# --------------------------------------------------------------------------- #
# W5.1 — intent-route bias (stable reorder; NO set change; verdict-safe)
# --------------------------------------------------------------------------- #
def _passage_facet_score(source: Dict[str, Any], facets: Dict[str, Any]) -> int:
    """Additive facet-match score for one passage (higher ⇒ boosted forward)."""
    score = 0
    ctypes = facets.get("chunk_types") or []
    if ctypes and str(source.get("chunk_type") or "") in ctypes:
        score += 2
    obj_ids = {o.lower() for o in (facets.get("objective_ids") or [])}
    if obj_ids:
        refs = {str(r).lower() for r in (source.get("learning_outcome_refs") or [])}
        if obj_ids & refs:
            score += 2
    bloom_levels = facets.get("bloom_levels") or set()
    if bloom_levels and str(source.get("bloom_level") or "") in bloom_levels:
        score += 1
    weeks = facets.get("weeks") or []
    if weeks:
        hay = " ".join(
            str(source.get(k) or "")
            for k in ("item_path", "module_title", "section_heading", "lesson_title")
        )
        found = {int(m) for m in _WEEK_RE.findall(hay)}
        if set(weeks) & found:
            score += 1
    return score


def bias_passages_by_intent(
    query: str,
    passages: Sequence[Any],
    *,
    capture: Optional[Any] = None,
    course_slug: str = "",
    query_sha: str = "",
) -> List[Any]:
    """Stable-reorder ``passages`` to bias facet-matching chunks forward.

    Classifies the query intent (``classify_intent``) and computes a per-passage
    facet-match score against each passage's (facet-enriched) ``source``. Sorts
    DESCENDING by that score with a stable tie-break on the original retrieval
    order, so the retrieved SET is unchanged (no drop, no add) — only the order
    shifts. That guarantees the confidence / refusal gate (which reads
    set-membership, never order) is byte-identical, so the arm can NEVER change a
    verdict. Fail-open: any classification error returns the input order.
    """
    passages = list(passages)
    if not passages:
        return passages
    try:
        from LibV2.tools.intent_router import classify_intent

        routed = classify_intent(query or "")
        ents = routed.get("extracted_entities", {}) or {}
        facets = {
            "chunk_types": list(ents.get("chunk_types") or []),
            "objective_ids": list(ents.get("objective_ids") or []),
            "weeks": list(ents.get("weeks") or []),
            "bloom_levels": {lvl for (_v, lvl) in (ents.get("bloom_verbs") or [])},
        }
        scores = [
            _passage_facet_score(dict(getattr(p, "source", {}) or {}), facets)
            for p in passages
        ]
    except Exception:  # noqa: BLE001 - fail OPEN, original order
        _safe_log(
            capture,
            decision_type=DECISION_TYPE_INTENT_ROUTE,
            decision="intent_route:fallback",
            rationale=(
                f"intent-route bias fell open for query {query_sha or '?'} "
                f"(course={course_slug or '?'}): classification unavailable; "
                f"retrieved order preserved byte-identical ({len(passages)} passages)"
            ),
            context="fallback=1",
        )
        return passages

    order = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    reordered = [passages[i] for i in order]
    n_boosted = sum(1 for s in scores if s > 0)
    churn = sum(1 for a, b in zip(passages, reordered) if a is not b)
    _emit_intent_route(
        capture,
        course_slug=course_slug,
        query_sha=query_sha,
        intent_class=str(routed.get("intent_class", "?")),
        confidence=float(routed.get("confidence", 0.0) or 0.0),
        facets=facets,
        n_passages=len(passages),
        n_boosted=n_boosted,
        churn=churn,
    )
    return reordered


def _emit_intent_route(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    intent_class: str,
    confidence: float,
    facets: Dict[str, Any],
    n_passages: int,
    n_boosted: int,
    churn: int,
) -> None:
    facet_str = (
        f"chunk_types={facets.get('chunk_types')} "
        f"objective_ids={facets.get('objective_ids')} "
        f"weeks={facets.get('weeks')} "
        f"bloom_levels={sorted(facets.get('bloom_levels') or [])}"
    )
    rationale = (
        f"intent-route bias for query {query_sha or '?'} "
        f"(course={course_slug or '?'}) intent_class={intent_class} "
        f"confidence={confidence:.2f}; facets[{facet_str}]; "
        f"n_passages={n_passages} n_boosted={n_boosted} rank_churn={churn}; "
        f"stable reorder only (no set change) → refusal verdict byte-identical"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_INTENT_ROUTE,
        decision=f"intent_route:{intent_class}",
        rationale=rationale,
        context=f"boosted={n_boosted} churn={churn}",
    )


# --------------------------------------------------------------------------- #
# W5.2 — multi-turn deterministic antecedent rewrite (retrieval-only)
# --------------------------------------------------------------------------- #
def _looks_like_followup(query: str) -> bool:
    """A query is a follow-up if it carries a standalone antecedent pronoun or is
    very short (≤4 tokens) — both signal an unresolved reference to prior turns."""
    toks = [t.lower() for t in _WORD_RE.findall(query or "")]
    if not toks:
        return False
    if any(t in _FOLLOWUP_PRONOUNS for t in toks):
        return True
    return len(toks) <= 4


def _salient_terms(text: str, *, max_terms: int = 6) -> List[str]:
    """Deterministic salient-term extraction from a prior turn.

    Prefers capitalized multi-word phrases (proper nouns / titled concepts);
    falls back to content tokens (len>3, non-stopword) in first-seen order. Both
    deduped, order-preserving, capped — no model, fully deterministic.
    """
    out: List[str] = []
    seen: set = set()
    for phrase in _CAP_PHRASE_RE.findall(text or ""):
        key = phrase.lower()
        if key in seen or key in _STOPWORDS:
            continue
        seen.add(key)
        out.append(phrase)
        if len(out) >= max_terms:
            return out
    for tok in _WORD_RE.findall(text or ""):
        low = tok.lower()
        if len(low) <= 3 or low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        out.append(tok)
        if len(out) >= max_terms:
            break
    return out


def rewrite_query_with_context(
    query: str,
    prior_turns: Optional[Sequence[str]],
    *,
    capture: Optional[Any] = None,
    course_slug: str = "",
    query_sha: str = "",
) -> str:
    """Deterministically enrich a follow-up query with prior-turn salient terms.

    RETRIEVAL-ONLY: the returned string is used solely to drive retrieval; the
    ORIGINAL query still drives compose + the grounding gate. No model call. When
    the query is not a follow-up, there are no prior turns, or no salient terms
    resolve, the ORIGINAL query is returned unchanged (byte-identical retrieval).
    The rewrite is purely ADDITIVE (appends context terms) so it can only add
    retrieval signal, never fabricate a citation.
    """
    turns = [str(t) for t in (prior_turns or []) if str(t).strip()]
    if not turns or not _looks_like_followup(query):
        _emit_multiturn(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:not_followup_or_no_history",
            original=query, rewritten=query, terms=[], n_turns=len(turns),
        )
        return query
    # Most-recent prior turn is the strongest antecedent source; walk backwards
    # until some salient term resolves.
    terms: List[str] = []
    for turn in reversed(turns):
        terms = _salient_terms(turn)
        if terms:
            break
    if not terms:
        _emit_multiturn(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:no_salient_terms",
            original=query, rewritten=query, terms=[], n_turns=len(turns),
        )
        return query
    rewritten = f"{query.strip()} {' '.join(terms)}".strip()
    _emit_multiturn(
        capture, course_slug=course_slug, query_sha=query_sha,
        outcome="rewritten",
        original=query, rewritten=rewritten, terms=terms, n_turns=len(turns),
    )
    return rewritten


def _emit_multiturn(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    outcome: str,
    original: str,
    rewritten: str,
    terms: Sequence[str],
    n_turns: int,
) -> None:
    rationale = (
        f"multiturn antecedent rewrite {outcome} for query {query_sha or '?'} "
        f"(course={course_slug or '?'}); n_prior_turns={n_turns} "
        f"added_terms={list(terms)}; original={original!r} rewritten={rewritten!r}; "
        f"retrieval-only (original query still drives compose + grounding gate)"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_MULTITURN,
        decision=f"multiturn:{outcome}",
        rationale=rationale,
        context=f"terms={len(terms)} turns={n_turns}",
    )


# --------------------------------------------------------------------------- #
# Verdict-safe merge (shared by decompose + HyDE)
# --------------------------------------------------------------------------- #
def merge_passage_candidates(
    flat: Sequence[Any],
    extra_lists: Sequence[Sequence[Any]],
    *,
    limit: int,
) -> List[Any]:
    """Union ``flat`` with ``extra_lists``, dedup by chunk id, rank, trim.

    Dedup keeps the higher-``score`` instance for a chunk seen in more than one
    retrieval (a real score from a real retrieval — never fabricated). The merged
    list is sorted DESCENDING by native score with a STABLE tie-break that keeps
    first-seen (flat-first) order, then trimmed to ``limit`` (always keeping ≥1).

    Verdict-safety: the flat set's max-score passage is the global maximum and is
    always in the top-``limit`` slice, so the confidence gate's ``top_score`` and
    ``n_above_floor`` can only stay equal or grow — the merge NEVER makes the
    refusal verdict worse.
    """
    best: Dict[str, Any] = {}
    order: List[str] = []
    for p in flat:
        cid = str(getattr(p, "chunk_id", "") or "")
        if cid not in best:
            best[cid] = p
            order.append(cid)
    for lst in extra_lists:
        for p in lst:
            cid = str(getattr(p, "chunk_id", "") or "")
            cur = best.get(cid)
            if cur is None:
                best[cid] = p
                order.append(cid)
            elif float(getattr(p, "score", 0.0) or 0.0) > float(
                getattr(cur, "score", 0.0) or 0.0
            ):
                best[cid] = p
    merged = [best[cid] for cid in order]
    merged.sort(key=lambda p: -float(getattr(p, "score", 0.0) or 0.0))
    if limit and limit > 0:
        merged = merged[: max(1, int(limit))]
    return merged


# --------------------------------------------------------------------------- #
# W5.6 — decompose a multi-part question into sub-queries and merge
# --------------------------------------------------------------------------- #
def decompose_and_merge(
    query: str,
    flat_passages: Sequence[Any],
    retrieve_fn: Callable[[str, int], List[Any]],
    *,
    limit: int,
    capture: Optional[Any] = None,
    course_slug: str = "",
    query_sha: str = "",
) -> List[Any]:
    """Split a multi-part question, retrieve per sub-query, and merge (verdict-safe).

    Reuses the pure-lexical ``split_question_parts`` under-split splitter. A
    single-part question is a strict no-op (returns ``flat_passages``). Each
    sub-query is retrieved via ``retrieve_fn(subquery, limit)``; a per-part
    retrieval failure is swallowed (fail-open) so a bad sub-query never sinks the
    answer. Results union with the flat set via :func:`merge_passage_candidates`.
    """
    flat = list(flat_passages)
    try:
        from lib.retrieval.answer_completeness import split_question_parts

        parts = split_question_parts(query or "")
    except Exception:  # noqa: BLE001 - fail OPEN
        parts = [query]
    if len(parts) < 2:
        _emit_decompose(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:single_part", parts=parts,
            n_flat=len(flat), n_merged=len(flat), added_ids=[],
        )
        return flat

    flat_ids = {str(getattr(p, "chunk_id", "") or "") for p in flat}
    extra_lists: List[List[Any]] = []
    for part in parts:
        try:
            extra_lists.append(list(retrieve_fn(part, limit)))
        except Exception:  # noqa: BLE001 - per-part fail-open
            continue
    merged = merge_passage_candidates(flat, extra_lists, limit=limit)
    added_ids = [
        str(getattr(p, "chunk_id", "") or "")
        for p in merged
        if str(getattr(p, "chunk_id", "") or "") not in flat_ids
    ]
    _emit_decompose(
        capture, course_slug=course_slug, query_sha=query_sha,
        outcome="merged", parts=parts,
        n_flat=len(flat), n_merged=len(merged), added_ids=added_ids,
    )
    return merged


def _emit_decompose(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    outcome: str,
    parts: Sequence[str],
    n_flat: int,
    n_merged: int,
    added_ids: Sequence[str],
) -> None:
    parts_str = " | ".join(str(p) for p in parts)
    rationale = (
        f"decompose {outcome} for query {query_sha or '?'} "
        f"(course={course_slug or '?'}); n_parts={len(parts)} parts=[{parts_str}]; "
        f"n_flat={n_flat} n_merged={n_merged} added_chunk_ids={list(added_ids)}; "
        f"union+dedup+top-limit (flat top-scorer retained → verdict never worse)"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_DECOMPOSE,
        decision=f"decompose:{outcome}",
        rationale=rationale,
        context=f"parts={len(parts)} added={len(added_ids)}",
    )


# --------------------------------------------------------------------------- #
# W5.7 — HyDE (hypothetical-document-embedding) retrieval arm
# --------------------------------------------------------------------------- #
def generate_hypothetical_answer(client: Any, query: str) -> str:
    """Generate a short hypothetical answer passage via the LOCAL composer client.

    Uses the same ``chat_completion`` duck-type the composer calls. Deterministic
    temperature 0.0. The returned text is used ONLY as a retrieval query (the
    retriever embeds it) — it NEVER becomes a citation. Raises on any client
    failure so the caller can fail-open.
    """
    messages = [
        {"role": "system", "content": _HYDE_SYSTEM},
        {"role": "user", "content": str(query or "")},
    ]
    raw = client.chat_completion(
        messages, max_tokens=_HYDE_MAX_TOKENS, temperature=0.0
    )
    text = str(raw or "").strip()
    return text[:_HYDE_MAX_CHARS]


def hyde_augment(
    query: str,
    flat_passages: Sequence[Any],
    retrieve_fn: Callable[[str, int], List[Any]],
    gen_fn: Callable[[str], str],
    *,
    limit: int,
    capture: Optional[Any] = None,
    course_slug: str = "",
    query_sha: str = "",
) -> List[Any]:
    """Generate a hypothetical answer, retrieve with it, and union (verdict-safe).

    ``gen_fn(query) -> hypothetical_text`` is the LOCAL generation closure;
    ``retrieve_fn(text, limit)`` retrieves real chunks for that text. The
    hypothetical DOCUMENT is never returned / cited — only the real chunks it
    surfaces enter the candidate set, unioned with the flat set via
    :func:`merge_passage_candidates`. Fully fail-open: a generation error, an
    empty hypothetical, or a retrieval error returns ``flat_passages`` unchanged.
    """
    flat = list(flat_passages)
    try:
        hypo = gen_fn(query)
    except Exception as exc:  # noqa: BLE001 - fail OPEN
        _emit_hyde(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:generation_failed", hypo_len=0,
            n_flat=len(flat), n_merged=len(flat), added_ids=[],
            reason=type(exc).__name__,
        )
        return flat
    if not str(hypo or "").strip():
        _emit_hyde(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:empty_hypothetical", hypo_len=0,
            n_flat=len(flat), n_merged=len(flat), added_ids=[], reason=None,
        )
        return flat

    flat_ids = {str(getattr(p, "chunk_id", "") or "") for p in flat}
    try:
        hyde_results = list(retrieve_fn(str(hypo), limit))
    except Exception as exc:  # noqa: BLE001 - fail OPEN
        _emit_hyde(
            capture, course_slug=course_slug, query_sha=query_sha,
            outcome="noop:retrieval_failed", hypo_len=len(str(hypo)),
            n_flat=len(flat), n_merged=len(flat), added_ids=[],
            reason=type(exc).__name__,
        )
        return flat

    merged = merge_passage_candidates(flat, [hyde_results], limit=limit)
    added_ids = [
        str(getattr(p, "chunk_id", "") or "")
        for p in merged
        if str(getattr(p, "chunk_id", "") or "") not in flat_ids
    ]
    _emit_hyde(
        capture, course_slug=course_slug, query_sha=query_sha,
        outcome="merged", hypo_len=len(str(hypo)),
        n_flat=len(flat), n_merged=len(merged), added_ids=added_ids, reason=None,
    )
    return merged


def _emit_hyde(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    outcome: str,
    hypo_len: int,
    n_flat: int,
    n_merged: int,
    added_ids: Sequence[str],
    reason: Optional[str],
) -> None:
    reason_str = f" reason={reason}" if reason else ""
    rationale = (
        f"HyDE {outcome} for query {query_sha or '?'} "
        f"(course={course_slug or '?'}){reason_str}; hypothetical_len={hypo_len} "
        f"n_flat={n_flat} n_merged={n_merged} added_chunk_ids={list(added_ids)}; "
        f"hypothetical document never cited (retrieval-ranking only); "
        f"union+dedup+top-limit (flat top-scorer retained → verdict never worse)"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_HYDE,
        decision=f"hyde:{outcome}",
        rationale=rationale,
        context=f"added={len(added_ids)} hypo_len={hypo_len}",
    )


__all__ = [
    "ENV_INTENT_ROUTE",
    "ENV_MULTITURN",
    "ENV_DECOMPOSE",
    "ENV_HYDE",
    "DECISION_TYPE_INTENT_ROUTE",
    "DECISION_TYPE_MULTITURN",
    "DECISION_TYPE_DECOMPOSE",
    "DECISION_TYPE_HYDE",
    "intent_route_enabled",
    "multiturn_enabled",
    "decompose_enabled",
    "hyde_enabled",
    "enrich_passage_facets",
    "bias_passages_by_intent",
    "rewrite_query_with_context",
    "merge_passage_candidates",
    "decompose_and_merge",
    "generate_hypothetical_answer",
    "hyde_augment",
]
