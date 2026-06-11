"""Gold-candidate authoring (retrieval-answer-eval-set §2.2 steps 1-3 + 5).

P2 of the eval-set plan. Three surfaces, all driven by the license-clean LOCAL
provider only (never an Anthropic surface — the gold set is operator
MEASUREMENT data, but drafting routes through the local provider as defense in
depth per §2.1):

  1. **Stratified deterministic sampling** (:func:`sample_chunks`) over a
     course's pinned union chunkset by ``(week, CO, population, teaching_role)``
     with the §1.2-1.3 quotas (50-question taxonomy split; population quotas
     source>=10 / course>=25 / both>=5; 2x over-generation default n~=100).
     Seeded → deterministic.

  2. **Candidate drafting** (:func:`draft_candidates`) — for each sampled
     chunk the local model emits ``question_text``, ``question_type``, 2-4
     ``expected_key_points``, and a VERBATIM >=40-char quote it must copy from
     the chunk text. The deterministic fields (chunk anchors incl.
     ``content_sha256`` via :func:`gold_set.chunk_content_sha256`, population,
     ``objective_refs`` from the chunk's ``learning_outcome_refs``, difficulty
     heuristic per §1.3, ``synthesis_scope`` where applicable) are filled by
     this module, NOT the model.

  3. **Mechanical pre-screen** (:func:`prescreen_candidate`) — deterministic,
     runs after drafting: reject quote-not-contained (reuse
     ``gold_set._quote_in_chunk``), quote-ambiguous (>3 chunks, reuse
     ``gold_set._quote_chunk_match_count``), short question (<10 chars), and
     near-duplicates (shingle containment via ``lib.retrieval._text``).
     Rejected candidates are RECORDED with reasons in the output, not silently
     dropped.

Output: ``retrieval_eval/gold_candidates.json`` — a candidates wrapper carrying
the v1.1 question shape + a per-candidate ``authoring`` stamp
(``{method: llm_assisted, author: <model id + prompt version>, reviewed_by:
PENDING_REVIEW, status: draft}``) + ``prescreen`` verdicts. The promote step
(:func:`promote_candidates`) merges operator-accepted candidates into
``gold_set.json``.

Decision capture: drafting IS an LLM call path — every batch emits one
``gold_candidate_authoring`` decision (dynamic rationale: chunk ids, model,
accept/reject distribution). The embedded ``OpenAICompatibleClient`` ALSO emits
its own ``llm_chat_call`` per call; this module's per-batch event is the
authoring-semantics layer on top.

The candidates file and the promote merge are the operator-curation hinge
(§2.2 step 5): an operator edits ``gold_candidates.json`` in place — marking a
candidate accepted (see ACCEPT CONVENTION below) — then ``gold-promote`` renumbers
ids continuing from the existing gold set, backfills provenance, merges into
``gold_set.json``, and re-validates fail-closed.

ACCEPT CONVENTION
-----------------
A candidate is accepted when its ``authoring.status`` is edited from ``"draft"``
to ``"reviewed"`` AND ``authoring.reviewed_by`` is set to a non-PENDING handle.
``gold-promote`` promotes exactly the candidates that (a) are ``status:
reviewed``, (b) carry a non-``PENDING_REVIEW`` reviewer, and (c) passed
pre-screen. Everything else is left in the candidates file untouched. This reuses
the v1.1 ``authoring.status`` lifecycle (``seed`` / ``draft`` / ``reviewed``)
rather than inventing a parallel ``accept: true`` key — one lifecycle field, one
source of truth, and the promoted question lands in ``gold_set.json`` already
carrying the right ``status``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.retrieval._text import shingle_containment
from lib.retrieval.gold_coverage import (
    _population_of_chunk,
    _week_of_item_path,
)
from lib.retrieval.gold_set import (
    GOLD_SET_FILENAME,
    RETRIEVAL_EVAL_SUBDIR,
    _load_chunks_by_id,
    _quote_chunk_match_count,
    _quote_in_chunk,
    chunk_content_sha256,
    critical_issues,
    load_gold_set,
    validate_gold_set,
)
from lib.utils import sha256_file

# ---------------------------------------------------------------- constants

GOLD_CANDIDATES_FILENAME = "gold_candidates.json"

# Authoring prompt version — bump on any wording change to the drafting prompt
# so the ``authoring.author`` stamp records which prompt produced the question.
GOLD_AUTHORING_PROMPT_VERSION = "gold-author.v1"

# §1.2 answerable taxonomy split (per 50-question course target).
_TAXONOMY_TARGET = {
    "factual_recall": 15,
    "procedural": 10,
    "conceptual_synthesis": 12,
    "multi_part": 8,
    "where_covered": 5,
}
_DEFAULT_TARGET_QUESTIONS = 50
# 2x over-generation default (§2.2 step 3).
_DEFAULT_OVERGEN_FACTOR = 2

# §1.3 population quotas (union corpora).
_POPULATION_QUOTA = {"source": 10, "course": 25, "both": 5}

# Pre-screen thresholds.
_MIN_QUESTION_CHARS = 10
_MIN_QUOTE_CHARS = 40
_AMBIGUOUS_QUOTE_MAX_CHUNKS = 3
_NEAR_DUP_CONTAINMENT = 0.80   # shingle-containment >= this => near-duplicate
_NEAR_DUP_SHINGLE_SIZE = 4

# Deterministic key-point bounds (schema: 2-6; drafting targets 2-4).
_KEY_POINTS_MIN = 2
_KEY_POINTS_MAX = 4

_QUESTION_ID_NUM_RE = re.compile(r"-(\d{4})$")


# ---------------------------------------------------------------- data types


@dataclass(frozen=True)
class SampleSlot:
    """One sampled chunk + the stratum it represents."""

    chunk_id: str
    question_type: str
    week: Optional[str]
    population: str
    teaching_role: str
    objective_refs: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "question_type": self.question_type,
            "week": self.week,
            "population": self.population,
            "teaching_role": self.teaching_role,
            "objective_refs": list(self.objective_refs),
        }


@dataclass
class PrescreenVerdict:
    """Deterministic pre-screen outcome for one candidate."""

    passed: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons)}


# ---------------------------------------------------------------- sampler


def _teaching_role(chunk: Dict[str, Any]) -> str:
    """Best-effort teaching-role signal for stratification."""
    role = chunk.get("teaching_role")
    if isinstance(role, str) and role:
        return role
    ctl = chunk.get("content_type_label")
    if isinstance(ctl, str) and ctl:
        return ctl
    return "unknown"


def _objective_refs(chunk: Dict[str, Any]) -> Tuple[str, ...]:
    refs = chunk.get("learning_outcome_refs") or []
    out = tuple(sorted({r for r in refs if isinstance(r, str) and r}))
    return out


def _stratum_key(chunk: Dict[str, Any]) -> Tuple[Optional[str], str, str, str]:
    """The (week, CO, population, teaching_role) stratum for a chunk.

    CO is the chunk's lowest chapter-objective ref (deterministic pick) or the
    empty string when none — so a chunk with no CO still lands in a stratum.
    """
    source = chunk.get("source") or {}
    week = _week_of_item_path(source.get("item_path", ""))
    population = _population_of_chunk(chunk)
    role = _teaching_role(chunk)
    refs = _objective_refs(chunk)
    co = ""
    for r in refs:
        if r.lower().startswith("co"):
            co = r
            break
    return (week, co, population, role)


def _type_for_index(idx: int, plan: Sequence[str]) -> str:
    """Round-robin a question-type plan list, clamped to its length."""
    return plan[idx % len(plan)] if plan else "factual_recall"


def _build_type_plan(n: int) -> List[str]:
    """Build a question-type assignment list of length ``n`` honoring the
    §1.2 taxonomy split, scaled to ``n`` (over-generation keeps the ratio).

    Deterministic: types are emitted in a fixed canonical order, interleaved so
    the per-stratum assignment doesn't clump one type.
    """
    base_total = sum(_TAXONOMY_TARGET.values()) or 1
    # Scale each target to n, floor, then distribute the remainder by the
    # canonical type order so the sum is exactly n.
    scaled: Dict[str, int] = {}
    for t, q in _TAXONOMY_TARGET.items():
        scaled[t] = (q * n) // base_total
    remainder = n - sum(scaled.values())
    order = list(_TAXONOMY_TARGET.keys())
    i = 0
    while remainder > 0:
        scaled[order[i % len(order)]] += 1
        remainder -= 1
        i += 1
    # Interleave: round-robin draw one of each remaining type until exhausted.
    plan: List[str] = []
    pools = {t: scaled[t] for t in order}
    while len(plan) < n:
        progressed = False
        for t in order:
            if pools[t] > 0:
                plan.append(t)
                pools[t] -= 1
                progressed = True
                if len(plan) >= n:
                    break
        if not progressed:
            break
    return plan[:n]


def sample_chunks(
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    n: int,
    seed: int = 0,
    is_union: bool = True,
) -> List[SampleSlot]:
    """Stratified deterministic sample of ``n`` chunks for candidate drafting.

    Strata are ``(week, CO, population, teaching_role)``. The sampler walks the
    strata in a seed-rotated deterministic order and draws one chunk per
    stratum round-robin until ``n`` slots are filled (or chunks are exhausted),
    so coverage spreads across weeks / objectives / populations / roles rather
    than clumping on the densest stratum. Each slot is then assigned a
    question_type from the §1.2-scaled taxonomy plan.

    On a union corpus the population quotas (§1.3: source>=10 / course>=25 /
    both>=5, scaled to ``n``) bias the draw: a population that is below its
    scaled quota is preferred when breaking ties between otherwise-equal strata.

    Pure + deterministic for a fixed ``(chunks_by_id, n, seed)``.
    """
    # Build strata: ordered map stratum_key -> [chunk_id, ...] (ids sorted for
    # determinism).
    strata: Dict[Tuple[Optional[str], str, str, str], List[str]] = {}
    pop_of: Dict[str, str] = {}
    for cid in sorted(chunks_by_id):
        chunk = chunks_by_id[cid]
        key = _stratum_key(chunk)
        strata.setdefault(key, []).append(cid)
        pop_of[cid] = key[2]

    stratum_keys = sorted(strata.keys(), key=lambda k: (str(k[0]), k[1], k[2], k[3]))
    if not stratum_keys:
        return []

    # Seed-derived per-stratum tiebreak rank: a deterministic hash of the
    # stratum key salted by the seed. Under-quota population bias stays the
    # PRIMARY sort key; the seed only re-orders strata that are otherwise tied,
    # so two seeds explore the same strata in a different order without ever
    # violating the population quotas. (A bare rotate-by-seed was insufficient
    # because the per-iteration priority re-sort collapsed it on small corpora.)
    def _seed_rank(k: Tuple) -> int:
        h = hashlib.sha256(f"{seed}:{k[0]}|{k[1]}|{k[2]}|{k[3]}".encode()).hexdigest()
        return int(h[:8], 16)

    seed_rank = {k: _seed_rank(k) for k in stratum_keys}

    # Scaled population quotas (union only).
    pop_quota: Dict[str, int] = {}
    if is_union:
        base = sum(_POPULATION_QUOTA.values()) or 1
        for p, q in _POPULATION_QUOTA.items():
            pop_quota[p] = max(1, (q * n) // base)

    chosen: List[str] = []
    chosen_set = set()
    pop_count: Dict[str, int] = {"source": 0, "course": 0, "both": 0}
    cursor: Dict[Tuple, int] = {k: 0 for k in stratum_keys}

    def _stratum_priority(k: Tuple) -> Tuple:
        """Lower sorts first. Prefer strata whose population is under quota;
        break remaining ties by the seed-derived rank (then the key itself)."""
        pop = k[2]
        under = 0
        if is_union and pop in pop_quota and pop_count.get(pop, 0) < pop_quota[pop]:
            under = -1  # under-quota → higher priority (sorts first)
        return (under, seed_rank[k], str(k[0]), k[1], k[2], k[3])

    # Round-robin draw across strata until n filled or all exhausted.
    while len(chosen) < n:
        progressed = False
        for k in sorted(stratum_keys, key=_stratum_priority):
            ids = strata[k]
            idx = cursor[k]
            while idx < len(ids) and ids[idx] in chosen_set:
                idx += 1
            if idx < len(ids):
                cid = ids[idx]
                cursor[k] = idx + 1
                chosen.append(cid)
                chosen_set.add(cid)
                pop_count[pop_of[cid]] = pop_count.get(pop_of[cid], 0) + 1
                progressed = True
                if len(chosen) >= n:
                    break
            else:
                cursor[k] = idx
        if not progressed:
            break

    type_plan = _build_type_plan(len(chosen))
    slots: List[SampleSlot] = []
    for i, cid in enumerate(chosen):
        chunk = chunks_by_id[cid]
        key = _stratum_key(chunk)
        slots.append(
            SampleSlot(
                chunk_id=cid,
                question_type=_type_for_index(i, type_plan),
                week=key[0],
                population=key[2],
                teaching_role=key[3],
                objective_refs=_objective_refs(chunk),
            )
        )
    return slots


# ---------------------------------------------------------------- difficulty


def _difficulty_heuristic(question_type: str, n_passages: int, weeks: int) -> str:
    """§1.3 operationalized difficulty.

    easy   = verbatim in one chunk (single-passage factual_recall);
    medium = paraphrase / 2-chunk join;
    hard   = >=3 chunks, across-weeks, or a partially-covered multi_part.
    """
    if question_type == "multi_part":
        return "hard"
    if n_passages >= 3 or weeks >= 2:
        return "hard"
    if question_type == "factual_recall" and n_passages <= 1:
        return "easy"
    if n_passages <= 1:
        return "easy"
    return "medium"


# ---------------------------------------------------------------- drafting


class GoldDraftError(Exception):
    """Raised when the model response can't be parsed into a candidate draft."""


def _draft_prompt(chunk: Dict[str, Any], question_type: str) -> Tuple[str, str]:
    """Return ``(system, user)`` prompts for one candidate draft.

    The model emits ONLY question semantics; the deterministic fields (anchors,
    population, ids) are filled by this module.
    """
    system = (
        "You author one evaluation question grounded in a single source chunk. "
        "Output JSON only with keys: question_text (a clear question of at "
        "least 10 characters), expected_key_points (a JSON array of 2 to 4 "
        "short factual points the correct answer must state), and quote (a "
        "VERBATIM substring copied from the source chunk text, at least 40 "
        "characters, that contains the answer). Copy the quote character for "
        "character from the chunk — do not paraphrase it. Do not add facts "
        "not present in the chunk."
    )
    type_hint = {
        "factual_recall": "Ask for a single definitional or verbatim fact.",
        "procedural": "Ask about the ordered steps of a procedure; key points are the steps.",
        "conceptual_synthesis": "Ask a question whose answer synthesizes ideas in the chunk.",
        "multi_part": "Ask a question with two or more enumerable parts.",
        "where_covered": "Ask where the course covers a topic this chunk introduces.",
    }.get(question_type, "Ask a clear question answerable from the chunk.")
    chunk_text = str(chunk.get("text") or "")
    user = (
        f"Question type: {question_type}. {type_hint}\n\n"
        f"Source chunk text:\n{chunk_text}\n\n"
        'Output JSON only: {"question_text": "...", '
        '"expected_key_points": ["...", "..."], "quote": "<verbatim substring>"}'
    )
    return system, user


def _parse_draft(text: str) -> Dict[str, Any]:
    """Parse a model draft response into ``{question_text, expected_key_points,
    quote}``. Tolerates markdown-fenced / prose-wrapped JSON (the same 7B-Q4
    drift the local provider handles)."""
    raw = (text or "").strip()
    parsed: Optional[Dict[str, Any]] = None
    # Direct.
    try:
        cand = json.loads(raw)
        if isinstance(cand, dict):
            parsed = cand
    except json.JSONDecodeError:
        parsed = None
    # Fenced / embedded object scan.
    if parsed is None:
        start = raw.find("{")
        depth = 0
        for i in range(start, len(raw)) if start >= 0 else []:
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        cand = json.loads(raw[start:i + 1])
                        if isinstance(cand, dict):
                            parsed = cand
                    except json.JSONDecodeError:
                        parsed = None
                    break
    if not isinstance(parsed, dict):
        raise GoldDraftError(f"no JSON object recoverable from draft tail {raw[-120:]!r}")
    qt = parsed.get("question_text")
    kps = parsed.get("expected_key_points")
    quote = parsed.get("quote")
    if not isinstance(qt, str):
        raise GoldDraftError("draft missing question_text")
    if not isinstance(quote, str):
        raise GoldDraftError("draft missing quote")
    if not isinstance(kps, list):
        kps = []
    kps = [str(k).strip() for k in kps if isinstance(k, (str, int, float)) and str(k).strip()]
    # Clamp to schema bounds (2..6; we target 2..4).
    kps = kps[:_KEY_POINTS_MAX]
    return {"question_text": qt.strip(), "expected_key_points": kps, "quote": quote.strip()}


def _build_candidate(
    slot: SampleSlot,
    chunk: Dict[str, Any],
    draft: Dict[str, Any],
    *,
    model_id: str,
) -> Dict[str, Any]:
    """Assemble a v1.1-shaped candidate question from a slot + parsed draft."""
    source = chunk.get("source") or {}
    item_path = str(source.get("item_path") or "")
    quote = draft["quote"]
    anchor: Dict[str, Any] = {
        "item_path": item_path,
        "text_quote": quote,
        "content_sha256": chunk_content_sha256(chunk),
    }
    heading = source.get("section_heading")
    if isinstance(heading, str) and heading:
        anchor["section_heading"] = heading

    n_passages = 1
    weeks = 1 if slot.week else 0
    difficulty = _difficulty_heuristic(slot.question_type, n_passages, weeks)

    candidate: Dict[str, Any] = {
        # question_id is a placeholder until promotion renumbers it; kept
        # candidate-local-unique so the candidates file is self-consistent.
        "question_id": f"gqc-{slot.chunk_id}",
        "question_text": draft["question_text"],
        "question_type": slot.question_type,
        "difficulty": difficulty,
        "relevant_passages": [
            {"chunk_id": slot.chunk_id, "relevance": "primary", "anchor": anchor}
        ],
        "authoring": {
            "method": "llm_assisted",
            "author": f"{model_id}/{GOLD_AUTHORING_PROMPT_VERSION}",
            "reviewed_by": "PENDING_REVIEW",
            "status": "draft",
        },
    }
    if slot.objective_refs:
        candidate["objective_refs"] = list(slot.objective_refs)
    kps = draft.get("expected_key_points") or []
    if len(kps) >= _KEY_POINTS_MIN:
        candidate["expected_key_points"] = kps[:_KEY_POINTS_MAX]
    if slot.population in ("source", "course", "both"):
        candidate["expected_citation_population"] = slot.population
    # multi_part requires a parts[] block; supply a minimal 2-part scaffold the
    # operator fills in during curation (covered:true, anchored to the primary).
    if slot.question_type == "multi_part":
        candidate["parts"] = [
            {"part_id": "a", "part_text": "PART A — operator: fill in",
             "covered": True, "relevant_passage_refs": [slot.chunk_id]},
            {"part_id": "b", "part_text": "PART B — operator: fill in",
             "covered": True, "relevant_passage_refs": [slot.chunk_id]},
        ]
    return candidate


def draft_candidates(
    slots: Sequence[SampleSlot],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Draft a candidate question per slot via the local-provider ``client``.

    ``client`` is duck-typed on ``chat_completion(messages, *, max_tokens=,
    temperature=)`` (the :class:`OpenAICompatibleClient` /
    :class:`FakeAnswerClient` surface), so tests inject a deterministic fake and
    production passes a local-server-backed client. NEVER an Anthropic surface.

    A model call that fails to parse is recorded as a candidate carrying a
    ``draft_error`` + a failing pre-screen (so the rejection is audited, not
    silently dropped). One ``gold_candidate_authoring`` decision is emitted
    after the batch with a dynamic rationale.
    """
    resolved_model = model_id or getattr(client, "model", "local")
    candidates: List[Dict[str, Any]] = []
    errors = 0
    for slot in slots:
        chunk = chunks_by_id.get(slot.chunk_id) or {}
        system, user = _draft_prompt(chunk, slot.question_type)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = client.chat_completion(messages, max_tokens=512, temperature=0.2)
            draft = _parse_draft(text)
            cand = _build_candidate(slot, chunk, draft, model_id=resolved_model)
        except (GoldDraftError, Exception) as exc:  # noqa: BLE001 — record, don't drop
            errors += 1
            cand = {
                "question_id": f"gqc-{slot.chunk_id}",
                "question_text": "",
                "question_type": slot.question_type,
                "relevant_passages": [
                    {"chunk_id": slot.chunk_id, "relevance": "primary",
                     "anchor": {"item_path": str((chunk.get("source") or {}).get("item_path") or ""),
                                "text_quote": ""}}
                ],
                "authoring": {
                    "method": "llm_assisted",
                    "author": f"{resolved_model}/{GOLD_AUTHORING_PROMPT_VERSION}",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "draft",
                },
                "draft_error": str(exc),
            }
        candidates.append(cand)

    if capture is not None:
        accepted_drafts = len(candidates) - errors
        chunk_ids = ",".join(s.chunk_id for s in slots[:8])
        more = "" if len(slots) <= 8 else f" (+{len(slots) - 8} more)"
        try:
            capture.log_decision(
                decision_type="gold_candidate_authoring",
                decision=(
                    f"drafted {len(candidates)} gold candidate(s) via local "
                    f"model {resolved_model}; {errors} parse-failure(s)."
                ),
                rationale=(
                    f"Stratified gold-candidate drafting over {len(slots)} sampled "
                    f"chunk(s) [{chunk_ids}{more}] using license-clean local model "
                    f"{resolved_model} (prompt {GOLD_AUTHORING_PROMPT_VERSION}); "
                    f"parsed_ok={accepted_drafts}, parse_failures={errors}. "
                    f"question_type plan honours the §1.2 50-question taxonomy "
                    f"split; quotes are pre-screened deterministically for "
                    f"verbatim containment + ambiguity downstream."
                ),
            )
        except Exception:  # pragma: no cover — defensive
            pass
    return candidates


# ---------------------------------------------------------------- prescreen


def prescreen_candidate(
    candidate: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    prior_questions: Sequence[str] = (),
) -> PrescreenVerdict:
    """Deterministic mechanical pre-screen of one candidate (§2.2 step 3).

    Rejects (records ALL applicable reasons):
      * ``DRAFT_ERROR`` — the model response failed to parse.
      * ``QUESTION_TOO_SHORT`` — question_text < 10 chars.
      * ``QUOTE_TOO_SHORT`` — primary quote < 40 chars.
      * ``QUOTE_NOT_IN_CHUNK`` — quote not normalized-contained in its chunk.
      * ``QUOTE_AMBIGUOUS`` — quote contained in > 3 chunks of the set.
      * ``NEAR_DUPLICATE`` — question shingle-contained >= 0.80 in a prior
        question (drafted earlier this batch or already in the gold set).
    """
    reasons: List[str] = []
    if candidate.get("draft_error"):
        reasons.append("DRAFT_ERROR")
    qt = str(candidate.get("question_text") or "")
    if len(qt.strip()) < _MIN_QUESTION_CHARS:
        reasons.append("QUESTION_TOO_SHORT")

    passages = candidate.get("relevant_passages") or []
    primary = next(
        (p for p in passages if isinstance(p, dict) and p.get("relevance") == "primary"),
        passages[0] if passages and isinstance(passages[0], dict) else None,
    )
    if isinstance(primary, dict):
        anchor = primary.get("anchor") or {}
        quote = str(anchor.get("text_quote") or "")
        cid = primary.get("chunk_id")
        if len(quote.strip()) < _MIN_QUOTE_CHARS:
            reasons.append("QUOTE_TOO_SHORT")
        elif isinstance(cid, str) and cid in chunks_by_id:
            if not _quote_in_chunk(quote, chunks_by_id[cid]):
                reasons.append("QUOTE_NOT_IN_CHUNK")
            else:
                if _quote_chunk_match_count(quote, chunks_by_id) > _AMBIGUOUS_QUOTE_MAX_CHUNKS:
                    reasons.append("QUOTE_AMBIGUOUS")
        elif isinstance(cid, str):
            reasons.append("QUOTE_NOT_IN_CHUNK")  # chunk id absent => unanchorable

    # Near-dup: only meaningful for a non-empty question.
    if qt.strip():
        for prior in prior_questions:
            if not prior or not prior.strip():
                continue
            if shingle_containment(
                qt, prior, shingle_size=_NEAR_DUP_SHINGLE_SIZE
            ) >= _NEAR_DUP_CONTAINMENT:
                reasons.append("NEAR_DUPLICATE")
                break

    return PrescreenVerdict(passed=not reasons, reasons=reasons)


def prescreen_candidates(
    candidates: Sequence[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    existing_questions: Sequence[str] = (),
) -> List[Tuple[Dict[str, Any], PrescreenVerdict]]:
    """Pre-screen a batch; near-dup checks accumulate intra-batch + against
    ``existing_questions`` (the gold set's question_texts)."""
    seen: List[str] = list(existing_questions)
    out: List[Tuple[Dict[str, Any], PrescreenVerdict]] = []
    for cand in candidates:
        verdict = prescreen_candidate(cand, chunks_by_id, prior_questions=seen)
        out.append((cand, verdict))
        qt = str(cand.get("question_text") or "").strip()
        if qt:
            seen.append(qt)
    return out


# ---------------------------------------------------------------- write/build


def build_candidates_doc(
    course_slug: str,
    chunkset: Dict[str, Any],
    screened: Sequence[Tuple[Dict[str, Any], PrescreenVerdict]],
    *,
    n_requested: int,
    seed: int,
    model_id: str,
) -> Dict[str, Any]:
    """Build the ``gold_candidates.json`` wrapper doc."""
    candidates_out: List[Dict[str, Any]] = []
    for cand, verdict in screened:
        entry = dict(cand)
        entry["prescreen"] = verdict.to_dict()
        candidates_out.append(entry)
    passed = sum(1 for _, v in screened if v.passed)
    return {
        "schema_version": "1.1",
        "course_slug": course_slug,
        "chunkset": dict(chunkset),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authoring_run": {
            "n_requested": n_requested,
            "n_drafted": len(candidates_out),
            "n_prescreen_passed": passed,
            "seed": seed,
            "model_id": model_id,
            "prompt_version": GOLD_AUTHORING_PROMPT_VERSION,
        },
        "candidates": candidates_out,
    }


def generate_gold_candidates(
    course_dir: Path,
    *,
    client: Any,
    n: Optional[int] = None,
    seed: int = 0,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
    write: bool = True,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """End-to-end §2.2 steps 1-3: sample → draft → pre-screen → write doc.

    Loads the course's pinned chunkset from its gold set's ``chunkset`` pin,
    samples ``n`` chunks (default 2x the 50-question target), drafts a candidate
    per slot via ``client``, pre-screens deterministically, and writes
    ``retrieval_eval/gold_candidates.json``. Returns ``(doc, written_path)``.

    Raises ``FileNotFoundError`` when the gold set / chunkset is absent (the
    chunkset pin is the source of truth for which corpus to sample).
    """
    course_dir = Path(course_dir)
    gold, _ = load_gold_set(course_dir, verify=False)
    chunkset = (gold.get("chunkset") or {}) if gold else {}
    chunks_rel = chunkset.get("chunks_path")
    if not chunks_rel:
        raise FileNotFoundError(
            f"no chunkset pin in gold set at {course_dir}; run gold-validate / "
            f"gold-repin to pin a chunkset before drafting candidates."
        )
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        raise FileNotFoundError(f"pinned chunkset not found at {chunks_path}.")
    chunks_by_id = _load_chunks_by_id(chunks_path)

    n_resolved = int(n) if n else _DEFAULT_TARGET_QUESTIONS * _DEFAULT_OVERGEN_FACTOR
    is_union = chunkset.get("kind") == "corpus"
    slots = sample_chunks(chunks_by_id, n=n_resolved, seed=seed, is_union=is_union)
    candidates = draft_candidates(
        slots, chunks_by_id, client=client, model_id=model_id, capture=capture
    )
    existing = [
        str(q.get("question_text") or "")
        for q in (gold.get("questions") or [])
        if isinstance(q, dict)
    ]
    screened = prescreen_candidates(candidates, chunks_by_id, existing_questions=existing)
    resolved_model = model_id or getattr(client, "model", "local")
    doc = build_candidates_doc(
        gold.get("course_slug", course_dir.name),
        chunkset,
        screened,
        n_requested=n_resolved,
        seed=seed,
        model_id=resolved_model,
    )
    out_path: Optional[Path] = None
    if write:
        out_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / GOLD_CANDIDATES_FILENAME
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc, out_path


# ---------------------------------------------------------------- promote


class GoldPromoteError(Exception):
    """Raised on a fail-closed promotion refusal."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass
class PromoteReport:
    """Outcome of a candidate-promotion pass."""

    course_slug: str
    promoted_ids: List[str] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    refused: List[Dict[str, Any]] = field(default_factory=list)
    next_ordinal: int = 0
    frozen: bool = False
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "course_slug": self.course_slug,
            "generated_at": self.generated_at,
            "promoted_ids": list(self.promoted_ids),
            "promoted_count": len(self.promoted_ids),
            "skipped": self.skipped,
            "refused": self.refused,
            "next_ordinal": self.next_ordinal,
            "frozen": self.frozen,
        }


def _max_ordinal(questions: Sequence[Dict[str, Any]]) -> int:
    hi = 0
    for q in questions:
        qid = q.get("question_id") if isinstance(q, dict) else None
        if isinstance(qid, str):
            m = _QUESTION_ID_NUM_RE.search(qid)
            if m:
                hi = max(hi, int(m.group(1)))
    return hi


def _is_accepted(candidate: Dict[str, Any]) -> bool:
    """ACCEPT CONVENTION: status==reviewed AND reviewer != PENDING_REVIEW."""
    auth = candidate.get("authoring") or {}
    return (
        auth.get("status") == "reviewed"
        and isinstance(auth.get("reviewed_by"), str)
        and auth.get("reviewed_by")
        and auth.get("reviewed_by") != "PENDING_REVIEW"
    )


def _strip_candidate_fields(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Drop candidate-only fields (prescreen, draft_error) for the gold doc."""
    out = {k: v for k, v in candidate.items() if k not in ("prescreen", "draft_error")}
    return out


def promote_candidates_into_gold(
    gold: Dict[str, Any],
    candidates_doc: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    chunks_sha256: str,
    *,
    course_slug: str,
    freeze: bool = False,
) -> Tuple[Dict[str, Any], PromoteReport]:
    """Merge accepted + pre-screen-passing candidates into a gold doc (pure).

    Renumbers ``gq-<slug>-NNNN`` continuing from the existing max ordinal,
    backfills provenance, and RE-VALIDATES the merged doc fail-closed
    (:func:`validate_gold_set`). Refuses to promote any candidate that fails
    re-screen or whose merge would make the gold set invalid — a refusing
    candidate is recorded, the rest still land.

    Returns ``(new_gold, report)``. ``new_gold`` is a deep copy; the input is
    never mutated.
    """
    gold = json.loads(json.dumps(gold))
    report = PromoteReport(
        course_slug=course_slug,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        frozen=bool(gold.get("frozen")),
    )
    questions: List[Dict[str, Any]] = [
        q for q in (gold.get("questions") or []) if isinstance(q, dict)
    ]
    existing_texts = [str(q.get("question_text") or "") for q in questions]
    ordinal = _max_ordinal(questions)

    for cand in candidates_doc.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        cand_label = cand.get("question_id", "<unknown>")
        if not _is_accepted(cand):
            report.skipped.append(
                {"candidate": cand_label, "reason": "not_accepted"}
            )
            continue
        # Re-run the deterministic pre-screen against the live chunkset +
        # accumulated question texts (curation may have edited the quote).
        verdict = prescreen_candidate(cand, chunks_by_id, prior_questions=existing_texts)
        if not verdict.passed:
            report.refused.append(
                {"candidate": cand_label, "reason": "prescreen_failed",
                 "details": verdict.reasons}
            )
            continue
        ordinal += 1
        promoted = _strip_candidate_fields(cand)
        promoted["question_id"] = f"gq-{course_slug}-{ordinal:04d}"
        auth = dict(promoted.get("authoring") or {})
        auth.setdefault("method", "llm_assisted")
        auth.setdefault("status", "reviewed")
        promoted["authoring"] = auth
        questions.append(promoted)
        existing_texts.append(str(promoted.get("question_text") or ""))
        report.promoted_ids.append(promoted["question_id"])

    gold["questions"] = questions
    if freeze and report.promoted_ids:
        gold["frozen"] = True
        report.frozen = True
    # Refresh the chunkset sha pin (freeze records the live bytes).
    if isinstance(gold.get("chunkset"), dict):
        gold["chunkset"]["chunks_sha256"] = chunks_sha256
    report.next_ordinal = ordinal

    # Re-validate fail-closed; a critical issue refuses the WHOLE promotion.
    issues = validate_gold_set(gold, chunks_by_id, chunks_sha256)
    crit = critical_issues(issues)
    if crit:
        raise GoldPromoteError(
            "GOLD_PROMOTE_VALIDATION_FAILED",
            "; ".join(f"[{i.code}] {i.message}" for i in crit[:5]),
        )
    return gold, report


def promote_candidates(
    course_dir: Path,
    *,
    freeze: bool = False,
    dry_run: bool = False,
) -> Tuple[PromoteReport, Optional[Path]]:
    """I/O entry: merge accepted candidates from ``gold_candidates.json`` into
    ``gold_set.json``, re-validate, and (unless ``dry_run``) write both the
    updated gold set + a promote-report sidecar.

    Refuses on a frozen gold set (a frozen set is the canonical eval pin; adding
    questions silently would break comparability) — re-pin / unfreeze first.
    """
    course_dir = Path(course_dir)
    eval_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
    gold_path = eval_dir / GOLD_SET_FILENAME
    cand_path = eval_dir / GOLD_CANDIDATES_FILENAME
    if not gold_path.is_file():
        raise GoldPromoteError("GOLD_SET_NOT_FOUND", f"no gold set at {gold_path}")
    if not cand_path.is_file():
        raise GoldPromoteError(
            "GOLD_CANDIDATES_NOT_FOUND", f"no candidates file at {cand_path}"
        )
    try:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        cand_doc = json.loads(cand_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldPromoteError("INVALID_JSON", str(exc)) from exc

    if gold.get("frozen"):
        raise GoldPromoteError(
            "GOLD_SET_FROZEN",
            "gold set is frozen; re-pin / unfreeze before promoting candidates.",
        )

    course_slug = gold.get("course_slug") or course_dir.name
    chunks_rel = (gold.get("chunkset") or {}).get("chunks_path")
    if not chunks_rel:
        raise GoldPromoteError("NO_CHUNKSET_PIN", "gold set has no chunkset pin.")
    chunks_path = course_dir / chunks_rel
    if not chunks_path.is_file():
        raise GoldPromoteError(
            "GOLD_SET_CHUNKSET_NOT_FOUND", f"pinned chunkset not found at {chunks_path}."
        )
    chunks_sha = sha256_file(chunks_path)
    chunks_by_id = _load_chunks_by_id(chunks_path)

    new_gold, report = promote_candidates_into_gold(
        gold, cand_doc, chunks_by_id, chunks_sha,
        course_slug=course_slug, freeze=freeze,
    )

    if dry_run:
        return report, None
    gold_path.write_text(json.dumps(new_gold, indent=2) + "\n", encoding="utf-8")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = eval_dir / f"gold_promote_report_{ts}.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report, gold_path


__all__ = [
    "GOLD_CANDIDATES_FILENAME",
    "GOLD_AUTHORING_PROMPT_VERSION",
    "SampleSlot",
    "PrescreenVerdict",
    "PromoteReport",
    "GoldDraftError",
    "GoldPromoteError",
    "sample_chunks",
    "draft_candidates",
    "prescreen_candidate",
    "prescreen_candidates",
    "build_candidates_doc",
    "generate_gold_candidates",
    "promote_candidates_into_gold",
    "promote_candidates",
]
