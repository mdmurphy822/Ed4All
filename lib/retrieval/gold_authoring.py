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

# ---------------------------------------------------------------- template arms
#
# Beyond the stratified-sampler arm (sample_chunks -> draft_candidates), two
# corpus-shape-targeted template arms mine specific chunk structures and draft
# one question per mined seed:
#
#   * "definition"     — glossary-style chunks laid out as ``Term Definition.
#                        Term Definition.`` (the Week-N Summary glossary pages).
#                        Mines candidate key TERMS deterministically, batches
#                        them, and asks the local model to draft a definition
#                        question per term. question_type=factual_recall.
#   * "worked_example" — Problem/Solution/Step chunks. Mines the chunks whose
#                        text carries a worked example and asks the local model
#                        to draft a follow/explain question. question_type=
#                        procedural.
#
# Both reuse the deterministic candidate-assembly + pre-screen machinery; the
# model emits only question semantics + a verbatim quote (never the anchors,
# population, ids, or difficulty — those stay module-filled).
DEFINITION_TEMPLATE = "definition"
WORKED_EXAMPLE_TEMPLATE = "worked_example"
BOTH_POPULATION_TEMPLATE = "both_population"
DEFAULT_TEMPLATE = "stratified"

# ---------------------------------------------------------------- intent arms
#
# Per-learner_intent drafting arms (v1.2 learner_intent axis). Each arm mines a
# corpus-shape that maps to a real GUI learner intent, drafts ONE question per
# mined seed, and STAMPS the v1.2 ``learner_intent`` + ``expected_behavior``
# fields on the candidate (both schema-valid question fields) so the grounded
# eval's per-intent slice sees them. The arms reuse the deterministic
# candidate-assembly + pre-screen machinery; the model emits only question
# semantics + a verbatim quote (word_problem relaxes the quote pre-screen to
# numeric-literal containment — see the per-intent prescreen policy below).
#
#   * conceptual_why  — explanation chunks carrying because/reason cues; asks a
#                       "why does X hold" question. expected_behavior=answer_grounded.
#   * comparative     — TWO distinct-concept chunks paired; asks a contrast
#                       question whose answer cites both. answer_grounded.
#   * example_seeking — chunk_type/role=example chunks; asks for a worked/named
#                       example. answer_grounded.
#   * notation_symbol — chunks carrying math notation/symbols; asks what a
#                       symbol/notation means. answer_grounded.
#   * word_problem    — application chunks (numeric literals + a word-problem
#                       cue); asks a computational word problem.
#                       expected_behavior=answer_computational. The pre-screen
#                       is RELAXED to numeric-literal containment (a word
#                       problem's grounding is the shared numbers, not a
#                       >=40-char verbatim quote).
INTENT_CONCEPTUAL_WHY = "conceptual_why"
INTENT_COMPARATIVE = "comparative"
INTENT_EXAMPLE_SEEKING = "example_seeking"
INTENT_NOTATION_SYMBOL = "notation_symbol"
INTENT_WORD_PROBLEM = "word_problem"

# Per-intent arm policy: question_type, expected_behavior, and the pre-screen
# policy (``verbatim_quote`` = the default >=40-char verbatim containment;
# ``numeric_literal`` = the word-problem relaxation). ``comparative`` is a
# TWO-chunk arm handled separately (draft_comparative_candidates) and is not in
# this single-chunk table.
_INTENT_ARMS: Dict[str, Dict[str, str]] = {
    INTENT_CONCEPTUAL_WHY: {
        "question_type": "conceptual_synthesis",
        "expected_behavior": "answer_grounded",
        "prescreen_policy": "verbatim_quote",
    },
    INTENT_EXAMPLE_SEEKING: {
        "question_type": "factual_recall",
        "expected_behavior": "answer_grounded",
        "prescreen_policy": "verbatim_quote",
    },
    INTENT_NOTATION_SYMBOL: {
        "question_type": "factual_recall",
        "expected_behavior": "answer_grounded",
        "prescreen_policy": "verbatim_quote",
    },
    INTENT_WORD_PROBLEM: {
        "question_type": "procedural",
        "expected_behavior": "answer_computational",
        "prescreen_policy": "numeric_literal",
    },
}

# The learner_intents whose primary-passage pre-screen is numeric-literal
# relaxed rather than verbatim-quote. Derived from _INTENT_ARMS so the prescreen
# and the drafting stay in lockstep (one source of truth).
_NUMERIC_PRESCREEN_INTENTS = frozenset(
    intent for intent, arm in _INTENT_ARMS.items()
    if arm["prescreen_policy"] == "numeric_literal"
)

# conceptual_why: an explanation chunk that states a REASON — a because/since/
# therefore/so-that causal cue. Deliberately conservative; the model + quote
# pre-screen are the real gate.
_CAUSAL_CUE_RE = re.compile(
    r"\b(because|since|therefore|thus|hence|so that|in order to|"
    r"the reason|as a result|due to|which is why|this means)\b",
    re.IGNORECASE,
)
# example_seeking: an example chunk. STRUCTURAL signal first (chunk_type /
# teaching_role / content_type_label == example), then a textual "For example"/
# "e.g." fallback so a prose example without the structural label still mines.
_EXAMPLE_CHUNK_TYPES = frozenset({"example", "worked_example"})
_EXAMPLE_TEXT_RE = re.compile(
    r"\b(for example|for instance|e\.g\.|consider the|as an example)\b",
    re.IGNORECASE,
)
# notation_symbol: math notation / operator symbols. A chunk qualifies when it
# carries at least _NOTATION_MIN_SYMBOLS symbol occurrences (a single stray '='
# in prose is not a notation chunk).
_NOTATION_SYMBOL_RE = re.compile(
    r"[=≤≥≠±×÷√∑∏∫∞≈∝∈∉⊆⊂∪∩→←↔°πθλµ·^]|\\\(|\\\[|\\frac|\\sqrt|\\sum"
)
_NOTATION_MIN_SYMBOLS = 2
# word_problem: an application chunk — carries numeric literals AND a
# word-problem cue (how many / find / calculate / total / per / each …). Numbers
# alone (a page number, a year) do not make a word problem.
_NUMERIC_LITERAL_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_WORD_PROBLEM_CUE_RE = re.compile(
    r"\b(how many|how much|how long|find the|calculate|compute|"
    r"what is the total|altogether|each|per |average|how far|"
    r"how fast|at what|solve for)\b",
    re.IGNORECASE,
)
_WORD_PROBLEM_MIN_NUMERICS = 2

# A glossary chunk pairs a short Title-Case TERM with a sentence-y definition
# that follows it. We mine ``Term Definition`` boundaries: a Title-Case run of
# 1-4 words immediately followed by a capitalized definition sentence. The
# regex is deliberately conservative (it is a CANDIDATE miner — the local model
# + the verbatim-quote pre-screen are the real gate), but it matches loosely
# enough that it must only be applied to chunks that ALREADY look like a
# glossary (see :func:`_is_glossary_chunk`), else course prose floods it.
_GLOSSARY_TERM_RE = re.compile(
    r"(?<![A-Za-z])([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+"
    r"([A-Z][^.]{20,400}?\.)"
)
# Non-term Title-Case words that recur in course prose / headings and are NOT
# glossary terms — filtered so a heading like "Solution ..." or "Chapter ..."
# never mints a spurious definition candidate.
_GLOSSARY_TERM_STOPWORDS = frozenset(
    {
        "the", "a", "an", "this", "that", "these", "those", "solution",
        "problem", "step", "steps", "example", "answer", "show", "chapter",
        "section", "lesson", "module", "week", "objective", "objectives",
        "summary", "core", "elementary", "introduction", "overview", "note",
        "notes", "remember", "understand", "apply", "learning", "practice",
        "term", "definition", "key", "reference", "page", "title",
    }
)
# A chunk qualifies as a glossary only when it yields at least this many
# distinct term—definition pairs — a single stray "Term Sentence." in prose
# does not make a glossary.
_GLOSSARY_MIN_PAIRS = 4
# Glossary terms are short noun phrases; reject a "term" longer than this many
# words (a long Title-Case run is a heading / sentence fragment, not a term).
_GLOSSARY_TERM_MAX_WORDS = 4
# Structural glossary signal: the genuine term—definition pages are the
# Week-N Summary chunks (chunk_type "summary"; module/section "summary" /
# "glossary"). Gating on this STRUCTURAL marker — not just text density — keeps
# worked-example / prose chunks (which also pack Title-Case phrases) out.
_GLOSSARY_CHUNK_TYPES = frozenset({"summary", "glossary"})
_GLOSSARY_HEADING_MARKERS = ("summary", "glossary")
# A worked-example chunk announces a Problem and a Solution / enumerated-Step
# structure. The patterns are deliberately specific (not a bare word match):
#   * a "Problem" or "Problem N" LABEL (Title-case, line-label style);
#   * a "Solution" LABEL;
#   * an enumerated "Step N" (rules out the "one-step" / "multi-step" adjectives
#     that pepper objective-listing prose).
# A chunk qualifies when it has a Problem label AND (a Solution label OR an
# enumerated Step). Whole-word adjectives like "one-step" no longer trigger it.
_PROBLEM_RE = re.compile(r"\bProblem\b")
_SOLUTION_RE = re.compile(r"\bSolution\b")
_STEP_RE = re.compile(r"\bStep\s+\d+\b")

# How many glossary terms to put in front of the model per drafting call. One
# question is drafted per term, but terms are batched so a single chunk's
# glossary yields a focused, low-latency call per chunk.
_DEFINITION_TERMS_PER_CHUNK = 8
# Minimum definition length so a term whose "definition" is a stray fragment is
# skipped before it ever reaches the model.
_MIN_DEFINITION_CHARS = 25


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
            "template": DEFAULT_TEMPLATE,
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


# ---------------------------------------------------------------- definition arm


@dataclass(frozen=True)
class GlossarySeed:
    """One mined glossary ``(term, definition)`` pair from a glossary chunk."""

    chunk_id: str
    term: str
    definition: str


def _glossary_pairs(text: str) -> List[Tuple[str, str]]:
    """Extract de-duplicated ``(term, definition)`` pairs from one chunk's text.

    Filters stray prose: drops terms that are stopwords / heading words, terms
    longer than :data:`_GLOSSARY_TERM_MAX_WORDS` words, and definitions shorter
    than :data:`_MIN_DEFINITION_CHARS`. Deterministic (source order, first win
    per term)."""
    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    for m in _GLOSSARY_TERM_RE.finditer(text):
        term = m.group(1).strip()
        definition = m.group(2).strip()
        if len(definition) < _MIN_DEFINITION_CHARS:
            continue
        words = term.split()
        if len(words) > _GLOSSARY_TERM_MAX_WORDS:
            continue
        # Reject when the FIRST word of the term is a stopword / heading word
        # (e.g. "Solution ...", "Chapter ...") — a real glossary term leads
        # with the term itself.
        if words[0].lower() in _GLOSSARY_TERM_STOPWORDS:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        pairs.append((term, definition))
    return pairs


def _is_glossary_chunk(chunk: Dict[str, Any]) -> bool:
    """A chunk is a glossary when it carries the STRUCTURAL summary/glossary
    marker (chunk_type or module/section heading) AND packs >= the minimum
    term—definition pairs.

    The structural gate is the load-bearing one: a worked-example or prose chunk
    that happens to pack Title-Case phrases is rejected because it is not a
    summary/glossary page. The text-density floor is the secondary guard."""
    ctype = str(chunk.get("chunk_type") or "").lower()
    structural = ctype in _GLOSSARY_CHUNK_TYPES
    if not structural:
        source = chunk.get("source") or {}
        hay = " ".join(
            str(source.get(k) or "")
            for k in ("module_id", "module_title", "section_heading")
        ).lower()
        structural = any(m in hay for m in _GLOSSARY_HEADING_MARKERS)
    if not structural:
        return False
    return len(_glossary_pairs(str(chunk.get("text") or ""))) >= _GLOSSARY_MIN_PAIRS


def mine_glossary_terms(
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    per_chunk_cap: int = _DEFINITION_TERMS_PER_CHUNK,
) -> List[GlossarySeed]:
    """Deterministically mine ``(term, definition)`` seeds from glossary chunks.

    A glossary chunk lays out term—definition pairs as ``Term Definition.
    Term Definition.`` (e.g. the Week-N Summary pages). We walk every chunk in
    id order, and — only for chunks that read as a glossary
    (:func:`_is_glossary_chunk`: the STRUCTURAL summary/glossary marker plus a
    term-definition density floor) — extract the filtered Title-Case-term /
    capitalized-definition boundaries as seeds. Pure + deterministic.

    The glossary gate keeps worked-example / course prose from minting spurious
    "definition" candidates; the per-chunk cap keeps one dense glossary from
    dominating. A non-glossary chunk contributes nothing.
    """
    seeds: List[GlossarySeed] = []
    for cid in sorted(chunks_by_id):
        chunk = chunks_by_id[cid]
        if not _is_glossary_chunk(chunk):
            continue
        pairs = _glossary_pairs(str(chunk.get("text") or ""))
        for term, definition in pairs[:per_chunk_cap]:
            seeds.append(GlossarySeed(chunk_id=cid, term=term, definition=definition))
    return seeds


def _definition_prompt(term: str, chunk_text: str) -> Tuple[str, str]:
    """``(system, user)`` prompts asking the model to draft one definition
    question for ``term``, grounded in ``chunk_text``."""
    system = (
        "You author one definition question for a key term, grounded in a "
        "single source chunk. Output JSON only with keys: question_text (a "
        "clear question asking the learner to define or explain the term, at "
        "least 10 characters), expected_key_points (a JSON array of 2 to 4 "
        "short factual points the correct definition must state), and quote (a "
        "VERBATIM substring copied from the source chunk text, at least 40 "
        "characters, that states the term's definition). Copy the quote "
        "character for character from the chunk — do not paraphrase it. Do not "
        "add facts not present in the chunk."
    )
    user = (
        f"Key term to ask about: {term}\n\n"
        f"Source chunk text (a glossary):\n{chunk_text}\n\n"
        'Output JSON only: {"question_text": "...", '
        '"expected_key_points": ["...", "..."], "quote": "<verbatim substring>"}'
    )
    return system, user


def draft_definition_candidates(
    seeds: Sequence[GlossarySeed],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Draft one ``definition`` candidate per mined glossary term.

    Mirrors :func:`draft_candidates`: the model emits only question semantics +
    a verbatim quote; the deterministic fields (anchors, population, ids,
    difficulty) are module-filled. question_type=factual_recall,
    authoring.template=definition. One ``gold_candidate_authoring`` decision is
    emitted after the batch with a dynamic rationale (terms, model, failures).
    NEVER an Anthropic surface (``client`` is the loopback-local provider).
    """
    resolved_model = model_id or getattr(client, "model", "local")
    candidates: List[Dict[str, Any]] = []
    errors = 0
    for seed in seeds:
        chunk = chunks_by_id.get(seed.chunk_id) or {}
        system, user = _definition_prompt(seed.term, str(chunk.get("text") or ""))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = client.chat_completion(messages, max_tokens=512, temperature=0.2)
            draft = _parse_draft(text)
            cand = _build_template_candidate(
                seed.chunk_id, chunk, draft, model_id=resolved_model,
                question_type="factual_recall", template=DEFINITION_TEMPLATE,
            )
        except (GoldDraftError, Exception) as exc:  # noqa: BLE001 — record, don't drop
            errors += 1
            cand = _build_error_candidate(
                seed.chunk_id, chunk, resolved_model,
                question_type="factual_recall", template=DEFINITION_TEMPLATE,
            )
            cand["draft_error"] = str(exc)
        candidates.append(cand)

    _log_template_authoring(
        capture, template=DEFINITION_TEMPLATE, model=resolved_model,
        seeds=[s.chunk_id for s in seeds], n=len(candidates), errors=errors,
        detail=f"mined {len(seeds)} glossary term(s): "
               + ", ".join(s.term for s in seeds[:8]),
    )
    return candidates


# ---------------------------------------------------------------- worked-example arm


def mine_worked_example_chunks(
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Deterministically mine the chunk ids carrying a worked example.

    A worked-example chunk announces a ``Problem`` AND either a ``Solution`` or
    a ``Step`` structure. Returns the matching chunk ids in id order. Pure +
    deterministic — the local model + verbatim-quote pre-screen are the gate on
    whether the mined chunk actually yields a usable procedural question.
    """
    out: List[str] = []
    for cid in sorted(chunks_by_id):
        text = str(chunks_by_id[cid].get("text") or "")
        if _PROBLEM_RE.search(text) and (
            _SOLUTION_RE.search(text) or _STEP_RE.search(text)
        ):
            out.append(cid)
    return out


def _worked_example_prompt(chunk_text: str) -> Tuple[str, str]:
    """``(system, user)`` prompts asking the model to draft one procedural
    question about following / explaining a worked example."""
    system = (
        "You author one question that asks the learner to follow or explain a "
        "worked example present in a single source chunk. Output JSON only with "
        "keys: question_text (a clear question asking the learner to work "
        "through, follow, or explain the steps of the example, at least 10 "
        "characters), expected_key_points (a JSON array of 2 to 4 short points "
        "naming the ordered steps the correct answer must state), and quote (a "
        "VERBATIM substring copied from the source chunk text, at least 40 "
        "characters, drawn from the worked example). Copy the quote character "
        "for character from the chunk — do not paraphrase it. Do not add facts "
        "not present in the chunk."
    )
    user = (
        "Source chunk text (a worked example with a Problem and a "
        f"Solution/Step structure):\n{chunk_text}\n\n"
        'Output JSON only: {"question_text": "...", '
        '"expected_key_points": ["...", "..."], "quote": "<verbatim substring>"}'
    )
    return system, user


def draft_worked_example_candidates(
    chunk_ids: Sequence[str],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Draft one ``worked_example`` candidate per mined worked-example chunk.

    Mirrors :func:`draft_candidates`: the model emits only question semantics +
    a verbatim quote; deterministic fields are module-filled.
    question_type=procedural, authoring.template=worked_example. One
    ``gold_candidate_authoring`` decision is emitted after the batch with a
    dynamic rationale. NEVER an Anthropic surface.
    """
    resolved_model = model_id or getattr(client, "model", "local")
    candidates: List[Dict[str, Any]] = []
    errors = 0
    for cid in chunk_ids:
        chunk = chunks_by_id.get(cid) or {}
        system, user = _worked_example_prompt(str(chunk.get("text") or ""))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = client.chat_completion(messages, max_tokens=512, temperature=0.2)
            draft = _parse_draft(text)
            cand = _build_template_candidate(
                cid, chunk, draft, model_id=resolved_model,
                question_type="procedural", template=WORKED_EXAMPLE_TEMPLATE,
            )
        except (GoldDraftError, Exception) as exc:  # noqa: BLE001 — record, don't drop
            errors += 1
            cand = _build_error_candidate(
                cid, chunk, resolved_model,
                question_type="procedural", template=WORKED_EXAMPLE_TEMPLATE,
            )
            cand["draft_error"] = str(exc)
        candidates.append(cand)

    _log_template_authoring(
        capture, template=WORKED_EXAMPLE_TEMPLATE, model=resolved_model,
        seeds=list(chunk_ids), n=len(candidates), errors=errors,
        detail=f"mined {len(chunk_ids)} Problem/Solution/Step chunk(s)",
    )
    return candidates


# ---------------------------------------------------------------- both-population arm
#
# A union corpus mixes a generated course-population (``week_NN/`` pages,
# ``course_overview/``) with an original-document source-population (flat
# ``*.html`` textbook chunks). The §1.3 coverage report wants >= 5 ``both``
# questions — a synthesis whose COMPLETE answer requires citing one chunk of
# EACH population (e.g. relating the course page's treatment of a concept to
# the textbook's worked treatment of the same concept). This arm:
#
#   1. Pairs a course chunk with a source chunk on the SAME concept,
#      deterministically, via shared significant-token vocabulary
#      (:func:`mine_both_population_pairs`). concept_tags are empty on the
#      the union corpus, so the pairing leans on section-heading +
#      body-text content tokens (a small stopword filter keeps the signal on
#      domain vocabulary), and the highest-overlap source partner per course
#      chunk is chosen (ties broken by chunk id for determinism).
#   2. Asks the local model to draft ONE synthesis question whose answer needs
#      both chunks, plus a verbatim quote from EACH (so each passage anchors).
#
# The candidate carries BOTH chunks in ``relevant_passages`` (course=primary,
# source=supporting), ``expected_citation_population="both"``, and difficulty
# ``medium`` where honest (a 2-chunk, same-week join) — escalating to ``hard``
# only when the two chunks span >= 2 distinct content weeks. The model emits
# only question semantics + the two quotes; every deterministic field is
# module-filled. NEVER an Anthropic surface.

# Pairing parameters.
_BOTH_MIN_SHARED_TOKENS = 3       # require >= this many shared content tokens
_BOTH_MIN_JACCARD = 0.06          # and >= this token-Jaccard to pair
_BOTH_TERM_MIN_CHARS = 4          # ignore short tokens (the, and, for, ...)
# Lightweight stopword set so the overlap signal stays on domain vocabulary
# rather than function words. Deliberately small — the char floor does most of
# the filtering; this just drops the common >=4-char function words.
_BOTH_STOPWORDS = frozenset(
    {
        "this", "that", "these", "those", "there", "their", "them", "then",
        "than", "with", "which", "while", "when", "where", "what", "into",
        "from", "have", "here", "your", "also", "such", "they", "will",
        "would", "could", "should", "about", "above", "below", "between",
        "each", "other", "some", "more", "most", "must", "both", "been",
        "were", "does", "doing", "done", "using", "used", "uses", "over",
        "under", "page", "part", "section", "chapter", "lesson", "module",
        "week", "title", "summary", "preface", "example", "examples",
        "problem", "solution", "step", "steps", "learning", "objective",
        "objectives", "overview", "introduction", "figure", "table",
    }
)


@dataclass(frozen=True)
class BothPopulationPair:
    """A mined ``(course_chunk, source_chunk)`` pair on a shared concept.

    ``shared_terms`` are the content tokens both chunks carry (the pairing
    evidence, surfaced for the operator); ``jaccard`` is the token-Jaccard
    between the two chunks' content-token sets.
    """

    course_chunk_id: str
    source_chunk_id: str
    shared_terms: Tuple[str, ...]
    jaccard: float


def _content_tokens(chunk: Dict[str, Any]) -> set:
    """The set of significant content tokens for pairing.

    Union of the chunk's ``concept_tags`` (split on non-alphanumerics) and the
    significant tokens of its ``section_heading`` + body ``text``: lowercased
    alphabetic tokens >= :data:`_BOTH_TERM_MIN_CHARS` chars that are not in
    :data:`_BOTH_STOPWORDS`. Deterministic (set membership only)."""
    source = chunk.get("source") or {}
    parts: List[str] = []
    for tag in chunk.get("concept_tags") or []:
        if isinstance(tag, str):
            parts.append(tag)
    heading = source.get("section_heading")
    if isinstance(heading, str):
        parts.append(heading)
    parts.append(str(chunk.get("text") or ""))
    blob = " ".join(parts).lower()
    toks = re.findall(r"[a-z]+", blob)
    return {
        t for t in toks
        if len(t) >= _BOTH_TERM_MIN_CHARS and t not in _BOTH_STOPWORDS
    }


def mine_both_population_pairs(
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    max_pairs: int = 25,
) -> List[BothPopulationPair]:
    """Deterministically pair course-population chunks with source-population
    chunks on a shared concept.

    For each course chunk (id order) the best source partner is the source
    chunk with the highest content-token Jaccard (>= :data:`_BOTH_MIN_JACCARD`
    AND >= :data:`_BOTH_MIN_SHARED_TOKENS` shared tokens), ties broken by
    source chunk id. A source chunk is used at most once across pairs (so the
    arm spreads concepts rather than re-pairing one dense textbook chunk).
    Returns up to ``max_pairs`` pairs in course-chunk-id order. Pure +
    deterministic — the local model + per-passage verbatim-quote pre-screen are
    the real gate on whether the pair yields a usable synthesis question.
    """
    course_ids: List[str] = []
    source_ids: List[str] = []
    for cid in sorted(chunks_by_id):
        pop = _population_of_chunk(chunks_by_id[cid])
        if pop == "course":
            course_ids.append(cid)
        elif pop == "source":
            source_ids.append(cid)

    source_tokens = {sid: _content_tokens(chunks_by_id[sid]) for sid in source_ids}
    used_sources: set = set()
    pairs: List[BothPopulationPair] = []
    for cid in course_ids:
        if len(pairs) >= max_pairs:
            break
        ctoks = _content_tokens(chunks_by_id[cid])
        if not ctoks:
            continue
        best: Optional[Tuple[float, int, str, Tuple[str, ...]]] = None
        for sid in source_ids:
            if sid in used_sources:
                continue
            stoks = source_tokens[sid]
            if not stoks:
                continue
            shared = ctoks & stoks
            if len(shared) < _BOTH_MIN_SHARED_TOKENS:
                continue
            union = len(ctoks | stoks) or 1
            jac = len(shared) / union
            if jac < _BOTH_MIN_JACCARD:
                continue
            # Maximize (jaccard, shared_count); tie-break by source id ASC
            # (negate the id-string comparison by tracking it for a stable sort).
            cand_key = (jac, len(shared))
            if best is None or cand_key > (best[0], best[1]) or (
                cand_key == (best[0], best[1]) and sid < best[2]
            ):
                best = (jac, len(shared), sid, tuple(sorted(shared)))
        if best is not None:
            used_sources.add(best[2])
            pairs.append(
                BothPopulationPair(
                    course_chunk_id=cid,
                    source_chunk_id=best[2],
                    shared_terms=best[3][:12],
                    jaccard=round(best[0], 4),
                )
            )
    return pairs


def _both_population_prompt(
    course_text: str, source_text: str, shared_terms: Sequence[str]
) -> Tuple[str, str]:
    """``(system, user)`` prompts asking the model to draft ONE synthesis
    question whose complete answer requires BOTH chunks, with a verbatim quote
    from each."""
    system = (
        "You author one synthesis question whose COMPLETE answer requires TWO "
        "source chunks: a course page and the original textbook, both about the "
        "same concept. The question must NOT be answerable from either chunk "
        "alone — it must relate the course page's treatment to the textbook's "
        "treatment of the same concept. Output JSON only with keys: "
        "question_text (a clear question of at least 10 characters that needs "
        "both chunks), expected_key_points (a JSON array of 2 to 4 short "
        "factual points the correct answer must state), course_quote (a "
        "VERBATIM substring of at least 40 characters copied from the COURSE "
        "chunk), and source_quote (a VERBATIM substring of at least 40 "
        "characters copied from the TEXTBOOK chunk). Copy each quote character "
        "for character from its chunk — do not paraphrase. Do not add facts not "
        "present in the chunks."
    )
    terms = ", ".join(shared_terms[:10])
    user = (
        f"Shared concept vocabulary across the two chunks: {terms}\n\n"
        f"COURSE chunk text:\n{course_text}\n\n"
        f"TEXTBOOK (source) chunk text:\n{source_text}\n\n"
        'Output JSON only: {"question_text": "...", '
        '"expected_key_points": ["...", "..."], '
        '"course_quote": "<verbatim from course chunk>", '
        '"source_quote": "<verbatim from textbook chunk>"}'
    )
    return system, user


def _parse_both_draft(text: str) -> Dict[str, Any]:
    """Parse a both-population draft into ``{question_text,
    expected_key_points, course_quote, source_quote}``. Reuses the lenient
    JSON-object recovery of :func:`_parse_draft`."""
    raw = (text or "").strip()
    parsed: Optional[Dict[str, Any]] = None
    try:
        cand = json.loads(raw)
        if isinstance(cand, dict):
            parsed = cand
    except json.JSONDecodeError:
        parsed = None
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
        raise GoldDraftError(f"no JSON object recoverable from both-draft tail {raw[-120:]!r}")
    qt = parsed.get("question_text")
    course_quote = parsed.get("course_quote")
    source_quote = parsed.get("source_quote")
    if not isinstance(qt, str):
        raise GoldDraftError("both-draft missing question_text")
    if not isinstance(course_quote, str):
        raise GoldDraftError("both-draft missing course_quote")
    if not isinstance(source_quote, str):
        raise GoldDraftError("both-draft missing source_quote")
    kps = parsed.get("expected_key_points")
    if not isinstance(kps, list):
        kps = []
    kps = [str(k).strip() for k in kps if isinstance(k, (str, int, float)) and str(k).strip()]
    return {
        "question_text": qt.strip(),
        "expected_key_points": kps[:_KEY_POINTS_MAX],
        "course_quote": course_quote.strip(),
        "source_quote": source_quote.strip(),
    }


def _anchor_for(chunk: Dict[str, Any], quote: str) -> Dict[str, Any]:
    """Build the v1.1 passage anchor for ``chunk`` carrying ``quote``."""
    source = chunk.get("source") or {}
    anchor: Dict[str, Any] = {
        "item_path": str(source.get("item_path") or ""),
        "text_quote": quote,
        "content_sha256": chunk_content_sha256(chunk),
    }
    heading = source.get("section_heading")
    if isinstance(heading, str) and heading:
        anchor["section_heading"] = heading
    return anchor


def _build_both_candidate(
    pair: BothPopulationPair,
    course_chunk: Dict[str, Any],
    source_chunk: Dict[str, Any],
    draft: Dict[str, Any],
    *,
    model_id: str,
) -> Dict[str, Any]:
    """Assemble a v1.1-shaped ``both_population`` candidate with TWO passages
    (course=primary, source=supporting), ``expected_citation_population=both``,
    honest difficulty + synthesis_scope."""
    course_week = _week_of_item_path((course_chunk.get("source") or {}).get("item_path", ""))
    source_week = _week_of_item_path((source_chunk.get("source") or {}).get("item_path", ""))
    weeks = {w for w in (course_week, source_week) if w}
    n_weeks = len(weeks)
    # A 2-chunk synthesis: medium where same-week (or the source has no week,
    # which is the union norm — textbook chunks are weekless), hard when the
    # two cited chunks genuinely span >= 2 distinct content weeks.
    difficulty = "hard" if n_weeks >= 2 else "medium"
    candidate: Dict[str, Any] = {
        "question_id": f"gqc-both-{pair.course_chunk_id}-{pair.source_chunk_id}",
        "question_text": draft["question_text"],
        "question_type": "conceptual_synthesis",
        "difficulty": difficulty,
        "synthesis_scope": "across_weeks" if n_weeks >= 2 else "within_week",
        "expected_citation_population": "both",
        "relevant_passages": [
            {"chunk_id": pair.course_chunk_id, "relevance": "primary",
             "anchor": _anchor_for(course_chunk, draft["course_quote"])},
            {"chunk_id": pair.source_chunk_id, "relevance": "supporting",
             "anchor": _anchor_for(source_chunk, draft["source_quote"])},
        ],
        "authoring": {
            "method": "llm_assisted",
            "author": f"{model_id}/{GOLD_AUTHORING_PROMPT_VERSION}",
            "reviewed_by": "PENDING_REVIEW",
            "status": "draft",
            "template": BOTH_POPULATION_TEMPLATE,
        },
    }
    refs = tuple(sorted(set(_objective_refs(course_chunk)) | set(_objective_refs(source_chunk))))
    if refs:
        candidate["objective_refs"] = list(refs)
    kps = draft.get("expected_key_points") or []
    if len(kps) >= _KEY_POINTS_MIN:
        candidate["expected_key_points"] = kps[:_KEY_POINTS_MAX]
    return candidate


def draft_both_population_candidates(
    pairs: Sequence[BothPopulationPair],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Draft one ``both_population`` candidate per mined course/source pair.

    The model emits only question semantics + a verbatim quote per chunk; the
    deterministic fields (anchors, two passages, population=both, difficulty,
    synthesis_scope, ids) are module-filled. question_type=conceptual_synthesis,
    authoring.template=both_population. One ``gold_candidate_authoring``
    decision is emitted after the batch with a dynamic rationale. NEVER an
    Anthropic surface.
    """
    resolved_model = model_id or getattr(client, "model", "local")
    candidates: List[Dict[str, Any]] = []
    errors = 0
    for pair in pairs:
        course_chunk = chunks_by_id.get(pair.course_chunk_id) or {}
        source_chunk = chunks_by_id.get(pair.source_chunk_id) or {}
        system, user = _both_population_prompt(
            str(course_chunk.get("text") or ""),
            str(source_chunk.get("text") or ""),
            pair.shared_terms,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = client.chat_completion(messages, max_tokens=640, temperature=0.2)
            draft = _parse_both_draft(text)
            cand = _build_both_candidate(
                pair, course_chunk, source_chunk, draft, model_id=resolved_model
            )
        except (GoldDraftError, Exception) as exc:  # noqa: BLE001 — record, don't drop
            errors += 1
            cand = {
                "question_id": f"gqc-both-{pair.course_chunk_id}-{pair.source_chunk_id}",
                "question_text": "",
                "question_type": "conceptual_synthesis",
                "relevant_passages": [
                    {"chunk_id": pair.course_chunk_id, "relevance": "primary",
                     "anchor": {"item_path": str((course_chunk.get("source") or {}).get("item_path") or ""),
                                "text_quote": ""}},
                    {"chunk_id": pair.source_chunk_id, "relevance": "supporting",
                     "anchor": {"item_path": str((source_chunk.get("source") or {}).get("item_path") or ""),
                                "text_quote": ""}},
                ],
                "authoring": {
                    "method": "llm_assisted",
                    "author": f"{resolved_model}/{GOLD_AUTHORING_PROMPT_VERSION}",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "draft",
                    "template": BOTH_POPULATION_TEMPLATE,
                },
                "draft_error": str(exc),
            }
        candidates.append(cand)

    pair_labels = [f"{p.course_chunk_id}+{p.source_chunk_id}" for p in pairs]
    _log_template_authoring(
        capture, template=BOTH_POPULATION_TEMPLATE, model=resolved_model,
        seeds=pair_labels, n=len(candidates), errors=errors,
        detail=f"paired {len(pairs)} course/source chunk(s) on shared concept "
               f"vocabulary (each answer must cite BOTH populations)",
    )
    return candidates


# ---------------------------------------------------------------- arm helpers


def _build_template_candidate(
    chunk_id: str,
    chunk: Dict[str, Any],
    draft: Dict[str, Any],
    *,
    model_id: str,
    question_type: str,
    template: str,
) -> Dict[str, Any]:
    """Assemble a v1.1-shaped candidate for a template arm (definition /
    worked_example). Reuses the deterministic anchor + population + difficulty
    fill of :func:`_build_candidate`, then stamps ``authoring.template``.

    The seed chunk is always the single primary passage (these arms draft one
    question per seed chunk), so the SampleSlot is synthesized from the chunk's
    own week / population / objective_refs — same provenance the sampler arm
    derives, just without the round-robin question_type assignment.
    """
    key = _stratum_key(chunk)
    slot = SampleSlot(
        chunk_id=chunk_id,
        question_type=question_type,
        week=key[0],
        population=key[2],
        teaching_role=key[3],
        objective_refs=_objective_refs(chunk),
    )
    candidate = _build_candidate(slot, chunk, draft, model_id=model_id)
    candidate["authoring"]["template"] = template
    return candidate


def _build_error_candidate(
    chunk_id: str,
    chunk: Dict[str, Any],
    model_id: str,
    *,
    question_type: str,
    template: str,
) -> Dict[str, Any]:
    """A draft-failure candidate (recorded, not dropped) for a template arm."""
    return {
        "question_id": f"gqc-{chunk_id}",
        "question_text": "",
        "question_type": question_type,
        "relevant_passages": [
            {"chunk_id": chunk_id, "relevance": "primary",
             "anchor": {"item_path": str((chunk.get("source") or {}).get("item_path") or ""),
                        "text_quote": ""}}
        ],
        "authoring": {
            "method": "llm_assisted",
            "author": f"{model_id}/{GOLD_AUTHORING_PROMPT_VERSION}",
            "reviewed_by": "PENDING_REVIEW",
            "status": "draft",
            "template": template,
        },
    }


def _log_template_authoring(
    capture: Optional[Any],
    *,
    template: str,
    model: str,
    seeds: Sequence[str],
    n: int,
    errors: int,
    detail: str,
) -> None:
    """Emit one ``gold_candidate_authoring`` decision for a template-arm batch
    with a dynamic rationale (template, model, seed ids, accept/reject split)."""
    if capture is None:
        return
    seed_ids = ",".join(seeds[:8])
    more = "" if len(seeds) <= 8 else f" (+{len(seeds) - 8} more)"
    parsed_ok = n - errors
    try:
        capture.log_decision(
            decision_type="gold_candidate_authoring",
            decision=(
                f"drafted {n} '{template}' gold candidate(s) via local model "
                f"{model}; {errors} parse-failure(s)."
            ),
            rationale=(
                f"Template-arm '{template}' gold-candidate drafting over "
                f"{len(seeds)} deterministically-mined seed(s) "
                f"[{seed_ids}{more}] using license-clean local model {model} "
                f"(prompt {GOLD_AUTHORING_PROMPT_VERSION}); {detail}; "
                f"parsed_ok={parsed_ok}, parse_failures={errors}. Quotes are "
                f"pre-screened deterministically for verbatim containment + "
                f"ambiguity downstream before any promotion."
            ),
        )
    except Exception:  # pragma: no cover — defensive
        pass


# ---------------------------------------------------------------- intent miners


def mine_conceptual_why_chunks(
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Chunk ids of explanation chunks that state a REASON (a because/since/
    therefore causal cue). Id order, deterministic. Glossary + worked-example
    chunks are skipped (their arms own them)."""
    out: List[str] = []
    for cid in sorted(chunks_by_id):
        chunk = chunks_by_id[cid]
        if _is_glossary_chunk(chunk):
            continue
        text = str(chunk.get("text") or "")
        if _PROBLEM_RE.search(text) and (
            _SOLUTION_RE.search(text) or _STEP_RE.search(text)
        ):
            continue  # worked example — the worked_example arm owns it
        if _CAUSAL_CUE_RE.search(text):
            out.append(cid)
    return out


def mine_example_chunks(
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Chunk ids that read as an EXAMPLE — the structural chunk_type /
    teaching_role / content_type_label == example marker, or a textual
    "for example"/"e.g." cue as a fallback. Id order, deterministic."""
    out: List[str] = []
    for cid in sorted(chunks_by_id):
        chunk = chunks_by_id[cid]
        ctype = str(chunk.get("chunk_type") or "").lower()
        role = str(chunk.get("teaching_role") or "").lower()
        ctl = str(chunk.get("content_type_label") or "").lower()
        structural = (
            ctype in _EXAMPLE_CHUNK_TYPES
            or role in _EXAMPLE_CHUNK_TYPES
            or ctl in _EXAMPLE_CHUNK_TYPES
        )
        if structural or _EXAMPLE_TEXT_RE.search(str(chunk.get("text") or "")):
            out.append(cid)
    return out


def mine_notation_chunks(
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Chunk ids carrying math notation / operator symbols (>=
    :data:`_NOTATION_MIN_SYMBOLS` symbol occurrences). Id order, deterministic."""
    out: List[str] = []
    for cid in sorted(chunks_by_id):
        text = str(chunks_by_id[cid].get("text") or "")
        if len(_NOTATION_SYMBOL_RE.findall(text)) >= _NOTATION_MIN_SYMBOLS:
            out.append(cid)
    return out


def mine_word_problem_chunks(
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Chunk ids of APPLICATION chunks — >= :data:`_WORD_PROBLEM_MIN_NUMERICS`
    numeric literals AND a word-problem cue (how many / find / per / …). Numbers
    alone (a year, a page number) do not qualify. Id order, deterministic."""
    out: List[str] = []
    for cid in sorted(chunks_by_id):
        text = str(chunks_by_id[cid].get("text") or "")
        if (
            len(_NUMERIC_LITERAL_RE.findall(text)) >= _WORD_PROBLEM_MIN_NUMERICS
            and _WORD_PROBLEM_CUE_RE.search(text)
        ):
            out.append(cid)
    return out


_INTENT_MINERS = {
    INTENT_CONCEPTUAL_WHY: mine_conceptual_why_chunks,
    INTENT_EXAMPLE_SEEKING: mine_example_chunks,
    INTENT_NOTATION_SYMBOL: mine_notation_chunks,
    INTENT_WORD_PROBLEM: mine_word_problem_chunks,
}


def mine_intent_chunks(
    intent: str, chunks_by_id: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Dispatch to the single-chunk miner for a learner ``intent``."""
    miner = _INTENT_MINERS.get(intent)
    if miner is None:
        raise ValueError(f"no single-chunk miner for intent {intent!r}")
    return miner(chunks_by_id)


# ---------------------------------------------------------------- intent drafting


def _intent_prompt(intent: str, chunk: Dict[str, Any]) -> Tuple[str, str]:
    """``(system, user)`` prompts for a single-chunk intent arm. The model emits
    only question semantics + a verbatim quote; deterministic fields are
    module-filled. word_problem asks the quote to carry the numeric values."""
    hint, quote_hint = {
        INTENT_CONCEPTUAL_WHY: (
            "Ask a 'why' / 'what is the reason' question whose answer explains "
            "the cause or justification stated in the chunk.",
            "that states the reason",
        ),
        INTENT_EXAMPLE_SEEKING: (
            "Ask the learner to give or identify a concrete example of the "
            "concept the chunk illustrates.",
            "that contains the example",
        ),
        INTENT_NOTATION_SYMBOL: (
            "Ask what a symbol or piece of notation in the chunk means or how "
            "it is read.",
            "that contains the symbol / notation",
        ),
        INTENT_WORD_PROBLEM: (
            "Ask a computational word problem answerable with the numbers in "
            "the chunk; the answer is a computed value.",
            "that contains the relevant numbers",
        ),
    }.get(intent, ("Ask a clear question answerable from the chunk.", ""))
    system = (
        "You author one evaluation question of a specific kind, grounded in a "
        "single source chunk. Output JSON only with keys: question_text (a "
        "clear question of at least 10 characters), expected_key_points (a JSON "
        "array of 2 to 4 short factual points the correct answer must state), "
        "and quote (a VERBATIM substring copied from the source chunk text "
        f"{quote_hint}). Copy the quote character for character from the chunk "
        "— do not paraphrase it. Do not add facts not present in the chunk."
    )
    chunk_text = str(chunk.get("text") or "")
    user = (
        f"Question kind: {intent}. {hint}\n\n"
        f"Source chunk text:\n{chunk_text}\n\n"
        'Output JSON only: {"question_text": "...", '
        '"expected_key_points": ["...", "..."], "quote": "<verbatim substring>"}'
    )
    return system, user


def _build_intent_candidate(
    intent: str,
    chunk_id: str,
    chunk: Dict[str, Any],
    draft: Dict[str, Any],
    *,
    model_id: str,
) -> Dict[str, Any]:
    """Assemble a v1.2-shaped single-chunk intent candidate: reuse the template
    candidate machinery, then stamp ``learner_intent`` + ``expected_behavior``
    (both schema-valid v1.2 question fields) and ``authoring.template=<intent>``."""
    arm = _INTENT_ARMS[intent]
    candidate = _build_template_candidate(
        chunk_id, chunk, draft, model_id=model_id,
        question_type=arm["question_type"], template=intent,
    )
    candidate["learner_intent"] = intent
    candidate["expected_behavior"] = arm["expected_behavior"]
    return candidate


def draft_intent_candidates(
    intent: str,
    chunk_ids: Sequence[str],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Draft one candidate per mined seed chunk for a single-chunk learner
    ``intent`` arm (conceptual_why / example_seeking / notation_symbol /
    word_problem). Stamps ``learner_intent`` + ``expected_behavior``. One
    ``gold_candidate_authoring`` decision is emitted per batch with a dynamic
    rationale naming the intent arm. NEVER an Anthropic surface."""
    if intent not in _INTENT_ARMS:
        raise ValueError(f"{intent!r} is not a single-chunk intent arm")
    resolved_model = model_id or getattr(client, "model", "local")
    candidates: List[Dict[str, Any]] = []
    errors = 0
    for cid in chunk_ids:
        chunk = chunks_by_id.get(cid) or {}
        system, user = _intent_prompt(intent, chunk)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = client.chat_completion(messages, max_tokens=512, temperature=0.2)
            draft = _parse_draft(text)
            cand = _build_intent_candidate(
                intent, cid, chunk, draft, model_id=resolved_model
            )
        except (GoldDraftError, Exception) as exc:  # noqa: BLE001 — record, don't drop
            errors += 1
            cand = _build_error_candidate(
                cid, chunk, resolved_model,
                question_type=_INTENT_ARMS[intent]["question_type"], template=intent,
            )
            cand["learner_intent"] = intent
            cand["expected_behavior"] = _INTENT_ARMS[intent]["expected_behavior"]
            cand["draft_error"] = str(exc)
        candidates.append(cand)

    _log_template_authoring(
        capture, template=intent, model=resolved_model,
        seeds=list(chunk_ids), n=len(candidates), errors=errors,
        detail=f"learner_intent '{intent}' arm "
               f"(expected_behavior={_INTENT_ARMS[intent]['expected_behavior']}, "
               f"prescreen={_INTENT_ARMS[intent]['prescreen_policy']})",
    )
    return candidates


# ---------------------------------------------------------------- comparative arm
#
# The comparative intent asks a CONTRAST question across two DISTINCT-concept
# chunks ("how does X differ from Y"). Unlike both_population (which pairs the
# SAME concept across populations), comparative pairs chunks whose CONCEPT
# identity differs (distinct concept_tags) but which SHARE subject-area
# vocabulary (so the contrast is meaningful — prime vs composite, mean vs
# median). Related concepts routinely share vocabulary, so the pairing keys on
# concept-tag DISTINCTNESS plus a shared-body-token floor, NOT a low-overlap
# ceiling. A chunk with no concept_tags cannot be identity-distinguished and is
# skipped (comparative is opt-in; a tag-less corpus yields no comparative pairs).

_COMPARATIVE_MIN_SHARED_TOKENS = 2  # same subject area => >= this many shared tokens


def _concept_signature(chunk: Dict[str, Any]) -> frozenset:
    """The chunk's CONCEPT identity for comparative distinctness — the
    normalized ``concept_tags`` set (lowercased, whitespace-folded)."""
    sig: set = set()
    for tag in chunk.get("concept_tags") or []:
        if isinstance(tag, str) and tag.strip():
            sig.add(" ".join(tag.lower().split()))
    return frozenset(sig)


@dataclass(frozen=True)
class ComparativePair:
    """A mined ``(chunk_a, chunk_b)`` pair on two DISTINCT concepts that share
    subject-area vocabulary. ``shared`` is the shared content-token count."""

    chunk_a_id: str
    chunk_b_id: str
    shared: int


def mine_comparative_pairs(
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    max_pairs: int = 25,
) -> List[ComparativePair]:
    """Deterministically pair concept-bearing chunks whose concept_tags DIFFER
    (distinct concepts) but which share >= :data:`_COMPARATIVE_MIN_SHARED_TOKENS`
    content tokens (same subject area) for a contrast question. Each chunk is
    used at most once; the best (most shared vocabulary) partner is chosen, ties
    broken by partner id. Id order. Pure + deterministic."""
    ids = [
        cid for cid in sorted(chunks_by_id)
        if _concept_signature(chunks_by_id[cid])
    ]
    tokens = {cid: _content_tokens(chunks_by_id[cid]) for cid in ids}
    sigs = {cid: _concept_signature(chunks_by_id[cid]) for cid in ids}
    used: set = set()
    pairs: List[ComparativePair] = []
    for a in ids:
        if len(pairs) >= max_pairs:
            break
        if a in used:
            continue
        best: Optional[Tuple[int, str]] = None
        for b in ids:
            if b == a or b in used:
                continue
            if sigs[a] == sigs[b]:
                continue  # same concept identity => not a contrast pair
            shared = len(tokens[a] & tokens[b])
            if shared < _COMPARATIVE_MIN_SHARED_TOKENS:
                continue
            # Maximize shared vocabulary (most relatable contrast); tie-break by
            # partner id ASC for determinism.
            if best is None or shared > best[0] or (shared == best[0] and b < best[1]):
                best = (shared, b)
        if best is not None:
            used.add(a)
            used.add(best[1])
            pairs.append(ComparativePair(chunk_a_id=a, chunk_b_id=best[1],
                                         shared=best[0]))
    return pairs


def _comparative_prompt(text_a: str, text_b: str) -> Tuple[str, str]:
    """``(system, user)`` prompts asking for ONE contrast question whose answer
    requires BOTH chunks, with a verbatim quote from each."""
    system = (
        "You author one CONTRAST question that asks how two related but DISTINCT "
        "concepts differ. Its complete answer requires BOTH source chunks — it "
        "must NOT be answerable from either alone. Output JSON only with keys: "
        "question_text (a clear contrast question of at least 10 characters), "
        "expected_key_points (a JSON array of 2 to 4 short factual points the "
        "correct answer must state), quote_a (a VERBATIM substring of at least "
        "40 characters copied from the FIRST chunk), and quote_b (a VERBATIM "
        "substring of at least 40 characters copied from the SECOND chunk). Copy "
        "each quote character for character — do not paraphrase. Do not add "
        "facts not present in the chunks."
    )
    user = (
        f"FIRST chunk text:\n{text_a}\n\n"
        f"SECOND chunk text:\n{text_b}\n\n"
        'Output JSON only: {"question_text": "...", '
        '"expected_key_points": ["...", "..."], '
        '"quote_a": "<verbatim from first chunk>", '
        '"quote_b": "<verbatim from second chunk>"}'
    )
    return system, user


def _parse_comparative_draft(text: str) -> Dict[str, Any]:
    """Parse a comparative draft into ``{question_text, expected_key_points,
    quote_a, quote_b}``. Reuses the lenient JSON-object recovery pattern."""
    raw = (text or "").strip()
    parsed: Optional[Dict[str, Any]] = None
    try:
        cand = json.loads(raw)
        if isinstance(cand, dict):
            parsed = cand
    except json.JSONDecodeError:
        parsed = None
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
        raise GoldDraftError(f"no JSON object recoverable from comparative tail {raw[-120:]!r}")
    qt = parsed.get("question_text")
    quote_a = parsed.get("quote_a")
    quote_b = parsed.get("quote_b")
    if not isinstance(qt, str):
        raise GoldDraftError("comparative draft missing question_text")
    if not isinstance(quote_a, str):
        raise GoldDraftError("comparative draft missing quote_a")
    if not isinstance(quote_b, str):
        raise GoldDraftError("comparative draft missing quote_b")
    kps = parsed.get("expected_key_points")
    if not isinstance(kps, list):
        kps = []
    kps = [str(k).strip() for k in kps if isinstance(k, (str, int, float)) and str(k).strip()]
    return {
        "question_text": qt.strip(),
        "expected_key_points": kps[:_KEY_POINTS_MAX],
        "quote_a": quote_a.strip(),
        "quote_b": quote_b.strip(),
    }


def _build_comparative_candidate(
    pair: ComparativePair,
    chunk_a: Dict[str, Any],
    chunk_b: Dict[str, Any],
    draft: Dict[str, Any],
    *,
    model_id: str,
) -> Dict[str, Any]:
    """Assemble a v1.2 comparative candidate: two passages (a=primary,
    b=supporting), learner_intent=comparative, expected_behavior=answer_grounded,
    question_type=conceptual_synthesis."""
    week_a = _week_of_item_path((chunk_a.get("source") or {}).get("item_path", ""))
    week_b = _week_of_item_path((chunk_b.get("source") or {}).get("item_path", ""))
    n_weeks = len({w for w in (week_a, week_b) if w})
    candidate: Dict[str, Any] = {
        "question_id": f"gqc-cmp-{pair.chunk_a_id}-{pair.chunk_b_id}",
        "question_text": draft["question_text"],
        "question_type": "conceptual_synthesis",
        "difficulty": "hard" if n_weeks >= 2 else "medium",
        "synthesis_scope": "across_weeks" if n_weeks >= 2 else "within_week",
        "learner_intent": INTENT_COMPARATIVE,
        "expected_behavior": "answer_grounded",
        "relevant_passages": [
            {"chunk_id": pair.chunk_a_id, "relevance": "primary",
             "anchor": _anchor_for(chunk_a, draft["quote_a"])},
            {"chunk_id": pair.chunk_b_id, "relevance": "supporting",
             "anchor": _anchor_for(chunk_b, draft["quote_b"])},
        ],
        "authoring": {
            "method": "llm_assisted",
            "author": f"{model_id}/{GOLD_AUTHORING_PROMPT_VERSION}",
            "reviewed_by": "PENDING_REVIEW",
            "status": "draft",
            "template": INTENT_COMPARATIVE,
        },
    }
    refs = tuple(sorted(set(_objective_refs(chunk_a)) | set(_objective_refs(chunk_b))))
    if refs:
        candidate["objective_refs"] = list(refs)
    kps = draft.get("expected_key_points") or []
    if len(kps) >= _KEY_POINTS_MIN:
        candidate["expected_key_points"] = kps[:_KEY_POINTS_MAX]
    return candidate


def draft_comparative_candidates(
    pairs: Sequence[ComparativePair],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Draft one comparative (contrast) candidate per mined distinct-concept
    pair. Two passages + verbatim quote per chunk. One
    ``gold_candidate_authoring`` decision per batch. NEVER an Anthropic surface."""
    resolved_model = model_id or getattr(client, "model", "local")
    candidates: List[Dict[str, Any]] = []
    errors = 0
    for pair in pairs:
        chunk_a = chunks_by_id.get(pair.chunk_a_id) or {}
        chunk_b = chunks_by_id.get(pair.chunk_b_id) or {}
        system, user = _comparative_prompt(
            str(chunk_a.get("text") or ""), str(chunk_b.get("text") or "")
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = client.chat_completion(messages, max_tokens=640, temperature=0.2)
            draft = _parse_comparative_draft(text)
            cand = _build_comparative_candidate(
                pair, chunk_a, chunk_b, draft, model_id=resolved_model
            )
        except (GoldDraftError, Exception) as exc:  # noqa: BLE001 — record, don't drop
            errors += 1
            cand = {
                "question_id": f"gqc-cmp-{pair.chunk_a_id}-{pair.chunk_b_id}",
                "question_text": "",
                "question_type": "conceptual_synthesis",
                "learner_intent": INTENT_COMPARATIVE,
                "expected_behavior": "answer_grounded",
                "relevant_passages": [
                    {"chunk_id": pair.chunk_a_id, "relevance": "primary",
                     "anchor": {"item_path": str((chunk_a.get("source") or {}).get("item_path") or ""),
                                "text_quote": ""}},
                    {"chunk_id": pair.chunk_b_id, "relevance": "supporting",
                     "anchor": {"item_path": str((chunk_b.get("source") or {}).get("item_path") or ""),
                                "text_quote": ""}},
                ],
                "authoring": {
                    "method": "llm_assisted",
                    "author": f"{resolved_model}/{GOLD_AUTHORING_PROMPT_VERSION}",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "draft",
                    "template": INTENT_COMPARATIVE,
                },
                "draft_error": str(exc),
            }
        candidates.append(cand)

    pair_labels = [f"{p.chunk_a_id}+{p.chunk_b_id}" for p in pairs]
    _log_template_authoring(
        capture, template=INTENT_COMPARATIVE, model=resolved_model,
        seeds=pair_labels, n=len(candidates), errors=errors,
        detail=f"learner_intent 'comparative' arm — paired {len(pairs)} "
               f"distinct-concept chunk pair(s) for a contrast question",
    )
    return candidates


# ---------------------------------------------------------------- paraphraser
#
# A phrasing paraphraser takes ACCEPTED canonical gold questions and emits
# learner-phrasing VARIANTS (v1.1 ``phrasing`` axis) that reuse the original's
# relevant_passages VERBATIM — only the question_text changes. Two arms:
#
#   (a) malformed — DETERMINISTIC (seeded, no LLM) typo / dropped-article /
#       dropped-word / telegraphic templates. phrasing="malformed".
#   (b) colloquial — the local provider rewrites the question in a bare,
#       conversational learner voice. phrasing="colloquial". A paraphrase that
#       drifts off the original answer target (token-overlap below the floor)
#       is REJECTED — a colloquial rewrite must still ask the SAME thing.
#
# Both preserve relevant_passages byte-for-byte (the retrieval target is
# unchanged), carry a candidate-only ``derived_from_gold_question_id`` (stripped
# at promotion), and set authoring.template=phrasing_{malformed,colloquial}.

PHRASING_MALFORMED_TEMPLATE = "phrasing_malformed"
PHRASING_COLLOQUIAL_TEMPLATE = "phrasing_colloquial"

# Reject a colloquial paraphrase whose significant-token Jaccard vs the original
# question falls below this floor (it has drifted off the answer target). Set to
# catch a topic CHANGE, not to demand high surface similarity — a genuine
# colloquial rewrite legitimately swaps many surface words.
_PARAPHRASE_MIN_OVERLAP = 0.25
# Function words dropped before the token-overlap comparison so the floor keys
# on content words, not "the/is/of".
_PARAPHRASE_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
        "for", "and", "or", "do", "does", "did", "how", "what", "why", "when",
        "which", "that", "this", "with", "as", "at", "by", "be", "you", "your",
        "it", "its", "can", "could", "would", "should", "we", "i", "me",
    }
)
_ARTICLES = ("the", "a", "an")


def _overlap_tokens(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    return {t for t in toks if t not in _PARAPHRASE_STOPWORDS and len(t) > 1}


def _token_overlap(a: str, b: str) -> float:
    ta, tb = _overlap_tokens(a), _overlap_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def make_malformed_variant(question: str, *, seed: int = 0) -> str:
    """Deterministically produce ONE malformed learner-phrasing variant of
    ``question`` (typo / dropped-article / dropped-word / telegraphic). Seeded:
    the same ``(question, seed)`` always yields the same variant. Never returns
    an empty / <10-char string (falls back to a lowercased de-punctuated form)."""
    original = str(question or "").strip()
    words = original.split()
    # Deterministic template pick from a hash of the question + seed.
    h = int(hashlib.sha256(f"{seed}:{original}".encode()).hexdigest()[:8], 16)
    template = h % 4

    def _drop_article(ws: List[str]) -> List[str]:
        out = [w for w in ws if w.lower().strip("?.,") not in _ARTICLES]
        return out or ws

    def _drop_word(ws: List[str]) -> List[str]:
        # Drop one non-first, non-last content word (deterministic index).
        idxs = [
            i for i in range(1, len(ws) - 1)
            if ws[i].lower().strip("?.,") not in _PARAPHRASE_STOPWORDS
        ]
        if not idxs:
            return ws
        drop = idxs[h % len(idxs)]
        return ws[:drop] + ws[drop + 1:]

    def _typo(text: str) -> str:
        # Swap two adjacent characters in the longest word (deterministic).
        if not words:
            return text
        longest = max(range(len(words)), key=lambda i: len(words[i]))
        w = words[longest]
        core = w.rstrip("?.,")
        if len(core) >= 4:
            j = 1 + (h % (len(core) - 2))
            core = core[:j] + core[j + 1] + core[j] + core[j + 2:]
            ws = list(words)
            ws[longest] = core + w[len(core):]
            return " ".join(ws)
        return text

    def _telegraphic(ws: List[str]) -> List[str]:
        out = [w for w in ws if w.lower().strip("?.,") not in _PARAPHRASE_STOPWORDS]
        return out or ws

    if template == 0:
        variant = " ".join(_drop_article(words))
    elif template == 1:
        variant = " ".join(_drop_word(words))
    elif template == 2:
        variant = _typo(original)
    else:
        variant = " ".join(_telegraphic(words)).rstrip("?.").strip() + "?"

    variant = variant.strip()
    if len(variant) < _MIN_QUESTION_CHARS:
        # Fall back to a lowercased, de-punctuated malformation that never
        # under-runs the pre-screen length floor.
        variant = original.lower()
    return variant


def _variant_candidate(
    base: Dict[str, Any],
    variant_text: str,
    *,
    phrasing: str,
    template: str,
    method: str,
    model_id: str,
) -> Dict[str, Any]:
    """Build a phrasing-variant candidate from a base gold question, reusing its
    relevant_passages VERBATIM and stamping the phrasing axis + provenance."""
    cand: Dict[str, Any] = {
        "question_id": f"gqc-{phrasing}-{base.get('question_id', 'q')}",
        "question_text": variant_text,
        "question_type": base.get("question_type", "factual_recall"),
        # relevant_passages preserved byte-for-byte (the retrieval target is the
        # SAME as the canonical question — only the phrasing changes).
        "relevant_passages": json.loads(json.dumps(base.get("relevant_passages") or [])),
        "phrasing": phrasing,
        "authoring": {
            "method": method,
            "author": f"{model_id}/{GOLD_AUTHORING_PROMPT_VERSION}",
            "reviewed_by": "PENDING_REVIEW",
            "status": "draft",
            "template": template,
        },
    }
    # Carry forward the schema-valid axis / metadata fields when present.
    for key in ("objective_refs", "expected_citation_population",
                "expected_key_points", "learner_intent", "expected_behavior",
                "synthesis_scope", "difficulty", "parts"):
        if key in base:
            cand[key] = json.loads(json.dumps(base[key]))
    base_id = base.get("question_id")
    if isinstance(base_id, str):
        # Candidate-only provenance link (stripped at promotion — the gold
        # question schema has no derived-from field).
        cand["derived_from_gold_question_id"] = base_id
    return cand


def paraphrase_malformed(
    questions: Sequence[Dict[str, Any]],
    *,
    seed: int = 0,
    model_id: str = "deterministic",
) -> List[Dict[str, Any]]:
    """Deterministically emit ONE malformed phrasing variant per input gold
    question (no LLM). phrasing='malformed'. Pure + seeded."""
    out: List[Dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        qt = str(q.get("question_text") or "")
        if not qt.strip():
            continue
        variant = make_malformed_variant(qt, seed=seed)
        out.append(_variant_candidate(
            q, variant, phrasing="malformed",
            template=PHRASING_MALFORMED_TEMPLATE, method="manual",
            model_id="malformed-paraphraser",
        ))
    return out


def _colloquial_prompt(question: str) -> Tuple[str, str]:
    system = (
        "You rewrite a formal course question in a bare, conversational learner "
        "voice — the way a student would actually type it into a chat box. Keep "
        "the SAME meaning and ask for the SAME thing; only change the phrasing. "
        "Output JSON only: {\"question\": \"<colloquial rewrite>\"}."
    )
    user = (
        f"Formal question:\n{question}\n\n"
        'Rewrite it colloquially. Output JSON only: {"question": "..."}'
    )
    return system, user


def _parse_colloquial(text: str) -> Optional[str]:
    """Recover the colloquial rewrite string from a model response."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            val = parsed.get("question") or parsed.get("question_text")
            if isinstance(val, str) and val.strip():
                return val.strip().strip("\"'").strip()
    except json.JSONDecodeError:
        pass
    # Embedded-object scan.
    start = raw.find("{")
    depth = 0
    for i in range(start, len(raw)) if start >= 0 else []:
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(raw[start:i + 1])
                    if isinstance(parsed, dict):
                        val = parsed.get("question") or parsed.get("question_text")
                        if isinstance(val, str) and val.strip():
                            return val.strip().strip("\"'").strip()
                except json.JSONDecodeError:
                    pass
                break
    return None


def paraphrase_colloquial(
    questions: Sequence[Dict[str, Any]],
    *,
    client: Any,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
    min_overlap: float = _PARAPHRASE_MIN_OVERLAP,
) -> List[Dict[str, Any]]:
    """Emit colloquial phrasing variants via the local provider (one call per
    input question). phrasing='colloquial'; relevant_passages preserved verbatim.
    A rewrite whose significant-token overlap vs the original falls below
    ``min_overlap`` is REJECTED (it drifted off the answer target). One
    ``gold_candidate_authoring`` decision is emitted with a dynamic rationale.
    NEVER an Anthropic surface."""
    resolved_model = model_id or getattr(client, "model", "local")
    out: List[Dict[str, Any]] = []
    attempted = 0
    rejected = 0
    for q in questions:
        if not isinstance(q, dict):
            continue
        qt = str(q.get("question_text") or "").strip()
        if not qt:
            continue
        attempted += 1
        system, user = _colloquial_prompt(qt)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = client.chat_completion(messages, max_tokens=256, temperature=0.4)
            variant = _parse_colloquial(text)
        except Exception:  # noqa: BLE001 — a failed rewrite is skipped, not fatal
            variant = None
        if not variant or len(variant) < _MIN_QUESTION_CHARS:
            rejected += 1
            continue
        # Reject a paraphrase that changed the answer target.
        if _token_overlap(qt, variant) < min_overlap:
            rejected += 1
            continue
        out.append(_variant_candidate(
            q, variant, phrasing="colloquial",
            template=PHRASING_COLLOQUIAL_TEMPLATE, method="llm_assisted",
            model_id=resolved_model,
        ))

    if capture is not None:
        try:
            capture.log_decision(
                decision_type="gold_candidate_authoring",
                decision=(
                    f"drafted {len(out)} colloquial phrasing variant(s) via "
                    f"local model {resolved_model}; {rejected} rejected "
                    f"(parse-fail or off-target)."
                ),
                rationale=(
                    f"Colloquial phrasing-paraphrase pass over {attempted} "
                    f"canonical gold question(s) using license-clean local model "
                    f"{resolved_model} (prompt {GOLD_AUTHORING_PROMPT_VERSION}); "
                    f"accepted={len(out)}, rejected={rejected} at token-overlap "
                    f"floor {min_overlap}. Each variant preserves the original's "
                    f"relevant_passages verbatim (same retrieval target) and is "
                    f"stamped phrasing=colloquial so the eval's phrasing slice "
                    f"measures bare/learner voice, not only textbook phrasing."
                ),
            )
        except Exception:  # pragma: no cover — defensive
            pass
    return out


# ---------------------------------------------------------------- round-trip


# Env-overridable retrieval engine for the round-trip prescreen filter. Default
# ``lexical`` is CPU-safe (BM25, no embed model — offline-test-safe); set to
# ``hybrid-rrf`` (or ``semantic``) for a production authoring run with a built
# vector index. Parse-with-fallback: an unknown value → ``lexical``.
ENV_ROUNDTRIP_ENGINE = "ED4ALL_GOLD_ROUNDTRIP_ENGINE"
_ROUNDTRIP_ENGINES = ("lexical", "semantic", "hybrid-rrf")
_DEFAULT_ROUNDTRIP_ENGINE = "lexical"
_ROUNDTRIP_TOP_K = 8


def resolve_roundtrip_engine() -> str:
    """Resolve the round-trip retrieval engine from the environment
    (parse-with-fallback to ``lexical``)."""
    import os
    val = (os.environ.get(ENV_ROUNDTRIP_ENGINE) or "").strip().lower()
    return val if val in _ROUNDTRIP_ENGINES else _DEFAULT_ROUNDTRIP_ENGINE


def make_roundtrip_filter(
    retrieve_fn: Any,
    libv2_root: Any,
    course_slug: str,
    *,
    k: int = _ROUNDTRIP_TOP_K,
    engine: Optional[str] = None,
) -> Any:
    """Build a ``roundtrip_fn(question_text) -> List[chunk_id]`` closure over the
    LibV2 ``retrieve_chunks`` machinery for the prescreen round-trip filter.

    ``retrieve_fn`` is duck-typed on ``retrieve_fn(libv2_root, query,
    course_slug=, engine=, limit=)`` (i.e. ``retrieve_chunks``; tests inject a
    fake). The engine defaults to :func:`resolve_roundtrip_engine`
    (``ED4ALL_GOLD_ROUNDTRIP_ENGINE``, default CPU-safe ``lexical``). A retriever
    that raises records an EMPTY top-k (the candidate then fails the round-trip —
    fail-closed, never a silent pass)."""
    resolved_engine = engine or resolve_roundtrip_engine()

    def _roundtrip(question_text: str) -> List[str]:
        try:
            results = list(
                retrieve_fn(libv2_root, question_text, course_slug=course_slug,
                            engine=resolved_engine, limit=k)
            )
        except Exception:  # noqa: BLE001 — a missing index => empty top-k
            return []
        return [str(getattr(r, "chunk_id", "")) for r in results[:k]]

    return _roundtrip


# ---------------------------------------------------------------- prescreen


def _numeric_literals(text: str) -> set:
    """The set of normalized numeric literals in ``text`` (thousands/decimal
    separators folded so ``1,000`` and ``1000`` compare equal)."""
    return {
        m.replace(",", "") for m in _NUMERIC_LITERAL_RE.findall(str(text or ""))
    }


def prescreen_candidate(
    candidate: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    prior_questions: Sequence[str] = (),
    roundtrip_fn: Optional[Any] = None,
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
      * ``ROUNDTRIP_MISS`` — (when ``roundtrip_fn`` supplied) the primary source
        chunk is not in the retriever's top-k for the drafted question text.

    Per-intent prescreen policy: a ``learner_intent=word_problem`` candidate is
    scored with the NUMERIC-LITERAL policy (the quote need only share a numeric
    literal with its chunk — a word problem's grounding is the numbers, not a
    >=40-char verbatim quote); every other candidate uses the verbatim-quote
    policy. Phrasing VARIANT candidates (``phrasing`` in colloquial/malformed)
    are EXEMPT from the near-duplicate check — an intentional paraphrase of a
    canonical question is a near-duplicate BY DESIGN.
    """
    reasons: List[str] = []
    if candidate.get("draft_error"):
        reasons.append("DRAFT_ERROR")
    qt = str(candidate.get("question_text") or "")
    if len(qt.strip()) < _MIN_QUESTION_CHARS:
        reasons.append("QUESTION_TOO_SHORT")

    numeric_policy = candidate.get("learner_intent") in _NUMERIC_PRESCREEN_INTENTS
    passages = candidate.get("relevant_passages") or []
    primary = next(
        (p for p in passages if isinstance(p, dict) and p.get("relevance") == "primary"),
        passages[0] if passages and isinstance(passages[0], dict) else None,
    )
    if isinstance(primary, dict):
        anchor = primary.get("anchor") or {}
        quote = str(anchor.get("text_quote") or "")
        cid = primary.get("chunk_id")
        if numeric_policy:
            # word_problem: the quote must carry >=1 numeric literal that also
            # appears in the source chunk (shared numbers = grounding).
            q_nums = _numeric_literals(quote)
            if not q_nums:
                reasons.append("QUOTE_NO_NUMERIC")
            elif isinstance(cid, str) and cid in chunks_by_id:
                if not (q_nums & _numeric_literals(chunks_by_id[cid].get("text", ""))):
                    reasons.append("NUMERIC_NOT_IN_CHUNK")
            elif isinstance(cid, str):
                reasons.append("QUOTE_NOT_IN_CHUNK")  # chunk id absent => unanchorable
        elif len(quote.strip()) < _MIN_QUOTE_CHARS:
            reasons.append("QUOTE_TOO_SHORT")
        elif isinstance(cid, str) and cid in chunks_by_id:
            if not _quote_in_chunk(quote, chunks_by_id[cid]):
                reasons.append("QUOTE_NOT_IN_CHUNK")
            else:
                if _quote_chunk_match_count(quote, chunks_by_id) > _AMBIGUOUS_QUOTE_MAX_CHUNKS:
                    reasons.append("QUOTE_AMBIGUOUS")
        elif isinstance(cid, str):
            reasons.append("QUOTE_NOT_IN_CHUNK")  # chunk id absent => unanchorable

    # Multi-passage (e.g. both_population) candidates carry a SUPPORTING passage
    # whose quote must ALSO anchor — a both-population question whose source
    # quote isn't verbatim-contained can't honestly claim its citations span
    # both populations. Verify every non-primary passage carrying a quote; an
    # empty supporting quote (anchor without a quote) is left to the schema.
    for p in passages:
        if not isinstance(p, dict) or p is primary:
            continue
        s_anchor = p.get("anchor") or {}
        s_quote = str(s_anchor.get("text_quote") or "")
        s_cid = p.get("chunk_id")
        if not s_quote.strip():
            continue
        if len(s_quote.strip()) < _MIN_QUOTE_CHARS:
            reasons.append("SUPPORTING_QUOTE_TOO_SHORT")
        elif isinstance(s_cid, str) and s_cid in chunks_by_id:
            if not _quote_in_chunk(s_quote, chunks_by_id[s_cid]):
                reasons.append("SUPPORTING_QUOTE_NOT_IN_CHUNK")
        elif isinstance(s_cid, str):
            reasons.append("SUPPORTING_QUOTE_NOT_IN_CHUNK")

    # Near-dup: only meaningful for a non-empty question, and SKIPPED for
    # phrasing variants (an intentional paraphrase of a canonical question is a
    # near-duplicate by design — that is the whole point of the phrasing axis).
    is_phrasing_variant = candidate.get("phrasing") in ("colloquial", "malformed")
    if qt.strip() and not is_phrasing_variant:
        for prior in prior_questions:
            if not prior or not prior.strip():
                continue
            if shingle_containment(
                qt, prior, shingle_size=_NEAR_DUP_SHINGLE_SIZE
            ) >= _NEAR_DUP_CONTAINMENT:
                reasons.append("NEAR_DUPLICATE")
                break

    # Round-trip filter (optional): a draft survives only if its PRIMARY source
    # chunk appears in the existing retriever's top-k for the drafted question
    # text. Only meaningful for an anchorable question with a resolvable primary.
    if roundtrip_fn is not None and qt.strip() and isinstance(primary, dict):
        cid = primary.get("chunk_id")
        if isinstance(cid, str) and cid:
            try:
                top_ids = list(roundtrip_fn(qt))
            except Exception:  # noqa: BLE001 — a retriever failure fails closed
                top_ids = []
            if cid not in top_ids:
                reasons.append("ROUNDTRIP_MISS")

    return PrescreenVerdict(passed=not reasons, reasons=reasons)


def prescreen_candidates(
    candidates: Sequence[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    existing_questions: Sequence[str] = (),
    roundtrip_fn: Optional[Any] = None,
) -> List[Tuple[Dict[str, Any], PrescreenVerdict]]:
    """Pre-screen a batch; near-dup checks accumulate intra-batch + against
    ``existing_questions`` (the gold set's question_texts). ``roundtrip_fn`` (when
    supplied) enforces the round-trip retrieval filter per candidate."""
    seen: List[str] = list(existing_questions)
    out: List[Tuple[Dict[str, Any], PrescreenVerdict]] = []
    for cand in candidates:
        verdict = prescreen_candidate(
            cand, chunks_by_id, prior_questions=seen, roundtrip_fn=roundtrip_fn
        )
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
    by_template: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Build the ``gold_candidates.json`` wrapper doc.

    ``by_template`` records the per-arm drafted counts (stratified / definition
    / worked_example) so the operator sees how many candidates each template arm
    produced — a zero-yield arm (e.g. a corpus with no glossary chunks) is then
    visible rather than silently absent.
    """
    candidates_out: List[Dict[str, Any]] = []
    for cand, verdict in screened:
        entry = dict(cand)
        entry["prescreen"] = verdict.to_dict()
        candidates_out.append(entry)
    passed = sum(1 for _, v in screened if v.passed)
    authoring_run: Dict[str, Any] = {
        "n_requested": n_requested,
        "n_drafted": len(candidates_out),
        "n_prescreen_passed": passed,
        "seed": seed,
        "model_id": model_id,
        "prompt_version": GOLD_AUTHORING_PROMPT_VERSION,
    }
    if by_template is not None:
        authoring_run["by_template"] = dict(by_template)
    return {
        "schema_version": "1.1",
        "course_slug": course_slug,
        "chunkset": dict(chunkset),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authoring_run": authoring_run,
        "candidates": candidates_out,
    }


_DEFAULT_TEMPLATES: Tuple[str, ...] = (
    DEFAULT_TEMPLATE,
    DEFINITION_TEMPLATE,
    WORKED_EXAMPLE_TEMPLATE,
    BOTH_POPULATION_TEMPLATE,
)
# Opt-in per-learner_intent template arms (v1.2 learner_intent axis). NOT in
# _DEFAULT_TEMPLATES — a default run stays byte-identical; select them
# explicitly (``templates=[..., "conceptual_why", "comparative", ...]``).
_INTENT_TEMPLATES: Tuple[str, ...] = (
    INTENT_CONCEPTUAL_WHY,
    INTENT_COMPARATIVE,
    INTENT_EXAMPLE_SEEKING,
    INTENT_NOTATION_SYMBOL,
    INTENT_WORD_PROBLEM,
)
# Cap the per-template mined-seed counts so the template arms supplement the
# stratified arm without flooding it (they over-generate ~half the base target).
_DEFAULT_DEFINITION_CAP = 25
_DEFAULT_WORKED_EXAMPLE_CAP = 25
_DEFAULT_BOTH_POPULATION_CAP = 25
_DEFAULT_INTENT_CAP = 25


def generate_gold_candidates(
    course_dir: Path,
    *,
    client: Any,
    n: Optional[int] = None,
    seed: int = 0,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
    write: bool = True,
    templates: Optional[Sequence[str]] = None,
    definition_cap: int = _DEFAULT_DEFINITION_CAP,
    worked_example_cap: int = _DEFAULT_WORKED_EXAMPLE_CAP,
    both_population_cap: int = _DEFAULT_BOTH_POPULATION_CAP,
    intent_cap: int = _DEFAULT_INTENT_CAP,
    roundtrip_fn: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """End-to-end §2.2 steps 1-3: sample / mine → draft → pre-screen → write doc.

    Loads the course's pinned chunkset from its gold set's ``chunkset`` pin and
    runs one or more template arms (``templates``, default all three):

      * ``stratified`` — the §1.2 stratified sampler over ``n`` chunks (default
        2x the 50-question target).
      * ``definition`` — deterministically-mined glossary terms, one
        factual_recall question per term (capped at ``definition_cap``).
      * ``worked_example`` — deterministically-mined Problem/Solution/Step
        chunks, one procedural question per chunk (capped at
        ``worked_example_cap``).
      * ``both_population`` — course/source concept pairs (capped at
        ``both_population_cap``).

    The v1.2 learner_intent arms (``conceptual_why`` / ``comparative`` /
    ``example_seeking`` / ``notation_symbol`` / ``word_problem``) are OPT-IN
    (select them explicitly via ``templates=[...]``; NOT in the default set, so a
    default run is byte-identical). Each mines its corpus shape (capped at
    ``intent_cap``) and stamps the ``learner_intent`` + ``expected_behavior`` axes.

    All arms draft via the license-clean local ``client``, then the combined
    candidate list is pre-screened deterministically and written to
    ``retrieval_eval/gold_candidates.json``. When ``roundtrip_fn`` is supplied
    (build one with :func:`make_roundtrip_filter`) the pre-screen additionally
    enforces the round-trip retrieval filter. Returns ``(doc, written_path)``.

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

    selected = tuple(templates) if templates is not None else _DEFAULT_TEMPLATES

    n_resolved = int(n) if n else _DEFAULT_TARGET_QUESTIONS * _DEFAULT_OVERGEN_FACTOR
    is_union = chunkset.get("kind") == "corpus"
    candidates: List[Dict[str, Any]] = []
    by_template: Dict[str, int] = {}
    if DEFAULT_TEMPLATE in selected:
        slots = sample_chunks(chunks_by_id, n=n_resolved, seed=seed, is_union=is_union)
        strat = draft_candidates(
            slots, chunks_by_id, client=client, model_id=model_id, capture=capture
        )
        candidates.extend(strat)
        by_template[DEFAULT_TEMPLATE] = len(strat)
    if DEFINITION_TEMPLATE in selected:
        seeds = mine_glossary_terms(chunks_by_id)[:max(0, definition_cap)]
        defs = draft_definition_candidates(
            seeds, chunks_by_id, client=client, model_id=model_id, capture=capture
        )
        candidates.extend(defs)
        by_template[DEFINITION_TEMPLATE] = len(defs)
    if WORKED_EXAMPLE_TEMPLATE in selected:
        we_ids = mine_worked_example_chunks(chunks_by_id)[:max(0, worked_example_cap)]
        wes = draft_worked_example_candidates(
            we_ids, chunks_by_id, client=client, model_id=model_id, capture=capture
        )
        candidates.extend(wes)
        by_template[WORKED_EXAMPLE_TEMPLATE] = len(wes)
    if BOTH_POPULATION_TEMPLATE in selected:
        pairs = mine_both_population_pairs(
            chunks_by_id, max_pairs=max(0, both_population_cap)
        )
        boths = draft_both_population_candidates(
            pairs, chunks_by_id, client=client, model_id=model_id, capture=capture
        )
        candidates.extend(boths)
        by_template[BOTH_POPULATION_TEMPLATE] = len(boths)
    # Opt-in learner_intent arms (each stamps learner_intent + expected_behavior).
    for intent in _INTENT_ARMS:
        if intent in selected:
            ids = mine_intent_chunks(intent, chunks_by_id)[:max(0, intent_cap)]
            arm_cands = draft_intent_candidates(
                intent, ids, chunks_by_id, client=client,
                model_id=model_id, capture=capture,
            )
            candidates.extend(arm_cands)
            by_template[intent] = len(arm_cands)
    if INTENT_COMPARATIVE in selected:
        cmp_pairs = mine_comparative_pairs(chunks_by_id, max_pairs=max(0, intent_cap))
        cmp_cands = draft_comparative_candidates(
            cmp_pairs, chunks_by_id, client=client, model_id=model_id, capture=capture
        )
        candidates.extend(cmp_cands)
        by_template[INTENT_COMPARATIVE] = len(cmp_cands)

    existing = [
        str(q.get("question_text") or "")
        for q in (gold.get("questions") or [])
        if isinstance(q, dict)
    ]
    screened = prescreen_candidates(
        candidates, chunks_by_id, existing_questions=existing,
        roundtrip_fn=roundtrip_fn,
    )
    resolved_model = model_id or getattr(client, "model", "local")
    doc = build_candidates_doc(
        gold.get("course_slug", course_dir.name),
        chunkset,
        screened,
        n_requested=n_resolved,
        seed=seed,
        model_id=resolved_model,
        by_template=by_template,
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
    """Drop candidate-only fields (prescreen, draft_error, and the phrasing
    paraphraser's ``derived_from_gold_question_id`` provenance link) for the gold
    doc — the gold question schema is ``additionalProperties:false``."""
    out = {
        k: v for k, v in candidate.items()
        if k not in ("prescreen", "draft_error", "derived_from_gold_question_id")
    }
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
    "DEFAULT_TEMPLATE",
    "DEFINITION_TEMPLATE",
    "WORKED_EXAMPLE_TEMPLATE",
    "BOTH_POPULATION_TEMPLATE",
    "INTENT_CONCEPTUAL_WHY",
    "INTENT_COMPARATIVE",
    "INTENT_EXAMPLE_SEEKING",
    "INTENT_NOTATION_SYMBOL",
    "INTENT_WORD_PROBLEM",
    "PHRASING_MALFORMED_TEMPLATE",
    "PHRASING_COLLOQUIAL_TEMPLATE",
    "ENV_ROUNDTRIP_ENGINE",
    "SampleSlot",
    "GlossarySeed",
    "BothPopulationPair",
    "ComparativePair",
    "PrescreenVerdict",
    "PromoteReport",
    "GoldDraftError",
    "GoldPromoteError",
    "sample_chunks",
    "draft_candidates",
    "mine_glossary_terms",
    "draft_definition_candidates",
    "mine_worked_example_chunks",
    "draft_worked_example_candidates",
    "mine_both_population_pairs",
    "draft_both_population_candidates",
    "mine_conceptual_why_chunks",
    "mine_example_chunks",
    "mine_notation_chunks",
    "mine_word_problem_chunks",
    "mine_intent_chunks",
    "draft_intent_candidates",
    "mine_comparative_pairs",
    "draft_comparative_candidates",
    "make_malformed_variant",
    "paraphrase_malformed",
    "paraphrase_colloquial",
    "make_roundtrip_filter",
    "resolve_roundtrip_engine",
    "prescreen_candidate",
    "prescreen_candidates",
    "build_candidates_doc",
    "generate_gold_candidates",
    "promote_candidates_into_gold",
    "promote_candidates",
]
