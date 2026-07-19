"""Refusal-probe candidate authoring (retrieval-answer-eval-set §2.2 step 6).

P2 of the eval-set plan. Produces ``retrieval_eval/refusal_probe_candidates.json``
(NOT the final ``refusal_probes.json`` — candidates carry per-engine dry-run
evidence but leave the human ``top_passage_answers`` confirmation for the
operator). Three probe categories:

  * ``off_topic`` — drawn from a FIXED in-module cross-domain question bank
    (:data:`OFF_TOPIC_BANK`). Deterministic, no LLM.
  * ``adjacent_domain`` — the local provider is prompted with the course's
    ``concept_tags`` vocabulary to produce vocabulary-overlapping but uncovered
    neighbours (the hardest negatives — they share vocabulary). NEVER an
    Anthropic surface.
  * ``out_of_scope_detail`` — deterministic perturbation templates
    (:data:`OOS_DETAIL_TEMPLATES`) over accepted/draft gold questions; each
    carries ``derived_from_gold_question_id``.

A fourth category, ``ill_posed``, exists in the taxonomy (refusal_probes
schema v1.1 enum + the ``_is_refusal_question`` category set) but is NOT
auto-drafted here: it is a near-domain question referencing a non-existent or
self-contradictory concept whose vocabulary sounds algebraic (e.g. "the least
common factor of 95", "the largest common multiple of 6 and 8"). A correct
system must refuse / correct the premise rather than answer a neighbouring real
concept, so these are hand-authored (surfaced by real GUI learner phrasing) and
verified with the same read-only dry-run evidence.

For EVERY probe candidate the tool auto-runs the read-only retrieval dry-run
PER ENGINE (lexical / semantic / hybrid-rrf) via the existing ``retrieve_chunks``
machinery and fills the v1.1 ``dry_runs[]`` evidence (top chunk id + scores +
a top-passage excerpt for the reviewer). ``top_passage_answers`` is emitted as
``null`` (pending operator review) — the FINAL ``refusal_probes.json`` requires
``top_passage_answers: false``; the candidates file deliberately leaves it
unset so the operator must make the call (R5: probe mislabeling mitigation).

Decision capture: the adjacent_domain drafting IS an LLM call path — one
``probe_candidate_authoring`` decision is emitted per drafting batch (dynamic
rationale: concept-tag count, model, drafted count). off_topic +
out_of_scope_detail are deterministic and emit no LLM decision.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lib.retrieval.gold_set import (
    RETRIEVAL_EVAL_SUBDIR,
    _load_chunks_by_id,
    load_gold_set,
)

REFUSAL_PROBE_CANDIDATES_FILENAME = "refusal_probe_candidates.json"
PROBE_AUTHORING_PROMPT_VERSION = "probe-author.v1"

_ENGINES = ("lexical", "semantic", "hybrid-rrf")
_DEFAULT_DRY_RUN_LIMIT = 8
_TOP_PASSAGE_EXCERPT_CHARS = 240

# §1.2 refusal-probe split per 25-probe course target. ``off_topic_llm`` is the
# additive local-model-drafted pure-off-topic arm (sibling of the fixed-bank
# ``off_topic``): the model invents wholly-unrelated-discipline questions
# (cooking / geography / pop culture) rather than drawing from the fixed bank,
# so the off-topic negatives are not pinned to one hand-authored list.
_PROBE_TARGET = {
    "off_topic": 8,
    "off_topic_llm": 10,
    "adjacent_domain": 10,
    "out_of_scope_detail": 7,
}


# A fixed cross-domain off-topic bank. Deterministic; no course coupling. These
# are wholly-different-discipline questions a correctly-scoped course corpus
# cannot answer. The bank over-supplies so the §1.2 quota (8) can be met from a
# deterministic prefix.
OFF_TOPIC_BANK: Tuple[str, ...] = (
    "What is the boiling point of liquid nitrogen at one atmosphere?",
    "Who composed the opera The Magic Flute and in what year?",
    "How do you change the oil filter on a four-stroke motorcycle engine?",
    "What were the main causes of the French Revolution of 1789?",
    "Which vitamins are fat-soluble in human nutrition?",
    "How does a sourdough starter ferment over a multi-day schedule?",
    "What is the offside rule in association football?",
    "Describe the life cycle of a monarch butterfly from egg to adult.",
    "What is the difference between a sonnet and a villanelle?",
    "How is reinforced concrete cured on a large construction site?",
    "What tax form does a sole proprietor file for self-employment income?",
    "How do tides arise from the gravitational pull of the moon?",
)


# Deterministic out-of-scope-detail perturbation templates over a gold question.
# Each {q} interpolation pushes the course's own topic to a depth / specificity
# the corpus does not cover (a plausible but unanswerable detail twin).
OOS_DETAIL_TEMPLATES: Tuple[str, ...] = (
    "{q} — and what is the exact numerical value cited in the original source?",
    "{q} Provide the precise publication date and page number of the primary reference.",
    "{q} What proof or derivation does the course give for this, step by step?",
    "{q} How does this compare to the treatment in a graduate-level text not in this course?",
)


# ---------------------------------------------------------------- data types


@dataclass
class DryRunEvidence:
    """One per-engine read-only retrieval dry-run for a probe candidate."""

    engine: str
    limit: int
    n_results: int
    top_score: float
    top_chunk_id: str = ""
    top_passage_excerpt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            # NOTE: this is the CANDIDATE shape — the final refusal_probes.json
            # dry_runs[] additionally requires verified:true + top_passage_answers:
            # false, which the operator supplies during review. The candidate
            # records the empirical signal + an excerpt for that human call.
            "engine": self.engine,
            "limit": self.limit,
            "n_results": self.n_results,
            "top_score": round(self.top_score, 6),
            "top_chunk_id": self.top_chunk_id,
            "top_passage_excerpt": self.top_passage_excerpt,
            "top_passage_answers": None,  # pending operator review (R5)
        }


# ---------------------------------------------------------------- dry-runs


def _excerpt(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t[:_TOP_PASSAGE_EXCERPT_CHARS]


def run_dry_runs(
    question_text: str,
    *,
    retrieve_fn: Callable[..., Sequence[Any]],
    libv2_root: Path,
    course_slug: str,
    limit: int = _DEFAULT_DRY_RUN_LIMIT,
    engines: Sequence[str] = _ENGINES,
) -> List[DryRunEvidence]:
    """Run the read-only retrieval dry-run for ``question_text`` per engine.

    ``retrieve_fn`` is duck-typed on ``retrieve_fn(libv2_root, query,
    course_slug=, engine=, limit=)`` returning score-bearing results
    (``.score`` / ``.chunk_id`` / ``.text``) — i.e. ``retrieve_chunks``. Tests
    inject a deterministic fake. A retriever that raises (e.g. semantic index
    absent) records a zero-result dry-run rather than failing the whole batch,
    so the lexical evidence still lands.
    """
    out: List[DryRunEvidence] = []
    for engine in engines:
        try:
            results = list(
                retrieve_fn(libv2_root, question_text, course_slug=course_slug,
                            engine=engine, limit=limit)
            )
        except Exception:  # noqa: BLE001 — a missing index records 0 results
            out.append(DryRunEvidence(engine=engine, limit=limit, n_results=0,
                                      top_score=0.0))
            continue
        scores = [float(getattr(r, "score", 0.0) or 0.0) for r in results]
        top_score = max(scores) if scores else 0.0
        top_id = str(getattr(results[0], "chunk_id", "")) if results else ""
        top_text = str(getattr(results[0], "text", "")) if results else ""
        out.append(
            DryRunEvidence(
                engine=engine, limit=limit, n_results=len(results),
                top_score=top_score, top_chunk_id=top_id,
                top_passage_excerpt=_excerpt(top_text),
            )
        )
    return out


# ---------------------------------------------------------------- bank/perturb


def off_topic_candidates(n: int) -> List[Dict[str, Any]]:
    """Deterministic off_topic probe candidates from the fixed bank prefix."""
    n = max(0, min(n, len(OFF_TOPIC_BANK)))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append(
            {
                "question_text": OFF_TOPIC_BANK[i],
                "category": "off_topic",
                "why_unanswerable": (
                    "Wholly different discipline from the course corpus; no "
                    "passage shares the topic (deterministic cross-domain bank)."
                ),
                "authoring": {
                    "method": "manual",
                    "author": f"off-topic-bank/{PROBE_AUTHORING_PROMPT_VERSION}",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "draft",
                },
            }
        )
    return out


def out_of_scope_detail_candidates(
    gold_questions: Sequence[Dict[str, Any]], n: int
) -> List[Dict[str, Any]]:
    """Deterministic out_of_scope_detail twins of gold questions.

    Walks the gold questions in id order, applies the perturbation templates
    round-robin, and links each twin via ``derived_from_gold_question_id``.
    """
    usable = [
        q for q in gold_questions
        if isinstance(q, dict)
        and isinstance(q.get("question_text"), str)
        and isinstance(q.get("question_id"), str)
    ]
    usable.sort(key=lambda q: q["question_id"])
    out: List[Dict[str, Any]] = []
    ti = 0
    for q in usable:
        if len(out) >= n:
            break
        base = q["question_text"].rstrip("?. ")
        template = OOS_DETAIL_TEMPLATES[ti % len(OOS_DETAIL_TEMPLATES)]
        ti += 1
        out.append(
            {
                "question_text": template.format(q=base),
                "category": "out_of_scope_detail",
                "why_unanswerable": (
                    "The course's own topic at a depth/specificity the corpus "
                    "does not cover (deterministic perturbation of a gold "
                    "question); the precise detail asked for is absent."
                ),
                "derived_from_gold_question_id": q["question_id"],
                "authoring": {
                    "method": "manual",
                    "author": f"oos-perturb/{PROBE_AUTHORING_PROMPT_VERSION}",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "draft",
                },
            }
        )
    return out


# ---------------------------------------------------------------- adjacent LLM


def _course_concept_vocabulary(chunks_by_id: Dict[str, Dict[str, Any]],
                               *, cap: int = 60) -> List[str]:
    """Frequency-ordered ``concept_tags`` vocabulary across the chunkset."""
    counts: Dict[str, int] = {}
    for chunk in chunks_by_id.values():
        for tag in chunk.get("concept_tags") or []:
            if isinstance(tag, str) and tag.strip():
                counts[tag.strip()] = counts.get(tag.strip(), 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:cap]]


def _clean_question(s: str) -> str:
    """Normalize one drafted question: strip wrapping quotes/escapes/bullets
    and any trailing punctuation junk after the question mark."""
    s = str(s).strip().strip("\"'").replace('\\"', '"').strip()
    s = s.lstrip("-*0123456789. ").strip()
    # Truncate trailing junk after the LAST question mark (e.g. '..."?').
    if "?" in s:
        s = s[: s.rfind("?") + 1]
        # Collapse an escaped-quote echo before the final mark ('...?"?' →
        # '...?'), the observed 7B keys-as-questions artifact; when the inner
        # text ends in OTHER sentence punctuation ('...."?'), drop the stray
        # tail entirely so the ends-with-? filter rejects the non-question.
        s = re.sub(r"\?[\"']*\?$", "?", s)
        s = re.sub(r"([.!])[\"']*\?$", r"\1", s)
    return s.strip().strip("\"'").strip()


def _parse_adjacent(text: str) -> List[str]:
    """Parse a model response into a list of probe question strings.

    Lenient by design — small-quant local models emit JSON arrays, fenced
    arrays, ``{"questions": [...]}`` objects, and (observed live with a
    7B-Q4) degenerate objects whose KEYS are the escaped question strings
    with empty values. Harvest question-shaped strings from any of these;
    final fallbacks scan quoted spans and bare lines ending in '?'."""
    raw = (text or "").strip()
    # Strip markdown fences (```json ... ```).
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    found: List[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            found = [_clean_question(x) for x in parsed]
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("questions"), list):
                found = [_clean_question(x) for x in parsed["questions"]]
            else:
                # Degenerate keys-as-questions shape: harvest keys AND any
                # string values that read as questions.
                pool = list(parsed.keys()) + [
                    v for v in parsed.values() if isinstance(v, str)
                ]
                found = [_clean_question(x) for x in pool]
    except json.JSONDecodeError:
        pass
    found = [q for q in found if len(q) >= 10 and q.endswith("?")]
    if found:
        return found
    # Quoted-span fallback: every "..." span ending in '?'.
    spans = [
        _clean_question(m) for m in re.findall(r'"([^"\n]{10,300}?\?)"', raw)
    ]
    spans = [q for q in spans if len(q) >= 10 and q.endswith("?")]
    if spans:
        return spans
    # Newline fallback — drop obvious list bullets.
    lines = []
    for ln in raw.splitlines():
        s = _clean_question(ln)
        if len(s) >= 10 and s.endswith("?"):
            lines.append(s)
    return lines


def adjacent_domain_candidates(
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    client: Any,
    n: int,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Draft adjacent_domain probes via the local provider over the course's
    ``concept_tags`` vocabulary. NEVER an Anthropic surface.

    Emits one ``probe_candidate_authoring`` decision with a dynamic rationale.
    """
    vocab = _course_concept_vocabulary(chunks_by_id)
    resolved_model = model_id or getattr(client, "model", "local")
    if not vocab:
        return []
    vocab_str = ", ".join(vocab[:40])
    system = (
        "You author 'adjacent domain' refusal probes: plausible-sounding "
        "questions that use vocabulary OVERLAPPING a course but ask about a "
        "neighbouring topic the course does NOT cover. They must be the hardest "
        "negatives — sharing terms but unanswerable from the course corpus. "
        "Output JSON only: a JSON array of question strings, each ending in '?'."
    )
    user = (
        f"Course concept vocabulary (frequency-ordered):\n{vocab_str}\n\n"
        f"Write {n} adjacent-domain probe questions that reuse some of this "
        f"vocabulary but ask about a topic the course corpus does not contain. "
        f'Output JSON array only: ["...?", "...?"]'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    questions: List[str] = []
    try:
        text = client.chat_completion(messages, max_tokens=512, temperature=0.4)
        questions = _parse_adjacent(text)
    except Exception:  # noqa: BLE001 — empty draft is recorded as zero probes
        questions = []
    questions = [q for q in questions if len(q.strip()) >= 10][:n]

    out: List[Dict[str, Any]] = []
    for q in questions:
        out.append(
            {
                "question_text": q.strip(),
                "category": "adjacent_domain",
                "why_unanswerable": (
                    "Plausible neighbour topic that reuses course vocabulary but "
                    "is not covered by any passage (LLM-drafted against the "
                    "concept_tags vocabulary; operator confirms via dry-run)."
                ),
                "authoring": {
                    "method": "llm_assisted",
                    "author": f"{resolved_model}/{PROBE_AUTHORING_PROMPT_VERSION}",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "draft",
                },
            }
        )

    if capture is not None:
        try:
            capture.log_decision(
                decision_type="probe_candidate_authoring",
                decision=(
                    f"drafted {len(out)} adjacent_domain probe(s) via local "
                    f"model {resolved_model} over {len(vocab)} concept tag(s)."
                ),
                rationale=(
                    f"Adjacent-domain refusal-probe drafting using license-clean "
                    f"local model {resolved_model} (prompt "
                    f"{PROBE_AUTHORING_PROMPT_VERSION}) seeded by the course's top "
                    f"{min(40, len(vocab))} concept_tags ({vocab_str[:120]}...); "
                    f"requested={n}, drafted={len(out)}. These are the hardest "
                    f"negatives — vocabulary-overlapping but uncovered — and each "
                    f"gets a per-engine read-only dry-run before operator review."
                ),
            )
        except Exception:  # pragma: no cover — defensive
            pass
    return out


# ---------------------------------------------------------------- off-topic LLM


# Disciplines the local model is asked to draw the pure-off-topic questions
# from — wholly unrelated to any course corpus. Kept as a HINT list (the model
# invents the questions); deterministic only in that the SAME hint is sent each
# run, so the arm is reproducible at the prompt level while the text is drafted.
_OFF_TOPIC_DOMAINS = (
    "cooking and recipes",
    "world geography and capital cities",
    "pop culture, film, and music",
    "professional sports and their rules",
    "gardening and houseplants",
    "automotive maintenance",
)


def off_topic_llm_candidates(
    *,
    client: Any,
    n: int,
    model_id: Optional[str] = None,
    capture: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Draft pure off-topic refusal probes via the local provider.

    Distinct from the fixed-bank :func:`off_topic_candidates`: the local model
    INVENTS ``n`` questions that are completely unrelated to any course material
    (cooking, geography, pop culture, …) rather than drawing from the fixed
    bank. The point of a refusal probe is a question the corpus cannot answer;
    these have zero course coupling by construction. NEVER an Anthropic surface.

    Emits one ``probe_candidate_authoring`` decision with a dynamic rationale.
    """
    resolved_model = model_id or getattr(client, "model", "local")
    if n <= 0:
        return []
    domains_str = ", ".join(_OFF_TOPIC_DOMAINS)
    system = (
        "You author 'off topic' refusal probes: questions about everyday "
        "general-knowledge subjects that are COMPLETELY UNRELATED to any "
        "academic course — cooking, geography, sports, pop culture, gardening, "
        "and the like. They must share NO vocabulary or topic with a math, "
        "science, or humanities course. Each must be a real, answerable "
        "general-knowledge question (just not from any course corpus). "
        "Output JSON only: a JSON array of question strings, each ending in '?'."
    )
    user = (
        f"Draw from these everyday domains: {domains_str}.\n\n"
        f"Write {n} off-topic questions that are completely unrelated to any "
        f"course material (no math, science, or humanities course terms). "
        f'Output JSON array only: ["...?", "...?"]'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    questions: List[str] = []
    try:
        text = client.chat_completion(messages, max_tokens=512, temperature=0.5)
        questions = _parse_adjacent(text)
    except Exception:  # noqa: BLE001 — empty draft is recorded as zero probes
        questions = []
    questions = [q for q in questions if len(q.strip()) >= 10][:n]

    out: List[Dict[str, Any]] = []
    for q in questions:
        out.append(
            {
                "question_text": q.strip(),
                "category": "off_topic_llm",
                "why_unanswerable": (
                    "Wholly different discipline from the course corpus, "
                    "drafted by the local model from everyday off-topic domains "
                    "(no shared vocabulary or topic); operator confirms via dry-run."
                ),
                "authoring": {
                    "method": "llm_assisted",
                    "author": f"{resolved_model}/{PROBE_AUTHORING_PROMPT_VERSION}",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "draft",
                },
            }
        )

    if capture is not None:
        try:
            capture.log_decision(
                decision_type="probe_candidate_authoring",
                decision=(
                    f"drafted {len(out)} off_topic_llm probe(s) via local model "
                    f"{resolved_model} across {len(_OFF_TOPIC_DOMAINS)} everyday domains."
                ),
                rationale=(
                    f"Pure off-topic refusal-probe drafting using license-clean "
                    f"local model {resolved_model} (prompt "
                    f"{PROBE_AUTHORING_PROMPT_VERSION}) seeded by everyday "
                    f"off-topic domains ({domains_str[:120]}); requested={n}, "
                    f"drafted={len(out)}. These have zero course coupling by "
                    f"construction (cooking/geography/pop-culture) and each gets "
                    f"a per-engine read-only dry-run before operator review."
                ),
            )
        except Exception:  # pragma: no cover — defensive
            pass
    return out


# ---------------------------------------------------------------- assemble


def attach_dry_runs(
    candidates: Sequence[Dict[str, Any]],
    *,
    retrieve_fn: Callable[..., Sequence[Any]],
    libv2_root: Path,
    course_slug: str,
    limit: int = _DEFAULT_DRY_RUN_LIMIT,
    engines: Sequence[str] = _ENGINES,
) -> List[Dict[str, Any]]:
    """Attach a ``dry_runs[]`` block (one per engine) to each probe candidate."""
    out: List[Dict[str, Any]] = []
    for cand in candidates:
        entry = dict(cand)
        evidence = run_dry_runs(
            str(cand.get("question_text") or ""),
            retrieve_fn=retrieve_fn, libv2_root=libv2_root,
            course_slug=course_slug, limit=limit, engines=engines,
        )
        entry["dry_runs"] = [e.to_dict() for e in evidence]
        out.append(entry)
    return out


def build_probe_candidates_doc(
    course_slug: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    model_id: str,
) -> Dict[str, Any]:
    """Assemble + number the ``refusal_probe_candidates.json`` wrapper doc."""
    numbered: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidates, start=1):
        entry = dict(cand)
        entry["probe_id"] = f"rpc-{course_slug}-{i:04d}"
        numbered.append(entry)
    by_cat: Dict[str, int] = {}
    for c in numbered:
        cat = c.get("category", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
    return {
        "schema_version": "1.1",
        "course_slug": course_slug,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authoring_run": {
            # Zero-yield arms are reported EXPLICITLY (never silently
            # omitted): an empty adjacent_domain draft means the LLM arm
            # failed/parsed-to-nothing and the operator must know.
            "by_category": {
                cat: by_cat.get(cat, 0) for cat in _PROBE_TARGET
            },
            "underfilled_categories": [
                cat for cat, want in _PROBE_TARGET.items()
                if by_cat.get(cat, 0) < want
            ],
            "n_probes": len(numbered),
            "model_id": model_id,
            "prompt_version": PROBE_AUTHORING_PROMPT_VERSION,
        },
        "probe_candidates": numbered,
    }


def generate_probe_candidates(
    course_dir: Path,
    *,
    client: Any,
    retrieve_fn: Callable[..., Sequence[Any]],
    libv2_root: Path,
    targets: Optional[Dict[str, int]] = None,
    model_id: Optional[str] = None,
    limit: int = _DEFAULT_DRY_RUN_LIMIT,
    engines: Sequence[str] = _ENGINES,
    capture: Optional[Any] = None,
    write: bool = True,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """End-to-end §2.2 step 6: build the three probe categories, attach
    per-engine dry-runs, write ``refusal_probe_candidates.json``.

    ``retrieve_fn`` + ``libv2_root`` drive the read-only dry-runs (inject a
    deterministic fake in tests). Returns ``(doc, written_path)``.
    """
    course_dir = Path(course_dir)
    gold, _ = load_gold_set(course_dir, verify=False)
    course_slug = (gold.get("course_slug") if gold else None) or course_dir.name
    chunks_rel = ((gold.get("chunkset") or {}).get("chunks_path")) if gold else None
    chunks_by_id: Dict[str, Dict[str, Any]] = {}
    if chunks_rel and (course_dir / chunks_rel).is_file():
        chunks_by_id = _load_chunks_by_id(course_dir / chunks_rel)

    tgt = dict(_PROBE_TARGET)
    if targets:
        tgt.update(targets)

    candidates: List[Dict[str, Any]] = []
    candidates.extend(off_topic_candidates(tgt.get("off_topic", 0)))
    candidates.extend(
        off_topic_llm_candidates(
            client=client, n=tgt.get("off_topic_llm", 0),
            model_id=model_id, capture=capture,
        )
    )
    candidates.extend(
        adjacent_domain_candidates(
            chunks_by_id, client=client, n=tgt.get("adjacent_domain", 0),
            model_id=model_id, capture=capture,
        )
    )
    candidates.extend(
        out_of_scope_detail_candidates(
            (gold.get("questions") or []) if gold else [],
            tgt.get("out_of_scope_detail", 0),
        )
    )

    candidates = attach_dry_runs(
        candidates, retrieve_fn=retrieve_fn, libv2_root=libv2_root,
        course_slug=course_slug, limit=limit, engines=engines,
    )
    resolved_model = model_id or getattr(client, "model", "local")
    doc = build_probe_candidates_doc(course_slug, candidates, model_id=resolved_model)

    out_path: Optional[Path] = None
    if write:
        out_dir = course_dir / RETRIEVAL_EVAL_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / REFUSAL_PROBE_CANDIDATES_FILENAME
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc, out_path


__all__ = [
    "REFUSAL_PROBE_CANDIDATES_FILENAME",
    "PROBE_AUTHORING_PROMPT_VERSION",
    "OFF_TOPIC_BANK",
    "OOS_DETAIL_TEMPLATES",
    "DryRunEvidence",
    "run_dry_runs",
    "off_topic_candidates",
    "off_topic_llm_candidates",
    "out_of_scope_detail_candidates",
    "adjacent_domain_candidates",
    "attach_dry_runs",
    "build_probe_candidates_doc",
    "generate_probe_candidates",
]
