"""Deterministic "FAQ" page builder (feature W6.4).

A fully deterministic post-pass that, per terminal objective (week), authors a
``week_NN_faq.html`` page of grounded Frequently-Asked-Question cards — one card
per question, each carrying a short answer and (when resolvable) a deep-link back
to the source chunk that grounds the answer. NO 7B free authoring runs here:
every answer is lifted from / supported by a real source chunk or a real
misconception correction (anti-fabrication — a question whose answer resolves to
NEITHER a supporting chunk NOR a real correction is OMITTED, never invented).

The builder is intentionally GPU-free and side-effect-free so it can run as a
deterministic post-pass over an existing course export with no model server. It
mirrors the ``lib/generation/key_terms.py`` precedent (deterministic page
builder + a ``vocab_card``-reusing pre-rendered block whose ``template_type``
short-circuits the rewrite tier).

Question seeds (anti-fabrication — never invents a topic):

  1. MISCONCEPTIONS — each misconception block yields a "Is it true that …?"
     question whose answer is the block's real CORRECTION, kept only when a
     source chunk supports the correction (grounding gate).
  2. OBJECTIVES — each objective's key concepts yield a "What is …?" question
     whose answer is a definitional sentence LIFTED from a source chunk that
     mentions the concept (grounding is inherent — the chunk IS the answer).

Gated by ``ED4ALL_FAQ_PAGE`` (default OFF); when off ``build_faq_blocks``
returns ``[]`` so the emit stays byte-identical.
"""

from __future__ import annotations

import html as _html_mod
import os as _os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.ontology.slugs import canonical_slug

# Reuse the FROZEN source-link + heading-slug helpers so a FAQ deep-link
# round-trips through the learner source viewer byte-identically to a key-terms
# deep-link (same ``/api/learn/source/...`` shape, same ``heading_slug``).
from lib.generation.key_terms import (
    heading_slug,  # noqa: F401  (re-exported for parity / tests)
    resolve_source_link,
)

# The page-type string registered by the outline-phase call site (Lane A).
FAQ_PAGE_TYPE = "faq"

# The deterministic-block marker stamped onto ``Block.template_type`` so the
# rewrite tier recognises a pre-authored, grounded FAQ card and SKIPS the LLM
# dispatch (the answer is already grounded + verbatim). Mirrors
# ``lib.generation.key_terms.KEY_TERMS_TEMPLATE_TYPE``.
FAQ_TEMPLATE_TYPE = "faq"

# ED4ALL_FAQ_PAGE — the opt-in gate. Default OFF => no faq page, byte-identical.
FAQ_PAGE_ENV = "ED4ALL_FAQ_PAGE"
_FAQ_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Per-page cap on the number of FAQ cards (a real FAQ is a short list, not an
# exhaustive dump). Env-overridable; garbage / non-positive => default.
FAQ_MAX_PER_PAGE_ENV = "ED4ALL_FAQ_MAX_PER_PAGE"
FAQ_DEFAULT_MAX_PER_PAGE = 12

# Minimum answer length (chars) — a one-word fragment is not a usable answer.
_MIN_ANSWER_CHARS = 12

# Sentence splitter (deterministic, no NLP model) — mirrors key_terms.py.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Leaky-sentence rejection (exercise / heading / answer-key runs) — a copy of
# the key_terms.py P4 leakage discipline so a FAQ answer is a real definition,
# never an exercise stub.
_LEAK_RE = re.compile(
    r"\bTRY\s+IT\b|\bEXAMPLE\b|\bSolution\b|\bChapter\s+\d|::",
    re.IGNORECASE,
)
_LEAK_LEADING_RE = re.compile(r"^\s*(?:[\d+\-*/=.,]|[ⓐ-ⓩ⒜-⒵])")

# Content-token stopwords for the answer↔chunk grounding overlap test.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
        "were", "be", "been", "being", "to", "of", "in", "on", "at", "for",
        "with", "as", "by", "that", "this", "these", "those", "it", "its",
        "not", "no", "yes", "you", "your", "we", "they", "he", "she", "will",
        "can", "may", "do", "does", "did", "so", "than", "when", "which",
        "what", "how", "why", "who", "from", "into", "about", "there",
    }
)


def resolve_faq_page(override: Optional[bool] = None) -> bool:
    """Resolve the ED4ALL_FAQ_PAGE gate (explicit arg > env > default OFF).

    Parse-with-fallback: an explicit bool wins; otherwise the env var is read
    each call (so tests can toggle it). Falsey / garbage => off.
    """
    if isinstance(override, bool):
        return override
    return _os.environ.get(FAQ_PAGE_ENV, "").strip().lower() in _FAQ_TRUTHY


def resolve_max_faqs_per_page(override: Optional[int] = None) -> int:
    """Resolve the per-page FAQ cap (explicit arg > env > default 12)."""
    if isinstance(override, int) and override > 0:
        return override
    raw = _os.environ.get(FAQ_MAX_PER_PAGE_ENV, "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return FAQ_DEFAULT_MAX_PER_PAGE


def _strip_tags(text: str) -> str:
    return _TAG_STRIP_RE.sub(" ", str(text))


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", str(text).lower()).strip()


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", _strip_tags(text)).strip()


def _is_leaky(sentence: str) -> bool:
    s = str(sentence)
    return bool(_LEAK_RE.search(s) or _LEAK_LEADING_RE.match(s))


def _content_tokens(text: str) -> set:
    """Lowercased alnum content tokens (>=4 chars, minus stopwords)."""
    out = set()
    for tok in re.findall(r"[a-z0-9]+", _norm(text)):
        if len(tok) >= 4 and tok not in _STOPWORDS:
            out.add(tok)
    return out


def _chunk_text(chunk: Dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return ""
    return _clean(str(chunk.get("text") or chunk.get("content") or ""))


def _ground_answer(
    answer: str,
    *,
    chunks: Sequence[Dict[str, Any]],
    min_overlap: int = 2,
) -> Optional[Dict[str, Any]]:
    """Return the source chunk that best SUPPORTS ``answer``, or ``None``.

    Deterministic content-token overlap: a chunk supports the answer when it
    shares at least ``min_overlap`` content tokens with the answer (or covers
    >= 60% of the answer's content tokens for a short answer). The
    highest-overlap chunk wins; ties break on the chunk's ``id`` for
    determinism. ``None`` => ungrounded (anti-fabrication OMISSION).
    """
    ans_tokens = _content_tokens(answer)
    if not ans_tokens:
        return None
    best: Optional[Dict[str, Any]] = None
    best_score = 0
    best_id = ""
    for chunk in chunks or ():
        if not isinstance(chunk, dict):
            continue
        body = _chunk_text(chunk)
        if not body:
            continue
        shared = ans_tokens & _content_tokens(body)
        score = len(shared)
        threshold = min(
            min_overlap, max(1, int(round(0.60 * len(ans_tokens))))
        )
        if score < threshold:
            continue
        cid = str(chunk.get("id") or "")
        if score > best_score or (score == best_score and cid < best_id):
            best, best_score, best_id = chunk, score, cid
    return best


# --- misconception seeds ---------------------------------------------------

def _misconception_fields(block: Any) -> Tuple[str, str]:
    """Extract ``(misconception, correction)`` from a misconception block/dict.

    Handles a ``Block`` whose ``content`` is a dict, a bare dict, and the
    ``mc_*`` field surface. Returns ``("", "")`` when neither resolves.
    """
    mis = ""
    cor = ""
    content = getattr(block, "content", None)
    if content is None and isinstance(block, dict):
        content = block
    if isinstance(content, dict):
        mis = str(
            content.get("misconception") or content.get("claim") or ""
        )
        cor = str(
            content.get("correction") or content.get("reconcile") or ""
        )
    # mc_* attribute surface (misconception_rich) as a fallback.
    if not mis:
        mis = str(getattr(block, "mc_named_concept", "") or "")
    if not cor:
        cor = str(getattr(block, "mc_reconcile", "") or "")
    return _clean(mis), _clean(cor)


def _seed_from_misconceptions(
    misconception_blocks: Sequence[Any],
    *,
    chunks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in misconception_blocks or ():
        mis, cor = _misconception_fields(block)
        if not mis or len(cor) < _MIN_ANSWER_CHARS:
            continue
        # Grounding gate — the correction must be supported by a real chunk.
        chunk = _ground_answer(cor, chunks=chunks)
        if chunk is None:
            continue  # anti-fabrication OMISSION.
        question = mis.rstrip("?.! ")
        if not question:
            continue
        question = f"Is it true that {question[:1].lower()}{question[1:]}?"
        out.append(
            {
                "question": question,
                "answer": cor,
                "source_chunk": chunk,
                "slug": canonical_slug(mis)[:48] or canonical_slug(cor)[:48],
                "seed": "misconception",
            }
        )
    return out


# --- objective / key-concept seeds -----------------------------------------

def _objective_concepts(objective: Dict[str, Any]) -> List[str]:
    concepts: List[str] = []
    for key in ("keyConcepts", "key_concepts"):
        vals = objective.get(key) if isinstance(objective, dict) else None
        if isinstance(vals, (list, tuple)):
            for v in vals:
                if isinstance(v, str) and v.strip():
                    concepts.append(v.strip())
    return concepts


def _definition_for_concept(
    concept: str,
    *,
    chunks: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Lift a definitional sentence for ``concept`` from a source chunk.

    Rejects leaky exercise/heading sentences (mirrors key_terms.py). Returns
    ``{"answer", "chunk"}`` or ``None`` (OMIT — never invents).
    """
    surface = _norm(concept)
    if not surface:
        return None
    for chunk in chunks or ():
        if not isinstance(chunk, dict):
            continue
        body = _chunk_text(chunk)
        if not body:
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(body):
            s_norm = _norm(sentence)
            if surface not in s_norm:
                continue
            if _is_leaky(sentence):
                continue
            answer = _WS_RE.sub(" ", sentence).strip()
            if len(answer) >= _MIN_ANSWER_CHARS:
                return {"answer": answer, "chunk": chunk}
    return None


def _seed_from_objectives(
    week_objectives: Sequence[Dict[str, Any]],
    *,
    chunks: Sequence[Dict[str, Any]],
    seen_slugs: set,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for objective in week_objectives or ():
        if not isinstance(objective, dict):
            continue
        obj_id = str(objective.get("id") or "")
        for concept in _objective_concepts(objective):
            slug = canonical_slug(concept)
            if not slug or slug in seen_slugs:
                continue
            resolved = _definition_for_concept(concept, chunks=chunks)
            if resolved is None:
                continue  # anti-fabrication OMISSION.
            seen_slugs.add(slug)
            display = concept.strip()
            question = f"What is {display}?"
            out.append(
                {
                    "question": question,
                    "answer": resolved["answer"],
                    "source_chunk": resolved["chunk"],
                    "slug": slug[:48],
                    "objective_id": obj_id,
                    "seed": "objective",
                }
            )
    return out


def build_faq_entries(
    week_objectives: Sequence[Dict[str, Any]],
    misconception_blocks: Sequence[Any],
    chunks: Sequence[Dict[str, Any]],
    *,
    course_slug: str = "",
    max_faqs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build grounded FAQ entry dicts (pure, deterministic, testable).

    Returns a list of ``{question, answer, source_link?, slug, seed}`` dicts.
    Misconception-seeded questions come first (they are the distinctive FAQ
    entry), then objective/key-concept questions; deduped on slug; capped.
    """
    mc_entries = _seed_from_misconceptions(misconception_blocks, chunks=chunks)
    seen: set = set()
    for e in mc_entries:
        if e["slug"]:
            seen.add(e["slug"])
    obj_entries = _seed_from_objectives(
        week_objectives, chunks=chunks, seen_slugs=seen
    )
    combined = mc_entries + obj_entries

    cap = resolve_max_faqs_per_page(max_faqs)
    combined = combined[:cap]

    out: List[Dict[str, Any]] = []
    for e in combined:
        link = resolve_source_link(
            e.get("source_chunk"), course_slug=course_slug
        )
        entry: Dict[str, Any] = {
            "question": e["question"],
            "answer": e["answer"],
            "slug": e["slug"],
            "seed": e["seed"],
            "source_chunk": e.get("source_chunk"),
        }
        if link:
            entry["source_link"] = link
        if e.get("objective_id"):
            entry["objective_id"] = e["objective_id"]
        out.append(entry)
    return out


def render_faq_card(
    *,
    question: str,
    answer: str,
    source_link: Optional[str] = None,
) -> str:
    """Render ONE FAQ card as deterministic HTML (single source of the markup).

    Shape mirrors the key-terms vocab card: a ``data-cf-content-type="faq"``
    card carrying the question (``<p class="faq-question">``), the answer
    (``<p class="faq-answer">``), and (when resolvable) a source deep-link.
    """
    q_html = _html_mod.escape(str(question))
    a_html = _html_mod.escape(str(answer))
    link_html = ""
    if isinstance(source_link, str) and source_link:
        link_html = (
            '\n      <p class="faq-source">'
            f'<a href="{_html_mod.escape(source_link)}">View source</a></p>'
        )
    return (
        '<div class="callout faq-card" data-cf-content-type="faq">\n'
        f'      <p class="faq-question"><strong>{q_html}</strong></p>\n'
        f'      <p class="faq-answer">{a_html}</p>{link_html}\n'
        '    </div>'
    )


def build_faq_blocks(
    week_objectives: Sequence[Dict[str, Any]],
    misconception_blocks: Sequence[Any],
    chunks: Sequence[Dict[str, Any]],
    *,
    enabled: Optional[bool] = None,
    course_slug: str = "",
    page_id: str = "",
    max_faqs: Optional[int] = None,
) -> List[Any]:
    """Build a week's grounded FAQ ``Block`` list (the SHARED-CONTRACT entry).

    Deterministic + anti-fabrication: seeds questions from misconception blocks
    + objectives, grounds every answer to a real chunk (OMITs the ungrounded),
    and returns ``Block(block_type="vocab_card")`` cards carrying pre-rendered
    grounded HTML + ``template_type="faq"`` so the rewrite tier short-circuits.

    Gated by ``ED4ALL_FAQ_PAGE`` (default OFF): when off (or ``enabled=False``)
    returns ``[]`` so emit is byte-identical.
    """
    if not resolve_faq_page(enabled):
        return []

    entries = build_faq_entries(
        week_objectives,
        misconception_blocks,
        chunks,
        course_slug=course_slug,
        max_faqs=max_faqs,
    )
    if not entries:
        return []

    # Lazy import (mirrors the pipeline_tools / block_planner pattern) so this
    # library module stays import-light and cycle-safe.
    from Courseforge.scripts.blocks import Block  # noqa: PLC0415

    bound_page_id = page_id or "faq"
    blocks: List[Any] = []
    for idx, e in enumerate(entries):
        content = render_faq_card(
            question=e["question"],
            answer=e["answer"],
            source_link=e.get("source_link"),
        )
        # Resolve the grounding chunk's real SemantiK source id (when present) so the
        # block carries grounding the source-ref gate can resolve.
        source_ids: Tuple[str, ...] = ()
        chunk = e.get("source_chunk")
        if isinstance(chunk, dict):
            cid = chunk.get("id")
            if isinstance(cid, str) and cid and course_slug:
                source_ids = (f"semantik:{course_slug}#{cid}",)
        objective_ids: Tuple[str, ...] = ()
        if e.get("objective_id"):
            objective_ids = (str(e["objective_id"]),)
        slug = str(e.get("slug") or f"q{idx}")
        try:
            # Block identity: a FAQ card REUSES the ``vocab_card`` wrapper
            # (``block_type`` must be a canonical ``BLOCK_TYPES`` member; minting
            # a dedicated ``faq_card`` enum would have to learn ~a dozen
            # enum/schema/validator/framework-map surfaces incl. the hard-pinned
            # ``len(BLOCK_TYPES) == 30`` guards). The DISTINCT identity is the
            # ``template_type=FAQ_TEMPLATE_TYPE`` marker — it (a) survives into
            # the emitted blocks JSONL (so analytics can tell FAQ cards apart
            # from key-terms glossary cards, both of which are ``vocab_card``),
            # and (b) excludes the card from the glossary definition-quality
            # audit (``lib/validators/key_terms_definition_quality`` keys on this
            # marker). Do NOT drop the marker without updating that consumer.
            blk = Block(
                block_id=Block.stable_id(
                    bound_page_id, "vocab_card", slug, idx
                ),
                block_type="vocab_card",
                page_id=bound_page_id,
                sequence=idx,
                content=content,
                template_type=FAQ_TEMPLATE_TYPE,
                objective_ids=objective_ids,
                source_ids=source_ids,
                target_bloom="understand",
            )
        except (TypeError, ValueError):
            continue
        blocks.append(blk)
    return blocks
