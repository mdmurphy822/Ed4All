"""Layered gold-set decontamination for the SFT pair corpus (SFT data program S3).

Freeze/load the course's retrieval-eval ``gold_set.json`` BEFORE pair
generation, then screen every emitted training pair against the gold
questions in FOUR escalating layers:

  1. **exact-match**   — the normalized pair text equals a gold question.
  2. **sliding 8-gram**— any gold-question token 8-gram appears verbatim in
     the pair text (catches copied questions / near-verbatim reuse).
  3. **embedding top-k** — cosine >= floor against the nearest gold question
     (OPTIONAL — injected ``embedder`` seam; skipped offline when ``None``).
  4. **paraphrase hook** — a pluggable callable; OFF by default offline (a
     future fetch-window / LLM-judge arm plugs in here).

A pair that trips ANY layer is DROPPED + quarantined with the tripped-layer
reason; every survivor is stamped ``decontam_checked=True``.  N-gram alone
misses MetaMath-style rephrases (2311.04850), so the embedding + paraphrase
layers exist — but they are *injectable seams* so the default path stays
fully offline (``HF_HUB_OFFLINE=1``, no network, no model load).

This is the pair-side complement of the edge-based ``HoldoutBuilder`` (which
never touches pairs): it is the gold-set leakage gate the training corpus
otherwise lacks.  Wire it as a pre-train gate over the final
``instruction_pairs`` / ``preference_pairs`` buffers.

Deterministic, read-only w.r.t. the gold set — it NEVER mutates
``gold_set.json`` (the frozen set is the canonical eval pin).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Share the retrieval-layer normalization + shingle helpers so the decontam
# gate measures containment identically to the gold-set loader + citation
# anchor (one normalization contract across the retrieval + training surfaces).
from lib.retrieval._text import normalize_ws, shingles  # noqa: E402

DEFAULT_NGRAM_SIZE = 8
# Cosine floor for the (optional) embedding layer. Deliberately high — the
# embedding arm only exists to catch semantic rephrases the 8-gram misses, not
# to drop merely-topical overlap (which is expected: pairs and gold questions
# are drawn from the same course).
DEFAULT_EMBED_FLOOR = 0.92

# Reasons stamped on a quarantined pair (surfaced for the audit trail).
REASON_EXACT = "exact_match"
REASON_NGRAM = "sliding_8gram"
REASON_EMBED = "embedding_topk"
REASON_PARAPHRASE = "paraphrase_check"


# --------------------------------------------------------------------------- #
# Gold-question extraction
# --------------------------------------------------------------------------- #

def gold_question_texts(gold_doc: Any) -> List[str]:
    """Extract the ``question_text`` of every question in a gold-set doc.

    Tolerant of shape: accepts the canonical ``{"questions": [...]}`` doc, a
    bare list of question dicts, or a list of strings. Blank / non-string
    texts are dropped. Order preserved for deterministic downstream hashing.
    """
    out: List[str] = []
    if isinstance(gold_doc, dict):
        questions = gold_doc.get("questions")
    elif isinstance(gold_doc, list):
        questions = gold_doc
    else:
        questions = None
    for q in questions or []:
        if isinstance(q, str):
            t = q.strip()
        elif isinstance(q, dict):
            t = str(q.get("question_text") or q.get("question") or "").strip()
        else:
            t = ""
        if t:
            out.append(t)
    return out


def load_gold_questions(course_dir: Any) -> Tuple[List[str], List[Any]]:
    """Freeze/load ``<course_dir>/retrieval_eval/gold_set.json`` BEFORE pair-gen.

    Returns ``(question_texts, issues)``. On a missing / unparseable gold set
    the texts list is empty and the loader's issue records explain why — the
    caller decides whether an absent gold set is a no-op (default) or a
    hard pre-train stop. Uses ``verify=False`` so no chunkset I/O is needed
    (the decontam pass only reads question TEXT, never the pinned chunks).
    """
    try:
        from lib.retrieval.gold_set import load_gold_set
    except Exception as exc:  # noqa: BLE001 — import guard, keep offline-safe
        logger.warning("pair_decontamination: gold_set loader import failed: %s", exc)
        return [], []
    gold_doc, issues = load_gold_set(course_dir, verify=False)
    return gold_question_texts(gold_doc), issues


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #

def _norm(text: Any) -> str:
    return normalize_ws(str(text or "")).lower()


def _tokens(text: str) -> List[str]:
    return text.split()


def _pair_fields(pair: Dict[str, Any]) -> List[str]:
    """The individual text fields of a pair (prompt / completion / preference
    sides). Layer-1 exact-match compares these field-by-field so a pair whose
    PROMPT equals a gold question is caught even though its completion isn't."""
    return [
        str(pair.get("prompt") or ""),
        str(pair.get("completion") or ""),
        str(pair.get("chosen") or ""),
        str(pair.get("rejected") or ""),
    ]


def _pair_text(pair: Dict[str, Any]) -> str:
    """Default pair-text projection: prompt + completion (+ preference sides)."""
    return " ".join(f for f in _pair_fields(pair) if f)


def _cosine(a: Any, b: Any) -> float:
    """Cosine similarity of two vectors (lists or numpy arrays).

    Returns 0.0 on a zero-norm vector or a shape mismatch (never raises — the
    embedding layer is best-effort).
    """
    try:
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            fx = float(x)
            fy = float(y)
            dot += fx * fy
            na += fx * fx
            nb += fy * fy
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))
    except (TypeError, ValueError):
        return 0.0


def _gold_ngram_set(gold_norm_texts: Sequence[str], ngram_size: int) -> Set[Tuple[str, ...]]:
    grams: Set[Tuple[str, ...]] = set()
    for t in gold_norm_texts:
        toks = _tokens(t)
        if not toks:
            continue
        for g in shingles(toks, ngram_size):
            grams.add(g)
    return grams


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def decontaminate_pairs(
    pairs: Sequence[Dict[str, Any]],
    gold_questions: Sequence[str],
    *,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    embedder: Optional[Any] = None,
    embed_floor: float = DEFAULT_EMBED_FLOOR,
    paraphrase_check: Optional[Callable[[str, Sequence[str]], Optional[str]]] = None,
    text_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    capture: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the layered gold-set decontamination pass over ``pairs``.

    Args:
        pairs: The emitted training pairs to screen (instruction and/or
            preference). Each surviving pair is mutated in place to stamp
            ``decontam_checked=True``.
        gold_questions: The frozen gold-question TEXTS
            (``load_gold_questions`` output). Empty -> no drops (the pass
            still runs and stamps survivors).
        ngram_size: Sliding-shingle size for layer 2 (default 8).
        embedder: OPTIONAL object exposing ``encode(text) -> vector`` (a
            ``lib.embedding.SentenceEmbedder`` or an injected test double).
            ``None`` (default offline) skips layer 3 entirely.
        embed_floor: Cosine floor for a layer-3 drop.
        paraphrase_check: OPTIONAL ``(pair_text, gold_texts) -> reason|None``
            hook for layer 4. ``None`` (default) skips it.
        text_fn: OPTIONAL ``pair -> str`` projection (defaults to
            prompt+completion+chosen+rejected).
        capture: OPTIONAL ``DecisionCapture`` — one ``synthesis_leakage_check``
            event is logged summarising the pass (dropped count + per-layer
            reasons) when supplied.

    Returns:
        ``(survivors, quarantined)``. Survivors carry ``decontam_checked=True``;
        each quarantined pair carries ``_decontam_reason`` (the tripped layer)
        and ``_decontam_gold_hint`` (a short snippet of the matched gold text).
    """
    tf = text_fn or _pair_text
    gold_norm = [_norm(g) for g in gold_questions if str(g).strip()]
    gold_norm_set = set(gold_norm)
    gold_ngrams = _gold_ngram_set(gold_norm, ngram_size) if gold_norm else set()

    # Layer-3 setup: embed the gold questions once (best-effort).
    gold_vecs: List[Any] = []
    if embedder is not None and gold_norm:
        for g in gold_questions:
            try:
                gold_vecs.append(embedder.encode(str(g)))
            except Exception as exc:  # noqa: BLE001 — best-effort embed
                logger.warning("pair_decontamination: gold embed failed: %s", exc)
                gold_vecs = []
                break

    survivors: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}

    def _match_gold_hint(pair_norm: str) -> str:
        for g in gold_norm:
            if g and (g in pair_norm or pair_norm in g):
                return g[:60]
        return gold_norm[0][:60] if gold_norm else ""

    for pair in pairs:
        text = tf(pair)
        pair_norm = _norm(text)
        pair_toks = _tokens(pair_norm)
        reason: Optional[str] = None
        gold_hint = ""

        # Layer 1 — exact match: any INDIVIDUAL pair field (prompt / completion
        # / preference side) normalized-equals a gold question.
        if gold_norm_set:
            for fld in _pair_fields(pair):
                fn = _norm(fld)
                if fn and fn in gold_norm_set:
                    reason = REASON_EXACT
                    gold_hint = fn[:60]
                    break

        # Layer 2 — sliding n-gram overlap.
        if reason is None and gold_ngrams and pair_toks:
            for g in shingles(pair_toks, ngram_size):
                if g in gold_ngrams:
                    reason = REASON_NGRAM
                    gold_hint = " ".join(g)[:60]
                    break

        # Layer 3 — embedding top-k (optional, injected seam).
        if reason is None and gold_vecs:
            try:
                pv = embedder.encode(text)
                best = 0.0
                best_idx = 0
                for i, gv in enumerate(gold_vecs):
                    c = _cosine(pv, gv)
                    if c > best:
                        best = c
                        best_idx = i
                if best >= embed_floor:
                    reason = REASON_EMBED
                    gold_hint = _norm(gold_questions[best_idx])[:60]
            except Exception as exc:  # noqa: BLE001 — best-effort embed
                logger.warning("pair_decontamination: pair embed failed: %s", exc)

        # Layer 4 — paraphrase hook (optional, off by default offline).
        if reason is None and paraphrase_check is not None:
            try:
                pr = paraphrase_check(text, gold_questions)
            except Exception as exc:  # noqa: BLE001 — best-effort hook
                logger.warning("pair_decontamination: paraphrase hook failed: %s", exc)
                pr = None
            if pr:
                reason = REASON_PARAPHRASE
                gold_hint = str(pr)[:60]

        if reason is not None:
            pair["_decontam_reason"] = reason
            pair["_decontam_gold_hint"] = gold_hint
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            quarantined.append(pair)
        else:
            pair["decontam_checked"] = True
            survivors.append(pair)

    if capture is not None:
        try:
            capture.log_decision(
                decision_type="synthesis_leakage_check",
                decision=(
                    f"gold-set decontamination: kept {len(survivors)}, "
                    f"quarantined {len(quarantined)} of {len(pairs)} pairs"
                ),
                rationale=(
                    f"Layered gold-set decontam over {len(gold_norm)} frozen gold "
                    f"questions (ngram={ngram_size}, embed_floor={embed_floor}, "
                    f"embedder={'on' if gold_vecs else 'off'}, "
                    f"paraphrase_hook={'on' if paraphrase_check else 'off'}). "
                    f"Per-layer drops: {reason_counts or 'none'}. Survivors "
                    f"stamped decontam_checked=true; hits quarantined with reason."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — capture is best-effort
            logger.warning("pair_decontamination: capture log failed: %s", exc)

    logger.info(
        "pair_decontamination: %d kept / %d quarantined (%s) over %d gold questions",
        len(survivors), len(quarantined), reason_counts or "no drops", len(gold_norm),
    )
    return survivors, quarantined


__all__ = [
    "DEFAULT_NGRAM_SIZE",
    "DEFAULT_EMBED_FLOOR",
    "REASON_EXACT",
    "REASON_NGRAM",
    "REASON_EMBED",
    "REASON_PARAPHRASE",
    "gold_question_texts",
    "load_gold_questions",
    "decontaminate_pairs",
]
