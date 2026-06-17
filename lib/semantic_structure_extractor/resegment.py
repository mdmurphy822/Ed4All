"""WS6b — collapse re-segmentation.

When the DART heading-parser failure mode collapses a whole textbook into a
SINGLE chapter carrying many sections (the 141:1 collapse signature), this
module recovers coherent *pseudo-chapters* via contiguity-constrained Ward
clustering over section embeddings, BEFORE ``textbook_structure.json`` is
written.

The algorithm was de-risked on the real 141-section OpenStax algebra
structure (2026-06-17): contiguous Ward at K≈11 recovered clean
pseudo-chapters (whole numbers / variables / integers / fractions / decimals
/ roots / properties) that map almost 1:1 to OpenStax Ch.1's real structure.

Design contract:

* **Trigger only on the collapse signature** — ``len(chapters)==1`` AND that
  chapter carries more than ``_STRUCTURE_COLLAPSE_SECTION_THRESHOLD`` (40)
  sections. Otherwise pass-through unchanged (multi-chapter corpora are never
  touched).
* **Contiguity** — the chain-graph connectivity constraint forces each Ward
  cluster to be a contiguous run of sections (document order preserved).
* **Rides existing order_fallback** — each pseudo-chapter carries
  ``chapter_text`` (the join of its sections' ``section_text``) and NO
  ``source_file``, so ``lib/objectives/chunk_window.py::chunks_for_chapter``
  takes the proportional order_fallback path with no custom remapping.
* **Graceful** — sklearn/scipy/embeddings unavailable, flag off, or not
  collapsed → return ``chapters`` unchanged. Never crash extraction.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .semantic_structure_extractor import (
    _STRUCTURE_COLLAPSE_SECTION_THRESHOLD,
    _is_noncontent_heading,
)

logger = logging.getLogger(__name__)

# Gate flag — default ON. When off, pass-through unchanged.
_FLAG_RESEGMENT = "ED4ALL_RESEGMENT_COLLAPSED"
# Target sections-per-pseudo-chapter (drives K). 141/13 ≈ 11.
_FLAG_SECTIONS_PER_CHAPTER = "ED4ALL_RESEGMENT_SECTIONS_PER_CHAPTER"
_DEFAULT_SECTIONS_PER_CHAPTER = 13

# K is clamped into this range.
_K_MIN = 6
_K_MAX = 20

# Section embedding text is ``headingText + ". " + section_text`` truncated to
# this many characters (keeps the embed call cheap + focused on the lead).
_EMBED_TRUNCATE_CHARS = 600


def _flag_on(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _resolve_sections_per_chapter() -> int:
    raw = os.environ.get(_FLAG_SECTIONS_PER_CHAPTER)
    if raw is None or raw.strip() == "":
        return _DEFAULT_SECTIONS_PER_CHAPTER
    try:
        val = int(float(raw.strip()))
    except (TypeError, ValueError):
        return _DEFAULT_SECTIONS_PER_CHAPTER
    if val < 1:
        return _DEFAULT_SECTIONS_PER_CHAPTER
    return val


def _section_embed_text(section: Dict[str, Any]) -> str:
    """``headingText + ". " + section_text`` truncated to ~600 chars."""
    heading = str(section.get("headingText") or "").strip()
    body = str(section.get("section_text") or "").strip()
    if heading and body:
        combined = f"{heading}. {body}"
    else:
        combined = heading or body
    return combined[:_EMBED_TRUNCATE_CHARS]


def _section_title(section: Dict[str, Any]) -> str:
    return str(section.get("headingText") or "").strip()


# Pedagogical-scaffolding headings that are real content but make poor CHAPTER
# titles (they recur in every section). _is_noncontent_heading does NOT catch
# these (it targets answer-key / front-matter / donor noise), so the title
# derivation skips them separately to land on the segment's actual topic.
_SCAFFOLDING_TITLE_PREFIXES = (
    "self check",
    "solution",
    "example",
    "try it",
    "learning objective",
    "how to",
    "exercises",
    "practice",
)


def _is_scaffolding_title(title: str) -> bool:
    t = title.strip().lower()
    return any(t == p or t.startswith(p) for p in _SCAFFOLDING_TITLE_PREFIXES)


def _segment_title(sections: List[Dict[str, Any]], index: int) -> str:
    """First topic heading in the slice, else ``Unit {index}``.

    A heading is a usable chapter title when it's non-empty, passes
    ``_is_noncontent_heading`` (NOT answer-key / front-matter / donor noise),
    AND is not a recurring pedagogical-scaffolding heading (Self Check,
    Solution, Example, …) that would make a meaningless chapter name.
    Falls back to the first content-bearing-but-scaffolding heading before
    ``Unit {index}`` so a slice that is ALL scaffolding still gets a heading.
    """
    scaffolding_fallback = ""
    for sec in sections:
        title = _section_title(sec)
        if not title or _is_noncontent_heading(title):
            continue
        if _is_scaffolding_title(title):
            if not scaffolding_fallback:
                scaffolding_fallback = title
            continue
        return title
    return scaffolding_fallback or f"Unit {index}"


def _segment_chapter_text(sections: List[Dict[str, Any]]) -> str:
    """Join the slice's ``section_text``. REQUIRED — drives order_fallback."""
    parts = [
        str(sec.get("section_text") or "").strip()
        for sec in sections
    ]
    return "\n\n".join(p for p in parts if p).strip()


def _contiguous_runs(labels: List[int]) -> Optional[List[List[int]]]:
    """Split ``labels`` into contiguous index runs, one per label change.

    Returns a list of index-lists (each a contiguous run) IF every label is a
    single contiguous block (no label re-appears after a different one).
    Returns ``None`` when any cluster is non-contiguous (sklearn ever returns
    an interleaved labelling despite the chain constraint) so the caller can
    fall back.
    """
    if not labels:
        return []
    runs: List[List[int]] = []
    seen_labels: set = set()
    current_label = labels[0]
    current_run: List[int] = [0]
    for idx in range(1, len(labels)):
        lbl = labels[idx]
        if lbl == current_label:
            current_run.append(idx)
            continue
        # Boundary: close the current run.
        if current_label in seen_labels:
            return None  # label reappeared → non-contiguous
        seen_labels.add(current_label)
        runs.append(current_run)
        current_label = lbl
        current_run = [idx]
    if current_label in seen_labels:
        return None
    runs.append(current_run)
    return runs


def _fallback_runs_by_cosine_drops(
    norm_vecs: "Any", k: int
) -> List[List[int]]:
    """Split into ``k`` contiguous runs at the K-1 largest adjacent-cosine
    drops. Defensive fallback if sklearn ever returns a non-contiguous
    labelling under the chain constraint."""
    import numpy as np

    n = norm_vecs.shape[0]
    if n <= k:
        return [[i] for i in range(n)]
    # adjacent cosine (vectors already L2-normalized → dot product).
    adj_cos = np.sum(norm_vecs[:-1] * norm_vecs[1:], axis=1)
    # The K-1 lowest-cosine adjacencies are the segment boundaries.
    n_cuts = k - 1
    cut_after = sorted(
        int(i) for i in np.argsort(adj_cos)[:n_cuts]
    )
    runs: List[List[int]] = []
    start = 0
    for boundary in cut_after:
        runs.append(list(range(start, boundary + 1)))
        start = boundary + 1
    runs.append(list(range(start, n)))
    return runs


def resegment_collapsed_structure(
    chapters: List[Dict[str, Any]],
    *,
    embed_client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Recover pseudo-chapters from a collapsed single mega-chapter.

    Args:
        chapters: the merged chapter dicts (``to_dict()`` shape — camelCase
            ``headingText`` / ``section_text``).
        embed_client: optional pre-built embedding client (an object exposing
            ``encode_batch(List[str]) -> np.ndarray`` of L2-normalized rows).
            When ``None`` a ``build_embedding_client(offline=True)`` is built
            lazily. Injected by tests with deterministic vectors.

    Returns:
        The K pseudo-chapters when re-segmentation fires; ``chapters``
        unchanged on any non-trigger / graceful-skip path.
    """
    # --- Gate flag -----------------------------------------------------
    if not _flag_on(_FLAG_RESEGMENT, default=True):
        return chapters

    # --- Trigger only on the collapse signature -----------------------
    if not chapters or len(chapters) != 1:
        return chapters
    mega = chapters[0]
    if not isinstance(mega, dict):
        return chapters
    sections = mega.get("sections") or []
    n = len(sections)
    if n <= _STRUCTURE_COLLAPSE_SECTION_THRESHOLD:
        return chapters

    # --- Graceful lazy import of the numeric stack --------------------
    try:
        import numpy as np  # noqa: F401
        import scipy.sparse  # noqa: F401
        from sklearn.cluster import AgglomerativeClustering  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        logger.warning(
            "resegment_collapsed_structure: sklearn/scipy unavailable (%s); "
            "passing the collapsed %d-section chapter through unchanged.",
            exc,
            n,
        )
        return chapters

    # --- Embed each section -------------------------------------------
    try:
        if embed_client is None:
            from lib.embedding.providers import build_embedding_client

            embed_client = build_embedding_client(offline=True)
        embed_texts = [_section_embed_text(sec) for sec in sections]
        vecs = embed_client.encode_batch(embed_texts)
        vecs = np.asarray(vecs, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - never crash extraction
        logger.warning(
            "resegment_collapsed_structure: embedding the %d collapsed "
            "sections failed (%s); passing through unchanged.",
            n,
            exc,
        )
        return chapters

    if vecs.ndim != 2 or vecs.shape[0] != n:
        logger.warning(
            "resegment_collapsed_structure: unexpected embedding shape %s "
            "for %d sections; passing through unchanged.",
            getattr(vecs, "shape", None),
            n,
        )
        return chapters

    # L2-normalize (defensive — the client already normalizes, but injected
    # test vectors may not).
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    norm_vecs = vecs / norms

    # --- K = clamp(round(n / SEC_PER_CHAPTER), 6, 20) -----------------
    sec_per_chapter = _resolve_sections_per_chapter()
    k = max(_K_MIN, min(_K_MAX, round(n / sec_per_chapter)))
    k = min(k, n)  # never more clusters than sections.

    # --- Contiguity-constrained Ward ----------------------------------
    runs: Optional[List[List[int]]] = None
    try:
        # CHAIN = path graph: each section connected to its doc-order
        # neighbors via the +/-1 off-diagonals.
        ones = np.ones(n - 1)
        chain = scipy.sparse.diags(
            [ones, ones], [-1, 1], shape=(n, n)
        )
        model = AgglomerativeClustering(
            n_clusters=k, linkage="ward", connectivity=chain
        )
        labels = model.fit_predict(norm_vecs).tolist()
        runs = _contiguous_runs(labels)
    except Exception as exc:  # noqa: BLE001 - never crash extraction
        logger.warning(
            "resegment_collapsed_structure: Ward clustering failed (%s); "
            "falling back to cosine-drop segmentation.",
            exc,
        )
        runs = None

    if runs is None or len(runs) != k:
        # Connectivity constraint should yield contiguous runs; if sklearn
        # ever returns a non-contiguous labelling (or run count drift), fall
        # back to splitting at the K-1 largest adjacent-cosine drops.
        logger.warning(
            "resegment_collapsed_structure: Ward produced non-contiguous / "
            "off-count segments for %d sections; using cosine-drop fallback "
            "for K=%d.",
            n,
            k,
        )
        runs = _fallback_runs_by_cosine_drops(norm_vecs, k)

    # --- Build K pseudo-chapters --------------------------------------
    pseudo_chapters: List[Dict[str, Any]] = []
    for i, run in enumerate(runs, start=1):
        slice_sections = [sections[idx] for idx in run]
        chapter_text = _segment_chapter_text(slice_sections)
        title = _segment_title(slice_sections, i)
        pseudo = {
            # Preserve every original chapter-level field except the ones we
            # re-mint / drop (id, sections, chapter_text, source_file, title).
            "id": f"ch{i}",
            "headingLevel": mega.get("headingLevel", 2),
            "headingText": title,
            "headingId": None,
            "explicitObjectives": [],
            "contentBlocks": [],
            "sections": slice_sections,
            "chapter_text": chapter_text,
        }
        # NO source_file → chunks_for_chapter takes order_fallback.
        pseudo_chapters.append(pseudo)

    logger.warning(
        "resegment_collapsed_structure: recovered %d contiguous "
        "pseudo-chapters from a collapsed %d-section chapter (method="
        "contiguous_ward, K=%d).",
        len(pseudo_chapters),
        n,
        k,
    )
    return pseudo_chapters
