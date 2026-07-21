"""W2 §1 — chapter → chunk-window partitioning sized to fit a 7B serving window.

The textbook-synthesis MAP unit drops from CHAPTER granularity (24 KB chapter
prose → overflows a 4-8 K 7B window → silent head-truncation → fabrication) to a
**chunk-window**: an ordered group of SemantiK chunks whose combined rendered
prompt fits the resolved serving window. The objective the model emits MUST cite
``source_chunk_ids`` that resolve against the chunkset, so feeding the model the
chunks themselves (id + body) makes the allowed-id set explicit and the citation
a copy, not an inference (the OUTLINE-tier discipline lifted onto LO synthesis).

Two public symbols:

- :func:`chunks_for_chapter` — the chapter → chunks join. Primary key is the
  SemantiK slug shared by the chapter's ``source_file`` basename and each chunk's
  ``source.source_references[].sourceId`` (shape ``semantik:{slug}#{block_id}``).
  Order-fallback (document order, proportional to ``chapter_text`` length) when
  no sourceId join resolves (legacy corpora). Records ``join_method``.
- :func:`group_chunks_into_windows` — greedy-pack chunks (chapter document order)
  into windows under the per-window token budget, reusing
  :mod:`lib.retrieval._prompts`'s ``estimate_tokens`` (the 2.5 chars/token
  divisor) — the windowing math does NOT re-derive a divisor. Overlap = 0
  (cross-window duplicate objectives are Pass D's explicit job; §1.4).

The helpers are PURE (no provider dep, no env reads beyond the caller-supplied
``num_ctx``) so the budget invariant is unit-testable with a tiny ``num_ctx``.
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

from lib.ontology.slugs import canonical_slug
from lib.retrieval._prompts import estimate_tokens

# ---------------------------------------------------------------------------
# Module constants (W2 §1.3 / §1.5)
# ---------------------------------------------------------------------------

#: Grammar/schema preamble token reserve. The synthesis grammar payload
#: (response_format / format / grammar) adds a fixed cost the model server
#: prepends; budget it so a window never assumes the whole window is prose.
GRAMMAR_RESERVE_TOKENS = 120

#: Constant headroom on top of every per-window fixed cost. The token estimate
#: is an upper bound but rough; this keeps a window comfortably inside the wire
#: window even if the server tokenizer is denser than the 2.5 chars/token model.
SAFETY_MARGIN_TOKENS = 128

#: Per-chunk body truncation (characters). Mirrors the outline tier's 1200-char
#: per-chunk cap, bumped to 1500 because LO synthesis wants more context per
#: chunk while staying bounded. A single chunk whose truncated body alone
#: exceeds the window budget is still emitted as its own 1-chunk window — the
#: per-chunk cap bounds it; the prompt head is NEVER truncated.
MAX_CHUNK_BODY_CHARS = 1500


# ---------------------------------------------------------------------------
# ChunkWindow
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkWindow:
    """An ordered group of SemantiK chunks sized to fit a 7B serving window.

    A window never crosses a chapter boundary (so CO-NN hierarchy +
    ``ChapterObjectiveCoverageValidator`` still hold) and its chunks are
    disjoint from every other window of the chapter (overlap = 0; §1.4).
    """

    chapter_id: str
    window_index: int  # 0-based within the chapter
    chunk_ids: List[str]  # the allowed-id set for this window
    chunks: List[Dict[str, Any]]  # id + truncated text bodies (render input)
    join_method: str  # "sourceid" | "order_fallback"
    estimated_prompt_tokens: int  # for the decision event + tripwire
    dropped_trailing: int = 0  # chunks dropped from this chapter that didn't fit
    #: Per-section window mode (``ED4ALL_OBJECTIVE_WINDOW_PER_SECTION``): the
    #: source section's heading text. Empty on every legacy (chapter-packed)
    #: window — and elided from ``to_dict`` when empty, so the flag-off dict
    #: shape stays byte-identical.
    section_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "chapter_id": self.chapter_id,
            "window_index": self.window_index,
            "chunk_ids": list(self.chunk_ids),
            "chunks": [dict(c) for c in self.chunks],
            "join_method": self.join_method,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "dropped_trailing": self.dropped_trailing,
        }
        # Elide when empty: legacy windows stay byte-identical.
        if self.section_label:
            out["section_label"] = self.section_label
        return out


# ---------------------------------------------------------------------------
# Slug / sourceId helpers
# ---------------------------------------------------------------------------
#: ``semantik:{slug}#{block_id}`` — capture the slug between the prefix and
#: ``#``. Dual-READ: accepts both the current ``semantik:`` prefix and the
#: legacy pre-SemantiK ``dart:`` prefix.
_SOURCE_ID_RE = re.compile(r"^(?:dart|semantik):(?P<slug>[^#]+)#")


def _basename_stem(path_str: str) -> str:
    """Return the filename stem of a path string (OS-agnostic).

    The chapter ``source_file`` may be a POSIX or Windows path; resolve the
    basename without the suffix using both parsers so a Windows-authored
    structure still joins on a POSIX runner.
    """
    if not path_str:
        return ""
    raw = str(path_str)
    # Pick whichever parser yields the shorter (more-stripped) name — a
    # Windows path on POSIX leaves backslashes that PureWindowsPath strips.
    posix_name = PurePosixPath(raw).name
    win_name = PureWindowsPath(raw).name
    name = win_name if len(win_name) <= len(posix_name) else posix_name
    # Drop a single trailing suffix (.html / .htm / .xhtml etc.).
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return stem


def _chapter_slug_forms(chapter: Dict[str, Any]) -> List[str]:
    """Candidate SemantiK slugs for a chapter, derived from ``source_file``.

    Spec §1.2 mandates ``canonical_slug`` on the basename. A ``lowercase +
    space-to-hyphen`` normalization (per the project slug contract) differs from
    ``canonical_slug`` (which DELETES disallowed chars and fuses digits). We
    compute BOTH forms and match a chunk whose ``sourceId`` slug equals either —
    robust across the two normalizations without guessing which the chunker
    emitted.
    """
    src = str(chapter.get("source_file") or "")
    stem = _basename_stem(src)
    if not stem:
        return []
    forms: List[str] = []
    canon = canonical_slug(stem)
    if canon:
        forms.append(canon)
    # Hyphen style: lowercase + non-alnum runs → single hyphen + strip hyphens.
    hyphen_slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    if hyphen_slug and hyphen_slug not in forms:
        forms.append(hyphen_slug)
    # Raw lowercased stem (some chunkers slug the stem verbatim).
    low = stem.lower()
    if low and low not in forms:
        forms.append(low)
    return forms


def _chunk_source_slugs(chunk: Dict[str, Any]) -> List[str]:
    """Extract the SemantiK slug(s) from a chunk's ``source.source_references``."""
    source = chunk.get("source")
    if not isinstance(source, dict):
        return []
    refs = source.get("source_references")
    if not isinstance(refs, list):
        return []
    slugs: List[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        sid = str(ref.get("sourceId") or "")
        m = _SOURCE_ID_RE.match(sid)
        if m:
            slug = m.group("slug")
            if slug and slug not in slugs:
                slugs.append(slug)
    return slugs


def _chunk_id(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "")


def _chunk_body(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("text") or chunk.get("body") or "")


# ---------------------------------------------------------------------------
# Chapter → chunks join (W2 §1.2)
# ---------------------------------------------------------------------------
def chunks_for_chapter(
    chapter: Dict[str, Any],
    all_chunks: List[Dict[str, Any]],
    *,
    all_chapters: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Return ``(chapter_chunks, join_method)`` for one chapter.

    Primary key: the SemantiK slug shared by the chapter's ``source_file``
    basename and each chunk's ``source.source_references[].sourceId``. A chunk whose
    sourceId slug equals any candidate slug form of the chapter joins to it.

    Fallback (``join_method="order_fallback"``): when NO sourceId join resolves
    for ANY chapter (legacy corpora with no source_file / sourceId linkage),
    partition the document-ordered chunk list into chapter buckets proportional
    to each chapter's ``chapter_text`` length. ``all_chapters`` (the full ordered
    chapter list) is required for the proportional split; absent it, the fallback
    assigns ALL chunks to a single-chapter call (degenerate but never crashes).

    Returns the matched chunks in document order (``all_chunks`` order preserved).
    A chunk matching NO chapter under the sourceid path is simply not returned
    for this chapter (the caller logs the global drop count).
    """
    chapter_slugs = set(_chapter_slug_forms(chapter))
    if chapter_slugs:
        matched = [
            c
            for c in all_chunks
            if chapter_slugs.intersection(_chunk_source_slugs(c))
        ]
        if matched:
            return matched, "sourceid"

    # --- Order-fallback. ---------------------------------------------------
    return _order_fallback_for_chapter(chapter, all_chunks, all_chapters), (
        "order_fallback"
    )


def _order_fallback_for_chapter(
    chapter: Dict[str, Any],
    all_chunks: List[Dict[str, Any]],
    all_chapters: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Proportional document-order partition of ``all_chunks`` to ``chapter``."""
    if not all_chunks:
        return []
    chapters = [
        c for c in (all_chapters or []) if isinstance(c, dict)
    ]
    if not chapters:
        # No chapter list to proportion against — degenerate single bucket.
        return list(all_chunks)

    # Find this chapter's index by identity (id), else fall back to object eq.
    cid = str(chapter.get("id") or "")
    chapter_index: Optional[int] = None
    for i, ch in enumerate(chapters):
        if str(ch.get("id") or "") == cid and cid:
            chapter_index = i
            break
    if chapter_index is None:
        try:
            chapter_index = chapters.index(chapter)
        except ValueError:
            return []

    # Compute proportional boundaries by chapter_text length.
    lengths = [
        max(1, len(str(ch.get("chapter_text") or ""))) for ch in chapters
    ]
    total_len = float(sum(lengths))
    n_chunks = len(all_chunks)
    # Cumulative boundary (start, end) in chunk-index space for each chapter.
    boundaries: List[Tuple[int, int]] = []
    cursor = 0
    acc = 0.0
    for i, length in enumerate(lengths):
        acc += length
        if i == len(lengths) - 1:
            end = n_chunks
        else:
            end = int(math.floor((acc / total_len) * n_chunks))
        end = max(cursor, min(end, n_chunks))
        boundaries.append((cursor, end))
        cursor = end
    start, end = boundaries[chapter_index]
    return list(all_chunks[start:end])


# ---------------------------------------------------------------------------
# Windowing / budget math (W2 §1.3)
# ---------------------------------------------------------------------------
def _render_chunk_line(chunk_id: str, body: str) -> str:
    """Render one chunk's prompt line (mirrors the window-prompt enumeration)."""
    return f"  - [{chunk_id}] {body}"


def _allowed_id_reserve_tokens(chunk_ids: List[str]) -> int:
    """Token cost of enumerating the allowed-id list in the prompt tail."""
    if not chunk_ids:
        return 0
    joined = ", ".join(chunk_ids)
    return estimate_tokens(joined)


# ---------------------------------------------------------------------------
# Defect C — exercise-apparatus seed sanitation (flag-gated window render prep)
# ---------------------------------------------------------------------------
#: ``ED4ALL_OBJECTIVE_SEED_SANITIZE`` (default OFF). When on, the RENDERED
#: window body fed to the 7B synthesizer is stripped of exercise-/practice-
#: apparatus lines & sentences so a non-instructional banner ("In the following
#: exercises…") never SEEDS an objective. chunk_ids / citability / the chunkset
#: itself are UNTOUCHED — only the prompt render text changes. Because the window
#: chunk bodies feed ``_stage2_window_fingerprint``, flipping the flag flips the
#: fingerprint, so a stale per-window resume sidecar correctly invalidates.
#: Default OFF ⇒ byte-identical windows (proved by the ``to_dict()`` regression).
_SEED_SANITIZE_ENV = "ED4ALL_OBJECTIVE_SEED_SANITIZE"
_SEED_SANITIZE_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Render placeholder for a wholly-apparatus chunk (heading kept, body replaced).
APPARATUS_SENTINEL = "[practice exercises — not instructional prose]"

#: Sentence splitter (mirrors the lightweight boundary heuristic used elsewhere).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def resolve_seed_sanitize(value: Optional[bool] = None) -> bool:
    """Resolve the ``ED4ALL_OBJECTIVE_SEED_SANITIZE`` gate (default OFF).

    Explicit arg > env. Truthy tokens ``1``/``true``/``yes``/``on`` (any case)
    enable; falsey / garbage / unset → off (parse-with-fallback, mirroring the
    stage-2 flag resolvers). Read at window-build time so flipping the flag
    between runs re-derives the render (and thus the fingerprint).
    """
    if value is not None:
        return bool(value)
    return (
        os.environ.get(_SEED_SANITIZE_ENV, "").strip().lower()
        in _SEED_SANITIZE_TRUTHY
    )


def _chunk_heading(chunk: Dict[str, Any]) -> str:
    """Best-effort display heading for a chunk (for the wholly-apparatus render).

    Prefers a top-level ``heading`` / ``section_heading`` / ``title``; falls back
    to ``source.section_heading`` (the chunk-v4 relayed heading). Empty when none.
    """
    if not isinstance(chunk, dict):
        return ""
    for key in ("heading", "section_heading", "title"):
        val = chunk.get(key)
        if val:
            return str(val).strip()
    source = chunk.get("source")
    if isinstance(source, dict):
        val = source.get("section_heading")
        if val:
            return str(val).strip()
    return ""


def _wholly_apparatus_render(chunk: Dict[str, Any]) -> str:
    """Render a wholly-apparatus chunk as ``heading + sentinel`` (or bare sentinel)."""
    heading = _chunk_heading(chunk)
    if heading:
        return f"{heading}\n{APPARATUS_SENTINEL}"
    return APPARATUS_SENTINEL


def _strip_apparatus_from_body(text: str, profile: Any) -> str:
    """Drop apparatus lines / sentences from ``text``, preserving surrounding prose.

    Line-level first (a line that is wholly an apparatus banner is dropped),
    then sentence-level within each surviving line (an apparatus sentence embedded
    in a prose line is dropped, the prose kept). Returns the stripped body
    (may be ``""`` when every line/sentence was apparatus — caller sentinels it).
    """
    kept_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        sentences = _SENTENCE_SPLIT_RE.split(stripped)
        kept = [
            s.strip()
            for s in sentences
            if s.strip() and not profile.text_has_marker(s)
        ]
        if kept:
            kept_lines.append(" ".join(kept))
    return "\n".join(kept_lines).strip()


def _sanitize_window_body(chunk: Dict[str, Any], body: str, profile: Any) -> str:
    """Flag-on sanitation of ONE chunk's rendered body (structural first, lexical second).

    Detection order mirrors ``citation_reselect._is_exercise_like_chunk``:

    * structural signal ``True`` (``composite_unit == "exercise_set"`` /
      ``unit_roles ∋ readiness_check/answer_key/practice``) → wholly apparatus →
      ``heading + sentinel``;
    * structural signal ``False`` (declared instructional prose — wins over
      apparatus) → body returned unchanged (trust the metadata, no lexical strip);
    * structural signal ``None`` (no pedagogical metadata) → lexical line/sentence
      strip; if nothing survives, the chunk was wholly apparatus → sentinel.
    """
    signal = profile.structural_signal(chunk)
    if signal is False:
        return body
    if signal is True:
        return _wholly_apparatus_render(chunk)
    # No pedagogical metadata → lexical. Fast path: no marker anywhere in the
    # body leaves it byte-identical (the vendor-control invariant — a marker-free
    # corpus is never re-flowed), so sentence surgery runs only on a real hit.
    if not profile.text_has_marker(body):
        return body
    remaining = _strip_apparatus_from_body(body, profile)
    if not remaining:
        return _wholly_apparatus_render(chunk)
    return remaining


def group_chunks_into_windows(
    chapter: Dict[str, Any],
    chunks_for_chapter_list: List[Dict[str, Any]],
    *,
    num_ctx: int,
    system_prompt: str,
    draft_block: str,
    max_tokens: int,
    join_method: str = "sourceid",
    response_reserve: Optional[int] = None,
    sanitize_seeds: Optional[bool] = None,
) -> List[ChunkWindow]:
    """Greedy-pack a chapter's chunks into windows under the ``num_ctx`` budget.

    Per-window budget (W2 §1.3)::

        available = num_ctx
                  - estimate_tokens(system_prompt)
                  - estimate_tokens(draft_block)
                  - GRAMMAR_RESERVE_TOKENS
                  - allowed_id_reserve (grows with chunk count)
                  - response_reserve (== max_tokens, the model's output budget)
                  - SAFETY_MARGIN_TOKENS

    Chunks are packed in chapter document order until the next chunk's rendered
    line would exceed ``available``; then the window closes and a new one opens
    (overlap = 0). A single chunk whose truncated body alone exceeds the window
    budget is emitted as its OWN 1-chunk window (never head-truncate the chunk;
    the per-chunk cap already bounds it). ``dropped_trailing`` counts chunks that
    fell off the chapter's last window only when the chapter already produced ≥1
    full window (rare).

    The ``allowed_id_reserve`` term grows with chunk count, so the budget is
    re-checked against the WINDOW's running id set on each add (the reserve for a
    window of N ids is bounded; we recompute it as the window grows).
    """
    chapter_id = str(chapter.get("id") or "")
    reserve = int(response_reserve if response_reserve is not None else max_tokens)
    fixed = (
        estimate_tokens(system_prompt)
        + estimate_tokens(draft_block)
        + GRAMMAR_RESERVE_TOKENS
        + max(0, reserve)
        + SAFETY_MARGIN_TOKENS
    )
    base_available = int(num_ctx) - fixed
    # Floor the window budget at a small positive value so a tiny num_ctx still
    # yields at least one chunk per window (single-chunk-window contract).
    base_available = max(1, base_available)

    # Defect-C apparatus-seed sanitation of the RENDERED body (flag-gated,
    # default OFF ⇒ this branch is inert and the windows are byte-identical).
    # The lexicon profile is the union of every profile (over-match is safe —
    # only render text is edited, never chunk ids / citability).
    _sanitize = resolve_seed_sanitize(sanitize_seeds)
    _profile = None
    if _sanitize:
        from lib.objectives.apparatus_lexicon import compile_profile
        _profile = compile_profile()

    # Pre-truncate per-chunk bodies + precompute rendered lines / token costs.
    prepared: List[Tuple[str, str, int]] = []  # (chunk_id, body, line_tokens)
    for chunk in chunks_for_chapter_list:
        cid = _chunk_id(chunk)
        if not cid:
            continue
        body = _chunk_body(chunk)
        if _sanitize and _profile is not None:
            # Strip apparatus BEFORE the per-chunk cap so a wholly-apparatus
            # chunk collapses to the short sentinel and frees window budget.
            body = _sanitize_window_body(chunk, body, _profile)
        if len(body) > MAX_CHUNK_BODY_CHARS:
            body = body[:MAX_CHUNK_BODY_CHARS]
        line_tokens = estimate_tokens(_render_chunk_line(cid, body))
        prepared.append((cid, body, line_tokens))

    windows: List[ChunkWindow] = []
    cur_ids: List[str] = []
    cur_bodies: List[Tuple[str, str]] = []
    cur_tokens = 0

    def _close_window() -> None:
        nonlocal cur_ids, cur_bodies, cur_tokens
        if not cur_ids:
            return
        # Recompute the precise estimated prompt tokens for this window:
        # fixed + chunk lines + the allowed-id enumeration for THIS id set.
        chunk_line_tokens = sum(
            estimate_tokens(_render_chunk_line(cid, body))
            for cid, body in cur_bodies
        )
        est = fixed + chunk_line_tokens + _allowed_id_reserve_tokens(cur_ids)
        windows.append(
            ChunkWindow(
                chapter_id=chapter_id,
                window_index=len(windows),
                chunk_ids=list(cur_ids),
                chunks=[
                    {"id": cid, "text": body} for cid, body in cur_bodies
                ],
                join_method=join_method,
                estimated_prompt_tokens=est,
                dropped_trailing=0,
            )
        )
        cur_ids = []
        cur_bodies = []
        cur_tokens = 0

    for cid, body, line_tokens in prepared:
        # The allowed-id reserve grows with each added id; recompute the budget
        # net of the id enumeration for the prospective window.
        prospective_ids = cur_ids + [cid]
        id_reserve = _allowed_id_reserve_tokens(prospective_ids)
        available = base_available - id_reserve
        available = max(1, available)
        if cur_ids and (cur_tokens + line_tokens) > available:
            # Close the current window; start a fresh one with this chunk.
            _close_window()
            cur_ids = [cid]
            cur_bodies = [(cid, body)]
            cur_tokens = line_tokens
        else:
            cur_ids.append(cid)
            cur_bodies.append((cid, body))
            cur_tokens += line_tokens
    _close_window()

    return windows


# ---------------------------------------------------------------------------
# Vendor-depth per-SECTION windows (ED4ALL_OBJECTIVE_WINDOW_PER_SECTION)
# ---------------------------------------------------------------------------
#: Master gate for one-window-per-SECTION stage-2 map units. Default OFF →
#: the legacy chapter-greedy packing above runs byte-identical. When on, the
#: stage-2 window builder partitions each chapter's chunks along the
#: ``textbook_structure.json`` chapter→section tree (ordered walk anchored on
#: each chunk's ``section_heading`` provenance), so a 10-chapter / 71-section
#: corpus synthesizes ~71 section-scoped windows instead of ~23 giant packed
#: windows — the measured mechanism gap behind 52 vs ~800 COs (the vendor-super
#: depth came from 794 accidental one-chunk windows; this is the intentional,
#: structure-true version of that per-unit fan-out).
_WINDOW_PER_SECTION_ENV = "ED4ALL_OBJECTIVE_WINDOW_PER_SECTION"
_WINDOW_PER_SECTION_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Coalescing floor: a section that attracted fewer than this many chunks is
#: merged into its preceding sibling (the first group merges forward) so no
#: window is starved of source material.
SECTION_MIN_CHUNKS = 3

#: Ordered-walk lookahead: a chunk may advance the section pointer by at most
#: this many sections at once (recovers a single missed boundary without
#: letting a spurious heading echo jump the walk across the chapter).
_SECTION_LOOKAHEAD = 2

#: Leading ``N.M`` section-ordinal pattern (e.g. ``1.2 Use the Language …``).
_SECTION_ORDINAL_RE = re.compile(r"^(\d+)\.(\d+)\b")

#: How much of a chunk's TEXT head is scanned for the section heading when the
#: ``section_heading`` provenance carries a subsection heading instead.
_SECTION_TEXT_HEAD_CHARS = 300


def resolve_window_per_section(value: Optional[bool] = None) -> bool:
    """Resolve the ``ED4ALL_OBJECTIVE_WINDOW_PER_SECTION`` gate (default OFF).

    Explicit arg > env. Truthy tokens ``1``/``true``/``yes``/``on`` (any case)
    enable; falsey / garbage / unset → off (parse-with-fallback, mirroring
    :func:`resolve_seed_sanitize`). Read at window-build time.
    """
    if value is not None:
        return bool(value)
    return (
        os.environ.get(_WINDOW_PER_SECTION_ENV, "").strip().lower()
        in _WINDOW_PER_SECTION_TRUTHY
    )


def _norm_heading_text(text: Any) -> str:
    """NFKC + casefold + whitespace-collapse a heading for comparison."""
    s = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", s).strip()


def _section_heading_text(section: Dict[str, Any]) -> str:
    """Best-effort display heading for a structure-tree section."""
    if not isinstance(section, dict):
        return ""
    for key in ("headingText", "heading", "title"):
        val = section.get(key)
        if val:
            return str(val).strip()
    return ""


def _chunk_matches_section(chunk: Dict[str, Any], section: Dict[str, Any]) -> bool:
    """Does ``chunk`` look like the START (membership) of ``section``?

    Three arms, cheapest first:

    1. normalized ``section_heading`` provenance equals / prefixes the section
       heading (either direction — converted HTML may truncate);
    2. both carry the same leading ``N.M`` ordinal (scan corpora keep the
       numbering even when the title text drifts);
    3. the section heading appears verbatim in the chunk's TEXT head (the
       heading opens the section's first chunk when provenance carries a
       subsection heading instead).
    """
    sec_head = _norm_heading_text(_section_heading_text(section))
    if not sec_head:
        return False
    chunk_head = _norm_heading_text(_chunk_heading(chunk))
    if chunk_head:
        if chunk_head == sec_head or chunk_head.startswith(sec_head):
            return True
        if len(chunk_head) > 4 and sec_head.startswith(chunk_head):
            return True
        m_sec = _SECTION_ORDINAL_RE.match(sec_head)
        m_chk = _SECTION_ORDINAL_RE.match(chunk_head)
        if m_sec and m_chk and m_sec.groups() == m_chk.groups():
            return True
    text_head = _norm_heading_text(
        _chunk_body(chunk)[:_SECTION_TEXT_HEAD_CHARS]
    )
    if text_head and sec_head in text_head:
        return True
    return False


def partition_chunks_by_section(
    chapter: Dict[str, Any],
    chapter_chunks: List[Dict[str, Any]],
    *,
    min_chunks: int = SECTION_MIN_CHUNKS,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Partition one chapter's DOCUMENT-ORDERED chunks along its section tree.

    Ordered walk: a pointer over ``chapter["sections"]`` advances when a chunk
    matches the NEXT (lookahead ≤ :data:`_SECTION_LOOKAHEAD`) section's heading
    — via ``section_heading`` provenance, the shared ``N.M`` ordinal, or the
    chunk text head. Chunks before the first matched boundary belong to the
    first section. A section whose boundary never matches simply attracts no
    chunks and disappears (its chunks stay with the preceding section — the
    same coalescing outcome).

    Coalescing floor: any group with fewer than ``min_chunks`` chunks merges
    into the preceding surviving group (the first group merges forward), so no
    starved window is ever emitted. Document order and the disjoint-cover
    invariant are preserved: the concatenation of all returned groups equals
    ``chapter_chunks`` exactly.

    Returns ``[(section_label, chunks), ...]``; degenerates to a single
    ``("", chapter_chunks)`` group when the chapter declares fewer than two
    sections (the caller falls back to legacy packing).
    """
    sections = [
        s for s in (chapter.get("sections") or []) if isinstance(s, dict)
    ]
    if len(sections) < 2 or not chapter_chunks:
        return [("", list(chapter_chunks))]

    labels = [_section_heading_text(s) for s in sections]
    assigned: List[List[Dict[str, Any]]] = [[] for _ in sections]
    cur = -1
    for chunk in chapter_chunks:
        nxt = cur + 1
        advance: Optional[int] = None
        for j in range(nxt, min(nxt + _SECTION_LOOKAHEAD, len(sections))):
            if _chunk_matches_section(chunk, sections[j]):
                advance = j
                break
        if advance is not None:
            cur = advance
        assigned[max(cur, 0)].append(chunk)

    # Drop empty groups (missed boundaries) keeping document order.
    groups: List[Tuple[str, List[Dict[str, Any]]]] = [
        (labels[i], assigned[i]) for i in range(len(sections)) if assigned[i]
    ]
    if not groups:
        return [("", list(chapter_chunks))]

    # Coalesce starved groups into the PRECEDING survivor (first merges
    # forward). Single pass left-to-right; label of the ABSORBING group wins.
    floor = max(1, int(min_chunks))
    coalesced: List[Tuple[str, List[Dict[str, Any]]]] = []
    for label, chunks in groups:
        if coalesced and len(chunks) < floor:
            prev_label, prev_chunks = coalesced[-1]
            coalesced[-1] = (prev_label, prev_chunks + chunks)
        else:
            coalesced.append((label, list(chunks)))
    if len(coalesced) > 1 and len(coalesced[0][1]) < floor:
        # First group was starved: merge it FORWARD into its next sibling.
        first_label, first_chunks = coalesced.pop(0)
        nxt_label, nxt_chunks = coalesced[0]
        coalesced[0] = (nxt_label or first_label, first_chunks + nxt_chunks)
    return coalesced


def group_chunks_into_section_windows(
    chapter: Dict[str, Any],
    chunks_for_chapter_list: List[Dict[str, Any]],
    *,
    num_ctx: int,
    system_prompt: str,
    draft_block: str,
    max_tokens: int,
    join_method: str = "sourceid",
    response_reserve: Optional[int] = None,
    sanitize_seeds: Optional[bool] = None,
    min_chunks: int = SECTION_MIN_CHUNKS,
) -> List[ChunkWindow]:
    """Per-SECTION analogue of :func:`group_chunks_into_windows`.

    Partitions the chapter's chunks along its section tree
    (:func:`partition_chunks_by_section`), then reuses the SAME greedy budget
    packer WITHIN each section group — so the serving-window invariant still
    holds (an oversized section splits into >1 window instead of overflowing),
    while a normal section yields exactly one window. ``window_index`` runs
    sequentially across the whole chapter (unique per chapter, the checkpoint
    key contract); each window is stamped with its ``section_label``.

    A chapter with fewer than two declared sections degrades to the legacy
    packer output byte-identically (single unlabelled group).
    """
    groups = partition_chunks_by_section(
        chapter, chunks_for_chapter_list, min_chunks=min_chunks
    )
    windows: List[ChunkWindow] = []
    for label, group_chunks in groups:
        packed = group_chunks_into_windows(
            chapter,
            group_chunks,
            num_ctx=num_ctx,
            system_prompt=system_prompt,
            draft_block=draft_block,
            max_tokens=max_tokens,
            join_method=join_method,
            response_reserve=response_reserve,
            sanitize_seeds=sanitize_seeds,
        )
        for w in packed:
            windows.append(
                replace(
                    w,
                    window_index=len(windows),
                    section_label=label,
                )
            )
    return windows
