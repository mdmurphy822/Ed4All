"""Pass-2 verifier digest builder + spot-HTML slicer (SEMANTIK_SECOND_PASS).

Pure, GPU-free, deterministic helpers that turn an ``AssembledDoc`` (+ the
post-cap region list) into the structured digest the Pass-2 verifier judges,
and slice a flagged region's (or its enclosing section's) REAL assembled HTML
so the verifier can confirm a section it is about to FAIL against true bytes.

UNCONSUMED in Phases 1-2 (Phase 4 wires the dispatch) — so ``assemble_document``
is byte-identical. Everything here is a pure read: no LLM, no env side effect,
no region/HTML mutation.

Round-trip contract (load-bearing): the digest is keyed by ``region_index`` =
the VALUE stored in ``AssembledDoc.region_provenance`` (the ``capped`` index
space), NOT the digest-list position — so a future non-identity emission still
resolves ``regions[entry['region_index']]``. ``sub_task_log['region_html']`` is
index-aligned with ``region_provenance`` by POSITION (both built in emission
order in ``pass_9a``); ``anchors`` is heading_id -> region_index ``int``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .gold_shell_markup import _CONTAINER_SPECS
from .types import AssembledDoc

# Balance-scan token for the level-agnostic ``<section aria-labelledby>`` span
# (the re-level-immune heading/section slice path). Matches a section OPEN tag
# start or a section CLOSE tag.
_SECTION_TOKEN_RE = re.compile(r"<section\b|</section\s*>", re.IGNORECASE)

# First opening tag of a fragment (the aria_wrapper fallback for a
# semantic_class with no ``_CONTAINER_SPECS`` recipe — e.g. figure/section
# passthrough).
_OPENING_TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)\b")


def _region_html_list(assembled: AssembledDoc) -> list[str]:
    """Defensive read of the per-region fragment stash.

    ``sub_task_log['region_html']`` is annotated ``dict[str, str]`` at
    ``types.py`` but the runtime VALUE is a ``list[str]`` index-aligned with
    ``region_provenance`` (the de-facto shape; a future stringify cleanup must
    not silently break this read). Returns ``[]`` on any breach.
    """
    log = getattr(assembled, "sub_task_log", None) or {}
    rh = log.get("region_html")
    if isinstance(rh, list):
        return rh
    return []


def _position_for_region_index(
    region_provenance: Sequence[int], region_index: int
) -> int | None:
    """First POSITION ``p`` where ``region_provenance[p] == region_index``.

    ``region_html`` is position-aligned with ``region_provenance``, so the
    fragment for a provenance VALUE is found by its position (identity in v1,
    but this survives a future non-identity emission). ``None`` when absent.
    """
    for p, v in enumerate(region_provenance):
        if v == region_index:
            return p
    return None


def _opening_tag(fragment: str | None) -> str | None:
    """The first opening-tag name of ``fragment`` (e.g. ``<figure ...>`` -> ``figure``)."""
    if not fragment:
        return None
    m = _OPENING_TAG_RE.search(fragment)
    return m.group(1).lower() if m else None


def _aria_wrapper_for(semantic_class: str | None, fragment: str | None) -> str | None:
    """The ARIA wrapper element for a region.

    From ``_CONTAINER_SPECS[semantic_class].tag`` when the class has a gold
    container recipe; else (a passthrough class such as ``figure``/``section``)
    the opening tag of the region's HTML fragment. ``None`` when the region
    carries no ``semantic_class`` (a bare ``<p>`` fall-through).
    """
    if not semantic_class:
        return None
    spec = _CONTAINER_SPECS.get(semantic_class)
    if spec is not None:
        return spec.tag
    return _opening_tag(fragment)


def _head_tail(source: str, edge_tokens: int) -> tuple[str, str]:
    """Terse head/tail token edges of ``source`` (mirrors ``build_edge_input``)."""
    tokens = source.split()
    n = int(edge_tokens) if edge_tokens and int(edge_tokens) > 0 else 1
    head = " ".join(tokens[:n])
    tail = " ".join(tokens[-n:])
    return head, tail


def build_verifier_digest(
    assembled: AssembledDoc,
    regions: Sequence[Any],
    feature_blocks: Sequence[Any],
    *,
    edge_tokens: int,
) -> dict[str, Any]:
    """Distill the assembled doc into the structured verifier digest.

    One terse entry per EMITTED region in emission order (iterating
    ``region_provenance``), plus a document ``heading_outline``. Mirrors the
    SHAPE of ``cascade._build_region_provenance`` (pure, document-order,
    optional-key-additive distillation). Consumed by nobody in Phase 1.

    Each region entry: ``{region_index, kind, semantic_class, aria_wrapper,
    head, tail}`` where ``region_index`` is the ``region_provenance`` VALUE
    (NOT the digest-list position). Empty ``metadata_drop`` fragments (a region
    whose emitted HTML is ``''``) are SKIPPED — they are not phantom sections.
    """
    from ..qwen_specialists.reviewer import _joined_source_text  # lazy: break cycle

    region_provenance = list(getattr(assembled, "region_provenance", []) or [])
    region_html = _region_html_list(assembled)
    n_regions = len(regions)

    entries: list[dict[str, Any]] = []
    for pos, region_index in enumerate(region_provenance):
        if not (0 <= region_index < n_regions):
            continue
        frag = region_html[pos] if pos < len(region_html) else ""
        # SKIP an empty-fragment region (metadata_drop) — not a phantom section.
        if not frag:
            continue
        region = regions[region_index]
        payload = getattr(region, "payload", {}) or {}
        kind = getattr(region, "kind", "paragraph")
        semantic_class = payload.get("semantic_class")
        source = _joined_source_text(region, feature_blocks)
        head, tail = _head_tail(source, edge_tokens)
        entries.append(
            {
                "region_index": region_index,
                "kind": kind,
                "semantic_class": semantic_class,
                "aria_wrapper": _aria_wrapper_for(semantic_class, frag),
                "head": head,
                "tail": tail,
            }
        )

    # Heading outline — pair the final-html-accurate ``heading_tree`` (level,
    # text) with the heading region indices in heading document order
    # (``anchors`` is insertion-ordered by heading doc order in pass_9a).
    heading_tree = list(getattr(assembled, "heading_tree", []) or [])
    anchors = dict(getattr(assembled, "anchors", {}) or {})
    heading_outline: list[dict[str, Any]] = []
    for (level, text), (heading_id, h_region_index) in zip(heading_tree, anchors.items()):
        heading_outline.append(
            {
                "region_index": h_region_index,
                "level": level,
                "text": text,
                "heading_id": heading_id,
            }
        )

    return {"regions": entries, "heading_outline": heading_outline}


# ---------------------------------------------------------------------------
# Section-aligned digest windowing (Pass-2 verifier, this session).
#
# The committed ``run_second_pass_verify`` built ONE digest over the WHOLE
# assembled doc (a 2190-region OpenStax run distilled to ~1830 entries /
# ~112k tokens) and dispatched a SINGLE verifier call — but a 16k Ollama
# serving window head-truncates ~86% of that prompt, so the verifier judges a
# fragment and returns a non-committal pass (a seeded EXAMPLE-as-heading defect
# went uncaught). ``window_verifier_digest`` packs the digest's per-region
# entries into SECTION-ALIGNED windows that each fit the serving window, so the
# orchestrator can dispatch one bounded call per window and aggregate.
#
# Pure + GPU-free + deterministic — no LLM, no env read, no mutation.
# ---------------------------------------------------------------------------

# Coarse char->token divisor (mirrors the ``//4`` heuristic the answer-path
# token budgets use). The estimate only needs to be MONOTONE in size — it gates
# packing, not a hard serving-window assertion (the per-window prompt's own
# truncation tripwire is the real guard).
_TOKENS_PER_CHAR_DIVISOR = 4
# Rough token cost of the appended truncation-marker node (keeps the capped
# outline strictly under ``outline_token_cap``).
_OUTLINE_MARKER_TOKENS = 8


def _estimate_tokens(payload: Any) -> int:
    """Coarse token estimate of ``payload`` = ``len(compact-json) // 4``.

    Deterministic, dependency-free, and MONOTONE in serialized size — exactly
    the signal the section packer needs to bound a window. NOT a tokenizer.
    """
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(payload)
    return len(text) // _TOKENS_PER_CHAR_DIVISOR


def _split_into_sections(entries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Partition emitted-region entries into SECTIONS (heading + its content).

    A section = a ``kind == 'heading'`` entry plus the following content entries
    up to (not including) the next heading entry. Entries BEFORE the first
    heading form an initial front-matter pseudo-section. A heading-led run never
    yields an empty section (a heading with no following content is a size-1
    section); a doc that opens with a heading produces no spurious empty
    front-matter section.
    """
    sections: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for entry in entries:
        is_heading = isinstance(entry, dict) and entry.get("kind") == "heading"
        if is_heading:
            if current:
                sections.append(current)
            current = [entry]
        else:
            current.append(entry)
    if current:
        sections.append(current)
    return sections


def _subsplit_section(
    section: list[dict[str, Any]], window_tokens: int
) -> list[list[dict[str, Any]]]:
    """Slice an OVER-BUDGET single section into entry-range sub-windows.

    Greedy left-to-right: start a new slice when adding the next entry would
    exceed ``window_tokens`` (a lone entry larger than the budget still goes in
    its own slice — never dropped). Used only when one section alone exceeds the
    window budget (the design's over-budget fallback).
    """
    slices: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for entry in section:
        et = _estimate_tokens(entry)
        if current and current_tokens + et > window_tokens:
            slices.append(current)
            current, current_tokens = [], 0
        current.append(entry)
        current_tokens += et
    if current:
        slices.append(current)
    return slices


def _pack_sections(
    sections: list[list[dict[str, Any]]], window_tokens: int
) -> list[list[dict[str, Any]]]:
    """Pack consecutive WHOLE sections into windows under ``window_tokens``.

    A section is never split across windows UNLESS it alone exceeds the budget,
    in which case it is sub-split via :func:`_subsplit_section` (no entry is
    ever dropped). Returns a list of windows, each a flat list of entries.
    """
    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for section in sections:
        sec_tokens = _estimate_tokens(section)
        if sec_tokens > window_tokens:
            # Over-budget single section: flush, then entry-range sub-split it.
            if current:
                windows.append(current)
                current, current_tokens = [], 0
            windows.extend(_subsplit_section(section, window_tokens))
            continue
        if current and current_tokens + sec_tokens > window_tokens:
            windows.append(current)
            current, current_tokens = [], 0
        current.extend(section)
        current_tokens += sec_tokens
    if current:
        windows.append(current)
    return windows


def _cap_outline(
    full_outline: list[dict[str, Any]],
    window_ids: set[int],
    outline_token_cap: int,
) -> list[dict[str, Any]]:
    """Return the shared full-doc outline, capped at ``outline_token_cap``.

    Under budget → the full outline verbatim (so the single-window case stays
    byte-equivalent to the input digest). Over budget → keep the WINDOW-LOCAL
    headings (those whose ``region_index`` is in this window) + a truncation
    marker node recording how many were omitted, trimming further if the local
    set still overflows.
    """
    if _estimate_tokens(full_outline) <= outline_token_cap:
        return full_outline
    local = [
        node
        for node in full_outline
        if isinstance(node, dict) and node.get("region_index") in window_ids
    ]
    kept = list(local)
    while kept and _estimate_tokens(kept) + _OUTLINE_MARKER_TOKENS > outline_token_cap:
        kept.pop()
    omitted = len(full_outline) - len(kept)
    kept.append({"outline_truncated": True, "omitted_headings": omitted})
    return kept


def window_verifier_digest(
    digest: dict[str, Any],
    *,
    window_tokens: int,
    outline_token_cap: int = 3000,
) -> list[dict[str, Any]]:
    """Pack a verifier digest into SECTION-ALIGNED windows that fit the serving
    window.

    Each returned window is a sub-digest of the SAME shape
    :func:`reviewer_prompt.build_second_pass_verify_request` consumes —
    ``{"regions": [<this window's entries>], "heading_outline": <terse full-doc
    outline, capped at outline_token_cap>, "allowed_region_indices": [<the
    region_index VALUEs in this window>]}``. The full-doc outline is shared
    CONTEXT in every window (heading text/level/region_index — never per-region
    detail) so a verifier judging one window still knows the whole document's
    section structure; ``allowed_region_indices`` pins which ids THIS window may
    flag (it may only flag what it can see).

    Packing: consecutive WHOLE sections (a heading + its content) are packed
    until the next section would exceed ``window_tokens``; a section is NEVER
    split across windows unless it alone exceeds the budget, in which case it is
    entry-range sub-split (no entry dropped). Token budget is estimated over the
    ENTRIES (``len(compact-json)//4``); the shared outline is budgeted
    separately at ``outline_token_cap``.

    1-window invariant: when every entry fits one window AND the shared outline
    is under ``outline_token_cap`` (the small-doc case), the single returned
    window carries the FULL ``regions`` + the FULL ``heading_outline`` verbatim
    (``_pack_sections`` preserves document order; ``_cap_outline`` is a no-op
    under budget), so ``build_second_pass_verify_request(window)`` is byte-
    identical to ``build_second_pass_verify_request(digest)`` (the builder
    ignores ``allowed_region_indices`` in serialization and emits the allow-list
    only when the shared outline references regions OUTSIDE the window). Empty
    digest → ``[]``.
    """
    regions = list(digest.get("regions", []) or [])
    full_outline = list(digest.get("heading_outline", []) or [])
    if not regions:
        return []

    packed = _pack_sections(_split_into_sections(regions), window_tokens)

    windows: list[dict[str, Any]] = []
    for win_entries in packed:
        win_ids = {
            e["region_index"]
            for e in win_entries
            if isinstance(e, dict) and e.get("region_index") is not None
        }
        windows.append(
            {
                "regions": win_entries,
                "heading_outline": _cap_outline(full_outline, win_ids, outline_token_cap),
                "allowed_region_indices": sorted(win_ids),
            }
        )
    return windows


# ---------------------------------------------------------------------------
# Spot-HTML slicer (Phase 2) — pure, unconsumed.
# ---------------------------------------------------------------------------


def slice_section_html(assembled: AssembledDoc, heading_id: str) -> str:
    """Cut a section's REAL assembled HTML from its ``<section aria-labelledby>``.

    PRIMARY heading/section confirm path: locate the
    ``<section aria-labelledby="{heading_id}">`` open and balance-scan nested
    ``<section>`` opens / ``</section>`` closes to its matching close. The id
    is level-AGNOSTIC, so this is IMMUNE to ``normalize_document_heading_levels``
    (the post-9a ``<hN>`` re-level a ``region_html`` substring search would miss
    on exactly the heading-mistype sections the verifier most wants to confirm).
    ``''`` when the section is not found.
    """
    if not heading_id:
        return ""
    html = getattr(assembled, "html", "") or ""
    needle = f'<section aria-labelledby="{heading_id}">'
    start = html.find(needle)
    if start == -1:
        safe = heading_id.replace('"', "&quot;")
        needle = f'<section aria-labelledby="{safe}">'
        start = html.find(needle)
        if start == -1:
            return ""
    depth = 0
    end: int | None = None
    for m in _SECTION_TOKEN_RE.finditer(html, start):
        if m.group(0).lower().startswith("</section"):
            depth -= 1
            if depth == 0:
                end = m.end()
                break
        else:
            depth += 1
    if end is None:
        return html[start:]
    return html[start:end]


def enclosing_section_heading_id(
    assembled: AssembledDoc, region_index: int
) -> str | None:
    """The ``heading_id`` of the nearest PRECEDING heading region.

    Scans ``region_provenance`` document order back from ``region_index`` for
    the nearest heading region (a region that owns an ``anchors`` id) at or
    before it, returning that heading's id. ``None`` when none precedes (a
    front-matter / pre-first-heading body region). Maps a flagged CONTENT region
    back to the heading that owns its section (the per-SECTION confirm
    granularity).
    """
    region_provenance = list(getattr(assembled, "region_provenance", []) or [])
    anchors = dict(getattr(assembled, "anchors", {}) or {})
    # Reverse: region_index -> heading_id (anchors is heading_id -> region_index).
    rev: dict[int, str] = {ri: hid for hid, ri in anchors.items()}
    pos = _position_for_region_index(region_provenance, region_index)
    if pos is None:
        return None
    for p in range(pos, -1, -1):
        v = region_provenance[p]
        if v in rev:
            return rev[v]
    return None


def slice_region_html(assembled: AssembledDoc, region_index: int) -> str:
    """The REAL assembled HTML for ONE non-heading region (fast path).

    Returns the region's ``region_html`` fragment for a non-gap, non-heading
    region (a verbatim contiguous SUBSTRING of the final HTML). For a gap kind
    (title/author/citation/legal — ``region_html`` is PRE-9c-splice) the
    fragment is located in the FINAL ``assembled.html`` via ``str.find`` so the
    slice is the post-splice bytes, not the stale stash. ``''`` (never raises)
    for an empty/``metadata_drop`` region or an unresolvable index. For a heading
    confirm, prefer :func:`slice_section_html` (re-level-immune).
    """
    region_provenance = list(getattr(assembled, "region_provenance", []) or [])
    region_html = _region_html_list(assembled)
    pos = _position_for_region_index(region_provenance, region_index)
    if pos is None or pos >= len(region_html):
        return ""
    frag = region_html[pos]
    if not frag:
        return ""
    html = getattr(assembled, "html", "") or ""
    if frag in html:
        # Confirmed present in the final doc — the located slice IS the fragment.
        return frag
    # Gap kind: the persisted stash is PRE-9c-splice. Locate in the final HTML.
    idx = html.find(frag)
    if idx >= 0:
        return html[idx : idx + len(frag)]
    return ""


__all__ = [
    "build_verifier_digest",
    "window_verifier_digest",
    "slice_section_html",
    "slice_region_html",
    "enclosing_section_heading_id",
]
