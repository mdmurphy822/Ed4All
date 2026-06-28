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
    "slice_section_html",
    "slice_region_html",
    "enclosing_section_heading_id",
]
