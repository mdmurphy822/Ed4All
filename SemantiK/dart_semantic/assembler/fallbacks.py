"""Per-region-kind fallback emitters (Plans/04 §1, all 11 RegionKinds).

Used when Stage 7 dropped all K candidates for a region (the
``top_per_region[i] is None`` case). Output must conform to
docs/ontology.md §2 — every fallback emits a small, valid HTML5
fragment that will pass the per-region hard-gate's HTML5 well-formedness
check at minimum.

Coverage: every kind in
:data:`dart_semantic.structure_graph.REGION_KINDS` has an entry in
:data:`FALLBACKS`. ``emit_fallback`` is the public dispatch.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from html import escape
from typing import Any

from ..structure_graph import Region
from ..types import FeatureBlock


def _source_text(region: Region, feature_blocks: Sequence[FeatureBlock]) -> str:
    """Concatenate the raw text of every FeatureBlock the region claims."""
    parts: list[str] = []
    for i in region.feature_block_indices:
        if 0 <= i < len(feature_blocks):
            raw = getattr(feature_blocks[i], "raw", None)
            text = (getattr(raw, "text", None) or "") if raw is not None else ""
            t = text.strip()
            if t:
                parts.append(t)
    if parts:
        return " ".join(parts)
    # Fall back to payload-carried text (math regions in particular
    # only carry ``src_text`` in their payload).
    payload = region.payload or {}
    for key in ("text", "src_text"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ---------------------------------------------------------------------------
# Per-kind fallbacks
# ---------------------------------------------------------------------------


def fallback_paragraph(region: Region, feature_blocks: Sequence[FeatureBlock]) -> str:
    return f"<p>{escape(_source_text(region, feature_blocks))}</p>"


def fallback_heading(region: Region, feature_blocks: Sequence[FeatureBlock]) -> str:
    # Always emit h2; the heading tree normalizer will demote/promote as needed.
    return f"<h2>{escape(_source_text(region, feature_blocks))}</h2>"


def fallback_list(region: Region, feature_blocks: Sequence[FeatureBlock]) -> str:
    payload = region.payload or {}
    items = payload.get("items") or []
    ordered = bool(payload.get("ordered"))
    tag = "ol" if ordered else "ul"
    body_parts: list[str] = []
    if items:
        for item in items:
            text = ""
            if isinstance(item, dict):
                text = item.get("text", "") or ""
            else:
                text = str(item)
            body_parts.append(f"<li>{escape(text)}</li>")
    else:
        for fb_i in region.feature_block_indices:
            if 0 <= fb_i < len(feature_blocks):
                raw = getattr(feature_blocks[fb_i], "raw", None)
                t = (getattr(raw, "text", None) or "") if raw is not None else ""
                t = t.strip()
                if t:
                    body_parts.append(f"<li>{escape(t)}</li>")
    if not body_parts:
        # Lists with no items would fail axe ``list`` rule; emit a
        # placeholder so the gate doesn't blow up on an empty <ul></ul>.
        body_parts.append("<li></li>")
    return f"<{tag}>{''.join(body_parts)}</{tag}>"


def fallback_definition_list(
    region: Region, feature_blocks: Sequence[FeatureBlock],
) -> str:
    fbs = [
        feature_blocks[i]
        for i in region.feature_block_indices
        if 0 <= i < len(feature_blocks)
    ]
    if not fbs:
        return "<dl><dt></dt><dd></dd></dl>"
    raw0 = getattr(fbs[0], "raw", None)
    dt_text = (getattr(raw0, "text", None) or "") if raw0 is not None else ""
    dt = escape(dt_text.strip())
    dd_parts: list[str] = []
    for fb in fbs[1:]:
        raw = getattr(fb, "raw", None)
        t = (getattr(raw, "text", None) or "") if raw is not None else ""
        t = t.strip()
        if t:
            dd_parts.append(escape(t))
    dd = " ".join(dd_parts)
    return f"<dl><dt>{dt}</dt><dd>{dd}</dd></dl>"


def fallback_table(
    region: Region,
    feature_blocks: Sequence[FeatureBlock],
    *,
    cell_roles: list[list[str]] | None = None,
) -> str:
    """Emit ``<table>`` with caption / thead / tbody / per-cell ``<th scope=>``.

    When ``cell_roles`` (cell-role BERT output, 4-class:
    ``data | header_col | header_row | span``) is provided AND its shape
    matches ``cell_grid``, per-cell role drives header/data emission:

      * ``header_col`` → ``<th scope="col">``
      * ``header_row`` → ``<th scope="row">``
      * ``data`` / ``span`` / unknown → ``<td>``
        (``span`` flagged but emitted as ``<td>`` — rowspan/colspan can't be
        reconstructed without geometry the BERT doesn't see.)

    Header rows for the ``<thead>`` are those where ≥50% of cells are
    ``header_col``. When ``cell_roles`` is absent or shape-mismatched, falls
    back to the prior behavior (``header_row_indices`` payload key, default
    first row to header).
    """
    payload = region.payload or {}
    grid: list[list[Any]] = list(payload.get("cell_grid") or [])
    header_rows = set(payload.get("header_row_indices") or [])
    caption_idx = payload.get("caption_fb_index")
    parts: list[str] = ["<table>"]
    if caption_idx is not None and 0 <= caption_idx < len(feature_blocks):
        raw = getattr(feature_blocks[caption_idx], "raw", None)
        cap = (getattr(raw, "text", None) or "") if raw is not None else ""
        cap = cap.strip()
        if cap:
            parts.append(f"<caption>{escape(cap)}</caption>")
    if not grid:
        parts.append(
            '<thead><tr><th scope="col"></th></tr></thead>'
            "<tbody><tr><td></td></tr></tbody></table>"
        )
        return "".join(parts)

    # Use cell_roles only if shape matches the grid; otherwise drop to legacy.
    use_roles = (
        cell_roles is not None
        and len(cell_roles) == len(grid)
        and all(
            len(cell_roles[i] or []) == len(grid[i] or [])
            for i in range(len(grid))
        )
    )

    if use_roles:
        # Header row = row whose majority of cells are column-header-labeled.
        header_rows = {
            i for i, rr in enumerate(cell_roles)
            if rr and sum(1 for r in rr if r == "header_col") * 2 >= len(rr)
        }
    if not header_rows:
        header_rows = {0}

    def _cell_html(role: str | None, cell_text: str, *, in_thead: bool) -> str:
        s = escape(cell_text)
        if role == "header_row":
            return f'<th scope="row">{s}</th>'
        if role == "header_col":
            return f'<th scope="col">{s}</th>'
        if in_thead:
            # No explicit role in thead → default to column header.
            return f'<th scope="col">{s}</th>'
        return f"<td>{s}</td>"

    parts.append("<thead>")
    for i in sorted(header_rows):
        if i >= len(grid):
            continue
        row_cells = grid[i] or []
        row_roles = (cell_roles[i] if use_roles else [None] * len(row_cells))
        parts.append(
            "<tr>"
            + "".join(
                _cell_html(role, str(c or ""), in_thead=True)
                for c, role in zip(row_cells, row_roles)
            )
            + "</tr>"
        )
    parts.append("</thead><tbody>")
    body_emitted = False
    for i, row_cells in enumerate(grid):
        if i in header_rows:
            continue
        body_emitted = True
        row_cells = row_cells or []
        row_roles = (cell_roles[i] if use_roles else [None] * len(row_cells))
        parts.append(
            "<tr>"
            + "".join(
                _cell_html(role, str(c or ""), in_thead=False)
                for c, role in zip(row_cells, row_roles)
            )
            + "</tr>"
        )
    if not body_emitted:
        parts.append("<tr><td></td></tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def fallback_math(region: Region, feature_blocks: Sequence[FeatureBlock]) -> str:
    payload = region.payload or {}
    src = payload.get("src_text") or _source_text(region, feature_blocks)
    return f'<span class="math-source">{escape(src)}</span>'


def fallback_code_block(
    region: Region, feature_blocks: Sequence[FeatureBlock],
) -> str:
    return f"<pre><code>{escape(_source_text(region, feature_blocks))}</code></pre>"


def fallback_blockquote(
    region: Region, feature_blocks: Sequence[FeatureBlock],
) -> str:
    return f"<blockquote>{escape(_source_text(region, feature_blocks))}</blockquote>"


def fallback_figure(region: Region, feature_blocks: Sequence[FeatureBlock]) -> str:
    payload = region.payload or {}
    cap_idx = payload.get("caption_fb_index")
    cap = ""
    if cap_idx is not None and 0 <= cap_idx < len(feature_blocks):
        raw = getattr(feature_blocks[cap_idx], "raw", None)
        cap = (getattr(raw, "text", None) or "") if raw is not None else ""
        cap = cap.strip()
    figcap = f"<figcaption>{escape(cap)}</figcaption>" if cap else ""
    # alt_text from Stage 6b (figure_captioner) when present; otherwise empty
    # alt (the prior behaviour, which axe SC 1.1.1 flags for figure-bearing
    # docs — the gap Plans/09 closes).
    alt = (payload.get("alt_text") or "").strip()
    # extended_description from Stage 6b: a longer accessible description that
    # doesn't belong in the short `alt`. Mirror the form-field `help` idiom in
    # ``emit_html._emit_field`` — a sibling element with a generated id, pointed
    # at by ``aria-describedby`` on the <img>. When absent, behave exactly as
    # before (no extra element, no aria-describedby attribute). SC 1.1.1 /
    # WCAG 2.2 AA.
    ext = (payload.get("extended_description") or "").strip()
    # Caption-first + no-hallucination numeric guard (Plans/09 §5 gate 3 / §2.4).
    from ..figure_captioner import (
        alt_from_caption,
        guard_figure_alt,
        strip_numeric_hallucinations,
    )
    if cap:
        # Caption-first: a resolved <figcaption> is a more trustworthy accessible
        # name than a 256M VLM alt (which can hallucinate NON-numeric content the
        # numeric guard won't catch), so derive the short alt from the caption and
        # let the model contribute only the extended description. This matches
        # scripts.eval_figure_captioner.route_figure — the path that scored the
        # SHIP verdict — so production no longer diverges from the gate.
        alt = alt_from_caption(cap)
    else:
        # No caption to verify against: the model owns the alt. Strip any sentence
        # carrying a number absent from the (empty) caption — a wrong number is
        # worse than none for a screen-reader user — falling back to a type-level
        # alt only when nothing survives (see figure_captioner.guard_figure_alt).
        alt = guard_figure_alt(alt, cap)
    # The extended description is always numeric-guarded against the caption.
    ext = strip_numeric_hallucinations(ext, cap)
    if ext:
        # Deterministic id from the region's first FB index so the splice is
        # stable across runs and unique within the document.
        first_fb = (
            region.feature_block_indices[0]
            if region.feature_block_indices else 0
        )
        desc_id = f"dart-figdesc-{first_fb}"
        img = f'<img src="" alt="{escape(alt)}" aria-describedby="{desc_id}">'
        desc = (
            f'<p id="{desc_id}" class="dart-figure-desc">{escape(ext)}</p>'
        )
        return f"<figure>{img}{figcap}{desc}</figure>"
    return f'<figure><img src="" alt="{escape(alt)}">{figcap}</figure>'


def fallback_form(region: Region, feature_blocks: Sequence[FeatureBlock]) -> str:
    # v1: emit a labelled placeholder field so axe label-rules pass.
    return (
        '<form><fieldset><legend>Form</legend>'
        '<label for="dart-form-fallback">Field</label>'
        '<input id="dart-form-fallback" type="text">'
        '</fieldset></form>'
    )


def fallback_metadata_drop(
    region: Region, feature_blocks: Sequence[FeatureBlock],
) -> str:
    return ""


FALLBACKS: dict[str, Callable[..., str]] = {
    "paragraph":       fallback_paragraph,
    "heading":         fallback_heading,
    "list":            fallback_list,
    "definition_list": fallback_definition_list,
    "table":           fallback_table,
    "math":            fallback_math,
    "code_block":      fallback_code_block,
    "blockquote":      fallback_blockquote,
    "figure":          fallback_figure,
    "form":            fallback_form,
    "metadata_drop":   fallback_metadata_drop,
}


def emit_fallback(region: Region, feature_blocks: Sequence[FeatureBlock],
                  **kwargs) -> str:
    """Dispatch to the per-kind fallback. Defaults to ``<p>`` for unknown kinds.

    ``**kwargs`` are forwarded ONLY to ``fallback_table`` (currently
    ``cell_roles=...``). Other per-kind functions have the
    ``(region, feature_blocks)`` signature; kwargs are dropped for them so
    callers can pass ``cell_roles=...`` unconditionally without per-kind
    branching.
    """
    fn = FALLBACKS.get(region.kind)
    if fn is None:
        return f"<p>{escape(_source_text(region, feature_blocks))}</p>"
    if region.kind == "table":
        return fn(region, feature_blocks, **kwargs)
    return fn(region, feature_blocks)


__all__ = ["FALLBACKS", "emit_fallback"]
