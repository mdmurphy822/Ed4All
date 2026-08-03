"""Per-region-kind fallback emitters (Plans/04 §1, all 11 RegionKinds).

Used when Stage 7 dropped all K candidates for a region (the
``top_per_region[i] is None`` case). Output must conform to
docs/ontology.md §2 — every fallback emits a small, valid HTML5
fragment that will pass the per-region hard-gate's HTML5 well-formedness
check at minimum.

Coverage: every kind in
:data:`semantik_structure.structure_graph.REGION_KINDS` has an entry in
:data:`FALLBACKS`. ``emit_fallback`` is the public dispatch.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from html import escape
from typing import Any

from ..structure_graph import Region, resolve_bert_authoritative
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


def _containment_on() -> bool:
    """Read the ITEM4 containment gate (function-local import — cycle-safe)."""
    from ..containment import resolve_containment_mode

    return resolve_containment_mode()


def _fb_text_at(feature_blocks: Sequence[FeatureBlock], idx: Any) -> str:
    """Stripped raw text of the FB at ``idx`` (``""`` when out of range)."""
    if not isinstance(idx, int) or not (0 <= idx < len(feature_blocks)):
        return ""
    raw = getattr(feature_blocks[idx], "raw", None)
    return ((getattr(raw, "text", None) or "") if raw is not None else "").strip()


def _css_class_attr(region: Region) -> str:
    """`` class="…"`` for a region carrying a ``payload['css_class']`` hint.

    The deterministic structure-correction pass stamps a semantic class hint
    (e.g. ``"pedagogy-example"``) on pedagogical headings it demotes to
    paragraphs, so the worked-example / solution semantics survive the
    demotion. When present, the assembler emits ``<p class="…">``; when absent
    (every non-demoted paragraph), this returns the empty string so the emit
    is byte-identical to the prior plain-``<p>`` behaviour. A class attr on
    ``<p>`` is valid HTML5 and WCAG-neutral.

    RETIRED under ``SEMANTIK_BERT_AUTHORITATIVE``: this ``<p class="…">`` path is
    a band-aid for the old deterministic pedagogical-demotion (which is itself
    retired under the flag). With the authoritative head, a pedagogical kind
    carries ``payload['semantic_class']`` and is routed through the gold
    semantic-class wrapper (``_wrap_semantic_class``) instead, so this returns
    the empty string (no ``class`` attr on ``<p>``). Flag OFF -> byte-identical.
    """
    if resolve_bert_authoritative():
        return ""
    payload = region.payload or {}
    css_class = payload.get("css_class")
    if isinstance(css_class, str) and css_class.strip():
        return f' class="{escape(css_class.strip(), quote=True)}"'
    return ""


def fallback_paragraph(region: Region, feature_blocks: Sequence[FeatureBlock]) -> str:
    return (
        f"<p{_css_class_attr(region)}>"
        f"{escape(_source_text(region, feature_blocks))}</p>"
    )


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
    # ITEM4: when the containment builder stamped ``payload['dl_pairs']`` (council
    # structural_role term/def grouping), emit ONE <dt>/<dd> group per pair —
    # closing the dl-flattening gap (the legacy path below collapses every FB
    # after the first into ONE <dd>). Absent (no council state / flag off) ->
    # the legacy single-pair flatten (byte-identical).
    payload = region.payload or {}
    pairs = payload.get("dl_pairs")
    if pairs:
        groups: list[str] = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or not pair:
                continue
            dt_text = _fb_text_at(feature_blocks, pair[0])
            def_fbs = pair[1] if len(pair) > 1 and isinstance(pair[1], (list, tuple)) else []
            dd_text = " ".join(
                t for f in def_fbs if (t := _fb_text_at(feature_blocks, f))
            )
            groups.append(f"<dt>{escape(dt_text)}</dt><dd>{escape(dd_text)}</dd>")
        if groups:
            return "<dl>" + "".join(groups) + "</dl>"
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
    # ITEM4: suppress the duplicate <caption> when the containment walk nested
    # the caption region's <p> after the table (payload['caption_nested']); the
    # caption then ships exactly ONCE. Flag off / no caption edge -> legacy emit.
    suppress_caption = bool(payload.get("caption_nested")) and _containment_on()
    if (
        not suppress_caption
        and caption_idx is not None
        and 0 <= caption_idx < len(feature_blocks)
    ):
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
    # T4 — DETERMINISTIC-FIRST math reconstruction. When
    # SEMANTIK_MATH_RECONSTRUCT is on, NEVER emit the raw glyph-soup
    # <span class="math-source"> (it linearises 2-D structure into a flat,
    # screen-reader-hostile string). Prefer the Stage-5 gate-valid
    # deterministic candidate; else emit the uniform accessible
    # <math><mtext>/<merror> last resort (source carried as alttext). Flag
    # OFF -> the legacy soup (byte-identical).
    try:
        from ..math_reconstruct.policy import resolve_math_reconstruct_mode

        reconstruct_on = resolve_math_reconstruct_mode()
    except Exception:
        reconstruct_on = False
    if reconstruct_on:
        mr = payload.get("math_reconstruct")
        if isinstance(mr, dict):
            mathml = mr.get("mathml")
            if isinstance(mathml, str) and mathml.strip():
                return mathml
        try:
            from ..math_reconstruct.mathml_emit import accessible_uniform_mathml

            return accessible_uniform_mathml(
                src, display=bool(payload.get("display", True))
            )
        except Exception:
            # Even on an import failure, do not regress to glyph soup —
            # emit a minimal valid <math> with the source as alttext/mtext.
            safe = escape(src or "math expression")
            safe_attr = escape(src or "math expression", quote=True)
            return f'<math alttext="{safe_attr}"><mtext>{safe}</mtext></math>'
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
    # ITEM4: suppress the duplicate <figcaption> when the containment walk nested
    # the caption region's <p> inside this <figure> (payload['caption_nested']),
    # so the caption ships exactly ONCE. The short ``alt`` is still caption-derived
    # (alt_from_caption below reads ``cap``). Flag off / no caption edge -> legacy.
    suppress_caption = bool(payload.get("caption_nested")) and _containment_on()
    figcap = "" if suppress_caption else (f"<figcaption>{escape(cap)}</figcaption>" if cap else "")
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
        # scripts.eval.eval_figure_captioner.route_figure — the path that scored the
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
    # Part F — the Stage-F sidecar-write pass stamps ``image_src`` (relative
    # path to the written figure PNG) on the payload. Fill the previously-
    # empty ``src=""``. When no src resolved (figure path off, deferred, or
    # write failed), keep the historic empty-src behaviour byte-for-byte.
    image_src = (payload.get("image_src") or "").strip()
    src_attr = escape(image_src)
    if ext:
        # Deterministic id from the region's first FB index so the splice is
        # stable across runs and unique within the document.
        first_fb = (
            region.feature_block_indices[0]
            if region.feature_block_indices else 0
        )
        desc_id = f"semantik-figdesc-{first_fb}"
        img = f'<img src="{src_attr}" alt="{escape(alt)}" aria-describedby="{desc_id}">'
        desc = (
            f'<p id="{desc_id}" class="semantik-figure-desc">{escape(ext)}</p>'
        )
        return f"<figure>{img}{figcap}{desc}</figure>"
    return f'<figure><img src="{src_attr}" alt="{escape(alt)}">{figcap}</figure>'


def fallback_form(region: Region, feature_blocks: Sequence[FeatureBlock]) -> str:
    # v1: emit a labelled placeholder field so axe label-rules pass.
    return (
        '<form><fieldset><legend>Form</legend>'
        '<label for="semantik-form-fallback">Field</label>'
        '<input id="semantik-form-fallback" type="text">'
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
