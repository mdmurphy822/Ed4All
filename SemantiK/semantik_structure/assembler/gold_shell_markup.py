"""Gold-standard accessible document-shell markup constants (data only).

Self-contained, SemantiK-owned (Apache-2.0) vendored copy of the gold
reference document's ``<head>`` meta, schema.org accessibility JSON-LD, and
``<body>`` landmark skeleton. Phase 1 vendors the *data* only — NOTHING
imports this module yet; the assembler splices these constants in a later
phase gated behind ``SEMANTIK_GOLD_SHELL``, so the unconsumed path is
byte-stable.

Constants
---------
``GOLD_HEAD_META``
    The viewport meta tag (charset + title are emitted by the shell's
    ``.format()`` slots; this is the extra meta the gold ``<head>`` carries).
``GOLD_ACCESSIBILITY_JSONLD``
    The ``<script type="application/ld+json">`` schema.org accessibility
    block, mirroring the gold reference template's advertised feature set:
    ``accessibilityFeature: [alternativeText, readingOrder,
    structuralNavigation, tableOfContents, ARIA]``, ``accessibilityHazard:
    none``, ``accessMode``, ``accessibilitySummary``. Generic ``CreativeWork``
    typing (the gold template's sample-specific author/name fields are
    dropped — those are placeholder artifacts, not part of the accessibility
    contract).
``GOLD_DOC_OPEN`` / ``GOLD_DOC_CLOSE``
    The document skeleton. ``GOLD_DOC_OPEN`` keeps the EXACT ``.format()``
    placeholder contract of ``shell.DOC_OPEN`` — ``{lang}`` + ``{title}`` and
    NO other unescaped braces — so Phase 8 can swap it in via the same
    ``DOC_OPEN.format(lang=..., title=...)`` call. The ``<style>`` bundle and
    the JSON-LD block are spliced in Phase 8 at the brace-free
    ``GOLD_STYLE_SLOT`` / ``GOLD_JSONLD_SLOT`` sentinels (kept brace-free so
    they never collide with ``str.format``). The ``DART_TITLE_SLOT`` sentinel
    is preserved (mirrors ``shell.TITLE_SLOT_SENTINEL``) so Stage-9c title
    gap-fill lands inside ``<header>``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

# Reuse the canonical document-id scanner (do NOT reimplement the regex) — the
# same compiled pattern Stage-9c uses to dedup gap-fill ids. Importing it keeps
# id-uniqueness logic single-sourced (Phase 7 self-minted-label dedup mirrors
# ``pass_9c._ensure_h1_id``'s counter loop).
from .pass_9c import _DOC_IDS_RE
from .semantic_catalog import label_for


# Brace-free splice sentinels (Phase 8 replaces these — kept free of ``{}`` so
# ``GOLD_DOC_OPEN.format(...)`` never trips over them).
GOLD_STYLE_SLOT = "<!-- GOLD_STYLE_SLOT -->"
GOLD_JSONLD_SLOT = "<!-- GOLD_JSONLD_SLOT -->"
GOLD_TITLE_SLOT = "<!-- DART_TITLE_SLOT -->"


GOLD_HEAD_META = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'


GOLD_ACCESSIBILITY_JSONLD = """<script type="application/ld+json">
{
    "@context": "https://schema.org",
    "@type": "CreativeWork",
    "inLanguage": "en",
    "accessibilityFeature": [
        "alternativeText", "readingOrder", "structuralNavigation",
        "tableOfContents", "ARIA"
    ],
    "accessibilityHazard": "none",
    "accessMode": ["textual", "visual"],
    "accessibilitySummary": "Semantic HTML5 with ARIA landmarks, skip link, heading hierarchy, scoped tables, and reduced-motion/dark-mode support."
}
</script>"""


# Same ``.format(lang=..., title=...)`` contract as ``shell.DOC_OPEN``.
# ONLY ``{lang}`` and ``{title}`` are brace-bearing; every other slot is a
# brace-free HTML comment sentinel spliced in Phase 8.
GOLD_DOC_OPEN = (
    '<!DOCTYPE html>\n'
    '<html lang="{lang}">\n'
    '<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    '<title>{title}</title>\n'
    '<!-- GOLD_STYLE_SLOT -->\n'
    '<!-- GOLD_JSONLD_SLOT -->\n'
    '</head>\n'
    '<body>\n'
    '<a class="skip-link" href="#main-content">Skip to main content</a>\n'
    '<main id="main-content" role="main">\n'
    '<article>\n'
    '<header>\n'
    '<!-- DART_TITLE_SLOT -->\n'
    '</header>\n'
)


GOLD_DOC_CLOSE = (
    '<footer role="contentinfo">\n'
    '</footer>\n'
    '</article>\n'
    '</main>\n'
    '</body>\n'
    '</html>\n'
)


# ---------------------------------------------------------------------------
# Phase 7 — per-region gold ARIA wrap keyed by ``payload['semantic_class']``.
#
# Applied as a POST-SELECTION step in ``pass_9a`` Sub-task 1 over the SELECTED
# ``region_html[idx]`` (AFTER the stage6-LLM-HTML / deterministic-fallback
# two-lane if/else), gated on ``shell.resolve_gold_shell_mode()``. The wrap is
# additive — it adds tags AROUND the fragment and never mutates fragment bytes,
# so the downstream ``pass_9c`` flat string-replace splice keys still match and
# the flag-off path is byte-identical.
#
# Each ``role=region`` / labelled-section / nav container mints its OWN inner
# label heading ``<hN id="dart-{class}-{first_fb}">{LABEL}</hN>`` (LABEL from the
# Phase-2 catalog via ``semantic_catalog.label_for``) and points
# ``aria-labelledby`` at that minted id — so every container resolves to a real,
# non-empty-named id at ALL times, independent of Phase 9's outer-``section``
# grouping. The id is deduped against the accumulating doc-id set (NOT
# ``AssembledDoc.anchors``, which is id->region_index ``int``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ContainerSpec:
    """Markup recipe for one gold semantic-class container.

    ``tag`` outer element; ``css`` class attr value (None -> no class attr);
    ``role`` ARIA role attr (None -> omit); ``heading_tag`` the minted label
    heading element (None -> no label heading, e.g. footnote/pullquote);
    ``aria`` one of ``"labelledby"`` (mint id + heading + aria-labelledby),
    ``"label"`` (aria-label=LABEL, no heading), ``"none"`` (label heading with
    NO id, e.g. callouts).
    """

    tag: str
    css: str | None
    role: str | None
    heading_tag: str | None
    aria: str  # "labelledby" | "label" | "none"


# class -> container recipe. Mirrors the plan's wrap mapping; css values are the
# real vendored ``wcag22_css.WCAG22_CSS`` classes (catalog ``css_class``).
# ``figure`` (reuses ``fallback_figure``) and ``section`` (Phase 9 outer
# nesting) are intentionally ABSENT — they pass through unchanged.
_CONTAINER_SPECS: dict[str, _ContainerSpec] = {
    "definition_region": _ContainerSpec("div", "definition", "region", "h4", "labelledby"),
    "worked_example": _ContainerSpec("div", "algorithm worked-example", "region", "h4", "labelledby"),
    "callout_note": _ContainerSpec("div", "callout callout-info", None, "h4", "none"),
    "callout_tip": _ContainerSpec("div", "callout callout-tip", None, "h4", "none"),
    "callout_warning": _ContainerSpec("div", "callout callout-warning", None, "h4", "none"),
    "callout_danger": _ContainerSpec("div", "callout callout-danger", None, "h4", "none"),
    "footnote": _ContainerSpec("aside", None, "doc-footnote", None, "label"),
    "references": _ContainerSpec("section", "references", None, "h2", "labelledby"),
    "pullquote": _ContainerSpec("aside", "pullquote", None, None, "none"),
    "abstract": _ContainerSpec("section", "abstract", None, "h2", "labelledby"),
    "toc": _ContainerSpec("nav", "toc", "navigation", "h2", "labelledby"),
    "exercise": _ContainerSpec("div", "exercise", "region", "h4", "labelledby"),
    "objectives": _ContainerSpec("section", "objectives", None, "h3", "labelledby"),
}

# Classes handled OUTSIDE ``_CONTAINER_SPECS`` (pass-through / special markup).
_PASSTHROUGH_CLASSES = frozenset({"figure", "section"})


# ---------------------------------------------------------------------------
# Body-region absorption (the STOPGAP — SEMANTIK_GOLD_ABSORB).
#
# A live render exposed that a ``worked_example`` region wraps ONLY the example
# LABEL ("EXAMPLE 1.3") because the council emits the example's label and body
# (problem statement / "Solution" / steps) as SEPARATE following regions OUTSIDE
# the box. This pre-step lets a body-bearing component region ABSORB the
# immediately-following sibling regions into its container box, UNTIL a
# boundary. It is a conservative, gated heuristic — the PROPER reviewer-side
# merge/regroup fix is designed separately and will supersede this.
# ---------------------------------------------------------------------------

# Pedagogical-unit boundary single-source-of-truth — hoisted (Phase 1) to the
# neutral ``semantik_structure.pedagogical_units`` module so the assembler absorb and
# the Stage-5e cross-kind regroup share ONE boundary contract (and the
# cross-package import is cycle-free). Re-exported here under the historical
# private names so ``compute_absorption_runs`` is byte-identical. The positional
# call ``_is_absorption_boundary(regions[end], region_html[end])`` below resolves
# to ``is_unit_boundary(region, html)`` — html supplied -> the byte-emptiness
# arm, exactly as before the hoist.
from ..pedagogical_units import (  # noqa: E402
    ABSORB_MAX_RUN as _ABSORB_MAX_RUN,
    BODY_BEARING_COMPONENT_CLASSES as _BODY_BEARING_COMPONENT_CLASSES,
    UNIT_START_PEDAGOGY_CLASSES as _UNIT_START_PEDAGOGY_CLASSES,
    is_passthrough_region as _is_passthrough_region,
    is_unit_boundary as _is_absorption_boundary,
)


def _region_fb_first(region: Any) -> int:
    """``min(feature_block_indices)`` or ``-1`` when the region carries none —
    mirrors ``block_resegment._pair_is_continuation``'s ``cur_first`` read."""
    fbs = getattr(region, "feature_block_indices", None)
    return min(fbs) if fbs else -1


def _region_fb_last(region: Any) -> int:
    """``max(feature_block_indices)`` or ``-1`` — mirrors
    ``block_resegment._pair_is_continuation``'s ``prev_last`` read."""
    fbs = getattr(region, "feature_block_indices", None)
    return max(fbs) if fbs else -1


def _region_payload_get(region: Any, key: str) -> Any:
    return (getattr(region, "payload", None) or {}).get(key)


def compute_absorption_runs(
    regions: Sequence[Any],
    region_html: Sequence[str],
    *,
    max_run: int = _ABSORB_MAX_RUN,
    passthrough_only: bool = False,
) -> dict[int, int]:
    """Map each body-bearing component anchor index -> EXCLUSIVE end index of the
    contiguous following-region run it absorbs (``[anchor .. end)``).

    **DEPRECATED (ITEM4) — legacy; consulted only when SEMANTIK_CONTAINMENT=0.**
    The materialized containment tree's unit arm (``containment._unit_runs``)
    subsumes both the full absorb and the narrow-table passthrough absorb; under
    the default ``SEMANTIK_CONTAINMENT`` the ``shell.resolve_gold_absorb_mode`` /
    ``resolve_narrow_table_absorb_mode`` resolvers short-circuit ``False`` so this
    helper is never reached. Retained solely as the OFF-lever fallback pending
    the post-soak deletion (OPEN Q1).

    For each region whose ``payload['semantic_class']`` is a body-bearing
    component (and whose own fragment is non-empty), scan forward over the
    following regions, absorbing each until the FIRST
    :func:`_is_absorption_boundary` OR until ``max_run`` regions have been
    absorbed (the conservative cap). Only anchors that absorb ≥1 region appear
    in the result (``end > anchor + 1``); an anchor with no absorbable follower
    is omitted (it wraps exactly as the non-absorb path). Absorbed indices are
    never themselves re-used as anchors (a body-bearing follower is a boundary,
    so this is belt-and-suspenders).

    ``passthrough_only`` (table-v2, default ``False`` -> byte-identical to the
    committed full absorb): when ``True`` the forward walk absorbs a follower
    ONLY if it is a Stage-4 passthrough (``is_passthrough_region`` —
    ``source_region_id`` set: table/math/figure) AND it is FB-CONTIGUOUS with
    the previous region (``cur_first == prev_last + 1``, reading the same
    FB-index fields ``block_resegment._pair_is_continuation`` uses). The first
    NON-passthrough OR non-FB-adjacent follower is a hard stop, so the walk
    captures only the unit's OWN contiguous trailing passthrough(s) — a
    non-contiguous standalone section table, or the opening table of the NEXT
    unit, is never pulled in. Still capped at ``max_run``. The anchor scan is
    UNCHANGED (body-bearing components only).
    """
    n = len(region_html)
    runs: dict[int, int] = {}
    absorbed: set[int] = set()
    for i, region in enumerate(regions):
        if i in absorbed:
            continue
        if _region_payload_get(region, "semantic_class") not in _BODY_BEARING_COMPONENT_CLASSES:
            continue
        frag = region_html[i] if i < n else ""
        if not frag or not frag.strip():
            continue
        end = i + 1
        while end < n and (end - (i + 1)) < max_run:
            follower = regions[end]
            if _is_absorption_boundary(follower, region_html[end]):
                break
            if passthrough_only:
                # Narrow scope: absorb a follower only if it is a Stage-4
                # passthrough AND FB-contiguous with the previous region. The
                # first non-passthrough OR non-FB-adjacent follower hard-stops
                # the walk (the over-capture guard).
                if not _is_passthrough_region(follower):
                    break
                prev_last = _region_fb_last(regions[end - 1])
                cur_first = _region_fb_first(follower)
                if cur_first != prev_last + 1:
                    break
            end += 1
        if end > i + 1:
            runs[i] = end
            absorbed.update(range(i + 1, end))
    return runs


def collect_doc_ids(fragments: Iterable[str]) -> set[str]:
    """Seed the accumulating doc-id set from already-emitted region fragments.

    Reuses the single-sourced ``pass_9c._DOC_IDS_RE`` so minted container-label
    ids never collide with an id a Stage-6 candidate already emitted.
    """
    ids: set[str] = set()
    for frag in fragments:
        if frag:
            ids.update(_DOC_IDS_RE.findall(frag))
    return ids


def _mint_unique_id(base: str, doc_ids: set[str]) -> str:
    """Mint ``base`` (or ``base-2``, ``base-3``...) unique against ``doc_ids``.

    Mirrors ``pass_9c._ensure_h1_id``'s dedup counter loop; the minted id is
    ADDED to ``doc_ids`` so the next container in the same document dedups
    against it (the threaded accumulating set).
    """
    ident = base
    n = 2
    while ident in doc_ids:
        ident = f"{base}-{n}"
        n += 1
    doc_ids.add(ident)
    return ident


def _opening_signature(spec: _ContainerSpec) -> str:
    """The leading opening-tag prefix used for the idempotency guard."""
    if spec.css:
        return f'<{spec.tag} class="{spec.css}"'
    if spec.role:
        return f'<{spec.tag} role="{spec.role}"'
    return f"<{spec.tag}"


def _wrap_semantic_class(
    inner_html: str,
    region: Any,
    *,
    doc_ids: set[str],
) -> str:
    """Wrap ``inner_html`` in the gold ARIA container for the region's class.

    Reads ``region.payload['semantic_class']``. Returns ``inner_html``
    UNCHANGED when: the class is None (the default on ~all regions), the
    fragment is empty/whitespace, the class is ``figure`` (already emitted via
    ``fallback_figure``) or ``section`` (deferred to Phase 9), the fragment is
    already wrapped (idempotency), or the class is unknown. Otherwise emit the
    gold container with a self-minted, resolvable inner label heading.

    ``doc_ids`` is the accumulating set of ids already present in the document;
    a minted container-label id is deduped against it AND added to it.
    """
    sc = (getattr(region, "payload", None) or {}).get("semantic_class")
    if sc is None:
        return inner_html
    if not inner_html or not inner_html.strip():
        return inner_html
    if sc in _PASSTHROUGH_CLASSES:
        return inner_html

    stripped = inner_html.lstrip()

    # code_region: <pre role="region"> wrap, unless the fragment is already a
    # <pre> (idempotent — the fallback/stage6 code emit is already <pre>).
    if sc == "code_region":
        if stripped[:4].lower() == "<pre":
            return inner_html
        return f'<pre role="region">{inner_html}</pre>'

    spec = _CONTAINER_SPECS.get(sc)
    if spec is None:
        # Unknown / unmapped class — never invent markup (defensive no-op).
        return inner_html

    # Idempotency: a fragment that already opens with this container's
    # signature (e.g. a stage6 candidate that emitted the wrapper) is not
    # re-wrapped.
    sig = _opening_signature(spec)
    if stripped[: len(sig)].lower() == sig.lower():
        return inner_html

    label = label_for(sc) or sc
    esc_label = escape(label)
    first_fb = (
        region.feature_block_indices[0]
        if getattr(region, "feature_block_indices", None)
        else 0
    )

    attrs: list[str] = []
    if spec.css:
        attrs.append(f'class="{spec.css}"')
    if spec.role:
        attrs.append(f'role="{spec.role}"')

    heading = ""
    if spec.aria == "labelledby" and spec.heading_tag:
        ident = _mint_unique_id(f"dart-{sc}-{first_fb}", doc_ids)
        attrs.append(f'aria-labelledby="{ident}"')
        heading = f'<{spec.heading_tag} id="{ident}">{esc_label}</{spec.heading_tag}>'
    elif spec.aria == "label":
        attrs.append(f'aria-label="{escape(label, quote=True)}"')
    elif spec.heading_tag:  # aria "none" but a label heading (callouts)
        heading = f"<{spec.heading_tag}>{esc_label}</{spec.heading_tag}>"

    open_tag = f"<{spec.tag} {' '.join(attrs)}>" if attrs else f"<{spec.tag}>"
    return f"{open_tag}{heading}{inner_html}</{spec.tag}>"


__all__ = (
    "GOLD_ACCESSIBILITY_JSONLD",
    "GOLD_DOC_CLOSE",
    "GOLD_DOC_OPEN",
    "GOLD_HEAD_META",
    "GOLD_JSONLD_SLOT",
    "GOLD_STYLE_SLOT",
    "GOLD_TITLE_SLOT",
    "collect_doc_ids",
    "compute_absorption_runs",
)
