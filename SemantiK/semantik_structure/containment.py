"""ITEM4 — materialized containment tree over the flat ``Region`` list.

A single deterministic, CPU-only pass derives the document's containment
structure — an explicit parent/child **forest over the flat Region list** — so
the Stage-9 assembler can WALK that tree instead of re-deriving containment
three different ways at emit time (section stack grouping, gold-absorb runs,
narrow-table absorb).

This module is deliberately **dependency-light**: it imports only
:mod:`semantik_structure.pedagogical_units` (the pedagogical-unit boundary SoT) and
:func:`semantik_structure.assembler.heading_tree.normalize_heading_levels` (the
promote/demote/never-skip ladder), so a future Stage-5 consumer (ITEM5/ITEM6)
can call the builder without pulling in the assembler package.

NO LLM call site is added here — the pass is a pure function (deterministic,
CPU-only), so it carries NO ``DecisionCapture`` obligation (there are no
exceptions in ITEM4). No long-running loop is added (the builder is O(n) over an
in-memory region list), so no stop-sentinel seam is required.

The tree encodes four ownership relations (§ITEM4 D3), processed in ONE forward
pass with precedence **caption > unit > section**:

1. heading → section ownership (a heading owns every region until the next
   heading of level ≤ its own — the stack discipline
   ``pass_9a._group_regions_into_sections`` re-derives);
2. pedagogical unit label/anchor → body ownership (what
   ``gold_shell_markup.compute_absorption_runs`` re-derives);
3. figure/table → caption ownership (from ``payload['caption_fb_index']``);
4. definition-list term/def pair structure (intra-region ``payload['dl_pairs']``
   stamp — no edge).

The representation is a **derived sidecar forest, rebuilt at consumption time**
(§ITEM4 D1) — NOT a ``parent_region_id`` field on ``Region`` (region identity is
positional and shifts under every later mutation, so a stored index-valued field
goes stale; a derived tree has no staleness bug class).
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

from .assembler.heading_tree import normalize_heading_levels
from .pedagogical_units import (
    ABSORB_MAX_RUN,
    BODY_BEARING_COMPONENT_CLASSES,
    is_passthrough_region,
    is_unit_boundary,
)

logger = logging.getLogger(__name__)

# Default-ON parse: only an EXPLICIT falsey token reverts to the byte-identical
# legacy path (mirrors ``structure_graph.resolve_reading_order_fix`` /
# ``pedagogical_units.resolve_unit_regroup_mode``).
_CONTAINMENT_FALSEY = frozenset({"0", "false", "no", "off"})

# Matches the outermost ``<hN>`` opening tag of an HTML fragment — a private
# copy of ``pass_9a._HEADING_OPEN_RE`` so this module stays assembler-free.
_HEADING_OPEN_RE = re.compile(r"<h([1-6])\b([^>]*)>", re.IGNORECASE)
# Captures the id VALUE on the first ``<hN>`` opening tag (the resolvable
# ``aria-labelledby`` target). Mirrors ``pass_9a._HEADING_ID_VALUE_RE``.
_HEADING_ID_VALUE_RE = re.compile(
    r'<h[1-6]\b[^>]*\bid\s*=\s*"([^"]+)"',
    re.IGNORECASE,
)


def resolve_containment_mode() -> bool:
    """Return True when the materialized-containment pass is enabled.

    Reads ``SEMANTIK_CONTAINMENT``. **Default ON** (unset / blank / truthy /
    garbage -> on): only an EXPLICIT falsey value (``0``/``false``/``no``/``off``,
    case-insensitive) reverts to the byte-identical legacy path (every consuming
    call site branches on this resolver). Mirrors the default-ON falsey-set
    posture of ``structure_graph.resolve_reading_order_fix`` /
    ``pedagogical_units.resolve_unit_regroup_mode`` (house pattern), NOT a
    truthy-set opt-in.
    """
    raw = (os.environ.get("SEMANTIK_CONTAINMENT") or "").strip().lower()
    return raw not in _CONTAINMENT_FALSEY


def _resolve_reading_order_fix() -> bool:
    """Read the reading-order-fix gate without a module-load cycle.

    ``structure_graph`` is heavy; import it lazily so ``containment`` stays
    importable from both the assembler and the cascade. Mirrors the function-
    local import posture ``shell.resolve_gold_absorb_mode`` uses.
    """
    from .structure_graph import resolve_reading_order_fix

    return resolve_reading_order_fix()


@dataclass(frozen=True)
class ContainmentTree:
    """A derived parent/child forest over the flat Region list.

    ``parent[i]`` is region ``i``'s parent region index, or ``None`` when ``i``
    is a root. ``children[i]`` is the ascending tuple of ``i``'s child region
    indices (ascending == document order post reading-order sort). ``roots`` is
    the ascending tuple of parentless region indices. ``edge_kind[i]`` is the
    kind of the edge from ``i`` to its parent — ``"section"`` | ``"unit"`` |
    ``"caption"`` — or ``None`` for a root. ``levels`` maps each heading region
    index to its normalized structural level (Phase-3 consumer). ``diagnostics``
    carries per-arm counters (``dropped_backward_edges``, ``unit_runs``,
    ``caption_edges``, ...).

    TREE-COVER invariant (extends R-PART/coverage):
      1. every region index ``0..n-1`` appears exactly once in
         ``roots ∪ ⋃children`` (each node has exactly one parent-or-root);
      2. a pre-order walk from ``roots`` visits every node EXACTLY once
         (forest, acyclic);
      3. ORDER-CONSERVING RENDER: the pre-order emission sequence is the
         IDENTITY permutation of the region list (parents always precede
         children — any backward candidate edge is dropped to metadata).
    """

    parent: tuple[int | None, ...]
    children: tuple[tuple[int, ...], ...]
    roots: tuple[int, ...]
    edge_kind: tuple[str | None, ...]
    levels: dict[int, int] = field(default_factory=dict)
    diagnostics: dict[str, int] = field(default_factory=dict)


def _trivial_forest(n: int, *, diagnostics: dict[str, int] | None = None) -> ContainmentTree:
    """The all-roots forest: every region is its own root, no edges.

    The fail-closed return (reading-order off, or a TREE-COVER assertion
    failure) — a pre-order walk over it is the identity permutation, so a walk
    consuming it emits the flat concat (byte-identical legacy behavior).
    """
    diag = {"dropped_backward_edges": 0, "unit_runs": 0, "caption_edges": 0}
    diag.update(diagnostics or {})
    return ContainmentTree(
        parent=tuple(None for _ in range(n)),
        children=tuple(() for _ in range(n)),
        roots=tuple(range(n)),
        edge_kind=tuple(None for _ in range(n)),
        levels={},
        diagnostics=diag,
    )


def _payload_get(region: Any, key: str) -> Any:
    return (getattr(region, "payload", None) or {}).get(key)


def _extract_heading_level(html: str | None) -> int | None:
    """Integer level of the first ``<hN>`` tag, or ``None``.

    Byte-parity with ``pass_9a._extract_heading_level`` so the builder's raw
    levels equal Sub-task 2's.
    """
    if not html:
        return None
    m = _HEADING_OPEN_RE.search(html)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _heading_id_value(html: str | None) -> str | None:
    """Return the id on the fragment's first ``<hN>`` tag, or ``None``."""
    if not html:
        return None
    m = _HEADING_ID_VALUE_RE.search(html)
    return m.group(1) if m else None


def _coverage_map(regions: Sequence[Any]) -> dict[int, int]:
    """Map ``fb_index -> owning region index`` (unique by the Stage-5 coverage
    invariant: every FB appears in exactly one region's
    ``feature_block_indices``)."""
    cover: dict[int, int] = {}
    for ridx, region in enumerate(regions):
        for fb in getattr(region, "feature_block_indices", ()) or ():
            cover[fb] = ridx
    return cover


def _heading_levels(
    regions: Sequence[Any],
    region_html: Sequence[str] | None,
) -> tuple[list[int], dict[int, int]]:
    """Compute the normalized structural level of every heading region ONCE.

    Byte-parity with ``pass_9a`` Sub-task 2: raw level =
    ``payload['level_hint']`` if stamped, else the first ``<hN>`` in
    ``region_html[i]``, else 2 (the exact ladder at ``pass_9a.py:523-528``);
    then ``normalize_heading_levels`` (promote-first / demote-forward /
    never-skip). Returns ``(heading_indices, {heading_idx: normalized_level})``.
    """
    heading_indices: list[int] = []
    raw_levels: list[int] = []
    for idx, region in enumerate(regions):
        if getattr(region, "kind", None) != "heading":
            continue
        lvl_hint = _payload_get(region, "level_hint")
        if lvl_hint is not None:
            lvl = int(lvl_hint)
        else:
            html = region_html[idx] if region_html and idx < len(region_html) else None
            lvl = _extract_heading_level(html) or 2
        heading_indices.append(idx)
        raw_levels.append(lvl)
    normalized = normalize_heading_levels(raw_levels)
    return heading_indices, dict(zip(heading_indices, normalized))


def _section_parents(
    regions: Sequence[Any],
    levels: dict[int, int],
) -> list[int | None]:
    """Assign each region its nearest enclosing OPEN heading (section arm).

    Reproduces the stack discipline of
    ``pass_9a._group_regions_into_sections`` (:194-222): a heading of level L
    pops every open section of level ≥ L, then opens its own; each region's
    section parent is the heading whose section is open when it is reached.
    Heading H's own parent is the enclosing heading AFTER the pop (a heading of
    level M > L nests under the nearest preceding heading of level < M). Every
    parent index < child index (headings precede their content), so section
    edges are order-conserving by construction.
    """
    n = len(regions)
    parents: list[int | None] = [None] * n
    open_stack: list[int] = []  # region indices of currently-open heading sections
    for idx in range(n):
        if idx in levels:
            lvl = levels[idx]
            while open_stack and levels[open_stack[-1]] >= lvl:
                open_stack.pop()
            parents[idx] = open_stack[-1] if open_stack else None
            open_stack.append(idx)
        else:
            parents[idx] = open_stack[-1] if open_stack else None
    return parents


def _region_fb_first(region: Any) -> int:
    fbs = getattr(region, "feature_block_indices", None)
    return min(fbs) if fbs else -1


def _region_fb_last(region: Any) -> int:
    fbs = getattr(region, "feature_block_indices", None)
    return max(fbs) if fbs else -1


def _unit_runs(
    regions: Sequence[Any],
    region_html: Sequence[str] | None,
) -> dict[int, int]:
    """Map each body-bearing anchor index -> EXCLUSIVE end of the run it claims.

    Lifts ``gold_shell_markup.compute_absorption_runs``' loop shape to tree
    edges (importing the boundary SoT from ``pedagogical_units`` — never
    re-derived): for each region whose ``payload['semantic_class']`` is a
    body-bearing component (with a non-empty fragment), scan forward claiming
    each follower until the FIRST ``is_unit_boundary`` OR ``ABSORB_MAX_RUN``.
    INCLUDING the narrow-table FB-adjacency arm (subsumes
    ``SEMANTIK_UNIT_REGROUP_TABLE`` unconditionally): a Stage-4 passthrough
    follower (``is_passthrough_region``) is claimed ONLY when FB-contiguous
    with the previous region (``cur_first == prev_last + 1``); a non-adjacent
    passthrough hard-stops the walk (so a standalone section table / the next
    unit's opening table is never over-captured). Only anchors that claim ≥1
    follower appear (``end > anchor + 1``).
    """
    n = len(regions)
    runs: dict[int, int] = {}
    claimed: set[int] = set()
    for i, region in enumerate(regions):
        if i in claimed:
            continue
        if _payload_get(region, "semantic_class") not in BODY_BEARING_COMPONENT_CLASSES:
            continue
        frag = region_html[i] if region_html and i < len(region_html) else None
        # Anchor with an empty fragment claims nothing (matches the absorb).
        if region_html is not None and (not frag or not frag.strip()):
            continue
        end = i + 1
        while end < n and (end - (i + 1)) < ABSORB_MAX_RUN:
            follower = regions[end]
            follower_html = (
                region_html[end] if region_html and end < len(region_html) else None
            )
            if is_unit_boundary(follower, follower_html):
                break
            if is_passthrough_region(follower):
                prev_last = _region_fb_last(regions[end - 1])
                cur_first = _region_fb_first(follower)
                if cur_first != prev_last + 1:
                    break
            end += 1
        if end > i + 1:
            runs[i] = end
            claimed.update(range(i + 1, end))
    return runs


def _text_of_region(region: Any, feature_blocks: Sequence[Any]) -> str:
    """Concatenated stripped raw text of a region's FBs (for the caption
    single-owner text-equality guard)."""
    parts: list[str] = []
    for fb in sorted(getattr(region, "feature_block_indices", ()) or ()):
        if 0 <= fb < len(feature_blocks):
            raw = getattr(feature_blocks[fb], "raw", None)
            t = (getattr(raw, "text", None) or "") if raw is not None else ""
            t = t.strip()
            if t:
                parts.append(t)
    return " ".join(parts)


def _fb_text(feature_blocks: Sequence[Any], fb_idx: int) -> str:
    if not (0 <= fb_idx < len(feature_blocks)):
        return ""
    raw = getattr(feature_blocks[fb_idx], "raw", None)
    return ((getattr(raw, "text", None) or "") if raw is not None else "").strip()


def _dl_pairs_from_council(region: Any, council_state: Any) -> list[list[Any]] | None:
    """Read per-FB ``structural_role`` top-1 (definition_term / definition_def)
    from ``council_state`` and group FBs into ``[[term_fb, [def_fb, ...]], ...]``.

    Returns ``None`` (no stamp -> legacy flattened emit) when the council state
    has no usable structural_role signal for the region's FBs.
    """
    fb_indices = list(getattr(region, "feature_block_indices", ()) or ())
    if not fb_indices:
        return None
    outputs = getattr(council_state, "outputs", None) or {}
    bert_out = outputs.get("structure")
    if bert_out is None:
        return None
    # Map fb_index -> structural_role top-1 label.
    role_by_fb: dict[int, str] = {}
    for sig in getattr(bert_out, "signals", []) or []:
        if getattr(sig, "head_name", None) != "structural_role":
            continue
        rid = getattr(sig, "region_id", None)
        labels = getattr(sig, "top_k_labels", None) or []
        if rid is not None and labels:
            role_by_fb[rid] = labels[0]
    if not role_by_fb:
        return None
    pairs: list[list[Any]] = []
    cur_term: int | None = None
    cur_defs: list[int] = []
    for fb in fb_indices:
        role = role_by_fb.get(fb)
        if role == "definition_term":
            if cur_term is not None:
                pairs.append([cur_term, cur_defs])
            cur_term = fb
            cur_defs = []
        elif role == "definition_def":
            if cur_term is not None:
                cur_defs.append(fb)
        # any other role is skipped (not part of a term/def pair)
    if cur_term is not None:
        pairs.append([cur_term, cur_defs])
    return pairs or None


def _preorder(tree_children: Sequence[Sequence[int]], roots: Sequence[int]) -> list[int]:
    """Iterative pre-order walk (roots ascending, children ascending)."""
    out: list[int] = []
    # Reverse-push so ascending order pops first.
    stack: list[int] = list(reversed(roots))
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            # A cycle / re-entry — abort (the caller treats a short walk as a
            # cover failure).
            return out
        seen.add(node)
        out.append(node)
        for child in reversed(tree_children[node]):
            stack.append(child)
    return out


def _assert_tree_cover(tree: ContainmentTree, n: int) -> ContainmentTree:
    """Validate the three TREE-COVER invariants; on failure log + return the
    trivial all-roots forest (fail closed — never a broken render).

    Mirrors the reading-order sort's permutation assert (fail-closed to the
    safe forest instead of raising, so a real doc never crashes on the safety
    net doing its job).
    """
    try:
        walk = _preorder(tree.children, tree.roots)
        # Invariant 1 + 2: forest covers every node exactly once.
        if len(walk) != n or set(walk) != set(range(n)):
            raise AssertionError(
                f"TREE-COVER cover failure: walk={len(walk)} nodes, expected {n}"
            )
        # Invariant 3: ORDER-CONSERVING — pre-order == identity permutation.
        if walk != list(range(n)):
            raise AssertionError("TREE-COVER order failure: pre-order != identity")
    except AssertionError as exc:
        logger.warning("containment tree TREE-COVER assertion failed: %s", exc)
        diag = dict(tree.diagnostics)
        diag["tree_cover_failed"] = 1
        return _trivial_forest(n, diagnostics=diag)
    return tree


def build_containment_tree(
    regions: Sequence[Any],
    feature_blocks: Sequence[Any],
    *,
    region_html: Sequence[str] | None = None,
    council_state: Any | None = None,
) -> ContainmentTree:
    """Materialize the containment forest over ``regions`` (§ITEM4 D3).

    ONE forward pass, precedence **caption > unit > section**. Deterministic +
    CPU-only; no LLM. When the reading-order fix is OFF the builder degrades to
    the trivial all-roots forest (the region list is not in reading order, so a
    forward-adjacency walk is meaningless) -> a consuming walk == flat concat ->
    legacy behavior preserved without a special case.
    """
    n = len(regions)
    if n == 0:
        return _trivial_forest(0)
    if not _resolve_reading_order_fix():
        return _trivial_forest(n, diagnostics={"reading_order_off": 1})

    diagnostics = {"dropped_backward_edges": 0, "unit_runs": 0, "caption_edges": 0}

    # Levels (once) + section arm (default parent for every region).
    _heading_indices, levels = _heading_levels(regions, region_html)
    section_parent = _section_parents(regions, levels)

    parent: list[int | None] = list(section_parent)
    edge_kind: list[str | None] = [
        ("section" if p is not None else None) for p in parent
    ]

    # Caption arm (highest precedence) — figure/table -> owning paragraph.
    coverage = _coverage_map(regions)
    caption_claimed: set[int] = set()
    for i, region in enumerate(regions):
        if getattr(region, "kind", None) not in {"figure", "table"}:
            continue
        cap_idx = _payload_get(region, "caption_fb_index")
        if not isinstance(cap_idx, int):
            continue
        owner = coverage.get(cap_idx)
        if owner is None:
            continue
        # Order conservation (D1-3): a backward / self owner (the caption-ABOVE
        # case) is dropped to metadata-only.
        if owner <= i:
            diagnostics["dropped_backward_edges"] += 1
            continue
        if getattr(regions[owner], "kind", None) != "paragraph":
            continue
        if owner in caption_claimed:
            continue
        # Owner must be single-FB OR its concatenated text equals the caption FB
        # text (never nest a merged paragraph carrying more than the caption).
        owner_fbs = list(getattr(regions[owner], "feature_block_indices", ()) or ())
        if len(owner_fbs) != 1:
            cap_text = _fb_text(feature_blocks, cap_idx)
            if not cap_text or _text_of_region(regions[owner], feature_blocks) != cap_text:
                continue
        parent[owner] = i
        edge_kind[owner] = "caption"
        caption_claimed.add(owner)
        diagnostics["caption_edges"] += 1
        # Phase-2 stamp: the figure/table's own duplicate <figcaption>/<caption>
        # is suppressed at emit time (fallbacks reads this) because the caption
        # region's fragment is now nested inside the parent by the walk. In-place
        # payload stamp (the established pattern). Byte-inert until Phase 2's
        # assembler consumption reads it.
        payload = getattr(region, "payload", None)
        if isinstance(payload, dict):
            payload["caption_nested"] = True

    # Unit arm — body-bearing anchor -> following body regions. A caption child
    # is never re-claimed as a unit body (precedence).
    unit_runs = _unit_runs(regions, region_html)
    for anchor, end in unit_runs.items():
        claimed_any = False
        for follower in range(anchor + 1, end):
            if follower in caption_claimed:
                continue
            parent[follower] = anchor
            edge_kind[follower] = "unit"
            claimed_any = True
        if claimed_any:
            diagnostics["unit_runs"] += 1

    # dl pairs (intra-region, no edge) — in-place payload stamp.
    if council_state is not None:
        for region in regions:
            if getattr(region, "kind", None) != "definition_list":
                continue
            pairs = _dl_pairs_from_council(region, council_state)
            if pairs:
                payload = getattr(region, "payload", None)
                if isinstance(payload, dict):
                    payload["dl_pairs"] = pairs

    # Build children (ascending == document order) + roots.
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i in range(n):
        p = parent[i]
        if p is None:
            roots.append(i)
        elif 0 <= p < n and p < i:
            children[p].append(i)
        else:
            # Defensive: a backward / out-of-range parent would violate order
            # conservation — treat as a root (the assert catches any residual).
            parent[i] = None
            edge_kind[i] = None
            roots.append(i)

    tree = ContainmentTree(
        parent=tuple(parent),
        children=tuple(tuple(c) for c in children),
        roots=tuple(roots),
        edge_kind=tuple(edge_kind),
        levels=dict(levels),
        diagnostics=diagnostics,
    )
    return _assert_tree_cover(tree, n)


def _nest_caption_fragments(frag: str, caps: list[str]) -> str:
    """Nest caption child fragment(s) INSIDE the parent figure/table fragment.

    Inserts the caption bytes before the FINAL ``</figure>`` (so the caption
    ``<p>`` is flow content inside the ``<figure>``); for a ``<table>`` the
    caption ``<p>`` cannot live inside the table, so it is placed as a sibling
    immediately AFTER ``</table>``. Absent both tags (Stage-6 authored
    something else) -> degrade to a trailing sibling (D4). NEVER mutates the
    caption bytes — it splices whole fragments between markup, so pass_9c
    splice keys survive.
    """
    if not caps:
        return frag
    suffix = "".join(caps)
    low = frag.rfind("</figure>")
    if low != -1:
        return frag[:low] + suffix + frag[low:]
    low = frag.rfind("</table>")
    if low != -1:
        end = low + len("</table>")
        return frag[:end] + suffix + frag[end:]
    return frag + suffix


def render_tree(
    tree: ContainmentTree,
    region_html: Sequence[str],
    *,
    absorbed_indices: AbstractSet[int] | None = None,
) -> str:
    """Emit the body HTML by walking the tree (Phase-2 consumer; present from
    Phase 1 for unit-testability).

    Reproduces ``pass_9a._group_regions_into_sections`` exactly when the tree
    carries only section edges (the §5 equivalence test): open a
    ``<section aria-labelledby="{hid}">`` at each heading node (``hid`` = the id
    on that heading's ``<hN>``), close every open section of level ≥ the new
    heading's level, close all remaining at EOF. Because TREE-COVER(3)
    guarantees the pre-order emission is the identity permutation of the region
    list, this linear stack walk over indices IS the pre-order tree walk.

    ``absorbed_indices`` (unit/caption descendants already composed INTO their
    anchor's box, or caption children nested into a figure/table) are SKIPPED in
    the linear walk so they render exactly once. A CAPTION child whose parent
    figure/table IS emitted at top level (not itself absorbed) is nested INSIDE
    that parent's fragment (``_nest_caption_fragments``), fixing the caption
    double-render. A heading with no resolvable id is emitted un-sectioned
    (defensive). The wrap adds ``<section>`` tags BETWEEN whole fragments and
    NEVER mutates a fragment's bytes.
    """
    absorbed = set(absorbed_indices or ())
    levels = tree.levels
    edge_kind = tree.edge_kind
    parent = tree.parent
    # caption children to nest into their figure/table parent fragment
    caption_by_parent: dict[int, list[int]] = {}
    for c, ek in enumerate(edge_kind):
        if ek == "caption":
            p = parent[c]
            if p is not None:
                caption_by_parent.setdefault(p, []).append(c)
    parts: list[str] = []
    open_levels: list[int] = []
    for idx, html in enumerate(region_html):
        if not html:
            continue
        if idx in absorbed:
            continue
        frag = html
        caps = caption_by_parent.get(idx)
        if caps:
            frag = _nest_caption_fragments(
                frag, [region_html[c] for c in sorted(caps) if region_html[c]]
            )
        if idx in levels:
            lvl = levels[idx]
            while open_levels and open_levels[-1] >= lvl:
                parts.append("</section>")
                open_levels.pop()
            hid = _heading_id_value(html)
            if hid:
                safe = hid.replace('"', "&quot;")
                parts.append(f'<section aria-labelledby="{safe}">')
                open_levels.append(lvl)
            parts.append(frag)
        else:
            parts.append(frag)
    while open_levels:
        parts.append("</section>")
        open_levels.pop()
    return "".join(parts)


__all__ = [
    "ContainmentTree",
    "build_containment_tree",
    "render_tree",
    "resolve_containment_mode",
]
