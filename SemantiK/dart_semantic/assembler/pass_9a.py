"""Stage 9a — deterministic assembler pass.

Walks per-region top-1 candidates in input order, applies heading-tree
normalization, wires the doc shell + skip-link + ``<main>``, detects
landmarks (v1 scope: ``<address>`` for author doc-role only), and
flags these GapKinds:

  * ``missing_title`` — no Semantic.title with conf ≥ 0.70 AND no
    ``<h1>`` was present in the raw heading region inputs (i.e. before
    ``normalize_heading_levels`` promoted the first heading to h1).
  * ``author_block`` — Semantic top-1 == ``"author"`` with conf ≥ 0.60.
    Fires regardless of whether Sub-task 5 already wrapped the region
    in ``<address>``: the gap's ``fallback_html`` carries the current
    (possibly wrapped) HTML, and Stage 9c's splice replaces it with
    the Qwen-enriched ``<address>`` candidate. If Stage 9b's K=8
    candidates all fail the per-region gate, ``_apply_fallback`` is a
    no-op for AUTHOR_BLOCK and the deterministic wrap is preserved
    unchanged — no regression risk.
  * ``citation_unresolved`` — an inline ref (``Section X.Y`` / ``Figure
    N`` / ``Table N`` / ``[N]``) whose target is missing from the doc
    reference index AND a near-miss target exists (Levenshtein ≤ 2 on
    the section number, |ΔN| ≤ 1 for a numeric label). Dangling refs
    (no near-miss) and ambiguous cues ("see above") are NOT flagged
    (Plans/04 §1.4 + §2). See :mod:`.citation_detect`.
  * ``copyright_block`` / ``legal_disclaimer`` — paragraph region whose
    BERT-Semantic doc_role top-1 is ``"legal"`` with conf ≥ τ_gap
    (0.60), or whose Stage-5 payload carries a ``doc_role`` of
    ``"legal"``/``"copyright"``, and whose emitted HTML is not already
    wrapped in ``<aside>``/``<footer>``. The two kinds are split by the
    deterministic copyright sub-flag
    (:data:`~dart_semantic.qwen_specialists.prompts.COPYRIGHT_SUBFLAG_RE`,
    Plans/04 §2): sub-flag TRUE → copyright_block, FALSE →
    legal_disclaimer (with a ``legal_subkind`` discriminator).
    ``fallback_html`` is the deterministic
    ``<aside class="copyright|legal">`` wrap of the region's current
    HTML (Plans/04 §3.4).

    CAPABILITY NOTE (no-silent-fallbacks): the shipped gap_fill **v1**
    GGUF was trained before these two kinds existed — until the v2
    retrain (user-gated GPU step, Plans/08 §5) its real-runtime
    candidates for copyright_block / legal_disclaimer will be weak or
    gate-failing. That is expected and SAFE: every candidate still goes
    through the Stage 9b per-region hard gate, the 9c splice refuses
    non-``<aside>``/``<footer>`` fragments, and the honest deterministic
    ``fallback_html`` wrap carries the region — recorded in
    ``gaps_fallback``, never silently stamped as a resolution. Do not
    mistake this plumbing for model capability.

Returns ``(pre_doc, gaps_found, sub_task_log)`` — the pre_doc is a
fully-stitched :class:`AssembledDoc` ready for Stage 9c to mutate (or
ship as-is when no gaps were found).

V1 cuts honoured (per the user spec, do not widen):

  * No list continuation across regions.
  * Reference resolution is NOT performed deterministically — the
    assembler only DETECTS near-miss broken refs and defers the
    anchor-restore to the gap-fill specialist (Stage 9b/9c).
  * No ``<nav>`` wiring; the landmark sub-task (5) still emits no
    ``<aside class="legal">`` of its own — legal/copyright asides enter
    only via the gap path above (9c splice or deterministic fallback).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Any

from ..qwen_specialists.prompts import (
    COPYRIGHT_SUBFLAG_RE,
    COPYRIGHT_YEAR_RE,
    classify_legal_subkind,
)
from ..soft_reranker.types import RankedCandidate
from ..structure_graph import Region
from ..types import FeatureBlock
from .citation_detect import (
    build_reference_index,
    classify_reference,
    find_references,
)
from .fallbacks import emit_fallback
from .gold_shell_markup import (
    _wrap_semantic_class,
    collect_doc_ids,
    compute_absorption_runs,
)
from .heading_tree import assign_heading_ids, normalize_heading_levels
from .shell import (
    build_shell,
    detect_language,
    resolve_gold_absorb_mode,
    resolve_gold_shell_mode,
    resolve_narrow_table_absorb_mode,
)
from .types import AssembledDoc, GapKind, GapSlot

if TYPE_CHECKING:
    from .api import AssemblerConfig


# Matches the outermost ``<hN>`` opening tag of an HTML fragment.
_HEADING_OPEN_RE = re.compile(r"<h([1-6])\b([^>]*)>", re.IGNORECASE)
_HEADING_CLOSE_RE = re.compile(r"</h([1-6])\s*>", re.IGNORECASE)


def _extract_heading_level(html: str) -> int | None:
    """Return the integer level of the first ``<hN>`` tag, or ``None``."""
    if not html:
        return None
    m = _HEADING_OPEN_RE.search(html)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _rewrite_heading_level(html: str, new_level: int) -> str:
    """Rewrite the first ``<hN>...</hN>`` pair to ``<h{new_level}>``."""
    if not html:
        return html
    new_level = max(1, min(6, int(new_level)))
    open_m = _HEADING_OPEN_RE.search(html)
    if open_m is None:
        return html
    old_level = open_m.group(1)
    attrs = open_m.group(2) or ""
    rewritten_open = f"<h{new_level}{attrs}>"
    head = html[: open_m.start()]
    tail = html[open_m.end() :]
    # Rewrite the matching close tag (first close that matches old_level).
    close_pattern = re.compile(rf"</h{old_level}\s*>", re.IGNORECASE)
    close_m = close_pattern.search(tail)
    if close_m is not None:
        tail = tail[: close_m.start()] + f"</h{new_level}>" + tail[close_m.end() :]
    return head + rewritten_open + tail


# Detects an existing ``id="..."`` attr on the first ``<hN>`` opening tag.
_HEADING_HAS_ID_RE = re.compile(
    r'<h[1-6]\b[^>]*\bid\s*=\s*"[^"]*"',
    re.IGNORECASE,
)

# Captures the id VALUE on the first ``<hN>`` opening tag (the resolvable
# ``aria-labelledby`` target for the Phase-9 outer-section grouping).
_HEADING_ID_VALUE_RE = re.compile(
    r'<h[1-6]\b[^>]*\bid\s*=\s*"([^"]+)"',
    re.IGNORECASE,
)


def _heading_id_value(html: str) -> str | None:
    """Return the id on the fragment's first ``<hN>`` tag, or ``None``."""
    if not html:
        return None
    m = _HEADING_ID_VALUE_RE.search(html)
    return m.group(1) if m else None


def _group_regions_into_sections(
    region_html: Sequence[str],
    heading_indices: Sequence[int],
    normalized_levels: Sequence[int],
    *,
    absorbed_indices: AbstractSet[int] | None = None,
) -> str:
    """Phase 9 — outer ``<section aria-labelledby>`` document-outline grouping.

    Reconstructs a nested document outline from the index-aligned flat
    ``region_html`` list: open a ``<section aria-labelledby="{hid}">`` at each
    HEADING region (``hid`` = the id already injected on that heading's
    ``<hN>`` by Sub-task 2), and at each NEW heading close every open section
    whose grouping level is ``>=`` the new heading's normalized level; close
    all remaining open sections at EOF.

    The wrap adds ``<section>`` tags BETWEEN whole fragments and NEVER mutates
    a fragment's bytes, so every ``pass_9c`` flat string-replace splice key
    (``region_html[idx]`` / ``fallback_html`` / the legal ``region_html`` /
    ``match_text``) stays a contiguous substring of the assembled body, and the
    P1 1:1 index-stable contract holds (the region LIST is untouched; only the
    emitted HTML nests). A heading with no resolvable id is emitted un-sectioned
    (defensive; every heading gets an id from Sub-task 2 in practice).

    ``absorbed_indices`` (SEMANTIK_GOLD_ABSORB) are regions whose bytes were
    already concatenated INTO an earlier body-bearing component's box; they are
    SKIPPED here so they render exactly once (inside the box, in order), keeping
    the 1:1-region->output intent. Empty / None -> no-op (byte-identical).
    """
    level_by_idx = dict(zip(heading_indices, normalized_levels))
    absorbed = absorbed_indices or frozenset()
    parts: list[str] = []
    open_levels: list[int] = []
    for idx, html in enumerate(region_html):
        if not html:
            continue
        if idx in absorbed:
            # Already enclosed inside an earlier body-bearing component's box
            # (SEMANTIK_GOLD_ABSORB); its bytes were concatenated into the
            # anchor fragment, so re-emitting here would double-render it.
            continue
        if idx in level_by_idx:
            lvl = level_by_idx[idx]
            while open_levels and open_levels[-1] >= lvl:
                parts.append("</section>")
                open_levels.pop()
            hid = _heading_id_value(html)
            if hid:
                safe = hid.replace('"', "&quot;")
                parts.append(f'<section aria-labelledby="{safe}">')
                open_levels.append(lvl)
            parts.append(html)
        else:
            parts.append(html)
    while open_levels:
        parts.append("</section>")
        open_levels.pop()
    return "".join(parts)


def _inject_heading_id(html: str, ident: str) -> str:
    """Add ``id="ident"`` to the first ``<hN>`` opening tag if absent."""
    if not html or not ident:
        return html
    if _HEADING_HAS_ID_RE.search(html):
        return html
    open_m = _HEADING_OPEN_RE.search(html)
    if open_m is None:
        return html
    level = open_m.group(1)
    attrs = (open_m.group(2) or "").rstrip()
    safe = ident.replace('"', "&quot;")
    new_open = f'<h{level} id="{safe}"{(" " + attrs.lstrip()) if attrs else ""}>'
    return html[: open_m.start()] + new_open + html[open_m.end() :]


def _heading_text(html: str) -> str:
    """Pull the text between the first ``<hN>`` and ``</hN>`` (stripped of inner tags)."""
    open_m = _HEADING_OPEN_RE.search(html)
    if open_m is None:
        return ""
    close_m = _HEADING_CLOSE_RE.search(html, open_m.end())
    if close_m is None:
        return ""
    inner = html[open_m.end() : close_m.start()]
    # Strip inner tags crudely.
    text = re.sub(r"<[^>]+>", "", inner)
    return text.strip()


def _wrap_address(html: str) -> str:
    """Wrap a region's outer prose tag in ``<address>...</address>``."""
    return f"<address>{html}</address>"


def _semantic_top1_for_region(
    council_state: Any | None,
    region: Region,
) -> tuple[str | None, float]:
    """Best-effort: return Semantic doc_role top-1 for the region's first FB."""
    if council_state is None:
        return (None, 0.0)
    if not region.feature_block_indices:
        return (None, 0.0)
    fb_idx = region.feature_block_indices[0]
    outputs = getattr(council_state, "outputs", None) or {}
    bert_out = outputs.get("semantic")
    if bert_out is None:
        return (None, 0.0)
    for sig in getattr(bert_out, "signals", []) or []:
        if (
            getattr(sig, "head_name", None) == "doc_role"
            and getattr(sig, "region_id", None) == fb_idx
        ):
            labels = getattr(sig, "top_k_labels", None) or []
            confs = getattr(sig, "top_k_confidences", None) or []
            if labels and confs:
                return (labels[0], float(confs[0]))
            return (None, 0.0)
    return (None, 0.0)


def _payload_doc_role(region: Region) -> str | None:
    """Read the Semantic doc_role override Stage 5 stamped on the payload."""
    payload = region.payload or {}
    role = payload.get("doc_role")
    if isinstance(role, str) and role:
        return role
    return None


def _math_reconstruct_html(region: Region) -> str | None:
    """Return the Stage-5 deterministic-reconstruction <math> fragment for a
    math region, when SEMANTIK_MATH_RECONSTRUCT is on AND Stage-5 stamped a
    gate-valid deterministic candidate. ``None`` for every other case
    (non-math region, flag off, no stamp), keeping the flag-off path
    byte-identical."""
    if getattr(region, "kind", None) != "math":
        return None
    try:
        from ..math_reconstruct.policy import resolve_math_reconstruct_mode

        if not resolve_math_reconstruct_mode():
            return None
    except Exception:
        return None
    mr = (region.payload or {}).get("math_reconstruct")
    if isinstance(mr, dict) and mr.get("method") == "deterministic":
        mathml = mr.get("mathml")
        if isinstance(mathml, str) and mathml.strip():
            return mathml
    return None


def run_pass_9a(
    top_per_region: dict[int, RankedCandidate | None],
    regions: Sequence[Region],
    feature_blocks: Sequence[FeatureBlock],
    *,
    council_state: Any | None = None,
    config: "AssemblerConfig | None" = None,
) -> tuple[AssembledDoc, list[GapSlot], dict[str, str]]:
    """Run the deterministic 9a sub-tasks.

    Parameters
    ----------
    top_per_region
        Map ``region_index -> RankedCandidate | None``. ``None`` means
        Stage 7 dropped all K candidates; the assembler emits the
        per-kind fallback for that region.
    regions
        The Stage-5 typed regions list. ``top_per_region`` keys index
        into this list.
    feature_blocks
        The Stage-2 FeatureBlock stream — used for fallback emission
        and language detection.
    council_state
        Optional :class:`CouncilState` — used for Semantic doc_role
        lookup during ``author_block`` gap detection. ``None`` skips
        that gap.
    config
        :class:`AssemblerConfig` — when ``None`` the defaults from
        :func:`api.AssemblerConfig` apply.

    Returns
    -------
    (pre_doc, gaps_found, sub_task_log)
        ``pre_doc`` carries the fully-stitched HTML; Stage 9c either
        ships it as-is (no gaps) or splices into it.
    """
    if config is None:
        # Local import keeps the module loadable when api hasn't run yet.
        from .api import AssemblerConfig

        config = AssemblerConfig()

    sub_task_log: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Sub-task 1 — Per-region HTML stream + provenance.
    # ------------------------------------------------------------------
    region_html: list[str] = []
    region_provenance_kinds: list[str] = []  # "stage6" | "fallback" | "drop"
    n_fallback = 0
    n_stage6 = 0
    for idx, region in enumerate(regions):
        if region.kind == "metadata_drop":
            # A metadata_drop region is ALWAYS dropped from the render — even if
            # Stage-6 authored prose for it. It routes to the PROSE adapter
            # (routing.py) so Stage-6 sees it, but the clean pass already decided
            # it is a running header / page number / phantom-TOC line, NOT content.
            # Without this guard the real-Stage-6 path emits the AUTHORED running
            # header ("Chapter 1 Foundations 27") as a <p> (the empty
            # fallback_metadata_drop only fires when Stage-6 produced no candidate).
            region_html.append("")
            region_provenance_kinds.append("drop")
            continue
        # T4 — DETERMINISTIC-FIRST math. When SEMANTIK_MATH_RECONSTRUCT is on
        # and Stage-5 stamped a gate-valid deterministic <math> candidate on
        # this math region, prefer it over the Stage-6 generative candidate
        # (owner directive: deterministic-first). Counted in the fallback
        # family (a deterministic, non-LLM emitter) so the per_region_emit
        # drop arithmetic stays correct. Flag OFF -> no stamp -> this is a
        # no-op and the legacy selection runs (byte-identical).
        det_math = _math_reconstruct_html(region)
        if det_math is not None:
            region_html.append(det_math)
            region_provenance_kinds.append("fallback")
            n_fallback += 1
            continue
        ranked = top_per_region.get(idx)
        if ranked is not None and ranked.candidate.text:
            region_html.append(ranked.candidate.text)
            region_provenance_kinds.append("stage6")
            n_stage6 += 1
        else:
            # Pass cell-role BERT output through to fallback_table so the
            # deterministic table emitter uses per-cell <th scope=> from the
            # cell-role signal (Plans/06 §7 / task #15). None for non-table
            # regions or tables without the BERT signal — the fallback handles
            # None as the legacy "first row is header" path.
            cell_roles = None
            if region.kind == "table":
                cell_roles = (region.payload or {}).get("cell_roles")
            html_frag = emit_fallback(
                region,
                feature_blocks,
                cell_roles=cell_roles,
            )
            region_html.append(html_frag)
            # metadata_drop emits an empty string — preserve as a "drop".
            region_provenance_kinds.append("drop" if not html_frag else "fallback")
            if html_frag:
                n_fallback += 1
    sub_task_log["per_region_emit"] = (
        f"stage6={n_stage6} fallback={n_fallback} drop={len(regions) - n_stage6 - n_fallback}"
    )

    # ------------------------------------------------------------------
    # Sub-task 1b — Gold-standard per-region ARIA wrap (gated;
    # SEMANTIK_GOLD_SHELL). POST-SELECTION step over the SELECTED
    # ``region_html[idx]`` AFTER the stage6-LLM-HTML / deterministic-fallback
    # two-lane if/else above, so the wrap covers BOTH lanes (the re-typed
    # EXAMPLE/TRY-IT blocks return via the Stage-6 prose seat). Each container
    # mints its own resolvable label id (deduped against the accumulating
    # doc-id set). Wraps add tags AROUND the fragment and NEVER mutate fragment
    # bytes, so the 1:1 index-stable emit order and every pass_9c flat
    # string-replace splice key are preserved. Flag-off -> byte-identical.
    # ------------------------------------------------------------------
    #
    # Body-region ABSORPTION (the stopgap; gated on SEMANTIK_GOLD_ABSORB, which
    # additionally requires SEMANTIK_GOLD_SHELL) is a PRE-step here: for each
    # body-bearing component region (worked_example / definition_region /
    # exercise) it computes the contiguous run ``[i .. end)`` of following
    # sibling regions to pull into the box, then the wrap encloses the
    # CONCATENATION of ``region_html[i:end]`` (raw fragment bytes, never
    # mutated). Absorbed indices keep their original bytes in ``region_html``
    # (so every pass_9c splice key survives as a contiguous substring INSIDE the
    # box) but are SKIPPED here (no per-region wrap) and skipped again by the
    # Sub-task-6 section grouping (no double emit). Absorb OFF -> ``absorb_runs``
    # / ``absorbed_indices`` empty -> this loop is byte-identical to today.
    absorbed_indices: set[int] = set()
    if resolve_gold_shell_mode():
        doc_ids = collect_doc_ids(region_html)
        absorb_runs: dict[int, int] = {}
        # Branch precedence (table-v2 Phase 3): full-absorb (regroup inert) ->
        # narrow-table-absorb (regroup firing + sub-flag) -> no absorb. The two
        # are mutually exclusive — resolve_gold_absorb_mode short-circuits False
        # whenever the regroup fires (shell.py), and resolve_narrow_table_absorb_mode
        # requires the SAME regroup-firing composite, so they never co-fire. The
        # narrow absorb runs POST-Stage-6 on RENDERED BYTES: the in-box passthrough
        # table keeps its Stage-4 source_region_id (the absorb moves bytes, never
        # the Region) and no post-9a document-level pass keys on the table's
        # top-level position.
        if resolve_gold_absorb_mode():
            absorb_runs = compute_absorption_runs(regions, region_html)
        elif resolve_narrow_table_absorb_mode():
            absorb_runs = compute_absorption_runs(
                regions, region_html, passthrough_only=True
            )
        # Hoisted out of the branches so absorbed_indices is derived from
        # WHICHEVER absorb_runs resolved — else the narrow branch would leave it
        # empty and the table would double-render (standalone + in-box).
        absorbed_indices = {
            idx
            for start, end in absorb_runs.items()
            for idx in range(start + 1, end)
        }
        n_wrapped = 0
        for idx, region in enumerate(regions):
            if idx in absorbed_indices:
                # Already enclosed in an earlier anchor's box; do NOT wrap or
                # emit separately. Original bytes are left untouched so they
                # stay a contiguous substring of the box (splice-key safety).
                continue
            end = absorb_runs.get(idx)
            if end is not None:
                inner = "".join(region_html[i] for i in range(idx, end))
            else:
                inner = region_html[idx]
            wrapped = _wrap_semantic_class(inner, region, doc_ids=doc_ids)
            if wrapped != region_html[idx]:
                n_wrapped += 1
            region_html[idx] = wrapped
        sub_task_log["gold_shell_wrap"] = f"wrapped={n_wrapped}"
        if absorbed_indices:
            sub_task_log["gold_shell_absorb"] = (
                f"anchors={len(absorb_runs)} absorbed={len(absorbed_indices)}"
            )
            if resolve_narrow_table_absorb_mode():
                # table-v2 Phase 2 audit row: anchor region_index -> the
                # source_region_ids of the passthrough regions it absorbed, so
                # the Phase-5 in-box-table count + over-capture check are
                # mechanical (not manual HTML grepping).
                narrow_audit = {
                    start: [
                        getattr(regions[idx], "source_region_id", None)
                        for idx in range(start + 1, end)
                    ]
                    for start, end in absorb_runs.items()
                }
                sub_task_log["narrow_table_absorb"] = str(narrow_audit)

    # ------------------------------------------------------------------
    # Sub-task 2 — Heading-tree normalization (over HEADING regions only).
    # ------------------------------------------------------------------
    heading_indices: list[int] = []
    raw_levels: list[int] = []
    for idx, region in enumerate(regions):
        if region.kind != "heading":
            continue
        # Prefer Stage-5's ``level_hint`` over the candidate's ``<hN>``
        # tag: the font-ratio + numbered-section signal in
        # :func:`dart_semantic.structure_graph._level_hint` is
        # structurally more reliable than what Stage 6 generates
        # (mock runtime always emits ``<h2>``; the trained Qwen prose
        # head is content-focused, not structural). Fall back to the
        # candidate tag only if Stage 5 didn't stamp one.
        payload = region.payload or {}
        lvl_hint = payload.get("level_hint")
        if lvl_hint is not None:
            lvl = int(lvl_hint)
        else:
            lvl = _extract_heading_level(region_html[idx]) or 2
        heading_indices.append(idx)
        raw_levels.append(lvl)
    normalized = normalize_heading_levels(raw_levels)
    sub_task_log["heading_normalize"] = (
        f"in={raw_levels} out={normalized}" if heading_indices else "no headings"
    )
    # Apply the normalization back to the HTML stream, then assign
    # WCAG 2.4.5 anchor IDs over the post-normalization heading texts.
    heading_tree: list[tuple[int, str]] = []
    for region_idx, new_lvl in zip(heading_indices, normalized):
        old_html = region_html[region_idx]
        new_html = _rewrite_heading_level(old_html, new_lvl)
        region_html[region_idx] = new_html
        heading_tree.append((new_lvl, _heading_text(new_html)))
    heading_ids = assign_heading_ids([t for _, t in heading_tree])
    for region_idx, ident in zip(heading_indices, heading_ids):
        region_html[region_idx] = _inject_heading_id(
            region_html[region_idx],
            ident,
        )

    # ------------------------------------------------------------------
    # Sub-task 3 — Title detection (Plans/04 §1.5 ladder; v1 simplified).
    # ------------------------------------------------------------------
    title_text: str | None = None
    title_source = "none"
    # (1) Region whose payload carries doc_role == "title" (stamped by
    # Stage 5 on the paragraph, or Semantic.title for a heading region).
    for idx, region in enumerate(regions):
        role = _payload_doc_role(region)
        if role == "title":
            text = (region.payload or {}).get("text") or ""
            text = text.strip()
            if text:
                title_text = text
                title_source = "doc_role:title"
                break
    # (2) First <h1> in the normalized output.
    if title_text is None:
        for region_idx, lvl in zip(heading_indices, normalized):
            if lvl == 1:
                t = _heading_text(region_html[region_idx])
                if t:
                    title_text = t
                    title_source = "first_h1"
                break
    # (3) Default.
    if title_text is None:
        title_text = "Untitled document"
        title_source = "default"
    sub_task_log["title_detect"] = f"source={title_source}"

    # ------------------------------------------------------------------
    # Sub-task 4 — Language detection.
    # ------------------------------------------------------------------
    lang = detect_language(feature_blocks)
    sub_task_log["language_detect"] = f"lang={lang}"

    # ------------------------------------------------------------------
    # Sub-task 5 — Landmark wiring (v1: <address> for author doc_role only).
    # ------------------------------------------------------------------
    n_address = 0
    for idx, region in enumerate(regions):
        # Restrict to paragraph regions: wrapping a heading in <address>
        # is bizarre WCAG-wise (an h-tag inside <address> isn't a
        # contact block), and the new author_block gap-fill (Sub-task 8)
        # would splice <address>Mock Author</address> over the heading
        # — destroying the <hN> tag and producing a heading-tree skip
        # at Stage 10. Council mis-classifies some headings as author
        # (e.g. "a School of Electronic Information..."); leave those
        # alone here and let the structure_graph head re-classify.
        if region.kind != "paragraph":
            continue
        role = _payload_doc_role(region)
        sem_top, sem_conf = _semantic_top1_for_region(council_state, region)
        is_author_payload = role == "author"
        is_author_semantic = (
            sem_top == "author" and sem_conf >= config.landmark_confidence_threshold
        )
        if not (is_author_payload or is_author_semantic):
            continue
        html_frag = region_html[idx]
        if html_frag.startswith("<address"):
            continue
        region_html[idx] = _wrap_address(html_frag)
        n_address += 1
    sub_task_log["landmarks"] = f"address={n_address}"

    # ------------------------------------------------------------------
    # Sub-task 6 — Reading-order DOM placement.
    # The structure_graph already returned regions in reading order, so the
    # flat lane is a straight concatenate. Under the gold shell
    # (SEMANTIK_GOLD_SHELL) we additionally reconstruct the OUTER
    # ``<section aria-labelledby>`` document-outline grouping over the
    # index-aligned region list (Phase 9) — opening a section at each heading
    # and closing sections of level >= the new heading. This MUST run AFTER
    # Sub-task 2's heading-id injection (so ``aria-labelledby`` points at a
    # real id) and after Sub-task 5's <address> wrap (so the grouped fragment
    # bytes are final). The wrap nests emitted HTML, never the region list, and
    # never mutates a fragment's bytes -> flag-off byte-identical + every
    # pass_9c splice key survives.
    # ------------------------------------------------------------------
    if resolve_gold_shell_mode():
        body_html = _group_regions_into_sections(
            region_html, heading_indices, normalized,
            absorbed_indices=absorbed_indices,
        )
    else:
        body_html = "".join(html for html in region_html if html)

    # ------------------------------------------------------------------
    # Sub-task 7 — Doc shell.
    # ------------------------------------------------------------------
    doc_open, doc_close = build_shell(
        lang=lang,
        title=title_text,
        fabricated_title=(title_source == "default"),
    )
    full_html = doc_open + body_html + doc_close

    # ------------------------------------------------------------------
    # Sub-task 8 — Gap detection (v1: missing_title + author_block only).
    # ------------------------------------------------------------------
    gaps_found: list[GapSlot] = []

    # missing_title — no doc_role:title source AND no <h1> appeared in
    # the heading region INPUTS. ``normalize_heading_levels`` always
    # promotes the first heading to h1, so checking ``normalized`` makes
    # the gap structurally unreachable for any doc with at least one
    # heading region (Plans/04 §1.4: "no <h1> survived" means the input
    # had no h1, not that normalization produced one).
    has_h1 = any(lvl == 1 for lvl in raw_levels)
    has_title_role = title_source == "doc_role:title"
    if not has_h1 and not has_title_role:
        # Build context — the gap-fill prompt wants first paragraph
        # text + doc lang.
        first_para = ""
        for idx, region in enumerate(regions):
            if region.kind == "paragraph":
                t = (region.payload or {}).get("text") or ""
                t = t.strip()
                if t:
                    first_para = t[:500]
                    break
        gaps_found.append(
            GapSlot(
                kind=GapKind.MISSING_TITLE,
                region_index=-1,
                context={
                    "first_paragraph_text": first_para,
                    "doc_lang": lang,
                },
                detected_by="9a:title_ladder",
                semantic_top1=None,
                semantic_confidence=None,
                fallback_html=(
                    '<title data-dart-fabricated="title">Untitled document</title>'
                    '<h1 data-dart-fabricated="title">Untitled document</h1>'
                ),
            )
        )

    # author_block — Semantic top-1 == author with conf ≥ τ_gap (0.60).
    # Restricted to paragraph regions: see Sub-task 5 comment for why
    # heading regions are skipped (gap splice would destroy the <hN>).
    if council_state is not None:
        for idx, region in enumerate(regions):
            if region.kind != "paragraph":
                continue
            sem_top, sem_conf = _semantic_top1_for_region(council_state, region)
            if sem_top != "author":
                continue
            if sem_conf < config.gap_detection_threshold:
                continue
            # Fire the gap even if Sub-task 5 already wrapped this region in
            # <address>: ``fallback_html`` carries the current region HTML
            # (wrapped for sem_conf ≥ landmark_threshold, un-wrapped for the
            # narrow [0.60, 0.70) band). Stage 9c's _splice_author_block does
            # html.replace(fallback_html, <enriched-address>, 1), so either
            # shape splices cleanly. If all K candidates fail the per-region
            # gate, _apply_fallback is a no-op for AUTHOR_BLOCK and the
            # deterministic wrap is preserved.
            html_frag = region_html[idx]
            raw_text = (region.payload or {}).get("text") or ""
            gaps_found.append(
                GapSlot(
                    kind=GapKind.AUTHOR_BLOCK,
                    region_index=idx,
                    context={
                        "raw_text": raw_text[:1000],
                        "surrounding_context": "",
                        "semantic_confidence": sem_conf,
                    },
                    detected_by="9a:semantic_landmark",
                    semantic_top1=sem_top,
                    semantic_confidence=sem_conf,
                    fallback_html=html_frag,
                )
            )
    # citation_unresolved — an inline reference (Section X.Y / Figure N /
    # Table N / [N]) whose target is MISSING from the doc index but a
    # NEAR-MISS target exists (Plans/04 §1.4 + §2). Built on the heading
    # section-number index (+ caption/bib present-sets); dangling refs
    # (no near-miss) and ambiguous cues ("see above") are NOT flagged.
    #
    # The index is built from the post-normalization heading texts and
    # their assigned anchor ids, plus a scan of every region's prose text
    # for caption-defining ``Figure N``/``Table N`` leads and ``[N]``
    # bibliography leads. References are searched in paragraph-region text
    # only (headings/captions are reference TARGETS, not sources).
    heading_texts_for_index = [t for _, t in heading_tree]
    region_texts: list[str] = []
    for region in regions:
        payload = region.payload or {}
        rtext = payload.get("text")
        if not rtext:
            items = payload.get("items")
            if items:
                rtext = " ".join((it.get("text") or "") for it in items)
        region_texts.append((rtext or "").strip())
    ref_index = build_reference_index(
        heading_texts_for_index,
        heading_ids,
        region_texts=region_texts,
    )
    n_citation = 0
    for idx, region in enumerate(regions):
        if region.kind != "paragraph":
            continue
        text = region_texts[idx]
        if not text:
            continue
        for ref in find_references(text):
            cgap = classify_reference(ref, text, ref_index)
            if cgap is None:
                continue
            gaps_found.append(
                GapSlot(
                    kind=GapKind.CITATION_UNRESOLVED,
                    region_index=idx,
                    context={
                        "match_text": cgap.ref.match_text,
                        "surrounding_50_chars": cgap.surrounding_50_chars,
                        "candidate_targets": cgap.candidate_targets,
                        "reference_kind": cgap.ref.reference_kind,
                    },
                    detected_by="9a:reference_resolution",
                    semantic_top1=None,
                    semantic_confidence=None,
                    # No deterministic anchor splice without a Qwen choice:
                    # the no-op fallback is "leave the plain text unchanged"
                    # (Plans/04 §3.4) — which is exactly the body HTML's
                    # current state, so fallback_html is the original text
                    # span (used by 9c only as the replace() search key when
                    # a candidate is chosen; never auto-spliced).
                    fallback_html=None,
                )
            )
            n_citation += 1

    # copyright_block / legal_disclaimer — BERT-Semantic top-1 ==
    # "legal" with conf ≥ τ_gap (0.60), or a Stage-5 payload doc_role of
    # "legal"/"copyright"; the two kinds are split by the deterministic
    # copyright sub-flag regex (Plans/04 §2). Restricted to paragraph
    # regions (same rationale as author_block: a gap splice over a
    # heading would destroy the <hN>) whose emitted HTML is not already
    # wrapped in <aside>/<footer> by the prose top-1.
    #
    # NOTE: until the gap_fill v2 retrain, real-runtime candidates for
    # these kinds are weak (v1 GGUF never saw them); the per-region hard
    # gate + the deterministic fallback_html wrap below carry them, and
    # gaps_fallback records it. See the module docstring CAPABILITY NOTE.
    n_legal = 0
    for idx, region in enumerate(regions):
        if region.kind != "paragraph":
            continue
        text = region_texts[idx]
        if not text:
            continue
        role = _payload_doc_role(region)
        sem_top, sem_conf = _semantic_top1_for_region(council_state, region)
        is_legal_payload = role in ("legal", "copyright")
        is_legal_semantic = sem_top == "legal" and sem_conf >= config.gap_detection_threshold
        if not (is_legal_payload or is_legal_semantic):
            continue
        html_frag = region_html[idx]
        lead = html_frag.lstrip().lower()
        if lead.startswith("<aside") or lead.startswith("<footer"):
            # Prose top-1 already produced the wrap (Plans/04 §2 gate).
            continue
        is_copyright = bool(COPYRIGHT_SUBFLAG_RE.search(text))
        kind = GapKind.COPYRIGHT_BLOCK if is_copyright else GapKind.LEGAL_DISCLAIMER
        context: dict[str, Any] = {
            "raw_text": text[:1000],
            "surrounding_context": "",
            "semantic_confidence": sem_conf,
            # 9c splice search key — the region's current body HTML
            # (NOT part of the prompt envelope; build_gap_fill_request
            # reads only the per-kind keys above/below).
            "region_html": html_frag,
        }
        if kind is GapKind.COPYRIGHT_BLOCK:
            aside_class = "copyright"
            context["has_copyright_year_match"] = bool(COPYRIGHT_YEAR_RE.search(text))
        else:
            aside_class = "legal"
            context["legal_subkind"] = classify_legal_subkind(text)
        gaps_found.append(
            GapSlot(
                kind=kind,
                region_index=idx,
                context=context,
                detected_by="9a:semantic_legal",
                semantic_top1=sem_top if is_legal_semantic else role,
                semantic_confidence=sem_conf if is_legal_semantic else None,
                # Deterministic fallback (Plans/04 §3.4): wrap the region's
                # current HTML in the canonical aside. Honest path — real
                # text, deterministic structure, recorded in gaps_fallback.
                fallback_html=(f'<aside class="{aside_class}">{html_frag}</aside>'),
            )
        )
        n_legal += 1

    sub_task_log["gap_detect"] = (
        f"found={[g.kind.value for g in gaps_found]}" if gaps_found else "no gaps"
    )

    # ------------------------------------------------------------------
    # Assemble the AssembledDoc.
    # The locked field type ``region_provenance: list[int]`` carries
    # the original region index for each emitted block (i.e. an
    # identity permutation in v1). The "stage6 vs fallback" annotation
    # lives in ``sub_task_log["per_region_emit"]`` instead.
    # ------------------------------------------------------------------
    landmarks: dict[str, int] = {
        "main": 1,
        "address": n_address,
        "nav": 0,
        "footer": 0,
        "aside": 0,
    }
    # Gold shell adds a STATIC ``<footer role="contentinfo">`` landmark
    # (`gold_shell_markup.GOLD_DOC_CLOSE`). Account for it under the gold
    # branch ONLY so the flag-off landmarks dict stays byte-identical. The
    # dynamic ``nav``/``aside``/``region`` counts from the Phase-7/9 wraps
    # already increment as those wraps emit, so they are NOT bumped here
    # (no double-count).
    if resolve_gold_shell_mode():
        landmarks["footer"] = 1

    region_provenance = list(range(len(regions)))

    # Per-region emitted HTML, index-aligned with region_provenance. Theta's
    # semantic_preservation scorer reads this to map each region to its exact
    # candidate text (avoids the fragile re-parse of the joined html, which
    # misaligns when top-level child count != region count). Stashed in
    # sub_task_log to avoid changing the locked AssembledDoc field set; pass_9c
    # carries sub_task_log forward.
    sub_task_log["region_html"] = list(region_html)

    anchors: dict[str, int] = {}
    for region_idx, ident in zip(heading_indices, heading_ids):
        if ident and ident not in anchors:
            anchors[ident] = region_idx

    pre_doc = AssembledDoc(
        html=full_html,
        gaps_found=list(gaps_found),
        gaps_resolved=[],
        gaps_fallback=[],
        heading_tree=heading_tree,
        landmarks=landmarks,
        anchors=anchors,
        region_provenance=region_provenance,
        sub_task_log=sub_task_log,
    )
    return pre_doc, gaps_found, sub_task_log


__all__ = ["run_pass_9a", "_extract_heading_level"]
