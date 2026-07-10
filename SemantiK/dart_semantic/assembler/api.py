"""Stage 9 — Deterministic document assembler. Public API.

Three sub-passes (Plans/04 §1, §3, §4):

  * **9a deterministic** — concatenate per-region top-1 (or per-kind
    fallback when Stage 7 dropped all K), normalize the heading tree,
    wire the doc shell, detect all five GapKinds (``missing_title``,
    ``author_block``, ``citation_unresolved``, ``copyright_block``,
    ``legal_disclaimer`` — see :mod:`.pass_9a` for per-kind triggers
    and the v1-GGUF capability note on the legal kinds).
  * **9b GapFill** — drive K=8 mock-runtime generations through the
    GAP_FILL adapter; gate each candidate.
  * **9c merge** — score per gap, splice the argmax candidate, fall
    back to deterministic strings on zero-survivor gaps.

V1 cuts (do NOT widen — see user spec): no list continuation, no
deterministic reference resolution, no <nav> wiring, no heading-tree
re-run after splice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..soft_reranker.types import RankedCandidate
from ..structure_graph import Region
from ..types import FeatureBlock
from .types import AssembledDoc


@dataclass(frozen=True)
class AssemblerConfig:
    """Runtime-tunable knobs (Plans/04 defaults)."""

    skip_gap_fill: bool = False
    gap_fill_k: int = 8
    landmark_confidence_threshold: float = 0.70
    gap_detection_threshold: float = 0.60


def assemble_document(
    top_per_region: dict[int, RankedCandidate | None],
    regions: Sequence[Region],
    feature_blocks: Sequence[FeatureBlock],
    *,
    council_state: Any | None = None,
    runtime_mode: Literal["mock", "real"] = "mock",
    config: AssemblerConfig = AssemblerConfig(),
    config_path: Path | None = None,
    validator: Any | None = None,
) -> AssembledDoc:
    """Run all 3 sub-passes (9a + 9b + 9c) and return the final doc.

    When ``config.skip_gap_fill`` is True OR Stage 9a found zero gaps,
    the 9b/9c passes are skipped and 9a's output is returned as-is.

    ``validator``, when supplied, is reused by Stage 9b's gap-candidate
    gate. The eval / smoke drivers own one ``HtmlValidator`` for the
    whole cascade; passing it through avoids a nested
    ``sync_playwright().start()`` (Playwright crashes with
    "Sync API inside the asyncio loop" on the second open).
    """
    from .pass_9a import run_pass_9a
    from .pass_9b import run_pass_9b
    from .pass_9c import run_pass_9c

    pre, gaps_found, _sub_task_log = run_pass_9a(
        top_per_region,
        regions,
        feature_blocks,
        council_state=council_state,
        config=config,
    )
    if config.skip_gap_fill or not gaps_found:
        return _finalize_heading_contiguity(pre)

    candidates_per_gap = run_pass_9b(
        gaps_found,
        runtime_mode=runtime_mode,
        config=config,
        config_path=config_path,
        validator=validator,
    )
    post = run_pass_9c(
        pre,
        gaps_found,
        candidates_per_gap,
        regions,
        feature_blocks,
        config=config,
    )
    return _finalize_heading_contiguity(post)


def _count_structural_heading_drift(doc: AssembledDoc, new_html: str) -> int:
    """Count STRUCTURAL headings whose level the whole-doc contiguity pass
    CHANGED (ITEM4 Phase 3 diagnostic — expected 0).

    A structural heading is a ``sub_task_log['region_html']`` fragment carrying a
    Sub-task-2-injected ``<hN id="...">``. The id survives the contiguity pass
    (it re-levels only, never touches ids), so we match by id and compare the
    level. A change means the region-blind doc pass overrode a level the
    containment builder owns — a builder bug, surfaced loudly (not asserted).
    """
    import re as _re

    region_html = (doc.sub_task_log or {}).get("region_html") or []
    open_re = _re.compile(r"<h([1-6])\b([^>]*)>", _re.IGNORECASE)
    id_re = _re.compile(r'\bid\s*=\s*"([^"]+)"')
    drift = 0
    for frag in region_html:
        if not frag:
            continue
        om = open_re.search(frag)
        if om is None:
            continue
        idm = id_re.search(om.group(2) or "")
        if idm is None:
            continue
        hid = idm.group(1)
        orig_level = int(om.group(1))
        nm = _re.search(
            r'<h([1-6])\b[^>]*\bid\s*=\s*"' + _re.escape(hid) + r'"',
            new_html,
            _re.IGNORECASE,
        )
        if nm is not None and int(nm.group(1)) != orig_level:
            drift += 1
    return drift


def _finalize_heading_contiguity(doc: AssembledDoc) -> AssembledDoc:
    """Re-level EVERY ``<hN>`` in the final HTML to a contiguous tree.

    Stage 9a's heading-tree normalization only re-levels the STRUCTURAL
    heading regions; a Stage-6 specialist (notably the hosted 70B prose
    seat) can emit ``<hN>`` tags *inside* a non-heading region's body
    (e.g. ``<h6>EXAMPLE 1.6</h6>`` after an ``<h2>``), which bypasses
    that pass and produces a level skip the Stage-10 ``heading_tree``
    document gate rejects. This deterministic document-order pass closes
    that gap by judging the WHOLE document's heading levels in context.

    Idempotent: a doc with an already-contiguous heading tree is returned
    unchanged (``html`` byte-identical, ``heading_tree`` untouched). When
    a rewrite occurs, ``heading_tree`` is re-derived from the corrected
    HTML so the metadata matches the emitted levels.

    ITEM4 Phase 3 — this whole-doc pass is DEMOTED to verification + embedded-tag
    repair for STRUCTURAL headings: the containment builder owns structural
    heading levels (pass_9a Sub-task 2 applies ``tree.levels``), so if this pass
    re-levels a heading that IS a structural heading region (matched by its
    Sub-task-2-injected id), that is a builder disagreement — counted as
    ``sub_task_log["heading_contiguity_structural_drift"]`` (expected 0, a loud
    diagnostic NOT an assert). Re-levels of specialist-EMBEDDED ``<hN>`` tags
    (the only thing this region-blind pass can see that the tree cannot) remain
    silent normal operation.
    """
    from ..containment import resolve_containment_mode
    from .heading_contiguity import normalize_document_heading_levels
    from .pass_9a import _HEADING_CLOSE_RE, _HEADING_OPEN_RE

    new_html = normalize_document_heading_levels(doc.html)
    if resolve_containment_mode():
        drift = _count_structural_heading_drift(doc, new_html)
        doc.sub_task_log = dict(doc.sub_task_log or {})
        doc.sub_task_log["heading_contiguity_structural_drift"] = drift
    if new_html == doc.html:
        return doc

    # Re-derive (level, text) in document order from the corrected HTML so
    # downstream consumers (theta nav scoring, conformance audit) see the
    # emitted levels. Crude inner-text strip mirrors pass_9a._heading_text.
    import re as _re

    tree: list[tuple[int, str]] = []
    for open_m in _HEADING_OPEN_RE.finditer(new_html):
        lvl = int(open_m.group(1))
        close_m = _HEADING_CLOSE_RE.search(new_html, open_m.end())
        inner = new_html[open_m.end() : close_m.start()] if close_m else ""
        text = _re.sub(r"<[^>]+>", "", inner).strip()
        tree.append((lvl, text))

    doc.html = new_html
    doc.heading_tree = tree
    doc.sub_task_log = dict(doc.sub_task_log or {})
    doc.sub_task_log["heading_contiguity"] = "doc-order re-level applied"
    return doc


__all__ = ["AssemblerConfig", "assemble_document"]
