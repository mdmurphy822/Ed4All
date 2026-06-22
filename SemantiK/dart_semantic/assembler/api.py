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
        return pre

    candidates_per_gap = run_pass_9b(
        gaps_found,
        runtime_mode=runtime_mode,
        config=config,
        config_path=config_path,
        validator=validator,
    )
    return run_pass_9c(
        pre,
        gaps_found,
        candidates_per_gap,
        regions,
        feature_blocks,
        config=config,
    )


__all__ = ["AssemblerConfig", "assemble_document"]
