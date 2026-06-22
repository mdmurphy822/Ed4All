"""Assembler data contracts.

Shapes locked by Plans/04_assembler_layer_investigation.md §10. Field
names and types here are the source of truth for downstream phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GapKind(str, Enum):
    MISSING_TITLE = "missing_title"
    CITATION_UNRESOLVED = "citation_unresolved"
    AUTHOR_BLOCK = "author_block"
    COPYRIGHT_BLOCK = "copyright_block"
    LEGAL_DISCLAIMER = "legal_disclaimer"


@dataclass
class GapSlot:
    """One detected gap the assembler needs Qwen-GapFill (Phase 9b) to fill.

    Per Plans/04 §10.2 — `context` is per-kind (see §2 table); validation
    happens at gap-fill prompt-construction time.
    """

    kind: GapKind
    region_index: int  # -1 for whole-doc slots (e.g. missing_title)
    context: dict[str, Any]  # per-kind shape; see Plans/04 §2
    detected_by: str  # e.g. "9a:title_ladder" | "9a:reference_resolution"
    semantic_top1: str | None = None  # BERT-Semantic doc-role that triggered this (if any)
    semantic_confidence: float | None = None
    fallback_html: str | None = None  # deterministic splice if all K gap-fill candidates fail
    resolved_html: str | None = None  # the spliced winning candidate (set by 9c) — token-level
    # provenance for theta's hallucinated-structure anchoring


class CertificationStatus(str, Enum):
    SHIP_WITH_CONFIDENCE = "ship-with-confidence"
    OFFLINE_QWEN_LANE = "offline-qwen-lane"
    NON_CERTIFIED = "non-certified"


@dataclass
class AssembledDoc:
    """Assembler output — the post-9c HTML (or post-9a if no gaps) plus diagnostics."""

    html: str
    gaps_found: list[GapSlot]  # what 9a flagged; empty list if clean
    gaps_resolved: list[GapSlot]  # what 9c successfully filled
    gaps_fallback: list[GapSlot]  # what 9c filled with deterministic fallbacks
    heading_tree: list[tuple[int, str]]  # (level, text) in document order
    landmarks: dict[str, int]  # {"main": 1, "nav": 1, "footer": 0, "aside": 2, ...}
    anchors: dict[str, str]  # {"sec-3-2": "<h2 id='sec-3-2'>...</h2>", ...}
    region_provenance: list[int]  # index back into Phase 8 top-1 regions per emitted block
    sub_task_log: dict[str, str]  # per sub-task one-liner diagnostics


@dataclass
class DocCandidate:
    """One assembled document under consideration by Phase 11's reranker.

    `gate_result` is typed as a forward-string to break the circular
    import with `dart_semantic.gates.types.GateResult` — the gates
    package depends on no other dart_semantic package; assembler depends
    on gates' types but only via this string annotation.
    """

    assembled: AssembledDoc
    source: str  # "fast-lane" | "offline-lane"
    gate_result: "GateResult"  # owned by dart_semantic.gates.types
    soft_score: float | None = None  # composite from Plans/04 §6.2; None until ranker runs
    soft_axes: dict[str, float] | None = None  # per-axis breakdown for diagnostics
    confidence: float | None = None  # final §7.1 confidence (set when ranker has chosen)


@dataclass
class ExitDecision:
    """The Stage-12 verdict — which exit was chosen and why."""

    status: CertificationStatus
    chosen: DocCandidate  # the candidate selected to ship
    fast_lane_attempted: bool
    offline_lane_attempted: bool
    reasons: list[str] = field(default_factory=list)
    sidecar_json: dict[str, Any] = field(default_factory=dict)
