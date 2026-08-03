"""Council data contracts.

These are the typed signals produced by individual BERTs in the council
(Stage 3 of the v2 pipeline) and the aggregate state passed to Stage 4
(cross-BERT reranker).

Phase 0 scope: dataclass definitions only. No logic. No model imports.

Phase-4 (v1 cross-BERT reranker) extension: :class:`RoutingDecision`
is the per-region output of Stage 4's rule-based arbitration over a
:class:`CouncilState`. See ``semantik_structure/council/cross_reranker.py``
for the rules (architecture.md §3.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class TypedSignal:
    """One BERT's per-region output for one head.

    A BERT may emit multiple TypedSignals per region (e.g. a multi-head
    BERT producing both a coarse and fine-grained label).
    """
    head_name: str                          # e.g. "block_role" | "math_inline_vs_display"
    region_id: int                          # index into the Phase-2 region stream
    top_k_labels: list[str]                 # ranked, length k (typically 3)
    top_k_confidences: list[float]          # parallel to top_k_labels, in [0, 1]
    feature_provenance: dict[str, Any] = field(default_factory=dict)
    # ^ which features fed this prediction; per-BERT shape (e.g. font, geometry, OCR text).


@dataclass(frozen=True)
class BertOutput:
    """All TypedSignals produced by one BERT for one document."""
    bert_name: str                          # e.g. "structure" | "semantic" | "table_specialist"
    signals: list[TypedSignal]
    backbone_version: str = ""              # e.g. "deberta-v3-base@hf-rev"
    adapter_version: str = ""               # e.g. "role-lora-v1"


@dataclass(frozen=True)
class CouncilState:
    """The aggregate of every BERT's output for one document.

    Consumed by Stage 4 (cross-BERT reranker), Stage 5 (structure graph),
    and threaded through to Stage 9 (assembler) for provenance.

    ``signal_coverage`` is observability metadata populated by the
    orchestrator. For each council BERT (head) it records ``expected``
    (the number of FeatureBlocks/regions the head should have scored),
    ``observed`` (the number of distinct ``region_id``s the head
    actually emitted at least one signal for), and ``missing``
    (``expected - observed``, never negative). When a head fails to
    load or run entirely, the entry will be ``expected=N, observed=0``
    and an ``error`` key holds the exception's repr for debugging.
    Per the no-silent-fallbacks invariant, downstream eval code
    surfaces nonzero ``missing`` aggregates so partial-failure runs
    can't masquerade as healthy.
    """
    outputs: dict[str, BertOutput]          # keyed by bert_name
    document_id: str = ""
    council_version: str = ""               # e.g. "council-config-1.0"
    signal_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Region candidate types — canonical definitions live in region_detection.py
# ---------------------------------------------------------------------------
# Re-exported here so council consumers can import the typed-candidate
# hierarchy without reaching into the deeper module. Phase 0 originally
# defined a speculative RegionCandidate here; Phase 1 settled the canonical
# shape in `semantik_structure.region_detection` (kind, bbox, pages,
# evidence_features, member_block_indices) along with TableCandidate and
# MathCandidate subclasses carrying cell-grids and glyph-density features.
# That is now the single source of truth.
# RELATIVE imports (``..`` = the ``semantik_structure`` package) so the
# typed-candidate classes bind to the SAME module object the rest of the
# cascade builds candidates from. The cascade is importable under TWO
# top-level names — bare ``semantik_structure`` (the canonical
# ``run_cascade_json.py`` / ``SemantiK/`` on sys.path route) AND the
# package-prefixed ``SemantiK.semantik_structure`` (the in-process MCP seam
# ``_run_semantik_v2_conversion`` route). An ABSOLUTE ``from
# semantik_structure.region_detection import ...`` here would always bind the
# bare module, while ``cascade``/``features`` (loaded via relative imports
# under whichever top-level name) build candidates from the package-local
# module — splitting class identity so ``isinstance(region, TableCandidate)``
# / ``ImageCandidate`` in the cross-reranker returned False and EVERY table /
# figure candidate fell through to the prose route (the 0-tables / 0-figures
# regression). Relative imports bind the package-local module either way.
from ..region_detection import (  # noqa: E402
    ImageCandidate,
    MathCandidate,
    RegionCandidate,
    TableCandidate,
)
# Stage-5 typed region (deterministic structure-graph candidate). Re-
# exported here so council consumers can pull the post-arbitration
# Region type without reaching into the semantik_structure package root.
from ..structure_graph import Region  # noqa: E402


@dataclass(frozen=True)
class RoutingDecision:
    """Stage 4 (cross-BERT reranker) v1 output, one per region.

    ``region_id`` indexes into the **detected region stream** — i.e.
    the ``regions`` list passed to
    :func:`~semantik_structure.council.cross_reranker.arbitrate`, which in
    v1 is ``table_candidates + math_candidates`` (see
    :func:`~semantik_structure.council.orchestrator.run_council`). It is
    **not** an index into FeatureBlocks. Prose spans do **not** get
    RoutingDecisions in v1; per-span structure decisions are produced
    downstream by Stage 5 directly from Structure's per-span
    :class:`TypedSignal`s on
    ``CouncilState.outputs['structure'].signals`` (whose own
    ``region_id`` *is* the FeatureBlock index).

    The ``route`` is the downstream specialist track:

    * ``"prose"``  — flat-text path (Qwen prose specialist).
    * ``"table"``  — confirmed table; cells already parsed by
                     :class:`TableSpecialist` upstream.
    * ``"math"``   — confirmed math; consumed by the math specialist.
    * ``"figure"`` — DETERMINISTIC image-candidate route (an
                     :class:`ImageCandidate` extracted from a PDF IMAGE
                     page-object). The deterministic detector is
                     LOAD-BEARING here — Structure's ``is_image_block``
                     head is a secondary/confirmation strength flag, not
                     the trigger. Gated by ``SEMANTIK_DETECT_FIGURES``.
    * ``"drop"``   — region demoted out of the structure stream entirely
                     (header/footer artefact, page-number, etc.). v1
                     never emits ``drop`` from the rule arbiter; the
                     value is reserved for assemblers / future reranker
                     versions.

    ``final_role`` is the canonical role label after arbitration — for
    a prose region this is typically the winning ``structural_role`` or
    Semantic ``doc_role`` label; for a table it's ``"table"``; for math
    it's ``"math"``; for a ``drop`` it is ``None``. Consumers that need
    raw per-head distributions should pull them off the
    :class:`CouncilState` instead of decoding ``final_role``.

    ``flags`` enumerates which §3.2 rules fired during arbitration. The
    canonical strings (kept in sync with cross_reranker.py):

        ``"low_confidence"``                  top-1 < 0.5
        ``"math_matrix_override"``            math wins overlap with table
        ``"image_block_demoted"``             image-block detected by Structure
        ``"detector_demoted"``                table detector rejected region
        ``"math_detector_stand_in_demoted"``  math detector stand-in rejected
        ``"semantic_override"``               Semantic doc_role overrode Structure
        ``"no_signals"``                      no BERT emitted any signal
        ``"detector_signal_missing"``         detector should have run but didn't
    """
    region_id: int
    route: Literal["prose", "table", "math", "figure", "drop"]
    final_role: str | None = None
    flags: tuple[str, ...] = ()


__all__ = [
    "TypedSignal",
    "BertOutput",
    "CouncilState",
    "RoutingDecision",
    "RegionCandidate",
    "TableCandidate",
    "MathCandidate",
    "ImageCandidate",
    "Region",
]
