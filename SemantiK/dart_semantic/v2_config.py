"""v2 pipeline configuration — names which BERTs and Qwen specialists are loaded.

Each per-phase rollout flips a `None` to a concrete adapter path. Phase
0 ships defaults of None for every slot, so nothing model-shaped is
loaded by default and the v2 path is identical to v1 (which goes through
`pipeline_v2.run(pdf, mode='v1')`).

Hot-reload is intentionally not supported: changing this config requires
re-instantiating `V2Config` and passing it into a fresh `pipeline_v2.run`
call. The dataclass is frozen so accidental mutation is loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class V2Config:
    """Which v2 components are loaded for a given pipeline run.

    All fields default to None; the v1-mode passthrough does not consult
    any of these. Setting a path enables that component for v2-mode (once
    the corresponding rollout phase has implemented it).
    """

    # Council BERTs (Phase 2 / Phase 3 rollout). Each is the disk path
    # of a LoRA adapter directory.
    council_math_detector_adapter: Path | None = None
    council_span_adapter: Path | None = None
    council_order_adapter: Path | None = None
    council_role_adapter: Path | None = None
    council_math_class_adapter: Path | None = None
    council_table_class_adapter: Path | None = None
    council_semantic_adapter: Path | None = None

    # Qwen specialists (Phase 5 / 6 / 6c). LoRA adapter paths over the
    # shared Qwen 4B base.
    qwen_prose_adapter: Path | None = None
    qwen_table_adapter: Path | None = None
    qwen_math_adapter: Path | None = None
    qwen_gap_fill_adapter: Path | None = None

    # Cross-encoders (Phase 8 / 10 soft rerankers).
    soft_region_reranker_path: Path | None = None
    soft_document_reranker_path: Path | None = None

    # Theta scorer (Phase 11). DeBERTa-v3-small fine-tune.
    theta_semantic_scorer_path: Path | None = None

    # Shared backbones.
    council_backbone_hf_name: str = "microsoft/deberta-v3-base"
    cross_encoder_backbone_hf_name: str = "microsoft/deberta-v3-small"
    qwen_base_hf_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Free-form per-phase overrides (e.g. sampling temp, top_k). Empty in
    # Phase 0; populated by per-phase rollout PRs.
    overrides: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def is_council_loaded(self) -> bool:
        """True if any council adapter is configured."""
        return any(
            getattr(self, name) is not None
            for name in (
                "council_math_detector_adapter",
                "council_span_adapter",
                "council_order_adapter",
                "council_role_adapter",
                "council_math_class_adapter",
                "council_table_class_adapter",
                "council_semantic_adapter",
            )
        )

    def is_qwen_specialist_loaded(self) -> bool:
        """True if any Qwen specialist adapter is configured."""
        return any(
            getattr(self, name) is not None
            for name in (
                "qwen_prose_adapter",
                "qwen_table_adapter",
                "qwen_math_adapter",
                "qwen_gap_fill_adapter",
            )
        )


DEFAULT_V2_CONFIG = V2Config()
