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

    @classmethod
    def from_model_dir(cls, model_dir: Path | None = None) -> "V2Config":
        """Build a config populated from the on-disk VENDORED weights.

        This is the resolver the in-process LOCAL cascade needs. The
        cascade does NOT consume these adapter-path fields to *load* the
        models — the council BERTs resolve their own paths
        (``council/<name>/final`` via :func:`dart_semantic.paths.resolve_model`)
        and the Stage-6 Qwen specialists read ``qwen_specialists/config.yaml``
        — but :func:`dart_semantic.cascade.run_pipeline_v2` reads
        :meth:`is_qwen_specialist_loaded` to AUTO-DERIVE ``runtime_mode``:
        ``DEFAULT_V2_CONFIG`` (all-None) makes it default to ``"mock"`` even
        when the real GGUFs are on disk, so the local cascade silently
        produces mock HTML the Ed4All seam then refuses to ship.

        Each field is set to its canonical vendored path ONLY when that
        artifact actually exists on disk, and left ``None`` otherwise — so a
        partial install degrades gracefully (a missing specialist leaves its
        slot None) instead of fabricating a path that would mislead the gate.
        Path resolution goes through :func:`dart_semantic.paths.model_dir`
        (honors ``SEMANTIK_MODEL_DIR`` / ``SEMANTIK_HOME``); ``model_dir`` may
        be passed explicitly for tests.

        The Qwen GGUF paths are read from ``qwen_specialists/config.yaml``
        (the SAME source ``make_runtime('real')`` checks for presence) so the
        gate and the live load agree on which specialists exist.
        """
        # Imported here (not at module top) so importing v2_config stays
        # cheap and dependency-light, matching the paths-module contract.
        from . import paths as _paths

        root = Path(model_dir) if model_dir is not None else _paths.model_dir()

        def _dir(rel: str) -> Path | None:
            p = root / rel
            return p if p.exists() else None

        # Council adapter directories (LoRA + heads.pt + tokenizer). These
        # mirror each council module's DEFAULT_ADAPTER_DIR
        # (``resolve_model("council/<name>/...")``), so a populated council
        # slot reflects the artifact the council actually loads.
        council_math_detector = _dir("council/structure/final")
        council_role = _dir("council/structure/final")
        council_span = _dir("council/structure/final")
        council_order = _dir("council/merge_or_split/final")
        council_math_class = _dir("council/math_specialist/final")
        # table_specialist ships final_v5 as its canonical adapter (see
        # council/table_specialist.py::DEFAULT_ADAPTER_DIR), with final as the
        # legacy fallback.
        council_table_class = _dir("council/table_specialist/final_v5") or _dir(
            "council/table_specialist/final"
        )
        council_semantic = _dir("council/semantic/final")

        # Qwen specialist GGUFs — read the same config.yaml the Stage-6
        # presence check consults so the runtime_mode gate and the live load
        # never disagree. A missing/unreadable config or a null adapter_path
        # leaves the slot None (graceful partial install).
        qwen_paths = cls._scan_qwen_gguf_paths(root)

        # Theta semantic-preservation cross-encoder head (Stage-12). v8 is the
        # current vendored version; absence leaves the slot None (theta-stub).
        theta = _dir("theta/semantic_preservation/v8")

        return cls(
            council_math_detector_adapter=council_math_detector,
            council_span_adapter=council_span,
            council_order_adapter=council_order,
            council_role_adapter=council_role,
            council_math_class_adapter=council_math_class,
            council_table_class_adapter=council_table_class,
            council_semantic_adapter=council_semantic,
            qwen_prose_adapter=qwen_paths.get("prose"),
            qwen_table_adapter=qwen_paths.get("table"),
            qwen_math_adapter=qwen_paths.get("math"),
            qwen_gap_fill_adapter=qwen_paths.get("gap_fill"),
            theta_semantic_scorer_path=theta,
        )

    @staticmethod
    def _scan_qwen_gguf_paths(model_root: Path) -> dict[str, Path]:
        """Return ``{adapter_name: gguf_path}`` for each on-disk Qwen GGUF.

        Reads ``qwen_specialists/config.yaml`` (the single source of truth the
        Stage-6 ``make_runtime('real')`` presence check uses) and resolves each
        ``adapter_path`` literal through the model root, mirroring
        :func:`dart_semantic.qwen_specialists.runtime._scan_adapter_artifacts`.
        Only entries whose GGUF exists on disk are included; a missing config /
        yaml dep / null path / absent file is silently skipped so a partial
        install degrades gracefully.
        """
        config_path = (
            Path(__file__).resolve().parent
            / "qwen_specialists"
            / "config.yaml"
        )
        try:
            import yaml  # noqa: WPS433 — lazy; absent yaml just yields {}
        except ImportError:
            return {}
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except (OSError, ValueError):
            return {}
        if not isinstance(cfg, dict):
            return {}
        adapters = cfg.get("adapters") or {}
        out: dict[str, Path] = {}
        for name, entry in adapters.items():
            raw = (entry or {}).get("adapter_path") if isinstance(entry, dict) else None
            if not raw:
                continue
            # The config stores ``models/<...>.gguf`` literals; strip a single
            # leading ``models/`` segment and rejoin against the resolved model
            # root (matches paths.resolve_model_literal / runtime._scan).
            rel = Path(raw)
            if rel.is_absolute():
                p = rel
            else:
                parts = rel.parts
                if parts and parts[0] == "models":
                    rel = Path(*parts[1:]) if len(parts) > 1 else Path()
                p = model_root / rel
            if p.exists():
                out[name] = p
        return out


DEFAULT_V2_CONFIG = V2Config()


def resolve_local_v2_config(model_dir: Path | None = None) -> V2Config:
    """Convenience wrapper: build the populated LOCAL config from disk.

    Equivalent to :meth:`V2Config.from_model_dir`; provided as a module-level
    helper so callers (the Ed4All in-process conversion seam) read a single
    intention-revealing name. ``model_dir`` defaults to
    :func:`dart_semantic.paths.model_dir` (honors ``SEMANTIK_MODEL_DIR`` /
    ``SEMANTIK_HOME``).
    """
    return V2Config.from_model_dir(model_dir)
