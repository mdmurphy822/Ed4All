"""Wave 90 — training-config loader for ``Trainforge.training``.

YAML files in this package map base-model short names (e.g.
``qwen2.5-1.5b.yaml``) to a :class:`TrainingConfig` dataclass. The
loader merges optional course-level overrides on top so a single course
can pin a non-default LR / rank / epochs without forking the per-base
defaults.

Wave 90 ships configs for: ``qwen2.5-1.5b``, ``llama-3.2-1b``,
``smollm2-1.7b``. ``llama-3.2-3b`` and ``phi-3.5-mini`` registry
entries exist but their per-base configs land in a follow-up wave —
``load_config`` raises ``FileNotFoundError`` with a clear message so
the gap is loud, not silent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TrainingConfig:
    """Per-run training hyperparameters.

    Mirrors the ``training_config`` block of
    ``schemas/models/model_card.schema.json`` so the runner can pass
    the dataclass directly into ``model_card.training_config`` via
    :meth:`to_dict`.
    """

    base_model: str
    learning_rate: float
    epochs: int
    lora_rank: int
    lora_alpha: int
    max_seq_length: int
    batch_size: int
    seed: int
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])
    # SFT program recipe (scratchpad/sft_data_program.md §C): bf16 LoRA is
    # the DEFAULT (use_4bit=False) — the shipped adapter is a small 1.5B
    # course-tutor; QLoRA (4-bit) stays reachable behind this flag for
    # memory-constrained boxes. NB: LoRA already leaks less than full-FT;
    # keep epochs light (2-3) with early-stopping on.
    use_4bit: bool = False
    min_dpo_pairs: int = 50
    dpo_preference_filter: str = "editorial_or_misconception"
    dpo_fail_hard: bool = True

    # ------------------------------------------------------------------ #
    # S8 training-recipe orchestration knobs (NOT persisted to the model  #
    # card — the schema's training_config is additionalProperties:false;  #
    # the runner filters to the canonical schema keys before emit).       #
    # ------------------------------------------------------------------ #
    # Completion-only loss masking: train the loss over the assistant
    # completion only, never the prompt (scratchpad/sft_data_program.md §C).
    completion_only_loss: bool = True
    # Sample a batch and fail LOUD if the prompt isn't masked before the
    # full run (peft_trainer._verify_completion_only_loss_mask).
    verify_loss_mask: bool = True
    # Per-epoch checkpoint retention so the downstream-probe selector can
    # pick the best epoch out-of-band. Defaults to `epochs` when None.
    save_total_limit: Optional[int] = None
    # Early-stopping patience (epochs) for the narrow/repetitive-corpus
    # overfit guard; consumed by the checkpoint-selection scaffolding.
    early_stopping_patience: int = 1
    # Which downstream probe the checkpoint selector optimises. Canonical
    # values: "gold_keypoint_coverage" | "sympy_correctness" | "pair_loss".
    # Pair/held-out loss is only an overfit tripwire, never the selector.
    checkpoint_selection_metric: str = "gold_keypoint_coverage"


    # Fields the model_card schema's ``training_config`` object accepts
    # (schemas/models/model_card.schema.json, additionalProperties:false).
    # The S8 orchestration knobs above are deliberately NOT in this set —
    # they steer the run but don't belong in the provenance card.
    CARD_FIELDS = (
        "seed", "learning_rate", "epochs", "lora_rank", "lora_alpha",
        "max_seq_length", "batch_size", "lora_dropout",
        "gradient_accumulation_steps", "warmup_ratio", "weight_decay",
        "target_modules", "use_4bit", "min_dpo_pairs",
        "dpo_preference_filter", "dpo_fail_hard",
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_card_dict(self) -> Dict[str, Any]:
        """``to_dict`` filtered to the model_card-schema-accepted keys.

        The runner writes this into ``model_card.json::training_config`` so
        the S8 orchestration knobs never trip the schema's
        ``additionalProperties:false`` guard.
        """
        full = self.to_dict()
        return {k: full[k] for k in self.CARD_FIELDS if k in full}

    def merged(self, overrides: Dict[str, Any]) -> "TrainingConfig":
        """Return a new :class:`TrainingConfig` with ``overrides`` applied.

        Unknown keys in ``overrides`` raise ``ValueError`` so a typo
        fails loud rather than silently no-op-ing.
        """
        if not overrides:
            return self
        valid_fields = {f.name for f in fields(self)}
        unknown = set(overrides.keys()) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown TrainingConfig override key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_fields)}"
            )
        merged = self.to_dict()
        merged.update(overrides)
        # Re-cast so YAML floats aren't accidentally promoted to int etc.
        return TrainingConfig(**merged)


_CONFIG_DIR = Path(__file__).resolve().parent


def _yaml_filename_for(base_model: str) -> Path:
    """Return the canonical YAML path for a base model short-name.

    Uses the short name verbatim with ``.yaml`` appended. Wave 90's
    five registered bases all map cleanly with this rule.
    """
    return _CONFIG_DIR / f"{base_model}.yaml"


def load_config(
    base_model: str,
    course_overrides: Optional[Path] = None,
) -> TrainingConfig:
    """Load the per-base default config and merge any course overrides.

    Args:
        base_model: Short name registered in
            :data:`Trainforge.training.base_models._REGISTRY`. Typo
            here surfaces as ``FileNotFoundError`` listing the dir.
        course_overrides: Optional path to a YAML file whose top-level
            keys override the per-base defaults. Unknown keys fail loud
            via :meth:`TrainingConfig.merged`.

    Returns:
        The merged :class:`TrainingConfig`.

    Raises:
        FileNotFoundError: when no per-base YAML ships for the
            requested base. The error message points at the expected
            path so it's obvious where to drop the new config.
    """
    yaml_path = _yaml_filename_for(base_model)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"No training config for base_model={base_model!r}. "
            f"Expected {yaml_path}. Wave 90 ships configs for "
            f"qwen2.5-1.5b, llama-3.2-1b, smollm2-1.7b. Add a YAML "
            f"with the same shape to extend support."
        )
    base_payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    base_payload.setdefault("base_model", base_model)
    cfg = TrainingConfig(**base_payload)

    if course_overrides is None:
        return cfg
    overrides_path = Path(course_overrides)
    if not overrides_path.exists():
        raise FileNotFoundError(
            f"course_overrides path does not exist: {overrides_path}"
        )
    override_payload = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}
    return cfg.merged(override_payload)


__all__ = ["TrainingConfig", "load_config"]
