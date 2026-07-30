"""SFT-D S8 — training-recipe config defaults + card-field filtering."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.training.configs import TrainingConfig, load_config  # noqa: E402


def test_qwen_recipe_defaults():
    cfg = load_config("qwen2.5-1.5b")
    # SFT recipe (runtime/scratchpad/sft_data_program.md §C).
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 2 * cfg.lora_rank
    assert cfg.learning_rate == pytest.approx(2e-4)
    assert 2 <= cfg.epochs <= 3
    assert 0.05 <= cfg.lora_dropout <= 0.1
    assert cfg.weight_decay > 0
    # bf16 LoRA is the default (QLoRA behind the flag).
    assert cfg.use_4bit is False
    # Checkpoint-selection scaffolding knobs.
    assert cfg.save_total_limit == 3
    assert cfg.completion_only_loss is True
    assert cfg.verify_loss_mask is True
    assert cfg.checkpoint_selection_metric == "gold_keypoint_coverage"


def test_dataclass_use_4bit_default_is_false():
    cfg = TrainingConfig(
        base_model="x", learning_rate=2e-4, epochs=3, lora_rank=16,
        lora_alpha=32, max_seq_length=2048, batch_size=1, seed=42,
    )
    assert cfg.use_4bit is False
    assert cfg.completion_only_loss is True


def test_to_card_dict_only_schema_keys():
    cfg = load_config("qwen2.5-1.5b")
    card = cfg.to_card_dict()
    # S8 orchestration knobs must NOT leak into the model-card training_config.
    for forbidden in (
        "base_model", "completion_only_loss", "verify_loss_mask",
        "save_total_limit", "early_stopping_patience",
        "checkpoint_selection_metric",
    ):
        assert forbidden not in card, f"{forbidden!r} leaked into the card"
    # Canonical schema keys present.
    for required in ("use_4bit", "lora_rank", "learning_rate", "epochs"):
        assert required in card


def test_merged_accepts_new_knob_override(tmp_path):
    overrides = tmp_path / "o.yaml"
    overrides.write_text(
        "use_4bit: true\nsave_total_limit: 5\ncompletion_only_loss: false\n",
        encoding="utf-8",
    )
    cfg = load_config("qwen2.5-1.5b", course_overrides=overrides)
    assert cfg.use_4bit is True
    assert cfg.save_total_limit == 5
    assert cfg.completion_only_loss is False
