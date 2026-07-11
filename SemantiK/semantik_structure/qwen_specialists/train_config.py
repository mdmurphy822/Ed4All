"""Per-adapter training hyperparameters for Phase 1.5 LoRA rollout.

Single source of truth for max_len / epochs / lr / lora_*. The trainer
CLI (``scripts/train_qwen_lora.py``) reads from this module and refuses
to run an unknown adapter id.

Sizing rationale (per architecture.md §4.2 + Plans/01 hyperparam grid):

    math      — long base prompts (~310 char), short targets (~4K char
                 of MathML). max_len 2048 (was 1024): the v1 dataset's
                 1024-token filter censored 32.5% of math elements
                 (the long-MathML tail), so the adapter never saw the
                 hardest equations. The v2 dataset (qwen_math_dataset_v2)
                 uncensors targets to 2048, recovering ~46K rows; its
                 new target max is ~1,957 tokens, so 2048 clears the
                 distribution. The builder applies a tokenizer-aware
                 filter at this same 2048 cap so the trainer sees a
                 length-coherent dataset. Epochs 1 because the dataset
                 is huge (~233K+ rows post-filter).
    table     — caption + per-cell envelope easily clears 1K tokens.
                 Small-ish dataset; epochs 3 + lower lr (1e-4) + smaller
                 LoRA r=8 to avoid overfitting.
    prose     — long source paragraphs. Big dataset; 1 epoch at 2e-4 +
                 r=16 mirrors the v1 reasoner shape that worked.
    gap_fill  — ~12.4K train rows (capped V2, all 5 kinds: missing_title
                 + author_block + citation_unresolved + copyright_block
                 + legal_disclaimer; the dominant kind is capped via
                 --cap-per-kind auto so it can't swamp the minor kinds).
                 epochs 3 + r=8 + dropout 0.10; total_step_cap sized to
                 clear 3 full epochs (see below).

``total_step_cap`` is a fail-safe ceiling on optimizer steps; it
guarantees a runaway dataset (e.g. someone changes batch size and
forgets to recheck total steps) terminates instead of training for a
week. The trainer takes ``min(epochs * len/bs/grad_accum, cap)``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterTrainConfig:
    """Hyperparameters for one LoRA adapter."""

    max_len: int
    epochs: int
    grad_accum: int
    lr: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    eval_steps: int
    save_steps: int
    total_step_cap: int


ADAPTER_CONFIGS: dict[str, AdapterTrainConfig] = {
    "math": AdapterTrainConfig(
        max_len=2048,
        epochs=1,
        grad_accum=8,
        lr=2e-4,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        eval_steps=500,
        save_steps=1000,
        total_step_cap=30000,
    ),
    "table": AdapterTrainConfig(
        # max_len 2048 (was 1024): the multi-source table set is ~2× longer
        # than prose/math — the prompt carries the full cell_grid AND the
        # target repeats every cell (p50≈2135 tok; 1024 kept only 17% of
        # tables). See Plans/06 §4. VRAM-heavier on the 8GB 3070; the dry-run
        # at step 0 will OOM *loudly* if 2048 doesn't fit (the fallback ladder
        # is unused — no silent downgrade to 1024). lora_r kept at 8 for VRAM
        # headroom at 2048; revisit r=16 if it fits. total_step_cap 1500 ≥ the
        # ~1,300 steps that 3 epochs over ~3.46K train rows needs.
        max_len=2048,
        epochs=3,
        grad_accum=8,
        lr=1e-4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,
        eval_steps=50,
        save_steps=200,
        total_step_cap=1500,
    ),
    "prose": AdapterTrainConfig(
        max_len=1024,
        epochs=1,
        grad_accum=8,
        lr=2e-4,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        eval_steps=500,
        save_steps=1000,
        total_step_cap=8000,
    ),
    "gap_fill": AdapterTrainConfig(
        max_len=512,
        epochs=3,
        grad_accum=8,
        lr=1e-4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,
        eval_steps=100,
        save_steps=200,
        # ~12.4K train rows (V2 all-5-kinds build, 2026-06-10) @ bs=1 ×
        # grad_accum=8 → ~1,547 steps/epoch, so 3 epochs ≈ 4,640 steps.
        # Cap=5500 clears 3 full epochs with headroom while still
        # catching a runaway (the previous cap=4000 was sized for the
        # 8.9K-row 3-kind build and would stop at ~86% of epoch 3).
        total_step_cap=5500,
    ),
}


__all__ = ["AdapterTrainConfig", "ADAPTER_CONFIGS"]
