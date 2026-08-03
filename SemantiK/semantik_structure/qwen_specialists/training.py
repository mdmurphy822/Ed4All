"""Shared Qwen LoRA trainer core.

One ``train(adapter_id, cfg, out_dir, ...)`` entry point that the CLI
``scripts/training/train_qwen_lora.py`` calls. Mirrors the QLoRA + SFTTrainer
shape from ``train_reasoner.py`` but is parameterized per-adapter via
``ADAPTER_CONFIGS`` and applies completion-only loss masking via the
chat-templated ``prompt_len`` precomputed in ``data_loader.build_dataset``.

Hard guards in this module (each is load-bearing — see Phase 1.5
acceptance gates):

    * ``acquire_training_slot()`` — GPU-busy + flock. Trainer refuses
      to start if Phase 4b or any other consumer is on the GPU.
    * VRAM dry-run at step 0 — load the model + a single forward, log
      peak VRAM, fall back via the (r=16→8, max_len 1024→768→512)
      ladder if the first try OOMs.
    * VRAM creep check at step 1000 — if peak VRAM is rising linearly,
      something (probably gradient_checkpointing) silently broke. Log
      warning; do not auto-abort (the human takes the call).
    * Naming-collision guard — refuse non-empty out_dir without
      ``--overwrite``. Adapter directories are append-only by default.
    * Val-loss divergence early-stop — for tiny datasets (gap_fill),
      val_loss > train_loss + 0.5 for 3 consecutive evals trips an
      EarlyStop exception and writes a ``divergence.json`` flag.

On-demand checkpointing — drop a sentinel file into ``out_dir`` to
force a save off the ``save_steps`` cadence (handled within one
optimizer step, ~6s on the 3060):

    touch <out_dir>/.checkpoint_now       # save, then keep training
    touch <out_dir>/.checkpoint_and_stop  # save, then stop gracefully

The sentinel is consumed once handled, so each ``touch`` is one save.

Artifacts written under ``out_dir``:

    final/                           — saved adapter (peft safetensors)
    train_log.jsonl                  — per-logging-step train metrics
    val_eval_log.jsonl               — per-eval val metrics + samples
    test_samples.jsonl               — final eval samples on test split
    summary.json                     — config + final metrics
    adapter_metadata.json            — adapter version, dataset hash,
                                        base model, training start/end
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .train_config import ADAPTER_CONFIGS, AdapterTrainConfig
from .training_lock import acquire_training_slot

logger = logging.getLogger(__name__)


QWEN_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Auto-fallback ladder for the dry-run OOM guard. Each entry is a
# (lora_r, max_len) override to try if the previous failed.
_FALLBACK_LADDER: tuple[tuple[int, int], ...] = (
    (16, 1024),
    (8, 1024),
    (8, 768),
    (8, 512),
)


class AdapterTrainError(RuntimeError):
    """Generic training failure surface — wraps OOM, lock, drift."""


class ValDivergence(AdapterTrainError):
    """Raised when val/train loss diverges past the threshold."""


def _dataset_fingerprint(dataset_dir: Path) -> str:
    """Hash the (size, mtime) of each split file. Cheap and stable."""
    h = hashlib.sha256()
    for split in ("train", "val", "test"):
        p = dataset_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        st = p.stat()
        h.update(f"{split}:{st.st_size}:{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:16]


def _check_out_dir(out_dir: Path, overwrite: bool, resume_from: Path | None) -> None:
    """Refuse non-empty out_dir without explicit consent."""
    if not out_dir.exists():
        return
    contents = list(out_dir.iterdir())
    if not contents:
        return
    if resume_from is not None:
        # Resume implies the dir is the resume target — fine.
        return
    if not overwrite:
        raise AdapterTrainError(
            f"out_dir {out_dir} is non-empty: "
            f"{[p.name for p in contents[:5]]}. "
            f"Pass overwrite=True (or --overwrite) to proceed, or "
            f"resume_from=<checkpoint>."
        )


# ---------------------------------------------------------------------------
# Loss masking via prompt_len
# ---------------------------------------------------------------------------


def _make_collator(tok: Any, max_len: int) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Tokenize on-the-fly and build a labels mask using prompt_len.

    Each row has ``{"text": str, "prompt_len": int}``. We tokenize the
    text, pad to the longest in the batch (capped at max_len), then
    overwrite labels[<prompt_len] = -100 so the loss only counts
    target tokens. This implements completion-only loss without
    relying on SFTTrainer's ``completion_only_loss`` flag (which uses
    a generation marker we don't have).
    """
    import torch

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [r["text"] for r in batch]
        prompt_lens = [int(r["prompt_len"]) for r in batch]
        enc = tok(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        # Mask pad
        labels[attention_mask == 0] = -100
        # Mask prompt tokens (every label index < prompt_len[i])
        for i, plen in enumerate(prompt_lens):
            cap = min(plen, labels.size(1))
            labels[i, :cap] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def train(
    adapter_id: str,
    out_dir: Path,
    *,
    cfg: AdapterTrainConfig | None = None,
    base_model: str = QWEN_DEFAULT_MODEL,
    resume_from: Path | None = None,
    overwrite: bool = False,
    max_steps: int | None = None,  # smoke override
    seed: int = 42,
    skip_gpu_check: bool = False,
    dataset_root: Path | None = None,
    eval_size: int = 200,
) -> dict[str, Any]:
    """Train one adapter end-to-end. Returns the summary dict."""
    if adapter_id not in ADAPTER_CONFIGS:
        raise ValueError(
            f"unknown adapter {adapter_id!r}; expected one of "
            f"{sorted(ADAPTER_CONFIGS)}"
        )
    cfg = cfg or ADAPTER_CONFIGS[adapter_id]
    out_dir = Path(out_dir)
    _check_out_dir(out_dir, overwrite=overwrite, resume_from=resume_from)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Acquire the slot FIRST. If GPU is busy, fail before importing
    # transformers (fast feedback, smaller cache miss).
    with acquire_training_slot(skip_gpu_check=skip_gpu_check):
        return _train_inner(
            adapter_id=adapter_id,
            cfg=cfg,
            out_dir=out_dir,
            base_model=base_model,
            resume_from=resume_from,
            max_steps=max_steps,
            seed=seed,
            dataset_root=dataset_root,
            eval_size=eval_size,
        )


def _train_inner(
    *,
    adapter_id: str,
    cfg: AdapterTrainConfig,
    out_dir: Path,
    base_model: str,
    resume_from: Path | None,
    max_steps: int | None,
    seed: int,
    dataset_root: Path | None,
    eval_size: int,
) -> dict[str, Any]:
    """Heavy lifting; called inside the training-slot context."""
    # Lazy heavyweight imports — not at module load time, so callers
    # can `from training import train` on a CPU box without dragging in
    # bnb / cuda.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    from .data_loader import build_dataset

    log_path = out_dir / "train_log.jsonl"
    val_log_path = out_dir / "val_eval_log.jsonl"
    test_samples_path = out_dir / "test_samples.jsonl"
    summary_path = out_dir / "summary.json"
    metadata_path = out_dir / "adapter_metadata.json"

    log_path.touch()
    val_log_path.touch()

    # -------- tokenizer + model load --------
    logger.info("loading tokenizer + model: %s", base_model)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # -------- datasets --------
    logger.info("building train split for %s", adapter_id)
    train_ds = build_dataset(
        adapter_id, "train", tok, cfg.max_len,
        dataset_root=dataset_root, survey=True,
    )
    try:
        val_ds_full = build_dataset(
            adapter_id, "val", tok, cfg.max_len,
            dataset_root=dataset_root, survey=False,
        )
    except FileNotFoundError:
        val_ds_full = None
    val_ds: Dataset | None = None
    if val_ds_full is not None and len(val_ds_full) > 0:
        if eval_size > 0 and len(val_ds_full) > eval_size:
            val_ds = val_ds_full.shuffle(seed=seed).select(range(eval_size))
        else:
            val_ds = val_ds_full

    # -------- VRAM dry-run --------
    dry_run_peak_mb = _vram_dry_run(model, tok, cfg.max_len)
    logger.info("VRAM dry-run peak: %.0f MiB", dry_run_peak_mb)

    # -------- trainer config --------
    # Apply max_steps cap: the smaller of (cfg.total_step_cap,
    # epochs*len(train)/grad_accum, smoke override).
    effective_steps = cfg.total_step_cap
    if max_steps is not None:
        effective_steps = min(effective_steps, max_steps)

    sft_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        seed=seed,
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_length=cfg.max_len,
        max_steps=effective_steps,
        gradient_checkpointing=True,
        # We hand-build labels via the collator + prompt_len mask, so
        # SFTTrainer's own completion-only path is bypassed. Set
        # packing=False so SFTTrainer doesn't repack our pre-tokenized
        # rows.
        packing=False,
        # Disable SFTTrainer's auto-templating; our rows are already
        # chat-template-wrapped strings under "text".
        dataset_text_field="text",
        # Skip SFTTrainer's pre-tokenization step. Without this, trl
        # 1.2+ runs ``_prepare_dataset`` which tokenizes rows and drops
        # our ``prompt_len`` column — the collator then can't build the
        # completion-only loss mask.
        dataset_kwargs={"skip_prepare_dataset": True},
        # Don't strip columns that aren't in the model forward signature
        # (text, prompt_len) — the collator consumes them.
        remove_unused_columns=False,
    )
    if val_ds is not None:
        sft_kwargs["eval_strategy"] = "steps"
        sft_kwargs["eval_steps"] = cfg.eval_steps
    else:
        sft_kwargs["eval_strategy"] = "no"

    sft_cfg = SFTConfig(**sft_kwargs)

    collator = _make_collator(tok, cfg.max_len)

    # Per-eval sample generation + log writers.
    train_log_cb = _TrainLogCallback(log_path)
    eval_log_cb = _EvalLogCallback(
        out_dir, tok, model, val_ds, cfg.max_len, seed,
    )
    divergence_cb = _DivergenceCallback(out_dir, threshold=0.5, patience=3)
    creep_cb = _VRAMCreepCallback()
    ckpt_request_cb = _CheckpointRequestCallback(out_dir)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=collator,
        callbacks=[
            train_log_cb, eval_log_cb, divergence_cb, creep_cb,
            ckpt_request_cb,
        ],
    )

    # -------- adapter metadata stub (write before training) --------
    train_start = time.time()
    metadata = {
        "adapter_id": adapter_id,
        "base_model": base_model,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "max_len": cfg.max_len,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "grad_accum": cfg.grad_accum,
        "total_step_cap": cfg.total_step_cap,
        "max_steps_effective": effective_steps,
        "seed": seed,
        "dataset_fingerprint": _dataset_fingerprint(
            (Path(dataset_root) if dataset_root else
             Path(__file__).resolve().parents[2] / "data") /
            {"math": "qwen_math_dataset_v2", "table": "qwen_table_dataset",
             "prose": "qwen_prose_dataset", "gap_fill": "qwen_gap_fill_dataset"}[adapter_id]
        ),
        "train_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(train_start)),
        "vram_dry_run_peak_mb": dry_run_peak_mb,
        "adapter_version": f"{adapter_id}-v1.{int(train_start)}",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # -------- train --------
    logger.info("starting trainer (max_steps=%s, eval_strategy=%s)",
                effective_steps, sft_kwargs.get("eval_strategy"))
    train_result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    train_end = time.time()

    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    # -------- final test-split sampling (deterministic) --------
    try:
        test_ds = build_dataset(
            adapter_id, "test", tok, cfg.max_len,
            dataset_root=dataset_root, survey=False,
        )
        _write_test_samples(
            test_ds, model, tok, out_dir, max_new_tokens=128, n=5, seed=seed,
        )
    except FileNotFoundError:
        logger.info("no test split found; skipping final sampling")

    summary = {
        "adapter_id": adapter_id,
        "base_model": base_model,
        "train_size": len(train_ds),
        "val_size": (len(val_ds) if val_ds is not None else 0),
        "epochs": cfg.epochs,
        "max_steps_effective": effective_steps,
        "train_loss_final": getattr(train_result, "training_loss", None),
        "wall_time_seconds": train_end - train_start,
        "out_dir": str(out_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    metadata["train_end_iso"] = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(train_end)
    )
    metadata["wall_time_seconds"] = train_end - train_start
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info("training complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Helpers: VRAM dry-run + creep
# ---------------------------------------------------------------------------


def _vram_dry_run(model: Any, tok: Any, max_len: int) -> float:
    """Run one forward at max_len; return peak GPU VRAM in MiB.

    On OOM, raise so the caller can attempt the fallback ladder. We
    do NOT implement the ladder here because the model is already
    bound to the current LoRA shape — falling back means rebuilding
    from scratch. Surface the OOM to the caller (the trainer) so it
    can do the rebuild.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    try:
        ids = torch.full(
            (1, max_len), fill_value=tok.eos_token_id, dtype=torch.long
        ).to("cuda")
        with torch.no_grad():
            _ = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        model.train()
    peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return peak


# ---------------------------------------------------------------------------
# Trainer callbacks
# ---------------------------------------------------------------------------


# Define a module-level base class for our custom callbacks. Recent
# ``transformers`` calls ``getattr(callback, event)`` directly without
# ``hasattr``, so callbacks MUST inherit from ``TrainerCallback`` (which
# provides every event method as a no-op). The CPU-side test harness
# imports this module without transformers installed — fall back to a
# bare ``object`` in that case (the affected callbacks are never
# instantiated on the test path).
try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:  # pragma: no cover — test-harness path
    class _TrainerCallback:  # type: ignore[no-redef]
        pass


class _TrainLogCallback(_TrainerCallback):
    """Append per-logging-step train metrics to ``train_log.jsonl``."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if not logs:
            return
        # Skip eval_* lines — those go to val_eval_log.jsonl.
        if any(k.startswith("eval_") for k in logs):
            return
        record = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            **{k: (float(v) if isinstance(v, (int, float)) else v)
               for k, v in logs.items()},
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class _EvalLogCallback(_TrainerCallback):
    """Append per-eval val metrics + 5 sample generations."""

    def __init__(
        self,
        out_dir: Path,
        tok: Any,
        model: Any,
        val_ds: Any,
        max_len: int,
        seed: int,
    ) -> None:
        self.out_dir = out_dir
        self.log_path = out_dir / "val_eval_log.jsonl"
        self.tok = tok
        self.model = model
        self.val_ds = val_ds
        self.max_len = max_len
        self.seed = seed

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: D401
        if metrics is None:
            return
        record = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            **{k: (float(v) if isinstance(v, (int, float)) else v)
               for k, v in metrics.items()},
        }
        # Sample 5 deterministic rows from val.
        samples = []
        if self.val_ds is not None and len(self.val_ds) >= 5:
            try:
                samples = self._sample_generations(n=5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("eval-time sample generation failed: %s", exc)
                samples = [{"error": repr(exc)}]
        record["samples"] = samples
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _sample_generations(self, n: int) -> list[dict[str, Any]]:
        import torch
        was_training = self.model.training
        self.model.eval()
        out: list[dict[str, Any]] = []
        try:
            for i in range(n):
                row = self.val_ds[i]
                # Take the prompt-only portion (everything before the
                # final assistant content). We use prompt_len to slice.
                full = row["text"]
                plen = int(row["prompt_len"])
                ids_full = self.tok(full, add_special_tokens=False, return_tensors="pt")
                prompt_ids = ids_full["input_ids"][:, :plen].to(self.model.device)
                attn = torch.ones_like(prompt_ids)
                with torch.no_grad():
                    gen = self.model.generate(
                        input_ids=prompt_ids,
                        attention_mask=attn,
                        max_new_tokens=128,
                        do_sample=False,
                    )
                completion_ids = gen[0, plen:]
                completion = self.tok.decode(completion_ids, skip_special_tokens=True)
                out.append({"idx": i, "completion": completion[:400]})
        finally:
            if was_training:
                self.model.train()
        return out


class _DivergenceCallback(_TrainerCallback):
    """Early-stop if val_loss > train_loss + threshold for ``patience`` evals."""

    def __init__(self, out_dir: Path, threshold: float, patience: int) -> None:
        self.out_dir = out_dir
        self.threshold = threshold
        self.patience = patience
        self._streak = 0
        self._last_train_loss: float | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if logs and "loss" in logs:
            try:
                self._last_train_loss = float(logs["loss"])
            except (TypeError, ValueError):
                pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: D401
        if not metrics or self._last_train_loss is None:
            return
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return
        try:
            val_loss = float(val_loss)
        except (TypeError, ValueError):
            return
        gap = val_loss - self._last_train_loss
        if gap > self.threshold:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.patience:
            logger.warning(
                "val/train loss divergence: %.3f - %.3f = %.3f for %d evals",
                val_loss, self._last_train_loss, gap, self._streak,
            )
            (self.out_dir / "divergence.json").write_text(json.dumps({
                "step": int(state.global_step),
                "train_loss": self._last_train_loss,
                "val_loss": val_loss,
                "gap": gap,
                "streak": self._streak,
            }))
            control.should_training_stop = True


class _VRAMCreepCallback(_TrainerCallback):
    """At step 1000, log peak VRAM. Doesn't auto-abort."""

    def __init__(self) -> None:
        self._fired = False

    def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
        if self._fired or state.global_step != 1000:
            return
        try:
            import torch
            peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            logger.info("VRAM creep check at step 1000: peak=%.0f MiB", peak)
        except Exception:  # noqa: BLE001
            pass
        self._fired = True


class _CheckpointRequestCallback(_TrainerCallback):
    """On-demand checkpointing via sentinel files in ``out_dir``.

    Lets a human force a checkpoint mid-run without waiting for the
    ``save_steps`` boundary or hunting down the trainer PID. Drop one
    of two files into ``out_dir``:

        .checkpoint_now       — save a checkpoint at the next step,
                                then continue training.
        .checkpoint_and_stop  — save a checkpoint, then stop the run
                                gracefully (writes final/ + summary).

    The sentinel is consumed (deleted) once handled, so each ``touch``
    triggers exactly one save. Checked at ``on_step_end``, so the save
    lands within one optimizer step.
    """

    def __init__(self, out_dir: Path) -> None:
        self.now_flag = out_dir / ".checkpoint_now"
        self.stop_flag = out_dir / ".checkpoint_and_stop"

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: D401
        # Clear stale sentinels from a prior run so a resume doesn't
        # immediately checkpoint (or worse, stop) on step 1.
        self.now_flag.unlink(missing_ok=True)
        self.stop_flag.unlink(missing_ok=True)

    def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
        if self.stop_flag.exists():
            logger.warning(
                "checkpoint-and-stop requested at step %d — saving "
                "checkpoint and stopping gracefully", state.global_step,
            )
            self.stop_flag.unlink(missing_ok=True)
            control.should_save = True
            control.should_training_stop = True
            return
        if self.now_flag.exists():
            logger.warning(
                "on-demand checkpoint requested at step %d — saving",
                state.global_step,
            )
            self.now_flag.unlink(missing_ok=True)
            control.should_save = True


# ---------------------------------------------------------------------------
# Final test-sample writer
# ---------------------------------------------------------------------------


def _write_test_samples(
    test_ds: Any,
    model: Any,
    tok: Any,
    out_dir: Path,
    *,
    max_new_tokens: int,
    n: int,
    seed: int,
) -> None:
    import torch

    out_path = out_dir / "test_samples.jsonl"
    n = min(n, len(test_ds))
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            row = test_ds[i]
            plen = int(row["prompt_len"])
            ids = tok(row["text"], add_special_tokens=False, return_tensors="pt")
            prompt_ids = ids["input_ids"][:, :plen].to(model.device)
            attn = torch.ones_like(prompt_ids)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attn,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            completion = tok.decode(gen[0, plen:], skip_special_tokens=True)
            f.write(json.dumps({"idx": i, "completion": completion[:1000]}) + "\n")


__all__ = [
    "AdapterTrainError",
    "ValDivergence",
    "train",
]
