"""Fine-tune a Qwen 3 4B LoRA adapter for stage 3b structural reasoning.

Input dataset: JSONL where each row is {"messages": [system, user, assistant]}
              as produced by data/builders/build_qwen_data.py.

Output: a LoRA adapter under <output-dir>/final that can be loaded via
reason.load_reasoner(path) once adapter-loading support is wired in.

Design notes:
  - 4-bit NF4 quant to fit the 3060 8GB. LoRA on q/k/v/o projections.
  - completion_only_loss=True so loss is computed only over the
    assistant tokens (the JSON array), not the system prompt or the
    user's block dump.
  - Low LR (2e-4) + warmup; short runs (1-3 epochs). Over-training the
    adapter on sparse-target examples turns Qwen into a JSON-shape
    emitter that ignores the input.

Usage:
    python train_reasoner.py \\
        --dataset-dir data/qwen_dataset_smoke \\
        --output-dir models/reasoner_smoke \\
        --epochs 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


QWEN_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=QWEN_DEFAULT_MODEL)
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="Dir with train.jsonl, val.jsonl (val optional)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="Max tokens per training sequence. Examples longer "
                         "than this get truncated. Default is sized for "
                         "pages-per-chunk=1 data (p99 < 2K tokens).")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--packing", action="store_true",
                    help="Pack short sequences into max_length windows "
                         "(Best Fit Decreasing). Big throughput win when "
                         "most sequences are shorter than max_length.")
    ap.add_argument("--subsample-train", type=int, default=None,
                    help="Cap training set to N rows after shuffle. Useful "
                         "for time-boxing a local run.")
    ap.add_argument("--eval-size", type=int, default=200,
                    help="Cap eval set to this many rows (shuffled). Full val "
                         "set is ~1300 rows and costs ~2s each on 3070 — "
                         "capping keeps total wall-time reasonable.")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: cap steps, skip eval, log more often")
    args = ap.parse_args()

    print(f"[gpu] cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # ---------- model load (4-bit) ----------
    print(f"[load] {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # attn_implementation="sdpa" routes attention through PyTorch's native
    # Flash Attention 2 kernels (available since torch 2.0, auto-selected
    # when `sdpa` is set and the hardware supports it). Much faster than
    # the default "eager" path and no external `flash-attn` wheel needed.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---------- dataset ----------
    print(f"[data] loading {args.dataset_dir}")
    data_files = {"train": str(args.dataset_dir / "train.jsonl")}
    val_path = args.dataset_dir / "val.jsonl"
    if val_path.exists():
        data_files["val"] = str(val_path)
    ds = load_dataset("json", data_files=data_files)
    if args.subsample_train:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(
            range(min(args.subsample_train, len(ds["train"])))
        )
    # Cap eval size — on a 3070 each val example costs ~2s, so a full
    # 1336-row val set would dominate total wall-time. 200 is plenty for
    # a loss signal.
    if "val" in ds and len(ds["val"]) > args.eval_size:
        ds["val"] = ds["val"].shuffle(seed=args.seed).select(range(args.eval_size))
    print(f"  splits: {list(ds.keys())}")
    for split, d in ds.items():
        print(f"  {split}: {len(d)} rows")

    # ---------- trainer ----------
    cfg_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=5 if args.smoke else 25,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_length=args.max_length,
        # completion_only_loss masks loss to only the assistant response.
        # assistant_only_loss would be stricter but requires chat templates
        # with `{% generation %}` markers, which Qwen 3's template lacks.
        # Packing is incompatible with completion_only_loss (it packs
        # multiple sequences into one window, breaking the per-sample
        # prompt/completion boundary), so disable packing in that mode.
        completion_only_loss=True,
        gradient_checkpointing=True,
        packing=args.packing and False,  # forced off; see note above
    )
    if args.smoke:
        cfg_kwargs["max_steps"] = 20
        # Eval doubles smoke wall-time for negligible signal on 20 steps.
        cfg_kwargs["eval_strategy"] = "no"
    elif "val" in ds:
        cfg_kwargs["eval_strategy"] = "epoch"

    sft_cfg = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds.get("val"),
        processing_class=tok,
    )

    print("[train] starting...")
    trainer.train()

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    summary = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "dataset_dir": str(args.dataset_dir),
        "train_size": len(ds["train"]),
        "val_size": len(ds.get("val", [])),
        "epochs": args.epochs,
        "smoke": args.smoke,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] adapter -> {final_dir}")


if __name__ == "__main__":
    main()
