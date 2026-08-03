"""Train BERT-MathSpecialist (shared ModernBERT-base + LoRA, two heads).

Two heads:
    * math_type      ∈ {inline, display, numbered, multiline, matrix}    (5)
    * eq_num_assoc   ∈ {none, attached_left, attached_right, separate}   (4)

The base encoder uses bf16 while rank-16 LoRA adapters target the attention
modules, keeping the trainable parameter set compact.

Mirrors the structure of ``train_classifier.py`` (custom Trainer with
class-weighted CE + WeightedRandomSampler oversample) but replaces the
single-head ``AutoModelForSequenceClassification`` with a multi-head
wrapper around the council :class:`SharedBackbone`.

Outputs:
    models/council/math_specialist/final/adapter_model.safetensors
    models/council/math_specialist/final/adapter_config.json
    models/council/math_specialist/final/heads.pt           # nn.Linear weights
    models/council/math_specialist/final/tokenizer/         # HF tokenizer
    models/council/math_specialist/final/summary.json       # held-out F1s

Usage:
    python train_math_specialist.py
    python train_math_specialist.py --epochs 6 --lora-r 8
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

import sys

# This trainer lives under SemantiK/training/; ensure the SemantiK root is on
# the import path so the top-level ``data`` namespace resolves regardless of
# the invocation directory.
_SEMANTIK_ROOT = str(Path(__file__).resolve().parent.parent)
if _SEMANTIK_ROOT not in sys.path:
    sys.path.insert(0, _SEMANTIK_ROOT)

from data.builders.build_math_specialist_data import (  # noqa: E402  (after sys.path bootstrap)
    EQ_NUM_ASSOC,
    EQ_NUM_TO_ID,
    MATH_TYPE_TO_ID,
    MATH_TYPES,
)


def _label_count_table(labels: list[int], names: list[str]) -> str:
    counts = Counter(labels)
    lines = []
    for i, name in enumerate(names):
        c = counts.get(i, 0)
        lines.append(f"  {name:14} {c}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _format_input(row: dict) -> str:
    """Build the model input string.

    Inputs are a tagged concatenation of the math element's LaTeX
    ``alttext`` (or fallback text) plus its surrounding sentence
    context. Tags are token-level cues that survive bpe.

    Format:
        "[math] {alttext} [ctx] {context}"
    """
    alt = (row.get("alttext") or row.get("math_text") or "").strip()
    ctx = (row.get("context") or "").strip()
    parts = []
    if alt:
        parts.append(f"[math] {alt}")
    if ctx:
        parts.append(f"[ctx] {ctx}")
    return " ".join(parts) or "[math]"


def load_split(path: Path) -> Dataset:
    rows: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        mt = r.get("math_type")
        en = r.get("eq_num_assoc")
        if mt not in MATH_TYPE_TO_ID or en not in EQ_NUM_TO_ID:
            skipped += 1
            continue
        rows.append({
            "text": _format_input(r),
            "math_type_id": MATH_TYPE_TO_ID[mt],
            "eq_num_id": EQ_NUM_TO_ID[en],
        })
    if skipped:
        print(f"  skipped {skipped} rows with unknown labels in {path.name}")
    return Dataset.from_list(rows)


def compute_class_weights(
    labels: list[int],
    num_labels: int,
    *,
    weight_cap: float = 30.0,
) -> torch.Tensor:
    counts = Counter(labels)
    n = len(labels)
    k = num_labels
    weights = torch.ones(num_labels, dtype=torch.float32)
    for i in range(num_labels):
        c = counts.get(i, 0)
        if c == 0:
            continue
        w = n / (k * c)
        weights[i] = min(w, weight_cap)
    return weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MathSpecialistModel(nn.Module):
    """Shared configurable encoder with LoRA and two classification heads.

    Implemented as a single nn.Module so the standard PyTorch training
    loop does optimizer/save bookkeeping for both the LoRA matrices and
    the head Linears together.
    """

    def __init__(
        self,
        base_model_name: str,
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        base = AutoModel.from_pretrained(base_model_name, dtype=dtype)
        # Attention-projection module names depend on the encoder family.
        # ModernBERT uses a single fused Wqkv (and Wo for output); DeBERTa-v3
        # uses separate query_proj / key_proj / value_proj. PEFT will skip
        # names that don't match the loaded model, so listing both is safe.
        target_modules = [
            "Wqkv", "Wo",                          # ModernBERT
            "query_proj", "key_proj", "value_proj",  # DeBERTa-v3
        ]
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(base, peft_cfg)
        hidden = base.config.hidden_size
        self.head_math_type = nn.Linear(hidden, len(MATH_TYPES))
        self.head_eq_num = nn.Linear(hidden, len(EQ_NUM_ASSOC))
        # Heads are fp32 for stability; encoder runs in bf16.

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # (B, T, H)
        pooled = last[:, 0, :]        # CLS
        return {
            "math_type": self.head_math_type(pooled.float()),
            "eq_num_assoc": self.head_eq_num(pooled.float()),
        }

    def save_adapter(self, out_dir: Path) -> None:
        """Write adapter + heads + manifest into ``out_dir``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        # peft writes adapter_model.safetensors + adapter_config.json.
        self.encoder.save_pretrained(str(out_dir))
        torch.save(
            {
                "head_math_type.state_dict": self.head_math_type.state_dict(),
                "head_eq_num.state_dict": self.head_eq_num.state_dict(),
            },
            str(out_dir / "heads.pt"),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _make_loader(
    ds: Dataset,
    tok,
    *,
    batch_size: int,
    shuffle: bool,
    sample_weights: torch.Tensor | None = None,
) -> DataLoader:
    coll = DataCollatorWithPadding(tok, return_tensors="pt")

    # Tokenizer fields the collator pads. `text` is the raw string input
    # we tokenized into these fields; we drop it before padding.
    TOK_FIELDS = ("input_ids", "attention_mask", "token_type_ids")

    def collate(batch):
        ml = torch.tensor([b["math_type_id"] for b in batch], dtype=torch.long)
        el = torch.tensor([b["eq_num_id"] for b in batch], dtype=torch.long)
        feats = [
            {k: b[k] for k in TOK_FIELDS if k in b}
            for b in batch
        ]
        out = coll(feats)
        out["math_type_id"] = ml
        out["eq_num_id"] = el
        return out

    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          collate_fn=collate)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate)


def evaluate(
    model: MathSpecialistModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    all_mt_pred: list[int] = []
    all_mt_true: list[int] = []
    all_en_pred: list[int] = []
    all_en_true: list[int] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids, mask)
            all_mt_pred.extend(out["math_type"].argmax(-1).tolist())
            all_mt_true.extend(batch["math_type_id"].tolist())
            all_en_pred.extend(out["eq_num_assoc"].argmax(-1).tolist())
            all_en_true.extend(batch["eq_num_id"].tolist())
    mt_macro = f1_score(all_mt_true, all_mt_pred, average="macro", zero_division=0)
    en_macro = f1_score(all_en_true, all_en_pred, average="macro", zero_division=0)
    return {
        "math_type_macro_f1": float(mt_macro),
        "eq_num_assoc_macro_f1": float(en_macro),
        "math_type": (all_mt_true, all_mt_pred),
        "eq_num_assoc": (all_en_true, all_en_pred),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("data/math_specialist_dataset"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/council/math_specialist"))
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--weight-cap", type=float, default=30.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no-oversample", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gpu] cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    print(f"[data] loading {args.dataset_dir}")
    train_ds = load_split(args.dataset_dir / "train.jsonl")
    val_ds = load_split(args.dataset_dir / "val.jsonl")
    test_ds = load_split(args.dataset_dir / "test.jsonl")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    print(f"[balance] math_type (train):")
    print(_label_count_table(train_ds["math_type_id"], MATH_TYPES))
    print(f"[balance] eq_num_assoc (train):")
    print(_label_count_table(train_ds["eq_num_id"], EQ_NUM_ASSOC))

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    cw_mt = compute_class_weights(
        train_ds["math_type_id"], len(MATH_TYPES), weight_cap=args.weight_cap,
    ).to(device)
    cw_en = compute_class_weights(
        train_ds["eq_num_id"], len(EQ_NUM_ASSOC), weight_cap=args.weight_cap,
    ).to(device)
    print(f"[weights] math_type:    {cw_mt.tolist()}")
    print(f"[weights] eq_num_assoc: {cw_en.tolist()}")

    sample_w = None
    if not args.no_oversample:
        # Sample weight = the math_type weight at each row's label, so
        # rare classes (matrix) see proportionally more gradient steps.
        sample_w = torch.tensor(
            [cw_mt[int(i)].item() for i in train_ds["math_type_id"]],
            dtype=torch.float32,
        )
        print(f"[oversample] WeightedRandomSampler over math_type weights")

    train_loader = _make_loader(
        train_ds, tok, batch_size=args.batch_size, shuffle=True,
        sample_weights=sample_w,
    )
    val_loader = _make_loader(
        val_ds, tok, batch_size=args.batch_size * 2, shuffle=False,
    )
    test_loader = _make_loader(
        test_ds, tok, batch_size=args.batch_size * 2, shuffle=False,
    )

    print(f"[model] base={args.base_model} LoRA r={args.lora_r} α={args.lora_alpha}")
    model = MathSpecialistModel(
        args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)
    model.encoder.print_trainable_parameters()

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=total_steps,
    )

    best_val = -1.0
    best_state: dict[str, object] | None = None
    best_epoch = 0
    epochs_since_best = 0
    train_log: list[dict[str, object]] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        sum_loss = 0.0
        sum_mt = 0.0
        sum_en = 0.0
        n_batches = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mt = batch["math_type_id"].to(device)
            en = batch["eq_num_id"].to(device)
            out = model(ids, mask)
            loss_mt = F.cross_entropy(out["math_type"], mt, weight=cw_mt)
            loss_en = F.cross_entropy(out["eq_num_assoc"], en, weight=cw_en)
            loss = loss_mt + loss_en
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            sum_loss += float(loss.item())
            sum_mt += float(loss_mt.item())
            sum_en += float(loss_en.item())
            n_batches += 1

        avg_loss = sum_loss / max(1, n_batches)
        avg_mt = sum_mt / max(1, n_batches)
        avg_en = sum_en / max(1, n_batches)
        ep_dt = time.time() - ep_t0

        val_metrics = evaluate(model, val_loader, device)
        mt_f1 = val_metrics["math_type_macro_f1"]
        en_f1 = val_metrics["eq_num_assoc_macro_f1"]
        print(f"[epoch {epoch:02d}] loss={avg_loss:.4f} mt={avg_mt:.4f} en={avg_en:.4f} "
              f"val_mt_f1={mt_f1:.4f} val_en_f1={en_f1:.4f}  ({ep_dt:.1f}s)")
        train_log.append({
            "epoch": epoch,
            "loss": avg_loss, "loss_math_type": avg_mt, "loss_eq_num": avg_en,
            "val_math_type_macro_f1": mt_f1,
            "val_eq_num_assoc_macro_f1": en_f1,
            "epoch_time_s": ep_dt,
        })

        if mt_f1 > best_val + 1e-4:
            best_val = mt_f1
            best_epoch = epoch
            epochs_since_best = 0
            # Snapshot the trainable parameters (LoRA + heads).
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if any(s in k for s in ("lora_", "head_math_type", "head_eq_num"))
            }
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"[early-stop] no val_mt_f1 improvement for "
                      f"{args.patience} epochs (best epoch={best_epoch})")
                break

    total_dt = time.time() - t0
    print(f"[train] done in {total_dt:.1f}s; best epoch={best_epoch} "
          f"val_mt_f1={best_val:.4f}")

    # Restore best snapshot.
    if best_state is not None:
        full_state = model.state_dict()
        for k, v in best_state.items():
            full_state[k] = v.to(full_state[k].device)
        model.load_state_dict(full_state)

    # ------ Held-out test ------
    test_metrics = evaluate(model, test_loader, device)
    mt_y, mt_p = test_metrics["math_type"]
    en_y, en_p = test_metrics["eq_num_assoc"]
    test_mt_f1 = test_metrics["math_type_macro_f1"]
    test_en_f1 = test_metrics["eq_num_assoc_macro_f1"]
    print(f"\n[test] math_type macro-F1   = {test_mt_f1:.4f}")
    print(f"[test] eq_num_assoc macro-F1 = {test_en_f1:.4f}")
    print("[test] math_type per-class:")
    print(classification_report(
        mt_y, mt_p,
        labels=list(range(len(MATH_TYPES))),
        target_names=MATH_TYPES,
        zero_division=0, digits=3,
    ))
    print("[test] eq_num_assoc per-class:")
    print(classification_report(
        en_y, en_p,
        labels=list(range(len(EQ_NUM_ASSOC))),
        target_names=EQ_NUM_ASSOC,
        zero_division=0, digits=3,
    ))

    # ------ Save ------
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_adapter(final_dir)
    tok.save_pretrained(str(final_dir / "tokenizer"))
    summary = {
        "base_model": args.base_model,
        "math_types": MATH_TYPES,
        "eq_num_assoc": EQ_NUM_ASSOC,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_math_type_macro_f1": best_val,
        "test_math_type_macro_f1": test_mt_f1,
        "test_eq_num_assoc_macro_f1": test_en_f1,
        "total_train_time_s": total_dt,
        "epochs_run": len(train_log),
        "epoch_log": train_log,
    }
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] adapter + heads -> {final_dir}")
    print(f"[save] summary -> {final_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
