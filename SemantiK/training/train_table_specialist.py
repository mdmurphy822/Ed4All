"""Train BERT-TableSpecialist (Phase 4 — task #52).

Cell-level multi-class classifier. Single head in v1; caption_association
deferred to v2 (needs caption-bbox proximity, not per-cell signal).

Heads:
    * cell_role_scope    4-class    `data`, `header_col`, `header_row`,
                                    `span` — see
                                    data.build_table_specialist_data
                                    for the labeling rules.

Inputs:
    * text     — cell HTML text (truncated to 300 chars at build time).
    * layout   — 15-dim numeric vector (row/col index norms,
                 boundary flags, text-shape signals; see
                 data.build_table_specialist_data.LAYOUT_FEATURE_NAMES).

Side-channel:
    layout(15) → LayerNorm → MLP(15→64→64) → concat with ModernBERT
    pooled (CLS) → 1 classification head.

    Neighbors are NOT consumed in v1 — the layout features
    (is_first_row / is_first_col / row_idx_norm / col_idx_norm) carry
    most of the cell-role signal. v2 can extend the input string with
    "[ABOVE] <text> [LEFT] <text> [CELL] <text> [RIGHT] <text> [BELOW]
    <text>" tokens if the v1 confusion matrix shows row/col header
    confusion the layout vector can't resolve.

Hyperparameter recipe (matches Phase 3b/3c):
    * ModernBERT-base, bf16 encoder, fp32 head + side MLP
    * LoRA r=16 alpha=32, dropout=0.05
    * 6 epochs, AdamW lr=2e-4 weight_decay=0.01, cosine schedule
    * Class-weighted CE (weight_cap=15 — span is rare at 3.5%)
    * WeightedRandomSampler — per-row weight = class_weight[y]
    * Snapshot policy: max(val cell_role_scope macro-F1).

Outputs:
    models/council/table_specialist/final/adapter_model.safetensors
    models/council/table_specialist/final/heads.pt
    models/council/table_specialist/final/tokenizer/
    models/council/table_specialist/final/summary.json
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

from data.build_table_specialist_data import (  # noqa: E402  (after sys.path bootstrap)
    CELL_ROLE_SCOPE_NAMES,
    LAYOUT_FEATURE_DIM,
)


NUM_CELL_ROLES = len(CELL_ROLE_SCOPE_NAMES)
LAYOUT_MLP_HIDDEN = 64


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _format_input(row: dict) -> str:
    text = (row.get("text") or "").strip()
    nb = row.get("neighbors") or {}
    above = (nb.get("above") or "").strip()
    below = (nb.get("below") or "").strip()
    left = (nb.get("left") or "").strip()
    right = (nb.get("right") or "").strip()
    return (
        f"[ABOVE] {above} [LEFT] {left} "
        f"[CELL] {text} "
        f"[RIGHT] {right} [BELOW] {below}"
    )


def load_split(path: Path) -> Dataset:
    rows: list[dict] = []
    skipped = 0
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            r = json.loads(line)
            labels = r.get("labels") or {}
            if "cell_role_scope" not in labels:
                skipped += 1
                continue
            rows.append({
                "text": _format_input(r),
                "layout": r["layout"],
                "y": int(labels["cell_role_scope"]),
                "source": r.get("source", "unknown"),
            })
    if skipped:
        print(f"  skipped {skipped} rows missing labels in {path.name}")
    return Dataset.from_list(rows)


def compute_class_weights(
    labels: list[int], num_labels: int, *, weight_cap: float = 15.0,
) -> torch.Tensor:
    counts = Counter(labels)
    n = sum(counts.values())
    weights = torch.ones(num_labels, dtype=torch.float32)
    if n == 0:
        return weights
    for i in range(num_labels):
        c = counts.get(i, 0)
        if c == 0:
            continue
        w = n / (num_labels * c)
        weights[i] = min(w, weight_cap)
    return weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TableSpecialistModel(nn.Module):
    """ModernBERT-base + LoRA + 15-dim layout side-channel + 1 head."""

    def __init__(
        self, base_model_name: str, *,
        lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
        layout_dim: int = LAYOUT_FEATURE_DIM,
        layout_mlp_hidden: int = LAYOUT_MLP_HIDDEN,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        base = AutoModel.from_pretrained(base_model_name, dtype=dtype)
        target_modules = [
            "Wqkv", "Wo", "query_proj", "key_proj", "value_proj",
        ]
        peft_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(base, peft_cfg)
        hidden = base.config.hidden_size

        self.layout_dim = int(layout_dim)
        self.layout_mlp_hidden = int(layout_mlp_hidden)
        self.layout_norm = nn.LayerNorm(self.layout_dim)
        self.layout_mlp = nn.Sequential(
            nn.Linear(self.layout_dim, self.layout_mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.layout_mlp_hidden, self.layout_mlp_hidden),
            nn.ReLU(),
        )

        head_in = hidden + self.layout_mlp_hidden
        self.head_cell_role_scope = nn.Linear(head_in, NUM_CELL_ROLES)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        layout: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        last = out.last_hidden_state
        pooled = last[:, 0, :].float()
        layout_h = self.layout_mlp(self.layout_norm(layout.float()))
        h = torch.cat([pooled, layout_h], dim=-1)
        return {"cell_role_scope": self.head_cell_role_scope(h)}

    def save_adapter(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(str(out_dir))
        torch.save({
            "head_cell_role_scope.state_dict":
                self.head_cell_role_scope.state_dict(),
            "layout_norm.state_dict": self.layout_norm.state_dict(),
            "layout_mlp.state_dict": self.layout_mlp.state_dict(),
            "layout_dim": self.layout_dim,
            "layout_mlp_hidden": self.layout_mlp_hidden,
        }, str(out_dir / "heads.pt"))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _make_loader(
    ds: Dataset, tok, *, batch_size: int, shuffle: bool,
    sample_weights: torch.Tensor | None = None,
) -> DataLoader:
    coll = DataCollatorWithPadding(tok, return_tensors="pt")
    TOK_FIELDS = ("input_ids", "attention_mask", "token_type_ids")

    def collate(batch):
        feats = [{k: b[k] for k in TOK_FIELDS if k in b} for b in batch]
        out = coll(feats)
        out["y"] = torch.tensor([b["y"] for b in batch], dtype=torch.long)
        out["layout"] = torch.tensor(
            [b["layout"] for b in batch], dtype=torch.float32,
        )
        return out

    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(ds), replacement=True,
        )
        return DataLoader(
            ds, batch_size=batch_size, sampler=sampler,
            collate_fn=collate, num_workers=0,
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate, num_workers=0,
    )


def evaluate(
    model: TableSpecialistModel, loader: DataLoader, device: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            layout = batch["layout"].to(device)
            out = model(input_ids, attention_mask, layout)
            logits = out["cell_role_scope"]
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_p.extend(preds.tolist())
            all_y.extend(batch["y"].tolist())
    macro = f1_score(all_y, all_p, average="macro", zero_division=0)
    return float(macro), np.asarray(all_y), np.asarray(all_p)


def train(args) -> None:
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    print(f"[load] {dataset_dir}")
    ds_train = load_split(dataset_dir / "train.jsonl")
    ds_val = load_split(dataset_dir / "val.jsonl")
    ds_test = load_split(dataset_dir / "test.jsonl")
    print(f"  train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")

    label_counts = Counter(ds_train["y"])
    print("[balance] cell_role_scope (train):")
    for i, name in enumerate(CELL_ROLE_SCOPE_NAMES):
        print(f"  {name:14s} {label_counts.get(i, 0)}")

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(b):
        return tok(
            b["text"], padding=False, truncation=True,
            max_length=args.max_length,
        )

    ds_train = ds_train.map(tokenize, batched=True)
    ds_val = ds_val.map(tokenize, batched=True)
    ds_test = ds_test.map(tokenize, batched=True)

    cw = compute_class_weights(
        list(ds_train["y"]), NUM_CELL_ROLES, weight_cap=args.weight_cap,
    )
    print(f"[weights] cell_role_scope: {cw.tolist()}")

    if not args.no_oversample:
        sample_w = torch.tensor(
            [cw[int(y)].item() for y in ds_train["y"]],
            dtype=torch.float32,
        )
        pre_max = float(sample_w.max())
        sample_w = sample_w.clamp(max=args.sampler_weight_cap)
        post_max = float(sample_w.max())
        print(f"[oversample] sampler_weight_cap={args.sampler_weight_cap}  "
              f"pre-clip max={pre_max:.3f}  post-clip max={post_max:.3f}")
        train_loader = _make_loader(
            ds_train, tok, batch_size=args.batch_size, shuffle=False,
            sample_weights=sample_w,
        )
        print(f"[oversample] per-class weights (CE unchanged, sampler clamped):")
        for i, name in enumerate(CELL_ROLE_SCOPE_NAMES):
            n = label_counts.get(i, 0)
            ce_w = float(cw[i].item())
            samp_w = min(ce_w, args.sampler_weight_cap)
            print(f"  {name:14s} n={n} ce_w={ce_w:.3f} samp_w={samp_w:.3f}")
    else:
        train_loader = _make_loader(
            ds_train, tok, batch_size=args.batch_size, shuffle=True,
        )

    val_loader = _make_loader(
        ds_val, tok, batch_size=args.batch_size, shuffle=False,
    )
    test_loader = _make_loader(
        ds_test, tok, batch_size=args.batch_size, shuffle=False,
    )

    print(f"[model] base={args.base_model} LoRA r={args.lora_r} α={args.lora_alpha}")
    model = TableSpecialistModel(
        args.base_model,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.encoder.print_trainable_parameters()

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
    )
    n_steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps)
    cw_dev = cw.to(device)

    best_score = -1.0
    best_epoch = -1
    best_state: dict | None = None
    epoch_log = []
    t_start = time.time()
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_epoch = time.time()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            layout = batch["layout"].to(device)
            y = batch["y"].to(device)

            out = model(input_ids, attention_mask, layout)
            loss = F.cross_entropy(
                out["cell_role_scope"], y, weight=cw_dev,
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            epoch_loss += float(loss.item())

        epoch_loss /= max(1, len(train_loader))
        macro_f1, y_val, p_val = evaluate(model, val_loader, device)
        score = macro_f1
        epoch_time = time.time() - t_epoch
        print(
            f"[epoch {epoch:02d}] loss={epoch_loss:.4f}  "
            f"val macro_f1={macro_f1:.4f}  ({epoch_time:.1f}s)",
            flush=True,
        )
        epoch_log.append({
            "epoch": epoch, "loss": epoch_loss,
            "val_macro_f1": macro_f1,
            "epoch_time_s": epoch_time,
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[stop] early — no improvement for {args.patience} epochs")
                break

    total_time = time.time() - t_start
    print(f"[train] done in {total_time:.1f}s; best epoch={best_epoch} "
          f"val macro_f1={best_score:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_macro, y_test, p_test = evaluate(model, test_loader, device)
    print(f"\n[test] cell_role_scope macro-F1 = {test_macro:.4f}\n")
    print("[test] cell_role_scope per-class:")
    print(classification_report(
        y_test, p_test,
        labels=list(range(NUM_CELL_ROLES)),
        target_names=list(CELL_ROLE_SCOPE_NAMES),
        digits=3, zero_division=0,
    ))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_adapter(out_dir)
    tok_dir = out_dir / "tokenizer"
    tok.save_pretrained(str(tok_dir))

    summary = {
        "base_model": args.base_model,
        "cell_role_scope_names": list(CELL_ROLE_SCOPE_NAMES),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_size": len(ds_train),
        "val_size": len(ds_val),
        "test_size": len(ds_test),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "test_macro_f1": test_macro,
        "weight_cap": args.weight_cap,
        "sampler_weight_cap": args.sampler_weight_cap,
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
        "layout_mlp_hidden": LAYOUT_MLP_HIDDEN,
        "epochs_run": len(epoch_log),
        "epoch_log": epoch_log,
        "total_train_time_s": total_time,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] adapter + heads -> {out_dir}")
    print(f"[save] summary -> {out_dir / 'summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/table_specialist_dataset")
    ap.add_argument("--output-dir", default="models/council/table_specialist/final")
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--weight-cap", type=float, default=15.0)
    ap.add_argument(
        "--sampler-weight-cap", type=float, default=3.0,
        help="Cap on per-row sampler weight (separate from CE "
             "weight_cap). Mirrors train_structure's "
             "is_heading_sampler_cap and train_semantic's "
             "--sampler-weight-cap. v1 used uncapped sampler weights "
             "(max 7.08, ratio 21x); rare classes over-fired at "
             "inference (header_row P=0.50, span P=0.55).",
    )
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--no-oversample", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
