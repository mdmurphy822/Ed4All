"""Train BERT-MergeOrSplit (ModernBERT-base + LoRA, 3 multi-head).

Heads:
    * same_logical_block      ∈ {0, 1}                              (2)
    * join_type               ∈ {space, newline_within_p,
                                 list_continuation}                 (3)
    * hyphen_repair           ∈ {0, 1}                              (2)

Conditional head: ``join_type`` loss is masked when
``same_logical_block=0`` (its label is ``-100`` in the dataset and we
pass ``ignore_index=-100`` to ``F.cross_entropy``).

Heading detection (formerly ``heading_body_boundary`` here) moved to
BERT-Structure (Phase 3b) as a span-level ``is_heading`` head. Pair-
level boundary detection is structurally a worse framing — only ~2.6%
of pairs span a heading→body transition, vs ~5-15% of *spans* being
headings. The pair-level head trained but plateaued near F1≈0.4 with
high false positives. Removing it lets the encoder + layout MLP focus
on the heads that do work.

Hyperparameter recipe identical to Phase 2 (MathSpecialist):
    * ModernBERT-base backbone, bf16 encoder, fp32 heads
    * LoRA r=16 alpha=32, dropout=0.05, target attention modules
    * 6 epochs, AdamW lr=2e-4 weight_decay=0.01, cosine LR schedule
    * Class-weighted CE per head (weight cap=30)
    * Multi-head WeightedRandomSampler — per-row weight =
      ``max(cw_same[y_same], cw_hyphen[y_hyphen])`` so hyphen-repair
      positives (~1% of rows) still get oversampled.
    * Layout side-channel: 24-dim numeric features → LayerNorm → 64-dim
      MLP → concatenated with BERT pooled before all 3 heads.
    * Snapshot policy: 0.7·same + 0.3·hyphen weighted-macro by default.
    * Early stop on the snapshot score, patience=3.

Outputs (mirrors Phase 2):
    models/council/merge_or_split/final/adapter_model.safetensors
    models/council/merge_or_split/final/adapter_config.json
    models/council/merge_or_split/final/heads.pt
    models/council/merge_or_split/final/tokenizer/
    models/council/merge_or_split/final/summary.json
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

from data.build_merge_or_split_data import (
    JOIN_TYPES,
    JOIN_TYPE_IGNORE,
    JOIN_TYPE_TO_ID,
    LAYOUT_FEATURE_DIM,
    layout_features_from_row,
)


LAYOUT_MLP_HIDDEN = 64


SAME_LOGICAL_LABELS = ("not_same", "same")           # 2
HYPHEN_LABELS = ("no_repair", "repair")              # 2


def _label_count_table(labels: list[int], names) -> str:
    counts = Counter(labels)
    lines = []
    for i, name in enumerate(names):
        c = counts.get(i, 0)
        lines.append(f"  {name:24} {c}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _format_input(row: dict) -> str:
    """Build the model input string with a structured deterministic prefix
    per span:

        [col:{cid_a} art:{is_a} fs:{fs_a} bold:{...}] {span_a_text}
        [SEP]
        [col:{cid_b} art:{is_b} fs:{fs_b} bold:{...}] {span_b_text}

    The geometric prefix uses bracketed key:value pairs so the BPE keeps
    the column / artifact / font cues as a small finite vocabulary.
    """
    feats = row.get("det_features", {}) or {}
    a = feats.get("a", {}) or {}
    b = feats.get("b", {}) or {}
    y_gap = feats.get("y_gap", 0.0)
    lhr = feats.get("line_height_ratio", 1.0)
    txt_a = (row.get("span_a_text") or "").strip()
    txt_b = (row.get("span_b_text") or "").strip()

    def _bucket_size(fs: float) -> str:
        if fs <= 0:
            return "?"
        if fs < 9: return "xs"
        if fs < 11: return "sm"
        if fs < 13: return "md"
        if fs < 16: return "lg"
        return "xl"

    def _bucket_gap(g: float) -> str:
        if g <= 0:
            return "0"
        if g < 6:
            return "sm"
        if g < 18:
            return "md"
        return "lg"

    def _prefix(s: dict) -> str:
        return (
            f"[col:{int(s.get('column_id', 0))} "
            f"art:{int(bool(s.get('is_artifact', False)))} "
            f"fs:{_bucket_size(float(s.get('font_size', 0) or 0))} "
            f"bold:{int(bool(s.get('is_bold', False)))} "
            f"hy:{int(bool(s.get('ends_with_hyphen', False)))}]"
        )

    pa = _prefix(a)
    pb = _prefix(b)
    return (
        f"{pa} {txt_a} [SEP] [gap:{_bucket_gap(float(y_gap or 0))} "
        f"lhr:{float(lhr):.2f}] {pb} {txt_b}"
    )


def load_split(path: Path) -> Dataset:
    rows: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        labels = r.get("labels") or {}
        if "same_logical_block" not in labels:
            skipped += 1
            continue
        rows.append({
            "text": _format_input(r),
            "layout": layout_features_from_row(r),
            "y_same": int(labels["same_logical_block"]),
            "y_join": int(labels["join_type"]),  # may be -100
            "y_hyphen": int(labels["hyphen_repair"]),
            "noisy": bool(r.get("noisy", False)),
        })
    if skipped:
        print(f"  skipped {skipped} rows missing labels in {path.name}")
    return Dataset.from_list(rows)


def compute_class_weights(
    labels: list[int],
    num_labels: int,
    *,
    weight_cap: float = 30.0,
    ignore: int | None = None,
) -> torch.Tensor:
    counts = Counter(l for l in labels if ignore is None or l != ignore)
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


class MergeOrSplitModel(nn.Module):
    """ModernBERT-base + LoRA, three classification heads with an
    explicit layout side-channel.

    The 5-bucket prefix tokens (``fs:xs/sm/md/lg/xl``) lose too much of
    the font-size-jump signal that the heads — particularly join_type
    on heading→body adjacency — need. A small MLP consumes a fixed-dim
    numeric layout vector (``LAYOUT_FEATURE_DIM``) and its output is
    concatenated with the BERT pooled vector before the three heads.
    """

    def __init__(
        self,
        base_model_name: str,
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        layout_dim: int = LAYOUT_FEATURE_DIM,
        layout_mlp_hidden: int = LAYOUT_MLP_HIDDEN,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        base = AutoModel.from_pretrained(base_model_name, dtype=dtype)
        # Same target list as MathSpecialist — both ModernBERT and
        # DeBERTa-v3 attention names are listed; PEFT skips non-matches.
        target_modules = [
            "Wqkv", "Wo",
            "query_proj", "key_proj", "value_proj",
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
        self.head_same = nn.Linear(head_in, 2)
        self.head_join = nn.Linear(head_in, len(JOIN_TYPES))
        self.head_hyphen = nn.Linear(head_in, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layout: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        pooled = last[:, 0, :].float()
        layout_h = self.layout_mlp(self.layout_norm(layout.float()))
        h = torch.cat([pooled, layout_h], dim=-1)
        return {
            "same": self.head_same(h),
            "join": self.head_join(h),
            "hyphen": self.head_hyphen(h),
        }

    def save_adapter(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(str(out_dir))
        torch.save(
            {
                "head_same.state_dict": self.head_same.state_dict(),
                "head_join.state_dict": self.head_join.state_dict(),
                "head_hyphen.state_dict": self.head_hyphen.state_dict(),
                "layout_norm.state_dict": self.layout_norm.state_dict(),
                "layout_mlp.state_dict": self.layout_mlp.state_dict(),
                "layout_dim": self.layout_dim,
                "layout_mlp_hidden": self.layout_mlp_hidden,
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
    TOK_FIELDS = ("input_ids", "attention_mask", "token_type_ids")

    def collate(batch):
        feats = [{k: b[k] for k in TOK_FIELDS if k in b} for b in batch]
        out = coll(feats)
        out["y_same"] = torch.tensor([b["y_same"] for b in batch], dtype=torch.long)
        out["y_join"] = torch.tensor([b["y_join"] for b in batch], dtype=torch.long)
        out["y_hyphen"] = torch.tensor([b["y_hyphen"] for b in batch], dtype=torch.long)
        out["layout"] = torch.tensor([b["layout"] for b in batch], dtype=torch.float32)
        out["noisy"] = torch.tensor([1 if b.get("noisy") else 0 for b in batch], dtype=torch.long)
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
    model: MergeOrSplitModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    same_t, same_p = [], []
    join_t, join_p = [], []     # Only collected when y_join != IGNORE.
    hy_t, hy_p = [], []
    noisy_flag: list[int] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            layout = batch["layout"].to(device)
            out = model(ids, mask, layout)
            p_same = out["same"].argmax(-1).tolist()
            p_join = out["join"].argmax(-1).tolist()
            p_hy = out["hyphen"].argmax(-1).tolist()
            ys = batch["y_same"].tolist()
            yj = batch["y_join"].tolist()
            yh = batch["y_hyphen"].tolist()
            ns = batch["noisy"].tolist()
            for i in range(len(ys)):
                same_t.append(ys[i]); same_p.append(p_same[i])
                hy_t.append(yh[i]);   hy_p.append(p_hy[i])
                noisy_flag.append(ns[i])
                if yj[i] != JOIN_TYPE_IGNORE:
                    join_t.append(yj[i]); join_p.append(p_join[i])
    same_macro = f1_score(same_t, same_p, average="macro", zero_division=0)
    join_macro = (
        f1_score(join_t, join_p, average="macro", zero_division=0)
        if join_t else 0.0
    )
    hy_pos_f1 = f1_score(hy_t, hy_p, pos_label=1, zero_division=0)
    # Noisy subset macro-F1 for same_logical_block.
    noisy_t = [same_t[i] for i in range(len(same_t)) if noisy_flag[i]]
    noisy_p = [same_p[i] for i in range(len(same_p)) if noisy_flag[i]]
    noisy_macro = (
        f1_score(noisy_t, noisy_p, average="macro", zero_division=0)
        if noisy_t else 0.0
    )
    return {
        "same_macro_f1": float(same_macro),
        "join_macro_f1": float(join_macro),
        "hyphen_pos_f1": float(hy_pos_f1),
        "same_macro_f1_noisy": float(noisy_macro),
        "raw": {
            "same": (same_t, same_p),
            "join": (join_t, join_p),
            "hyphen": (hy_t, hy_p),
            "noisy_flag": noisy_flag,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("data/merge_or_split_dataset"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/council/merge_or_split"))
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--weight-cap", type=float, default=30.0)
    ap.add_argument("--hyphen-sampler-cap", type=float, default=30.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no-oversample", action="store_true")
    ap.add_argument("--snapshot-policy", choices=("same_only", "weighted_macro"),
                    default="weighted_macro",
                    help="Best-epoch snapshot selection: same_only saves "
                         "the epoch that maximizes val_same_macro_f1; "
                         "weighted_macro saves the epoch that maximizes "
                         "0.7*same + 0.3*hyphen so we don't throw away "
                         "hyphen-repair progress.")
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

    print("[balance] same_logical_block (train):")
    print(_label_count_table(train_ds["y_same"], SAME_LOGICAL_LABELS))
    train_join = [v for v in train_ds["y_join"] if v != JOIN_TYPE_IGNORE]
    print(f"[balance] join_type (train, positive only n={len(train_join)}):")
    print(_label_count_table(train_join, JOIN_TYPES))
    print("[balance] hyphen_repair (train):")
    print(_label_count_table(train_ds["y_hyphen"], HYPHEN_LABELS))

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    cw_same = compute_class_weights(
        train_ds["y_same"], 2, weight_cap=args.weight_cap,
    ).to(device)
    cw_join = compute_class_weights(
        train_ds["y_join"], len(JOIN_TYPES),
        weight_cap=args.weight_cap, ignore=JOIN_TYPE_IGNORE,
    ).to(device)
    cw_hyphen = compute_class_weights(
        train_ds["y_hyphen"], 2, weight_cap=args.weight_cap,
    ).to(device)
    print(f"[weights] same:    {cw_same.tolist()}")
    print(f"[weights] join:    {cw_join.tolist()}")
    print(f"[weights] hyphen:  {cw_hyphen.tolist()}")

    sample_w = None
    if not args.no_oversample:
        # Multi-head per-row weight: take the max of the per-class
        # weights across the two sparse heads (same_logical_block,
        # hyphen_repair). A row that is positive on EITHER head gets
        # sampled at the rare-head rate; the primary head's balance is
        # preserved on the bulk of rows.
        # join_type isn't included because it's masked on negatives,
        # and its positive-only distribution is already mild (~3:1 max).
        #
        # Per-head sampler caps are SEPARATE from the CE class-weight
        # cap. The CE cap controls loss magnitude; the sampler cap
        # controls exposure rate. Hyphen positives tolerated cap=30 in
        # the v2/v3b runs and the head learns cleanly.
        ys_arr = torch.tensor(train_ds["y_same"], dtype=torch.long)
        yh_arr = torch.tensor(train_ds["y_hyphen"], dtype=torch.long)
        cw_same_cpu = cw_same.detach().cpu()
        cw_hy_cpu = cw_hyphen.detach().cpu().clone()
        cw_hy_cpu[1] = min(float(cw_hy_cpu[1]), float(args.hyphen_sampler_cap))
        per_row = torch.stack([
            cw_same_cpu[ys_arr],
            cw_hy_cpu[yh_arr],
        ], dim=0)
        sample_w = per_row.max(dim=0).values.to(torch.float32)
        print(f"[oversample] sampler cap  hyphen={args.hyphen_sampler_cap}")
        for nm, arr in (
            ("y_same", ys_arr), ("y_hyphen", yh_arr),
        ):
            for cls in (0, 1):
                mask = arr == cls
                if mask.any():
                    mw = float(sample_w[mask].mean())
                    n = int(mask.sum())
                    print(f"[oversample] mean weight  {nm}={cls}  "
                          f"n={n}  w_mean={mw:.3f}")

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
    model = MergeOrSplitModel(
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
    peak_vram_bytes = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        sums = {"loss": 0.0, "same": 0.0, "join": 0.0, "hyphen": 0.0}
        n_batches = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            layout = batch["layout"].to(device)
            ys = batch["y_same"].to(device)
            yj = batch["y_join"].to(device)
            yh = batch["y_hyphen"].to(device)
            out = model(ids, mask, layout)
            loss_same = F.cross_entropy(out["same"], ys, weight=cw_same)
            loss_join = F.cross_entropy(
                out["join"], yj,
                weight=cw_join,
                ignore_index=JOIN_TYPE_IGNORE,
            )
            loss_hy = F.cross_entropy(out["hyphen"], yh, weight=cw_hyphen)
            # If all join labels in this batch are masked, F.cross_entropy
            # returns NaN — guard it.
            if torch.isnan(loss_join):
                loss_join = torch.zeros((), device=device)
            loss = loss_same + loss_join + loss_hy
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            sums["loss"] += float(loss.item())
            sums["same"] += float(loss_same.item())
            sums["join"] += float(loss_join.item())
            sums["hyphen"] += float(loss_hy.item())
            n_batches += 1
            if torch.cuda.is_available():
                peak_vram_bytes = max(
                    peak_vram_bytes, torch.cuda.max_memory_allocated(device)
                )

        ep_dt = time.time() - ep_t0
        avg = {k: v / max(1, n_batches) for k, v in sums.items()}
        val_metrics = evaluate(model, val_loader, device)
        same_f1 = val_metrics["same_macro_f1"]
        if args.snapshot_policy == "same_only":
            score = same_f1
        else:  # weighted_macro
            score = (
                0.7 * same_f1
                + 0.3 * val_metrics["hyphen_pos_f1"]
            )
        print(
            f"[epoch {epoch:02d}] loss={avg['loss']:.4f} "
            f"same={avg['same']:.4f} join={avg['join']:.4f} "
            f"hy={avg['hyphen']:.4f}  "
            f"val same={same_f1:.4f} join={val_metrics['join_macro_f1']:.4f} "
            f"hy_pos={val_metrics['hyphen_pos_f1']:.4f} "
            f"noisy_same={val_metrics['same_macro_f1_noisy']:.4f}  "
            f"score={score:.4f}  ({ep_dt:.1f}s)",
            flush=True,
        )
        train_log.append({
            "epoch": epoch,
            "loss": avg["loss"], "loss_same": avg["same"],
            "loss_join": avg["join"], "loss_hyphen": avg["hyphen"],
            "val_same_macro_f1": same_f1,
            "val_join_macro_f1": val_metrics["join_macro_f1"],
            "val_hyphen_pos_f1": val_metrics["hyphen_pos_f1"],
            "val_same_macro_f1_noisy": val_metrics["same_macro_f1_noisy"],
            "snapshot_score": score,
            "epoch_time_s": ep_dt,
        })

        if score > best_val + 1e-4:
            best_val = score
            best_epoch = epoch
            epochs_since_best = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if any(s in k for s in (
                    "lora_", "head_same", "head_join", "head_hyphen",
                    "layout_norm", "layout_mlp",
                ))
            }
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"[early-stop] no val_same improvement for "
                      f"{args.patience} epochs (best epoch={best_epoch})")
                break

    total_dt = time.time() - t0
    print(f"[train] done in {total_dt:.1f}s; best epoch={best_epoch} "
          f"snapshot_score={best_val:.4f}", flush=True)

    # Restore best snapshot.
    if best_state is not None:
        full_state = model.state_dict()
        for k, v in best_state.items():
            full_state[k] = v.to(full_state[k].device)
        model.load_state_dict(full_state)

    # ------ Held-out test ------
    test_metrics = evaluate(model, test_loader, device)
    test_same_f1 = test_metrics["same_macro_f1"]
    test_join_f1 = test_metrics["join_macro_f1"]
    test_hy_f1 = test_metrics["hyphen_pos_f1"]
    test_same_f1_noisy = test_metrics["same_macro_f1_noisy"]

    print(f"\n[test] same_logical_block macro-F1   = {test_same_f1:.4f}")
    print(f"[test] same_logical_block macro-F1 (noisy subset) = {test_same_f1_noisy:.4f}")
    print(f"[test] join_type macro-F1 (positive-primary only) = {test_join_f1:.4f}")
    print(f"[test] hyphen_repair F1 (pos class)               = {test_hy_f1:.4f}")

    same_t, same_p = test_metrics["raw"]["same"]
    print("[test] same_logical_block per-class:")
    print(classification_report(
        same_t, same_p,
        labels=[0, 1],
        target_names=SAME_LOGICAL_LABELS,
        zero_division=0, digits=3,
    ))
    join_t, join_p = test_metrics["raw"]["join"]
    if join_t:
        print("[test] join_type per-class (positive-primary only):")
        print(classification_report(
            join_t, join_p,
            labels=list(range(len(JOIN_TYPES))),
            target_names=JOIN_TYPES,
            zero_division=0, digits=3,
        ))
    hy_t, hy_p = test_metrics["raw"]["hyphen"]
    print("[test] hyphen_repair per-class:")
    print(classification_report(
        hy_t, hy_p,
        labels=[0, 1],
        target_names=HYPHEN_LABELS,
        zero_division=0, digits=3,
    ))

    # ------ Save ------
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_adapter(final_dir)
    tok.save_pretrained(str(final_dir / "tokenizer"))
    summary = {
        "base_model": args.base_model,
        "join_types": JOIN_TYPES,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "snapshot_policy": args.snapshot_policy,
        "best_epoch": best_epoch,
        "best_snapshot_score": best_val,
        "hyphen_sampler_cap": args.hyphen_sampler_cap,
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
        "layout_mlp_hidden": LAYOUT_MLP_HIDDEN,
        "test_same_macro_f1": test_same_f1,
        "test_same_macro_f1_noisy": test_same_f1_noisy,
        "test_join_macro_f1": test_join_f1,
        "test_hyphen_pos_f1": test_hy_f1,
        "total_train_time_s": total_dt,
        "epochs_run": len(train_log),
        "epoch_log": train_log,
        "peak_vram_bytes": int(peak_vram_bytes),
    }
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] adapter + heads -> {final_dir}")
    print(f"[save] summary -> {final_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
