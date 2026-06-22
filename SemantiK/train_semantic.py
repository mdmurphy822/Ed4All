"""Train BERT-Semantic (Phase 3c — ModernBERT-base + LoRA, 2 multi-head).

Span-level multi-head classifier for document-level role + boilerplate.
Consumes Phase 3b's Structure adapter as a teacher-forced cascade signal:
each training row carries an 8-dim ``cascade`` vector
[P(role)x6, P(is_heading=1), P(table_region=1)] computed at data-build
time. The cascade vector is concatenated with the 20-dim layout vector
to form a 28-dim side-channel input. Cascade is therefore baked into the
JSONL — the Phase 3c trainer never invokes the Structure adapter.

Heads:
    * doc_role            7-class     (title, author, body, citation,
                                      footer, legal, metadata) — see
                                      data.build_semantic_data for
                                      vocab rationale (``abstract``
                                      removed at 0.16% prevalence).
    * boilerplate         binary      Independent of doc_role:
                                      copyright lines, page numbers,
                                      arxiv banners, DOI strings, etc.
                                      Naturally balanced at ~5.6% in
                                      the training set — we DO NOT
                                      oversample on boilerplate (see
                                      sampler comment).

Side-channel layout: 20-dim layout || 8-dim cascade || 3-dim positional
= 31-dim → LayerNorm → MLP(31 -> 64 -> 64) → concatenated with BERT
pooled (CLS) before the heads. Same MLP-hidden as Phase 3b for parity.
The 3-dim positional vec (block_index_norm, page_index_norm,
doc_position_norm) is Semantic-ONLY — Structure stays on the 20-dim
layout contract so the trained Structure adapter doesn't need rebuild.

Hyperparameter recipe (matches Phase 3b):
    * ModernBERT-base, bf16 encoder, fp32 heads
    * LoRA r=16 alpha=32, dropout=0.05
    * 6 epochs, AdamW lr=2e-4 weight_decay=0.01, cosine LR schedule
    * Class-weighted CE per head (weight_cap=30)
    * WeightedRandomSampler — per-row weight = cw_doc_role[y_doc_role].
      Boilerplate is NOT in the sampler weight: it's already at 5.6%
      (well-balanced binary), and oversampling on it would over-rotate
      doc_role training. Same lesson as Phase 3a: only oversample
      classes with severe imbalance.
    * Snapshot policy (weighted): 0.7*doc_role_macro_f1 +
      0.3*boilerplate_pos_f1.

Outputs (mirrors Structure):
    models/council/semantic/final/adapter_model.safetensors
    models/council/semantic/final/heads.pt
    models/council/semantic/final/tokenizer/
    models/council/semantic/final/summary.json
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

from data.build_semantic_data import (
    CASCADE_DIM,
    DOC_ROLE_NAMES,
    LAYOUT_FEATURE_DIM,
    NUM_DOC_ROLES,
    POSITIONAL_FEATURE_DIM,
    POSITIONAL_FEATURE_NAMES,
)


# Side-channel layout: 20-dim layout || 8-dim cascade || 3-dim positional
# = 31-dim. Phase 3c precision-improvement: positional features expose
# title/author's strong ordinal-position signal, which the layout vec
# (page-relative only) cannot. Positional is Semantic-ONLY; the trained
# Structure adapter remains on a 20-dim layout contract.
SIDE_CHANNEL_DIM = LAYOUT_FEATURE_DIM + CASCADE_DIM + POSITIONAL_FEATURE_DIM
SIDE_MLP_HIDDEN = 64

BOILERPLATE_LABELS = ("not_boilerplate", "boilerplate")


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
    """Phase 3c is span-level; layout + cascade carry the structural
    signal directly. The model input is just the span text."""
    return (row.get("text") or "").strip()


def load_split(path: Path, *, limit: int | None = None) -> Dataset:
    rows: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        labels = r.get("labels") or {}
        if "doc_role" not in labels or "boilerplate" not in labels:
            skipped += 1
            continue
        layout = r.get("layout")
        cascade = r.get("cascade")
        if layout is None or cascade is None:
            skipped += 1
            continue
        # Positional is a Phase 3c addition; default to zeros for back-
        # compat with datasets built before POSITIONAL_FEATURE_DIM landed.
        # New builds always emit a 3-dim float vector.
        positional = r.get("positional")
        if positional is None:
            positional = [0.0] * POSITIONAL_FEATURE_DIM
        rows.append({
            "text": _format_input(r),
            "layout": list(layout),
            "cascade": list(cascade),
            "positional": list(positional),
            "y_doc_role": int(labels["doc_role"]),
            "y_boilerplate": int(labels["boilerplate"]),
            "source": r.get("source", "unknown"),
        })
        if limit is not None and len(rows) >= limit:
            break
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


def snapshot_score_weighted(
    doc_role_macro_f1: float, boilerplate_pos_f1: float,
) -> float:
    """Phase 3c snapshot score: 0.7*doc_role_macro + 0.3*boilerplate_pos."""
    return 0.7 * doc_role_macro_f1 + 0.3 * boilerplate_pos_f1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SemanticModel(nn.Module):
    """ModernBERT-base + LoRA, two span-level classification heads with a
    31-dim numeric side-channel = 20-dim layout || 8-dim cascade ||
    3-dim positional.

    The cascade vector is teacher-forced: it's [P(role)x6, P(is_heading),
    P(table_region)] from the trained Structure adapter, computed at
    data-build time and baked into the JSONL. Phase 3c never re-runs
    Structure.

    The positional vector (block_index_norm, page_index_norm,
    doc_position_norm) is Semantic-only. It exposes ordinal location in
    the document — title is doc-block 0-3, author is 1-5, etc. — which
    the page-relative layout vec cannot encode. Adding it bumps the
    side-channel to 31-dim WITHOUT touching the Structure adapter's
    20-dim layout contract.
    """

    def __init__(
        self,
        base_model_name: str,
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        side_channel_dim: int = SIDE_CHANNEL_DIM,
        side_mlp_hidden: int = SIDE_MLP_HIDDEN,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        base = AutoModel.from_pretrained(base_model_name, dtype=dtype)
        target_modules = [
            "Wqkv", "Wo",
            "query_proj", "key_proj", "value_proj",
        ]
        peft_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(base, peft_cfg)
        hidden = base.config.hidden_size

        self.side_channel_dim = int(side_channel_dim)
        self.side_mlp_hidden = int(side_mlp_hidden)
        self.side_norm = nn.LayerNorm(self.side_channel_dim)
        self.side_mlp = nn.Sequential(
            nn.Linear(self.side_channel_dim, self.side_mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.side_mlp_hidden, self.side_mlp_hidden),
            nn.ReLU(),
        )

        head_in = hidden + self.side_mlp_hidden
        self.head_doc_role = nn.Linear(head_in, NUM_DOC_ROLES)
        self.head_boilerplate = nn.Linear(head_in, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        side_channel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        pooled = last[:, 0, :].float()
        side_h = self.side_mlp(self.side_norm(side_channel.float()))
        h = torch.cat([pooled, side_h], dim=-1)
        return {
            "doc_role": self.head_doc_role(h),
            "boilerplate": self.head_boilerplate(h),
        }

    def save_adapter(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(str(out_dir))
        torch.save(
            {
                "head_doc_role.state_dict": self.head_doc_role.state_dict(),
                "head_boilerplate.state_dict":
                    self.head_boilerplate.state_dict(),
                "side_norm.state_dict": self.side_norm.state_dict(),
                "side_mlp.state_dict": self.side_mlp.state_dict(),
                "side_channel_dim": self.side_channel_dim,
                "side_mlp_hidden": self.side_mlp_hidden,
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
        out["y_doc_role"] = torch.tensor(
            [b["y_doc_role"] for b in batch], dtype=torch.long,
        )
        out["y_boilerplate"] = torch.tensor(
            [b["y_boilerplate"] for b in batch], dtype=torch.long,
        )
        layout = torch.tensor(
            [b["layout"] for b in batch], dtype=torch.float32,
        )
        cascade = torch.tensor(
            [b["cascade"] for b in batch], dtype=torch.float32,
        )
        positional = torch.tensor(
            [b["positional"] for b in batch], dtype=torch.float32,
        )
        out["side_channel"] = torch.cat(
            [layout, cascade, positional], dim=-1,
        )
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
    model: SemanticModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    dr_t, dr_p = [], []
    bp_t, bp_p = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            side = batch["side_channel"].to(device)
            out = model(ids, mask, side)
            p_dr = out["doc_role"].argmax(-1).tolist()
            p_bp = out["boilerplate"].argmax(-1).tolist()
            ydr = batch["y_doc_role"].tolist()
            ybp = batch["y_boilerplate"].tolist()
            for i in range(len(ydr)):
                dr_t.append(ydr[i]); dr_p.append(p_dr[i])
                bp_t.append(ybp[i]); bp_p.append(p_bp[i])
    dr_macro = f1_score(dr_t, dr_p, average="macro", zero_division=0)
    bp_pos = f1_score(bp_t, bp_p, pos_label=1, zero_division=0)
    return {
        "doc_role_macro_f1": float(dr_macro),
        "boilerplate_pos_f1": float(bp_pos),
        "raw": {
            "doc_role": (dr_t, dr_p),
            "boilerplate": (bp_t, bp_p),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("data/semantic_dataset"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/council/semantic"))
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--weight-cap", type=float, default=30.0)
    ap.add_argument("--sampler-weight-cap", type=float, default=8.0,
                    help="Cap on per-row WeightedRandomSampler weight "
                         "(applied AFTER class-weight lookup, BEFORE "
                         "sampler construction). Mirrors Phase 3b's "
                         "is_heading_sampler_cap pattern. Phase 3c test "
                         "diagnosis: author was sampled 8.97x and title "
                         "P=0.679 / R=0.967 — over-prediction. CE class "
                         "weights stay at weight_cap=30 (unaffected); "
                         "this just throttles the sampler.")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no-oversample", action="store_true")
    ap.add_argument("--snapshot-policy",
                    choices=("role_only", "weighted"),
                    default="weighted",
                    help="role_only: max(val_doc_role_macro_f1). "
                         "weighted: 0.7*doc_role + 0.3*boilerplate_pos.")
    ap.add_argument("--limit-train-rows", type=int, default=None,
                    help="Smoke flag: cap train_ds rows for fast wiring "
                         "tests. Do not use for production training.")
    ap.add_argument("--limit-val-rows", type=int, default=None,
                    help="Smoke flag: cap val_ds rows.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gpu] cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    print(f"[data] loading {args.dataset_dir}")
    train_ds = load_split(
        args.dataset_dir / "train.jsonl", limit=args.limit_train_rows,
    )
    val_ds = load_split(
        args.dataset_dir / "val.jsonl", limit=args.limit_val_rows,
    )
    test_ds = load_split(args.dataset_dir / "test.jsonl")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    print("[balance] doc_role (train):")
    print(_label_count_table(train_ds["y_doc_role"], DOC_ROLE_NAMES))
    print("[balance] boilerplate (train):")
    print(_label_count_table(train_ds["y_boilerplate"], BOILERPLATE_LABELS))

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    cw_doc_role = compute_class_weights(
        train_ds["y_doc_role"], NUM_DOC_ROLES, weight_cap=args.weight_cap,
    ).to(device)
    cw_boilerplate = compute_class_weights(
        train_ds["y_boilerplate"], 2, weight_cap=args.weight_cap,
    ).to(device)
    print(f"[weights] doc_role:    {cw_doc_role.tolist()}")
    print(f"[weights] boilerplate: {cw_boilerplate.tolist()}")

    sample_w = None
    if not args.no_oversample:
        # Per-row weight = cw_doc_role[y_doc_role]. Boilerplate is NOT
        # included — naturally balanced at 5.6%, and oversampling on it
        # would over-rotate doc_role training (Phase 3a's lesson).
        ydr_arr = torch.tensor(train_ds["y_doc_role"], dtype=torch.long)
        cw_dr_cpu = cw_doc_role.detach().cpu()
        sample_w = cw_dr_cpu[ydr_arr].to(torch.float32)

        # Pre-clip stats — visible in train log so the diff is clear.
        pre_max = float(sample_w.max())
        pre_mean = float(sample_w.mean())
        # Apply sampler cap. Phase 3c diagnosis: title P=0.679 / R=0.967,
        # author P=0.746 / R=0.995 — high recall + low precision driven
        # by an 8.97x oversample on author. Clipping the per-row weight
        # (NOT the CE class weight) throttles the sampler without
        # weakening the loss signal. Same logic as Phase 3b's
        # is_heading_sampler_cap.
        sample_w = sample_w.clamp(max=args.sampler_weight_cap)
        post_max = float(sample_w.max())
        post_mean = float(sample_w.mean())
        print(f"[oversample] sampler_weight_cap={args.sampler_weight_cap}  "
              f"pre-clip max={pre_max:.3f} mean={pre_mean:.3f}  "
              f"post-clip max={post_max:.3f} mean={post_mean:.3f}")
        for cls in sorted(set(ydr_arr.tolist())):
            mask = ydr_arr == cls
            if mask.any():
                mw = float(sample_w[mask].mean())
                n = int(mask.sum())
                print(f"[oversample] mean weight  doc_role={cls}  "
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
    print(f"[model] side_channel_dim={SIDE_CHANNEL_DIM} "
          f"(layout={LAYOUT_FEATURE_DIM} + cascade={CASCADE_DIM} + "
          f"positional={POSITIONAL_FEATURE_DIM})")
    model = SemanticModel(
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
        sums = {"loss": 0.0, "doc_role": 0.0, "boilerplate": 0.0}
        n_batches = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            side = batch["side_channel"].to(device)
            ydr = batch["y_doc_role"].to(device)
            ybp = batch["y_boilerplate"].to(device)
            out = model(ids, mask, side)
            loss_dr = F.cross_entropy(
                out["doc_role"], ydr, weight=cw_doc_role,
            )
            loss_bp = F.cross_entropy(
                out["boilerplate"], ybp, weight=cw_boilerplate,
            )
            loss = loss_dr + loss_bp
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            sums["loss"] += float(loss.item())
            sums["doc_role"] += float(loss_dr.item())
            sums["boilerplate"] += float(loss_bp.item())
            n_batches += 1
            if torch.cuda.is_available():
                peak_vram_bytes = max(
                    peak_vram_bytes,
                    torch.cuda.max_memory_allocated(device),
                )

        ep_dt = time.time() - ep_t0
        avg = {k: v / max(1, n_batches) for k, v in sums.items()}
        val_metrics = evaluate(model, val_loader, device)

        if args.snapshot_policy == "role_only":
            score = val_metrics["doc_role_macro_f1"]
        else:  # weighted
            score = snapshot_score_weighted(
                val_metrics["doc_role_macro_f1"],
                val_metrics["boilerplate_pos_f1"],
            )

        print(
            f"[epoch {epoch:02d}] loss={avg['loss']:.4f} "
            f"dr={avg['doc_role']:.4f} bp={avg['boilerplate']:.4f}  "
            f"val dr_macro={val_metrics['doc_role_macro_f1']:.4f} "
            f"bp_pos={val_metrics['boilerplate_pos_f1']:.4f}  "
            f"score={score:.4f}  ({ep_dt:.1f}s)",
            flush=True,
        )
        train_log.append({
            "epoch": epoch,
            "loss": avg["loss"],
            "loss_doc_role": avg["doc_role"],
            "loss_boilerplate": avg["boilerplate"],
            "val_doc_role_macro_f1": val_metrics["doc_role_macro_f1"],
            "val_boilerplate_pos_f1": val_metrics["boilerplate_pos_f1"],
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
                    "lora_", "head_doc_role", "head_boilerplate",
                    "side_norm", "side_mlp",
                ))
            }
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"[early-stop] no improvement for "
                      f"{args.patience} epochs (best epoch={best_epoch})",
                      flush=True)
                break

    total_dt = time.time() - t0
    print(f"[train] done in {total_dt:.1f}s; best epoch={best_epoch} "
          f"snapshot_score={best_val:.4f}", flush=True)

    if best_state is not None:
        full_state = model.state_dict()
        for k, v in best_state.items():
            full_state[k] = v.to(full_state[k].device)
        model.load_state_dict(full_state)

    # ------ Held-out test ------
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n[test] doc_role macro-F1            = "
          f"{test_metrics['doc_role_macro_f1']:.4f}")
    print(f"[test] boilerplate pos-F1           = "
          f"{test_metrics['boilerplate_pos_f1']:.4f}")

    # Per-class reports.
    dr_t, dr_p = test_metrics["raw"]["doc_role"]
    print("\n[test] doc_role per-class:")
    print(classification_report(
        dr_t, dr_p,
        labels=list(range(NUM_DOC_ROLES)),
        target_names=list(DOC_ROLE_NAMES),
        zero_division=0, digits=3,
    ))
    bp_t, bp_p = test_metrics["raw"]["boilerplate"]
    print("[test] boilerplate per-class:")
    print(classification_report(
        bp_t, bp_p,
        labels=[0, 1],
        target_names=BOILERPLATE_LABELS,
        zero_division=0, digits=3,
    ))

    # ------ Save ------
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_adapter(final_dir)
    tok.save_pretrained(str(final_dir / "tokenizer"))
    summary = {
        "base_model": args.base_model,
        "role_names": list(DOC_ROLE_NAMES),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "snapshot_policy": args.snapshot_policy,
        "snapshot_formula": "0.7*doc_role_macro_f1 + 0.3*boilerplate_pos_f1",
        "best_epoch": best_epoch,
        "best_snapshot_score": best_val,
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
        "cascade_dim": CASCADE_DIM,
        "positional_dim": POSITIONAL_FEATURE_DIM,
        "positional_feature_names": list(POSITIONAL_FEATURE_NAMES),
        "side_channel_dim": SIDE_CHANNEL_DIM,
        "side_mlp_hidden": SIDE_MLP_HIDDEN,
        "sampler_weight_cap": args.sampler_weight_cap,
        "test_doc_role_macro_f1": test_metrics["doc_role_macro_f1"],
        "test_boilerplate_pos_f1": test_metrics["boilerplate_pos_f1"],
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
