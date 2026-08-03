"""Train BERT-Structure (Phase 3b — ModernBERT-base + LoRA, six heads).

Train the council's span-level structure specialist. Its role head provides
the council counterpart to the compatibility v1 single-head classifier, while
five additional heads supply routing, hierarchy, and pedagogical signals.

Heads:
    * structural_role     9-class     Active Role enum subset only
                                      (paragraph, heading, list_item,
                                      form_label, blockquote, code_block,
                                      definition_term, definition_def,
                                      caption) — see
                                      data.builders.build_structure_data for
                                      rationale. Non-authoritative
                                      recommendation; downstream
                                      specialists override per their
                                      domain.
    * is_heading          binary      Span-level heading detection,
                                      used to gate heading-level analysis.
                                      heading-level (h1..h6) is the
                                      HeadingSpecialist's job, gated on
                                      is_heading=1.
    * table_region        binary      Span-level table-membership
                                      detection. table_region=1 hands
                                      the span to the TableSpecialist
                                      (Phase 3e) for cell-level role +
                                      scope; same gating pattern as
                                      is_heading→HeadingSpecialist.
    * is_image_block      binary      Span-level figure/image-block
                                      membership. Runtime routing treats this
                                      as secondary confirmation for extracted
                                      image candidates and preserves strong
                                      prose-path hits for figure formation.
    * list_nesting        4-class     <ul>/<ol> ancestor count of <li>:
                                      0=not in list, 1=top-level li,
                                      2=nested, 3=3+ deep.
    * pedagogical_role   10-class     Orthogonal teaching-function signal:
                                      none plus example, exercise, solution,
                                      practice, and related opening roles.

Layout side-channel: 20-dim numeric vector → LayerNorm → 64-dim MLP →
concatenated with BERT pooled before all six heads. Same pattern as
Phase 3a v4 MergeOrSplit. Order in
``data.builders.build_structure_data.LAYOUT_FEATURE_NAMES``.

Hyperparameter recipe (matches Phase 2/3a):
    * ModernBERT-base, bf16 encoder, fp32 heads
    * LoRA r=16 alpha=32, dropout=0.05
    * 6 epochs, AdamW lr=2e-4 weight_decay=0.01, cosine LR schedule
    * Class-weighted CE per head (weight_cap=30)
    * Multi-head WeightedRandomSampler — per-row weight =
      max(cw_role[y_role], cw_is_heading[y_is_heading]) so heading
      positives (~5-15% of spans) and rare structural roles get
      sampled at the rare-class rate. table_region (~12%) and
      list_nesting deliberately excluded from sampler weighting
      (binary balance is OK from CE alone; depth=2/3 over-sampling
      would replay the Phase-3a heading-precision crash).
    * Snapshot policy (weighted_macro): 0.45*role_macro_f1 +
      0.20*is_heading_f1 + 0.18*table_region_f1 +
      0.12*is_image_block_f1 + 0.05*(1 - ln_mae/3).

Outputs (mirrors MergeOrSplit):
    models/council/structure/final/adapter_model.safetensors
    models/council/structure/final/heads.pt
    models/council/structure/final/tokenizer/
    models/council/structure/final/summary.json
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

from data.builders.build_structure_data import (  # noqa: E402  (after sys.path bootstrap)
    LAYOUT_FEATURE_DIM,
    LIST_NESTING_BUCKETS,
    NUM_PEDAGOGICAL_ROLES,
    NUM_ROLES,
    PEDAGOGICAL_ROLE_NAMES,
    ROLE_NAMES,
)


LAYOUT_MLP_HIDDEN = 64

NUM_LIST_NESTING = len(LIST_NESTING_BUCKETS)  # 4

IS_HEADING_LABELS = ("not_heading", "heading")
TABLE_REGION_LABELS = ("not_table_region", "table_region")
IS_IMAGE_BLOCK_LABELS = ("not_image_block", "image_block")
LIST_NESTING_NAMES = [f"depth={d}" for d in LIST_NESTING_BUCKETS]
PEDAGOGICAL_ROLE_LABELS = PEDAGOGICAL_ROLE_NAMES


def _label_count_table(labels: list[int], names, *, ignore: int | None = None) -> str:
    counts = Counter(l for l in labels if ignore is None or l != ignore)
    lines = []
    for i, name in enumerate(names):
        c = counts.get(i, 0)
        lines.append(f"  {name:24} {c}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _format_input(row: dict) -> str:
    """Build the model input string. Phase 3b is span-level (not pair-
    level), so the input is just the span text — the layout side-channel
    carries the geometric signal directly without lossy bucketed prefix
    tokens. We add the html_tag as a coarse prior token only when it's
    meaningful (heading / list / table cell)."""
    text = (row.get("text") or "").strip()
    return text


def load_split(path: Path) -> Dataset:
    rows: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        labels = r.get("labels") or {}
        if "structural_role" not in labels:
            skipped += 1
            continue
        rows.append({
            "text": _format_input(r),
            "layout": r["layout"],
            "y_role": int(labels["structural_role"]),
            "y_is_heading": int(labels["is_heading"]),
            "y_table_region": int(labels["table_region"]),
            # Backwards-compat: older datasets may not carry is_image_block.
            # Default to 0 (negative) so the trainer can still ingest them
            # — but the dataset SHOULD be rebuilt for a real Phase 3f run.
            "y_is_image_block": int(labels.get("is_image_block", 0)),
            "y_list_nesting": int(labels["list_nesting"]),
            # AXIS-2: backward-compat — pre-AXIS-2 rows (and the byte-stable
            # greedy dataset) carry no pedagogical_role; default to 0 (none),
            # mirroring is_image_block. A v3 row carries the real label.
            "y_pedagogical_role": int(labels.get("pedagogical_role", 0)),
            "source": r.get("source", "unknown"),
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


class StructureModel(nn.Module):
    """ModernBERT-base + LoRA, four span-level classification heads with
    an explicit numeric layout side-channel.

    The layout vector (20-dim, see
    ``data.builders.build_structure_data.LAYOUT_FEATURE_NAMES``) carries the raw
    font_size_relative_to_page_median, bold/italic flags, position, and
    text-shape features that the encoder otherwise has to read out of
    the text alone. Same pattern as Phase 3a v4 MergeOrSplit; same
    rationale (bucketed prefix tokens lost too much of the heading
    signal).
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
        # ModernBERT defaults reference_compile=True when CUDA is present, which
        # routes the forward through torch.compile -> Triton -> a JIT gcc build
        # of cuda_utils.c. That JIT build fails on some boxes (e.g. WSL2: the
        # triton driver's `gcc ... -l:libcuda.so.1` returns non-zero), aborting
        # training the instant the Trainer touches a compiled kernel. Load in
        # eager mode so training never depends on the Triton JIT toolchain.
        # Tolerate non-ModernBERT bases that don't carry the config field.
        try:
            base = AutoModel.from_pretrained(
                base_model_name, dtype=dtype, reference_compile=False
            )
        except (TypeError, ValueError):
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
        self.head_role = nn.Linear(head_in, NUM_ROLES)
        self.head_is_heading = nn.Linear(head_in, 2)
        self.head_table_region = nn.Linear(head_in, 2)
        self.head_is_image_block = nn.Linear(head_in, 2)
        self.head_list_nesting = nn.Linear(head_in, NUM_LIST_NESTING)
        self.head_pedagogical_role = nn.Linear(head_in, NUM_PEDAGOGICAL_ROLES)

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
            "role": self.head_role(h),
            "is_heading": self.head_is_heading(h),
            "table_region": self.head_table_region(h),
            "is_image_block": self.head_is_image_block(h),
            "list_nesting": self.head_list_nesting(h),
            "pedagogical_role": self.head_pedagogical_role(h),
        }

    def save_adapter(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(str(out_dir))
        torch.save(
            {
                "head_role.state_dict": self.head_role.state_dict(),
                "head_is_heading.state_dict":
                    self.head_is_heading.state_dict(),
                "head_table_region.state_dict":
                    self.head_table_region.state_dict(),
                "head_is_image_block.state_dict":
                    self.head_is_image_block.state_dict(),
                "head_list_nesting.state_dict":
                    self.head_list_nesting.state_dict(),
                "head_pedagogical_role.state_dict":
                    self.head_pedagogical_role.state_dict(),
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
        out["y_role"] = torch.tensor(
            [b["y_role"] for b in batch], dtype=torch.long,
        )
        out["y_is_heading"] = torch.tensor(
            [b["y_is_heading"] for b in batch], dtype=torch.long,
        )
        out["y_table_region"] = torch.tensor(
            [b["y_table_region"] for b in batch], dtype=torch.long,
        )
        out["y_is_image_block"] = torch.tensor(
            [b["y_is_image_block"] for b in batch], dtype=torch.long,
        )
        out["y_list_nesting"] = torch.tensor(
            [b["y_list_nesting"] for b in batch], dtype=torch.long,
        )
        out["y_pedagogical_role"] = torch.tensor(
            [b["y_pedagogical_role"] for b in batch], dtype=torch.long,
        )
        out["layout"] = torch.tensor(
            [b["layout"] for b in batch], dtype=torch.float32,
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


def metrics_from_collected(
    role_t, role_p,
    is_h_t, is_h_p,
    tr_t, tr_p,
    ib_t, ib_p,
    ln_t, ln_p,
    pr_t, pr_p,
) -> dict[str, object]:
    """Compute the per-head metric dict from already-collected (true, pred)
    label lists. Factored out of ``evaluate`` so an external benchmark harness
    (``SemantiK/scripts/calibration/benchmark_structure_backbones.py``) can score every
    backbone arm through the SAME metric path — apples-to-apples — without
    re-implementing (and silently drifting from) this formula. No behaviour
    change: ``evaluate`` delegates here."""
    role_macro = f1_score(role_t, role_p, average="macro", zero_division=0)
    is_h_f1 = f1_score(is_h_t, is_h_p, pos_label=1, zero_division=0)
    tr_f1 = f1_score(tr_t, tr_p, pos_label=1, zero_division=0)
    ib_f1 = f1_score(ib_t, ib_p, pos_label=1, zero_division=0)
    ln_mae = (
        sum(abs(t - p) for t, p in zip(ln_t, ln_p)) / max(1, len(ln_t))
    )
    # pedagogical_role: macro-F1 over the NON-none classes only (class 0 is
    # the overwhelming majority and its F1 would mask the rare-positive
    # performance that matters). Empty positive set -> 0.0.
    ped_pos_labels = list(range(1, NUM_PEDAGOGICAL_ROLES))
    pr_macro = f1_score(
        pr_t, pr_p, labels=ped_pos_labels, average="macro", zero_division=0
    )
    return {
        "role_macro_f1": float(role_macro),
        "is_heading_pos_f1": float(is_h_f1),
        "table_region_pos_f1": float(tr_f1),
        "is_image_block_pos_f1": float(ib_f1),
        "list_nesting_mae": float(ln_mae),
        "pedagogical_role_macro_f1": float(pr_macro),
        "raw": {
            "role": (role_t, role_p),
            "is_heading": (is_h_t, is_h_p),
            "table_region": (tr_t, tr_p),
            "is_image_block": (ib_t, ib_p),
            "list_nesting": (ln_t, ln_p),
            "pedagogical_role": (pr_t, pr_p),
        },
    }


def evaluate(
    model: StructureModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    role_t, role_p = [], []
    is_h_t, is_h_p = [], []
    tr_t, tr_p = [], []
    ib_t, ib_p = [], []
    ln_t, ln_p = [], []
    pr_t, pr_p = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            layout = batch["layout"].to(device)
            out = model(ids, mask, layout)
            p_role = out["role"].argmax(-1).tolist()
            p_h = out["is_heading"].argmax(-1).tolist()
            p_tr = out["table_region"].argmax(-1).tolist()
            p_ib = out["is_image_block"].argmax(-1).tolist()
            p_ln = out["list_nesting"].argmax(-1).tolist()
            p_pr = out["pedagogical_role"].argmax(-1).tolist()
            yr = batch["y_role"].tolist()
            yh = batch["y_is_heading"].tolist()
            ytr = batch["y_table_region"].tolist()
            yib = batch["y_is_image_block"].tolist()
            yln = batch["y_list_nesting"].tolist()
            ypr = batch["y_pedagogical_role"].tolist()
            for i in range(len(yr)):
                # -100 = masked synth table-cell role/is_heading: exclude from
                # the metric so role macro-F1 is measured on real prose roles.
                if yr[i] != -100:
                    role_t.append(yr[i]); role_p.append(p_role[i])
                if yh[i] != -100:
                    is_h_t.append(yh[i]); is_h_p.append(p_h[i])
                tr_t.append(ytr[i]); tr_p.append(p_tr[i])
                ib_t.append(yib[i]); ib_p.append(p_ib[i])
                ln_t.append(yln[i]); ln_p.append(p_ln[i])
                pr_t.append(ypr[i]); pr_p.append(p_pr[i])
    return metrics_from_collected(
        role_t, role_p, is_h_t, is_h_p, tr_t, tr_p,
        ib_t, ib_p, ln_t, ln_p, pr_t, pr_p,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path("data/structure_dataset"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("models/council/structure"))
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--weight-cap", type=float, default=30.0)
    ap.add_argument("--is-heading-sampler-cap", type=float, default=8.0,
                    help="Cap on is_heading-positive sample weight in the "
                         "multi-head sampler — lower than the CE class "
                         "weight cap to avoid the precision-crash that "
                         "killed Phase 3a's pair-level heading head.")
    ap.add_argument("--pedagogical-sampler-cap", type=float, default=8.0,
                    help="Cap on the per-positive-class pedagogical_role "
                         "sample weight in the multi-head sampler (AXIS-2). "
                         "Mirrors --is-heading-sampler-cap: oversamples the "
                         "rare EXAMPLE/TRY-IT/Solution labels without letting "
                         "them over-distort the role objective.")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no-oversample", action="store_true")
    ap.add_argument("--snapshot-policy",
                    choices=("role_only", "weighted_macro"),
                    default="weighted_macro",
                    help="role_only: max(val_role_macro_f1). "
                         "weighted_macro: 0.5*role + 0.25*is_heading + "
                         "0.20*table_region + 0.05*(1-ln_mae/3).")
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

    print("[balance] structural_role (train):")
    print(_label_count_table(train_ds["y_role"], ROLE_NAMES))
    print("[balance] is_heading (train):")
    print(_label_count_table(train_ds["y_is_heading"], IS_HEADING_LABELS))
    print("[balance] table_region (train):")
    print(_label_count_table(train_ds["y_table_region"], TABLE_REGION_LABELS))
    print("[balance] is_image_block (train):")
    print(_label_count_table(
        train_ds["y_is_image_block"], IS_IMAGE_BLOCK_LABELS,
    ))
    print("[balance] list_nesting (train):")
    print(_label_count_table(train_ds["y_list_nesting"], LIST_NESTING_NAMES))
    print("[balance] pedagogical_role (train):")
    print(_label_count_table(
        train_ds["y_pedagogical_role"], PEDAGOGICAL_ROLE_LABELS,
    ))

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    # role/is_heading carry -100 on synth table-cell rows (masked in the
    # dataset merge so the table augmentation can't skew the role objective);
    # exclude them from the class-weight counts. -100 is the CE ignore_index.
    cw_role = compute_class_weights(
        train_ds["y_role"], NUM_ROLES, weight_cap=args.weight_cap,
        ignore=-100,
    ).to(device)
    cw_is_heading = compute_class_weights(
        train_ds["y_is_heading"], 2, weight_cap=args.weight_cap,
        ignore=-100,
    ).to(device)
    cw_table_region = compute_class_weights(
        train_ds["y_table_region"], 2, weight_cap=args.weight_cap,
    ).to(device)
    cw_is_image_block = compute_class_weights(
        train_ds["y_is_image_block"], 2, weight_cap=args.weight_cap,
    ).to(device)
    cw_list_nesting = compute_class_weights(
        train_ds["y_list_nesting"], NUM_LIST_NESTING,
        weight_cap=args.weight_cap,
    ).to(device)
    cw_pedagogical_role = compute_class_weights(
        train_ds["y_pedagogical_role"], NUM_PEDAGOGICAL_ROLES,
        weight_cap=args.weight_cap,
    ).to(device)
    print(f"[weights] role:            {cw_role.tolist()}")
    print(f"[weights] is_heading:      {cw_is_heading.tolist()}")
    print(f"[weights] table_region:    {cw_table_region.tolist()}")
    print(f"[weights] is_image_block:  {cw_is_image_block.tolist()}")
    print(f"[weights] list_nesting:    {cw_list_nesting.tolist()}")
    print(f"[weights] pedagogical_role:{cw_pedagogical_role.tolist()}")

    sample_w = None
    if not args.no_oversample:
        # Multi-head per-row weight: max(cw_role[y_role],
        # cw_is_heading[y_is_heading]). Heading positives get sampled
        # at the rare-class rate; rare structural roles get sampled at
        # their own rate. The is_heading sampler cap prevents heading
        # positives from over-distorting role training (Phase 3a's
        # lesson: cap=24 made the heading head over-predict; cap=8 was
        # the right balance).
        yr_arr = torch.tensor(train_ds["y_role"], dtype=torch.long)
        yh_arr = torch.tensor(train_ds["y_is_heading"], dtype=torch.long)
        ypr_arr = torch.tensor(train_ds["y_pedagogical_role"], dtype=torch.long)
        cw_role_cpu = cw_role.detach().cpu()
        cw_h_cpu = cw_is_heading.detach().cpu().clone()
        cw_h_cpu[1] = min(float(cw_h_cpu[1]),
                          float(args.is_heading_sampler_cap))
        # AXIS-2: oversample rare pedagogical_role POSITIVES (classes 1..N) so
        # the new head sees TRY-IT/EXAMPLE/Solution at the rare-class rate; the
        # `none` class (0) keeps weight 1.0 (it is the majority and must not be
        # up-sampled). Capped (mirrors the is_heading sampler cap) so the rare
        # pedagogical labels do not over-distort the role objective.
        cw_pr_cpu = cw_pedagogical_role.detach().cpu().clone()
        cw_pr_cpu[0] = 1.0
        for c in range(1, NUM_PEDAGOGICAL_ROLES):
            cw_pr_cpu[c] = min(float(cw_pr_cpu[c]),
                               float(args.pedagogical_sampler_cap))
        # Masked rows (role/is_heading == -100, synth table cells) must not
        # index cw[-100] (silent negative-index of the last class). Clamp the
        # index to 0 for the lookup, then force those rows' role/heading
        # sampling weight to a neutral 1.0 — they're kept at base sampling
        # rate to feed table_region/image/nesting, not up-weighted by a role
        # they don't supervise.
        role_w = cw_role_cpu[yr_arr.clamp(min=0)]
        head_w = cw_h_cpu[yh_arr.clamp(min=0)]
        ped_w = cw_pr_cpu[ypr_arr.clamp(min=0)]
        role_w = torch.where(yr_arr < 0, torch.ones_like(role_w), role_w)
        head_w = torch.where(yh_arr < 0, torch.ones_like(head_w), head_w)
        ped_w = torch.where(ypr_arr < 0, torch.ones_like(ped_w), ped_w)
        per_row = torch.stack([role_w, head_w, ped_w], dim=0)
        sample_w = per_row.max(dim=0).values.to(torch.float32)
        print(f"[oversample] is_heading sampler cap = "
              f"{args.is_heading_sampler_cap}")
        for nm, arr in (
            ("y_role", yr_arr),
            ("y_is_heading", yh_arr),
            ("y_pedagogical_role", ypr_arr),
        ):
            for cls in sorted(set(arr.tolist())):
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
    model = StructureModel(
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
        sums = {"loss": 0.0, "role": 0.0, "is_heading": 0.0,
                "table_region": 0.0, "is_image_block": 0.0,
                "list_nesting": 0.0, "pedagogical_role": 0.0}
        n_batches = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            layout = batch["layout"].to(device)
            yr = batch["y_role"].to(device)
            yh = batch["y_is_heading"].to(device)
            ytr = batch["y_table_region"].to(device)
            yib = batch["y_is_image_block"].to(device)
            yln = batch["y_list_nesting"].to(device)
            ypr = batch["y_pedagogical_role"].to(device)
            out = model(ids, mask, layout)
            loss_role = F.cross_entropy(out["role"], yr, weight=cw_role)
            loss_h = F.cross_entropy(
                out["is_heading"], yh, weight=cw_is_heading,
            )
            loss_tr = F.cross_entropy(
                out["table_region"], ytr, weight=cw_table_region,
            )
            loss_ib = F.cross_entropy(
                out["is_image_block"], yib, weight=cw_is_image_block,
            )
            loss_ln = F.cross_entropy(
                out["list_nesting"], yln, weight=cw_list_nesting,
            )
            loss_pr = F.cross_entropy(
                out["pedagogical_role"], ypr, weight=cw_pedagogical_role,
            )
            loss = loss_role + loss_h + loss_tr + loss_ib + loss_ln + loss_pr
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            sums["loss"] += float(loss.item())
            sums["role"] += float(loss_role.item())
            sums["is_heading"] += float(loss_h.item())
            sums["table_region"] += float(loss_tr.item())
            sums["is_image_block"] += float(loss_ib.item())
            sums["list_nesting"] += float(loss_ln.item())
            sums["pedagogical_role"] += float(loss_pr.item())
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
            score = val_metrics["role_macro_f1"]
        else:  # weighted_macro
            # MAE-to-score: ln_mae of 0 → 1.0, ln_mae of 3 → 0.0.
            ln_score = max(0.0, 1.0 - val_metrics["list_nesting_mae"] / 3.0)
            # AXIS-2: pedagogical_role folded into the snapshot score (weights
            # renormalized to sum to 1.0 with the new 0.12 term).
            score = (
                0.40 * val_metrics["role_macro_f1"]
                + 0.18 * val_metrics["is_heading_pos_f1"]
                + 0.15 * val_metrics["table_region_pos_f1"]
                + 0.10 * val_metrics["is_image_block_pos_f1"]
                + 0.12 * val_metrics["pedagogical_role_macro_f1"]
                + 0.05 * ln_score
            )

        print(
            f"[epoch {epoch:02d}] loss={avg['loss']:.4f} "
            f"role={avg['role']:.4f} ih={avg['is_heading']:.4f} "
            f"tr={avg['table_region']:.4f} "
            f"ib={avg['is_image_block']:.4f} "
            f"ln={avg['list_nesting']:.4f} "
            f"pr={avg['pedagogical_role']:.4f}  "
            f"val role={val_metrics['role_macro_f1']:.4f} "
            f"ih_pos={val_metrics['is_heading_pos_f1']:.4f} "
            f"tr_pos={val_metrics['table_region_pos_f1']:.4f} "
            f"ib_pos={val_metrics['is_image_block_pos_f1']:.4f} "
            f"pr_macro={val_metrics['pedagogical_role_macro_f1']:.4f} "
            f"ln_mae={val_metrics['list_nesting_mae']:.4f}  "
            f"score={score:.4f}  ({ep_dt:.1f}s)",
            flush=True,
        )
        train_log.append({
            "epoch": epoch,
            "loss": avg["loss"], "loss_role": avg["role"],
            "loss_is_heading": avg["is_heading"],
            "loss_table_region": avg["table_region"],
            "loss_is_image_block": avg["is_image_block"],
            "loss_list_nesting": avg["list_nesting"],
            "loss_pedagogical_role": avg["pedagogical_role"],
            "val_role_macro_f1": val_metrics["role_macro_f1"],
            "val_is_heading_pos_f1": val_metrics["is_heading_pos_f1"],
            "val_table_region_pos_f1": val_metrics["table_region_pos_f1"],
            "val_is_image_block_pos_f1":
                val_metrics["is_image_block_pos_f1"],
            "val_pedagogical_role_macro_f1":
                val_metrics["pedagogical_role_macro_f1"],
            "val_list_nesting_mae": val_metrics["list_nesting_mae"],
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
                    "lora_", "head_role", "head_is_heading",
                    "head_table_region", "head_is_image_block",
                    "head_list_nesting", "head_pedagogical_role",
                    "layout_norm", "layout_mlp",
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
    print(f"\n[test] role macro-F1                = "
          f"{test_metrics['role_macro_f1']:.4f}")
    print(f"[test] is_heading pos-F1            = "
          f"{test_metrics['is_heading_pos_f1']:.4f}")
    print(f"[test] table_region pos-F1          = "
          f"{test_metrics['table_region_pos_f1']:.4f}")
    print(f"[test] is_image_block pos-F1        = "
          f"{test_metrics['is_image_block_pos_f1']:.4f}")
    print(f"[test] pedagogical_role macro-F1    = "
          f"{test_metrics['pedagogical_role_macro_f1']:.4f}")
    print(f"[test] list_nesting MAE             = "
          f"{test_metrics['list_nesting_mae']:.4f}")

    # Per-class reports.
    role_t, role_p = test_metrics["raw"]["role"]
    print("\n[test] structural_role per-class:")
    print(classification_report(
        role_t, role_p,
        labels=list(range(NUM_ROLES)),
        target_names=list(ROLE_NAMES),
        zero_division=0, digits=3,
    ))
    is_h_t, is_h_p = test_metrics["raw"]["is_heading"]
    print("[test] is_heading per-class:")
    print(classification_report(
        is_h_t, is_h_p,
        labels=[0, 1],
        target_names=IS_HEADING_LABELS,
        zero_division=0, digits=3,
    ))
    tr_t, tr_p = test_metrics["raw"]["table_region"]
    print("[test] table_region per-class:")
    print(classification_report(
        tr_t, tr_p,
        labels=[0, 1],
        target_names=TABLE_REGION_LABELS,
        zero_division=0, digits=3,
    ))
    ib_t, ib_p = test_metrics["raw"]["is_image_block"]
    print("[test] is_image_block per-class:")
    print(classification_report(
        ib_t, ib_p,
        labels=[0, 1],
        target_names=IS_IMAGE_BLOCK_LABELS,
        zero_division=0, digits=3,
    ))
    ln_t, ln_p = test_metrics["raw"]["list_nesting"]
    print("[test] list_nesting per-class:")
    print(classification_report(
        ln_t, ln_p,
        labels=list(range(NUM_LIST_NESTING)),
        target_names=LIST_NESTING_NAMES,
        zero_division=0, digits=3,
    ))
    pr_t, pr_p = test_metrics["raw"]["pedagogical_role"]
    print("[test] pedagogical_role per-class:")
    print(classification_report(
        pr_t, pr_p,
        labels=list(range(NUM_PEDAGOGICAL_ROLES)),
        target_names=list(PEDAGOGICAL_ROLE_LABELS),
        zero_division=0, digits=3,
    ))

    # ------ Save ------
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_adapter(final_dir)
    tok.save_pretrained(str(final_dir / "tokenizer"))
    summary = {
        "base_model": args.base_model,
        "role_names": ROLE_NAMES,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "snapshot_policy": args.snapshot_policy,
        "best_epoch": best_epoch,
        "best_snapshot_score": best_val,
        "is_heading_sampler_cap": args.is_heading_sampler_cap,
        "pedagogical_sampler_cap": args.pedagogical_sampler_cap,
        "pedagogical_role_names": list(PEDAGOGICAL_ROLE_LABELS),
        "layout_feature_dim": LAYOUT_FEATURE_DIM,
        "layout_mlp_hidden": LAYOUT_MLP_HIDDEN,
        "test_role_macro_f1": test_metrics["role_macro_f1"],
        "test_is_heading_pos_f1": test_metrics["is_heading_pos_f1"],
        "test_table_region_pos_f1": test_metrics["table_region_pos_f1"],
        "test_is_image_block_pos_f1":
            test_metrics["is_image_block_pos_f1"],
        "test_pedagogical_role_macro_f1":
            test_metrics["pedagogical_role_macro_f1"],
        "test_list_nesting_mae": test_metrics["list_nesting_mae"],
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
