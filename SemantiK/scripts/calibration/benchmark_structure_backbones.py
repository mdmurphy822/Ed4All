"""Phase 4 backbone benchmark — layout-native backbones vs ModernBERT+sidechannel.

Roadmap: ``plans/semantik-structure-rearchitecture-2026-06-28.md`` §"Phase 4 —
Backbone benchmark". A go/no-go report (NOT a production swap) that measures
whether a layout-native backbone closes the real-PDF deploy gap on the
genuinely layout-hard heads (``is_image_block`` 0.708, ``table_region``) and
improves cross-corpus generalization.

All arms run in **BLOCK-POOLED Mode B** — one dataset row = one block; the
block text is tokenized, run through the backbone, pooled to a single vector,
and fed the SAME six multi-task heads as
``training/train_structure.py::StructureModel`` so the comparison is
apples-to-apples. Every arm is SCORED through the SAME metric path
(``train_structure.metrics_from_collected``, imported — not re-implemented),
so a cross-arm delta cannot be an artifact of two metric formulas drifting.

Arms (all Mode B):
  * **A0** — ModernBERT-base + the 20-dim layout side-channel = the existing
    ``StructureModel``, reused verbatim and RE-TRAINED under this harness (not
    the cached checkpoint) so it is comparable to A1/A2/A3 under identical
    splits/epochs/seed.
  * **A1** — LiLT (``SCUT-DLVCLab/lilt-roberta-en-base``, MIT). Block text
    tokens + per-token bbox (the block's normalized bbox replicated to every
    token). No 20-dim side-channel. LoRA over the discovered attention
    projections (text + layout flow). Mask-aware mean-pool → six heads.
  * **A2** — LayoutLM v1 (``microsoft/layoutlm-base-uncased``, MIT). input_ids
    + per-token bbox (block bbox replicated, normalized 0-1000). Mean-pool →
    six heads. LoRA over query/key/value.
  * **A3** — ABLATION: A2 (LayoutLM v1) but every token's bbox is the CONSTANT
    ``[0, 0, 1000, 1000]`` (geometry ablated). Isolates whether the
    layout-PRETRAINED backbone helps even WITHOUT real bbox input — attributes
    a win to the backbone vs the token geometry.

LayoutLMv2 / LayoutLMv3 are BARRED (CC BY-NC-SA) — they are never added.

bbox normalization (no page dims are available; documented approximation):
  for each pair (document) compute the max x-extent and max y-extent across
  THAT pair's blocks in the loaded set, then map each block bbox to integer
  [0, 1000] (``x0n = round(1000 * x0 / max_x)`` …), clamp to [0, 1000], and
  enforce ``x0 <= x1`` / ``y0 <= y1``. Per-document (not global) because render
  scale varies between documents. See ``_normalize_doc_bboxes``.

Splits:
  * ``standard`` — the existing train/val/test split: headline per-head metrics.
  * ``loco`` — leave-one-corpus-out, keyed on the per-row ``source`` field: for
    each corpus C with >= ``--loco-min-rows`` rows, train on all-rows-except-C,
    test on C; reports per-(arm × held-out-corpus) role macro-F1 (+ every head).
  * Every trained arm is ALSO scored on the real-PDF v3 gold
    (``data/structure_dataset_realpdf_v3`` shippable +
    ``…_internal_v3`` internal) → union + gold-restricted role macro-F1,
    generalizing ``eval/measure_realpdf_structure``'s predictor to any arm.

License partition (``--license-partition``):
  * ``shippable`` — EXCLUDES ``source == 'openstax'`` (OpenStax is CC-BY-NC-SA,
    internal-calibration-only — must never train a shippable arm) and scores
    real-PDF only on the shippable v3 root.
  * ``internal`` — all corpora; real-PDF on both v3 roots.
  * ``both`` (default) — runs both partitions and labels the results.

VRAM discipline: exactly one backbone is resident at a time
(build → train → eval → free CUDA cache → next). Resumable: a cell
(arm × split × partition [× held-out corpus]) whose result already exists in
the output JSON is skipped, and the JSON is written incrementally after every
cell, so a crash/kill resumes.

Smoke (``--smoke``): tiny offline CI/validation regime — ~200 train rows, 1
epoch, CPU, arms A0 + A2 (two DISTINCT architectures), single ``internal``
partition, ``standard`` split only. The layout arm is built from a TINY random-
init config and uses a cached BERT-WordPiece tokenizer, and A0 reuses the
already-cached ModernBERT-base, so smoke runs fully offline (no weight
downloads) and exercises the real train+eval+realpdf+resume loop end to end.

Usage (real GPU benchmark is wrapped by ``scripts/ops/gpu_guard.sh`` — this script
does NOT self-flock):
    cd SemantiK && ../scripts/ops/gpu_guard.sh run --task bench-standard -- \
      ../.venv/bin/python -m scripts.calibration.benchmark_structure_backbones \
        --splits standard --license-partition both
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import sys

# This benchmark lives under SemantiK/scripts/; put the SemantiK root on the
# import path so the top-level ``data`` / ``training`` namespaces resolve
# regardless of the invocation directory (mirrors train_structure.py).
_SEMANTIK_ROOT = str(Path(__file__).resolve().parents[2])
if _SEMANTIK_ROOT not in sys.path:
    sys.path.insert(0, _SEMANTIK_ROOT)

from data.builders.build_structure_data import (  # noqa: E402
    LIST_NESTING_BUCKETS,
    NUM_PEDAGOGICAL_ROLES,
    NUM_ROLES,
)
from training.train_structure import (  # noqa: E402  (single source of truth)
    StructureModel,
    compute_class_weights,
    metrics_from_collected,
)

NUM_LIST_NESTING = len(LIST_NESTING_BUCKETS)  # 4

SCHEMA = "structure_backbone_benchmark/1.0"

# Real-PDF v3 gold roots (mirrors eval/measure_realpdf_structure defaults, v3).
REALPDF_SHIPPABLE_ROOT = Path("data/structure_dataset_realpdf_v3")
REALPDF_INTERNAL_ROOT = Path("data/structure_dataset_realpdf_internal_v3")

# OpenStax is CC-BY-NC-SA — never enters a shippable arm (training OR eval).
INTERNAL_ONLY_SOURCES = {"openstax"}

# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------
# kind: "modernbert" (A0, layout side-channel) | "layout" (A1/A2/A3, per-token
# bbox). model_cls: HF model class for layout arms. ablate_bbox: A3.
ARM_SPECS: dict[str, dict] = {
    "A0": {
        "kind": "modernbert",
        "base_model": "answerdotai/ModernBERT-base",
        "desc": "ModernBERT-base + 20-dim layout side-channel (StructureModel)",
    },
    "A1": {
        "kind": "layout",
        "hf_model": "SCUT-DLVCLab/lilt-roberta-en-base",
        "model_cls": "LiltModel",
        "tokenizer": "SCUT-DLVCLab/lilt-roberta-en-base",
        "ablate_bbox": False,
        "desc": "LiLT (lilt-roberta-en-base, MIT) — text+layout, per-token bbox",
    },
    "A2": {
        "kind": "layout",
        "hf_model": "microsoft/layoutlm-base-uncased",
        "model_cls": "LayoutLMModel",
        "tokenizer": "microsoft/layoutlm-base-uncased",
        "ablate_bbox": False,
        "desc": "LayoutLM-v1 (layoutlm-base-uncased, MIT) — per-token bbox",
    },
    "A3": {
        "kind": "layout",
        "hf_model": "microsoft/layoutlm-base-uncased",
        "model_cls": "LayoutLMModel",
        "tokenizer": "microsoft/layoutlm-base-uncased",
        "ablate_bbox": True,
        "desc": "ABLATION: LayoutLM-v1 with bbox==[0,0,1000,1000] (geometry off)",
    },
}

# Candidate LoRA target last-segment names (text + LiLT layout flow). Discovered
# per-model against the actually-present nn.Linear modules so the config is
# correct for whichever backbone is loaded; the resolved set is recorded.
_LORA_TARGET_CANDIDATES = (
    "query", "key", "value",
    "layout_query", "layout_key", "layout_value",
)


# ---------------------------------------------------------------------------
# Data loading / bbox normalization
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        labels = r.get("labels") or {}
        if "structural_role" not in labels:
            continue
        rows.append(r)
    return rows


def _doc_key(row: dict) -> str:
    """Stable per-document key for bbox normalization. v3 rows carry ``pair``;
    real-PDF rows additionally carry ``doc_id``/``domain``."""
    if row.get("pair"):
        return f"pair:{row['pair']}"
    return f"doc:{row.get('domain', '?')}/{row.get('doc_id', '?')}"


def _clamp_int(v: float) -> int:
    return max(0, min(1000, int(round(v))))


def _normalize_doc_bboxes(rows: list[dict]) -> None:
    """Stamp ``row['_bbox']`` = per-document-normalized integer [x0,y0,x1,y1]
    in [0, 1000]. Per-document (the ``_doc_key`` grouping) because render scale
    varies between documents; no page dimensions are available, so we use the
    max x/y extent observed across that document's blocks as the scale. A
    document with a degenerate (zero/negative) extent falls back to a zero box
    (geometry simply uninformative for that doc — never crashes)."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_doc[_doc_key(r)].append(r)
    for _, doc_rows in by_doc.items():
        max_x = 0.0
        max_y = 0.0
        for r in doc_rows:
            bb = ((r.get("provenance") or {}).get("bbox")) or [0, 0, 0, 0]
            x0, y0, x1, y1 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
            max_x = max(max_x, x0, x1)
            max_y = max(max_y, y0, y1)
        sx = 1000.0 / max_x if max_x > 0 else 0.0
        sy = 1000.0 / max_y if max_y > 0 else 0.0
        for r in doc_rows:
            bb = ((r.get("provenance") or {}).get("bbox")) or [0, 0, 0, 0]
            x0 = _clamp_int(float(bb[0]) * sx)
            y0 = _clamp_int(float(bb[1]) * sy)
            x1 = _clamp_int(float(bb[2]) * sx)
            y1 = _clamp_int(float(bb[3]) * sy)
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            r["_bbox"] = [x0, y0, x1, y1]


def _label_ints(row: dict) -> dict:
    labels = row["labels"]
    return {
        "y_role": int(labels["structural_role"]),
        "y_is_heading": int(labels.get("is_heading", 0)),
        "y_table_region": int(labels.get("table_region", 0)),
        "y_is_image_block": int(labels.get("is_image_block", 0)),
        "y_list_nesting": int(labels.get("list_nesting", 0)),
        "y_pedagogical_role": int(labels.get("pedagogical_role", 0)),
    }


def _partition_filter(rows: list[dict], partition: str) -> list[dict]:
    if partition == "shippable":
        return [r for r in rows if r.get("source") not in INTERNAL_ONLY_SOURCES]
    return rows  # internal: all corpora


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _discover_lora_targets(model: nn.Module) -> list[str]:
    """Return the subset of ``_LORA_TARGET_CANDIDATES`` that are present as the
    LAST name-segment of an ``nn.Linear`` module — robust across LayoutLM
    (query/key/value) and LiLT (+ layout_query/layout_key/layout_value)."""
    present: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            seg = name.split(".")[-1]
            if seg in _LORA_TARGET_CANDIDATES:
                present.add(seg)
    # Stable order matching the candidate tuple.
    return [c for c in _LORA_TARGET_CANDIDATES if c in present]


class LayoutBackboneModel(nn.Module):
    """Generic Mode-B head wrapper for a layout-native backbone (LiLT /
    LayoutLM). Tokenized block text + per-token bbox → backbone →
    mask-aware mean-pool → the SAME six heads as ``StructureModel`` (minus the
    20-dim side-channel; the backbone consumes geometry through ``bbox``)."""

    def __init__(self, backbone: nn.Module, lora_targets: list[str]):
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model

        peft_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=lora_targets, task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(backbone, peft_cfg)
        hidden = backbone.config.hidden_size
        self.head_role = nn.Linear(hidden, NUM_ROLES)
        self.head_is_heading = nn.Linear(hidden, 2)
        self.head_table_region = nn.Linear(hidden, 2)
        self.head_is_image_block = nn.Linear(hidden, 2)
        self.head_list_nesting = nn.Linear(hidden, NUM_LIST_NESTING)
        self.head_pedagogical_role = nn.Linear(hidden, NUM_PEDAGOGICAL_ROLES)

    def forward(self, input_ids, attention_mask, bbox):
        out = self.encoder(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
        )
        last = out.last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return {
            "role": self.head_role(pooled),
            "is_heading": self.head_is_heading(pooled),
            "table_region": self.head_table_region(pooled),
            "is_image_block": self.head_is_image_block(pooled),
            "list_nesting": self.head_list_nesting(pooled),
            "pedagogical_role": self.head_pedagogical_role(pooled),
        }


def _build_layout_backbone(arm: str, smoke: bool):
    """Build the layout backbone + tokenizer. In smoke mode build a TINY
    random-init config (no weight download) and use a cached BERT-WordPiece
    tokenizer; otherwise load the real MIT weights/tokenizer from HF."""
    import transformers
    from transformers import AutoTokenizer

    spec = ARM_SPECS[arm]
    model_cls = getattr(transformers, spec["model_cls"])
    if smoke:
        cfg_cls = getattr(transformers, spec["model_cls"].replace("Model", "Config"))
        if spec["model_cls"] == "LiltModel":
            cfg = cfg_cls(
                vocab_size=30522, hidden_size=24, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=48,
                max_position_embeddings=130, max_2d_position_embeddings=1024,
                channel_shrink_ratio=2,
            )
        else:  # LayoutLMModel
            cfg = cfg_cls(
                vocab_size=30522, hidden_size=32, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=64,
                max_position_embeddings=130, max_2d_position_embeddings=1024,
            )
        backbone = model_cls(cfg)
        # distilbert-base-uncased is the cached BERT-WordPiece tokenizer
        # (vocab 30522) — offline, matches the tiny config vocab.
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    else:
        backbone = model_cls.from_pretrained(spec["hf_model"])
        tok = AutoTokenizer.from_pretrained(spec["tokenizer"])
    lora_targets = _discover_lora_targets(backbone)
    model = LayoutBackboneModel(backbone, lora_targets)
    return model, tok, lora_targets


def _build_arm(arm: str, smoke: bool, device: torch.device):
    """Return (model, tok, info). ``info`` records arm metadata (incl. resolved
    LoRA targets) for the report. ModernBERT (A0) is reused verbatim."""
    spec = ARM_SPECS[arm]
    if spec["kind"] == "modernbert":
        from transformers import AutoTokenizer
        base = spec["base_model"]
        # fp32 on CPU/smoke (bf16 CPU matmul is slow); bf16 on CUDA (the
        # StructureModel default) for the real run.
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = StructureModel(base, dtype=dtype)
        tok = AutoTokenizer.from_pretrained(base)
        info = {
            "kind": "modernbert", "base_model": base,
            "lora_targets": ["Wqkv", "Wo", "query_proj", "key_proj", "value_proj"],
            "layout_side_channel": True, "ablate_bbox": False,
        }
        return model.to(device), tok, info
    model, tok, lora_targets = _build_layout_backbone(arm, smoke)
    info = {
        "kind": "layout",
        "hf_model": "(tiny-from-config)" if smoke else spec["hf_model"],
        "model_cls": spec["model_cls"],
        "lora_targets": lora_targets,
        "layout_side_channel": False,
        "ablate_bbox": bool(spec.get("ablate_bbox")),
    }
    return model.to(device), tok, info


# ---------------------------------------------------------------------------
# Collation (tokenize on the fly so per-token bbox stays aligned)
# ---------------------------------------------------------------------------


def _collate_modernbert(rows, tok, max_length):
    enc = tok([(r.get("text") or "").strip() for r in rows], padding=True,
              truncation=True, max_length=max_length, return_tensors="pt")
    batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    batch["layout"] = torch.tensor([r["layout"] for r in rows], dtype=torch.float32)
    _attach_labels(batch, rows)
    return batch


def _collate_layout(rows, tok, max_length, ablate):
    enc = tok([(r.get("text") or "").strip() for r in rows], padding=True,
              truncation=True, max_length=max_length, return_tensors="pt")
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    bsz, seq = ids.shape
    bbox = torch.zeros((bsz, seq, 4), dtype=torch.long)
    for i, r in enumerate(rows):
        box = [0, 0, 1000, 1000] if ablate else r.get("_bbox", [0, 0, 0, 0])
        row_box = torch.tensor(box, dtype=torch.long)
        # Real (non-pad) tokens get the block bbox; pad tokens stay [0,0,0,0].
        real = mask[i].bool()
        bbox[i][real] = row_box
    batch = {"input_ids": ids, "attention_mask": mask, "bbox": bbox}
    _attach_labels(batch, rows)
    return batch


def _attach_labels(batch, rows):
    li = [_label_ints(r) for r in rows]
    for key in ("y_role", "y_is_heading", "y_table_region", "y_is_image_block",
                "y_list_nesting", "y_pedagogical_role"):
        batch[key] = torch.tensor([d[key] for d in li], dtype=torch.long)


def _make_loader(rows, tok, *, kind, ablate, batch_size, max_length,
                 sample_weights=None, shuffle=False):
    if kind == "modernbert":
        coll = lambda b: _collate_modernbert(b, tok, max_length)  # noqa: E731
    else:
        coll = lambda b: _collate_layout(b, tok, max_length, ablate)  # noqa: E731
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(rows, batch_size=batch_size, sampler=sampler,
                          collate_fn=coll)
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=coll)


def _forward(model, batch, kind, device):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    if kind == "modernbert":
        return model(ids, mask, batch["layout"].to(device))
    return model(ids, mask, batch["bbox"].to(device))


# ---------------------------------------------------------------------------
# Sampler weights (mirrors train_structure.main's multi-head WeightedRandomSampler)
# ---------------------------------------------------------------------------


def _build_sample_weights(rows, cw_role, cw_is_heading, cw_ped, *,
                          is_heading_cap=8.0, ped_cap=8.0):
    yr = torch.tensor([_label_ints(r)["y_role"] for r in rows], dtype=torch.long)
    yh = torch.tensor([_label_ints(r)["y_is_heading"] for r in rows], dtype=torch.long)
    yp = torch.tensor([_label_ints(r)["y_pedagogical_role"] for r in rows], dtype=torch.long)
    cw_h = cw_is_heading.detach().cpu().clone()
    cw_h[1] = min(float(cw_h[1]), float(is_heading_cap))
    cw_p = cw_ped.detach().cpu().clone()
    cw_p[0] = 1.0
    for c in range(1, NUM_PEDAGOGICAL_ROLES):
        cw_p[c] = min(float(cw_p[c]), float(ped_cap))
    role_w = cw_role.detach().cpu()[yr.clamp(min=0)]
    head_w = cw_h[yh.clamp(min=0)]
    ped_w = cw_p[yp.clamp(min=0)]
    role_w = torch.where(yr < 0, torch.ones_like(role_w), role_w)
    head_w = torch.where(yh < 0, torch.ones_like(head_w), head_w)
    ped_w = torch.where(yp < 0, torch.ones_like(ped_w), ped_w)
    return torch.stack([role_w, head_w, ped_w], dim=0).max(dim=0).values.to(torch.float32)


# ---------------------------------------------------------------------------
# Train / eval one arm
# ---------------------------------------------------------------------------


def _class_weights(rows, device):
    li = [_label_ints(r) for r in rows]

    def col(k):
        return [d[k] for d in li]

    return {
        "role": compute_class_weights(col("y_role"), NUM_ROLES, ignore=-100).to(device),
        "is_heading": compute_class_weights(col("y_is_heading"), 2, ignore=-100).to(device),
        "table_region": compute_class_weights(col("y_table_region"), 2).to(device),
        "is_image_block": compute_class_weights(col("y_is_image_block"), 2).to(device),
        "list_nesting": compute_class_weights(col("y_list_nesting"), NUM_LIST_NESTING).to(device),
        "pedagogical_role": compute_class_weights(col("y_pedagogical_role"), NUM_PEDAGOGICAL_ROLES).to(device),
    }


@torch.no_grad()
def _evaluate(model, rows, tok, *, kind, ablate, device, batch_size, max_length):
    model.eval()
    cols = {k: ([], []) for k in (
        "role", "is_heading", "table_region", "is_image_block",
        "list_nesting", "pedagogical_role")}
    loader = _make_loader(rows, tok, kind=kind, ablate=ablate,
                          batch_size=batch_size, max_length=max_length)
    for batch in loader:
        out = _forward(model, batch, kind, device)
        preds = {
            "role": out["role"].argmax(-1).tolist(),
            "is_heading": out["is_heading"].argmax(-1).tolist(),
            "table_region": out["table_region"].argmax(-1).tolist(),
            "is_image_block": out["is_image_block"].argmax(-1).tolist(),
            "list_nesting": out["list_nesting"].argmax(-1).tolist(),
            "pedagogical_role": out["pedagogical_role"].argmax(-1).tolist(),
        }
        ys = {
            "role": batch["y_role"].tolist(),
            "is_heading": batch["y_is_heading"].tolist(),
            "table_region": batch["y_table_region"].tolist(),
            "is_image_block": batch["y_is_image_block"].tolist(),
            "list_nesting": batch["y_list_nesting"].tolist(),
            "pedagogical_role": batch["y_pedagogical_role"].tolist(),
        }
        for k in cols:
            t, p = cols[k]
            for i in range(len(ys[k])):
                # role/is_heading carry -100 on synth table-cell rows: exclude.
                if k in ("role", "is_heading") and ys[k][i] == -100:
                    continue
                t.append(ys[k][i]); p.append(preds[k][i])
    m = metrics_from_collected(
        *cols["role"], *cols["is_heading"], *cols["table_region"],
        *cols["is_image_block"], *cols["list_nesting"], *cols["pedagogical_role"],
    )
    m.pop("raw", None)
    return m


def _snapshot_score(m):
    ln_score = max(0.0, 1.0 - m["list_nesting_mae"] / 3.0)
    return (
        0.40 * m["role_macro_f1"]
        + 0.18 * m["is_heading_pos_f1"]
        + 0.15 * m["table_region_pos_f1"]
        + 0.10 * m["is_image_block_pos_f1"]
        + 0.12 * m["pedagogical_role_macro_f1"]
        + 0.05 * ln_score
    )


def _trainable_state(model):
    return {n: p.detach().cpu().clone()
            for n, p in model.state_dict().items()
            if any(s in n for s in (
                "lora_", "head_role", "head_is_heading", "head_table_region",
                "head_is_image_block", "head_list_nesting",
                "head_pedagogical_role", "layout_norm", "layout_mlp"))}


def _train_arm(model, train_rows, val_rows, tok, *, kind, ablate, device,
               args):
    cws = _class_weights(train_rows, device)
    sample_w = _build_sample_weights(
        train_rows, cws["role"], cws["is_heading"], cws["pedagogical_role"])
    train_loader = _make_loader(
        train_rows, tok, kind=kind, ablate=ablate, batch_size=args.batch_size,
        max_length=args.max_length, sample_weights=sample_w)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_score = -1.0
    best_state = None
    best_epoch = 0
    epochs_run = 0
    head_keys = ("role", "is_heading", "table_region", "is_image_block",
                 "list_nesting", "pedagogical_role")
    y_keys = ("y_role", "y_is_heading", "y_table_region", "y_is_image_block",
              "y_list_nesting", "y_pedagogical_role")
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        for batch in train_loader:
            out = _forward(model, batch, kind, device)
            loss = 0.0
            for hk, yk in zip(head_keys, y_keys):
                loss = loss + F.cross_entropy(
                    out[hk], batch[yk].to(device), weight=cws[hk])
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
        epochs_run = epoch
        val_m = _evaluate(model, val_rows, tok, kind=kind, ablate=ablate,
                          device=device, batch_size=args.batch_size * 2,
                          max_length=args.max_length) if val_rows else None
        score = _snapshot_score(val_m) if val_m else 0.0
        print(f"    [epoch {epoch:02d}] "
              f"val_role={val_m['role_macro_f1']:.4f} score={score:.4f} "
              f"({time.time() - ep_t0:.1f}s)" if val_m else
              f"    [epoch {epoch:02d}] (no val) ({time.time() - ep_t0:.1f}s)",
              flush=True)
        if score > best_score + 1e-4 or best_state is None:
            best_score = score
            best_epoch = epoch
            best_state = _trainable_state(model)
    if best_state is not None:
        full = model.state_dict()
        for k, v in best_state.items():
            full[k] = v.to(full[k].device)
        model.load_state_dict(full)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return epochs_run, best_epoch, peak


# ---------------------------------------------------------------------------
# Real-PDF eval (generalized from eval/measure_realpdf_structure)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _realpdf_eval(model, tok, realpdf_rows, *, kind, ablate, device,
                  batch_size, max_length):
    """Role-only union + gold-restricted macro-F1 on the real-PDF v3 gold,
    mirroring measure_realpdf_structure (which only scores ``structural_role``)
    but generalized to any arm's forward."""
    from sklearn.metrics import f1_score
    if not realpdf_rows:
        return {"union_macro_f1": None, "gold_restricted_macro_f1": None,
                "n_rows": 0}
    model.eval()
    y_true, y_pred = [], []
    loader = _make_loader(realpdf_rows, tok, kind=kind, ablate=ablate,
                          batch_size=batch_size, max_length=max_length)
    for batch in loader:
        out = _forward(model, batch, kind, device)
        preds = out["role"].argmax(-1).tolist()
        y_pred.extend(preds)
        y_true.extend(batch["y_role"].tolist())
    union = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    gold = float(f1_score(y_true, y_pred, average="macro",
                          labels=sorted(set(y_true)), zero_division=0))
    return {"union_macro_f1": union, "gold_restricted_macro_f1": gold,
            "n_rows": len(realpdf_rows)}


def _load_realpdf_rows(partition: str) -> list[dict]:
    roots = [REALPDF_SHIPPABLE_ROOT]
    if partition == "internal":
        roots.append(REALPDF_INTERNAL_ROOT)
    rows: list[dict] = []
    for root in roots:
        rows.extend(_read_rows(root / "test.jsonl"))
    # Defensive: shippable real-PDF excludes any OpenStax-sourced rows.
    if partition == "shippable":
        rows = [r for r in rows if r.get("source") not in INTERNAL_ONLY_SOURCES]
    _normalize_doc_bboxes(rows)
    return rows


# ---------------------------------------------------------------------------
# Report I/O + cell orchestration
# ---------------------------------------------------------------------------


def _cell_key(cell: dict) -> tuple:
    return (cell["arm"], cell["split"], cell["partition"],
            cell.get("held_out_corpus"))


def _load_report(path: Path, dataset_meta: dict, arms: list[str],
                 partitions: list[str]) -> dict:
    if path.exists():
        try:
            rep = json.loads(path.read_text())
            rep.setdefault("per_cell", [])
            return rep
        except Exception:
            pass
    return {
        "schema": SCHEMA,
        "dataset": dataset_meta,
        "arms_run": arms,
        "partitions": partitions,
        "per_cell": [],
        "deltas_vs_A0": {},
    }


def _recompute_deltas(report: dict) -> None:
    """deltas_vs_A0[partition][split][held_out_corpus or '*'][arm] = {metric: Δ}."""
    a0 = {}
    for c in report["per_cell"]:
        if c["arm"] == "A0":
            a0[(c["split"], c["partition"], c.get("held_out_corpus"))] = c
    deltas: dict = {}
    for c in report["per_cell"]:
        if c["arm"] == "A0":
            continue
        base = a0.get((c["split"], c["partition"], c.get("held_out_corpus")))
        if not base:
            continue
        d = {}
        for m in ("role_macro_f1", "is_heading_pos_f1", "table_region_pos_f1",
                  "is_image_block_pos_f1", "pedagogical_role_macro_f1"):
            bv = base["per_head_metrics"].get(m)
            cv = c["per_head_metrics"].get(m)
            if bv is not None and cv is not None:
                d[m] = round(cv - bv, 6)
        bru = (base.get("realpdf") or {}).get("union_macro_f1")
        cru = (c.get("realpdf") or {}).get("union_macro_f1")
        if bru is not None and cru is not None:
            d["realpdf_union_macro_f1"] = round(cru - bru, 6)
        corp = c.get("held_out_corpus") or "*"
        deltas.setdefault(c["partition"], {}).setdefault(c["split"], {}) \
            .setdefault(corp, {})[c["arm"]] = d
    report["deltas_vs_A0"] = deltas


def _write_report(report: dict, path: Path) -> None:
    _recompute_deltas(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def _run_cell(*, arm, split, partition, held_out_corpus, train_rows, val_rows,
              test_rows, realpdf_rows, smoke, device, args) -> dict:
    spec = ARM_SPECS[arm]
    kind = spec["kind"]
    ablate = bool(spec.get("ablate_bbox"))
    t0 = time.time()
    model, tok, info = _build_arm(arm, smoke, device)
    try:
        epochs_run, best_epoch, peak = _train_arm(
            model, train_rows, val_rows, tok, kind=kind, ablate=ablate,
            device=device, args=args)
        head_m = _evaluate(model, test_rows, tok, kind=kind, ablate=ablate,
                           device=device, batch_size=args.batch_size * 2,
                           max_length=args.max_length)
        realpdf = _realpdf_eval(model, tok, realpdf_rows, kind=kind,
                                ablate=ablate, device=device,
                                batch_size=args.batch_size * 2,
                                max_length=args.max_length)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cell = {
        "arm": arm,
        "split": split,
        "partition": partition,
        "held_out_corpus": held_out_corpus,
        "arm_info": info,
        "per_head_metrics": head_m,
        "realpdf": realpdf,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "epochs": epochs_run,
        "best_epoch": best_epoch,
        "peak_vram_bytes": peak,
        "wall_time_s": round(time.time() - t0, 1),
    }
    return cell


def _loco_corpora(rows, partition, min_rows) -> list[str]:
    counts = Counter(r.get("source", "unknown") for r in rows)
    out = []
    for src, n in counts.items():
        if n < min_rows:
            continue
        if partition == "shippable" and src in INTERNAL_ONLY_SOURCES:
            continue
        out.append(src)
    return sorted(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/structure_dataset_v3"))
    ap.add_argument("--arms", default="A0,A1,A2,A3",
                    help="Comma list subset of A0,A1,A2,A3 (default all).")
    ap.add_argument("--splits", default="standard,loco",
                    help="Comma list subset of standard,loco (default both).")
    ap.add_argument("--license-partition", choices=("shippable", "internal", "both"),
                    default="both")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="Cap training rows (fast / LOCO regime). Default None.")
    ap.add_argument("--max-eval-rows", type=int, default=None,
                    help="Cap val/test rows per cell (fast / smoke regime). "
                         "Default None = full split.")
    ap.add_argument("--loco-min-rows", type=int, default=2000,
                    help="Min rows for a corpus to be a LOCO held-out target.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path,
                    default=Path("data/eval_reports/structure_backbone_benchmark.json"))
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny offline CI run: ~200 rows, 1 epoch, CPU, A0+A2, "
                         "internal/standard only.")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    partitions = (["shippable", "internal"] if args.license_partition == "both"
                  else [args.license_partition])

    if args.smoke:
        arms = ["A0", "A2"]
        splits = ["standard"]
        partitions = ["internal"]
        args.epochs = 1
        args.device = "cpu"
        if args.max_train_rows is None:
            args.max_train_rows = 200
        if args.max_eval_rows is None:
            args.max_eval_rows = 200
        args.batch_size = min(args.batch_size, 8)
        # The smoke layout arms use a tiny from-config backbone with
        # max_position_embeddings=130; keep seq below that.
        args.max_length = min(args.max_length, 128)

    for a in arms:
        if a not in ARM_SPECS:
            raise SystemExit(f"unknown arm {a!r} (valid: {sorted(ARM_SPECS)})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[bench] device={device} arms={arms} splits={splits} "
          f"partitions={partitions} smoke={args.smoke}", flush=True)

    # Load + bbox-normalize the standard split once (LOCO subsets reuse it; the
    # per-doc normalization is independent of split membership).
    print(f"[data] loading {args.dataset_dir}", flush=True)
    train_all = _read_rows(args.dataset_dir / "train.jsonl")
    val_all = _read_rows(args.dataset_dir / "val.jsonl")
    test_all = _read_rows(args.dataset_dir / "test.jsonl")
    all_rows = train_all + val_all + test_all
    _normalize_doc_bboxes(all_rows)
    print(f"  train={len(train_all)} val={len(val_all)} test={len(test_all)}",
          flush=True)

    realpdf_by_partition = {p: _load_realpdf_rows(p) for p in partitions}
    for p, rr in realpdf_by_partition.items():
        print(f"  realpdf[{p}]={len(rr)} rows", flush=True)

    dataset_meta = {
        "dataset_dir": str(args.dataset_dir),
        "train": len(train_all), "val": len(val_all), "test": len(test_all),
        "schema_version": (train_all[0].get("schema_version") if train_all else None),
        "epochs": args.epochs, "max_length": args.max_length,
        "batch_size": args.batch_size, "max_train_rows": args.max_train_rows,
        "seed": args.seed, "smoke": args.smoke,
        "role_classes": NUM_ROLES, "pedagogical_classes": NUM_PEDAGOGICAL_ROLES,
    }
    report = _load_report(args.output, dataset_meta, arms, partitions)
    done = {_cell_key(c) for c in report["per_cell"]}

    def cap(rows):
        if args.max_train_rows and len(rows) > args.max_train_rows:
            return rows[:args.max_train_rows]
        return rows

    def cap_eval(rows):
        if args.max_eval_rows and len(rows) > args.max_eval_rows:
            return rows[:args.max_eval_rows]
        return rows

    for p in list(realpdf_by_partition):
        realpdf_by_partition[p] = cap_eval(realpdf_by_partition[p])

    # Build the plan of cells (partition × split × arm [× corpus]). A0 first
    # within each (partition,split[,corpus]) group so deltas resolve on write.
    for partition in partitions:
        tr_p = _partition_filter(train_all, partition)
        va_p = _partition_filter(val_all, partition)
        te_p = _partition_filter(test_all, partition)
        realpdf_rows = realpdf_by_partition[partition]
        for split in splits:
            if split == "standard":
                groups = [(None, cap(tr_p), cap_eval(va_p), cap_eval(te_p))]
            else:  # loco
                corpora = _loco_corpora(tr_p, partition, args.loco_min_rows)
                print(f"[loco] partition={partition} held-out corpora={corpora}",
                      flush=True)
                groups = []
                for c in corpora:
                    tr_c = cap([r for r in tr_p if r.get("source") != c])
                    va_c = cap_eval([r for r in va_p if r.get("source") != c])
                    # Test = all rows of C across train+val+test (held out by
                    # construction — C is excluded from training).
                    te_c = cap_eval([r for r in all_rows if r.get("source") == c
                            and r.get("source") not in (INTERNAL_ONLY_SOURCES
                                                        if partition == "shippable" else set())])
                    groups.append((c, tr_c, va_c, te_c))
            for corpus, tr_rows, va_rows, te_rows in groups:
                for arm in sorted(arms, key=lambda a: (a != "A0", a)):
                    key = (arm, split, partition, corpus)
                    if key in done:
                        print(f"[skip] {key} already in report", flush=True)
                        continue
                    if not tr_rows or not te_rows:
                        print(f"[skip] {key} empty train/test", flush=True)
                        continue
                    label = f"{arm} {split} {partition}" + (f" heldout={corpus}" if corpus else "")
                    print(f"[cell] {label}  train={len(tr_rows)} test={len(te_rows)}",
                          flush=True)
                    try:
                        cell = _run_cell(
                            arm=arm, split=split, partition=partition,
                            held_out_corpus=corpus, train_rows=tr_rows,
                            val_rows=va_rows, test_rows=te_rows,
                            realpdf_rows=realpdf_rows, smoke=args.smoke,
                            device=device, args=args)
                    except Exception as exc:  # noqa: BLE001
                        # One arm's failure (e.g. an 8GB OOM on an fp32 layout
                        # backbone) must NOT abort the remaining arms. Record
                        # nothing in `done` so a re-run (e.g. smaller --batch-size)
                        # retries this cell; free the card and move on.
                        import traceback
                        print(f"[cell-error] {label}: {type(exc).__name__}: {exc}",
                              flush=True)
                        traceback.print_exc()
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    report["per_cell"].append(cell)
                    done.add(key)
                    _write_report(report, args.output)
                    rp = cell["realpdf"]
                    print(f"  -> role_macro={cell['per_head_metrics']['role_macro_f1']:.4f} "
                          f"img={cell['per_head_metrics']['is_image_block_pos_f1']:.4f} "
                          f"tbl={cell['per_head_metrics']['table_region_pos_f1']:.4f} "
                          f"realpdf_union={rp.get('union_macro_f1')} "
                          f"peakVRAM={cell['peak_vram_bytes']/1e9:.2f}GB "
                          f"({cell['wall_time_s']}s)", flush=True)

    _write_report(report, args.output)
    print(f"[bench] wrote {args.output}  ({len(report['per_cell'])} cells)",
          flush=True)


if __name__ == "__main__":
    main()
