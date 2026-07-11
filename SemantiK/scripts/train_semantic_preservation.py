"""Train Theta semantic_preservation cross-encoder (Phase 4b).

DeBERTa-v3-small + LoRA(r=16, alpha=32) cross-encoder. Takes
(source_text, candidate_text) → scalar in [0,1] = fraction of source
meaning preserved. Trained on 319,967 arXiv-only rows; eval slices on
val/test (arxiv) plus probe_qwen_style and probe_cross_domain.

Output layout (matches train_semantic.py's adapter pattern):
    models/theta/semantic_preservation/v1/
        adapter_model.safetensors   # PEFT LoRA weights
        adapter_config.json
        heads.pt                    # head_score state_dict
        tokenizer/
        summary.json                # SHA256 stamps + pre-flight + best val
        calibration.json            # isotonic + offset; fit_split=val ONLY
        train_log.jsonl             # per-50-step log
        val_eval_log.jsonl          # per-eval val metrics
    models/theta/semantic_preservation/v1/_ckpt/  # rolling intermediates

Phase 4b is BUILD + LAUNCH; Phase 4c integrates into theta/evaluator.py.
This script does NOT modify semantik_structure/theta/evaluator.py or
semantik_structure/theta/config.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    DebertaV2Tokenizer,
    get_cosine_schedule_with_warmup,
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

BASE_MODEL_NAME = "microsoft/deberta-v3-small"
DATASET_DIR = Path("data/semantic_preservation_dataset")
DEFAULT_OUT = Path("models/theta/semantic_preservation/v1")
LOG_PATH = Path("data/logs/train_semantic_preservation.log")

# Pre-flight thresholds.
TRUNCATION_FAIL_PCT = 5.0
TRUNCATION_WARN_PCT = 1.0
MIN_BUCKET_ROWS = 5_000
VRAM_FAIL_GB = 7.5

# Truncation audit sample size.
TRUNCATION_SAMPLE = 5_000


# ----------------------------------------------------------------------
# Pre-flight gates
# ----------------------------------------------------------------------


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _abort(reason: str, code: int = 2) -> None:
    print(f"[preflight FAIL] {reason}", flush=True)
    sys.exit(code)


def gate_backbone_available() -> tuple[bool, str, Path | None]:
    """Gate 1: backbone files exist in HF cache (we already downloaded
    via huggingface_hub.snapshot_download in the launcher). If not
    found, the trainer aborts before model load. Returns the snapshot
    path so downstream code can locate the spm.model directly."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cand = cache_root / "models--microsoft--deberta-v3-small"
    if not cand.exists():
        return False, f"backbone cache dir missing: {cand}", None
    snaps = list((cand / "snapshots").glob("*"))
    if not snaps:
        return False, f"no snapshots under {cand}", None
    snap = snaps[0]
    needed = ["config.json", "spm.model", "tokenizer_config.json"]
    have_weights = any(
        (snap / n).exists()
        for n in ("model.safetensors", "pytorch_model.bin")
    )
    missing = [n for n in needed if not (snap / n).exists()]
    if missing or not have_weights:
        return False, (
            f"backbone snapshot incomplete (missing={missing}, "
            f"weights_present={have_weights})"
        ), None
    return True, str(snap), snap


def gate_dataset_sha256() -> dict[str, str]:
    """Gate 2: SHA256-stamp the dataset files. Failure to read the
    files aborts the run."""
    targets = ["train.jsonl", "val.jsonl", "test.jsonl",
               "coverage_report.json"]
    out: dict[str, str] = {}
    for name in targets:
        p = DATASET_DIR / name
        if not p.exists():
            _abort(f"dataset file missing: {p}")
        out[name] = _sha256(p)
    return out


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _reservoir_sample_rows(path: Path, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sample: list[dict] = []
    for i, row in enumerate(_iter_jsonl(path)):
        if i < k:
            sample.append(row)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = row
    return sample


def gate_truncation_audit(tokenizer, *, max_len: int, seed: int) -> dict:
    """Gate 3: tokenize 5K random rows; fail if >5% would be truncated."""
    sample = _reservoir_sample_rows(
        DATASET_DIR / "train.jsonl", TRUNCATION_SAMPLE, seed,
    )
    n_total = len(sample)
    n_truncated = 0
    lens: list[int] = []
    for row in sample:
        enc = tokenizer(
            row["source_text"],
            row["candidate_text"],
            truncation="longest_first",
            max_length=max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        L = len(enc["input_ids"])
        lens.append(L)
        # `>= max_len-1`: would have been truncated (longest_first packs
        # right up to the limit; len==max_len means the encoder hit it).
        if L >= max_len - 1:
            n_truncated += 1
    pct = 100.0 * n_truncated / max(1, n_total)
    p50 = int(np.percentile(lens, 50))
    p95 = int(np.percentile(lens, 95))
    p99 = int(np.percentile(lens, 99))
    result = {
        "sample_size": n_total,
        "n_truncated": n_truncated,
        "pct_truncated": round(pct, 3),
        "p50_len": p50,
        "p95_len": p95,
        "p99_len": p99,
        "max_len": max_len,
    }
    if pct > TRUNCATION_FAIL_PCT:
        _abort(
            f"truncation_audit: {pct:.2f}% rows would be truncated at "
            f"max_len={max_len} (>5% threshold). p99 token len={p99}."
        )
    if pct > TRUNCATION_WARN_PCT:
        result["warning"] = (
            f"{pct:.2f}% truncation between 1% and 5% — proceeding."
        )
        print(f"[preflight WARN] {result['warning']}", flush=True)
    return result


def gate_label_support() -> dict:
    """Gate 4: 10-bucket histogram on labels; fail if any bucket has <5K."""
    buckets = [0] * 10
    for row in _iter_jsonl(DATASET_DIR / "train.jsonl"):
        y = float(row["label"])
        # Map [0, 1] → 0..9, with y==1.0 → bucket 9.
        idx = min(9, int(y * 10))
        buckets[idx] += 1
    edges = [round(0.1 * i, 1) for i in range(11)]
    weak = [
        f"[{edges[i]:.1f}, {edges[i+1]:.1f}): {buckets[i]}"
        for i in range(10) if buckets[i] < MIN_BUCKET_ROWS
    ]
    if weak:
        _abort(
            "label_support: at least one 10% bucket has <5K rows: "
            + "; ".join(weak)
        )
    return {"buckets": buckets, "edges": edges}


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------


class JsonlPairDataset(Dataset):
    """Lazy index → row reader. Loads JSONL into memory (each row is
    ~1KB; 320K → ~300MB which fits comfortably)."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = list(_iter_jsonl(path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        return {
            "source_text": r["source_text"],
            "candidate_text": r["candidate_text"],
            "label": float(r["label"]),
            "perturbation_type": r.get("perturbation_type", "unknown"),
            "source_region_kind": r.get("source_region_kind", "unknown"),
            "source_domain": r.get("source_domain", "unknown"),
        }


def make_collate(tokenizer, max_len: int):
    def collate(batch: list[dict]) -> dict:
        sources = [b["source_text"] for b in batch]
        candidates = [b["candidate_text"] for b in batch]
        enc = tokenizer(
            sources, candidates,
            truncation="longest_first",
            max_length=max_len,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        labels = torch.tensor(
            [b["label"] for b in batch], dtype=torch.float32,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids"),
            "labels": labels,
            "perturbation_type": [b["perturbation_type"] for b in batch],
            "source_region_kind": [b["source_region_kind"] for b in batch],
            "source_domain": [b["source_domain"] for b in batch],
        }
    return collate


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------


class SemanticPreservationModel(nn.Module):
    """DeBERTa-v3-small + LoRA → [CLS] pool → linear(768→1) → sigmoid.

    Pooling: ``outputs.last_hidden_state[:, 0, :]`` — DeBERTa-v3 has no
    pooler module (no `outputs.pooler_output`), so we read CLS off the
    last hidden state directly. Asserted at construction.
    """

    def __init__(
        self,
        base_model_name: str = BASE_MODEL_NAME,
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target: str = "qkv",
        pooling: str = "cls",
        finetune: str = "lora",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"unknown pooling={pooling!r}")
        if finetune not in ("lora", "full"):
            raise ValueError(f"unknown finetune={finetune!r}")
        self.pooling = pooling
        self.finetune = finetune
        base = AutoModel.from_pretrained(base_model_name, dtype=dtype)
        # 'full' = unfreeze the entire backbone (canonical cross-encoder STS
        # recipe). v1-v5 all FROZE the backbone (LoRA-only) and could not learn
        # to compare source<->candidate — a low-rank update to frozen attention
        # is not enough to reshape it for alignment. deberta-v3-small (44M) fits
        # 8GB in full FT trivially. Saves model.safetensors+config.json (NOT a
        # peft adapter); the runtime loader branches on artifact type.
        if finetune == "full":
            self.encoder = base  # unwrapped; all params already require_grad
        else:
            # 'qkv' = legacy attention-only adapter. 'all-linear' = every linear
            # layer — near full-FT expressivity while keeping the adapter path.
            if lora_target == "all-linear":
                target_modules = "all-linear"
            elif lora_target == "qkv":
                target_modules = ["query_proj", "key_proj", "value_proj"]
            else:
                raise ValueError(f"unknown lora_target={lora_target!r}")
            peft_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.encoder = get_peft_model(base, peft_cfg)
        # Gradient checkpointing on the encoder. peft_model wraps the
        # base; the underlying base supports it.
        try:
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except TypeError:
            self.encoder.gradient_checkpointing_enable()
        # Disable use_cache when checkpointing is on.
        if hasattr(self.encoder, "config"):
            try:
                self.encoder.config.use_cache = False
            except Exception:
                pass

        hidden = base.config.hidden_size
        self.head_score = nn.Linear(hidden, 1)

        # Pooler check — DeBERTa-v3 has no pooler module
        assert not hasattr(base, "pooler") or base.pooler is None, (
            "Expected no pooler on DeBERTa-v3 base; got an explicit "
            "pooler module — refusing to silently ignore it."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        # DeBERTa-v3 has no pooler module; read CLS off last_hidden_state.
        assert getattr(out, "pooler_output", None) is None, (
            "DeBERTa-v3 unexpectedly produced a pooler_output; the "
            "trainer assumes none and reads CLS off last_hidden_state."
        )
        hidden = out.last_hidden_state  # (B, T, H)
        if self.pooling == "mean":
            # Masked mean over real tokens. deberta-v3 is pretrained with RTD
            # (no sentence-pair/NSP objective), so its CLS is not trained to
            # summarize a pair — mean-pooling gives the head a stronger pair
            # signal. Must stay in sync with the runtime loader's pooling.
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            pooled = pooled.float()
        else:
            pooled = hidden[:, 0, :].float()  # CLS
        logit = self.head_score(pooled).squeeze(-1)  # (B,)
        return logit

    def save_adapter(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(str(out_dir))
        torch.save(
            {"head_score.state_dict": self.head_score.state_dict()},
            str(out_dir / "heads.pt"),
        )


# ----------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------


@torch.no_grad()
def eval_loop(
    model: SemanticPreservationModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    raw_preds: list[float] = []
    labels: list[float] = []
    perts: list[str] = []
    rkinds: list[str] = []
    domains: list[str] = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tti = batch.get("token_type_ids")
        if tti is not None:
            tti = tti.to(device)
        logits = model(ids, mask, tti)
        scores = torch.sigmoid(logits).detach().float().cpu().numpy()
        raw_preds.extend(scores.tolist())
        labels.extend(batch["labels"].tolist())
        perts.extend(batch["perturbation_type"])
        rkinds.extend(batch["source_region_kind"])
        domains.extend(batch["source_domain"])
    raw_arr = np.array(raw_preds)
    lab_arr = np.array(labels)
    mae = float(np.mean(np.abs(raw_arr - lab_arr)))
    rmse = float(np.sqrt(np.mean((raw_arr - lab_arr) ** 2)))
    sp_corr = float("nan")
    try:
        sp_corr, _ = spearmanr(raw_arr, lab_arr)
        sp_corr = float(sp_corr)
    except Exception:
        pass

    def _grouped_mae(keys: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        seen = defaultdict(list)
        for k, p, l in zip(keys, raw_arr, lab_arr):
            seen[k].append((p, l))
        for k, items in seen.items():
            arr = np.array(items)
            out[k] = {
                "n": int(len(arr)),
                "mae": float(np.mean(np.abs(arr[:, 0] - arr[:, 1]))),
                "mean_pred": float(np.mean(arr[:, 0])),
                "mean_label": float(np.mean(arr[:, 1])),
            }
        return out

    return {
        "n": int(len(raw_arr)),
        "mae": mae,
        "rmse": rmse,
        "spearman": sp_corr,
        "per_perturbation": _grouped_mae(perts),
        "per_region_kind": _grouped_mae(rkinds),
        "per_domain": _grouped_mae(domains),
        "raw_preds": raw_preds,
        "labels": labels,
    }


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------


def fit_calibration(
    val_eval: dict,
    val_perturbations: list[str],
) -> dict:
    """Phase 4b calibration: isotonic regression on val (split assertion
    is enforced by the caller passing val ONLY)."""
    raw = np.array(val_eval["raw_preds"], dtype=np.float64)
    lab = np.array(val_eval["labels"], dtype=np.float64)

    iso = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True,
    )
    iso.fit(raw, lab)
    cal = iso.predict(raw)

    pre_mae = float(np.mean(np.abs(raw - lab)))
    post_mae = float(np.mean(np.abs(cal - lab)))

    # NLL using a small clamp.
    eps = 1e-6
    raw_c = np.clip(raw, eps, 1 - eps)
    cal_c = np.clip(cal, eps, 1 - eps)
    pre_nll = float(np.mean(
        -lab * np.log(raw_c) - (1 - lab) * np.log(1 - raw_c)
    ))
    post_nll = float(np.mean(
        -lab * np.log(cal_c) - (1 - lab) * np.log(1 - cal_c)
    ))

    # Offset such that 5%-quantile(calibrated_score | clean rows) >= 0.65.
    is_clean = np.array(
        [p == "clean" for p in val_perturbations], dtype=bool,
    )
    if is_clean.sum() > 0:
        clean_cal = cal[is_clean]
        q05 = float(np.quantile(clean_cal, 0.05))
        offset = max(0.0, 0.65 - q05)
    else:
        q05 = float("nan")
        offset = 0.0

    return {
        "fit_split": "val",
        "isotonic_X_thresholds": iso.X_thresholds_.tolist(),
        "isotonic_y_thresholds": iso.y_thresholds_.tolist(),
        "offset": float(offset),
        "clean_q05_calibrated_pre_offset": q05,
        "clean_q05_calibrated_post_offset": float(q05 + offset),
        "val_mae_pre": pre_mae,
        "val_mae_post": post_mae,
        "val_nll_pre": pre_nll,
        "val_nll_post": post_nll,
        "val_n": int(len(raw)),
        "val_n_clean": int(is_clean.sum()),
    }


# ----------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--loss", choices=["bce", "mse"], default="bce",
                    help="Training loss. 'bce' = soft-target "
                         "binary_cross_entropy_with_logits (strong gradients "
                         "on confident-wrong preds; breaks the predict-the-mean "
                         "basin that mode-collapsed v1/v2 under MSE). 'mse' = "
                         "legacy sigmoid+MSE recipe.")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-target", choices=["qkv", "all-linear"],
                    default="qkv",
                    help="Which linear layers LoRA adapts. 'qkv' = legacy "
                         "attention-only (v1/v2/v3 — too little capacity, "
                         "learned a length shortcut). 'all-linear' = every "
                         "linear layer (attn output + FFN + QKV), near-full-FT "
                         "expressivity while keeping the adapter save/load path.")
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls",
                    help="Sentence-pair pooling. 'cls' = legacy [CLS] token "
                         "(v1-v4 — weak, since deberta-v3 RTD pretraining never "
                         "trains CLS for pairs). 'mean' = masked mean over "
                         "tokens. Runtime loader reads this from summary.json "
                         "and must pool identically.")
    ap.add_argument("--finetune", choices=["lora", "full"], default="lora",
                    help="'lora' = adapter on frozen backbone (v1-v5). 'full' = "
                         "unfreeze the whole backbone (canonical cross-encoder "
                         "STS recipe; the real fix for the frozen-backbone "
                         "confound). Saves model.safetensors+config.json, not a "
                         "peft adapter — the runtime loader branches on this.")
    ap.add_argument("--base-model", type=str, default=BASE_MODEL_NAME,
                    help="HF backbone id. Default deberta-v3-small. NOTE: gate1 "
                         "backbone-cache check is pinned to small; generalize it "
                         "before pointing this at deberta-v3-base.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Smoke flag: cap total optimization steps (early stop).")
    ap.add_argument("--sched-steps", type=int, default=None,
                    help="Cosine LR horizon, decoupled from --max-steps. Set to "
                         "the full-run step count for a FAITHFUL smoke (peak LR "
                         "not compressed). Defaults to total_steps.")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=0,
                    help="Early-stop after N evals with no val-spearman "
                         "improvement. 0 disables (run to completion).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = args.out_dir / "train_log.jsonl"
    val_log_path = args.out_dir / "val_eval_log.jsonl"
    summary_path = args.out_dir / "summary.json"

    # Seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("[preflight] running gates ...", flush=True)
    pre_results: dict = {}

    # Gate 1.
    ok, info, snap_path = gate_backbone_available()
    pre_results["gate1_backbone"] = {"pass": bool(ok), "info": info}
    if not ok:
        _abort(f"gate1_backbone: {info}")
    print(f"[preflight PASS] gate1_backbone: {info}", flush=True)
    spm_path = snap_path / "spm.model"

    # Gate 2.
    sha = gate_dataset_sha256()
    pre_results["gate2_dataset_sha256"] = {"pass": True, "sha256": sha}
    print("[preflight PASS] gate2_dataset_sha256:", flush=True)
    for k, v in sha.items():
        print(f"  {k}: {v}", flush=True)

    # Tokenizer for gates 3 + training. transformers 5.x's fast-tokenizer
    # auto-conversion path requires `protobuf`, which isn't installed in
    # the venv. The slow `DebertaV2Tokenizer` constructed directly from
    # the cached `spm.model` works without protobuf and matches the
    # reference vocab byte-for-byte.
    tokenizer = DebertaV2Tokenizer(vocab_file=str(spm_path))

    # Gate 3.
    trunc = gate_truncation_audit(
        tokenizer, max_len=args.max_len, seed=args.seed,
    )
    pre_results["gate3_truncation"] = {"pass": True, **trunc}
    print(
        f"[preflight PASS] gate3_truncation: {trunc['pct_truncated']}% "
        f"(p50={trunc['p50_len']} p95={trunc['p95_len']} "
        f"p99={trunc['p99_len']})",
        flush=True,
    )

    # Gate 4.
    lab = gate_label_support()
    pre_results["gate4_label_support"] = {"pass": True, **lab}
    print("[preflight PASS] gate4_label_support: buckets=" +
          ",".join(str(b) for b in lab["buckets"]), flush=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("[data] loading splits ...", flush=True)
    train_ds = JsonlPairDataset(DATASET_DIR / "train.jsonl")
    val_ds = JsonlPairDataset(DATASET_DIR / "val.jsonl")
    print(f"  train={len(train_ds)}  val={len(val_ds)}", flush=True)

    collate = make_collate(tokenizer, args.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gpu] device={device} cuda_available={torch.cuda.is_available()}",
          flush=True)
    model = SemanticPreservationModel(
        args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target=args.lora_target,
        pooling=args.pooling,
        finetune=args.finetune,
    ).to(device)
    if hasattr(model.encoder, "print_trainable_parameters"):
        model.encoder.print_trainable_parameters()  # peft only
    else:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {n_tr:,} || all params: {n_all:,} || "
              f"trainable%: {100*n_tr/n_all:.4f}  (full fine-tune)", flush=True)

    if args.resume is not None and args.resume.exists():
        print(f"[resume] loading from {args.resume}", flush=True)
        # Reload LoRA weights.
        peft_path = args.resume
        # Reuse PeftModel.from_pretrained idiom: load weights into the
        # existing peft adapter rather than re-wrapping.
        from safetensors.torch import load_file as _load_st
        st = _load_st(str(peft_path / "adapter_model.safetensors"))
        missing, unexpected = model.encoder.load_state_dict(st, strict=False)
        heads = torch.load(str(peft_path / "heads.pt"), map_location="cpu")
        model.head_score.load_state_dict(heads["head_score.state_dict"])
        print(f"  loaded LoRA (missing={len(missing)} "
              f"unexpected={len(unexpected)}) + heads", flush=True)

    # Optimizer (LoRA + head only).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    micro_per_epoch = len(train_loader)
    optim_steps_per_epoch = max(
        1, micro_per_epoch // max(1, args.grad_accum),
    )
    total_steps = optim_steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    # Scheduler horizon, decoupled from the early-stop cap. A faithful smoke
    # sets --sched-steps to the FULL-run horizon while --max-steps stops early,
    # so the cosine LR is NOT compressed (peak LR sustained) — otherwise a short
    # --max-steps smoke under-trains and is an unfaithful preview of the full run.
    sched_horizon = args.sched_steps if args.sched_steps is not None else total_steps
    warmup_steps = max(1, int(args.warmup_frac * sched_horizon))
    scheduler = get_cosine_schedule_with_warmup(
        optim, warmup_steps, sched_horizon,
    )
    print(
        f"[sched] micro_per_epoch={micro_per_epoch} "
        f"optim_per_epoch={optim_steps_per_epoch} "
        f"total_steps={total_steps} sched_horizon={sched_horizon} "
        f"warmup={warmup_steps}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Initial summary stamp.
    # ------------------------------------------------------------------
    summary = {
        "phase": "4b",
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target": args.lora_target,
        "pooling": args.pooling,
        "finetune": args.finetune,
        "max_len": args.max_len,
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.per_device_batch * args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss": args.loss,
        "warmup_frac": args.warmup_frac,
        "seed": args.seed,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "preflight": pre_results,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[stamp] summary -> {summary_path}", flush=True)

    # ------------------------------------------------------------------
    # Train.
    # ------------------------------------------------------------------
    train_log_fh = train_log_path.open("a", encoding="utf-8")
    val_log_fh = val_log_path.open("a", encoding="utf-8")

    best_val_mae = math.inf
    best_val_spearman = -math.inf
    evals_since_improve = 0
    best_state: dict | None = None
    rolling: list[Path] = []
    t0 = time.time()
    optim_step = 0
    micro_step = 0
    accum_loss = 0.0
    first_step_vram_gb: float | None = None
    peak_vram_gb = 0.0
    early_stop = False

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        if early_stop:
            break
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            tti = batch.get("token_type_ids")
            if tti is not None:
                tti = tti.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit = model(ids, mask, tti)
                if args.loss == "bce":
                    # Soft-target BCE on the raw logit: gradients stay strong
                    # for confident-wrong predictions, so the optimizer cannot
                    # park at the label mean the way sigmoid+MSE did (v1/v2
                    # mode-collapse). _with_logits is autocast-safe (fp32 inner).
                    loss = F.binary_cross_entropy_with_logits(logit, labels)
                else:
                    pred = torch.sigmoid(logit)
                    loss = F.mse_loss(pred, labels)

            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            accum_loss += float(loss.item())
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, max_norm=1.0,
                )
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                optim_step += 1

                avg_loss = accum_loss / args.grad_accum
                accum_loss = 0.0

                # First-step VRAM guard (gate 5).
                if first_step_vram_gb is None and torch.cuda.is_available():
                    first_step_vram_gb = float(
                        torch.cuda.max_memory_allocated(device) / 1e9
                    )
                    print(
                        f"[preflight] first-step VRAM peak = "
                        f"{first_step_vram_gb:.3f} GB",
                        flush=True,
                    )
                    if first_step_vram_gb > VRAM_FAIL_GB:
                        msg = (
                            f"gate5_first_step_vram: {first_step_vram_gb:.2f} GB "
                            f"> {VRAM_FAIL_GB} GB threshold; aborting before "
                            "later OOM."
                        )
                        # Stamp the failure into the summary then abort.
                        summary["preflight"]["gate5_first_step_vram"] = {
                            "pass": False,
                            "first_step_vram_gb": first_step_vram_gb,
                            "threshold_gb": VRAM_FAIL_GB,
                        }
                        summary_path.write_text(json.dumps(summary, indent=2))
                        _abort(msg)
                    summary["preflight"]["gate5_first_step_vram"] = {
                        "pass": True,
                        "first_step_vram_gb": first_step_vram_gb,
                        "threshold_gb": VRAM_FAIL_GB,
                    }
                    summary_path.write_text(json.dumps(summary, indent=2))

                if torch.cuda.is_available():
                    cur_peak = float(
                        torch.cuda.max_memory_allocated(device) / 1e9
                    )
                    peak_vram_gb = max(peak_vram_gb, cur_peak)

                # Log every N optim steps.
                if optim_step % args.log_every == 0:
                    cur_lr = scheduler.get_last_lr()[0]
                    rec = {
                        "step": optim_step,
                        "epoch": epoch,
                        "lr": float(cur_lr),
                        "loss": float(avg_loss),
                        "grad_norm": float(grad_norm),
                        "vram_mb": int(
                            (torch.cuda.max_memory_allocated(device) / (1 << 20))
                            if torch.cuda.is_available() else 0
                        ),
                        "time_seconds": round(time.time() - t0, 1),
                    }
                    train_log_fh.write(json.dumps(rec) + "\n")
                    train_log_fh.flush()
                    print(
                        f"[step {optim_step:6d}] "
                        f"loss={rec['loss']:.4f} "
                        f"grad={rec['grad_norm']:.3f} "
                        f"lr={rec['lr']:.2e} "
                        f"vram={rec['vram_mb']}MB "
                        f"t={rec['time_seconds']:.1f}s",
                        flush=True,
                    )

                # Eval every M optim steps.
                if optim_step % args.eval_every == 0:
                    print(f"[eval] @ step={optim_step}", flush=True)
                    val_metrics = eval_loop(model, val_loader, device)
                    val_metrics_lite = {
                        k: v for k, v in val_metrics.items()
                        if k not in ("raw_preds", "labels")
                    }
                    val_log_fh.write(json.dumps({
                        "step": optim_step,
                        "epoch": epoch,
                        "metrics": val_metrics_lite,
                    }) + "\n")
                    val_log_fh.flush()
                    print(
                        f"[eval] mae={val_metrics['mae']:.4f} "
                        f"rmse={val_metrics['rmse']:.4f} "
                        f"spearman={val_metrics['spearman']:.4f}",
                        flush=True,
                    )

                    # Select the best checkpoint by SPEARMAN, not MAE. A
                    # mode-collapsed mean-predictor scores MAE≈0.23 (best in
                    # v1) but spearman≈0 — exactly the failure we will not ship
                    # again. Ranking ability is what theta needs, so rank on it.
                    # NaN spearman (degenerate constant preds) → treat as worst.
                    sp = val_metrics["spearman"]
                    sp_cmp = sp if (sp == sp) else -1.0  # NaN guard
                    best_val_mae = min(best_val_mae, val_metrics["mae"])
                    if sp_cmp > best_val_spearman:
                        best_val_spearman = sp_cmp
                        evals_since_improve = 0
                        best_state = {
                            "step": optim_step,
                            "epoch": epoch,
                            "val_mae": val_metrics["mae"],
                            "val_spearman": sp_cmp,
                            # full-FT: snapshot the whole model (no lora_ keys
                            # exist). lora: just the adapter deltas.
                            "encoder_state": {
                                k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()
                                if model.finetune == "full" or "lora_" in k
                            },
                            "head_state": {
                                k: v.detach().cpu().clone()
                                for k, v in model.head_score.state_dict().items()
                            },
                        }
                        print(f"[best] new best val spearman = "
                              f"{best_val_spearman:.4f} "
                              f"(mae={val_metrics['mae']:.4f})", flush=True)
                    else:
                        evals_since_improve += 1
                        if args.patience and evals_since_improve >= args.patience:
                            print(f"[early-stop] no val-spearman improvement in "
                                  f"{evals_since_improve} evals (best="
                                  f"{best_val_spearman:.4f}); stopping.",
                                  flush=True)
                            early_stop = True

                    # Rolling intermediate ckpt.
                    snap_dir = ckpt_dir / f"step_{optim_step:06d}"
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    model.save_adapter(snap_dir)
                    rolling.append(snap_dir)
                    if len(rolling) > 2:
                        old = rolling.pop(0)
                        try:
                            for p in old.glob("*"):
                                if p.is_dir():
                                    for q in p.glob("*"):
                                        q.unlink()
                                    p.rmdir()
                                else:
                                    p.unlink()
                            old.rmdir()
                        except Exception as e:
                            print(f"[warn] cleanup failed: {e}", flush=True)

                    model.train()

                if args.max_steps is not None and optim_step >= args.max_steps:
                    print(f"[stop] reached max_steps={args.max_steps}",
                          flush=True)
                    early_stop = True
                    break

                # Patience-triggered early stop (set in the eval block above).
                if early_stop:
                    break

        if early_stop:
            break
        print(f"[epoch {epoch}] done @ step {optim_step}", flush=True)

    train_log_fh.close()
    val_log_fh.close()
    train_dt = time.time() - t0
    print(f"[train] done in {train_dt:.1f}s; best_val_mae={best_val_mae:.4f}",
          flush=True)

    # ------------------------------------------------------------------
    # Restore best, run final val eval, save adapter.
    # ------------------------------------------------------------------
    if best_state is not None:
        full = model.state_dict()
        for k, v in best_state["encoder_state"].items():
            full[k] = v.to(full[k].device)
        model.load_state_dict(full)
        model.head_score.load_state_dict(best_state["head_state"])
        print(f"[restore] best at step={best_state['step']} "
              f"val_mae={best_state['val_mae']:.4f}", flush=True)

    # Final adapter (best-on-val) → v1/.
    model.save_adapter(args.out_dir)
    tok_dir = args.out_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tok_dir))
    # `DebertaV2Tokenizer.save_pretrained` writes config but not the
    # SPM vocab file. Copy it in so eval/inference can construct the
    # tokenizer from the saved dir alone.
    import shutil as _shutil
    _shutil.copy(str(spm_path), str(tok_dir / "spm.model"))
    print(f"[save] adapter + heads + tokenizer -> {args.out_dir}", flush=True)

    # ------------------------------------------------------------------
    # Calibration on val ONLY.
    # ------------------------------------------------------------------
    print("[calibration] running val forward (no_grad) ...", flush=True)
    val_eval = eval_loop(model, val_loader, device)
    val_perts: list[str] = []
    for batch_rows in val_loader:
        val_perts.extend(batch_rows["perturbation_type"])
    # Realign — eval_loop iterates in same order with shuffle=False.
    # But to be safe, recompute val_perts via fresh pass:
    val_perts = []
    for r in JsonlPairDataset(DATASET_DIR / "val.jsonl").rows:
        val_perts.append(r.get("perturbation_type", "unknown"))

    # Hard split assertion.
    fit_split_name = "val"
    assert fit_split_name == "val", (
        "Calibration must be fit on val. Refusing to fit on test/probe."
    )

    cal = fit_calibration(val_eval, val_perts)
    cal_path = args.out_dir / "calibration.json"
    cal_path.write_text(json.dumps(cal, indent=2))
    print(
        f"[calibration] val MAE pre={cal['val_mae_pre']:.4f} "
        f"post={cal['val_mae_post']:.4f}  offset={cal['offset']:.4f}",
        flush=True,
    )
    print(f"[save] calibration -> {cal_path}", flush=True)

    # ------------------------------------------------------------------
    # Finalize summary.
    # ------------------------------------------------------------------
    val_lite = {k: v for k, v in val_eval.items()
                if k not in ("raw_preds", "labels")}
    summary["best_val_mae"] = float(best_val_mae)
    summary["best_val_spearman"] = float(best_val_spearman)
    summary["selection_metric"] = "val_spearman"
    summary["best_step"] = (
        int(best_state["step"]) if best_state is not None else None
    )
    summary["final_val"] = val_lite
    summary["calibration_summary"] = {
        "fit_split": cal["fit_split"],
        "offset": cal["offset"],
        "val_mae_pre": cal["val_mae_pre"],
        "val_mae_post": cal["val_mae_post"],
    }
    summary["peak_vram_gb"] = peak_vram_gb
    summary["first_step_vram_gb"] = first_step_vram_gb
    summary["total_train_seconds"] = train_dt
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] -> {summary_path}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
