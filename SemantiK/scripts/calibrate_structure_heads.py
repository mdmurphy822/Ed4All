"""Post-hoc temperature calibration for the BERT-Structure ``is_heading`` head.

Walks ``data/structure_dataset/val.jsonl`` through the trained Structure
adapter (LoRA + heads + layout MLP), collects raw ``is_heading`` logits +
labels, fits a single scalar temperature ``T`` (LBFGS + grid cross-check),
and rewrites ``models/council/structure/final/heads.pt`` with an extra
``is_heading_temperature`` key so the runtime can plumb it through.

This is post-hoc: NO weights move, NO retraining. The script is fit
ONLY on val.jsonl (asserted in code) and is hard-failing on a few
sanity guards (T >5.0, >30% of val samples landing in the ambiguous
[0.45, 0.55] band) so a bad fit can't silently land in production.

Usage:
    .venv/bin/python scripts/calibrate_structure_heads.py
    .venv/bin/python scripts/calibrate_structure_heads.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from training.train_structure import (  # noqa: E402  (after sys.path tweak)
    StructureModel,
    load_split,
)


# Sanity guards (gate 3).
_T_MAX = 5.0
_AMBIGUOUS_BAND = (0.45, 0.55)
_AMBIGUOUS_MAX_FRAC = 0.30


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_loader(
    val_ds: Dataset,
    tok,
    *,
    max_length: int,
    batch_size: int,
) -> DataLoader:
    """Mirrors train_structure._make_loader (eval branch) — same fields,
    same DataCollatorWithPadding, no sampler.
    """
    coll = DataCollatorWithPadding(tok, return_tensors="pt")
    TOK_FIELDS = ("input_ids", "attention_mask", "token_type_ids")

    def collate(batch):
        feats = [{k: b[k] for k in TOK_FIELDS if k in b} for b in batch]
        out = coll(feats)
        out["y_is_heading"] = torch.tensor(
            [b["y_is_heading"] for b in batch], dtype=torch.long,
        )
        out["layout"] = torch.tensor(
            [b["layout"] for b in batch], dtype=torch.float32,
        )
        return out

    return DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
    )


def _load_lora_into_model(
    model: StructureModel,
    adapter_dir: Path,
    device: torch.device,
) -> None:
    """Replace the model's encoder with the saved PEFT adapter.

    Mirrors how `dart_semantic.council.structure._load_heads` + the
    council runner do it: PEFT's encoder is saved by
    `model.encoder.save_pretrained(adapter_dir)`, which writes
    `adapter_config.json` + `adapter_model.safetensors`. We use
    `PeftModel.from_pretrained` to re-attach.

    NOTE: train_structure builds a fresh PEFT model on construction;
    we throw that away and load the saved one on top of the same base.
    """
    from peft import PeftModel
    from transformers import AutoModel

    # Pull the *base* (unwrapped) encoder out — train_structure's
    # StructureModel.__init__ already wrapped it. The base is at
    # `model.encoder.base_model.model` (peft layout) on construction.
    # Cleaner: build a fresh AutoModel + attach the saved adapter, then
    # swap it in. That avoids any reuse-of-existing-LoRA-state risks.
    base_name = "answerdotai/ModernBERT-base"  # train_structure default
    base = AutoModel.from_pretrained(base_name, dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    peft_model.eval()
    model.encoder = peft_model
    model.to(device)
    model.eval()


def _load_heads_into_model(
    model: StructureModel,
    heads_path: Path,
) -> None:
    """Load classifier heads + layout MLP from heads.pt into ``model``."""
    state = torch.load(str(heads_path), map_location="cpu")
    model.head_role.load_state_dict(state["head_role.state_dict"])
    model.head_is_heading.load_state_dict(state["head_is_heading.state_dict"])
    model.head_table_region.load_state_dict(
        state["head_table_region.state_dict"],
    )
    if "head_is_image_block.state_dict" in state:
        model.head_is_image_block.load_state_dict(
            state["head_is_image_block.state_dict"],
        )
    model.head_list_nesting.load_state_dict(
        state["head_list_nesting.state_dict"],
    )
    model.layout_norm.load_state_dict(state["layout_norm.state_dict"])
    model.layout_mlp.load_state_dict(state["layout_mlp.state_dict"])


def _collect_logits(
    model: StructureModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run val through ``model`` and return (logits[N,2], labels[N])."""
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            layout = batch["layout"].to(device)
            yh = batch["y_is_heading"].to(device)
            out = model(ids, mask, layout)
            logits = out["is_heading"].float().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(yh.cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# Calibration — temperature fit + metrics
# ---------------------------------------------------------------------------


def _nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    """Cross-entropy of softmax(logits/T) against labels (mean over N)."""
    z = torch.tensor(logits, dtype=torch.float64) / float(T)
    y = torch.tensor(labels, dtype=torch.long)
    return float(F.cross_entropy(z, y, reduction="mean").item())


def _ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Guo et al. 2017), the standard
    temperature-scaling diagnostic. Bins are over the **top-1 predicted-
    class confidence**, and per-bin "accuracy" is the fraction of
    predictions in that bin where the argmax matches the label.

    ``probs`` is the full softmax distribution shape ``(N, K)``;
    ``labels`` is the integer-class array shape ``(N,)``.
    """
    if len(probs) == 0:
        return 0.0
    pred = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _brier(probs_pos: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs_pos - labels) ** 2))


def _pr_at_threshold(
    probs_pos: np.ndarray,
    labels: np.ndarray,
    thr: float,
) -> dict[str, float]:
    pred = (probs_pos >= thr).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-12, p + r)
    return {
        "threshold": float(thr),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "n_predicted_positive": int(pred.sum()),
        "rate_predicted_positive": float(pred.mean()),
    }


def _stats_at_T(
    logits: np.ndarray,
    labels: np.ndarray,
    T: float,
) -> dict:
    """All-in-one calibration stats at temperature T."""
    z = torch.tensor(logits, dtype=torch.float64) / float(T)
    probs = F.softmax(z, dim=-1).numpy()
    probs_pos = probs[:, 1]
    out = {
        "T": float(T),
        "nll": _nll(logits, labels, T),
        "ece_15bin": _ece(probs, labels, n_bins=15),
        "brier": _brier(probs_pos, labels),
        "pos_rate_pred_top1": float((probs_pos >= 0.5).mean()),
    }
    out["pr_at_threshold"] = {
        f"{thr:.1f}": _pr_at_threshold(probs_pos, labels, thr)
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
    return out


def _fit_lbfgs(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    n_iter: int = 50,
) -> float:
    """Fit T = exp(log_T) by LBFGS on CE loss. Returns scalar T."""
    Z = torch.tensor(logits, dtype=torch.float64)
    Y = torch.tensor(labels, dtype=torch.long)
    log_T = torch.zeros((), dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [log_T],
        lr=0.1,
        max_iter=n_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(Z / T, Y, reduction="mean")
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_T).item())


def _fit_grid(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[float, float]]:
    """Coarse grid scan; return (best_T, {T → NLL})."""
    grid = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
    nlls = {T: _nll(logits, labels, T) for T in grid}
    best_T = min(grid, key=lambda T: nlls[T])
    return best_T, nlls


def _check_sanity_guards(
    T: float,
    logits: np.ndarray,
) -> tuple[bool, list[str]]:
    """Return (ok, list_of_failure_messages)."""
    failures: list[str] = []
    if T > _T_MAX:
        failures.append(
            f"GUARD T-max: T={T:.4f} > {_T_MAX} (post-hoc temperature too "
            f"large — model is severely under-confident; investigate before "
            f"trusting this fit)"
        )
    z = torch.tensor(logits, dtype=torch.float64) / float(T)
    probs_pos = F.softmax(z, dim=-1)[:, 1].numpy()
    band_lo, band_hi = _AMBIGUOUS_BAND
    in_band = ((probs_pos >= band_lo) & (probs_pos <= band_hi)).mean()
    if in_band > _AMBIGUOUS_MAX_FRAC:
        failures.append(
            f"GUARD ambiguous-band: {100 * in_band:.2f}% of val samples "
            f"land in P(heading=1) ∈ [{band_lo}, {band_hi}] "
            f"(>{100 * _AMBIGUOUS_MAX_FRAC:.0f}% — over-flattening; "
            f"calibration would make the gate near-useless)"
        )
    return (len(failures) == 0, failures)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("models/council/structure/final"),
    )
    ap.add_argument(
        "--val-path", type=Path,
        default=Path("data/structure_dataset/val.jsonl"),
    )
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Don't rewrite heads.pt or summary.json; print only.",
    )
    args = ap.parse_args()

    val_path: Path = args.val_path.resolve()
    if val_path.name != "val.jsonl":
        print(
            f"[fatal] --val-path must point to a file named 'val.jsonl' "
            f"(got '{val_path.name}'). Calibration MUST be fit on the "
            f"validation split — fitting on train would over-estimate "
            f"confidence; fitting on test would leak into the final "
            f"acceptance gate.",
            file=sys.stderr,
        )
        return 1
    if not val_path.exists():
        print(f"[fatal] {val_path} does not exist", file=sys.stderr)
        return 1

    ckpt: Path = args.checkpoint_dir.resolve()
    heads_path = ckpt / "heads.pt"
    summary_path = ckpt / "summary.json"
    if not heads_path.exists():
        print(f"[fatal] {heads_path} does not exist", file=sys.stderr)
        return 1
    if not summary_path.exists():
        print(
            f"[fatal] {summary_path} does not exist (need to update its "
            f"calibration block)",
            file=sys.stderr,
        )
        return 1

    device = torch.device(args.device)
    print(f"[gpu] device={device}")
    print(f"[ckpt] {ckpt}")
    print(f"[val ] {val_path}")

    val_sha = _sha256(val_path)
    print(f"[val ] sha256={val_sha}")

    print("[load] dataset (load_split mirrors train_structure)")
    val_ds = load_split(val_path)
    print(f"  val size = {len(val_ds)}")
    if len(val_ds) == 0:
        print("[fatal] val split is empty after load_split", file=sys.stderr)
        return 1

    print(f"[load] tokenizer ({args.base_model})")
    tok_dir = ckpt / "tokenizer"
    if tok_dir.exists():
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
    else:
        tok = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tok(
            batch["text"], truncation=True, max_length=args.max_length,
        )

    val_ds = val_ds.map(tokenize, batched=True)

    val_loader = _build_loader(
        val_ds, tok, max_length=args.max_length, batch_size=args.batch_size,
    )

    print("[load] StructureModel + saved LoRA + heads")
    model = StructureModel(args.base_model)
    _load_lora_into_model(model, ckpt, device)
    _load_heads_into_model(model, heads_path)
    model.to(device)
    model.eval()

    print("[forward] collecting val logits …")
    t0 = time.time()
    logits, labels = _collect_logits(model, val_loader, device)
    elapsed = time.time() - t0
    print(
        f"  collected logits shape={logits.shape}  labels shape={labels.shape}"
        f"  ({elapsed:.1f}s)"
    )
    print(f"  positive-class fraction (truth) = {labels.mean():.4f}")

    # Cache logits for forensic re-fits.
    cache_dir = _REPO_ROOT / "data/eval_reports"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "is_heading_val_logits.npz"
    np.savez(npz_path, logits=logits, labels=labels)
    print(f"[save] {npz_path}")

    # ----- Pre-calibration stats -----
    pre = _stats_at_T(logits, labels, T=1.0)
    print(
        f"\n[pre-cal] T=1.000  NLL={pre['nll']:.4f}  "
        f"ECE15={pre['ece_15bin']:.4f}  Brier={pre['brier']:.4f}  "
        f"pred_pos_rate={pre['pos_rate_pred_top1']:.4f}"
    )
    for thr_str, m in pre["pr_at_threshold"].items():
        print(
            f"  thr={thr_str}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  pred_pos_rate={m['rate_predicted_positive']:.4f}"
        )

    # ----- Fit T two ways -----
    print("\n[fit] LBFGS …")
    T_lbfgs = _fit_lbfgs(logits, labels, n_iter=50)
    print(f"  T_lbfgs = {T_lbfgs:.6f}  NLL = {_nll(logits, labels, T_lbfgs):.6f}")

    print("[fit] grid …")
    T_grid, grid_nlls = _fit_grid(logits, labels)
    print(f"  T_grid = {T_grid:.4f}  NLL = {grid_nlls[T_grid]:.6f}")
    for T, nll in sorted(grid_nlls.items()):
        marker = "  <-- best" if T == T_grid else ""
        print(f"    T={T:.2f}  NLL={nll:.6f}{marker}")

    if abs(T_lbfgs - T_grid) > 0.5:
        # The grid is coarse; a 0.5 mismatch on the grid spacing is ok if
        # LBFGS sits within one bin of the grid argmin. Tighter bound
        # checks NLL agreement.
        nll_lbfgs = _nll(logits, labels, T_lbfgs)
        nll_grid = grid_nlls[T_grid]
        if abs(nll_lbfgs - nll_grid) > 0.02:
            print(
                f"[fatal] LBFGS vs grid disagree by NLL > 0.02: "
                f"T_lbfgs={T_lbfgs:.4f} (NLL={nll_lbfgs:.4f}) vs "
                f"T_grid={T_grid:.4f} (NLL={nll_grid:.4f}). Aborting.",
                file=sys.stderr,
            )
            return 3

    T = float(T_lbfgs)

    # ----- Sanity guards -----
    ok, failures = _check_sanity_guards(T, logits)
    if not ok:
        for msg in failures:
            print(f"[guard FAIL] {msg}", file=sys.stderr)
        print(
            f"[fatal] sanity guards tripped at T={T:.4f}; refusing to "
            f"write heads.pt.",
            file=sys.stderr,
        )
        return 2

    # ----- Post-calibration stats -----
    post = _stats_at_T(logits, labels, T=T)
    print(
        f"\n[post-cal] T={T:.4f}  NLL={post['nll']:.4f}  "
        f"ECE15={post['ece_15bin']:.4f}  Brier={post['brier']:.4f}  "
        f"pred_pos_rate={post['pos_rate_pred_top1']:.4f}"
    )
    for thr_str, m in post["pr_at_threshold"].items():
        print(
            f"  thr={thr_str}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  pred_pos_rate={m['rate_predicted_positive']:.4f}"
        )

    nll_drop = pre["nll"] - post["nll"]
    ece_pre = pre["ece_15bin"] or 1e-9
    ece_ratio = ece_pre / max(post["ece_15bin"], 1e-9)
    print(f"\n[Δ] NLL drop  = {nll_drop:+.4f}  (must be > 0)")
    print(f"[Δ] ECE ratio = {ece_ratio:.2f}× (must be ≥ 3×)")

    gate_nll_ok = nll_drop > 0.0
    gate_ece_ok = ece_ratio >= 3.0
    if not gate_nll_ok:
        print(
            "[gate FAIL] NLL did not drop after calibration. Either the "
            "model is already calibrated (T≈1) or there's a bug in the "
            "forward path. Refusing to write heads.pt.",
            file=sys.stderr,
        )
        return 4
    if not gate_ece_ok:
        print(
            f"[gate FAIL] ECE ratio {ece_ratio:.2f}× < 3× — calibration "
            f"helped but not enough. Refusing to write heads.pt.",
            file=sys.stderr,
        )
        return 5

    # ----- Build calibration metadata block -----
    metadata = {
        "method": "single_scalar_temperature",
        "T": T,
        "fit": {
            "lbfgs_T": T_lbfgs,
            "grid_T": T_grid,
            "grid_nlls": {f"{T:.2f}": v for T, v in grid_nlls.items()},
        },
        "val_path": str(val_path),
        "val_sha256": val_sha,
        "val_size": int(len(val_ds)),
        "before": pre,
        "after": post,
        "guards": {
            "T_max": _T_MAX,
            "ambiguous_band": list(_AMBIGUOUS_BAND),
            "ambiguous_max_frac": _AMBIGUOUS_MAX_FRAC,
            "passed": True,
        },
        "logits_npz": str(npz_path),
        "fit_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if args.dry_run:
        print("\n[dry-run] would write heads.pt + summary.json. Skipping.")
        print(json.dumps({"T": T, "calibration_summary": {
            "before_nll": pre["nll"], "after_nll": post["nll"],
            "before_ece": pre["ece_15bin"], "after_ece": post["ece_15bin"],
        }}, indent=2))
        return 0

    # ----- Backup heads.pt -----
    bak_path = heads_path.with_suffix(heads_path.suffix + ".bak")
    if not bak_path.exists():
        shutil.copy2(heads_path, bak_path)
        print(f"[backup] {bak_path}")
    else:
        print(f"[backup] {bak_path} already exists; not overwriting")

    # ----- Rewrite heads.pt with the temperature key -----
    state = torch.load(str(heads_path), map_location="cpu")
    state["is_heading_temperature"] = float(T)
    state["is_heading_calibration"] = {
        "method": metadata["method"],
        "val_sha256": val_sha,
        "val_size": metadata["val_size"],
        "T": T,
        "fit_at": metadata["fit_at"],
    }
    torch.save(state, str(heads_path))
    print(f"[save] {heads_path} (now carries is_heading_temperature={T:.6f})")

    # ----- Append calibration block to summary.json -----
    summary = json.loads(summary_path.read_text())
    summary["calibration"] = {"is_heading": metadata}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] {summary_path}")

    print(
        f"\n[done] T={T:.4f}  "
        f"NLL {pre['nll']:.4f}->{post['nll']:.4f}  "
        f"ECE {pre['ece_15bin']:.4f}->{post['ece_15bin']:.4f} "
        f"({ece_ratio:.1f}× drop)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
