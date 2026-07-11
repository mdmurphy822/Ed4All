"""Phase 1.5 — train one Qwen LoRA adapter (math/table/prose/gap_fill).

Thin wrapper around ``semantik_structure.qwen_specialists.training.train``.
The shared trainer holds all the per-adapter logic; this CLI just picks
the adapter id and routes the args.

Usage::

    .venv/bin/python scripts/train_qwen_lora.py \\
        --adapter math \\
        --out-dir models/qwen_specialists/math/v1 \\
        [--resume-from <ckpt>] \\
        [--overwrite] \\
        [--max-steps 50]   # smoke

Smoke gate: ``--max-steps`` lets us prove the trainer launches +
descends loss on a 50-step slice without committing to a 30K-step run.

The trainer refuses to start if the GPU is in use (``training_lock``)
and refuses to overwrite an adapter directory unless ``--overwrite``
is passed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from semantik_structure.qwen_specialists.train_config import ADAPTER_CONFIGS
from semantik_structure.qwen_specialists.training import (
    AdapterTrainError,
    train,
)
from semantik_structure.qwen_specialists.training_lock import TrainingSlotBusy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--adapter",
        required=True,
        choices=sorted(ADAPTER_CONFIGS.keys()),
        help="Adapter id (math/table/prose/gap_fill).",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Where to write adapter, logs, summary.",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional resume-from-checkpoint path.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow non-empty out_dir.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap optimizer steps (smoke runs).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip the nvidia-smi GPU-busy poll. Use for CPU-only "
             "lock-only smoke; the trainer itself still requires GPU.",
    )
    ap.add_argument(
        "--eval-size",
        type=int,
        default=200,
        help="Cap val split to N rows. 0 = full split.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    cfg = ADAPTER_CONFIGS[args.adapter]
    print(f"[train] adapter={args.adapter} out_dir={args.out_dir}")
    print(f"[train] cfg={cfg}")

    try:
        summary = train(
            adapter_id=args.adapter,
            out_dir=args.out_dir,
            cfg=cfg,
            resume_from=args.resume_from,
            overwrite=args.overwrite,
            max_steps=args.max_steps,
            seed=args.seed,
            skip_gpu_check=args.skip_gpu_check,
            eval_size=args.eval_size,
        )
    except TrainingSlotBusy as exc:
        print(f"[train] ABORT: training slot busy: {exc}", file=sys.stderr)
        return 2
    except AdapterTrainError as exc:
        print(f"[train] ABORT: {exc}", file=sys.stderr)
        return 3

    print(f"[train] done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
