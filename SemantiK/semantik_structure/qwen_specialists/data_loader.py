"""Phase 1.5 dataset loader for Qwen LoRA training.

Loads ``data/qwen_<adapter>_dataset/<split>.jsonl``, applies
``wrap_for_qwen`` around each row's stored
``request_payload.payload.prompt`` + ``target_html``, and yields rows
shaped for SFTTrainer with a precomputed ``prompt_len`` so the loss
mask can be computed deterministically (the chat template's tokenizer
output is whitespace-sensitive; recomputing the boundary at training
time risks drift between batch indices).

The output schema (per row) is::

    {
      "text": str,        # full chat-templated string (user+assistant)
      "prompt_len": int,  # token count of user-turn-only template wrap
                          # — every label index < prompt_len gets -100.
    }

A pre-flight survey runs at load time:

    * sample 200 rows
    * tokenize each, record p50/p99 lengths
    * if p99 > 2 * max_len: ABORT — the configured max_len is too
      small and most rows will be truncated, killing the training
      signal silently.

The survey is intentionally noisy in logs so a regression in dataset
shape (e.g. someone bumps caption_text length cap) trips a human
eyeball before the trainer wastes 8 hours.
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

from datasets import Dataset

from .chat_format import wrap_for_qwen


logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data"

_DATASET_DIR_FOR_ADAPTER = {
    # v2 @ max_len 2048 (~233K rows) — uncensors the long-MathML tail the
    # v1 1024-token filter dropped (32.5% of math elements). Supersedes
    # the 1024-capped qwen_math_dataset (v1, kept on disk for the loss-mask
    # test fixture). See Plans/01 + train_config math rationale.
    "math": "qwen_math_dataset_v2",
    # multi-source (arxiv+openstax+pmc+cfr+fedreg) @ max_len 2048, 4,337 rows
    # — supersedes the arxiv-only qwen_table_dataset (1,315 rows, kept as a
    # backup on disk). See Plans/06.
    "table": "qwen_table_dataset_multi",
    "prose": "qwen_prose_dataset",
    "gap_fill": "qwen_gap_fill_dataset",
}


class DatasetBlowupError(RuntimeError):
    """Raised when token-length p99 exceeds 2 * max_len."""


def _row_prompt(row: dict[str, Any]) -> str:
    """Pull the raw SYSTEM/USER prompt the prompt-builders emitted."""
    rp = row.get("request_payload") or {}
    payload = rp.get("payload") or {}
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            "row.request_payload.payload.prompt missing or empty; "
            "dataset row not built by prompts.build_*_request"
        )
    return prompt


def _row_target(row: dict[str, Any]) -> str:
    """Pull the assistant-turn target (HTML / MathML fragment)."""
    target = row.get("target_html")
    if not isinstance(target, str) or not target:
        raise ValueError(
            "row.target_html missing or empty; dataset row not "
            "wired for completion-only training"
        )
    return target


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _tokenize_blowup_survey(
    rows: list[dict[str, Any]],
    tok: Any,
    max_len: int,
    sample: int = 200,
) -> dict[str, float]:
    """Return p50/p99/max token-length stats over a sample. Raises if
    p99 exceeds 2 * max_len."""
    n = min(sample, len(rows))
    if n == 0:
        return {"p50": 0.0, "p99": 0.0, "max": 0.0, "n": 0}
    lengths: list[int] = []
    for row in rows[:n]:
        wrapped = wrap_for_qwen(tok, _row_prompt(row), target=_row_target(row))
        ids = tok(wrapped, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
    lengths.sort()
    p50 = float(lengths[len(lengths) // 2])
    p99_idx = max(0, int(len(lengths) * 0.99) - 1)
    p99 = float(lengths[p99_idx])
    mx = float(lengths[-1])
    stats = {"p50": p50, "p99": p99, "max": mx, "n": float(len(lengths))}
    logger.info(
        "tokenizer-blowup survey: n=%d max_len=%d p50=%.0f p99=%.0f max=%.0f",
        len(lengths), max_len, p50, p99, mx,
    )
    # The 2x slack is intentional: the dataset builder applies a
    # tokenizer-aware filter at the SAME max_len (so on a clean dataset
    # p99 should already be <= max_len). The 2x ceiling here is a
    # second-line defense against tokenizer drift between build-time
    # and train-time (e.g. someone bumps the base model and the
    # tokenizer's chat template grows by a few tokens per row). If
    # build-time and train-time tokenizers ever disagree by more than
    # 2x at p99, we want to fail loudly rather than silently truncate.
    if p99 > 2 * max_len:
        raise DatasetBlowupError(
            f"p99 token length {p99:.0f} > 2 * max_len ({max_len}); "
            f"most rows will be truncated. Increase max_len in "
            f"train_config.py or rebuild the dataset with shorter rows."
        )
    return stats


def _build_row(row: dict[str, Any], tok: Any, max_len: int) -> dict[str, Any]:
    """Compute the trainer row: full text + prompt_len for masking."""
    raw = _row_prompt(row)
    target = _row_target(row)
    # Inference-shape wrap (user only, with assistant prefix). Token
    # count of THIS is the prompt boundary; everything after is target.
    prompt_only = wrap_for_qwen(tok, raw, target=None, add_generation_prompt=True)
    full = wrap_for_qwen(tok, raw, target=target)
    prompt_ids = tok(prompt_only, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    return {
        "text": full,
        "prompt_len": prompt_len,
    }


def build_dataset(
    adapter: str,
    split: str,
    tok: Any,
    max_len: int,
    *,
    dataset_root: Path | None = None,
    survey: bool = True,
) -> Dataset:
    """Load and shape a split for Qwen LoRA training.

    Parameters
    ----------
    adapter
        One of ``{math, table, prose, gap_fill}``.
    split
        ``train`` | ``val`` | ``test``.
    tok
        HF tokenizer with chat_template.
    max_len
        Max sequence length (informational; the trainer enforces).
    dataset_root
        Override ``data/`` directory (tests).
    survey
        If True, run the blowup survey. Disable for tiny in-memory
        fixtures where 200 rows aren't available.
    """
    if adapter not in _DATASET_DIR_FOR_ADAPTER:
        raise ValueError(
            f"unknown adapter {adapter!r}; expected one of "
            f"{sorted(_DATASET_DIR_FOR_ADAPTER)}"
        )
    root = Path(dataset_root) if dataset_root else DATASET_ROOT
    path = root / _DATASET_DIR_FOR_ADAPTER[adapter] / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"dataset split missing: {path}")

    rows = _load_jsonl(path)
    logger.info("loaded %d rows from %s", len(rows), path)

    if survey:
        _tokenize_blowup_survey(rows, tok, max_len)

    shaped = [_build_row(r, tok, max_len) for r in rows]
    return Dataset.from_list(shaped)


__all__ = ["build_dataset", "DatasetBlowupError"]
