"""SFT-D S8 — memorization-probe held-out assessment-item slice.

The SFT program (runtime/scratchpad/sft_data_program.md §C) requires a memorization
probe: hold out a slice of assessment items that is NEVER used to generate any
training pair, then each run compare adapter accuracy on the held-out items vs
the trained ones to quantify memorization-vs-generalization.

This module owns the deterministic CONTRACT both sides read:

* ``select_holdout_item_ids`` — deterministic (seeded) selection of a fraction
  of assessment-item ids to withhold. Pure + stable across runs given the same
  ids + seed + fraction.
* ``write_holdout_exclusion`` / ``load_holdout_exclusion`` — persist / read the
  exclusion list at the canonical path
  ``training_specs/.memorization_holdout.json``. The SFT pair generators
  (assessment_sft_generator + any assessment-derived generator) MUST call
  ``load_holdout_exclusion`` and skip any item whose id is in the returned set
  BEFORE emitting a pair, so the held-out slice is genuinely unseen.
* ``evaluate_memorization`` — compute the generalization gap from the two
  post-training accuracies.

No torch / trl / network. The actual per-item adapter accuracy is measured
out-of-band (GPU); this module only owns the id bookkeeping + the gap math.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# Canonical on-disk contract path (relative to a LibV2 course dir).
HOLDOUT_REL_PATH = "training_specs/.memorization_holdout.json"

# Default fraction of assessment items to withhold from pair-gen.
DEFAULT_HOLDOUT_FRACTION = 0.10
# Never withhold so much that pair-gen starves; never zero on a real corpus.
_MIN_TRAIN_ITEMS = 5


def _stable_rank(item_id: str, seed: int) -> str:
    """Deterministic per-item sort key (seeded sha256 hex).

    Independent of dict / list ordering so the same ids + seed always select
    the same slice on any machine.
    """
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def select_holdout_item_ids(
    item_ids: Iterable[str],
    *,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = 42,
) -> List[str]:
    """Deterministically choose a held-out slice of assessment-item ids.

    Args:
        item_ids: every assessment item's id (deduped internally).
        fraction: share to withhold (clamped to ``[0, 0.5]``).
        seed: RNG seed folded into the stable per-item hash.

    Returns the sorted held-out id list. Guarantees at least
    ``_MIN_TRAIN_ITEMS`` items remain for pair-gen (a tiny corpus withholds
    fewer, or none, rather than starving training).
    """
    unique = sorted({str(i) for i in item_ids if str(i).strip()})
    n = len(unique)
    if n == 0:
        return []
    frac = max(0.0, min(0.5, float(fraction)))
    k = int(n * frac)
    # Never leave fewer than _MIN_TRAIN_ITEMS for training.
    k = min(k, max(0, n - _MIN_TRAIN_ITEMS))
    if k <= 0:
        return []
    ordered = sorted(unique, key=lambda i: _stable_rank(i, seed))
    return sorted(ordered[:k])


def write_holdout_exclusion(
    course_dir: Path,
    item_ids: List[str],
    *,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = 42,
) -> Path:
    """Persist the exclusion list to the canonical contract path.

    Atomic tmp+replace. Returns the written path. The payload carries the
    seed + fraction so a re-run (or an auditor) can confirm the slice is
    reproducible.
    """
    path = Path(course_dir) / HOLDOUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v1",
        "purpose": "memorization_probe_holdout",
        "seed": int(seed),
        "fraction": float(fraction),
        "held_out_item_ids": sorted({str(i) for i in item_ids}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def load_holdout_exclusion(course_dir: Path) -> Set[str]:
    """Read the held-out item-id set (empty when absent / unreadable).

    **SFT pair generators MUST call this and skip any item whose id is in the
    returned set before emitting a pair.** Empty set == no holdout configured
    (byte-identical legacy behaviour — every item is pair-eligible).
    """
    path = Path(course_dir) / HOLDOUT_REL_PATH
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "memorization_probe: failed to read holdout exclusion %s (%s); "
            "treating as empty (no items withheld).", path, exc,
        )
        return set()
    ids = payload.get("held_out_item_ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list):
        return set()
    return {str(i) for i in ids}


def evaluate_memorization(
    *,
    accuracy_held_out: Optional[float],
    accuracy_trained: Optional[float],
) -> Dict[str, Any]:
    """Compute the memorization-vs-generalization gap.

    ``gap = accuracy_trained - accuracy_held_out``. A large positive gap means
    the adapter learned the trained items far better than unseen ones —
    memorization. Both accuracies are measured out-of-band on the same
    base+adapter. Returns a dict shaped to drop into ``eval_report.json``.
    Missing either input → ``gap=None`` (not fabricated).
    """
    gap: Optional[float] = None
    if accuracy_held_out is not None and accuracy_trained is not None:
        gap = float(accuracy_trained) - float(accuracy_held_out)
    return {
        "accuracy_trained_items": (
            float(accuracy_trained) if accuracy_trained is not None else None
        ),
        "accuracy_held_out_items": (
            float(accuracy_held_out) if accuracy_held_out is not None else None
        ),
        "memorization_gap": gap,
    }


__all__ = [
    "HOLDOUT_REL_PATH",
    "DEFAULT_HOLDOUT_FRACTION",
    "select_holdout_item_ids",
    "write_holdout_exclusion",
    "load_holdout_exclusion",
    "evaluate_memorization",
]
