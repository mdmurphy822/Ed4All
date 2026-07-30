"""Bloom-label dataset preparation (CPU-only; no ML deps at import time).

Reads the harvester's row schema (``lib.bloom_labels.harvester.BloomLabel`` /
``docs``: ``text`` / ``bloom_level`` / ``artifact_kind`` / ``content_sha256`` /
...) straight off a ``labels.jsonl`` path the caller supplies -- never a
hardcoded location -- and turns it into what a downstream trainer needs:

* **de-dup** by ``content_sha256`` (the harvester's own dedupe key; a store
  that was harvested across several runs may still carry a stray duplicate
  from a manual copy/merge).
* a **seeded, per-level stratified train/val split** so the thin
  create/evaluate tail is represented in both splits instead of a pooled
  random split occasionally dropping the rarest level from val entirely.
* **per-level one-vs-rest binary views** (``positive`` = rows at that level,
  ``negative`` = everything else) -- the framing a per-level binary head
  trains against.
* a **multiclass view** (all six levels as a single labeled dataset,
  :data:`BLOOM_LEVEL_TO_ID` giving the canonical 0-5 label id per level) --
  the alternative framing a single 6-way softmax head trains against
  instead of six independent one-vs-rest heads (Bloom-ladder addendum
  WI-AD-01).
* **class-weight computation** (balanced inverse-frequency, multiclass and
  per-view binary) so a trainer can counteract the imbalance without
  resampling.
* :func:`class_balance_report` -- the operator-facing corpus census. Always
  reports all six canonical levels (a level nobody has harvested yet reads as
  an explicit zero, never a silent absence) and flags any level that falls
  below half the balanced "fair share" as ``thin`` -- on the live corpus this
  is exactly the create/evaluate tail, and it must stay visible rather than
  disappearing into an aggregate accuracy number.

This module is deliberately import-light: it is pure stdlib (``json`` /
``random`` / ``dataclasses`` / ``pathlib``), so it is importable -- and its
splits/reports computable -- without the ``[training]`` extra. Any actual
tensor construction / model training is out of scope here and belongs to a
later, torch-importing module.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Canonical Bloom levels, low -> high complexity. Mirrors
#: ``lib.ontology.bloom.BLOOM_LEVELS`` / ``lib.bloom_labels.harvester._BLOOM_LEVELS``
#: (inlined import-light, same convention the harvester already uses).
BLOOM_LEVELS: Tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)
_BLOOM_LEVEL_SET = frozenset(BLOOM_LEVELS)

#: Canonical multiclass label ids -- BLOOM_LEVELS order defines the label
#: space a single 6-way softmax head trains against (label id 3 ==
#: "analyze" everywhere: dataset prep, the trainer's id2label/label2id, and
#: the loader's argmax). Mirrors the one-vs-rest LABEL2ID/ID2LABEL
#: convention in ``lib.classifiers.training.train_bloom_deberta``, just for
#: all six levels instead of one binary head at a time.
BLOOM_LEVEL_TO_ID: Dict[str, int] = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}
ID_TO_BLOOM_LEVEL: Dict[int, str] = {idx: level for level, idx in BLOOM_LEVEL_TO_ID.items()}

#: A level is flagged "thin" in :func:`class_balance_report` when its count
#: falls below this fraction of the balanced "fair share" (total / 6 levels).
#: Self-scaling -- not tuned to any one corpus snapshot -- so it keeps
#: flagging a create/evaluate-shaped imbalance whether the corpus is 300 rows
#: or 30,000.
_THIN_CLASS_FAIR_SHARE_FRACTION = 0.5

#: Default val-split fraction and RNG seed for :func:`stratified_split` /
#: :func:`prepare_dataset`.
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# row shape
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelRow:
    """One de-duplicated Bloom-label row (the fields this module reads out of
    the harvester's wider :class:`lib.bloom_labels.harvester.BloomLabel`)."""

    text: str
    bloom_level: str
    artifact_kind: str
    content_sha256: str
    source_id: str = ""
    model_provenance: str = ""
    license_tag: str = ""
    run_id: str = ""
    harvested_at: str = ""
    schema_version: str = ""

    @classmethod
    def from_row(cls, row: Any) -> Optional["LabelRow"]:
        """Build a :class:`LabelRow` from one parsed JSONL row.

        Tolerant, skip-not-fail -- mirrors the harvester's own contract: a
        row missing ``text`` / a canonical ``bloom_level`` /
        ``content_sha256`` is not a usable labeled example, so it returns
        ``None`` (the caller counts it) rather than raising.
        """
        if not isinstance(row, dict):
            return None
        text = row.get("text")
        level = row.get("bloom_level")
        sha = row.get("content_sha256")
        if not isinstance(text, str) or not text.strip():
            return None
        if not isinstance(level, str) or level.strip().lower() not in _BLOOM_LEVEL_SET:
            return None
        if not isinstance(sha, str) or not sha.strip():
            return None
        return cls(
            text=text,
            bloom_level=level.strip().lower(),
            artifact_kind=str(row.get("artifact_kind") or ""),
            content_sha256=sha,
            source_id=str(row.get("source_id") or ""),
            model_provenance=str(row.get("model_provenance") or ""),
            license_tag=str(row.get("license_tag") or ""),
            run_id=str(row.get("run_id") or ""),
            harvested_at=str(row.get("harvested_at") or ""),
            schema_version=str(row.get("schema_version") or ""),
        )


@dataclass
class LoadResult:
    """Result of :func:`load_labels`."""

    rows: List[LabelRow] = field(default_factory=list)
    total_lines: int = 0
    malformed_skipped: int = 0
    duplicates_skipped: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "row_count": len(self.rows),
            "total_lines": self.total_lines,
            "malformed_skipped": self.malformed_skipped,
            "duplicates_skipped": self.duplicates_skipped,
        }


# ---------------------------------------------------------------------------
# load + de-dup
# ---------------------------------------------------------------------------
def load_labels(store_path: str | os.PathLike[str]) -> LoadResult:
    """Read + de-duplicate a harvested ``labels.jsonl`` file.

    ``store_path`` is always caller-supplied (a positional parameter) --
    never a hardcoded location -- so a caller (or a test) can point this at
    any labels store, including a ``tmp_path`` synthetic fixture.

    De-dup is by ``content_sha256`` (the harvester's own dedupe key), with
    first-occurrence-wins: the earliest row in file order is kept, a later
    duplicate is counted, not appended. A structurally malformed line (bad
    JSON, non-dict, or missing/invalid required field) is skipped and
    counted, never raised -- the harvester's own skip-not-fail contract,
    mirrored here for the reader side. A missing file returns an empty
    result rather than raising (nothing has been harvested yet).
    """
    path = Path(store_path)
    result = LoadResult()
    if not path.exists():
        return result
    seen_shas: set = set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("bloom-labels dataset: unreadable store %s: %s", path, exc)
        return result
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        result.total_lines += 1
        try:
            raw = json.loads(line)
        except ValueError:
            result.malformed_skipped += 1
            continue
        label = LabelRow.from_row(raw)
        if label is None:
            result.malformed_skipped += 1
            continue
        if label.content_sha256 in seen_shas:
            result.duplicates_skipped += 1
            continue
        seen_shas.add(label.content_sha256)
        result.rows.append(label)
    return result


# ---------------------------------------------------------------------------
# class-balance census
# ---------------------------------------------------------------------------
def class_balance_report(rows: Sequence[LabelRow]) -> Dict[str, Any]:
    """Per-level count/proportion breakdown -- the operator-facing corpus census.

    Always reports all six canonical levels (a level with zero harvested rows
    reads as an explicit ``0``, never a silent absence). ``thin_levels``
    names every level whose count falls below half the balanced "fair share"
    (``total / 6``) -- on the live corpus this is exactly the
    create/evaluate tail (roughly 930 / 790 / 590 /
    350 / 140 / 80 for understand / apply / remember / analyze /
    evaluate / create), surfaced explicitly so a
    downstream trainer never silently starves those heads.
    """
    counts: Dict[str, int] = {level: 0 for level in BLOOM_LEVELS}
    for row in rows:
        counts[row.bloom_level] = counts.get(row.bloom_level, 0) + 1
    total = sum(counts.values())
    fair_share = (total / len(BLOOM_LEVELS)) if total else 0.0
    thin_floor = fair_share * _THIN_CLASS_FAIR_SHARE_FRACTION
    proportions = {
        level: (counts[level] / total if total else 0.0) for level in BLOOM_LEVELS
    }
    thin_levels = [level for level in BLOOM_LEVELS if total and counts[level] < thin_floor]
    missing_levels = [level for level in BLOOM_LEVELS if counts[level] == 0]
    present_counts = [c for c in counts.values() if c > 0]
    imbalance_ratio: Optional[float] = (
        max(present_counts) / min(present_counts) if present_counts else None
    )
    return {
        "total": total,
        "counts": counts,
        "proportions": proportions,
        "thin_levels": thin_levels,
        "missing_levels": missing_levels,
        "min_level": min(counts, key=lambda lv: counts[lv]) if total else None,
        "max_level": max(counts, key=lambda lv: counts[lv]) if total else None,
        "imbalance_ratio": imbalance_ratio,
    }


# ---------------------------------------------------------------------------
# stratified split
# ---------------------------------------------------------------------------
@dataclass
class DatasetSplit:
    """Seeded stratified train/val split over a de-duplicated row set."""

    train: List[LabelRow]
    val: List[LabelRow]
    seed: int
    val_fraction: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "train_count": len(self.train),
            "val_count": len(self.val),
            "seed": self.seed,
            "val_fraction": self.val_fraction,
        }


def stratified_split(
    rows: Sequence[LabelRow],
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> DatasetSplit:
    """Seeded, per-level stratified train/val split.

    Splitting independently per Bloom level (rather than over the pooled
    corpus) keeps the thin create/evaluate classes represented in BOTH
    splits instead of a global random split occasionally starving val (or
    train) of the rarest level entirely.

    Deterministic: the same ``(rows, val_fraction, seed)`` always produces
    the same split, regardless of the incoming row order or the platform's
    dict-iteration order -- each level gets its own ``random.Random`` seeded
    off ``f"{seed}:{level}"`` (a string, so it seeds identically everywhere;
    plain ints/tuples are not a portable ``random.Random`` seed across
    interpreter versions).

    A level with fewer than 2 rows keeps everything in train -- there is no
    way to hold out a val example without leaving that level with zero train
    examples.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction!r}")

    by_level: Dict[str, List[LabelRow]] = {level: [] for level in BLOOM_LEVELS}
    for row in rows:
        by_level.setdefault(row.bloom_level, []).append(row)

    train: List[LabelRow] = []
    val: List[LabelRow] = []
    for level in BLOOM_LEVELS:
        group = by_level.get(level, [])
        if len(group) < 2:
            train.extend(group)
            continue
        rng = random.Random(f"{seed}:{level}")
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        n_val = min(n_val, len(shuffled) - 1)  # always leave >= 1 in train
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return DatasetSplit(train=train, val=val, seed=seed, val_fraction=val_fraction)


# ---------------------------------------------------------------------------
# one-vs-rest binary views
# ---------------------------------------------------------------------------
@dataclass
class OneVsRestView:
    """Binary positive/negative split for one Bloom level."""

    level: str
    positive: List[LabelRow]
    negative: List[LabelRow]

    @property
    def n_positive(self) -> int:
        return len(self.positive)

    @property
    def n_negative(self) -> int:
        return len(self.negative)


def one_vs_rest_view(rows: Sequence[LabelRow], level: str) -> OneVsRestView:
    """Binary positive/negative view for one Bloom level.

    ``positive`` = rows asserted AT ``level``; ``negative`` = every other
    row, regardless of its own level -- the standard one-vs-rest framing for
    training a single per-level binary head.
    """
    normalized = level.strip().lower()
    if normalized not in _BLOOM_LEVEL_SET:
        raise ValueError(f"unknown bloom level {level!r}; expected one of {BLOOM_LEVELS}")
    positive = [row for row in rows if row.bloom_level == normalized]
    negative = [row for row in rows if row.bloom_level != normalized]
    return OneVsRestView(level=normalized, positive=positive, negative=negative)


def all_one_vs_rest_views(rows: Sequence[LabelRow]) -> Dict[str, OneVsRestView]:
    """One-vs-rest view for every canonical level, keyed by level name."""
    return {level: one_vs_rest_view(rows, level) for level in BLOOM_LEVELS}


# ---------------------------------------------------------------------------
# multiclass view -- all six levels as a single labeled dataset
# ---------------------------------------------------------------------------
@dataclass
class MulticlassView:
    """All six Bloom levels framed as a single labeled (multiclass) dataset.

    Unlike :class:`OneVsRestView` (one level vs. everything else), ``rows``
    here is the FULL row set, unsplit -- the multiclass framing needs no
    positive/negative partition, only a canonical numeric label per row
    (:data:`BLOOM_LEVEL_TO_ID`), which a downstream trainer applies at
    tensor-construction time (kept out of this stdlib-only module).
    """

    rows: List[LabelRow]

    @property
    def label_counts(self) -> Dict[str, int]:
        """Per-level row count within this view -- same shape as
        :func:`class_balance_report`'s ``counts``, but scoped to just this
        view's rows (typically one split, e.g. train or val)."""
        counts: Dict[str, int] = {level: 0 for level in BLOOM_LEVELS}
        for row in self.rows:
            counts[row.bloom_level] = counts.get(row.bloom_level, 0) + 1
        return counts


def multiclass_view(rows: Sequence[LabelRow]) -> MulticlassView:
    """Wrap ``rows`` as a :class:`MulticlassView`.

    The six-way framing a single-head multiclass trainer wants, offered
    alongside the six per-level :func:`one_vs_rest_view` binary framings --
    same input rows, different downstream label shape.
    """
    return MulticlassView(rows=list(rows))


# ---------------------------------------------------------------------------
# class weights
# ---------------------------------------------------------------------------
def compute_class_weights(rows: Sequence[LabelRow]) -> Dict[str, float]:
    """Balanced (inverse-frequency) multiclass weights over the 6 levels.

    Standard "balanced" formula: ``weight(level) = total / (n_present *
    count(level))``, using only the levels PRESENT in ``rows`` for
    ``n_present`` -- matching ``sklearn``'s ``class_weight="balanced"``
    semantics. A level with zero rows gets weight ``0.0`` (there is nothing
    to weight; a caller must not train against a class with no support).
    """
    counts: Dict[str, int] = {level: 0 for level in BLOOM_LEVELS}
    for row in rows:
        counts[row.bloom_level] = counts.get(row.bloom_level, 0) + 1
    total = sum(counts.values())
    n_present = sum(1 for c in counts.values() if c > 0)
    if total == 0 or n_present == 0:
        return {level: 0.0 for level in BLOOM_LEVELS}
    return {
        level: (total / (n_present * counts[level])) if counts[level] > 0 else 0.0
        for level in BLOOM_LEVELS
    }


def compute_binary_class_weights(view: OneVsRestView) -> Dict[str, float]:
    """Balanced positive/negative weights for one one-vs-rest view.

    Same balanced formula collapsed to two classes:
    ``weight(class) = total / (2 * count(class))``.
    """
    n_pos, n_neg = view.n_positive, view.n_negative
    total = n_pos + n_neg
    if total == 0:
        return {"positive": 0.0, "negative": 0.0}
    return {
        "positive": (total / (2 * n_pos)) if n_pos > 0 else 0.0,
        "negative": (total / (2 * n_neg)) if n_neg > 0 else 0.0,
    }


def compute_multiclass_class_weights(view: MulticlassView) -> Dict[str, float]:
    """Balanced per-level class weights for one :class:`MulticlassView`.

    Thin wrapper over :func:`compute_class_weights` (same balanced formula,
    unchanged) -- kept as its own named entry point so a multiclass trainer
    reads weights off a ``view`` object the same way
    :func:`compute_binary_class_weights` does for one-vs-rest, rather than
    reaching past the view to the raw row list.
    """
    return compute_class_weights(view.rows)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
@dataclass
class BloomDataset:
    """Full dataset-preparation bundle for one harvested ``labels.jsonl``."""

    rows: List[LabelRow]
    load_result: LoadResult
    split: DatasetSplit
    balance_report: Dict[str, Any]
    class_weights: Dict[str, float]
    one_vs_rest: Dict[str, OneVsRestView]
    multiclass: MulticlassView


def prepare_dataset(
    store_path: str | os.PathLike[str],
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> BloomDataset:
    """Load, de-dupe, split, and summarize one harvested Bloom-label store.

    One call assembles everything a downstream (torch-importing, out of
    scope here) training script needs: the de-duplicated row set, the seeded
    stratified train/val split, the per-level class-balance census,
    multiclass class weights, the six one-vs-rest binary views, and the
    single multiclass view -- all computed over the de-duplicated corpus
    (never the raw file, which may carry a cross-run duplicate
    ``content_sha256`` row).
    """
    load_result = load_labels(store_path)
    split = stratified_split(load_result.rows, val_fraction=val_fraction, seed=seed)
    balance_report = class_balance_report(load_result.rows)
    class_weights = compute_class_weights(load_result.rows)
    views = all_one_vs_rest_views(load_result.rows)
    multiclass = multiclass_view(load_result.rows)
    return BloomDataset(
        rows=load_result.rows,
        load_result=load_result,
        split=split,
        balance_report=balance_report,
        class_weights=class_weights,
        one_vs_rest=views,
        multiclass=multiclass,
    )


__all__ = [
    "BLOOM_LEVELS",
    "BLOOM_LEVEL_TO_ID",
    "ID_TO_BLOOM_LEVEL",
    "DEFAULT_SEED",
    "DEFAULT_VAL_FRACTION",
    "LabelRow",
    "LoadResult",
    "DatasetSplit",
    "OneVsRestView",
    "MulticlassView",
    "BloomDataset",
    "load_labels",
    "class_balance_report",
    "stratified_split",
    "one_vs_rest_view",
    "all_one_vs_rest_views",
    "multiclass_view",
    "compute_class_weights",
    "compute_binary_class_weights",
    "compute_multiclass_class_weights",
    "prepare_dataset",
]
