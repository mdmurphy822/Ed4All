"""Tests for the Bloom-label dataset-preparation module (lib.bloom_labels.dataset).

Every ``labels.jsonl`` path exercised here is a synthetic ``tmp_path``
fixture -- never a real course/campaign path -- per the no-course-data-in-
tracked-tests contract.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from lib.bloom_labels.dataset import (
    BLOOM_LEVEL_TO_ID,
    BLOOM_LEVELS,
    ID_TO_BLOOM_LEVEL,
    BloomDataset,
    DatasetSplit,
    LabelRow,
    LoadResult,
    MulticlassView,
    OneVsRestView,
    all_one_vs_rest_views,
    class_balance_report,
    compute_binary_class_weights,
    compute_class_weights,
    compute_multiclass_class_weights,
    load_labels,
    multiclass_view,
    one_vs_rest_view,
    prepare_dataset,
    stratified_split,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _harvester_row(
    *,
    text: str,
    bloom_level: str,
    content_sha256: str,
    artifact_kind: str = "objective",
    source_id: str = "TO-01",
    **overrides,
) -> Dict[str, object]:
    """One row in the harvester's on-disk shape (see harvester.py:90-106)."""
    row = {
        "text": text,
        "bloom_level": bloom_level,
        "artifact_kind": artifact_kind,
        "source_id": source_id,
        "model_provenance": "super-120b-nvfp4",
        "license_tag": "nvidia-open-model-local",
        "content_sha256": content_sha256,
        "harvested_at": "2026-07-29T00:00:00+00:00",
        "run_id": "WF-test-00000001",
        "schema_version": "1.0",
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _make_row(level: str, idx: int) -> LabelRow:
    return LabelRow(
        text=f"Statement {idx} at level {level}.",
        bloom_level=level,
        artifact_kind="objective",
        content_sha256=f"sha_{level}_{idx}",
        source_id=f"TO-{level}-{idx}",
    )


def _rows_for_counts(counts: Dict[str, int]) -> List[LabelRow]:
    rows: List[LabelRow] = []
    for level, n in counts.items():
        rows.extend(_make_row(level, i) for i in range(n))
    return rows


# ---------------------------------------------------------------------------
# load_labels — reads the harvester row schema from a caller-supplied path
# ---------------------------------------------------------------------------
def test_load_labels_reads_caller_supplied_path(tmp_path):
    store = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            _harvester_row(
                text="Model real-world scenarios as algebraic equations.",
                bloom_level="apply",
                content_sha256="sha-apply-1",
            ),
            _harvester_row(
                text="Identify place value of digits.",
                bloom_level="remember",
                content_sha256="sha-remember-1",
                artifact_kind="block",
                source_id="week_01#objective_0",
            ),
        ],
    )
    result = load_labels(store)
    assert isinstance(result, LoadResult)
    assert len(result.rows) == 2
    assert result.malformed_skipped == 0
    assert result.duplicates_skipped == 0
    levels = {row.bloom_level for row in result.rows}
    assert levels == {"apply", "remember"}
    block_row = next(r for r in result.rows if r.artifact_kind == "block")
    assert block_row.source_id == "week_01#objective_0"


def test_load_labels_missing_file_returns_empty(tmp_path):
    result = load_labels(tmp_path / "does_not_exist.jsonl")
    assert result.rows == []
    assert result.as_dict() == {
        "row_count": 0,
        "total_lines": 0,
        "malformed_skipped": 0,
        "duplicates_skipped": 0,
    }


def test_load_labels_dedupes_by_content_sha256(tmp_path):
    # Same content_sha256, different surface text — dedupe is on the sha,
    # first occurrence wins (mirrors the harvester's own dedupe contract).
    store = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            _harvester_row(
                text="First seen text.",
                bloom_level="analyze",
                content_sha256="dup-sha",
                source_id="first",
            ),
            _harvester_row(
                text="A later duplicate row.",
                bloom_level="analyze",
                content_sha256="dup-sha",
                source_id="second",
            ),
        ],
    )
    result = load_labels(store)
    assert len(result.rows) == 1
    assert result.duplicates_skipped == 1
    assert result.rows[0].source_id == "first"


def test_load_labels_skips_malformed_rows(tmp_path):
    lines = [
        "not json at all {{{",
        json.dumps(["a", "bare", "list"]),  # not a dict
        json.dumps(_harvester_row(text="", bloom_level="apply", content_sha256="s1")),
        json.dumps(
            {"bloom_level": "apply", "content_sha256": "s2"}
        ),  # missing text
        json.dumps(
            _harvester_row(text="No level here.", bloom_level="frobnicate", content_sha256="s3")
        ),
        json.dumps({"text": "No sha.", "bloom_level": "apply"}),  # missing sha
        json.dumps(
            _harvester_row(text="Valid row.", bloom_level="create", content_sha256="s4")
        ),
    ]
    store = tmp_path / "labels.jsonl"
    store.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = load_labels(store)
    assert len(result.rows) == 1
    assert result.rows[0].bloom_level == "create"
    assert result.malformed_skipped == 6
    assert result.total_lines == 7


def test_load_labels_normalizes_bloom_level_case(tmp_path):
    store = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            _harvester_row(
                text="Evaluate competing solution strategies.",
                bloom_level="Evaluate",
                content_sha256="sha-mixed-case",
            )
        ],
    )
    result = load_labels(store)
    assert result.rows[0].bloom_level == "evaluate"


# ---------------------------------------------------------------------------
# class_balance_report — the operator-facing corpus census
# ---------------------------------------------------------------------------
def test_class_balance_report_includes_all_levels_with_zero_when_absent():
    rows = _rows_for_counts({"apply": 3, "remember": 2})
    report = class_balance_report(rows)
    assert report["total"] == 5
    assert set(report["counts"]) == set(BLOOM_LEVELS)
    assert report["counts"]["apply"] == 3
    assert report["counts"]["remember"] == 2
    for level in ("understand", "analyze", "evaluate", "create"):
        assert report["counts"][level] == 0
    assert set(report["missing_levels"]) == {"understand", "analyze", "evaluate", "create"}


def test_class_balance_report_flags_thin_levels_like_live_corpus():
    # Synthetic distribution with the shape a real harvest shows — dense
    # low-Bloom levels, thin create/evaluate tail (the property the report
    # must flag); the exact values are arbitrary.
    counts = {
        "understand": 93,
        "apply": 79,
        "remember": 59,
        "analyze": 35,
        "evaluate": 14,
        "create": 8,
    }
    rows = _rows_for_counts(counts)
    report = class_balance_report(rows)
    assert report["total"] == sum(counts.values())
    assert set(report["thin_levels"]) == {"evaluate", "create"}
    assert "analyze" not in report["thin_levels"]
    assert report["min_level"] == "create"
    assert report["max_level"] == "understand"
    assert report["imbalance_ratio"] == pytest.approx(93 / 8)


def test_class_balance_report_empty_corpus():
    report = class_balance_report([])
    assert report["total"] == 0
    assert all(c == 0 for c in report["counts"].values())
    assert report["thin_levels"] == []
    assert report["min_level"] is None
    assert report["max_level"] is None
    assert report["imbalance_ratio"] is None


# ---------------------------------------------------------------------------
# stratified_split — seeded per-level train/val split
# ---------------------------------------------------------------------------
def test_stratified_split_is_deterministic_for_a_fixed_seed():
    rows = _rows_for_counts({"apply": 20, "remember": 20, "create": 3})
    split_a = stratified_split(rows, val_fraction=0.2, seed=42)
    split_b = stratified_split(rows, val_fraction=0.2, seed=42)
    assert [r.content_sha256 for r in split_a.train] == [r.content_sha256 for r in split_b.train]
    assert [r.content_sha256 for r in split_a.val] == [r.content_sha256 for r in split_b.val]


def test_stratified_split_different_seeds_diverge():
    rows = _rows_for_counts({"apply": 30, "remember": 30})
    split_a = stratified_split(rows, val_fraction=0.3, seed=1)
    split_b = stratified_split(rows, val_fraction=0.3, seed=2)
    val_a = [r.content_sha256 for r in split_a.val]
    val_b = [r.content_sha256 for r in split_b.val]
    assert val_a != val_b


def test_stratified_split_covers_every_row_exactly_once():
    counts = {"understand": 15, "apply": 10, "remember": 7, "analyze": 5, "evaluate": 3, "create": 2}
    rows = _rows_for_counts(counts)
    split = stratified_split(rows, val_fraction=0.25, seed=7)
    assert isinstance(split, DatasetSplit)
    train_shas = {r.content_sha256 for r in split.train}
    val_shas = {r.content_sha256 for r in split.val}
    assert train_shas.isdisjoint(val_shas)
    assert train_shas | val_shas == {r.content_sha256 for r in rows}
    assert len(split.train) + len(split.val) == len(rows)


def test_stratified_split_is_per_level_stratified():
    # Every level with enough rows must contribute to BOTH train and val —
    # a pooled (non-stratified) random split could easily drop the rarest
    # level from val entirely.
    counts = {"understand": 40, "apply": 40, "remember": 40, "analyze": 40, "evaluate": 10, "create": 10}
    rows = _rows_for_counts(counts)
    split = stratified_split(rows, val_fraction=0.2, seed=42)
    train_levels = {r.bloom_level for r in split.train}
    val_levels = {r.bloom_level for r in split.val}
    assert train_levels == set(counts)
    assert val_levels == set(counts)


def test_stratified_split_thin_level_keeps_at_least_one_train_row():
    rows = _rows_for_counts({"create": 2})
    split = stratified_split(rows, val_fraction=0.5, seed=42)
    assert len(split.train) == 1
    assert len(split.val) == 1


def test_stratified_split_single_row_level_goes_entirely_to_train():
    rows = _rows_for_counts({"create": 1})
    split = stratified_split(rows, val_fraction=0.5, seed=42)
    assert len(split.train) == 1
    assert len(split.val) == 0


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_stratified_split_rejects_out_of_range_val_fraction(bad_fraction):
    rows = _rows_for_counts({"apply": 5})
    with pytest.raises(ValueError):
        stratified_split(rows, val_fraction=bad_fraction, seed=42)


# ---------------------------------------------------------------------------
# one-vs-rest binary views
# ---------------------------------------------------------------------------
def test_one_vs_rest_view_splits_positive_and_negative():
    rows = _rows_for_counts({"apply": 4, "remember": 3, "create": 1})
    view = one_vs_rest_view(rows, "apply")
    assert isinstance(view, OneVsRestView)
    assert view.n_positive == 4
    assert view.n_negative == 4
    assert all(r.bloom_level == "apply" for r in view.positive)
    assert all(r.bloom_level != "apply" for r in view.negative)


def test_one_vs_rest_view_is_case_and_whitespace_tolerant():
    rows = _rows_for_counts({"apply": 2})
    view = one_vs_rest_view(rows, "  Apply  ")
    assert view.level == "apply"
    assert view.n_positive == 2


def test_one_vs_rest_view_unknown_level_raises():
    with pytest.raises(ValueError):
        one_vs_rest_view([], "frobnicate")


def test_all_one_vs_rest_views_covers_every_canonical_level():
    rows = _rows_for_counts({"apply": 2, "remember": 3})
    views = all_one_vs_rest_views(rows)
    assert set(views) == set(BLOOM_LEVELS)
    assert views["apply"].n_positive == 2
    assert views["apply"].n_negative == 3
    assert views["create"].n_positive == 0
    assert views["create"].n_negative == 5


# ---------------------------------------------------------------------------
# BLOOM_LEVEL_TO_ID / ID_TO_BLOOM_LEVEL -- canonical multiclass label ids
# ---------------------------------------------------------------------------
def test_bloom_level_to_id_covers_all_six_levels_in_order():
    assert list(BLOOM_LEVEL_TO_ID.keys()) == list(BLOOM_LEVELS)
    assert BLOOM_LEVEL_TO_ID == {level: i for i, level in enumerate(BLOOM_LEVELS)}


def test_id_to_bloom_level_is_the_inverse_mapping():
    assert ID_TO_BLOOM_LEVEL == {i: level for level, i in BLOOM_LEVEL_TO_ID.items()}
    for level in BLOOM_LEVELS:
        assert ID_TO_BLOOM_LEVEL[BLOOM_LEVEL_TO_ID[level]] == level


# ---------------------------------------------------------------------------
# multiclass view -- all six levels as a single labeled dataset
# ---------------------------------------------------------------------------
def test_multiclass_view_wraps_all_rows_unsplit():
    rows = _rows_for_counts({"apply": 4, "remember": 3, "create": 1})
    view = multiclass_view(rows)
    assert isinstance(view, MulticlassView)
    assert len(view.rows) == 8
    assert {r.content_sha256 for r in view.rows} == {r.content_sha256 for r in rows}


def test_multiclass_view_label_counts_matches_class_balance_shape():
    counts = {"understand": 9, "apply": 7, "remember": 5, "analyze": 3, "evaluate": 1, "create": 1}
    rows = _rows_for_counts(counts)
    view = multiclass_view(rows)
    label_counts = view.label_counts
    assert set(label_counts) == set(BLOOM_LEVELS)
    for level, n in counts.items():
        assert label_counts[level] == n


def test_multiclass_view_empty_rows():
    view = multiclass_view([])
    assert view.rows == []
    assert all(c == 0 for c in view.label_counts.values())


def test_compute_multiclass_class_weights_matches_compute_class_weights():
    rows = _rows_for_counts({"apply": 10, "remember": 5, "create": 2})
    view = multiclass_view(rows)
    assert compute_multiclass_class_weights(view) == compute_class_weights(rows)


def test_compute_multiclass_class_weights_empty_view_returns_all_zero():
    weights = compute_multiclass_class_weights(MulticlassView(rows=[]))
    assert set(weights) == set(BLOOM_LEVELS)
    assert all(w == 0.0 for w in weights.values())


# ---------------------------------------------------------------------------
# class-weight computation
# ---------------------------------------------------------------------------
def test_compute_class_weights_balanced_formula():
    rows = _rows_for_counts({"apply": 10, "remember": 5})
    weights = compute_class_weights(rows)
    # total=15, n_present=2 -> weight(level) = 15 / (2 * count(level))
    assert weights["apply"] == pytest.approx(15 / (2 * 10))
    assert weights["remember"] == pytest.approx(15 / (2 * 5))
    for level in ("understand", "analyze", "evaluate", "create"):
        assert weights[level] == 0.0


def test_compute_class_weights_no_rows_returns_all_zero():
    weights = compute_class_weights([])
    assert set(weights) == set(BLOOM_LEVELS)
    assert all(w == 0.0 for w in weights.values())


def test_compute_binary_class_weights_balanced_formula():
    rows = _rows_for_counts({"create": 2, "apply": 8})
    view = one_vs_rest_view(rows, "create")
    weights = compute_binary_class_weights(view)
    total = 10
    assert weights["positive"] == pytest.approx(total / (2 * 2))
    assert weights["negative"] == pytest.approx(total / (2 * 8))


def test_compute_binary_class_weights_empty_view_returns_zero():
    empty_view = OneVsRestView(level="create", positive=[], negative=[])
    weights = compute_binary_class_weights(empty_view)
    assert weights == {"positive": 0.0, "negative": 0.0}


# ---------------------------------------------------------------------------
# prepare_dataset — the end-to-end orchestrator
# ---------------------------------------------------------------------------
def test_prepare_dataset_end_to_end(tmp_path):
    rows = []
    for level, n in {
        "understand": 12,
        "apply": 10,
        "remember": 8,
        "analyze": 6,
        "evaluate": 3,
        "create": 2,
    }.items():
        for i in range(n):
            rows.append(
                _harvester_row(
                    text=f"{level} statement number {i}.",
                    bloom_level=level,
                    content_sha256=f"sha-{level}-{i}",
                )
            )
    # One cross-run duplicate that must not double-count.
    rows.append(
        _harvester_row(
            text="understand statement number 0.",
            bloom_level="understand",
            content_sha256="sha-understand-0",
        )
    )
    store = _write_jsonl(tmp_path / "labels.jsonl", rows)

    bundle = prepare_dataset(store, val_fraction=0.2, seed=7)
    assert isinstance(bundle, BloomDataset)
    assert bundle.load_result.duplicates_skipped == 1
    assert len(bundle.rows) == 41
    assert len(bundle.split.train) + len(bundle.split.val) == 41
    assert bundle.balance_report["total"] == 41
    assert set(bundle.balance_report["thin_levels"]) >= {"create"}
    assert set(bundle.class_weights) == set(BLOOM_LEVELS)
    assert set(bundle.one_vs_rest) == set(BLOOM_LEVELS)
    assert bundle.one_vs_rest["apply"].n_positive == 10
    assert isinstance(bundle.multiclass, MulticlassView)
    assert len(bundle.multiclass.rows) == 41
    assert bundle.multiclass.label_counts["apply"] == 10


def test_prepare_dataset_is_reproducible_across_calls(tmp_path):
    rows = [
        _harvester_row(
            text=f"apply statement {i}.", bloom_level="apply", content_sha256=f"sha-apply-{i}"
        )
        for i in range(20)
    ] + [
        _harvester_row(
            text=f"create statement {i}.", bloom_level="create", content_sha256=f"sha-create-{i}"
        )
        for i in range(5)
    ]
    store = _write_jsonl(tmp_path / "labels.jsonl", rows)

    bundle_a = prepare_dataset(store, val_fraction=0.25, seed=99)
    bundle_b = prepare_dataset(store, val_fraction=0.25, seed=99)
    assert [r.content_sha256 for r in bundle_a.split.val] == [
        r.content_sha256 for r in bundle_b.split.val
    ]
    assert bundle_a.class_weights == bundle_b.class_weights


# ---------------------------------------------------------------------------
# import-lightness — no torch/transformers at module top level
# ---------------------------------------------------------------------------
def test_importable_without_training_deps(monkeypatch):
    """dataset.py must import cleanly with torch/transformers UNAVAILABLE.

    Setting ``sys.modules["torch"] = None`` is the standard idiom to make any
    ``import torch`` raise ``ImportError`` — so re-executing the module body
    under that condition proves no top-level import of the heavy [training]
    extras.
    """
    for name in ("torch", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "lib.bloom_labels.dataset", raising=False)
    reloaded = importlib.import_module("lib.bloom_labels.dataset")
    try:
        assert reloaded.BLOOM_LEVELS == BLOOM_LEVELS
    finally:
        # Restore the real module object for any test collected after this
        # one in the same session.
        monkeypatch.delitem(sys.modules, "lib.bloom_labels.dataset", raising=False)
        importlib.import_module("lib.bloom_labels.dataset")
