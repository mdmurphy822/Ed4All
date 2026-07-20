"""SFT-D S8 — memorization-probe held-out assessment-item slice contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.training.memorization_probe import (  # noqa: E402
    HOLDOUT_REL_PATH,
    evaluate_memorization,
    load_holdout_exclusion,
    select_holdout_item_ids,
    write_holdout_exclusion,
)


def test_selection_deterministic():
    ids = [f"item_{i}" for i in range(100)]
    a = select_holdout_item_ids(ids, fraction=0.1, seed=42)
    b = select_holdout_item_ids(ids, fraction=0.1, seed=42)
    assert a == b
    assert len(a) == 10
    assert set(a).issubset(set(ids))


def test_selection_seed_changes_slice():
    ids = [f"item_{i}" for i in range(100)]
    a = select_holdout_item_ids(ids, fraction=0.1, seed=1)
    b = select_holdout_item_ids(ids, fraction=0.1, seed=2)
    assert a != b


def test_selection_never_starves_training():
    ids = [f"item_{i}" for i in range(6)]
    # fraction 0.5 -> 3, but must leave >= 5 for training -> withhold <= 1.
    held = select_holdout_item_ids(ids, fraction=0.5, seed=42)
    assert len(held) <= 1
    assert len(ids) - len(held) >= 5


def test_selection_tiny_corpus_withholds_nothing():
    assert select_holdout_item_ids(["a", "b", "c"], fraction=0.5) == []


def test_write_load_roundtrip(tmp_path):
    course = tmp_path / "course"
    (course / "training_specs").mkdir(parents=True)
    ids = ["item_3", "item_1", "item_2"]
    path = write_holdout_exclusion(course, ids, fraction=0.1, seed=7)
    assert path == course / HOLDOUT_REL_PATH
    loaded = load_holdout_exclusion(course)
    assert loaded == {"item_1", "item_2", "item_3"}


def test_load_absent_returns_empty(tmp_path):
    assert load_holdout_exclusion(tmp_path) == set()


def test_evaluate_memorization_gap():
    out = evaluate_memorization(accuracy_held_out=0.6, accuracy_trained=0.9)
    assert out["memorization_gap"] == pytest.approx(0.3)


def test_evaluate_memorization_missing_input():
    out = evaluate_memorization(accuracy_held_out=None, accuracy_trained=0.9)
    assert out["memorization_gap"] is None
