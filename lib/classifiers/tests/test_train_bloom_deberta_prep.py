"""Bloom-ladder initiative WI-13 -- ``train_bloom_deberta`` prep-path tests.

Covers everything reachable WITHOUT executing a real training run: the
module stays importable on CPU without the ``[training]`` extra, argparse
defaults/choices, the ``--output-dir`` default (the exact
``<heads-dir>/<level>`` shape the WI-05 registry expects), the
``HF_HUB_OFFLINE`` hub-id rejection contract, and the pure-stdlib
one-vs-rest example builder. No real model weights are loaded and
``main()`` is never invoked end-to-end -- the trainer script must NEVER be
executed in this suite (owner directive); only its preparation surface is
exercised.

Every ``labels.jsonl`` path here is a synthetic ``tmp_path`` fixture, never
a real course/campaign path, per the no-course-data-in-tracked-tests
contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Repo root on path so the imports work from a fresh checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.bloom_labels.dataset import (  # noqa: E402
    BLOOM_LEVEL_TO_ID,
    BLOOM_LEVELS,
    multiclass_view,
    one_vs_rest_view,
    prepare_dataset,
)
from lib.classifiers.training import train_bloom_deberta as trainer_mod  # noqa: E402


# --------------------------------------------------------------------- #
# import-lightness: no torch/transformers/datasets/sklearn at module level
# --------------------------------------------------------------------- #


def test_module_imports_without_training_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-importing the module must not touch any [training]-extra package.

    Sabotages import of the four heavy packages the module docstring
    promises stay deferred inside ``main()``; a fresh, forced re-import
    that raised here would prove one of them leaked to module scope.
    """
    heavy = {"torch", "transformers", "datasets", "sklearn"}
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _failing_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        top = name.split(".")[0]
        if top in heavy:
            raise AssertionError(
                f"train_bloom_deberta imported {name!r} at module scope -- "
                f"heavy [training]-extra imports must be deferred inside main()."
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)
    monkeypatch.delitem(sys.modules, "lib.classifiers.training.train_bloom_deberta", raising=False)
    import importlib

    reloaded = importlib.import_module("lib.classifiers.training.train_bloom_deberta")
    assert reloaded is not None
    # Restore the normal module object for any later test in this process.
    monkeypatch.delitem(sys.modules, "lib.classifiers.training.train_bloom_deberta", raising=False)
    importlib.import_module("lib.classifiers.training.train_bloom_deberta")


def test_labels_and_id_maps_are_binary() -> None:
    assert trainer_mod.LABELS == ["negative", "positive"]
    assert trainer_mod.LABEL2ID == {"negative": 0, "positive": 1}
    assert trainer_mod.ID2LABEL == {0: "negative", 1: "positive"}


# --------------------------------------------------------------------- #
# default_output_dir -- the WI-05 registry shape
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("level", list(BLOOM_LEVELS))
def test_default_output_dir_matches_registry_shape(level: str) -> None:
    result = trainer_mod.default_output_dir(level)
    assert result == Path("models/bloom_classifiers") / level
    # The registry appends "final" itself (BloomDebertaHeads.default_registry);
    # this function must NOT double up on that suffix.
    assert result.name == level


def test_default_multiclass_output_dir_matches_registry_shape() -> None:
    result = trainer_mod.default_multiclass_output_dir()
    assert result == Path("models/bloom_classifiers") / "multiclass"
    assert result.name == "multiclass"


# --------------------------------------------------------------------- #
# validate_base_model_path -- HF_HUB_OFFLINE hub-id rejection
# --------------------------------------------------------------------- #


def test_validate_base_model_offline_unset_allows_hub_id() -> None:
    # No HF_HUB_OFFLINE at all -> a bare hub id is fine (network path).
    # Passed as an explicit falsy sentinel rather than relying on the
    # live-env read (this sandbox exports HF_HUB_OFFLINE=1 machine-wide
    # per the "no models phoning home" posture) -- the live-env-read
    # path itself is covered separately by
    # test_validate_base_model_default_reads_live_env.
    trainer_mod.validate_base_model_path(
        "microsoft/deberta-v3-base", hf_hub_offline="",
    )


@pytest.mark.parametrize("falsy", ["0", "false", "no", "off", "", "   "])
def test_validate_base_model_offline_falsy_allows_hub_id(falsy: str) -> None:
    trainer_mod.validate_base_model_path(
        "microsoft/deberta-v3-base", hf_hub_offline=falsy,
    )


@pytest.mark.parametrize("truthy", ["1", "true", "True", "yes", "on"])
def test_validate_base_model_offline_truthy_rejects_hub_id(truthy: str) -> None:
    with pytest.raises(ValueError, match="HF_HUB_OFFLINE"):
        trainer_mod.validate_base_model_path(
            "microsoft/deberta-v3-base", hf_hub_offline=truthy,
        )


def test_validate_base_model_offline_truthy_accepts_local_dir(tmp_path: Path) -> None:
    local_dir = tmp_path / "deberta-v3-base"
    local_dir.mkdir()
    # Must not raise -- a real local directory satisfies the offline contract.
    trainer_mod.validate_base_model_path(str(local_dir), hf_hub_offline="1")


def test_validate_base_model_offline_truthy_rejects_missing_local_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="not a local directory"):
        trainer_mod.validate_base_model_path(str(missing), hf_hub_offline="1")


def test_validate_base_model_default_reads_live_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """With no explicit hf_hub_offline arg, the live env is read at call time."""
    monkeypatch.setenv(trainer_mod.HF_HUB_OFFLINE_ENV, "1")
    with pytest.raises(ValueError):
        trainer_mod.validate_base_model_path("microsoft/deberta-v3-base")

    monkeypatch.delenv(trainer_mod.HF_HUB_OFFLINE_ENV, raising=False)
    trainer_mod.validate_base_model_path("microsoft/deberta-v3-base")  # no raise


# --------------------------------------------------------------------- #
# build_binary_examples -- pure-stdlib one-vs-rest example construction
# --------------------------------------------------------------------- #


def _write_labels_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            row.setdefault("content_sha256", f"sha-{i:04d}")
            fh.write(json.dumps(row) + "\n")


def _row(text: str, level: str, idx: int) -> Dict[str, object]:
    return {
        "text": text,
        "bloom_level": level,
        "artifact_kind": "objective",
        "content_sha256": f"sha-{level}-{idx:04d}",
    }


def test_build_binary_examples_labels_positive_and_negative(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    rows = (
        [_row(f"apply text {i}", "apply", i) for i in range(4)]
        + [_row(f"remember text {i}", "remember", i) for i in range(3)]
        + [_row(f"create text {i}", "create", i) for i in range(2)]
    )
    _write_labels_jsonl(labels_path, rows)

    dataset = prepare_dataset(labels_path, val_fraction=0.5, seed=7)
    train_view = one_vs_rest_view(dataset.split.train, "apply")

    examples = trainer_mod.build_binary_examples(train_view)
    assert len(examples) == len(train_view.positive) + len(train_view.negative)

    positive_texts = {row.text for row in train_view.positive}
    negative_texts = {row.text for row in train_view.negative}
    for ex in examples:
        if ex["label"] == trainer_mod.LABEL2ID["positive"]:
            assert ex["text"] in positive_texts
        else:
            assert ex["label"] == trainer_mod.LABEL2ID["negative"]
            assert ex["text"] in negative_texts

    n_positive = sum(1 for ex in examples if ex["label"] == 1)
    n_negative = sum(1 for ex in examples if ex["label"] == 0)
    assert n_positive == train_view.n_positive
    assert n_negative == train_view.n_negative


def test_build_binary_examples_empty_view_returns_empty_list(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    _write_labels_jsonl(labels_path, [])
    dataset = prepare_dataset(labels_path)
    view = one_vs_rest_view(dataset.split.train, "apply")
    assert trainer_mod.build_binary_examples(view) == []


# --------------------------------------------------------------------- #
# build_multiclass_examples -- pure-stdlib six-way example construction
# --------------------------------------------------------------------- #


def test_build_multiclass_examples_labels_by_canonical_id(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    rows = (
        [_row(f"apply text {i}", "apply", i) for i in range(4)]
        + [_row(f"remember text {i}", "remember", i) for i in range(3)]
        + [_row(f"create text {i}", "create", i) for i in range(2)]
    )
    _write_labels_jsonl(labels_path, rows)

    dataset = prepare_dataset(labels_path, val_fraction=0.5, seed=7)
    train_view = multiclass_view(dataset.split.train)

    examples = trainer_mod.build_multiclass_examples(train_view)
    assert len(examples) == len(train_view.rows)

    by_sha = {row.text: row.bloom_level for row in train_view.rows}
    for ex in examples:
        expected_level = by_sha[ex["text"]]
        assert ex["label"] == BLOOM_LEVEL_TO_ID[expected_level]

    # Every emitted label is one of the six canonical ids, never a
    # positive/negative binary label.
    assert set(ex["label"] for ex in examples) <= set(BLOOM_LEVEL_TO_ID.values())


def test_build_multiclass_examples_empty_view_returns_empty_list(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    _write_labels_jsonl(labels_path, [])
    dataset = prepare_dataset(labels_path)
    view = multiclass_view(dataset.split.train)
    assert trainer_mod.build_multiclass_examples(view) == []


# --------------------------------------------------------------------- #
# _parse_args -- CLI surface (defaults + required/choices validation)
# --------------------------------------------------------------------- #


def test_parse_args_defaults() -> None:
    args = trainer_mod._parse_args(["--bloom-level", "create"])
    assert args.labels_path == Path("runtime/state/bloom_labels/labels.jsonl")
    assert args.head == "one-vs-rest"
    assert args.bloom_level == "create"
    assert args.base_model == trainer_mod.DEFAULT_BASE_MODEL
    assert args.output_dir is None
    assert args.seed == 42
    assert args.epochs == 3
    assert args.lr == 5e-5
    assert args.batch_size == 16
    assert args.max_length == 256


def test_parse_args_bloom_level_required() -> None:
    with pytest.raises(SystemExit):
        trainer_mod._parse_args([])


def test_parse_args_bloom_level_rejects_unknown_choice() -> None:
    with pytest.raises(SystemExit):
        trainer_mod._parse_args(["--bloom-level", "not-a-level"])


@pytest.mark.parametrize("level", list(BLOOM_LEVELS))
def test_parse_args_accepts_every_canonical_level(level: str) -> None:
    args = trainer_mod._parse_args(["--bloom-level", level])
    assert args.bloom_level == level


# --------------------------------------------------------------------- #
# --head -- multiclass mode + --bloom-level mutual-exclusion contract
# --------------------------------------------------------------------- #


def test_parse_args_head_defaults_to_one_vs_rest() -> None:
    args = trainer_mod._parse_args(["--bloom-level", "apply"])
    assert args.head == "one-vs-rest"


def test_parse_args_head_multiclass_with_no_bloom_level() -> None:
    args = trainer_mod._parse_args(["--head", "multiclass"])
    assert args.head == "multiclass"
    assert args.bloom_level is None


def test_parse_args_head_multiclass_rejects_bloom_level() -> None:
    with pytest.raises(SystemExit):
        trainer_mod._parse_args(["--head", "multiclass", "--bloom-level", "create"])


def test_parse_args_head_one_vs_rest_still_requires_bloom_level() -> None:
    with pytest.raises(SystemExit):
        trainer_mod._parse_args(["--head", "one-vs-rest"])


def test_parse_args_head_rejects_unknown_choice() -> None:
    with pytest.raises(SystemExit):
        trainer_mod._parse_args(["--head", "not-a-head", "--bloom-level", "apply"])


def test_parse_args_head_multiclass_output_dir_override(tmp_path: Path) -> None:
    out_dir = tmp_path / "custom_multiclass_out"
    args = trainer_mod._parse_args([
        "--head", "multiclass",
        "--output-dir", str(out_dir),
    ])
    assert args.head == "multiclass"
    assert args.bloom_level is None
    assert args.output_dir == out_dir


def test_parse_args_overrides(tmp_path: Path) -> None:
    labels_path = tmp_path / "custom_labels.jsonl"
    out_dir = tmp_path / "custom_out"
    args = trainer_mod._parse_args([
        "--bloom-level", "evaluate",
        "--labels-path", str(labels_path),
        "--base-model", "models/base/custom-model",
        "--output-dir", str(out_dir),
        "--seed", "7",
        "--epochs", "5",
        "--lr", "1e-5",
        "--batch-size", "8",
        "--max-length", "128",
    ])
    assert args.labels_path == labels_path
    assert args.bloom_level == "evaluate"
    assert args.base_model == "models/base/custom-model"
    assert args.output_dir == out_dir
    assert args.seed == 7
    assert args.epochs == 5
    assert args.lr == 1e-5
    assert args.batch_size == 8
    assert args.max_length == 128


# --------------------------------------------------------------------- #
# main() must never actually train -- fail loudly on empty splits before
# any heavy import is attempted (proves the guard fires ahead of the
# deferred-import block, without needing to stub the model at all).
# --------------------------------------------------------------------- #


def test_main_raises_before_heavy_import_when_no_training_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels_path = tmp_path / "labels.jsonl"
    _write_labels_jsonl(labels_path, [])  # nothing harvested at all

    heavy = {"torch", "transformers", "datasets", "sklearn"}
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _failing_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.split(".")[0] in heavy:
            raise AssertionError(
                f"main() imported {name!r} despite an empty training split "
                f"-- the empty-split guard must fire before any heavy import."
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)
    monkeypatch.delenv(trainer_mod.HF_HUB_OFFLINE_ENV, raising=False)

    with pytest.raises(RuntimeError, match="no training examples"):
        trainer_mod.main([
            "--bloom-level", "create",
            "--labels-path", str(labels_path),
            "--output-dir", str(tmp_path / "out"),
        ])


def test_main_multiclass_raises_before_heavy_import_when_no_training_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels_path = tmp_path / "labels.jsonl"
    _write_labels_jsonl(labels_path, [])  # nothing harvested at all

    heavy = {"torch", "transformers", "datasets", "sklearn"}
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _failing_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.split(".")[0] in heavy:
            raise AssertionError(
                f"main() imported {name!r} despite an empty training split "
                f"-- the empty-split guard must fire before any heavy import."
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)
    monkeypatch.delenv(trainer_mod.HF_HUB_OFFLINE_ENV, raising=False)

    with pytest.raises(RuntimeError, match="no training examples"):
        trainer_mod.main([
            "--head", "multiclass",
            "--labels-path", str(labels_path),
            "--output-dir", str(tmp_path / "out"),
        ])
