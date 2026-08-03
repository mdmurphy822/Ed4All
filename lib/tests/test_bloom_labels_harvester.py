"""Tests for the deterministic Bloom-label harvester (lib.bloom_labels)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.bloom_labels import (
    LICENSE_TAG,
    SCHEMA_VERSION,
    harvest_bloom_labels,
)


# ---------------------------------------------------------------------------
# synthetic export fixture — all three artifact kinds
# ---------------------------------------------------------------------------
def _write_objectives(export: Path) -> None:
    # Real synthesized_objectives.json carries a populated flat
    # ``learning_outcomes`` union (the preferred surface) alongside the grouped
    # terminal/chapter views.
    doc = {
        "course_name": "SAMPLE",
        "model": "super-120b-nvfp4",
        "learning_outcomes": [
            {
                "id": "TO-01",
                "statement": "Model real-world scenarios as algebraic equations.",
                "bloom_level": "apply",
            },
            {
                # camelCase bloomLevel tolerance + no-id fallback source_id.
                "statement": "Evaluate competing solution strategies.",
                "bloomLevel": "Evaluate",
            },
            {
                "id": "CO-01",
                "statement": "Identify place value of digits.",
                "bloom_level": "remember",
            },
            {
                # No bloom -> skipped (not a label, not malformed).
                "id": "CO-02",
                "statement": "Some statement without a bloom level.",
            },
            {
                # Non-canonical bloom -> skipped.
                "id": "CO-03",
                "statement": "Statement with a bogus level.",
                "bloom_level": "frobnicate",
            },
        ],
    }
    d = export / "01_learning_objectives"
    d.mkdir(parents=True, exist_ok=True)
    (d / "synthesized_objectives.json").write_text(json.dumps(doc), encoding="utf-8")


def _write_blocks(export: Path) -> None:
    rows = [
        {
            "block_id": "week_01_overview#objective_0",
            "block_type": "objective",
            "bloom_level": "understand",
            "content": "<h2>Overview</h2><p>Understand the number line.</p>",
        },
        {
            # dict content + target_bloom fallback (no bloom_level).
            "block_id": "week_01_content#assessment_1",
            "block_type": "assessment_item",
            "target_bloom": "analyze",
            "content": {"statement": "Compare two graphing approaches."},
        },
        {
            # No bloom of any kind -> skipped.
            "block_id": "week_01_content#example_2",
            "block_type": "example",
            "content": "<p>An example with no bloom metadata.</p>",
        },
    ]
    d = export / "04_rewrite"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "blocks_final.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        # A malformed (non-JSON) line -> counted as malformed, skipped.
        fh.write("{not valid json\n")


def _write_assessments(export: Path) -> None:
    d = export / "06_assessments"
    d.mkdir(parents=True, exist_ok=True)
    # Manifest (no per-item bloom) — present but yields no labels.
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "assessments": [
                    {"file": "quiz_w1.xml", "type": "qti", "title": "Quiz 1"}
                ],
            }
        ),
        encoding="utf-8",
    )
    # Checkpoint sidecar — assessment.to_dict() rows carrying questions[].
    checkpoint = [
        {
            "assessment_id": "A1",
            "provider": "local",
            "questions": [
                {
                    "question_id": "Q1",
                    "stem": "What is the place value of 7 in 4,700?",
                    "bloom_level": "remember",
                },
                {
                    "question_id": "Q2",
                    "question_text": "Analyze why the strategy fails.",
                    "bloom_level": "analyze",
                },
                {
                    # No bloom -> skipped.
                    "question_id": "Q3",
                    "stem": "A stem with no bloom.",
                },
            ],
        }
    ]
    with (d / ".assessments_checkpoint.jsonl").open("w", encoding="utf-8") as fh:
        for row in checkpoint:
            fh.write(json.dumps(row) + "\n")


@pytest.fixture()
def sample_export(tmp_path: Path) -> Path:
    export = tmp_path / "project-zeta"
    _write_objectives(export)
    _write_blocks(export)
    _write_assessments(export)
    return export


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_harvest_all_three_artifact_kinds(sample_export: Path, tmp_path: Path):
    store = tmp_path / "store" / "labels.jsonl"
    result = harvest_bloom_labels(sample_export, store_path=store, run_id="run-1")

    # objectives: TO-01, TO(no-id), CO-01  == 3
    assert result.per_artifact_counts["objective"] == 3
    # blocks: overview(understand), assessment(analyze via target_bloom) == 2
    assert result.per_artifact_counts["block"] == 2
    # assessments: Q1(remember), Q2(analyze) == 2
    assert result.per_artifact_counts["assessment_item"] == 2
    assert result.rows_added == 7
    assert result.duplicates_skipped == 0
    # one malformed JSONL block line was counted.
    assert result.malformed_skipped == 1
    assert result.missing_artifacts == []

    rows = [json.loads(ln) for ln in store.read_text().splitlines()]
    assert len(rows) == 7
    for row in rows:
        assert row["license_tag"] == LICENSE_TAG
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["run_id"] == "run-1"
        assert row["bloom_level"] in {
            "remember", "understand", "apply", "analyze", "evaluate", "create"
        }
        assert row["content_sha256"]
        assert row["harvested_at"]

    # provenance resolved from artifact metadata where present.
    obj_rows = [r for r in rows if r["artifact_kind"] == "objective"]
    assert all(r["model_provenance"] == "super-120b-nvfp4" for r in obj_rows)
    assess_rows = [r for r in rows if r["artifact_kind"] == "assessment_item"]
    assert all(r["model_provenance"] == "local" for r in assess_rows)

    # no-id objective fell back to a synthetic source id.
    to_no_id = [r for r in obj_rows if "Evaluate competing" in r["text"]]
    assert to_no_id and to_no_id[0]["bloom_level"] == "evaluate"


def test_dedupe_across_two_runs(sample_export: Path, tmp_path: Path):
    store = tmp_path / "store" / "labels.jsonl"
    first = harvest_bloom_labels(sample_export, store_path=store, run_id="r1")
    assert first.rows_added == 7

    # Second harvest of the SAME export: everything is a cross-run duplicate.
    second = harvest_bloom_labels(sample_export, store_path=store, run_id="r2")
    assert second.rows_added == 0
    assert second.duplicates_skipped == 7

    # Store did not grow.
    assert len(store.read_text().splitlines()) == 7


def test_dedupe_within_run(tmp_path: Path):
    export = tmp_path / "project-eta"
    d = export / "01_learning_objectives"
    d.mkdir(parents=True)
    # Two objectives files with an identical (text, level) pair.
    same = {"id": "X", "statement": "Identical statement.", "bloom_level": "apply"}
    (d / "synthesized_objectives.json").write_text(json.dumps([same]))
    d2 = export / "sub" / "01_learning_objectives"
    d2.mkdir(parents=True)
    (d2 / "objectives.json").write_text(json.dumps([same]))

    store = tmp_path / "labels.jsonl"
    result = harvest_bloom_labels(export, store_path=store)
    assert result.rows_added == 1
    assert result.duplicates_skipped == 1


def test_missing_artifacts_are_counted_not_raised(tmp_path: Path):
    # Empty export dir -> all three families missing, no exception.
    empty = tmp_path / "empty"
    empty.mkdir()
    store = tmp_path / "labels.jsonl"
    result = harvest_bloom_labels(empty, store_path=store)
    assert result.rows_added == 0
    assert set(result.missing_artifacts) == {
        "objective", "block", "assessment_item"
    }
    assert not store.exists()  # nothing written when there is nothing to add


def test_only_some_artifacts_present(sample_export: Path, tmp_path: Path):
    # Remove the assessments dir -> assessment_item reported missing, others ok.
    import shutil

    shutil.rmtree(sample_export / "06_assessments")
    store = tmp_path / "labels.jsonl"
    result = harvest_bloom_labels(sample_export, store_path=store)
    assert result.missing_artifacts == ["assessment_item"]
    assert result.per_artifact_counts["objective"] == 3
    assert result.per_artifact_counts["block"] == 2


def test_dry_run_writes_nothing(sample_export: Path, tmp_path: Path):
    store = tmp_path / "store" / "labels.jsonl"
    result = harvest_bloom_labels(
        sample_export, store_path=store, dry_run=True, run_id="r1"
    )
    # Counts are still computed...
    assert result.rows_added == 7
    assert result.dry_run is True
    # ...but nothing is written.
    assert not store.exists()

    # A subsequent real run then writes all 7 (dry-run did not poison dedupe).
    real = harvest_bloom_labels(sample_export, store_path=store, run_id="r2")
    assert real.rows_added == 7
    assert store.exists()


def test_model_provenance_override(sample_export: Path, tmp_path: Path):
    store = tmp_path / "labels.jsonl"
    result = harvest_bloom_labels(
        sample_export, store_path=store, model_provenance="forced-model"
    )
    assert result.rows_added == 7
    rows = [json.loads(ln) for ln in store.read_text().splitlines()]
    assert all(r["model_provenance"] == "forced-model" for r in rows)


def test_single_file_input(sample_export: Path, tmp_path: Path):
    # Passing a single objectives file (not a dir) also works.
    obj_file = sample_export / "01_learning_objectives" / "synthesized_objectives.json"
    store = tmp_path / "labels.jsonl"
    result = harvest_bloom_labels(obj_file, store_path=store)
    assert result.per_artifact_counts["objective"] == 3
    assert set(result.missing_artifacts) == {"block", "assessment_item"}


# ---------------------------------------------------------------------------
# CLI verb
# ---------------------------------------------------------------------------
def test_cli_harvest_and_dry_run(sample_export: Path, tmp_path: Path):
    from click.testing import CliRunner

    from cli.commands.harvest_bloom_labels import harvest_bloom_labels_command

    store = tmp_path / "cli_labels.jsonl"
    runner = CliRunner()

    # Dry-run prints a summary but writes nothing.
    dry = runner.invoke(
        harvest_bloom_labels_command,
        [str(sample_export), "--store", str(store), "--dry-run"],
    )
    assert dry.exit_code == 0, dry.output
    assert "[dry-run]" in dry.output
    assert "Harvested 7 new label(s)" in dry.output
    assert not store.exists()

    # Real run writes the store + reports per-artifact counts.
    real = runner.invoke(
        harvest_bloom_labels_command,
        [str(sample_export), "--store", str(store)],
    )
    assert real.exit_code == 0, real.output
    assert "Harvested 7 new label(s)" in real.output
    assert "objective: 3" in real.output
    assert store.exists()
    assert len(store.read_text().splitlines()) == 7
