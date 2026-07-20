"""Integration wiring test for the SFT data program (S1/S3/S4/S5) inside
``run_synthesis``.

Exercises the end-to-end hoist + reduction + decontam path with the mock
provider (fully offline, no LLM):

  * ``with_assessment_sft`` (and its ED4ALL_WITH_ASSESSMENT_SFT env mirror)
    appends assessment->SFT pairs to instruction_pairs.jsonl;
  * ``with_graph_sft`` appends concept-graph->SFT pairs over the
    holdout-REDUCED concept graph;
  * the holdout split reduces the graph (holdout_edges_excluded > 0);
  * the S3 decontamination gate runs and stamps decontam_checked=True on
    every surviving instruction pair.

Deterministic — no network, no model, no course slugs.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import run_synthesis  # noqa: E402

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "mini_course_training"


def _make_working_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "mini_course_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
    ):
        if stale.exists():
            stale.unlink()
    return dst


def _write_assessments(course: Path) -> None:
    (course / "training_specs").mkdir(parents=True, exist_ok=True)
    doc = {
        "assessment_id": "a1",
        "questions": [
            {
                "question_id": "q-001",
                "question_type": "short_answer",
                "objective_id": "TO-01",
                "stem": "Explain how cognitive load affects working memory.",
                "bloom_level": "understand",
                "correct_answer": "It limits how much a learner can process at once.",
                "feedback": "<ol><li>Working memory is finite.</li><li>Overload degrades retention.</li></ol>",
                "source_chunk_ids": ["chunk_mc_01"],
            },
            {
                "question_id": "q-002",
                "question_type": "multiple_choice",
                "objective_id": "TO-01",
                "stem": "Which design principle reduces extraneous load?",
                "bloom_level": "apply",
                "choices": [
                    {"text": "Segmenting content", "is_correct": True},
                    {"text": "Adding decorative images", "is_correct": False,
                     "misconception_note": "Decorative media increases extraneous load."},
                ],
            },
        ],
    }
    (course / "training_specs" / "assessments.json").write_text(
        json.dumps(doc), encoding="utf-8",
    )


def _write_concept_graph(course: Path) -> None:
    (course / "graph").mkdir(parents=True, exist_ok=True)
    graph = {
        "kind": "concept_semantic",
        "nodes": [
            {"id": "cognitive-load", "label": "Cognitive Load", "occurrences": ["chunk_mc_01"]},
            {"id": "working-memory", "label": "Working Memory", "occurrences": ["chunk_mc_01"]},
            {"id": "retention", "label": "Retention", "occurrences": []},
        ],
        "edges": [
            {"source": "cognitive-load", "target": "working-memory", "type": "related-to",
             "edge_status": "supported"},
            {"source": "retention", "target": "cognitive-load", "type": "prerequisite",
             "edge_status": "confirmed"},
        ],
    }
    (course / "graph" / "concept_graph_semantic.json").write_text(
        json.dumps(graph), encoding="utf-8",
    )


def _write_holdout(course: Path) -> None:
    (course / "eval").mkdir(parents=True, exist_ok=True)
    (course / "eval" / "holdout_split.json").write_text(
        json.dumps({"withheld_edges": [
            {"source": "cognitive-load", "target": "working-memory", "relation_type": "related_to"},
        ]}),
        encoding="utf-8",
    )


def _write_gold(course: Path) -> None:
    (course / "retrieval_eval").mkdir(parents=True, exist_ok=True)
    (course / "retrieval_eval" / "gold_set.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "questions": [
                {"question_id": "gq-mini-0001",
                 "question_text": "What does cognitive load theory describe about memory?"},
            ],
        }),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ED4ALL_WITH_ASSESSMENT_SFT", raising=False)
    monkeypatch.delenv("ED4ALL_WITH_GRAPH_SFT", raising=False)


def _instruction_pairs(course: Path):
    p = course / "training_specs" / "instruction_pairs.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_full_sft_program_wiring(tmp_path):
    course = _make_working_copy(tmp_path)
    _write_assessments(course)
    _write_concept_graph(course)
    _write_holdout(course)
    _write_gold(course)

    stats = run_synthesis(
        corpus_dir=course,
        course_code="MINI_TRAINING_101",
        provider="mock",
        with_assessment_sft=True,
        with_graph_sft=True,
    )

    pairs = _instruction_pairs(course)
    templates = {p.get("template_id", "") for p in pairs}
    assert any(t.startswith("assessment_sft.") for t in templates)
    assert any(t.startswith("graph_sft.") for t in templates)

    assert stats.assessment_sft_pairs_emitted > 0
    assert stats.graph_sft_pairs_emitted > 0
    # S4: the withheld cognitive-load<->working-memory edge was excluded.
    assert stats.holdout_edges_excluded >= 1
    # No graph_sft relation_qa pair may verbalize the withheld a<->b relation.
    for p in pairs:
        if p.get("template_id") == "graph_sft.relation_qa":
            body = p.get("completion", "")
            assert not ("Cognitive Load" in body and "Working Memory" in body)

    # S3: every emitted instruction pair carries decontam_checked=True.
    assert all(p.get("decontam_checked") is True for p in pairs)


def test_env_flag_drives_assessment_sft(tmp_path, monkeypatch):
    course = _make_working_copy(tmp_path)
    _write_assessments(course)
    monkeypatch.setenv("ED4ALL_WITH_ASSESSMENT_SFT", "1")

    stats = run_synthesis(
        corpus_dir=course,
        course_code="MINI_TRAINING_101",
        provider="mock",
        # kwarg left False — env must drive it on.
    )
    assert stats.assessment_sft_pairs_emitted > 0


def test_assessment_sft_pairs_carry_license_stamp(tmp_path):
    """SFT-C stitch: stamp_pair_license() runs on emit, so every assessment->SFT
    pair carries a fine-grained ``license`` tag (in addition to the coarse
    ``seat_license`` / ``provider``)."""
    course = _make_working_copy(tmp_path)
    _write_assessments(course)

    run_synthesis(
        corpus_dir=course,
        course_code="MINI_TRAINING_101",
        provider="mock",
        with_assessment_sft=True,
    )

    asft = [
        p for p in _instruction_pairs(course)
        if str(p.get("template_id", "")).startswith("assessment_sft.")
    ]
    assert asft, "expected at least one assessment->SFT pair"
    for p in asft:
        assert p.get("license"), (
            "stamp_pair_license must add a non-empty license tag on emit"
        )
        # generating_seat is preserved byte-identical (the local template seat).
        assert p.get("generating_seat") == "local"


def _write_memorization_holdout(course: Path, item_ids) -> None:
    (course / "training_specs").mkdir(parents=True, exist_ok=True)
    (course / "training_specs" / ".memorization_holdout.json").write_text(
        json.dumps({
            "schema_version": "v1",
            "purpose": "memorization_probe_holdout",
            "seed": 42,
            "fraction": 1.0,
            "held_out_item_ids": sorted(item_ids),
        }),
        encoding="utf-8",
    )


def test_memorization_holdout_excludes_assessment_sft_pairs(tmp_path):
    """SFT-C/S8 stitch: run_synthesis loads training_specs/.memorization_holdout
    .json and threads held-out item ids into the assessment->SFT generator, so a
    reserved item never produces a training pair (the probe's slice stays
    genuinely unseen)."""
    course = _make_working_copy(tmp_path)
    _write_assessments(course)
    # Reserve BOTH assessment items (question_id == item_id) → zero pairs.
    _write_memorization_holdout(course, ["q-001", "q-002"])

    stats = run_synthesis(
        corpus_dir=course,
        course_code="MINI_TRAINING_101",
        provider="mock",
        with_assessment_sft=True,
    )
    assert stats.assessment_sft_pairs_emitted == 0
    assert not any(
        str(p.get("template_id", "")).startswith("assessment_sft.")
        for p in _instruction_pairs(course)
    )


def test_no_memorization_holdout_is_byte_identical_default(tmp_path):
    """Absent .memorization_holdout.json → empty set → every item is
    pair-eligible (byte-identical legacy behaviour)."""
    course = _make_working_copy(tmp_path)
    _write_assessments(course)

    stats = run_synthesis(
        corpus_dir=course,
        course_code="MINI_TRAINING_101",
        provider="mock",
        with_assessment_sft=True,
    )
    assert stats.assessment_sft_pairs_emitted > 0
