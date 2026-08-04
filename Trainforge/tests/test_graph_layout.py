"""SFT-D B3/B4 — deterministic graph-layout normalization for a LibV2 course."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Trainforge.training.graph_layout as graph_layout  # noqa: E402
from Trainforge.training.graph_layout import ensure_graph_layout  # noqa: E402


def _mk_course(tmp_path: Path) -> Path:
    course = tmp_path / "course"
    (course / "concept_graph").mkdir(parents=True)
    (course / "graph").mkdir(parents=True)
    (course / "imscc_chunks").mkdir(parents=True)
    # concept graph only in concept_graph/ (modern pipeline layout).
    (course / "concept_graph" / "concept_graph_semantic.json").write_text(
        json.dumps({"kind": "concept_semantic", "nodes": [], "edges": []}),
        encoding="utf-8",
    )
    (course / "imscc_chunks" / "chunks.jsonl").write_text(
        json.dumps({
            "id": "crs_chunk_00001", "text": "fixture",
            "concept_tags": ["alpha"], "learning_outcome_refs": ["TO-01"],
        }) + "\n",
        encoding="utf-8",
    )
    return course


def test_b4_copies_concept_graph_into_graph_dir(tmp_path, monkeypatch):
    course = _mk_course(tmp_path)
    monkeypatch.setattr(
        graph_layout, "build_pedagogy_graph", None, raising=False,
    )
    report = ensure_graph_layout(course)
    assert report["concept_graph_copied"] is True
    graph_copy = course / "graph" / "concept_graph_semantic.json"
    assert graph_copy.exists()
    assert json.loads(graph_copy.read_text())["kind"] == "concept_semantic"


def test_b3_emits_pedagogy_graph(tmp_path, monkeypatch):
    course = _mk_course(tmp_path)

    def _fake_builder(chunks, objectives=None, *, course_id=None,
                      concept_classes=None):
        assert course_id == "course"
        return {"kind": "pedagogy", "nodes": [{"id": "n1"}], "edges": []}

    monkeypatch.setattr(
        "Trainforge.rag.graphs.pedagogy_graph_builder.build_pedagogy_graph",
        _fake_builder,
    )
    report = ensure_graph_layout(course)
    assert report["pedagogy_graph_emitted"] is True
    ped = course / "graph" / "pedagogy_graph.json"
    assert ped.exists()
    assert json.loads(ped.read_text())["kind"] == "pedagogy"


def test_idempotent_when_both_present(tmp_path):
    course = _mk_course(tmp_path)
    # Pre-place both artifacts.
    (course / "graph" / "concept_graph_semantic.json").write_text(
        '{"kind": "concept_semantic", "sentinel": "keep"}', encoding="utf-8",
    )
    (course / "graph" / "pedagogy_graph.json").write_text(
        '{"kind": "pedagogy", "sentinel": "keep"}', encoding="utf-8",
    )
    report = ensure_graph_layout(course)
    assert report["concept_graph_copied"] is False
    assert report["pedagogy_graph_emitted"] is False
    # Untouched.
    assert json.loads(
        (course / "graph" / "pedagogy_graph.json").read_text()
    )["sentinel"] == "keep"
    assert json.loads(
        (course / "graph" / "concept_graph_semantic.json").read_text()
    )["sentinel"] == "keep"


def test_b3_skipped_without_chunkset(tmp_path):
    course = tmp_path / "course"
    (course / "concept_graph").mkdir(parents=True)
    (course / "graph").mkdir(parents=True)
    (course / "concept_graph" / "concept_graph_semantic.json").write_text(
        '{"kind": "concept_semantic"}', encoding="utf-8",
    )
    report = ensure_graph_layout(course)
    # B4 still fires; B3 skips (no chunkset) without raising.
    assert report["concept_graph_copied"] is True
    assert report["pedagogy_graph_emitted"] is False
    assert "pedagogy_graph_skipped_no_chunkset" in report["notes"]
    assert not (course / "graph" / "pedagogy_graph.json").exists()
