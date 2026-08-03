"""Hermetic archive factory for Trainforge regression tests.

Operator course archives are private runtime data and are never test inputs.
Tests pass neutral records explicitly and receive paths beneath ``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def make_synthetic_archive(
    tmp_path: Path,
    *,
    chunks: list[dict[str, Any]],
    objectives: dict[str, Any],
    concept_classes: dict[str, str] | None = None,
) -> tuple[Path, Path, Path, str]:
    """Materialize a neutral, test-scoped archive and return its artifacts."""
    archive = tmp_path / "sample-course"
    corpus = archive / "corpus"
    graph = archive / "graph"
    corpus.mkdir(parents=True)
    graph.mkdir(parents=True)

    chunks_path = corpus / "chunks.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    objectives_path = archive / "synthesized_objectives.json"
    objectives_path.write_text(json.dumps(objectives), encoding="utf-8")
    concept_graph_path = graph / "concept_graph.json"
    concept_graph_path.write_text(
        json.dumps({
            "nodes": [
                {"id": concept_id, "class": concept_class}
                for concept_id, concept_class in (concept_classes or {}).items()
            ],
            "edges": [],
        }),
        encoding="utf-8",
    )
    return chunks_path, objectives_path, concept_graph_path, "SAMPLE_COURSE"
