"""Fix-2 — `_generate_semantic_concept_graph` upstream-consumption.

Locks the contract that ``CourseProcessor._generate_semantic_concept_graph``
short-circuits the in-process ``build_semantic_graph`` call when an
upstream-emitted ``kind: "concept_semantic"`` file is supplied via the
``concept_graph_path`` constructor kwarg (the path the
``concept_extraction`` workflow phase writes at
``LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json``).

Mirrors the ``_generate_pedagogy_graph`` ST-13 short-circuit test
(``test_concept_graph_consumption.py``).

Contract assertions:

1. **Skip-when-semantic-file-supplied.** ``concept_graph_path`` set to a
   ``kind: "concept_semantic"`` JSON file → method returns it verbatim
   WITHOUT invoking ``build_semantic_graph``.
2. **Fall-through when path is None.** Legacy corpora rebuild in-process.
3. **Fall-through on missing file.** A stale handoff doesn't crash.
4. **Fall-through on parse failure.** Malformed JSON doesn't crash.
5. **Fall-through on wrong `kind`.** A ``kind: "pedagogy"`` file at the
   path is NOT adopted — it falls through to the rebuild (the
   disambiguation guard).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import CourseProcessor  # noqa: E402


def _make_processor(concept_graph_path: Any = None) -> CourseProcessor:
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "FIX2_SEMANTIC_TEST"
    proc.objectives = {}
    proc.typed_edges_llm = False
    proc.capture = None
    proc.concept_graph_path = (
        Path(concept_graph_path) if concept_graph_path else None
    )
    return proc


def _minimal_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "id": "chunk_a",
            "chunk_type": "explanation",
            "concept_tags": ["alpha"],
            "learning_outcome_refs": [],
            "source": {"module_id": "week_01", "item_path": "week_01/p.html"},
        },
    ]


def _semantic_graph_payload() -> Dict[str, Any]:
    """A genuine ``kind: "concept_semantic"`` payload with a sentinel."""
    return {
        "kind": "concept_semantic",
        "_fix2_marker": "loaded_from_disk",
        "nodes": [
            {"id": "alpha", "class": "DomainConcept", "label": "Alpha"},
            {"id": "beta", "class": "DomainConcept", "label": "Beta"},
        ],
        "edges": [
            {"source": "alpha", "target": "beta", "type": "related-to"},
        ],
        "generated_at": "2026-05-15T00:00:00",
    }


def _pedagogy_graph_payload() -> Dict[str, Any]:
    """A ``kind: "pedagogy"`` payload — must NOT be adopted by the
    semantic short-circuit (the disambiguation guard).
    """
    return {
        "kind": "pedagogy",
        "_fix2_marker": "wrong_kind_should_not_load",
        "nodes": [{"id": "x", "class": "BloomLevel"}],
        "edges": [],
        "generated_at": "2026-05-15T00:00:00",
    }


def test_consumes_upstream_semantic_graph_when_present(tmp_path: Path) -> None:
    """A valid ``kind: "concept_semantic"`` file at ``concept_graph_path``
    is returned verbatim WITHOUT calling ``build_semantic_graph``.
    """
    graph_path = tmp_path / "concept_graph_semantic.json"
    graph_path.write_text(json.dumps(_semantic_graph_payload()), encoding="utf-8")

    proc = _make_processor(concept_graph_path=graph_path)

    def _sentinel(*args, **kwargs):
        raise AssertionError(
            "build_semantic_graph called despite a kind='concept_semantic' "
            "file being supplied — Fix-2 short-circuit failed."
        )

    with mock.patch(
        "Trainforge.rag.typed_edge_inference.build_semantic_graph", _sentinel
    ):
        graph = proc._generate_semantic_concept_graph(
            _minimal_chunks(), None, {"kind": "concept", "nodes": [], "edges": []}
        )

    assert graph.get("_fix2_marker") == "loaded_from_disk", (
        "Returned graph does not match the upstream payload."
    )
    assert graph.get("kind") == "concept_semantic"


def test_falls_through_when_path_is_none() -> None:
    """No ``concept_graph_path`` → in-process ``build_semantic_graph`` runs."""
    proc = _make_processor(concept_graph_path=None)

    called = {"count": 0}

    def _capturing(*args, **kwargs):
        called["count"] += 1
        return {"kind": "concept_semantic", "_built_in_process": True,
                "nodes": [], "edges": []}

    with mock.patch(
        "Trainforge.rag.typed_edge_inference.build_semantic_graph", _capturing
    ):
        graph = proc._generate_semantic_concept_graph(
            _minimal_chunks(), None, {"kind": "concept", "nodes": [], "edges": []}
        )

    assert called["count"] == 1
    assert graph.get("_built_in_process") is True


def test_falls_through_when_path_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A stale handoff (file missing) degrades to the in-process build."""
    proc = _make_processor(concept_graph_path=tmp_path / "missing.json")

    called = {"count": 0}

    def _capturing(*args, **kwargs):
        called["count"] += 1
        return {"kind": "concept_semantic", "_built_in_process": True,
                "nodes": [], "edges": []}

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        with mock.patch(
            "Trainforge.rag.typed_edge_inference.build_semantic_graph",
            _capturing,
        ):
            graph = proc._generate_semantic_concept_graph(
                _minimal_chunks(), None,
                {"kind": "concept", "nodes": [], "edges": []},
            )

    assert called["count"] == 1
    assert graph.get("_built_in_process") is True
    assert any(
        "does not exist" in r.getMessage() for r in caplog.records
    )


def test_falls_through_on_malformed_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Malformed JSON at the path falls through to the in-process build."""
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json {{{", encoding="utf-8")
    proc = _make_processor(concept_graph_path=bad)

    called = {"count": 0}

    def _capturing(*args, **kwargs):
        called["count"] += 1
        return {"kind": "concept_semantic", "_built_in_process": True,
                "nodes": [], "edges": []}

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        with mock.patch(
            "Trainforge.rag.typed_edge_inference.build_semantic_graph",
            _capturing,
        ):
            graph = proc._generate_semantic_concept_graph(
                _minimal_chunks(), None,
                {"kind": "concept", "nodes": [], "edges": []},
            )

    assert called["count"] == 1
    assert graph.get("_built_in_process") is True
    assert any("failed to load" in r.getMessage().lower() for r in caplog.records)


def test_falls_through_on_wrong_kind(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A ``kind: "pedagogy"`` file at ``concept_graph_path`` is NOT
    adopted by the semantic short-circuit — it falls through to the
    in-process rebuild. This is the disambiguation guard: a legacy
    pedagogy artifact must never be served as the semantic graph.
    """
    ped = tmp_path / "concept_graph_semantic.json"
    ped.write_text(json.dumps(_pedagogy_graph_payload()), encoding="utf-8")
    proc = _make_processor(concept_graph_path=ped)

    called = {"count": 0}

    def _capturing(*args, **kwargs):
        called["count"] += 1
        return {"kind": "concept_semantic", "_built_in_process": True,
                "nodes": [], "edges": []}

    with caplog.at_level(logging.WARNING, logger="Trainforge.process_course"):
        with mock.patch(
            "Trainforge.rag.typed_edge_inference.build_semantic_graph",
            _capturing,
        ):
            graph = proc._generate_semantic_concept_graph(
                _minimal_chunks(), None,
                {"kind": "concept", "nodes": [], "edges": []},
            )

    assert called["count"] == 1, (
        "build_semantic_graph should run when the upstream file is the "
        "wrong kind — the pedagogy artifact must not be mis-adopted."
    )
    assert graph.get("_built_in_process") is True
    assert graph.get("_fix2_marker") != "wrong_kind_should_not_load"
    assert any("not a kind" in r.getMessage() for r in caplog.records)
