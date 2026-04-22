"""Wave 11 — archive_to_libv2 features.evidence_source_provenance manifest flag.

Contract: when ``MCP/tools/pipeline_tools.py::archive_to_libv2`` emits the
LibV2 manifest, it scans the archived concept_graph_semantic.json for
edges whose ``provenance.evidence.source_references`` is a non-empty
array. If at least one edge has the field populated,
``manifest.features.evidence_source_provenance = true``. Otherwise
(missing file, malformed json, no refs anywhere) it's ``false``.

This advisory flag lets LibV2 retrieval callers distinguish chunk-level
(Wave 10, ``features.source_provenance``) from evidence-level (Wave 11,
this flag) provenance, so they can target queries appropriately.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Callable, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _detect_evidence_source_provenance,
    _graph_has_evidence_refs,
    register_pipeline_tools,
)


class _CapturingMCP:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def tool(self):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


@pytest.fixture
def archive_tool(monkeypatch, tmp_path):
    """Return archive_to_libv2 coroutine rooted at tmp_path."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    mcp = _CapturingMCP()
    register_pipeline_tools(mcp)
    return mcp.tools["archive_to_libv2"]


# --------------------------------------------------------------------- #
# Unit tests on _graph_has_evidence_refs
# --------------------------------------------------------------------- #


def test_graph_has_evidence_refs_false_on_non_dict():
    assert _graph_has_evidence_refs(None) is False
    assert _graph_has_evidence_refs([]) is False
    assert _graph_has_evidence_refs("graph") is False
    assert _graph_has_evidence_refs(42) is False


def test_graph_has_evidence_refs_false_on_missing_edges():
    assert _graph_has_evidence_refs({"kind": "concept_semantic"}) is False


def test_graph_has_evidence_refs_false_on_edges_without_provenance():
    graph = {"edges": [{"source": "a", "target": "b", "type": "is-a"}]}
    assert _graph_has_evidence_refs(graph) is False


def test_graph_has_evidence_refs_false_on_legacy_evidence_shapes():
    """Wave-6 evidence (no source_references) → false."""
    graph = {
        "edges": [
            {
                "source": "a",
                "target": "b",
                "type": "is-a",
                "provenance": {
                    "rule": "is_a_from_key_terms",
                    "rule_version": 1,
                    "evidence": {
                        "chunk_id": "c1",
                        "term": "x",
                        "definition_excerpt": "y",
                        "pattern": "p",
                    },
                },
            },
        ],
    }
    assert _graph_has_evidence_refs(graph) is False


def test_graph_has_evidence_refs_true_on_wave11_evidence():
    graph = {
        "edges": [
            {
                "source": "a",
                "target": "b",
                "type": "is-a",
                "provenance": {
                    "rule": "is_a_from_key_terms",
                    "rule_version": 2,
                    "evidence": {
                        "chunk_id": "c1",
                        "term": "x",
                        "definition_excerpt": "y",
                        "pattern": "p",
                        "source_references": [
                            {"sourceId": "dart:s#b", "role": "primary"},
                        ],
                    },
                },
            },
        ],
    }
    assert _graph_has_evidence_refs(graph) is True


def test_graph_has_evidence_refs_false_on_empty_source_references():
    """Empty source_references array does NOT count as populated."""
    graph = {
        "edges": [
            {
                "source": "a",
                "target": "b",
                "type": "is-a",
                "provenance": {
                    "rule": "is_a_from_key_terms",
                    "rule_version": 2,
                    "evidence": {
                        "chunk_id": "c1",
                        "term": "x",
                        "definition_excerpt": "y",
                        "pattern": "p",
                        "source_references": [],
                    },
                },
            },
        ],
    }
    assert _graph_has_evidence_refs(graph) is False


def test_graph_has_evidence_refs_true_with_mixed_edges():
    """Single edge carrying refs is enough."""
    graph = {
        "edges": [
            {
                "source": "a",
                "target": "b",
                "type": "is-a",
                "provenance": {
                    "rule": "is_a_from_key_terms",
                    "rule_version": 1,
                    "evidence": {
                        "chunk_id": "c1",
                        "term": "x",
                        "definition_excerpt": "y",
                        "pattern": "p",
                    },
                },
            },
            {
                "source": "c",
                "target": "d",
                "type": "exemplifies",
                "provenance": {
                    "rule": "exemplifies_from_example_chunks",
                    "rule_version": 2,
                    "evidence": {
                        "chunk_id": "c_ex",
                        "concept_slug": "z",
                        "content_type": "chunk_type",
                        "source_references": [
                            {"sourceId": "dart:a#b", "role": "contributing"}
                        ],
                    },
                },
            },
        ],
    }
    assert _graph_has_evidence_refs(graph) is True


def test_graph_has_evidence_refs_tolerates_malformed_entries():
    """Non-dict edges / provenance / evidence don't crash the scanner."""
    graph = {
        "edges": [
            "not-a-dict",
            {"source": "a"},  # no provenance
            {"provenance": None},
            {"provenance": {"evidence": "string-evidence"}},
            {"provenance": {"evidence": {"source_references": "not-a-list"}}},
            {
                "provenance": {
                    "evidence": {
                        "source_references": [
                            {"sourceId": "dart:x#y", "role": "primary"}
                        ]
                    }
                }
            },
        ]
    }
    assert _graph_has_evidence_refs(graph) is True


# --------------------------------------------------------------------- #
# _detect_evidence_source_provenance — file-based scans
# --------------------------------------------------------------------- #


def test_detect_evidence_returns_false_no_graph_file(tmp_path):
    course_dir = tmp_path / "courses" / "slug"
    (course_dir / "corpus").mkdir(parents=True)
    (course_dir / "graph").mkdir(parents=True)
    assert _detect_evidence_source_provenance(course_dir) is False


def test_detect_evidence_reads_graph_subdir(tmp_path):
    course_dir = tmp_path / "courses" / "slug"
    (course_dir / "graph").mkdir(parents=True)
    (course_dir / "corpus").mkdir(parents=True)
    graph = {
        "kind": "concept_semantic",
        "edges": [
            {
                "provenance": {
                    "evidence": {
                        "source_references": [
                            {"sourceId": "dart:x#y", "role": "primary"}
                        ]
                    }
                }
            },
        ],
    }
    (course_dir / "graph" / "concept_graph_semantic.json").write_text(
        json.dumps(graph)
    )
    assert _detect_evidence_source_provenance(course_dir) is True


def test_detect_evidence_falls_back_to_corpus_subdir(tmp_path):
    course_dir = tmp_path / "courses" / "slug"
    (course_dir / "corpus").mkdir(parents=True)
    graph = {
        "kind": "concept_semantic",
        "edges": [
            {
                "provenance": {
                    "evidence": {
                        "source_references": [
                            {"sourceId": "dart:x#y", "role": "primary"}
                        ]
                    }
                }
            },
        ],
    }
    (course_dir / "corpus" / "concept_graph_semantic.json").write_text(
        json.dumps(graph)
    )
    assert _detect_evidence_source_provenance(course_dir) is True


def test_detect_evidence_false_when_graph_has_only_legacy_edges(tmp_path):
    course_dir = tmp_path / "courses" / "slug"
    (course_dir / "graph").mkdir(parents=True)
    graph = {
        "kind": "concept_semantic",
        "edges": [
            {
                "provenance": {
                    "rule": "is_a_from_key_terms",
                    "evidence": {
                        "chunk_id": "c1",
                        "term": "x",
                        "definition_excerpt": "y",
                        "pattern": "p",
                    },
                }
            }
        ],
    }
    (course_dir / "graph" / "concept_graph_semantic.json").write_text(
        json.dumps(graph)
    )
    assert _detect_evidence_source_provenance(course_dir) is False


def test_detect_evidence_false_on_malformed_json(tmp_path):
    course_dir = tmp_path / "courses" / "slug"
    (course_dir / "graph").mkdir(parents=True)
    (course_dir / "graph" / "concept_graph_semantic.json").write_text(
        "{ broken json"
    )
    assert _detect_evidence_source_provenance(course_dir) is False


# --------------------------------------------------------------------- #
# End-to-end archive_to_libv2 manifest flag
# --------------------------------------------------------------------- #


def test_archive_manifest_includes_evidence_flag_key(archive_tool, tmp_path):
    """Even with no corpus + no graph the flag key is present (False)."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_E11",
        domain="test-domain",
    ))
    result = json.loads(result_str)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert "features" in manifest
    assert "evidence_source_provenance" in manifest["features"]
    assert manifest["features"]["evidence_source_provenance"] is False


def test_archive_manifest_keeps_both_wave10_and_wave11_flags(
    archive_tool, tmp_path
):
    """Both flags coexist in features block."""
    result_str = asyncio.run(archive_tool(
        course_name="TEST_E11b",
        domain="test-domain",
    ))
    result = json.loads(result_str)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    features = manifest["features"]
    # Wave 10 flag
    assert "source_provenance" in features
    # Wave 11 flag
    assert "evidence_source_provenance" in features
    # Both are boolean (not None / missing / stringy)
    assert isinstance(features["source_provenance"], bool)
    assert isinstance(features["evidence_source_provenance"], bool)


def test_archive_manifest_evidence_flag_true_with_wave11_graph(
    archive_tool, tmp_path
):
    """When the graph subdir (pre-populated into the archive's course dir)
    carries evidence refs, the flag is true."""
    # The archive_to_libv2 tool builds the course dir at LibV2/courses/<slug>.
    # Since we monkeypatched PROJECT_ROOT to tmp_path, we can pre-create the
    # course_dir/graph and drop a semantic graph there BEFORE running the
    # archive.
    course_dir = tmp_path / "LibV2" / "courses" / "test-e11c"
    (course_dir / "graph").mkdir(parents=True)
    graph = {
        "kind": "concept_semantic",
        "edges": [
            {
                "provenance": {
                    "evidence": {
                        "source_references": [
                            {"sourceId": "dart:a#b", "role": "primary"},
                        ]
                    }
                }
            }
        ],
    }
    (course_dir / "graph" / "concept_graph_semantic.json").write_text(
        json.dumps(graph)
    )

    result_str = asyncio.run(archive_tool(
        course_name="TEST_E11c",
        domain="test-domain",
    ))
    result = json.loads(result_str)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["features"]["evidence_source_provenance"] is True


def test_archive_manifest_evidence_flag_false_with_legacy_graph(
    archive_tool, tmp_path
):
    """Legacy (Wave 6) graph with no evidence refs → flag false."""
    course_dir = tmp_path / "LibV2" / "courses" / "test-e11d"
    (course_dir / "graph").mkdir(parents=True)
    graph = {
        "kind": "concept_semantic",
        "edges": [
            {
                "provenance": {
                    "rule": "is_a_from_key_terms",
                    "evidence": {
                        "chunk_id": "c1",
                        "term": "x",
                        "definition_excerpt": "y",
                        "pattern": "p",
                    },
                }
            }
        ],
    }
    (course_dir / "graph" / "concept_graph_semantic.json").write_text(
        json.dumps(graph)
    )

    result_str = asyncio.run(archive_tool(
        course_name="TEST_E11d",
        domain="test-domain",
    ))
    result = json.loads(result_str)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["features"]["evidence_source_provenance"] is False


def test_archive_manifest_features_is_dict(archive_tool, tmp_path):
    result_str = asyncio.run(archive_tool(
        course_name="TEST_E11e",
        domain="test-domain",
    ))
    manifest = json.loads(
        Path(json.loads(result_str)["manifest_path"]).read_text()
    )
    assert isinstance(manifest["features"], dict)
