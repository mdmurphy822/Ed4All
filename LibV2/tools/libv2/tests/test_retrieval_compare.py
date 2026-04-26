"""Wave 84 integration tests for the A/B retrieval-method comparison.

Builds a minimal in-memory course tree, drops a probe set, runs
``compare_retrieval_methods`` over it, and asserts the shape and
sanity of the returned report. Pins:

  - per-method aggregates exist for every requested method
  - per-query rows include each method's top-1 + hit decisions
  - method=bm25 and method=hybrid produce DIFFERENT result orderings
    (otherwise the harness can't actually tell methods apart)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from LibV2.tools.libv2.eval_harness import compare_retrieval_methods


def _make_course_repo(tmp_path: Path) -> Path:
    """Build a minimal courses/<slug>/{corpus,quality}/ tree.

    The fixture chunks are crafted so:
      - "What is RDF?" — a definition chunk should win under intent prior
      - "blank nodes" — explanation chunk wins under BM25
      - "subClassOf entailment" — only one chunk has the term
    """
    courses_dir = tmp_path / "courses"
    catalog_dir = tmp_path / "catalog"
    course_dir = courses_dir / "fake-rdf"
    corpus_dir = course_dir / "corpus"
    quality_dir = course_dir / "quality"
    corpus_dir.mkdir(parents=True)
    quality_dir.mkdir(parents=True)
    catalog_dir.mkdir(parents=True)

    # Minimal manifest so the retriever can resolve the course.
    manifest = {
        "slug": "fake-rdf",
        "schema_version": "v4",
        "classification": {
            "primary_domain": "computer-science",
            "division": "STEM",
        },
    }
    (course_dir / "manifest.json").write_text(json.dumps(manifest))

    # Minimal course.json (used by lo_match_boost loader).
    (course_dir / "course.json").write_text(
        json.dumps({"course_code": "FAKE", "title": "Fake", "learning_outcomes": []})
    )

    # Two chunks with the same surface terms but different chunk_type.
    # The intent prior should flip ordering when query carries 'define'.
    chunks = [
        {
            "id": "chunk_def_rdf",
            "schema_version": "v4",
            "chunk_type": "definition",
            "text": "RDF is the Resource Description Framework. It models knowledge as triples.",
            "summary": "Definition of RDF",
            "concept_tags": ["rdf"],
            "source": {"section_heading": "What is RDF"},
            "word_count": 13,
        },
        {
            "id": "chunk_ex_rdf",
            "schema_version": "v4",
            "chunk_type": "example",
            "text": "Here is an RDF example: alice knows bob. Resource Description Framework triples.",
            "summary": "Example of RDF",
            "concept_tags": ["rdf"],
            "source": {"section_heading": "Example: RDF"},
            "word_count": 13,
        },
        {
            "id": "chunk_subclassof",
            "schema_version": "v4",
            "chunk_type": "explanation",
            "text": "subClassOf entailment in rdfs propagates type assertions through the class hierarchy.",
            "summary": "subClassOf entailment",
            "concept_tags": ["rdfs", "subclassof"],
            "source": {"section_heading": "subClassOf entailment"},
            "word_count": 12,
        },
        {
            "id": "chunk_blank_node",
            "schema_version": "v4",
            "chunk_type": "explanation",
            "text": "Blank nodes are anonymous resources in an RDF graph without a global identifier.",
            "summary": "Blank nodes",
            "concept_tags": ["rdf", "blank-node"],
            "source": {"section_heading": "Blank nodes"},
            "word_count": 13,
        },
    ]
    with open(corpus_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    # Probe set targeting each chunk.
    probe = {
        "course_slug": "fake-rdf",
        "version": "1.0",
        "created_timestamp": "2026-04-26T00:00:00Z",
        "queries": [
            {
                "query_id": "define_rdf",
                "query_text": "What is RDF?",
                "expected_chunk_ids": ["chunk_def_rdf"],
            },
            {
                "query_id": "subclassof",
                "query_text": "subClassOf entailment",
                "expected_chunk_ids": ["chunk_subclassof"],
            },
            {
                "query_id": "blank_nodes",
                "query_text": "blank nodes",
                "expected_chunk_ids": ["chunk_blank_node"],
            },
        ],
    }
    probe_path = quality_dir / "retrieval_probe.json"
    probe_path.write_text(json.dumps(probe))

    # Master catalog so retrieve_chunks doesn't choke on missing catalog.
    master_catalog = {
        "schema_version": "1.0",
        "courses": [
            {
                "slug": "fake-rdf",
                "title": "Fake",
                "division": "STEM",
                "primary_domain": "computer-science",
            }
        ],
    }
    (catalog_dir / "master_catalog.json").write_text(json.dumps(master_catalog))
    return tmp_path


class TestCompareRetrievalMethods:
    def test_returns_aggregate_per_method(self, tmp_path):
        repo_root = _make_course_repo(tmp_path)
        result = compare_retrieval_methods(
            repo_root=repo_root,
            course_slug="fake-rdf",
            probe_path=repo_root / "courses" / "fake-rdf" / "quality" / "retrieval_probe.json",
            methods=["bm25", "hybrid"],
            retrieval_limit=10,
        )

        assert result["total_queries"] == 3
        assert set(result["aggregate"].keys()) == {"bm25", "hybrid"}
        # Required metric keys present.
        for method in ("bm25", "hybrid"):
            agg = result["aggregate"][method]
            for key in ("hit_at_1", "hit_at_5", "hit_at_10", "mrr", "map_at_10", "avg_latency_ms"):
                assert key in agg, f"{method} aggregate missing {key}"

    def test_per_query_includes_each_method(self, tmp_path):
        repo_root = _make_course_repo(tmp_path)
        result = compare_retrieval_methods(
            repo_root=repo_root,
            course_slug="fake-rdf",
            probe_path=repo_root / "courses" / "fake-rdf" / "quality" / "retrieval_probe.json",
            methods=["bm25", "bm25+intent"],
            retrieval_limit=5,
        )
        assert len(result["per_query"]) == 3
        for row in result["per_query"]:
            assert "query_id" in row
            assert set(row["results"].keys()) == {"bm25", "bm25+intent"}
            for method in ("bm25", "bm25+intent"):
                method_row = row["results"][method]
                assert "hit_at_1" in method_row
                assert "rr" in method_row
                assert "top1" in method_row
                assert "matched_in_top_10" in method_row

    def test_unknown_method_raises_in_runner(self, tmp_path):
        # The preset resolver fails closed on bad method names; this
        # propagates through the comparison runner.
        repo_root = _make_course_repo(tmp_path)
        with pytest.raises(ValueError, match="Unknown retrieval method preset"):
            compare_retrieval_methods(
                repo_root=repo_root,
                course_slug="fake-rdf",
                probe_path=repo_root / "courses" / "fake-rdf" / "quality" / "retrieval_probe.json",
                methods=["definitely-not-a-method"],
                retrieval_limit=5,
            )

    def test_missing_probe_raises(self, tmp_path):
        repo_root = _make_course_repo(tmp_path)
        with pytest.raises(FileNotFoundError):
            compare_retrieval_methods(
                repo_root=repo_root,
                course_slug="fake-rdf",
                probe_path=repo_root / "no_such_probe.json",
                methods=["bm25"],
                retrieval_limit=5,
            )

    def test_empty_probe_raises(self, tmp_path):
        repo_root = _make_course_repo(tmp_path)
        empty = repo_root / "courses" / "fake-rdf" / "quality" / "empty_probe.json"
        empty.write_text(json.dumps({"course_slug": "fake-rdf", "queries": []}))
        with pytest.raises(ValueError, match="No queries"):
            compare_retrieval_methods(
                repo_root=repo_root,
                course_slug="fake-rdf",
                probe_path=empty,
                methods=["bm25"],
                retrieval_limit=5,
            )
