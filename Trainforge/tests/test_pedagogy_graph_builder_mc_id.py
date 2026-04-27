"""Wave 99 — pedagogy_graph_builder._mc_id parity with the canonical algorithm.

Pre-Wave-99 the builder hashed misconception text only (lowercased,
stripped). The two other call sites (``process_course._build_misconceptions
_for_graph`` and ``preference_factory._misconception_id``) hash the
3-input seed ``statement|correction|bloom_level``. Drift across the three
sites caused 34 ``mc_*`` nodes in the ``rdf-shacl-551-2`` pedagogy graph
to disagree with chunk-level + DPO-pair IDs (Wave 97 rebuilt the on-disk
file as a one-shot).

This module is the regression test that locks the three sites to a single
algorithm. It synthesizes a misconception entry, runs the canonical
``_build_misconceptions_for_graph`` pipeline, and asserts that
``pedagogy_graph_builder._mc_id`` invoked with the same input produces the
identical ``mc_<hex>`` ID. Parametrized variants cover the with-bloom and
without-bloom branches plus whitespace normalization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Project root (Ed4All/). This file lives at
# Ed4All/Trainforge/tests/test_pedagogy_graph_builder_mc_id.py -> parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.pedagogy_graph_builder import (  # noqa: E402
    _mc_id,
    build_pedagogy_graph,
)
from Trainforge.generators.preference_factory import (  # noqa: E402
    _misconception_id,
)
from lib.ontology.misconception_id import canonical_mc_id  # noqa: E402


def _chunk_with_misconception(
    statement: str,
    correction: str,
    bloom_level: str,
) -> dict:
    """Synthesize a v4 chunk carrying a single misconception entry.

    The chunk uses the minimum keys the builder + the
    ``_build_misconceptions_for_graph`` graph emitter both consume.
    """

    return {
        "id": "ck_001",
        "course_code": "T_001",
        "concept_tags": ["triples"],
        "objective_refs": ["TO-01"],
        "bloom_level": "understand",
        "misconceptions": [
            {
                "misconception": statement,
                "correction": correction,
                "bloom_level": bloom_level,
            }
        ],
        "source": {"module_id": "week_01", "item_path": "week_01/page_01.html"},
        "content_type_label": "explanation",
    }


@pytest.mark.parametrize(
    "statement, correction, bloom",
    [
        (
            "Triples are like rows in a table.",
            "Triples are graph statements with subject, predicate, object.",
            "understand",
        ),
        (
            "Blank nodes and IRIs are the same.",
            "Blank nodes are local; IRIs are global identifiers.",
            "remember",
        ),
        (
            "RDF graphs require a fixed schema.",
            "RDF graphs are schemaless — SHACL adds shape constraints.",
            "analyze",
        ),
        (
            "SHACL validates SPARQL queries.",
            "SHACL validates RDF data against shapes, not queries.",
            "apply",
        ),
        (
            "Every triple needs a named graph.",
            "Default graphs exist; named graphs are optional.",
            "",  # bloom-less path: 2-segment seed
        ),
    ],
)
def test_mc_id_matches_canonical_helper(statement, correction, bloom):
    """``_mc_id`` is byte-equivalent to ``canonical_mc_id``."""

    builder_id = _mc_id(statement, correction, bloom)
    canonical_id = canonical_mc_id(statement, correction, bloom)
    assert builder_id == canonical_id


@pytest.mark.parametrize(
    "statement, correction, bloom",
    [
        (
            "Triples are like rows in a table.",
            "Triples are graph statements.",
            "understand",
        ),
        (
            "RDF graphs require a fixed schema.",
            "RDF graphs are schemaless.",
            "analyze",
        ),
    ],
)
def test_mc_id_matches_preference_factory(statement, correction, bloom):
    """All three canonical sites produce the same hash for the same input.

    Closes the Wave 95 + Wave 97 drift class: ``preference_factory`` and
    ``pedagogy_graph_builder`` must mint identical ``mc_*`` IDs so DPO
    pairs link cleanly to graph nodes.
    """

    builder_id = _mc_id(statement, correction, bloom)
    pref_id = _misconception_id(statement, correction, bloom)
    assert builder_id == pref_id


def test_builder_mc_node_id_matches_canonical_pipeline():
    """End-to-end: graph-emitted ``mc_*`` node ID matches canonical hash.

    Builds a graph from a synthesized chunk, extracts the Misconception
    node ID, computes the canonical ID via the helper for the same
    inputs, and asserts equality.
    """

    statement = "Triples are like rows in a table."
    correction = "Triples are graph statements."
    bloom = "understand"
    chunk = _chunk_with_misconception(statement, correction, bloom)

    objectives = {
        "terminal_objectives": [
            {"id": "TO-01", "text": "Understand RDF triples.", "bloom_level": "understand"}
        ],
        "chapter_objectives": [],
    }

    graph = build_pedagogy_graph([chunk], objectives, course_id="T_001")
    mc_nodes = [n for n in graph["nodes"] if n.get("class") == "Misconception"]
    assert len(mc_nodes) == 1, mc_nodes

    expected = canonical_mc_id(statement, correction, bloom)
    assert mc_nodes[0]["id"] == expected


def test_mc_id_string_misconception_uses_empty_fallbacks():
    """A bare-string misconception (no dict) hashes with empty correction/bloom.

    Mirrors ``_build_misconceptions_for_graph``'s string-branch fallback.
    """

    text = "RDF is a serialization format."
    builder_id = _mc_id(text)  # uses defaulted correction="" and bloom_level=""
    canonical_id = canonical_mc_id(text, "", "")
    assert builder_id == canonical_id


def test_mc_id_outer_whitespace_normalised():
    """Outer whitespace stripped — IDs invariant to surrounding spaces."""

    a = _mc_id("  triples are rows  ", "  graph statements  ", "  Understand  ")
    b = _mc_id("triples are rows", "graph statements", "understand")
    assert a == b
