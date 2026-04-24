"""Wave 69 — targetedConcepts[] flow end-to-end from JSON-LD to KG edges.

Prior to Wave 69, the Wave 66 ``targets_concept_from_lo`` rule was wired
into ``build_semantic_graph`` and had full unit-test coverage, but the
call site in ``Trainforge.process_course._generate_semantic_concept_graph``
passed ``objectives_metadata=None``. So on every real pipeline run the
rule fired with an empty input and produced zero edges. The Wave 57
emit was silently discarded.

Wave 69 threads the parsed JSON-LD ``learningObjectives[]`` — lifted
from every page by ``html_content_parser`` — into the graph builder as
``objectives_metadata``. This test locks in the full flow:

1. A minimal HTML page with JSON-LD ``targetedConcepts[]`` lands as
   ``LearningObjective.targeted_concepts`` on the parser output.
2. ``process_course._build_objectives_metadata_for_graph`` packages
   those LOs back into the rule's expected JSON-LD-ish shape.
3. ``build_semantic_graph`` produces N ``targets-concept`` edges matching
   the input.

Also covers the "propagation to chunks" half of the wave: the same
targeted_concepts list flows onto chunks whose ``learning_outcome_refs``
cite the source LO (via ``process_course._create_chunk``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trainforge.parsers.html_content_parser import HTMLContentParser  # noqa: E402
from Trainforge.process_course import CourseProcessor  # noqa: E402
from Trainforge.rag.inference_rules.targets_concept_from_lo import (  # noqa: E402
    EDGE_TYPE as TARGETS_CONCEPT_EDGE_TYPE,
)
from Trainforge.rag.typed_edge_inference import build_semantic_graph  # noqa: E402


def _bare_processor() -> CourseProcessor:
    """Instantiate without running __init__ — mirrors existing test pattern
    in test_assesses_misconception_edge_emit.py."""
    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "TST_101"
    return proc


def _page_with_jsonld(json_ld: dict) -> str:
    return (
        "<html><head><title>p</title>"
        f'<script type="application/ld+json">{json.dumps(json_ld)}</script>'
        "</head><body><h1>p</h1></body></html>"
    )


def _parsed_item(json_ld: dict, lesson_id: str = "lesson-1") -> dict:
    """Mimic the dict shape ``process_course._parse_html`` emits per page."""
    parsed = HTMLContentParser().parse(_page_with_jsonld(json_ld))
    return {
        "item_id": lesson_id,
        "item_path": f"{lesson_id}.html",
        "title": parsed.title,
        "resource_type": "page",
        "module_id": "m1",
        "module_title": "Module 1",
        "week_num": 1,
        "word_count": parsed.word_count,
        "sections": parsed.sections,
        "learning_objectives": parsed.learning_objectives,
        "key_concepts": parsed.key_concepts,
        "interactive_components": parsed.interactive_components,
        "raw_html": "",
        "page_id": parsed.page_id,
        "misconceptions": parsed.misconceptions,
        "suggested_assessment_types": parsed.suggested_assessment_types,
        "courseforge_metadata": parsed.metadata.get("courseforge"),
        "objective_refs": parsed.objective_refs,
        "source_references": parsed.source_references,
        "_jsonld_tag_present": True,
        "_jsonld_parse_failed": False,
    }


def test_build_objectives_metadata_reuses_jsonld_payload():
    """_build_objectives_metadata_for_graph should copy through the raw
    JSON-LD learningObjectives[] untouched (so the Wave 66 rule sees the
    exact shape it was written against)."""
    proc = _bare_processor()
    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "TO-01",
                "statement": "Analyze frameworks",
                "bloomLevel": "analyze",
                "cognitiveDomain": "conceptual",
                "targetedConcepts": [
                    {"concept": "framework", "bloomLevel": "apply"},
                    {"concept": "ecosystem-flow", "bloomLevel": "analyze"},
                ],
            }
        ],
    }
    item = _parsed_item(json_ld)
    out = proc._build_objectives_metadata_for_graph([item])
    assert len(out) == 1
    lo = out[0]
    assert lo["id"] == "TO-01"
    assert lo["targetedConcepts"] == [
        {"concept": "framework", "bloomLevel": "apply"},
        {"concept": "ecosystem-flow", "bloomLevel": "analyze"},
    ]


def test_build_objectives_metadata_dedupes_across_pages():
    """Same LO appearing on multiple pages only contributes once."""
    proc = _bare_processor()
    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "TO-01",
                "statement": "s",
                "bloomLevel": "apply",
                "cognitiveDomain": "conceptual",
                "targetedConcepts": [
                    {"concept": "x", "bloomLevel": "apply"},
                ],
            }
        ],
    }
    item_a = _parsed_item(json_ld, lesson_id="p-a")
    item_b = _parsed_item(json_ld, lesson_id="p-b")
    out = proc._build_objectives_metadata_for_graph([item_a, item_b])
    assert len(out) == 1
    assert out[0]["id"] == "TO-01"


def test_end_to_end_targets_concept_edges_materialize_from_jsonld():
    """Full stack: JSON-LD input → parser → build_objectives_metadata →
    build_semantic_graph emits N targets-concept edges matching the input."""
    proc = _bare_processor()
    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "TO-01",
                "statement": "Apply frameworks",
                "bloomLevel": "apply",
                "cognitiveDomain": "conceptual",
                "targetedConcepts": [
                    {"concept": "framework", "bloomLevel": "apply"},
                    {"concept": "ecosystem-flow", "bloomLevel": "analyze"},
                ],
            },
            {
                "id": "CO-01",
                "statement": "Recognize frameworks",
                "bloomLevel": "remember",
                "cognitiveDomain": "factual",
                "targetedConcepts": [
                    {"concept": "framework", "bloomLevel": "remember"},
                ],
            },
        ],
    }
    item = _parsed_item(json_ld)
    objectives_metadata = proc._build_objectives_metadata_for_graph([item])

    # Pre-Wave-69 baseline: calling build_semantic_graph without
    # objectives_metadata produces zero targets-concept edges.
    graph_without = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph={"nodes": [], "edges": []},
        now=None,
    )
    zero_edges = [
        e for e in graph_without.get("edges", [])
        if e["type"] == TARGETS_CONCEPT_EDGE_TYPE
    ]
    assert zero_edges == [], "Baseline: no metadata → no targets-concept edges"

    # Wave 69: passing objectives_metadata materializes the edges.
    graph = build_semantic_graph(
        chunks=[],
        course=None,
        concept_graph={"nodes": [], "edges": []},
        objectives_metadata=objectives_metadata,
        now=None,
    )
    target_edges = [
        e for e in graph.get("edges", [])
        if e["type"] == TARGETS_CONCEPT_EDGE_TYPE
    ]
    # 2 targetedConcepts on TO-01 + 1 on CO-01 = 3 edges total.
    assert len(target_edges) == 3
    # Verify exact shape: lowercased LO ID source, concept target, bloom
    # carried on provenance.evidence.
    triples = sorted(
        (e["source"], e["target"],
         e["provenance"]["evidence"]["bloom_level"])
        for e in target_edges
    )
    assert triples == [
        ("co-01", "framework", "remember"),
        ("to-01", "ecosystem-flow", "analyze"),
        ("to-01", "framework", "apply"),
    ]


def test_chunk_level_targeted_concepts_propagation():
    """_create_chunk should copy targeted_concepts from cited LOs onto
    chunks via learning_outcome_refs."""
    proc = _bare_processor()
    # Stand up the bits of state _create_chunk touches.
    from collections import defaultdict
    proc.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": defaultdict(int),
        "difficulty_distribution": defaultdict(int),
    }
    proc._all_concept_tags = set()
    # Keep _extract_concept_tags / _determine_difficulty deterministic
    # (no LLM, no fallback gating).
    proc._valid_outcome_ids = {"to-01"}
    # _extract_concept_tags iterates self.domain_concept_seeds;
    # an empty list is the "no curriculum seeds loaded" state.
    proc.domain_concept_seeds = []

    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "TO-01",
                "statement": "Apply frameworks",
                "bloomLevel": "apply",
                "cognitiveDomain": "conceptual",
                "targetedConcepts": [
                    {"concept": "framework", "bloomLevel": "apply"},
                    {"concept": "ecosystem-flow", "bloomLevel": "analyze"},
                ],
            }
        ],
    }
    item = _parsed_item(json_ld)

    # _create_chunk relies on item carrying data-cf-objective-ref values
    # or a LO list referenced by ID. The parser populates item's
    # learning_objectives; item["objective_refs"] is the page union.
    # _extract_objective_refs prefers page-level objective_refs when the
    # section isn't mapped — for this test we stuff a LO-ref into the
    # page-level list so the chunk gets learning_outcome_refs=["to-01"].
    item["objective_refs"] = ["TO-01"]

    chunk = proc._create_chunk(
        chunk_id="tst_101_chunk_00001",
        text="Frameworks describe the ecosystem flow of a domain.",
        html="<p>Frameworks describe the ecosystem flow of a domain.</p>",
        item=item,
        section_heading="Section A",
        chunk_type="explanation",
    )
    # learning_outcome_refs should carry the LO id (case depends on
    # _extract_objective_refs normalization).
    refs = chunk.get("learning_outcome_refs") or []
    refs_lower = {str(r).lower() for r in refs}
    assert "to-01" in refs_lower, f"Expected to-01 in refs, got: {refs}"

    targeted = chunk.get("targeted_concepts") or []
    assert targeted == [
        {"concept": "ecosystem-flow", "bloom_level": "analyze"},
        {"concept": "framework", "bloom_level": "apply"},
    ]


def test_chunks_without_matching_lo_refs_do_not_get_targeted_concepts():
    """When a chunk's learning_outcome_refs cite an LO with no
    targetedConcepts, no targeted_concepts field is emitted (elided)."""
    proc = _bare_processor()
    from collections import defaultdict
    proc.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": defaultdict(int),
        "difficulty_distribution": defaultdict(int),
    }
    proc._all_concept_tags = set()
    proc._valid_outcome_ids = {"to-99"}
    proc.domain_concept_seeds = []

    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "TO-99",
                "statement": "no-targets LO",
                "bloomLevel": "understand",
                "cognitiveDomain": "conceptual",
                # No targetedConcepts[] — legacy / pre-Wave-57 emit.
            }
        ],
    }
    item = _parsed_item(json_ld)
    item["objective_refs"] = ["TO-99"]

    chunk = proc._create_chunk(
        chunk_id="tst_101_chunk_00002",
        text="Some text.",
        html="<p>Some text.</p>",
        item=item,
        section_heading="Section B",
        chunk_type="explanation",
    )
    assert "targeted_concepts" not in chunk
