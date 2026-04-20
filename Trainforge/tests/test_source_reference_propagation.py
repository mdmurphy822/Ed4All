"""Wave 10 — end-to-end source_reference propagation tests.

Contracts:

- HTML with page-level JSON-LD ``sourceReferences`` → parser captures the
  list verbatim on ``ParsedHTMLModule.source_references``.
- HTML with section-level ``data-cf-source-ids`` attrs → parser captures
  them on ``ContentSection.source_references`` as stringified sourceIds.
- Chunker emits ``source.source_references[]`` on chunks whose page has
  any Wave-9 provenance.
- JSON-LD (full shape) has precedence over ``data-cf-source-ids`` (auto-
  roled as contributing) — first-seen wins on sourceId collision.
- Legacy HTML without any provenance → chunks carry no
  ``source.source_references`` field (absence = pre-Wave-9 corpus).
- Graph builder copies ``occurrences[0]`` chunk's
  ``source.source_references`` to the node as ``source_refs[]``.
- Concept nodes whose first chunk has no refs stay legacy-shaped.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------- #
# HTML fixtures
# --------------------------------------------------------------------- #


WAVE9_HTML_FULL = """<!DOCTYPE html>
<html>
<head>
  <script type="application/ld+json">
  {
    "@context": "https://ed4all.dev/ns/courseforge/v1",
    "@type": "CourseModule",
    "courseCode": "SAMPLE_101",
    "weekNumber": 3,
    "moduleType": "content",
    "pageId": "week_03_content_01_cognitive_load",
    "sourceReferences": [
      {
        "sourceId": "dart:science_of_learning#s5_p2",
        "role": "primary",
        "weight": 0.8,
        "confidence": 0.92,
        "pages": [14, 15],
        "extractor": "pdfplumber"
      }
    ],
    "sections": [
      {
        "heading": "Cognitive Load Types",
        "contentType": "explanation",
        "bloomRange": ["understand"],
        "sourceReferences": [
          {
            "sourceId": "dart:science_of_learning#s6_p1",
            "role": "contributing"
          }
        ]
      }
    ]
  }
  </script>
</head>
<body>
<main>
<h1 data-cf-source-ids="dart:science_of_learning#s5_p2">Cognitive Load</h1>
<section data-cf-source-ids="dart:science_of_learning#s5_p2,dart:new_source#s2_p0">
<h2 data-cf-content-type="explanation" data-cf-source-ids="dart:science_of_learning#s6_p1">Cognitive Load Types</h2>
<p>Cognitive load theory divides mental effort into three categories: intrinsic, extraneous, and germane load.</p>
<p>Intrinsic load is determined by the inherent complexity of the material being learned.</p>
<p>Extraneous load comes from poorly designed instruction that distracts from the learning objective.</p>
<p>Germane load is productive — it's the effort devoted to constructing mental schemas.</p>
</section>
</main>
</body>
</html>
"""


LEGACY_HTML_NO_PROVENANCE = """<!DOCTYPE html>
<html>
<head><title>Legacy Page</title></head>
<body>
<main>
<h1>Legacy Heading</h1>
<h2>Legacy Section</h2>
<p>Legacy content from pre-Wave-9 corpus. No source provenance anywhere.
This is the sole paragraph in the legacy section with plenty of words to
avoid the minimum chunk size gate and to make cognitive load and other
concept tags surface in the concept graph builder.</p>
<p>A second paragraph to reach word counts that produce at least one chunk.
Cognitive load appears here to drive concept-graph construction.</p>
</main>
</body>
</html>
"""


WAVE9_HTML_DATA_ATTR_ONLY = """<!DOCTYPE html>
<html>
<head>
  <script type="application/ld+json">
  {
    "@context": "https://ed4all.dev/ns/courseforge/v1",
    "@type": "CourseModule",
    "courseCode": "SAMPLE_101",
    "weekNumber": 4,
    "moduleType": "content",
    "pageId": "week_04_attr_only"
  }
  </script>
</head>
<body>
<main>
<h1>Data Attribute Only Page</h1>
<section data-cf-source-ids="dart:slug_a#s0_p0">
<h2 data-cf-content-type="explanation">Attribute-Only Section</h2>
<p>This page has no JSON-LD sourceReferences — only data-cf-source-ids.
Cognitive load makes a concept tag here for the graph builder.
Cognitive load, cognitive load, cognitive load to push frequency>=2.</p>
<p>Another paragraph with cognitive load and enough content to keep the
chunker from merging or dropping this section.</p>
</section>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------- #
# Parser-level tests
# --------------------------------------------------------------------- #


def test_parser_captures_page_level_jsonld_source_references():
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parser = HTMLContentParser()
    parsed = parser.parse(WAVE9_HTML_FULL)

    page_ids = [r["sourceId"] for r in parsed.source_references]
    assert "dart:science_of_learning#s5_p2" in page_ids, page_ids
    # Section-level JSON-LD ref should also aggregate up.
    assert "dart:science_of_learning#s6_p1" in page_ids, page_ids
    # data-cf-source-ids new block (not in JSON-LD) should also appear
    # via the HTML-attr fallback.
    assert "dart:new_source#s2_p0" in page_ids, page_ids


def test_parser_preserves_jsonld_role_over_html_attr_default():
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parser = HTMLContentParser()
    parsed = parser.parse(WAVE9_HTML_FULL)

    by_sid = {r["sourceId"]: r for r in parsed.source_references}
    # JSON-LD said 'primary' for s5_p2 — must NOT be overridden to
    # contributing even though an HTML data-cf-source-ids also lists it.
    assert by_sid["dart:science_of_learning#s5_p2"]["role"] == "primary"
    # HTML-only refs default to 'contributing'.
    assert by_sid["dart:new_source#s2_p0"]["role"] == "contributing"


def test_parser_captures_section_level_source_ids():
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parser = HTMLContentParser()
    parsed = parser.parse(WAVE9_HTML_FULL)

    # Should have "Cognitive Load" (h1) and "Cognitive Load Types" (h2).
    sections_by_heading = {s.heading: s for s in parsed.sections}
    assert "Cognitive Load" in sections_by_heading
    assert "Cognitive Load Types" in sections_by_heading
    types_section = sections_by_heading["Cognitive Load Types"]
    # The heading carried data-cf-source-ids="dart:science_of_learning#s6_p1"
    assert "dart:science_of_learning#s6_p1" in types_section.source_references


def test_parser_legacy_html_empty_source_references():
    """Pre-Wave-9 HTML returns an empty source_references list — no error."""
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parser = HTMLContentParser()
    parsed = parser.parse(LEGACY_HTML_NO_PROVENANCE)
    assert parsed.source_references == []
    for section in parsed.sections:
        assert section.source_references == []


def test_parser_data_attr_only_auto_roles_contributing():
    """HTML attrs without JSON-LD get synthesised as contributing."""
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parser = HTMLContentParser()
    parsed = parser.parse(WAVE9_HTML_DATA_ATTR_ONLY)

    by_sid = {r["sourceId"]: r for r in parsed.source_references}
    assert "dart:slug_a#s0_p0" in by_sid
    assert by_sid["dart:slug_a#s0_p0"]["role"] == "contributing"


def test_parser_dedupes_repeated_source_ids():
    """Same sourceId listed multiple times only appears once."""
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parser = HTMLContentParser()
    # Same ID appears at page-level, section-level, and HTML attr.
    parsed = parser.parse(WAVE9_HTML_FULL)
    sids = [r["sourceId"] for r in parsed.source_references]
    assert len(sids) == len(set(sids)), f"Duplicate sourceIds leaked: {sids}"


# --------------------------------------------------------------------- #
# Chunker propagation tests (through the full _chunk_content path)
# --------------------------------------------------------------------- #


def _build_parsed_item(html: str, parser) -> Dict[str, Any]:
    """Mirror the dict shape _chunk_content expects."""
    parsed = parser.parse(html)
    return {
        "item_id": "item_1",
        "item_path": "content/week_03/cognitive_load.html",
        "title": parsed.title,
        "resource_type": "page",
        "module_id": "week_03",
        "module_title": "Week 3",
        "week_num": 3,
        "word_count": parsed.word_count,
        "sections": parsed.sections,
        "learning_objectives": parsed.learning_objectives,
        "key_concepts": parsed.key_concepts,
        "interactive_components": parsed.interactive_components,
        "raw_html": html,
        "page_id": parsed.page_id,
        "misconceptions": parsed.misconceptions,
        "suggested_assessment_types": parsed.suggested_assessment_types,
        "courseforge_metadata": parsed.metadata.get("courseforge"),
        "objective_refs": parsed.objective_refs,
        "source_references": parsed.source_references,
        "_jsonld_tag_present": True,
        "_jsonld_parse_failed": False,
    }


def _make_processor():
    from collections import defaultdict
    from types import SimpleNamespace

    from Trainforge.process_course import CourseProcessor

    processor = object.__new__(CourseProcessor)
    processor.capture = SimpleNamespace(
        run_id="test_run_w10",
        log_decision=lambda **kwargs: None,
    )
    processor.course_code = "sample_101"
    processor.stats = {
        "total_chunks": 0,
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": defaultdict(int),
        "difficulty_distribution": defaultdict(int),
        "modules_processed": 0,
        "quizzes_processed": 0,
        "sections_processed": 0,
    }
    processor._boilerplate_spans = []
    processor._all_concept_tags = set()
    processor.domain_concept_seeds = []
    processor.objectives = None
    processor.OBJECTIVE_CODE_RE = CourseProcessor.OBJECTIVE_CODE_RE
    processor.WEEK_PREFIX_RE = CourseProcessor.WEEK_PREFIX_RE
    processor.NON_CONCEPT_TAGS = CourseProcessor.NON_CONCEPT_TAGS
    processor.MIN_CHUNK_SIZE = CourseProcessor.MIN_CHUNK_SIZE
    processor.MAX_CHUNK_SIZE = CourseProcessor.MAX_CHUNK_SIZE
    processor.TARGET_CHUNK_SIZE = CourseProcessor.TARGET_CHUNK_SIZE
    return processor


def _chunk_from_html(html: str) -> List[Dict[str, Any]]:
    from Trainforge.parsers.html_content_parser import HTMLContentParser

    parser = HTMLContentParser()
    processor = _make_processor()
    item = _build_parsed_item(html, parser)
    return processor._chunk_content([item])


def test_chunker_writes_source_references_on_chunks():
    chunks = _chunk_from_html(WAVE9_HTML_FULL)
    assert chunks, "Expected at least one chunk"
    # At least one chunk must carry source.source_references.
    with_refs = [c for c in chunks if c["source"].get("source_references")]
    assert with_refs, "No chunks carry source_references — propagation broken"


def test_chunker_preserves_authoritative_role_on_chunks():
    chunks = _chunk_from_html(WAVE9_HTML_FULL)
    # Find any chunk that carries s5_p2 and check its role stays 'primary'.
    found = False
    for chunk in chunks:
        refs = chunk["source"].get("source_references", [])
        for ref in refs:
            if ref["sourceId"] == "dart:science_of_learning#s5_p2":
                assert ref["role"] == "primary"
                found = True
    assert found, "s5_p2 primary-roled ref never landed on a chunk"


def test_chunker_html_attr_refs_auto_role_contributing():
    chunks = _chunk_from_html(WAVE9_HTML_DATA_ATTR_ONLY)
    assert chunks
    # Find the HTML-attr-only refs and confirm contributing role.
    for chunk in chunks:
        refs = chunk["source"].get("source_references", [])
        for ref in refs:
            if ref["sourceId"] == "dart:slug_a#s0_p0":
                assert ref["role"] == "contributing"


def test_legacy_chunk_has_no_source_references_field():
    """Pre-Wave-9 HTML → chunks lack source.source_references (absence)."""
    chunks = _chunk_from_html(LEGACY_HTML_NO_PROVENANCE)
    assert chunks
    for chunk in chunks:
        assert "source_references" not in chunk["source"], (
            "Legacy chunk unexpectedly carries source_references"
        )


def test_chunker_dedupes_source_ids_in_chunk():
    """Chunk should not carry the same sourceId twice."""
    chunks = _chunk_from_html(WAVE9_HTML_FULL)
    for chunk in chunks:
        refs = chunk["source"].get("source_references", [])
        sids = [r["sourceId"] for r in refs]
        assert len(sids) == len(set(sids)), f"Duplicate sourceIds: {sids}"


# --------------------------------------------------------------------- #
# Graph builder tests — node source_refs[]
# --------------------------------------------------------------------- #


def _build_graph(chunks, course_id=""):
    from Trainforge.process_course import CourseProcessor

    processor = CourseProcessor.__new__(CourseProcessor)
    processor.course_code = course_id
    return processor._build_tag_graph(chunks)


def _mk_chunk(chunk_id, tags, refs=None):
    source: Dict[str, Any] = {
        "course_id": "sample_101",
        "module_id": "m",
        "lesson_id": "l",
    }
    if refs:
        source["source_references"] = refs
    return {"id": chunk_id, "concept_tags": list(tags), "source": source}


def test_node_source_refs_copied_from_first_occurrence(monkeypatch):
    from Trainforge.rag import typed_edge_inference

    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    chunks = [
        _mk_chunk(
            "c_00001",
            ["cognitive-load"],
            refs=[
                {"sourceId": "dart:a#s0_p0", "role": "primary"},
                {"sourceId": "dart:a#s1_p0", "role": "contributing"},
            ],
        ),
        _mk_chunk(
            "c_00002",
            ["cognitive-load"],
            refs=[
                {"sourceId": "dart:b#s0_p0", "role": "primary"},
            ],
        ),
    ]

    graph = _build_graph(chunks)
    by_id = {n["id"]: n for n in graph["nodes"]}
    node = by_id["cognitive-load"]
    # occurrences[0] is c_00001 (sorted ASC). Its refs should be copied.
    assert node.get("source_refs")
    copied_sids = [r["sourceId"] for r in node["source_refs"]]
    assert copied_sids == ["dart:a#s0_p0", "dart:a#s1_p0"]
    # Role precedence preserved from the chunk.
    assert node["source_refs"][0]["role"] == "primary"


def test_node_without_source_refs_when_chunk_has_none(monkeypatch):
    """Pre-Wave-9 chunks → nodes stay legacy-shaped (no source_refs)."""
    from Trainforge.rag import typed_edge_inference

    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    chunks = [
        _mk_chunk("c_00001", ["cognitive-load"]),
        _mk_chunk("c_00002", ["cognitive-load"]),
    ]
    graph = _build_graph(chunks)
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert "source_refs" not in by_id["cognitive-load"]


def test_node_source_refs_deterministic_sort_order(monkeypatch):
    """Occurrences are sorted ASC → source_refs come from the lowest ID."""
    from Trainforge.rag import typed_edge_inference

    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    # Insert chunks in reverse order to verify sort ordering of occurrences.
    chunks = [
        _mk_chunk(
            "c_00003",
            ["cognitive-load"],
            refs=[{"sourceId": "dart:c#s0_p0", "role": "primary"}],
        ),
        _mk_chunk(
            "c_00001",
            ["cognitive-load"],
            refs=[{"sourceId": "dart:a#s0_p0", "role": "primary"}],
        ),
        _mk_chunk(
            "c_00002",
            ["cognitive-load"],
            refs=[{"sourceId": "dart:b#s0_p0", "role": "primary"}],
        ),
    ]
    graph = _build_graph(chunks)
    by_id = {n["id"]: n for n in graph["nodes"]}
    node = by_id["cognitive-load"]
    # Occurrences sorted ASC → c_00001 first → its refs copied.
    assert node["occurrences"][0] == "c_00001"
    assert node["source_refs"][0]["sourceId"] == "dart:a#s0_p0"


def test_graph_node_source_refs_independent_of_chunk_mutation(monkeypatch):
    """Mutating node.source_refs must NOT change the underlying chunk."""
    from Trainforge.rag import typed_edge_inference

    monkeypatch.setattr(typed_edge_inference, "SCOPE_CONCEPT_IDS", False)

    ref_dict = {"sourceId": "dart:a#s0_p0", "role": "primary"}
    chunks = [
        _mk_chunk("c_00001", ["cognitive-load"], refs=[ref_dict]),
        _mk_chunk("c_00002", ["cognitive-load"], refs=[ref_dict]),
    ]
    graph = _build_graph(chunks)
    by_id = {n["id"]: n for n in graph["nodes"]}
    node = by_id["cognitive-load"]

    # Mutate the node copy.
    node["source_refs"][0]["role"] = "contributing"
    # Original chunk ref must stay 'primary'.
    assert chunks[0]["source"]["source_references"][0]["role"] == "primary"


# --------------------------------------------------------------------- #
# Schema round-trip — produced chunk validates under chunk_v4
# --------------------------------------------------------------------- #


def test_produced_chunks_validate_against_chunk_v4_schema():
    """Round-trip: process Wave-9 HTML → chunks produced pass strict."""
    jsonschema = pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator, RefResolver

    schemas_dir = PROJECT_ROOT / "schemas"
    chunk_schema = jsonschema.Draft202012Validator(
        __import__("json").loads(
            (schemas_dir / "knowledge" / "chunk_v4.schema.json").read_text()
        )
    )
    # Build resolver with remote $refs preloaded (source_reference + taxonomies).
    import json as _json

    with open(schemas_dir / "knowledge" / "chunk_v4.schema.json") as f:
        schema = _json.load(f)
    store: Dict[str, Any] = {}
    for p in schemas_dir.rglob("*.json"):
        try:
            with open(p) as f:
                s = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            store[sid] = s
    resolver = RefResolver.from_schema(schema, store=store)
    validator = Draft202012Validator(schema, resolver=resolver)

    chunks = _chunk_from_html(WAVE9_HTML_FULL)
    assert chunks
    for chunk in chunks:
        errors = list(validator.iter_errors(chunk))
        assert errors == [], [e.message for e in errors]
