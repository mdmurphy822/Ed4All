"""Wave 10 — _merge_small_sections source_references union tests.

When ``CourseProcessor._merge_small_sections`` collapses 2+ small
adjacent sections into a single chunk, the chunk's ``source_references[]``
must be the UNION of every merged section's sourceIds (deduped,
insertion-order preserved). Role-precedence is carried by the
underlying SourceReference entries — sections that contribute stay as
'contributing', primary sections stay primary.

Three contracts locked here:

1. Array shape: a chunk that absorbed multiple sections carries
   ``source_references`` as an array with every participating sourceId.
2. Dedupe: if two merged sections reference the same sourceId, the
   chunk lists it once.
3. Role precedence preservation: when a JSON-LD page-level ref says
   "primary" for a sourceId and a section-level data-cf-source-ids attr
   also lists it, the chunk keeps 'primary' (first-seen wins, JSON-LD
   comes first).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------- #
# Helpers — build a processor ready to exercise _merge_small_sections
# --------------------------------------------------------------------- #


def _make_processor(min_size: int = 200, max_size: int = 1000):
    from Trainforge.process_course import CourseProcessor

    processor = object.__new__(CourseProcessor)
    processor.capture = SimpleNamespace(
        run_id="test_run_w10_merge",
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
    processor.MIN_CHUNK_SIZE = min_size
    processor.MAX_CHUNK_SIZE = max_size
    processor.TARGET_CHUNK_SIZE = CourseProcessor.TARGET_CHUNK_SIZE
    return processor


def _mk_section(heading: str, content: str, source_ids: List[str]):
    """Shape a ContentSection-like object (duck-typed)."""
    from Trainforge.parsers.html_content_parser import ContentSection
    return ContentSection(
        heading=heading,
        level=2,
        content=content,
        word_count=len(content.split()),
        source_references=list(source_ids),
    )


# --------------------------------------------------------------------- #
# Unit tests on _merge_small_sections output shape
# --------------------------------------------------------------------- #


def test_merge_returns_4_tuples_with_source_ids():
    """Contract: (heading, text, chunk_type, merged_source_ids)."""
    processor = _make_processor()
    sections = [
        _mk_section("Sec A", "short text A", ["dart:a#s0_p0"]),
        _mk_section("Sec B", "short text B", ["dart:b#s0_p0"]),
    ]
    merged = processor._merge_small_sections(sections)
    assert merged, "expected at least one merged tuple"
    for entry in merged:
        assert len(entry) == 4
        heading, text, chunk_type, source_ids = entry
        assert isinstance(source_ids, list)


def test_merge_unions_source_ids_across_sections():
    """Two merged sections → union of their sourceIds on the chunk."""
    processor = _make_processor(max_size=1000)
    # Small sections so they merge into one chunk.
    sections = [
        _mk_section("Intro", "short intro text", ["dart:a#s0_p0"]),
        _mk_section("Body", "short body text", ["dart:b#s0_p0"]),
        _mk_section("Outro", "short outro text", ["dart:c#s0_p0"]),
    ]
    merged = processor._merge_small_sections(sections)
    assert len(merged) == 1, f"Expected 1 merged tuple, got {len(merged)}"
    _, _, _, source_ids = merged[0]
    assert set(source_ids) == {
        "dart:a#s0_p0",
        "dart:b#s0_p0",
        "dart:c#s0_p0",
    }


def test_merge_dedupes_duplicate_source_ids():
    """If adjacent sections share a sourceId, it's listed once."""
    processor = _make_processor(max_size=1000)
    sections = [
        _mk_section("A", "sA text", ["dart:shared#s0_p0", "dart:a#s0_p0"]),
        _mk_section("B", "sB text", ["dart:shared#s0_p0", "dart:b#s0_p0"]),
    ]
    merged = processor._merge_small_sections(sections)
    _, _, _, source_ids = merged[0]
    assert source_ids.count("dart:shared#s0_p0") == 1
    # All three unique IDs present.
    assert set(source_ids) == {
        "dart:shared#s0_p0",
        "dart:a#s0_p0",
        "dart:b#s0_p0",
    }


def test_merge_preserves_insertion_order():
    """Dedupe retains first-seen order so downstream diffs stay stable."""
    processor = _make_processor(max_size=1000)
    sections = [
        _mk_section("A", "text a", ["dart:first#s0_p0"]),
        _mk_section("B", "text b", ["dart:second#s0_p0"]),
        _mk_section("C", "text c", ["dart:first#s0_p0", "dart:third#s0_p0"]),
    ]
    merged = processor._merge_small_sections(sections)
    _, _, _, source_ids = merged[0]
    assert source_ids == [
        "dart:first#s0_p0",
        "dart:second#s0_p0",
        "dart:third#s0_p0",
    ]


def test_merge_empty_source_ids_when_no_refs():
    """Sections without source_references → empty list on merged tuple."""
    processor = _make_processor(max_size=1000)
    sections = [
        _mk_section("A", "text a", []),
        _mk_section("B", "text b", []),
    ]
    merged = processor._merge_small_sections(sections)
    _, _, _, source_ids = merged[0]
    assert source_ids == []


def test_merge_respects_max_chunk_size_boundary():
    """Sections that would exceed MAX merge size stay in separate chunks."""
    # word counts: 150 + 150 > 200, so they split.
    processor = _make_processor(max_size=200)
    sections = [
        _mk_section(
            "First", " ".join(["w"] * 150), ["dart:a#s0_p0"]
        ),
        _mk_section(
            "Second", " ".join(["w"] * 150), ["dart:b#s0_p0"]
        ),
    ]
    merged = processor._merge_small_sections(sections)
    # Two separate chunks → two separate source_ids lists.
    assert len(merged) == 2
    assert merged[0][3] == ["dart:a#s0_p0"]
    assert merged[1][3] == ["dart:b#s0_p0"]


# --------------------------------------------------------------------- #
# Round-trip: merged chunk's source_references reflects role precedence
# --------------------------------------------------------------------- #


def test_merged_chunk_preserves_primary_role_from_page_jsonld():
    """Section-level data-cf-source-ids (auto-roled contributing) must NOT
    downgrade a page-level JSON-LD 'primary' reference to contributing.

    The precedence order in _resolve_chunk_source_references is:
      1. page-level JSON-LD refs (full shape)
      2. section-level JSON-LD refs (per heading match)
      3. section-level data-cf-source-ids strings (auto-roled contributing)

    First-seen wins on sourceId collision.
    """

    processor = _make_processor(max_size=1000)
    item: Dict[str, Any] = {
        "module_id": "m",
        "module_title": "M",
        "item_id": "l",
        "title": "L",
        "resource_type": "page",
        "learning_objectives": [],
        "courseforge_metadata": {
            "sourceReferences": [
                {"sourceId": "dart:shared#s0_p0", "role": "primary"},
            ],
        },
        "sections": [],
        "misconceptions": [],
        "item_path": "m/l.html",
        "source_references": [
            {"sourceId": "dart:shared#s0_p0", "role": "primary"},
        ],
    }

    refs = processor._resolve_chunk_source_references(
        item=item,
        section_heading="Some Section",
        section_source_ids=["dart:shared#s0_p0"],  # also seen in HTML
    )
    # dart:shared must appear once and keep 'primary'.
    assert len(refs) == 1
    assert refs[0]["sourceId"] == "dart:shared#s0_p0"
    assert refs[0]["role"] == "primary"


def test_merged_chunk_contributing_role_for_html_only_refs():
    """HTML-attr-only ids (not in page JSON-LD) become 'contributing'."""
    processor = _make_processor(max_size=1000)
    item: Dict[str, Any] = {
        "module_id": "m",
        "module_title": "M",
        "item_id": "l",
        "title": "L",
        "resource_type": "page",
        "learning_objectives": [],
        "courseforge_metadata": {},
        "sections": [],
        "misconceptions": [],
        "item_path": "m/l.html",
        "source_references": [],
    }
    refs = processor._resolve_chunk_source_references(
        item=item,
        section_heading="Section",
        section_source_ids=["dart:html_only#s0_p0"],
    )
    assert len(refs) == 1
    assert refs[0]["role"] == "contributing"
    assert refs[0]["sourceId"] == "dart:html_only#s0_p0"


def test_merged_chunk_multi_role_mix():
    """Merged chunk carries primary (JSON-LD) + contributing (HTML)."""
    processor = _make_processor(max_size=1000)
    item: Dict[str, Any] = {
        "module_id": "m",
        "module_title": "M",
        "item_id": "l",
        "title": "L",
        "resource_type": "page",
        "learning_objectives": [],
        "courseforge_metadata": {},
        "sections": [],
        "misconceptions": [],
        "item_path": "m/l.html",
        "source_references": [
            {"sourceId": "dart:primary_ref#s0_p0", "role": "primary"},
        ],
    }
    refs = processor._resolve_chunk_source_references(
        item=item,
        section_heading="Section",
        section_source_ids=[
            "dart:primary_ref#s0_p0",  # shared with JSON-LD
            "dart:html_ref#s0_p0",     # HTML-only
        ],
    )
    by_sid = {r["sourceId"]: r for r in refs}
    assert by_sid["dart:primary_ref#s0_p0"]["role"] == "primary"
    assert by_sid["dart:html_ref#s0_p0"]["role"] == "contributing"


def test_section_jsonld_override_resolves():
    """Section JSON-LD refs resolve when heading matches."""
    processor = _make_processor(max_size=1000)
    item: Dict[str, Any] = {
        "module_id": "m",
        "module_title": "M",
        "item_id": "l",
        "title": "L",
        "resource_type": "page",
        "learning_objectives": [],
        "courseforge_metadata": {
            "sections": [
                {
                    "heading": "Target Section",
                    "sourceReferences": [
                        {
                            "sourceId": "dart:section_only#s0_p0",
                            "role": "corroborating",
                        }
                    ],
                }
            ],
        },
        "sections": [],
        "misconceptions": [],
        "item_path": "m/l.html",
        "source_references": [],  # page-level empty
    }
    refs = processor._resolve_chunk_source_references(
        item=item,
        section_heading="Target Section",
        section_source_ids=[],
    )
    assert len(refs) == 1
    assert refs[0]["sourceId"] == "dart:section_only#s0_p0"
    assert refs[0]["role"] == "corroborating"


def test_part_suffix_strips_when_matching_section_heading():
    """(part 2) suffix on chunk heading strips for section lookup."""
    processor = _make_processor(max_size=1000)
    item: Dict[str, Any] = {
        "module_id": "m",
        "module_title": "M",
        "item_id": "l",
        "title": "L",
        "resource_type": "page",
        "learning_objectives": [],
        "courseforge_metadata": {
            "sections": [
                {
                    "heading": "Long Section",
                    "sourceReferences": [
                        {"sourceId": "dart:long#s0_p0", "role": "primary"},
                    ],
                }
            ],
        },
        "sections": [],
        "misconceptions": [],
        "item_path": "m/l.html",
        "source_references": [],
    }
    refs = processor._resolve_chunk_source_references(
        item=item,
        section_heading="Long Section (part 2)",
        section_source_ids=[],
    )
    assert len(refs) == 1
    assert refs[0]["sourceId"] == "dart:long#s0_p0"
