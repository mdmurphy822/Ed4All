"""
Tests for Courseforge metadata extraction in Trainforge pipeline.

Verifies:
  - JSON-LD extraction from HTML
  - data-cf-* attribute parsing
  - Priority chain: JSON-LD > data-attr > regex
  - Backward compatibility with non-Courseforge HTML
  - Chunk metadata enrichment (bloom_level, content_type_label, key_terms)
  - ContentExtractor metadata-first extraction
"""


from Trainforge.generators.content_extractor import ContentExtractor
from Trainforge.parsers.html_content_parser import (
    HTMLContentParser,
)

# ---------------------------------------------------------------------------
# Fixtures: HTML samples
# ---------------------------------------------------------------------------

COURSEFORGE_HTML_WITH_JSONLD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Week 2: Constructivism &mdash; SYNTHETIC_COURSE</title>
  <script type="application/ld+json">
  {
    "@context": "https://ed4all.dev/ns/courseforge/v1",
    "@type": "CourseModule",
    "courseCode": "SYNTHETIC_COURSE",
    "weekNumber": 2,
    "moduleType": "content",
    "pageId": "week_02_content_01_constructivism",
    "learningObjectives": [
      {
        "id": "CO-01",
        "statement": "Describe the key principles of constructivist learning theory",
        "bloomLevel": "understand",
        "bloomVerb": "describe",
        "cognitiveDomain": "conceptual",
        "keyConcepts": ["constructivism", "scaffolding"],
        "assessmentSuggestions": ["multiple_choice", "short_answer"]
      },
      {
        "id": "CO-02",
        "statement": "Apply scaffolding techniques in instructional design",
        "bloomLevel": "apply",
        "bloomVerb": "apply",
        "cognitiveDomain": "procedural",
        "assessmentSuggestions": ["short_answer", "essay"]
      }
    ],
    "sections": [
      {
        "heading": "Constructivist Learning Theory",
        "contentType": "explanation",
        "bloomRange": ["understand"],
        "keyTerms": [
          {"term": "constructivism", "definition": "A theory that learners actively construct knowledge through experience"},
          {"term": "scaffolding", "definition": "Temporary support structures that help learners reach higher levels"}
        ]
      }
    ],
    "misconceptions": [
      {"misconception": "Constructivism means no direct instruction", "correction": "Guided discovery includes teacher facilitation"}
    ],
    "suggestedAssessmentTypes": ["multiple_choice", "short_answer", "essay"]
  }
  </script>
</head>
<body>
  <main>
    <h1>Week 2: Constructivism</h1>
    <div class="objectives" role="region" aria-label="Learning Objectives">
      <h2>Learning Objectives</h2>
      <ul>
        <li data-cf-objective-id="CO-01" data-cf-bloom-level="understand"
            data-cf-bloom-verb="describe" data-cf-cognitive-domain="conceptual">
          <strong>CO-01:</strong> Describe the key principles of constructivist learning theory</li>
        <li data-cf-objective-id="CO-02" data-cf-bloom-level="apply"
            data-cf-bloom-verb="apply" data-cf-cognitive-domain="procedural">
          <strong>CO-02:</strong> Apply scaffolding techniques in instructional design</li>
      </ul>
    </div>
    <h2 data-cf-content-type="explanation" data-cf-key-terms="constructivism,scaffolding"
        data-cf-bloom-range="understand">Constructivist Learning Theory</h2>
    <p>Constructivism is a theory that learners actively construct knowledge.</p>
    <p>Scaffolding provides temporary support to help learners.</p>
  </main>
</body>
</html>"""

COURSEFORGE_HTML_DATA_ATTRS_ONLY = """<!DOCTYPE html>
<html lang="en">
<head><title>Week 3 &mdash; TEST_101</title></head>
<body>
  <main>
    <h1>Week 3: ADDIE Model</h1>
    <div class="objectives" role="region" aria-label="Learning Objectives">
      <h2>Learning Objectives</h2>
      <ul>
        <li data-cf-objective-id="CO-05" data-cf-bloom-level="analyze"
            data-cf-bloom-verb="analyze" data-cf-cognitive-domain="conceptual">
          <strong>CO-05:</strong> Analyze the strengths and limitations of the ADDIE model</li>
      </ul>
    </div>
    <h2 data-cf-content-type="comparison" data-cf-key-terms="addie,sam">Comparing ADDIE and SAM</h2>
    <p>The ADDIE model follows a linear approach to instructional design.</p>
  </main>
</body>
</html>"""

PLAIN_HTML_NO_METADATA = """<!DOCTYPE html>
<html lang="en">
<head><title>Generic Course Module</title></head>
<body>
  <main>
    <h1>Introduction to Assessment</h1>
    <div>
      <h2>Learning Objectives</h2>
      <ul>
        <li>Define formative and summative assessment</li>
        <li>Explain the role of rubrics in grading</li>
      </ul>
    </div>
    <h2>Formative Assessment</h2>
    <p>Formative assessment is an ongoing process of evaluation.</p>
  </main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Tests: JSON-LD extraction
# ---------------------------------------------------------------------------

class TestJSONLDExtraction:
    def setup_method(self):
        self.parser = HTMLContentParser()

    def test_extracts_json_ld_metadata(self):
        result = self.parser.parse(COURSEFORGE_HTML_WITH_JSONLD)
        assert "courseforge" in result.metadata
        cf = result.metadata["courseforge"]
        assert cf["courseCode"] == "SYNTHETIC_COURSE"
        assert cf["weekNumber"] == 2
        assert cf["moduleType"] == "content"
        assert cf["pageId"] == "week_02_content_01_constructivism"

    def test_extracts_page_id(self):
        result = self.parser.parse(COURSEFORGE_HTML_WITH_JSONLD)
        assert result.page_id == "week_02_content_01_constructivism"

    def test_extracts_misconceptions(self):
        result = self.parser.parse(COURSEFORGE_HTML_WITH_JSONLD)
        assert len(result.misconceptions) == 1
        assert result.misconceptions[0]["misconception"] == "Constructivism means no direct instruction"

    def test_extracts_suggested_assessment_types(self):
        result = self.parser.parse(COURSEFORGE_HTML_WITH_JSONLD)
        assert "multiple_choice" in result.suggested_assessment_types
        assert "essay" in result.suggested_assessment_types

    def test_no_json_ld_returns_empty_metadata(self):
        result = self.parser.parse(PLAIN_HTML_NO_METADATA)
        assert "courseforge" not in result.metadata
        assert result.page_id is None
        assert result.misconceptions == []


# ---------------------------------------------------------------------------
# Tests: Objective extraction priority chain
# ---------------------------------------------------------------------------

class TestObjectiveExtraction:
    def setup_method(self):
        self.parser = HTMLContentParser()

    def test_json_ld_objectives_take_priority(self):
        result = self.parser.parse(COURSEFORGE_HTML_WITH_JSONLD)
        assert len(result.learning_objectives) == 2
        co1 = result.learning_objectives[0]
        assert co1.id == "CO-01"
        assert co1.bloom_level == "understand"
        assert co1.bloom_verb == "describe"
        assert co1.cognitive_domain == "conceptual"
        assert "constructivism" in co1.key_concepts
        assert "multiple_choice" in co1.assessment_suggestions

    def test_data_attr_objectives_when_no_json_ld(self):
        result = self.parser.parse(COURSEFORGE_HTML_DATA_ATTRS_ONLY)
        assert len(result.learning_objectives) >= 1
        co5 = result.learning_objectives[0]
        assert co5.id == "CO-05"
        assert co5.bloom_level == "analyze"
        assert co5.bloom_verb == "analyze"

    def test_regex_fallback_for_plain_html(self):
        result = self.parser.parse(PLAIN_HTML_NO_METADATA)
        assert len(result.learning_objectives) >= 1
        # Should detect bloom from verb
        lo = result.learning_objectives[0]
        assert lo.text.startswith("Define")
        assert lo.bloom_level == "remember"
        assert lo.bloom_verb == "define"


# ---------------------------------------------------------------------------
# Tests: Section metadata extraction
# ---------------------------------------------------------------------------

class TestSectionMetadata:
    def setup_method(self):
        self.parser = HTMLContentParser()

    def test_data_cf_content_type_on_sections(self):
        result = self.parser.parse(COURSEFORGE_HTML_WITH_JSONLD)
        explanation_sections = [s for s in result.sections if s.content_type == "explanation"]
        assert len(explanation_sections) >= 1

    def test_data_cf_key_terms_on_sections(self):
        result = self.parser.parse(COURSEFORGE_HTML_WITH_JSONLD)
        section = next(
            (s for s in result.sections if s.content_type == "explanation"), None
        )
        assert section is not None
        assert "constructivism" in section.key_terms
        assert "scaffolding" in section.key_terms

    def test_comparison_content_type(self):
        result = self.parser.parse(COURSEFORGE_HTML_DATA_ATTRS_ONLY)
        comparison_sections = [s for s in result.sections if s.content_type == "comparison"]
        assert len(comparison_sections) >= 1

    def test_plain_html_sections_have_no_content_type(self):
        result = self.parser.parse(PLAIN_HTML_NO_METADATA)
        for section in result.sections:
            assert section.content_type is None


# ---------------------------------------------------------------------------
# Tests: ContentExtractor metadata-first extraction
# ---------------------------------------------------------------------------

class TestContentExtractorMetadata:
    def setup_method(self):
        self.extractor = ContentExtractor()

    def test_extract_from_metadata_returns_key_terms(self):
        chunks = [
            {
                "id": "chunk_001",
                "text": "Some text about constructivism.",
                "key_terms": [
                    {"term": "constructivism", "definition": "A theory of active knowledge construction"},
                    {"term": "scaffolding", "definition": "Temporary learning support"},
                ],
            }
        ]
        result = self.extractor.extract_from_metadata(chunks)
        assert len(result["key_terms"]) == 2
        assert result["key_terms"][0].term == "constructivism"
        assert result["key_terms"][0].definition == "A theory of active knowledge construction"

    def test_extract_from_metadata_returns_misconceptions(self):
        chunks = [
            {
                "id": "chunk_001",
                "text": "Text",
                "misconceptions": [
                    {"misconception": "Learning is passive", "correction": "Learning is active"},
                ],
            }
        ]
        result = self.extractor.extract_from_metadata(chunks)
        assert len(result["misconceptions"]) == 1
        assert result["misconceptions"][0]["misconception"] == "Learning is passive"

    def test_extract_from_metadata_returns_bloom_levels(self):
        chunks = [
            {"id": "c1", "text": "Text", "bloom_level": "understand"},
            {"id": "c2", "text": "Text", "bloom_level": "apply"},
        ]
        result = self.extractor.extract_from_metadata(chunks)
        assert "understand" in result["bloom_levels"]
        assert "apply" in result["bloom_levels"]

    def test_extract_key_terms_prefers_metadata(self):
        """When chunks have structured key_terms, regex extraction is bypassed."""
        chunks = [
            {
                "id": "chunk_001",
                "text": "Random text with no definition patterns.",
                "key_terms": [
                    {"term": "metacognition", "definition": "Thinking about thinking"},
                ],
            }
        ]
        terms = self.extractor.extract_key_terms(chunks)
        assert len(terms) == 1
        assert terms[0].term == "metacognition"

    def test_extract_key_terms_falls_back_to_regex(self):
        """Without metadata, regex extraction is used."""
        chunks = [
            {
                "id": "chunk_001",
                "text": "Behaviorism is defined as a learning theory focused on observable behavior.",
                "concept_tags": ["behaviorism"],
            }
        ]
        terms = self.extractor.extract_key_terms(chunks)
        assert len(terms) >= 1
        assert any("behaviorism" in t.term.lower() for t in terms)

    def test_empty_metadata_falls_back(self):
        """Chunks without key_terms metadata use regex."""
        chunks = [
            {
                "id": "chunk_001",
                "text": "Scaffolding refers to temporary support structures in education.",
                "concept_tags": ["scaffolding"],
            }
        ]
        terms = self.extractor.extract_key_terms(chunks)
        assert len(terms) >= 1


# ---------------------------------------------------------------------------
# Tests: Backward compatibility
# ---------------------------------------------------------------------------

class TestMultiPartChunkMetadata:
    """Verify that multi-part chunk headings with (part N) suffix still match metadata."""

    def test_part_suffix_stripped_for_matching(self):
        import re
        heading = "Constructivist Learning Theory (part 2)"
        normalized = re.sub(r'\s*\(part\s+\d+\)\s*$', '', heading).lower()
        assert normalized == "constructivist learning theory"

    def test_heading_without_part_unchanged(self):
        import re
        heading = "Constructivist Learning Theory"
        normalized = re.sub(r'\s*\(part\s+\d+\)\s*$', '', heading).lower()
        assert normalized == "constructivist learning theory"

    def test_high_part_number(self):
        import re
        heading = "Advanced Topics (part 15)"
        normalized = re.sub(r'\s*\(part\s+\d+\)\s*$', '', heading).lower()
        assert normalized == "advanced topics"

    def test_parenthetical_not_part_preserved(self):
        """Headings with non-part parenthetical should NOT be stripped."""
        import re
        heading = "ADDIE Model (Overview)"
        normalized = re.sub(r'\s*\(part\s+\d+\)\s*$', '', heading).lower()
        assert normalized == "addie model (overview)"

    def test_none_key_terms_handled(self):
        """Chunks with explicit None key_terms should not crash extractor."""
        extractor = ContentExtractor()
        chunks = [{"id": "c1", "text": "Some text.", "key_terms": None}]
        result = extractor.extract_from_metadata(chunks)
        assert result["key_terms"] == []

    def test_none_misconceptions_handled(self):
        extractor = ContentExtractor()
        chunks = [{"id": "c1", "text": "Some text.", "misconceptions": None}]
        result = extractor.extract_from_metadata(chunks)
        assert result["misconceptions"] == []


class TestBackwardCompatibility:
    def setup_method(self):
        self.parser = HTMLContentParser()

    def test_plain_html_still_extracts_objectives(self):
        result = self.parser.parse(PLAIN_HTML_NO_METADATA)
        assert len(result.learning_objectives) >= 1

    def test_plain_html_still_extracts_sections(self):
        result = self.parser.parse(PLAIN_HTML_NO_METADATA)
        assert len(result.sections) >= 1

    def test_plain_html_still_extracts_concepts(self):
        result = self.parser.parse(PLAIN_HTML_NO_METADATA)
        # word_count should be reasonable
        assert result.word_count > 0

    def test_new_fields_default_empty(self):
        result = self.parser.parse(PLAIN_HTML_NO_METADATA)
        assert result.page_id is None
        assert result.misconceptions == []
        assert result.prerequisite_pages == []
        assert result.suggested_assessment_types == []


# ---------------------------------------------------------------------------
# Wave 5 W5.B — chunker ingestion mirror for W1.5 key_claims + W1.7
# objective_alignment audit fields
# ---------------------------------------------------------------------------

class TestExtractSectionMetadataW5B:
    """W5.B chunker-side mirror of W5.A. ``_extract_section_metadata``
    bumped from a 4-tuple to a 5-tuple so the chunker can stamp the
    optional ``key_claims`` / ``objective_alignment`` audit arrays onto
    every chunk dict whose originating JSON-LD ``blocks[]`` entry
    declares them. Fixtures mirror the W5.A test patterns under
    ``test_html_content_parser.py``."""

    def _bare_processor(self):
        from Trainforge.process_course import CourseProcessor

        proc = CourseProcessor.__new__(CourseProcessor)
        proc.MIN_CHUNK_SIZE = 100
        proc.MAX_CHUNK_SIZE = 800
        return proc

    def _item_with_jsonld_blocks(self, *blocks):
        """Build an item whose courseforge_metadata has a JSON-LD
        ``blocks[]`` array of the given shape."""
        return {
            "title": "Page Title",
            "courseforge_metadata": {"blocks": list(blocks)},
            "sections": [],
        }

    def test_extract_section_metadata_carries_key_claims_from_jsonld_blocks(self):
        """When a JSON-LD ``blocks[]`` entry matches the section heading and
        carries ``keyClaims``, the 4th tuple element (section_metadata_extras)
        carries them under the snake_case key ``key_claims``."""
        proc = self._bare_processor()
        block = {
            "heading": "Constructivist Theory",
            "contentType": "explanation",
            "keyClaims": [
                {
                    "claim": "Knowledge is actively constructed by learners.",
                    "source_chunk_ids": ["chunk_001", "chunk_002"],
                },
                {"claim": "Scaffolding fades as expertise grows."},
            ],
        }
        item = self._item_with_jsonld_blocks(block)
        bloom, ctype, kt, extras, trace = proc._extract_section_metadata(
            item, "Constructivist Theory"
        )
        assert ctype == "explanation"
        assert extras["key_claims"] == [
            {
                "claim": "Knowledge is actively constructed by learners.",
                "source_chunk_ids": ["chunk_001", "chunk_002"],
            },
            {"claim": "Scaffolding fades as expertise grows."},
        ]
        # Trace pins which path populated the field.
        assert trace["key_claims"] == "jsonld_blocks_match"
        # objective_alignment was absent on the block — extras default to [].
        assert extras["objective_alignment"] == []
        assert trace["objective_alignment"] == "none"

    def test_extract_section_metadata_carries_objective_alignment_from_jsonld_blocks(self):
        """Same shape as key_claims — when a matched block carries
        ``objectiveAlignment``, the extras dict surfaces it under the
        snake_case key ``objective_alignment``."""
        proc = self._bare_processor()
        block = {
            "heading": "ADDIE Model",
            "contentType": "comparison",
            "objectiveAlignment": [
                {
                    "objective_id": "CO-01",
                    "status": "delivered",
                    "declared_bloom": "analyze",
                },
                {
                    "objective_id": "CO-02",
                    "status": "partial",
                    "declared_bloom": "apply",
                },
            ],
        }
        item = self._item_with_jsonld_blocks(block)
        _, _, _, extras, trace = proc._extract_section_metadata(item, "ADDIE Model")
        assert extras["objective_alignment"] == [
            {
                "objective_id": "CO-01",
                "status": "delivered",
                "declared_bloom": "analyze",
            },
            {
                "objective_id": "CO-02",
                "status": "partial",
                "declared_bloom": "apply",
            },
        ]
        assert trace["objective_alignment"] == "jsonld_blocks_match"
        # key_claims absent on the block.
        assert extras["key_claims"] == []
        assert trace["key_claims"] == "none"

    def test_extract_section_metadata_omits_extras_when_block_lacks_fields(self):
        """Back-compat: a matched block with no keyClaims / objectiveAlignment
        leaves the extras dict at its default empty-list shape."""
        proc = self._bare_processor()
        block = {
            "heading": "Legacy Block",
            "contentType": "example",
            "keyTerms": [{"term": "alpha", "definition": "first letter"}],
        }
        item = self._item_with_jsonld_blocks(block)
        _, ctype, kt, extras, trace = proc._extract_section_metadata(
            item, "Legacy Block"
        )
        # Pre-W5 fields populated as before.
        assert ctype == "example"
        assert kt[0]["term"] == "alpha"
        # New fields default to empty lists, traces to "none".
        assert extras == {"key_claims": [], "objective_alignment": []}
        assert trace["key_claims"] == "none"
        assert trace["objective_alignment"] == "none"

    def test_extract_section_metadata_filters_non_dict_audit_entries(self):
        """Defensive filter — payloads that slip past schema validation
        (string entries in keyClaims / objectiveAlignment arrays) are
        silently dropped, not propagated. Mirrors the dict-only filter
        in ``html_content_parser._content_sections_from_blocks``."""
        proc = self._bare_processor()
        block = {
            "heading": "Defensive Block",
            "contentType": "explanation",
            "keyClaims": [
                {"claim": "valid claim"},
                "string entry that should be filtered",
                None,
            ],
            "objectiveAlignment": [
                {"objective_id": "TO-01", "status": "delivered"},
                42,
            ],
        }
        item = self._item_with_jsonld_blocks(block)
        _, _, _, extras, _ = proc._extract_section_metadata(item, "Defensive Block")
        assert extras["key_claims"] == [{"claim": "valid claim"}]
        assert extras["objective_alignment"] == [
            {"objective_id": "TO-01", "status": "delivered"},
        ]


def _create_chunk_processor():
    """Stand up a CourseProcessor minimally enough for ``_create_chunk``
    to run end-to-end. Mirrors the pattern in
    ``test_targeted_concepts_end_to_end.py::test_chunk_level_targeted_concepts_propagation``."""
    from collections import defaultdict

    from Trainforge.process_course import CourseProcessor

    proc = CourseProcessor.__new__(CourseProcessor)
    proc.course_code = "TST_101"
    proc.MIN_CHUNK_SIZE = 100
    proc.MAX_CHUNK_SIZE = 800
    proc.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": defaultdict(int),
        "difficulty_distribution": defaultdict(int),
    }
    proc._all_concept_tags = set()
    proc._valid_outcome_ids = set()
    proc.domain_concept_seeds = []
    # ``__init__`` would have set these; the bare-processor pattern bypasses
    # __init__ so we stub the minimum surface ``_create_chunk`` touches.
    proc.objectives = None
    proc._lo_parent_map = {}
    return proc


def _item_for_create_chunk(blocks=None, sections=None):
    """Minimal item dict shape that ``_create_chunk`` will accept."""
    cf_meta = {"blocks": blocks} if blocks is not None else None
    return {
        "item_id": "lesson-1",
        "item_path": "lesson-1.html",
        "module_id": "m1",
        "module_title": "Module 1",
        "title": "Page Title",
        "resource_type": "page",
        "raw_html": "",
        "sections": sections or [],
        "learning_objectives": [],
        "courseforge_metadata": cf_meta,
        "misconceptions": [],
        "_jsonld_tag_present": cf_meta is not None,
        "_jsonld_parse_failed": False,
    }


class TestCreateChunkW5BStamp:
    """End-to-end: ContentSection -> JSON-LD blocks -> metadata extraction
    -> chunk dict carries the W5.B audit fields. Conditional on truthy —
    empty lists are absent (preserves back-compat)."""

    def test_create_chunk_stamps_key_claims_when_present(self):
        proc = _create_chunk_processor()
        block = {
            "heading": "Constructivist Theory",
            "contentType": "explanation",
            "keyClaims": [
                {
                    "claim": "Knowledge is actively constructed by learners.",
                    "source_chunk_ids": ["chunk_001"],
                },
                {"claim": "Scaffolding fades as expertise grows."},
            ],
        }
        item = _item_for_create_chunk(blocks=[block])

        chunk = proc._create_chunk(
            chunk_id="tst_101_chunk_00001",
            text=(
                "Constructivism holds that learners actively build knowledge "
                "through direct experience and scaffolded reflection."
            ),
            html="<p>Constructivism holds that learners actively build knowledge.</p>",
            item=item,
            section_heading="Constructivist Theory",
            chunk_type="explanation",
        )
        assert chunk["key_claims"] == [
            {
                "claim": "Knowledge is actively constructed by learners.",
                "source_chunk_ids": ["chunk_001"],
            },
            {"claim": "Scaffolding fades as expertise grows."},
        ]
        # _metadata_trace surfaces the W5.B trace entry.
        assert chunk["_metadata_trace"]["key_claims"] == "jsonld_blocks_match"
        # objective_alignment absent on the source block — chunk should
        # NOT carry the key at all (back-compat: not even as []).
        assert "objective_alignment" not in chunk
        assert chunk["_metadata_trace"]["objective_alignment"] == "none"

    def test_create_chunk_stamps_objective_alignment_when_present(self):
        proc = _create_chunk_processor()
        block = {
            "heading": "ADDIE Model",
            "contentType": "comparison",
            "objectiveAlignment": [
                {
                    "objective_id": "CO-01",
                    "status": "delivered",
                    "declared_bloom": "analyze",
                },
            ],
        }
        item = _item_for_create_chunk(blocks=[block])

        chunk = proc._create_chunk(
            chunk_id="tst_101_chunk_00002",
            text=(
                "The ADDIE model decomposes instructional design into five "
                "phases: analyze, design, develop, implement, and evaluate."
            ),
            html="<p>The ADDIE model decomposes instructional design.</p>",
            item=item,
            section_heading="ADDIE Model",
            chunk_type="explanation",
        )
        assert chunk["objective_alignment"] == [
            {
                "objective_id": "CO-01",
                "status": "delivered",
                "declared_bloom": "analyze",
            },
        ]
        assert chunk["_metadata_trace"]["objective_alignment"] == "jsonld_blocks_match"
        # key_claims absent on the source block.
        assert "key_claims" not in chunk
        assert chunk["_metadata_trace"]["key_claims"] == "none"

    def test_create_chunk_omits_new_fields_when_absent(self):
        """Back-compat: a chunk derived from a section / block with NO
        key_claims and NO objective_alignment must not carry the new keys
        at all (not even as []). Legacy-corpus chunks stay byte-identical."""
        proc = _create_chunk_processor()
        # No blocks[] at all → the JSON-LD blocks-walk doesn't enter; the
        # data-cf-* fallback returns a section with no audit arrays.
        item = _item_for_create_chunk(blocks=None, sections=[])

        chunk = proc._create_chunk(
            chunk_id="tst_101_chunk_00003",
            text=(
                "This passage carries no Wave-1.5 key-claim audit and no "
                "Wave-1.7 objective-alignment audit; the chunker should "
                "elide both keys entirely from the resulting chunk dict."
            ),
            html="<p>This passage carries no audit metadata.</p>",
            item=item,
            section_heading="Some Section",
            chunk_type="explanation",
        )
        assert "key_claims" not in chunk
        assert "objective_alignment" not in chunk
        # _metadata_trace still carries the trace entries (set to "none")
        # so a downstream observability surface can attribute the absence.
        assert chunk["_metadata_trace"]["key_claims"] == "none"
        assert chunk["_metadata_trace"]["objective_alignment"] == "none"


# Wave 5 W5.G — per-chunk learning_outcome_source_refs reverse map.
# Builds a {lo_id: [source_chunk_id, ...]} projection from the page-level
# JSON-LD ``learningObjectives[].sourceReferences[]`` so downstream
# consumers can resolve "which SemantiK/Courseforge chunk(s) source TO-05?"
# without re-loading synthesized_objectives.json off disk.


def _lo_obj(lo_id):
    """Minimal LearningObjective stand-in that exposes both the `.id`
    attribute (read by ``_extract_objective_refs``) AND the
    `.bloom_level` attribute (read by ``_determine_difficulty``)."""
    from types import SimpleNamespace
    return SimpleNamespace(id=lo_id, bloom_level=None)


def _item_for_w5g(jsonld_los, learning_objective_ids=None):
    """Item shape that exercises _create_chunk's W5.G stamp.

    ``jsonld_los`` populates ``courseforge_metadata.learningObjectives``
    (the source of the reverse map). ``learning_objective_ids`` is the
    list of LO id strings the parser would have produced; each one is
    wrapped into a ``SimpleNamespace`` LO stand-in so
    ``_extract_objective_refs`` populates the chunk's
    ``learning_outcome_refs`` and ``_determine_difficulty`` doesn't
    AttributeError on missing ``bloom_level``.
    """
    cf_meta: Dict[str, Any] = {"learningObjectives": list(jsonld_los)}
    los = [_lo_obj(lid) for lid in (learning_objective_ids or [])]
    return {
        "item_id": "lesson-1",
        "item_path": "lesson-1.html",
        "module_id": "m1",
        "module_title": "Module 1",
        "title": "Page Title",
        "resource_type": "page",
        "raw_html": "",
        "sections": [],
        "learning_objectives": los,
        "courseforge_metadata": cf_meta,
        "misconceptions": [],
        "_jsonld_tag_present": True,
        "_jsonld_parse_failed": False,
    }


class TestCreateChunkW5GLearningOutcomeSourceRefs:
    """End-to-end: page-level JSON-LD ``learningObjectives[].sourceReferences[]``
    + chunk's ``learning_outcome_refs[]`` -> chunk dict carries
    ``learning_outcome_source_refs`` reverse map. Conditional on truthy
    output — empty maps are elided so legacy corpora stay byte-identical."""

    def test_create_chunk_stamps_learning_outcome_source_refs_when_jsonld_carries_sourceReferences(self):
        proc = _create_chunk_processor()
        item = _item_for_w5g(
            jsonld_los=[
                {
                    "id": "TO-05",
                    "sourceReferences": [
                        {"sourceId": "semantik:rdf-primer-ch3#sec-2"},
                    ],
                },
            ],
            learning_objective_ids=["TO-05"],
        )
        chunk = proc._create_chunk(
            chunk_id="tst_101_chunk_00010",
            text=(
                "RDF triples bind a subject, predicate, and object into a "
                "single statement that downstream graphs can compose."
            ),
            html="<p>RDF triples bind subject + predicate + object.</p>",
            item=item,
            section_heading="RDF Graph",
            chunk_type="explanation",
        )
        assert chunk["learning_outcome_source_refs"] == {
            "TO-05": ["semantik:rdf-primer-ch3#sec-2"]
        }

    def test_create_chunk_omits_reverse_map_when_jsonld_lacks_sourceReferences(self):
        """Pre-Wave-1.6 page-level JSON-LD doesn't carry sourceReferences[]
        on its learningObjectives[] entries; chunk MUST NOT carry the
        reverse-map field at all (back-compat: not even as {})."""
        proc = _create_chunk_processor()
        item = _item_for_w5g(
            jsonld_los=[
                {"id": "TO-05"},  # no sourceReferences key at all
                {"id": "TO-06", "sourceReferences": []},  # empty list
            ],
            learning_objective_ids=["TO-05", "TO-06"],
        )
        chunk = proc._create_chunk(
            chunk_id="tst_101_chunk_00011",
            text=(
                "Legacy chunk text body without any per-LO source-reference "
                "attribution at the page-level JSON-LD layer."
            ),
            html="<p>Legacy chunk.</p>",
            item=item,
            section_heading="Legacy Section",
            chunk_type="explanation",
        )
        assert "learning_outcome_source_refs" not in chunk

    def test_create_chunk_handles_case_insensitive_lo_match(self):
        """Chunk's ``learning_outcome_refs`` lowercases by default
        (TRAINFORGE_PRESERVE_LO_CASE unset). The reverse map MUST still
        match those lowercased refs against the canonical-case JSON-LD
        ``id`` values, AND the output key must preserve the JSON-LD
        emit case (``TO-05``, not ``to-05``) so downstream consumers
        can canonicalise via the LO id pattern."""
        proc = _create_chunk_processor()
        item = _item_for_w5g(
            jsonld_los=[
                {
                    "id": "TO-05",  # canonical-cased emit form
                    "sourceReferences": [
                        {"sourceId": "semantik:chunk_alpha"},
                        {"sourceId": "semantik:chunk_beta"},
                    ],
                },
            ],
            # The parsed LearningObjective dict feeds _extract_objective_refs,
            # which lowercases under the default case policy. The chunk's
            # learning_outcome_refs ends up as ["to-05"] post-normalize.
            learning_objective_ids=["TO-05"],
        )
        chunk = proc._create_chunk(
            chunk_id="tst_101_chunk_00012",
            text=(
                "Sample text body for a chunk whose LO refs should match "
                "case-insensitively against the JSON-LD canonical-case ids."
            ),
            html="<p>Case-insensitive match.</p>",
            item=item,
            section_heading="Case Match",
            chunk_type="explanation",
        )
        # Chunk's own refs are lowercase per default case policy.
        assert "to-05" in chunk["learning_outcome_refs"]
        # But the reverse-map output key preserves the JSON-LD emit case.
        assert chunk["learning_outcome_source_refs"] == {
            "TO-05": ["semantik:chunk_alpha", "semantik:chunk_beta"],
        }

    def test_create_chunk_defensive_skips_malformed_jsonld_entries(self):
        """The helper MUST be defensive against missing/empty ``id``,
        non-list ``sourceReferences``, non-dict ``sourceReferences[]``
        entries, and missing/empty ``sourceId`` — all skipped silently;
        valid entries still process and land on the chunk dict.

        Test directly via the helper so the assertions exercise the
        helper's defensive surface independent of any orthogonal
        brittleness in sibling chunk-emit helpers (e.g.
        ``_determine_difficulty`` walks ``learningObjectives[]`` with
        ``.get(...)`` and would AttributeError on a non-dict entry).
        The end-to-end ``_create_chunk`` integration is covered by the
        three sibling W5.G tests above; this one tightens the helper's
        defensive contract one level deeper."""
        proc = _create_chunk_processor()
        item = _item_for_w5g(
            jsonld_los=[
                "not-a-dict",  # non-dict LO entry
                None,
                {"id": ""},  # empty id
                {"id": "TO-07", "sourceReferences": "not-a-list"},
                {
                    "id": "TO-08",
                    "sourceReferences": [
                        "not-a-dict-ref",
                        {"sourceId": ""},  # empty sourceId
                        {"sourceId": "semantik:valid_ref"},
                        {"notSourceId": "ignored"},
                    ],
                },
            ],
            learning_objective_ids=["TO-07", "TO-08"],
        )
        # Both lo_refs are present (case-folded under default policy):
        # the helper should match TO-07 and TO-08 against jsonld_los.
        reverse_map = proc._build_learning_outcome_source_refs(
            item, ["to-07", "to-08"]
        )
        # TO-07 has a non-list sourceReferences -> entirely skipped.
        # TO-08 has one valid sourceId among the malformed items.
        assert reverse_map == {"TO-08": ["semantik:valid_ref"]}

        # Bonus contract: an empty lo_refs returns {} (NOT None) so the
        # call site's ``if reverse_map:`` gate elides the field cleanly.
        assert proc._build_learning_outcome_source_refs(item, []) == {}
