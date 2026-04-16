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

import json
import pytest

from Trainforge.parsers.html_content_parser import (
    ContentSection,
    HTMLContentParser,
    LearningObjective,
    ParsedHTMLModule,
)
from Trainforge.generators.content_extractor import ContentExtractor


# ---------------------------------------------------------------------------
# Fixtures: HTML samples
# ---------------------------------------------------------------------------

COURSEFORGE_HTML_WITH_JSONLD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Week 2: Constructivism &mdash; DIGPED_101</title>
  <script type="application/ld+json">
  {
    "@context": "https://ed4all.dev/ns/courseforge/v1",
    "@type": "CourseModule",
    "courseCode": "DIGPED_101",
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
    <div class="objectives" role="region" aria-label="Learning Objectives"
         data-cf-objectives-count="2">
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
        assert cf["courseCode"] == "DIGPED_101"
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
