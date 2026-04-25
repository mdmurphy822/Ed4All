"""
Wave 81 Worker C — content-generator spec misconception dual-emit tests.

These tests pin the **forward-looking** dual-emit contract documented in
``Courseforge/templates/chunk_templates.md`` Template 3 and the Wave 79
Template Catalog section of ``Courseforge/agents/content-generator.md``.

The contract: every ``common_pitfall`` chunk MUST emit BOTH the
``data-cf-misconception="true"`` HTML attribute AND a corresponding
JSON-LD ``misconceptions[]`` entry. The two arms are equivalent
semantics; both are required.

We can't test the live content-generator subagent here (that requires a
real Anthropic dispatch and is out of scope). What we CAN test is that:

  1. The spec text in ``chunk_templates.md`` documents the dual-emit
     requirement and includes a canonical JSON-LD example.
  2. The spec text in ``content-generator.md`` repeats the requirement
     under the Wave 79 Template Catalog section.
  3. The example JSON-LD in the spec parses cleanly and matches the
     misconceptions[] shape consumed by Trainforge's HTMLContentParser.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from Trainforge.parsers.html_content_parser import HTMLContentParser

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNK_TEMPLATES_MD = (
    REPO_ROOT / "Courseforge" / "templates" / "chunk_templates.md"
)
CONTENT_GENERATOR_MD = (
    REPO_ROOT / "Courseforge" / "agents" / "content-generator.md"
)


# ---------------------------------------------------------------------------
# Spec presence
# ---------------------------------------------------------------------------

class TestChunkTemplatesSpec:
    """The chunk_templates.md spec must explicitly mandate dual-emit."""

    def test_chunk_templates_md_mentions_dual_emit(self):
        text = CHUNK_TEMPLATES_MD.read_text(encoding="utf-8")
        # The spec must use the phrase "dual-emit" (or close cognate)
        # AND mention the data-cf-misconception attribute alongside the
        # JSON-LD misconceptions[] array.
        assert "dual-emit" in text.lower()
        assert "data-cf-misconception" in text
        assert "misconceptions[]" in text or '"misconceptions"' in text

    def test_chunk_templates_md_has_jsonld_example_for_template3(self):
        text = CHUNK_TEMPLATES_MD.read_text(encoding="utf-8")
        # Locate the Template 3 section.
        t3_match = re.search(
            r"## Template 3 — Common Pitfall(.*?)(?:^---$|## Template 4)",
            text,
            re.DOTALL | re.MULTILINE,
        )
        assert t3_match, "Template 3 section not found in chunk_templates.md"
        section = t3_match.group(1)

        # The Template 3 section must contain a JSON-LD example with
        # a misconceptions[] entry and the canonical fields.
        assert 'application/ld+json' in section
        assert '"misconceptions"' in section
        assert '"misconception"' in section
        assert '"correction"' in section
        assert '"bloom_level"' in section

    def test_chunk_templates_md_jsonld_example_parses_and_matches_shape(self):
        text = CHUNK_TEMPLATES_MD.read_text(encoding="utf-8")
        t3_match = re.search(
            r"## Template 3 — Common Pitfall(.*?)(?:^## Template 4)",
            text,
            re.DOTALL | re.MULTILINE,
        )
        assert t3_match, "Template 3 section not found"
        section = t3_match.group(1)

        # Find the JSON-LD block inside the Template 3 section.
        jsonld_match = re.search(
            r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
            section,
            re.DOTALL,
        )
        assert jsonld_match, "Template 3 JSON-LD example not found"

        # The example must parse as JSON.
        data = json.loads(jsonld_match.group(1))
        assert isinstance(data, dict)
        assert "misconceptions" in data
        assert isinstance(data["misconceptions"], list)
        assert len(data["misconceptions"]) >= 1

        # Each misconception entry must carry the required fields per
        # the spec's field-contract table.
        entry = data["misconceptions"][0]
        assert "misconception" in entry
        assert "correction" in entry
        assert "bloom_level" in entry

    def test_jsonld_example_consumable_by_trainforge_parser(self):
        """The exact JSON-LD example in the spec must round-trip through
        the same parser that harvests misconceptions for chunks. This is
        the load-bearing contract — the spec example is only useful if
        Trainforge can actually consume it."""
        text = CHUNK_TEMPLATES_MD.read_text(encoding="utf-8")
        t3_match = re.search(
            r"## Template 3 — Common Pitfall(.*?)(?:^## Template 4)",
            text,
            re.DOTALL | re.MULTILINE,
        )
        section = t3_match.group(1)
        jsonld_match = re.search(
            r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
            section,
            re.DOTALL,
        )
        jsonld_blob = jsonld_match.group(1)

        # Wrap the JSON-LD inside a minimal HTML5 page and feed it
        # through the parser.
        html = (
            "<!DOCTYPE html>"
            "<html><head>"
            "<meta charset=\"UTF-8\"><title>spec example</title>"
            f'<script type="application/ld+json">{jsonld_blob}</script>'
            "</head><body><h1>Spec Example</h1></body></html>"
        )
        parsed = HTMLContentParser().parse(html)

        assert len(parsed.misconceptions) >= 1
        mc = parsed.misconceptions[0]
        # Trainforge normalizes camelCase bloomLevel -> snake_case
        # bloom_level. The spec example uses snake_case directly so the
        # field appears verbatim.
        assert "misconception" in mc
        assert "correction" in mc
        assert mc.get("bloom_level") == "analyze"


class TestContentGeneratorSpec:
    """The content-generator.md spec must repeat the dual-emit
    requirement under the Wave 79 Template Catalog section."""

    def test_content_generator_md_mentions_dual_emit(self):
        text = CONTENT_GENERATOR_MD.read_text(encoding="utf-8")
        # Find the Common Pitfall bullet under Wave 79 Template Catalog
        # (numbered list item 3 in the catalog).
        assert "Common Pitfall" in text
        assert "dual-emit" in text.lower()
        # Both arms must be referenced explicitly.
        assert "data-cf-misconception" in text
        assert "misconceptions[]" in text or "misconceptions" in text

    def test_content_generator_md_references_template3_canonical_shape(self):
        text = CONTENT_GENERATOR_MD.read_text(encoding="utf-8")
        # The agent spec should point readers at chunk_templates.md
        # Template 3 for the canonical JSON-LD shape rather than
        # restating it in two places.
        assert "chunk_templates.md" in text
        assert "Template 3" in text
