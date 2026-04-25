"""
Wave 81 Worker C — dual-path misconception extraction tests.

Verifies the bridging fallback in ``HTMLContentParser`` that rescues
``data-cf-misconception="true"`` paragraphs when the page's JSON-LD
``misconceptions[]`` array is absent or partial.

Background: Wave 79 content-generator subagents emit the
``common_pitfall`` template's misconception paragraph with the
``data-cf-misconception="true"`` HTML attribute but do NOT always
populate the JSON-LD ``misconceptions[]`` array. Trainforge has
historically harvested JSON-LD only, so those misconceptions silently
dropped during Path B regen (rdf-shacl-551-2: 67 -> 45). Wave 81
Worker C's bridging fix lets the parser fall back to the HTML attr
when JSON-LD is absent. The forward-looking spec (see
``Courseforge/templates/chunk_templates.md`` Template 3) now mandates
dual-emit; this fallback only rescues older archives.

Test matrix:
  1. JSON-LD present     -> JSON-LD wins, no HTML-attr fallback fires.
  2. JSON-LD absent      -> HTML-attr fallback extracts a misconception.
  3. Both present (same) -> JSON-LD wins, no duplicate from fallback.
  4. Neither present     -> empty misconceptions list.
"""

from __future__ import annotations

from Trainforge.parsers.html_content_parser import HTMLContentParser


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _wrap(body: str, jsonld: str | None = None) -> str:
    head_extra = (
        f'<script type="application/ld+json">{jsonld}</script>'
        if jsonld
        else ""
    )
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\"><head>"
        "<meta charset=\"UTF-8\"><title>Wave 81 dual-path test</title>"
        f"{head_extra}"
        "</head><body>"
        f"{body}"
        "</body></html>"
    )


# Common pitfall section body — used by tests 2 and 3.
PITFALL_BODY = """
<section data-cf-template-type="common_pitfall"
         data-cf-pitfall-concept="rdf-blank-node"
         data-cf-confused-with="rdf-named-node"
         data-cf-objective-id="CO-02"
         data-cf-bloom-level="analyze"
         data-cf-content-type="explanation">
  <h3 data-cf-content-type="explanation">Common Pitfall: blank nodes vs named nodes</h3>
  <p>When learners first model an object, they reach for blank nodes.</p>
  <h4>What looks like the right answer</h4>
  <p data-cf-misconception="true">A blank node is just an anonymous URI; downstream consumers can dereference it the same way.</p>
  <h4>Why it's wrong</h4>
  <p>Blank-node identifiers are scoped to the graph that emits them.</p>
  <h4>The right approach</h4>
  <p>Mint a named node with a stable URI under a controlled namespace whenever the resource needs to be referenced from outside its graph.</p>
</section>
"""

JSONLD_WITH_MC = """
{
  "@context": "https://ed4all.dev/ns/courseforge/v1",
  "@type": "CourseModule",
  "courseCode": "TEST_101",
  "weekNumber": 2,
  "moduleType": "content",
  "pageId": "week_02_pitfall_01",
  "misconceptions": [
    {
      "misconception": "A blank node is just an anonymous URI; downstream consumers can dereference it the same way.",
      "correction": "Blank-node identifiers are scoped to the graph that emits them.",
      "bloomLevel": "analyze"
    }
  ]
}
"""

JSONLD_NO_MC = """
{
  "@context": "https://ed4all.dev/ns/courseforge/v1",
  "@type": "CourseModule",
  "courseCode": "TEST_101",
  "weekNumber": 2,
  "moduleType": "content",
  "pageId": "week_02_pitfall_01"
}
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJSONLDPath:
    """Test 1: JSON-LD present, no HTML-attr -> JSON-LD entries returned."""

    def test_jsonld_misconceptions_extracted(self):
        # No data-cf-misconception attribute in the body — only JSON-LD.
        body = "<section><p>Plain prose, no misconception attr.</p></section>"
        html = _wrap(body, jsonld=JSONLD_WITH_MC)

        parsed = HTMLContentParser().parse(html)

        assert len(parsed.misconceptions) == 1
        mc = parsed.misconceptions[0]
        assert mc["misconception"].startswith("A blank node is just")
        assert mc["correction"].startswith("Blank-node identifiers")
        assert mc["bloom_level"] == "analyze"


class TestHTMLAttrFallback:
    """Test 2: no JSON-LD, has data-cf-misconception -> fallback extracts."""

    def test_attr_fallback_extracts_misconception(self):
        html = _wrap(PITFALL_BODY, jsonld=JSONLD_NO_MC)

        parsed = HTMLContentParser().parse(html)

        assert len(parsed.misconceptions) == 1
        mc = parsed.misconceptions[0]
        assert "blank node is just an anonymous URI" in mc["misconception"]
        # Correction should be sourced from the "right approach" paragraph.
        assert "named node" in mc["correction"].lower()
        # Default bloom_level for the fallback is "analyze".
        assert mc["bloom_level"] == "analyze"

    def test_attr_fallback_no_jsonld_at_all(self):
        # Page has NO JSON-LD block whatsoever.
        html = _wrap(PITFALL_BODY, jsonld=None)

        parsed = HTMLContentParser().parse(html)

        assert len(parsed.misconceptions) == 1
        assert (
            "blank node"
            in parsed.misconceptions[0]["misconception"].lower()
        )

    def test_attr_fallback_correct_approach_synonym(self):
        # Some content variants use "Correct approach" instead of
        # "The right approach". The fallback should match either.
        body = PITFALL_BODY.replace("The right approach", "Correct approach")
        html = _wrap(body, jsonld=None)

        parsed = HTMLContentParser().parse(html)

        assert len(parsed.misconceptions) == 1
        assert "named node" in parsed.misconceptions[0]["correction"].lower()


class TestDualEmitNoDuplicates:
    """Test 3: JSON-LD AND data-cf-misconception present (same text) ->
    only the JSON-LD entry surfaces, no duplicate."""

    def test_jsonld_wins_no_duplicate(self):
        html = _wrap(PITFALL_BODY, jsonld=JSONLD_WITH_MC)

        parsed = HTMLContentParser().parse(html)

        # Even though both JSON-LD and the HTML attr name the same
        # misconception, the parser MUST NOT emit two entries.
        assert len(parsed.misconceptions) == 1
        # JSON-LD's correction wins (shorter, structured form) rather than
        # the HTML-attr fallback's "right approach" paragraph.
        mc = parsed.misconceptions[0]
        assert mc["misconception"].startswith("A blank node is just")
        assert mc["correction"].startswith("Blank-node identifiers")


class TestNeitherPath:
    """Test 4: no JSON-LD misconceptions[] AND no data-cf-misconception
    attr -> empty misconceptions list."""

    def test_no_misconceptions_extracted(self):
        body = (
            "<section><h3>Plain section</h3>"
            "<p>Boring prose with no pitfall structure.</p></section>"
        )
        html = _wrap(body, jsonld=JSONLD_NO_MC)

        parsed = HTMLContentParser().parse(html)

        assert parsed.misconceptions == []

    def test_no_jsonld_no_attr(self):
        body = "<section><p>Just text.</p></section>"
        html = _wrap(body, jsonld=None)

        parsed = HTMLContentParser().parse(html)

        assert parsed.misconceptions == []
