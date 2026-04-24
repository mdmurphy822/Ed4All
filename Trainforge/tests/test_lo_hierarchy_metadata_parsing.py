"""Wave 69 — LO hierarchy metadata + targeted_concepts land on the dataclass.

Courseforge's Wave 57 + Wave 59 emit additions (``targetedConcepts[]``,
``hierarchyLevel``, ``parentObjectiveId``) are present in the JSON-LD but
were previously discarded by ``Trainforge/parsers/html_content_parser.py``.
This test suite locks in that the parser now lifts each field onto the
``LearningObjective`` dataclass with the Trainforge snake_case convention
(and lowercased Bloom level on ``targeted_concepts``).

Covers:
* ``hierarchyLevel`` → ``LearningObjective.hierarchy_level``
* ``parentObjectiveId`` → ``LearningObjective.parent_objective_id``
* ``targetedConcepts[]`` → ``LearningObjective.targeted_concepts`` (each
  entry ``{"concept": slug, "bloom_level": lowered_level}``)
* Legacy corpora (no Wave 57 / Wave 59 fields) parse with the new
  attributes defaulting to None / []
* Malformed ``targetedConcepts[]`` entries (missing concept, non-dict,
  blank Bloom) are silently skipped rather than raising
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trainforge.parsers.html_content_parser import (  # noqa: E402
    HTMLContentParser,
    LearningObjective,
)


def _page_html(json_ld: dict) -> str:
    """Wrap a JSON-LD dict in a minimal HTML page."""
    return (
        "<html><head><title>Page</title>"
        f'<script type="application/ld+json">{json.dumps(json_ld)}</script>'
        "</head><body><h1>Page</h1></body></html>"
    )


def test_hierarchy_level_and_parent_lo_id_land_on_dataclass():
    """Wave 59 fields surface on LearningObjective."""
    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "CO-05",
                "statement": "Apply frameworks",
                "bloomLevel": "apply",
                "cognitiveDomain": "procedural",
                "hierarchyLevel": "chapter",
                "parentObjectiveId": "TO-01",
            },
            {
                "id": "TO-01",
                "statement": "Do terminal things",
                "bloomLevel": "understand",
                "cognitiveDomain": "conceptual",
                "hierarchyLevel": "terminal",
                # No parentObjectiveId on terminals — they're KG roots.
            },
        ],
    }
    parsed = HTMLContentParser().parse(_page_html(json_ld))
    by_id = {lo.id: lo for lo in parsed.learning_objectives}
    assert set(by_id.keys()) == {"CO-05", "TO-01"}

    co05 = by_id["CO-05"]
    assert co05.hierarchy_level == "chapter"
    assert co05.parent_objective_id == "TO-01"

    to01 = by_id["TO-01"]
    assert to01.hierarchy_level == "terminal"
    assert to01.parent_objective_id is None


def test_targeted_concepts_land_on_dataclass_with_lowercased_bloom():
    """Wave 57 targetedConcepts[] surface with snake_case keys + lower Bloom."""
    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "TO-01",
                "statement": "Analyze systems",
                "bloomLevel": "Analyze",  # mixed case — parser should lower
                "cognitiveDomain": "conceptual",
                "targetedConcepts": [
                    {"concept": "framework", "bloomLevel": "Apply"},
                    {"concept": "ecosystem-flow", "bloomLevel": "ANALYZE"},
                ],
            }
        ],
    }
    parsed = HTMLContentParser().parse(_page_html(json_ld))
    (lo,) = parsed.learning_objectives
    assert isinstance(lo, LearningObjective)
    assert lo.targeted_concepts == [
        {"concept": "framework", "bloom_level": "apply"},
        {"concept": "ecosystem-flow", "bloom_level": "analyze"},
    ]


def test_legacy_corpus_without_wave57_or_wave59_fields_defaults_cleanly():
    """Pre-Wave-57/59 JSON-LD leaves the new fields at their defaults."""
    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "TO-01",
                "statement": "Old-style objective",
                "bloomLevel": "understand",
                "cognitiveDomain": "conceptual",
            }
        ],
    }
    parsed = HTMLContentParser().parse(_page_html(json_ld))
    (lo,) = parsed.learning_objectives
    assert lo.hierarchy_level is None
    assert lo.parent_objective_id is None
    assert lo.targeted_concepts == []


def test_malformed_targeted_concepts_entries_skipped():
    """Non-dict / missing-field entries drop silently, not raise."""
    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "TO-01",
                "statement": "Mixed quality input",
                "bloomLevel": "apply",
                "cognitiveDomain": "procedural",
                "targetedConcepts": [
                    {"concept": "valid", "bloomLevel": "apply"},
                    "string-not-dict",
                    {"bloomLevel": "apply"},  # missing concept
                    {"concept": "no-bloom"},  # missing bloomLevel
                    {"concept": "", "bloomLevel": "apply"},  # blank concept
                    {"concept": "blank-bloom", "bloomLevel": ""},
                ],
            }
        ],
    }
    parsed = HTMLContentParser().parse(_page_html(json_ld))
    (lo,) = parsed.learning_objectives
    assert lo.targeted_concepts == [
        {"concept": "valid", "bloom_level": "apply"},
    ]


def test_non_jsonld_path_still_emits_empty_targeted_concepts():
    """When the LO is pulled from data-cf-* HTML attributes (no JSON-LD),
    targeted_concepts defaults to []. Guards the fallback path."""
    html = (
        "<html><body>"
        '<ul><li data-cf-objective-id="TO-01" data-cf-bloom-level="apply" '
        'data-cf-bloom-verb="apply" data-cf-cognitive-domain="procedural">'
        "TO-01: Apply things</li></ul>"
        "</body></html>"
    )
    parsed = HTMLContentParser().parse(html)
    (lo,) = parsed.learning_objectives
    assert lo.id == "TO-01"
    assert lo.bloom_level == "apply"
    # New Wave 69 fields default to None / [].
    assert lo.hierarchy_level is None
    assert lo.parent_objective_id is None
    assert lo.targeted_concepts == []


def test_misconception_bloom_level_and_cognitive_domain_normalized():
    """Wave 60 fields on page-level misconceptions translate to snake_case
    and lowercase bloom_level.

    The parser is the authoritative normalizer — downstream process_course
    expects snake_case everywhere.
    """
    json_ld = {
        "pageId": "p1",
        "learningObjectives": [
            {
                "id": "TO-01",
                "statement": "s",
                "bloomLevel": "apply",
                "cognitiveDomain": "procedural",
            }
        ],
        "misconceptions": [
            {
                "misconception": "Thinking X is Y",
                "correction": "X is actually Z",
                "bloomLevel": "Apply",
                "cognitiveDomain": "procedural",
            },
            {
                "misconception": "No bloom here",
                "correction": "Still no bloom",
                # No bloomLevel / cognitiveDomain — pre-Wave-60
            },
        ],
    }
    parsed = HTMLContentParser().parse(_page_html(json_ld))
    assert len(parsed.misconceptions) == 2
    first, second = parsed.misconceptions
    assert first["misconception"] == "Thinking X is Y"
    assert first["correction"] == "X is actually Z"
    assert first["bloom_level"] == "apply"  # lowercased
    assert first["cognitive_domain"] == "procedural"
    # Pre-Wave-60 entry stays lean.
    assert "bloom_level" not in second
    assert "cognitive_domain" not in second
    assert second["misconception"] == "No bloom here"
    assert second["correction"] == "Still no bloom"
