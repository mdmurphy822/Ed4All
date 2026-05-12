"""GPT Feedback (May 12) item 5 — HTMLContentParser normalizes the new
``cognitiveTaskType`` JSON-LD camelCase field to ``cognitive_task_type``
snake_case on misconception sub-objects.

Mirrors the existing ``bloomLevel`` / ``cognitiveDomain`` normalization
behavior that lives a few lines above the new arm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from Trainforge.parsers.html_content_parser import HTMLContentParser  # noqa: E402


def _page_with_jsonld(json_ld: dict) -> str:
    return (
        "<html><head><title>page</title>"
        f'<script type="application/ld+json">{json.dumps(json_ld)}</script>'
        "</head><body><h1>page</h1>"
        "<p>Body paragraph so the parser counts words.</p>"
        "</body></html>"
    )


def test_misconception_with_camelcase_cognitive_task_type_normalized():
    """JSON-LD camelCase ``cognitiveTaskType`` → snake_case ``cognitive_task_type``."""
    json_ld = {
        "pageId": "p1",
        "misconceptions": [
            {
                "misconception": "All inheritance hierarchies are trees.",
                "correction": "Multiple inheritance produces a DAG.",
                "bloomLevel": "Analyze",
                "cognitiveDomain": "conceptual",
                "cognitiveTaskType": "debug",
            }
        ],
    }
    parsed = HTMLContentParser().parse(_page_with_jsonld(json_ld))
    assert parsed.misconceptions, "expected at least one misconception entry"
    mc = parsed.misconceptions[0]
    assert mc["cognitive_task_type"] == "debug"
    # And the sibling Bloom-axis fields are still normalized as before.
    assert mc["bloom_level"] == "analyze"
    assert mc["cognitive_domain"] == "conceptual"
    # The camelCase original is filtered out (mirrors bloomLevel handling).
    assert "cognitiveTaskType" not in mc


def test_misconception_with_snakecase_cognitive_task_type_preserved():
    """Authors who already emit snake_case ``cognitive_task_type`` pass through."""
    json_ld = {
        "pageId": "p1",
        "misconceptions": [
            {
                "misconception": "All inheritance hierarchies are trees.",
                "correction": "Multiple inheritance produces a DAG.",
                "cognitive_task_type": "classify",
            }
        ],
    }
    parsed = HTMLContentParser().parse(_page_with_jsonld(json_ld))
    mc = parsed.misconceptions[0]
    assert mc["cognitive_task_type"] == "classify"


def test_misconception_without_cognitive_task_type_field_silently_absent():
    """Legacy misconception (no field at all) doesn't grow a null slot."""
    json_ld = {
        "pageId": "p1",
        "misconceptions": [
            {
                "misconception": "All inheritance hierarchies are trees.",
                "correction": "Multiple inheritance produces a DAG.",
            }
        ],
    }
    parsed = HTMLContentParser().parse(_page_with_jsonld(json_ld))
    mc = parsed.misconceptions[0]
    assert "cognitive_task_type" not in mc


def test_misconception_with_non_string_cognitive_task_type_ignored():
    """Non-string ``cognitiveTaskType`` (e.g. number, null) doesn't crash."""
    for bad_value in [None, 42, ["debug"], {"verb": "debug"}, ""]:
        json_ld = {
            "pageId": "p1",
            "misconceptions": [
                {
                    "misconception": "Misc.",
                    "correction": "Correction.",
                    "cognitiveTaskType": bad_value,
                }
            ],
        }
        parsed = HTMLContentParser().parse(_page_with_jsonld(json_ld))
        mc = parsed.misconceptions[0]
        assert "cognitive_task_type" not in mc, (
            f"non-string value {bad_value!r} should be silently ignored"
        )


def test_cognitive_task_type_lowercased():
    """Mixed-case input (``Debug``) lowercased to canonical (``debug``)."""
    json_ld = {
        "pageId": "p1",
        "misconceptions": [
            {
                "misconception": "Misc.",
                "correction": "Correction.",
                "cognitiveTaskType": "Debug",
            }
        ],
    }
    parsed = HTMLContentParser().parse(_page_with_jsonld(json_ld))
    mc = parsed.misconceptions[0]
    assert mc["cognitive_task_type"] == "debug"
