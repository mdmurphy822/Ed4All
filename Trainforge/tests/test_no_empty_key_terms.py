"""Chunks must never emit ``key_terms[*].definition == ""``.

The chunk_v4 schema requires ``KeyTerm.definition`` with ``minLength: 1``.
The data-cf-* fallback path in ``_extract_section_metadata`` historically
synthesised ``{"term": t, "definition": ""}`` because data-cf-* carries
only term slugs. Under ``TRAINFORGE_VALIDATE_CHUNKS=true`` this tripped
schema validation and aborted every run whose chunks fell through to
that fallback.

The fix filters empty definitions in ``_fill_or_drop_empty_key_term_definitions``:
for each empty entry, attempt a best-effort definition lookup from the
chunk text (first sentence mentioning the term); drop when unrecoverable.
These tests lock in both branches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import CourseProcessor  # noqa: E402


@pytest.fixture
def helper():
    return CourseProcessor._fill_or_drop_empty_key_term_definitions


def test_empty_definition_is_extracted_from_text_when_term_is_mentioned(helper):
    """When the chunk text contains a sentence mentioning the term, lift
    that sentence as the definition."""
    text = (
        "The alt text attribute provides a textual alternative for images. "
        "It enables screen readers to describe visual content."
    )
    kt = [{"term": "alt text", "definition": ""}]
    out = helper(kt, text)
    assert len(out) == 1
    assert out[0]["term"] == "alt text"
    # The first sentence mentions the term; it becomes the definition.
    assert "alt text" in out[0]["definition"].lower()
    assert out[0]["definition"] != ""


def test_empty_definition_drops_entry_when_term_absent_from_text(helper):
    """When extraction can't find the term, omit the entry. Never emit ""."""
    text = "Some totally unrelated prose about Saturn's rings."
    kt = [{"term": "wcag compliance", "definition": ""}]
    out = helper(kt, text)
    assert out == [], (
        f"Expected empty list when term absent; got {out!r}. The fix must "
        "omit entries rather than leave an empty definition placeholder."
    )


def test_existing_definitions_are_preserved(helper):
    """Entries with a non-empty definition pass through unchanged."""
    text = "Unrelated text."
    kt = [
        {"term": "aria role", "definition": "A role attribute for assistive tech."},
        {"term": "heading", "definition": "A semantic section header element."},
    ]
    out = helper(kt, text)
    assert len(out) == 2
    assert out[0]["definition"] == "A role attribute for assistive tech."
    assert out[1]["definition"] == "A semantic section header element."


def test_whitespace_only_definition_treated_as_empty(helper):
    """A definition of ``"   "`` is equivalent to empty per schema (length 1
    of whitespace would technically pass minLength but is semantically
    empty). The filter normalises."""
    text = "The contrast ratio measures luminance differences between colours."
    kt = [{"term": "contrast ratio", "definition": "   "}]
    out = helper(kt, text)
    assert len(out) == 1
    # Whitespace-only was replaced by the sentence mentioning the term.
    assert "contrast ratio" in out[0]["definition"].lower()


def test_mixed_entries_some_filled_some_dropped(helper):
    """A batch with populated, recoverable, and unrecoverable entries ends
    up as the union of populated + recoverable — never an empty string."""
    text = (
        "The semantic heading structure is critical for navigation. "
        "Focus order must follow the reading order."
    )
    kt = [
        {"term": "existing", "definition": "an explicit definition"},
        {"term": "focus order", "definition": ""},       # recoverable
        {"term": "landmark region", "definition": ""},   # unrecoverable
    ]
    out = helper(kt, text)
    assert len(out) == 2
    terms = [e["term"] for e in out]
    assert "existing" in terms
    assert "focus order" in terms
    assert "landmark region" not in terms
    # And nothing carries an empty definition.
    for entry in out:
        assert entry["definition"], (
            f"Entry {entry!r} leaked an empty definition through the filter."
        )


def test_validate_chunks_env_true_does_not_trip_on_fallback_pairs(
    helper, monkeypatch
):
    """End-to-end-ish: under TRAINFORGE_VALIDATE_CHUNKS=true the chunk the
    helper produced must validate against chunk_v4.schema.json's KeyTerm
    shape (``definition`` minLength 1)."""
    monkeypatch.setenv("TRAINFORGE_VALIDATE_CHUNKS", "true")
    text = "A heading landmark defines the primary region of a page."
    kt = [{"term": "heading landmark", "definition": ""}]
    out = helper(kt, text)
    assert out and out[0]["definition"], out
    # A manual schema check against the KeyTerm slice.
    from jsonschema import validate

    key_term_schema = {
        "type": "object",
        "required": ["term", "definition"],
        "additionalProperties": False,
        "properties": {
            "term": {"type": "string", "minLength": 1},
            "definition": {"type": "string", "minLength": 1},
        },
    }
    for entry in out:
        validate(instance=entry, schema=key_term_schema)
