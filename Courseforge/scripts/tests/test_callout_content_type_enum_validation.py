"""Wave 56 — CalloutContentType enum validation at Courseforge emit.

Companion to ``test_content_type_enum_validation.py`` (Wave 50, sections).
Covers the callout emit site at ``_render_content_sections`` which used to
hardcode ``"application-note"`` / ``"note"`` without routing through a
validator — any future typo or new callout subtype could silently ship an
ad-hoc value outside the taxonomy.

Covers:

* Hardcoded ``CALLOUT_CONTENT_TYPE_ENUM`` in ``generate_course.py`` matches
  ``schemas/taxonomies/content_type.json::$defs.CalloutContentType``
  (schema-code drift guard).
* ``_validate_callout_content_type`` accepts every enum member.
* With ``TRAINFORGE_ENFORCE_CONTENT_TYPE`` truthy, an unknown value raises
  ``ValueError``.
* Without the flag, an unknown value logs a WARNING and defaults to
  ``"note"`` (the neutral callout, safer than ``"application-note"``).
* The emit path at ``_render_content_sections`` routes through the
  validator so mis-typed callout ``type`` values don't leak into HTML.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_course  # noqa: E402
from generate_course import (  # noqa: E402
    CALLOUT_CONTENT_TYPE_ENUM,
    _render_content_sections,
    _validate_callout_content_type,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONTENT_TYPE_SCHEMA = (
    _PROJECT_ROOT / "schemas" / "taxonomies" / "content_type.json"
)


# ---------------------------------------------------------------------- #
# 1. Drift guard: hardcoded constant matches taxonomy schema
# ---------------------------------------------------------------------- #


def test_callout_enum_matches_taxonomy_schema():
    """The hardcoded callout enum must equal the taxonomy file."""
    with open(_CONTENT_TYPE_SCHEMA, encoding="utf-8") as f:
        schema = json.load(f)
    schema_enum = frozenset(schema["$defs"]["CalloutContentType"]["enum"])
    assert CALLOUT_CONTENT_TYPE_ENUM == schema_enum, (
        f"Drift: hardcoded={sorted(CALLOUT_CONTENT_TYPE_ENUM)} vs "
        f"schema={sorted(schema_enum)}"
    )


# ---------------------------------------------------------------------- #
# 2. Validator accepts every enum member
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("value", sorted(CALLOUT_CONTENT_TYPE_ENUM))
def test_validator_accepts_enum_members(value, monkeypatch):
    """Every enum member round-trips unchanged with or without enforcement."""
    monkeypatch.delenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", raising=False)
    assert _validate_callout_content_type(value) == value

    monkeypatch.setenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", "1")
    assert _validate_callout_content_type(value) == value


# ---------------------------------------------------------------------- #
# 3. Enforcement flag: truthy → raise ValueError on unknown value
# ---------------------------------------------------------------------- #


def test_enforce_flag_raises_on_unknown_callout_value(monkeypatch):
    """Enforcement on + bogus value → ValueError mentioning the callout enum."""
    monkeypatch.setenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", "1")
    with pytest.raises(ValueError, match="Unknown callout content type"):
        _validate_callout_content_type("bogus-callout")


def test_enforce_flag_rejects_section_value_in_callout_slot(monkeypatch):
    """Section-enum members aren't valid in the callout slot, and vice versa.

    This is the payoff of having two enums — a SectionContentType like
    ``"example"`` is valid for ``<h2>`` / ``<h3>`` but must be rejected at
    callout emit time, since callouts aren't sections in the heading-
    heuristic sense.
    """
    monkeypatch.setenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", "1")
    with pytest.raises(ValueError, match="Unknown callout content type"):
        _validate_callout_content_type("example")


# ---------------------------------------------------------------------- #
# 4. Enforcement unset → WARNING + fallback to "note"
# ---------------------------------------------------------------------- #


def test_unenforced_flag_logs_warning_and_defaults_to_note(monkeypatch, caplog):
    """With enforcement off, a bogus callout value warns and returns 'note'."""
    monkeypatch.delenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", raising=False)
    with caplog.at_level(logging.WARNING, logger=generate_course.logger.name):
        result = _validate_callout_content_type("bogus-callout")
    assert result == "note"
    assert any(
        "bogus-callout" in rec.getMessage() and rec.levelno == logging.WARNING
        for rec in caplog.records
    ), f"Expected WARNING log mentioning 'bogus-callout', got {caplog.records!r}"


# ---------------------------------------------------------------------- #
# 5. Emit-path integration: _render_content_sections routes through validator
# ---------------------------------------------------------------------- #


def test_render_content_sections_routes_callout_through_validator(monkeypatch):
    """Callout emit path must invoke the validator — drift guard.

    Substitutes a tracking wrapper and confirms every callout produced by
    ``_render_content_sections`` passes through it at least once. Without
    this, a future refactor could restore the hardcoded emit without
    surfacing a test failure.
    """
    seen: list[str] = []
    real_validator = generate_course._validate_callout_content_type

    def tracking(value: str) -> str:
        seen.append(value)
        return real_validator(value)

    monkeypatch.setattr(generate_course, "_validate_callout_content_type", tracking)

    sections = [
        {
            "heading": "A Section With a Note",
            "paragraphs": ["Body text."],
            "callout": {
                "type": "callout-info",
                "heading": "Note",
                "items": ["Be aware."],
            },
        },
        {
            "heading": "A Section With a Warning",
            "paragraphs": ["Body text."],
            "callout": {
                "type": "callout-warning",
                "heading": "Careful",
                "items": ["Watch out."],
            },
        },
    ]
    _render_content_sections(sections)

    assert "note" in seen, (
        f"Expected callout validator to see 'note' for callout-info section; "
        f"saw {seen!r}"
    )
    assert "application-note" in seen, (
        f"Expected callout validator to see 'application-note' for "
        f"callout-warning section; saw {seen!r}"
    )


def test_render_content_sections_callout_emit_has_data_cf_content_type():
    """HTML output must carry data-cf-content-type on every callout div."""
    sections = [
        {
            "heading": "Section",
            "paragraphs": ["p"],
            "callout": {
                "type": "callout-warning",
                "heading": "Heads Up",
                "items": ["text"],
            },
        }
    ]
    html = _render_content_sections(sections)
    assert 'data-cf-content-type="application-note"' in html, (
        f"callout-warning section did not emit data-cf-content-type="
        f"application-note; rendered HTML was:\n{html}"
    )
