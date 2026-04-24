"""Wave 50 — SectionContentType enum validation at Courseforge emit.

Covers:

* Hardcoded ``SECTION_CONTENT_TYPE_ENUM`` in ``generate_course.py`` matches
  ``schemas/taxonomies/content_type.json::$defs.SectionContentType``
  (schema-code drift guard).
* ``_infer_content_type`` always returns a value in the enum across a
  sampling of 8+ section-fixture shapes.
* With ``TRAINFORGE_ENFORCE_CONTENT_TYPE`` truthy, an unknown value from
  the inner heuristic raises ``ValueError``.
* Without the flag, an unknown value logs a WARNING and defaults to
  ``"explanation"`` so emit still succeeds.
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
    SECTION_CONTENT_TYPE_ENUM,
    _infer_content_type,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONTENT_TYPE_SCHEMA = (
    _PROJECT_ROOT / "schemas" / "taxonomies" / "content_type.json"
)


# ---------------------------------------------------------------------- #
# 1. Drift guard: hardcoded constant matches taxonomy schema
# ---------------------------------------------------------------------- #


def test_enum_matches_taxonomy_schema():
    """The hardcoded enum in generate_course.py must equal the taxonomy file."""
    with open(_CONTENT_TYPE_SCHEMA, encoding="utf-8") as f:
        schema = json.load(f)
    schema_enum = frozenset(schema["$defs"]["SectionContentType"]["enum"])
    assert SECTION_CONTENT_TYPE_ENUM == schema_enum, (
        f"Drift: hardcoded={sorted(SECTION_CONTENT_TYPE_ENUM)} vs "
        f"schema={sorted(schema_enum)}"
    )


# ---------------------------------------------------------------------- #
# 2. Heuristic always returns an enum member
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "section,expected",
    [
        # overview branch
        ({"heading": "Course Overview"}, "overview"),
        ({"heading": "Introduction to the Topic"}, "overview"),
        # summary branch
        ({"heading": "Module Summary"}, "summary"),
        ({"heading": "Key Takeaways"}, "summary"),
        # definition branch (flip_cards triggers definition regardless of heading)
        ({"heading": "Glossary", "flip_cards": [{"term": "x", "definition": "y"}]}, "definition"),
        # example branch
        ({"heading": "Worked Example"}, "example"),
        ({"heading": "Case Study: ACME"}, "example"),
        # procedure branch
        ({"heading": "Steps to Configure"}, "procedure"),
        ({"heading": "How To Install"}, "procedure"),
        # comparison branch
        ({"heading": "TCP vs UDP"}, "comparison"),
        ({"heading": "Contrast Between Models"}, "comparison"),
        # exercise branch
        ({"heading": "Practice Activity"}, "exercise"),
        # explanation (default fallthrough)
        ({"heading": "Core Concepts"}, "explanation"),
        ({"heading": "Historical Background"}, "explanation"),
    ],
)
def test_infer_content_type_always_returns_enum_member(section, expected):
    """Every fixture must produce a value inside SECTION_CONTENT_TYPE_ENUM."""
    result = _infer_content_type(section)
    assert result in SECTION_CONTENT_TYPE_ENUM, (
        f"{result!r} from section {section!r} is not in enum "
        f"{sorted(SECTION_CONTENT_TYPE_ENUM)}"
    )
    assert result == expected


# ---------------------------------------------------------------------- #
# 3. Enforcement flag: truthy → raise ValueError on unknown value
# ---------------------------------------------------------------------- #


def test_enforce_flag_raises_on_unknown_value(monkeypatch):
    """With enforcement on, a bogus raw value must raise ValueError."""
    monkeypatch.setenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", "1")
    # Force the raw heuristic to return an enum-miss value.
    monkeypatch.setattr(
        generate_course,
        "_infer_content_type_raw",
        lambda section: "application-note",  # belongs to CalloutContentType, not Section
    )
    with pytest.raises(ValueError, match="Unknown content type"):
        _infer_content_type({"heading": "anything"})


# ---------------------------------------------------------------------- #
# 4. Enforcement flag: unset → WARNING + fallback to "explanation"
# ---------------------------------------------------------------------- #


def test_unenforced_flag_logs_warning_and_defaults_to_explanation(monkeypatch, caplog):
    """With enforcement off, a bogus value logs a WARNING and returns fallback."""
    monkeypatch.delenv("TRAINFORGE_ENFORCE_CONTENT_TYPE", raising=False)
    monkeypatch.setattr(
        generate_course,
        "_infer_content_type_raw",
        lambda section: "bogus-type",
    )
    with caplog.at_level(logging.WARNING, logger=generate_course.logger.name):
        result = _infer_content_type({"heading": "anything"})
    assert result == "explanation"
    assert any(
        "bogus-type" in rec.getMessage() and rec.levelno == logging.WARNING
        for rec in caplog.records
    ), f"Expected WARNING log mentioning 'bogus-type', got {caplog.records!r}"
