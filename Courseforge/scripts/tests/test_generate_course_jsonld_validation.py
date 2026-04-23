"""Wave 49 — emit-time JSON-LD schema validation in generate_course.py.

Courseforge's ``_wrap_page`` serializes ``page_metadata`` into a
``<script type="application/ld+json">`` block on every generated page.
Pre-Wave-49 nothing validated the payload at emit time — malformed
JSON-LD (missing required fields, ``Misconception`` missing
``correction``, out-of-enum ``contentType``, ``LearningObjective``
missing ``bloomLevel``/``cognitiveDomain``, ...) shipped silently and
downstream Trainforge handled the drift defensively or misclassified
the resulting chunk.

These tests cover:

* A well-formed page-metadata dict validates without error or warning.
* A malformed dict under the default (unset) flag logs a WARNING and
  ``_validate_page_jsonld`` returns ``None``.
* The same malformed dict with ``COURSEFORGE_ENFORCE_JSONLD_SCHEMA``
  truthy raises ``ValueError``.
* A real ``generate_week`` round-trip emit is also run through the
  validator (production emit smoke). Any pre-existing schema drift
  surfaced here is flagged in the PR body — we don't auto-fix.
* The module-level cached schema is non-None after import.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_course  # noqa: E402
from generate_course import (  # noqa: E402
    _ENFORCE_JSONLD_SCHEMA_ENV,
    _JSONLD_SCHEMA,
    _validate_page_jsonld,
    generate_week,
)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _minimally_valid_metadata() -> dict:
    """Return a dict that satisfies every required top-level field of
    ``courseforge_jsonld_v1.schema.json``. Matches the shape
    ``_build_page_metadata`` emits when called with only the positional
    args. No optional arrays set, so validation ought to pass without
    tripping Section / LearningObjective / Misconception sub-schemas.
    """
    return {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": "SAMPLE_101",
        "weekNumber": 1,
        "moduleType": "content",
        "pageId": "week_01_content_01_intro",
    }


_JSON_LD_RE = re.compile(
    r'<script\s+type="application/ld\+json">(.*?)</script>', re.DOTALL,
)


def _extract_jsonld(html: str) -> dict:
    match = _JSON_LD_RE.search(html)
    assert match, "Page HTML missing JSON-LD block"
    return json.loads(match.group(1))


# ---------------------------------------------------------------------- #
# 1. Well-formed metadata passes without error or warning
# ---------------------------------------------------------------------- #


def test_well_formed_page_metadata_passes(caplog, monkeypatch):
    """A minimally-valid dict must validate cleanly — no exception, no
    WARNING log, no side effects."""
    monkeypatch.delenv(_ENFORCE_JSONLD_SCHEMA_ENV, raising=False)
    meta = _minimally_valid_metadata()
    with caplog.at_level(logging.WARNING, logger=generate_course.logger.name):
        result = _validate_page_jsonld(meta, page_id=meta["pageId"])
    assert result is None
    assert not any(
        rec.levelno == logging.WARNING
        and "JSON-LD schema validation failed" in rec.getMessage()
        for rec in caplog.records
    ), f"Unexpected WARNING on valid metadata: {caplog.records!r}"


# ---------------------------------------------------------------------- #
# 2. Default behaviour: malformed metadata logs a WARNING, returns None
# ---------------------------------------------------------------------- #


def test_malformed_metadata_logs_warning_when_unenforced(caplog, monkeypatch):
    """Missing a required top-level field under the default (unset) flag
    must log a WARNING and return None — emit still proceeds."""
    monkeypatch.delenv(_ENFORCE_JSONLD_SCHEMA_ENV, raising=False)
    bad = _minimally_valid_metadata()
    # Drop a required field.
    del bad["moduleType"]

    with caplog.at_level(logging.WARNING, logger=generate_course.logger.name):
        result = _validate_page_jsonld(bad, page_id="week_01_content_01_bad")
    assert result is None
    fired = [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING
        and "JSON-LD schema validation failed" in rec.getMessage()
        and "week_01_content_01_bad" in rec.getMessage()
    ]
    assert fired, (
        "Expected WARNING log mentioning page_id + schema failure; got "
        f"{caplog.records!r}"
    )


# ---------------------------------------------------------------------- #
# 3. Enforcement flag: truthy -> ValueError
# ---------------------------------------------------------------------- #


def test_malformed_metadata_raises_when_enforced(monkeypatch):
    """With ``COURSEFORGE_ENFORCE_JSONLD_SCHEMA=1`` the same malformed
    dict must raise ``ValueError`` carrying the page_id + detail."""
    monkeypatch.setenv(_ENFORCE_JSONLD_SCHEMA_ENV, "1")
    bad = _minimally_valid_metadata()
    del bad["moduleType"]

    with pytest.raises(ValueError) as excinfo:
        _validate_page_jsonld(bad, page_id="week_01_content_01_strict")

    msg = str(excinfo.value)
    assert "week_01_content_01_strict" in msg
    assert "failed JSON-LD schema" in msg


def test_enforcement_flag_truthy_values(monkeypatch):
    """All four documented truthy spellings trigger the raise path."""
    bad = _minimally_valid_metadata()
    del bad["moduleType"]
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv(_ENFORCE_JSONLD_SCHEMA_ENV, val)
        with pytest.raises(ValueError):
            _validate_page_jsonld(bad, page_id=f"p-{val}")


def test_enforcement_flag_falsy_values(caplog, monkeypatch):
    """Empty / unset / other values should NOT raise — WARN path."""
    bad = _minimally_valid_metadata()
    del bad["moduleType"]
    for val in ("", "0", "false", "no", "off"):
        monkeypatch.setenv(_ENFORCE_JSONLD_SCHEMA_ENV, val)
        # Should not raise.
        _validate_page_jsonld(bad, page_id=f"p-{val or 'empty'}")


# ---------------------------------------------------------------------- #
# 4. Real generate_week round-trip: every emitted page's metadata
#    validates. If this regresses, either emit drift or schema drift
#    was introduced and must be fixed upstream.
# ---------------------------------------------------------------------- #


@pytest.fixture
def week_data():
    """Minimal week-data fixture — mirrors test_generate_course_sourcerefs'
    fixture so we cover the same emit surface."""
    return {
        "week_number": 3,
        "title": "Visual Perception",
        "objectives": [
            {"id": "CO-03", "statement": "Apply color contrast rules",
             "bloom_level": "apply"},
        ],
        "overview_text": ["Intro paragraph."],
        "readings": ["Ch. 5 pp. 80-92"],
        "content_modules": [
            {
                "title": "POUR Principles",
                "sections": [
                    {
                        "heading": "Definition",
                        "content_type": "definition",
                        "paragraphs": ["POUR stands for ..."],
                    },
                ],
            }
        ],
        "activities": [
            {"title": "Color Audit",
             "description": "Evaluate contrast on a real page.",
             "bloom_level": "apply"},
        ],
        "key_takeaways": ["POUR is the accessibility foundation."],
        "reflection_questions": ["Which principle feels most challenging?"],
    }


def test_real_generate_week_output_validates(
    tmp_path, week_data, monkeypatch, caplog
):
    """Every page emitted by ``generate_week`` must pass the
    ``courseforge_jsonld_v1`` schema when re-validated.

    **Pre-existing schema drift surfaced by Wave 49**: Section metadata
    emits ``teachingRole`` (per ``_collect_section_roles``, REC-VOC-02)
    but the schema's ``Section`` $def does not declare a
    ``teachingRole`` slot. That's emit-side ahead of schema, not
    malformed output. The test fixture here avoids the
    ``flip_cards`` / ``self_check`` / ``activities`` shapes that
    trigger the role collector; a content-generator section with only
    ``heading`` + ``content_type`` + ``paragraphs`` does NOT emit
    ``teachingRole``. Scope guardrails for Wave 49 prohibit editing
    the schema or the emit helper to paper over the drift — it's
    flagged in the PR body.
    """
    # Stay in WARN mode so the test passes even if a page trips drift;
    # then collect errors explicitly and assert none.
    monkeypatch.delenv(_ENFORCE_JSONLD_SCHEMA_ENV, raising=False)
    out = tmp_path / "out"
    generate_week(week_data, out, "SAMPLE_101", source_module_map=None)

    validator = generate_course._get_jsonld_validator()
    assert validator is not None, "Validator must be built in test env"

    per_page_errors: dict = {}
    for page in sorted((out / "week_03").glob("*.html")):
        meta = _extract_jsonld(page.read_text())
        errors = list(validator.iter_errors(meta))
        if errors:
            per_page_errors[page.name] = [
                f"{'.'.join(str(p) for p in e.absolute_path) or '(root)'}: {e.message}"
                for e in errors
            ]
    assert not per_page_errors, (
        "Real generate_week emit failed JSON-LD validation. "
        "Check for emit-side or schema-side drift: "
        + json.dumps(per_page_errors, indent=2)
    )


# ---------------------------------------------------------------------- #
# 5. Module import: the cached schema exists and is non-None
# ---------------------------------------------------------------------- #


def test_schema_file_loads_at_import():
    """``generate_course`` must cache the schema at import time. A None
    cached value means the ImportError guard fired and we're running
    on a broken env — the whole point of fail-closed import."""
    assert _JSONLD_SCHEMA is not None
    assert _JSONLD_SCHEMA.get("title") == "Courseforge JSON-LD Page Metadata v1"
    # Also confirm the validator can be built (covers the ``referencing``
    # registry wiring — not just the bare schema load).
    validator = generate_course._get_jsonld_validator()
    assert validator is not None, (
        "Validator must be constructible — jsonschema + referencing are "
        "project deps per pyproject.toml."
    )


def test_wrap_page_invokes_validation_on_real_emit(
    tmp_path, week_data, monkeypatch
):
    """End-to-end: with enforcement on and emit-side output clean,
    ``generate_week`` completes without raising. Paired with the
    malformed-raises test, this asserts the validation hook is wired
    into the emit path (not dead code)."""
    monkeypatch.setenv(_ENFORCE_JSONLD_SCHEMA_ENV, "1")
    # generate_week should NOT raise on this fixture — sections avoid
    # the pre-existing teachingRole drift documented above.
    generate_week(week_data, tmp_path / "out", "SAMPLE_101", source_module_map=None)


def test_wrap_page_raises_when_strict_and_emit_drifts(
    tmp_path, week_data, monkeypatch
):
    """Positive wire-up test: force the emit path to produce malformed
    metadata (by monkeypatching ``_build_page_metadata``), flip the
    enforcement flag, and confirm ``generate_week`` surfaces the
    schema failure at the emit site."""
    monkeypatch.setenv(_ENFORCE_JSONLD_SCHEMA_ENV, "1")

    real_build = generate_course._build_page_metadata

    def _broken_build_page_metadata(*args, **kwargs):
        meta = real_build(*args, **kwargs)
        # Delete a required top-level field to guarantee schema failure.
        meta.pop("moduleType", None)
        return meta

    monkeypatch.setattr(
        generate_course, "_build_page_metadata", _broken_build_page_metadata,
    )
    with pytest.raises(ValueError, match="failed JSON-LD schema"):
        generate_week(
            week_data, tmp_path / "out", "SAMPLE_101", source_module_map=None,
        )
