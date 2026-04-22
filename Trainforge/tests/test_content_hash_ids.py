"""Worker N — REC-ID-01 content-hash chunk IDs (opt-in).

Regression tests for the ``_generate_chunk_id`` helper in
``Trainforge.process_course`` and for the relaxed ``chunk_v4.schema.json``
``id`` pattern.

Default behavior (env var unset or not "true") must produce legacy
position-based IDs matching ``^<prefix>\\d{5}$``. When
``TRAINFORGE_CONTENT_HASH_IDS=true`` is set, IDs must be content-addressed:
hex suffix derived from ``sha256(text|source_locator|schema_version)``.

The helper reads the env var on each call (not at module-import) so tests
can flip the flag via ``monkeypatch.setenv`` without reload gymnastics.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Project root (Ed4All/). This file lives at
# Ed4All/Trainforge/tests/test_content_hash_ids.py → parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.process_course import _generate_chunk_id  # noqa: E402

SCHEMAS_DIR = PROJECT_ROOT / "schemas"
CHUNK_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "chunk_v4.schema.json"


# ---------------------------------------------------------------------------
# Tests: _generate_chunk_id behavior under flag-off / flag-on
# ---------------------------------------------------------------------------


def test_flag_off_uses_position(monkeypatch):
    """Default env (flag unset) → legacy 5-digit position-based IDs."""
    monkeypatch.delenv("TRAINFORGE_CONTENT_HASH_IDS", raising=False)
    chunk_id = _generate_chunk_id(
        prefix="wcag_201_chunk_",
        start_id=42,
        text="some text",
        source_locator="course_content/page_01.html",
    )
    assert chunk_id == "wcag_201_chunk_00042"
    assert re.match(r"^wcag_201_chunk_\d{5}$", chunk_id)


def test_flag_off_explicit_false(monkeypatch):
    """Explicit non-truthy env values must preserve legacy behavior too."""
    for value in ("", "false", "False", "0", "no"):
        monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", value)
        chunk_id = _generate_chunk_id(
            prefix="wcag_201_chunk_", start_id=7, text="t", source_locator="p",
        )
        assert chunk_id == "wcag_201_chunk_00007", (
            f"Expected legacy form with env value {value!r}, got {chunk_id!r}"
        )


def test_flag_on_uses_content_hash(monkeypatch):
    """Flag on → 16-hex-char content-hash suffix after prefix."""
    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "true")
    chunk_id = _generate_chunk_id(
        prefix="wcag_201_chunk_",
        start_id=42,
        text="some text",
        source_locator="course_content/page_01.html",
    )
    assert re.match(r"^wcag_201_chunk_[0-9a-f]{16}$", chunk_id), chunk_id
    # Position-derived digits must NOT appear as the whole suffix under
    # flag-on mode.
    assert not chunk_id.endswith("00042")


def test_content_hash_stable_across_runs(monkeypatch):
    """Same (text, source_locator) → same ID across repeated calls."""
    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "true")
    a = _generate_chunk_id(
        prefix="wcag_201_chunk_", start_id=0,
        text="stable content for hashing", source_locator="path/a.html",
    )
    b = _generate_chunk_id(
        prefix="wcag_201_chunk_", start_id=999,  # start_id must not affect hash
        text="stable content for hashing", source_locator="path/a.html",
    )
    assert a == b, "Content-hash IDs must be independent of start_id"


def test_content_hash_differs_on_text_change(monkeypatch):
    """One-character edit to text → different ID."""
    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "true")
    a = _generate_chunk_id(
        prefix="wcag_201_chunk_", start_id=0,
        text="original text", source_locator="path/a.html",
    )
    b = _generate_chunk_id(
        prefix="wcag_201_chunk_", start_id=0,
        text="original text!", source_locator="path/a.html",
    )
    assert a != b


def test_content_hash_differs_on_source_change(monkeypatch):
    """Same text, different source locator → different ID.

    Guards against boilerplate-chunk collisions across pages.
    """
    monkeypatch.setenv("TRAINFORGE_CONTENT_HASH_IDS", "true")
    a = _generate_chunk_id(
        prefix="wcag_201_chunk_", start_id=0,
        text="boilerplate footer text", source_locator="path/a.html",
    )
    b = _generate_chunk_id(
        prefix="wcag_201_chunk_", start_id=0,
        text="boilerplate footer text", source_locator="path/b.html",
    )
    assert a != b


# ---------------------------------------------------------------------------
# Schema-pattern relaxation: both legacy and hash forms must validate.
# ---------------------------------------------------------------------------


def _load_chunk_id_pattern() -> str:
    """Load the regex pattern for the chunk ``id`` field from the schema."""
    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    return schema["properties"]["id"]["pattern"]


def test_schema_accepts_both_formats():
    """chunk_v4 ``id`` pattern matches both 5-digit and 16-hex suffixes;
    rejects malformed suffixes."""
    pattern = _load_chunk_id_pattern()
    regex = re.compile(pattern)

    # Legacy position-based form — must pass.
    assert regex.match("wcag_201_chunk_00001")
    assert regex.match("wcag_201_chunk_99999")

    # New content-hash form — must pass.
    assert regex.match("wcag_201_chunk_a3f2b9c8d1e4f567")
    assert regex.match("wcag_201_chunk_0123456789abcdef")

    # Malformed suffixes — must NOT match (negative controls).
    assert not regex.match("wcag_201_chunk_123")            # too few digits
    assert not regex.match("wcag_201_chunk_invalid")        # non-hex letters
    assert not regex.match("wcag_201_chunk_a3f2b9c8d1e4f5") # 14 hex chars
    assert not regex.match("wcag_201_chunk_A3F2B9C8D1E4F567")  # uppercase hex


# ---------------------------------------------------------------------------
# End-to-end: when installed, ensure jsonschema also accepts both forms
# against the full schema (not just the pattern in isolation).
# ---------------------------------------------------------------------------


def _require_jsonschema():
    try:
        import jsonschema  # noqa: F401
        return jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema not installed")


def _build_validator():
    jsonschema = _require_jsonschema()
    from jsonschema import Draft202012Validator, RefResolver

    with open(CHUNK_SCHEMA_PATH) as f:
        schema = json.load(f)
    store: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            store[sid] = s
    resolver = RefResolver.from_schema(schema, store=store)
    return Draft202012Validator(schema, resolver=resolver)


def _make_valid_chunk(chunk_id: str) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Sample chunk text.",
        "html": "<p>Sample chunk text.</p>",
        "follows_chunk": None,
        "source": {
            "course_id": "TEST_101",
            "module_id": "m1",
            "lesson_id": "l1",
        },
        "concept_tags": ["sample"],
        "learning_outcome_refs": [],
        "difficulty": "foundational",
        "tokens_estimate": 3,
        "word_count": 3,
        "bloom_level": "understand",
    }


def test_full_schema_validates_legacy_and_hash_ids():
    """Full-schema validation (not just pattern) accepts both ID forms."""
    validator = _build_validator()
    # Legacy form
    legacy = _make_valid_chunk("wcag_201_chunk_00001")
    legacy_errors = list(validator.iter_errors(legacy))
    assert legacy_errors == [], (
        f"Legacy-form ID should validate: {[e.message for e in legacy_errors]}"
    )
    # Content-hash form
    hashed = _make_valid_chunk("wcag_201_chunk_a3f2b9c8d1e4f567")
    hash_errors = list(validator.iter_errors(hashed))
    assert hash_errors == [], (
        f"Hash-form ID should validate: {[e.message for e in hash_errors]}"
    )
