"""Wave 22 DC4 — course_code normalization covers real-world PDF names.

The canonical ``course_id`` pattern at
``schemas/events/decision_event.schema.json`` is
``^[A-Z]{2,8}_[0-9]{3}$``. PDF-derived codes like ``"Ed4All"``,
``"long_slug_style_textbook_name"``, or ``"arxiv-0000.00000"`` fail that
regex. Pre-Wave-22, roughly half of a recent run's decision records
carried ``course_id`` validation issues as a result.

``MCP.tools.dart_tools.normalize_course_code`` folds any input into
a schema-valid form deterministically — same input always produces
the same output, so runs are reproducible and captures are joinable.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from MCP.tools.dart_tools import normalize_course_code  # noqa: E402

_CANON = re.compile(r"^[A-Z]{2,8}_[0-9]{3}$")


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        "Ed4All",
        "long_slug_style_textbook_name",
        "arxiv-0000.00000",
        "ontology_engineering_textbook",
        "MINI_TRAINING_101_PYTEST",
        "textbook",
        "  leading-space-name  ",
        "123numeric-leading",
        "X",  # single char
        "",  # empty
    ],
)
def test_normalize_produces_canonical_course_code(raw):
    """Every real-world PDF name must normalise to the canonical pattern."""
    normalised = normalize_course_code(raw)
    assert _CANON.match(normalised), (
        f"normalize_course_code({raw!r}) returned {normalised!r} which "
        f"does not match the canonical pattern ^[A-Z]{{2,8}}_[0-9]{{3}}$"
    )


@pytest.mark.unit
def test_already_canonical_codes_are_preserved():
    """Inputs that already match should round-trip unchanged."""
    for canon in ("MTH_101", "BIO_201", "CHEM_001", "PHYS_999"):
        assert normalize_course_code(canon) == canon, (
            f"Canonical code {canon} should round-trip, got "
            f"{normalize_course_code(canon)}"
        )


@pytest.mark.unit
def test_normalization_is_deterministic():
    """Same input must always produce the same output (hash-based suffix)."""
    names = [
        "Ed4All",
        "long_slug_style_textbook_name",
        "arxiv-0000.00000",
    ]
    for name in names:
        first = normalize_course_code(name)
        for _ in range(5):
            assert normalize_course_code(name) == first, (
                f"normalize_course_code({name!r}) is non-deterministic"
            )


@pytest.mark.unit
def test_different_inputs_generally_yield_different_codes():
    """Distinct PDFs should usually produce distinct codes (low collision)."""
    inputs = [
        "Ed4All",
        "long_slug_style_textbook_name",
        "arxiv-0000.00000",
        "ontology_engineering_textbook",
        "textbook_a",
        "textbook_b",
        "science_of_learning",
        "principles_of_accessible_design",
    ]
    normalised = {normalize_course_code(x) for x in inputs}
    # With a 3-digit hash space (1000 values) and 8 inputs, collisions
    # are unlikely but theoretically possible. Assert at least 6 of the
    # 8 are distinct to give a stable test while still catching a
    # regression that makes everything collapse to a single value.
    assert len(normalised) >= 6, (
        f"normalize_course_code collapsed {len(inputs)} distinct inputs "
        f"into only {len(normalised)} unique codes: {normalised}"
    )


@pytest.mark.unit
def test_normalization_validates_under_decision_event_schema():
    """The normalised value must pass the live schema validator.

    This is an integration-ish smoke test — it loads the real schema and
    validates a synthetic decision event carrying the normalised code.
    If the schema's regex ever drifts from the test regex above, this
    catches it.
    """
    import json

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "schemas"
        / "events"
        / "decision_event.schema.json"
    )
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    pattern = schema["properties"]["course_id"]["pattern"]

    live_regex = re.compile(pattern)
    for raw in ("Ed4All", "long_slug_style_textbook_name", "arxiv-0000.00000"):
        normalised = normalize_course_code(raw)
        assert live_regex.match(normalised), (
            f"Normalised code {normalised!r} fails live schema pattern "
            f"{pattern}"
        )
