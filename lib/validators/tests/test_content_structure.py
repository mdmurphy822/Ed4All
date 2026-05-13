"""Regression coverage for ``ContentStructureValidator._check_headings``.

Wave3-I4 — dispatch-7 audit Finding F4 / Execution-inspection Finding 4
flagged that operators could not audit heading-skip completeness in
one pass: every skipped heading must surface as its own ``GateIssue``
with a distinguishing locator so the gate report enumerates ALL
violations on a page, not just the first.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.content import ContentStructureValidator  # noqa: E402


# Three distinct h1→h3 (or h1→h4) skips at known line positions.
# Filler <p> blocks keep the validator's THIN_CONTENT (≥50 words)
# warning out of the picture so the assertions stay focused on the
# heading-skip surface.
_FILLER = " ".join(["word"] * 60)
_THREE_SKIPS_HTML = (
    "<html><body>\n"
    "<h1>Chapter one</h1>\n"
    f"<p>{_FILLER}</p>\n"
    "<h3>Skip A — h1 to h3</h3>\n"
    f"<p>{_FILLER}</p>\n"
    "<h1>Chapter two</h1>\n"
    f"<p>{_FILLER}</p>\n"
    "<h4>Skip B — h1 to h4</h4>\n"
    f"<p>{_FILLER}</p>\n"
    "<h1>Chapter three</h1>\n"
    f"<p>{_FILLER}</p>\n"
    "<h3>Skip C — h1 to h3</h3>\n"
    f"<p>{_FILLER}</p>\n"
    "</body></html>\n"
)


def test_all_heading_skips_surface_not_just_first() -> None:
    """All 3 known heading-skips MUST surface in one validate() call."""
    validator = ContentStructureValidator()
    result = validator.validate(
        {"html_content": _THREE_SKIPS_HTML, "gate_id": "content_structure"}
    )

    skip_issues = [i for i in result.issues if i.code == "HEADING_SKIP"]
    assert len(skip_issues) == 3, (
        f"expected 3 HEADING_SKIP issues, got {len(skip_issues)}: "
        f"{[i.message for i in skip_issues]}"
    )

    # Spot-check the per-skip messages so a future regression that
    # collapses every skip into the same canned message is caught.
    messages = [i.message for i in skip_issues]
    assert messages == [
        "Heading skips from h1 to h3",
        "Heading skips from h1 to h4",
        "Heading skips from h1 to h3",
    ], messages

    # The validator MUST fail on heading-skip (severity=error).
    assert result.passed is False
    assert all(i.severity == "error" for i in skip_issues)


def test_each_skip_carries_distinct_locator() -> None:
    """Each HEADING_SKIP issue must carry a unique location locator."""
    validator = ContentStructureValidator()
    result = validator.validate(
        {"html_content": _THREE_SKIPS_HTML, "gate_id": "content_structure"}
    )

    skip_issues = [i for i in result.issues if i.code == "HEADING_SKIP"]
    locators = [i.location for i in skip_issues]

    # No None entries — every issue must carry a locator.
    assert all(loc is not None for loc in locators), locators

    # Locators must be unique so an operator can pin each violation.
    assert len(set(locators)) == len(locators), (
        f"expected unique locators, got duplicates: {locators}"
    )

    # Locator shape: "heading#<N> line:<L> offset:<O>" — sanity-check
    # the per-violation index increases monotonically with HTML order.
    for loc in locators:
        assert loc.startswith("heading#"), loc
        assert " line:" in loc, loc
        assert " offset:" in loc, loc

    # Pull out heading indexes and assert they're in source order.
    indexes = [int(loc.split("#", 1)[1].split(" ", 1)[0]) for loc in locators]
    assert indexes == sorted(indexes), indexes
    # Pull out line numbers and assert they're strictly increasing
    # (each skip lives on a later line than the previous one in the
    # fixture).
    lines = [
        int(loc.split("line:", 1)[1].split(" ", 1)[0]) for loc in locators
    ]
    assert lines == sorted(lines) and len(set(lines)) == len(lines), lines


def test_clean_hierarchy_emits_no_skip_issues() -> None:
    """h1→h2→h3 (no skips) should emit zero HEADING_SKIP issues."""
    clean_html = (
        "<html><body>\n"
        "<h1>Title</h1>\n"
        f"<p>{_FILLER}</p>\n"
        "<h2>Section</h2>\n"
        f"<p>{_FILLER}</p>\n"
        "<h3>Subsection</h3>\n"
        f"<p>{_FILLER}</p>\n"
        "</body></html>\n"
    )
    validator = ContentStructureValidator()
    result = validator.validate(
        {"html_content": clean_html, "gate_id": "content_structure"}
    )
    skip_issues = [i for i in result.issues if i.code == "HEADING_SKIP"]
    assert skip_issues == []
