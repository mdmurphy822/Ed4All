"""Validate the content-generator's mandatory page-authoring directives.

The agent spec must retain heading hierarchy, source attribution, and
objective attribution requirements so emitted pages remain compatible with
the validators and downstream chunker.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Resolve the agent specification from the supported Courseforge layout.
_AGENTS_DIR = Path(__file__).resolve().parents[3] / "agents"
_PROMPT_PATH = _AGENTS_DIR / "content-generator.md"


@pytest.fixture(scope="module")
def prompt_text() -> str:
    assert _PROMPT_PATH.exists(), (
        f"content-generator.md not found at {_PROMPT_PATH}; "
        "the agent prompt file may have been moved or renamed."
    )
    return _PROMPT_PATH.read_text(encoding="utf-8")


# Heading hierarchy contract.


def test_heading_hierarchy_directive_present(prompt_text: str) -> None:
    """Prompt must contain a MANDATORY heading-hierarchy section."""
    assert "MANDATORY: Heading Hierarchy" in prompt_text, (
        "Missing 'MANDATORY: Heading Hierarchy' section in content-generator.md; "
        "the directive prevents HEADING_SKIP violations."
    )


def test_heading_hierarchy_no_skip_rule(prompt_text: str) -> None:
    """Prompt must explicitly forbid level-skipping (h1 → h2 → h3 chain)."""
    assert "h1 → h2 → h3" in prompt_text, (
        "Missing 'h1 → h2 → h3' sequence rule in content-generator.md. "
        "Subagents must see the explicit descent chain to avoid HEADING_SKIP."
    )


def test_heading_hierarchy_references_heading_skip_error(prompt_text: str) -> None:
    """Prompt must cite the HEADING_SKIP validator error code."""
    assert "HEADING_SKIP" in prompt_text, (
        "Missing 'HEADING_SKIP' error code in content-generator.md. "
        "Referencing the validator error code makes the rule actionable."
    )


# Source-attribution contract.


def test_source_ids_attribute_directive_present(prompt_text: str) -> None:
    """Prompt must contain a MANDATORY source-id stamping section."""
    assert "MANDATORY: Source-ID Stamping" in prompt_text, (
        "Missing 'MANDATORY: Source-ID Stamping' section in content-generator.md; "
        "the directive prevents EMPTY_SOURCE_REFS violations."
    )


def test_source_ids_attribute_name_present(prompt_text: str) -> None:
    """Prompt must explicitly name the data-cf-source-ids attribute."""
    assert "data-cf-source-ids" in prompt_text, (
        "Missing 'data-cf-source-ids' attribute name in content-generator.md."
    )


def test_source_ids_references_empty_source_refs_code(prompt_text: str) -> None:
    """Prompt must cite the EMPTY_SOURCE_REFS validator code."""
    assert "EMPTY_SOURCE_REFS" in prompt_text, (
        "Missing 'EMPTY_SOURCE_REFS' error code in content-generator.md. "
        "Referencing the validator code makes the enforcement path clear."
    )


def test_source_ids_empty_string_allowed(prompt_text: str) -> None:
    """Prompt must clarify that data-cf-source-ids="" (empty) is valid for boilerplate."""
    assert 'data-cf-source-ids=""' in prompt_text, (
        "Missing explicit 'data-cf-source-ids=\"\"' (empty-string) allowance. "
        "Without this, subagents may omit the attribute on boilerplate sections."
    )


# Objective-attribution contract.


def test_objective_id_attribute_directive_present(prompt_text: str) -> None:
    """Prompt must contain a MANDATORY objective-id stamping section."""
    assert "MANDATORY: Objective-ID Stamping" in prompt_text, (
        "Missing 'MANDATORY: Objective-ID Stamping' section in content-generator.md; "
        "the directive populates learning_outcome_refs correctly."
    )


def test_objective_id_attribute_name_present(prompt_text: str) -> None:
    """Prompt must explicitly name the data-cf-objective-id attribute."""
    assert "data-cf-objective-id" in prompt_text, (
        "Missing 'data-cf-objective-id' attribute name in content-generator.md."
    )


def test_objective_id_lo_pattern_present(prompt_text: str) -> None:
    """Prompt must show the canonical TO-NN / CO-NN LO ID pattern."""
    # Accept either the regex pattern or the concrete example forms
    assert ("TO-NN" in prompt_text or "TO-01" in prompt_text), (
        "Missing canonical LO pattern (TO-NN / TO-01) in content-generator.md. "
        "Subagents need concrete ID examples to emit correct objective IDs."
    )
