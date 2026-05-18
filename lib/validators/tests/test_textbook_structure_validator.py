"""Tests for ``TextbookOutlineValidator`` (Stage-1 outline enrichment).

Three-stage large-LLM textbook synthesis architecture, Wave A — see
``plans/textbook-llm-synthesis-3stage-2026-05.md`` §8 row 1.

Cases:

1. ``test_skip_with_pass_when_enrichment_keys_absent`` — a default-off
   textbook_structure (no `semantic_outline`, no
   `draft_terminal_objectives`) yields a no-op ``passed=True``.
2. ``test_outline_themes_empty`` — ``semantic_outline.themes`` empty →
   ``OUTLINE_THEMES_EMPTY``.
3. ``test_theme_chapter_unresolved`` — a theme references a chapter ID
   not on ``chapters[]`` → ``OUTLINE_THEME_CHAPTER_UNRESOLVED``.
4. ``test_draft_to_empty`` — ``draft_terminal_objectives`` empty →
   ``OUTLINE_DRAFT_TO_EMPTY``.
5. ``test_draft_to_malformed`` — a draft TO missing `statement` /
   carrying an invalid `bloom_level` → ``OUTLINE_DRAFT_TO_MALFORMED``.
6. ``test_happy_path_passes`` — a well-formed enrichment passes clean.
7. ``test_decision_capture_emitted`` — the validator emits exactly one
   ``content_structure_check`` event per run.
8. ``test_path_loaded_from_disk`` — ``textbook_structure_path`` is
   loaded + flattened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.textbook_structure import (  # noqa: E402
    TextbookOutlineValidator,
)


class _StubDecisionCapture:
    """Records every ``log_decision`` call so tests can assert capture."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def _base_structure() -> Dict[str, Any]:
    """A well-formed Stage-1-enriched textbook_structure."""
    return {
        "chapters": [
            {"id": "ch1", "title": "Foundations"},
            {"id": "ch2", "title": "Linear Equations"},
        ],
        "semantic_outline": {
            "course_summary": "An intro algebra course.",
            "themes": [
                {
                    "title": "Linear Equations",
                    "chapter_ids": ["ch1", "ch2"],
                    "summary": "Solving linear equations.",
                }
            ],
        },
        "draft_terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Solve linear equations in one variable.",
                "bloom_level": "apply",
                "draft": True,
            }
        ],
        "structure_enrichment": {
            "enriched": True,
            "schema_version": "v1",
        },
    }


def _codes(result: Any) -> List[str]:
    return [i.code for i in result.issues]


def test_skip_with_pass_when_enrichment_keys_absent() -> None:
    legacy = {"chapters": [{"id": "ch1", "title": "Foundations"}]}
    result = TextbookOutlineValidator().validate(
        {"textbook_structure": legacy}
    )
    assert result.passed is True
    assert result.issues == []


def test_outline_themes_empty() -> None:
    structure = _base_structure()
    structure["semantic_outline"]["themes"] = []
    result = TextbookOutlineValidator().validate(
        {"textbook_structure": structure}
    )
    assert "OUTLINE_THEMES_EMPTY" in _codes(result)
    # Day-1 warning severity — never fails the gate.
    assert result.passed is True


def test_theme_chapter_unresolved() -> None:
    structure = _base_structure()
    structure["semantic_outline"]["themes"][0]["chapter_ids"] = [
        "ch1",
        "ch99",  # not on chapters[]
    ]
    result = TextbookOutlineValidator().validate(
        {"textbook_structure": structure}
    )
    assert "OUTLINE_THEME_CHAPTER_UNRESOLVED" in _codes(result)


def test_draft_to_empty() -> None:
    structure = _base_structure()
    structure["draft_terminal_objectives"] = []
    result = TextbookOutlineValidator().validate(
        {"textbook_structure": structure}
    )
    assert "OUTLINE_DRAFT_TO_EMPTY" in _codes(result)


def test_draft_to_malformed() -> None:
    structure = _base_structure()
    structure["draft_terminal_objectives"] = [
        {"id": "TO-01", "statement": "", "bloom_level": "wizardry"}
    ]
    result = TextbookOutlineValidator().validate(
        {"textbook_structure": structure}
    )
    assert "OUTLINE_DRAFT_TO_MALFORMED" in _codes(result)


def test_happy_path_passes() -> None:
    result = TextbookOutlineValidator().validate(
        {"textbook_structure": _base_structure()}
    )
    assert result.passed is True
    assert result.issues == []
    assert result.score == 1.0


def test_decision_capture_emitted() -> None:
    capture = _StubDecisionCapture()
    TextbookOutlineValidator().validate(
        {
            "textbook_structure": _base_structure(),
            "decision_capture": capture,
        }
    )
    assert len(capture.events) == 1
    assert capture.events[0]["decision_type"] == "content_structure_check"
    assert len(capture.events[0]["rationale"]) >= 20


def test_decision_capture_emitted_on_skip() -> None:
    capture = _StubDecisionCapture()
    TextbookOutlineValidator().validate(
        {
            "textbook_structure": {"chapters": []},
            "decision_capture": capture,
        }
    )
    assert len(capture.events) == 1
    assert capture.events[0]["decision_type"] == "content_structure_check"


def test_path_loaded_from_disk(tmp_path: Path) -> None:
    p = tmp_path / "textbook_structure.json"
    p.write_text(json.dumps(_base_structure()), encoding="utf-8")
    result = TextbookOutlineValidator().validate(
        {"textbook_structure_path": str(p)}
    )
    assert result.passed is True
    assert result.issues == []


def test_missing_path_fails_closed(tmp_path: Path) -> None:
    result = TextbookOutlineValidator().validate(
        {"textbook_structure_path": str(tmp_path / "nope.json")}
    )
    assert result.passed is False
    assert "OUTLINE_STRUCTURE_PATH_MISSING" in _codes(result)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
