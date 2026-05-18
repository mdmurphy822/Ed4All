"""Tests for ``ChapterObjectiveCoverageValidator`` (Stage-2 coverage).

Three-stage large-LLM textbook synthesis architecture, Wave A — see
``plans/textbook-llm-synthesis-3stage-2026-05.md`` §8 row 3.

Cases:

1. ``test_skip_with_pass_when_stage2_signal_absent`` — chapters without
   `chapter_text` (a default-off run) → no-op ``passed=True``.
2. ``test_chapter_obj_missing`` — a chapter with text, NOT in
   chapter_synthesis_failures, produced 0 COs → ``CHAPTER_OBJ_MISSING``.
3. ``test_chapter_obj_partial_coverage`` — a chapter with text that IS
   in chapter_synthesis_failures produced 0 COs →
   ``CHAPTER_OBJ_PARTIAL_COVERAGE`` (informational).
4. ``test_reconciled_to_empty`` — empty reconciled
   ``terminal_objectives`` → ``RECONCILED_TO_EMPTY``.
5. ``test_happy_path_passes`` — full coverage + non-empty terminals.
6. ``test_decision_capture_emitted`` — one ``content_structure_check``
   event per run.
7. ``test_path_loaded_from_disk`` — both `*_path` inputs are loaded.
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

from lib.validators.chapter_objective_coverage import (  # noqa: E402
    ChapterObjectiveCoverageValidator,
)


class _StubDecisionCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def _structure() -> Dict[str, Any]:
    """A Stage-2 textbook_structure: two chapters, both with text."""
    return {
        "chapters": [
            {"id": "ch1", "title": "Foundations", "chapter_text": "x" * 200},
            {"id": "ch2", "title": "Equations", "chapter_text": "y" * 200},
        ]
    }


def _objectives(*, ch1_cos: int = 1, ch2_cos: int = 1,
                terminals: int = 2) -> Dict[str, Any]:
    """A synthesized_objectives dict with per-chapter CO attribution."""
    chapter_objectives: List[Dict[str, Any]] = []
    for _ in range(ch1_cos):
        chapter_objectives.append(
            {"id": "CO-01", "statement": "...", "chapter_id": "ch1"}
        )
    for _ in range(ch2_cos):
        chapter_objectives.append(
            {"id": "CO-02", "statement": "...", "chapter_id": "ch2"}
        )
    terminal_objectives = [
        {"id": f"TO-{i+1:02d}", "statement": "..."}
        for i in range(terminals)
    ]
    return {
        "chapter_objectives": chapter_objectives,
        "terminal_objectives": terminal_objectives,
    }


def _codes(result: Any) -> List[str]:
    return [i.code for i in result.issues]


def test_skip_with_pass_when_stage2_signal_absent() -> None:
    # Chapters carry NO chapter_text → default-off run.
    legacy = {"chapters": [{"id": "ch1", "title": "Foundations"}]}
    result = ChapterObjectiveCoverageValidator().validate(
        {
            "textbook_structure": legacy,
            "synthesized_objectives": _objectives(),
        }
    )
    assert result.passed is True
    assert result.issues == []


def test_chapter_obj_missing() -> None:
    # ch2 has text, is NOT in failures, yet produced 0 COs.
    result = ChapterObjectiveCoverageValidator().validate(
        {
            "textbook_structure": _structure(),
            "synthesized_objectives": _objectives(ch2_cos=0),
        }
    )
    assert "CHAPTER_OBJ_MISSING" in _codes(result)


def test_chapter_obj_partial_coverage() -> None:
    # ch2 produced 0 COs but IS in chapter_synthesis_failures.
    result = ChapterObjectiveCoverageValidator().validate(
        {
            "textbook_structure": _structure(),
            "synthesized_objectives": _objectives(ch2_cos=0),
            "chapter_synthesis_failures": ["ch2"],
        }
    )
    codes = _codes(result)
    assert "CHAPTER_OBJ_PARTIAL_COVERAGE" in codes
    assert "CHAPTER_OBJ_MISSING" not in codes
    assert result.passed is True


def test_reconciled_to_empty() -> None:
    result = ChapterObjectiveCoverageValidator().validate(
        {
            "textbook_structure": _structure(),
            "synthesized_objectives": _objectives(terminals=0),
        }
    )
    assert "RECONCILED_TO_EMPTY" in _codes(result)


def test_happy_path_passes() -> None:
    result = ChapterObjectiveCoverageValidator().validate(
        {
            "textbook_structure": _structure(),
            "synthesized_objectives": _objectives(),
        }
    )
    assert result.passed is True
    assert result.issues == []
    assert result.score == 1.0


def test_decision_capture_emitted() -> None:
    capture = _StubDecisionCapture()
    ChapterObjectiveCoverageValidator().validate(
        {
            "textbook_structure": _structure(),
            "synthesized_objectives": _objectives(),
            "decision_capture": capture,
        }
    )
    assert len(capture.events) == 1
    assert capture.events[0]["decision_type"] == "content_structure_check"
    assert len(capture.events[0]["rationale"]) >= 20


def test_decision_capture_emitted_on_skip() -> None:
    capture = _StubDecisionCapture()
    ChapterObjectiveCoverageValidator().validate(
        {
            "textbook_structure": {"chapters": []},
            "synthesized_objectives": _objectives(),
            "decision_capture": capture,
        }
    )
    assert len(capture.events) == 1
    assert capture.events[0]["decision_type"] == "content_structure_check"


def test_path_loaded_from_disk(tmp_path: Path) -> None:
    sp = tmp_path / "textbook_structure.json"
    op = tmp_path / "synthesized_objectives.json"
    sp.write_text(json.dumps(_structure()), encoding="utf-8")
    op.write_text(json.dumps(_objectives()), encoding="utf-8")
    result = ChapterObjectiveCoverageValidator().validate(
        {
            "textbook_structure_path": str(sp),
            "synthesized_objectives_path": str(op),
        }
    )
    assert result.passed is True
    assert result.issues == []


def test_missing_path_fails_closed(tmp_path: Path) -> None:
    result = ChapterObjectiveCoverageValidator().validate(
        {
            "textbook_structure_path": str(tmp_path / "nope.json"),
            "synthesized_objectives": _objectives(),
        }
    )
    assert result.passed is False
    assert "CHAPTER_OBJ_STRUCTURE_PATH_MISSING" in _codes(result)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
