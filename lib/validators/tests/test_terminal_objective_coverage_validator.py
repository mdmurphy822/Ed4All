"""Tests for ``TerminalObjectiveCoverageValidator`` (fail-fast gate).

This is the Layer-B planning-time gate that guarantees the
``UNCOVERED_TERMINAL_OUTCOME`` archival failure cannot reoccur silently.

Cases:

1. ``test_co_bearing_orphan_terminal_blocks`` — a CO-bearing course with a
   terminal that no chapter objective rolls up to → critical
   ``ORPHAN_TERMINAL_NO_CHAPTER_OBJECTIVE`` (the systemic guarantee).
2. ``test_co_bearing_consistent_passes`` — every terminal has a rolling-up
   CO → pass (backward-compat for RDF/SHACL-style courses).
3. ``test_co_less_chapter_in_structure_passes`` — CO-less course whose
   terminals point at chapters that exist (modulo label normalisation) →
   pass (backward-compat for OpenStax / well-formed reuse runs).
4. ``test_co_less_chapter_missing_blocks`` — CO-less terminal pointing at a
   chapter NOT in the structure → critical ``ORPHAN_TERMINAL_NO_CHAPTER``.
5. ``test_co_less_no_structure_warns_and_passes`` — CO-less course with no
   textbook_structure → ``TERMINAL_COVERAGE_UNVERIFIED`` warning + pass.
6. ``test_empty_objectives_passes`` — no terminals → no-op pass.
7. ``test_no_objectives_input_passes`` — absent objectives → no-op pass.
8. ``test_decision_capture_emitted`` — one ``content_structure_check``
   event per run (LLM-instrumentation regression contract).
9. ``test_paths_loaded_from_disk`` — both ``*_path`` inputs are loaded.
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

from lib.validators.terminal_objective_coverage import (  # noqa: E402
    TerminalObjectiveCoverageValidator,
)


class _StubDecisionCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def _co_bearing(*, orphan: bool) -> Dict[str, Any]:
    """A CO-bearing objectives set; ``orphan`` adds an uncovered TO-99."""
    terminals = [{"id": "TO-01"}, {"id": "TO-02"}]
    if orphan:
        terminals.append({"id": "TO-99"})
    return {
        "terminal_objectives": terminals,
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {"id": "CO-01", "parent_to": "TO-01"},
                    {"id": "CO-02", "parent_to": "TO-02"},
                ],
            }
        ],
    }


def _co_less(*, chapters=("1", "2")) -> Dict[str, Any]:
    """A CO-less objectives set; terminals carry a ``chapter`` pointer."""
    return {
        "terminal_objectives": [
            {"id": f"TO-{i:02d}", "chapter": c}
            for i, c in enumerate(chapters, start=1)
        ],
        "chapter_objectives": [],
    }


def _structure(chapter_ids=("ch1", "ch2")) -> Dict[str, Any]:
    return {
        "chapters": [
            {"id": cid, "chapter_text": "x" * 50} for cid in chapter_ids
        ]
    }


def test_co_bearing_orphan_terminal_blocks() -> None:
    r = TerminalObjectiveCoverageValidator().validate(
        {"synthesized_objectives": _co_bearing(orphan=True)}
    )
    assert r.passed is False
    assert r.action == "block"
    codes = [i.code for i in r.issues]
    assert "ORPHAN_TERMINAL_NO_CHAPTER_OBJECTIVE" in codes
    # The orphan ID is named in the message.
    assert any("TO-99" in i.message for i in r.issues)


def test_co_bearing_consistent_passes() -> None:
    r = TerminalObjectiveCoverageValidator().validate(
        {"synthesized_objectives": _co_bearing(orphan=False)}
    )
    assert r.passed is True
    assert r.critical_count == 0


def test_co_less_chapter_in_structure_passes() -> None:
    # Terminals point at "1"/"2"; structure has "ch1"/"ch2". The label
    # normalisation must reconcile these so the gate does NOT over-fire.
    r = TerminalObjectiveCoverageValidator().validate(
        {
            "synthesized_objectives": _co_less(chapters=("1", "2")),
            "textbook_structure": _structure(("ch1", "ch2")),
        }
    )
    assert r.passed is True
    assert r.critical_count == 0


def test_co_less_chapter_missing_blocks() -> None:
    # Terminal TO-02 points at chapter "9" which is not in the structure.
    r = TerminalObjectiveCoverageValidator().validate(
        {
            "synthesized_objectives": _co_less(chapters=("1", "9")),
            "textbook_structure": _structure(("ch1", "ch2")),
        }
    )
    assert r.passed is False
    codes = [i.code for i in r.issues]
    assert "ORPHAN_TERMINAL_NO_CHAPTER" in codes
    assert any("TO-02" in i.message for i in r.issues)


def test_co_less_no_structure_warns_and_passes() -> None:
    r = TerminalObjectiveCoverageValidator().validate(
        {"synthesized_objectives": _co_less(chapters=("1", "2"))}
    )
    assert r.passed is True
    codes = [i.code for i in r.issues]
    assert "TERMINAL_COVERAGE_UNVERIFIED" in codes


def test_co_bearing_no_backpointers_warns_and_passes() -> None:
    # CO-bearing course where ZERO chapter objectives carry a
    # parent-terminal key (what the deterministic synthesizer / Stage-2 /
    # _normalize_objective_entry produce). The roll-up model is not in
    # use, so orphanhood is unprovable. The gate must pass with a
    # TERMINAL_COVERAGE_UNVERIFIED warning rather than false-flag every
    # terminal as an orphan.
    objectives = {
        "terminal_objectives": [{"id": "TO-01"}, {"id": "TO-02"}],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [{"id": "CO-01"}, {"id": "CO-02"}],
            }
        ],
    }
    r = TerminalObjectiveCoverageValidator().validate(
        {"synthesized_objectives": objectives}
    )
    assert r.passed is True
    assert r.critical_count == 0
    codes = [i.code for i in r.issues]
    assert "TERMINAL_COVERAGE_UNVERIFIED" in codes
    # No orphan codes should be emitted for any terminal.
    assert "ORPHAN_TERMINAL_NO_CHAPTER_OBJECTIVE" not in codes


def test_empty_objectives_passes() -> None:
    r = TerminalObjectiveCoverageValidator().validate(
        {"synthesized_objectives": {"terminal_objectives": []}}
    )
    assert r.passed is True
    assert r.issues == []


def test_no_objectives_input_passes() -> None:
    r = TerminalObjectiveCoverageValidator().validate({})
    assert r.passed is True


def test_decision_capture_emitted() -> None:
    cap = _StubDecisionCapture()
    TerminalObjectiveCoverageValidator().validate(
        {
            "synthesized_objectives": _co_bearing(orphan=True),
            "decision_capture": cap,
        }
    )
    assert len(cap.events) == 1
    evt = cap.events[0]
    assert evt["decision_type"] == "content_structure_check"
    assert len(evt["rationale"]) >= 20


def test_paths_loaded_from_disk(tmp_path: Path) -> None:
    obj_path = tmp_path / "synthesized_objectives.json"
    ts_path = tmp_path / "textbook_structure.json"
    obj_path.write_text(
        json.dumps(_co_less(chapters=("1", "9"))), encoding="utf-8"
    )
    ts_path.write_text(
        json.dumps(_structure(("ch1", "ch2"))), encoding="utf-8"
    )
    r = TerminalObjectiveCoverageValidator().validate(
        {
            "synthesized_objectives_path": str(obj_path),
            "textbook_structure_path": str(ts_path),
        }
    )
    assert r.passed is False
    assert "ORPHAN_TERMINAL_NO_CHAPTER" in [i.code for i in r.issues]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
