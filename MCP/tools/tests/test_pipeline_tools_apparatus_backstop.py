"""Defect-C apparatus-seed backstop + 2-chunk window integration (hermetic).

Covers the stage-2 candidate backstop
``MCP.tools.pipeline_tools._drop_apparatus_seeded_candidates`` — it drops any
Pass-C survivor whose STATEMENT is exercise-/practice-apparatus text (matched
against the shared ``lib/objectives/apparatus_lexicon`` profile), with keep-≥1
protection and a decision capture — and the end-to-end window-prep integration
on a synthetic 2-chunk chapter (one exercise_set chunk + one prose chunk).

No course slugs / publisher names / data paths — all fixtures synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.chunk_window import (  # noqa: E402
    APPARATUS_SENTINEL,
    group_chunks_into_windows,
)
from MCP.tools.pipeline_tools import (  # noqa: E402
    _drop_apparatus_seeded_candidates,
)


class _FakeCapture:
    """Records ``log_decision`` calls for assertion."""

    def __init__(self) -> None:
        self.calls: list = []

    def log_decision(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _co(statement: str) -> dict:
    return {"statement": statement, "source_chunk_ids": ["c1"]}


# ---------------------------------------------------------------------------
# Backstop unit
# ---------------------------------------------------------------------------
def test_backstop_drops_apparatus_statement_keeps_prose():
    pool = [
        _co("Explain how a denominator names equal parts of a whole."),
        _co("In the following exercises, add the given fractions."),
        _co("Determine the least common denominator of two fractions."),
    ]
    cap = _FakeCapture()
    kept = _drop_apparatus_seeded_candidates(pool, capture=cap)
    stmts = [c["statement"] for c in kept]
    assert "In the following exercises, add the given fractions." not in stmts
    # Order preserved among survivors.
    assert stmts == [
        "Explain how a denominator names equal parts of a whole.",
        "Determine the least common denominator of two fractions.",
    ]
    # One content_selection decision with dynamic counts (≥20-char rationale).
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["decision_type"] == "content_selection"
    assert "dropped 1 of 3" in call["decision"]
    assert len(call["rationale"]) >= 20


def test_backstop_no_apparatus_is_noop():
    pool = [_co("Solve a linear equation in one variable.")]
    cap = _FakeCapture()
    kept = _drop_apparatus_seeded_candidates(pool, capture=cap)
    assert kept == pool
    assert cap.calls == []  # nothing dropped → no decision


def test_backstop_keep_at_least_one_when_all_flagged():
    """Every survivor is apparatus → keep the longest-statement one (never empty)."""
    pool = [
        _co("In the following exercises, simplify."),
        _co("In the following exercises, add the fractions and reduce fully."),
        _co("Practice Makes Perfect"),
    ]
    cap = _FakeCapture()
    kept = _drop_apparatus_seeded_candidates(pool, capture=cap)
    assert len(kept) == 1
    # The longest-statement flagged candidate is rescued.
    assert kept[0]["statement"] == (
        "In the following exercises, add the fractions and reduce fully."
    )
    assert "dropped 2 of 3" in cap.calls[0]["decision"]


def test_backstop_empty_pool_is_noop():
    assert _drop_apparatus_seeded_candidates([], capture=_FakeCapture()) == []


# ---------------------------------------------------------------------------
# Synthetic 2-chunk window integration (part (a) end-to-end)
# ---------------------------------------------------------------------------
def _chunk(cid: str, slug: str, body: str, **meta) -> dict:
    c = {
        "id": cid,
        "text": body,
        "source": {
            "source_references": [
                {"sourceId": f"semantik:{slug}#{cid}", "role": "primary"}
            ]
        },
    }
    c.update(meta)
    return c


def test_two_chunk_window_sanitation_integration():
    """One exercise_set chunk + one prose chunk pack into one window; only the
    apparatus chunk is sentineled, the prose chunk survives byte-clean."""
    chapter = {"id": "ch1", "source_file": "one.html", "chapter_text": "x"}
    chunks = [
        _chunk(
            "prose1", "one",
            "A fraction represents part of a whole; the denominator names "
            "the number of equal parts.",
        ),
        _chunk(
            "ex1", "one",
            "1. Add 1/2 + 1/3. 2. Simplify 6/8. 3. Reduce 10/15.",
            composite_unit="exercise_set",
            heading="Section Exercises",
        ),
    ]
    windows = group_chunks_into_windows(
        chapter, chunks,
        num_ctx=4096,
        system_prompt="You are a synthesis author.",
        draft_block="  (none)",
        max_tokens=128,
        sanitize_seeds=True,
    )
    # Single window holds both chunks (budget is ample).
    assert len(windows) == 1
    by_id = {c["id"]: c["text"] for c in windows[0].chunks}
    assert by_id["prose1"].startswith("A fraction represents part of a whole")
    assert APPARATUS_SENTINEL not in by_id["prose1"]
    assert by_id["ex1"] == f"Section Exercises\n{APPARATUS_SENTINEL}"
    # chunk_ids / citability untouched (both ids remain the allowed-id set).
    assert set(windows[0].chunk_ids) == {"prose1", "ex1"}


def test_backstop_keeps_callout_english_statements():
    """Statements containing callout-label English ("how to", "learning
    objectives") are legitimate COs and must survive the backstop."""
    pool = [
        _co("Explain how to solve a two-step linear equation."),
        _co("Identify the learning objectives addressed by each module."),
        _co("In the following exercises, solve each system."),
    ]
    kept = _drop_apparatus_seeded_candidates(pool)
    stmts = [c["statement"] for c in kept]
    assert "Explain how to solve a two-step linear equation." in stmts
    assert (
        "Identify the learning objectives addressed by each module." in stmts
    )
    assert "In the following exercises, solve each system." not in stmts
