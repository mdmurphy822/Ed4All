"""W10 — cross-quiz exact-duplicate stem dedup guard.

Book-1 canary defect: 325/765 emitted quiz items sat in exact-duplicate
groups across quizzes (every per-TO quiz shares the corpus-wide chunk pool,
so deterministic templates minted the same stem in many quizzes). The
pre-emit guard ``_dedup_assessment_stems`` collapses exact-duplicate stems
across the WHOLE quiz set — first occurrence (week order) wins, later
duplicates are dropped, quizzes emptied by the collapse are removed, and any
objective whose only item(s) were duplicates is surfaced in the report
(honest count reduction beats salad duplication).

Fixtures are synthetic — no course slugs, no corpus content.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _dedup_assessment_stems,
    _normalize_stem_for_dedup,
)


@dataclass
class _Q:
    stem: str
    objective_id: str = "TO-01"


@dataclass
class _Quiz:
    assessment_id: str
    questions: List[Any] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# _normalize_stem_for_dedup
# --------------------------------------------------------------------------- #

def test_normalize_strips_html_case_and_whitespace():
    a = _normalize_stem_for_dedup("<p>Briefly  explain <em>X</em>.</p>")
    b = _normalize_stem_for_dedup("briefly explain X .")
    assert a == b


def test_normalize_empty_stem_yields_empty_key():
    assert _normalize_stem_for_dedup(None) == ""
    assert _normalize_stem_for_dedup("<p> </p>") == ""


# --------------------------------------------------------------------------- #
# _dedup_assessment_stems
# --------------------------------------------------------------------------- #

def test_first_occurrence_wins_later_duplicates_dropped():
    q1 = _Quiz("q1", [_Q("<p>Explain replication.</p>", "TO-01"),
                      _Q("<p>Define sharding.</p>", "CO-01")])
    q2 = _Quiz("q2", [_Q("<p>Explain  replication.</p>", "TO-02"),
                      _Q("<p>Describe caching.</p>", "CO-05")])
    survivors, report = _dedup_assessment_stems([q1, q2])
    assert len(survivors) == 2
    assert [len(s.questions) for s in survivors] == [2, 1]
    assert report["total_before"] == 4
    assert report["total_after"] == 3
    assert report["duplicates_removed"] == 1
    assert report["duplicate_groups"] == 1


def test_quiz_emptied_by_collapse_is_dropped():
    q1 = _Quiz("q1", [_Q("<p>Explain replication.</p>", "TO-01")])
    q2 = _Quiz("q2", [_Q("<p>Explain replication.</p>", "TO-02")])
    survivors, report = _dedup_assessment_stems([q1, q2])
    assert [s.assessment_id for s in survivors] == ["q1"]
    assert report["dropped_assessments"] == ["q2"]


def test_objective_losing_all_coverage_is_reported():
    # TO-02's ONLY item is a duplicate of TO-01's — the drop is honest but
    # must be REPORTED, never silent.
    q1 = _Quiz("q1", [_Q("<p>Explain replication.</p>", "TO-01")])
    q2 = _Quiz("q2", [_Q("<p>Explain replication.</p>", "TO-02"),
                      _Q("<p>Define quorum.</p>", "CO-09")])
    _, report = _dedup_assessment_stems([q1, q2])
    assert report["objectives_losing_coverage"] == ["TO-02"]


def test_objective_covered_elsewhere_not_reported():
    q1 = _Quiz("q1", [_Q("<p>Explain replication.</p>", "TO-01")])
    q2 = _Quiz("q2", [_Q("<p>Explain replication.</p>", "TO-02"),
                      _Q("<p>Define quorum.</p>", "TO-02")])
    _, report = _dedup_assessment_stems([q1, q2])
    assert report["objectives_losing_coverage"] == []


def test_accepts_cached_dict_shape():
    # Checkpoint-replayed units arrive as to_dict() payloads.
    q1 = {"assessment_id": "q1",
          "questions": [{"stem": "<p>Explain replication.</p>",
                         "objective_id": "TO-01"}]}
    q2 = {"assessment_id": "q2",
          "questions": [{"stem": "Explain   replication.",
                         "objective_id": "TO-02"},
                        {"stem": "<p>Define quorum.</p>",
                         "objective_id": "CO-09"}]}
    survivors, report = _dedup_assessment_stems([q1, q2])
    assert report["duplicates_removed"] == 1
    assert len(survivors) == 2
    assert len(survivors[1]["questions"]) == 1


def test_mixed_dataclass_and_dict_shapes():
    q1 = _Quiz("q1", [_Q("<p>Explain replication.</p>", "TO-01")])
    q2 = {"assessment_id": "q2",
          "questions": [{"stem": "Explain replication.",
                         "objective_id": "TO-02"}]}
    survivors, report = _dedup_assessment_stems([q1, q2])
    assert [getattr(s, "assessment_id", None) or s.get("assessment_id")
            for s in survivors] == ["q1"]
    assert report["dropped_assessments"] == ["q2"]


def test_no_duplicates_is_a_no_op():
    q1 = _Quiz("q1", [_Q("<p>Explain replication.</p>", "TO-01")])
    q2 = _Quiz("q2", [_Q("<p>Define quorum.</p>", "TO-02")])
    survivors, report = _dedup_assessment_stems([q1, q2])
    assert len(survivors) == 2
    assert report["duplicates_removed"] == 0
    assert report["duplicate_groups"] == 0
    assert report["dropped_assessments"] == []


def test_empty_stems_never_collapse_each_other():
    q1 = _Quiz("q1", [_Q("", "TO-01"), _Q("", "TO-02")])
    survivors, report = _dedup_assessment_stems([q1])
    assert report["duplicates_removed"] == 0
    assert len(survivors[0].questions) == 2


def test_deterministic_across_repeat_invocations():
    def build():
        return [
            _Quiz("q1", [_Q("<p>Explain replication.</p>", "TO-01"),
                         _Q("<p>Define sharding.</p>", "CO-01")]),
            _Quiz("q2", [_Q("<p>Explain replication.</p>", "TO-02")]),
        ]
    s1, r1 = _dedup_assessment_stems(build())
    s2, r2 = _dedup_assessment_stems(build())
    assert r1 == r2
    assert [q.stem for a in s1 for q in a.questions] == \
        [q.stem for a in s2 for q in a.questions]


# --------------------------------------------------------------------------- #
# Decision-capture enum registration
# --------------------------------------------------------------------------- #

def test_assessment_stem_dedup_decision_type_registered():
    schema = json.loads(
        (PROJECT_ROOT / "schemas" / "events"
         / "decision_event.schema.json").read_text(encoding="utf-8")
    )
    enum = schema["properties"]["decision_type"]["enum"]
    assert "assessment_stem_dedup" in enum
