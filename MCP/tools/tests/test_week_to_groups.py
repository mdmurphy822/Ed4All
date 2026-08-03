"""ED4ALL_WEEK_TO_GROUPS — TO-membership per-week CO grouping (opt-in).

The outline tier builds each ``"Week N"`` chapter_objectives group from a flat
ceil-stride CO slice (positional), which ignores TO cluster boundaries so a
week's pages bind COs that roll up to a DIFFERENT terminal objective. The
``ED4ALL_WEEK_TO_GROUPS`` gate (default OFF) rebuilds the groups by TO
membership when ``duration_weeks == len(terminal_objectives)``: week N's group
is TO-N's child COs.

The grouping is single-sourced in
``MCP.tools.pipeline_tools._week_co_groups`` — BOTH the persist path
(``_plan_course_structure`` → ``synthesized_objectives.json``) and the emit
path (``_generate_course_content`` → ``week_chapter_cos``) consume it, and the
validator reads the persisted ``"Week N"`` groups directly (the
``load_canonical_objectives`` week-range branch), so emit ↔ on-disk ↔ validator
stay identical by construction.

Pure-helper unit tests — no LLM client, no router.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools as _pt  # noqa: E402

# Shared single-sourced slicer (what the ceil-stride fallback delegates to).
_CF_SCRIPTS = PROJECT_ROOT / "Courseforge" / "scripts" / "rendering"
if str(_CF_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CF_SCRIPTS))
from generate_course import (  # noqa: E402
    _slice_cos_for_week,
    load_canonical_objectives,
    resolve_week_objectives,
)

_ENV = "ED4ALL_WEEK_TO_GROUPS"


def _tos(n):
    return [{"id": f"TO-{i:02d}", "statement": f"terminal {i}"} for i in range(1, n + 1)]


def _interleaved_cos():
    """6 COs across 3 TOs, INTERLEAVED so ceil-stride != TO-membership.

    ceil-stride(3wk, step 2): W1=[CO-01,CO-02] W2=[CO-03,CO-04] W3=[CO-05,CO-06]
    TO-membership:            W1=[CO-01,CO-04] W2=[CO-02,CO-05] W3=[CO-03,CO-06]
    """
    cos = []
    for i in range(1, 7):
        tid = f"TO-{((i - 1) % 3) + 1:02d}"
        cos.append({"id": f"CO-{i:02d}", "statement": f"co {i}", "terminal_id": tid})
    return cos


def _ids(groups):
    return {w: [c["id"] for c in v] for w, v in groups.items()}


# --------------------------------------------------------------------------- #
# 1. Flag off → byte-identical to the ceil-stride slicer (regression).
# --------------------------------------------------------------------------- #
def test_flag_off_matches_ceil_stride(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    cos = _interleaved_cos()
    tos = _tos(3)
    got = _pt._week_co_groups(cos, tos, 3)

    # Reproduce the legacy inline calls: shared placed-ids across weeks.
    placed: set = set()
    expected = {
        w: _slice_cos_for_week(cos, 3, w, placed_ids=placed)
        for w in range(1, 4)
    }
    assert _ids(got) == _ids(expected)
    # And it is the POSITIONAL slice (not TO-membership).
    assert _ids(got) == {
        1: ["CO-01", "CO-02"], 2: ["CO-03", "CO-04"], 3: ["CO-05", "CO-06"],
    }


def test_flag_off_via_explicit_arg(monkeypatch):
    monkeypatch.setenv(_ENV, "1")  # env on, but explicit arg wins
    cos = _interleaved_cos()
    got = _pt._week_co_groups(cos, _tos(3), 3, enabled=False)
    assert _ids(got) == {
        1: ["CO-01", "CO-02"], 2: ["CO-03", "CO-04"], 3: ["CO-05", "CO-06"],
    }


# --------------------------------------------------------------------------- #
# 2. Flag on, weeks == num_tos → each week == TO-N's child COs, covered once.
# --------------------------------------------------------------------------- #
def test_flag_on_to_membership(monkeypatch):
    monkeypatch.setenv(_ENV, "on")
    cos = _interleaved_cos()
    tos = _tos(3)
    got = _pt._week_co_groups(cos, tos, 3)
    assert _ids(got) == {
        1: ["CO-01", "CO-04"], 2: ["CO-02", "CO-05"], 3: ["CO-03", "CO-06"],
    }


def test_flag_on_coverage_exactly_once(monkeypatch):
    """All COs placed exactly once — no drop, no dup (77-style shape)."""
    monkeypatch.setenv(_ENV, "true")
    tos = _tos(11)
    cos = []
    for i in range(1, 78):  # 77 COs
        tid = f"TO-{((i - 1) % 11) + 1:02d}"
        cos.append({"id": f"CO-{i:02d}", "statement": f"co {i}", "terminal_id": tid})
    got = _pt._week_co_groups(cos, tos, 11)
    placed = [c for g in got.values() for c in g]
    assert len(placed) == 77
    assert sorted(c["id"] for c in placed) == sorted(c["id"] for c in cos)
    # Each week's group == exactly that TO's children (in flat CO order).
    for w in range(1, 12):
        tid = f"TO-{w:02d}"
        expect = [c["id"] for c in cos if c["terminal_id"] == tid]
        assert _ids(got)[w] == expect


def test_flag_on_within_group_stable_flat_order(monkeypatch):
    monkeypatch.setenv(_ENV, "yes")
    # CO-06 appears BEFORE CO-01 in the flat list but both roll up to TO-01.
    cos = [
        {"id": "CO-06", "statement": "x", "terminal_id": "TO-01"},
        {"id": "CO-02", "statement": "x", "terminal_id": "TO-02"},
        {"id": "CO-01", "statement": "x", "terminal_id": "TO-01"},
    ]
    got = _pt._week_co_groups(cos, _tos(2), 2)
    # TO-01 group preserves flat order: CO-06 before CO-01.
    assert _ids(got)[1] == ["CO-06", "CO-01"]
    assert _ids(got)[2] == ["CO-02"]


def test_flag_on_orphan_co_appended_to_last_week(monkeypatch):
    """A CO whose parent TO doesn't resolve is kept (last week), never dropped."""
    monkeypatch.setenv(_ENV, "1")
    cos = [
        {"id": "CO-01", "statement": "x", "terminal_id": "TO-01"},
        {"id": "CO-02", "statement": "x", "terminal_id": "TO-02"},
        {"id": "CO-99", "statement": "x", "terminal_id": "TO-NOPE"},
    ]
    got = _pt._week_co_groups(cos, _tos(2), 2)
    assert _ids(got)[2] == ["CO-02", "CO-99"]
    # No drop.
    assert {c["id"] for g in got.values() for c in g} == {"CO-01", "CO-02", "CO-99"}


def test_case_insensitive_terminal_backlink(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    cos = [
        {"id": "CO-01", "statement": "x", "terminal_id": "to-01"},  # lower
        {"id": "CO-02", "statement": "x", "terminal_id": "TO-02"},
    ]
    got = _pt._week_co_groups(cos, _tos(2), 2)
    assert _ids(got) == {1: ["CO-01"], 2: ["CO-02"]}


# --------------------------------------------------------------------------- #
# 3. Flag on, weeks != num_tos → ceil-stride fallback + warning.
# --------------------------------------------------------------------------- #
def test_flag_on_weeks_ne_num_tos_falls_back(monkeypatch, caplog):
    monkeypatch.setenv(_ENV, "1")
    cos = _interleaved_cos()
    tos = _tos(3)  # 3 TOs but ask for 4 weeks
    import logging
    with caplog.at_level(logging.WARNING):
        got = _pt._week_co_groups(cos, tos, 4)

    # Byte-identical to the flag-off ceil-stride over 4 weeks.
    placed: set = set()
    expected = {
        w: _slice_cos_for_week(cos, 4, w, placed_ids=placed) for w in range(1, 5)
    }
    assert _ids(got) == _ids(expected)
    # Warning names BOTH counts.
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "4" in msgs and "3" in msgs
    assert "ED4ALL_WEEK_TO_GROUPS" in msgs


def test_resolve_flag_parse_with_fallback(monkeypatch):
    for tok in ("1", "true", "TRUE", "yes", "on", "On"):
        monkeypatch.setenv(_ENV, tok)
        assert _pt.resolve_week_to_groups() is True
    for tok in ("0", "false", "no", "off", "garbage", ""):
        monkeypatch.setenv(_ENV, tok)
        assert _pt.resolve_week_to_groups() is False
    monkeypatch.delenv(_ENV, raising=False)
    assert _pt.resolve_week_to_groups() is False
    # Explicit arg wins over env.
    monkeypatch.setenv(_ENV, "1")
    assert _pt.resolve_week_to_groups(False) is False


# --------------------------------------------------------------------------- #
# 4. PARITY: the validator (load_canonical_objectives) reads the persisted
#    "Week N" groups the helper emits — so validator-week-N == helper-week-N.
# --------------------------------------------------------------------------- #
def test_parity_validator_reads_persisted_to_membership(monkeypatch, tmp_path):
    monkeypatch.setenv(_ENV, "1")
    cos = _interleaved_cos()
    tos = _tos(3)

    # PERSIST path shape: one "Week N" group per week from the helper.
    groups = _pt._week_co_groups(cos, tos, 3)
    synthesized = {
        "course_name": "PARITY_TEST",
        "duration_weeks": 3,
        "terminal_objectives": tos,
        "chapter_objectives": [
            {"chapter": f"Week {w}", "objectives": groups[w]}
            for w in range(1, 4)
        ],
    }
    p = tmp_path / "synthesized_objectives.json"
    p.write_text(json.dumps(synthesized), encoding="utf-8")

    canonical = load_canonical_objectives(p)
    for w in range(1, 4):
        resolved = resolve_week_objectives(w, canonical)
        resolved_cos = [o["id"] for o in resolved if str(o["id"]).startswith("CO-")]
        assert resolved_cos == _ids(groups)[w], f"validator disagrees on week {w}"


def test_empty_cos_returns_empty_week_buckets(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    got = _pt._week_co_groups([], _tos(3), 3)
    assert got == {1: [], 2: [], 3: []}
