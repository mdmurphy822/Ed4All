"""Feature 1 — deterministic Bloom-relevel invariants (hermetic, no LLM).

Verifies: (1) default-OFF is a byte-stable no-op (bloom_level untouched, no
capture), (2) relevel correctness on the measured mismatch cases (apply-verb
mislabelled understand → apply; translate-verb mislabelled apply → understand),
(3) abcd.behavior.verb takes precedence over the statement's detected verb,
(4) an objective already in agreement is NOT touched, (5) an unknown / missing
verb or an invalid declared level is left alone, (6) the parse-with-fallback
resolver, (7) the decision-capture fires per releveled objective.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.bloom_relevel import (  # noqa: E402
    derive_level,
    relevel_objectives,
    resolve_bloom_relevel,
)


class _Capture:
    """Minimal DecisionCapture recorder."""

    def __init__(self) -> None:
        self.events = []

    def log_decision(self, **kwargs) -> None:  # noqa: ANN003
        self.events.append(kwargs)


def _obj(statement, bloom_level, *, abcd_verb=None, oid="CO-01"):
    o = {"id": oid, "statement": statement, "bloom_level": bloom_level}
    if abcd_verb is not None:
        o["abcd"] = {"behavior": {"verb": abcd_verb}}
    return o


def test_flag_off_noop_bytestable():
    """enabled=None + env unset → no relevel, statements + levels untouched."""
    objs = [_obj("Apply the theorem to the data", "understand")]
    before = [dict(o) for o in objs]
    result = relevel_objectives(objs, enabled=None, capture=_Capture())
    assert result.available is False
    assert result.releveled_count == 0
    assert objs == before  # byte-identical


def test_relevel_apply_verb_mislabelled_understand():
    """An apply-verb statement stamped 'understand' is releveled to 'apply'."""
    cap = _Capture()
    objs = [_obj("Apply the distributive property to the expression", "understand")]
    result = relevel_objectives(objs, enabled=True, capture=cap)
    assert result.available is True
    assert result.releveled_count == 1
    assert objs[0]["bloom_level"] == "apply"
    # Statement untouched.
    assert objs[0]["statement"] == (
        "Apply the distributive property to the expression"
    )
    # One capture, dynamic rationale.
    assert len(cap.events) == 1
    ev = cap.events[0]
    assert ev["decision_type"] == "bloom_level_assignment"
    assert "understand" in ev["rationale"] and "apply" in ev["rationale"]
    assert len(ev["rationale"]) >= 20
    alternative = ev["alternatives_considered"][0]
    assert alternative["option"]
    assert "apply" in alternative["reason_rejected"]


def test_relevel_translate_verb_mislabelled_apply():
    """A translate-verb (understand) statement stamped 'apply' → 'understand'."""
    objs = [_obj("Translate the phrase into an algebraic expression", "apply")]
    result = relevel_objectives(objs, enabled=True)
    assert result.releveled_count == 1
    assert objs[0]["bloom_level"] == "understand"


def test_abcd_verb_takes_precedence_over_statement():
    """abcd.behavior.verb wins over the statement's detected verb."""
    # Statement main verb detects 'evaluate'; abcd verb pins 'analyze'.
    objs = [
        _obj("Evaluate the competing arguments", "understand", abcd_verb="analyze")
    ]
    level, verb = derive_level(objs[0])
    assert (level, verb) == ("analyze", "analyze")
    result = relevel_objectives(objs, enabled=True)
    assert result.releveled_count == 1
    assert objs[0]["bloom_level"] == "analyze"


def test_agreement_not_releveled():
    """A correctly-labelled objective is left alone (scanned, not releveled)."""
    objs = [_obj("Analyze the data set for outliers", "analyze")]
    result = relevel_objectives(objs, enabled=True)
    assert result.scanned_count == 1
    assert result.releveled_count == 0
    assert objs[0]["bloom_level"] == "analyze"


def test_unknown_verb_left_alone():
    """No canonical verb in the statement → no derivable level → no change."""
    objs = [_obj("Foobar the widget appropriately", "apply")]
    result = relevel_objectives(objs, enabled=True)
    assert result.releveled_count == 0
    assert objs[0]["bloom_level"] == "apply"


def test_invalid_declared_level_skipped():
    """A missing / non-canonical declared level is not scanned (nothing to
    disagree)."""
    objs = [_obj("Apply the theorem", "")]
    result = relevel_objectives(objs, enabled=True)
    assert result.scanned_count == 0
    assert result.releveled_count == 0
    assert objs[0]["bloom_level"] == ""


def test_resolve_parse_with_fallback(monkeypatch):
    """Truthy tokens enable; garbage / unset → default OFF."""
    monkeypatch.delenv("ED4ALL_OBJECTIVE_BLOOM_RELEVEL", raising=False)
    assert resolve_bloom_relevel() is False
    for tok in ("1", "true", "YES", "on"):
        monkeypatch.setenv("ED4ALL_OBJECTIVE_BLOOM_RELEVEL", tok)
        assert resolve_bloom_relevel() is True
    for tok in ("0", "false", "banana", ""):
        monkeypatch.setenv("ED4ALL_OBJECTIVE_BLOOM_RELEVEL", tok)
        assert resolve_bloom_relevel() is False
    # Explicit arg wins.
    assert resolve_bloom_relevel(True) is True
    assert resolve_bloom_relevel(False) is False
