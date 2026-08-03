"""Phase 6 Subtask 5 — tests for ``AbcdObjectiveValidator``.

Cases covered (mirrors the Subtask 5 spec):

1. ``test_passes_when_verb_matches_bloom_level`` — happy path; ABCD
   verb is in ``BLOOMS_VERBS[level]``; ``passed=True, action=None``.
2. ``test_returns_regenerate_on_verb_mismatch`` — ABCD verb is NOT in
   the level's verb set; emits ``ABCD_VERB_BLOOM_MISMATCH`` warning
   and routes ``action="regenerate"``.
3. ``test_returns_warning_when_abcd_field_absent`` — ``require_abcd=True``
   and the LO has no ``abcd`` field; emits ``ABCD_MISSING`` warning,
   no regenerate action.
4. ``test_handles_capitalization_normalization`` — uppercase verb
   ``"IDENTIFY"`` resolves to lowercase ``"identify"`` against the
   verb set; happy path.
5. ``test_compose_abcd_prose_round_trip`` — integration with
   ``compose_abcd_prose``: a LO whose ABCD round-trips through the
   prose composer and back into the validator with verb in the canonical
   set still passes.
6. ``test_legacy_lo_without_abcd_skipped`` — ``require_abcd=False``
   (default) and the LO has no ``abcd`` field; silently skipped (no
   issue, score reflects the LO as passing).
7. ``test_empty_objectives_no_op_pass`` — empty LO list yields
   ``passed=True`` with no issues.
8. ``test_malformed_abcd_emits_block_action`` — ``abcd`` is not a
   mapping (or behavior.verb is missing); emits ``ABCD_MALFORMED``
   critical issue and routes ``action="block"``.
9. ``test_decision_capture_emitted_on_mismatch`` — verb mismatch wires
   a stub DecisionCapture; the ``log_decision`` call lands with
   ``decision_type="abcd_verb_bloom_mismatch"``.
10. ``test_decision_capture_emitted_on_pass`` — happy path emits a
    positive-path ``decision_type="abcd_authored"`` event.
11. ``test_synthesized_objectives_path_loaded`` — when
    ``inputs["synthesized_objectives_path"]`` points at an on-disk
    JSON file, the validator loads + flattens both
    ``terminal_objectives`` and ``chapter_objectives``.
12. ``test_camelcase_bloomlevel_accepted`` — LO with ``bloomLevel``
    (camelCase) instead of ``bloom_level`` is accepted.
13. ``test_no_bloom_level_emits_warning`` — LO has ABCD but no
    bloom_level → warning ``ABCD_NO_BLOOM_LEVEL`` and per-LO pass.

The tests use the real ``BLOOMS_VERBS`` table — Wave 6-A2 made the
loader deterministic so we don't need to mock it. A captured
``BLOOMS_VERBS["remember"]`` lookup confirms the verbs (``define``,
``identify``, ``list``, …) the tests expect to land.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Repo root on path for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.ontology.learning_objectives import (  # noqa: E402
    BLOOMS_VERBS,
    compose_abcd_prose,
)
from lib.validators.abcd_objective import (  # noqa: E402
    AbcdObjectiveValidator,
)


# --------------------------------------------------------------------- #
# Stub DecisionCapture — records calls so we can assert log_decision
# was invoked with the expected decision_type.
# --------------------------------------------------------------------- #


class _StubDecisionCapture:
    """Minimal DecisionCapture stand-in.

    Records every ``log_decision(...)`` invocation in
    ``self.events`` so tests can assert which decision types fired
    (and how many of each).
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
        context: Any = None,
        alternatives_considered: Any = None,
        **kwargs: Any,
    ) -> None:
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                "context": context,
                "alternatives_considered": alternatives_considered,
                **kwargs,
            }
        )


# --------------------------------------------------------------------- #
# Pick canonical verbs from the real BLOOMS_VERBS table so the tests
# stay aligned with whatever schemas/taxonomies/bloom_verbs.json says.
# --------------------------------------------------------------------- #


def _pick_verb(level: str) -> str:
    """Pick a deterministic verb for ``level`` from the real verb set.

    Sorted to be reproducible across runs (frozenset iteration order is
    not guaranteed). Used by happy-path tests.
    """
    verbs = sorted(BLOOMS_VERBS[level])
    assert verbs, f"BLOOMS_VERBS[{level!r}] is unexpectedly empty"
    return verbs[0]


def _make_abcd(verb: str) -> Dict[str, Any]:
    """Return a well-formed ABCD dict with the given verb."""
    return {
        "audience": "Students",
        "behavior": {"verb": verb, "action_object": "the parts of a cell"},
        "condition": "from a labeled diagram",
        "degree": "with 90% accuracy",
    }


def _make_lo(
    lo_id: str = "TO-01",
    *,
    bloom_level: str = "remember",
    verb: str = None,  # type: ignore[assignment]
    abcd: Any = "auto",
    statement: str = "Students will identify the parts of a cell.",
    use_camel_case_bloom: bool = False,
    requires_abcd: bool = False,
) -> Dict[str, Any]:
    """Build an LO dict for the validator's input contract."""
    lo: Dict[str, Any] = {
        "id": lo_id,
        "statement": statement,
    }
    if use_camel_case_bloom:
        lo["bloomLevel"] = bloom_level
    else:
        lo["bloom_level"] = bloom_level
    if abcd == "auto":
        lo["abcd"] = _make_abcd(verb or _pick_verb(bloom_level))
    elif abcd is not None:
        lo["abcd"] = abcd
    if requires_abcd:
        lo["requires_abcd"] = True
    return lo


# --------------------------------------------------------------------- #
# Cases.
# --------------------------------------------------------------------- #


def test_passes_when_verb_matches_bloom_level():
    """Happy path: verb in BLOOMS_VERBS[level] → pass."""
    validator = AbcdObjectiveValidator()
    lo = _make_lo(verb=_pick_verb("remember"))
    result = validator.validate({"objectives": [lo]})

    assert result.passed is True
    assert result.action is None
    # No issues emitted on the happy path.
    assert result.issues == []
    assert result.score == 1.0


def test_returns_regenerate_on_verb_mismatch():
    """Verb mismatch → ABCD_VERB_BLOOM_MISMATCH + action='regenerate'."""
    validator = AbcdObjectiveValidator()
    # Pick a verb that's NOT in BLOOMS_VERBS["remember"]. "create" verbs
    # like "design" / "compose" are not in the remember bucket.
    create_verbs = sorted(BLOOMS_VERBS["create"])
    remember_verbs = BLOOMS_VERBS["remember"]
    mismatch_verb = next(
        v for v in create_verbs if v not in remember_verbs
    )

    lo = _make_lo(bloom_level="remember", verb=mismatch_verb)
    result = validator.validate({"objectives": [lo]})

    assert result.action == "regenerate"
    # passed=True per warning-severity contract; the gate's severity
    # is governed by the issues, not by action="regenerate".
    assert result.passed is True
    codes = [i.code for i in result.issues]
    assert "ABCD_VERB_BLOOM_MISMATCH" in codes
    # The LO ID + verb + valid-set preview should appear in the message.
    msg = next(i.message for i in result.issues if i.code == "ABCD_VERB_BLOOM_MISMATCH")
    assert "TO-01" in msg
    assert mismatch_verb in msg
    assert "remember" in msg


def test_returns_warning_when_abcd_field_absent():
    """require_abcd=True + missing abcd → ABCD_MISSING warning."""
    validator = AbcdObjectiveValidator()
    lo = _make_lo(abcd=None)
    result = validator.validate(
        {"objectives": [lo], "require_abcd": True}
    )

    assert result.passed is True  # warning, not critical
    assert result.action is None
    codes = [i.code for i in result.issues]
    assert "ABCD_MISSING" in codes
    assert all(i.severity == "warning" for i in result.issues)


def test_handles_capitalization_normalization():
    """Uppercase verb 'IDENTIFY' → lowercased + matches verb set."""
    validator = AbcdObjectiveValidator()
    verb = _pick_verb("remember")
    lo = _make_lo(verb=verb.upper())
    result = validator.validate({"objectives": [lo]})

    assert result.passed is True
    assert result.action is None
    assert result.issues == []


def test_compose_abcd_prose_round_trip():
    """compose_abcd_prose composes prose; validator still passes."""
    validator = AbcdObjectiveValidator()
    abcd = {
        "audience": "Students",
        "behavior": {
            "verb": _pick_verb("remember"),
            "action_object": "cell parts",
        },
        "condition": "from a labeled diagram",
        "degree": "with 90% accuracy",
    }
    # Confirm prose composer still emits a stable sentence.
    prose = compose_abcd_prose(abcd)
    assert prose.endswith(".")
    assert abcd["behavior"]["verb"] in prose

    # Validator is happy with the same ABCD dict.
    lo = {
        "id": "TO-02",
        "statement": prose,
        "bloom_level": "remember",
        "abcd": abcd,
    }
    result = validator.validate({"objectives": [lo]})
    assert result.passed is True
    assert result.action is None


def test_legacy_lo_without_abcd_skipped():
    """Default (require_abcd=False) → legacy LO with no abcd → silent pass."""
    validator = AbcdObjectiveValidator()
    lo = _make_lo(abcd=None)  # no abcd field
    result = validator.validate({"objectives": [lo]})

    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0


def test_empty_objectives_no_op_pass():
    """Empty LO list → no-op pass."""
    validator = AbcdObjectiveValidator()
    result = validator.validate({"objectives": []})

    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0


def test_malformed_abcd_emits_block_action():
    """abcd is not a mapping → ABCD_MALFORMED critical + action='block'."""
    validator = AbcdObjectiveValidator()
    lo = _make_lo(abcd="not-a-dict")
    result = validator.validate({"objectives": [lo]})

    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues]
    assert "ABCD_MALFORMED" in codes
    assert any(i.severity == "critical" for i in result.issues)


def test_malformed_abcd_missing_behavior():
    """abcd present but behavior is missing → ABCD_MALFORMED."""
    validator = AbcdObjectiveValidator()
    lo = _make_lo(
        abcd={
            "audience": "Students",
            "condition": "from a diagram",
            "degree": "with 90% accuracy",
            # missing behavior
        }
    )
    result = validator.validate({"objectives": [lo]})

    assert result.action == "block"
    assert result.passed is False
    assert any(i.code == "ABCD_MALFORMED" for i in result.issues)


def test_malformed_abcd_missing_verb():
    """behavior present but verb empty → ABCD_MALFORMED."""
    validator = AbcdObjectiveValidator()
    lo = _make_lo(
        abcd={
            "audience": "Students",
            "behavior": {"verb": "", "action_object": "cells"},
            "condition": "",
            "degree": "",
        }
    )
    result = validator.validate({"objectives": [lo]})

    assert result.action == "block"
    assert any(i.code == "ABCD_MALFORMED" for i in result.issues)


def test_decision_capture_emitted_on_mismatch():
    """Mismatch path emits decision_type='abcd_verb_bloom_mismatch'."""
    validator = AbcdObjectiveValidator()
    create_verbs = sorted(BLOOMS_VERBS["create"])
    remember_verbs = BLOOMS_VERBS["remember"]
    mismatch_verb = next(
        v for v in create_verbs if v not in remember_verbs
    )
    lo = _make_lo(bloom_level="remember", verb=mismatch_verb)
    capture = _StubDecisionCapture()

    result = validator.validate(
        {"objectives": [lo], "decision_capture": capture}
    )

    assert result.action == "regenerate"
    types = [e["decision_type"] for e in capture.events]
    assert "abcd_verb_bloom_mismatch" in types
    # Rationale must be ≥ 20 chars per project decision-capture contract.
    mismatch_event = next(
        e for e in capture.events
        if e["decision_type"] == "abcd_verb_bloom_mismatch"
    )
    assert len(mismatch_event["rationale"]) >= 20


def test_mismatch_rationales_distinct_per_objective():
    """Regression — DUPLICATE_BOILERPLATE_RATIONALE (live-run finding).

    The mismatch rationale used to open with a ~107-char static preamble,
    so every ``abcd_verb_bloom_mismatch`` event shared the same
    first-80-char prefix and tripped the DecisionCapture boilerplate
    scanner (``lib/decision_capture.py::_scan_boilerplate_rationales``,
    which keys on ``_BOILERPLATE_PREFIX_LEN=80``). Two DIFFERENT
    mismatching objectives must now yield two DIFFERENT rationales —
    distinct even in their first 80 chars — each ≥ 20 chars and each
    interpolating that objective's id, verb, and declared Bloom level.
    """
    validator = AbcdObjectiveValidator()
    create_verbs = sorted(BLOOMS_VERBS["create"])
    remember_verbs = BLOOMS_VERBS["remember"]
    # Two DISTINCT mismatching verbs (both create-level, neither in remember).
    mismatch_pool = [v for v in create_verbs if v not in remember_verbs]
    assert len(mismatch_pool) >= 2, "need two distinct mismatch verbs"
    verb_a, verb_b = mismatch_pool[0], mismatch_pool[1]

    lo_a = _make_lo(lo_id="CO-01", bloom_level="remember", verb=verb_a)
    lo_b = _make_lo(lo_id="CO-02", bloom_level="remember", verb=verb_b)
    capture = _StubDecisionCapture()

    result = validator.validate(
        {"objectives": [lo_a, lo_b], "decision_capture": capture}
    )

    assert result.action == "regenerate"
    rationales = [
        e["rationale"]
        for e in capture.events
        if e["decision_type"] == "abcd_verb_bloom_mismatch"
    ]
    assert len(rationales) == 2
    r_a, r_b = rationales
    # Distinct as whole strings AND within the scanner's 80-char prefix
    # window (the property the boilerplate detector actually checks).
    assert r_a != r_b
    assert r_a[:80] != r_b[:80]
    for rationale, lo_id, verb in ((r_a, "CO-01", verb_a), (r_b, "CO-02", verb_b)):
        assert len(rationale) >= 20
        assert lo_id in rationale
        assert verb in rationale
        assert "remember" in rationale  # the declared level it mismatched


def test_mismatch_same_verb_rationales_still_distinct_by_id():
    """Two LOs with the SAME mismatching verb still get distinct rationales.

    The per-objective id leads the rationale, so even a shared verb +
    level cannot collapse two objectives to one boilerplate prefix.
    """
    validator = AbcdObjectiveValidator()
    create_verbs = sorted(BLOOMS_VERBS["create"])
    remember_verbs = BLOOMS_VERBS["remember"]
    verb = next(v for v in create_verbs if v not in remember_verbs)

    lo_a = _make_lo(lo_id="CO-11", bloom_level="remember", verb=verb)
    lo_b = _make_lo(lo_id="CO-12", bloom_level="remember", verb=verb)
    capture = _StubDecisionCapture()
    validator.validate({"objectives": [lo_a, lo_b], "decision_capture": capture})

    rationales = [
        e["rationale"]
        for e in capture.events
        if e["decision_type"] == "abcd_verb_bloom_mismatch"
    ]
    assert len(rationales) == 2
    assert rationales[0][:80] != rationales[1][:80]


def test_pass_rationales_distinct_per_objective():
    """abcd_authored rationales are distinct even for a shared verb+level.

    Same boilerplate-scanner regression guard for the positive path: the
    pass rationale used to omit the objective id entirely, so a corpus of
    LOs sharing one verb collapsed to a single 80-char prefix.
    """
    validator = AbcdObjectiveValidator()
    verb = _pick_verb("remember")
    lo_a = _make_lo(lo_id="CO-21", verb=verb)
    lo_b = _make_lo(lo_id="CO-22", verb=verb)
    capture = _StubDecisionCapture()
    validator.validate({"objectives": [lo_a, lo_b], "decision_capture": capture})

    rationales = [
        e["rationale"]
        for e in capture.events
        if e["decision_type"] == "abcd_authored"
    ]
    assert len(rationales) == 2
    assert rationales[0] != rationales[1]
    assert rationales[0][:80] != rationales[1][:80]
    for rationale, lo_id in ((rationales[0], "CO-21"), (rationales[1], "CO-22")):
        assert len(rationale) >= 20
        assert lo_id in rationale
        assert verb in rationale


def test_decision_capture_emitted_on_pass():
    """Happy path emits decision_type='abcd_authored'."""
    validator = AbcdObjectiveValidator()
    lo = _make_lo(verb=_pick_verb("remember"))
    capture = _StubDecisionCapture()

    result = validator.validate(
        {"objectives": [lo], "decision_capture": capture}
    )

    assert result.passed is True
    types = [e["decision_type"] for e in capture.events]
    assert "abcd_authored" in types
    pass_event = next(
        e for e in capture.events
        if e["decision_type"] == "abcd_authored"
    )
    assert len(pass_event["rationale"]) >= 20


def test_synthesized_objectives_path_loaded(tmp_path: Path):
    """Path-based input loads + flattens terminal + chapter objectives."""
    validator = AbcdObjectiveValidator()
    payload = {
        "course_name": "TST_905",
        "duration_weeks": 8,
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Students will identify cell parts.",
                "bloom_level": "remember",
                "abcd": _make_abcd(_pick_verb("remember")),
            }
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {
                        "id": "CO-01",
                        "statement": "Students will list the organelles.",
                        "bloom_level": "remember",
                        "abcd": _make_abcd(_pick_verb("remember")),
                    }
                ],
            }
        ],
    }
    objectives_path = tmp_path / "synthesized_objectives.json"
    objectives_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validator.validate(
        {"synthesized_objectives_path": str(objectives_path)}
    )
    assert result.passed is True
    assert result.action is None
    # Two LOs audited, both pass.
    assert result.score == 1.0


def test_synthesized_objectives_path_missing(tmp_path: Path):
    """Non-existent path → critical ABCD_OBJECTIVES_PATH_MISSING + action=block."""
    validator = AbcdObjectiveValidator()
    result = validator.validate(
        {"synthesized_objectives_path": str(tmp_path / "does_not_exist.json")}
    )
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues]
    assert "ABCD_OBJECTIVES_PATH_MISSING" in codes


def test_synthesized_objectives_path_unparseable(tmp_path: Path):
    """Malformed JSON → critical ABCD_OBJECTIVES_PATH_UNREADABLE + action=block."""
    validator = AbcdObjectiveValidator()
    bad = tmp_path / "synthesized_objectives.json"
    bad.write_text("{not valid json", encoding="utf-8")
    result = validator.validate(
        {"synthesized_objectives_path": str(bad)}
    )
    assert result.passed is False
    assert result.action == "block"
    codes = [i.code for i in result.issues]
    assert "ABCD_OBJECTIVES_PATH_UNREADABLE" in codes


def test_camelcase_bloomlevel_accepted():
    """LO with bloomLevel (camelCase) is normalized to lowercase."""
    validator = AbcdObjectiveValidator()
    lo = _make_lo(verb=_pick_verb("remember"), use_camel_case_bloom=True)
    result = validator.validate({"objectives": [lo]})

    assert result.passed is True
    assert result.action is None


def test_no_bloom_level_emits_warning():
    """LO has abcd but no bloom_level → ABCD_NO_BLOOM_LEVEL warning."""
    validator = AbcdObjectiveValidator()
    lo = {
        "id": "TO-03",
        "statement": "Students will do something.",
        "abcd": _make_abcd(_pick_verb("remember")),
        # no bloom_level
    }
    result = validator.validate({"objectives": [lo]})

    assert result.passed is True
    assert result.action is None
    codes = [i.code for i in result.issues]
    assert "ABCD_NO_BLOOM_LEVEL" in codes
    assert all(i.severity == "warning" for i in result.issues)


def test_per_lo_requires_abcd_flag():
    """A single LO can opt-in via requires_abcd=True even when global is False."""
    validator = AbcdObjectiveValidator()
    lo = _make_lo(abcd=None, requires_abcd=True)
    result = validator.validate({"objectives": [lo]})

    codes = [i.code for i in result.issues]
    assert "ABCD_MISSING" in codes


def test_mixed_legacy_and_abcd_los():
    """A corpus with both legacy (no-abcd) and ABCD LOs → mixed pass/fail."""
    validator = AbcdObjectiveValidator()
    legacy = _make_lo(lo_id="TO-01", abcd=None)
    good = _make_lo(
        lo_id="TO-02",
        bloom_level="remember",
        verb=_pick_verb("remember"),
    )
    create_verbs = sorted(BLOOMS_VERBS["create"])
    remember_verbs = BLOOMS_VERBS["remember"]
    mismatch_verb = next(
        v for v in create_verbs if v not in remember_verbs
    )
    bad = _make_lo(
        lo_id="TO-03",
        bloom_level="remember",
        verb=mismatch_verb,
    )

    result = validator.validate(
        {"objectives": [legacy, good, bad]}
    )
    # Legacy + good pass; bad triggers regenerate.
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "ABCD_VERB_BLOOM_MISMATCH" in codes
    # Only one mismatch issue (TO-03), legacy was skipped.
    assert sum(1 for c in codes if c == "ABCD_VERB_BLOOM_MISMATCH") == 1


def test_validator_has_expected_protocol():
    """Smoke: the class instance has the documented validate() method."""
    v = AbcdObjectiveValidator()
    assert hasattr(v, "validate")
    assert v.name == "abcd_objective"
    assert v.version == "0.1.0"


def test_decision_capture_failure_does_not_break_gate():
    """A capture that raises in log_decision must not break the gate."""

    class _RaisingCapture:
        def log_decision(self, **kwargs: Any) -> None:
            raise RuntimeError("simulated capture failure")

    validator = AbcdObjectiveValidator()
    lo = _make_lo(verb=_pick_verb("remember"))
    result = validator.validate(
        {"objectives": [lo], "decision_capture": _RaisingCapture()}
    )
    # Gate still passes despite the capture error.
    assert result.passed is True
    assert result.action is None


# --------------------------------------------------------------------- #
# IB6.5 — B01 both-axes (cognitive-process × knowledge-type) assertion.
# Gated behind ED4ALL_BLOCK_QUALITY_RUBRIC; default-off byte-stable.
# --------------------------------------------------------------------- #


def test_ib6_knowledge_type_missing_flagged_when_flag_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_RUBRIC", "1")
    validator = AbcdObjectiveValidator()
    lo = _make_lo(verb=_pick_verb("remember"))  # no cognitive_domain
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_KNOWLEDGE_TYPE_MISSING" in codes
    # warning-day-1: still passes.
    assert result.passed is True


def test_ib6_both_axes_present_passes(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_RUBRIC", "1")
    validator = AbcdObjectiveValidator()
    lo = _make_lo(verb=_pick_verb("remember"))
    lo["cognitive_domain"] = "factual"
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_KNOWLEDGE_TYPE_MISSING" not in codes
    assert result.passed is True


def test_ib6_knowledge_type_check_off_by_default(monkeypatch):
    monkeypatch.delenv("ED4ALL_BLOCK_QUALITY_RUBRIC", raising=False)
    validator = AbcdObjectiveValidator()
    lo = _make_lo(verb=_pick_verb("remember"))  # no cognitive_domain
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    # default-off: no knowledge-type flag, byte-stable with the legacy path.
    assert "OBJECTIVE_KNOWLEDGE_TYPE_MISSING" not in codes
    assert result.issues == []


def test_ib6_camelcase_cognitive_domain_accepted(monkeypatch):
    monkeypatch.setenv("ED4ALL_BLOCK_QUALITY_RUBRIC", "1")
    validator = AbcdObjectiveValidator()
    lo = _make_lo(verb=_pick_verb("understand"), bloom_level="understand")
    lo["cognitiveDomain"] = "conceptual"
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_KNOWLEDGE_TYPE_MISSING" not in codes


# --------------------------------------------------------------------- #
# W1.1 — ED4ALL_OBJECTIVE_OBSERVABLE_VERB: non-observable ("fuzzy") main
# verb scan on the legacy no-ABCD path. Default-off byte-stable.
# --------------------------------------------------------------------- #


def _legacy_lo(statement: str, lo_id: str = "TO-01") -> Dict[str, Any]:
    """A legacy (no-ABCD) LO carrying just id + statement."""
    return {"id": lo_id, "statement": statement}


def test_observable_verb_off_by_default_byte_stable(monkeypatch):
    monkeypatch.delenv("ED4ALL_OBJECTIVE_OBSERVABLE_VERB", raising=False)
    validator = AbcdObjectiveValidator()
    lo = _legacy_lo("Students will understand the water cycle.")
    result = validator.validate({"objectives": [lo]})
    # Default-off: legacy LO silently passes, no vague-verb issue.
    assert result.passed is True
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0


def test_observable_verb_flags_vague_main_verb(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_OBSERVABLE_VERB", "1")
    validator = AbcdObjectiveValidator()
    lo = _legacy_lo("Students will understand the water cycle.")
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_VAGUE_VERB" in codes
    # Warning-only: the legacy LO still passes.
    assert result.passed is True
    assert result.action is None
    issue = next(i for i in result.issues if i.code == "OBJECTIVE_VAGUE_VERB")
    assert issue.severity == "warning"
    assert "understand" in issue.message
    # Suggestion names a same-level (understand) observable replacement.
    assert issue.suggestion


def test_observable_verb_flags_be_aware_phrase(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_OBSERVABLE_VERB", "on")
    validator = AbcdObjectiveValidator()
    lo = _legacy_lo("Learners will be aware of the safety protocols.")
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_VAGUE_VERB" in codes


def test_observable_verb_no_flag_on_observable_verb(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_OBSERVABLE_VERB", "1")
    validator = AbcdObjectiveValidator()
    lo = _legacy_lo("Students will identify the parts of a cell.")
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    assert "OBJECTIVE_VAGUE_VERB" not in codes
    assert result.passed is True


def test_observable_verb_decision_capture_fires(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_OBSERVABLE_VERB", "1")
    validator = AbcdObjectiveValidator()
    lo = _legacy_lo("Students will appreciate the beauty of algebra.")
    capture = _StubDecisionCapture()
    validator.validate({"objectives": [lo], "decision_capture": capture})
    types = [e["decision_type"] for e in capture.events]
    assert "objective_vague_verb" in types
    event = next(
        e for e in capture.events if e["decision_type"] == "objective_vague_verb"
    )
    assert len(event["rationale"]) >= 20


def test_observable_verb_not_scanned_on_require_abcd_missing(monkeypatch):
    # require_abcd routes a no-ABCD LO to the ABCD_MISSING branch, NOT the
    # legacy silent-pass branch — the vague-verb scan must not fire there.
    monkeypatch.setenv("ED4ALL_OBJECTIVE_OBSERVABLE_VERB", "1")
    validator = AbcdObjectiveValidator()
    lo = _legacy_lo("Students will understand the water cycle.")
    result = validator.validate({"objectives": [lo], "require_abcd": True})
    codes = [i.code for i in result.issues]
    assert "ABCD_MISSING" in codes
    assert "OBJECTIVE_VAGUE_VERB" not in codes


# --------------------------------------------------------------------- #
# W1.3 — ABCD field-presence completeness (ABCD_INCOMPLETE), gated by
# the existing require_abcd input. Byte-stable for legacy runs.
# --------------------------------------------------------------------- #


def _incomplete_abcd(verb: str) -> Dict[str, Any]:
    """A well-formed-but-incomplete ABCD (verb OK, C+D empty)."""
    return {
        "audience": "Students",
        "behavior": {"verb": verb, "action_object": "the parts of a cell"},
        "condition": "",
        "degree": "",
    }


def test_abcd_incomplete_flagged_when_require_abcd(monkeypatch):
    validator = AbcdObjectiveValidator()
    lo = _make_lo(abcd=_incomplete_abcd(_pick_verb("remember")))
    result = validator.validate({"objectives": [lo], "require_abcd": True})
    codes = [i.code for i in result.issues]
    assert "ABCD_INCOMPLETE" in codes
    issue = next(i for i in result.issues if i.code == "ABCD_INCOMPLETE")
    assert issue.severity == "warning"
    assert "condition" in issue.message and "degree" in issue.message
    # Warning-only + verb is aligned → gate still passes.
    assert result.passed is True


def test_abcd_incomplete_not_flagged_by_default(monkeypatch):
    # require_abcd defaults False → legacy byte-identical (no ABCD_INCOMPLETE).
    validator = AbcdObjectiveValidator()
    lo = _make_lo(abcd=_incomplete_abcd(_pick_verb("remember")))
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    assert "ABCD_INCOMPLETE" not in codes
    assert result.passed is True


def test_abcd_complete_no_incomplete_flag(monkeypatch):
    validator = AbcdObjectiveValidator()
    lo = _make_lo(verb=_pick_verb("remember"))  # _make_abcd is complete
    result = validator.validate({"objectives": [lo], "require_abcd": True})
    codes = [i.code for i in result.issues]
    assert "ABCD_INCOMPLETE" not in codes


def test_abcd_incomplete_missing_action_object(monkeypatch):
    validator = AbcdObjectiveValidator()
    abcd = {
        "audience": "Students",
        "behavior": {"verb": _pick_verb("remember"), "action_object": ""},
        "condition": "from a diagram",
        "degree": "with 90% accuracy",
    }
    lo = _make_lo(abcd=abcd)
    result = validator.validate({"objectives": [lo], "require_abcd": True})
    codes = [i.code for i in result.issues]
    assert "ABCD_INCOMPLETE" in codes
    issue = next(i for i in result.issues if i.code == "ABCD_INCOMPLETE")
    assert "behavior.action_object" in issue.message


def test_abcd_incomplete_decision_capture_fires(monkeypatch):
    validator = AbcdObjectiveValidator()
    lo = _make_lo(abcd=_incomplete_abcd(_pick_verb("remember")))
    capture = _StubDecisionCapture()
    validator.validate(
        {"objectives": [lo], "require_abcd": True, "decision_capture": capture}
    )
    types = [e["decision_type"] for e in capture.events]
    assert "abcd_incomplete" in types
    event = next(e for e in capture.events if e["decision_type"] == "abcd_incomplete")
    assert len(event["rationale"]) >= 20


# --------------------------------------------------------------------- #
# W1.4 — ED4ALL_OBJECTIVE_INFER_BLOOM: infer a null bloom_level from the
# declared ABCD verb rather than skipping the audit. Default-off byte-stable.
# --------------------------------------------------------------------- #


def _abcd_lo_no_bloom(verb: str, lo_id: str = "TO-05") -> Dict[str, Any]:
    return {
        "id": lo_id,
        "statement": "Students will do something.",
        "abcd": _make_abcd(verb),
        # deliberately no bloom_level
    }


def test_infer_bloom_off_by_default_emits_no_bloom_level(monkeypatch):
    monkeypatch.delenv("ED4ALL_OBJECTIVE_INFER_BLOOM", raising=False)
    validator = AbcdObjectiveValidator()
    lo = _abcd_lo_no_bloom(_pick_verb("apply"))
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    # Default-off byte-identical: legacy warning-skip.
    assert "ABCD_NO_BLOOM_LEVEL" in codes
    assert "ABCD_BLOOM_LEVEL_INFERRED" not in codes
    assert result.passed is True


def test_infer_bloom_infers_level_from_verb(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_INFER_BLOOM", "1")
    validator = AbcdObjectiveValidator()
    lo = _abcd_lo_no_bloom(_pick_verb("apply"))
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    assert "ABCD_BLOOM_LEVEL_INFERRED" in codes
    assert "ABCD_NO_BLOOM_LEVEL" not in codes
    # Inferred level aligns with the verb by construction → passes.
    assert result.passed is True
    assert result.action is None
    inferred = next(
        i for i in result.issues if i.code == "ABCD_BLOOM_LEVEL_INFERRED"
    )
    assert inferred.severity == "info"
    assert "apply" in inferred.message


def test_infer_bloom_decision_capture_fires(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_INFER_BLOOM", "1")
    validator = AbcdObjectiveValidator()
    lo = _abcd_lo_no_bloom(_pick_verb("apply"))
    capture = _StubDecisionCapture()
    validator.validate({"objectives": [lo], "decision_capture": capture})
    types = [e["decision_type"] for e in capture.events]
    assert "abcd_bloom_inferred" in types
    event = next(
        e for e in capture.events if e["decision_type"] == "abcd_bloom_inferred"
    )
    assert len(event["rationale"]) >= 20


def test_infer_bloom_falls_back_when_verb_not_canonical(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_INFER_BLOOM", "1")
    validator = AbcdObjectiveValidator()
    # A non-canonical verb can't be mapped to a Bloom level.
    lo = _abcd_lo_no_bloom("flibber")
    result = validator.validate({"objectives": [lo]})
    codes = [i.code for i in result.issues]
    assert "ABCD_NO_BLOOM_LEVEL" in codes
    assert "ABCD_BLOOM_LEVEL_INFERRED" not in codes
    assert result.passed is True
