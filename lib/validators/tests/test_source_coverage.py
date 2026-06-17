"""WS6a — tests for SourceCoverageValidator.

Verifies the source→objective coverage-audit gate: section→nearest-objective
max-cosine scoring, the per-section ``SOURCE_SECTION_UNCOVERED`` + aggregate
``SOURCE_COVERAGE_LOW`` warnings, graceful EMBEDDING_DEPS_MISSING degrade,
empty/missing-objectives-or-structure no-op pass, decision-capture firing,
the config→inputs threshold merge, and the router-registration binding.

A fake embedder (`_StubEmbedder`) returns deterministic per-key vectors so
the suite pins specific cosine outcomes WITHOUT the sentence-transformers
extras installed (the deps-missing path is exercised separately by passing
``embedder=None`` against a validator with no override + monkeypatching
``try_load_embedder`` to return None).

Mirrors ``lib/validators/tests/test_co_terminal_alignment.py``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from lib.validators.source_coverage import (
    DEFAULT_COVERAGE_FLOOR,
    SourceCoverageValidator,
)


# --------------------------------------------------------------------- #
# Stub embedder — deterministic unit vectors on a 2D circle. Each key maps
# to an angle (radians); ``encode`` returns (cos θ, sin θ). cosine between
# two keys is then cos(θ_a − θ_b), so the tests pin exact alignment by
# choosing angles. Same angle → cosine 1.0; orthogonal (π/2 apart) → 0.0.
# --------------------------------------------------------------------- #


class _StubEmbedder:
    def __init__(self, angle_map: Dict[str, float]) -> None:
        self.angle_map = angle_map
        self.calls: List[str] = []

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        self.calls.append(text)
        # Longest prefix-match so we can key on a leading tag.
        match_len = -1
        angle = 0.0
        for key, ang in self.angle_map.items():
            if text.startswith(key) and len(key) > match_len:
                match_len = len(key)
                angle = ang
        return [math.cos(angle), math.sin(angle)]


_BODY = " body text padding to clear the 40-char content floor."


def _to(tid: str, statement: str) -> Dict[str, Any]:
    return {"id": tid, "statement": statement}


def _co(cid: str, statement: str) -> Dict[str, Any]:
    return {"id": cid, "statement": statement}


def _section(heading: str, text: str) -> Dict[str, Any]:
    return {"headingText": heading, "section_text": text}


def _objectives(
    terminals: List[Dict[str, Any]],
    cos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "terminal_objectives": terminals,
        "chapter_objectives": cos,  # flat list shape (flatten tolerates)
    }


def _structure(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"chapters": [{"id": "ch1", "sections": sections}]}


def _codes(result) -> List[str]:
    return [i.code for i in result.issues]


class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, *, decision_type, decision, rationale, **kw):
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
            }
        )


# --------------------------------------------------------------------- #
# 1. Fully-covered set passes
# --------------------------------------------------------------------- #


def test_fully_covered_set_passes():
    # Every section shares an objective's angle → cosine 1.0 (>= 0.45).
    embedder = _StubEmbedder({"OBJ": 0.0, "SEC": 0.0})
    objectives = _objectives(
        [_to("TO-01", "OBJ terminal statement")],
        [
            _co("CO-01", "OBJ chapter one"),
            _co("CO-02", "OBJ chapter two"),
        ],
    )
    structure = _structure(
        [
            _section("SEC alpha", "SEC alpha" + _BODY),
            _section("SEC beta", "SEC beta" + _BODY),
            _section("SEC gamma", "SEC gamma" + _BODY),
        ]
    )
    v = SourceCoverageValidator(embedder=embedder)
    result = v.validate(
        {
            "synthesized_objectives": objectives,
            "textbook_structure": structure,
        }
    )
    assert result.passed is True
    assert result.issues == []
    assert result.score == 1.0


# --------------------------------------------------------------------- #
# 2. A section far from every objective → uncovered (+ rate when high)
# --------------------------------------------------------------------- #


def test_uncovered_section_warns_and_rate_low_when_high():
    # All 3 sections orthogonal (π/2) to every objective → cosine 0.0.
    embedder = _StubEmbedder({"OBJ": 0.0, "SEC": math.pi / 2})
    objectives = _objectives(
        [_to("TO-01", "OBJ terminal statement")],
        [_co("CO-01", "OBJ chapter one")],
    )
    structure = _structure(
        [
            _section("SEC one", "SEC one" + _BODY),
            _section("SEC two", "SEC two" + _BODY),
            _section("SEC three", "SEC three" + _BODY),
        ]
    )
    v = SourceCoverageValidator(embedder=embedder)
    result = v.validate(
        {
            "synthesized_objectives": objectives,
            "textbook_structure": structure,
        }
    )
    codes = _codes(result)
    assert codes.count("SOURCE_SECTION_UNCOVERED") == 3
    # 3/3 uncovered (100%) > 0.15 → aggregate warning.
    assert "SOURCE_COVERAGE_LOW" in codes
    # Warning-only → no hard-block.
    assert result.passed is True
    assert all(i.severity == "warning" for i in result.issues)
    assert result.score == 0.0


# --------------------------------------------------------------------- #
# 3. Partial-uncovered below the rate floor → section issues, no rate warn
# --------------------------------------------------------------------- #


def test_partial_uncovered_below_rate_floor_no_coverage_low():
    # 10 sections, 1 uncovered (rate 0.1 < 0.15). 9 aligned, 1 orthogonal.
    embedder = _StubEmbedder(
        {"OBJ": 0.0, "GOOD": 0.0, "BAD": math.pi / 2}
    )
    objectives = _objectives(
        [_to("TO-01", "OBJ terminal statement")], []
    )
    sections = [
        _section(f"GOOD sec {i}", f"GOOD sec {i}" + _BODY)
        for i in range(9)
    ]
    sections.append(_section("BAD sec", "BAD sec" + _BODY))
    structure = _structure(sections)
    v = SourceCoverageValidator(embedder=embedder)
    result = v.validate(
        {
            "synthesized_objectives": objectives,
            "textbook_structure": structure,
        }
    )
    codes = _codes(result)
    assert codes.count("SOURCE_SECTION_UNCOVERED") == 1
    assert "SOURCE_COVERAGE_LOW" not in codes
    assert result.score == 0.9
    assert result.passed is True


# --------------------------------------------------------------------- #
# 4. Embedding deps missing → graceful degrade
# --------------------------------------------------------------------- #


def test_embedding_deps_missing_graceful_degrade(monkeypatch):
    import lib.validators.source_coverage as mod

    monkeypatch.setattr(mod, "try_load_embedder", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_strict_mode", lambda: False)
    objectives = _objectives(
        [_to("TO-01", "OBJ terminal statement")], []
    )
    structure = _structure(
        [_section("SEC any", "SEC any" + _BODY)]
    )
    v = SourceCoverageValidator()  # no embedder override
    result = v.validate(
        {
            "synthesized_objectives": objectives,
            "textbook_structure": structure,
        }
    )
    assert result.passed is True
    assert _codes(result) == ["EMBEDDING_DEPS_MISSING"]
    assert result.score == 1.0


# --------------------------------------------------------------------- #
# 5. Empty / missing objectives or structure → no-op pass
# --------------------------------------------------------------------- #


def test_empty_or_missing_inputs_noop_pass():
    embedder = _StubEmbedder({})
    v = SourceCoverageValidator(embedder=embedder)
    structure = _structure([_section("SEC a", "SEC a" + _BODY)])
    objectives = _objectives([_to("TO-01", "OBJ x")], [])

    # No objectives at all.
    r1 = v.validate({"textbook_structure": structure})
    assert r1.passed is True and r1.issues == [] and r1.score == 1.0

    # No structure at all.
    r2 = v.validate({"synthesized_objectives": objectives})
    assert r2.passed is True and r2.issues == [] and r2.score == 1.0

    # Objectives present but zero statements.
    r3 = v.validate(
        {
            "synthesized_objectives": {
                "terminal_objectives": [],
                "chapter_objectives": [],
            },
            "textbook_structure": structure,
        }
    )
    assert r3.passed is True and r3.issues == [] and r3.score == 1.0

    # Structure present but no content-bearing section (text too short).
    r4 = v.validate(
        {
            "synthesized_objectives": objectives,
            "textbook_structure": _structure(
                [_section("SEC stub", "tiny")]
            ),
        }
    )
    assert r4.passed is True and r4.issues == [] and r4.score == 1.0

    # Wholly absent inputs.
    r5 = v.validate({})
    assert r5.passed is True and r5.issues == [] and r5.score == 1.0

    # No encode should have happened on any no-op path.
    assert embedder.calls == []


# --------------------------------------------------------------------- #
# 6. Decision capture fires with dynamic signals
# --------------------------------------------------------------------- #


def test_decision_capture_fires_with_dynamic_signals():
    embedder = _StubEmbedder({"OBJ": 0.0, "SEC": math.pi / 2})
    objectives = _objectives(
        [_to("TO-01", "OBJ terminal statement")], []
    )
    structure = _structure(
        [
            _section("SEC one", "SEC one" + _BODY),
            _section("SEC two", "SEC two" + _BODY),
        ]
    )
    cap = _RecordingCapture()
    v = SourceCoverageValidator(embedder=embedder)
    v.validate(
        {
            "synthesized_objectives": objectives,
            "textbook_structure": structure,
            "decision_capture": cap,
        }
    )
    cov_events = [
        e for e in cap.events
        if e["decision_type"] == "source_coverage_check"
    ]
    # One per audited section.
    assert len(cov_events) == 2
    for e in cov_events:
        assert len(e["rationale"]) >= 20
        assert "max_cosine_to_objective=" in e["rationale"]
        assert "coverage_floor=" in e["rationale"]
        assert "covered=" in e["rationale"]


# --------------------------------------------------------------------- #
# 7. Threshold override from inputs (config→inputs merge)
# --------------------------------------------------------------------- #


def test_coverage_floor_override_from_inputs():
    # Section at 60° from the objective → cosine cos(60°)=0.5. Default 0.45
    # would pass; an inputs coverage_floor of 0.60 flags it.
    embedder = _StubEmbedder({"OBJ": 0.0, "SEC": math.pi / 3})
    objectives = _objectives(
        [_to("TO-01", "OBJ terminal statement")], []
    )
    structure = _structure(
        [_section("SEC half", "SEC half" + _BODY)]
    )
    v = SourceCoverageValidator(embedder=embedder)
    assert DEFAULT_COVERAGE_FLOOR == 0.45
    # Under the default floor (0.45) the 0.5-cosine section passes.
    r_default = v.validate(
        {
            "synthesized_objectives": objectives,
            "textbook_structure": structure,
        }
    )
    assert "SOURCE_SECTION_UNCOVERED" not in _codes(r_default)
    # With a top-level coverage_floor of 0.60, the 0.5-cosine section flags.
    r_strict = v.validate(
        {
            "synthesized_objectives": objectives,
            "textbook_structure": structure,
            "coverage_floor": 0.60,
        }
    )
    assert "SOURCE_SECTION_UNCOVERED" in _codes(r_strict)


# --------------------------------------------------------------------- #
# 8. TO-only coverage counts (a section covered only by a TO is covered)
# --------------------------------------------------------------------- #


def test_section_covered_by_terminal_objective_only():
    # No COs; the only objective is a TO that the section aligns with.
    embedder = _StubEmbedder({"OBJ": 0.0, "SEC": 0.0})
    objectives = _objectives([_to("TO-01", "OBJ terminal only")], [])
    structure = _structure([_section("SEC a", "SEC a" + _BODY)])
    v = SourceCoverageValidator(embedder=embedder)
    result = v.validate(
        {
            "synthesized_objectives": objectives,
            "textbook_structure": structure,
        }
    )
    assert result.passed is True
    assert _codes(result) == []
    assert result.score == 1.0


# --------------------------------------------------------------------- #
# Router registration
# --------------------------------------------------------------------- #


def test_router_resolves_source_coverage_to_chapter_objective_builder():
    from MCP.hardening.gate_input_routing import (
        _build_chapter_objective_coverage_inputs,
        default_router,
    )

    r = default_router()
    dotted = "lib.validators.source_coverage.SourceCoverageValidator"
    assert dotted in r.builders, (
        f"WS6a regression: no builder registered for {dotted}; gate would "
        f"silently skip via __no_builder_registered__."
    )
    assert r.builders[dotted] is _build_chapter_objective_coverage_inputs
