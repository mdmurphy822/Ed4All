"""WS3 §8 — tests for CoTerminalAlignmentValidator.

Verifies the CO↔TO semantic-alignment detection gate: recompute-cosine
scoring, the per-CO ``CO_TERMINAL_SEMANTIC_MISMATCH`` + aggregate
``CO_TERMINAL_WEAK_LINK_RATE_HIGH`` warnings, the
``CO_NO_ASSIGNED_TERMINAL`` skip, graceful EMBEDDING_DEPS_MISSING degrade,
empty-objectives no-op pass, decision-capture firing, the config→inputs
threshold merge, and the optional WS2 stamped-score fast-path.

A fake embedder (`_StubEmbedder`) returns deterministic per-key vectors so
the suite pins specific cosine outcomes WITHOUT the sentence-transformers
extras installed (the deps-missing path is exercised separately by passing
``embedder=None`` against a validator with no override).

Plus a router-registration test: ``CoTerminalAlignmentValidator`` resolves
to ``_build_chapter_objective_coverage_inputs`` (not
``__no_builder_registered__``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from lib.validators.co_terminal_alignment import (
    DEFAULT_THRESHOLD,
    CoTerminalAlignmentValidator,
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


def _to(tid: str, statement: str) -> Dict[str, Any]:
    return {"id": tid, "statement": statement}


def _co(
    cid: str,
    statement: str,
    *,
    terminal_id: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    co: Dict[str, Any] = {"id": cid, "statement": statement}
    if terminal_id is not None:
        co["terminal_id"] = terminal_id
    co.update(extra)
    return co


def _objectives(
    terminals: List[Dict[str, Any]],
    cos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "terminal_objectives": terminals,
        "chapter_objectives": cos,  # flat list shape (flatten tolerates)
    }


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
# 1. Aligned set passes
# --------------------------------------------------------------------- #


def test_aligned_set_passes():
    # All COs share the TO's angle → cosine 1.0 (>= 0.45).
    embedder = _StubEmbedder({"TO": 0.0, "CO": 0.0})
    objectives = _objectives(
        [_to("TO-01", "TO terminal statement")],
        [
            _co("CO-01", "CO aligned one", terminal_id="TO-01"),
            _co("CO-02", "CO aligned two", terminal_id="TO-01"),
            _co("CO-03", "CO aligned three", terminal_id="TO-01"),
        ],
    )
    v = CoTerminalAlignmentValidator(embedder=embedder)
    result = v.validate({"synthesized_objectives": objectives})
    assert result.passed is True
    assert result.issues == []
    assert result.score == 1.0


# --------------------------------------------------------------------- #
# 2. Mismatched set warns (keystone)
# --------------------------------------------------------------------- #


def test_mismatched_set_warns():
    # COs are orthogonal (π/2) to the TO → cosine 0.0 (< 0.45).
    embedder = _StubEmbedder({"TO": 0.0, "CO": math.pi / 2})
    objectives = _objectives(
        [_to("TO-01", "TO terminal statement")],
        [
            _co("CO-01", "CO mismatch one", terminal_id="TO-01"),
            _co("CO-02", "CO mismatch two", terminal_id="TO-01"),
            _co("CO-03", "CO mismatch three", terminal_id="TO-01"),
        ],
    )
    v = CoTerminalAlignmentValidator(embedder=embedder)
    result = v.validate({"synthesized_objectives": objectives})
    codes = _codes(result)
    assert codes.count("CO_TERMINAL_SEMANTIC_MISMATCH") == 3
    assert "CO_TERMINAL_WEAK_LINK_RATE_HIGH" in codes
    # Warning-only → no hard-block.
    assert result.passed is True
    assert all(i.severity == "warning" for i in result.issues)
    assert result.score == 0.0


# --------------------------------------------------------------------- #
# 3. Partial weak below warn floor → no rate warning
# --------------------------------------------------------------------- #


def test_partial_weak_below_warn_floor_no_rate_warning():
    # 10 COs, 2 weak (rate 0.2 < 0.30). 8 aligned, 2 orthogonal.
    embedder = _StubEmbedder(
        {"TO": 0.0, "GOOD": 0.0, "BAD": math.pi / 2}
    )
    cos = [
        _co(f"CO-{i:02d}", f"GOOD aligned {i}", terminal_id="TO-01")
        for i in range(8)
    ]
    cos += [
        _co("CO-08", "BAD mismatch a", terminal_id="TO-01"),
        _co("CO-09", "BAD mismatch b", terminal_id="TO-01"),
    ]
    objectives = _objectives([_to("TO-01", "TO terminal statement")], cos)
    v = CoTerminalAlignmentValidator(embedder=embedder)
    result = v.validate({"synthesized_objectives": objectives})
    codes = _codes(result)
    assert codes.count("CO_TERMINAL_SEMANTIC_MISMATCH") == 2
    assert "CO_TERMINAL_WEAK_LINK_RATE_HIGH" not in codes
    assert result.score == 0.8
    assert result.passed is True


# --------------------------------------------------------------------- #
# 4. Embedding deps missing → graceful degrade
# --------------------------------------------------------------------- #


def test_embedding_deps_missing_graceful_degrade(monkeypatch):
    # No embedder override + force try_load_embedder to return None.
    import lib.validators.co_terminal_alignment as mod

    monkeypatch.setattr(mod, "try_load_embedder", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_strict_mode", lambda: False)
    objectives = _objectives(
        [_to("TO-01", "TO terminal statement")],
        [_co("CO-01", "CO any", terminal_id="TO-01")],
    )
    v = CoTerminalAlignmentValidator()  # no embedder override
    result = v.validate({"synthesized_objectives": objectives})
    assert result.passed is True
    assert _codes(result) == ["EMBEDDING_DEPS_MISSING"]
    assert result.score == 1.0


# --------------------------------------------------------------------- #
# 5. CO with no assigned terminal → skipped (warning, not counted weak)
# --------------------------------------------------------------------- #


def test_co_with_no_assigned_terminal_skipped():
    embedder = _StubEmbedder({"TO": 0.0, "CO": 0.0})
    objectives = _objectives(
        [_to("TO-01", "TO terminal statement")],
        [
            _co("CO-01", "CO aligned", terminal_id="TO-01"),
            # No terminal_id at all.
            _co("CO-02", "CO orphan"),
        ],
    )
    v = CoTerminalAlignmentValidator(embedder=embedder)
    result = v.validate({"synthesized_objectives": objectives})
    codes = _codes(result)
    assert codes.count("CO_NO_ASSIGNED_TERMINAL") == 1
    assert "CO_TERMINAL_SEMANTIC_MISMATCH" not in codes
    # Only the 1 assigned CO is audited (passed) → score 1.0.
    assert result.score == 1.0
    assert result.passed is True


# --------------------------------------------------------------------- #
# 6. Empty objectives → no-op pass
# --------------------------------------------------------------------- #


def test_empty_objectives_noop_pass():
    embedder = _StubEmbedder({})
    v = CoTerminalAlignmentValidator(embedder=embedder)
    # No terminals.
    r1 = v.validate({"synthesized_objectives": {"chapter_objectives": []}})
    assert r1.passed is True and r1.issues == [] and r1.score == 1.0
    # No chapter objectives.
    r2 = v.validate(
        {"synthesized_objectives": {"terminal_objectives": [_to("TO-01", "x")]}}
    )
    assert r2.passed is True and r2.issues == [] and r2.score == 1.0
    # Wholly absent objectives input.
    r3 = v.validate({})
    assert r3.passed is True and r3.issues == [] and r3.score == 1.0
    # No encode should have happened.
    assert embedder.calls == []


# --------------------------------------------------------------------- #
# 7. Decision capture fires with dynamic signals
# --------------------------------------------------------------------- #


def test_decision_capture_fires_with_dynamic_signals():
    embedder = _StubEmbedder({"TO": 0.0, "CO": math.pi / 2})
    objectives = _objectives(
        [_to("TO-01", "TO terminal statement")],
        [
            _co("CO-01", "CO one", terminal_id="TO-01"),
            _co("CO-02", "CO two", terminal_id="TO-01"),
        ],
    )
    cap = _RecordingCapture()
    v = CoTerminalAlignmentValidator(embedder=embedder)
    v.validate({"synthesized_objectives": objectives, "decision_capture": cap})
    align_events = [
        e for e in cap.events
        if e["decision_type"] == "co_terminal_alignment_check"
    ]
    # One per audited CO.
    assert len(align_events) == 2
    for e in align_events:
        assert len(e["rationale"]) >= 20
        # Dynamic signals interpolated.
        assert "co_to_cosine=" in e["rationale"]
        assert "threshold=" in e["rationale"]
        assert "weak=" in e["rationale"]


# --------------------------------------------------------------------- #
# 8. Threshold override from inputs (config→inputs merge)
# --------------------------------------------------------------------- #


def test_threshold_override_from_inputs():
    # CO at 60° from TO → cosine cos(60°)=0.5. Default 0.45 would pass;
    # an inputs threshold of 0.60 flags it.
    embedder = _StubEmbedder({"TO": 0.0, "CO": math.pi / 3})
    objectives = _objectives(
        [_to("TO-01", "TO terminal statement")],
        [_co("CO-01", "CO half", terminal_id="TO-01")],
    )
    v = CoTerminalAlignmentValidator(embedder=embedder)
    # Sanity: under the default floor (0.45) the 0.5-cosine CO passes.
    assert DEFAULT_THRESHOLD == 0.45
    r_default = v.validate({"synthesized_objectives": objectives})
    assert "CO_TERMINAL_SEMANTIC_MISMATCH" not in _codes(r_default)
    # With a top-level threshold of 0.60, the 0.5-cosine CO is flagged.
    r_strict = v.validate(
        {"synthesized_objectives": objectives, "threshold": 0.60}
    )
    assert "CO_TERMINAL_SEMANTIC_MISMATCH" in _codes(r_strict)


# --------------------------------------------------------------------- #
# 9. WS2 fast-path reads stamped score (no encode)
# --------------------------------------------------------------------- #


def test_ws2_fast_path_reads_stamped_score():
    embedder = _StubEmbedder({"TO": 0.0, "CO": 0.0})  # would score 1.0
    objectives = _objectives(
        [_to("TO-01", "TO terminal statement")],
        [
            _co(
                "CO-01",
                "CO stamped",
                terminal_id="TO-01",
                terminal_link_score=0.20,  # stamped weak score
            ),
        ],
    )
    v = CoTerminalAlignmentValidator(embedder=embedder)
    result = v.validate({"synthesized_objectives": objectives})
    # Flagged via the stamped 0.20 (< 0.45) despite the aligned encode.
    assert "CO_TERMINAL_SEMANTIC_MISMATCH" in _codes(result)
    # No encode call — the fast-path short-circuits before embedding.
    assert embedder.calls == []


# --------------------------------------------------------------------- #
# Router registration
# --------------------------------------------------------------------- #


def test_router_resolves_co_terminal_alignment_to_chapter_objective_builder():
    from MCP.hardening.gate_input_routing import (
        _build_chapter_objective_coverage_inputs,
        default_router,
    )

    r = default_router()
    dotted = (
        "lib.validators.co_terminal_alignment.CoTerminalAlignmentValidator"
    )
    assert dotted in r.builders, (
        f"WS3 regression: no builder registered for {dotted}; gate would "
        f"silently skip via __no_builder_registered__."
    )
    assert r.builders[dotted] is _build_chapter_objective_coverage_inputs
