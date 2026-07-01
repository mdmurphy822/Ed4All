"""W8.9 — the C1/C3/C4 technique levers actually have effect at the router.

Before W8.9 ``technique_modes.apply_mode_to_env`` projected
``COURSEFORGE_CHUNK_SCOPED`` (C1) / ``COURSEFORGE_SELF_VERIFY`` (C3) /
``COURSEFORGE_REFINE_ROUNDS`` (C4) onto the env but NO consumer read them —
dead levers. These tests prove the router now honors them:

* C1 chunk-scoping narrows the per-block ``source_chunks`` to the block's own
  cited chunks (filter-only; never strips all grounding).
* C3 self-verify + C4 refine run bounded post-winner passes that can only
  IMPROVE a validator-passing winner (never-regress).
* Default OFF (all knobs unset) → byte-identical: no scoping, no extra
  dispatch.

All dispatch is faked — no LLM / torch import on the path.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.router import CourseforgeRouter  # noqa: E402
from MCP.hardening.validation_gates import GateResult  # noqa: E402
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(*, source_ids=(), source_references=()) -> Block:
    return Block(
        block_id="page1#concept_intro_0",
        block_type="concept",
        page_id="page1",
        sequence=0,
        content="v0",
        source_ids=tuple(source_ids),
        source_references=tuple(source_references),
    )


class _EchoProvider:
    """Records the ``source_chunks`` each dispatch received; echoes a fresh
    block whose content is ``v{call_index}`` so acceptance is observable."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def _run(self, block: Block, source_chunks, objectives, **kwargs) -> Block:
        idx = len(self.calls) + 1
        self.calls.append(
            {
                "source_chunks": source_chunks,
                "remediation_suffix": kwargs.get("remediation_suffix"),
            }
        )
        return dataclasses.replace(block, content=f"v{idx}")

    def generate_outline(self, block, *, source_chunks, objectives, **kwargs):
        return self._run(block, source_chunks, objectives, **kwargs)

    def generate_rewrite(self, block, *, source_chunks, objectives, **kwargs):
        return self._run(block, source_chunks, objectives, **kwargs)


class _StatefulValidator:
    """Passes the first N validate() calls, fails the rest (never-regress probe)."""

    def __init__(self, *, pass_first: int) -> None:
        self.validator_name = "outline_curie_anchoring"
        self._pass_first = pass_first
        self.calls = 0

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        self.calls += 1
        passed = self.calls <= self._pass_first
        return GateResult(
            gate_id=self.validator_name,
            validator_name=self.validator_name,
            validator_version="1.0.0",
            passed=passed,
            action=None if passed else "regenerate",
        )


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


_CHUNKS = [
    {"id": "c1", "text": "alpha"},
    {"id": "c2", "text": "beta"},
    {"id": "c3", "text": "gamma"},
]


def _clear_levers(mp):
    for k in (
        "COURSEFORGE_CHUNK_SCOPED",
        "COURSEFORGE_SELF_VERIFY",
        "COURSEFORGE_REFINE_ROUNDS",
        "COURSEFORGE_BEST_OF_N_SELECT_BY",
        "COURSEFORGE_OUTLINE_N_CANDIDATES",
    ):
        mp.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# C1 chunk-scoping
# ---------------------------------------------------------------------------


def test_chunk_scoping_off_passes_full_list(monkeypatch):
    _clear_levers(monkeypatch)
    prov = _EchoProvider()
    router = CourseforgeRouter(outline_provider=prov)
    router.route(
        _block(source_ids=("c1",)),
        tier="outline",
        source_chunks=list(_CHUNKS),
    )
    # OFF → full list unchanged.
    assert [c["id"] for c in prov.calls[0]["source_chunks"]] == ["c1", "c2", "c3"]


def test_chunk_scoping_on_narrows_to_cited(monkeypatch):
    _clear_levers(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_CHUNK_SCOPED", "true")
    prov = _EchoProvider()
    router = CourseforgeRouter(outline_provider=prov)
    router.route(
        _block(source_ids=("c1",), source_references=({"sourceId": "c3"},)),
        tier="outline",
        source_chunks=list(_CHUNKS),
    )
    # ON → only the block's own cited chunks (c1 via source_ids, c3 via refs).
    assert [c["id"] for c in prov.calls[0]["source_chunks"]] == ["c1", "c3"]


def test_chunk_scoping_on_no_refs_keeps_full_list(monkeypatch):
    _clear_levers(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_CHUNK_SCOPED", "true")
    prov = _EchoProvider()
    router = CourseforgeRouter(outline_provider=prov)
    router.route(_block(), tier="outline", source_chunks=list(_CHUNKS))
    # No referenced ids → cannot scope safely → unchanged.
    assert [c["id"] for c in prov.calls[0]["source_chunks"]] == ["c1", "c2", "c3"]


def test_chunk_scoping_on_no_overlap_keeps_full_list(monkeypatch):
    _clear_levers(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_CHUNK_SCOPED", "true")
    prov = _EchoProvider()
    router = CourseforgeRouter(outline_provider=prov)
    router.route(
        _block(source_ids=("zzz",)),
        tier="outline",
        source_chunks=list(_CHUNKS),
    )
    # No id overlap → never strip all grounding → unchanged.
    assert [c["id"] for c in prov.calls[0]["source_chunks"]] == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# C3/C4 refine + verify
# ---------------------------------------------------------------------------


def test_refine_verify_off_no_extra_dispatch(monkeypatch):
    _clear_levers(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")
    prov = _EchoProvider()
    router = CourseforgeRouter(outline_provider=prov)
    out = router.route_with_self_consistency(
        _block(), n_candidates=1, validators=[], source_chunks=list(_CHUNKS),
    )
    # Only the single in-loop dispatch; no refine/verify pass.
    assert len(prov.calls) == 1
    assert out.content == "v1"


def test_refine_verify_on_runs_bounded_passes_and_accepts(monkeypatch):
    _clear_levers(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")
    monkeypatch.setenv("COURSEFORGE_SELF_VERIFY", "true")
    monkeypatch.setenv("COURSEFORGE_REFINE_ROUNDS", "2")
    prov = _EchoProvider()
    cap = _FakeCapture()
    router = CourseforgeRouter(outline_provider=prov, capture=cap)
    out = router.route_with_self_consistency(
        _block(), n_candidates=1, validators=[], source_chunks=list(_CHUNKS),
    )
    # 1 in-loop + 1 verify + 2 refine = 4 dispatches; empty validators → each
    # refined candidate accepted → final content is the last dispatch.
    assert len(prov.calls) == 4
    assert out.content == "v4"
    # A verify pass carried the self-critique suffix.
    assert any("SELF-VERIFY" in (c["remediation_suffix"] or "") for c in prov.calls)
    # Refine/verify decision-capture event fired.
    assert any(
        e.get("ml_features", {}).get("phase_kind") == "refine_verify"
        for e in cap.events
    )
    assert any(t.purpose == "refine_verify" for t in out.touched_by)


def test_refine_verify_never_regresses_a_passing_winner(monkeypatch):
    _clear_levers(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")
    monkeypatch.setenv("COURSEFORGE_SELF_VERIFY", "true")
    monkeypatch.setenv("COURSEFORGE_REFINE_ROUNDS", "2")
    prov = _EchoProvider()
    # Validator passes ONLY the first (in-loop winner) call; every refine/verify
    # candidate fails its chain → must be rejected, keeping the winner.
    validator = _StatefulValidator(pass_first=1)
    router = CourseforgeRouter(outline_provider=prov)
    out = router.route_with_self_consistency(
        _block(), n_candidates=1, validators=[validator],
        source_chunks=list(_CHUNKS),
    )
    # Winner was candidate v1; refine passes ran but none re-passed → v1 kept.
    assert out.content == "v1"
    assert len(prov.calls) >= 2  # at least one refine/verify dispatch attempted


def test_refine_verify_on_rewrite_tier(monkeypatch):
    _clear_levers(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")
    monkeypatch.setenv("COURSEFORGE_REFINE_ROUNDS", "1")
    prov = _EchoProvider()
    router = CourseforgeRouter(rewrite_provider=prov)
    out = router.route_rewrite_with_remediation(
        _block(), n_candidates=1, validators=[], source_chunks=list(_CHUNKS),
    )
    # 1 in-loop rewrite + 1 refine round.
    assert len(prov.calls) == 2
    assert out.content == "v2"
