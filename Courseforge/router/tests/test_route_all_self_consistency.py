"""Tests for ``route_all``'s self-consistency dispatch.

C1 fix (2026-05): ``route_all`` now consults the canonical
:meth:`CourseforgeRouter._resolve_n_candidates` chain, which terminates
at layer 5's hardcoded ``_DEFAULT_OUTLINE_N_CANDIDATES = 3``. As a
result every block routes through
:meth:`CourseforgeRouter.route_with_self_consistency` by default — the
prior asymmetric ``_resolve_explicit_n_candidates`` gate has been
removed because it made the doc-claimed default of 3 inert in
production. Per-block-type opt-out via
``block_routing.yaml::blocks.{type}.n_candidates: 1`` (resolves at
layer 2) short-circuits to the single-candidate direct
:meth:`CourseforgeRouter.route` dispatch.

Also pins the asymmetric default: the rewrite pass stays on the direct
:meth:`route` call regardless of n_candidates resolution. Operators that
want rewrite-tier multi-candidate sampling must call
:meth:`route_rewrite_with_remediation` explicitly. The asymmetry lives
in the ``route_all`` docstring so downstream readers don't trip on the
missing rewrite-mirror path; this test documents the expected behaviour
in code so a future refactor that accidentally widens the rewrite path
trips a regression failure.

Test count target: ≥5 covering:

- ``test_route_all_dispatches_through_self_consistency_when_n_candidates_gt_1``
- ``test_route_all_dispatches_direct_when_n_candidates_eq_1``
- ``test_route_all_per_block_n_candidates_one_skips_self_consistency``
- ``test_route_all_outline_pass_uses_self_consistency``
- ``test_route_all_rewrite_pass_stays_direct``
- ``test_route_all_default_routes_through_self_consistency`` — the
  hardcoded default of 3 fires by default; closes the C1 audit gap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.router import (  # noqa: E402
    BlockProviderSpec,
    CourseforgeRouter,
)
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _block(
    *,
    block_type: str = "concept",
    block_id: str = "page1#concept_intro_0",
    content: Any = "hello",
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page1",
        sequence=0,
        content=content,
    )


class _RecordingProvider:
    """Stub provider exposing both outline + rewrite surfaces.

    Tracks every call's kwargs so tests can assert which surface fired
    + how many times. Returns the input block unchanged so the loop
    exits after one dispatch when paired with a passing validator
    (or an empty validator list, which is "trivially passes").
    """

    def __init__(self) -> None:
        self.outline_calls: List[Dict[str, Any]] = []
        self.rewrite_calls: List[Dict[str, Any]] = []

    def generate_outline(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        **kwargs: Any,
    ) -> Block:
        self.outline_calls.append(
            {
                "block": block,
                "source_chunks": source_chunks,
                "objectives": objectives,
                **{k: v for k, v in kwargs.items() if k == "remediation_suffix"},
            }
        )
        return block

    def generate_rewrite(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        **kwargs: Any,
    ) -> Block:
        self.rewrite_calls.append(
            {
                "block": block,
                "source_chunks": source_chunks,
                "objectives": objectives,
                **{k: v for k, v in kwargs.items() if k == "remediation_suffix"},
            }
        )
        return block


class _StubPolicy:
    """Minimal policy stand-in carrying just the
    ``n_candidates_by_block_type`` fast-lookup map.

    Mirrors the surface :meth:`CourseforgeRouter._resolve_n_candidates`
    reaches into. ``resolve`` is wired so spec-resolution doesn't crash;
    it returns ``None`` (the "fall through to env / hardcoded default"
    signal).
    """

    def __init__(
        self,
        *,
        n_candidates_by_block_type: Optional[Dict[str, int]] = None,
        regen_budget_by_block_type: Optional[Dict[str, int]] = None,
    ) -> None:
        self.n_candidates_by_block_type = n_candidates_by_block_type or {}
        self.regen_budget_by_block_type = regen_budget_by_block_type or {}
        # Symmetric to the policy's empty/honour shape; the router's
        # self-consistency loop doesn't need a non-None spec for these
        # tests (the providers are injected so the spec only feeds the
        # Touch.model + Touch.provider audit fields).
        self.escalate_immediately_by_block_type: Dict[str, bool] = {}
        self.regen_budget_rewrite_by_block_type: Dict[str, int] = {}

    def resolve(self, block_id: str, block_type: str, tier: str) -> Any:
        return None


# ---------------------------------------------------------------------------
# Subtask-33 contract: opt-in self-consistency dispatch in route_all
# ---------------------------------------------------------------------------


def test_route_all_dispatches_through_self_consistency_when_n_candidates_gt_1(
    monkeypatch,
):
    """When the env var ``COURSEFORGE_OUTLINE_N_CANDIDATES`` is set
    to a value > 1, ``route_all`` dispatches each block's outline pass
    via ``route_with_self_consistency`` (which fires N candidate
    dispatches per block). With an empty validator list the
    self-consistency loop exits after the first candidate (trivial
    pass), so the per-block dispatch count is 1 — but the audit chain
    carries a ``self_consistency_winner`` Touch that the legacy
    direct path does NOT emit.
    """
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "3")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", raising=False)
    provider = _RecordingProvider()
    rewrite_provider = _RecordingProvider()
    r = CourseforgeRouter(
        outline_provider=provider,
        rewrite_provider=rewrite_provider,
    )
    blocks = [
        _block(block_type="concept", block_id="page1#concept_a_0"),
        _block(block_type="example", block_id="page1#example_b_0"),
    ]
    out = r.route_all(blocks)
    # One outline call per block (validator list empty → first
    # candidate trivially passes → self-consistency loop exits early).
    assert len(provider.outline_calls) == 2
    # Every returned block carries a ``self_consistency_winner`` Touch
    # — that's the signal the dispatch went through
    # ``route_with_self_consistency`` (the direct ``route`` path
    # appends only an outline-tier Touch with a different purpose).
    for b in out:
        purposes = {t.purpose for t in b.touched_by}
        assert "self_consistency_winner" in purposes, (
            f"Block {b.block_id} missing self_consistency_winner Touch; "
            f"touched_by={b.touched_by!r}"
        )


def test_route_all_dispatches_direct_when_n_candidates_eq_1(monkeypatch):
    """When the env var resolves to exactly 1, ``route_all`` falls
    through to the direct ``route(block, tier="outline")`` dispatch
    path — the resolved value is NOT > 1, so the self-consistency
    branch is skipped. Returned blocks carry an outline Touch with
    a NON-``self_consistency_winner`` purpose (the route() path
    emits a different purpose tag)."""
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")
    provider = _RecordingProvider()
    rewrite_provider = _RecordingProvider()
    r = CourseforgeRouter(
        outline_provider=provider,
        rewrite_provider=rewrite_provider,
    )
    blocks = [_block(block_type="concept", block_id="page1#concept_a_0")]
    out = r.route_all(blocks)
    assert len(provider.outline_calls) == 1
    # Direct ``route()`` path does NOT append a
    # ``self_consistency_winner`` Touch.
    for b in out:
        purposes = {t.purpose for t in b.touched_by}
        assert "self_consistency_winner" not in purposes, (
            f"Block {b.block_id} accidentally went through "
            f"self-consistency at n=1; touched_by={b.touched_by!r}"
        )


def test_route_all_per_block_n_candidates_one_skips_self_consistency(monkeypatch):
    """C1 fix (2026-05): per-block-type ``n_candidates: 1`` is the
    operator opt-out from the default-on self-consistency dispatch.
    When a policy entry pins ``example`` to ``n_candidates=1`` (and
    ``concept`` carries no entry, falling through to layer-5 default
    of 3), the ``concept`` block dispatches via self-consistency while
    the ``example`` block short-circuits to the direct route() path.
    Documents per-block resolution under mixed policy + default."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    policy = _StubPolicy(
        n_candidates_by_block_type={"example": 1},
    )
    provider = _RecordingProvider()
    rewrite_provider = _RecordingProvider()
    r = CourseforgeRouter(
        outline_provider=provider,
        rewrite_provider=rewrite_provider,
        policy=policy,
    )
    blocks = [
        _block(block_type="concept", block_id="page1#concept_a_0"),
        _block(block_type="example", block_id="page1#example_b_0"),
    ]
    out = r.route_all(blocks)
    by_id = {b.block_id: b for b in out}
    # ``concept`` block dispatched through self-consistency (layer-5
    # default of 3 fires).
    concept_purposes = {t.purpose for t in by_id["page1#concept_a_0"].touched_by}
    assert "self_consistency_winner" in concept_purposes, (
        f"concept block missing self_consistency_winner; "
        f"touched_by={by_id['page1#concept_a_0'].touched_by!r}"
    )
    # ``example`` block short-circuited to direct route() via
    # per-block-type ``n_candidates: 1`` opt-out — no winner Touch.
    example_purposes = {t.purpose for t in by_id["page1#example_b_0"].touched_by}
    assert "self_consistency_winner" not in example_purposes, (
        f"example block unexpectedly went through self-consistency "
        f"despite n_candidates=1 opt-out; "
        f"touched_by={by_id['page1#example_b_0'].touched_by!r}"
    )


def test_route_all_outline_pass_uses_self_consistency(monkeypatch):
    """Direct dispatch-method observation: with the explicit override
    set, every block's outline pass routes through
    ``route_with_self_consistency`` (verified via spy). Pins the
    Subtask-33 surface contract — the helper dispatched is the
    n-candidate sampler, not the single-shot dispatcher."""
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "2")
    provider = _RecordingProvider()
    rewrite_provider = _RecordingProvider()
    r = CourseforgeRouter(
        outline_provider=provider,
        rewrite_provider=rewrite_provider,
    )
    # Spy on both dispatch surfaces so we can prove which one fired
    # for the outline pass.
    self_consistency_calls: List[Block] = []
    direct_route_calls: List[tuple] = []
    original_self_consistency = r.route_with_self_consistency
    original_route = r.route

    def spy_self_consistency(blk: Block, **kwargs: Any) -> Block:
        self_consistency_calls.append(blk)
        return original_self_consistency(blk, **kwargs)

    def spy_route(blk: Block, **kwargs: Any) -> Block:
        direct_route_calls.append((blk, kwargs.get("tier")))
        return original_route(blk, **kwargs)

    r.route_with_self_consistency = spy_self_consistency  # type: ignore[assignment]
    r.route = spy_route  # type: ignore[assignment]

    blocks = [
        _block(block_type="concept", block_id="page1#concept_a_0"),
        _block(block_type="example", block_id="page1#example_b_0"),
    ]
    r.route_all(blocks)

    # 2 outline-pass dispatches went through route_with_self_consistency.
    assert len(self_consistency_calls) == 2
    # The rewrite-pass tier dispatches still go through route() (direct).
    rewrite_direct = [
        call for call in direct_route_calls if call[1] == "rewrite"
    ]
    assert len(rewrite_direct) == 2, (
        f"rewrite-pass dispatches must still go through direct route(); "
        f"saw {direct_route_calls!r}"
    )
    # Outline-pass direct calls: zero from route_all itself, but
    # route_with_self_consistency internally calls route(tier="outline")
    # — those are the ONLY outline-tier route() calls allowed. So
    # outline route() calls == self-consistency dispatches (each
    # self-consistency loop fires at least one route() call).
    outline_direct = [
        call for call in direct_route_calls if call[1] == "outline"
    ]
    # Self-consistency loop with empty validators dispatches exactly
    # ONE outline candidate per block (first candidate "passes"
    # trivially), so we expect one route(tier="outline") per block.
    assert len(outline_direct) == 2


def test_route_all_rewrite_pass_stays_direct(monkeypatch):
    """Asymmetric default: even when n_candidates > 1 resolves for a
    block type, the rewrite pass dispatches through the direct
    ``route(block, tier="rewrite")`` call — NOT through
    ``route_rewrite_with_remediation``. Pins the Subtask-33
    asymmetric contract documented in the ``route_all`` docstring."""
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "3")
    monkeypatch.delenv("COURSEFORGE_REWRITE_REGEN_BUDGET", raising=False)
    provider = _RecordingProvider()
    rewrite_provider = _RecordingProvider()
    r = CourseforgeRouter(
        outline_provider=provider,
        rewrite_provider=rewrite_provider,
    )
    # Spy on the rewrite-tier remediation surface — it must NOT be
    # invoked from route_all even with the outline-tier opt-in set.
    rewrite_remediation_calls: List[Block] = []
    original_rewrite_remediation = r.route_rewrite_with_remediation

    def spy_rewrite_remediation(blk: Block, **kwargs: Any) -> Block:
        rewrite_remediation_calls.append(blk)
        return original_rewrite_remediation(blk, **kwargs)

    r.route_rewrite_with_remediation = spy_rewrite_remediation  # type: ignore[assignment]

    blocks = [_block(block_type="concept", block_id="page1#concept_a_0")]
    r.route_all(blocks)

    # The rewrite-tier remediation method MUST NOT have been touched.
    assert len(rewrite_remediation_calls) == 0, (
        f"route_all unexpectedly dispatched rewrite pass through "
        f"route_rewrite_with_remediation; calls={rewrite_remediation_calls!r}"
    )
    # But the rewrite tier WAS dispatched (legacy direct path).
    assert len(rewrite_provider.rewrite_calls) == 1


def test_route_all_default_routes_through_self_consistency(monkeypatch):
    """C1 fix sentinel (2026-05): with no policy + no env var + no
    constructor override, the layer-5 hardcoded default of 3 fires
    via :meth:`CourseforgeRouter._resolve_n_candidates`, so every
    block routes through ``route_with_self_consistency``. Replaces
    the previous "stays-direct" sentinel that pinned the inert
    asymmetric ``_resolve_explicit_n_candidates`` behaviour. Closes
    the C1 audit gap where ``CLAUDE.md``'s "Default 3" claim was
    contradicted by the de facto runtime default of 1."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_N_CANDIDATES", raising=False)
    provider = _RecordingProvider()
    rewrite_provider = _RecordingProvider()
    r = CourseforgeRouter(
        outline_provider=provider,
        rewrite_provider=rewrite_provider,
        # No n_candidates kwarg, no policy with n_candidates_by_block_type.
    )
    blocks = [
        _block(block_type="concept", block_id="page1#concept_a_0"),
        _block(block_type="example", block_id="page1#example_b_0"),
    ]
    out = r.route_all(blocks)

    # Every block goes through self-consistency by default.
    for b in out:
        purposes = {t.purpose for t in b.touched_by}
        assert "self_consistency_winner" in purposes, (
            f"Block {b.block_id} missing self_consistency_winner Touch "
            f"under default-on routing; touched_by={b.touched_by!r}"
        )
    # Both blocks reached both tiers (validator list empty → first
    # candidate trivially passes → one outline call per block).
    assert len(provider.outline_calls) == 2
    assert len(rewrite_provider.rewrite_calls) == 2
