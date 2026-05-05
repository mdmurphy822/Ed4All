"""Regression tests for C2 silent-degradation fix in
:meth:`CourseforgeRouter.route_all`.

Pre-fix: per-block dispatch exceptions in the outline / rewrite loop
were logged and the block was DROPPED from the return list. Because
the block never re-emerged with an ``escalation_marker``, the W5
packager-side filter (``escalation_marker is not None``) didn't catch
it, the IMSCC shipped without the block, and downstream consumers had
no fail-closed signal — silent-degradation pattern W7 of the W1-W9
plan, recurring at the router surface.

Post-fix: dispatch failures stamp the failing block with one of two
dedicated markers from ``Block._ESCALATION_MARKERS``:

- ``outline_dispatch_error`` — outline-tier exception (network
  failure / provider raise / unhandled exception inside the outline
  call path). Distinct from ``outline_budget_exhausted`` which is
  reserved for the regen-budget exhaustion path in
  ``route_with_self_consistency``.
- ``rewrite_dispatch_error`` — rewrite-tier exception. Distinct from
  ``validator_consensus_fail`` which is reserved for the rewrite-tier
  regen-budget exhaustion path in
  ``route_rewrite_with_remediation``.

Both markers ride the standard
:meth:`lib.validators.imscc.IMSCCValidator._check_escalated_blocks_absent`
filter (``escalation_marker is not None``) so a leaked dispatch-error
block trips ``ESCALATED_BLOCK_IN_IMSCC`` at the packaging gate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.router.router import CourseforgeRouter  # noqa: E402
from blocks import Block, _ESCALATION_MARKERS  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _block(
    *,
    block_type: str = "concept",
    block_id: str = "page1#concept_intro_0",
    content: Any = "hello",
    escalation_marker: Optional[str] = None,
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page1",
        sequence=0,
        content=content,
        escalation_marker=escalation_marker,
    )


class _RaisingOutlineProvider:
    """Outline provider that raises ``RuntimeError`` on every call.

    Simulates a transient network failure / provider crash that the
    pre-fix router silently swallowed.
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls: List[str] = []

    def generate_outline(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        **kwargs: Any,
    ) -> Block:
        self.calls.append(block.block_id)
        raise self._exc


class _PassthroughOutlineProvider:
    """Outline provider that returns the block unmodified."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def generate_outline(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        **kwargs: Any,
    ) -> Block:
        self.calls.append(block.block_id)
        return block


class _RaisingRewriteProvider:
    """Rewrite provider that raises ``RuntimeError`` on every call."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls: List[str] = []

    def generate_rewrite(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        **kwargs: Any,
    ) -> Block:
        self.calls.append(block.block_id)
        raise self._exc


class _PassthroughRewriteProvider:
    """Rewrite provider that returns the block unmodified."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def generate_rewrite(
        self,
        block: Block,
        *,
        source_chunks: Any,
        objectives: Any,
        **kwargs: Any,
    ) -> Block:
        self.calls.append(block.block_id)
        return block


# ---------------------------------------------------------------------------
# Marker enum
# ---------------------------------------------------------------------------


def test_dispatch_error_markers_registered_in_canonical_set():
    """Both new markers must live in ``Block._ESCALATION_MARKERS`` so
    ``Block.__post_init__`` doesn't raise on the
    ``dataclasses.replace`` inside the router exception handlers."""
    assert "outline_dispatch_error" in _ESCALATION_MARKERS
    assert "rewrite_dispatch_error" in _ESCALATION_MARKERS


# ---------------------------------------------------------------------------
# Outline-tier dispatch-error stamping
# ---------------------------------------------------------------------------


def test_route_all_outline_dispatch_error_stamps_marker():
    """Per-block outline-tier exception → block is preserved at its
    original index with ``escalation_marker="outline_dispatch_error"``;
    the rewrite tier is NOT invoked on the failed block."""
    raising = _RaisingOutlineProvider(RuntimeError("network failure"))
    rewrite = _PassthroughRewriteProvider()
    r = CourseforgeRouter(
        outline_provider=raising,
        rewrite_provider=rewrite,
    )
    blk = _block(block_type="concept", block_id="page1#concept_x_0")
    out = r.route_all([blk])

    assert raising.calls == ["page1#concept_x_0"]
    # Failed block bypasses rewrite tier entirely.
    assert rewrite.calls == []
    # Block survives in the return list (NOT silently dropped).
    assert len(out) == 1
    # Stamped with the dedicated dispatch-error marker.
    assert out[0].escalation_marker == "outline_dispatch_error"
    # Block identity preserved.
    assert out[0].block_id == "page1#concept_x_0"


def test_route_all_outline_dispatch_error_preserves_ordering_in_mixed_batch():
    """A mixed batch of passing + failing outline dispatches returns
    every block at its original index; failing blocks carry
    ``outline_dispatch_error``; passing blocks reach the rewrite tier
    cleanly."""

    class _MixedOutline:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def generate_outline(
            self,
            block: Block,
            *,
            source_chunks: Any,
            objectives: Any,
            **kwargs: Any,
        ) -> Block:
            self.calls.append(block.block_id)
            if "failure" in block.block_id:
                raise RuntimeError("simulated outline crash")
            return block

    mixed = _MixedOutline()
    rewrite = _PassthroughRewriteProvider()
    r = CourseforgeRouter(outline_provider=mixed, rewrite_provider=rewrite)

    b1 = _block(block_type="concept", block_id="page1#concept_a_0")
    b2 = _block(block_type="concept", block_id="page1#concept_failure_1")
    b3 = _block(block_type="concept", block_id="page1#concept_c_2")
    out = r.route_all([b1, b2, b3])

    # Every block returned, in input order.
    assert [b.block_id for b in out] == [
        "page1#concept_a_0",
        "page1#concept_failure_1",
        "page1#concept_c_2",
    ]
    # Only the surviving two reached rewrite.
    assert rewrite.calls == [
        "page1#concept_a_0",
        "page1#concept_c_2",
    ]
    # Failing block stamped with the dedicated marker.
    failed = next(b for b in out if "failure" in b.block_id)
    assert failed.escalation_marker == "outline_dispatch_error"
    # Surviving blocks unmarked.
    survivors = [b for b in out if "failure" not in b.block_id]
    assert all(b.escalation_marker is None for b in survivors)


# ---------------------------------------------------------------------------
# Rewrite-tier dispatch-error stamping
# ---------------------------------------------------------------------------


def test_route_all_rewrite_dispatch_error_stamps_marker():
    """Per-block rewrite-tier exception (after a clean outline pass) →
    the OUTLINED block is preserved at its original index with
    ``escalation_marker="rewrite_dispatch_error"``."""
    outline = _PassthroughOutlineProvider()
    raising = _RaisingRewriteProvider(RuntimeError("rewrite network failure"))
    r = CourseforgeRouter(
        outline_provider=outline,
        rewrite_provider=raising,
    )
    blk = _block(block_type="concept", block_id="page1#concept_y_0")
    out = r.route_all([blk])

    # Outline ran cleanly.
    assert outline.calls == ["page1#concept_y_0"]
    # Rewrite was attempted (and raised).
    assert raising.calls == ["page1#concept_y_0"]
    # Block survives in the return list (NOT silently dropped).
    assert len(out) == 1
    # Stamped with the dedicated dispatch-error marker.
    assert out[0].escalation_marker == "rewrite_dispatch_error"
    # Block identity preserved.
    assert out[0].block_id == "page1#concept_y_0"


def test_route_all_rewrite_dispatch_error_does_not_overwrite_outline_marker():
    """If the outline tier already stamped a marker on the block, a
    rewrite-tier exception MUST NOT overwrite it. The router's rewrite
    branch only stamps when the outlined block carries
    ``escalation_marker is None``. (In practice this branch is
    unreachable on the same block today because outline-failed blocks
    skip the rewrite tier entirely — but the guard exists in the
    handler so the contract is worth pinning.)"""
    # An outline provider that returns a block with a pre-stamped
    # ``outline_budget_exhausted`` marker (mimicking the
    # ``escalate_immediately: true`` policy short-circuit, where the
    # outlined block IS routed through the rewrite tier).
    pre_stamped_marker = "outline_budget_exhausted"

    class _PreStampingOutline:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def generate_outline(
            self,
            block: Block,
            *,
            source_chunks: Any,
            objectives: Any,
            **kwargs: Any,
        ) -> Block:
            self.calls.append(block.block_id)
            return Block(
                block_id=block.block_id,
                block_type=block.block_type,
                page_id=block.page_id,
                sequence=block.sequence,
                content=block.content,
                escalation_marker=pre_stamped_marker,
            )

    outline = _PreStampingOutline()
    raising = _RaisingRewriteProvider(RuntimeError("rewrite fail"))
    r = CourseforgeRouter(
        outline_provider=outline,
        rewrite_provider=raising,
    )
    blk = _block(block_type="concept", block_id="page1#concept_z_0")
    out = r.route_all([blk])

    assert len(out) == 1
    # Pre-stamped outline marker survives — the rewrite-handler guard
    # ``if outlined.escalation_marker is not None`` keeps the original
    # marker rather than clobbering it with ``rewrite_dispatch_error``.
    assert out[0].escalation_marker == pre_stamped_marker


def test_route_all_rewrite_dispatch_error_preserves_ordering_in_mixed_batch():
    """A mixed batch where one block's rewrite raises returns every
    block at its original index; failing block carries
    ``rewrite_dispatch_error``; surviving blocks pass through cleanly."""

    class _MixedRewrite:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def generate_rewrite(
            self,
            block: Block,
            *,
            source_chunks: Any,
            objectives: Any,
            **kwargs: Any,
        ) -> Block:
            self.calls.append(block.block_id)
            if "rwfail" in block.block_id:
                raise RuntimeError("rewrite tier crash")
            return block

    outline = _PassthroughOutlineProvider()
    mixed = _MixedRewrite()
    r = CourseforgeRouter(outline_provider=outline, rewrite_provider=mixed)

    b1 = _block(block_type="concept", block_id="page1#concept_a_0")
    b2 = _block(block_type="concept", block_id="page1#concept_rwfail_1")
    b3 = _block(block_type="concept", block_id="page1#concept_c_2")
    out = r.route_all([b1, b2, b3])

    # Every block returned, in input order.
    assert [b.block_id for b in out] == [
        "page1#concept_a_0",
        "page1#concept_rwfail_1",
        "page1#concept_c_2",
    ]
    # Failing block stamped with the dedicated rewrite marker.
    failed = next(b for b in out if "rwfail" in b.block_id)
    assert failed.escalation_marker == "rewrite_dispatch_error"
    # Surviving blocks unmarked.
    survivors = [b for b in out if "rwfail" not in b.block_id]
    assert all(b.escalation_marker is None for b in survivors)


# ---------------------------------------------------------------------------
# Symmetric coverage — per-tier markers must be DIFFERENT
# ---------------------------------------------------------------------------


def test_outline_and_rewrite_dispatch_error_markers_are_distinct():
    """Outline + rewrite dispatch errors get DIFFERENT markers so a
    postmortem can tell them apart in the IMSCC W5 audit trail."""
    outline_raising = _RaisingOutlineProvider(RuntimeError("outline fail"))
    rewrite_passing = _PassthroughRewriteProvider()
    r1 = CourseforgeRouter(
        outline_provider=outline_raising,
        rewrite_provider=rewrite_passing,
    )
    out1 = r1.route_all([_block(block_id="page1#concept_outline_0")])

    outline_passing = _PassthroughOutlineProvider()
    rewrite_raising = _RaisingRewriteProvider(RuntimeError("rewrite fail"))
    r2 = CourseforgeRouter(
        outline_provider=outline_passing,
        rewrite_provider=rewrite_raising,
    )
    out2 = r2.route_all([_block(block_id="page1#concept_rewrite_0")])

    assert out1[0].escalation_marker == "outline_dispatch_error"
    assert out2[0].escalation_marker == "rewrite_dispatch_error"
    assert out1[0].escalation_marker != out2[0].escalation_marker
