"""dispatch-resilience-2026-07 — outline per-block dispatch resilience.

Regression suite for the outline-tier dispatch-resilience fix in
``MCP/tools/pipeline_tools.py::_run_content_generation_outline``. Before the
fix, ANY exception from ``route_with_self_consistency`` was caught by a bare
``except Exception`` that stamped the block ``outline_budget_exhausted`` (the
SEMANTIC regen-exhaustion marker) and continued — so an intermittent endpoint
outage silently chewed through the whole course, mislabelling every block and
corrupting postmortems, with no retry and no circuit breaker.

The fix:
  (a) classifies the exception (transient vs permanent via the shared MCP error
      taxonomy) and does a bounded in-loop retry on transient errors;
  (b) stamps the canonical ``outline_dispatch_error`` marker on give-up (NOT
      ``outline_budget_exhausted``);
  (c) a consecutive-failure circuit breaker (``COURSEFORGE_OUTLINE_DISPATCH_BREAKER``,
      default 5, 0=disabled) halts the phase LOUDLY instead of stamping the
      rest of the course, leaving already-authored/checkpointed blocks
      resumable;
  (d) genuine SEMANTIC budget exhaustion (the router RETURNS a marker block,
      no exception) still yields ``outline_budget_exhausted`` untouched.

We stub ``CourseforgeRouter.route_with_self_consistency`` per-test so the loop's
resilience path is exercised deterministically (no real LLM dispatch), and
monkeypatch ``time.sleep`` so the backoff ladder never actually waits.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block  # noqa: E402
from MCP.tools import pipeline_tools as _pt  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixture + patch helpers (mirror test_pipeline_tools_outline_self_consistency)
# ---------------------------------------------------------------------- #


def _seed_project(tmp_path: Path, project_id: str) -> Path:
    exports_root = tmp_path / "Courseforge" / "exports"
    project_path = exports_root / project_id
    (project_path / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    (project_path / "01_learning_objectives" / "synthesized_objectives.json").write_text(
        json.dumps({
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Describe core concept A in detail."}
            ],
            "chapter_objectives": [],
        }),
        encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": project_id, "duration_weeks": 1}),
        encoding="utf-8",
    )
    return project_path


def _patch_project_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)


def _no_sleep(monkeypatch) -> List[float]:
    """Replace ``time.sleep`` on the pipeline_tools module; record the waits."""
    waits: List[float] = []

    def _fake_sleep(secs: float = 0.0, *a: Any, **k: Any) -> None:
        waits.append(secs)

    monkeypatch.setattr(_pt.time, "sleep", _fake_sleep)
    return waits


def _install_scripted_route(
    monkeypatch, handler: Callable[[Block, int], Block],
) -> List[str]:
    """Replace ``CourseforgeRouter.route_with_self_consistency`` with a stub.

    ``handler(block, attempt_index)`` is invoked with the 0-based retry index
    for that block; it either returns a ``Block`` or raises. The returned list
    records the block_id of every dispatch attempt (retries included).
    """
    from Courseforge.router import router as _router_mod

    calls: List[str] = []

    def _scripted(self, block: Block, **kwargs: Any) -> Block:
        attempt_index = calls.count(block.block_id)
        calls.append(block.block_id)
        return handler(block, attempt_index)

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter,
        "route_with_self_consistency",
        _scripted,
    )
    return calls


def _run(project_id: str, **kwargs: Any) -> Dict[str, Any]:
    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
        **kwargs,
    ))
    return json.loads(result)


def _read_blocks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks_path = Path(payload["blocks_outline_path"])
    return [
        json.loads(ln)
        for ln in blocks_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


# ---------------------------------------------------------------------- #
# Unit tests — classification + breaker resolver
# ---------------------------------------------------------------------- #


def test_resolve_breaker_default_and_overrides(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_OUTLINE_DISPATCH_BREAKER", raising=False)
    assert _pt._resolve_outline_dispatch_breaker() == 5
    monkeypatch.setenv("COURSEFORGE_OUTLINE_DISPATCH_BREAKER", "0")
    assert _pt._resolve_outline_dispatch_breaker() == 0  # disabled
    monkeypatch.setenv("COURSEFORGE_OUTLINE_DISPATCH_BREAKER", "3")
    assert _pt._resolve_outline_dispatch_breaker() == 3
    # Parse-with-fallback: garbage / negative → default.
    monkeypatch.setenv("COURSEFORGE_OUTLINE_DISPATCH_BREAKER", "banana")
    assert _pt._resolve_outline_dispatch_breaker() == 5
    monkeypatch.setenv("COURSEFORGE_OUTLINE_DISPATCH_BREAKER", "-4")
    assert _pt._resolve_outline_dispatch_breaker() == 5


def test_transient_classification():
    assert _pt._outline_dispatch_is_transient(
        ConnectionError("connection refused by endpoint")
    ) is True
    assert _pt._outline_dispatch_is_transient(
        TimeoutError("request timed out")
    ) is True
    assert _pt._outline_dispatch_is_transient(
        RuntimeError("429 too many requests")
    ) is True
    # Permanent — not retried.
    assert _pt._outline_dispatch_is_transient(
        ValueError("invalid schema: malformed field")
    ) is False
    assert _pt._outline_dispatch_is_transient(None) is False


# ---------------------------------------------------------------------- #
# (a) transient error that recovers on retry → block authored, no marker
# ---------------------------------------------------------------------- #


def test_transient_error_recovers_on_retry(tmp_path, monkeypatch):
    project_id = "TEST_DR_TRANSIENT_RECOVER"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    waits = _no_sleep(monkeypatch)

    def handler(block: Block, attempt_index: int) -> Block:
        if attempt_index == 0:
            raise ConnectionError("connection reset by peer")
        # Second attempt succeeds.
        return dataclasses.replace(block, content="recovered prose body")

    calls = _install_scripted_route(monkeypatch, handler)

    payload = _run(project_id)
    assert payload["success"] is True, payload

    blocks = _read_blocks(payload)
    assert blocks, "expected at least one outline block"
    # No block carries a dispatch-error marker — every block recovered.
    assert all(
        b.get("escalation_marker") != "outline_dispatch_error" for b in blocks
    ), [b.get("escalation_marker") for b in blocks]
    # The retry fired (exactly one backoff wait per block).
    assert waits, "expected at least one retry backoff sleep"
    # Each dispatched block was attempted twice (fail then success).
    from collections import Counter
    per_block = Counter(calls)
    assert all(n == 2 for n in per_block.values()), per_block


# ---------------------------------------------------------------------- #
# (b) persistent transient error → outline_dispatch_error after bounded retries
# ---------------------------------------------------------------------- #


def test_persistent_transient_stamps_dispatch_error(tmp_path, monkeypatch):
    project_id = "TEST_DR_PERSISTENT"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    waits = _no_sleep(monkeypatch)
    # Disable the breaker so we can observe the marker on every block
    # (chew-through) rather than a mid-course halt.
    monkeypatch.setenv("COURSEFORGE_OUTLINE_DISPATCH_BREAKER", "0")

    def handler(block: Block, attempt_index: int) -> Block:
        raise ConnectionError("connection refused — endpoint down")

    calls = _install_scripted_route(monkeypatch, handler)

    payload = _run(project_id)
    assert payload["success"] is True, payload  # breaker disabled → completes

    blocks = _read_blocks(payload)
    dispatched = [b for b in blocks if b.get("escalation_marker")]
    assert dispatched, "expected at least one dispatched (marker-bearing) block"
    # The CORRECTED marker is used — never the semantic budget marker.
    assert all(
        b.get("escalation_marker") == "outline_dispatch_error"
        for b in dispatched
    ), [b.get("escalation_marker") for b in dispatched]
    assert not any(
        b.get("escalation_marker") == "outline_budget_exhausted"
        for b in dispatched
    )
    # Bounded retry ladder: 3 retries per block ⇒ 4 attempts, 3 sleeps each.
    from collections import Counter
    per_block = Counter(calls)
    assert all(n == 4 for n in per_block.values()), per_block
    assert waits, "expected retry backoff sleeps"


# ---------------------------------------------------------------------- #
# (c) breaker trips after N consecutive failures → phase halts loudly,
#     previously-authored blocks stay checkpointed
# ---------------------------------------------------------------------- #


def test_breaker_trips_and_preserves_checkpoint(tmp_path, monkeypatch):
    project_id = "TEST_DR_BREAKER"
    project_path = _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    _no_sleep(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_DISPATCH_BREAKER", "2")

    state = {"first": None}

    def handler(block: Block, attempt_index: int) -> Block:
        # First distinct block authors cleanly (and checkpoints); every
        # subsequent block fails persistently so the breaker trips.
        if state["first"] is None:
            state["first"] = block.block_id
        if block.block_id == state["first"]:
            return dataclasses.replace(block, content="authored ok body")
        raise ConnectionError("connection refused — endpoint down")

    _install_scripted_route(monkeypatch, handler)

    payload = _run(project_id)
    # Loud phase failure envelope (executor maps success=False → FAILED).
    assert payload["success"] is False, payload
    assert payload.get("error_code") == "OUTLINE_DISPATCH_CIRCUIT_BREAKER", payload
    assert "circuit breaker tripped" in payload.get("error", "")

    # The already-authored block stays checkpointed → a plain --resume
    # rehydrates it and only re-attempts the failed / never-reached blocks.
    sidecar = project_path / "01_outline" / _pt._OUTLINE_CHECKPOINT_NAME
    assert sidecar.exists(), "checkpoint sidecar must survive the breaker halt"
    lines = [
        json.loads(ln)
        for ln in sidecar.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(lines) >= 1, "expected the successfully-authored block checkpointed"
    assert all(
        entry.get("escalation_marker") in (None, "")
        for entry in lines
    ), "marker-bearing blocks must never be checkpointed"


# ---------------------------------------------------------------------- #
# (d) breaker disabled (=0) → legacy chew-through with the CORRECTED marker
# ---------------------------------------------------------------------- #


def test_breaker_disabled_chews_through_with_corrected_marker(
    tmp_path, monkeypatch,
):
    project_id = "TEST_DR_BREAKER_OFF"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    _no_sleep(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_DISPATCH_BREAKER", "0")

    def handler(block: Block, attempt_index: int) -> Block:
        raise ConnectionError("connection refused — endpoint down")

    _install_scripted_route(monkeypatch, handler)

    payload = _run(project_id)
    # No breaker → the run completes (legacy chew-through), but every stamped
    # block carries the CORRECTED dispatch marker.
    assert payload["success"] is True, payload
    blocks = _read_blocks(payload)
    dispatched = [b for b in blocks if b.get("escalation_marker")]
    assert dispatched, "expected marker-bearing blocks"
    assert all(
        b.get("escalation_marker") == "outline_dispatch_error"
        for b in dispatched
    )
    assert not any(
        b.get("escalation_marker") == "outline_budget_exhausted"
        for b in dispatched
    )


# ---------------------------------------------------------------------- #
# (e) semantic budget exhaustion (RETURNED marker, no exception) is untouched
# ---------------------------------------------------------------------- #


def test_semantic_budget_exhaustion_marker_untouched(tmp_path, monkeypatch):
    project_id = "TEST_DR_SEMANTIC"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    _no_sleep(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_DISPATCH_BREAKER", "2")

    def handler(block: Block, attempt_index: int) -> Block:
        # The router RETURNS a semantically-escalated block — no exception.
        # This is genuine regen-budget exhaustion, not a dispatch failure.
        return dataclasses.replace(
            block, escalation_marker="outline_budget_exhausted",
        )

    _install_scripted_route(monkeypatch, handler)

    payload = _run(project_id)
    # A returned marker block means the endpoint is alive → no breaker trip.
    assert payload["success"] is True, payload
    blocks = _read_blocks(payload)
    escalated = [b for b in blocks if b.get("escalation_marker")]
    assert escalated, "expected escalated blocks"
    # The SEMANTIC marker is preserved verbatim; the dispatch path never fires.
    assert all(
        b.get("escalation_marker") == "outline_budget_exhausted"
        for b in escalated
    )
    assert not any(
        b.get("escalation_marker") == "outline_dispatch_error" for b in blocks
    )


# ---------------------------------------------------------------------- #
# Consumer-set parity — outline_dispatch_error resolves rewrite-tier context
# ---------------------------------------------------------------------- #


def test_rewrite_marker_context_covers_dispatch_error():
    from Courseforge.generators.rewrite._rewrite_provider import (
        _ESCALATION_MARKER_CONTEXT,
    )
    assert "outline_dispatch_error" in _ESCALATION_MARKER_CONTEXT
    # Non-empty, from-scratch guidance (symmetric with the budget marker).
    assert _ESCALATION_MARKER_CONTEXT["outline_dispatch_error"].strip()


def test_outline_dispatch_error_is_canonical_marker():
    from Courseforge.scripts.blocks import _ESCALATION_MARKERS
    assert "outline_dispatch_error" in _ESCALATION_MARKERS
