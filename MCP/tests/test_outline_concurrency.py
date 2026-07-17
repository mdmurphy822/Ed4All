"""COURSEFORGE_OUTLINE_CONCURRENCY opt-in outline-tier concurrency tests.

Exercises ``MCP/tools/pipeline_tools.py::_run_content_generation_outline``
with a FAKE outline path (no real LLM call) to prove:

1. With ``COURSEFORGE_OUTLINE_CONCURRENCY=4`` and N blocks, all blocks are
   authored and the output order is IDENTICAL to the sequential path (the
   concurrent ``blocks_outline.jsonl`` is byte-for-byte equal to the
   sequential one for the same inputs → deterministic).
2. Decision captures still fire under concurrency (no lost/corrupted events).
3. A per-block dispatch failure is isolated (siblings still complete; the
   failed block is stamped ``outline_dispatch_error`` — the exact sequential
   fail-soft contract).
4. Unset / ``1`` / garbage is the EXACT sequential path — no
   ``ThreadPoolExecutor`` is constructed.

These tests never touch a real model: ``route_with_self_consistency`` is
monkeypatched to a deterministic in-memory fake that records the worker
thread id per call. Structure mirrors ``MCP/tests/test_rewrite_concurrency.py``
(the reference implementation this knob copies) with the project scaffold from
``MCP/tools/tests/test_pipeline_tools_outline_dispatch_resilience.py``.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

import MCP.tools.pipeline_tools as pipeline_tools
from Courseforge.router.router import CourseforgeRouter
from Courseforge.scripts.blocks import Block

# Force the CPU NLI device + small local model so no test path can reach a
# heavyweight GPU load even if best-of-N were somehow enabled (it is not here).
pytestmark = pytest.mark.usefixtures("_pin_cpu_env")


@pytest.fixture()
def _pin_cpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cpu")
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:7b-instruct-q4_K_M")


def _seed_project(tmp_path: Path, project_id: str) -> Path:
    """Seed a minimal outline-phase project export (objectives + config)."""
    exports_root = tmp_path / "Courseforge" / "exports"
    project_path = exports_root / project_id
    (project_path / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    (project_path / "01_learning_objectives" / "synthesized_objectives.json").write_text(
        json.dumps({
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Describe core concept A in detail."},
                {"id": "TO-02", "statement": "Apply core concept B to problems."},
            ],
            "chapter_objectives": [],
        }),
        encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": project_id, "duration_weeks": 2}),
        encoding="utf-8",
    )
    return project_path


def _fake_outline_factory(
    *,
    fail_block_id: str | None = None,
    record: List[Any] | None = None,
    delay: float = 0.0,
):
    """Build a fake ``route_with_self_consistency`` bound method.

    Deterministic: the authored content is a pure function of the input
    block_id, so the output is identical regardless of dispatch order.
    Records ``(thread_id, block_id)`` per call so a test can prove real
    overlap. ``fail_block_id`` raises a PERMANENT (non-transient) error so
    the resilience path gives up after one attempt (no retry sleeps).
    """

    def _fake(self, block: Block, **kwargs: Any) -> Block:  # type: ignore[no-untyped-def]
        if record is not None:
            record.append((threading.get_ident(), block.block_id))
        if delay:
            time.sleep(delay)
        if fail_block_id is not None and block.block_id == fail_block_id:
            raise ValueError(f"invalid schema: injected failure for {block.block_id}")
        return dataclasses.replace(
            block, content=f"authored outline body for {block.block_id}"
        )

    return _fake


def _run(project_id: str, **kwargs: Any) -> Dict[str, Any]:
    result = asyncio.run(pipeline_tools._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
        **kwargs,
    ))
    return json.loads(result)


def _read_outline(payload: Dict[str, Any]) -> str:
    return Path(payload["blocks_outline_path"]).read_text(encoding="utf-8")


def _block_ids(outline_text: str) -> List[str]:
    return [
        json.loads(ln)["block_id"]
        for ln in outline_text.splitlines() if ln.strip()
    ]


def test_concurrent_output_identical_to_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrency=4 produces blocks_outline byte-identical to sequential."""
    monkeypatch.setattr(
        CourseforgeRouter, "route_with_self_consistency",
        _fake_outline_factory(),
    )

    # --- Sequential run (concurrency unset = 1) ---
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "seq")
    _seed_project(tmp_path / "seq", "TEST_OC_SEQ")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_CONCURRENCY", raising=False)
    seq_res = _run("TEST_OC_SEQ")
    assert seq_res.get("success") is True, seq_res
    seq_outline = _read_outline(seq_res)
    # Normalize the project-id difference out (only the course/project name
    # differs between the two runs; block bodies/order must match).
    seq_norm = seq_outline.replace("TEST_OC_SEQ", "PROJ")

    # --- Concurrent run (concurrency=4) ---
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "conc")
    _seed_project(tmp_path / "conc", "TEST_OC_CONC")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "4")
    conc_res = _run("TEST_OC_CONC")
    assert conc_res.get("success") is True, conc_res
    conc_outline = _read_outline(conc_res)
    conc_norm = conc_outline.replace("TEST_OC_CONC", "PROJ")

    # Byte-identical (modulo project id) → deterministic ordering regardless
    # of completion order.
    assert conc_norm == seq_norm
    # Same non-trivial block-id sequence in the original order.
    assert _block_ids(conc_outline) == _block_ids(seq_outline)
    assert len(_block_ids(conc_outline)) > 1


def test_concurrent_real_overlap_and_all_authored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent dispatch genuinely overlaps (>1 worker thread) + all done."""
    record: List[Any] = []
    monkeypatch.setattr(
        CourseforgeRouter, "route_with_self_consistency",
        _fake_outline_factory(record=record, delay=0.05),
    )
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    _seed_project(tmp_path, "TEST_OC_OVERLAP")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "4")
    res = _run("TEST_OC_OVERLAP")
    assert res.get("success") is True, res
    ids = _block_ids(_read_outline(res))
    assert len(ids) > 1
    # Every block dispatched exactly once, none dropped.
    assert sorted(bid for _tid, bid in record) == sorted(ids)
    # More than one worker thread was actually used (real concurrency).
    assert len({tid for tid, _bid in record}) > 1


def test_per_block_failure_isolated_under_concurrency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One block's dispatch error doesn't abort the others (fail-soft)."""
    # First discover the deterministic block-id universe with a clean run.
    monkeypatch.setattr(
        CourseforgeRouter, "route_with_self_consistency",
        _fake_outline_factory(),
    )
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "probe")
    _seed_project(tmp_path / "probe", "TEST_OC_FAIL")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_CONCURRENCY", raising=False)
    probe_ids = _block_ids(_read_outline(_run("TEST_OC_FAIL")))
    assert len(probe_ids) >= 3, probe_ids
    victim = probe_ids[1]

    # Now fail exactly that block under concurrency.
    monkeypatch.setattr(
        CourseforgeRouter, "route_with_self_consistency",
        _fake_outline_factory(fail_block_id=victim),
    )
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "fail")
    _seed_project(tmp_path / "fail", "TEST_OC_FAIL")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "4")
    res = _run("TEST_OC_FAIL")
    assert res.get("success") is True, res
    entries = {
        json.loads(ln)["block_id"]: json.loads(ln)
        for ln in _read_outline(res).splitlines() if ln.strip()
    }
    # All blocks still present in the original order.
    assert list(entries.keys()) == probe_ids
    # The failed block carries the canonical dispatch-error marker (exact
    # sequential fail-soft contract).
    assert entries[victim].get("escalation_marker") == "outline_dispatch_error"
    # Siblings completed normally (real authored bodies, no marker).
    survivors = [bid for bid in probe_ids if bid != victim]
    for bid in survivors:
        assert not entries[bid].get("escalation_marker")
        assert f"authored outline body for {bid}" in str(
            entries[bid].get("content", "")
        )


def test_failure_isolation_matches_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent + sequential agree on a per-block failure outcome."""
    # Discover the deterministic victim id first.
    monkeypatch.setattr(
        CourseforgeRouter, "route_with_self_consistency",
        _fake_outline_factory(),
    )
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "probe")
    _seed_project(tmp_path / "probe", "TEST_OC_FPAR")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_CONCURRENCY", raising=False)
    probe_ids = _block_ids(_read_outline(_run("TEST_OC_FPAR")))
    victim = probe_ids[0]

    monkeypatch.setattr(
        CourseforgeRouter, "route_with_self_consistency",
        _fake_outline_factory(fail_block_id=victim),
    )
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "fseq")
    _seed_project(tmp_path / "fseq", "TEST_OC_FPAR")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_CONCURRENCY", raising=False)
    seq_outline = _read_outline(_run("TEST_OC_FPAR"))

    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "fconc")
    _seed_project(tmp_path / "fconc", "TEST_OC_FPAR")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "4")
    conc_outline = _read_outline(_run("TEST_OC_FPAR"))
    assert conc_outline == seq_outline


def test_sequential_path_constructs_no_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset / 1 / garbage must never construct a ThreadPoolExecutor."""
    monkeypatch.setattr(
        CourseforgeRouter, "route_with_self_consistency",
        _fake_outline_factory(),
    )

    import concurrent.futures as _cf

    constructed = {"count": 0}
    _real_tpe = _cf.ThreadPoolExecutor

    class _SpyTPE(_real_tpe):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            constructed["count"] += 1
            super().__init__(*a, **k)

    monkeypatch.setattr(_cf, "ThreadPoolExecutor", _SpyTPE)

    # Unset → default 1.
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "noenv")
    _seed_project(tmp_path / "noenv", "TEST_OC_NOPOOL")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_CONCURRENCY", raising=False)
    assert _run("TEST_OC_NOPOOL").get("success") is True
    assert constructed["count"] == 0

    # Explicit 1 → still no pool.
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "one")
    _seed_project(tmp_path / "one", "TEST_OC_NOPOOL")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "1")
    assert _run("TEST_OC_NOPOOL").get("success") is True
    assert constructed["count"] == 0

    # Garbage → falls back to 1, still no pool.
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "garbage")
    _seed_project(tmp_path / "garbage", "TEST_OC_NOPOOL")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "not-an-int")
    assert _run("TEST_OC_NOPOOL").get("success") is True
    assert constructed["count"] == 0

    # Negative → falls back to 1, still no pool.
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path / "neg")
    _seed_project(tmp_path / "neg", "TEST_OC_NOPOOL")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "-3")
    assert _run("TEST_OC_NOPOOL").get("success") is True
    assert constructed["count"] == 0


def test_captures_fire_under_concurrency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DecisionCapture events fire (and are well-formed) under concurrency."""
    captured: List[Dict[str, Any]] = []

    import lib.decision_capture as dc_mod

    _real_log = dc_mod.DecisionCapture.log_decision

    def _spy_log(self, decision_type, decision, rationale, **kw):  # type: ignore[no-untyped-def]
        captured.append({"decision_type": decision_type, "rationale": rationale})
        return _real_log(self, decision_type, decision, rationale, **kw)

    monkeypatch.setattr(dc_mod.DecisionCapture, "log_decision", _spy_log)
    monkeypatch.setattr(
        CourseforgeRouter, "route_with_self_consistency",
        _fake_outline_factory(),
    )
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    _seed_project(tmp_path, "TEST_OC_CAP")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "4")
    res = _run("TEST_OC_CAP")
    assert res.get("success") is True, res
    # The concurrency-enabled decision event fired.
    assert any(
        "COURSEFORGE_OUTLINE_CONCURRENCY=4" in c["rationale"]
        for c in captured
    )
    # Every captured rationale meets the 20-char minimum (no corruption).
    assert all(len(c["rationale"]) >= 20 for c in captured)
