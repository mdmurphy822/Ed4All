"""COURSEFORGE_REWRITE_CONCURRENCY opt-in rewrite-tier concurrency tests.

Exercises ``MCP/tools/pipeline_tools.py::_run_content_generation_rewrite``
with a FAKE rewrite path (no real NVIDIA / LLM call) to prove:

1. With ``COURSEFORGE_REWRITE_CONCURRENCY=4`` and N blocks, all blocks are
   rewritten and the output order is IDENTICAL to the sequential path (the
   concurrent ``blocks_final.jsonl`` is byte-for-byte equal to the sequential
   one for the same inputs → deterministic).
2. Decision captures still fire under concurrency (no lost/corrupted events).
3. A per-block dispatch failure is isolated (siblings still complete; the
   failed block is stamped ``validator_consensus_fail``).
4. Unset / ``1`` is the EXACT sequential path — no ``ThreadPoolExecutor`` is
   constructed.

These tests never touch a real model: ``route_rewrite_with_remediation`` is
monkeypatched to a deterministic in-memory fake that records the worker
thread id per call.
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

# Force the CPU NLI device + small local model so no test path can reach a
# heavyweight GPU load even if best-of-N were somehow enabled (it is not here).
pytestmark = pytest.mark.usefixtures("_pin_cpu_env")


@pytest.fixture()
def _pin_cpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ED4ALL_NLI_DEVICE", "cpu")
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:7b-instruct-q4_K_M")


def _make_block(block_id: str, page_id: str, seq: int) -> dict:
    """Minimal outline-tier block entry (dict content carrying objective_ids)."""
    return {
        "block_id": block_id,
        "block_type": "objective",
        "page_id": page_id,
        "sequence": seq,
        "content": {"objective_ids": ["CO-01"], "body": f"outline {block_id}"},
        "objective_ids": ["CO-01"],
    }


def _write_validated(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            entry = _make_block(f"blk{i:03d}", f"week_01_content_01", i)
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _fake_rewrite_factory(
    *,
    fail_block_id: str | None = None,
    record: List[Any] | None = None,
    delay: float = 0.0,
):
    """Build a fake ``route_rewrite_with_remediation`` bound method.

    Deterministic: the rewritten str content is a pure function of the input
    block_id, so the output is identical regardless of dispatch order. Records
    ``(thread_id, block_id)`` per call so a test can prove real overlap.
    """

    def _fake(self, block, *, validators=None, source_chunks=None,
              objectives=None, **kw):  # type: ignore[no-untyped-def]
        if record is not None:
            record.append((threading.get_ident(), block.block_id))
        if delay:
            time.sleep(delay)
        if fail_block_id is not None and block.block_id == fail_block_id:
            raise RuntimeError(f"injected failure for {block.block_id}")
        # Deterministic str body derived purely from the block id.
        html = (
            f'<section data-cf-objective-id="CO-01">'
            f"<p>rewritten body for {block.block_id}</p></section>"
        )
        return dataclasses.replace(block, content=html)

    return _fake


def _setup_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                   n_blocks: int) -> Dict[str, Any]:
    """Redirect exports into tmp_path and write the validated-blocks input."""
    # PROJECT_ROOT is the canonical phase-handler test seam (see
    # courseforge_exports_dir()).
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)

    project_id = "TESTPROJ"
    project_path = tmp_path / "Courseforge" / "exports" / project_id
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": "FXTEST_101"}), encoding="utf-8",
    )
    validated = project_path / "blocks_validated.jsonl"
    _write_validated(validated, n_blocks)
    return {
        "project_id": project_id,
        "blocks_validated_path": str(validated),
        "project_path": project_path,
    }


def _run(kwargs: Dict[str, Any]) -> dict:
    out = asyncio.run(pipeline_tools._run_content_generation_rewrite(**kwargs))
    return json.loads(out)


def _read_final(project_path: Path) -> str:
    return (project_path / "04_rewrite" / "blocks_final.jsonl").read_text(
        encoding="utf-8"
    )


def test_concurrent_output_identical_to_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrency=4 produces blocks_final byte-identical to sequential."""
    monkeypatch.setattr(
        CourseforgeRouter, "route_rewrite_with_remediation",
        _fake_rewrite_factory(),
    )

    # --- Sequential run (concurrency unset = 1) ---
    seq_setup = _setup_project(monkeypatch, tmp_path / "seq", 12)
    monkeypatch.delenv("COURSEFORGE_REWRITE_CONCURRENCY", raising=False)
    seq_res = _run({
        "project_id": seq_setup["project_id"],
        "blocks_validated_path": seq_setup["blocks_validated_path"],
    })
    assert seq_res.get("success") is not False
    seq_final = _read_final(seq_setup["project_path"])

    # --- Concurrent run (concurrency=4) ---
    conc_setup = _setup_project(monkeypatch, tmp_path / "conc", 12)
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", "4")
    conc_res = _run({
        "project_id": conc_setup["project_id"],
        "blocks_validated_path": conc_setup["blocks_validated_path"],
    })
    assert conc_res.get("success") is not False
    conc_final = _read_final(conc_setup["project_path"])

    # Byte-identical → deterministic ordering regardless of completion order.
    assert conc_final == seq_final
    # All 12 blocks present in original order.
    ids = [json.loads(line)["block_id"]
           for line in conc_final.splitlines() if line.strip()]
    assert ids == [f"blk{i:03d}" for i in range(12)]


def test_concurrent_real_overlap_and_all_rewritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent dispatch genuinely overlaps (>1 worker thread) + all done."""
    record: List[Any] = []
    monkeypatch.setattr(
        CourseforgeRouter, "route_rewrite_with_remediation",
        _fake_rewrite_factory(record=record, delay=0.05),
    )
    setup = _setup_project(monkeypatch, tmp_path / "ov", 8)
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", "4")
    res = _run({
        "project_id": setup["project_id"],
        "blocks_validated_path": setup["blocks_validated_path"],
    })
    assert res.get("success") is not False
    # Every block dispatched exactly once.
    assert len(record) == 8
    assert {bid for _tid, bid in record} == {f"blk{i:03d}" for i in range(8)}
    # More than one worker thread was actually used (real concurrency).
    assert len({tid for tid, _bid in record}) > 1


def test_per_block_failure_isolated_under_concurrency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One block's dispatch error doesn't abort the others (fail-soft)."""
    monkeypatch.setattr(
        CourseforgeRouter, "route_rewrite_with_remediation",
        _fake_rewrite_factory(fail_block_id="blk003"),
    )
    setup = _setup_project(monkeypatch, tmp_path / "fail", 8)
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", "4")
    res = _run({
        "project_id": setup["project_id"],
        "blocks_validated_path": setup["blocks_validated_path"],
    })
    assert res.get("success") is not False
    final = _read_final(setup["project_path"])
    entries = {json.loads(line)["block_id"]: json.loads(line)
               for line in final.splitlines() if line.strip()}
    # All 8 blocks still present in original order.
    assert list(entries.keys()) == [f"blk{i:03d}" for i in range(8)]
    # The failed block carries the consensus-fail escalation marker.
    assert entries["blk003"].get("escalation_marker") == "validator_consensus_fail"
    # A sibling completed normally (real rewritten body).
    assert "rewritten body for blk004" in entries["blk004"].get("content", "")


def test_failure_isolation_matches_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent + sequential agree on a per-block failure outcome."""
    monkeypatch.setattr(
        CourseforgeRouter, "route_rewrite_with_remediation",
        _fake_rewrite_factory(fail_block_id="blk002"),
    )
    seq_setup = _setup_project(monkeypatch, tmp_path / "fseq", 6)
    monkeypatch.delenv("COURSEFORGE_REWRITE_CONCURRENCY", raising=False)
    _run({
        "project_id": seq_setup["project_id"],
        "blocks_validated_path": seq_setup["blocks_validated_path"],
    })
    seq_final = _read_final(seq_setup["project_path"])

    conc_setup = _setup_project(monkeypatch, tmp_path / "fconc", 6)
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", "4")
    _run({
        "project_id": conc_setup["project_id"],
        "blocks_validated_path": conc_setup["blocks_validated_path"],
    })
    conc_final = _read_final(conc_setup["project_path"])
    assert conc_final == seq_final


def test_sequential_path_constructs_no_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset / 1 must never construct a ThreadPoolExecutor."""
    monkeypatch.setattr(
        CourseforgeRouter, "route_rewrite_with_remediation",
        _fake_rewrite_factory(),
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
    setup = _setup_project(monkeypatch, tmp_path / "noenv", 5)
    monkeypatch.delenv("COURSEFORGE_REWRITE_CONCURRENCY", raising=False)
    _run({
        "project_id": setup["project_id"],
        "blocks_validated_path": setup["blocks_validated_path"],
    })
    assert constructed["count"] == 0

    # Explicit 1 → default 1, still no pool.
    setup2 = _setup_project(monkeypatch, tmp_path / "one", 5)
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", "1")
    _run({
        "project_id": setup2["project_id"],
        "blocks_validated_path": setup2["blocks_validated_path"],
    })
    assert constructed["count"] == 0

    # Garbage → falls back to 1, still no pool.
    setup3 = _setup_project(monkeypatch, tmp_path / "garbage", 5)
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", "not-an-int")
    _run({
        "project_id": setup3["project_id"],
        "blocks_validated_path": setup3["blocks_validated_path"],
    })
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
        CourseforgeRouter, "route_rewrite_with_remediation",
        _fake_rewrite_factory(),
    )
    setup = _setup_project(monkeypatch, tmp_path / "cap", 10)
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", "4")
    _run({
        "project_id": setup["project_id"],
        "blocks_validated_path": setup["blocks_validated_path"],
    })
    # The concurrency-enabled decision event fired.
    assert any(
        "COURSEFORGE_REWRITE_CONCURRENCY=4" in c["rationale"]
        for c in captured
    )
    # Every captured rationale meets the 20-char minimum (no corruption).
    assert all(len(c["rationale"]) >= 20 for c in captured)
