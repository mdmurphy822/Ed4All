"""Graceful-stop + resume-sidecar invariants for the sub-objective derivation.

The per-CO resume sidecar (``sub_objectives.populate_sub_objectives``) must
checkpoint every completed CO and raise ``GracefulStopRequested`` at the next CO
boundary — with the stop check at the loop top, BEFORE the per-CO ``except``
guard, so a graceful stop is never swallowed. Invariant: **sidecar records ==
provider calls** for the LLM arm (stop after N → N records + N calls; resume →
total-N calls, byte-equivalent COs; pre-armed → 0).

Full canonical invariant set + the three stop legs, hermetic (fake LLM arm via
the class-level ``chat_completion`` monkeypatch the sibling ``test_sub_objectives``
suite uses; no network, no course slugs). Sentinel isolation via the top-level
``state_runs_isolated`` fixture + a synthetic ``ED4ALL_RUN_ID``.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402
from lib.objectives import sub_objectives  # noqa: E402
from lib.objectives.sub_objectives import populate_sub_objectives  # noqa: E402

_RUN_ID = "STOP_SUBOBJ_TESTRUN"


class _Capture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def _cos() -> List[Dict[str, Any]]:
    """Four flat COs, each grounded on its own single chunk (1 LLM call each)."""
    return [{
        "id": f"CO-0{i}",
        "statement": f"Objective number {i} covering distinct topic {i}",
        "source_chunk_ids": [f"k{i}"],
        "sub_objectives": [],
    } for i in range(1, 5)]


def _chunks() -> Dict[str, Any]:
    return {f"k{i}": {
        "id": f"k{i}",
        "chapter_id": "ch1",
        "text": f"Body text for chunk {i} describing topic {i} in detail. " * 4,
    } for i in range(1, 5)}


def _patch(monkeypatch, counter: Dict[str, int], *, arm_after: Optional[int] = None):
    """Patch chat_completion: count calls, return a grounded sub-objective."""

    def _fake_chat(self, messages, *, max_tokens=0, temperature=0.0, **kw):
        counter["calls"] = counter.get("calls", 0) + 1
        user = next(m for m in messages if m["role"] == "user")
        payload = json.loads(user["content"].split("INPUT:\n", 1)[1])
        src = payload.get("source_chunks") or []
        cid = src[0]["chunk_id"] if src else None
        if arm_after is not None and counter["calls"] == arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        subs = (
            [{"statement": f"Concept grounded in {cid}", "source_chunk_ids": [cid]}]
            if cid else []
        )
        return json.dumps({"sub_objectives": subs})

    import Trainforge.generators._openai_compatible_client as _oac

    monkeypatch.setattr(_oac.OpenAICompatibleClient, "chat_completion", _fake_chat)


def _records(path: Path) -> int:
    return len(sub_objectives._sub_objectives_store(path).load())


def _run(cos, sidecar: Path):
    return populate_sub_objectives(
        chapter_objectives=cos,
        chunks_by_id=_chunks(),
        embedder=None,  # explicit → hermetic (LLM arm needs no embedder)
        provider="nvidia",
        client=object(),
        capture=_Capture(),
        checkpoint_path=sidecar,
    )


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_SUB_OBJECTIVE_MODEL", raising=False)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


# --------------------------------------------------------------------------- #
# Leg 1 — stop after N → sidecar exactly N + provider exactly N
# --------------------------------------------------------------------------- #
def test_subobj_stop_after_n_exact(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / sub_objectives.SUB_OBJECTIVES_CHECKPOINT_NAME
    counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, counter, arm_after=2)

    with pytest.raises(GracefulStopRequested):
        _run(_cos(), sidecar)

    assert counter["calls"] == 2
    assert _records(sidecar) == 2


# --------------------------------------------------------------------------- #
# Leg 2 — resume → total-N calls, byte-equivalent COs
# --------------------------------------------------------------------------- #
def test_subobj_resume_after_stop_byte_equivalent(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / sub_objectives.SUB_OBJECTIVES_CHECKPOINT_NAME

    base = _cos()
    base_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, base_counter)
    _run(base, sidecar)
    n_total = base_counter["calls"]
    assert n_total == 4
    assert not sidecar.exists()

    interrupted = _cos()
    int_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, int_counter, arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _run(interrupted, sidecar)
    assert int_counter["calls"] == 2
    assert _records(sidecar) == 2

    stop_control.clear_stop(include_global=True)
    resume = _cos()
    res_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, res_counter)
    _run(resume, sidecar)
    assert res_counter["calls"] == n_total - 2
    assert int_counter["calls"] + res_counter["calls"] == n_total
    assert resume == base  # populated sub_objectives identical to the oracle
    assert not sidecar.exists()


# --------------------------------------------------------------------------- #
# Leg 3 — pre-armed sentinel → 0 calls, no sidecar
# --------------------------------------------------------------------------- #
def test_subobj_pre_armed_zero_calls(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / sub_objectives.SUB_OBJECTIVES_CHECKPOINT_NAME
    counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, counter)
    cos = _cos()
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        _run(cos, sidecar)

    assert counter["calls"] == 0
    assert not sidecar.exists()
    assert all(c["sub_objectives"] == [] for c in cos)  # untouched


# --------------------------------------------------------------------------- #
# Fingerprint mismatch → the stale records re-run
# --------------------------------------------------------------------------- #
def test_subobj_fingerprint_mismatch_reruns(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / sub_objectives.SUB_OBJECTIVES_CHECKPOINT_NAME

    monkeypatch.setenv("ED4ALL_SUB_OBJECTIVE_MODEL", "model-a")
    int_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, int_counter, arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _run(_cos(), sidecar)
    assert _records(sidecar) == 2

    stop_control.clear_stop(include_global=True)
    monkeypatch.setenv("ED4ALL_SUB_OBJECTIVE_MODEL", "model-b")
    res_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, res_counter)
    _run(_cos(), sidecar)
    assert res_counter["calls"] == 4  # not 2 — stale (model-a) records ignored


# --------------------------------------------------------------------------- #
# Torn trailing line tolerated on resume
# --------------------------------------------------------------------------- #
def test_subobj_torn_trailing_line_tolerated(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / sub_objectives.SUB_OBJECTIVES_CHECKPOINT_NAME

    int_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, int_counter, arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _run(_cos(), sidecar)
    assert _records(sidecar) == 2

    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "unit_id": "CO-03", "fingerp')

    stop_control.clear_stop(include_global=True)
    res_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, res_counter)
    _run(_cos(), sidecar)
    assert res_counter["calls"] == 2  # 2 intact records reused; torn line skipped


# --------------------------------------------------------------------------- #
# Family-flag opt-out → no sidecar, no reuse
# --------------------------------------------------------------------------- #
def test_subobj_family_flag_opt_out(tmp_path, monkeypatch, _armed_env):
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    sidecar = tmp_path / sub_objectives.SUB_OBJECTIVES_CHECKPOINT_NAME

    c1: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, c1)
    _run(_cos(), sidecar)
    assert c1["calls"] == 4
    assert not sidecar.exists()

    c2: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, c2)
    _run(_cos(), sidecar)
    assert c2["calls"] == 4  # no sidecar → no reuse
