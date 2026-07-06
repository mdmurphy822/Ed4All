"""Graceful-stop + resume-sidecar invariants for the objective-review pass.

The per-review-chunk resume sidecar (``objective_review.review_objectives``)
must, at the next chunk boundary, checkpoint every completed chunk and raise
``GracefulStopRequested`` — never losing a completed review call, never leaving
a half-applied merge. The mechanical guarantee asserted throughout:
**sidecar records == provider calls** (stop after N → exactly N records + N
calls; resume → total-N calls, byte-equivalent objectives; pre-armed → 0).

Full canonical invariant set + the three stop legs, hermetic (fake provider via
the class-level ``chat_completion`` monkeypatch used by the sibling
``test_objective_review`` suite; no network, no course slugs). Sentinel
isolation via the top-level ``state_runs_isolated`` fixture + a synthetic
``ED4ALL_RUN_ID``.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402
from lib.objectives import objective_review  # noqa: E402
from lib.objectives.objective_review import review_objectives  # noqa: E402

_RUN_ID = "STOP_REVIEW_TESTRUN"


# --------------------------------------------------------------------------- #
# Deterministic embedder (token-overlap cosine — sibling suite's _Embed).
# --------------------------------------------------------------------------- #
class _Embed:
    _DIM = 512

    @staticmethod
    def _bucket(token: str) -> int:
        import hashlib

        h = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % _Embed._DIM

    def encode(self, text: str) -> List[float]:
        toks = [w.lower().strip(".,;:!?()'\"") for w in str(text).split()]
        toks = [w for w in toks if len(w) > 2]
        vec = [0.0] * self._DIM
        for w in toks:
            vec[self._bucket(w)] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# Benign per-id aligned rewrites (share the parent-TO anchoring tokens so the
# cosine guardrail accepts every statement edit — deterministic merge).
_ADJ = {
    "CO-01": {"statement": "Describe cellular respiration energy pathways clearly"},
    "CO-02": {"statement": "List cellular respiration energy pathways in order"},
    "CO-03": {"statement": "Trace cellular respiration energy pathways stepwise"},
    "CO-04": {"statement": "Outline cellular respiration energy pathways briefly"},
    "TO-01": {"statement": "Analyze cellular respiration energy pathways in depth"},
    "TO-02": {"statement": "Evaluate cellular respiration energy pathways rigorously"},
}


def _objectives() -> Dict[str, Any]:
    """Two TOs + four single-CO chapter groups → 4 CO chunks + 1 TO chunk = 5."""
    terminals = [
        {
            "id": "TO-01",
            "statement": "Analyze cellular respiration energy pathways",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
            "source_refs": [],
        },
        {
            "id": "TO-02",
            "statement": "Evaluate cellular respiration energy pathways",
            "bloom_level": "evaluate",
            "bloom_verb": "evaluate",
            "source_refs": [],
        },
    ]
    chapter_objectives = []
    for i in range(1, 5):
        cid = f"CO-0{i}"
        chapter_objectives.append({
            "chapter": f"C{i}",
            "objectives": [{
                "id": cid,
                "statement": "Recall cellular respiration energy pathways",
                "bloom_level": "understand",
                "bloom_verb": "recall",
                "terminal_id": "TO-01",
                "source_refs": [{"ref": "ch1", "chunk_ids": [f"k{i}"]}],
            }],
        })
    return {"terminals": terminals, "chapter_objectives": chapter_objectives}


def _patch(monkeypatch, counter: Dict[str, int], *, arm_after: Optional[int] = None):
    """Patch OpenAICompatibleClient.chat_completion: count calls, optional arm."""

    def _fake_chat(self, messages, *, max_tokens=0, temperature=0.0, **kw):
        counter["calls"] = counter.get("calls", 0) + 1
        user = next(m for m in messages if m["role"] == "user")
        payload = json.loads(user["content"].split("INPUT:\n", 1)[1])
        ids = [it.get("id") for it in (payload.get("review") or [])]
        if arm_after is not None and counter["calls"] == arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return json.dumps({"adjusted": {i: _ADJ[i] for i in ids if i in _ADJ}})

    import Trainforge.generators._openai_compatible_client as _oac

    monkeypatch.setattr(_oac.OpenAICompatibleClient, "chat_completion", _fake_chat)


def _records(path: Path) -> int:
    return len(objective_review._review_checkpoint_store(path).load())


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_OBJECTIVE_REVIEW_MODEL", raising=False)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


def _run(objs: Dict[str, Any], sidecar: Path):
    return review_objectives(
        terminals=objs["terminals"],
        chapter_objectives=objs["chapter_objectives"],
        course_name="BIO_101",
        embedder=_Embed(),
        client=object(),
        checkpoint_path=sidecar,
    )


# --------------------------------------------------------------------------- #
# Leg 1 — stop after N → sidecar exactly N + provider exactly N
# --------------------------------------------------------------------------- #
def test_review_stop_after_n_exact(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / "01_learning_objectives" / \
        objective_review.REVIEW_CHECKPOINT_NAME
    counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, counter, arm_after=2)

    with pytest.raises(GracefulStopRequested):
        _run(_objectives(), sidecar)

    assert counter["calls"] == 2
    assert _records(sidecar) == 2


# --------------------------------------------------------------------------- #
# Leg 2 — resume → total-N calls, byte-equivalent objectives (per-unit write)
# --------------------------------------------------------------------------- #
def test_review_resume_after_stop_byte_equivalent(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / objective_review.REVIEW_CHECKPOINT_NAME

    # Uninterrupted oracle on a fresh copy (sidecar removed on full completion).
    base = _objectives()
    base_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, base_counter)
    _run(base, sidecar)
    n_total = base_counter["calls"]
    assert n_total == 5
    assert not sidecar.exists()  # removed on fully-complete review

    # Interrupted leg: stop after 2 chunks.
    interrupted = _objectives()
    int_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, int_counter, arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _run(interrupted, sidecar)
    assert int_counter["calls"] == 2
    assert _records(sidecar) == 2

    # Resume leg: clear sentinel, rerun on a fresh copy → only the un-reviewed
    # chunks dispatch; total across legs == baseline; objectives byte-equivalent.
    stop_control.clear_stop(include_global=True)
    resume = _objectives()
    res_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, res_counter)
    _run(resume, sidecar)
    assert res_counter["calls"] == n_total - 2
    assert int_counter["calls"] + res_counter["calls"] == n_total
    assert resume == base  # in-place mutation identical to the oracle
    assert not sidecar.exists()  # cleaned up on the completed resume


# --------------------------------------------------------------------------- #
# Leg 3 — pre-armed sentinel → 0 calls, no sidecar
# --------------------------------------------------------------------------- #
def test_review_pre_armed_zero_calls(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / objective_review.REVIEW_CHECKPOINT_NAME
    counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, counter)
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        _run(_objectives(), sidecar)

    assert counter["calls"] == 0
    assert not sidecar.exists()


# --------------------------------------------------------------------------- #
# Fingerprint mismatch → the stale records re-run
# --------------------------------------------------------------------------- #
def test_review_fingerprint_mismatch_reruns(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / objective_review.REVIEW_CHECKPOINT_NAME

    # Interrupted leg with model A → 2 records fingerprinted on model A.
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_MODEL", "model-a")
    int_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, int_counter, arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _run(_objectives(), sidecar)
    assert _records(sidecar) == 2

    # Resume with model B → every fingerprint mismatches → all 5 chunks re-run.
    stop_control.clear_stop(include_global=True)
    monkeypatch.setenv("ED4ALL_OBJECTIVE_REVIEW_MODEL", "model-b")
    res_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, res_counter)
    _run(_objectives(), sidecar)
    assert res_counter["calls"] == 5  # not 3 — stale (model-a) records ignored


# --------------------------------------------------------------------------- #
# Torn trailing line tolerated on resume
# --------------------------------------------------------------------------- #
def test_review_torn_trailing_line_tolerated(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / objective_review.REVIEW_CHECKPOINT_NAME

    int_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, int_counter, arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _run(_objectives(), sidecar)
    assert _records(sidecar) == 2

    # Append a torn (half-written) trailing line — must be skipped, not crash.
    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "unit_id": "co_group:tor')

    stop_control.clear_stop(include_global=True)
    res_counter: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, res_counter)
    _run(_objectives(), sidecar)
    # The 2 intact records still reused → only the remaining 3 chunks dispatch.
    assert res_counter["calls"] == 3


# --------------------------------------------------------------------------- #
# Family-flag opt-out → no sidecar, no reuse
# --------------------------------------------------------------------------- #
def test_review_family_flag_opt_out(tmp_path, monkeypatch, _armed_env):
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    sidecar = tmp_path / objective_review.REVIEW_CHECKPOINT_NAME

    c1: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, c1)
    _run(_objectives(), sidecar)
    assert c1["calls"] == 5
    assert not sidecar.exists()  # disabled store never writes

    # A second run cannot reuse (no sidecar) → full re-run.
    c2: Dict[str, int] = {"calls": 0}
    _patch(monkeypatch, c2)
    _run(_objectives(), sidecar)
    assert c2["calls"] == 5
