"""Graceful-stop + resume-sidecar invariants for the bloom-complement pass.

The per-LLM-call resume sidecar (``bloom_complement.complement_bloom_profile``)
must checkpoint every completed complement call and raise
``GracefulStopRequested`` at the next call boundary — never losing a completed
call's candidates. Invariant: **sidecar records == provider calls** (stop after
N → N records + N calls; resume → total-N calls, byte-equivalent canonical;
pre-armed → 0).

Full canonical invariant set + the three stop legs, hermetic (fake
``_dispatch_call`` provider, FakeEmbed; no network, no course slugs). Sentinel
isolation via the top-level ``state_runs_isolated`` fixture + a synthetic
``ED4ALL_RUN_ID``.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402
from lib.objectives import bloom_complement  # noqa: E402
from lib.objectives.bloom_complement import complement_bloom_profile  # noqa: E402
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402

_RUN_ID = "STOP_BLOOM_TESTRUN"

# Distinct analyze statements (all detect as analyze; mutual cosine ~0.14 <<
# the 0.88 dedup floor; distinct verb+object → distinct skill signatures) so
# every call's candidate is accepted (one complement added per call).
_STMTS = [
    "Analyze numerator scaling within magnitude comparison relationships",
    "Examine denominator selection across equivalent representation strategies",
    "Differentiate proper improper quantities using benchmark reasoning",
    "Categorize operation preconditions among mixed rational forms",
    "Contrast regrouping procedures between borrowing conversion methods",
    "Investigate error patterns during simplification factor cancellation",
]

_AVOID_RE = re.compile(r"^  - (?!\[)(.+)$", re.M)


class _Capture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


class _ArmProvider:
    """Provider stub whose complement is keyed to the canonical STATE.

    The returned statement's index = (# avoid statements in the prompt) - 1 =
    the count of complements already present. This makes the response
    deterministic given the canonical set (not the provider's own call counter),
    so a fresh resume provider replaying the SAME loop positions returns the SAME
    statements as the uninterrupted oracle.
    """

    def __init__(self, *, model: str = "qwen", arm_after: Optional[int] = None) -> None:
        self._provider = "local"
        self._model = model
        self.calls = 0
        self._arm_after = arm_after

    def _dispatch_call(self, prompt, *, extra_payload=None):  # noqa: ANN001
        self.calls += 1
        avoid = _AVOID_RE.findall(prompt)
        idx = max(0, len(avoid) - 1) % len(_STMTS)
        stmt = _STMTS[idx]
        if self._arm_after is not None and self.calls == self._arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        payload = {
            "complement_objectives": [{
                "statement": stmt,
                "bloom_level": "analyze",
                "bloom_verb": "analyze",
                "source_chunk_ids": ["c1"],
            }]
        }
        return json.dumps(payload), 0


def _canonical() -> List[Dict[str, Any]]:
    return [{
        "statement": "Simplify fractions using common factors",
        "bloom_level": "apply",
        "source_chunk_ids": ["c1"],
        "chapter_id": "ch1",
    }]


def _chunks() -> Dict[str, Any]:
    return {"c1": {
        "id": "c1",
        "chapter_id": "ch1",
        "chunk_type": "explanation",
        "text": (
            "A fraction represents a part of a whole; its magnitude depends on "
            "numerator and denominator, a foundational rational-number idea. " * 4
        ),
    }}


def _records(path: Path) -> int:
    return len(bloom_complement._bloom_complement_store(path).load())


def _run(canonical, provider, sidecar: Path, *, max_add: int = 6):
    return complement_bloom_profile(
        canonical=canonical,
        chunks_by_id=_chunks(),
        provider=provider,
        course_name="ALG",
        capture=_Capture(),
        embed=FakeEmbed(),
        enabled=True,
        min_share=0.99,
        max_additions=max_add,
        checkpoint_path=sidecar,
    )


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


# --------------------------------------------------------------------------- #
# Leg 1 — stop after N → sidecar exactly N + provider exactly N
# --------------------------------------------------------------------------- #
def test_bloom_stop_after_n_exact(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / bloom_complement.BLOOM_COMPLEMENT_CHECKPOINT_NAME
    canonical = _canonical()
    provider = _ArmProvider(arm_after=2)

    with pytest.raises(GracefulStopRequested):
        _run(canonical, provider, sidecar)

    assert provider.calls == 2
    assert _records(sidecar) == 2


# --------------------------------------------------------------------------- #
# Leg 2 — resume → total-N calls, byte-equivalent canonical
# --------------------------------------------------------------------------- #
def test_bloom_resume_after_stop_byte_equivalent(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / bloom_complement.BLOOM_COMPLEMENT_CHECKPOINT_NAME

    base = _canonical()
    base_prov = _ArmProvider()
    _run(base, base_prov, sidecar)
    n_total = base_prov.calls
    assert n_total == 6
    assert not sidecar.exists()  # removed on normal completion

    interrupted = _canonical()
    int_prov = _ArmProvider(arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _run(interrupted, int_prov, sidecar)
    assert int_prov.calls == 2
    assert _records(sidecar) == 2

    stop_control.clear_stop(include_global=True)
    resume = _canonical()
    res_prov = _ArmProvider()
    _run(resume, res_prov, sidecar)
    assert res_prov.calls == n_total - 2
    assert int_prov.calls + res_prov.calls == n_total
    assert resume == base  # appended complements identical to the oracle
    assert not sidecar.exists()


# --------------------------------------------------------------------------- #
# Leg 3 — pre-armed sentinel → 0 calls, no sidecar
# --------------------------------------------------------------------------- #
def test_bloom_pre_armed_zero_calls(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / bloom_complement.BLOOM_COMPLEMENT_CHECKPOINT_NAME
    canonical = _canonical()
    provider = _ArmProvider()
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        _run(canonical, provider, sidecar)

    assert provider.calls == 0
    assert not sidecar.exists()
    assert not any(c.get("bloom_complement") for c in canonical)


# --------------------------------------------------------------------------- #
# Fingerprint mismatch → the stale records re-run
# --------------------------------------------------------------------------- #
def test_bloom_fingerprint_mismatch_reruns(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / bloom_complement.BLOOM_COMPLEMENT_CHECKPOINT_NAME

    interrupted = _canonical()
    int_prov = _ArmProvider(model="model-a", arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _run(interrupted, int_prov, sidecar)
    assert _records(sidecar) == 2

    # A different provider model flips every call fingerprint → full re-run.
    stop_control.clear_stop(include_global=True)
    resume = _canonical()
    res_prov = _ArmProvider(model="model-b")
    _run(resume, res_prov, sidecar)
    assert res_prov.calls == 6  # not 4 — stale (model-a) records ignored


# --------------------------------------------------------------------------- #
# Torn trailing line tolerated on resume
# --------------------------------------------------------------------------- #
def test_bloom_torn_trailing_line_tolerated(tmp_path, monkeypatch, _armed_env):
    sidecar = tmp_path / bloom_complement.BLOOM_COMPLEMENT_CHECKPOINT_NAME

    interrupted = _canonical()
    int_prov = _ArmProvider(arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _run(interrupted, int_prov, sidecar)
    assert _records(sidecar) == 2

    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "unit_id": "call#2", "fingerpr')

    stop_control.clear_stop(include_global=True)
    resume = _canonical()
    res_prov = _ArmProvider()
    _run(resume, res_prov, sidecar)
    assert res_prov.calls == 4  # 2 intact records reused; torn line skipped


# --------------------------------------------------------------------------- #
# Family-flag opt-out → no sidecar, no reuse
# --------------------------------------------------------------------------- #
def test_bloom_family_flag_opt_out(tmp_path, monkeypatch, _armed_env):
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    sidecar = tmp_path / bloom_complement.BLOOM_COMPLEMENT_CHECKPOINT_NAME

    c1 = _canonical()
    p1 = _ArmProvider()
    _run(c1, p1, sidecar)
    assert p1.calls == 6
    assert not sidecar.exists()

    c2 = _canonical()
    p2 = _ArmProvider()
    _run(c2, p2, sidecar)
    assert p2.calls == 6  # no sidecar → no reuse
