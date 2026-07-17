"""``score_candidate`` NLI-micro-batch routing (``ED4ALL_NLI_MICROBATCH``).

Asserts the router call path in ``Courseforge/router/router.py::score_candidate``:

* flag OFF (default) → the dispatcher is NEVER constructed; the legacy
  ``_NLI_SCORE_LOCK`` path runs and returns the same verdict.
* flag ON → the dispatcher is constructed for the resolved NLI and the verdict
  is routed through it (byte-equivalent grounding result).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from Courseforge.router.router import score_candidate  # noqa: E402
from lib.classifiers import nli_microbatch as mb  # noqa: E402
from blocks import Block  # noqa: E402


class _FakeScore:
    def __init__(self, ent: float, con: float = 0.0) -> None:
        self.entailment = ent
        self.neutral = max(0.0, 1.0 - ent - con)
        self.contradiction = con


class _ScriptedNLI:
    """HIGH/LOW/FABRICATED token-driven fake (mirrors the best-of-N test)."""

    _revision = "fake-nli"
    device = "cpu"

    def score_batch(self, *, pairs: List[Tuple[str, str]], batch_size=None):
        out = []
        for _premise, hyp in pairs:
            h = str(hyp)
            if "FABRICATED" in h:
                out.append(_FakeScore(0.05, con=0.95))
            elif "HIGH" in h:
                out.append(_FakeScore(0.95))
            else:
                out.append(_FakeScore(0.20))
        return out


_HIGH = "This HIGH supported sentence matches the cited reference material closely."
_SOURCE_CHUNKS = [{"id": "c1", "text": "Reference text describing the concept."}]
_OBJECTIVES = [{"id": "CO-01", "statement": "Understand the concept."}]


def _candidate(prose: str) -> Block:
    return Block(
        block_id="page1#concept_intro_0",
        block_type="concept",
        page_id="page1",
        sequence=0,
        content=prose,
        objective_ids=("CO-01",),
        source_ids=("c1",),
    )


@pytest.fixture(autouse=True)
def _reset_registry():
    mb._reset_for_tests()
    yield
    mb._reset_for_tests()


def test_flag_off_dispatcher_never_constructed(monkeypatch):
    monkeypatch.delenv("ED4ALL_NLI_MICROBATCH", raising=False)
    calls = {"n": 0}
    real_get = mb.get_dispatcher

    def _spy(nli):
        calls["n"] += 1
        return real_get(nli)

    monkeypatch.setattr(mb, "get_dispatcher", _spy)

    nli = _ScriptedNLI()
    verdict = score_candidate(_candidate(_HIGH), _SOURCE_CHUNKS, _OBJECTIVES, nli=nli)

    assert calls["n"] == 0, "dispatcher was constructed with the flag off"
    assert not mb._REGISTRY, "registry populated with the flag off"
    # Legacy locked path still produces a grounded verdict.
    assert verdict.entailment_rate == pytest.approx(1.0)
    assert verdict.contradicted is False


def test_flag_on_routes_through_dispatcher(monkeypatch):
    monkeypatch.setenv("ED4ALL_NLI_MICROBATCH", "1")
    nli = _ScriptedNLI()
    verdict = score_candidate(_candidate(_HIGH), _SOURCE_CHUNKS, _OBJECTIVES, nli=nli)

    # A dispatcher was constructed for this NLI and cached in the registry.
    assert id(nli) in mb._REGISTRY
    # Same grounded verdict as the locked path.
    assert verdict.entailment_rate == pytest.approx(1.0)
    assert verdict.contradicted is False


def test_flag_on_contradiction_still_detected(monkeypatch):
    monkeypatch.setenv("ED4ALL_NLI_MICROBATCH", "1")
    nli = _ScriptedNLI()
    prose = "This FABRICATED sentence conflicts with the cited reference material."
    verdict = score_candidate(_candidate(prose), _SOURCE_CHUNKS, _OBJECTIVES, nli=nli)
    assert verdict.contradicted is True
