"""Rewrite-chain wiring tests for the prose-stutter candidate check.

Book-1 canary keystone fix: ``ProseStutterValidator`` runs in the rewrite
router's per-candidate validator chain so a stuttered candidate fires
``action="regenerate"`` and ``route_rewrite_with_remediation`` re-rolls it
at authoring time (best-of-N until a clean candidate) instead of shipping
the slop into the retrieval corpus.

Uses the canned-provider pattern from ``test_validator_action.py`` — no
LLM dispatch anywhere; fixtures are synthetic (no course slugs).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SCRIPTS = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from Courseforge.router.router import CourseforgeRouter  # noqa: E402
from lib.validators.prose_stutter import ProseStutterValidator  # noqa: E402
from blocks import Block  # noqa: E402


_STUTTERED_HTML = (
    "<p>The pipeline consists of four stages: review, build, staging "
    "rollout, and final rollout review, build, staging rollout, and final "
    "rollout.</p>"
)
_CLEAN_HTML = (
    "<p>Tests are classified as small, intermediate, or large based on "
    "resource requirements and execution environment alone.</p>"
)


def _candidate(idx: int, *, prose: str) -> Block:
    return Block(
        block_id="page1#concept_intro_0",
        block_type="concept",
        page_id="page1",
        sequence=idx,
        content=prose,
    )


class _SequenceRewriteProvider:
    """Stub RewriteProvider returning canned Blocks in call order."""

    def __init__(self, outputs: List[Block]) -> None:
        self._outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def generate_rewrite(self, block, *, source_chunks, objectives, **kwargs):
        idx = min(len(self.calls), len(self._outputs) - 1)
        self.calls.append({
            "block": block,
            **{k: v for k, v in kwargs.items() if k == "remediation_suffix"},
        })
        return self._outputs[idx]


class _FakeCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def _router(outputs: List[Block]) -> tuple:
    provider = _SequenceRewriteProvider(outputs)
    router = CourseforgeRouter(
        rewrite_provider=provider,
        capture=_FakeCapture(),
        n_candidates=len(outputs),
    )
    return router, provider


def test_stuttered_candidate_rejected_and_rerolled(monkeypatch):
    """Candidate 1 stutters -> chain fires regenerate -> candidate 2 (clean)
    wins. The winning block carries the clean prose and no marker."""
    monkeypatch.delenv("COURSEFORGE_BEST_OF_N_SELECT_BY", raising=False)
    monkeypatch.delenv("COURSEFORGE_REWRITE_N_CANDIDATES", raising=False)
    outputs = [
        _candidate(0, prose=_STUTTERED_HTML),
        _candidate(1, prose=_CLEAN_HTML),
    ]
    router, provider = _router(outputs)
    out = router.route_rewrite_with_remediation(
        outputs[0],
        validators=[ProseStutterValidator()],
        regen_budget=5,
    )
    # The stuttered candidate consumed one dispatch; the clean re-roll won.
    assert len(provider.calls) == 2
    assert out.content == _CLEAN_HTML
    assert out.escalation_marker is None
    assert any(
        t.purpose == "self_consistency_winner" for t in out.touched_by
    )
    # The re-roll saw a remediation suffix naming the stutter gate.
    suffix = provider.calls[1].get("remediation_suffix")
    assert suffix and "block_prose_stutter" in suffix


def test_clean_candidate_passes_first_roll(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_BEST_OF_N_SELECT_BY", raising=False)
    monkeypatch.delenv("COURSEFORGE_REWRITE_N_CANDIDATES", raising=False)
    outputs = [_candidate(0, prose=_CLEAN_HTML)]
    router, provider = _router(outputs)
    out = router.route_rewrite_with_remediation(
        outputs[0],
        validators=[ProseStutterValidator()],
    )
    assert len(provider.calls) == 1
    assert out.content == _CLEAN_HTML
    assert out.escalation_marker is None


def test_persistent_stutter_exhausts_budget_with_marker(monkeypatch):
    """Every candidate stutters -> budget exhausts -> the surviving
    best-effort candidate is stamped validator_consensus_fail (never a
    silent pass)."""
    monkeypatch.delenv("COURSEFORGE_BEST_OF_N_SELECT_BY", raising=False)
    monkeypatch.delenv("COURSEFORGE_REWRITE_N_CANDIDATES", raising=False)
    outputs = [_candidate(i, prose=_STUTTERED_HTML) for i in range(3)]
    router, provider = _router(outputs)
    out = router.route_rewrite_with_remediation(
        outputs[0],
        validators=[ProseStutterValidator()],
        regen_budget=3,
    )
    assert out.escalation_marker == "validator_consensus_fail"


def test_pipeline_inloop_chain_carries_stutter_validator():
    """``_run_content_generation_rewrite`` threads the stutter check into
    the per-candidate chain via ``_build_rewrite_inloop_validators``."""
    from MCP.tools.pipeline_tools import _build_rewrite_inloop_validators

    chain = _build_rewrite_inloop_validators()
    assert any(isinstance(v, ProseStutterValidator) for v in chain)
