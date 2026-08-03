"""ED4ALL_PLANNING_REROLL_SALT + _FEEDBACK plumbing tests.

The course_planning gate-retry loop in ``MCP/core/workflow_runner.py`` sets
``ED4ALL_PLANNING_REROLL_SALT=attempt-N`` PLUS the failure-directed
``ED4ALL_PLANNING_REROLL_FEEDBACK`` remediation digest before each
re-dispatch. Two downstream consumers make the retry attempt genuinely
differ from (and be DIRECTED at the failure modes of) the cached/previous
attempt:

1. ``MCP/tools/pipeline_tools.py`` folds the salt (verbatim) + the feedback
   digest (hashed) into the stage-2 CLUSTER sidecar fingerprint (both
   TO-derivation call sites), so a salted/feedback-directed retry never
   reuses a prior attempt's cached TO — even when the sidecar eviction was
   lost. Both unset → the fingerprint dict carries neither key
   (byte-identical to legacy) and sidecar reuse works as before.
2. ``Courseforge/generators/_textbook_synthesis_provider.py`` folds the
   salt (a ``[Re-roll …]`` directive) + the feedback digest (a delimited
   REMEDIATION section) into the SYSTEM prompt at construction, so every
   dispatched synthesis call differs per attempt and sees the concrete
   failure signals. Both unset → the prompt is the unsalted module
   constant (byte-identical).

Hermetic — FakeEmbed + a fake provider; no LLM, no GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402

_SALT_ENV = "ED4ALL_PLANNING_REROLL_SALT"
_FEEDBACK_ENV = "ED4ALL_PLANNING_REROLL_FEEDBACK"

_DIGEST = (
    "- [objective_entailment] TO-08 OBJECTIVE_UNENTAILED: the statement "
    "must be directly supported by its cited chunks"
)


class _FakeProvider:
    """Authors one TO per cluster; records every author call."""

    _model = "fake-model"

    def __init__(self) -> None:
        self.author_calls: List[int] = []

    def author_terminal_for_cluster(
        self,
        cluster_cos: List[Dict[str, Any]],
        *,
        course_name: str,
        cluster_index: int,
    ) -> Dict[str, Any]:
        self.author_calls.append(cluster_index)
        rep = max(
            cluster_cos, key=lambda c: len(str(c.get("statement") or ""))
        )
        return {
            "statement": f"TO summarising: {rep.get('statement')}",
            "bloom_level": "understand",
            "source_refs": [],
        }


def _mint(kind: str, idx: int) -> str:
    prefix = {"terminal": "TO", "chapter": "CO"}.get(kind, "XX")
    return f"{prefix}-{idx:02d}"


def _co(statement: str) -> Dict[str, Any]:
    return {"statement": statement}


def _theme_cos() -> List[Dict[str, Any]]:
    """Three tight themes of COs (distinct vocab per theme) → ~3 clusters."""
    return [
        _co("Define a prime number and identify its factors."),
        _co("Define what a prime number is and list its factors."),
        _co("Compute the area of a triangle using base and height."),
        _co("Compute the area of a right triangle from base height."),
        _co("Photosynthesis converts light energy in the chloroplast."),
        _co("Photosynthesis uses light energy inside the chloroplast cell."),
    ]


def _derive(provider: _FakeProvider, checkpoint_path: Path):
    return pt._derive_terminals_bottom_up(
        provider=provider,
        chapter_cos=_theme_cos(),
        embed=FakeEmbed(),
        course_name="FXMATH_101",
        mint_lo_id=_mint,
        checkpoint_path=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# _planning_reroll_salt resolver
# ---------------------------------------------------------------------------


def test_planning_reroll_salt_empty_by_default(monkeypatch) -> None:
    monkeypatch.delenv(_SALT_ENV, raising=False)
    assert pt._planning_reroll_salt() == ""
    monkeypatch.setenv(_SALT_ENV, "  attempt-3  ")
    assert pt._planning_reroll_salt() == "attempt-3"


def test_planning_reroll_feedback_empty_by_default(monkeypatch) -> None:
    monkeypatch.delenv(_FEEDBACK_ENV, raising=False)
    assert pt._planning_reroll_feedback() == ""
    monkeypatch.setenv(_FEEDBACK_ENV, f"  {_DIGEST}  ")
    assert pt._planning_reroll_feedback() == _DIGEST


# ---------------------------------------------------------------------------
# Cluster sidecar fingerprint — salt invalidates reuse; unsalted is stable
# ---------------------------------------------------------------------------


def test_unsalted_reentry_reuses_cluster_sidecar(tmp_path, monkeypatch) -> None:
    """Control: without the salt, sidecar reuse is byte-identical legacy."""
    monkeypatch.delenv(_SALT_ENV, raising=False)
    monkeypatch.delenv(_FEEDBACK_ENV, raising=False)
    cp = tmp_path / ".stage2_clusters_checkpoint.jsonl"

    p1 = _FakeProvider()
    terminals1, _ = _derive(p1, cp)
    assert p1.author_calls, "first run must author TOs"
    assert cp.exists(), "sidecar must be written"

    p2 = _FakeProvider()
    terminals2, signals2 = _derive(p2, cp)
    assert p2.author_calls == [], "unsalted re-entry must reuse every cluster"
    assert signals2["to_clusters_reused_from_checkpoint"] == len(terminals1)
    assert [t["statement"] for t in terminals2] == [
        t["statement"] for t in terminals1
    ]


def test_salted_reentry_invalidates_cluster_sidecar_reuse(
    tmp_path, monkeypatch
) -> None:
    """With the salt set, a surviving sidecar never satisfies reuse — every
    cluster TO re-authors (the retry genuinely re-rolls even when the
    runner-side eviction was lost)."""
    monkeypatch.delenv(_SALT_ENV, raising=False)
    monkeypatch.delenv(_FEEDBACK_ENV, raising=False)
    cp = tmp_path / ".stage2_clusters_checkpoint.jsonl"

    p1 = _FakeProvider()
    _derive(p1, cp)
    assert cp.exists()

    monkeypatch.setenv(_SALT_ENV, "attempt-1")
    p2 = _FakeProvider()
    _, signals2 = _derive(p2, cp)
    assert len(p2.author_calls) == len(p1.author_calls), (
        "salted re-entry must re-author every cluster (no sidecar reuse)"
    )
    assert signals2["to_clusters_reused_from_checkpoint"] == 0

    # And a SECOND run under the SAME salt reuses again (the salt is part of
    # the fingerprint, not a reuse kill-switch).
    p3 = _FakeProvider()
    _, signals3 = _derive(p3, cp)
    assert p3.author_calls == []
    assert signals3["to_clusters_reused_from_checkpoint"] == len(
        p1.author_calls
    )


def test_feedback_change_invalidates_cluster_sidecar_reuse(
    tmp_path, monkeypatch
) -> None:
    """The failure-directed feedback digest is part of the cluster
    fingerprint: a DIFFERENT digest re-authors every cluster (no
    stale-cache reuse); the SAME digest reuses."""
    monkeypatch.delenv(_SALT_ENV, raising=False)
    monkeypatch.setenv(_FEEDBACK_ENV, _DIGEST)
    cp = tmp_path / ".stage2_clusters_checkpoint.jsonl"

    p1 = _FakeProvider()
    _derive(p1, cp)
    assert p1.author_calls, "first run must author TOs"

    # SAME digest → reuse (the digest is fingerprint input, not a
    # kill-switch).
    p2 = _FakeProvider()
    _, signals2 = _derive(p2, cp)
    assert p2.author_calls == []
    assert signals2["to_clusters_reused_from_checkpoint"] == len(
        p1.author_calls
    )

    # DIFFERENT digest → every cluster re-authors.
    monkeypatch.setenv(
        _FEEDBACK_ENV,
        "- [objective_entailment] TO-02 OBJECTIVE_CONTRADICTED: restate",
    )
    p3 = _FakeProvider()
    _, signals3 = _derive(p3, cp)
    assert len(p3.author_calls) == len(p1.author_calls), (
        "a changed feedback digest must invalidate cluster sidecar reuse"
    )
    assert signals3["to_clusters_reused_from_checkpoint"] == 0

    # Feedback CLEARED (normal run) → the feedback key leaves the
    # fingerprint, so the cleared-state fingerprint differs from the
    # feedback-stamped one and the clusters re-author once more.
    monkeypatch.delenv(_FEEDBACK_ENV, raising=False)
    p4 = _FakeProvider()
    _, signals4 = _derive(p4, cp)
    assert len(p4.author_calls) == len(p1.author_calls)
    assert signals4["to_clusters_reused_from_checkpoint"] == 0


# ---------------------------------------------------------------------------
# TextbookSynthesisProvider — salt folded into the system prompt
# ---------------------------------------------------------------------------


def test_provider_system_prompt_unsalted_by_default(monkeypatch) -> None:
    from Courseforge.generators import _textbook_synthesis_provider as tsp

    monkeypatch.delenv(_SALT_ENV, raising=False)
    monkeypatch.delenv(_FEEDBACK_ENV, raising=False)
    monkeypatch.delenv(tsp.ENV_PROVIDER, raising=False)
    p = tsp.TextbookSynthesisProvider(anthropic_client=object())
    assert p._system_prompt == tsp._TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT


def test_provider_system_prompt_carries_reroll_salt(monkeypatch) -> None:
    from Courseforge.generators import _textbook_synthesis_provider as tsp

    monkeypatch.setenv(_SALT_ENV, "attempt-2")
    monkeypatch.delenv(_FEEDBACK_ENV, raising=False)
    monkeypatch.delenv(tsp.ENV_PROVIDER, raising=False)
    p = tsp.TextbookSynthesisProvider(anthropic_client=object())
    # The canonical prompt survives verbatim as the head...
    assert p._system_prompt.startswith(
        tsp._TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT
    )
    # ...with the per-attempt re-roll directive appended.
    assert "Re-roll attempt-2" in p._system_prompt
    # Salt WITHOUT feedback (digest-less retry) → no REMEDIATION section.
    assert "REMEDIATION" not in p._system_prompt


def test_provider_system_prompt_carries_remediation_section(
    monkeypatch,
) -> None:
    """Feedback env set → the digest rides in a clearly-delimited
    REMEDIATION section appended after the salt directive."""
    from Courseforge.generators import _textbook_synthesis_provider as tsp

    monkeypatch.setenv(_SALT_ENV, "attempt-2")
    monkeypatch.setenv(_FEEDBACK_ENV, _DIGEST)
    monkeypatch.delenv(tsp.ENV_PROVIDER, raising=False)
    p = tsp.TextbookSynthesisProvider(anthropic_client=object())
    assert p._system_prompt.startswith(
        tsp._TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT
    )
    assert "Re-roll attempt-2" in p._system_prompt
    assert "=== REMEDIATION" in p._system_prompt
    assert _DIGEST in p._system_prompt
    assert "=== END REMEDIATION ===" in p._system_prompt
    # The digest sits AFTER the re-roll directive (the remediation section
    # closes the prompt).
    assert p._system_prompt.index("Re-roll attempt-2") < (
        p._system_prompt.index("=== REMEDIATION")
    )


def test_provider_prompt_byte_identical_when_feedback_unset(
    monkeypatch,
) -> None:
    """Feedback (and salt) unset → the prompt is the module constant,
    byte-identical to legacy."""
    from Courseforge.generators import _textbook_synthesis_provider as tsp

    monkeypatch.delenv(_SALT_ENV, raising=False)
    monkeypatch.delenv(_FEEDBACK_ENV, raising=False)
    monkeypatch.delenv(tsp.ENV_PROVIDER, raising=False)
    p = tsp.TextbookSynthesisProvider(anthropic_client=object())
    assert p._system_prompt == tsp._TEXTBOOK_SYNTHESIS_SYSTEM_PROMPT
    assert "REMEDIATION" not in p._system_prompt
