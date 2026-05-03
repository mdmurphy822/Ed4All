"""Tests for ``lib.classifiers.bloom_bert_ensemble`` (Phase 4 Subtask 29).

Five tests covering:
- :func:`test_unanimous_high_confidence_returns_winner`
- :func:`test_split_vote_resolves_via_confidence_weighting`
- :func:`test_dispersion_high_when_split_vote_3_distinct_levels`
- :func:`test_member_failure_falls_through_silently_with_warning`
- :func:`test_sha_pinning_recorded_in_decision_event`

The ensemble's :meth:`_load_members` is mocked in every test so the
suite never touches the real ``transformers`` extras (the CI worker
doesn't have them installed). Real model inference is exercised by
the integration smoke tests when the optional ``[bert]`` extra is
available.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from lib.classifiers.bloom_bert_ensemble import (
    BertClassifier,
    BloomBertEnsemble,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_member(name: str, revision: str = "abc123") -> BertClassifier:
    """Create a stub :class:`BertClassifier` with mocked model + tokenizer.

    The model + tokenizer references are unused once
    :meth:`_classify_with_member` is patched in the test, so the stubs
    can be plain :class:`MagicMock` instances.
    """
    return BertClassifier(
        name=name,
        revision=revision,
        model=MagicMock(),
        tokenizer=MagicMock(),
    )


class _ScriptedEnsemble(BloomBertEnsemble):
    """Subclass that returns scripted member-vote tuples in order.

    Bypasses the real ``transformers`` import so the suite runs on a
    slim install. The ``votes`` argument is a list of
    ``(level, confidence)`` tuples; the ensemble returns them in
    order, one per member load + classify call. The default member
    list is the canonical 3-member registry; override via the
    ``members=`` constructor kwarg when a test needs to verify
    failure-fallthrough behavior.
    """

    def __init__(
        self,
        votes_per_member: List[Tuple[str, float]],
        members: List[Dict[str, str]] | None = None,
    ) -> None:
        super().__init__(members=members)
        self._scripted_votes = list(votes_per_member)
        self._vote_idx = 0

    def _load_members(self) -> List[BertClassifier]:
        if self._loaded is not None:
            return self._loaded
        # One stub per scripted vote — caller controls how many members
        # the ensemble has via the votes-list length.
        loaded = [
            _stub_member(name=f"stub-member-{i}", revision=f"sha-{i}")
            for i in range(len(self._scripted_votes))
        ]
        self._loaded = loaded
        return loaded

    def _classify_with_member(
        self, member: BertClassifier, text: str
    ) -> Tuple[str, float]:
        if self._vote_idx >= len(self._scripted_votes):
            raise IndexError("Scripted ensemble ran out of votes")
        vote = self._scripted_votes[self._vote_idx]
        self._vote_idx += 1
        return vote


# ---------------------------------------------------------------------------
# Subtask 29 tests
# ---------------------------------------------------------------------------


def test_unanimous_high_confidence_returns_winner() -> None:
    """Three members all vote ``remember`` → winner=remember, dispersion=0."""
    ensemble = _ScriptedEnsemble(
        votes_per_member=[
            ("remember", 0.92),
            ("remember", 0.88),
            ("remember", 0.95),
        ]
    )
    result = ensemble.classify("List the four steps of cellular respiration.")

    assert result["winner_level"] == "remember"
    assert result["winner_score"] == pytest.approx(1.0, abs=1e-3)
    assert result["dispersion"] == pytest.approx(0.0, abs=1e-6)
    assert len(result["per_member"]) == 3
    # Per-member ordering preserves registry order.
    assert all(level == "remember" for level, _ in result["per_member"])


def test_split_vote_resolves_via_confidence_weighting() -> None:
    """2 vote ``remember`` (high conf) vs 1 votes ``apply`` (medium conf).

    The ``remember`` block accumulates 0.85 + 0.78 = 1.63 vs ``apply``'s
    0.55. Winner = remember; dispersion is non-zero but well below 1.0
    (one level dominates). Confirms ``_aggregate``'s argmax logic uses
    confidence weighting, not raw vote count.
    """
    ensemble = _ScriptedEnsemble(
        votes_per_member=[
            ("remember", 0.85),
            ("remember", 0.78),
            ("apply", 0.55),
        ]
    )
    result = ensemble.classify("Identify the three states of matter.")

    assert result["winner_level"] == "remember"
    # Winner score = 1.63 / (1.63 + 0.55) = 0.7477...
    assert result["winner_score"] == pytest.approx(0.7477, abs=1e-3)
    # Dispersion is non-zero (two distinct levels) but < 1.0 (not uniform).
    assert 0.0 < result["dispersion"] < 1.0


def test_dispersion_high_when_split_vote_3_distinct_levels() -> None:
    """Three members vote three distinct levels with equal confidence.

    Score per level: 0.5 each across {remember, apply, analyze}. The
    distribution is uniform, so normalised entropy hits 1.0 — the
    "high dispersion" sentinel that the validator uses to fire
    ``BERT_ENSEMBLE_DISPERSION_HIGH``.
    """
    ensemble = _ScriptedEnsemble(
        votes_per_member=[
            ("remember", 0.5),
            ("apply", 0.5),
            ("analyze", 0.5),
        ]
    )
    result = ensemble.classify("Some ambiguous text.")

    # Uniform distribution → max-entropy = 1.0 after normalisation.
    assert result["dispersion"] == pytest.approx(1.0, abs=1e-3)
    # Tie-broken winner is the lex-first level (analyze < apply < remember).
    assert result["winner_level"] == "analyze"
    assert result["winner_score"] == pytest.approx(1.0 / 3.0, abs=1e-3)
    assert len(result["per_member"]) == 3


def test_member_failure_falls_through_silently_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A member that raises during classify is silently omitted from the vote."""
    import logging

    class _PartiallyFailingEnsemble(_ScriptedEnsemble):
        def _classify_with_member(
            self, member: BertClassifier, text: str
        ) -> Tuple[str, float]:
            # Member 1 fails; members 0 + 2 vote normally.
            if self._vote_idx == 1:
                self._vote_idx += 1
                raise RuntimeError("simulated transformers OOM")
            return super()._classify_with_member(member, text)

    ensemble = _PartiallyFailingEnsemble(
        votes_per_member=[
            ("remember", 0.9),
            ("apply", 0.7),  # Will be skipped (member raises).
            ("remember", 0.85),
        ]
    )
    with caplog.at_level(logging.WARNING):
        result = ensemble.classify("Some text.")

    # Only members 0 + 2 voted; both for "remember" → unanimous.
    assert result["winner_level"] == "remember"
    assert result["winner_score"] == pytest.approx(1.0, abs=1e-3)
    assert result["dispersion"] == pytest.approx(0.0, abs=1e-6)
    assert len(result["per_member"]) == 2
    # Warning was logged for the failed member.
    assert any(
        "failed to classify" in rec.message for rec in caplog.records
    ), "Expected a warning log for the failed member"


def test_sha_pinning_recorded_in_decision_event() -> None:
    """The :meth:`_emit_member_loaded` capture call carries name + revision."""
    captured_events: List[Dict[str, Any]] = []

    class _RecordingCapture:
        def log_decision(
            self,
            *,
            decision_type: str,
            decision: str,
            rationale: str,
            metadata: Dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> None:
            captured_events.append(
                {
                    "decision_type": decision_type,
                    "decision": decision,
                    "rationale": rationale,
                    "metadata": metadata or {},
                }
            )

    members = [
        {"name": "kabir5297/bloom_taxonomy_classifier", "revision": "deadbeef00"},
        {"name": "distilbert-base-uncased-finetuned-sst-2-english", "revision": "feedface11"},
    ]

    class _LoadEmittingEnsemble(BloomBertEnsemble):
        """Ensemble whose ``_load_members`` walks ``self.members`` + emits."""

        def _load_one_member(self, member: Dict[str, str]) -> BertClassifier | None:
            # Fake a successful load so the success-path emit fires.
            return _stub_member(member["name"], member["revision"])

        def _load_members(self) -> List[BertClassifier]:
            if self._loaded is not None:
                return self._loaded
            loaded: List[BertClassifier] = []
            for member in self.members:
                clf = self._load_one_member(member)
                if clf is not None:
                    loaded.append(clf)
                    self._emit_member_loaded(member, success=True)
                else:
                    self._emit_member_loaded(member, success=False)
            self._loaded = loaded
            return loaded

    ensemble = _LoadEmittingEnsemble(members=members)
    ensemble.attach_capture(_RecordingCapture())
    loaded = ensemble._load_members()

    assert len(loaded) == 2
    assert len(captured_events) == 2

    for event, member in zip(captured_events, members):
        assert event["decision_type"] == "bert_ensemble_member_loaded"
        # Decision string carries name@revision so an audit trail
        # reader can extract the SHA without parsing metadata.
        assert member["name"] in event["decision"]
        assert member["revision"] in event["decision"]
        # Rationale carries the same plus cache_dir + success flag.
        assert "revision=" in event["rationale"]
        assert member["revision"] in event["rationale"]
        # Structured metadata carries the canonical keys.
        assert event["metadata"]["member_name"] == member["name"]
        assert event["metadata"]["member_revision"] == member["revision"]
        assert event["metadata"]["success"] is True
