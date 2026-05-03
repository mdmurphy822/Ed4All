"""Tests for ``lib.classifiers.bloom_bert_ensemble`` (Phase 4 Subtask 29).

Five tests covering:
- :func:`test_unanimous_high_confidence_returns_winner`
- :func:`test_split_vote_resolves_via_confidence_weighting`
- :func:`test_dispersion_high_when_split_vote_3_distinct_levels`
- :func:`test_member_failure_falls_through_silently_with_warning`
- :func:`test_sha_pinning_recorded_in_decision_event`

Phase 8 Subtask 4 extends with SHA-pin + label-map regression tests:
- :func:`test_default_ensemble_revisions_are_concrete_commit_shas`
- :func:`test_default_ensemble_first_member_is_cip29_replacement`
- :func:`test_cip29_to_bloom_covers_all_canonical_levels`
- :func:`test_cip29_to_bloom_keys_are_label_n_form`

The ensemble's :meth:`_load_members` is mocked in every test so the
suite never touches the real ``transformers`` extras (the CI worker
doesn't have them installed). Real model inference is exercised by
the integration smoke tests when the optional ``[bert]`` extra is
available.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from lib.classifiers.bloom_bert_ensemble import (
    _BLOOM_LEVELS,
    _CIP29_TO_BLOOM,
    _DEFAULT_ENSEMBLE_MEMBERS,
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


def test_temperature_scaling_can_flip_winner() -> None:
    """Subtask 33 — per-member temperature scaling.

    Single-class temperature scaling (raising a confidence in (0, 1]
    to power 1/T) sharpens (T<1) or softens (T>1) the ranking
    relative to other members. We use a scenario where the
    ``apply``-side has two moderate votes vs ``remember``'s single
    high-conf vote, then crank the ``remember`` member's T up to
    soften it enough to flip the winner.

    Setup: votes = [(remember, 0.95), (apply, 0.55), (apply, 0.55)].
    Baseline T=1.0: remember=0.95, apply=0.55+0.55=1.10. apply wins.
    To FLIP back to remember-winner we'd need to *sharpen* apply's
    side (T<1) which would shrink the apply confidences. But T<1
    *raises* the gap (0.55^2=0.3025 each → 0.605 total < 0.95).
    Verify both directions:

    1. Baseline (T=None / 1.0): apply wins (1.10 > 0.95).
    2. Sharpened apply (T=[1.0, 0.5, 0.5]): apply confidences
       become 0.55^2=0.3025 each → sum 0.605, remember stays 0.95
       → remember wins. Temperature actually flipped the outcome.
    3. Per-member-list shorter than members: pad with 1.0 (no-op
       for the missing entries; should match baseline).
    """
    ensemble = _ScriptedEnsemble(
        votes_per_member=[
            ("remember", 0.95),
            ("apply", 0.55),
            ("apply", 0.55),
        ]
    )
    # Baseline: no temperature → apply wins (1.10 > 0.95).
    baseline = ensemble.classify("Some text.")
    assert baseline["winner_level"] == "apply"

    # Sharpened apply members → 0.55^2 = 0.3025 each, sum 0.605
    # < remember's 0.95 → remember wins. Confirms per-member T flips.
    ensemble2 = _ScriptedEnsemble(
        votes_per_member=[
            ("remember", 0.95),
            ("apply", 0.55),
            ("apply", 0.55),
        ]
    )
    ensemble2.set_temperature([1.0, 0.5, 0.5])
    flipped = ensemble2.classify("Some text.")
    assert flipped["winner_level"] == "remember"

    # Per-member-list shorter than members → pad with 1.0; result
    # equals baseline.
    ensemble3 = _ScriptedEnsemble(
        votes_per_member=[
            ("remember", 0.95),
            ("apply", 0.55),
            ("apply", 0.55),
        ]
    )
    ensemble3.set_temperature([1.0])  # only first member; rest padded
    short = ensemble3.classify("Some text.")
    assert short["winner_level"] == baseline["winner_level"]
    assert short["winner_score"] == pytest.approx(baseline["winner_score"], abs=1e-4)


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


# ---------------------------------------------------------------------------
# Phase 8 Subtask 4 — SHA-pin + cip29 swap + label-map regression tests
# ---------------------------------------------------------------------------

#: Regex shape of a HuggingFace git commit SHA: 40 lowercase hex chars.
_HF_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_default_ensemble_revisions_are_concrete_commit_shas() -> None:
    """Every default ensemble member's ``revision`` is a 40-hex-char SHA.

    Phase 8 ST 4 replaced the placeholder ``"main"`` revisions with
    concrete SHAs resolved via
    ``huggingface_hub.HfApi().model_info(repo_id).sha``. This test
    locks the contract: any future drift back to a tag-style ref
    (e.g. ``"main"``, ``"v1.0"``) trips the regex assertion. SHAs
    pinned at land time:

    - cip29/bert-blooms-taxonomy-classifier: ae343e4f...
    - distilbert-base-uncased-finetuned-sst-2-english: 714eb0fa...
    - MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli: 6f5cf0a2...
    """
    assert len(_DEFAULT_ENSEMBLE_MEMBERS) == 3, (
        "Default ensemble must have exactly 3 members; "
        f"got {len(_DEFAULT_ENSEMBLE_MEMBERS)}"
    )
    for entry in _DEFAULT_ENSEMBLE_MEMBERS:
        assert "revision" in entry, f"Missing 'revision' field on entry {entry}"
        rev = entry["revision"]
        assert _HF_COMMIT_SHA_RE.match(rev), (
            f"Member {entry.get('name')!r} has revision {rev!r} which is "
            f"not a 40-hex-char HuggingFace commit SHA. Phase 8 ST 4 "
            f"contract: every revision MUST be a concrete commit SHA."
        )


def test_default_ensemble_first_member_is_cip29_replacement() -> None:
    """The first ensemble member is the cip29 Bloom classifier.

    Phase 8 ST 4 replaced the deleted ``kabir5297/bloom_taxonomy_classifier``
    with ``cip29/bert-blooms-taxonomy-classifier`` per operator
    decision logged 2026-05-03. This test guards against accidental
    rollback to the deleted upstream repo.
    """
    first = _DEFAULT_ENSEMBLE_MEMBERS[0]
    assert first["name"] == "cip29/bert-blooms-taxonomy-classifier", (
        f"First member should be cip29/bert-blooms-taxonomy-classifier; "
        f"got {first['name']!r}. The legacy kabir5297 repo was deleted "
        f"upstream — do not roll back to it."
    )
    # The kabir5297 reference must NOT appear anywhere in the registry.
    for entry in _DEFAULT_ENSEMBLE_MEMBERS:
        assert "kabir5297" not in entry.get("name", ""), (
            f"kabir5297 reference resurfaced in {entry!r}; the upstream "
            f"repo was deleted — Phase 8 ST 4 swapped to cip29."
        )


def test_cip29_to_bloom_covers_all_canonical_levels() -> None:
    """Every value in :data:`_CIP29_TO_BLOOM` is a canonical Bloom level.

    The cip29 model emits 6 generic ``LABEL_0`` ... ``LABEL_5`` labels
    that the table translates to canonical Bloom levels. The full
    canonical 6-level enum (``remember``, ``understand``, ``apply``,
    ``analyze``, ``evaluate``, ``create``) MUST appear among the table
    values so the validator's downstream regression suite stays stable.
    """
    bloom_levels_set = set(_BLOOM_LEVELS)
    table_values = set(_CIP29_TO_BLOOM.values())
    # Every value in the table must be a canonical level.
    for v in table_values:
        assert v in bloom_levels_set, (
            f"_CIP29_TO_BLOOM value {v!r} is not in canonical _BLOOM_LEVELS "
            f"({bloom_levels_set}). Translation table values MUST be "
            f"canonical Bloom levels."
        )
    # Every canonical level appears in the table values (full coverage).
    for level in bloom_levels_set:
        assert level in table_values, (
            f"Canonical Bloom level {level!r} missing from _CIP29_TO_BLOOM "
            f"values {table_values}. Per Phase 8 ST 4 contract, the table "
            f"MUST cover all six canonical levels."
        )


def test_cip29_to_bloom_keys_are_label_n_form() -> None:
    """Every key in :data:`_CIP29_TO_BLOOM` matches the ``LABEL_N`` form.

    The cip29 model's ``id2label`` config emits generic ``LABEL_0``
    ... ``LABEL_5`` strings (verified at SHA-pin time via
    ``huggingface_hub.hf_hub_download`` against ``config.json``). The
    translation table's keys MUST mirror this generic form so that
    :meth:`_classify_with_member` can look them up directly off the
    model's argmax output without further string munging.
    """
    label_n_re = re.compile(r"^LABEL_[0-5]$")
    for key in _CIP29_TO_BLOOM.keys():
        assert label_n_re.match(key), (
            f"_CIP29_TO_BLOOM key {key!r} does not match expected "
            f"LABEL_N form (N in [0,5]). Verified at SHA-pin time: "
            f"cip29/bert-blooms-taxonomy-classifier emits LABEL_0 ... "
            f"LABEL_5 generic labels."
        )
    # Exactly six keys, one per canonical Bloom level.
    assert len(_CIP29_TO_BLOOM) == 6, (
        f"_CIP29_TO_BLOOM must have exactly 6 entries (one per "
        f"canonical Bloom level); got {len(_CIP29_TO_BLOOM)}."
    )
