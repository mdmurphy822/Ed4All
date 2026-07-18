"""Tests for ``lib.classifiers.bloom_zero_shot`` — the trivote voters.

Covers (no GPU / no model load — the NLI singleton is stubbed or the
``nli=`` seam is injected):

- Zero-shot template scoring argmax + per-level softmax shape.
- Zero-shot abstention paths (empty text, NLI unavailable, score_batch
  raises, wrong-length response).
- Verb-ontology voter delegation + abstention.
- Trivote disagreement math (agreement, full scatter, partial, single-
  voter skip, non-canonical normalisation).
- The ``ED4ALL_BLOOM_TRIVOTE`` flag resolver (parse-with-fallback).
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from lib.classifiers.bloom_zero_shot import (
    _BLOOM_HYPOTHESIS_TEMPLATES,
    resolve_trivote_mode,
    trivote_disagreement,
    verb_vote,
    zero_shot_bloom,
)
from lib.ontology.bloom import BLOOM_LEVELS


# --------------------------------------------------------------------- #
# Stub NLI — no DeBERTa in CI.
# --------------------------------------------------------------------- #


class _FakeNliScore:
    def __init__(self, entailment: float, neutral: float = 0.0,
                 contradiction: float = 0.0) -> None:
        self.entailment = entailment
        self.neutral = neutral
        self.contradiction = contradiction


class _TargetNli:
    """Scores high entailment for the hypothesis of ``target_level``.

    ``zero_shot_bloom`` builds one (premise, hypothesis) pair per
    canonical level in ``BLOOM_LEVELS`` order; this stub matches the
    hypothesis string to decide which pair gets the high entailment.
    """

    def __init__(self, target_level: str, high: float = 0.9,
                 low: float = 0.1) -> None:
        self.target_level = target_level
        self.high = high
        self.low = low
        self.batches: List[List[Tuple[str, str]]] = []

    def score_batch(self, *, pairs, batch_size=None):  # noqa: ANN001
        self.batches.append(list(pairs))
        target_hyp = _BLOOM_HYPOTHESIS_TEMPLATES[self.target_level]
        return [
            _FakeNliScore(self.high if hyp == target_hyp else self.low)
            for _premise, hyp in pairs
        ]


# --------------------------------------------------------------------- #
# zero_shot_bloom
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("target", list(BLOOM_LEVELS))
def test_zero_shot_argmax_picks_target_level(target: str) -> None:
    nli = _TargetNli(target)
    result = zero_shot_bloom("Some task text for the student.", nli=nli)
    assert result is not None
    level, conf, per_level = result
    assert level == target
    # Winner confidence is the largest per-level share (rounded 4dp) and
    # exceeds uniform.
    assert conf == pytest.approx(max(per_level.values()), abs=1e-3)
    assert conf > 1.0 / len(BLOOM_LEVELS)
    # Full canonical distribution returned; sums to ~1.0.
    assert set(per_level) == set(BLOOM_LEVELS)
    assert sum(per_level.values()) == pytest.approx(1.0, abs=1e-6)
    # Exactly six pairs scored (one per level).
    assert len(nli.batches[0]) == len(BLOOM_LEVELS)


def test_zero_shot_empty_text_abstains() -> None:
    assert zero_shot_bloom("", nli=_TargetNli("apply")) is None
    assert zero_shot_bloom("   ", nli=_TargetNli("apply")) is None


def test_zero_shot_nli_unavailable_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    """No injected NLI + singleton returns None => abstain (no model load)."""
    from lib.classifiers import nli_classifier

    monkeypatch.setattr(
        nli_classifier.NliClassifier,
        "get_or_load",
        classmethod(lambda cls: None),
    )
    assert zero_shot_bloom("Solve the problem.", nli=None) is None


def test_zero_shot_score_batch_raises_abstains() -> None:
    class _BoomNli:
        def score_batch(self, *, pairs, batch_size=None):  # noqa: ANN001
            raise RuntimeError("simulated NLI failure")

    assert zero_shot_bloom("Analyze the passage.", nli=_BoomNli()) is None


def test_zero_shot_wrong_length_response_abstains() -> None:
    class _ShortNli:
        def score_batch(self, *, pairs, batch_size=None):  # noqa: ANN001
            return [_FakeNliScore(0.9)]  # only one score, not six

    assert zero_shot_bloom("Describe the cycle.", nli=_ShortNli()) is None


# --------------------------------------------------------------------- #
# verb_vote
# --------------------------------------------------------------------- #


def test_verb_vote_detects_canonical_verb() -> None:
    level, verb = verb_vote("Calculate the resulting velocity.")
    assert level == "apply"
    assert verb == "calculate"


def test_verb_vote_abstains_without_canonical_verb() -> None:
    assert verb_vote("The weather is nice today with clouds overhead.") == (None, None)
    assert verb_vote("") == (None, None)


# --------------------------------------------------------------------- #
# trivote_disagreement
# --------------------------------------------------------------------- #


def test_trivote_unanimous_agreement() -> None:
    td = trivote_disagreement("apply", "apply", "apply")
    assert td["n_present"] == 3
    assert td["disagreement_score"] == 0.0
    assert td["skipped"] is False
    assert td["asserted_supported"] is True
    assert td["asserted_unsupported"] is False
    assert td["majority_level"] == "apply"


def test_trivote_full_scatter() -> None:
    td = trivote_disagreement("apply", "analyze", "create")
    assert td["n_present"] == 3
    assert td["pairwise_total"] == 3
    assert td["pairwise_mismatch"] == 3
    assert td["disagreement_score"] == 1.0
    assert td["asserted_unsupported"] is True
    # Lexicographic tie-break among three singleton counts.
    assert td["majority_level"] == "analyze"


def test_trivote_asserted_supported_partial() -> None:
    """Asserted agrees with one independent; the other differs."""
    td = trivote_disagreement("apply", "apply", "create")
    assert td["disagreement_score"] == pytest.approx(2 / 3, abs=1e-4)
    assert td["asserted_supported"] is True
    assert td["asserted_unsupported"] is False
    assert td["majority_level"] == "apply"


def test_trivote_asserted_unsupported_by_agreeing_independents() -> None:
    """Both independents agree with each other but not with asserted."""
    td = trivote_disagreement("remember", "apply", "apply")
    assert td["disagreement_score"] == pytest.approx(2 / 3, abs=1e-4)
    assert td["asserted_supported"] is False
    assert td["asserted_unsupported"] is True
    assert td["majority_level"] == "apply"


def test_trivote_one_abstention_is_not_skipped() -> None:
    td = trivote_disagreement("apply", None, "apply")
    assert td["n_present"] == 2
    assert td["skipped"] is False
    assert td["disagreement_score"] == 0.0
    assert td["asserted_supported"] is True
    assert td["abstentions"] == ["zero_shot"]


def test_trivote_two_abstentions_skipped() -> None:
    td = trivote_disagreement("apply", None, None)
    assert td["n_present"] == 1
    assert td["skipped"] is True
    assert td["disagreement_score"] == 0.0


def test_trivote_all_abstain_skipped() -> None:
    td = trivote_disagreement(None, None, None)
    assert td["n_present"] == 0
    assert td["skipped"] is True


def test_trivote_non_canonical_asserted_normalises_to_abstain() -> None:
    td = trivote_disagreement("Application", "apply", "apply")
    assert td["votes"]["asserted"] is None
    assert td["asserted_present"] is False
    # Only the two independents remain — they agree, so no disagreement.
    assert td["n_present"] == 2
    assert td["disagreement_score"] == 0.0


# --------------------------------------------------------------------- #
# resolve_trivote_mode
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True),
        ("on", True), ("On", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("garbage", False), ("", False),
    ],
)
def test_resolve_trivote_mode(monkeypatch: pytest.MonkeyPatch, value: str,
                              expected: bool) -> None:
    monkeypatch.setenv("ED4ALL_BLOOM_TRIVOTE", value)
    assert resolve_trivote_mode() is expected


def test_resolve_trivote_mode_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ED4ALL_BLOOM_TRIVOTE", raising=False)
    assert resolve_trivote_mode() is False
