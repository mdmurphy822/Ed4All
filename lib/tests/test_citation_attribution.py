"""Pure unit tests for claim-level citation attribution (2026-06).

Deterministic, stdlib-only: no model, no network, no LibV2. Fixtures encode BOTH
live cases that motivated the work:

  * case 1 (2026-06-11, "what is rag?"): a definitional chunk backs every answer
    sentence; an evaluation chunk sharing topical tokens backs none → claim-less
    → prune candidate.
  * case 2 (2026-06-11, "five step problem solving strategy"): the model cites
    three top-ranked passages that merely *mention* the strategy and NOT the
    uncited definitional supporter (rank 7/8) the answer steps derive from →
    attribution must mark the supporter as the high-support uncited passage so
    the pipeline can ADD it.
"""
from __future__ import annotations

from dataclasses import dataclass

from lib.retrieval.citation_attribution import (
    ADD_MIN_SHINGLE,
    DEFAULT_MIN_OVERLAP,
    DEFAULT_MODE,
    MODE_OFF,
    MODE_ON,
    MODE_SHADOW,
    attribute_citations,
    resolve_min_overlap,
    resolve_prune_mode,
)


@dataclass
class _P:
    """Minimal duck-type of a retrieved passage (chunk_id + text)."""

    chunk_id: str
    text: str


# --------------------------------------------------------------------------- #
# Flag / knob resolution
# --------------------------------------------------------------------------- #


def test_mode_default_is_shadow(monkeypatch):
    monkeypatch.delenv("ED4ALL_ANSWER_CITATION_PRUNE", raising=False)
    assert resolve_prune_mode() == MODE_SHADOW
    assert DEFAULT_MODE == MODE_SHADOW


def test_mode_explicit_wins_over_env(monkeypatch):
    monkeypatch.setenv("ED4ALL_ANSWER_CITATION_PRUNE", "on")
    assert resolve_prune_mode("off") == MODE_OFF
    assert resolve_prune_mode() == MODE_ON


def test_mode_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ED4ALL_ANSWER_CITATION_PRUNE", "loud")
    assert resolve_prune_mode() == MODE_SHADOW


def test_min_overlap_default_and_env(monkeypatch):
    monkeypatch.delenv("ED4ALL_ANSWER_PRUNE_MIN_OVERLAP", raising=False)
    assert resolve_min_overlap() == DEFAULT_MIN_OVERLAP
    monkeypatch.setenv("ED4ALL_ANSWER_PRUNE_MIN_OVERLAP", "0.4")
    assert resolve_min_overlap() == 0.4
    # explicit wins
    assert resolve_min_overlap(0.1) == 0.1


def test_min_overlap_out_of_range_falls_back(monkeypatch):
    monkeypatch.setenv("ED4ALL_ANSWER_PRUNE_MIN_OVERLAP", "9.0")
    assert resolve_min_overlap() == DEFAULT_MIN_OVERLAP
    monkeypatch.setenv("ED4ALL_ANSWER_PRUNE_MIN_OVERLAP", "junk")
    assert resolve_min_overlap() == DEFAULT_MIN_OVERLAP


# --------------------------------------------------------------------------- #
# Case 1: claim-less citation → prune candidate (the "what is rag?" shape)
# --------------------------------------------------------------------------- #


_DEFINITIONAL = _P(
    "chunk_def",
    "Retrieval augmented generation is a technique that retrieves relevant "
    "passages from a knowledge source and supplies them to a language model "
    "so the model generates an answer grounded in the retrieved passages.",
)
_EVALUATION = _P(
    "chunk_eval",
    "Evaluating retrieval augmented generation systems requires measuring "
    "faithfulness, answer relevance, and context precision on a benchmark "
    "of held-out questions with curated gold passages.",
)
# Near-verbatim of the definitional chunk (the model's answer).
_RAG_ANSWER = (
    "Retrieval augmented generation is a technique that retrieves relevant "
    "passages from a knowledge source and supplies them to a language model. "
    "The model generates an answer grounded in the retrieved passages."
)


def test_case1_evaluation_chunk_is_claimless():
    report = attribute_citations(
        _RAG_ANSWER,
        [_DEFINITIONAL, _EVALUATION],
        cited_ids=["chunk_def", "chunk_eval"],
    )
    assert report.n_claims >= 2
    # The definitional chunk supports >= 1 claim; the evaluation chunk supports
    # none → it is the claim-less prune candidate.
    assert not report.supports["chunk_def"].is_claimless
    assert report.supports["chunk_eval"].is_claimless
    assert report.cited_claimless_ids() == ["chunk_eval"]
    assert "chunk_def" in report.cited_supported_ids()


def test_case1_excerpt_is_a_real_supporting_sentence():
    report = attribute_citations(
        _RAG_ANSWER, [_DEFINITIONAL], cited_ids=["chunk_def"]
    )
    support = report.supports["chunk_def"]
    assert support.supporting_excerpt is not None
    # The excerpt is a sentence drawn from the chunk, never fabricated.
    assert support.supporting_excerpt in _DEFINITIONAL.text or (
        support.supporting_excerpt.rstrip(".") in _DEFINITIONAL.text
    )
    assert "retrieval augmented generation" in support.supporting_excerpt.lower()


def test_claimless_chunk_has_no_excerpt():
    report = attribute_citations(
        _RAG_ANSWER, [_EVALUATION], cited_ids=["chunk_eval"]
    )
    assert report.supports["chunk_eval"].supporting_excerpt is None


# --------------------------------------------------------------------------- #
# Paraphrase: token-coverage arm keeps a lightly-reworded supporter
# --------------------------------------------------------------------------- #


def test_paraphrase_kept_by_token_coverage_arm():
    chunk = _P(
        "chunk_para",
        "Photosynthesis converts light energy into chemical energy stored in "
        "glucose using carbon dioxide and water inside the chloroplasts.",
    )
    # Reordered paraphrase: low 4-shingle overlap, but the answer's content
    # tokens are all present in the chunk → the token-coverage OR-arm fires.
    answer = (
        "Inside the chloroplasts, photosynthesis converts light energy and "
        "stored chemical energy using water carbon dioxide glucose."
    )
    report = attribute_citations(answer, [chunk], cited_ids=["chunk_para"])
    support = report.supports["chunk_para"]
    assert not support.is_claimless  # the token-coverage OR-arm saved it
    assert support.best_coverage >= 0.80


# --------------------------------------------------------------------------- #
# Multi-chunk: a claim supported by two chunks keeps both (threshold, not argmax)
# --------------------------------------------------------------------------- #


def test_multi_chunk_support_keeps_both():
    a = _P("chunk_a", "The mitochondria is the powerhouse of the cell.")
    b = _P("chunk_b", "The mitochondria is the powerhouse of the cell indeed.")
    answer = "The mitochondria is the powerhouse of the cell."
    report = attribute_citations(answer, [a, b], cited_ids=["chunk_a", "chunk_b"])
    assert not report.supports["chunk_a"].is_claimless
    assert not report.supports["chunk_b"].is_claimless
    # Both recorded as supporters of claim 0.
    assert set(report.claim_supporters[0]) == {"chunk_a", "chunk_b"}


# --------------------------------------------------------------------------- #
# Empty / short answers → no scorable claims
# --------------------------------------------------------------------------- #


def test_empty_answer_yields_no_claims():
    report = attribute_citations("", [_DEFINITIONAL], cited_ids=["chunk_def"])
    assert report.n_claims == 0
    assert report.supports["chunk_def"].is_claimless


def test_short_fragment_below_min_tokens_yields_no_claims():
    report = attribute_citations("Yes.", [_DEFINITIONAL], cited_ids=["chunk_def"])
    assert report.n_claims == 0


# --------------------------------------------------------------------------- #
# Case 2: uncited definitional supporter that out-supports the cited top-3
# --------------------------------------------------------------------------- #
#
# The model cited three top-ranked passages that merely MENTION the five-step
# strategy; the uncited supporter (rank 7) carries the step definitions the
# answer derives from. Attribution must mark the supporter non-claimless and
# uncited, with a high shingle score above ADD_MIN_SHINGLE, while the cited
# top-3 score low.


_FIVE_STEP_ANSWER = (
    "The five step problem solving strategy is a structured method. "
    "First, understand the problem and identify what is being asked. "
    "Second, devise a plan by choosing a strategy that fits the problem. "
    "Third, carry out the plan and perform the calculations carefully. "
    "Fourth, look back and check whether the answer is reasonable."
)
# Top-3 cited passages: each merely mentions the strategy by name.
_MENTION_1 = _P(
    "chunk_top1",
    "Many textbooks recommend the five step problem solving strategy when "
    "approaching unfamiliar exercises in the chapter review.",
)
_MENTION_2 = _P(
    "chunk_top2",
    "The five step problem solving strategy appears again in the worked "
    "examples that close each section of the unit.",
)
_MENTION_3 = _P(
    "chunk_top3",
    "Students who practice the five step problem solving strategy often report "
    "more confidence on the end of chapter assessment.",
)
# Uncited supporter (rank 7): the actual step definitions.
_SUPPORTER = _P(
    "chunk_supporter",
    "First, understand the problem and identify what is being asked. "
    "Second, devise a plan by choosing a strategy that fits the problem. "
    "Third, carry out the plan and perform the calculations carefully. "
    "Fourth, look back and check whether the answer is reasonable. "
    "Fifth, reflect on the method so it transfers to new problems.",
)


def test_case2_uncited_supporter_outsupports_cited_top3():
    gate_eligible = [_MENTION_1, _MENTION_2, _MENTION_3, _SUPPORTER]
    cited = ["chunk_top1", "chunk_top2", "chunk_top3"]
    report = attribute_citations(_FIVE_STEP_ANSWER, gate_eligible, cited_ids=cited)

    supporter = report.supports["chunk_supporter"]
    assert supporter.cited is False
    assert not supporter.is_claimless
    # The supporter clears the high ADD bar (its step text is near-verbatim).
    assert supporter.best_shingle >= ADD_MIN_SHINGLE
    # The cited top-3 each support far fewer step-claims than the supporter.
    for cid in cited:
        s = report.supports[cid]
        assert s.supported_claim_count < supporter.supported_claim_count
        assert s.best_shingle < ADD_MIN_SHINGLE
    # rank is recorded (supporter is last in the gate-eligible order).
    assert supporter.rank == 3


def test_uncited_passage_recorded_with_cited_false():
    report = attribute_citations(
        _RAG_ANSWER,
        [_DEFINITIONAL, _EVALUATION],
        cited_ids=["chunk_def"],  # only the definitional chunk cited
    )
    assert report.supports["chunk_def"].cited is True
    assert report.supports["chunk_eval"].cited is False


# =========================================================================== #
# NLI-ADD composite-criterion helpers (2026-06-12 under-citing investigation)
# =========================================================================== #

import os

import pytest

from lib.retrieval.citation_attribution import (
    DEFAULT_NLI_ADD_MODE,
    ENV_NLI_ADD,
    NLI_ADD_ENTAILMENT_FLOOR,
    NLI_ADD_TOKEN_COVERAGE_FLOOR,
    claim_numeric_literals,
    claim_numerics_present,
    passage_token_coverage,
    resolve_nli_add_mode,
)


# --- resolve_nli_add_mode: default OFF (not shadow — the NLI model is heavy) -- #


def test_nli_add_mode_default_is_off(monkeypatch):
    monkeypatch.delenv(ENV_NLI_ADD, raising=False)
    assert DEFAULT_NLI_ADD_MODE == MODE_OFF
    assert resolve_nli_add_mode() == MODE_OFF


@pytest.mark.parametrize("val,expected", [
    ("off", MODE_OFF), ("shadow", MODE_SHADOW), ("on", MODE_ON),
    ("ON", MODE_ON), ("  Shadow  ", MODE_SHADOW),
])
def test_nli_add_mode_recognized_values(monkeypatch, val, expected):
    monkeypatch.setenv(ENV_NLI_ADD, val)
    assert resolve_nli_add_mode() == expected


def test_nli_add_mode_garbage_falls_back_to_off(monkeypatch):
    monkeypatch.setenv(ENV_NLI_ADD, "garbage-typo")
    # A typo must land on the SAFE off, never silently shadow/on.
    assert resolve_nli_add_mode() == MODE_OFF


def test_nli_add_mode_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv(ENV_NLI_ADD, "on")
    assert resolve_nli_add_mode("off") == MODE_OFF


# --- numeric-literal guard (NLI is blind to numbers) ------------------------ #


def test_numerics_no_numerics_in_claim_passes_vacuously():
    assert claim_numerics_present("a claim with no digits", "also no digits") is True
    assert claim_numeric_literals("no digits here") == []


def test_numerics_currency_symbol_matches_bare_decimal():
    # "$1.50" in the claim must match a bare "1.50" in the chunk (normalized).
    assert claim_numerics_present(
        "the price of one pen is $1.50", "each pen sells for 1.50 dollars"
    ) is True


def test_numerics_decimal_must_match_exactly_not_rounded():
    # 27.78 must NOT match 27.8 — decimals are literal, never rounded.
    assert claim_numerics_present("the result is 27.78", "approximately 27.8") is False


def test_numerics_missing_literal_fails():
    # The hand-judged false add: "$1.50" absent from the chunk → guard fails.
    assert claim_numerics_present(
        "the price of one pen is $1.50",
        "this chapter covers buying pens and pencils",
    ) is False


def test_numerics_percent_and_thousands_separator_normalize():
    assert claim_numerics_present("50% of 1,234 items", "50 percent of 1234 items") is True
    assert sorted(claim_numeric_literals("50% off 1,234")) == ["1234", "50"]


def test_numerics_all_must_be_present():
    # Two numerics in the claim; only one in the chunk → fails (ALL required).
    assert claim_numerics_present("3 apples and 5 pears", "we have 3 apples") is False
    assert claim_numerics_present("3 apples and 5 pears", "3 apples, 5 pears") is True


# --- public token-coverage wrapper reuses the lexical machinery ------------- #


def test_passage_token_coverage_matches_lexical_definition():
    cov = passage_token_coverage(
        "add the numerators over the common denominator",
        "add numerators and place the result over the common denominator",
    )
    assert cov >= NLI_ADD_TOKEN_COVERAGE_FLOOR
    # Disjoint content tokens → zero coverage.
    assert passage_token_coverage("photosynthesis glucose", "vector store faiss") == 0.0


def test_criterion_constants_are_the_hand_judged_values():
    # The 2026-06-12 hand sample separated 4/4 genuine from 3/3 false adds at
    # exactly these thresholds; pin them so a silent drift is caught.
    assert NLI_ADD_ENTAILMENT_FLOOR == 0.75
    assert NLI_ADD_TOKEN_COVERAGE_FLOOR == 0.65
