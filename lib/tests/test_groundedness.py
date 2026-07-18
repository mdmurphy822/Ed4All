"""Tests for the WS3 groundedness scoring harness (D6).

CI-safe: a deterministic ``FakeNli`` is injected explicitly via the
``score_groundedness(nli=...)`` seam. ``lib/tests`` has no autouse NLI-stub
conftest (unlike Trainforge/MCP/LibV2), so every test that exercises the scored
path injects its fake; the degrade-path test monkeypatches the singleton loader
to return ``None``. One ``@pytest.mark.real_models`` smoke loads the real
DeBERTa (skipped without the weights/extras).
"""
from __future__ import annotations

import pytest

from lib.retrieval.answer_composer import RetrievedPassage
from lib.retrieval.groundedness import (
    ENV_REQUIRE_EMBEDDINGS,
    GroundednessReport,
    THRESHOLDS_PROVENANCE_DEFAULTS,
    VERDICT_CONTRADICTED,
    VERDICT_ENTAILED,
    VERDICT_UNSUPPORTED,
    score_groundedness,
    split_claims,
)
from lib.validators.pair._claim_support_thresholds import (
    _DEFAULT_CONTRADICTION_FLOOR,
    _DEFAULT_ENTAILMENT_FLOOR,
)


# ===========================================================================
# Deterministic fake NLI (no DeBERTa in CI)
# ===========================================================================

class _FakeNliScore:
    def __init__(self, entailment, neutral, contradiction):
        self.entailment = entailment
        self.neutral = neutral
        self.contradiction = contradiction


class FakeNli:
    """Deterministic NLI stand-in.

    Scores high entailment (0.9) when the hypothesis (claim) is a normalized
    case-insensitive substring of the premise (passage); high contradiction
    (0.6) when the claim contains the sentinel token ``CONTRADICTS``; else low
    (0.1) entailment, neutral. Records every batch for assertions.
    """

    _revision = "fake-nli-revision-0"

    def __init__(self):
        self.batches = []

    def score_batch(self, *, pairs, batch_size=8):
        self.batches.append(list(pairs))
        out = []
        for premise, hypothesis in pairs:
            h_norm = " ".join(hypothesis.lower().split())
            p_norm = " ".join(premise.lower().split())
            if "contradicts" in h_norm:
                out.append(_FakeNliScore(0.1, 0.3, 0.6))
            elif h_norm and h_norm in p_norm:
                out.append(_FakeNliScore(0.9, 0.05, 0.05))
            else:
                out.append(_FakeNliScore(0.1, 0.8, 0.1))
        return out


def _passage(chunk_id, text):
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=text,
        score=1.0,
        engine="lexical",
        item_path="alpha.html",
        section_heading="Sec",
        module_id="m1",
        source={"item_path": "alpha.html"},
    )


# ===========================================================================
# split_claims
# ===========================================================================

def test_split_claims_filters_short_sentences():
    text = "A vector store indexes embedding vectors for similarity search. Yes."
    claims = split_claims(text)
    # First sentence has enough content tokens; "Yes." does not.
    assert len(claims) == 1
    assert claims[0].startswith("A vector store indexes")


def test_split_claims_empty():
    assert split_claims("") == []
    assert split_claims("   ") == []


# ===========================================================================
# Abbreviation guard (2026-07-01 CO-13 false-CONTRADICTED regression)
# ===========================================================================

def test_split_claims_us_abbreviation_verbatim_co13():
    """The verbatim CO-13 statement must survive as ONE claim.

    The canonical ``_SENTENCE_SPLIT_RE`` split at the abbreviation period in
    "U.S.", beheading the statement into "…both U.S." + "and metric systems
    to solve real-world problems." — and NLI scored the beheaded fragment
    CONTRADICTED against a perfectly-supporting chunk (the observed
    OBJECTIVE_CONTRADICTED false failure on CO-13).
    """
    text = (
        "Apply the process of unit conversions in both U.S. and metric "
        "systems to solve real-world problems."
    )
    claims = split_claims(text)
    assert claims == [text]


def test_split_claims_abbreviation_variety():
    """Common abbreviations (e.g., i.e., vs., Dr., Fig. N, No. N) don't split."""
    cases = [
        "Solving equations is common, e.g. balancing chemical formulas today.",
        "Rational numbers, i.e. quotients of integers, form a dense set.",
        "Compare the substitution method vs. the elimination method carefully.",
        "Dr. Euler proved this identity using infinite series expansions.",
        "See Fig. 3 for the graph of the quadratic parent function.",
        "Problem No. 5 requires the distributive property applied twice.",
        "Convert between U.K. and metric units in the practice problems.",
    ]
    for text in cases:
        assert split_claims(text) == [text], text


def test_split_claims_us_at_sentence_end_still_splits():
    """Ambiguous enders (U.S., etc.) followed by a capitalized sentence keep
    the split — the guard only merges when the next fragment starts lowercase
    or with a digit."""
    text = (
        "The customary system is used in the U.S. The metric system is "
        "used almost everywhere else in the world."
    )
    claims = split_claims(text)
    assert len(claims) == 2
    assert claims[0].endswith("U.S.")
    assert claims[1].startswith("The metric system")


def test_split_claims_normal_two_sentence_split_unchanged():
    """The guard does not affect ordinary sentence boundaries."""
    text = (
        "A vector store indexes embedding vectors for similarity search. "
        "The retriever returns the top passages ranked by cosine score."
    )
    claims = split_claims(text)
    assert len(claims) == 2


# ===========================================================================
# split_claims=False (single-hypothesis path, ObjectiveEntailmentValidator)
# ===========================================================================

def test_score_groundedness_no_split_scores_single_hypothesis():
    """``split_claims=False`` scores the WHOLE text as one claim."""
    statement = (
        "Apply the process of unit conversions in both U.S. and metric "
        "systems to solve real-world problems."
    )
    nli = FakeNli()
    # Passage contains the full statement → FakeNli entails the whole-text
    # hypothesis. The load-bearing assertion is the HYPOTHESIS SET the NLI
    # saw: exactly one, equal to the full statement (no fragment scored).
    report = score_groundedness(
        statement,
        [_passage("c1", f"Unit conversions chapter. {statement} More text.")],
        nli=nli,
        split_claims=False,
    )
    assert report.available is True
    assert report.scored_count == 1
    assert len(report.claims) == 1
    assert report.claims[0].claim_text == statement
    assert report.claims[0].verdict == "entailed"
    hypotheses = {h for batch in nli.batches for (_p, h) in batch}
    assert hypotheses == {statement}


def test_score_groundedness_default_split_unchanged():
    """Default ``split_claims=True`` still splits multi-sentence answers."""
    answer = (
        "A vector store indexes embedding vectors for similarity search. "
        "The retriever returns the top passages ranked by cosine score."
    )
    nli = FakeNli()
    report = score_groundedness(
        answer,
        [_passage("c1", answer)],
        nli=nli,
    )
    assert report.scored_count == 2


# ===========================================================================
# Verdict mapping
# ===========================================================================

def test_entailed_claim_scores_grounded():
    passage = _passage(
        "c1",
        "A vector store indexes embedding vectors for nearest neighbour search.",
    )
    answer = "A vector store indexes embedding vectors for nearest neighbour search."
    fake = FakeNli()
    report = score_groundedness(answer, [passage], nli=fake)

    assert report.available is True
    assert report.scored_count == 1
    assert report.claims[0].verdict == VERDICT_ENTAILED
    assert report.claims[0].best_chunk_id == "c1"
    assert report.groundedness_rate == 1.0
    assert report.unsupported_count == 0
    assert report.contradicted_count == 0
    assert report.nli_model_revision == "fake-nli-revision-0"


def test_unsupported_claim():
    passage = _passage("c1", "The sky is blue and the grass is green today.")
    answer = "A completely unrelated statement about quantum chromodynamics here."
    report = score_groundedness(answer, [passage], nli=FakeNli())
    assert report.available is True
    assert report.scored_count == 1
    assert report.claims[0].verdict == VERDICT_UNSUPPORTED
    assert report.unsupported_count == 1
    assert report.groundedness_rate == 0.0


def test_contradicted_claim():
    passage = _passage("c1", "Embeddings are dense numerical vectors here.")
    answer = "This statement CONTRADICTS the cited passage about embeddings."
    report = score_groundedness(answer, [passage], nli=FakeNli())
    assert report.claims[0].verdict == VERDICT_CONTRADICTED
    assert report.contradicted_count == 1
    assert report.unsupported_count == 0
    assert report.groundedness_rate == 0.0


def test_max_entailment_over_passages_picks_best_chunk():
    p1 = _passage("c1", "Irrelevant passage about something else entirely now.")
    p2 = _passage(
        "c2",
        "Retrieval quality is commonly measured with recall at k metrics.",
    )
    answer = "Retrieval quality is commonly measured with recall at k metrics."
    report = score_groundedness(answer, [p1, p2], nli=FakeNli())
    assert report.claims[0].verdict == VERDICT_ENTAILED
    assert report.claims[0].best_chunk_id == "c2"


def test_rate_math_mixed_claims():
    passage = _passage(
        "c1",
        "A vector store indexes embedding vectors for similarity search here.",
    )
    answer = (
        "A vector store indexes embedding vectors for similarity search here. "
        "An unrelated fact about marine biology and deep sea creatures here."
    )
    report = score_groundedness(answer, [passage], nli=FakeNli())
    assert report.scored_count == 2
    assert report.groundedness_rate == 0.5


def test_thresholds_recorded_with_provenance():
    passage = _passage("c1", "Some passage text with enough words to be valid.")
    report = score_groundedness("Some passage text with enough words.", [passage], nli=FakeNli())
    assert report.thresholds == {
        "entailment_floor": _DEFAULT_ENTAILMENT_FLOOR,
        "contradiction_floor": _DEFAULT_CONTRADICTION_FLOOR,
    }
    assert report.thresholds_provenance == THRESHOLDS_PROVENANCE_DEFAULTS


def test_custom_thresholds_passed_through():
    passage = _passage("c1", "Some passage text with enough words to be valid.")
    report = score_groundedness(
        "Some passage text with enough words.",
        [passage],
        nli=FakeNli(),
        entailment_floor=0.95,
        contradiction_floor=0.4,
    )
    assert report.thresholds["entailment_floor"] == 0.95
    assert report.thresholds["contradiction_floor"] == 0.4


def test_no_claims_or_no_passages_available_true_rate_zero():
    # No passages → available True (NLI present) but nothing scored.
    report = score_groundedness("A scorable claim with enough tokens here.", [], nli=FakeNli())
    assert report.available is True
    assert report.scored_count == 0
    assert report.groundedness_rate == 0.0
    # Empty answer → no claims.
    report2 = score_groundedness("", [_passage("c1", "text here")], nli=FakeNli())
    assert report2.available is True
    assert report2.scored_count == 0


def test_batches_one_pass_grid():
    # Use passages that DO entail the claims so stage-1 clears the floor and the
    # stage-2 windowed rescue never fires (so this stays a single-batch grid).
    p1 = _passage("c1", "First scorable claim sentence with enough content tokens here.")
    p2 = _passage("c2", "Second scorable claim sentence with enough content tokens here.")
    answer = (
        "First scorable claim sentence with enough content tokens here. "
        "Second scorable claim sentence with enough content tokens here."
    )
    fake = FakeNli()
    report = score_groundedness(answer, [p1, p2], nli=fake)
    # 2 claims x 2 passages = 4 pairs, all in one stage-1 score_batch call; both
    # claims entailed in stage 1 so the stage-2 rescue pass is skipped entirely.
    assert report.groundedness_rate == 1.0
    assert len(fake.batches) == 1
    assert len(fake.batches[0]) == 4


def test_report_to_dict_shape():
    passage = _passage("c1", "A vector store indexes embedding vectors here.")
    report = score_groundedness(
        "A vector store indexes embedding vectors here.", [passage], nli=FakeNli()
    )
    d = report.to_dict()
    # v2 is ADDITIVE: every pre-existing key must still be present (same names)
    # alongside the new v2 keys (computational_count / filtered_count /
    # scorer_version on the report; windowed / best_chunk_cited on the claim).
    v1_report_keys = {
        "available", "claims", "groundedness_rate", "unsupported_count",
        "contradicted_count", "scored_count", "thresholds",
        "thresholds_provenance", "nli_model_revision", "reason",
    }
    assert v1_report_keys <= set(d)
    assert set(d) == v1_report_keys | {
        "computational_count", "filtered_count", "scorer_version",
        "nli_device",
    }
    v1_claim_keys = {
        "claim_text", "verdict", "entailment", "contradiction", "best_chunk_id",
    }
    assert v1_claim_keys <= set(d["claims"][0])
    assert set(d["claims"][0]) == v1_claim_keys | {"windowed", "best_chunk_cited"}


# ===========================================================================
# Degrade path + strict mode
# ===========================================================================

def test_degrade_when_nli_unavailable(monkeypatch):
    # nli=None and the singleton loader returns None → available=False.
    from lib.classifiers import nli_classifier

    monkeypatch.setattr(
        nli_classifier.NliClassifier, "get_or_load", classmethod(lambda cls: None)
    )
    passage = _passage("c1", "A passage with text and enough words to score it.")
    report = score_groundedness(
        "A scorable claim with enough content tokens here.", [passage], nli=None
    )
    assert report.available is False
    assert report.reason == "nli_unavailable"
    assert report.claims == []
    assert report.groundedness_rate == 0.0
    # Thresholds still recorded even on degrade (audit pin).
    assert report.thresholds_provenance == THRESHOLDS_PROVENANCE_DEFAULTS


def test_strict_mode_raises_when_nli_absent(monkeypatch):
    from lib.classifiers import nli_classifier

    monkeypatch.setattr(
        nli_classifier.NliClassifier, "get_or_load", classmethod(lambda cls: None)
    )
    monkeypatch.setenv(ENV_REQUIRE_EMBEDDINGS, "true")
    passage = _passage("c1", "A passage with text and enough words to score it.")
    with pytest.raises(RuntimeError) as exc:
        score_groundedness(
            "A scorable claim with enough content tokens here.",
            [passage],
            nli=None,
        )
    assert "DeBERTa" in str(exc.value) or "NLI" in str(exc.value)


def test_strict_mode_does_not_raise_when_nli_injected(monkeypatch):
    # Strict flag set but an NLI is injected → no raise, scores normally.
    monkeypatch.setenv(ENV_REQUIRE_EMBEDDINGS, "1")
    passage = _passage("c1", "A vector store indexes embedding vectors here.")
    report = score_groundedness(
        "A vector store indexes embedding vectors here.", [passage], nli=FakeNli()
    )
    assert report.available is True
    assert report.scored_count == 1


# ===========================================================================
# v2 — claim-artifact filtering
# ===========================================================================

def test_artifact_filtering_drops_dict_literal_and_enum_stub():
    from lib.retrieval.groundedness import (
        VERDICT_ENTAILED,
        _split_claims_for_scoring,
    )

    # A normal claim, a dict-literal claim-splitter artifact, and an enumeration
    # stub. Only the normal claim survives; filtered_count counts the 2 dropped.
    answer = (
        "A vector store indexes embedding vectors for similarity search here. "
        "{'lower_bound': 25, 'upper_bound': 28}. "
        "The steps to compute the range are as follows: 1."
    )
    kept, filtered = _split_claims_for_scoring(answer)
    assert filtered == 2
    assert len(kept) == 1
    assert kept[0].startswith("A vector store indexes")

    passage = _passage(
        "c1",
        "A vector store indexes embedding vectors for similarity search here.",
    )
    report = score_groundedness(answer, [passage], nli=FakeNli())
    assert report.filtered_count == 2
    # The two artifacts are gone; only the one real claim is scored.
    assert report.scored_count == 1
    assert report.claims[0].verdict == VERDICT_ENTAILED


def test_artifact_filtering_keeps_normal_sentences():
    from lib.retrieval.groundedness import _split_claims_for_scoring

    answer = (
        "Embeddings are dense numerical vectors used for retrieval here. "
        "Cosine similarity compares the angle between two such vectors."
    )
    kept, filtered = _split_claims_for_scoring(answer)
    assert filtered == 0
    assert len(kept) == 2


# ===========================================================================
# v2 — computational-claim exemption
# ===========================================================================

class _CountingFakeNli(FakeNli):
    """FakeNli that also records the total number of pairs ever scored."""

    def __init__(self):
        super().__init__()
        self.total_pairs = 0

    def score_batch(self, *, pairs, batch_size=8):
        self.total_pairs += len(list(pairs))
        return super().score_batch(pairs=pairs, batch_size=batch_size)


def test_computational_claim_exempted_from_nli_and_denominator():
    from lib.retrieval.groundedness import VERDICT_COMPUTATIONAL

    passage = _passage(
        "c1",
        "Combine the like terms to find the simplified algebraic expression.",
    )
    # A single computational claim — must never reach NLI.
    answer = "The simplified form is 3x + 10y + 4."
    fake = _CountingFakeNli()
    report = score_groundedness(answer, [passage], nli=fake)

    assert report.computational_count == 1
    assert report.claims[0].verdict == VERDICT_COMPUTATIONAL
    # Excluded from the rate denominator and the un/contra counts.
    assert report.scored_count == 0
    assert report.unsupported_count == 0
    assert report.contradicted_count == 0
    assert report.groundedness_rate == 0.0
    # NLI was never called for the computational claim (no pairs scored at all).
    assert fake.total_pairs == 0
    assert fake.batches == []


def test_computational_claim_mixed_with_scorable():
    from lib.retrieval.groundedness import (
        VERDICT_COMPUTATIONAL,
        VERDICT_ENTAILED,
    )

    passage = _passage(
        "c1",
        "A vector store indexes embedding vectors for similarity search here.",
    )
    answer = (
        "A vector store indexes embedding vectors for similarity search here. "
        "The simplified form is 3x + 10y + 4."
    )
    report = score_groundedness(answer, [passage], nli=FakeNli())
    # One scorable (entailed) + one computational (exempt). Order preserved.
    assert report.scored_count == 1
    assert report.computational_count == 1
    assert report.groundedness_rate == 1.0
    assert report.claims[0].verdict == VERDICT_ENTAILED
    assert report.claims[1].verdict == VERDICT_COMPUTATIONAL


# ===========================================================================
# v2 — windowed rescue pass
# ===========================================================================

class _WindowRescueNli:
    """NLI fake that returns LOW entailment for long premises and HIGH for a
    short window containing the support sentence — the glossary-chunk failure
    mode the stage-2 windowed rescue targets.
    """

    _revision = "fake-window-nli-0"

    SUPPORT = "recall at k measures retrieval completeness"
    #: A premise longer than this is treated as a "whole multi-topic chunk" the
    #: whole-chunk NLI cannot resolve; a tighter (shorter) window scores high.
    MAX_RESOLVABLE_CHARS = 130

    def __init__(self):
        self.batches = []

    def score_batch(self, *, pairs, batch_size=8):
        self.batches.append(list(pairs))
        out = []
        for premise, _hypothesis in pairs:
            p_norm = " ".join(premise.lower().split())
            contains = self.SUPPORT in p_norm
            # Support present AND the premise is tight enough to resolve → high
            # entailment; the long whole-chunk premise (over the char budget)
            # scores low even though it contains the support (glossary noise).
            if contains and len(premise) <= self.MAX_RESOLVABLE_CHARS:
                out.append(_FakeNliScore(0.9, 0.05, 0.05))
            else:
                out.append(_FakeNliScore(0.15, 0.8, 0.05))
        return out


def test_windowed_rescue_entails_via_sentence_window():
    from lib.retrieval.groundedness import VERDICT_ENTAILED

    # A glossary-style multi-topic chunk: the support sentence is buried among
    # several unrelated sentences, so the whole-chunk premise scores low but a
    # 3-sentence window isolating it scores high.
    glossary = (
        "Embeddings are dense numerical vectors. "
        "A vector store indexes them for search. "
        "Recall at k measures retrieval completeness. "
        "Precision at k measures retrieval correctness. "
        "Latency is the time to answer a query."
    )
    passage = _passage("c1", glossary)
    answer = "Recall at k measures retrieval completeness."
    nli = _WindowRescueNli()
    report = score_groundedness(answer, [passage], nli=nli)

    assert report.scored_count == 1
    assert report.claims[0].verdict == VERDICT_ENTAILED
    assert report.claims[0].windowed is True
    assert report.claims[0].best_chunk_id == "c1"
    # Two batches: stage-1 whole-chunk grid + stage-2 windowed rescue.
    assert len(nli.batches) == 2


# ===========================================================================
# v2 — wider evidence pool + cited flag
# ===========================================================================

def test_best_chunk_cited_flag_set_when_cited_ids_supplied():
    p1 = _passage("c1", "Irrelevant passage about something else entirely now.")
    p2 = _passage(
        "c2",
        "Retrieval quality is commonly measured with recall at k metrics.",
    )
    answer = "Retrieval quality is commonly measured with recall at k metrics."
    # The supporting chunk (c2) is NOT in the cited set → best_chunk_cited False.
    report = score_groundedness(
        answer, [p1, p2], nli=FakeNli(), cited_chunk_ids={"c1"}
    )
    assert report.claims[0].best_chunk_id == "c2"
    assert report.claims[0].best_chunk_cited is False

    # Now c2 IS cited → True.
    report2 = score_groundedness(
        answer, [p1, p2], nli=FakeNli(), cited_chunk_ids={"c1", "c2"}
    )
    assert report2.claims[0].best_chunk_cited is True


def test_best_chunk_cited_none_when_kwarg_omitted():
    passage = _passage(
        "c1",
        "A vector store indexes embedding vectors for similarity search here.",
    )
    answer = "A vector store indexes embedding vectors for similarity search here."
    report = score_groundedness(answer, [passage], nli=FakeNli())
    assert report.claims[0].best_chunk_cited is None


# ===========================================================================
# v2 — report version stamp
# ===========================================================================

def test_report_carries_scorer_version_two():
    from lib.retrieval.groundedness import SCORER_VERSION

    passage = _passage("c1", "Some passage text with enough words to be valid.")
    report = score_groundedness(
        "Some passage text with enough words.", [passage], nli=FakeNli()
    )
    assert report.scorer_version == "2"
    assert SCORER_VERSION == "2"
    assert report.to_dict()["scorer_version"] == "2"


# ===========================================================================
# W1.8 — ED4ALL_GROUNDEDNESS_COMPUTATIONAL numeric-grounding check
# ===========================================================================

from lib.retrieval.groundedness import (  # noqa: E402
    ENV_GROUNDEDNESS_COMPUTATIONAL,
    VERDICT_COMPUTATIONAL,
    resolve_groundedness_computational,
)

# A computational sentence with >= 4 content tokens (survives split_claims) whose
# numeric literals are 3, 10 and 27.
_COMP_CLAIM = "The computed sum is 3 plus 10 equals 27 for this problem."


def test_resolve_computational_flag_parse_with_fallback():
    assert resolve_groundedness_computational({}) is False
    assert resolve_groundedness_computational(
        {ENV_GROUNDEDNESS_COMPUTATIONAL: "garbage"}
    ) is False
    for tok in ("1", "true", "YES", "On"):
        assert resolve_groundedness_computational(
            {ENV_GROUNDEDNESS_COMPUTATIONAL: tok}
        ) is True


def test_computational_check_absent_when_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_GROUNDEDNESS_COMPUTATIONAL, raising=False)
    passage = _passage("c1", "Add the numbers 3 and 10 in this worked section.")
    report = score_groundedness(_COMP_CLAIM, [passage], nli=FakeNli())
    # Historical exemption byte-identical: no block written, verdict unchanged,
    # denominator untouched.
    assert report.computational_numeric_check is None
    assert "computational_numeric_check" not in report.to_dict()
    assert report.computational_count == 1
    assert report.scored_count == 0
    assert report.claims[0].verdict == VERDICT_COMPUTATIONAL


def test_computational_check_flags_ungrounded_numeric_when_on(monkeypatch):
    monkeypatch.setenv(ENV_GROUNDEDNESS_COMPUTATIONAL, "1")
    # Cited chunk carries 3 and 10 but NOT the derived result 27.
    passage = _passage("c1", "Add the numbers 3 and 10 in this worked section.")
    report = score_groundedness(_COMP_CLAIM, [passage], nli=FakeNli())
    block = report.computational_numeric_check
    assert block is not None
    assert block["enabled"] is True
    assert block["computational_claims_checked"] == 1
    assert block["claims_with_ungrounded_numeric"] == 1
    assert block["ungrounded_numeric_count"] == 1
    assert block["claims"][0]["ungrounded_numeric_literals"] == ["27"]
    assert block["claims"][0]["grounded"] is False
    # The block round-trips through to_dict only when present.
    assert "computational_numeric_check" in report.to_dict()
    # WARNING-ONLY: verdict + denominator unchanged vs the off path.
    assert report.claims[0].verdict == VERDICT_COMPUTATIONAL
    assert report.scored_count == 0
    assert report.groundedness_rate == 0.0


def test_computational_check_grounded_when_numerics_in_chunk(monkeypatch):
    monkeypatch.setenv(ENV_GROUNDEDNESS_COMPUTATIONAL, "on")
    # Cited chunk carries all three literals 3, 10 and 27.
    passage = _passage(
        "c1", "The section states 3, 10 and 27 as the relevant example values."
    )
    report = score_groundedness(_COMP_CLAIM, [passage], nli=FakeNli())
    block = report.computational_numeric_check
    assert block is not None
    assert block["claims_with_ungrounded_numeric"] == 0
    assert block["ungrounded_numeric_count"] == 0
    assert block["claims"][0]["grounded"] is True


def test_computational_check_restricts_to_cited_chunk(monkeypatch):
    monkeypatch.setenv(ENV_GROUNDEDNESS_COMPUTATIONAL, "true")
    # The literal 27 lives ONLY in an UNCITED passage; restricting to the cited
    # chunk makes it ungrounded (the check is about CITED support).
    cited = _passage("c1", "Add the numbers 3 and 10 in this worked section.")
    uncited = _passage("c2", "Elsewhere the value 27 appears out of context.")
    report = score_groundedness(
        _COMP_CLAIM, [cited, uncited], nli=FakeNli(), cited_chunk_ids={"c1"}
    )
    block = report.computational_numeric_check
    assert block["cited_pool_size"] == 1
    assert block["claims"][0]["ungrounded_numeric_literals"] == ["27"]


def test_computational_check_never_flips_scorable_verdict(monkeypatch):
    monkeypatch.setenv(ENV_GROUNDEDNESS_COMPUTATIONAL, "1")
    passage = _passage(
        "c1",
        "A vector store indexes embedding vectors for similarity search here.",
    )
    answer = (
        "A vector store indexes embedding vectors for similarity search here. "
        + _COMP_CLAIM
    )
    report = score_groundedness(answer, [passage], nli=FakeNli())
    # The scorable claim is still entailed; groundedness_rate reflects ONLY the
    # scorable claim (the computational one stays out of the denominator).
    assert report.scored_count == 1
    assert report.groundedness_rate == 1.0
    assert report.claims[0].verdict == VERDICT_ENTAILED
    assert report.claims[1].verdict == VERDICT_COMPUTATIONAL
    # The numeric check ran on the one computational claim.
    assert report.computational_numeric_check["computational_claims_checked"] == 1


class _RecordingCapture:
    """Minimal DecisionCapture stand-in recording log_decision calls."""

    def __init__(self):
        self.calls = []

    def log_decision(self, *, decision_type, decision, rationale, **kwargs):
        self.calls.append(
            {"decision_type": decision_type, "decision": decision,
             "rationale": rationale}
        )


def test_capture_fires_when_flag_on(monkeypatch):
    monkeypatch.setenv(ENV_GROUNDEDNESS_COMPUTATIONAL, "1")
    cap = _RecordingCapture()
    passage = _passage("c1", "Add the numbers 3 and 10 in this worked section.")
    score_groundedness(_COMP_CLAIM, [passage], nli=FakeNli(), capture=cap)
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["decision_type"] == "groundedness_computational_check"
    assert len(call["rationale"]) >= 20


def test_capture_silent_when_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_GROUNDEDNESS_COMPUTATIONAL, raising=False)
    cap = _RecordingCapture()
    passage = _passage("c1", "Add the numbers 3 and 10 in this worked section.")
    score_groundedness(_COMP_CLAIM, [passage], nli=FakeNli(), capture=cap)
    assert cap.calls == []


def test_capture_silent_when_no_computational_claim(monkeypatch):
    monkeypatch.setenv(ENV_GROUNDEDNESS_COMPUTATIONAL, "1")
    cap = _RecordingCapture()
    passage = _passage(
        "c1",
        "A vector store indexes embedding vectors for similarity search here.",
    )
    # No computational claim → nothing to check → no decision, no block.
    report = score_groundedness(
        "A vector store indexes embedding vectors for similarity search here.",
        [passage],
        nli=FakeNli(),
        capture=cap,
    )
    assert cap.calls == []
    assert report.computational_numeric_check is None


# ===========================================================================
# Opt-in real-model DeBERTa smoke (skipped without weights/extras)
# ===========================================================================

@pytest.mark.real_models
def test_real_deberta_verdict_ordering():
    from lib.classifiers.nli_classifier import NliClassifier

    nli = NliClassifier.get_or_load()
    if nli is None:
        pytest.skip("NLI model/extras unavailable (uncached or not installed)")

    passage = _passage(
        "c1",
        "Paris is the capital of France and its largest city by population.",
    )
    supported = score_groundedness(
        "Paris is the capital of France.", [passage], nli=nli
    )
    contradicted = score_groundedness(
        "Paris is not the capital of France.", [passage], nli=nli
    )
    assert supported.available is True
    assert supported.claims[0].entailment > contradicted.claims[0].entailment
    assert supported.claims[0].verdict == VERDICT_ENTAILED


# --------------------------------------------------------------------------- #
# ED4ALL_GROUNDEDNESS_S1_TOPK — top-K windowed stage 1 (opt-in)
# --------------------------------------------------------------------------- #

class _FakeS1Embedder:
    """Deterministic embedder: vector = [len(text) % 7, 1.0] (normalized later)."""

    def encode_batch(self, texts, normalize=True):
        return [[(len(t) % 7) + 1.0, 1.0] for t in texts]


def test_s1_topk_resolver_parse_with_fallback(monkeypatch):
    from lib.retrieval.groundedness import resolve_groundedness_s1_topk

    assert resolve_groundedness_s1_topk({}) == 0
    assert resolve_groundedness_s1_topk({"ED4ALL_GROUNDEDNESS_S1_TOPK": ""}) == 0
    assert resolve_groundedness_s1_topk({"ED4ALL_GROUNDEDNESS_S1_TOPK": "garbage"}) == 0
    assert resolve_groundedness_s1_topk({"ED4ALL_GROUNDEDNESS_S1_TOPK": "-3"}) == 0
    assert resolve_groundedness_s1_topk({"ED4ALL_GROUNDEDNESS_S1_TOPK": "16"}) == 16


def test_s1_topk_off_is_legacy_grid(monkeypatch):
    """Flag unset → pair volume equals the full claims × passages grid."""
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_S1_TOPK", raising=False)
    nli = FakeNli()
    from lib.retrieval.groundedness import score_groundedness

    passages = [
        {"chunk_id": "c1", "text": "The sky is blue. Water is wet. Grass is green."},
        {"chunk_id": "c2", "text": "Cats purr. Dogs bark. Fish swim."},
    ]
    report = score_groundedness("The sky is blue.", passages, nli=nli)
    assert report.available is True
    # 1 scorable claim × 2 whole-chunk passages (+ any stage-2 rescue pairs).
    assert len([pr for b in nli.batches for pr in b]) >= 2
    assert any(p[0].startswith("The sky is blue.") for p in [pr for b in nli.batches for pr in b])


def test_s1_topk_on_scores_windows_and_caps_pairs(monkeypatch):
    """Flag on + injected embedder → stage-1 premises are WINDOWS, ≤K per claim."""
    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_S1_TOPK", "2")
    import lib.retrieval.groundedness as g

    monkeypatch.setattr(
        "lib.embedding.sentence_embedder.try_load_embedder",
        lambda *a, **k: _FakeS1Embedder(),
    )
    nli = FakeNli()
    long_a = " ".join(f"The sky is blue on day {i}." for i in range(12))
    long_b = " ".join(f"Cats purr at hour {i}." for i in range(12))
    passages = [
        {"chunk_id": "c1", "text": long_a},
        {"chunk_id": "c2", "text": long_b},
    ]
    report = g.score_groundedness("The sky is blue on day 3.", passages, nli=nli)
    assert report.available is True
    # Stage-1 pairs capped at K=2 per claim; premises are windows (short),
    # never the full 12-sentence passage. Rescue may add widened pairs.
    s1_premises = [p[0] for p in [pr for b in nli.batches for pr in b]]
    assert all(len(prem) < len(long_a) for prem in s1_premises)
    # Verdict still folds to a real chunk id.
    scored = [c for c in report.claims if c.verdict != "computational_unverified"]
    assert scored and scored[0].best_chunk_id in {"c1", "c2"}


def test_s1_topk_degrades_to_legacy_without_embedder(monkeypatch):
    """Embedder unavailable → silently falls back to the legacy whole-chunk grid."""
    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_S1_TOPK", "4")
    import lib.retrieval.groundedness as g

    monkeypatch.setattr(
        "lib.embedding.sentence_embedder.try_load_embedder",
        lambda *a, **k: None,
    )
    nli = FakeNli()
    passages = [{"chunk_id": "c1", "text": "The sky is blue. Water is wet."}]
    report = g.score_groundedness("The sky is blue.", passages, nli=nli)
    assert report.available is True
    # Legacy grid premise = whole passage text.
    assert any(p[0] == "The sky is blue. Water is wet." for p in [pr for b in nli.batches for pr in b])


def test_s1_topk_changes_prose_entailment_fingerprint(monkeypatch):
    """Sidecar fingerprints must differ between legacy and top-K scoring."""
    from lib.validators.block_prose_entailment import _block_fingerprint

    class _NliStub:
        device = "cpu"
        _revision = "rev1"

    passages = [{"chunk_id": "c1", "text": "alpha"}]
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_S1_TOPK", raising=False)
    fp_legacy = _block_fingerprint("prose", passages, 0.7, 0.5, _NliStub())
    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_S1_TOPK", "16")
    fp_topk = _block_fingerprint("prose", passages, 0.7, 0.5, _NliStub())
    assert fp_legacy != fp_topk
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_S1_TOPK", raising=False)
    assert _block_fingerprint("prose", passages, 0.7, 0.5, _NliStub()) == fp_legacy


# ===========================================================================
# FRONTIER stage-1 (ED4ALL_GROUNDEDNESS_FRONTIER) — parity + early-stop
# ===========================================================================
#
# The frontier scores each claim against its passage pool in a priority order
# and retires it the instant a premise entails it. Parity argument (enforced by
# the tests below): entailed claims produce an identical VERDICT (best-passage
# metadata may differ); non-entailed claims exhaust the WHOLE pool so their
# (entailment, contradiction, best_chunk_id) — under the legacy passage-index
# tiebreak — are bitwise-identical to the grid, and the stage-2 rescue set is
# unchanged. All tests inject a deterministic fake NLI and force the cosine
# tiebreaker off (``_frontier_embed -> (None, None)``) so no embedding model
# loads (lexical-only ordering; cosine never affects a verdict).


import hashlib as _hashlib


class _CountingBatchesNli(FakeNli):
    """FakeNli recording every score_batch pair count (across rounds)."""

    def __init__(self):
        super().__init__()
        self.total_pairs = 0

    def score_batch(self, *, pairs, batch_size=8):
        p = list(pairs)
        self.total_pairs += len(p)
        return super().score_batch(pairs=p, batch_size=batch_size)


class _VaryingNli:
    """Deterministic NLI whose entailment/contradiction vary per (premise, claim).

    Entailment is capped BELOW the 0.7 floor so every claim is non-entailed —
    the case where the frontier must reproduce the grid's argmax (ent, con,
    best_chunk_id) exactly. Pseudo-scores are hash-derived, so exact ties are
    effectively impossible and the argmax is unique.
    """

    _revision = "fake-varying-nli-0"
    device = "cpu"

    def __init__(self):
        self.total_pairs = 0

    def score_batch(self, *, pairs, batch_size=8):
        out = []
        for premise, hyp in pairs:
            self.total_pairs += 1
            d = _hashlib.sha256((premise[:64] + "|" + hyp[:64]).encode()).digest()
            ent = (d[0] / 255.0) * 0.6  # in [0, 0.6) — never reaches the floor
            con = d[1] / 255.0
            out.append(_FakeNliScore(ent, max(0.0, 1.0 - ent - con), con))
        return out


def _no_embed(monkeypatch):
    """Force the frontier's cosine tiebreaker off (no model load)."""
    monkeypatch.setattr(
        "lib.retrieval.groundedness._frontier_embed",
        lambda passage_texts, claim_texts: (None, None),
    )


def _verdict_tuples(report):
    return [
        (c.claim_text, c.verdict, round(c.entailment, 6), round(c.contradiction, 6),
         c.best_chunk_id, c.windowed)
        for c in report.claims
    ]


# --------------------------------------------------------------------------- #
# (f) resolver parse-with-fallback
# --------------------------------------------------------------------------- #

def test_frontier_resolver_parse_with_fallback():
    from lib.retrieval.groundedness import (
        resolve_groundedness_frontier,
        resolve_groundedness_frontier_width,
    )

    assert resolve_groundedness_frontier({}) is False
    assert resolve_groundedness_frontier({"ED4ALL_GROUNDEDNESS_FRONTIER": ""}) is False
    assert resolve_groundedness_frontier({"ED4ALL_GROUNDEDNESS_FRONTIER": "garbage"}) is False
    for tok in ("1", "true", "yes", "on", "ON", "Yes"):
        assert resolve_groundedness_frontier({"ED4ALL_GROUNDEDNESS_FRONTIER": tok}) is True

    assert resolve_groundedness_frontier_width({}) == 4
    assert resolve_groundedness_frontier_width({"ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH": "8"}) == 8
    assert resolve_groundedness_frontier_width({"ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH": "0"}) == 4
    assert resolve_groundedness_frontier_width({"ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH": "-2"}) == 4
    assert resolve_groundedness_frontier_width({"ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH": "nope"}) == 4


# --------------------------------------------------------------------------- #
# (a) frontier off => legacy grid byte-identical
# --------------------------------------------------------------------------- #

def test_frontier_off_builds_full_legacy_grid(monkeypatch):
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_FRONTIER", raising=False)
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_S1_TOPK", raising=False)
    passages = [
        _passage("c0", "Alpha content sentence one here today."),
        _passage("c1", "Beta content sentence two here today."),
        _passage("c2", "Gamma content sentence three here today."),
    ]
    answer = (
        "The first scorable claim mentions something here. "
        "The second scorable claim mentions another thing."
    )
    nli = FakeNli()
    score_groundedness(answer, passages, nli=nli)
    # Legacy stage-1 builds the FULL claim-major grid in the first batch.
    assert len(nli.batches) >= 1
    grid = nli.batches[0]
    # 2 claims x 3 passages = 6 pairs, claim-major, passage order.
    expected = [
        (p.text, claim)
        for claim in [
            "The first scorable claim mentions something here.",
            "The second scorable claim mentions another thing.",
        ]
        for p in passages
    ]
    assert grid == expected


# --------------------------------------------------------------------------- #
# (c) early-stop actually stops (pairs < full grid)
# --------------------------------------------------------------------------- #

def test_frontier_early_stop_scores_fewer_pairs(monkeypatch):
    _no_embed(monkeypatch)
    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_FRONTIER", "1")
    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH", "4")
    # 10 passages; exactly one entails the claim AND shares the most tokens
    # (so candidate-ordering floats it into the first round), the rest are
    # unrelated filler. Frontier retires in round 1 (<= 4 pairs) vs 10 grid.
    claim = "vector store indexes embedding vectors similarity search here"
    passages = [_passage(f"f{i}", f"unrelated filler passage number {i} words") for i in range(9)]
    passages.append(_passage("hit", "vector store indexes embedding vectors similarity search here. More tail text."))
    nli = _CountingBatchesNli()
    report = score_groundedness(claim + ".", passages, nli=nli)
    assert report.claims[0].verdict == VERDICT_ENTAILED
    assert nli.total_pairs <= 4
    assert nli.total_pairs < len(passages)  # strictly fewer than the full grid


# --------------------------------------------------------------------------- #
# (b) verdict parity vs legacy on a synthetic corpus
# --------------------------------------------------------------------------- #

def _run_both(monkeypatch, answer, passages, nli_factory, **kw):
    """Score once legacy, once frontier (same fake), return (legacy, frontier)."""
    _no_embed(monkeypatch)
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_FRONTIER", raising=False)
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_S1_TOPK", raising=False)
    legacy = score_groundedness(answer, passages, nli=nli_factory(), **kw)
    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_FRONTIER", "1")
    frontier = score_groundedness(answer, passages, nli=nli_factory(), **kw)
    return legacy, frontier


def _assert_verdict_parity(legacy, frontier):
    lc, fc = legacy.claims, frontier.claims
    assert [c.claim_text for c in lc] == [c.claim_text for c in fc]
    for lcl, fcl in zip(lc, fc):
        # Verdict is identical for EVERY claim.
        assert lcl.verdict == fcl.verdict, (lcl.claim_text, lcl.verdict, fcl.verdict)
        if fcl.verdict != VERDICT_ENTAILED:
            # Non-entailed claims exhaust the whole pool => (ent, con, best_chunk_id)
            # bitwise-identical to the grid (legacy passage-index tiebreak).
            assert round(lcl.entailment, 6) == round(fcl.entailment, 6)
            assert round(lcl.contradiction, 6) == round(fcl.contradiction, 6)
            assert lcl.best_chunk_id == fcl.best_chunk_id
            assert lcl.windowed == fcl.windowed
    assert legacy.groundedness_rate == frontier.groundedness_rate
    assert legacy.scored_count == frontier.scored_count
    assert legacy.contradicted_count == frontier.contradicted_count
    assert legacy.unsupported_count == frontier.unsupported_count


def test_frontier_verdict_parity_mixed_corpus(monkeypatch):
    passages = [
        _passage("c0", "Alpha beta gamma delta unrelated content one here."),
        _passage("c1", "A vector store indexes embedding vectors for similarity search here."),
        _passage("c2", "Recall measures retrieval completeness across ranking systems today."),
        _passage("c3", "Some entirely different unrelated material lives here now."),
    ]
    answer = (
        "A vector store indexes embedding vectors for similarity search here. "
        "Quantum entanglement links distant particles instantly across the void forever. "
        "This statement CONTRADICTS the cited source material entirely today here."
    )
    legacy, frontier = _run_both(monkeypatch, answer, passages, FakeNli)
    # Sanity: the three claims cover entailed / unsupported / contradicted.
    verdicts = [c.verdict for c in legacy.claims]
    assert VERDICT_ENTAILED in verdicts
    assert VERDICT_UNSUPPORTED in verdicts
    assert VERDICT_CONTRADICTED in verdicts
    _assert_verdict_parity(legacy, frontier)


def test_frontier_verdict_parity_rescue_path(monkeypatch):
    # Whole-chunk premise scores low; stage-2 windowed rescue entails. The
    # frontier exhausts the pool (no stage-1 entail) then runs the UNCHANGED
    # legacy rescue, so the rescued verdict is identical.
    glossary = (
        "Embeddings are dense numerical vectors. "
        "A vector store indexes them for search. "
        "Recall at k measures retrieval completeness. "
        "Precision at k measures retrieval correctness. "
        "Latency is the time to answer a query."
    )
    passages = [_passage("c1", glossary)]
    answer = "Recall at k measures retrieval completeness."
    legacy, frontier = _run_both(monkeypatch, answer, passages, _WindowRescueNli)
    assert legacy.claims[0].verdict == VERDICT_ENTAILED
    assert legacy.claims[0].windowed is True
    _assert_verdict_parity(legacy, frontier)
    assert frontier.claims[0].windowed is True


# --------------------------------------------------------------------------- #
# (d) exhausted claim's (ent, con) equals the legacy argmax
# --------------------------------------------------------------------------- #

def test_frontier_exhausted_claim_matches_legacy_argmax(monkeypatch):
    # Varying scores, all below the entailment floor => every claim exhausts its
    # pool. The frontier's fold must reproduce the grid's unique argmax exactly.
    passages = [_passage(f"c{i}", f"passage body number {i} distinct tokens {i*7}") for i in range(12)]
    answer = (
        "First non entailing claim with several content tokens present here. "
        "Second non entailing claim carrying other distinct content tokens now."
    )
    legacy, frontier = _run_both(monkeypatch, answer, passages, _VaryingNli)
    # No claim is entailed (fake caps entailment under the floor).
    assert all(c.verdict != VERDICT_ENTAILED for c in legacy.claims)
    assert _verdict_tuples(legacy) == _verdict_tuples(frontier)


# --------------------------------------------------------------------------- #
# (e) fingerprint changes only when the flag is on
# --------------------------------------------------------------------------- #

def test_frontier_changes_prose_entailment_fingerprint(monkeypatch):
    from lib.validators.block_prose_entailment import _block_fingerprint

    class _NliStub:
        device = "cpu"
        _revision = "rev1"

    passages = [{"chunk_id": "c1", "text": "alpha"}]
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_FRONTIER", raising=False)
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_S1_TOPK", raising=False)
    fp_legacy = _block_fingerprint("prose", passages, 0.7, 0.5, _NliStub())

    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_FRONTIER", "1")
    fp_frontier = _block_fingerprint("prose", passages, 0.7, 0.5, _NliStub())
    assert fp_frontier != fp_legacy

    # Width folds into the key too.
    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH", "8")
    fp_frontier_w8 = _block_fingerprint("prose", passages, 0.7, 0.5, _NliStub())
    assert fp_frontier_w8 != fp_frontier

    # Frontier supersedes top-K: with frontier on, the top-K key never appears,
    # so adding S1_TOPK does not change the frontier fingerprint.
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_FRONTIER_WIDTH", raising=False)
    monkeypatch.setenv("ED4ALL_GROUNDEDNESS_S1_TOPK", "16")
    assert _block_fingerprint("prose", passages, 0.7, 0.5, _NliStub()) == fp_frontier

    # Flag off again => byte-identical to the pre-flag legacy fingerprint.
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_FRONTIER", raising=False)
    monkeypatch.delenv("ED4ALL_GROUNDEDNESS_S1_TOPK", raising=False)
    assert _block_fingerprint("prose", passages, 0.7, 0.5, _NliStub()) == fp_legacy
