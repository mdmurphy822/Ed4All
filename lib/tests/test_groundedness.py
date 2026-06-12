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
