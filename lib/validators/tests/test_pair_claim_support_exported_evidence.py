"""Producer-exported evidence quotes reach the NLI premise set.

The staged provider verifies every ``evidence_quote`` against the source
window before accepting a plan, then emits a paraphrased answer. Because a
paraphrase never restates its claims verbatim, the sealed micro bridge
(``_verified_micro_claim_premises``) cannot match it, so before this change
every such pair was scored against whole-chunk text alone — and a faithful
sentence supported by one line of a long chunk was recorded as unsupported.

These tests pin the two halves of the fix: the producer exports the quotes,
and the validator offers them as premises without ever trusting a quote that
is not literally present in the source.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.pair.claim_support import (  # noqa: E402
    _MAX_EXPORTED_EVIDENCE_PREMISES,
    _exported_claim_evidence_premises,
)

_QUOTE = "A system whose equations describe the same line has infinitely many solutions."
_FILLER = "Unrelated background sentence about notation. " * 60
_CHUNK = f"{_FILLER}{_QUOTE} {_FILLER}"


def _pair(evidence) -> dict:
    return {"provenance": {"claim_evidence": evidence}}


def test_exact_quote_is_offered_as_a_premise() -> None:
    premises = _exported_claim_evidence_premises(
        _pair([{"claim": "Same line means infinite solutions.",
                "evidence_quote": _QUOTE,
                "source_block_id": "block-1"}]),
        _CHUNK,
    )
    assert premises == [_QUOTE]


def test_paraphrased_answer_does_not_have_to_restate_its_claim() -> None:
    """The sealed micro bridge's two extra conditions must not apply here.

    ``_verified_micro_claim_premises`` additionally requires a
    ``claim_realizations`` map, a matching ``provenance_sha256`` seal, and the
    realization to appear verbatim among the completion's sentences. A V4 pair
    satisfies none of those and must still get its premise.
    """
    premises = _exported_claim_evidence_premises(
        _pair([{"claim": "Same line means infinite solutions.",
                "evidence_quote": _QUOTE}]),
        _CHUNK,
    )
    assert premises == [_QUOTE]


def test_quote_absent_from_source_is_never_trusted() -> None:
    """Anti-fabrication: an invented quote must not become a premise."""
    premises = _exported_claim_evidence_premises(
        _pair([{"claim": "Fabricated.",
                "evidence_quote": "This sentence is nowhere in the source."}]),
        _CHUNK,
    )
    assert premises == []


def test_one_malformed_row_does_not_discard_the_valid_ones() -> None:
    """Per-row tolerance — the sealed bridge's all-or-nothing return was a
    silent way to lose every premise in an otherwise healthy pair."""
    premises = _exported_claim_evidence_premises(
        _pair([
            "not-a-dict",
            {"claim": "no quote key"},
            {"claim": "ok", "evidence_quote": _QUOTE},
            {"claim": "bad type", "evidence_quote": 17},
        ]),
        _CHUNK,
    )
    assert premises == [_QUOTE]


def test_duplicate_quotes_collapse_and_count_is_bounded() -> None:
    sentences = [
        f"Distinct source sentence number {i} appears here."
        for i in range(_MAX_EXPORTED_EVIDENCE_PREMISES + 4)
    ]
    chunk = " ".join(sentences)
    rows = [{"claim": f"c{i}", "evidence_quote": s}
            for i, s in enumerate(sentences)]
    rows.append({"claim": "dupe", "evidence_quote": sentences[0]})

    premises = _exported_claim_evidence_premises(_pair(rows), chunk)

    assert len(premises) == _MAX_EXPORTED_EVIDENCE_PREMISES
    assert len(set(premises)) == len(premises)


# --------------------------------------------------------------------- #
# Raw-whitespace containment. Every fixture above joins already-clean
# sentences, so the collapse mismatch between the producer's cleaned quote
# (`" ".join(v.split())`) and the RAW `chunk["text"]` never surfaced. These
# chunks carry real embedded newlines and double spaces ACROSS the quote.
# --------------------------------------------------------------------- #

_RAW_CHUNK = (
    "3.2 Systems of equations.  A system may have one solution,\n"
    "no solution, or infinitely many.\n\n"
    "A system whose equations describe\n"
    "the  same line has infinitely many solutions.\n\n"
    "The next section introduces matrix notation."
)
#: What the producer stamps: whitespace-collapsed, so it is NOT a raw
#: substring of ``_RAW_CHUNK`` even though it is genuinely present there.
_COLLAPSED_QUOTE = (
    "A system whose equations describe the same line has "
    "infinitely many solutions."
)


def test_quote_spanning_raw_newlines_and_double_spaces_is_admitted() -> None:
    """The producer collapses whitespace; the raw chunk does not.

    Concentrated in long multi-paragraph chunks — exactly the ones whose
    dilution the exported-evidence premise exists to fix.
    """
    assert _COLLAPSED_QUOTE not in _RAW_CHUNK  # the defect's precondition
    premises = _exported_claim_evidence_premises(
        _pair([{"claim": "Same line means infinite solutions.",
                "evidence_quote": _COLLAPSED_QUOTE}]),
        _RAW_CHUNK,
    )
    assert premises == [_COLLAPSED_QUOTE]


def test_raw_whitespace_quote_is_admitted_in_cleaned_form() -> None:
    """A quote still carrying raw whitespace normalizes to the same premise."""
    premises = _exported_claim_evidence_premises(
        _pair([{"claim": "c",
                "evidence_quote": "A system whose equations describe\n"
                                  "the  same line has infinitely many "
                                  "solutions."}]),
        _RAW_CHUNK,
    )
    assert premises == [_COLLAPSED_QUOTE]


def test_whitespace_insensitivity_does_not_weaken_anti_fabrication() -> None:
    """Whitespace-insensitive is not the same as permissive."""
    assert _exported_claim_evidence_premises(
        _pair([{"claim": "Fabricated.",
                "evidence_quote": "A system whose equations describe\n"
                                  "parallel  lines has no solution."}]),
        _RAW_CHUNK,
    ) == []


def test_legacy_pair_without_provenance_is_unchanged() -> None:
    assert _exported_claim_evidence_premises({}, _CHUNK) == []
    assert _exported_claim_evidence_premises({"provenance": None}, _CHUNK) == []
    assert _exported_claim_evidence_premises(
        {"provenance": {}}, _CHUNK,
    ) == []


def test_v4_provider_exports_the_quotes_it_verified() -> None:
    """Producer half: the plan's verified quotes land on the emitted pair."""
    from Trainforge.generators.staged_synthesis_provider import (
        StagedSynthesisProvider,
    )

    pair: dict = {}
    StagedSynthesisProvider._stamp_claim_evidence(pair, {
        "supported_claims": [
            {"claim": "Same line means infinite solutions.",
             "evidence_quote": _QUOTE,
             "source_block_id": "block-1"},
            {"claim": "  ", "evidence_quote": _QUOTE},
        ],
    })

    assert pair["provenance"]["claim_evidence"] == [{
        "claim": "Same line means infinite solutions.",
        "evidence_quote": _QUOTE,
        "source_block_id": "block-1",
    }]


def test_v4_provider_stamp_preserves_existing_provenance() -> None:
    from Trainforge.generators.staged_synthesis_provider import (
        StagedSynthesisProvider,
    )

    pair = {"provenance": {"existing_key": "kept"}}
    StagedSynthesisProvider._stamp_claim_evidence(pair, {
        "supported_claims": [
            {"claim": "c", "evidence_quote": _QUOTE},
        ],
    })

    assert pair["provenance"]["existing_key"] == "kept"
    assert len(pair["provenance"]["claim_evidence"]) == 1


def test_v4_provider_emits_nothing_when_no_claim_carries_evidence() -> None:
    from Trainforge.generators.staged_synthesis_provider import (
        StagedSynthesisProvider,
    )

    pair: dict = {}
    StagedSynthesisProvider._stamp_claim_evidence(pair, {"supported_claims": []})
    assert pair == {}


# --------------------------------------------------------------------- #
# Additive-premise invariant. The safety argument for offering exported
# quotes as extra premises is "aggregation is max-entailment, so a sentence
# can only score the same or better". That is only true of ENTAILMENT.
# Contradiction is read off the argmax-entailment premise (non-structured
# branch) or maxed independently (structured branch), so an added premise
# with higher entailment AND higher contradiction than the base chunk
# premise could flip a sentence unsupported -> contradicted — and
# claim_contradicted_rate is checked FIRST against a 0.05 ceiling.
# --------------------------------------------------------------------- #

_ADV_QUOTE = "Row reduction terminates in finitely many pivot operations."
_ADV_CHUNK_TEXT = (
    f"Chapter 4 opens with elimination. {_ADV_QUOTE} "
    "Pivoting order affects numerical stability."
)
_SHAKY_SENTENCE = (
    "The pivot ordering guarantees a unique reduced echelon form."
)


class _AdversarialNli:
    """Scores the added quote ABOVE the chunk on entailment AND contradiction.

    Everything else is keyed off the hypothesis so the supporting sentences
    stay comfortably entailed by the chunk premise.
    """

    def __init__(self) -> None:
        self.pairs: list = []

    def score_pair(self, *, premise, hypothesis):
        from lib.classifiers.nli_classifier import NliScore

        self.pairs.append((premise, hypothesis))
        if premise == _ADV_QUOTE:
            # Higher entailment than the chunk premise below (so it wins the
            # argmax) AND above the 0.50 contradiction floor.
            return NliScore(entailment=0.55, neutral=0.0, contradiction=0.60)
        if _SHAKY_SENTENCE.split(".")[0] in hypothesis:
            return NliScore(entailment=0.10, neutral=0.80, contradiction=0.10)
        return NliScore(entailment=0.95, neutral=0.05, contradiction=0.0)

    def score_batch(self, *, pairs):
        return [
            self.score_pair(premise=premise, hypothesis=hypothesis)
            for premise, hypothesis in pairs
        ]


def _adversarial_pair(sentence_count: int) -> dict:
    supported = [
        f"Elimination step number {i} clears the column below the pivot."
        for i in range(sentence_count - 1)
    ]
    return {
        "chunk_id": "chunk-adv",
        "completion": " ".join(supported + [_SHAKY_SENTENCE]),
        "provenance": {"claim_evidence": [
            {"claim": "Row reduction terminates.",
             "evidence_quote": _ADV_QUOTE},
        ]},
    }


def _validate_adversarial(pair: dict):
    from lib.validators.pair.claim_support import PairClaimSupportValidator

    nli = _AdversarialNli()
    status, reason, fields = PairClaimSupportValidator(nli=nli).validate_pair(
        pair, kind="instruction",
        chunk={"id": "chunk-adv", "text": _ADV_CHUNK_TEXT},
    )
    assert any(premise == _ADV_QUOTE for premise, _ in nli.pairs), (
        "fixture is inert — the exported quote never became a premise"
    )
    return status, reason, fields


def test_added_premise_never_degrades_a_sentence_outcome() -> None:
    """An added premise that contradicts must not sink the sentence.

    Without the clamp the argmax-entailment premise becomes the added quote,
    so the sentence reads contradiction 0.60 (>= the 0.50 floor) instead of
    the base chunk's 0.10 and flips ``unsupported`` -> ``contradicted``.
    """
    _, _, fields = _validate_adversarial(_adversarial_pair(2))
    shaky = [
        entry for entry in fields["per_claim_support"]
        if entry["sentence"] == _SHAKY_SENTENCE
    ]
    assert len(shaky) == 1
    assert shaky[0]["outcome"] == "unsupported"
    assert shaky[0]["contradiction"] <= 0.10
    assert fields["claim_contradicted_rate"] == 0.0


def test_added_premise_cannot_reject_a_pair_that_would_have_validated() -> None:
    """Pair level: 1 flip in 19 sentences is 0.0526 > the 0.05 ceiling."""
    status, reason, fields = _validate_adversarial(_adversarial_pair(19))
    assert fields["claim_contradicted_rate"] == 0.0
    assert (status, reason) == ("validated", None)
