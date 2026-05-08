"""Wave W-D11 T11.1 — ``ClaimSupportValidator`` evidence-quote tests.

Pins the substring + char_span enforcement T11.1 layered on top of the
existing per-block-per-claim NLI gate. Sibling of
``test_claim_support.py`` — kept separate so the W-D11 surface stays
clean and the W2.F suite is byte-stable.

T11.0 already pinned the schema (``chunk_v4.schema.json::key_claims``
structured arm + ``pair_audit_fields.PerClaimSupport``); this file
pins validator behaviour.

Test coverage (six cases per the W-D11 brief):

(a) Valid quote + valid char_span on a real fixture chunk → no new
    GateIssue; ``claims_with_valid_quote`` increments;
    ``evidence_quote_coverage_rate`` reflects it.
(b) Quote not in chunk → ``EVIDENCE_QUOTE_NOT_SUBSTRING`` warning
    emitted; ``passed`` unchanged (still True if NLI passed);
    ``evidence_quote_substring_fail_rate > 0``.
(c) char_span mismatch → ``EVIDENCE_CHAR_SPAN_MISMATCH`` warning;
    both rates correctly computed.
(d) Missing quote (legacy chunk without the field) →
    ``claims_without_quote`` increments; no GateIssue (day-1
    warning-only-via-threshold; threshold not yet wired).
(e) Legacy ``List[str]`` ``key_claims`` arm → graceful degrade;
    ``evidence_quote_coverage_rate=0.0`` and counter shapes still
    well-formed; no failures.
(f) Mixed claims (some with quote, some without) → counters reflect
    the mix; rate math is correct.

Day-1 contract: warning-only severity. The validator's overall
``passed`` field MUST NOT flip to False solely because of
evidence-quote findings. Calibration-deferred per W1.7.C / W4.A /
W4.C precedent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Wave W-D10 T10.2 marker parity with the sibling W2.F test file —
# the validator wraps the [embedding] NLI deps via a stub here.
pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.classifiers.nli_classifier import NliScore  # noqa: E402
from lib.validators.claim_support import (  # noqa: E402
    ClaimSupportValidator,
    _CODE_CONTRADICTED_CLAIM,
    _CODE_EVIDENCE_CHAR_SPAN_MISMATCH,
    _CODE_EVIDENCE_QUOTE_MISSING,
    _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING,
    _CODE_UNSUPPORTED_CLAIM,
)


# --------------------------------------------------------------------- #
# Stubs / fixture builders
# --------------------------------------------------------------------- #


class _StubNli:
    """Drop-in NLI stub. Defaults to high entailment so evidence-quote
    findings are isolated from NLI-outcome findings."""

    def __init__(
        self,
        response_map: Optional[Dict[Tuple[str, str], NliScore]] = None,
        default_score: Optional[NliScore] = None,
    ) -> None:
        self.response_map: Dict[Tuple[str, str], NliScore] = response_map or {}
        self.default_score = default_score or NliScore(
            entailment=0.92, neutral=0.05, contradiction=0.03,
        )
        self.batch_calls: List[List[Tuple[str, str]]] = []

    def score_batch(
        self, *, pairs: List[Tuple[str, str]]
    ) -> List[NliScore]:
        self.batch_calls.append(list(pairs))
        return [
            self.response_map.get(pair, self.default_score) for pair in pairs
        ]

    def score_pair(self, *, premise: str, hypothesis: str) -> NliScore:
        return self.response_map.get(
            (premise, hypothesis), self.default_score,
        )


def _make_block(
    *,
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
    page_id: str = "page_01",
    sequence: int = 0,
    key_claims: Optional[List[Any]] = None,
    source_ids: Tuple[str, ...] = ("dart:slug#blk_0",),
) -> Block:
    content: Dict[str, Any] = {"statement": "Block intro statement."}
    if key_claims is not None:
        content["key_claims"] = key_claims
    refs = tuple({"sourceId": sid} for sid in source_ids)
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=sequence,
        content=content,
        source_ids=source_ids,
        source_references=refs,
    )


# --------------------------------------------------------------------- #
# (a) Valid quote + valid char_span → no new GateIssue, counters align
# --------------------------------------------------------------------- #


def test_valid_quote_and_char_span_no_issue() -> None:
    chunk_text = "RDF is a graph data model. SHACL validates RDF graphs."
    quote = "RDF is a graph data model."
    start = chunk_text.index(quote)
    end = start + len(quote)

    nli = _StubNli()
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=[
            {
                "claim": "RDF is a graph data model.",
                "source_chunk_ids": ["dart:slug#blk_0"],
                "evidence_quote": quote,
                "evidence_char_span": [start, end],
            }
        ],
        source_ids=("dart:slug#blk_0",),
    )
    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": chunk_text},
        }
    )

    # No evidence-quote issues.
    quote_issues = [
        i
        for i in result.issues
        if i.code in (
            _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING,
            _CODE_EVIDENCE_CHAR_SPAN_MISMATCH,
            _CODE_EVIDENCE_QUOTE_MISSING,
        )
    ]
    assert quote_issues == []
    assert result.passed is True
    md = result.metadata or {}
    assert md["total_claims"] == 1
    assert md["claims_with_valid_quote"] == 1
    assert md["claims_without_quote"] == 0
    assert md["evidence_quote_coverage_rate"] == 1.0
    assert md["evidence_quote_substring_fail_rate"] == 0.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.0


# --------------------------------------------------------------------- #
# (b) Quote not in chunk → EVIDENCE_QUOTE_NOT_SUBSTRING warning
# --------------------------------------------------------------------- #


def test_quote_not_in_chunk_emits_substring_fail_warning() -> None:
    chunk_text = "RDF is a graph data model."
    bogus_quote = "Quote that is nowhere in the chunk."

    nli = _StubNli()
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=[
            {
                "claim": "Some claim.",
                "source_chunk_ids": ["dart:slug#blk_0"],
                "evidence_quote": bogus_quote,
            }
        ],
        source_ids=("dart:slug#blk_0",),
    )
    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": chunk_text},
        }
    )

    substring_issues = [
        i for i in result.issues if i.code == _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING
    ]
    assert len(substring_issues) == 1
    assert substring_issues[0].severity == "warning"
    # passed stays True — evidence-quote findings are warning-only.
    assert result.passed is True
    md = result.metadata or {}
    assert md["claims_with_valid_quote"] == 0
    assert md["evidence_quote_substring_attempts"] == 1
    assert md["evidence_quote_substring_fails"] == 1
    assert md["evidence_quote_substring_fail_rate"] == 1.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.0


# --------------------------------------------------------------------- #
# (c) char_span mismatch → EVIDENCE_CHAR_SPAN_MISMATCH warning
# --------------------------------------------------------------------- #


def test_char_span_mismatch_emits_warning() -> None:
    chunk_text = "RDF is a graph data model. SHACL validates RDF graphs."
    quote = "RDF is a graph data model."
    # Quote is a valid substring, but the span points at a different region.
    bad_start = 27  # offset of "SHACL"
    bad_end = bad_start + len(quote)

    nli = _StubNli()
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=[
            {
                "claim": "RDF is a graph data model.",
                "source_chunk_ids": ["dart:slug#blk_0"],
                "evidence_quote": quote,
                "evidence_char_span": [bad_start, bad_end],
            }
        ],
        source_ids=("dart:slug#blk_0",),
    )
    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": chunk_text},
        }
    )

    span_issues = [
        i for i in result.issues if i.code == _CODE_EVIDENCE_CHAR_SPAN_MISMATCH
    ]
    assert len(span_issues) == 1
    assert span_issues[0].severity == "warning"
    # No substring fail — quote was in the chunk.
    assert not any(
        i.code == _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING for i in result.issues
    )
    assert result.passed is True
    md = result.metadata or {}
    assert md["evidence_quote_substring_attempts"] == 1
    assert md["evidence_quote_substring_fails"] == 0
    assert md["evidence_quote_char_span_attempts"] == 1
    assert md["evidence_quote_char_span_mismatches"] == 1
    assert md["evidence_quote_substring_fail_rate"] == 0.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 1.0
    # Substring matched but span didn't → claim is NOT counted as valid.
    assert md["claims_with_valid_quote"] == 0


# --------------------------------------------------------------------- #
# (d) Missing quote → claims_without_quote increments, no GateIssue
# --------------------------------------------------------------------- #


def test_missing_quote_no_issue_counter_increments() -> None:
    chunk_text = "Source chunk text."
    nli = _StubNli()
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=[
            {
                "claim": "Some claim.",
                "source_chunk_ids": ["dart:slug#blk_0"],
                # No evidence_quote, no evidence_char_span (legacy /
                # pre-W-D11 corpora).
            }
        ],
        source_ids=("dart:slug#blk_0",),
    )
    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": chunk_text},
        }
    )

    # No evidence-quote issues at all (day-1 contract — threshold-gated
    # warning is reserved for T11.5+ wiring).
    assert not any(
        i.code in (
            _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING,
            _CODE_EVIDENCE_CHAR_SPAN_MISMATCH,
            _CODE_EVIDENCE_QUOTE_MISSING,
        )
        for i in result.issues
    )
    assert result.passed is True
    md = result.metadata or {}
    assert md["claims_without_quote"] == 1
    assert md["claims_with_valid_quote"] == 0
    assert md["evidence_quote_substring_attempts"] == 0
    assert md["evidence_quote_substring_fails"] == 0
    assert md["evidence_quote_coverage_rate"] == 0.0
    assert md["evidence_quote_substring_fail_rate"] == 0.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.0


# --------------------------------------------------------------------- #
# (e) Legacy List[str] arm → graceful degrade, well-formed counters
# --------------------------------------------------------------------- #


def test_legacy_list_str_arm_graceful_degrade() -> None:
    chunk_text = "Source chunk text."
    nli = _StubNli()
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=["A bare-string legacy claim.", "Another bare-string."],
        source_ids=("dart:slug#blk_0",),
    )
    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": chunk_text},
        }
    )

    # Legacy List[str] entries cannot carry evidence_quote → all are
    # counted as ``claims_without_quote`` (no GateIssue emitted).
    assert not any(
        i.code in (
            _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING,
            _CODE_EVIDENCE_CHAR_SPAN_MISMATCH,
            _CODE_EVIDENCE_QUOTE_MISSING,
        )
        for i in result.issues
    )
    assert result.passed is True
    md = result.metadata or {}
    # Counter shapes are well-formed (T11.5 contract).
    assert md["total_claims"] == 2
    assert md["claims_without_quote"] == 2
    assert md["claims_with_valid_quote"] == 0
    assert md["evidence_quote_coverage_rate"] == 0.0
    assert md["evidence_quote_substring_fail_rate"] == 0.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.0
    # All five rate / counter keys are always present.
    for key in (
        "total_claims",
        "claims_with_valid_quote",
        "claims_without_quote",
        "evidence_quote_coverage_rate",
        "evidence_quote_substring_fail_rate",
        "evidence_quote_char_span_mismatch_rate",
    ):
        assert key in md, f"metadata key {key!r} missing"


# --------------------------------------------------------------------- #
# (f) Mixed claims (some with quote, some without) → counters reflect mix
# --------------------------------------------------------------------- #


def test_mixed_claims_counters_reflect_mix() -> None:
    chunk_text = "RDF is a graph data model. SHACL validates RDF graphs."
    quote_a = "RDF is a graph data model."
    start_a = chunk_text.index(quote_a)
    end_a = start_a + len(quote_a)

    nli = _StubNli()
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=[
            # 1: Valid quote + valid char_span → counts as valid.
            {
                "claim": "Claim about RDF.",
                "source_chunk_ids": ["dart:slug#blk_0"],
                "evidence_quote": quote_a,
                "evidence_char_span": [start_a, end_a],
            },
            # 2: No quote at all → claims_without_quote.
            {
                "claim": "Claim with no quote.",
                "source_chunk_ids": ["dart:slug#blk_0"],
            },
            # 3: Bogus quote → substring fail.
            {
                "claim": "Claim with bogus quote.",
                "source_chunk_ids": ["dart:slug#blk_0"],
                "evidence_quote": "Nowhere-in-chunk text.",
            },
            # 4: Valid quote but bad char_span → char_span mismatch.
            {
                "claim": "Claim with bad span.",
                "source_chunk_ids": ["dart:slug#blk_0"],
                "evidence_quote": quote_a,
                "evidence_char_span": [0, 5],
            },
        ],
        source_ids=("dart:slug#blk_0",),
    )
    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": chunk_text},
        }
    )

    # One substring-fail issue (claim 3) + one char-span-mismatch issue
    # (claim 4). No issue for claim 1 or 2.
    substring_issues = [
        i for i in result.issues if i.code == _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING
    ]
    span_issues = [
        i for i in result.issues if i.code == _CODE_EVIDENCE_CHAR_SPAN_MISMATCH
    ]
    assert len(substring_issues) == 1
    assert len(span_issues) == 1
    assert result.passed is True

    md = result.metadata or {}
    assert md["total_claims"] == 4
    assert md["claims_with_valid_quote"] == 1
    assert md["claims_without_quote"] == 1
    assert md["evidence_quote_substring_attempts"] == 3  # claims 1/3/4
    assert md["evidence_quote_substring_fails"] == 1  # claim 3
    # char_span only attempted when substring matched (claims 1 + 4).
    assert md["evidence_quote_char_span_attempts"] == 2
    assert md["evidence_quote_char_span_mismatches"] == 1  # claim 4
    # Coverage = 1 valid out of 4 total claims = 0.25.
    assert md["evidence_quote_coverage_rate"] == 0.25
    # Substring fail rate = 1 fail / 3 attempts ≈ 0.3333.
    assert md["evidence_quote_substring_fail_rate"] == round(1 / 3, 4)
    # Char-span mismatch rate = 1 / 2 = 0.5.
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.5


# --------------------------------------------------------------------- #
# Regression guard: NLI-outcome failures still gate `passed` semantics
# --------------------------------------------------------------------- #


def test_nli_outcome_passed_unchanged_by_evidence_quote_findings() -> None:
    """Evidence-quote warnings must NOT contribute to action='regenerate'.

    A block whose NLI outcomes all entail (no UNSUPPORTED_CLAIM /
    CONTRADICTED_CLAIM) but whose evidence_quote is bogus must still
    return ``passed=True, action=None`` (only the warning is recorded).
    """
    chunk_text = "Source text fully entailing the claim."
    nli = _StubNli(
        default_score=NliScore(entailment=0.95, neutral=0.03, contradiction=0.02),
    )
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=[
            {
                "claim": "Claim entailed by source.",
                "source_chunk_ids": ["dart:slug#blk_0"],
                "evidence_quote": "Quote nowhere in the chunk.",
            }
        ],
        source_ids=("dart:slug#blk_0",),
    )
    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": chunk_text},
        }
    )

    # Substring fail emits warning but does NOT trigger regenerate.
    assert any(
        i.code == _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING for i in result.issues
    )
    assert not any(
        i.code in (_CODE_UNSUPPORTED_CLAIM, _CODE_CONTRADICTED_CLAIM)
        for i in result.issues
    )
    assert result.passed is True
    assert result.action is None
