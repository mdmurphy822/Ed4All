"""Wave W-D11 T11.2 — pair_claim_support evidence_quote enforcement tests.

Mirror of the T11.1 block-side test file (`test_claim_support_evidence_quote.py`)
on the pair surface. Exercises the gate-runner walk's substring +
char_span enforcement against `per_claim_support[].evidence_quote` /
`evidence_char_span` fields stamped by `validate_pair`.

Day-1 contract: warning-only — `passed` MUST stay True for evidence-
quote findings; only the existing structural-key-presence audit
(`MISSING_PER_CLAIM_SUPPORT` / `MISSING_CLAIM_SUPPORT_RATE`) flips
`passed` to False. Calibration plan: rebuild rdf-shacl-551-2 with
synthesis-provider quote emit → tune corpus baseline → flip warning
→ critical in a micro-wave (mirrors W1.7.C / W4.A / W4.C precedent).

Six cases per the T11.2 brief:

  a. valid quote + valid char_span → no new GateIssue, counters
     increment correctly.
  b. quote not in any cited chunk → ``EVIDENCE_QUOTE_NOT_SUBSTRING``
     warning; `passed` unchanged.
  c. char_span mismatch → ``EVIDENCE_CHAR_SPAN_MISMATCH`` warning.
  d. missing quote → counter increments, no GateIssue (warning-only-
     via-threshold; T11.5 / T11.6 wires the threshold).
  e. multiple cited chunks, quote matches the SECOND cited chunk →
     no failure (quote matches some cited chunk, not all).
  f. mixed pairs in a corpus → corpus-wide rate computed correctly
     across multiple pairs / multiple per_claim entries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.validators.pair.claim_support import (  # noqa: E402
    PairClaimSupportValidator,
    _CODE_EVIDENCE_CHAR_SPAN_MISMATCH,
    _CODE_EVIDENCE_QUOTE_MISSING,
    _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING,
)


# Canonical chunk text used across cases (a, b, c, d, e):
# Selected so:
#   - "ground truth statement" is at offset 12 (length 22) → valid
#     substring + valid char_span.
#   - "fabricated quote" is NOT a substring → case (b).
#   - "ground truth statement" with offset (5, 27) → wrong slice →
#     case (c).
_CHUNK_PRIMARY = "Chunk one: ground truth statement here for testing."
# Alternate chunk for case (e). The quote "evidence in the second"
# is at offset 12 (length 21).
_CHUNK_SECONDARY = "Chunk two: evidence in the second cited chunk text."

# Quote anchors for the canonical chunk.
_VALID_QUOTE = "ground truth statement"  # offset 11, length 22
_VALID_QUOTE_OFFSET = (
    _CHUNK_PRIMARY.index(_VALID_QUOTE),
    _CHUNK_PRIMARY.index(_VALID_QUOTE) + len(_VALID_QUOTE),
)
assert _CHUNK_PRIMARY[_VALID_QUOTE_OFFSET[0]:_VALID_QUOTE_OFFSET[1]] == _VALID_QUOTE


def _pair_with_per_claim_support(
    *,
    chunk_id: str = "c1",
    per_claim_entries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a minimal pair carrying the structural per_claim_support
    + claim_support_rate fields the on-disk audit walk requires."""
    return {
        "prompt": "x" * 50,
        "completion": "x" * 60,
        "chunk_id": chunk_id,
        "lo_refs": ["TO-01"],
        "bloom_level": "remember",
        "content_type": "explanation",
        "seed": 17,
        "decision_capture_id": "evt_test",
        "per_claim_support": per_claim_entries or [],
        "claim_support_rate": 1.0,
    }


def _per_claim_entry(
    *,
    sentence: str = "Some sentence backed by source.",
    outcome: str = "entailed",
    source_chunk_ids: Optional[List[str]] = None,
    evidence_quote: Optional[str] = None,
    evidence_char_span: Optional[List[int]] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "sentence": sentence,
        "entailment": 0.9,
        "contradiction": 0.05,
        "outcome": outcome,
        "source_chunk_ids": source_chunk_ids,
    }
    if evidence_quote is not None:
        entry["evidence_quote"] = evidence_quote
    if evidence_char_span is not None:
        entry["evidence_char_span"] = list(evidence_char_span)
    return entry


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------- #
# Case (a) — valid quote + valid char_span
# --------------------------------------------------------------------- #


def test_case_a_valid_quote_and_char_span_passes(tmp_path: Path) -> None:
    """Valid quote + char_span: no new GateIssue, counters increment."""
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            _per_claim_entry(
                source_chunk_ids=["c1"],
                evidence_quote=_VALID_QUOTE,
                evidence_char_span=list(_VALID_QUOTE_OFFSET),
            ),
        ],
    )
    _write_jsonl(inst, [pair])

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {"c1": _CHUNK_PRIMARY},
    })

    # Structural audit passes.
    assert result.passed is True
    # No evidence-quote GateIssue codes fired.
    codes = [i.code for i in result.issues]
    assert _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING not in codes
    assert _CODE_EVIDENCE_CHAR_SPAN_MISMATCH not in codes
    assert _CODE_EVIDENCE_QUOTE_MISSING not in codes

    # Counters land on metadata.
    md = result.metadata
    assert md is not None
    assert md["total_claims"] == 1
    assert md["claims_with_valid_quote"] == 1
    assert md["claims_without_quote"] == 0
    assert md["evidence_quote_substring_attempts"] == 1
    assert md["evidence_quote_substring_fails"] == 0
    assert md["evidence_quote_char_span_attempts"] == 1
    assert md["evidence_quote_char_span_mismatches"] == 0
    assert md["evidence_quote_coverage_rate"] == 1.0
    assert md["evidence_quote_substring_fail_rate"] == 0.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.0


# --------------------------------------------------------------------- #
# Case (b) — quote not in any cited chunk
# --------------------------------------------------------------------- #


def test_case_b_quote_not_substring_fires_warning(tmp_path: Path) -> None:
    """Quote absent from cited chunks: warning + counter, passed True."""
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            _per_claim_entry(
                source_chunk_ids=["c1"],
                evidence_quote="fabricated quote not in chunk",
            ),
        ],
    )
    _write_jsonl(inst, [pair])

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {"c1": _CHUNK_PRIMARY},
    })

    # Day-1 warning contract — passed unchanged.
    assert result.passed is True
    codes = [i.code for i in result.issues]
    assert _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING in codes
    # The fired issue is warning-severity.
    not_substring_issues = [
        i for i in result.issues
        if i.code == _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING
    ]
    assert len(not_substring_issues) == 1
    assert not_substring_issues[0].severity == "warning"

    md = result.metadata
    assert md is not None
    assert md["total_claims"] == 1
    assert md["claims_with_valid_quote"] == 0
    assert md["evidence_quote_substring_attempts"] == 1
    assert md["evidence_quote_substring_fails"] == 1
    assert md["evidence_quote_char_span_attempts"] == 0
    assert md["evidence_quote_substring_fail_rate"] == 1.0
    assert md["evidence_quote_coverage_rate"] == 0.0


# --------------------------------------------------------------------- #
# Case (c) — char_span mismatch
# --------------------------------------------------------------------- #


def test_case_c_char_span_mismatch_fires_warning(tmp_path: Path) -> None:
    """char_span mismatch: warning + counter, passed True."""
    inst = tmp_path / "instruction_pairs.jsonl"
    # Quote is a valid substring, but the char_span slice does NOT
    # equal the quote (off by 5 chars).
    bad_span = (
        _VALID_QUOTE_OFFSET[0] - 5,
        _VALID_QUOTE_OFFSET[1] - 5,
    )
    assert (
        _CHUNK_PRIMARY[bad_span[0]:bad_span[1]] != _VALID_QUOTE
    )  # pre-check: this span doesn't slice to the quote
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            _per_claim_entry(
                source_chunk_ids=["c1"],
                evidence_quote=_VALID_QUOTE,
                evidence_char_span=list(bad_span),
            ),
        ],
    )
    _write_jsonl(inst, [pair])

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {"c1": _CHUNK_PRIMARY},
    })

    assert result.passed is True
    codes = [i.code for i in result.issues]
    # Substring check passes (quote IS in chunk) but span doesn't slice.
    assert _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING not in codes
    assert _CODE_EVIDENCE_CHAR_SPAN_MISMATCH in codes
    span_mismatch_issues = [
        i for i in result.issues
        if i.code == _CODE_EVIDENCE_CHAR_SPAN_MISMATCH
    ]
    assert len(span_mismatch_issues) == 1
    assert span_mismatch_issues[0].severity == "warning"

    md = result.metadata
    assert md is not None
    assert md["evidence_quote_substring_attempts"] == 1
    assert md["evidence_quote_substring_fails"] == 0
    assert md["evidence_quote_char_span_attempts"] == 1
    assert md["evidence_quote_char_span_mismatches"] == 1
    # Quote is NOT a "valid quote" because the char_span check failed.
    assert md["claims_with_valid_quote"] == 0
    assert md["evidence_quote_char_span_mismatch_rate"] == 1.0


# --------------------------------------------------------------------- #
# Case (d) — missing quote
# --------------------------------------------------------------------- #


def test_case_d_missing_quote_increments_counter_no_issue(
    tmp_path: Path,
) -> None:
    """Missing quote: counter increments, NO GateIssue (warning-only-
    via-threshold; T11.5 / T11.6 wires the threshold)."""
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            # No evidence_quote / evidence_char_span set — legacy /
            # graceful-degrade arm.
            _per_claim_entry(source_chunk_ids=["c1"]),
        ],
    )
    _write_jsonl(inst, [pair])

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {"c1": _CHUNK_PRIMARY},
    })

    assert result.passed is True
    codes = [i.code for i in result.issues]
    # No evidence-quote issue codes fired (missing quote is silent at
    # day-1; T11.5 / T11.6 wires the threshold-gated EVIDENCE_QUOTE_MISSING
    # surface).
    assert _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING not in codes
    assert _CODE_EVIDENCE_CHAR_SPAN_MISMATCH not in codes
    assert _CODE_EVIDENCE_QUOTE_MISSING not in codes

    md = result.metadata
    assert md is not None
    assert md["total_claims"] == 1
    assert md["claims_without_quote"] == 1
    assert md["claims_with_valid_quote"] == 0
    assert md["evidence_quote_substring_attempts"] == 0
    assert md["evidence_quote_coverage_rate"] == 0.0
    # Substring fail rate is 0.0 (no attempts) — important: division-by-
    # zero defends to 0.0, not NaN.
    assert md["evidence_quote_substring_fail_rate"] == 0.0


# --------------------------------------------------------------------- #
# Case (e) — quote matches the SECOND cited chunk
# --------------------------------------------------------------------- #


def test_case_e_quote_matches_second_cited_chunk_passes(
    tmp_path: Path,
) -> None:
    """Multiple cited chunks: quote matches the SECOND cited chunk
    (the brief's "match SOME cited chunk's text, not all" semantic).
    """
    secondary_quote = "evidence in the second"
    secondary_offset = (
        _CHUNK_SECONDARY.index(secondary_quote),
        _CHUNK_SECONDARY.index(secondary_quote) + len(secondary_quote),
    )
    assert (
        _CHUNK_SECONDARY[secondary_offset[0]:secondary_offset[1]]
        == secondary_quote
    )

    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            _per_claim_entry(
                # Two cited chunks; quote is in the SECOND, not first.
                source_chunk_ids=["c1", "c2"],
                evidence_quote=secondary_quote,
                evidence_char_span=list(secondary_offset),
            ),
        ],
    )
    _write_jsonl(inst, [pair])

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {
            "c1": _CHUNK_PRIMARY,
            "c2": _CHUNK_SECONDARY,
        },
    })

    assert result.passed is True
    codes = [i.code for i in result.issues]
    # No failure — the quote matches SOME cited chunk's text (the
    # second one), and the char_span slices correctly within that
    # matched chunk.
    assert _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING not in codes
    assert _CODE_EVIDENCE_CHAR_SPAN_MISMATCH not in codes

    md = result.metadata
    assert md is not None
    assert md["claims_with_valid_quote"] == 1
    assert md["evidence_quote_substring_attempts"] == 1
    assert md["evidence_quote_substring_fails"] == 0
    assert md["evidence_quote_char_span_attempts"] == 1
    assert md["evidence_quote_char_span_mismatches"] == 0


# --------------------------------------------------------------------- #
# Case (f) — mixed pairs across a corpus
# --------------------------------------------------------------------- #


def test_case_f_mixed_corpus_rate_computed_correctly(
    tmp_path: Path,
) -> None:
    """Mixed corpus: one valid pair, one substring-fail pair, one
    missing-quote pair, one char_span-mismatch pair → corpus-wide rates
    computed correctly across all four."""
    bad_span = (
        _VALID_QUOTE_OFFSET[0] - 5,
        _VALID_QUOTE_OFFSET[1] - 5,
    )

    pairs = [
        # Pair 1: one valid claim.
        _pair_with_per_claim_support(
            per_claim_entries=[
                _per_claim_entry(
                    source_chunk_ids=["c1"],
                    evidence_quote=_VALID_QUOTE,
                    evidence_char_span=list(_VALID_QUOTE_OFFSET),
                ),
            ],
        ),
        # Pair 2: one substring-fail claim.
        _pair_with_per_claim_support(
            per_claim_entries=[
                _per_claim_entry(
                    source_chunk_ids=["c1"],
                    evidence_quote="not present in chunk",
                ),
            ],
        ),
        # Pair 3: one missing-quote claim.
        _pair_with_per_claim_support(
            per_claim_entries=[
                _per_claim_entry(source_chunk_ids=["c1"]),
            ],
        ),
        # Pair 4: one char_span-mismatch claim.
        _pair_with_per_claim_support(
            per_claim_entries=[
                _per_claim_entry(
                    source_chunk_ids=["c1"],
                    evidence_quote=_VALID_QUOTE,
                    evidence_char_span=list(bad_span),
                ),
            ],
        ),
    ]
    inst = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst, pairs)

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {"c1": _CHUNK_PRIMARY},
    })

    assert result.passed is True

    md = result.metadata
    assert md is not None
    # 4 claims total (one per pair).
    assert md["total_claims"] == 4
    # 1 valid (pair 1).
    assert md["claims_with_valid_quote"] == 1
    # 1 missing (pair 3).
    assert md["claims_without_quote"] == 1
    # 3 substring-attempts (pairs 1, 2, 4).
    assert md["evidence_quote_substring_attempts"] == 3
    # 1 substring-fail (pair 2).
    assert md["evidence_quote_substring_fails"] == 1
    # 2 char-span-attempts (pairs 1, 4 — both have spans + substring
    # passed).
    assert md["evidence_quote_char_span_attempts"] == 2
    # 1 char-span-mismatch (pair 4).
    assert md["evidence_quote_char_span_mismatches"] == 1

    # Rates: coverage = 1/4, substring_fail = 1/3, char_span_mismatch = 1/2.
    assert md["evidence_quote_coverage_rate"] == 0.25
    assert abs(
        md["evidence_quote_substring_fail_rate"] - 1 / 3
    ) < 1e-3
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.5

    codes = [i.code for i in result.issues]
    # Both real-failure GateIssue codes fired exactly once each.
    assert codes.count(_CODE_EVIDENCE_QUOTE_NOT_SUBSTRING) == 1
    assert codes.count(_CODE_EVIDENCE_CHAR_SPAN_MISMATCH) == 1


# --------------------------------------------------------------------- #
# Graceful-degrade — legacy pairs without evidence_quote pass cleanly
# --------------------------------------------------------------------- #


def test_legacy_pairs_no_evidence_fields_pass_cleanly(
    tmp_path: Path,
) -> None:
    """Pre-W-D11 corpora — pairs whose per_claim_support entries have
    NEITHER evidence_quote NOR evidence_char_span pass the gate clean.
    The structural-key-presence audit still passes, and the metadata
    surface reports 0.0 rates (not NaN, not error)."""
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            _per_claim_entry(source_chunk_ids=["c1"]),
            _per_claim_entry(source_chunk_ids=["c1"]),
        ],
    )
    _write_jsonl(inst, [pair])

    # Legacy walk — no chunk_id_to_text_map threaded.
    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
    })

    assert result.passed is True
    # No GateIssue codes specific to W-D11 fired.
    codes = [i.code for i in result.issues]
    assert _CODE_EVIDENCE_QUOTE_NOT_SUBSTRING not in codes
    assert _CODE_EVIDENCE_CHAR_SPAN_MISMATCH not in codes

    md = result.metadata
    assert md is not None
    # Both claims counted as missing-quote (no evidence_quote field).
    assert md["total_claims"] == 2
    assert md["claims_without_quote"] == 2
    assert md["claims_with_valid_quote"] == 0
    # Rates default to 0.0 — division-by-zero defended.
    assert md["evidence_quote_coverage_rate"] == 0.0
    assert md["evidence_quote_substring_fail_rate"] == 0.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.0


# --------------------------------------------------------------------- #
# No chunks threaded — quote present but unverifiable
# --------------------------------------------------------------------- #


def test_quote_present_but_no_chunks_threaded_skips_check(
    tmp_path: Path,
) -> None:
    """When the gate-runner walk doesn't have a chunk_id_to_text_map
    threaded, evidence_quote checks are silently skipped — neither
    substring_attempts nor substring_fails increment. The pair still
    passes the structural audit."""
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            _per_claim_entry(
                source_chunk_ids=["c1"],
                evidence_quote=_VALID_QUOTE,
                evidence_char_span=list(_VALID_QUOTE_OFFSET),
            ),
        ],
    )
    _write_jsonl(inst, [pair])

    # Note: NO chunk_id_to_text_map.
    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
    })

    assert result.passed is True
    md = result.metadata
    assert md is not None
    # Quote was present so claims_without_quote stays 0; substring
    # check was skipped because no chunk surface to verify against.
    assert md["claims_without_quote"] == 0
    assert md["evidence_quote_substring_attempts"] == 0
    assert md["evidence_quote_substring_fails"] == 0
    assert md["claims_with_valid_quote"] == 0


# ===================================================================== #
# Wave W-D11 T11.4 — gate-walk decision-capture rationale signals
# ===================================================================== #
#
# Asserts that ``pair_claim_support_check`` decision events emitted by
# the on-disk gate-walk (``validate()``) interpolate the three
# evidence-quote rates (coverage / substring_fail / char_span_mismatch)
# in the rationale string. Mirrors the W5.D pattern of piggybacking on
# existing ``decision_type`` enum values.


class _CaptureStub:
    """Minimal DecisionCapture stand-in for rationale-content tests."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(
        self,
        *,
        decision_type: str,
        decision: str,
        rationale: str,
        **_: Any,
    ) -> None:
        self.calls.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
            }
        )


def test_t11_4_gate_walk_rationale_carries_evidence_quote_signals(
    tmp_path: Path,
) -> None:
    """Gate-walk valid quote + char_span: rationale carries
    coverage=100%, substring_fail=0%, char_span_mismatch=0%; per-corpus
    metadata rates match."""
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            _per_claim_entry(
                source_chunk_ids=["c1"],
                evidence_quote=_VALID_QUOTE,
                evidence_char_span=list(_VALID_QUOTE_OFFSET),
            ),
        ],
    )
    _write_jsonl(inst, [pair])
    capture = _CaptureStub()

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {"c1": _CHUNK_PRIMARY},
        "decision_capture": capture,
    })

    events = [
        e for e in capture.calls
        if e["decision_type"] == "pair_claim_support_check"
    ]
    assert len(events) == 1
    rationale = events[0]["rationale"]
    md = result.metadata or {}
    # Format token: matches the block-side + per-pair format so a
    # single grep finds all three surfaces.
    assert "evidence_quote: coverage=" in rationale
    assert "substring_fail=" in rationale
    assert "char_span_mismatch=" in rationale
    assert md["evidence_quote_coverage_rate"] == 1.0
    assert md["evidence_quote_substring_fail_rate"] == 0.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.0
    assert "coverage=100.00%" in rationale
    assert "substring_fail=0.00%" in rationale
    assert "char_span_mismatch=0.00%" in rationale


def test_t11_4_gate_walk_rationale_records_substring_fail_above_zero(
    tmp_path: Path,
) -> None:
    """Substring-fail fixture: gate-walk rationale records
    substring_fail>0%."""
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            _per_claim_entry(
                source_chunk_ids=["c1"],
                evidence_quote="fabricated quote not in chunk",
            ),
        ],
    )
    _write_jsonl(inst, [pair])
    capture = _CaptureStub()

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {"c1": _CHUNK_PRIMARY},
        "decision_capture": capture,
    })

    events = [
        e for e in capture.calls
        if e["decision_type"] == "pair_claim_support_check"
    ]
    assert len(events) == 1
    rationale = events[0]["rationale"]
    md = result.metadata or {}
    assert md["evidence_quote_substring_fail_rate"] == 1.0
    assert "substring_fail=100.00%" in rationale
    assert "coverage=0.00%" in rationale


def test_t11_4_gate_walk_rationale_records_char_span_mismatch_above_zero(
    tmp_path: Path,
) -> None:
    """Char-span-mismatch fixture: gate-walk rationale records
    char_span_mismatch>0%."""
    inst = tmp_path / "instruction_pairs.jsonl"
    bad_span = (
        _VALID_QUOTE_OFFSET[0] - 5,
        _VALID_QUOTE_OFFSET[1] - 5,
    )
    pair = _pair_with_per_claim_support(
        per_claim_entries=[
            _per_claim_entry(
                source_chunk_ids=["c1"],
                evidence_quote=_VALID_QUOTE,
                evidence_char_span=list(bad_span),
            ),
        ],
    )
    _write_jsonl(inst, [pair])
    capture = _CaptureStub()

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {"c1": _CHUNK_PRIMARY},
        "decision_capture": capture,
    })

    events = [
        e for e in capture.calls
        if e["decision_type"] == "pair_claim_support_check"
    ]
    assert len(events) == 1
    rationale = events[0]["rationale"]
    md = result.metadata or {}
    assert md["evidence_quote_char_span_mismatch_rate"] == 1.0
    assert "char_span_mismatch=100.00%" in rationale
    assert "substring_fail=0.00%" in rationale


def test_t11_4_gate_walk_legacy_pair_rationale_constructs_cleanly(
    tmp_path: Path,
) -> None:
    """Legacy pair without evidence_quote: gate-walk rationale still
    constructs cleanly with coverage=0.00%, no exception."""
    inst = tmp_path / "instruction_pairs.jsonl"
    pair = _pair_with_per_claim_support(
        # No per_claim entries with evidence_quote.
        per_claim_entries=[_per_claim_entry(source_chunk_ids=["c1"])],
    )
    _write_jsonl(inst, [pair])
    capture = _CaptureStub()

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst),
        "chunk_id_to_text_map": {"c1": _CHUNK_PRIMARY},
        "decision_capture": capture,
    })

    events = [
        e for e in capture.calls
        if e["decision_type"] == "pair_claim_support_check"
    ]
    assert len(events) == 1
    rationale = events[0]["rationale"]
    assert "coverage=0.00%" in rationale
    assert "substring_fail=0.00%" in rationale
    assert "char_span_mismatch=0.00%" in rationale
    assert result.passed is True


# ===================================================================== #
# Wave W-D11 T11.4 — per-pair (validate_pair) rationale signals
# ===================================================================== #
#
# Asserts that ``pair_claim_support_check`` decision events emitted by
# ``validate_pair`` (per-pair, in-process during synthesis) interpolate
# the three evidence-quote rates. Per-pair surface only computes
# coverage_rate (substring + char_span verification happens at the
# gate-walk surface, not here); the verification-derived rates default
# to 0.0 in the rationale per the metadata default-key contract.


def test_t11_4_validate_pair_rationale_carries_evidence_quote_signals() -> None:
    """validate_pair on a deps-missing branch (no NLI) still emits a
    rationale with the three evidence_quote signals defaulted to 0.0%
    — the deps-missing arm short-circuits before counters populate,
    and the format token must still appear so downstream rationale
    parsers see a stable shape across both branches."""
    validator = PairClaimSupportValidator()
    capture = _CaptureStub()

    # Provide an instruction pair + chunk so validate_pair fires; pass
    # a chunk that triggers the deps-missing branch by passing nli=None
    # via the validator-internal _get_nli() — done by NOT installing
    # the embedding extras (which is the default for the test
    # environment). The deps-missing branch emits the decision event
    # with all rates defaulted to 0.0.
    pair = {
        "prompt": "x" * 50,
        "completion": "x" * 60,
        "chunk_id": "c1",
        "lo_refs": ["TO-01"],
        "bloom_level": "remember",
        "content_type": "explanation",
    }
    chunk = {
        "id": "c1",
        "text": "Some chunk text for grounding.",
        "source": {},
    }
    _result = validator.validate_pair(
        pair=pair,
        kind="instruction",
        chunk=chunk,
        decision_capture=capture,
    )
    events = [
        e for e in capture.calls
        if e["decision_type"] == "pair_claim_support_check"
    ]
    assert len(events) >= 1
    rationale = events[-1]["rationale"]
    # The format token is present regardless of which branch fired.
    assert "evidence_quote: coverage=" in rationale
    assert "substring_fail=" in rationale
    assert "char_span_mismatch=" in rationale
    # Rationale stays well past the 20-character minimum.
    assert len(rationale) >= 100
