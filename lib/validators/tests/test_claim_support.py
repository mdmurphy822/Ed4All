"""GPT Feedback v2 Wave 2 W2.F — ``ClaimSupportValidator`` tests.

Covers the per-block-per-claim NLI entailment gate. Mocks the NLI
classifier via a stub so CI fresh-checkout doesn't need the
DeBERTa-v3-base model. The real loader's deps-missing branch is
exercised via the ``get_or_load returns None`` fixture (Test 6) and
the strict-mode arm (Test 7).

Test cases (10 total per Wave 2 W2.F brief):

1. All-entailed block → passed=True, action=None.
2. Above unsupported threshold → UNSUPPORTED_CLAIM warning, action=regenerate.
3. Above contradicted-only threshold → CONTRADICTED_CLAIM warning,
   action=regenerate.
4. Legacy List[str] key_claims (back-compat) → block-level chunk universe.
5. Structured List[{claim, source_chunk_ids[]}] (Wave 1.5 shape) →
   per-claim chunk scoping.
6. NLI deps missing (get_or_load returns None) → NLI_DEPS_MISSING
   warning, passed=True, action=None.
7. Strict mode (TRAINFORGE_REQUIRE_EMBEDDINGS=true) + missing deps →
   raises RuntimeError.
8. Skip-set block types (assessment_item, objective) → no events,
   no issues.
9. Decision capture fires per block with the 6 required signals.
10. Empty key_claims → silent pass, no events.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

# Wave W-D10 T10.2: validator wraps the [embedding] NLI deps (DeBERTa-v3
# via lib.classifiers.nli_classifier); tests stub the loader today but
# the file is slow-marked so the nightly extras-installed run picks them
# up via -m "slow".
pytestmark = pytest.mark.slow

# Repo root + scripts dir on path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.classifiers.nli_classifier import NliClassifier, NliScore  # noqa: E402
from lib.validators.claim_support import (  # noqa: E402
    ClaimSupportValidator,
    _CODE_CONTRADICTED_CLAIM,
    _CODE_NLI_DEPS_MISSING,
    _CODE_UNSUPPORTED_CLAIM,
)


# --------------------------------------------------------------------- #
# Stubs / fixtures
# --------------------------------------------------------------------- #


class _StubNli:
    """Drop-in test stub for ``NliClassifier``.

    Returns deterministic per-(premise, hypothesis) NliScores via a
    response map keyed on the (premise, hypothesis) tuple. Falls back
    to a high-entailment default for unmapped pairs. Records every
    ``score_batch`` / ``score_pair`` call so tests can assert fan-out.
    """

    def __init__(
        self,
        response_map: Optional[Dict[Tuple[str, str], NliScore]] = None,
        default_score: Optional[NliScore] = None,
    ) -> None:
        self.response_map: Dict[Tuple[str, str], NliScore] = (
            response_map or {}
        )
        self.default_score = default_score or NliScore(
            entailment=0.85, neutral=0.10, contradiction=0.05,
        )
        self.batch_calls: List[List[Tuple[str, str]]] = []
        self.pair_calls: List[Tuple[str, str]] = []

    def score_pair(self, *, premise: str, hypothesis: str) -> NliScore:
        self.pair_calls.append((premise, hypothesis))
        return self.response_map.get(
            (premise, hypothesis), self.default_score,
        )

    def score_batch(
        self, *, pairs: List[Tuple[str, str]]
    ) -> List[NliScore]:
        self.batch_calls.append(list(pairs))
        return [
            self.response_map.get(pair, self.default_score) for pair in pairs
        ]


def _make_block(
    *,
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
    page_id: str = "page_01",
    sequence: int = 0,
    key_claims: Optional[List[Any]] = None,
    source_ids: Tuple[str, ...] = ("semantik:slug#blk_0",),
    source_references: Optional[Tuple[Dict[str, Any], ...]] = None,
) -> Block:
    """Build a Block with dict content + ``key_claims`` populated.

    ``key_claims`` accepts either the legacy ``List[str]`` or the
    Wave 1.5 structured ``List[{claim, source_chunk_ids[]}]`` shape.
    """
    content: Dict[str, Any] = {"statement": "Block intro statement."}
    if key_claims is not None:
        content["key_claims"] = key_claims
    refs = source_references if source_references is not None else tuple(
        {"sourceId": sid} for sid in source_ids
    )
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
# Test 1: All-entailed block → passes
# --------------------------------------------------------------------- #


def test_all_entailed_block_passes() -> None:
    """Every claim scores entailment >= 0.7 against cited chunks → pass."""
    nli = _StubNli(
        default_score=NliScore(entailment=0.90, neutral=0.05, contradiction=0.05)
    )
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=["Claim A about X.", "Claim B about Y."],
        source_ids=("semantik:slug#blk_0",),
    )
    source_chunks = {
        "semantik:slug#blk_0": "Source chunk text covering X and Y in detail.",
    }

    result = validator.validate(
        {"blocks": [block], "source_chunks": source_chunks}
    )

    assert result.passed is True
    assert result.action is None
    assert not any(
        i.code in (_CODE_UNSUPPORTED_CLAIM, _CODE_CONTRADICTED_CLAIM)
        for i in result.issues
    )


# --------------------------------------------------------------------- #
# Test 2: Above unsupported threshold → UNSUPPORTED_CLAIM warning
# --------------------------------------------------------------------- #


def test_above_unsupported_threshold_fires_warning_and_regenerate() -> None:
    """3/4 unsupported claims (75%) > 20% ceiling → UNSUPPORTED_CLAIM."""
    chunk_text = "Some background text not covering the claims."
    # Build 4 claims where 3 score low entailment + low contradiction
    # (= unsupported), 1 scores high entailment (= entailed).
    claims = ["c1", "c2", "c3", "c4"]
    response_map = {
        (chunk_text, "c1"): NliScore(entailment=0.10, neutral=0.85, contradiction=0.05),
        (chunk_text, "c2"): NliScore(entailment=0.20, neutral=0.70, contradiction=0.10),
        (chunk_text, "c3"): NliScore(entailment=0.15, neutral=0.80, contradiction=0.05),
        (chunk_text, "c4"): NliScore(entailment=0.95, neutral=0.03, contradiction=0.02),
    }
    nli = _StubNli(response_map=response_map)
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(key_claims=claims, source_ids=("semantik:slug#blk_0",))
    source_chunks = {"semantik:slug#blk_0": chunk_text}

    result = validator.validate(
        {"blocks": [block], "source_chunks": source_chunks}
    )

    unsupported_issues = [
        i for i in result.issues if i.code == _CODE_UNSUPPORTED_CLAIM
    ]
    assert len(unsupported_issues) == 1
    assert unsupported_issues[0].severity == "warning"
    assert result.action == "regenerate"
    # ``passed`` stays True because every issue is warning-severity.
    assert result.passed is True


# --------------------------------------------------------------------- #
# Test 3: Above contradicted-only threshold → CONTRADICTED_CLAIM warning
# --------------------------------------------------------------------- #


def test_above_contradicted_threshold_fires_warning_and_regenerate() -> None:
    """1/10 contradicted claims (10%) > 5% ceiling → CONTRADICTED_CLAIM,
    even when total unsupported rate (1/10 = 10%) is below 20% ceiling."""
    chunk_text = "Source chunk text."
    claims = [f"c{i}" for i in range(10)]
    response_map: Dict[Tuple[str, str], NliScore] = {}
    # 9 claims entailed (high entailment); 1 contradicted (high
    # contradiction). Total unsupported_rate = 1/10 = 10% (< 20%
    # ceiling), but contradicted_rate = 1/10 = 10% (> 5% ceiling).
    for i, c in enumerate(claims):
        if i == 0:
            response_map[(chunk_text, c)] = NliScore(
                entailment=0.10, neutral=0.10, contradiction=0.80,
            )
        else:
            response_map[(chunk_text, c)] = NliScore(
                entailment=0.95, neutral=0.03, contradiction=0.02,
            )
    nli = _StubNli(response_map=response_map)
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(key_claims=claims, source_ids=("semantik:slug#blk_0",))
    source_chunks = {"semantik:slug#blk_0": chunk_text}

    result = validator.validate(
        {"blocks": [block], "source_chunks": source_chunks}
    )

    contradicted_issues = [
        i for i in result.issues if i.code == _CODE_CONTRADICTED_CLAIM
    ]
    assert len(contradicted_issues) == 1
    assert contradicted_issues[0].severity == "warning"
    assert result.action == "regenerate"


# --------------------------------------------------------------------- #
# Test 4: Legacy List[str] key_claims (back-compat) → block-level fallback
# --------------------------------------------------------------------- #


def test_legacy_list_str_key_claims_uses_block_level_chunks() -> None:
    """Legacy shape: each claim fans out against the union of all cited
    chunks (block-level fallback)."""
    chunk_a = "Source chunk A."
    chunk_b = "Source chunk B."
    nli = _StubNli(
        default_score=NliScore(entailment=0.90, neutral=0.05, contradiction=0.05)
    )
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=["Single claim text."],
        source_ids=("semantik:slug#blk_a", "semantik:slug#blk_b"),
    )
    source_chunks = {
        "semantik:slug#blk_a": chunk_a,
        "semantik:slug#blk_b": chunk_b,
    }
    result = validator.validate(
        {"blocks": [block], "source_chunks": source_chunks}
    )

    # The single claim fanned out against BOTH cited chunks → 1
    # batch call with 2 pairs (both chunks paired with the single claim).
    assert len(nli.batch_calls) == 1
    assert len(nli.batch_calls[0]) == 2
    pairs = nli.batch_calls[0]
    premise_set = {pair[0] for pair in pairs}
    assert premise_set == {chunk_a, chunk_b}
    assert result.passed is True


# --------------------------------------------------------------------- #
# Test 5: Structured List[{claim, source_chunk_ids[]}] → per-claim scoping
# --------------------------------------------------------------------- #


def test_structured_key_claims_uses_per_claim_chunk_scoping() -> None:
    """Wave 1.5 W1.5.A shape: each claim fans out against ONLY the
    chunks named in its source_chunk_ids[]."""
    chunk_a = "Source chunk A — about RDF."
    chunk_b = "Source chunk B — about SHACL."
    nli = _StubNli(
        default_score=NliScore(entailment=0.90, neutral=0.05, contradiction=0.05)
    )
    validator = ClaimSupportValidator(nli=nli)

    block = _make_block(
        key_claims=[
            {
                "claim": "Claim about RDF.",
                "source_chunk_ids": ["semantik:slug#blk_a"],
            },
            {
                "claim": "Claim about SHACL.",
                "source_chunk_ids": ["semantik:slug#blk_b"],
            },
        ],
        source_ids=("semantik:slug#blk_a", "semantik:slug#blk_b"),
    )
    source_chunks = {
        "semantik:slug#blk_a": chunk_a,
        "semantik:slug#blk_b": chunk_b,
    }

    result = validator.validate(
        {"blocks": [block], "source_chunks": source_chunks}
    )

    # Two claims, each scoped to ONE chunk → two batch calls each
    # with ONE pair (per-claim attribution).
    assert len(nli.batch_calls) == 2
    for batch in nli.batch_calls:
        assert len(batch) == 1
    # Confirm per-claim scoping: claim A maps to chunk_a, claim B to chunk_b.
    rdf_batch = next(b for b in nli.batch_calls if b[0][1] == "Claim about RDF.")
    shacl_batch = next(b for b in nli.batch_calls if b[0][1] == "Claim about SHACL.")
    assert rdf_batch[0][0] == chunk_a
    assert shacl_batch[0][0] == chunk_b
    assert result.passed is True


# --------------------------------------------------------------------- #
# Test 6: NLI deps missing → graceful-degrade warning
# --------------------------------------------------------------------- #


def test_nli_deps_missing_emits_warning_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_or_load returns None → NLI_DEPS_MISSING warning, passed=True."""
    from lib.classifiers import nli_classifier as mod

    monkeypatch.setattr(
        mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )
    # Ensure strict-mode is OFF.
    monkeypatch.delenv("TRAINFORGE_REQUIRE_EMBEDDINGS", raising=False)

    validator = ClaimSupportValidator()  # nli=None → calls real loader

    block = _make_block(
        key_claims=["A claim."],
        source_ids=("semantik:slug#blk_0",),
    )
    result = validator.validate(
        {"blocks": [block], "source_chunks": {"semantik:slug#blk_0": "Source."}}
    )

    deps_issues = [
        i for i in result.issues if i.code == _CODE_NLI_DEPS_MISSING
    ]
    assert len(deps_issues) == 1
    assert deps_issues[0].severity == "warning"
    assert result.passed is True
    assert result.action is None


# --------------------------------------------------------------------- #
# Test 7: Strict mode + missing deps → raises RuntimeError
# --------------------------------------------------------------------- #


def test_strict_mode_with_missing_deps_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TRAINFORGE_REQUIRE_EMBEDDINGS=true + None NLI → RuntimeError."""
    from lib.classifiers import nli_classifier as mod

    monkeypatch.setattr(
        mod.NliClassifier, "get_or_load", classmethod(lambda cls: None),
    )
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")

    validator = ClaimSupportValidator()  # nli=None

    block = _make_block(
        key_claims=["A claim."],
        source_ids=("semantik:slug#blk_0",),
    )
    with pytest.raises(RuntimeError) as excinfo:
        validator.validate(
            {"blocks": [block], "source_chunks": {"semantik:slug#blk_0": "Source."}}
        )
    assert "TRAINFORGE_REQUIRE_EMBEDDINGS" in str(excinfo.value)


# --------------------------------------------------------------------- #
# Test 8: Skip-set block types (assessment_item, objective) → no audit
# --------------------------------------------------------------------- #


def test_skip_set_block_types_emit_no_events_or_issues() -> None:
    """assessment_item / objective / chrome / recap → no NLI dispatch."""
    nli = _StubNli()
    validator = ClaimSupportValidator(nli=nli)

    blocks = []
    for bt in ("assessment_item", "objective", "chrome", "recap"):
        blocks.append(
            _make_block(
                block_id=f"page_01#{bt}_x_0",
                block_type=bt,
                key_claims=["Claim that should NOT be audited."],
            )
        )

    result = validator.validate(
        {
            "blocks": blocks,
            "source_chunks": {"semantik:slug#blk_0": "Source."},
        }
    )

    # No NLI dispatch — every block was in the skip set.
    assert nli.batch_calls == []
    assert nli.pair_calls == []
    # No claim-support issues.
    assert not any(
        i.code in (_CODE_UNSUPPORTED_CLAIM, _CODE_CONTRADICTED_CLAIM)
        for i in result.issues
    )
    assert result.passed is True
    assert result.action is None


# --------------------------------------------------------------------- #
# Test 9: Decision-capture fires per block with the 6 required signals
# --------------------------------------------------------------------- #


def test_decision_capture_fires_per_block_with_required_signals() -> None:
    """One claim_support_check event per audited block with all 6 fields."""
    captured: List[Dict[str, Any]] = []

    class _CaptureStub:
        def log_decision(
            self,
            *,
            decision_type: str,
            decision: str,
            rationale: str,
            **_: Any,
        ) -> None:
            captured.append(
                {
                    "decision_type": decision_type,
                    "decision": decision,
                    "rationale": rationale,
                }
            )

    nli = _StubNli(
        default_score=NliScore(entailment=0.90, neutral=0.05, contradiction=0.05)
    )
    validator = ClaimSupportValidator(nli=nli)

    block_a = _make_block(
        block_id="page_01#concept_a_0",
        key_claims=["Claim A1.", "Claim A2."],
        source_ids=("semantik:slug#blk_0",),
    )
    block_b = _make_block(
        block_id="page_01#concept_b_1",
        key_claims=["Claim B1."],
        source_ids=("semantik:slug#blk_0",),
    )

    validator.validate(
        {
            "blocks": [block_a, block_b],
            "source_chunks": {"semantik:slug#blk_0": "Source text."},
            "decision_capture": _CaptureStub(),
        }
    )

    events = [e for e in captured if e["decision_type"] == "claim_support_check"]
    # One event per audited block.
    assert len(events) == 2
    # Rationale carries the 6 required signals.
    for event in events:
        rationale = event["rationale"]
        assert "block_id" in rationale.lower() or "block " in rationale.lower()
        assert "total_claims" in rationale
        assert "entailed_count" in rationale
        assert "unsupported_count" in rationale
        assert "contradicted_count" in rationale
        assert "unsupported_claim_rate" in rationale


# --------------------------------------------------------------------- #
# Test 10: Empty key_claims → silent pass, no events
# --------------------------------------------------------------------- #


def test_empty_key_claims_silent_pass_no_events() -> None:
    """Block with no key_claims field → not audited, no events."""
    captured: List[Dict[str, Any]] = []

    class _CaptureStub:
        def log_decision(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    nli = _StubNli()
    validator = ClaimSupportValidator(nli=nli)

    # Block with NO key_claims at all.
    block = _make_block(
        key_claims=None,  # → no "key_claims" field set
        source_ids=("semantik:slug#blk_0",),
    )

    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"semantik:slug#blk_0": "Source."},
            "decision_capture": _CaptureStub(),
        }
    )

    # No audit, no events, no NLI dispatch.
    assert nli.batch_calls == []
    claim_support_events = [
        c for c in captured if c.get("decision_type") == "claim_support_check"
    ]
    assert claim_support_events == []
    assert result.passed is True
    assert result.action is None


# --------------------------------------------------------------------- #
# Bonus coverage: gate-config threshold overrides via inputs dict
# --------------------------------------------------------------------- #


def test_gate_config_thresholds_override_defaults() -> None:
    """Operators can tighten/loosen thresholds via the executor's
    gate.config merge into the inputs dict."""
    chunk_text = "Source chunk."
    nli = _StubNli(
        response_map={
            (chunk_text, "c1"): NliScore(
                entailment=0.30, neutral=0.65, contradiction=0.05,
            ),
        },
    )
    # With default 0.7 entailment floor, this claim is "unsupported"
    # (0.30 < 0.70, 0.05 < 0.50). With a tighter ceiling of 0.05
    # max_unsupported, 1/1 unsupported (100%) > 5% should fire.
    validator = ClaimSupportValidator(nli=nli)
    block = _make_block(key_claims=["c1"], source_ids=("semantik:slug#blk_0",))
    result = validator.validate(
        {
            "blocks": [block],
            "source_chunks": {"semantik:slug#blk_0": chunk_text},
            "max_unsupported_claim_rate": 0.05,
        }
    )
    assert any(
        i.code == _CODE_UNSUPPORTED_CLAIM for i in result.issues
    )
    assert result.action == "regenerate"
