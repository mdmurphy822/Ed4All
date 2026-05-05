"""Phase 4 Subtask 27 — tests for ``BloomClassifierDisagreementValidator``.

Restores per-validator test parity with the other three Phase 4
embedding validators (objective-assessment, concept-example,
objective-roundtrip). The validator wraps
:class:`lib.classifiers.bloom_bert_ensemble.BloomBertEnsemble` so the
suite stubs the ensemble surface (``_load_members`` + ``classify``)
to avoid pulling HuggingFace weights at test time — CI fresh-checkout
must not depend on the ``[embedding]`` extras for this module.

Cases covered:

1. Happy path — ensemble agrees with declared ``bloom_level`` and
   dispersion is below threshold; ``passed=True, action=None``.
2. Majority disagreement — ensemble winner != declared level (above
   the confidence floor); emits ``BERT_ENSEMBLE_DISAGREEMENT`` and
   sets ``action="regenerate"``.
3. High dispersion — split votes across multiple levels push entropy
   above the dispersion threshold; emits
   ``BERT_ENSEMBLE_DISPERSION_HIGH`` and sets ``action="regenerate"``.
4. Graceful-degrade — ``_load_members`` returns ``[]`` (default-mode
   missing-extras path); emits a single warning-severity
   ``BERT_ENSEMBLE_DEPS_MISSING`` GateIssue with ``passed=True,
   action=None``.
5. Strict-mode — ``_load_members`` raises
   :class:`BertEnsembleDepsMissing`; surfaces a critical-severity
   GateIssue with ``passed=False, action="block"``.
6. Threshold knob — ``dispersion_threshold=0.99`` (loose) lets a
   split-vote case pass that would otherwise have flagged.
7. Skips non-audited block types — ``concept`` blocks (and any
   block_type outside ``_AUDITED_BLOCK_TYPES``) are not classified
   even when present alongside audited blocks.
8. Skips audited blocks with empty ``bloom_level`` — the validator
   can't disagree with an unstated level, so the block is silently
   skipped (no classify call, no event).
9. Missing ``blocks`` input — returns critical
   ``MISSING_BLOCKS_INPUT`` issue with ``passed=False, action="block"``.

The stub ensemble class records every ``classify`` invocation so each
test can assert which blocks reached the classifier (vs which the
per-block-type / declared-level filters short-circuited past).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Repo root on path for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Block lives at Courseforge/scripts/blocks.py — import bridge mirror.
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.classifiers.bloom_bert_ensemble import (  # noqa: E402
    BertEnsembleDepsMissing,
)
from lib.validators.bloom_classifier_disagreement import (  # noqa: E402
    _DISAGREEMENT_CONFIDENCE_FLOOR,
    _DISPERSION_THRESHOLD,
    BloomClassifierDisagreementValidator,
)


# --------------------------------------------------------------------- #
# Stub ensemble — records calls + returns deterministic per-text
# classify results without instantiating any HuggingFace model.
# --------------------------------------------------------------------- #


class _StubEnsemble:
    """Drop-in test stub for ``BloomBertEnsemble``.

    The validator's contract on its ensemble dependency is just
    ``_load_members()`` (returns a truthy list when ready) and
    ``classify(text) -> {winner_level, winner_score, dispersion,
    per_member}``. The stub emulates both with a per-text response map
    so each test can pin specific aggregation outcomes.

    ``load_behavior`` controls the ``_load_members`` outcome:

    - ``"loaded"`` (default): returns a single sentinel member so the
      validator proceeds to the per-block classify loop.
    - ``"empty"``: returns ``[]`` to drive the graceful-degrade path.
    - ``"raise"``: raises :class:`BertEnsembleDepsMissing` to drive
      the strict-mode critical path.
    """

    def __init__(
        self,
        response_map: Optional[Dict[str, Dict[str, Any]]] = None,
        default_response: Optional[Dict[str, Any]] = None,
        load_behavior: str = "loaded",
    ) -> None:
        self.response_map = response_map or {}
        self.default_response = default_response or {
            "winner_level": "remember",
            "winner_score": 0.5,
            "dispersion": 0.0,
            "per_member": [("remember", 0.5)],
        }
        self.load_behavior = load_behavior
        self.classify_calls: List[str] = []

    def _load_members(self) -> List[Any]:
        if self.load_behavior == "empty":
            return []
        if self.load_behavior == "raise":
            raise BertEnsembleDepsMissing(
                "transformers extras missing in strict mode (test stub)"
            )
        # Sentinel single-member list — the validator only checks
        # truthiness, so the list contents don't need to be real.
        return [object()]

    def classify(self, text: str) -> Dict[str, Any]:
        self.classify_calls.append(text)
        # Match by exact text; fall back to the default response.
        return dict(self.response_map.get(text, self.default_response))


def _make_assessment_block(
    *,
    block_id: str = "page_01#assessment_item_quiz_0",
    page_id: str = "page_01",
    sequence: int = 0,
    statement: str = "Identify the main themes of the passage.",
    bloom_level: Optional[str] = "remember",
) -> Block:
    """Build an audited (``assessment_item``) Block with the given
    declared ``bloom_level`` + ``content["statement"]``.

    ``content["statement"]`` is the canonical surface the validator's
    ``_extract_text_for_classification`` pulls first; the test response
    map keys against this text so the stub's classify outcome is
    reproducible.
    """
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id=page_id,
        sequence=sequence,
        content={"statement": statement},
        bloom_level=bloom_level,
    )


def _make_objective_block(
    *,
    block_id: str = "page_01#objective_intro_0",
    statement: str = "Define the role of federated identity in SSO.",
    bloom_level: Optional[str] = "understand",
) -> Block:
    """Build an audited (``objective``) Block."""
    return Block(
        block_id=block_id,
        block_type="objective",
        page_id="page_01",
        sequence=0,
        content={"statement": statement},
        bloom_level=bloom_level,
    )


def _make_concept_block(
    *,
    block_id: str = "page_01#concept_intro_0",
) -> Block:
    """Non-audited Block (block_type outside ``_AUDITED_BLOCK_TYPES``)."""
    return Block(
        block_id=block_id,
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={"key_claims": ["Federation requires trust."]},
    )


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_passes_when_ensemble_agrees_with_declared_bloom_level() -> None:
    """Ensemble winner == declared level + low dispersion -> pass."""
    text = "Recall the definition of single sign-on."
    ensemble = _StubEnsemble(
        response_map={
            text: {
                "winner_level": "remember",
                "winner_score": 0.92,
                "dispersion": 0.1,
                "per_member": [("remember", 0.95), ("remember", 0.89)],
            }
        }
    )
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)

    block = _make_assessment_block(statement=text, bloom_level="remember")
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert result.issues == []
    # Ensemble was actually consulted on the audited block.
    assert ensemble.classify_calls == [text]


def test_majority_disagreement_emits_regenerate() -> None:
    """Ensemble winner != declared (above confidence floor) -> regenerate."""
    text = "Construct a novel argument refuting the textbook claim."
    ensemble = _StubEnsemble(
        response_map={
            text: {
                "winner_level": "create",
                # Above the 0.4 confidence floor so the disagreement
                # signal isn't filtered as low-confidence noise.
                "winner_score": 0.78,
                "dispersion": 0.2,
                "per_member": [("create", 0.85), ("create", 0.71)],
            }
        }
    )
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)

    block = _make_assessment_block(statement=text, bloom_level="remember")
    result = validator.validate({"blocks": [block]})

    # The validator's pass field stays True (warning-severity issues
    # only) but action='regenerate' tells the router to re-roll.
    assert result.passed
    assert result.action == "regenerate"
    assert any(
        issue.code == "BERT_ENSEMBLE_DISAGREEMENT"
        and issue.severity == "warning"
        for issue in result.issues
    )


def test_low_confidence_disagreement_does_not_regenerate() -> None:
    """Disagreement below the confidence floor is treated as noise."""
    text = "Restate the lecture's opening argument in your own words."
    ensemble = _StubEnsemble(
        response_map={
            text: {
                "winner_level": "evaluate",
                # Strictly below the 0.4 confidence floor — the
                # validator filters this as low-confidence noise.
                "winner_score": 0.3,
                "dispersion": 0.2,
                "per_member": [("evaluate", 0.3), ("understand", 0.25)],
            }
        }
    )
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)

    block = _make_assessment_block(statement=text, bloom_level="understand")
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    # No DISAGREEMENT issue and dispersion is below threshold so no
    # DISPERSION_HIGH issue either.
    assert not any(
        issue.code == "BERT_ENSEMBLE_DISAGREEMENT" for issue in result.issues
    )


def test_high_dispersion_emits_dispersion_high() -> None:
    """Dispersion > threshold -> regenerate even when the winner agrees."""
    text = "Apply the framework to a complex contemporary case."
    ensemble = _StubEnsemble(
        response_map={
            text: {
                # Winner agrees with the declared level so there's NO
                # disagreement signal — the dispersion check is the
                # ONLY thing that fires here.
                "winner_level": "apply",
                "winner_score": 0.4,
                # Strictly above the default 0.7 dispersion threshold.
                "dispersion": 0.95,
                "per_member": [
                    ("apply", 0.4),
                    ("analyze", 0.35),
                    ("understand", 0.3),
                ],
            }
        }
    )
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)

    block = _make_assessment_block(statement=text, bloom_level="apply")
    result = validator.validate({"blocks": [block]})

    assert result.action == "regenerate"
    assert any(
        issue.code == "BERT_ENSEMBLE_DISPERSION_HIGH"
        and issue.severity == "warning"
        for issue in result.issues
    )
    # No disagreement issue (winner == declared).
    assert not any(
        issue.code == "BERT_ENSEMBLE_DISAGREEMENT"
        for issue in result.issues
    )


def test_dispersion_threshold_knob_respected() -> None:
    """Loosening ``dispersion_threshold`` lets a noisy split-vote pass."""
    text = "Identify three implications of the policy change."
    ensemble = _StubEnsemble(
        response_map={
            text: {
                "winner_level": "analyze",
                "winner_score": 0.4,
                # Same dispersion as the previous test but the
                # validator now tolerates it.
                "dispersion": 0.95,
                "per_member": [
                    ("analyze", 0.4),
                    ("evaluate", 0.35),
                    ("apply", 0.3),
                ],
            }
        }
    )
    validator = BloomClassifierDisagreementValidator(
        ensemble=ensemble,
        dispersion_threshold=0.99,
    )

    block = _make_assessment_block(statement=text, bloom_level="analyze")
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert not any(
        issue.code == "BERT_ENSEMBLE_DISPERSION_HIGH"
        for issue in result.issues
    )


def test_graceful_degrade_when_extras_missing() -> None:
    """``_load_members`` returns [] -> single warning, passes."""
    ensemble = _StubEnsemble(load_behavior="empty")
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)

    block = _make_assessment_block()
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.code == "BERT_ENSEMBLE_DEPS_MISSING"
    assert issue.severity == "warning"
    # The classify path must NOT have been entered when no members
    # loaded.
    assert ensemble.classify_calls == []


def test_strict_mode_raises_surface_as_critical() -> None:
    """``_load_members`` raises -> critical issue, passed=False, blocks."""
    ensemble = _StubEnsemble(load_behavior="raise")
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)

    block = _make_assessment_block()
    result = validator.validate({"blocks": [block]})

    assert not result.passed
    assert result.action == "block"
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.code == "BERT_ENSEMBLE_DEPS_MISSING"
    assert issue.severity == "critical"


def test_skips_non_audited_block_types() -> None:
    """``concept`` blocks bypass the classifier even when audited
    blocks are present alongside them."""
    audited_text = "Recall the canonical SAML assertion fields."
    ensemble = _StubEnsemble(
        response_map={
            audited_text: {
                "winner_level": "remember",
                "winner_score": 0.8,
                "dispersion": 0.1,
                "per_member": [("remember", 0.85)],
            }
        }
    )
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)

    blocks = [
        _make_concept_block(),  # Skipped — block_type='concept'.
        _make_assessment_block(statement=audited_text, bloom_level="remember"),
    ]
    result = validator.validate({"blocks": blocks})

    assert result.passed
    assert result.action is None
    # The concept block's text was NEVER passed to classify; only the
    # audited assessment block's statement reached the ensemble.
    assert ensemble.classify_calls == [audited_text]


def test_skips_audited_blocks_without_declared_bloom_level() -> None:
    """Audited block with empty/None ``bloom_level`` is silently skipped."""
    ensemble = _StubEnsemble()
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)

    block = _make_objective_block(
        statement="Recognize the components of a federated identity flow.",
        bloom_level=None,
    )
    result = validator.validate({"blocks": [block]})

    assert result.passed
    assert result.action is None
    assert result.issues == []
    # No classifier call — undeclared level short-circuits before
    # text extraction.
    assert ensemble.classify_calls == []


def test_missing_blocks_input_is_critical_block() -> None:
    """``inputs['blocks']`` absent -> critical, passed=False, action='block'."""
    validator = BloomClassifierDisagreementValidator(ensemble=_StubEnsemble())
    result = validator.validate({})

    assert not result.passed
    assert result.action == "block"
    assert len(result.issues) == 1
    assert result.issues[0].code == "MISSING_BLOCKS_INPUT"
    assert result.issues[0].severity == "critical"


def test_default_thresholds_match_module_constants() -> None:
    """The validator's defaults must agree with the module-level constants
    documented in the docstring + Phase 4 plan."""
    validator = BloomClassifierDisagreementValidator()
    assert validator._dispersion_threshold == _DISPERSION_THRESHOLD
    assert validator._confidence_floor == _DISAGREEMENT_CONFIDENCE_FLOOR


# --------------------------------------------------------------------- #
# H3 Wave W2 — DecisionCapture wiring smoke test.
# --------------------------------------------------------------------- #


class _StubCapture:
    """Records every log_decision invocation."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, str]] = []

    def log_decision(self, decision_type, decision, rationale, **kwargs):
        self.calls.append((decision_type, decision, rationale))


def test_decision_capture_emits_one_event_per_validate_call() -> None:
    """A single audited block yields exactly one
    ``bloom_classifier_disagreement_check`` decision capture event, with
    rationale interpolating ensemble winner + dispersion + declared
    bloom_level + member-vote dynamic signals."""
    text = "Recall the definition of single sign-on."
    ensemble = _StubEnsemble(
        response_map={
            text: {
                "winner_level": "remember",
                "winner_score": 0.92,
                "dispersion": 0.1,
                "per_member": [("remember", 0.95), ("remember", 0.89)],
            }
        }
    )
    capture = _StubCapture()
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)
    block = _make_assessment_block(statement=text, bloom_level="remember")
    validator.validate({"blocks": [block], "decision_capture": capture})

    assert len(capture.calls) == 1
    decision_type, decision, rationale = capture.calls[0]
    assert decision_type == "bloom_classifier_disagreement_check"
    assert decision == "passed"
    assert len(rationale) >= 20
    # Dynamic signals appear in the rationale.
    assert "declared_level=remember" in rationale
    assert "ensemble_winner=remember" in rationale
    assert "dispersion=" in rationale
    assert "dispersion_threshold=" in rationale
    assert "member_votes=" in rationale
    assert "members_loaded=" in rationale


def test_decision_capture_emits_for_disagreement_failure() -> None:
    """A disagreeing block yields exactly one capture with
    decision='failed:...' carrying the BERT_ENSEMBLE_DISAGREEMENT code
    and the ensemble's winner level in the rationale."""
    text = "Construct a novel argument refuting the textbook claim."
    ensemble = _StubEnsemble(
        response_map={
            text: {
                "winner_level": "create",
                "winner_score": 0.78,
                "dispersion": 0.2,
                "per_member": [("create", 0.85), ("create", 0.71)],
            }
        }
    )
    capture = _StubCapture()
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)
    block = _make_assessment_block(statement=text, bloom_level="remember")
    validator.validate({"blocks": [block], "decision_capture": capture})

    assert len(capture.calls) == 1
    _, decision, rationale = capture.calls[0]
    assert decision.startswith("failed:")
    assert "BERT_ENSEMBLE_DISAGREEMENT" in decision
    assert "ensemble_winner=create" in rationale
    assert "declared_level=remember" in rationale


def test_per_block_classify_failure_is_silently_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception raised by ``ensemble.classify`` for one block does
    not abort the validator — that block is logged and skipped, the
    remaining blocks proceed normally."""

    text_ok = "Define the components of an identity provider."
    text_bad = "Construct a counterexample to the textbook proof."

    class _PartiallyBrokenEnsemble(_StubEnsemble):
        def classify(self, text: str) -> Dict[str, Any]:
            self.classify_calls.append(text)
            if text == text_bad:
                raise RuntimeError("simulated per-block failure")
            return dict(self.default_response)

    ensemble = _PartiallyBrokenEnsemble(
        default_response={
            "winner_level": "remember",
            "winner_score": 0.8,
            "dispersion": 0.1,
            "per_member": [("remember", 0.8)],
        }
    )
    validator = BloomClassifierDisagreementValidator(ensemble=ensemble)

    blocks = [
        _make_assessment_block(
            block_id="page_01#assessment_item_q1_0",
            statement=text_ok,
            bloom_level="remember",
        ),
        _make_assessment_block(
            block_id="page_01#assessment_item_q2_1",
            sequence=1,
            statement=text_bad,
            bloom_level="create",
        ),
    ]
    result = validator.validate({"blocks": blocks})

    # Both blocks were attempted; the failing one was logged + skipped
    # without aborting the whole gate.
    assert ensemble.classify_calls == [text_ok, text_bad]
    assert result.passed
    # No issues from either block — the OK block agreed with its
    # declared level; the failing block was skipped before any issue
    # could be emitted against it.
    assert result.issues == []
