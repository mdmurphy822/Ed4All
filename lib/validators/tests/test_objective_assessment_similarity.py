"""Phase 4 Wave N2 Subtask 19 — tests for
ObjectiveAssessmentSimilarityValidator.

Verifies the validator's input handling, embedding-tier dispatch, and
the GateResult.action mapping (regenerate / pass) per Wave N2's
PoC contract. Fixtures construct outline-tier ``Block`` instances
matching ``Courseforge.scripts.blocks.Block`` for the assessment_item
block_type so the validator's per-block-type filter fires.

Tests use a stub embedder (`_StubEmbedder`) returning deterministic
vectors so the suite runs WITHOUT the sentence-transformers extras
installed — Wave N1's fallback contract (warning issue when extras
absent) is exercised separately by the deps-missing test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from lib.validators.objective_assessment_similarity import (  # noqa: E402
    DEFAULT_THRESHOLD,
    ObjectiveAssessmentSimilarityValidator,
)


# --------------------------------------------------------------------- #
# Stub embedder — deterministic vectors keyed on prefix-of-text so the
# tests can pin specific cosine outcomes without running a real model.
# --------------------------------------------------------------------- #


class _StubEmbedder:
    """Returns a per-key vector. The default vector is unit-length on
    a single dimension; identical keys collide on the same vector so
    cosine_similarity returns 1.0 for matched pairs and the configured
    similarity for mismatched pairs.

    The vector_map is keyed on text prefixes — ``encode(text)`` looks
    up the longest matching prefix and returns its vector. Missing keys
    fall back to a default orthogonal vector (cosine ~ 0 against named
    keys).
    """

    def __init__(self, vector_map: Dict[str, List[float]]) -> None:
        self.vector_map = vector_map
        self.calls: List[str] = []

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        self.calls.append(text)
        # Find the longest prefix-match in vector_map.
        match: Tuple[int, str] = (-1, "")
        for key in self.vector_map:
            if text.startswith(key) and len(key) > match[0]:
                match = (len(key), key)
        if match[0] >= 0:
            return self.vector_map[match[1]]
        # Default orthogonal vector.
        return [0.0, 0.0, 1.0]


def _make_assessment_block(
    *,
    block_id: str = "page_01#assessment_item_quiz_0",
    page_id: str = "page_01",
    sequence: int = 0,
    stem: str = "ASSESSMENT: What is the role of federated identity?",
    answer_key: str = "ASSESSMENT: It enables single sign-on across providers.",
    objective_ids: Tuple[str, ...] = ("TO-01",),
) -> Block:
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id=page_id,
        sequence=sequence,
        content={
            "stem": stem,
            "answer_key": answer_key,
        },
        objective_ids=objective_ids,
    )


def _make_concept_block(
    *,
    block_id: str = "page_01#concept_intro_0",
) -> Block:
    """Non-assessment block — should be ignored by the validator."""
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


def test_passes_when_assessment_aligns_with_objective() -> None:
    """High cosine similarity between stem+answer and objective statement
    yields ``passed=True`` and ``action=None``."""
    # Both keys map to the same unit vector → cosine = 1.0.
    embedder = _StubEmbedder(
        vector_map={
            "ASSESSMENT:": [1.0, 0.0, 0.0],
            "OBJECTIVE:": [1.0, 0.0, 0.0],
        }
    )
    validator = ObjectiveAssessmentSimilarityValidator(embedder=embedder)

    block = _make_assessment_block(objective_ids=("TO-01",))
    result = validator.validate(
        {
            "blocks": [block],
            "objective_statements": {
                "TO-01": "OBJECTIVE: Define the role of federated identity in SSO.",
            },
        }
    )

    assert result.passed
    assert result.action is None
    assert result.score == 1.0
    # No critical issues, but warnings are also absent on a happy path.
    assert all(i.severity != "critical" for i in result.issues)


def test_low_similarity_emits_action_regenerate() -> None:
    """Low cosine similarity emits a critical issue with action='regenerate'."""
    # Orthogonal vectors → cosine = 0.0, well below the 0.55 default.
    embedder = _StubEmbedder(
        vector_map={
            "ASSESSMENT:": [1.0, 0.0, 0.0],
            "OBJECTIVE:": [0.0, 1.0, 0.0],
        }
    )
    validator = ObjectiveAssessmentSimilarityValidator(embedder=embedder)

    block = _make_assessment_block(objective_ids=("TO-01",))
    result = validator.validate(
        {
            "blocks": [block],
            "objective_statements": {
                "TO-01": "OBJECTIVE: An unrelated concept on disk encryption.",
            },
        }
    )

    assert not result.passed
    assert result.action == "regenerate"
    assert any(
        i.code == "ASSESSMENT_OBJECTIVE_LOW_SIMILARITY" and i.severity == "critical"
        for i in result.issues
    )


def test_skips_non_assessment_blocks() -> None:
    """Blocks with block_type != 'assessment_item' are ignored entirely."""
    embedder = _StubEmbedder(vector_map={})
    validator = ObjectiveAssessmentSimilarityValidator(embedder=embedder)

    # Only a concept block — no assessments to audit.
    result = validator.validate({"blocks": [_make_concept_block()]})

    assert result.passed
    assert result.action is None
    assert result.issues == []
    assert result.score == 1.0
    # No embedder call — empty assessment list short-circuits before
    # any encode() runs.
    assert embedder.calls == []


def test_missing_blocks_input_returns_regenerate_action() -> None:
    """No 'blocks' key in inputs -> passed=False, action='regenerate'."""
    validator = ObjectiveAssessmentSimilarityValidator(
        embedder=_StubEmbedder(vector_map={})
    )
    result = validator.validate({})

    assert not result.passed
    assert result.action == "regenerate"
    assert len(result.issues) == 1
    assert result.issues[0].code == "MISSING_BLOCKS_INPUT"
    assert result.issues[0].severity == "critical"


def test_threshold_override_via_inputs() -> None:
    """Per-call threshold override via inputs['threshold'] works."""
    # cos = ~0.7 between the two keys (same vector + small noise).
    embedder = _StubEmbedder(
        vector_map={
            "ASSESSMENT:": [1.0, 0.0, 0.0],
            # cosine([1,0,0],[0.7,0.7,0]) = 0.7 / sqrt(0.98) = ~0.707
            "OBJECTIVE:": [0.7, 0.7, 0.0],
        }
    )

    block = _make_assessment_block(objective_ids=("TO-01",))
    inputs: Dict[str, Any] = {
        "blocks": [block],
        "objective_statements": {
            "TO-01": "OBJECTIVE: An assessment-related but distinct objective.",
        },
    }

    # Default threshold (0.55) — passes (0.707 > 0.55).
    v1 = ObjectiveAssessmentSimilarityValidator(embedder=embedder)
    r1 = v1.validate(inputs)
    assert r1.passed, f"Expected pass at threshold {DEFAULT_THRESHOLD}; got {r1.issues}"

    # Override threshold to 0.9 — should fail (0.707 < 0.9).
    inputs_strict = dict(inputs)
    inputs_strict["threshold"] = 0.9
    r2 = v1.validate(inputs_strict)
    assert not r2.passed
    assert r2.action == "regenerate"


def test_warns_when_objective_statement_unresolved() -> None:
    """An assessment referencing an unmapped objective_id yields a warning."""
    embedder = _StubEmbedder(
        vector_map={
            "ASSESSMENT:": [1.0, 0.0, 0.0],
            "OBJECTIVE:": [1.0, 0.0, 0.0],
        }
    )
    validator = ObjectiveAssessmentSimilarityValidator(embedder=embedder)

    block = _make_assessment_block(objective_ids=("TO-01", "TO-99"))
    result = validator.validate(
        {
            "blocks": [block],
            "objective_statements": {
                # Only TO-01 mapped; TO-99 is unresolved.
                "TO-01": "OBJECTIVE: Aligned objective statement.",
            },
        }
    )

    # Passes overall (TO-01 cosine = 1.0 > threshold), but a warning
    # surfaces for TO-99 which has no statement to embed against.
    assert result.passed
    assert any(
        i.code == "OBJECTIVE_STATEMENT_UNRESOLVED" and i.severity == "warning"
        for i in result.issues
    )


def test_deps_missing_emits_warning_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding extras missing -> single warning, passed=True, no action."""
    from lib.validators import objective_assessment_similarity as mod

    # Force try_load_embedder to return None (extras-missing path).
    monkeypatch.setattr(mod, "try_load_embedder", lambda: None)

    validator = ObjectiveAssessmentSimilarityValidator()
    result = validator.validate(
        {
            "blocks": [_make_assessment_block()],
            "objective_statements": {
                "TO-01": "OBJECTIVE: Doesn't matter; embedder missing."
            },
        }
    )

    assert result.passed
    assert result.action is None
    assert len(result.issues) == 1
    assert result.issues[0].code == "EMBEDDING_DEPS_MISSING"
    assert result.issues[0].severity == "warning"


# --------------------------------------------------------------------- #
# H3 Wave W2 — DecisionCapture wiring smoke test.
# --------------------------------------------------------------------- #


class _StubCapture:
    """Stub DecisionCapture that records every log_decision invocation."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, str]] = []

    def log_decision(
        self,
        decision_type: str,
        decision: str,
        rationale: str,
        **kwargs: Any,
    ) -> None:
        self.calls.append((decision_type, decision, rationale))


def test_decision_capture_emits_one_event_per_validate_call() -> None:
    """A single audited block yields exactly one
    ``objective_assessment_similarity_check`` decision capture event,
    with rationale interpolating the cosine + threshold + above/below
    flag dynamic signals."""
    embedder = _StubEmbedder(
        vector_map={
            "ASSESSMENT:": [1.0, 0.0, 0.0],
            "OBJECTIVE:": [1.0, 0.0, 0.0],
        }
    )
    capture = _StubCapture()
    validator = ObjectiveAssessmentSimilarityValidator(embedder=embedder)
    validator.validate(
        {
            "blocks": [_make_assessment_block(objective_ids=("TO-01",))],
            "objective_statements": {
                "TO-01": "OBJECTIVE: Aligned objective.",
            },
            "decision_capture": capture,
        }
    )

    assert len(capture.calls) == 1
    decision_type, decision, rationale = capture.calls[0]
    assert decision_type == "objective_assessment_similarity_check"
    assert decision == "passed"
    assert len(rationale) >= 20
    # Dynamic signals appear in the rationale.
    assert "min_pair_cosine=" in rationale
    assert "threshold=" in rationale
    assert "above_threshold=True" in rationale
    assert "TO-01" in rationale


def test_decision_capture_emits_for_low_similarity_failure() -> None:
    """A failing block yields exactly one capture with decision='failed:...'
    and below-threshold signal in the rationale."""
    embedder = _StubEmbedder(
        vector_map={
            "ASSESSMENT:": [1.0, 0.0, 0.0],
            "OBJECTIVE:": [0.0, 1.0, 0.0],
        }
    )
    capture = _StubCapture()
    validator = ObjectiveAssessmentSimilarityValidator(embedder=embedder)
    validator.validate(
        {
            "blocks": [_make_assessment_block(objective_ids=("TO-01",))],
            "objective_statements": {
                "TO-01": "OBJECTIVE: Unrelated topic.",
            },
            "decision_capture": capture,
        }
    )
    assert len(capture.calls) == 1
    _, decision, rationale = capture.calls[0]
    assert decision.startswith("failed:")
    assert "ASSESSMENT_OBJECTIVE_LOW_SIMILARITY" in decision
    assert "above_threshold=False" in rationale
