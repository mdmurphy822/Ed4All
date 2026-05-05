"""Worker W3a — tests for ``DistractorMisconceptionAlignmentValidator``.

Covers the four required regression surfaces from the W3a plan:

  - **Happy path** (embedder loaded, refs resolve, similarity above
    threshold) → ``passed=True``, no critical issues.
  - **Unresolved ref** → critical
    ``DISTRACTOR_MISCONCEPTION_REF_UNRESOLVED``.
  - **Misalignment** (cosine below threshold) → critical
    ``DISTRACTOR_MISCONCEPTION_MISALIGNED``.
  - **Embedder missing** (Jaccard fallback) → warning
    ``EMBEDDING_DEPS_MISSING`` issue but the validator continues
    running and Jaccard determines pass/fail.
  - **Distractor without ``misconception_ref``** is skipped (W3b's
    surface, not W3a's).
  - **Decision capture** fires exactly once per ``validate()`` call.

Stub embedder + capture mirror the patterns from the sibling
``test_concept_example_similarity.py`` /
``test_objective_assessment_similarity.py`` so the suite runs WITHOUT
the sentence-transformers extras installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# Repo root + scripts dir on path for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.distractor_misconception_alignment import (  # noqa: E402
    DEFAULT_MIN_COSINE,
    DEFAULT_MIN_JACCARD,
    DistractorMisconceptionAlignmentValidator,
    _build_misconception_index,
    _jaccard,
)


# --------------------------------------------------------------------- #
# Stub embedder + capture (mirrors sibling-validator fixtures).
# --------------------------------------------------------------------- #


class _StubEmbedder:
    """Deterministic embedding stub keyed on text-prefix lookups.

    ``encode(text)`` returns the longest matching prefix's vector;
    misses fall back to a default orthogonal vector so unrelated texts
    cluster at cosine ~0.
    """

    def __init__(self, vector_map: Dict[str, List[float]]) -> None:
        self.vector_map = vector_map
        self.calls: List[str] = []

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        self.calls.append(text)
        match: Tuple[int, str] = (-1, "")
        for key in self.vector_map:
            if text.startswith(key) and len(key) > match[0]:
                match = (len(key), key)
        if match[0] >= 0:
            return self.vector_map[match[1]]
        return [0.0, 0.0, 1.0]


class _RecordingCapture:
    """Minimal stand-in for ``DecisionCapture``; records log_decision calls."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------- #
# Block fixtures.
# --------------------------------------------------------------------- #


def _make_assessment_block(
    *,
    block_id: str = "page_01#assessment_demo_0",
    distractors: List[Dict[str, Any]] = None,
    objective_ids: Tuple[str, ...] = ("TO-01",),
    correct_answer_index: int = 0,
) -> Block:
    distractors = distractors if distractors is not None else [
        {"text": "fallback option a"},
        {"text": "fallback option b"},
    ]
    return Block(
        block_id=block_id,
        block_type="assessment_item",
        page_id="page_01",
        sequence=0,
        objective_ids=objective_ids,
        content={
            "stem": "Stem question?",
            "answer_key": "Correct answer.",
            "distractors": distractors,
            "correct_answer_index": correct_answer_index,
        },
    )


# --------------------------------------------------------------------- #
# Happy path — embedder loaded, ref resolves, similarity above threshold.
# --------------------------------------------------------------------- #


def test_passes_when_distractor_aligns_with_misconception() -> None:
    """High cosine between distractor text and resolved misconception
    statement → ``passed=True``, no critical issues."""
    embedder = _StubEmbedder(
        vector_map={
            "Forces always cause acceleration": [1.0, 0.0, 0.0],
            "Newton's first law: an object at rest": [1.0, 0.0, 0.0],
        }
    )
    validator = DistractorMisconceptionAlignmentValidator(embedder=embedder)

    block = _make_assessment_block(
        distractors=[
            {"text": "Correct answer text"},
            {
                "text": "Forces always cause acceleration in every case",
                "misconception_ref": "TO-01#m1",
            },
        ],
        correct_answer_index=0,
    )
    result = validator.validate({
        "blocks": [block],
        "misconceptions": {
            "TO-01#m1": (
                "Newton's first law: an object at rest stays at rest unless"
                " acted upon by a net external force"
            ),
        },
    })

    assert result.passed
    assert result.action is None
    assert result.score == 1.0
    # No critical issues.
    assert all(i.severity != "critical" for i in result.issues)


# --------------------------------------------------------------------- #
# Unresolved ref → critical.
# --------------------------------------------------------------------- #


def test_unresolved_ref_emits_critical() -> None:
    """A distractor with a ``misconception_ref`` that doesn't exist in
    the inventory → ``DISTRACTOR_MISCONCEPTION_REF_UNRESOLVED`` critical."""
    embedder = _StubEmbedder(vector_map={})
    validator = DistractorMisconceptionAlignmentValidator(embedder=embedder)

    block = _make_assessment_block(
        distractors=[
            {"text": "Correct answer"},
            {
                "text": "Some distractor text",
                "misconception_ref": "PHYS-99#m9",
            },
        ],
    )
    result = validator.validate({
        "blocks": [block],
        # Empty inventory.
        "misconceptions": {},
    })

    assert not result.passed
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "DISTRACTOR_MISCONCEPTION_REF_UNRESOLVED" in codes
    unresolved_issue = next(
        i for i in result.issues
        if i.code == "DISTRACTOR_MISCONCEPTION_REF_UNRESOLVED"
    )
    assert unresolved_issue.severity == "critical"


# --------------------------------------------------------------------- #
# Misalignment (cosine below threshold) → critical.
# --------------------------------------------------------------------- #


def test_low_cosine_emits_misaligned_critical() -> None:
    """Cosine below ``min_cosine`` → ``DISTRACTOR_MISCONCEPTION_MISALIGNED``."""
    embedder = _StubEmbedder(
        vector_map={
            "An entirely unrelated topic": [1.0, 0.0, 0.0],
            "The misconception statement here": [0.0, 1.0, 0.0],
        }
    )
    validator = DistractorMisconceptionAlignmentValidator(embedder=embedder)

    block = _make_assessment_block(
        distractors=[
            {"text": "Correct answer"},
            {
                "text": "An entirely unrelated topic",
                "misconception_ref": "TO-01#m1",
            },
        ],
    )
    result = validator.validate({
        "blocks": [block],
        "misconceptions": {
            "TO-01#m1": "The misconception statement here",
        },
    })

    assert not result.passed
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "DISTRACTOR_MISCONCEPTION_MISALIGNED" in codes
    miss_issue = next(
        i for i in result.issues
        if i.code == "DISTRACTOR_MISCONCEPTION_MISALIGNED"
    )
    assert miss_issue.severity == "critical"
    # Message references metric (cosine) for embedder-loaded path.
    assert "cosine" in miss_issue.message


# --------------------------------------------------------------------- #
# Distractor without misconception_ref is W3b's surface, NOT W3a's.
# --------------------------------------------------------------------- #


def test_distractor_without_ref_is_skipped() -> None:
    """A distractor without a ``misconception_ref`` is not flagged
    by W3a — that's W3b's plausibility axis."""
    embedder = _StubEmbedder(vector_map={})
    validator = DistractorMisconceptionAlignmentValidator(embedder=embedder)

    block = _make_assessment_block(
        distractors=[
            {"text": "Correct answer"},
            {"text": "Distractor without a misconception ref"},
            {"text": "Another distractor without ref"},
        ],
    )
    result = validator.validate({
        "blocks": [block],
        "misconceptions": {},  # Empty inventory; no refs to resolve.
    })

    assert result.passed
    assert result.action is None
    # No critical issues — the no-ref distractors are skipped cleanly.
    assert all(i.severity != "critical" for i in result.issues)


# --------------------------------------------------------------------- #
# Embedder None → Jaccard fallback + EMBEDDING_DEPS_MISSING warning.
# --------------------------------------------------------------------- #


def test_deps_missing_falls_back_to_jaccard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``try_load_embedder()`` returns None, the validator emits
    a single ``EMBEDDING_DEPS_MISSING`` *warning* and falls back to
    Jaccard token-overlap (does NOT fire a critical, does NOT
    short-circuit)."""
    from lib.validators import distractor_misconception_alignment as mod

    monkeypatch.setattr(
        mod._sentence_embedder_mod,
        "try_load_embedder",
        lambda *a, **kw: None,
    )

    validator = DistractorMisconceptionAlignmentValidator()

    # High lexical overlap → Jaccard above min_jaccard floor (0.05).
    block = _make_assessment_block(
        distractors=[
            {"text": "Correct answer"},
            {
                "text": "forces always cause acceleration in every case",
                "misconception_ref": "TO-01#m1",
            },
        ],
    )
    result = validator.validate({
        "blocks": [block],
        "misconceptions": {
            "TO-01#m1": (
                "forces always cause acceleration even at constant velocity"
            ),
        },
    })

    # Warning issue surfaces the silent-degrade signal.
    deps_issues = [
        i for i in result.issues if i.code == "EMBEDDING_DEPS_MISSING"
    ]
    assert len(deps_issues) == 1
    assert deps_issues[0].severity == "warning"

    # NO critical issue for the Jaccard-passing pair.
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []
    assert result.passed
    assert result.action is None


def test_deps_missing_jaccard_below_floor_fires_critical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Jaccard fallback path must still fire critical for genuinely
    misaligned (zero-overlap) distractor/misconception pairs."""
    from lib.validators import distractor_misconception_alignment as mod

    monkeypatch.setattr(
        mod._sentence_embedder_mod,
        "try_load_embedder",
        lambda *a, **kw: None,
    )

    validator = DistractorMisconceptionAlignmentValidator()

    block = _make_assessment_block(
        distractors=[
            {"text": "Correct answer"},
            {
                "text": "wholly different topic xyzzy plugh frobnicate",
                "misconception_ref": "TO-01#m1",
            },
        ],
    )
    result = validator.validate({
        "blocks": [block],
        "misconceptions": {
            "TO-01#m1": "alpha beta gamma delta epsilon",
        },
    })

    assert not result.passed
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "DISTRACTOR_MISCONCEPTION_MISALIGNED" in codes
    miss_issue = next(
        i for i in result.issues
        if i.code == "DISTRACTOR_MISCONCEPTION_MISALIGNED"
    )
    # Jaccard metric reflected in message.
    assert "jaccard" in miss_issue.message


# --------------------------------------------------------------------- #
# Decision capture — exactly one event per validate() call.
# --------------------------------------------------------------------- #


def test_decision_capture_emits_one_event_per_validate_call() -> None:
    """Per CLAUDE.md call-site instrumentation contract: exactly one
    ``distractor_misconception_alignment_check`` decision per
    ``validate()`` call, with rationale carrying replayable signals."""
    embedder = _StubEmbedder(vector_map={})
    validator = DistractorMisconceptionAlignmentValidator(embedder=embedder)
    capture = _RecordingCapture()

    block = _make_assessment_block(
        distractors=[
            {"text": "Correct answer"},
            {
                "text": "Some text",
                "misconception_ref": "TO-01#m1",
            },
            {
                "text": "Another text",
                "misconception_ref": "TO-99#m9",  # unresolved
            },
        ],
    )
    validator.validate({
        "blocks": [block],
        "misconceptions": {"TO-01#m1": "Some text mirror"},
        "decision_capture": capture,
    })

    assert len(capture.events) == 1
    event = capture.events[0]
    assert event["decision_type"] == "distractor_misconception_alignment_check"
    rationale = event["rationale"]
    # Rationale interpolates dynamic signals (per the call-site
    # instrumentation contract — block_id, audited count, threshold,
    # metric, embedder flag).
    assert "audited_distractors_with_ref=2" in rationale
    assert "unresolved_refs=1" in rationale
    assert "metric=cosine" in rationale
    assert "threshold=" in rationale
    assert "embedder_loaded=True" in rationale
    # ≥20-char rationale per the schema's minLength contract.
    assert len(rationale) >= 20


def test_decision_capture_under_jaccard_path_records_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decision-capture rationale records the metric switch (jaccard)
    when the embedder is unavailable — operators can audit silent-
    degrade events from the JSONL stream alone."""
    from lib.validators import distractor_misconception_alignment as mod

    monkeypatch.setattr(
        mod._sentence_embedder_mod,
        "try_load_embedder",
        lambda *a, **kw: None,
    )

    validator = DistractorMisconceptionAlignmentValidator()
    capture = _RecordingCapture()

    block = _make_assessment_block(
        distractors=[{"text": "Some text"}],
    )
    validator.validate({
        "blocks": [block],
        "misconceptions": {},
        "decision_capture": capture,
    })

    assert len(capture.events) == 1
    rationale = capture.events[0]["rationale"]
    assert "metric=jaccard" in rationale
    assert "embedder_loaded=False" in rationale


# --------------------------------------------------------------------- #
# Helper coverage.
# --------------------------------------------------------------------- #


def test_jaccard_helper_basic() -> None:
    """Sanity check for the Jaccard token-overlap helper."""
    assert _jaccard("the quick brown fox", "the quick brown fox") == 1.0
    assert _jaccard("alpha beta", "gamma delta") == 0.0
    # Partial overlap: 2/4.
    overlap = _jaccard("alpha beta gamma", "alpha beta delta")
    assert overlap == pytest.approx(2 / 4)
    # Empty short-circuits to 0.
    assert _jaccard("", "anything") == 0.0


def test_build_misconception_index_from_chunks() -> None:
    """When ``inputs['chunks']`` is supplied, the validator synthesises
    refs of the form ``{lo_ref}#m{N}`` (1-based) for every chunk
    misconception."""
    chunks = [
        {
            "learning_outcome_refs": ["TO-01"],
            "misconceptions": [
                {
                    "misconception": "First misconception statement",
                    "correction": "Correct it like this",
                },
                {
                    "statement": "Second misconception (statement form)",
                    "correction": "Correct it like that",
                },
            ],
        }
    ]
    index = _build_misconception_index({"chunks": chunks})
    assert index["TO-01#m1"] == "First misconception statement"
    assert index["TO-01#m2"] == "Second misconception (statement form)"


def test_build_misconception_index_explicit_wins() -> None:
    """Explicit ``inputs['misconceptions']`` map takes priority over
    chunk-derived synthesis."""
    chunks = [
        {
            "learning_outcome_refs": ["TO-01"],
            "misconceptions": [
                {"misconception": "From chunks", "correction": "x"},
            ],
        }
    ]
    index = _build_misconception_index({
        "chunks": chunks,
        "misconceptions": {"TO-01#m1": "Explicit override"},
    })
    assert index["TO-01#m1"] == "Explicit override"


# --------------------------------------------------------------------- #
# Defensive: missing inputs / non-assessment blocks.
# --------------------------------------------------------------------- #


def test_missing_blocks_input_returns_regenerate_action() -> None:
    """No 'blocks' key -> passed=False, action='regenerate'."""
    validator = DistractorMisconceptionAlignmentValidator(
        embedder=_StubEmbedder(vector_map={})
    )
    result = validator.validate({})
    assert not result.passed
    assert result.action == "regenerate"
    assert result.issues[0].code == "MISSING_BLOCKS_INPUT"


def test_no_assessment_blocks_passes() -> None:
    """A corpus without assessment_item blocks is a no-op (passes)."""
    embedder = _StubEmbedder(vector_map={})
    validator = DistractorMisconceptionAlignmentValidator(embedder=embedder)
    block = Block(
        block_id="page_01#concept_intro_0",
        block_type="concept",
        page_id="page_01",
        sequence=0,
        content={"key_claims": ["Foo."]},
    )
    result = validator.validate({"blocks": [block]})
    assert result.passed
    assert result.action is None
    assert result.issues == []


def test_default_thresholds_are_calibrated() -> None:
    """Sanity: default cosine floor (0.45) > Jaccard floor (0.05) since
    embedder-loaded path is the precision-stricter measurement."""
    assert DEFAULT_MIN_COSINE > DEFAULT_MIN_JACCARD
    assert DEFAULT_MIN_COSINE == 0.45
    assert DEFAULT_MIN_JACCARD == 0.05


def test_action_on_fail_override_for_post_rewrite_tier() -> None:
    """Rewrite + assessment tiers override action_on_fail to 'block'
    via the input dict; outline default is 'regenerate'."""
    embedder = _StubEmbedder(vector_map={})
    validator = DistractorMisconceptionAlignmentValidator(embedder=embedder)
    block = _make_assessment_block(
        distractors=[
            {"text": "Correct"},
            {"text": "Distractor", "misconception_ref": "TO-99#m9"},  # unresolved
        ],
    )
    result = validator.validate({
        "blocks": [block],
        "misconceptions": {},
        "action_on_fail": "block",
    })
    assert not result.passed
    assert result.action == "block"
