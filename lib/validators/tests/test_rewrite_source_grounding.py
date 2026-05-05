"""Tests for RewriteSourceGroundingValidator (Plan §3.5 followup §3.3).

Six required cases:
1. Block fully grounded in source chunks → pass with high rate.
2. Hallucinated content (no overlap) → fail with
   ``REWRITE_SENTENCE_GROUNDING_LOW``.
3. Mixed grounded/hallucinated above 0.60 rate → pass.
4. Block with content_type=assessment_item → skip (no critical
   issue, decision emitted as passed=True / SKIPPED_CONTENT_TYPE).
5. Embedding deps missing (mock try_load_embedder None) →
   warning, passed=True, action=None.
6. Strict mode (TRAINFORGE_REQUIRE_EMBEDDINGS=true) + missing deps
   → critical fail.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

# Repo root + scripts dir on path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.rewrite_source_grounding import (  # noqa: E402
    DEFAULT_MIN_GROUNDED_SENTENCE_RATE,
    DEFAULT_MIN_GROUNDING_COSINE,
    RewriteSourceGroundingValidator,
)
from lib.embedding.sentence_embedder import EmbeddingDepsMissing  # noqa: E402


# ---------------------------------------------------------------------- #
# Stub embedder
# ---------------------------------------------------------------------- #


class _StubEmbedder:
    """Deterministic embedder keyed on text-prefix lookups.

    Vectors that share a leading prefix are similar (high cosine);
    distinct prefixes are orthogonal. Lets us simulate
    paraphrase-of-source vs. fabricated content without loading
    sentence-transformers.
    """

    def __init__(self, vector_map: Dict[str, List[float]]) -> None:
        self.vector_map = vector_map
        self.calls: List[str] = []

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        self.calls.append(text)
        # Longest-prefix match to allow targeted overrides.
        match_key = ""
        for key in self.vector_map:
            if text.startswith(key) and len(key) > len(match_key):
                match_key = key
        if match_key:
            return self.vector_map[match_key]
        # Default orthogonal vector.
        return [0.0, 0.0, 0.0, 1.0]


def _make_block(
    *,
    block_id: str = "page_01#explanation_demo_0",
    block_type: str = "explanation",
    content: str = "<p>Some prose.</p>",
    source_ids=("dart:slug#blk_0",),
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=content,
        source_ids=source_ids,
    )


# ---------------------------------------------------------------------- #
# Helper: synthesize a paragraph of >= 10 content-words sentences.
# ---------------------------------------------------------------------- #


_GROUNDED_HTML = (
    "<section>"
    "<p>Federation describes a trust relationship across multiple independent security domains today. "
    "Identity providers issue cryptographic assertions about authenticated subjects within shared trust anchors. "
    "Service providers verify those assertions against pre-established cryptographic trust anchors carefully.</p>"
    "</section>"
)

_HALLUCINATED_HTML = (
    "<section>"
    "<p>Blockchain consensus mechanisms validate distributed ledger entries among participating network nodes everywhere. "
    "Smart contracts execute deterministic state transitions on virtual machines without trusted central authorities. "
    "Cryptocurrency wallets manage private keys for transaction signing across multiple blockchain networks securely.</p>"
    "</section>"
)

_GROUNDING_SOURCE = (
    "Federation establishes trust between independent security domains via cryptographic assertions widely. "
    "An identity provider issues signed assertions about authenticated users to participating relying parties. "
    "Service providers validate those assertions using shared cryptographic trust anchors today."
)

_GROUNDED_VECTOR = [1.0, 0.0, 0.0, 0.0]
_HALLUCINATED_VECTOR = [0.0, 1.0, 0.0, 0.0]


# ---------------------------------------------------------------------- #
# 1. Grounded block passes
# ---------------------------------------------------------------------- #


def test_block_fully_grounded_passes() -> None:
    embedder = _StubEmbedder(
        vector_map={
            # Source chunk + every sentence of the grounded HTML start
            # with prefixes that map to the same vector.
            "Federation": _GROUNDED_VECTOR,
            "An identity": _GROUNDED_VECTOR,
            "Service providers": _GROUNDED_VECTOR,
            "Identity providers": _GROUNDED_VECTOR,
        }
    )
    block = _make_block(content=_GROUNDED_HTML)
    result = RewriteSourceGroundingValidator(embedder=embedder).validate({
        "blocks": [block],
        "source_chunks": {"dart:slug#blk_0": _GROUNDING_SOURCE},
    })
    assert result.passed is True
    assert result.action is None
    assert all(i.severity != "critical" for i in result.issues)


# ---------------------------------------------------------------------- #
# 2. Hallucinated block fails
# ---------------------------------------------------------------------- #


def test_hallucinated_block_fails_with_grounding_low() -> None:
    embedder = _StubEmbedder(
        vector_map={
            # Source vector + grounded prefixes
            "Federation": _GROUNDED_VECTOR,
            # Hallucinated sentences map to an orthogonal vector.
            "Blockchain": _HALLUCINATED_VECTOR,
            "Smart contracts": _HALLUCINATED_VECTOR,
            "Cryptocurrency": _HALLUCINATED_VECTOR,
        }
    )
    block = _make_block(content=_HALLUCINATED_HTML)
    result = RewriteSourceGroundingValidator(embedder=embedder).validate({
        "blocks": [block],
        "source_chunks": {"dart:slug#blk_0": _GROUNDING_SOURCE},
    })
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "REWRITE_SENTENCE_GROUNDING_LOW" in codes


# ---------------------------------------------------------------------- #
# 3. Mixed grounded / hallucinated above the 0.60 rate passes
# ---------------------------------------------------------------------- #


def test_mostly_grounded_with_one_hallucinated_passes() -> None:
    """3 grounded + 1 hallucinated = 75% rate ≥ 0.60 threshold."""
    mixed_html = (
        "<section>"
        "<p>Federation describes a trust relationship across multiple independent security domains today. "
        "Blockchain wallets manage private keys for transaction signing across multiple networks securely. "
        "Identity providers issue cryptographic assertions about authenticated subjects within shared trust anchors. "
        "Service providers verify those assertions against pre-established cryptographic trust anchors carefully.</p>"
        "</section>"
    )
    embedder = _StubEmbedder(
        vector_map={
            "Federation": _GROUNDED_VECTOR,
            "Blockchain": _HALLUCINATED_VECTOR,
            "Identity providers": _GROUNDED_VECTOR,
            "Service providers": _GROUNDED_VECTOR,
            "An identity": _GROUNDED_VECTOR,
        }
    )
    block = _make_block(content=mixed_html)
    result = RewriteSourceGroundingValidator(embedder=embedder).validate({
        "blocks": [block],
        "source_chunks": {"dart:slug#blk_0": _GROUNDING_SOURCE},
    })
    assert result.passed is True
    assert result.action is None


# ---------------------------------------------------------------------- #
# 4. assessment_item is skipped
# ---------------------------------------------------------------------- #


def test_assessment_item_skipped() -> None:
    embedder = _StubEmbedder(vector_map={})  # never called
    block = _make_block(
        block_type="assessment_item",
        content=_HALLUCINATED_HTML,  # would fail, but skipped
        source_ids=("dart:slug#blk_0",),
    )
    result = RewriteSourceGroundingValidator(embedder=embedder).validate({
        "blocks": [block],
        "source_chunks": {"dart:slug#blk_0": _GROUNDING_SOURCE},
    })
    assert result.passed is True
    assert result.action is None
    # No critical issues — the block was skipped.
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []


# ---------------------------------------------------------------------- #
# 4b. example block is skipped (URI-literal payload class)
# ---------------------------------------------------------------------- #


def test_example_block_skipped() -> None:
    """``example`` blocks legitimately carry payload-heavy content
    (URI literals, code fragments, schematic triples) whose per-sentence
    cosine against source prose drops below the floor even when the
    framing prose is grounded. ``concept_example_similarity`` is the
    sibling validator that gates the right signal for this block type.
    """
    embedder = _StubEmbedder(vector_map={})  # never called
    block = _make_block(
        block_type="example",
        content=_HALLUCINATED_HTML,  # would fail, but skipped
        source_ids=("dart:slug#blk_0",),
    )
    result = RewriteSourceGroundingValidator(embedder=embedder).validate({
        "blocks": [block],
        "source_chunks": {"dart:slug#blk_0": _GROUNDING_SOURCE},
    })
    assert result.passed is True
    assert result.action is None
    # No critical issues — the block was skipped.
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []


def test_example_block_skip_emits_decision_with_skip_code() -> None:
    """``example`` block skip emits a passed=True decision with
    SKIPPED_CONTENT_TYPE so the audit trail records the skip."""

    class _StubCapture:
        def __init__(self) -> None:
            self.calls = []

        def log_decision(self, decision_type, decision, rationale, **kw):
            self.calls.append((decision_type, decision, rationale))

    block = _make_block(block_type="example", content=_HALLUCINATED_HTML)
    capture = _StubCapture()
    RewriteSourceGroundingValidator(embedder=_StubEmbedder({})).validate({
        "blocks": [block],
        "source_chunks": {},
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0][1] == "passed"
    assert "SKIPPED_CONTENT_TYPE" in capture.calls[0][2]


# ---------------------------------------------------------------------- #
# 5. Embedding deps missing → warning, passed=True
# ---------------------------------------------------------------------- #


def test_embedding_deps_missing_warns_and_passes() -> None:
    """When try_load_embedder returns None, the gate degrades to warn."""
    block = _make_block(content=_GROUNDED_HTML)
    # Patch the loader at the validator's import site.
    with patch(
        "lib.validators.rewrite_source_grounding.try_load_embedder",
        return_value=None,
    ):
        # Ensure strict mode is off for this test.
        prev = os.environ.pop("TRAINFORGE_REQUIRE_EMBEDDINGS", None)
        try:
            result = RewriteSourceGroundingValidator().validate({
                "blocks": [block],
                "source_chunks": {"dart:slug#blk_0": _GROUNDING_SOURCE},
            })
        finally:
            if prev is not None:
                os.environ["TRAINFORGE_REQUIRE_EMBEDDINGS"] = prev
    assert result.passed is True
    assert result.action is None
    codes = [i.code for i in result.issues if i.severity == "warning"]
    assert "EMBEDDING_DEPS_MISSING" in codes


# ---------------------------------------------------------------------- #
# 6. Strict mode + missing deps → critical fail
# ---------------------------------------------------------------------- #


def test_strict_mode_missing_deps_fails_critical() -> None:
    """try_load_embedder raises EmbeddingDepsMissing in strict mode."""
    block = _make_block(content=_GROUNDED_HTML)

    def _raise_strict(*args, **kwargs):
        raise EmbeddingDepsMissing("sentence-transformers not installed")

    with patch(
        "lib.validators.rewrite_source_grounding.try_load_embedder",
        side_effect=_raise_strict,
    ):
        result = RewriteSourceGroundingValidator().validate({
            "blocks": [block],
            "source_chunks": {"dart:slug#blk_0": _GROUNDING_SOURCE},
        })
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues if i.severity == "critical"]
    assert "EMBEDDING_DEPS_MISSING" in codes


# ---------------------------------------------------------------------- #
# Decision-capture wiring smoke test
# ---------------------------------------------------------------------- #


def test_decision_capture_emits_per_block() -> None:
    class _StubCapture:
        def __init__(self) -> None:
            self.calls = []

        def log_decision(
            self,
            decision_type: str,
            decision: str,
            rationale: str,
            **kwargs,
        ) -> None:
            self.calls.append((decision_type, decision, rationale))

    embedder = _StubEmbedder(vector_map={
        "Federation": _GROUNDED_VECTOR,
        "Identity providers": _GROUNDED_VECTOR,
        "Service providers": _GROUNDED_VECTOR,
        "An identity": _GROUNDED_VECTOR,
    })
    capture = _StubCapture()
    block = _make_block(content=_GROUNDED_HTML)
    RewriteSourceGroundingValidator(embedder=embedder).validate({
        "blocks": [block],
        "source_chunks": {"dart:slug#blk_0": _GROUNDING_SOURCE},
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0][0] == "rewrite_source_grounding_check"
    rationale = capture.calls[0][2]
    assert len(rationale) >= 20
    # Rationale carries dynamic signals (block id, block type, grounded rate, threshold).
    assert "page_01#explanation_demo_0" in rationale
    assert "block_type=" in rationale
    assert "grounded_rate=" in rationale
    assert "min_rate_threshold=" in rationale


def test_assessment_item_skip_emits_decision_with_skip_code() -> None:
    """assessment_item is skipped but emits a passed=True decision."""

    class _StubCapture:
        def __init__(self) -> None:
            self.calls = []

        def log_decision(self, decision_type, decision, rationale, **kw):
            self.calls.append((decision_type, decision, rationale))

    block = _make_block(block_type="assessment_item", content=_HALLUCINATED_HTML)
    capture = _StubCapture()
    RewriteSourceGroundingValidator(embedder=_StubEmbedder({})).validate({
        "blocks": [block],
        "source_chunks": {},
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0][1] == "passed"
    assert "SKIPPED_CONTENT_TYPE" in capture.calls[0][2]


def test_no_grounding_source_warns_but_passes() -> None:
    """Block with no source chunks emits a warning but doesn't critical-fail."""
    embedder = _StubEmbedder(vector_map={})
    block = _make_block(content=_GROUNDED_HTML, source_ids=())
    result = RewriteSourceGroundingValidator(embedder=embedder).validate({
        "blocks": [block],
        "source_chunks": {},
    })
    assert result.passed is True
    assert result.action is None
    codes = [i.code for i in result.issues if i.severity == "warning"]
    assert "REWRITE_NO_GROUNDING_SOURCE" in codes
