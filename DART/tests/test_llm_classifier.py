"""Tests for the Wave 14 Claude-backed DART block classifier.

These tests exclusively use :class:`MockBackend` from Wave 7's LLM
abstraction — no real API traffic ever leaves the test process. The
goal is to exercise the contract:

* require an injected backend;
* batch blocks correctly into ~20-block chunks;
* parse the canonical JSON response shape into ``ClassifiedBlock``;
* fall back to the heuristic classifier on any partial failure,
  never crashing the pipeline;
* expose the factory gate (``DART_LLM_CLASSIFICATION``) via
  ``default_classifier``.
"""

from __future__ import annotations

import json
import math
from typing import List

import pytest

from DART.converter import (
    BlockRole,
    HeuristicClassifier,
    LLMClassifier,
    RawBlock,
    default_classifier,
)
from MCP.orchestrator.llm_backend import MockBackend


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_blocks(n: int, text_prefix: str = "Sample block") -> List[RawBlock]:
    """Build ``n`` RawBlocks with stable IDs and neighbour context.

    Neighbour text is populated because the classifier prompt
    intentionally leans on it; tests that assert on prompt content
    verify this survives batching.
    """
    blocks: List[RawBlock] = []
    for i in range(n):
        blocks.append(
            RawBlock(
                text=f"{text_prefix} {i}",
                block_id=f"blk{i:06d}",
                page=1,
                extractor="pdftotext",
                neighbors={
                    "prev": f"Prev context {i}" if i > 0 else "",
                    "next": f"Next context {i}" if i < n - 1 else "",
                },
            )
        )
    return blocks


def _all_paragraph_response(blocks: List[RawBlock]) -> str:
    """Return a valid LLM-shape JSON response tagging every block as paragraph."""
    payload = [
        {
            "block_id": b.block_id,
            "role": BlockRole.PARAGRAPH.value,
            "confidence": 0.9,
            "attributes": {},
        }
        for b in blocks
    ]
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# LLMClassifier — construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestLLMClassifierConstruction:
    def test_llm_classifier_requires_backend(self):
        """Instantiating without ``llm=`` must raise a clear error."""
        with pytest.raises(ValueError) as exc_info:
            LLMClassifier()
        message = str(exc_info.value)
        assert "LLMBackend" in message
        assert "llm=" in message

    def test_llm_classifier_rejects_non_positive_batch(self):
        backend = MockBackend(responses=["[]"])
        with pytest.raises(ValueError):
            LLMClassifier(llm=backend, batch_size=0)


# ---------------------------------------------------------------------------
# LLMClassifier — batching + happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestLLMClassifierBatching:
    @pytest.mark.asyncio
    async def test_llm_classifier_batches_correctly(self):
        """50 blocks at batch_size=20 => ceil(50/20)=3 backend calls."""
        blocks = _make_blocks(50)
        # Provide one canned response per expected batch.
        responses = [
            _all_paragraph_response(blocks[0:20]),
            _all_paragraph_response(blocks[20:40]),
            _all_paragraph_response(blocks[40:50]),
        ]
        backend = MockBackend(responses=responses)

        classifier = LLMClassifier(llm=backend, batch_size=20)
        result = await classifier.classify(blocks)

        expected_calls = math.ceil(50 / 20)
        assert expected_calls == 3
        assert len(backend.calls) == expected_calls
        assert len(result) == 50
        assert all(cb.role == BlockRole.PARAGRAPH for cb in result)
        assert all(cb.classifier_source == "llm" for cb in result)

    @pytest.mark.asyncio
    async def test_llm_classifier_handles_empty_input(self):
        """Empty input must yield empty output without hitting the backend."""
        backend = MockBackend(responses=[])
        classifier = LLMClassifier(llm=backend)
        result = await classifier.classify([])
        assert result == []
        assert backend.calls == []

    @pytest.mark.asyncio
    async def test_llm_classifier_respects_custom_batch_size(self):
        blocks = _make_blocks(10)
        responses = [
            _all_paragraph_response(blocks[0:3]),
            _all_paragraph_response(blocks[3:6]),
            _all_paragraph_response(blocks[6:9]),
            _all_paragraph_response(blocks[9:10]),
        ]
        backend = MockBackend(responses=responses)
        classifier = LLMClassifier(llm=backend, batch_size=3)
        result = await classifier.classify(blocks)
        assert len(backend.calls) == 4
        assert len(result) == 10


# ---------------------------------------------------------------------------
# LLMClassifier — response parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestLLMClassifierParsing:
    @pytest.mark.asyncio
    async def test_llm_classifier_parses_valid_response(self):
        """Canned reply yields ClassifiedBlock list with llm source + attrs."""
        blocks = _make_blocks(2)
        reply = json.dumps(
            [
                {
                    "block_id": blocks[0].block_id,
                    "role": "chapter_opener",
                    "confidence": 0.92,
                    "attributes": {"heading_text": "Foundations"},
                },
                {
                    "block_id": blocks[1].block_id,
                    "role": "paragraph",
                    "confidence": 0.7,
                    "attributes": {},
                },
            ]
        )
        backend = MockBackend(responses=[reply])
        classifier = LLMClassifier(llm=backend)
        result = await classifier.classify(blocks)

        assert len(result) == 2
        assert result[0].role == BlockRole.CHAPTER_OPENER
        assert result[0].confidence == pytest.approx(0.92)
        assert result[0].classifier_source == "llm"
        assert result[0].attributes["heading_text"] == "Foundations"
        assert result[1].role == BlockRole.PARAGRAPH

    @pytest.mark.asyncio
    async def test_llm_classifier_tolerates_missing_confidence_and_attributes(self):
        """Missing optional keys default sensibly; role still wins."""
        blocks = _make_blocks(1)
        reply = json.dumps(
            [{"block_id": blocks[0].block_id, "role": "paragraph"}]
        )
        backend = MockBackend(responses=[reply])
        classifier = LLMClassifier(llm=backend)
        result = await classifier.classify(blocks)
        assert result[0].role == BlockRole.PARAGRAPH
        assert result[0].classifier_source == "llm"
        assert 0.0 <= result[0].confidence <= 1.0
        assert result[0].attributes == {}

    @pytest.mark.asyncio
    async def test_llm_classifier_strips_markdown_fences(self):
        """Models sometimes wrap JSON in ```json ... ```; we must tolerate it."""
        blocks = _make_blocks(1)
        inner = json.dumps(
            [{"block_id": blocks[0].block_id, "role": "paragraph"}]
        )
        reply = f"```json\n{inner}\n```"
        backend = MockBackend(responses=[reply])
        classifier = LLMClassifier(llm=backend)
        result = await classifier.classify(blocks)
        assert result[0].role == BlockRole.PARAGRAPH
        assert result[0].classifier_source == "llm"


# ---------------------------------------------------------------------------
# LLMClassifier — fallback behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestLLMClassifierFallback:
    @pytest.mark.asyncio
    async def test_llm_classifier_falls_back_on_invalid_json(self):
        """Unparseable response routes the full batch to the heuristic."""
        blocks = _make_blocks(2)
        # Text that would be tagged PARAGRAPH by the heuristic classifier.
        blocks[0].text = "This is ordinary prose for the classifier."
        blocks[1].text = "Chapter 1: Foundations"
        backend = MockBackend(responses=["not valid json at all"])
        classifier = LLMClassifier(llm=backend)
        result = await classifier.classify(blocks)

        assert len(result) == 2
        assert all(cb.classifier_source == "heuristic" for cb in result)
        # Heuristic still produces meaningful roles.
        assert result[1].role == BlockRole.CHAPTER_OPENER

    @pytest.mark.asyncio
    async def test_llm_classifier_falls_back_on_missing_block_ids(self):
        """Response omits some blocks => heuristic fills the missing ones."""
        blocks = _make_blocks(3)
        blocks[0].text = "Abstract"  # heuristic => ABSTRACT
        blocks[1].text = "Chapter 1: Bar"  # heuristic => CHAPTER_OPENER
        blocks[2].text = "Ordinary paragraph body."

        # LLM only classifies the first and third blocks; middle is missing.
        reply = json.dumps(
            [
                {
                    "block_id": blocks[0].block_id,
                    "role": "abstract",
                    "confidence": 0.95,
                },
                {
                    "block_id": blocks[2].block_id,
                    "role": "paragraph",
                    "confidence": 0.8,
                },
            ]
        )
        backend = MockBackend(responses=[reply])
        classifier = LLMClassifier(llm=backend)
        result = await classifier.classify(blocks)

        # Result order is position-stable with the input.
        assert [cb.raw.block_id for cb in result] == [
            b.block_id for b in blocks
        ]
        assert result[0].classifier_source == "llm"
        assert result[0].role == BlockRole.ABSTRACT

        # Middle block came from the heuristic fallback.
        assert result[1].classifier_source == "heuristic"
        assert result[1].role == BlockRole.CHAPTER_OPENER

        assert result[2].classifier_source == "llm"

    @pytest.mark.asyncio
    async def test_llm_classifier_falls_back_on_unknown_role_string(self):
        """An unknown role value falls back for that block only."""
        blocks = _make_blocks(2)
        blocks[0].text = "Abstract"
        blocks[1].text = "Chapter 1: Foo"

        reply = json.dumps(
            [
                {
                    "block_id": blocks[0].block_id,
                    "role": "not_a_real_role",
                },
                {
                    "block_id": blocks[1].block_id,
                    "role": "chapter_opener",
                },
            ]
        )
        backend = MockBackend(responses=[reply])
        classifier = LLMClassifier(llm=backend)
        result = await classifier.classify(blocks)

        assert result[0].classifier_source == "heuristic"
        assert result[0].role == BlockRole.ABSTRACT  # heuristic wins
        assert result[1].classifier_source == "llm"
        assert result[1].role == BlockRole.CHAPTER_OPENER

    @pytest.mark.asyncio
    async def test_llm_classifier_falls_back_when_backend_raises(self):
        """Any backend exception drops the batch to the heuristic without crashing."""

        class ExplodingBackend:
            def __init__(self):
                self.calls = 0

            async def complete(self, *args, **kwargs):
                self.calls += 1
                raise RuntimeError("simulated API outage")

            def complete_sync(self, *args, **kwargs):
                raise RuntimeError("simulated API outage")

        blocks = _make_blocks(2)
        blocks[0].text = "Abstract"
        blocks[1].text = "Paragraph body."
        backend = ExplodingBackend()
        classifier = LLMClassifier(llm=backend)
        result = await classifier.classify(blocks)
        assert len(result) == 2
        assert all(cb.classifier_source == "heuristic" for cb in result)
        assert backend.calls == 1


# ---------------------------------------------------------------------------
# LLMClassifier — prompt shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestLLMClassifierPrompt:
    @pytest.mark.asyncio
    async def test_llm_classifier_passes_neighbor_context(self):
        """Prompt includes neighbors.prev/next excerpts for disambiguation."""
        blocks = _make_blocks(2)
        blocks[0].neighbors = {"prev": "PREV_ALPHA_MARKER", "next": "NEXT_BETA_MARKER"}
        blocks[1].neighbors = {"prev": "PREV_GAMMA_MARKER", "next": "NEXT_DELTA_MARKER"}
        reply = _all_paragraph_response(blocks)
        backend = MockBackend(responses=[reply])
        classifier = LLMClassifier(llm=backend)
        await classifier.classify(blocks)

        assert len(backend.calls) == 1
        user_prompt = backend.calls[0].user
        assert "neighbors.prev:" in user_prompt
        assert "neighbors.next:" in user_prompt
        assert "PREV_ALPHA_MARKER" in user_prompt
        assert "NEXT_BETA_MARKER" in user_prompt
        assert "PREV_GAMMA_MARKER" in user_prompt
        assert "NEXT_DELTA_MARKER" in user_prompt

    @pytest.mark.asyncio
    async def test_llm_classifier_system_prompt_lists_all_roles(self):
        """Every BlockRole enum value must appear in the system message."""
        blocks = _make_blocks(1)
        backend = MockBackend(responses=[_all_paragraph_response(blocks)])
        classifier = LLMClassifier(llm=backend)
        await classifier.classify(blocks)

        system = backend.calls[0].system
        for role in BlockRole:
            assert role.value in system, f"missing role {role.value} in system prompt"

    @pytest.mark.asyncio
    async def test_llm_classifier_truncates_long_block_text(self):
        """Very long blocks are truncated in the prompt to keep batches small."""
        blocks = _make_blocks(1)
        blocks[0].text = "A" * 5000  # way over the 500-char cap
        backend = MockBackend(responses=[_all_paragraph_response(blocks)])
        classifier = LLMClassifier(llm=backend)
        await classifier.classify(blocks)
        user_prompt = backend.calls[0].user
        # 500-char truncation + ellipsis marker => never the full 5000.
        assert "A" * 5000 not in user_prompt
        assert "..." in user_prompt

    @pytest.mark.asyncio
    async def test_llm_classifier_uses_zero_temperature(self):
        """Classification pins temperature at 0.0 for deterministic tagging."""
        blocks = _make_blocks(1)
        backend = MockBackend(responses=[_all_paragraph_response(blocks)])
        classifier = LLMClassifier(llm=backend)
        await classifier.classify(blocks)
        assert backend.calls[0].temperature == 0.0


# ---------------------------------------------------------------------------
# default_classifier — flag routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.dart
class TestDefaultClassifierRouting:
    def test_default_classifier_uses_heuristic_when_flag_off(self, monkeypatch):
        monkeypatch.setenv("DART_LLM_CLASSIFICATION", "false")
        backend = MockBackend(responses=["[]"])
        classifier = default_classifier(llm=backend)
        assert isinstance(classifier, HeuristicClassifier)

    def test_default_classifier_uses_heuristic_when_flag_unset(self, monkeypatch):
        monkeypatch.delenv("DART_LLM_CLASSIFICATION", raising=False)
        backend = MockBackend(responses=["[]"])
        classifier = default_classifier(llm=backend)
        assert isinstance(classifier, HeuristicClassifier)

    def test_default_classifier_uses_llm_when_flag_on_and_backend_provided(
        self, monkeypatch
    ):
        monkeypatch.setenv("DART_LLM_CLASSIFICATION", "true")
        backend = MockBackend(responses=["[]"])
        classifier = default_classifier(llm=backend)
        assert isinstance(classifier, LLMClassifier)

    def test_default_classifier_uses_heuristic_when_flag_on_but_no_backend(
        self, monkeypatch
    ):
        """Flag on + no backend must degrade gracefully to heuristic."""
        monkeypatch.setenv("DART_LLM_CLASSIFICATION", "true")
        classifier = default_classifier(llm=None)
        assert isinstance(classifier, HeuristicClassifier)

    def test_default_classifier_flag_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DART_LLM_CLASSIFICATION", "TRUE")
        backend = MockBackend(responses=["[]"])
        classifier = default_classifier(llm=backend)
        assert isinstance(classifier, LLMClassifier)
