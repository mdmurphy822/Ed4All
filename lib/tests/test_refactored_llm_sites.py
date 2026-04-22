"""Tests that the three refactored LLM call sites route through LLMBackend.

Covers ClaudeProcessor, AltTextGenerator, and classify_teaching_roles —
each should accept an injected backend and avoid the direct
``anthropic.Anthropic()`` path.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from MCP.orchestrator.llm_backend import MockBackend


# ============================================================================
# ClaudeProcessor (DART/pdf_converter/claude_processor.py)
# ============================================================================


class TestClaudeProcessorWithBackend:
    def test_processor_accepts_llm_kwarg(self):
        from DART.pdf_converter.claude_processor import ClaudeProcessor

        backend = MockBackend(responses=["{}"])
        processor = ClaudeProcessor(llm=backend, enable_cache=False)
        assert processor._llm is backend

    def test_processor_with_backend_no_api_key_works(self, monkeypatch):
        """With an injected backend, no API key should be required."""
        from DART.pdf_converter.claude_processor import ClaudeProcessor

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        canned = json.dumps(
            {
                "title": "Test",
                "authors": ["A"],
                "abstract": None,
                "blocks": [
                    {"block_type": "paragraph", "content": "Hello world."},
                ],
                "metadata": {},
            }
        )
        backend = MockBackend(responses=[canned])
        processor = ClaudeProcessor(llm=backend, enable_cache=False)

        doc = processor.process_text("some raw text")
        assert doc.title == "Test"
        assert len(doc.blocks) == 1
        # Verify the backend was actually called
        assert len(backend.calls) == 1
        assert backend.calls[0].system.startswith(
            "You are a document structure analyzer"
        )

    def test_processor_existing_api_key_path_unchanged(self, monkeypatch):
        """Callers passing api_key (old path) still work — no regression."""
        from DART.pdf_converter.claude_processor import ClaudeProcessor

        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
        processor = ClaudeProcessor(enable_cache=False)
        assert processor._llm is None
        assert processor.api_key == "fake-key-for-test"

    def test_processor_no_key_no_backend_still_errors(self, monkeypatch):
        from DART.pdf_converter.claude_processor import (
            ClaudeProcessingError,
            ClaudeProcessor,
        )

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        processor = ClaudeProcessor(enable_cache=False)
        with pytest.raises(ClaudeProcessingError, match="API key"):
            processor.process_text("something")


# ============================================================================
# AltTextGenerator (DART/pdf_converter/alt_text_generator.py)
# ============================================================================


class _FakeImage:
    """Minimal ExtractedImage stand-in for tests."""

    def __init__(self):
        self.data = b"fake"
        self.data_uri = None
        self.format = "png"
        self.page = 1
        self.width = 100
        self.height = 100
        self.nearby_caption = None


class TestAltTextGeneratorWithBackend:
    def test_generator_accepts_llm_kwarg(self):
        from DART.pdf_converter.alt_text_generator import AltTextGenerator

        backend = MockBackend(responses=["ALT: fake\nLONG: a long description"])
        gen = AltTextGenerator(llm=backend, use_ocr_fallback=False)
        assert gen._llm is backend
        assert gen.use_ai is True

    def test_generate_with_backend_returns_alt_text(self, monkeypatch):
        from DART.pdf_converter.alt_text_generator import AltTextGenerator

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        backend = MockBackend(
            responses=["ALT: a histogram of scores\nLONG: bars represent counts"]
        )
        gen = AltTextGenerator(llm=backend, use_ocr_fallback=False)
        result = gen.generate(_FakeImage())
        assert result.success is True
        assert result.source == "claude"
        assert "histogram" in result.alt_text

        # Verify backend got image payload
        assert len(backend.calls) == 1
        call = backend.calls[0]
        assert call.images is not None
        assert call.images[0]["media_type"] == "image/png"

    def test_no_backend_no_key_still_falls_back_to_generic(self, monkeypatch):
        """With neither backend nor API key, use_ai=False and falls through."""
        from DART.pdf_converter.alt_text_generator import AltTextGenerator

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        gen = AltTextGenerator(use_ocr_fallback=False)
        assert gen.use_ai is False
        result = gen.generate(_FakeImage())
        # No AI, no OCR, no caption => generic fallback succeeds
        assert result.source == "generic"


# ============================================================================
# classify_teaching_roles (Trainforge/align_chunks.py)
# ============================================================================


class TestAlignChunksWithBackend:
    def _chunks(self):
        """Produce chunks where the heuristic can't classify — forces LLM path."""
        return [
            {
                "id": "c1",
                "_position": 0,
                "chunk_type": "mixed",
                "source": {"resource_type": "html", "lesson_title": "Week 1"},
                "concept_tags": ["ambiguous"],
                "prereq_concepts": [],
                "text": "some mixed educational content",
            },
            {
                "id": "c2",
                "_position": 1,
                "chunk_type": "mixed",
                "source": {"resource_type": "html", "lesson_title": "Week 2"},
                "concept_tags": ["ambiguous"],
                "prereq_concepts": [],
                "text": "more ambiguous content",
            },
        ]

    def test_classify_accepts_llm_backend(self):
        from Trainforge.align_chunks import classify_teaching_roles

        response = json.dumps(
            [{"id": "c1", "role": "introduce"}, {"id": "c2", "role": "elaborate"}]
        )
        backend = MockBackend(responses=[response])
        chunks = self._chunks()
        classify_teaching_roles(chunks, llm=backend, verbose=False)
        # Every chunk should get a teaching_role (either from LLM or fallback)
        assert all("teaching_role" in c for c in chunks)
        # At least one should be an LLM-assigned role
        assert len(backend.calls) >= 1

    def test_classify_fallback_when_backend_fails(self):
        from Trainforge.align_chunks import classify_teaching_roles

        def crashy(system, user):
            raise RuntimeError("network dead")

        backend = MockBackend(response_fn=crashy)
        chunks = self._chunks()
        classify_teaching_roles(chunks, llm=backend)
        # Even though the LLM failed, every chunk must have a teaching_role
        # (mock fallback handles it)
        assert all("teaching_role" in c for c in chunks)

    def test_classify_mock_provider_unchanged(self):
        """llm_provider='mock' without an injected backend uses heuristic only."""
        from Trainforge.align_chunks import classify_teaching_roles

        chunks = self._chunks()
        # No backend, mock provider
        classify_teaching_roles(chunks, llm_provider="mock")
        assert all("teaching_role" in c for c in chunks)
