"""Tests that the refactored LLM call sites route through LLMBackend.

Covers ``classify_teaching_roles`` (Trainforge/alignment/align_chunks.py) — it
should accept an injected backend and avoid the direct
``anthropic.Anthropic()`` path.

SemantiK's PDF-to-accessible-HTML conversion has its own LLM-backend
tests under ``lib/semantik/tests``; this file covers only
``classify_teaching_roles``.
"""
from __future__ import annotations

import json

from MCP.orchestrator.llm_backend import MockBackend

# ============================================================================
# classify_teaching_roles (Trainforge/alignment/align_chunks.py)
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
        from Trainforge.alignment.align_chunks import classify_teaching_roles

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
        from Trainforge.alignment.align_chunks import classify_teaching_roles

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
        from Trainforge.alignment.align_chunks import classify_teaching_roles

        chunks = self._chunks()
        # No backend, mock provider
        classify_teaching_roles(chunks, llm_provider="mock")
        assert all("teaching_role" in c for c in chunks)
