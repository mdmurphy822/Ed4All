"""Regression: teaching_role_source labels failed-LLM chunks as llm_failed.

Pre-fix bug: ``classify_teaching_roles`` blanket-set
``teaching_role_source = "llm"`` on every ambiguous chunk after the
LLM helper returned, even though
``_classify_with_curriculum_provider`` (and the legacy
``_classify_with_llm``) silently fell back to ``_mock_role`` on any
exception. Result: chunks whose LLM call raised got heuristic roles
wearing an "llm" badge, with no corresponding decision-capture event
— a silent-fallback-masquerading-as-LLM provenance hole.

These tests pin both helpers to the post-fix contract: success path
labels "llm"; exception path labels "llm_failed" and attaches a
``teaching_role_failure`` dict with ``error_class`` +
``error_message`` so audits can find these without grepping
decision captures. The label deliberately avoids "mock_fallback" —
no mock provider exists on this path; the deterministic heuristic
just supplies a placeholder role so downstream consumers don't
break. The caller no longer touches ``teaching_role_source``
(helpers own the field).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge import align_chunks  # noqa: E402


def _chunk(idx: int, text: str = "x" * 50) -> dict:
    return {
        "id": f"c{idx:05d}",
        "text": text,
        "_position": idx,
        "concept_tags": [],
        "prereq_concepts": [],
        "chunk_type": "content",
        "source": {"resource_type": "html"},
    }


# ---------------------------------------------------------------------------
# _classify_with_curriculum_provider
# ---------------------------------------------------------------------------


def test_curriculum_helper_labels_success_as_llm():
    chunks = [_chunk(1), _chunk(2)]
    provider = MagicMock()
    provider.classify_teaching_role.side_effect = ["introduce", "elaborate"]

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider, verbose=False,
    )

    assert chunks[0]["teaching_role"] == "introduce"
    assert chunks[0]["teaching_role_source"] == "llm"
    assert chunks[1]["teaching_role"] == "elaborate"
    assert chunks[1]["teaching_role_source"] == "llm"


def test_curriculum_helper_labels_exception_as_llm_failed_with_metadata():
    """The provenance bug regression: failed LLM calls must NOT be
    labeled as 'llm'. They get the deterministic-heuristic role and
    a 'llm_failed' source plus a teaching_role_failure dict so audits
    can find them without grepping decision captures."""
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    provider = MagicMock()
    provider.classify_teaching_role.side_effect = [
        "introduce",
        RuntimeError("simulated transport failure"),
        "reinforce",
    ]

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider, verbose=False,
    )

    assert chunks[0]["teaching_role_source"] == "llm"
    assert "teaching_role_failure" not in chunks[0]
    assert chunks[1]["teaching_role_source"] == "llm_failed"
    assert chunks[1]["teaching_role"] in align_chunks.VALID_ROLES
    assert chunks[1]["teaching_role_failure"] == {
        "error_class": "RuntimeError",
        "error_message": "simulated transport failure",
    }
    assert chunks[2]["teaching_role_source"] == "llm"
    assert "teaching_role_failure" not in chunks[2]


def test_curriculum_helper_handles_invalid_role_response():
    """``CurriculumAlignmentProvider`` raises
    ``SynthesisProviderError(code='invalid_role_response')`` on bad
    output. The helper's blanket ``except Exception`` catches it and
    must label the chunk mock_fallback, not llm."""
    from Trainforge.generators._curriculum_provider import (
        SynthesisProviderError,
    )

    chunks = [_chunk(1)]
    provider = MagicMock()
    provider.classify_teaching_role.side_effect = SynthesisProviderError(
        "bad role", code="invalid_role_response", chunk_id="c00001",
    )

    align_chunks._classify_with_curriculum_provider(
        chunks, concept_first_seen={}, provider=provider, verbose=False,
    )

    assert chunks[0]["teaching_role_source"] == "llm_failed"


# ---------------------------------------------------------------------------
# _classify_with_llm (legacy batch path)
# ---------------------------------------------------------------------------


def test_legacy_llm_helper_labels_success_as_llm():
    chunks = [_chunk(1), _chunk(2)]
    llm = MagicMock()
    llm.complete_sync.return_value = (
        '[{"id": "c00001", "role": "introduce"}, '
        '{"id": "c00002", "role": "elaborate"}]'
    )

    align_chunks._classify_with_llm(
        chunks, concept_first_seen={}, model="claude-haiku-4-5-20251001",
        verbose=False, llm=llm,
    )

    assert chunks[0]["teaching_role_source"] == "llm"
    assert chunks[1]["teaching_role_source"] == "llm"


def test_legacy_llm_helper_labels_exception_as_llm_failed():
    chunks = [_chunk(1), _chunk(2)]
    llm = MagicMock()
    llm.complete_sync.side_effect = RuntimeError("simulated batch failure")

    align_chunks._classify_with_llm(
        chunks, concept_first_seen={}, model="claude-haiku-4-5-20251001",
        verbose=False, llm=llm,
    )

    assert chunks[0]["teaching_role_source"] == "llm_failed"
    assert chunks[1]["teaching_role_source"] == "llm_failed"


def test_legacy_llm_helper_labels_no_json_match_as_llm_failed():
    """When the model returns text without a JSON array, the helper's
    ``else: # Fallback`` branch fires. That's still a failed-LLM
    outcome, so the source must be mock_fallback, not llm."""
    chunks = [_chunk(1)]
    llm = MagicMock()
    llm.complete_sync.return_value = "I cannot do that, Dave."

    align_chunks._classify_with_llm(
        chunks, concept_first_seen={}, model="claude-haiku-4-5-20251001",
        verbose=False, llm=llm,
    )

    assert chunks[0]["teaching_role_source"] == "llm_failed"
