"""Tests for ``DomainConceptVocabularyValidator`` (Stage-3 vocabulary).

Three-stage large-LLM textbook synthesis architecture, Wave A — see
``plans/textbook-llm-synthesis-3stage-2026-05.md`` §8 row 2.

Cases:

1. ``test_skip_with_pass_when_vocabulary_absent`` — no vocabulary
   artifact supplied → no-op ``passed=True``.
2. ``test_vocab_thin`` — fewer than ``_MIN_CONCEPTS`` (8) concepts →
   ``VOCAB_THIN``.
3. ``test_vocab_concept_no_canonical`` — a concept missing `canonical`
   → ``VOCAB_CONCEPT_NO_CANONICAL``.
4. ``test_vocab_count_mismatch`` — ``concept_count`` != len(concepts)
   → ``VOCAB_COUNT_MISMATCH``.
5. ``test_vocab_partial_coverage`` — non-empty
   ``chapter_synthesis_failures`` → ``VOCAB_PARTIAL_COVERAGE``.
6. ``test_happy_path_passes`` — a well-formed vocabulary passes clean.
7. ``test_decision_capture_emitted`` — one ``content_structure_check``
   event per run.
8. ``test_path_loaded_from_disk`` — ``domain_concept_vocabulary_path``
   is loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.domain_concept_vocabulary import (  # noqa: E402
    DomainConceptVocabularyValidator,
)


class _StubDecisionCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def _concepts(n: int) -> List[Dict[str, Any]]:
    return [
        {"canonical": f"concept-{i}", "aliases": [f"alias-{i}"]}
        for i in range(n)
    ]


def _base_vocabulary(n: int = 10) -> Dict[str, Any]:
    concepts = _concepts(n)
    return {
        "schema_version": "v1",
        "course_id": "ALG_101",
        "course_slug": "alg-101",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "chapter_call_count": 5,
        "concept_count": len(concepts),
        "concepts": concepts,
    }


def _codes(result: Any) -> List[str]:
    return [i.code for i in result.issues]


def test_skip_with_pass_when_vocabulary_absent() -> None:
    result = DomainConceptVocabularyValidator().validate({})
    assert result.passed is True
    assert result.issues == []


def test_vocab_thin() -> None:
    vocab = _base_vocabulary(n=3)
    result = DomainConceptVocabularyValidator().validate(
        {"domain_concept_vocabulary": vocab}
    )
    assert "VOCAB_THIN" in _codes(result)
    assert result.passed is True  # warning severity


def test_vocab_concept_no_canonical() -> None:
    vocab = _base_vocabulary(n=10)
    vocab["concepts"][2] = {"aliases": ["orphan"]}  # no canonical
    result = DomainConceptVocabularyValidator().validate(
        {"domain_concept_vocabulary": vocab}
    )
    assert "VOCAB_CONCEPT_NO_CANONICAL" in _codes(result)


def test_vocab_count_mismatch() -> None:
    vocab = _base_vocabulary(n=10)
    vocab["concept_count"] = 99  # out of sync with len(concepts)==10
    result = DomainConceptVocabularyValidator().validate(
        {"domain_concept_vocabulary": vocab}
    )
    assert "VOCAB_COUNT_MISMATCH" in _codes(result)


def test_vocab_partial_coverage() -> None:
    vocab = _base_vocabulary(n=10)
    vocab["chapter_synthesis_failures"] = ["ch3", "ch7"]
    result = DomainConceptVocabularyValidator().validate(
        {"domain_concept_vocabulary": vocab}
    )
    assert "VOCAB_PARTIAL_COVERAGE" in _codes(result)
    # Partial coverage is informational — does not fail the gate.
    assert result.passed is True


def test_happy_path_passes() -> None:
    result = DomainConceptVocabularyValidator().validate(
        {"domain_concept_vocabulary": _base_vocabulary(n=12)}
    )
    assert result.passed is True
    assert result.issues == []
    assert result.score == 1.0


def test_decision_capture_emitted() -> None:
    capture = _StubDecisionCapture()
    DomainConceptVocabularyValidator().validate(
        {
            "domain_concept_vocabulary": _base_vocabulary(n=10),
            "decision_capture": capture,
        }
    )
    assert len(capture.events) == 1
    assert capture.events[0]["decision_type"] == "content_structure_check"
    assert len(capture.events[0]["rationale"]) >= 20


def test_decision_capture_emitted_on_skip() -> None:
    capture = _StubDecisionCapture()
    DomainConceptVocabularyValidator().validate(
        {"decision_capture": capture}
    )
    assert len(capture.events) == 1
    assert capture.events[0]["decision_type"] == "content_structure_check"


def test_path_loaded_from_disk(tmp_path: Path) -> None:
    p = tmp_path / "domain_concept_vocabulary.json"
    p.write_text(json.dumps(_base_vocabulary(n=10)), encoding="utf-8")
    result = DomainConceptVocabularyValidator().validate(
        {"domain_concept_vocabulary_path": str(p)}
    )
    assert result.passed is True
    assert result.issues == []


def test_missing_path_fails_closed(tmp_path: Path) -> None:
    result = DomainConceptVocabularyValidator().validate(
        {"domain_concept_vocabulary_path": str(tmp_path / "nope.json")}
    )
    assert result.passed is False
    assert "VOCAB_PATH_MISSING" in _codes(result)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
