"""Wave 22 DC1 — LLMClassifier emits one decision per batch.

Pre-Wave-22, every DART Claude call site was uninstrumented. A Bates
run with dozens of Claude decisions produced two static boilerplate
capture records from the MCP wrapper. Wave 22 DC1 threads an optional
``capture`` kwarg through ``LLMClassifier``; when supplied, every
batch emits one ``structure_detection`` decision with dynamic
rationale (block-ID range, fallback fraction, average confidence,
token payload, model).

These tests use :class:`MockBackend` + a mock capture so no real API
traffic and no real disk I/O happens.
"""
from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock

import pytest

from DART.converter import BlockRole, LLMClassifier, RawBlock
from MCP.orchestrator.llm_backend import MockBackend


def _make_blocks(n: int) -> List[RawBlock]:
    blocks: List[RawBlock] = []
    for i in range(n):
        blocks.append(
            RawBlock(
                text=f"Sample block content {i}. Lorem ipsum dolor.",
                block_id=f"blk{i:06d}",
                page=1,
                extractor="pdftotext",
                neighbors={"prev": "", "next": ""},
            )
        )
    return blocks


def _all_paragraph_response(blocks: List[RawBlock]) -> str:
    payload = [
        {
            "block_id": b.block_id,
            "role": BlockRole.PARAGRAPH.value,
            "confidence": 0.88,
            "attributes": {},
        }
        for b in blocks
    ]
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_llm_classifier_emits_capture_per_batch():
    """Each LLM batch must trigger one structure_detection decision."""
    blocks = _make_blocks(20)
    backend = MockBackend(responses=[_all_paragraph_response(blocks)])
    capture = MagicMock()

    classifier = LLMClassifier(llm=backend, capture=capture)
    await classifier.classify(blocks)

    # One batch of 20 blocks → one capture emit.
    assert capture.log_decision.call_count == 1, (
        f"Expected exactly 1 capture emit for a 20-block batch, "
        f"got {capture.log_decision.call_count}"
    )
    call = capture.log_decision.call_args
    assert call.kwargs["decision_type"] == "structure_detection"
    rationale = call.kwargs["rationale"]
    # Dynamic rationale signals (DC1 audit).
    assert len(rationale) >= 20
    assert "Block range" in rationale
    assert "LLM=" in rationale
    assert "heuristic_fallback=" in rationale
    assert "avg confidence" in rationale
    assert "char prompt payload" in rationale


@pytest.mark.asyncio
async def test_llm_classifier_emits_capture_per_batch_multi_batch():
    """A 45-block input with batch_size=20 must fire 3 capture emits."""
    blocks = _make_blocks(45)
    responses = [
        _all_paragraph_response(blocks[0:20]),
        _all_paragraph_response(blocks[20:40]),
        _all_paragraph_response(blocks[40:45]),
    ]
    backend = MockBackend(responses=responses)
    capture = MagicMock()

    classifier = LLMClassifier(llm=backend, batch_size=20, capture=capture)
    await classifier.classify(blocks)

    assert capture.log_decision.call_count == 3, (
        f"45 blocks at batch_size=20 should fire 3 capture emits, "
        f"got {capture.log_decision.call_count}"
    )


@pytest.mark.asyncio
async def test_llm_classifier_fires_capture_on_backend_failure():
    """Backend failure → heuristic fallback → still one capture emit."""
    blocks = _make_blocks(5)

    class FailingBackend:
        async def complete(self, **kwargs):
            raise RuntimeError("simulated backend outage")

    capture = MagicMock()
    classifier = LLMClassifier(llm=FailingBackend(), capture=capture)
    await classifier.classify(blocks)

    assert capture.log_decision.call_count == 1, (
        "Even on backend failure, the fallback path must emit one "
        "decision so captures stay exhaustive."
    )
    rationale = capture.log_decision.call_args.kwargs["rationale"]
    # 100% fallback fraction is the key signal a failure happened.
    assert "100% fallback" in rationale or "100.0% fallback" in rationale.replace("0%", "100%"), (
        f"Fallback fraction should read 100% on a total backend "
        f"failure; got rationale={rationale!r}"
    )


@pytest.mark.asyncio
async def test_llm_classifier_silently_skips_when_no_capture():
    """Without an injected capture, the classifier must not crash or log."""
    blocks = _make_blocks(3)
    backend = MockBackend(responses=[_all_paragraph_response(blocks)])

    # No capture kwarg — existing behaviour must be preserved.
    classifier = LLMClassifier(llm=backend)
    classified = await classifier.classify(blocks)

    assert len(classified) == 3
    # Smoke check: construction succeeded, classify succeeded, no capture
    # side effect.


def test_default_classifier_forwards_capture():
    """``default_classifier`` must forward ``capture`` into LLMClassifier."""
    import os

    os.environ["DART_LLM_CLASSIFICATION"] = "true"
    try:
        from DART.converter import default_classifier

        backend = MockBackend(responses=["[]"])
        capture = MagicMock()
        classifier = default_classifier(llm=backend, capture=capture)
        assert isinstance(classifier, LLMClassifier)
        assert classifier.capture is capture, (
            "default_classifier should forward `capture` into the "
            "LLMClassifier constructor."
        )
    finally:
        os.environ.pop("DART_LLM_CLASSIFICATION", None)
