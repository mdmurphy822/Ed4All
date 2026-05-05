"""H4 regression test — typed exhaustion exception parity.

``Trainforge/generators/_anthropic_provider.py`` previously raised
plain ``RuntimeError`` after ``MAX_PARSE_RETRIES`` malformed responses,
forcing the caller's recovery path
(``run_synthesis``'s paraphrase-exhaustion soft-skip) to dispatch on
provider type. ``LocalSynthesisProvider`` already raises typed
``SynthesisProviderError(code="paraphrase_invalid_after_retry")`` on the
same exhaustion condition; this test pins the Anthropic provider to the
same typed contract so caller-side recovery is provider-agnostic.

The pre-existing ``test_anthropic_synthesis_provider.py::
test_malformed_json_exhausts_retries_then_raises`` asserts that
``RuntimeError`` is raised, which keeps passing because
``SynthesisProviderError`` is a ``RuntimeError`` subclass. This file
adds the type + code + cause-chain assertions that pin the new contract.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._anthropic_provider import (  # noqa: E402
    MAX_PARSE_RETRIES,
    AnthropicSynthesisProvider,
    SynthesisProviderError,
)


# ---------------------------------------------------------------------- #
# Helpers — duplicated from the sibling test file to keep this regression
# self-contained (so a future edit to the shared helpers can't silently
# break the contract this test pins).
# ---------------------------------------------------------------------- #


def _mock_response(text: str) -> Any:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 100
    response.usage = usage
    return response


def _client_returning(*texts: str) -> Any:
    client = MagicMock()
    client.messages.create.side_effect = [_mock_response(t) for t in texts]
    return client


def _instruction_draft() -> dict:
    return {
        "prompt": "Define the central concept behind topic X.",
        "completion": (
            "Topic X is a central idea in this material. Learners should "
            "be able to recall and restate it."
        ),
        "chunk_id": "chunk_001",
        "lo_refs": ["TO-01"],
        "bloom_level": "remember",
        "content_type": "explanation",
        "seed": 17,
        "decision_capture_id": "",
        "template_id": "remember.explanation",
        "provider": "mock",
        "schema_version": "v1",
    }


def _chunk() -> dict:
    return {
        "id": "chunk_001",
        "text": (
            "Topic X is the foundational concept in chapter one. Learners "
            "encounter it in every subsequent chapter."
        ),
        "learning_outcome_refs": ["TO-01"],
    }


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #


def test_exhaustion_raises_typed_synthesis_provider_error():
    """After MAX_PARSE_RETRIES malformed responses, raise the typed
    ``SynthesisProviderError`` — not a bare ``RuntimeError`` — so the
    caller can dispatch on ``code`` without branching by provider."""
    bad_responses = ["this is not JSON"] * MAX_PARSE_RETRIES
    client = _client_returning(*bad_responses)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)

    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(_instruction_draft(), _chunk())

    err = excinfo.value
    # Code matches the LocalSynthesisProvider exhaustion code exactly so
    # the caller's recovery branch doesn't have to dispatch on provider.
    assert err.code == "paraphrase_invalid_after_retry"
    # And the SDK was called exactly MAX_PARSE_RETRIES times.
    assert client.messages.create.call_count == MAX_PARSE_RETRIES


def test_exhaustion_preserves_cause_chain():
    """The original parse-failure exception is preserved as ``__cause__``
    so postmortem audits can inspect the underlying parse trace."""
    bad_responses = ["not JSON"] * MAX_PARSE_RETRIES
    client = _client_returning(*bad_responses)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)

    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(_instruction_draft(), _chunk())

    cause = excinfo.value.__cause__
    # The last-attempt parse-failure exception is preserved.
    assert cause is not None
    assert isinstance(cause, ValueError)


def test_exhaustion_is_runtime_error_subclass_for_back_compat():
    """``SynthesisProviderError`` MUST stay a ``RuntimeError`` subclass so
    legacy callers that ``except RuntimeError`` keep working without
    re-pointing at the new type."""
    assert issubclass(SynthesisProviderError, RuntimeError)
    bad_responses = ["nope"] * MAX_PARSE_RETRIES
    client = _client_returning(*bad_responses)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)

    # Catch as RuntimeError — the back-compat surface.
    with pytest.raises(RuntimeError):
        p.paraphrase_instruction(_instruction_draft(), _chunk())


def test_exhaustion_message_carries_diagnostic_tail():
    """The error message MUST surface the last response tail and the
    retry budget so an operator can diagnose which response shape the
    model kept failing on."""
    bad_responses = ["totally bogus"] * MAX_PARSE_RETRIES
    client = _client_returning(*bad_responses)
    p = AnthropicSynthesisProvider(api_key="sk-test", client=client)

    with pytest.raises(SynthesisProviderError) as excinfo:
        p.paraphrase_instruction(_instruction_draft(), _chunk())

    msg = str(excinfo.value)
    assert str(MAX_PARSE_RETRIES) in msg
    assert "totally bogus" in msg
