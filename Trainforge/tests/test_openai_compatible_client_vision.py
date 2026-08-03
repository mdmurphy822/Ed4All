"""Wave W-D13 T13.1 — vision content-block regression tests for the
LLM-agnostic ``OpenAICompatibleClient``.

The client is provider-agnostic: the same ``chat_completion(images=...)``
shape must work for OpenAI's vision API, Together AI's Llama-3.2-Vision
endpoints, and Ollama's vision-capable models (llava, llama3.2-vision,
qwen2.5-vl). These tests assert the wire-payload shape against a mocked
``httpx`` transport — no real HTTP, no real model needed.

Coverage:

- Vision-disabled callers (``images=None`` / empty list) keep the legacy
  ``content: <str>`` shape — non-regression for every existing call site.
- ``images=`` rewrites the LAST user message to OpenAI's content-block
  list shape: ``{"type": "text", "text": ...}`` followed by one
  ``{"type": "image_url", "image_url": {"url": "data:<mt>;base64,<d>"}}``
  per image.
- Multiple images attach in declaration order.
- Non-user messages (system) pass through verbatim — only the user-side
  content-block translation fires.
- Pre-existing content-block lists in the user message merge with the
  appended vision blocks (caller's contract honoured).
- ``vision_capable`` flag is stored + readable but does NOT enforce at
  the client layer — that's the backend's job.
- Malformed image entries (missing ``media_type`` or ``data``) raise
  ``ValueError`` BEFORE the wire round-trip.
- Vision without a user message in the chain raises ``ValueError``.

The ``_extract_text`` / retry / decision-capture wiring is exercised by
the broader suite at ``Trainforge/tests/test_openai_compatible_client.py``;
this file focuses solely on the vision payload translation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.providers._openai_compatible_client import (  # noqa: E402
    OpenAICompatibleClient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str = "ok") -> dict:
    return {
        "id": "cmpl-vision",
        "model": "vision-test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
        },
    }


def _build(
    transport_handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = "test-key",
    base_url: str = "https://api.example.com/v1",
    model: str = "vision-test-model",
    provider_label: str = "test_vision_provider",
    vision_capable: bool = True,
) -> OpenAICompatibleClient:
    transport = httpx.MockTransport(transport_handler)
    return OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        provider_label=provider_label,
        client=httpx.Client(transport=transport),
        sleep_fn=lambda _s: None,
        vision_capable=vision_capable,
    )


# ---------------------------------------------------------------------------
# vision_capable flag
# ---------------------------------------------------------------------------


def test_vision_capable_defaults_to_false():
    """Default constructor argument keeps the flag off — back-compat."""

    client = OpenAICompatibleClient(base_url="http://x/v1", model="m")
    assert client.vision_capable is False


def test_vision_capable_flag_round_trips_when_set():
    client = OpenAICompatibleClient(
        base_url="http://x/v1", model="m", vision_capable=True
    )
    assert client.vision_capable is True


# ---------------------------------------------------------------------------
# images=None / empty preserves the legacy shape
# ---------------------------------------------------------------------------


def test_no_images_keeps_legacy_string_content_shape():
    """``images=None`` produces a wire payload with content as a string."""

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("hello"))

    client = _build(handler)
    out = client.chat_completion(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        max_tokens=32,
        temperature=0.0,
    )
    assert out == "hello"

    body = _read_request_json(seen[0])
    msgs = body["messages"]
    # Both messages keep their ``content: <str>`` shape — no rewrite.
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_empty_images_list_keeps_legacy_string_content_shape():
    """``images=[]`` is treated identically to ``images=None``."""

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("hello"))

    client = _build(handler)
    client.chat_completion(
        [{"role": "user", "content": "hi"}],
        images=[],
    )
    body = _read_request_json(seen[0])
    assert body["messages"] == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# images= rewrites the LAST user message
# ---------------------------------------------------------------------------


def test_single_image_rewrites_user_message_to_content_blocks():
    """One image → text block + one image_url block in OpenAI vision shape."""

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("described"))

    client = _build(handler)
    out = client.chat_completion(
        [{"role": "user", "content": "describe this"}],
        images=[{"media_type": "image/jpeg", "data": "AAA="}],
    )
    assert out == "described"

    body = _read_request_json(seen[0])
    msgs = body["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AAA="},
    }


def test_multiple_images_attach_in_declaration_order():
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(handler)
    client.chat_completion(
        [{"role": "user", "content": "compare these"}],
        images=[
            {"media_type": "image/jpeg", "data": "AAA="},
            {"media_type": "image/png", "data": "BBB="},
            {"media_type": "image/jpeg", "data": "CCC="},
        ],
    )

    body = _read_request_json(seen[0])
    content = body["messages"][0]["content"]
    # 1 text + 3 images.
    assert len(content) == 4
    assert content[0] == {"type": "text", "text": "compare these"}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,AAA="
    assert content[2]["image_url"]["url"] == "data:image/png;base64,BBB="
    assert content[3]["image_url"]["url"] == "data:image/jpeg;base64,CCC="


def test_system_message_passes_through_verbatim_with_vision():
    """Only the LAST user message gets rewritten; system stays string."""

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(handler)
    client.chat_completion(
        [
            {"role": "system", "content": "be a vision expert"},
            {"role": "user", "content": "describe"},
        ],
        images=[{"media_type": "image/png", "data": "DDD="}],
    )

    body = _read_request_json(seen[0])
    msgs = body["messages"]
    # System retains the legacy string-content shape.
    assert msgs[0] == {"role": "system", "content": "be a vision expert"}
    # User gets rewritten.
    assert isinstance(msgs[1]["content"], list)
    assert msgs[1]["content"][0]["text"] == "describe"


def test_vision_attaches_to_last_user_message_not_first():
    """Multiple user messages → only the LAST one carries the image blocks."""

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(handler)
    client.chat_completion(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "second — describe this"},
        ],
        images=[{"media_type": "image/jpeg", "data": "EEE="}],
    )

    body = _read_request_json(seen[0])
    msgs = body["messages"]
    # First user is left as a string.
    assert msgs[0] == {"role": "user", "content": "first"}
    # Assistant message untouched.
    assert msgs[1] == {"role": "assistant", "content": "ack"}
    # LAST user is rewritten with the image attached.
    assert isinstance(msgs[2]["content"], list)
    assert msgs[2]["content"][0]["text"] == "second — describe this"


def test_pre_existing_content_block_list_appends_image_blocks():
    """Caller pre-built a content-block list → images append to it."""

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body("ok"))

    client = _build(handler)
    client.chat_completion(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "text", "text": "and this"},
                ],
            }
        ],
        images=[{"media_type": "image/jpeg", "data": "FFF="}],
    )

    body = _read_request_json(seen[0])
    content = body["messages"][0]["content"]
    # Original 2 text blocks + 1 image block appended.
    assert len(content) == 3
    assert content[0] == {"type": "text", "text": "look at this"}
    assert content[1] == {"type": "text", "text": "and this"}
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,FFF="


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_image_missing_media_type_raises_value_error():
    client = _build(lambda r: httpx.Response(200, json=_success_body("ok")))
    with pytest.raises(ValueError, match="media_type"):
        client.chat_completion(
            [{"role": "user", "content": "describe"}],
            images=[{"data": "AAA="}],
        )


def test_image_missing_data_raises_value_error():
    client = _build(lambda r: httpx.Response(200, json=_success_body("ok")))
    with pytest.raises(ValueError, match="data"):
        client.chat_completion(
            [{"role": "user", "content": "describe"}],
            images=[{"media_type": "image/png"}],
        )


def test_vision_without_user_message_raises_value_error():
    client = _build(lambda r: httpx.Response(200, json=_success_body("ok")))
    with pytest.raises(ValueError, match="user message"):
        client.chat_completion(
            [{"role": "system", "content": "system only"}],
            images=[{"media_type": "image/png", "data": "AAA="}],
        )


def test_caller_messages_list_not_mutated_in_place():
    """Vision rewrite builds a fresh messages list; caller's keeps shape."""

    client = _build(lambda r: httpx.Response(200, json=_success_body("ok")))
    original = [{"role": "user", "content": "describe"}]
    client.chat_completion(
        original,
        images=[{"media_type": "image/png", "data": "AAA="}],
    )
    # Caller's reference still points at the original string-content
    # shape — we never mutate the input list in place.
    assert original[0]["content"] == "describe"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_request_json(request: httpx.Request) -> Dict[str, Any]:
    """``httpx.Request.read()`` returns bytes; parse to dict for assertions."""

    import json

    body_bytes = request.read()
    return json.loads(body_bytes.decode("utf-8"))
