"""Per-provider constrained-decoding payload routing tests (Phase 3 Subtask 56).

End-to-end coverage of the constrained-decoding wire path:

    OutlineProvider._build_grammar_payload(block_type)
        -> _BaseLLMProvider._dispatch_call(extra_payload=...)
        -> OpenAICompatibleClient._post_with_retry(payload)
        -> httpx POST body

Each test installs an ``httpx.MockTransport`` handler that captures
the outbound request body, drives a single ``OutlineProvider``
dispatch (via either ``generate_outline`` or a direct
``_dispatch_call`` invocation when ``generate_outline`` would
require a fully-populated source-chunk fixture), and asserts the
captured POST body carries the exact per-provider grammar field name
the router routes through. Six tests:

- ``test_grammar_payload_for_local_with_gbnf_mode_includes_grammar_field``
  — ``provider="local"`` + ``grammar_mode="gbnf"`` puts the GBNF
  grammar string under the top-level ``grammar`` key (llama.cpp /
  LM Studio convention).

- ``test_grammar_payload_for_ollama_json_schema_mode_includes_format_dict``
  — ``provider="local"`` + ``grammar_mode="json_schema"`` puts the
  Draft 2020-12 schema dict under the top-level ``format`` key
  (Ollama 0.5+ convention; the dict-shape distinguishes from
  Wave-113 ``json_mode`` which sets ``format="json"`` literal).

- ``test_grammar_payload_for_together_includes_response_format_json_schema``
  — ``provider="together"`` (auto-detect) puts the schema under
  ``response_format.json_schema.schema`` with ``"type": "json_schema"``
  and ``"strict": true`` (OpenAI-style strict structured-output path
  that Together AI implements).

- ``test_grammar_payload_for_vllm_includes_extra_body_guided_json``
  — ``provider="local"`` (or ``openai_compatible``) with a base URL
  containing ``vllm`` puts the schema under
  ``extra_body.guided_json`` (vLLM convention; the harness wraps the
  passthrough container under ``extra_body`` per vLLM's API surface).

- ``test_grammar_payload_for_anthropic_falls_back_to_json_mode_only``
  — ``provider="anthropic"`` returns an empty payload from
  ``_build_grammar_payload`` (the Anthropic SDK doesn't accept
  arbitrary OpenAI-compatible keys). This test verifies the
  ``_build_grammar_payload`` return value directly because Anthropic
  routes through the SDK, not the OpenAI-compatible wire path.

- ``test_grammar_mode_env_var_overrides_autodetect``
  — ``COURSEFORGE_OUTLINE_GRAMMAR_MODE`` env var (or the
  ``grammar_mode`` constructor kwarg) forces the explicit-mode
  branch and overrides the provider/base-url autodetect (e.g.
  ``provider="local"`` with no autodetect-friendly base URL still
  emits ``{"grammar": ...}`` when the env var is set to ``"gbnf"``).

The tests exercise the load-bearing wire-through that Subtask 21's
``extra_payload`` kwarg (verified again in Subtask 55 — Worker 2E's
work was complete; no additional code change needed) makes possible.
A regression in any of the four code paths (router build → dispatch
merge → client POST → wire body) fails one of these six tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._outline_provider import (  # noqa: E402
    OutlineProvider,
    _BLOCK_TYPE_GBNF,
    _BLOCK_TYPE_JSON_SCHEMAS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(content: str = "{}") -> dict:
    """Minimal OpenAI-shaped 200 body the OA client unwraps."""
    return {
        "id": "cmpl-cd-test",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _make_capturing_client() -> tuple[httpx.Client, List[Dict[str, Any]]]:
    """Build a client whose handler records every outbound JSON body.

    Returns ``(client, captured_bodies)``; ``captured_bodies`` is the
    list each test inspects after the dispatch call returns. The
    handler always returns a 200 with a syntactically-valid (but
    schema-invalid) ``"{}"`` body so the OA client unwraps cleanly.
    """
    captured: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # Decode the JSON request body the client POSTed.
        import json as _json

        captured.append(_json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=_success_body("{}"))

    return httpx.Client(transport=httpx.MockTransport(handler)), captured


def _make_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: str,
    base_url: str | None = None,
    grammar_mode: str | None = None,
    client: httpx.Client | None = None,
) -> OutlineProvider:
    """Construct an ``OutlineProvider`` with deterministic env state.

    Clears every relevant env var so the tests don't accidentally
    pick up operator-set defaults. Anthropic builds without an HTTP
    client (it routes through the SDK).
    """
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_GRAMMAR_MODE", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_MODEL", raising=False)
    monkeypatch.delenv("TOGETHER_SYNTHESIS_MODEL", raising=False)

    if provider == "anthropic":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    if provider == "together":
        monkeypatch.setenv("TOGETHER_API_KEY", "tk-test")

    return OutlineProvider(
        provider=provider,
        base_url=base_url,
        grammar_mode=grammar_mode,
        client=client,
    )


# ---------------------------------------------------------------------------
# Test 1 — local + gbnf mode → top-level ``grammar`` field
# ---------------------------------------------------------------------------


def test_grammar_payload_for_local_with_gbnf_mode_includes_grammar_field(
    monkeypatch,
):
    """``grammar_mode="gbnf"`` puts the GBNF grammar string under the
    top-level ``grammar`` key on every dispatched POST body. This is
    the llama.cpp / LM Studio convention; the field passes through
    ``OpenAICompatibleClient`` unchanged because Subtask 21's
    ``extra_payload`` merge runs BEFORE the POST."""
    client, captured = _make_capturing_client()
    provider = _make_provider(
        monkeypatch,
        provider="local",
        base_url="http://localhost:8080/v1",  # llama.cpp default
        grammar_mode="gbnf",
        client=client,
    )

    payload = provider._build_grammar_payload("concept")
    # Direct dispatch — drive the wire end-to-end.
    provider._dispatch_call("user prompt", extra_payload=payload)

    assert len(captured) == 1
    body = captured[0]
    assert "grammar" in body, (
        f"expected 'grammar' field in POST body; got keys={list(body.keys())}"
    )
    assert body["grammar"] == _BLOCK_TYPE_GBNF["concept"]
    # Sanity: standard OpenAI-compatible fields still present.
    assert body["model"]
    assert body["messages"]


# ---------------------------------------------------------------------------
# Test 2 — local + json_schema mode → top-level ``format`` dict
# ---------------------------------------------------------------------------


def test_grammar_payload_for_ollama_json_schema_mode_includes_format_dict(
    monkeypatch,
):
    """``grammar_mode="json_schema"`` puts the Draft 2020-12 JSON
    Schema dict under the top-level ``format`` key (Ollama 0.5+
    convention). The dict-shape distinguishes this from Wave-113's
    ``json_mode=True`` injection which sets ``format="json"`` (the
    string literal), so the test asserts the value is a dict carrying
    ``$schema`` / ``type`` / ``properties``."""
    client, captured = _make_capturing_client()
    provider = _make_provider(
        monkeypatch,
        provider="local",
        base_url="http://localhost:11434/v1",  # Ollama default
        grammar_mode="json_schema",
        client=client,
    )

    payload = provider._build_grammar_payload("concept")
    provider._dispatch_call("user prompt", extra_payload=payload)

    assert len(captured) == 1
    body = captured[0]
    assert "format" in body, (
        f"expected 'format' field in POST body; got keys={list(body.keys())}"
    )
    fmt = body["format"]
    # Must be the schema dict, NOT the Wave-113 ``"json"`` literal.
    assert isinstance(fmt, dict), (
        f"json_schema mode must emit a dict; got {type(fmt).__name__}={fmt!r}"
    )
    assert fmt.get("type") == "object"
    assert fmt.get("$schema", "").startswith("https://json-schema.org/")
    assert "properties" in fmt
    assert "block_type" in fmt["properties"]


# ---------------------------------------------------------------------------
# Test 3 — together → response_format.json_schema (strict)
# ---------------------------------------------------------------------------


def test_grammar_payload_for_together_includes_response_format_json_schema(
    monkeypatch,
):
    """``provider="together"`` autodetect emits an OpenAI-style
    ``response_format`` object with ``type=json_schema`` and the
    nested ``json_schema.schema`` carrying the per-block-type Draft
    2020-12 schema. This is Together AI's strict structured-output
    path — verified by asserting the full nesting shape, not just
    the field name."""
    client, captured = _make_capturing_client()
    provider = _make_provider(
        monkeypatch,
        provider="together",
        client=client,
    )

    payload = provider._build_grammar_payload("concept")
    provider._dispatch_call("user prompt", extra_payload=payload)

    assert len(captured) == 1
    body = captured[0]
    assert "response_format" in body, (
        f"expected 'response_format' field in POST body; got keys="
        f"{list(body.keys())}"
    )
    rf = body["response_format"]
    assert isinstance(rf, dict), (
        f"response_format must be a dict; got {type(rf).__name__}"
    )
    assert rf.get("type") == "json_schema"
    nested = rf.get("json_schema")
    assert isinstance(nested, dict), (
        f"response_format.json_schema must be a dict; got {nested!r}"
    )
    assert nested.get("strict") is True
    assert nested.get("name") == "OutlineBlock_concept"
    schema = nested.get("schema")
    assert isinstance(schema, dict)
    assert "block_type" in schema.get("properties", {})


# ---------------------------------------------------------------------------
# Test 4 — vLLM → extra_body.guided_json
# ---------------------------------------------------------------------------


def test_grammar_payload_for_vllm_includes_extra_body_guided_json(
    monkeypatch,
):
    """``provider="local"`` with a vLLM-flavoured base URL puts the
    schema under ``extra_body.guided_json``. vLLM's API surface
    expects guided-decoding fields nested under an ``extra_body``
    container — the test asserts the exact nesting so a regression
    that flattens ``guided_json`` to the top level fails loudly."""
    client, captured = _make_capturing_client()
    provider = _make_provider(
        monkeypatch,
        provider="local",
        base_url="http://localhost:8000/v1/vllm",
        client=client,
    )

    payload = provider._build_grammar_payload("concept")
    provider._dispatch_call("user prompt", extra_payload=payload)

    assert len(captured) == 1
    body = captured[0]
    assert "extra_body" in body, (
        f"expected 'extra_body' field in POST body; got keys="
        f"{list(body.keys())}"
    )
    eb = body["extra_body"]
    assert isinstance(eb, dict), (
        f"extra_body must be a dict; got {type(eb).__name__}"
    )
    assert "guided_json" in eb, (
        f"expected 'guided_json' under extra_body; got keys={list(eb.keys())}"
    )
    schema = eb["guided_json"]
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    assert "block_type" in schema.get("properties", {})
    # And the field MUST NOT be flattened to the top level.
    assert "guided_json" not in body


# ---------------------------------------------------------------------------
# Test 5 — anthropic → empty payload (SDK route, no OpenAI-compat fields)
# ---------------------------------------------------------------------------


def test_grammar_payload_for_anthropic_falls_back_to_json_mode_only(
    monkeypatch,
):
    """``provider="anthropic"`` MUST return an empty grammar payload —
    the Anthropic SDK does not accept arbitrary OpenAI-compatible
    keys (``grammar`` / ``format`` / ``response_format`` / ``extra_body``).
    The constraint is carried by Wave-113's ``json_mode=True`` path on
    the OA client for OpenAI-compatible backends, but Anthropic routes
    through the SDK directly so there is no equivalent field to inject.

    This test asserts the return value of ``_build_grammar_payload``
    is empty (no opportunity for ``_dispatch_call`` to merge anything)
    and that none of the recognised grammar field names appear in the
    payload. Anthropic dispatch goes through the SDK, not the OA
    wire path, so we don't run a wire dispatch — the empty-payload
    return value is the guarantee."""
    provider = _make_provider(
        monkeypatch,
        provider="anthropic",
    )

    payload = provider._build_grammar_payload("concept")

    assert payload == {}, (
        f"anthropic must emit empty payload; got {payload!r}"
    )
    # Sanity: none of the recognised grammar fields are present.
    for forbidden in (
        "grammar",
        "format",
        "guided_grammar",
        "guided_json",
        "guided_regex",
        "extra_body",
        "response_format",
    ):
        assert forbidden not in payload


# ---------------------------------------------------------------------------
# Test 6 — env-var override of autodetect
# ---------------------------------------------------------------------------


def test_grammar_mode_env_var_overrides_autodetect(monkeypatch):
    """``COURSEFORGE_OUTLINE_GRAMMAR_MODE=gbnf`` forces the explicit-
    mode branch even when the provider/base-url combination would
    have autodetected a different field (e.g. a Together provider
    with no llama-flavoured base_url would normally emit
    ``response_format``; with the env var set, it falls into the
    ``mode=="gbnf"`` early-return branch instead).

    Verifies the precedence contract documented in
    ``OutlineProvider._build_grammar_payload``: explicit mode wins
    over the per-provider autodetect. Drives the env var path
    directly (NOT the constructor kwarg), so a regression in
    ``OutlineProvider.__init__``'s env-var resolution would fail
    this test.
    """
    # Clear every relevant env var EXCEPT the grammar-mode one we're
    # testing, then set the grammar-mode env var BEFORE constructing
    # the provider — the provider's ``__init__`` reads
    # ``COURSEFORGE_OUTLINE_GRAMMAR_MODE`` to seed ``self._grammar_mode``.
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_MODEL", raising=False)
    monkeypatch.delenv("TOGETHER_SYNTHESIS_MODEL", raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "tk-test")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_GRAMMAR_MODE", "gbnf")

    client, captured = _make_capturing_client()
    # Together would normally autodetect to ``response_format`` — the
    # env-var override forces ``grammar`` instead. Construct directly
    # (NOT via ``_make_provider``) so the helper's env-clearing
    # doesn't wipe the env var we're testing.
    provider = OutlineProvider(
        provider="together",
        client=client,
    )
    # Sanity-check: the env-var bled through to the provider state.
    assert provider._grammar_mode == "gbnf"

    payload = provider._build_grammar_payload("concept")
    provider._dispatch_call("user prompt", extra_payload=payload)

    assert len(captured) == 1
    body = captured[0]
    assert "grammar" in body, (
        f"env-var override should force 'grammar' field; got keys="
        f"{list(body.keys())}"
    )
    assert body["grammar"] == _BLOCK_TYPE_GBNF["concept"]
    # And the autodetect path's strict ``response_format`` json_schema
    # nesting MUST NOT appear (the GBNF override won the build path).
    # Plan §3.2 companion: the Wave-113 ``json_mode=True`` opt-in on
    # the OpenAICompatibleClient does inject a generic
    # ``response_format: {"type": "json_object"}`` default — that's
    # the JSON-only constraint, not the schema-nested form. Assert the
    # field is either absent OR carries the generic shape (NOT the
    # autodetect's strict json_schema form).
    rf = body.get("response_format")
    if rf is not None:
        assert rf == {"type": "json_object"}, (
            f"unexpected response_format shape under gbnf override: {rf!r}"
        )
