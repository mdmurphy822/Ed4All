"""Regression: ``RewriteProvider._num_ctx_options_payload`` honors the
strict-OpenAI opt-out (``ED4ALL_LLM_OMIT_OLLAMA_FORMAT``).

Bug: on a LOCAL LOOPBACK seat the method unconditionally returned the
Ollama-style ``{"options": {"num_ctx": ...}}`` request-body extra, on the
assumption that a local server is Ollama and ignores an unknown field. A
strict-OpenAI local server (vLLM / TRT-LLM) does NOT ignore it — it rejects
the request with HTTP 400 ``extra_forbidden`` (``loc: ('body','options')``),
400-ing EVERY rewrite-tier dispatch. This payload bypasses the client-level
``format`` guard that ``ED4ALL_LLM_OMIT_OLLAMA_FORMAT`` already governs in
``Trainforge/generators/_openai_compatible_client.py``, so the resolver must
be honored here too.

Fix: when ``ED4ALL_LLM_OMIT_OLLAMA_FORMAT`` is truthy, the method returns
``None`` (no Ollama-only field). When unset/falsey, behavior is
byte-identical to before (real Ollama seats still get ``options.num_ctx``).

The method feeds BOTH the single-block (``generate_rewrite``) and batch
(``generate_rewrite_batch``) dispatch paths, so fixing the method covers
both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._rewrite_provider import RewriteProvider  # noqa: E402


class _FakeOA:
    """Minimal OpenAI-compatible client stub pinning a localhost base_url —
    the local vLLM / TRT-LLM seat the campaign serves at :8123."""

    model = "spark-super"
    base_url = "http://localhost:8123/v1"


def _local_provider() -> RewriteProvider:
    return RewriteProvider(provider="local", client=_FakeOA())


def test_num_ctx_payload_sent_when_flag_unset(monkeypatch):
    """Legacy behavior preserved: a localhost seat with the flag UNSET still
    receives the Ollama-style ``options.num_ctx`` payload."""
    monkeypatch.delenv("ED4ALL_LLM_OMIT_OLLAMA_FORMAT", raising=False)
    provider = _local_provider()

    payload = provider._num_ctx_options_payload()

    assert payload is not None
    assert payload == {"options": {"num_ctx": int(provider._rewrite_num_ctx)}}


def test_num_ctx_payload_omitted_when_flag_set(monkeypatch):
    """Strict-OpenAI opt-out: the SAME localhost seat with
    ``ED4ALL_LLM_OMIT_OLLAMA_FORMAT=1`` receives NO ``options`` payload, so a
    vLLM / TRT-LLM seat no longer 400s on every rewrite dispatch."""
    monkeypatch.setenv("ED4ALL_LLM_OMIT_OLLAMA_FORMAT", "1")
    provider = _local_provider()

    assert provider._num_ctx_options_payload() is None


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE", "On"])
def test_num_ctx_payload_omitted_for_all_truthy_tokens(monkeypatch, truthy):
    """The opt-out honors the canonical ``1/true/yes/on`` truthy semantics of
    the shared ``_omit_ollama_format`` resolver."""
    monkeypatch.setenv("ED4ALL_LLM_OMIT_OLLAMA_FORMAT", truthy)
    assert _local_provider()._num_ctx_options_payload() is None
