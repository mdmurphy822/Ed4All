"""Regression tests for the per-request timeout resolution.

Covers the cross-cutting ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS`` knob
honored by :class:`OpenAICompatibleClient` and the explicit generous
timeout the Courseforge content-generation tiers (outline / rewrite)
pass so local 7B prose generation isn't capped at the 60s client
default.

Resolution / precedence (high → low) for the client:

    explicit per-call ``timeout`` arg
    > ``ED4ALL_LLM_REQUEST_TIMEOUT_SECONDS``
    > ``DEFAULT_TIMEOUT_SECONDS`` (60.0)

Garbage / non-positive env values fall back to the default (mirrors the
parse-with-fallback pattern of the other ``ED4ALL_*`` timeout knobs).

The outline / rewrite providers source their default from the SAME env
var but fall back to 300.0 (not the bare 60s client default) when it is
unset.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators._openai_compatible_client import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    ENV_REQUEST_TIMEOUT,
    OpenAICompatibleClient,
    resolve_default_timeout,
)


def _make_client(**kwargs) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b-instruct-q4_K_M",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# (a) Env var honored when no explicit timeout passed.
# ---------------------------------------------------------------------------


def test_env_var_used_when_no_explicit_timeout(monkeypatch):
    monkeypatch.setenv(ENV_REQUEST_TIMEOUT, "275")
    client = _make_client()
    assert client._timeout == pytest.approx(275.0)


def test_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV_REQUEST_TIMEOUT, raising=False)
    client = _make_client()
    assert client._timeout == pytest.approx(DEFAULT_TIMEOUT_SECONDS)
    assert client._timeout == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# (b) Explicit per-call timeout wins over the env var.
# ---------------------------------------------------------------------------


def test_explicit_timeout_wins_over_env(monkeypatch):
    monkeypatch.setenv(ENV_REQUEST_TIMEOUT, "275")
    client = _make_client(timeout=42.0)
    assert client._timeout == pytest.approx(42.0)


def test_explicit_timeout_wins_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV_REQUEST_TIMEOUT, raising=False)
    client = _make_client(timeout=123.0)
    assert client._timeout == pytest.approx(123.0)


# ---------------------------------------------------------------------------
# (c) Garbage / non-positive env values fall back to the default.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["not-a-number", "", "   ", "0", "-5", "nan-ish", "nan", "inf"]
)
def test_garbage_env_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv(ENV_REQUEST_TIMEOUT, bad)
    # Unparseable / non-positive / non-finite (nan, inf) → default.
    assert resolve_default_timeout() == pytest.approx(DEFAULT_TIMEOUT_SECONDS)
    client = _make_client()
    assert client._timeout == pytest.approx(DEFAULT_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# (d) Outline / rewrite provider path ends up with a generous timeout
#     (>= 300) rather than 60.
# ---------------------------------------------------------------------------


def test_rewrite_provider_local_timeout_is_generous(monkeypatch):
    monkeypatch.delenv(ENV_REQUEST_TIMEOUT, raising=False)
    monkeypatch.delenv("COURSEFORGE_REWRITE_MODEL", raising=False)
    from Courseforge.generators.rewrite._rewrite_provider import RewriteProvider

    provider = RewriteProvider(provider="local")
    assert provider._oa_client is not None
    assert provider._oa_client._timeout >= 300.0
    # Specifically NOT the bare 60s client default.
    assert provider._oa_client._timeout != pytest.approx(60.0)


def test_outline_provider_local_timeout_is_generous(monkeypatch):
    monkeypatch.delenv(ENV_REQUEST_TIMEOUT, raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_MODEL", raising=False)
    from Courseforge.generators.outline._outline_provider import OutlineProvider

    provider = OutlineProvider(provider="local")
    assert provider._oa_client is not None
    assert provider._oa_client._timeout >= 300.0
    assert provider._oa_client._timeout != pytest.approx(60.0)


def test_rewrite_provider_env_overrides_default(monkeypatch):
    monkeypatch.setenv(ENV_REQUEST_TIMEOUT, "450")
    from Courseforge.generators.rewrite._rewrite_provider import RewriteProvider

    provider = RewriteProvider(provider="local")
    assert provider._oa_client is not None
    assert provider._oa_client._timeout == pytest.approx(450.0)


def test_outline_provider_explicit_kwarg_wins(monkeypatch):
    monkeypatch.setenv(ENV_REQUEST_TIMEOUT, "450")
    from Courseforge.generators.outline._outline_provider import OutlineProvider

    provider = OutlineProvider(provider="local", timeout=88.0)
    assert provider._oa_client is not None
    assert provider._oa_client._timeout == pytest.approx(88.0)


def test_rewrite_provider_garbage_env_falls_back_to_300(monkeypatch):
    monkeypatch.setenv(ENV_REQUEST_TIMEOUT, "garbage")
    from Courseforge.generators.rewrite._rewrite_provider import RewriteProvider

    provider = RewriteProvider(provider="local")
    assert provider._oa_client is not None
    # Provider-level fallback is 300.0, NOT the 60s client default.
    assert provider._oa_client._timeout == pytest.approx(300.0)
