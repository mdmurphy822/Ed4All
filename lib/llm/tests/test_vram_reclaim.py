"""Unit tests for :mod:`lib.llm.vram_reclaim`.

Hermetic — the ollama HTTP surface (``/api/ps`` + ``/api/generate``) is
mocked via a fake ``httpx.Client``, so no real ollama server is needed.
Covers: base-URL resolution (``/v1`` stripped to the native root),
model discovery from ``/api/ps``, the ``keep_alive: 0`` unload, the
True/False return contract, and graceful degradation when ollama is down
/ httpx is missing / no model is resident.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from lib.llm import vram_reclaim
from lib.llm.vram_reclaim import (
    DEFAULT_BASE_URL,
    ENV_BASE_URL,
    evict_local_llm,
    resolve_ollama_root,
)


@pytest.fixture(autouse=True)
def _clear_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BASE_URL, raising=False)


# --------------------------------------------------------------------- #
# Base-URL resolution
# --------------------------------------------------------------------- #


def test_resolve_ollama_root_strips_v1_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI-compatible /v1 suffix is stripped to reach /api/*."""
    assert resolve_ollama_root() == "http://localhost:11434"
    monkeypatch.setenv(ENV_BASE_URL, "http://localhost:8000/v1")
    assert resolve_ollama_root() == "http://localhost:8000"
    # No /v1 suffix → returned as-is (trailing slash trimmed).
    monkeypatch.setenv(ENV_BASE_URL, "http://gpu-box:11434/")
    assert resolve_ollama_root() == "http://gpu-box:11434"
    # explicit arg beats env
    monkeypatch.setenv(ENV_BASE_URL, "http://localhost:11434/v1")
    assert resolve_ollama_root("http://other:9999/v1") == "http://other:9999"


def test_default_base_url_is_ollama() -> None:
    assert DEFAULT_BASE_URL == "http://localhost:11434/v1"


# --------------------------------------------------------------------- #
# Fake httpx client
# --------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Records GET/POST calls; replays a scripted /api/ps payload."""

    def __init__(self, ps_models: List[Dict[str, Any]]) -> None:
        self._ps_models = ps_models
        self.get_calls: List[str] = []
        self.post_calls: List[Dict[str, Any]] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get(self, url: str) -> _FakeResponse:
        self.get_calls.append(url)
        return _FakeResponse({"models": self._ps_models})

    def post(self, url: str, json: Optional[Dict[str, Any]] = None) -> _FakeResponse:
        self.post_calls.append({"url": url, "json": json})
        return _FakeResponse({"done": True})


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch, client: Any
) -> None:
    """Make ``import httpx`` inside the helper yield a module whose
    ``Client(...)`` returns ``client``."""
    fake_httpx = MagicMock()
    fake_httpx.Client.return_value = client
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)


# --------------------------------------------------------------------- #
# evict_local_llm — happy path + model resolution from /api/ps
# --------------------------------------------------------------------- #


def test_evict_resolves_model_from_api_ps_and_unloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resident model discovered via /api/ps → keep_alive:0 unload → True."""
    client = _FakeClient(ps_models=[{"name": "qwen2.5:7b-instruct-q4_K_M"}])
    _install_fake_httpx(monkeypatch, client)

    assert evict_local_llm() is True
    # Discovery hit /api/ps at the native root.
    assert client.get_calls == ["http://localhost:11434/api/ps"]
    # Unload POSTed keep_alive:0 for the resident model.
    assert len(client.post_calls) == 1
    post = client.post_calls[0]
    assert post["url"] == "http://localhost:11434/api/generate"
    assert post["json"] == {
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "keep_alive": 0,
    }


def test_evict_unloads_every_resident_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple resident models → each unloaded; accepts 'model' key too."""
    client = _FakeClient(
        ps_models=[{"name": "qwen2.5:7b"}, {"model": "nomic-embed-text"}]
    )
    _install_fake_httpx(monkeypatch, client)

    assert evict_local_llm() is True
    unloaded = {p["json"]["model"] for p in client.post_calls}
    assert unloaded == {"qwen2.5:7b", "nomic-embed-text"}


def test_evict_honors_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOCAL_SYNTHESIS_BASE_URL drives the reclaim host (v1 stripped)."""
    monkeypatch.setenv(ENV_BASE_URL, "http://gpu-box:11434/v1")
    client = _FakeClient(ps_models=[{"name": "qwen2.5:14b"}])
    _install_fake_httpx(monkeypatch, client)

    assert evict_local_llm() is True
    assert client.get_calls == ["http://gpu-box:11434/api/ps"]
    assert client.post_calls[0]["url"] == "http://gpu-box:11434/api/generate"


# --------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------- #


def test_evict_no_resident_models_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty /api/ps → nothing to free → False, no unload POST."""
    client = _FakeClient(ps_models=[])
    _install_fake_httpx(monkeypatch, client)

    assert evict_local_llm() is False
    assert client.post_calls == []


def test_evict_ollama_down_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/api/ps raises (server down) → graceful False, never raises."""

    class _DownClient(_FakeClient):
        def get(self, url: str) -> _FakeResponse:
            raise ConnectionError("connection refused")

    _install_fake_httpx(monkeypatch, _DownClient(ps_models=[]))
    assert evict_local_llm() is False


def test_evict_unload_post_fails_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model resident but the unload POST fails → False (nothing freed)."""

    class _PostFailClient(_FakeClient):
        def post(
            self, url: str, json: Optional[Dict[str, Any]] = None
        ) -> _FakeResponse:
            raise TimeoutError("unload timed out")

    _install_fake_httpx(
        monkeypatch, _PostFailClient(ps_models=[{"name": "qwen2.5:7b"}])
    )
    assert evict_local_llm() is False


def test_evict_httpx_missing_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx import failure → graceful False (no crash)."""
    real_import = __import__

    def _no_httpx(name: str, *a: Any, **k: Any) -> Any:
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", _no_httpx)
    monkeypatch.delitem(sys.modules, "httpx", raising=False)
    assert evict_local_llm() is False


def test_evict_malformed_ps_payload_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/api/ps returns an unexpected shape → no models parsed → False."""

    class _BadShapeClient(_FakeClient):
        def get(self, url: str) -> _FakeResponse:
            self.get_calls.append(url)
            return _FakeResponse({"unexpected": "shape"})

    _install_fake_httpx(monkeypatch, _BadShapeClient(ps_models=[]))
    assert evict_local_llm() is False
