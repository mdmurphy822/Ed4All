"""Tests for protocol-first local model-server discovery.

The GUI must not assume the ``local`` provider IS Ollama. Discovery probes the
OpenAI-compatible ``GET {root}/v1/models`` (served by vLLM, Ollama, llama.cpp,
LM Studio) FIRST and falls back to Ollama's native ``GET {root}/api/tags`` only
when the OpenAI path fails or returns nothing. These tests pin that contract:
protocol-first order, the ``/v1`` suffix is never doubled, an unreachable server
degrades with a VENDOR-NEUTRAL message, and the deprecated ``/ollama-models``
alias still resolves.

The service is stdlib-only, so most tests drive it directly by monkeypatching the
single bounded HTTP helper. Two router tests need fastapi and are skipped on a
default install.
"""

from __future__ import annotations

import pytest

from gui.services import settings_service


# --------------------------------------------------------------- fake transport


def _fake_transport(responses):
    """Build a fake ``_http_get_json`` that returns/raises per URL substring.

    ``responses`` maps a URL SUBSTRING → either a payload dict (returned) or an
    Exception instance (raised). The recorded ``calls`` list captures every URL
    the service actually requested (used to assert the ``/v1`` suffix is never
    doubled and the probe ORDER is protocol-first).
    """
    calls = []

    def _fake(url, key):  # noqa: ANN001 — test double
        calls.append(url)
        for needle, result in responses.items():
            if needle in url:
                if isinstance(result, Exception):
                    raise result
                return result
        raise ConnectionError(f"no fake response for {url}")

    return _fake, calls


# ------------------------------------------------------------------- unit tests


def test_v1_models_is_probed_first(monkeypatch):
    """A vLLM seat answering /v1/models resolves WITHOUT any /api/tags probe."""
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8001")
    fake, calls = _fake_transport(
        {"/v1/models": {"data": [{"id": "nemotron-3-super"}, {"id": "aux"}]}}
    )
    monkeypatch.setattr(settings_service, "_http_get_json", fake)

    res = settings_service.list_local_models()

    assert res["available"] is True
    assert res["backend"] == "openai-compatible"
    assert res["models"] == ["nemotron-3-super", "aux"]
    assert res["host"] == "http://localhost:8001"
    # Protocol-first: the /v1/models endpoint answered, so /api/tags is never hit.
    assert calls == ["http://localhost:8001/v1/models"]


def test_ollama_tags_fallback(monkeypatch):
    """When /v1/models fails, the native Ollama /api/tags is the fallback."""
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")
    fake, calls = _fake_transport(
        {
            "/v1/models": ConnectionError("refused"),
            "/api/tags": {"models": [{"name": "llava:13b"}, {"name": "qwen2.5:7b"}]},
        }
    )
    monkeypatch.setattr(settings_service, "_http_get_json", fake)

    res = settings_service.list_local_models()

    assert res["available"] is True
    assert res["backend"] == "ollama"
    assert res["models"] == ["llava:13b", "qwen2.5:7b"]
    # v1 first (failed), THEN the tags fallback.
    assert calls == [
        "http://localhost:11434/v1/models",
        "http://localhost:11434/api/tags",
    ]


def test_v1_suffix_is_never_doubled(monkeypatch):
    """A base URL already carrying /v1 must not compose /v1/v1/models."""
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8001/v1")
    fake, calls = _fake_transport(
        {"/v1/models": {"data": [{"id": "m"}]}}
    )
    monkeypatch.setattr(settings_service, "_http_get_json", fake)

    res = settings_service.list_local_models()

    assert res["host"] == "http://localhost:8001"
    for url in calls:
        assert "/v1/v1" not in url, url
    assert calls[0] == "http://localhost:8001/v1/models"


def test_empty_openai_list_falls_back_to_tags(monkeypatch):
    """A reachable /v1/models with an EMPTY list still tries the tags fallback."""
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")
    fake, calls = _fake_transport(
        {
            "/v1/models": {"data": []},
            "/api/tags": {"models": [{"name": "phi3"}]},
        }
    )
    monkeypatch.setattr(settings_service, "_http_get_json", fake)

    res = settings_service.list_local_models()

    assert res["available"] is True
    assert res["backend"] == "ollama"
    assert res["models"] == ["phi3"]
    assert calls[0].endswith("/v1/models")
    assert calls[1].endswith("/api/tags")


def test_reachable_but_empty_openai_when_tags_also_empty(monkeypatch):
    """Both endpoints reachable-but-empty → available, openai-compatible, no models."""
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8001")
    fake, _calls = _fake_transport(
        {"/v1/models": {"data": []}, "/api/tags": {"models": []}}
    )
    monkeypatch.setattr(settings_service, "_http_get_json", fake)

    res = settings_service.list_local_models()

    assert res["available"] is True
    assert res["backend"] == "openai-compatible"
    assert res["models"] == []


def test_unreachable_is_graceful_and_vendor_neutral(monkeypatch):
    """A down local server → available False, backend None, no vendor blame."""
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8001")
    fake, _calls = _fake_transport(
        {
            "/v1/models": ConnectionError("refused"),
            "/api/tags": ConnectionError("refused"),
        }
    )
    monkeypatch.setattr(settings_service, "_http_get_json", fake)

    res = settings_service.list_local_models()

    assert res["available"] is False
    assert res["backend"] is None
    assert res["models"] == []
    # Vendor-neutral: names the local server + base URL, not "Ollama not reachable".
    assert "Local model server not reachable" in res["detail"]
    assert "http://localhost:8001" in res["detail"]
    assert "Ollama not reachable" not in res["detail"]


def test_deprecated_alias_delegates(monkeypatch):
    """``list_ollama_models`` is a thin delegate to the protocol-first path."""
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:8001")
    fake, _calls = _fake_transport(
        {"/v1/models": {"data": [{"id": "served"}]}}
    )
    monkeypatch.setattr(settings_service, "_http_get_json", fake)

    res = settings_service.list_ollama_models()

    assert res["backend"] == "openai-compatible"
    assert res["models"] == ["served"]


# ---------------------------------------------------------------- router tests

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402


@pytest.fixture
def client(state_dir, libv2_root):
    return TestClient(create_app())


def test_local_models_endpoint(client, monkeypatch):
    """GET /api/settings/local-models returns the protocol-first shape."""
    monkeypatch.setattr(
        settings_service,
        "list_local_models",
        lambda: {
            "available": True,
            "models": ["m1"],
            "detail": "ok",
            "host": "http://localhost:8001",
            "backend": "openai-compatible",
        },
    )
    resp = client.get("/api/settings/local-models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "openai-compatible"
    assert body["models"] == ["m1"]


def test_deprecated_ollama_models_alias_still_resolves(client, monkeypatch):
    """GET /api/settings/ollama-models still resolves (deprecated alias)."""
    monkeypatch.setattr(
        settings_service,
        "list_local_models",
        lambda: {
            "available": False,
            "models": [],
            "detail": "down",
            "host": "http://localhost:8001",
            "backend": None,
        },
    )
    resp = client.get("/api/settings/ollama-models")
    assert resp.status_code == 200
    assert resp.json()["backend"] is None
