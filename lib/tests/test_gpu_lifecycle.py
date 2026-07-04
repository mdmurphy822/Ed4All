"""Unit tests for the deterministic GPU-lifecycle lease helper.

Contract under test (``lib/gpu_lifecycle.py``):

  * ``release_ollama_models`` enumerates the resident models via a mocked
    ``GET /api/ps`` and POSTs ``keep_alive:0`` once per resident model,
    returning the evicted names; fail-soft (ollama down / httpx absent / bad
    JSON / unload error → ``[]`` or a partial list, never raises).
  * ``release_torch`` is fail-soft with torch absent, runs every registered
    releaser (one raising does not stop the rest), and reclaims the allocator
    cache.
  * ``stage_lease`` releases on clean exit AND on exception; a pure no-op when
    the lifecycle mode is off.
  * ``resolve_gpu_lifecycle_mode`` default-ON / explicit-falsey-off / garbage-ON
    parse semantics.

No live model / no live ollama — the httpx client and torch are mocked.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

import lib.gpu_lifecycle as gl


# --------------------------------------------------------------------------
# A fake httpx module the helper's ``import httpx`` picks up.
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload=None, *, raise_on_status=False):
        self._payload = payload
        self._raise = raise_on_status

    def raise_for_status(self):
        if self._raise:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._payload


class _FakeClient:
    """Records /api/ps + /api/generate calls; scripted ps payload."""

    def __init__(self, ps_models, *, get_raises=False, post_fail_for=None):
        self._ps_models = ps_models
        self._get_raises = get_raises
        self._post_fail_for = set(post_fail_for or ())
        self.unloaded = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url):
        assert url.endswith("/api/ps")
        if self._get_raises:
            raise RuntimeError("connection refused")
        return _FakeResp({"models": self._ps_models})

    def post(self, url, json=None):
        assert url.endswith("/api/generate")
        assert json.get("keep_alive") == 0
        model = json.get("model")
        if model in self._post_fail_for:
            return _FakeResp(raise_on_status=True)
        self.unloaded.append(model)
        return _FakeResp({})


def _install_fake_httpx(monkeypatch, client):
    fake = types.ModuleType("httpx")
    fake.Client = lambda *a, **k: client
    monkeypatch.setitem(sys.modules, "httpx", fake)


# --------------------------------------------------------------------------
# resolve_gpu_lifecycle_mode — DEFAULT ON parse semantics
# --------------------------------------------------------------------------
def test_mode_default_on_when_unset(monkeypatch):
    monkeypatch.delenv("ED4ALL_GPU_LIFECYCLE", raising=False)
    assert gl.resolve_gpu_lifecycle_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "FALSE", " Off "])
def test_mode_explicit_falsey_off(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", val)
    assert gl.resolve_gpu_lifecycle_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "garbage", "", "   "])
def test_mode_truthy_blank_garbage_on(monkeypatch, val):
    # Blank / garbage / truthy all resolve ON (default-on parse-with-fallback).
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", val)
    assert gl.resolve_gpu_lifecycle_mode() is True


def test_mode_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "off")
    assert gl.resolve_gpu_lifecycle_mode("on") is True
    assert gl.resolve_gpu_lifecycle_mode("off") is False


# --------------------------------------------------------------------------
# release_ollama_models
# --------------------------------------------------------------------------
def test_release_ollama_multi_model_unloads_each(monkeypatch):
    client = _FakeClient(
        [{"name": "qwen2.5:7b"}, {"name": "qwen2.5-vl:7b"}]
    )
    _install_fake_httpx(monkeypatch, client)

    evicted = gl.release_ollama_models()

    assert evicted == ["qwen2.5:7b", "qwen2.5-vl:7b"]
    assert client.unloaded == ["qwen2.5:7b", "qwen2.5-vl:7b"]


def test_release_ollama_empty_when_none_resident(monkeypatch):
    client = _FakeClient([])
    _install_fake_httpx(monkeypatch, client)
    assert gl.release_ollama_models() == []
    assert client.unloaded == []


def test_release_ollama_failsoft_on_ps_error(monkeypatch):
    client = _FakeClient([{"name": "x"}], get_raises=True)
    _install_fake_httpx(monkeypatch, client)
    # /api/ps blows up → empty list, never raises.
    assert gl.release_ollama_models() == []


def test_release_ollama_partial_when_one_unload_fails(monkeypatch):
    client = _FakeClient(
        [{"name": "good"}, {"name": "bad"}], post_fail_for=["bad"]
    )
    _install_fake_httpx(monkeypatch, client)
    # Only the model that unloaded cleanly is reported.
    assert gl.release_ollama_models() == ["good"]


def test_release_ollama_failsoft_when_httpx_absent(monkeypatch):
    # Simulate httpx import failing.
    monkeypatch.setitem(sys.modules, "httpx", None)
    # ``import httpx`` with a None entry raises ImportError → caught.
    assert gl.release_ollama_models() == []


# --------------------------------------------------------------------------
# release_torch + the releaser registry
# --------------------------------------------------------------------------
def test_release_torch_failsoft_without_torch(monkeypatch):
    # No torch installed / broken import → must not raise.
    monkeypatch.setitem(sys.modules, "torch", None)
    gl.release_torch()  # no raise


def test_registered_releasers_all_invoked_one_raising(monkeypatch):
    monkeypatch.setattr(gl, "_RELEASERS", [])
    calls = []
    gl.register_releaser(lambda: calls.append("a"))

    def _boom():
        calls.append("b")
        raise RuntimeError("releaser exploded")

    gl.register_releaser(_boom)
    gl.register_releaser(lambda: calls.append("c"))

    # torch absent so the tail is a no-op; the point is all 3 releasers ran.
    monkeypatch.setitem(sys.modules, "torch", None)
    gl.release_torch()

    assert calls == ["a", "b", "c"]


def test_register_releaser_is_idempotent(monkeypatch):
    monkeypatch.setattr(gl, "_RELEASERS", [])
    fn = lambda: None
    gl.register_releaser(fn)
    gl.register_releaser(fn)
    gl.register_releaser("not callable")  # ignored
    assert gl._RELEASERS == [fn]


def test_release_torch_calls_empty_cache_when_cuda(monkeypatch):
    monkeypatch.setattr(gl, "_RELEASERS", [])
    fake_torch = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=MagicMock(name="empty_cache"),
    )
    fake_torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    gl.release_torch(stage="unit")

    cuda.empty_cache.assert_called_once()


# --------------------------------------------------------------------------
# stage_lease
# --------------------------------------------------------------------------
def test_stage_lease_releases_on_clean_exit(monkeypatch):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "on")
    ollama = MagicMock(name="release_ollama_models", return_value=[])
    torch = MagicMock(name="release_torch")
    monkeypatch.setattr(gl, "release_ollama_models", ollama)
    monkeypatch.setattr(gl, "release_torch", torch)

    with gl.stage_lease("s"):
        pass

    ollama.assert_called_once()
    torch.assert_called_once()


def test_stage_lease_releases_on_exception(monkeypatch):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "on")
    ollama = MagicMock(return_value=[])
    torch = MagicMock()
    monkeypatch.setattr(gl, "release_ollama_models", ollama)
    monkeypatch.setattr(gl, "release_torch", torch)

    with pytest.raises(ValueError):
        with gl.stage_lease("s"):
            raise ValueError("body blew up")

    # Release still fired on the exception path, and the body's exception
    # propagated unchanged.
    ollama.assert_called_once()
    torch.assert_called_once()


def test_stage_lease_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "off")
    ollama = MagicMock(return_value=[])
    torch = MagicMock()
    monkeypatch.setattr(gl, "release_ollama_models", ollama)
    monkeypatch.setattr(gl, "release_torch", torch)

    with gl.stage_lease("s"):
        pass

    ollama.assert_not_called()
    torch.assert_not_called()


def test_stage_lease_arms_selectable(monkeypatch):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "on")
    ollama = MagicMock(return_value=[])
    torch = MagicMock()
    monkeypatch.setattr(gl, "release_ollama_models", ollama)
    monkeypatch.setattr(gl, "release_torch", torch)

    with gl.stage_lease("captioner", ollama=False, torch=True):
        pass

    ollama.assert_not_called()
    torch.assert_called_once()


def test_stage_lease_release_failure_is_swallowed(monkeypatch):
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "on")
    monkeypatch.setattr(
        gl, "release_ollama_models", MagicMock(side_effect=RuntimeError("x"))
    )
    monkeypatch.setattr(
        gl, "release_torch", MagicMock(side_effect=RuntimeError("y"))
    )
    # A release failure inside the lease must not propagate.
    with gl.stage_lease("s"):
        pass


def test_release_ollama_delegates_to_vram_reclaim_primitives():
    # Regression guard: gpu_lifecycle imports the SHARED vram_reclaim parser +
    # unloader rather than forking them (keeps test_vram_reclaim.py the SoT).
    from lib.llm import vram_reclaim

    assert gl._fetch_resident_models is vram_reclaim._fetch_resident_models
    assert gl._unload_model is vram_reclaim._unload_model
    assert gl.resolve_ollama_root is vram_reclaim.resolve_ollama_root
