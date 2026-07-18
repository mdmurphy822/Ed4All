"""Unit tests for the declarative SEAT SCHEDULE primitives.

Contract under test (``lib/vllm_container_lifecycle.py`` seat-schedule half):

  * ``resolve_seat_schedule_mode`` default-OFF / truthy-ON parse.
  * ``parse_seat_registry`` good / garbage / empty (``seat_name=base_url``).
  * ``resolve_seat_base_url`` / ``all_registered_seat_names``.
  * ``start_seat`` already-serving → 0.0; cold start + poll → seconds;
    unresolvable seat / missing container / start-fail / timeout → None.
  * ``stop_seat`` resolves + ``docker stop``; unresolvable → False.
  * ``coherence_probe`` sane content → True; empty / soup / no-choices / no
    model-id / HTTP error → False (NEVER raises).
  * ``_looks_coherent`` heuristic.

NO live docker / NO live network — subprocess + urlopen are monkeypatched.
"""

from __future__ import annotations

import io
import json

import pytest

import lib.vllm_container_lifecycle as vcl


_SEATS = "spark-super=http://localhost:8001,spark-glm=http://localhost:8002/"
_CONTAINERS = "http://localhost:8001=vllm-super,http://localhost:8002=vllm-glm"


@pytest.fixture(autouse=True)
def _reset_warn_flags():
    vcl._REGISTRY_WARNED = False
    vcl._SEAT_REGISTRY_WARNED = False
    yield
    vcl._REGISTRY_WARNED = False
    vcl._SEAT_REGISTRY_WARNED = False


@pytest.fixture
def _seat_env(monkeypatch):
    monkeypatch.setenv(vcl.ENV_SEAT_BASE_URLS, _SEATS)
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _CONTAINERS)


# --------------------------------------------------------------------------
# resolve_seat_schedule_mode
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False), ("", False), ("   ", False), ("0", False),
        ("false", False), ("off", False), ("garbage", False),
        ("1", True), ("true", True), ("TRUE", True), ("Yes", True), ("on", True),
    ],
)
def test_resolve_seat_schedule_mode(value, expected):
    assert vcl.resolve_seat_schedule_mode(value) is expected


# --------------------------------------------------------------------------
# parse_seat_registry / resolve_seat_base_url / all_registered_seat_names
# --------------------------------------------------------------------------
def test_parse_seat_registry_good():
    reg = vcl.parse_seat_registry(_SEATS)
    assert reg == {
        "spark-super": "http://localhost:8001",
        "spark-glm": "http://localhost:8002",   # trailing / stripped
    }


def test_parse_seat_registry_empty():
    assert vcl.parse_seat_registry("") == {}
    assert vcl.parse_seat_registry(None) == {}


def test_parse_seat_registry_partly_garbage():
    reg = vcl.parse_seat_registry("spark-super=http://x:1, bad, =y, z=, ok=http://y:2")
    assert reg == {"spark-super": "http://x:1", "ok": "http://y:2"}


def test_resolve_seat_base_url():
    assert vcl.resolve_seat_base_url("spark-super", value=_SEATS) == "http://localhost:8001"
    assert vcl.resolve_seat_base_url("nope", value=_SEATS) is None


def test_all_registered_seat_names():
    assert vcl.all_registered_seat_names(_SEATS) == {"spark-super", "spark-glm"}


# --------------------------------------------------------------------------
# _looks_coherent
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,ok",
    [
        ("SEATOK", True), ("  ok now  ", True), ("The answer is 42.", True),
        (None, False), ("", False), ("    ", False),
        ("!!!!", False), ("aaaa", False), ("        ", False),
        (".", False),
    ],
)
def test_looks_coherent(text, ok):
    assert vcl._looks_coherent(text) is ok


# --------------------------------------------------------------------------
# start_seat
# --------------------------------------------------------------------------
def test_start_seat_already_serving(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "_probe_ready", lambda base_url, **k: True)
    calls = []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: calls.append(args) or True)
    assert vcl.start_seat("spark-super") == 0.0
    assert calls == []  # no docker start when already serving


def test_start_seat_cold_start(monkeypatch, _seat_env):
    # not ready first probe, ready second → measured load
    probes = iter([False, True])
    monkeypatch.setattr(vcl, "_probe_ready", lambda base_url, **k: next(probes))
    started = []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: started.append(args) or True)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)
    load = vcl.start_seat("spark-super")
    assert isinstance(load, float) and load >= 0.0
    assert started == [["start", "vllm-super"]]


def test_start_seat_unresolvable_seat(monkeypatch, _seat_env):
    # never touches docker/probe when the seat name is not mapped
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: pytest.fail("probed"))
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: pytest.fail("dockered"))
    assert vcl.start_seat("ghost") is None


def test_start_seat_no_container_mapping(monkeypatch):
    monkeypatch.setenv(vcl.ENV_SEAT_BASE_URLS, "spark-super=http://localhost:9999")
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _CONTAINERS)  # 9999 not mapped
    assert vcl.start_seat("spark-super") is None


def test_start_seat_docker_start_fails(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: False)
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: False)
    assert vcl.start_seat("spark-super") is None


def test_start_seat_timeout(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "_probe_ready", lambda *a, **k: False)
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: True)
    monkeypatch.setattr(vcl.time, "sleep", lambda *_a, **_k: None)
    assert vcl.start_seat("spark-super", timeout_seconds=0) is None


# --------------------------------------------------------------------------
# stop_seat
# --------------------------------------------------------------------------
def test_stop_seat(monkeypatch, _seat_env):
    calls = []
    monkeypatch.setattr(vcl, "_run_docker", lambda args, **k: calls.append(args) or True)
    assert vcl.stop_seat("spark-glm") is True
    assert calls == [["stop", "vllm-glm"]]


def test_stop_seat_unresolvable(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "_run_docker", lambda *a, **k: pytest.fail("dockered"))
    assert vcl.stop_seat("ghost") is False


# --------------------------------------------------------------------------
# coherence_probe
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen_factory(models_payload, chat_payload, *, chat_error=None):
    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if url.endswith("/v1/models"):
            return _FakeResp(models_payload)
        if url.endswith("/v1/chat/completions"):
            if chat_error is not None:
                raise chat_error
            return _FakeResp(chat_payload)
        raise AssertionError(f"unexpected url {url}")
    return _fake


_MODELS_OK = {"data": [{"id": "nemotron-super"}]}


def test_coherence_probe_ok(monkeypatch):
    chat = {"choices": [{"message": {"role": "assistant", "content": "SEATOK"}}]}
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, chat))
    assert vcl.coherence_probe("http://localhost:8001") is True


def test_coherence_probe_empty_content(monkeypatch):
    chat = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, chat))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_null_content(monkeypatch):
    chat = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, chat))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_soup(monkeypatch):
    chat = {"choices": [{"message": {"role": "assistant", "content": "!!!!!!!!"}}]}
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, chat))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_no_choices(monkeypatch):
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory(_MODELS_OK, {"choices": []}))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_no_model_id(monkeypatch):
    monkeypatch.setattr(vcl.urllib.request, "urlopen",
                        _fake_urlopen_factory({"data": []}, {}))
    assert vcl.coherence_probe("http://localhost:8001") is False


def test_coherence_probe_http_error(monkeypatch):
    monkeypatch.setattr(
        vcl.urllib.request, "urlopen",
        _fake_urlopen_factory(_MODELS_OK, {}, chat_error=OSError("connreset")),
    )
    assert vcl.coherence_probe("http://localhost:8001") is False
