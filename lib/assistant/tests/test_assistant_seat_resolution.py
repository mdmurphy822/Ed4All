"""Dynamic seat/model resolution (S1 shared contract).

Priority walk, model-id read from the seat (never hardcoded), env fallback,
loopback guard, the single-entry TTL cache, and the dynamic-vs-static
AssistantClient behavior. Fully hermetic: probe + model_reader are injected
callables (no network, no real seats), the /v1/models id is read from the
injected reader, and the module TTL cache is reset around every case."""

from __future__ import annotations

import pytest

from lib.assistant.client import (
    ENV_ASSISTANT_BASE_URL,
    AssistantClient,
    AssistantProviderNotLocal,
    ResolvedSeat,
    reset_seat_cache,
    resolve_active_seat,
)

SUPER_URL = "http://localhost:8001/v1"
NANO_URL = "http://localhost:8004/v1"


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_seat_cache()
    yield
    reset_seat_cache()


@pytest.fixture(autouse=True)
def _default_registry(monkeypatch):
    """Default priority walk (spark-super, spark-nano) mapped to loopback URLs."""
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS",
        f"spark-super={SUPER_URL},spark-nano={NANO_URL}",
    )
    monkeypatch.delenv("ED4ALL_ASSISTANT_SEAT_PRIORITY", raising=False)
    monkeypatch.delenv("ED4ALL_ASSISTANT_SEAT", raising=False)
    monkeypatch.delenv("ED4ALL_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv(ENV_ASSISTANT_BASE_URL, raising=False)
    yield


def _reader(mapping):
    return lambda url: mapping.get(url)


# --------------------------------------------------------------------------- #
# Priority order
# --------------------------------------------------------------------------- #


def test_super_live_is_chosen():
    seat = resolve_active_seat(
        force=True,
        probe=lambda url: True,  # both live → first priority wins
        model_reader=_reader({SUPER_URL: "super-served", NANO_URL: "nano-served"}),
    )
    assert seat.seat_name == "spark-super"
    assert seat.base_url == SUPER_URL
    assert seat.model == "super-served"
    assert seat.live is True
    assert seat.source == "priority"


def test_super_down_nano_live_falls_to_nano():
    seat = resolve_active_seat(
        force=True,
        probe=lambda url: url == NANO_URL,  # super down, nano live
        model_reader=_reader({NANO_URL: "nano-served"}),
    )
    assert seat.seat_name == "spark-nano"
    assert seat.base_url == NANO_URL
    assert seat.model == "nano-served"
    assert seat.live is True
    assert seat.source == "priority"


# --------------------------------------------------------------------------- #
# Model id is READ, never hardcoded
# --------------------------------------------------------------------------- #


def test_model_id_read_from_injected_reader():
    seat = resolve_active_seat(
        force=True,
        probe=lambda url: url == SUPER_URL,
        model_reader=_reader({SUPER_URL: "whatever-the-seat-serves"}),
    )
    assert seat.model == "whatever-the-seat-serves"


def test_model_id_reader_none_falls_back_to_env_model(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_MODEL", "env-model-x")
    seat = resolve_active_seat(
        force=True,
        probe=lambda url: url == SUPER_URL,
        model_reader=lambda url: None,  # seat did not report a model id
    )
    assert seat.model == "env-model-x"
    assert seat.live is True


# --------------------------------------------------------------------------- #
# Fallback (registry empty / no seat live)
# --------------------------------------------------------------------------- #


def test_empty_registry_falls_back_to_env_base_url(monkeypatch):
    monkeypatch.delenv("ED4ALL_SEAT_BASE_URLS", raising=False)
    monkeypatch.setenv(ENV_ASSISTANT_BASE_URL, "http://127.0.0.1:9999/v1")
    seat = resolve_active_seat(
        force=True,
        probe=lambda url: url == "http://127.0.0.1:9999/v1",
        model_reader=_reader({"http://127.0.0.1:9999/v1": "fb-model"}),
    )
    assert seat.base_url == "http://127.0.0.1:9999/v1"
    assert seat.model == "fb-model"
    assert seat.live is True
    assert seat.source == "fallback"


def test_no_seat_live_returns_fallback_seat_not_live(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_MODEL", "env-model")
    seat = resolve_active_seat(
        force=True,
        probe=lambda url: False,  # nothing live anywhere
        model_reader=lambda url: "unused",  # not consulted when not live
    )
    assert seat.live is False
    assert seat.source == "fallback"
    # Falls back to the ED4ALL_ASSISTANT_SEAT default + env model resolver.
    assert seat.seat_name == "spark-nano"
    assert seat.model == "env-model"


# --------------------------------------------------------------------------- #
# Loopback guard — LOUD, never silently skipped
# --------------------------------------------------------------------------- #


def test_non_loopback_priority_registry_url_raises(monkeypatch):
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS", "spark-super=http://evil.example.com:8001/v1"
    )
    probed = {"n": 0}

    def probe(url):
        probed["n"] += 1
        return True

    with pytest.raises(AssistantProviderNotLocal) as exc:
        resolve_active_seat(force=True, probe=probe, model_reader=lambda u: "m")
    # Names the seat + the registry env var, and never reached the probe.
    assert "spark-super" in str(exc.value)
    assert "ED4ALL_SEAT_BASE_URLS" in str(exc.value)
    assert probed["n"] == 0


def test_non_loopback_fallback_registry_url_raises(monkeypatch):
    # Priority list references a seat that is NOT in the registry, so the walk
    # yields nothing; the fallback seat itself maps to a non-loopback URL.
    monkeypatch.setenv("ED4ALL_ASSISTANT_SEAT_PRIORITY", "absent-seat")
    monkeypatch.setenv("ED4ALL_ASSISTANT_SEAT", "cloudy")
    monkeypatch.setenv(
        "ED4ALL_SEAT_BASE_URLS", "cloudy=http://10.0.0.5:8001/v1"
    )
    with pytest.raises(AssistantProviderNotLocal) as exc:
        resolve_active_seat(force=True, probe=lambda u: True, model_reader=lambda u: "m")
    assert "cloudy" in str(exc.value)


# --------------------------------------------------------------------------- #
# TTL cache
# --------------------------------------------------------------------------- #


def test_ttl_cache_returns_cached_and_force_bypasses():
    calls = {"n": 0}

    def probe_super(url):
        calls["n"] += 1
        return url == SUPER_URL

    first = resolve_active_seat(
        force=True, probe=probe_super, model_reader=_reader({SUPER_URL: "m1"})
    )
    n_after_first = calls["n"]

    # Cache hit within TTL: probe is NOT consulted again, same object returned.
    second = resolve_active_seat(
        probe=probe_super, model_reader=_reader({SUPER_URL: "m1"})
    )
    assert second is first
    assert calls["n"] == n_after_first

    # force=True bypasses the cache and re-resolves against the new probe.
    third = resolve_active_seat(
        force=True,
        probe=lambda url: url == NANO_URL,
        model_reader=_reader({NANO_URL: "m2"}),
    )
    assert third is not first
    assert third.seat_name == "spark-nano"
    assert third.model == "m2"


def test_reset_seat_cache_clears():
    calls = {"n": 0}

    def probe(url):
        calls["n"] += 1
        return url == SUPER_URL

    resolve_active_seat(force=True, probe=probe, model_reader=_reader({SUPER_URL: "m"}))
    n_after_first = calls["n"]
    reset_seat_cache()
    # After a reset, a plain (non-force) call re-resolves → probe consulted again.
    resolve_active_seat(probe=probe, model_reader=_reader({SUPER_URL: "m"}))
    assert calls["n"] > n_after_first


# --------------------------------------------------------------------------- #
# Dynamic vs static AssistantClient
# --------------------------------------------------------------------------- #


def _final_body(content="hi"):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_dynamic_client_reresolves_per_chat_and_stamps_last_seat(monkeypatch):
    # Default probe/model_reader are the module functions the dynamic chat path
    # uses (no injected callables there) — stub them, no network.
    monkeypatch.setattr(
        "lib.assistant.client._default_probe", lambda url: url == SUPER_URL
    )
    monkeypatch.setattr(
        "lib.assistant.client._default_model_reader",
        lambda url: "super-served" if url == SUPER_URL else None,
    )

    wire = {}

    def transport(url, payload, timeout):
        wire["url"] = url
        wire["model"] = payload["model"]
        return _final_body()

    client = AssistantClient(transport=transport)
    assert client.dynamic is True
    assert client.last_seat is None

    client.chat([{"role": "user", "content": "hello"}])

    assert isinstance(client.last_seat, ResolvedSeat)
    assert client.last_seat.seat_name == "spark-super"
    assert client.base_url == SUPER_URL
    assert client.model == "super-served"
    assert wire["url"] == f"{SUPER_URL}/chat/completions"
    assert wire["model"] == "super-served"


def test_static_client_never_probes_and_last_seat_stays_none(monkeypatch):
    def _explode(url):  # pragma: no cover - asserts it is never called
        raise AssertionError("static client must not probe")

    monkeypatch.setattr("lib.assistant.client._default_probe", _explode)
    monkeypatch.setattr("lib.assistant.client._default_model_reader", _explode)

    wire = {}

    def transport(url, payload, timeout):
        wire["url"] = url
        wire["model"] = payload["model"]
        return _final_body()

    client = AssistantClient(
        base_url="http://127.0.0.1:7000/v1", model="pinned-model", transport=transport
    )
    assert client.dynamic is False

    client.chat([{"role": "user", "content": "hello"}])

    assert client.last_seat is None
    assert wire["url"] == "http://127.0.0.1:7000/v1/chat/completions"
    assert wire["model"] == "pinned-model"
