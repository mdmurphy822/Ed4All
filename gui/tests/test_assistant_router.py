"""Endpoint + settings tests for the Studio Assistant chat panel.

The engine is mocked (``assistant_service._build_engine`` monkeypatched) so no
seat, network, or model is ever touched — these tests exercise the GUI adapter
contract only: status-code mapping (seat down → 503 with start instructions,
non-loopback config → 500), the stateless history round-trip, the
``assistant.autostart`` setting plumbing, the ``assistant``-sourced activity
event, the "Seat model" seat-start contract (refusals, single-flight,
background start), and the served panel assets. Skipped entirely without the
``gui`` extra.

Every seat test is HERMETIC: the seat registry, the liveness probe, the
launch-spec lookup, and the lifecycle coherent-start call are all monkeypatched
— no docker, no subprocess, no ``/v1/models`` request is ever made.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402
from gui.services import assistant_service  # noqa: E402
from lib.assistant import (  # noqa: E402
    AssistantProviderNotLocal,
    AssistantSeatUnavailable,
    AssistantTurn,
)
from lib.assistant.debug_context import DebugContextUnavailable  # noqa: E402

SEAT_HINT = (
    "The assistant seat at http://localhost:8004/v1 is not serving. "
    "Start the nano vLLM seat (registry name 'spark-nano')."
)

DEBUG_RUN_ID = "WF-20260720-abcd1234"

#: What ``lib.assistant.debug_context.build_debug_context`` returns (subset).
DEBUG_CTX = {
    "run_id": DEBUG_RUN_ID,
    "source": "newest_failed",
    "slug": "demo-course",
    "failed_phase": "course_planning",
    "summary": f"Run {DEBUG_RUN_ID} (course demo-course) FAILED in phase course_planning.",
    "text": "=== FAILED RUN DEBUG CONTEXT (newest_failed) ===\nRun report body.",
}


class _FakeClient:
    base_url = "http://localhost:8004/v1"
    model = "nemotron-3-nano"


class _FakeEngine:
    """Engine double recording ``ensure_seat`` / ``run_turn`` calls."""

    def __init__(self, turn=None, seat_error=None):
        self.client = _FakeClient()
        self.turn = turn
        self.seat_error = seat_error
        self.ensure_calls: list = []
        self.run_calls: list = []

    def ensure_seat(self, *, allow_autostart=True):
        self.ensure_calls.append(allow_autostart)
        if self.seat_error is not None:
            raise self.seat_error

    def run_turn(self, user_text, history=None):
        self.run_calls.append((user_text, history))
        return self.turn


def _turn(
    reply="Seat spark-nano is serving.",
    tool_calls=None,
    *,
    model="nemotron-3-nano",
    seat_name="spark-nano",
):
    tool_calls = tool_calls if tool_calls is not None else []
    turn = AssistantTurn(
        reply=reply,
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": reply},
        ],
        tool_calls=tool_calls,
        rounds=1 + len(tool_calls),
        prompt_tokens=12,
        completion_tokens=7,
    )
    # S2 appends ``seat_name`` / ``model`` to ``AssistantTurn``; set them as
    # attributes so the stub carries the metadata whether or not that field
    # addition has landed (the service reads them via ``getattr(..., None)``).
    turn.seat_name = seat_name
    turn.model = model
    return turn


def _seat(*, live, model, seat_name):
    """A fake ``ResolvedSeat`` (S1) for the ``resolve_active_seat`` monkeypatch."""
    return SimpleNamespace(live=live, model=model, seat_name=seat_name)


@pytest.fixture
def install_engine(monkeypatch):
    """Install a fake engine factory; the cached engine is reset either side."""

    def _install(engine):
        monkeypatch.setattr(assistant_service, "_build_engine", lambda: engine)
        assistant_service.reset_engine()
        return engine

    assistant_service.reset_engine()
    yield _install
    assistant_service.reset_engine()


@pytest.fixture
def client(state_dir):
    """A TestClient over a fresh full-mode app with isolated state."""
    return TestClient(create_app())


# ------------------------------------------------------------------ POST /chat


def test_chat_happy_path_round_trips_history(client, install_engine):
    engine = install_engine(
        _FakeEngine(
            turn=_turn(
                tool_calls=[
                    {"tool": "seat_status", "arguments": "{}", "result": "serving"},
                ]
            )
        )
    )
    resp = client.post("/api/assistant/chat", json={"message": "Are the seats up?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reply"] == "Seat spark-nano is serving."
    assert body["rounds"] == 2
    assert body["tool_calls"][0]["tool"] == "seat_status"
    assert body["history"][-1] == {"role": "assistant", "content": body["reply"]}
    # The seat/model that answered round-trip from the turn (S11).
    assert body["model"] == "nemotron-3-nano"
    assert body["seat"] == "spark-nano"

    # Round-trip: the returned history is the next request's history, verbatim.
    resp2 = client.post(
        "/api/assistant/chat",
        json={"message": "And the campaign?", "history": body["history"]},
    )
    assert resp2.status_code == 200
    assert engine.run_calls[1] == ("And the campaign?", body["history"])


def test_chat_round_trips_distinct_seat_metadata(client, install_engine):
    """model/seat come from the turn, not a hardcoded default (S11)."""
    install_engine(
        _FakeEngine(turn=_turn(model="spark-super-120b", seat_name="spark-super"))
    )
    body = client.post("/api/assistant/chat", json={"message": "hi"}).json()
    assert body["model"] == "spark-super-120b"
    assert body["seat"] == "spark-super"


def test_chat_absent_turn_metadata_is_null(client, install_engine):
    """A stubbed turn without seat_name/model still works (getattr → None)."""
    bare = AssistantTurn(
        reply="ok",
        messages=[{"role": "assistant", "content": "ok"}],
        tool_calls=[],
        rounds=1,
    )
    install_engine(_FakeEngine(turn=bare))
    body = client.post("/api/assistant/chat", json={"message": "hi"}).json()
    assert body["model"] is None
    assert body["seat"] is None


def test_chat_seat_down_maps_503_with_instructions(client, install_engine):
    install_engine(_FakeEngine(seat_error=AssistantSeatUnavailable(SEAT_HINT)))
    resp = client.post("/api/assistant/chat", json={"message": "hello"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "assistant_seat_unavailable"
    # The 503 detail IS the operator-facing start instructions.
    assert "Start the nano vLLM seat" in body["detail"]


def test_chat_provider_not_local_maps_500(client, install_engine, monkeypatch):
    def _boom():
        raise AssistantProviderNotLocal("non-loopback base_url")

    monkeypatch.setattr(assistant_service, "_build_engine", _boom)
    assistant_service.reset_engine()
    resp = client.post("/api/assistant/chat", json={"message": "hello"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "assistant_config_error"
    assert "non-loopback" in body["detail"]


def test_chat_empty_message_is_422(client, install_engine):
    install_engine(_FakeEngine(turn=_turn()))
    resp = client.post("/api/assistant/chat", json={"message": "   "})
    assert resp.status_code == 422
    assert resp.json()["error"] == "empty_message"


def test_chat_autostart_setting_governs_ensure_seat(client, install_engine):
    from gui import settings_store

    engine = install_engine(_FakeEngine(turn=_turn()))
    # Default (no setting saved) → the engine is NOT allowed to autostart.
    client.post("/api/assistant/chat", json={"message": "status?"})
    assert engine.ensure_calls == [False]

    # assistant.autostart=true (the gui_set_setting deep-patch path) → allowed.
    settings_store.update_settings({"assistant": {"autostart": True}})
    client.post("/api/assistant/chat", json={"message": "status?"})
    assert engine.ensure_calls == [False, True]


def test_chat_appends_assistant_activity_event(client, install_engine):
    from gui import shared_state

    install_engine(
        _FakeEngine(
            turn=_turn(
                tool_calls=[
                    {"tool": "campaign_status", "arguments": "{}", "result": "idle"},
                ]
            )
        )
    )
    resp = client.post("/api/assistant/chat", json={"message": "Campaign status?"})
    assert resp.status_code == 200
    events = [e for e in shared_state.read_events() if e["source"] == "assistant"]
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "assistant_exchange"
    assert event["payload"]["question"] == "Campaign status?"
    assert event["payload"]["tools"] == ["campaign_status"]
    assert "Campaign status?" in event["payload"]["summary"]
    # And the bridge read path (what gui_read_events / the Activity tab see)
    # returns it alongside gui/claude events.
    shared_state.append_event("gui", "note", {"text": "hi"})
    sources = {e["source"] for e in shared_state.read_events()}
    assert {"assistant", "gui"} <= sources


# -------------------------------------------------------- debug mode (contexts)


@pytest.fixture
def install_debug_context(monkeypatch, install_engine):
    """Patch ``build_debug_context`` on the service; returns the call list.

    Piggybacks on ``install_engine`` so the debug caches are reset either side
    (``reset_engine`` clears them too).
    """
    install_engine(_FakeEngine(turn=_turn()))
    calls: list = []

    def _install(result=None, error=None):
        def _fake_build(run_id=None):
            calls.append(run_id)
            if error is not None:
                raise error
            return dict(result if result is not None else DEBUG_CTX)

        monkeypatch.setattr(assistant_service, "build_debug_context", _fake_build)
        return calls

    return _install


def test_debug_context_probe_available(client, install_debug_context):
    calls = install_debug_context()
    resp = client.get("/api/assistant/debug-context")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "available": True,
        "run_id": DEBUG_RUN_ID,
        "course": "demo-course",
        "failed_phase": "course_planning",
        "summary": DEBUG_CTX["summary"],
    }
    assert calls == [None]  # auto-resolve (no run_id given)
    # A second probe reuses the cached context — the (slow) builder ran once.
    assert client.get("/api/assistant/debug-context").status_code == 200
    assert calls == [None]


def test_debug_context_probe_explicit_run_id(client, install_debug_context):
    calls = install_debug_context()
    resp = client.get(f"/api/assistant/debug-context?run_id={DEBUG_RUN_ID}")
    assert resp.status_code == 200
    assert resp.json()["available"] is True
    assert calls == [DEBUG_RUN_ID]


def test_debug_context_probe_none_available(client, install_debug_context):
    install_debug_context(error=DebugContextUnavailable("No failed run found."))
    resp = client.get("/api/assistant/debug-context")
    assert resp.status_code == 200  # 404-free contract
    body = resp.json()
    assert body["available"] is False
    assert body["summary"] is None
    assert "No failed run" in body["reason"]


def test_debug_context_probe_bad_run_id_422(client, install_debug_context):
    calls = install_debug_context()
    resp = client.get("/api/assistant/debug-context?run_id=not-a-run-id")
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_run_id"
    assert calls == []  # garbage never reaches the debug machinery


# ------------------------------------------------------------ POST /chat (debug)


@pytest.fixture
def install_debug_engine(monkeypatch):
    """Patch ``AssistantEngine`` on the service with a kwargs-recording fake."""
    created: list = []

    def _fake_engine_cls(**kwargs):
        engine = _FakeEngine(turn=_turn(reply="Root cause: the planning gate failed."))
        engine.init_kwargs = kwargs
        created.append(engine)
        return engine

    monkeypatch.setattr(assistant_service, "AssistantEngine", _fake_engine_cls)
    return created


def test_chat_debug_happy_path(client, install_debug_context, install_debug_engine):
    build_calls = install_debug_context()
    created = install_debug_engine
    resp = client.post(
        "/api/assistant/chat",
        json={"message": "Why did it fail?", "debug": True, "run_id": DEBUG_RUN_ID},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reply"] == "Root cause: the planning gate failed."
    assert body["debug"] == {
        "run_id": DEBUG_RUN_ID,
        "failed_phase": "course_planning",
        "summary": DEBUG_CTX["summary"],
    }
    # The engine was constructed in DEBUG mode with the built context threaded.
    assert len(created) == 1
    assert created[0].init_kwargs == {
        "mode": "debug",
        "debug_context": DEBUG_CTX["text"],
    }
    assert created[0].run_calls == [("Why did it fail?", None)]

    # Turn 1: history round-trips; the context+engine are reused (built once).
    resp2 = client.post(
        "/api/assistant/chat",
        json={
            "message": "What next?",
            "history": body["history"],
            "debug": True,
            "run_id": DEBUG_RUN_ID,
        },
    )
    assert resp2.status_code == 200
    assert len(created) == 1 and build_calls == [DEBUG_RUN_ID]
    assert created[0].run_calls[1] == ("What next?", body["history"])


def test_chat_debug_unavailable_maps_409(client, install_debug_context, install_debug_engine):
    install_debug_context(error=DebugContextUnavailable("No failed run found."))
    resp = client.post("/api/assistant/chat", json={"message": "debug it", "debug": True})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "debug_context_unavailable"
    assert "No failed run" in body["detail"]


def test_chat_debug_bad_run_id_422(client, install_debug_context, install_debug_engine):
    calls = install_debug_context()
    resp = client.post(
        "/api/assistant/chat",
        json={"message": "debug it", "debug": True, "run_id": "WF-garbage"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_run_id"
    assert calls == []


def test_chat_non_debug_ignores_run_id(client, install_engine):
    """A plain chat with debug unset never touches the debug machinery."""
    engine = install_engine(_FakeEngine(turn=_turn()))
    resp = client.post("/api/assistant/chat", json={"message": "hello"})
    assert resp.status_code == 200
    assert "debug" not in resp.json()
    assert engine.run_calls == [("hello", None)]


# ------------------------------------------------------------------ GET /status


def test_status_seat_live(client, install_engine, monkeypatch):
    """A live resolved seat → serving + its model + seat name (S11)."""
    install_engine(_FakeEngine(turn=_turn()))
    monkeypatch.setattr(
        assistant_service,
        "resolve_active_seat",
        lambda *a, **k: _seat(live=True, model="spark-super-120b", seat_name="spark-super"),
    )
    resp = client.get("/api/assistant/status")
    assert resp.status_code == 200
    assert resp.json() == {
        "seat_serving": True,
        "model": "spark-super-120b",
        "seat": "spark-super",
    }


def test_status_seat_down(client, install_engine, monkeypatch):
    """A not-live resolved seat → not serving, model null, seat still named."""
    install_engine(_FakeEngine(turn=_turn()))
    monkeypatch.setattr(
        assistant_service,
        "resolve_active_seat",
        lambda *a, **k: _seat(live=False, model="nemotron-3-nano", seat_name="spark-nano"),
    )
    resp = client.get("/api/assistant/status")
    assert resp.status_code == 200
    assert resp.json() == {
        "seat_serving": False,
        "model": None,
        "seat": "spark-nano",
    }


def test_status_config_error_reports_not_serving(client, install_engine, monkeypatch):
    """Non-loopback engine construction → detail path, seat/model null."""

    def _boom():
        raise AssistantProviderNotLocal("non-loopback base_url")

    monkeypatch.setattr(assistant_service, "_build_engine", _boom)
    assistant_service.reset_engine()
    resp = client.get("/api/assistant/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["seat_serving"] is False
    assert body["model"] is None
    assert body["seat"] is None
    assert "non-loopback" in body["detail"]


def test_status_resolver_config_error_reports_not_serving(client, install_engine, monkeypatch):
    """A bad seat-registry entry (resolver raises) also lands on the detail path."""
    install_engine(_FakeEngine(turn=_turn()))

    def _boom(*a, **k):
        raise AssistantProviderNotLocal("non-loopback seat registry entry 'spark-super'")

    monkeypatch.setattr(assistant_service, "resolve_active_seat", _boom)
    resp = client.get("/api/assistant/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["seat_serving"] is False
    assert body["model"] is None
    assert body["seat"] is None
    assert "non-loopback" in body["detail"]


# ------------------------------------------------------- settings persistence


def test_default_settings_carry_assistant_section(state_dir):
    from gui import env_catalog, settings_store

    assert env_catalog.default_settings()["assistant"] == {"autostart": False}
    # A fresh (no file on disk) load carries the section too.
    assert settings_store.load_settings()["assistant"] == {"autostart": False}


def test_assistant_section_persists_and_patches(client):
    from gui import settings_store

    # HTTP deep-patch (what the GUI would send).
    resp = client.patch("/api/settings", json={"assistant": {"autostart": True}})
    assert resp.status_code == 200
    assert resp.json()["assistant"]["autostart"] is True
    assert settings_store.load_settings()["assistant"]["autostart"] is True

    # The MCP-bridge dotted path (gui_set_setting("assistant.autostart", ...))
    # builds exactly this nested patch through update_settings.
    stored = settings_store.update_settings({"assistant": {"autostart": False}})
    assert stored["assistant"]["autostart"] is False
    assert settings_store.load_settings()["assistant"]["autostart"] is False


# -------------------------------------------------- "Seat model" (POST /seat)

#: The hermetic seat topology every seat test starts from: the priority walk
#: (large seat, then the assistant's own small seat) and their registry URLs.
SEAT_PRIORITY = ["spark-super", "spark-nano"]
OWN_SEAT = "spark-nano"
SEAT_REGISTRY = {
    "spark-super": "http://localhost:8001",
    "spark-nano": "http://localhost:8004",
}


@pytest.fixture
def seat_env(monkeypatch, install_engine):
    """Install a hermetic seat topology; returns a knob object.

    ``knobs.live`` is the set of live seat names, ``knobs.launch_specs`` the
    ``ED4ALL_SEAT_LAUNCH_SPECS`` map, ``knobs.start_calls`` records every
    lifecycle coherent-start call, and ``knobs.start_result`` / ``start_gate``
    control what that call returns / when it returns.
    """
    install_engine(_FakeEngine(turn=_turn()))
    assistant_service.reset_seat_start_state()

    knobs = SimpleNamespace(
        live=set(),
        launch_specs={OWN_SEAT: "/opt/seats/launch-nano.sh"},
        start_calls=[],
        start_result=SimpleNamespace(ok=True, reason="warm_start_coherent"),
        start_gate=None,  # optional threading.Event the worker waits on
    )

    def _resolve_active(*_a, **_k):
        for name in SEAT_PRIORITY:
            if name in knobs.live:
                return _seat(live=True, model=f"{name}-model", seat_name=name)
        return _seat(live=False, model="nemotron-3-nano", seat_name=OWN_SEAT)

    def _autostart(*_a, **_k):
        knobs.start_calls.append(OWN_SEAT)
        if knobs.start_gate is not None:
            knobs.start_gate.wait(timeout=5)
        return knobs.start_result

    monkeypatch.setattr(assistant_service, "resolve_active_seat", _resolve_active)
    monkeypatch.setattr(assistant_service, "parse_seat_registry", lambda *a, **k: dict(SEAT_REGISTRY))
    monkeypatch.setattr(assistant_service, "resolve_assistant_seat_priority", lambda *a, **k: list(SEAT_PRIORITY))
    monkeypatch.setattr(assistant_service, "resolve_assistant_seat", lambda *a, **k: OWN_SEAT)
    monkeypatch.setattr(
        assistant_service,
        "seat_is_serving",
        lambda base_url: any(SEAT_REGISTRY.get(n) == base_url for n in knobs.live),
    )
    monkeypatch.setattr(
        assistant_service,
        "resolve_seat_launch_spec",
        lambda seat, **k: knobs.launch_specs.get(seat),
    )
    monkeypatch.setattr(assistant_service, "autostart_seat", _autostart)
    monkeypatch.setattr(assistant_service, "reset_seat_cache", lambda: None)

    yield knobs

    gate = knobs.start_gate
    if gate is not None:
        gate.set()
    _join_seat_start()
    assistant_service.reset_seat_start_state()


def _join_seat_start(timeout: float = 5.0) -> None:
    """Join the background seat-start worker (tests stay deterministic)."""
    thread = assistant_service._SEAT_START.get("thread")
    if isinstance(thread, threading.Thread):
        thread.join(timeout=timeout)


def test_seat_status_shape_no_seat_live(client, seat_env):
    resp = client.get("/api/assistant/seat/status")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "live_seat": None,
        "seats": {"spark-super": False, "spark-nano": False},
        "starting": False,
        "last_seat_error": None,
    }


def test_seat_status_reports_live_seat(client, seat_env):
    seat_env.live = {"spark-super"}
    body = client.get("/api/assistant/seat/status").json()
    assert body["live_seat"] == "spark-super"
    assert body["seats"] == {"spark-super": True, "spark-nano": False}


def test_seat_start_refuses_when_super_live(client, seat_env):
    seat_env.live = {"spark-super"}
    resp = client.post("/api/assistant/seat")
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["error"] == "assistant_seat_already_live"
    assert "spark-super" in body["detail"]
    # REFUSED with NO side effect — the lifecycle was never called.
    assert seat_env.start_calls == []


def test_seat_start_refuses_when_nano_live(client, seat_env):
    seat_env.live = {"spark-nano"}
    resp = client.post("/api/assistant/seat")
    assert resp.status_code == 409
    assert "spark-nano" in resp.json()["detail"]
    assert seat_env.start_calls == []


def test_seat_start_refuses_without_launch_spec(client, seat_env):
    """No ED4ALL_SEAT_LAUNCH_SPECS entry → the registry error, verbatim."""
    seat_env.launch_specs = {}
    resp = client.post("/api/assistant/seat")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "assistant_seat_no_launch_spec"
    assert "ED4ALL_SEAT_LAUNCH_SPECS" in body["detail"]
    assert "spark-nano" in body["detail"]
    assert seat_env.start_calls == []


def test_seat_start_happy_path_starts_in_background(client, seat_env):
    resp = client.post("/api/assistant/seat")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "ok": True,
        "state": "starting",
        "seat": "spark-nano",
        "error": None,
    }
    _join_seat_start()
    # The lifecycle coherent-start path ran exactly once for the resolved seat
    # (never a script path chosen by the GUI).
    assert seat_env.start_calls == ["spark-nano"]
    # A successful start leaves no error behind.
    assert client.get("/api/assistant/seat/status").json()["last_seat_error"] is None


def test_seat_start_does_not_require_autostart_flag(client, seat_env, monkeypatch):
    """The explicit human action ignores ED4ALL_ASSISTANT_AUTOSTART."""
    monkeypatch.delenv("ED4ALL_ASSISTANT_AUTOSTART", raising=False)
    assert client.post("/api/assistant/seat").status_code == 200
    _join_seat_start()
    assert seat_env.start_calls == ["spark-nano"]


def test_seat_start_is_single_flight(client, seat_env):
    """A second click while starting observes the in-flight start, never a 2nd."""
    seat_env.start_gate = threading.Event()
    first = client.post("/api/assistant/seat")
    assert first.status_code == 200

    status = client.get("/api/assistant/seat/status").json()
    assert status["starting"] is True

    second = client.post("/api/assistant/seat")
    assert second.status_code == 200
    assert second.json() == {
        "ok": True,
        "state": "starting",
        "seat": "spark-nano",
        "error": None,
    }
    assert seat_env.start_calls == ["spark-nano"]  # still exactly ONE start

    seat_env.start_gate.set()
    _join_seat_start()
    assert client.get("/api/assistant/seat/status").json()["starting"] is False


def test_seat_start_failure_surfaces_on_next_status(client, seat_env):
    seat_env.start_result = SimpleNamespace(ok=False, reason="warm_incoherent_no_spec")
    assert client.post("/api/assistant/seat").status_code == 200
    _join_seat_start()
    body = client.get("/api/assistant/seat/status").json()
    assert body["starting"] is False
    assert "warm_incoherent_no_spec" in body["last_seat_error"]
    # ...and the failed start does not block a retry.
    assert client.post("/api/assistant/seat").status_code == 200
    _join_seat_start()
    assert seat_env.start_calls == ["spark-nano", "spark-nano"]


def test_seat_start_lifecycle_exception_is_recorded_not_raised(client, seat_env, monkeypatch):
    def _boom(*_a, **_k):
        seat_env.start_calls.append(OWN_SEAT)
        raise RuntimeError("docker daemon unreachable")

    monkeypatch.setattr(assistant_service, "autostart_seat", _boom)
    assert client.post("/api/assistant/seat").status_code == 200
    _join_seat_start()
    body = client.get("/api/assistant/seat/status").json()
    assert body["starting"] is False
    assert "docker daemon unreachable" in body["last_seat_error"]


def test_seat_status_config_error_reports_detail(client, install_engine, monkeypatch):
    """A non-loopback seat config → detail, no seats, and a refused start."""
    install_engine(_FakeEngine(turn=_turn()))
    assistant_service.reset_seat_start_state()

    def _boom(*a, **k):
        raise AssistantProviderNotLocal("non-loopback seat registry entry 'spark-super'")

    monkeypatch.setattr(assistant_service, "resolve_active_seat", _boom)
    body = client.get("/api/assistant/seat/status").json()
    assert body["live_seat"] is None
    assert body["seats"] == {}
    assert "non-loopback" in body["detail"]

    resp = client.post("/api/assistant/seat")
    assert resp.status_code == 500
    assert resp.json()["error"] == "assistant_config_error"


def test_seat_endpoints_stay_open_under_the_token_gate(state_dir, seat_env, monkeypatch):
    """Auth posture matches the other assistant endpoints (deliberately open)."""
    from gui import auth

    monkeypatch.setenv(auth.TOKEN_ENV, "s3cret-token")
    monkeypatch.setattr("gui.auth._settings_token", lambda: None)
    gated = TestClient(create_app())
    assert gated.get("/api/assistant/seat/status").status_code == 200
    assert gated.post("/api/assistant/seat").status_code == 200
    _join_seat_start()


def test_seat_endpoints_mounted_in_studio_mode(state_dir, seat_env):
    studio_client = TestClient(create_app(mode="studio"))
    assert studio_client.get("/api/assistant/seat/status").status_code == 200


def test_learner_mode_does_not_mount_seat_endpoints(state_dir):
    learner_client = TestClient(create_app(mode="learner"))
    assert learner_client.get("/api/assistant/seat/status").status_code == 404
    # POST answers 405 (the learner static mount owns the path for GET only) —
    # either way the assistant seat route is NOT reachable in learner mode.
    assert learner_client.post("/api/assistant/seat").status_code in (404, 405)


# ---------------------------------------------------------- mounts + assets


def test_studio_mode_mounts_assistant(state_dir, install_engine, monkeypatch):
    install_engine(_FakeEngine(turn=_turn()))
    monkeypatch.setattr(
        assistant_service,
        "resolve_active_seat",
        lambda *a, **k: _seat(live=True, model="nemotron-3-nano", seat_name="spark-nano"),
    )
    studio_client = TestClient(create_app(mode="studio"))
    resp = studio_client.get("/api/assistant/status")
    assert resp.status_code == 200
    assert resp.json()["seat_serving"] is True


def test_learner_mode_does_not_mount_assistant(state_dir):
    learner_client = TestClient(create_app(mode="learner"))
    resp = learner_client.get("/api/assistant/status")
    assert resp.status_code == 404


def test_assistant_panel_assets_served(client):
    # The panel module is served and exports the renderer.
    js = client.get("/studio/assistant.js")
    assert js.status_code == 200
    assert "renderAssistant" in js.text
    assert "/api/assistant/chat" in js.text
    # The debug-last-failure flow: probe endpoint + button + banner bubble.
    assert "/api/assistant/debug-context" in js.text
    assert "Debug last failure" in js.text
    assert "chat-banner" in js.text
    # The per-reply seat/model badge (S11).
    assert "assistant-seat-badge" in js.text
    # The "Seat model" control: real <button>, both seat endpoints, the busy
    # copy, and the poll interval (no color-only signal anywhere).
    assert "Seat model" in js.text
    assert "/api/assistant/seat/status" in js.text
    assert "apiJSON('/api/assistant/seat', 'POST'" in js.text
    assert "Seating nano…" in js.text
    assert "aria-busy" in js.text
    # ...and its styling ships in the panel stylesheet.
    css = client.get("/studio/studio.css")
    assert css.status_code == 200
    assert ".assistant-seat-btn" in css.text
    # The Studio shell links the route + the router wires it.
    shell = client.get("/studio/")
    assert shell.status_code == 200
    assert "#/assistant" in shell.text
    studio_js = client.get("/studio/studio.js")
    assert studio_js.status_code == 200
    assert "/studio/assistant.js" in studio_js.text
