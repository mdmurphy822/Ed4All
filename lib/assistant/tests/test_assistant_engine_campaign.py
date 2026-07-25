"""Campaign-mode engine tests — the S2 contract exercised directly:

* campaign mode sends ``[*TOOL_SCHEMAS, *CAMPAIGN_TOOL_SCHEMAS]`` on the wire;
* a campaign-registered tool name routes to ``dispatch_campaign_tool`` in
  campaign mode, a base name still hits ``dispatch_tool``, and a campaign
  name in operator mode falls to ``dispatch_tool`` (unreachable → refused);
* ``CAMPAIGN_SYSTEM_PROMPT`` is the wire system prompt in campaign mode and
  states the load-bearing division of labor;
* ``AssistantTurn`` carries the resolved ``seat_name`` / ``model`` (dynamic
  client) and ``None`` for a static client;
* ``ensure_seat`` honours the dynamic-seat + autostart policy;
* the DecisionCapture regression net fires for a campaign turn (real
  ``DecisionCapture``), interpolating ``mode=campaign`` + the resolved seat.

All hermetic: fake client (no transport / network), campaign symbols on the
engine module monkeypatched to sentinels, no real seats, no campaign_tools
dependency.
"""

from __future__ import annotations

import json

import pytest

import lib.assistant.engine as engine_mod
from lib.assistant.engine import (
    CAMPAIGN_SYSTEM_PROMPT,
    AssistantEngine,
)
from lib.assistant.tools import TOOL_SCHEMAS


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def _tool_call_body(name, arguments, call_id="call_1"):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    }


def _final_body(content):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 12},
    }


class _FakeSeat:
    def __init__(self, seat_name="spark-super"):
        self.seat_name = seat_name


class _FakeClient:
    """A hermetic AssistantClient stand-in (no transport / network).

    ``last_seat``/``model`` model the two client states: a dynamically
    resolved client (``last_seat`` set, ``model`` populated) vs a static one
    (no ``last_seat``, no ``model`` attribute → both surface as ``None``).
    """

    def __init__(self, bodies, *, model="super-served-xyz", last_seat=None, has_model=True):
        self._queue = list(bodies)
        self.calls = []
        self.base_url = "http://localhost:8004/v1"
        self.max_tokens = 256
        self.last_seat = last_seat
        if has_model:
            self.model = model

    def chat(self, messages, *, tools=None, temperature=0.2):
        self.calls.append({"messages": messages, "tools": tools})
        if not self._queue:
            raise AssertionError("fake client exhausted — unexpected extra call")
        return self._queue.pop(0)


class _RecordingCapture:
    def __init__(self):
        self.decisions = []

    def log_decision(self, **kwargs):
        self.decisions.append(kwargs)


# --------------------------------------------------------------------------- #
# Wire tools: campaign mode merges the campaign schema list
# --------------------------------------------------------------------------- #


def test_campaign_mode_sends_merged_tool_schemas(monkeypatch):
    sentinel = {"type": "function", "function": {"name": "campaign_queue"}}
    monkeypatch.setattr(engine_mod, "CAMPAIGN_TOOL_SCHEMAS", [sentinel])

    client = _FakeClient([_final_body("queue is clear")])
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="campaign")

    engine.run_turn("how is the campaign?")

    assert client.calls[0]["tools"] == [*TOOL_SCHEMAS, sentinel]


def test_operator_mode_sends_only_base_tool_schemas(monkeypatch):
    sentinel = {"type": "function", "function": {"name": "campaign_queue"}}
    monkeypatch.setattr(engine_mod, "CAMPAIGN_TOOL_SCHEMAS", [sentinel])

    client = _FakeClient([_final_body("ok")])
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="operator")

    engine.run_turn("status?")

    assert client.calls[0]["tools"] == TOOL_SCHEMAS
    assert sentinel not in client.calls[0]["tools"]


# --------------------------------------------------------------------------- #
# Dispatch routing: campaign names route to the campaign dispatcher only in
# campaign mode; base names + operator mode go to dispatch_tool
# --------------------------------------------------------------------------- #


def test_campaign_name_routes_to_campaign_dispatch(monkeypatch):
    monkeypatch.setattr(
        engine_mod, "CAMPAIGN_TOOL_REGISTRY", {"campaign_queue": lambda: "x"}
    )
    routed = {}

    def _campaign(n, a):
        routed["campaign"] = (n, a)
        return "3 pending"

    def _base(n, a):
        routed["base"] = (n, a)
        return "base"

    monkeypatch.setattr(engine_mod, "dispatch_campaign_tool", _campaign)
    monkeypatch.setattr(engine_mod, "dispatch_tool", _base)

    client = _FakeClient(
        [_tool_call_body("campaign_queue", {}), _final_body("done")]
    )
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="campaign")

    turn = engine.run_turn("organize the queue")

    assert routed["campaign"] == ("campaign_queue", {})
    assert "base" not in routed
    assert turn.tool_calls[0]["result"] == "3 pending"


def test_base_name_routes_to_base_dispatch_in_campaign_mode(monkeypatch):
    monkeypatch.setattr(
        engine_mod, "CAMPAIGN_TOOL_REGISTRY", {"campaign_queue": lambda: "x"}
    )
    routed = {}

    def _campaign(n, a, **kwargs):
        routed["campaign"] = (n, a)
        return "campaign"

    def _base(n, a, **kwargs):
        # campaign mode routes base tools with campaign_mode=True (the
        # start_seat own-seat guard); accept + record it.
        routed["base"] = (n, a)
        routed["base_kwargs"] = kwargs
        return "recent runs"

    monkeypatch.setattr(engine_mod, "dispatch_campaign_tool", _campaign)
    monkeypatch.setattr(engine_mod, "dispatch_tool", _base)

    client = _FakeClient(
        [_tool_call_body("run_status", {}), _final_body("done")]
    )
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="campaign")

    turn = engine.run_turn("what is running?")

    assert routed["base"] == ("run_status", {})
    assert routed["base_kwargs"] == {"campaign_mode": True}
    assert "campaign" not in routed
    assert turn.tool_calls[0]["result"] == "recent runs"


def test_campaign_name_in_operator_mode_hits_base_dispatch(monkeypatch):
    # A campaign-registered name is UNREACHABLE outside campaign mode: it
    # falls to the base dispatcher, whose unknown-name refusal is desired.
    monkeypatch.setattr(
        engine_mod, "CAMPAIGN_TOOL_REGISTRY", {"campaign_queue": lambda: "x"}
    )

    def _campaign_explode(n, a):  # pragma: no cover - must never fire
        raise AssertionError("campaign dispatch must not fire in operator mode")

    monkeypatch.setattr(engine_mod, "dispatch_campaign_tool", _campaign_explode)
    routed = {}

    def _base(n, a):
        routed["base"] = (n, a)
        return f"Refused: tool {n!r} is not in the whitelist."

    monkeypatch.setattr(engine_mod, "dispatch_tool", _base)

    client = _FakeClient(
        [_tool_call_body("campaign_queue", {}), _final_body("understood")]
    )
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="operator")

    turn = engine.run_turn("run a campaign tool")

    assert routed["base"] == ("campaign_queue", {})
    assert turn.tool_calls[0]["result"].startswith("Refused:")


# --------------------------------------------------------------------------- #
# campaign-tick mode: the pilot's restricted OBSERVE + REPORT surface
# --------------------------------------------------------------------------- #


def test_campaign_tick_mode_is_a_valid_mode():
    assert "campaign-tick" in AssistantEngine.MODES
    eng = AssistantEngine(client=_FakeClient([]), capture=_RecordingCapture(),
                          mode="campaign-tick")
    assert eng.mode == "campaign-tick"


def test_campaign_tick_sends_only_readonly_merged_schemas():
    client = _FakeClient([_final_body("nothing to flag")])
    engine = AssistantEngine(client=client, capture=_RecordingCapture(),
                             mode="campaign-tick")
    engine.run_turn("review the snapshot")
    expected = [*engine_mod.READONLY_TOOL_SCHEMAS,
                *engine_mod.CAMPAIGN_READONLY_TOOL_SCHEMAS]
    assert client.calls[0]["tools"] == expected
    # No mutating tool leaks onto the wire.
    names = {s["function"]["name"] for s in client.calls[0]["tools"]}
    for mut in ("start_seat", "resume_run", "stop_run", "start_book",
                "campaign_launch_run", "campaign_resume_run", "campaign_stop_run",
                "campaign_prepare_run"):
        assert mut not in names


def test_campaign_tick_system_prompt_states_observe_report():
    engine = AssistantEngine(client=_FakeClient([]), capture=_RecordingCapture(),
                             mode="campaign-tick")
    prompt = engine.system_prompt
    assert prompt == engine_mod.CAMPAIGN_TICK_SYSTEM_PROMPT
    assert "OBSERVE and REPORT ONLY" in prompt
    assert "DETERMINISTIC" in prompt


def test_campaign_tick_mutating_campaign_tool_routed_readonly(monkeypatch):
    monkeypatch.setattr(
        engine_mod, "CAMPAIGN_TOOL_REGISTRY",
        {"campaign_queue": lambda: "x", "campaign_launch_run": lambda **k: "x"},
    )
    routed = {}

    def _campaign(n, a, **kwargs):
        routed["campaign"] = (n, a, kwargs)
        return "Refused: mutating tool" if kwargs.get("readonly") else "launched"

    monkeypatch.setattr(engine_mod, "dispatch_campaign_tool", _campaign)

    client = _FakeClient(
        [_tool_call_body("campaign_launch_run", {"name": "x"}), _final_body("done")]
    )
    engine = AssistantEngine(client=client, capture=_RecordingCapture(),
                             mode="campaign-tick")
    engine.run_turn("launch a book")
    # The engine passes readonly=True in tick mode; the dispatcher enforces it.
    assert routed["campaign"] == ("campaign_launch_run", {"name": "x"},
                                  {"readonly": True})


def test_campaign_tick_base_tool_routed_readonly(monkeypatch):
    monkeypatch.setattr(
        engine_mod, "CAMPAIGN_TOOL_REGISTRY", {"campaign_queue": lambda: "x"}
    )
    routed = {}

    def _base(n, a, **kwargs):
        routed["base"] = (n, a, kwargs)
        return "recent runs"

    monkeypatch.setattr(engine_mod, "dispatch_tool", _base)

    client = _FakeClient(
        [_tool_call_body("run_status", {}), _final_body("done")]
    )
    engine = AssistantEngine(client=client, capture=_RecordingCapture(),
                             mode="campaign-tick")
    engine.run_turn("what is running?")
    assert routed["base"] == ("run_status", {}, {"readonly": True})


# --------------------------------------------------------------------------- #
# System prompt: campaign prompt on the wire + load-bearing phrases
# --------------------------------------------------------------------------- #


def test_campaign_system_prompt_on_the_wire():
    client = _FakeClient([_final_body("hi")])
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="campaign")

    engine.run_turn("hello")

    system_msg = client.calls[0]["messages"][0]
    assert system_msg["role"] == "system"
    assert system_msg["content"] == CAMPAIGN_SYSTEM_PROMPT


def test_campaign_system_prompt_states_the_division_of_labor():
    # Load-bearing phrases the owner spec requires stated explicitly.
    assert "organizes" in CAMPAIGN_SYSTEM_PROMPT
    assert "Claude" in CAMPAIGN_SYSTEM_PROMPT
    assert "never author" in CAMPAIGN_SYSTEM_PROMPT
    assert "plain resume" in CAMPAIGN_SYSTEM_PROMPT
    # And the campaign tool catalog is present.
    assert "campaign_prepare_run" in CAMPAIGN_SYSTEM_PROMPT
    assert "campaign_report" in CAMPAIGN_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Turn metadata: resolved seat_name / model
# --------------------------------------------------------------------------- #


def test_turn_carries_seat_and_model_for_dynamic_client():
    client = _FakeClient(
        [_final_body("answer")],
        model="super-served-xyz",
        last_seat=_FakeSeat("spark-super"),
    )
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="campaign")

    turn = engine.run_turn("status")

    assert turn.seat_name == "spark-super"
    assert turn.model == "super-served-xyz"
    # JSON round-trippable.
    json.dumps({"seat": turn.seat_name, "model": turn.model})


def test_turn_carries_none_seat_and_model_for_static_client():
    client = _FakeClient([_final_body("answer")], last_seat=None, has_model=False)
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="operator")

    turn = engine.run_turn("status")

    assert turn.seat_name is None
    assert turn.model is None


# --------------------------------------------------------------------------- #
# ensure_seat: dynamic resolution + autostart policy
# --------------------------------------------------------------------------- #


class _Seat:
    def __init__(self, *, live, base_url="http://localhost:8004/v1", seat_name="spark-nano"):
        self.live = live
        self.base_url = base_url
        self.seat_name = seat_name


def test_ensure_seat_live_resolved_seat_no_autostart(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_AUTOSTART", "1")
    monkeypatch.setattr(
        engine_mod,
        "resolve_active_seat",
        lambda *, force=True: _Seat(live=True, seat_name="spark-super"),
    )

    def _explode():  # pragma: no cover - must never fire
        raise AssertionError("autostart must not fire when a seat is live")

    monkeypatch.setattr(engine_mod, "autostart_seat", _explode)

    client = _FakeClient([])
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="campaign")

    engine.ensure_seat()  # must not raise, must not autostart


def test_ensure_seat_dead_autostarts_then_resets_cache(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_AUTOSTART", "true")
    monkeypatch.setattr(
        engine_mod, "resolve_active_seat", lambda *, force=True: _Seat(live=False)
    )

    class _Result:
        ok = True
        reason = "warm_start_coherent"

    fired = {}

    def _autostart():
        fired["autostart"] = True
        return _Result()

    monkeypatch.setattr(engine_mod, "autostart_seat", _autostart)
    monkeypatch.setattr(
        engine_mod, "reset_seat_cache", lambda: fired.__setitem__("reset", True)
    )

    client = _FakeClient([])
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="campaign")

    engine.ensure_seat()
    assert fired.get("autostart") is True
    assert fired.get("reset") is True


def test_ensure_seat_dead_and_autostart_disabled_raises(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSISTANT_AUTOSTART", raising=False)
    monkeypatch.setattr(
        engine_mod, "resolve_active_seat", lambda *, force=True: _Seat(live=False)
    )

    from lib.assistant.client import AssistantSeatUnavailable

    client = _FakeClient([])
    engine = AssistantEngine(client=client, capture=_RecordingCapture(), mode="campaign")

    with pytest.raises(AssistantSeatUnavailable):
        engine.ensure_seat()


# --------------------------------------------------------------------------- #
# DecisionCapture regression net for a campaign turn (real DecisionCapture)
# --------------------------------------------------------------------------- #


def test_capture_fires_for_campaign_turn_real_capture():
    from lib.decision_capture import DecisionCapture

    client = _FakeClient(
        [_final_body("queue reviewed")],
        model="super-served-xyz",
        last_seat=_FakeSeat("spark-super"),
    )
    capture = DecisionCapture(
        course_code="assistant", phase="assistant_session", tool="assistant"
    )
    try:
        engine = AssistantEngine(client=client, capture=capture, mode="campaign")
        engine.run_turn("review the campaign snapshot")

        assert len(capture.decisions) == 1
        record = capture.decisions[0]
        assert record["decision_type"] == "llm_chat_call"
        assert len(record["rationale"]) >= 20
        # Dynamic rationale interpolates the mode + the resolved seat.
        assert "mode=campaign" in record["rationale"]
        assert "spark-super" in record["rationale"]
    finally:
        capture.close()
