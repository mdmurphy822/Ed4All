"""Engine tests — the UI-agnostic chat/tool loop exercised directly (no
CLI): message in → tool dispatched through the whitelist → reply out; the
DecisionCapture regression net (capture MUST fire on the LLM call path);
the 6-round tool-loop cap; serializable caller-held conversation state."""

from __future__ import annotations

import json

import pytest

from lib.assistant.client import AssistantClient
from lib.assistant.engine import MAX_TOOL_ROUNDS, AssistantEngine


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


def _scripted_client(bodies):
    """AssistantClient whose transport plays back ``bodies`` in order.

    Constructed STATIC (explicit loopback base_url + model) so it never
    triggers dynamic seat resolution / a real liveness probe — the tool-loop
    and capture tests stay hermetic (no network) independent of the dynamic
    seat-resolution client path.
    """
    queue = list(bodies)
    calls = []

    def transport(url, payload, timeout):
        calls.append(payload)
        if not queue:
            raise AssertionError("transport exhausted — unexpected extra call")
        return queue.pop(0)

    client = AssistantClient(
        base_url="http://localhost:8004/v1",
        model="nemotron-3-nano",
        transport=transport,
    )
    return client, calls


class _FakeSeat:
    """Minimal S1-shaped ResolvedSeat stand-in for engine seat tests."""

    def __init__(self, *, live, base_url="http://localhost:8004/v1", seat_name="spark-nano"):
        self.live = live
        self.base_url = base_url
        self.seat_name = seat_name


class _RecordingCapture:
    def __init__(self):
        self.decisions = []

    def log_decision(self, **kwargs):
        self.decisions.append(kwargs)


# --------------------------------------------------------------------------- #
# Message in → tool dispatched → reply out (the GUI-consumer contract)
# --------------------------------------------------------------------------- #


def test_turn_with_tool_call_dispatches_and_replies(monkeypatch):
    dispatched = {}

    def fake_dispatch(name, arguments):
        dispatched["name"] = name
        dispatched["arguments"] = arguments
        return "3 book(s) pending"

    monkeypatch.setattr("lib.assistant.engine.dispatch_tool", fake_dispatch)
    client, wire_calls = _scripted_client(
        [
            _tool_call_body("campaign_status", {}),
            _final_body("Three books are pending."),
        ]
    )
    engine = AssistantEngine(client=client, capture=_RecordingCapture())

    turn = engine.run_turn("how is the campaign going?")

    assert dispatched["name"] == "campaign_status"
    assert turn.reply == "Three books are pending."
    assert turn.rounds == 2
    assert [t["tool"] for t in turn.tool_calls] == ["campaign_status"]
    assert turn.tool_calls[0]["result"] == "3 book(s) pending"
    # The tool result went back to the model as a role=tool message.
    second_wire = wire_calls[1]["messages"]
    assert any(
        m.get("role") == "tool" and m.get("content") == "3 book(s) pending"
        for m in second_wire
    )
    # System prompt is engine-owned and prepended on the wire.
    assert wire_calls[0]["messages"][0]["role"] == "system"
    assert "Ed4All operator assistant" in wire_calls[0]["messages"][0]["content"]


def test_conversation_state_round_trips(monkeypatch):
    client, _ = _scripted_client([_final_body("First."), _final_body("Second.")])
    engine = AssistantEngine(client=client, capture=_RecordingCapture())

    turn1 = engine.run_turn("q1")
    # Caller-held state is plain JSON-serializable dicts.
    json.dumps(turn1.messages)
    turn2 = engine.run_turn("q2", history=turn1.messages)

    assert [m["role"] for m in turn2.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert turn2.messages[0]["content"] == "q1"
    assert turn2.reply == "Second."


def test_unknown_tool_requested_by_model_is_refused_back(monkeypatch):
    client, wire_calls = _scripted_client(
        [
            _tool_call_body("run_shell", {"cmd": "rm -rf /"}),
            _final_body("Understood."),
        ]
    )
    engine = AssistantEngine(client=client, capture=_RecordingCapture())

    turn = engine.run_turn("delete everything")

    assert turn.tool_calls[0]["tool"] == "run_shell"
    assert turn.tool_calls[0]["result"].startswith("Refused: tool 'run_shell'")


def test_malformed_tool_arguments_refused(monkeypatch):
    body = _tool_call_body("get_help", {})
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    client, _ = _scripted_client([body, _final_body("ok")])
    engine = AssistantEngine(client=client, capture=_RecordingCapture())

    turn = engine.run_turn("help")
    assert turn.tool_calls[0]["result"].startswith("Refused: malformed tool arguments")


def test_tool_loop_capped_at_six_rounds(monkeypatch):
    monkeypatch.setattr(
        "lib.assistant.engine.dispatch_tool", lambda n, a: "still going"
    )
    client, wire_calls = _scripted_client(
        [_tool_call_body("run_status", {}) for _ in range(MAX_TOOL_ROUNDS)]
    )
    engine = AssistantEngine(client=client, capture=_RecordingCapture())

    turn = engine.run_turn("loop forever")

    assert turn.rounds == MAX_TOOL_ROUNDS == 6
    assert len(wire_calls) == MAX_TOOL_ROUNDS  # never a 7th LLM call
    assert "round limit" in turn.reply


# --------------------------------------------------------------------------- #
# DecisionCapture regression net (project law: capture fires on the call path)
# --------------------------------------------------------------------------- #


def test_capture_fires_with_dynamic_rationale():
    client, _ = _scripted_client(
        [
            _tool_call_body("seat_status", {}),
            _final_body("All seats serving."),
        ]
    )
    capture = _RecordingCapture()
    # Real dispatch would probe seats; stub it via the whitelist boundary.
    engine = AssistantEngine(client=client, capture=capture)

    engine.run_turn("are the seats up?")

    assert len(capture.decisions) == 1
    record = capture.decisions[0]
    assert record["decision_type"] == "llm_chat_call"
    assert len(record["rationale"]) >= 20
    # Dynamic rationale: interpolates the live model id, seat URL, tool
    # names, and token counts — never a static boilerplate string.
    assert "nemotron-3-nano" in record["rationale"]
    assert "localhost:8004" in record["rationale"]
    assert "seat_status" in record["rationale"]
    assert "prompt_tokens=50" in record["rationale"]
    assert "completion_tokens=20" in record["rationale"]


def test_capture_fires_with_real_decision_capture(tmp_path, monkeypatch):
    """End-to-end capture net: the REAL DecisionCapture (isolated dirs via
    the repo conftest) records one llm_chat_call decision per turn."""
    from lib.decision_capture import DecisionCapture

    client, _ = _scripted_client([_final_body("Hello operator.")])
    capture = DecisionCapture(
        course_code="assistant", phase="assistant_session", tool="assistant"
    )
    try:
        engine = AssistantEngine(client=client, capture=capture)
        turn = engine.run_turn("hello")
        assert turn.reply == "Hello operator."
        assert len(capture.decisions) == 1
        assert capture.decisions[0]["decision_type"] == "llm_chat_call"
        assert len(capture.decisions[0]["rationale"]) >= 20
    finally:
        capture.close()


# --------------------------------------------------------------------------- #
# Seat policy (engine-owned)
# --------------------------------------------------------------------------- #


def test_ensure_seat_raises_when_down_and_autostart_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSISTANT_AUTOSTART", raising=False)
    monkeypatch.setattr(
        "lib.assistant.engine.resolve_active_seat",
        lambda *, force=True: _FakeSeat(live=False),
    )
    client, _ = _scripted_client([])
    engine = AssistantEngine(client=client, capture=_RecordingCapture())

    from lib.assistant.client import AssistantSeatUnavailable

    with pytest.raises(AssistantSeatUnavailable) as exc_info:
        engine.ensure_seat()
    assert "ED4ALL_ASSISTANT_AUTOSTART" in str(exc_info.value)


def test_ensure_seat_forbids_autostart_when_disallowed(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_AUTOSTART", "1")
    monkeypatch.setattr(
        "lib.assistant.engine.resolve_active_seat",
        lambda *, force=True: _FakeSeat(live=False),
    )

    def _explode():  # pragma: no cover
        raise AssertionError("autostart must not fire when disallowed")

    monkeypatch.setattr("lib.assistant.engine.autostart_seat", _explode)
    client, _ = _scripted_client([])
    engine = AssistantEngine(client=client, capture=_RecordingCapture())

    from lib.assistant.client import AssistantSeatUnavailable

    with pytest.raises(AssistantSeatUnavailable):
        engine.ensure_seat(allow_autostart=False)


def test_ensure_seat_autostarts_when_enabled(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_AUTOSTART", "true")
    monkeypatch.setattr(
        "lib.assistant.engine.resolve_active_seat",
        lambda *, force=True: _FakeSeat(live=False),
    )

    class _Result:
        ok = True
        reason = "warm_start_coherent"

    started = {}

    def fake_autostart():
        started["fired"] = True
        return _Result()

    monkeypatch.setattr("lib.assistant.engine.autostart_seat", fake_autostart)
    reset_calls = {"n": 0}
    monkeypatch.setattr(
        "lib.assistant.engine.reset_seat_cache",
        lambda: reset_calls.__setitem__("n", reset_calls["n"] + 1),
    )
    client, _ = _scripted_client([])
    engine = AssistantEngine(client=client, capture=_RecordingCapture())

    engine.ensure_seat()  # must not raise
    assert started["fired"] is True
    # A successful autostart clears the seat cache so the next resolution
    # sees the freshly-started seat.
    assert reset_calls["n"] == 1
