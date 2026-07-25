"""Tests for ``MCP/tools/assistant_tools.py`` — the operator-assistant MCP surface.

Hermetic: every ``lib.assistant`` collaborator is monkeypatched (the campaign
dispatch choke point, the seat resolver, the engine), so no real seat is probed,
no subprocess is spawned, and no LLM is called. These tests assert the wrapper's
ONLY job — faithful, non-widening delegation to the already-hardened
``lib.assistant`` layer.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.assistant_tools import register_assistant_tools  # noqa: E402


class _CapturingMCP:
    """Minimal stand-in for the FastMCP registration surface.

    ``register_assistant_tools`` only needs an object whose ``tool()`` returns a
    decorator; FastMCP keeps the wrapped coroutine callable, so a pass-through
    decorator that records ``fn`` by name reproduces the registration contract
    without requiring the optional ``mcp`` package to be installed.
    """

    def __init__(self):
        self.captured = {}

    def tool(self, *a, **k):
        def decorator(fn):
            self.captured[fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture
def tools():
    """Register the assistant tools on the capturing shim; expose them by name."""
    mcp = _CapturingMCP()
    register_assistant_tools(mcp)
    return mcp.captured


@pytest.fixture
def spy_dispatch(monkeypatch):
    """Replace the campaign dispatch choke point with a recording spy."""
    calls = []

    def _fake(name, arguments, *, readonly=False):
        calls.append({"name": name, "arguments": dict(arguments), "readonly": readonly})
        return f"OK:{name}"

    import lib.assistant.campaign_tools as ct

    monkeypatch.setattr(ct, "dispatch_campaign_tool", _fake)
    return calls


# --------------------------------------------------------------------------- #
# Campaign tools — faithful delegation to dispatch_campaign_tool
# --------------------------------------------------------------------------- #


async def test_queue_delegates(tools, spy_dispatch):
    out = await tools["assistant_campaign_queue"]()
    assert out == "OK:campaign_queue"
    assert spy_dispatch == [{"name": "campaign_queue", "arguments": {}, "readonly": False}]


async def test_run_status_maps_wf_id_to_run_id(tools, spy_dispatch):
    await tools["assistant_campaign_run_status"]("WF-20260722-abc12345")
    assert spy_dispatch[-1]["name"] == "campaign_run_status"
    assert spy_dispatch[-1]["arguments"] == {"run_id": "WF-20260722-abc12345"}


async def test_run_status_default_empty(tools, spy_dispatch):
    await tools["assistant_campaign_run_status"]()
    assert spy_dispatch[-1]["arguments"] == {"run_id": ""}


async def test_prepare_run_parses_overrides(tools, spy_dispatch):
    await tools["assistant_campaign_prepare_run"](
        "/corpus/book.pdf", '{"ED4ALL_TO_CHAPTER_ANCHOR": "1"}', "a note"
    )
    call = spy_dispatch[-1]
    assert call["name"] == "campaign_prepare_run"
    assert call["arguments"]["corpus"] == "/corpus/book.pdf"
    assert call["arguments"]["env_overlay"] == {"ED4ALL_TO_CHAPTER_ANCHOR": "1"}
    assert call["arguments"]["note"] == "a note"


async def test_prepare_run_bad_json_typed_error_no_dispatch(tools, spy_dispatch):
    out = await tools["assistant_campaign_prepare_run"]("/corpus/book.pdf", "{not json")
    err = json.loads(out)
    assert err["detail"] == "validation_error"
    assert spy_dispatch == []  # never reached the choke point


async def test_prepare_run_non_object_overrides_error(tools, spy_dispatch):
    out = await tools["assistant_campaign_prepare_run"]("/corpus/book.pdf", "[1,2]")
    assert "error" in json.loads(out)
    assert spy_dispatch == []


async def test_launch_run_maps_prepared_id_to_name(tools, spy_dispatch):
    await tools["assistant_campaign_launch_run"]("mybook")
    assert spy_dispatch[-1] == {
        "name": "campaign_launch_run",
        "arguments": {"name": "mybook"},
        "readonly": False,
    }


async def test_resume_run_never_carries_force(tools, spy_dispatch):
    await tools["assistant_campaign_resume_run"]("WF-20260722-deadbeef")
    call = spy_dispatch[-1]
    assert call["name"] == "campaign_resume_run"
    assert call["arguments"] == {"run_id": "WF-20260722-deadbeef"}
    assert "force" not in call["arguments"]  # plain-resume-only, structurally


async def test_stop_run_maps_wf_id(tools, spy_dispatch):
    await tools["assistant_campaign_stop_run"]("WF-20260722-deadbeef")
    assert spy_dispatch[-1] == {
        "name": "campaign_stop_run",
        "arguments": {"run_id": "WF-20260722-deadbeef"},
        "readonly": False,
    }


async def test_report_merges_kind_and_payload(tools, spy_dispatch):
    await tools["assistant_campaign_report"](
        "book_complete", '{"summary": "done", "book_slug": "wow-lore"}'
    )
    call = spy_dispatch[-1]
    assert call["name"] == "campaign_report"
    assert call["arguments"] == {
        "kind": "book_complete",
        "summary": "done",
        "book_slug": "wow-lore",
    }


async def test_report_bad_payload_typed_error(tools, spy_dispatch):
    out = await tools["assistant_campaign_report"]("book_complete", "{oops")
    assert json.loads(out)["detail"] == "validation_error"
    assert spy_dispatch == []


# --------------------------------------------------------------------------- #
# Stage-B training wrappers — faithful delegation, nothing added
# --------------------------------------------------------------------------- #


async def test_prepare_training_delegates(tools, spy_dispatch):
    out = await tools["assistant_campaign_prepare_training"]("sample-book-a")
    assert out == "OK:campaign_prepare_training"
    assert spy_dispatch[-1] == {
        "name": "campaign_prepare_training",
        "arguments": {"slug": "sample-book-a"},
        "readonly": False,
    }


async def test_launch_training_delegates_slug_only(tools, spy_dispatch):
    out = await tools["assistant_campaign_launch_training"]("sample-book-a")
    assert out == "OK:campaign_launch_training"
    call = spy_dispatch[-1]
    assert call["name"] == "campaign_launch_training"
    assert call["arguments"] == {"slug": "sample-book-a"}
    # No base-model / force / flag parameters exist on the wrapper surface.
    assert set(call["arguments"]) == {"slug"}


async def test_training_status_delegates_with_and_without_slug(tools, spy_dispatch):
    await tools["assistant_campaign_training_status"]("sample-book-a")
    assert spy_dispatch[-1] == {
        "name": "campaign_training_status",
        "arguments": {"slug": "sample-book-a"},
        "readonly": False,
    }
    await tools["assistant_campaign_training_status"]()
    assert spy_dispatch[-1]["arguments"] == {}


# --------------------------------------------------------------------------- #
# Seat status — read-only probe
# --------------------------------------------------------------------------- #


@dataclass
class _FakeSeat:
    seat_name: Optional[str]
    base_url: str
    model: str
    live: bool
    source: str


async def test_seat_status_reports_resolved_seat(tools, monkeypatch):
    import lib.assistant.client as cl

    monkeypatch.setattr(
        cl,
        "resolve_active_seat",
        lambda *, force=False: _FakeSeat("spark-nano", "http://localhost:8004/v1", "nemotron-3-nano", True, "priority"),
    )
    out = json.loads(await tools["assistant_seat_status"]())
    assert out == {
        "seat_name": "spark-nano",
        "base_url": "http://localhost:8004/v1",
        "model": "nemotron-3-nano",
        "live": True,
        "source": "priority",
    }


async def test_seat_status_non_loopback_typed_error(tools, monkeypatch):
    import lib.assistant.client as cl

    def _raise(*, force=False):
        raise cl.AssistantProviderNotLocal("non-loopback seat")

    monkeypatch.setattr(cl, "resolve_active_seat", _raise)
    out = json.loads(await tools["assistant_seat_status"]())
    assert out["detail"] == "provider_not_local"


# --------------------------------------------------------------------------- #
# assistant_ask — one-shot READONLY campaign-tick turn
# --------------------------------------------------------------------------- #


@dataclass
class _FakeTurn:
    reply: str = "observed queue"
    rounds: int = 1
    tool_calls: Optional[list] = None
    prompt_tokens: int = 10
    completion_tokens: int = 5
    seat_name: Optional[str] = "spark-nano"
    model: Optional[str] = "nemotron-3-nano"


class _FakeEngine:
    """Records the mode it was built in and the ensure_seat call shape."""

    last_instance = None

    def __init__(self, *, mode="operator", **_kw):
        self.mode = mode
        self.ensure_seat_kwargs = None
        _FakeEngine.last_instance = self

    def ensure_seat(self, *, allow_autostart=True):
        self.ensure_seat_kwargs = {"allow_autostart": allow_autostart}

    def run_turn(self, question, history=None):
        return _FakeTurn(tool_calls=[{"tool": "campaign_queue", "arguments": "{}", "result": "..."}])


async def test_ask_runs_readonly_campaign_tick(tools, monkeypatch):
    import lib.assistant.engine as eng

    monkeypatch.setattr(eng, "AssistantEngine", _FakeEngine)
    out = json.loads(await tools["assistant_ask"]("what is queued?"))
    assert out["reply"] == "observed queue"
    assert out["mode"] == "campaign-tick"
    assert out["tools_called"] == ["campaign_queue"]
    assert out["seat_name"] == "spark-nano"
    # Built in campaign-tick mode and never allowed to autostart from an MCP call.
    assert _FakeEngine.last_instance.mode == "campaign-tick"
    assert _FakeEngine.last_instance.ensure_seat_kwargs == {"allow_autostart": False}


async def test_ask_empty_question_error(tools):
    out = json.loads(await tools["assistant_ask"]("  "))
    assert "error" in out


async def test_ask_seat_unavailable_typed_error(tools, monkeypatch):
    import lib.assistant.client as cl
    import lib.assistant.engine as eng

    class _DownEngine(_FakeEngine):
        def ensure_seat(self, *, allow_autostart=True):
            raise cl.AssistantSeatUnavailable("seat down; start it")

    monkeypatch.setattr(eng, "AssistantEngine", _DownEngine)
    out = json.loads(await tools["assistant_ask"]("anything"))
    assert out["detail"] == "seat_unavailable"
