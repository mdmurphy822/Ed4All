"""Tests for ``ed4all assistant`` — the thin CLI consumer of AssistantEngine.

Verifies (1) the one-shot path returns the model text (mocked transport, no
network), (2) a down seat with ``--no-seat-start`` exits loudly WITHOUT
attempting autostart, (3) a down seat without autostart enabled exits with
the start instructions, and (4) a non-loopback base URL is a usage error.
"""

from __future__ import annotations

import json
import sys
import types

from click.testing import CliRunner

from cli.commands.assistant import assistant_command


def _final_body(content):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6},
    }


def _install_fake_transport(monkeypatch, bodies):
    """Route the client's default transport at a scripted playback."""
    import lib.assistant.client as client_mod

    monkeypatch.delenv("ED4ALL_ASSISTANT_BASE_URL", raising=False)
    queue = list(bodies)

    def transport(url, payload, timeout):
        assert url.startswith("http://localhost:8004/v1")
        return queue.pop(0)

    monkeypatch.setattr(client_mod, "_urllib_transport", transport)


def _patch_active_seat(monkeypatch, *, live, base_url="http://localhost:8004/v1"):
    """Repoint the S1 dynamic seat resolver at a fixed ``ResolvedSeat``.

    The seat-resolution refactor replaced the module-level
    ``lib.assistant.engine.seat_is_serving`` boolean probe with
    ``resolve_active_seat`` (returning a ``ResolvedSeat`` whose ``.live`` field
    gates :meth:`AssistantEngine.ensure_seat`). Patch that seam in BOTH the
    engine namespace (``ensure_seat`` reads its own import) and the client
    namespace (the dynamic client re-resolves per ``chat``) so the whole path
    stays hermetic — no real seat, no real ``/v1/models`` probe.
    """
    from lib.assistant.client import ResolvedSeat

    seat = ResolvedSeat(
        seat_name="spark-nano",
        base_url=base_url,
        model="nemotron-3-nano",
        live=live,
        source="priority",
    )
    monkeypatch.setattr(
        "lib.assistant.engine.resolve_active_seat", lambda *a, **k: seat
    )
    monkeypatch.setattr(
        "lib.assistant.client.resolve_active_seat", lambda *a, **k: seat
    )


def test_one_shot_returns_model_text(monkeypatch):
    _patch_active_seat(monkeypatch, live=True)
    _install_fake_transport(
        monkeypatch, [_final_body("Hello from the nano seat.")]
    )

    result = CliRunner().invoke(assistant_command, ["--once", "hello"])

    assert result.exit_code == 0, result.output
    assert "Hello from the nano seat." in result.output


def test_positional_question_is_one_shot(monkeypatch):
    _patch_active_seat(monkeypatch, live=True)
    _install_fake_transport(monkeypatch, [_final_body("42 runs on record.")])

    result = CliRunner().invoke(assistant_command, ["how many runs?"])

    assert result.exit_code == 0, result.output
    assert "42 runs on record." in result.output


def test_seat_down_no_seat_start_exits_without_autostart(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_AUTOSTART", "1")
    _patch_active_seat(monkeypatch, live=False)

    def _explode():  # pragma: no cover
        raise AssertionError("--no-seat-start must forbid autostart")

    monkeypatch.setattr("lib.assistant.engine.autostart_seat", _explode)

    result = CliRunner().invoke(
        assistant_command, ["--once", "hi", "--no-seat-start"]
    )

    assert result.exit_code == 1
    assert "not serving" in result.output


def test_seat_down_without_autostart_prints_start_instructions(monkeypatch):
    monkeypatch.delenv("ED4ALL_ASSISTANT_AUTOSTART", raising=False)
    _patch_active_seat(monkeypatch, live=False)

    result = CliRunner().invoke(assistant_command, ["--once", "hi"])

    assert result.exit_code == 1
    assert "ED4ALL_ASSISTANT_AUTOSTART" in result.output
    assert "spark-nano" in result.output


def test_non_loopback_base_url_is_usage_error(monkeypatch):
    monkeypatch.setenv("ED4ALL_ASSISTANT_BASE_URL", "http://spark:8004/v1")

    result = CliRunner().invoke(assistant_command, ["--once", "hi"])

    assert result.exit_code == 2
    assert "loopback" in result.output


# --------------------------------------------------------------------------- #
# Debug mode
# --------------------------------------------------------------------------- #


def test_run_without_debug_is_usage_error():
    result = CliRunner().invoke(
        assistant_command, ["--run", "WF-20260721-aaaa1111", "--once", "hi"]
    )
    assert result.exit_code == 2
    assert "--run requires --debug" in result.output


def test_debug_with_no_failed_run_is_clear_error(monkeypatch):
    from lib.assistant.debug_context import DebugContextUnavailable

    def _raise(run_id):
        raise DebugContextUnavailable("No failed run found: nothing to debug.")

    monkeypatch.setattr(
        "lib.assistant.debug_context.build_debug_context", _raise
    )
    result = CliRunner().invoke(assistant_command, ["--debug", "--once", "hi"])
    assert result.exit_code == 2
    assert "No failed run found" in result.output


def test_debug_prints_banner_and_uses_debug_engine(monkeypatch):
    captured = {}

    def _fake_build(run_id):
        captured["requested_run"] = run_id
        return {
            "run_id": "WF-20260721-aaaa1111",
            "source": "explicit",
            "slug": "camp-book",
            "failed_phase": "course_planning",
            "summary": "Run WF-20260721-aaaa1111 FAILED in phase course_planning.",
            "text": "=== FAILED RUN DEBUG CONTEXT ===",
        }

    monkeypatch.setattr(
        "lib.assistant.debug_context.build_debug_context", _fake_build
    )
    _patch_active_seat(monkeypatch, live=True)

    wire = {}
    import lib.assistant.client as client_mod

    monkeypatch.delenv("ED4ALL_ASSISTANT_BASE_URL", raising=False)

    def transport(url, payload, timeout):
        wire.setdefault("payloads", []).append(payload)
        return _final_body("Root cause: the planning gate blocked.")

    monkeypatch.setattr(client_mod, "_urllib_transport", transport)

    result = CliRunner().invoke(
        assistant_command,
        ["--debug", "--run", "WF-20260721-aaaa1111", "--once", "why?"],
    )

    assert result.exit_code == 0, result.output
    assert captured["requested_run"] == "WF-20260721-aaaa1111"
    assert "FAILURE SUMMARY: Run WF-20260721-aaaa1111 FAILED" in result.output
    assert "Root cause: the planning gate blocked." in result.output
    # Debug engine on the wire: debug prompt + injected context block.
    messages = wire["payloads"][0]["messages"]
    assert "DEBUG MODE" in messages[0]["content"]
    assert "FAILED-RUN DEBUG CONTEXT" in messages[1]["content"]


# --------------------------------------------------------------------------- #
# Campaign mode + campaign tick
# --------------------------------------------------------------------------- #


class _RecordingEngine:
    """Stub AssistantEngine that records init kwargs and never touches a seat.

    Fully replaces the real engine so tests NEVER build a client or resolve a
    seat. ``run_turn`` returns a turn carrying seat/model metadata so the
    campaign-mode surfacing paths are exercised without a live seat.
    """

    inits: list = []

    def __init__(self, **kwargs):
        _RecordingEngine.inits.append(kwargs)
        self.mode = kwargs.get("mode")
        self.client = types.SimpleNamespace(
            model="stub-model", base_url="http://localhost:8001/v1"
        )

    def ensure_seat(self, *, allow_autostart: bool = True) -> None:
        return None

    def run_turn(self, user_text, history=None):
        return types.SimpleNamespace(
            reply="campaign reply text.",
            messages=[],
            tool_calls=[],
            seat_name="spark-super",
            model="stub-model",
        )


def _install_recording_engine(monkeypatch):
    _RecordingEngine.inits = []
    monkeypatch.setattr(
        "lib.assistant.engine.AssistantEngine", _RecordingEngine
    )
    return _RecordingEngine


def _install_fake_campaign_tools(monkeypatch, campaign_dir):
    """Inject a hermetic ``lib.assistant.campaign_tools`` exposing CAMPAIGN_DIR.

    Task 3 owns the real module; the CLI only imports CAMPAIGN_DIR lazily, so
    a tmp-path stub keeps this test independent of that module AND guarantees
    the operator's real campaign tree is never touched.
    """
    mod = types.ModuleType("lib.assistant.campaign_tools")
    mod.CAMPAIGN_DIR = campaign_dir
    monkeypatch.setitem(sys.modules, "lib.assistant.campaign_tools", mod)
    return mod


def test_campaign_builds_engine_with_campaign_mode(monkeypatch):
    engine = _install_recording_engine(monkeypatch)

    result = CliRunner().invoke(assistant_command, ["--campaign", "--once", "hi"])

    assert result.exit_code == 0, result.output
    assert "campaign reply text." in result.output
    assert engine.inits, "engine was never constructed"
    assert engine.inits[-1]["mode"] == "campaign"


def test_campaign_positional_question_uses_campaign_engine(monkeypatch):
    engine = _install_recording_engine(monkeypatch)

    result = CliRunner().invoke(assistant_command, ["--campaign", "status?"])

    assert result.exit_code == 0, result.output
    assert engine.inits[-1]["mode"] == "campaign"


def test_campaign_with_debug_is_usage_error(monkeypatch):
    # Must NOT construct an engine when the flag combination is rejected.
    def _explode(**kwargs):  # pragma: no cover - asserts it is never called
        raise AssertionError("engine must not build on a usage error")

    monkeypatch.setattr("lib.assistant.engine.AssistantEngine", _explode)

    result = CliRunner().invoke(
        assistant_command, ["--campaign", "--debug", "--once", "hi"]
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_campaign_tick_with_question_is_usage_error():
    result = CliRunner().invoke(
        assistant_command, ["--campaign-tick", "run a book"]
    )
    assert result.exit_code == 2
    assert "--campaign-tick" in result.output


def test_campaign_tick_with_once_is_usage_error():
    result = CliRunner().invoke(
        assistant_command, ["--campaign-tick", "--once", "go"]
    )
    assert result.exit_code == 2


def test_campaign_tick_with_campaign_is_usage_error():
    result = CliRunner().invoke(
        assistant_command, ["--campaign-tick", "--campaign"]
    )
    assert result.exit_code == 2


def test_campaign_tick_happy_path_prints_json(monkeypatch, tmp_path):
    _install_fake_campaign_tools(monkeypatch, tmp_path)
    pilot = tmp_path / "pilot.py"
    pilot.write_text(
        "def run_tick():\n"
        "    return {'ts': 'now', 'stopped': False, 'actions': ['report:x']}\n"
    )

    result = CliRunner().invoke(assistant_command, ["--campaign-tick"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["stopped"] is False
    assert payload["actions"] == ["report:x"]


def test_campaign_tick_happy_path_via_monkeypatched_helper(
    monkeypatch, tmp_path
):
    _install_fake_campaign_tools(monkeypatch, tmp_path)
    (tmp_path / "pilot.py").write_text("def run_tick():\n    return {}\n")

    monkeypatch.setattr(
        "cli.commands.assistant._run_campaign_tick",
        lambda pilot_path: {"ok": True, "seat": "spark-nano"},
    )

    result = CliRunner().invoke(assistant_command, ["--campaign-tick"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"ok": True, "seat": "spark-nano"}


def test_campaign_tick_missing_pilot_exits_2(monkeypatch, tmp_path):
    _install_fake_campaign_tools(monkeypatch, tmp_path)  # no pilot.py written

    result = CliRunner().invoke(assistant_command, ["--campaign-tick"])

    assert result.exit_code == 2
    assert "no campaign pilot" in result.output


def test_campaign_tick_pilot_exception_exits_1(monkeypatch, tmp_path):
    _install_fake_campaign_tools(monkeypatch, tmp_path)
    (tmp_path / "pilot.py").write_text(
        "def run_tick():\n"
        "    raise RuntimeError('pilot boom')\n"
    )

    result = CliRunner().invoke(assistant_command, ["--campaign-tick"])

    assert result.exit_code == 1
    assert "campaign tick failed" in result.output
