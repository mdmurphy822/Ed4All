"""Debug-mode tests: the context builder (pointer / newest-failed / explicit
resolution, bounded + redacted output) and the engine's debug mode (system
prompt shifts, context injected on the wire, no new capabilities)."""

from __future__ import annotations

import json

import pytest

from lib.assistant import debug_context as debug_mod
from lib.assistant import tools as assistant_tools
from lib.assistant.client import AssistantClient
from lib.assistant.debug_context import (
    MAX_DEBUG_CONTEXT_CHARS,
    DebugContextUnavailable,
    build_debug_context,
)
from lib.assistant.engine import (
    DEBUG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    AssistantEngine,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def debug_env(monkeypatch, tmp_path):
    """Isolated state + campaign roots wired into BOTH modules."""
    state = tmp_path / "state"
    (state / "workflows").mkdir(parents=True)
    (state / "runs").mkdir(parents=True)
    campaign = tmp_path / "campaign"
    (campaign / "logs").mkdir(parents=True)
    for module in (assistant_tools, debug_mod):
        monkeypatch.setattr(module, "STATE_PATH", state)
        monkeypatch.setattr(module, "CAMPAIGN_DIR", campaign)
    monkeypatch.setattr(
        assistant_tools, "LIBV2_COURSES", tmp_path / "libv2" / "courses"
    )
    monkeypatch.setattr(
        debug_mod, "LAST_FAILURE_POINTER", campaign / "last_failure.json"
    )
    # doctor post-mortem stubbed (no subprocess in unit tests)
    monkeypatch.setattr(
        debug_mod, "_doctor_postmortem", lambda run_id: f"postmortem for {run_id}: OK"
    )
    return {"state": state, "campaign": campaign}


def _write_failed(state, run_id, *, mtime_bump=0, slug="camp-book"):
    doc = {
        "id": run_id,
        "status": "FAILED",
        "type": "textbook_to_course",
        "params": {"course_name": slug},
        "failed_phase": "semantik_conversion",
        "failure_reason": "seat 'spark-glm' could not be brought up coherent",
        "phase_outputs": {
            "semantik_conversion": {
                "_completed": False,
                "_gates_passed": False,
                "_gate_results": [
                    {
                        "gate_id": "chunkset_manifest",
                        "passed": False,
                        "severity": "critical",
                        "issues": [
                            {
                                "severity": "critical",
                                "code": "CHUNKSET_MISSING",
                                "message": "no chunks emitted",
                            }
                        ],
                    }
                ],
            }
        },
    }
    path = state / "workflows" / f"{run_id}.json"
    path.write_text(json.dumps(doc))
    if mtime_bump:
        import os

        stat = path.stat()
        os.utime(path, (stat.st_atime + mtime_bump, stat.st_mtime + mtime_bump))
    return path


# --------------------------------------------------------------------------- #
# build_debug_context
# --------------------------------------------------------------------------- #


def test_explicit_run_id_context_sections(debug_env):
    _write_failed(debug_env["state"], "WF-20260721-aaaa1111")
    (debug_env["campaign"] / "logs" / "camp-book.log").write_text(
        "\n".join(f"log-{i}" for i in range(120))
    )
    context = build_debug_context("WF-20260721-aaaa1111")
    assert context["run_id"] == "WF-20260721-aaaa1111"
    assert context["source"] == "explicit"
    assert context["failed_phase"] == "semantik_conversion"
    assert "FAILED in phase semantik_conversion" in context["summary"]
    assert "spark-glm" in context["summary"]
    text = context["text"]
    assert "run report" in text and "gate report" in text
    assert "CHUNKSET_MISSING" in text
    assert "log-119" in text  # log tail (via the campaign slug log)
    assert "postmortem for WF-20260721-aaaa1111" in text
    assert len(text) <= MAX_DEBUG_CONTEXT_CHARS + 100


def test_explicit_bad_run_id_raises(debug_env):
    with pytest.raises(DebugContextUnavailable):
        build_debug_context("WF-bogus")


def test_pointer_file_preferred(debug_env):
    _write_failed(debug_env["state"], "WF-20260721-aaaa1111")
    _write_failed(debug_env["state"], "WF-20260721-bbbb2222", mtime_bump=100)
    (debug_env["campaign"] / "last_failure.json").write_text(
        json.dumps({"slug": "camp-book", "run_id": "WF-20260721-aaaa1111"})
    )
    context = build_debug_context(None)
    assert context["run_id"] == "WF-20260721-aaaa1111"  # pointer wins over newest
    assert context["source"] == "last_failure_pointer"
    assert context["slug"] == "camp-book"


def test_newest_failed_fallback(debug_env):
    _write_failed(debug_env["state"], "WF-20260721-aaaa1111")
    _write_failed(debug_env["state"], "WF-20260721-bbbb2222", mtime_bump=100)
    context = build_debug_context(None)
    assert context["run_id"] == "WF-20260721-bbbb2222"
    assert context["source"] == "newest_failed"


def test_no_failed_run_raises(debug_env):
    with pytest.raises(DebugContextUnavailable) as exc_info:
        build_debug_context(None)
    assert "No failed run found" in str(exc_info.value)


def test_pointer_with_invalid_run_id_falls_back(debug_env):
    _write_failed(debug_env["state"], "WF-20260721-cccc3333")
    (debug_env["campaign"] / "last_failure.json").write_text(
        json.dumps({"slug": "camp-book", "run_id": "../../etc/passwd"})
    )
    context = build_debug_context(None)
    assert context["run_id"] == "WF-20260721-cccc3333"
    assert context["source"] == "newest_failed"


# --------------------------------------------------------------------------- #
# Engine debug mode
# --------------------------------------------------------------------------- #


def _final_body(content):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _scripted_client(bodies):
    queue = list(bodies)
    calls = []

    def transport(url, payload, timeout):
        calls.append(payload)
        return queue.pop(0)

    return AssistantClient(transport=transport), calls


class _RecordingCapture:
    def __init__(self):
        self.decisions = []

    def log_decision(self, **kwargs):
        self.decisions.append(kwargs)


def test_engine_debug_mode_prompt_and_context_injection():
    client, wire = _scripted_client([_final_body("Diagnosis.")])
    engine = AssistantEngine(
        client=client,
        capture=_RecordingCapture(),
        mode="debug",
        debug_context="=== FAILED RUN DEBUG CONTEXT ===\nRun WF-x FAILED.",
    )
    turn = engine.run_turn("what went wrong?")
    assert turn.reply == "Diagnosis."
    messages = wire[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "DEBUG MODE" in messages[0]["content"]
    assert messages[0]["content"] != SYSTEM_PROMPT
    assert engine.system_prompt == DEBUG_SYSTEM_PROMPT
    # Context is the first assistant-visible context block after the prompt.
    assert messages[1]["role"] == "system"
    assert "FAILED-RUN DEBUG CONTEXT" in messages[1]["content"]
    assert "Run WF-x FAILED." in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "what went wrong?"}
    # Context stays wire-only — never leaks into caller-held history.
    assert all("DEBUG CONTEXT" not in str(m.get("content")) for m in turn.messages)


def test_engine_operator_mode_unchanged_default(monkeypatch):
    # Hermetic: the reasoning-off directive (set in production/campaign envs)
    # legitimately prepends into messages[0]; this test asserts the pure
    # operator-mode wire shape, so pin the flag off (sibling pattern:
    # test_assistant_model_agnostic.py).
    monkeypatch.delenv("ED4ALL_REASONING_THINKING_OFF", raising=False)
    client, wire = _scripted_client([_final_body("ok")])
    engine = AssistantEngine(client=client, capture=_RecordingCapture())
    assert engine.mode == "operator"
    engine.run_turn("hello")
    messages = wire[0]["messages"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"  # no context block in operator mode


def test_engine_rejects_unknown_mode():
    client, _ = _scripted_client([])
    with pytest.raises(ValueError):
        AssistantEngine(client=client, capture=_RecordingCapture(), mode="root")


def test_debug_mode_capture_rationale_names_mode():
    client, _ = _scripted_client([_final_body("done")])
    capture = _RecordingCapture()
    engine = AssistantEngine(
        client=client, capture=capture, mode="debug", debug_context="ctx"
    )
    engine.run_turn("diagnose")
    assert "mode=debug" in capture.decisions[0]["rationale"]


def test_debug_prompts_reference_only_registered_tools():
    """The debug prompt proposes actions ONLY from the whitelist — every
    tool name it mentions must exist in the registry."""
    import re

    mentioned = set(re.findall(r"\b([a-z_]{3,})\b(?= <| \()", DEBUG_SYSTEM_PROMPT))
    known = set(assistant_tools.TOOL_REGISTRY)
    assert mentioned & known  # the prompt does name tools
    for name in ("resume_run", "start_book", "doctor", "tail_log", "flag_lookup"):
        assert name in DEBUG_SYSTEM_PROMPT
