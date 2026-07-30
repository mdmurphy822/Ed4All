"""Roadmap OP2 — OpenAI-compatible client usage-tap tests.

When ``ED4ALL_RUN_ID`` is set, every ``chat_completion`` call appends one
metering row to ``runtime/state/runs/<run_id>/llm_usage.jsonl``; when unset it is a
strict no-op (byte-identical for bare library callers).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from Trainforge.generators._openai_compatible_client import (
    OpenAICompatibleClient,
)


def _mock_client(**kwargs) -> OpenAICompatibleClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 7,
                    "total_tokens": 49,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    return OpenAICompatibleClient(
        base_url="http://localhost:1234/v1",
        model="test-model",
        provider_label="local",
        client=httpx.Client(transport=transport),
        **kwargs,
    )


def test_usage_row_written_when_run_id_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-OP2")
    client = _mock_client()
    out = client.chat_completion([{"role": "user", "content": "hi"}])
    assert out == "hello"

    usage_path = tmp_path / "runs" / "RUN-OP2" / "llm_usage.jsonl"
    assert usage_path.exists()
    rows = [
        json.loads(line)
        for line in usage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "local"
    assert row["model"] == "test-model"
    assert row["prompt_tokens"] == 42
    assert row["completion_tokens"] == 7
    assert isinstance(row["duration_ms"], (int, float))
    assert "ts" in row


def test_no_row_when_run_id_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    client = _mock_client()
    client.chat_completion([{"role": "user", "content": "hi"}])
    # No run id → no runs dir mutation at all.
    assert not (tmp_path / "runs").exists()


def test_multiple_calls_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-OP2b")
    client = _mock_client()
    client.chat_completion([{"role": "user", "content": "a"}])
    client.chat_completion([{"role": "user", "content": "b"}])
    usage_path = tmp_path / "runs" / "RUN-OP2b" / "llm_usage.jsonl"
    rows = [
        line
        for line in usage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2


# --------------------------------------------------------------------------
# Metering-correctness — the row stamps the SPENDING phase from the executor-
# published active-phase env, exactly as the SemantiK cascade taps do.
# --------------------------------------------------------------------------


def _one_row(usage_path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in usage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    return rows[0]


def test_row_stamps_active_phase_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-PHASE")
    monkeypatch.setenv("ED4ALL_ACTIVE_PHASE", "course_planning")
    client = _mock_client()
    client.chat_completion([{"role": "user", "content": "hi"}])
    row = _one_row(tmp_path / "runs" / "RUN-PHASE" / "llm_usage.jsonl")
    assert row["phase"] == "course_planning"


def test_row_omits_phase_when_context_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # No active-phase env — the row degrades to the legacy shape (no ``phase``
    # key), never raises. This is today's null-phase behaviour, preserved.
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-NOPHASE")
    monkeypatch.delenv("ED4ALL_ACTIVE_PHASE", raising=False)
    client = _mock_client()
    client.chat_completion([{"role": "user", "content": "hi"}])
    row = _one_row(tmp_path / "runs" / "RUN-NOPHASE" / "llm_usage.jsonl")
    assert "phase" not in row


def test_row_omits_phase_when_context_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A blank / whitespace active-phase env is treated as absent (never
    # fabricated into a phase, never raises).
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-BLANK")
    monkeypatch.setenv("ED4ALL_ACTIVE_PHASE", "   ")
    client = _mock_client()
    client.chat_completion([{"role": "user", "content": "hi"}])
    row = _one_row(tmp_path / "runs" / "RUN-BLANK" / "llm_usage.jsonl")
    assert "phase" not in row


def test_module_tap_stamps_active_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The module-level tap (the seam the Courseforge two-pass dispatch calls
    # directly, bypassing ``chat_completion``) also stamps the active phase.
    from Trainforge.generators._openai_compatible_client import (
        maybe_append_usage_row,
    )

    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-MOD")
    monkeypatch.setenv("ED4ALL_ACTIVE_PHASE", "content_generation_rewrite")
    maybe_append_usage_row(
        provider_label="local",
        model="m",
        usage={"prompt_tokens": 3, "completion_tokens": 1},
        duration_ms=1.0,
    )
    row = _one_row(tmp_path / "runs" / "RUN-MOD" / "llm_usage.jsonl")
    assert row["phase"] == "content_generation_rewrite"
    assert row["prompt_tokens"] == 3
