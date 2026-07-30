"""OP2 usage tap on the ``_BaseLLMProvider`` dispatch seam.

Regression net for the metering gap where every Courseforge tier dispatch
(outline / rewrite / content-generator) POSTed via
``_dispatch_call_with_usage`` → ``OpenAICompatibleClient._post_with_retry``
DIRECTLY — bypassing ``chat_completion`` and therefore the OP2 usage tap —
so ``runtime/state/runs/<run_id>/llm_usage.jsonl`` went silent for the whole
two-pass Courseforge surface while ``BuildCostAggregator`` undercounted.

The fix mirrors the tap at the ``_base`` seam through the SHARED
module-level helper
``Trainforge.generators._openai_compatible_client.maybe_append_usage_row``.
Contract under test (identical to the ``chat_completion`` tap):

- gated on ``ED4ALL_RUN_ID`` — no run id, no row, no runs-dir mutation;
- best-effort — a tap failure never perturbs the real call;
- real server-reported token counts + ``finish_reason`` on the row;
- the row is appended BEFORE ``_extract_text`` raises on
  ``finish_reason == "length"`` so a truncated call still meters.

Mirrors the ``httpx.MockTransport`` fixture pattern of
``MCP/tests/test_op2_usage_tap.py`` and
``Courseforge/generators/tests/test_base_llm_provider.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._base import _BaseLLMProvider  # noqa: E402
from Trainforge.generators._synthesis_common import (  # noqa: E402
    SynthesisProviderError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_body(
    content: str = "oa-reply",
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 42,
    completion_tokens: int = 7,
) -> dict:
    return {
        "id": "cmpl-tap-test",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _mock_httpx_client(body: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


class _MinimalProvider(_BaseLLMProvider):
    """Concrete subclass exercising ONLY the base's dispatch plumbing."""

    def _render_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        return "test prompt"

    def _emit_per_call_decision(
        self,
        *,
        raw_text: str,
        retry_count: int,
        **call_context: Any,
    ) -> None:  # pragma: no cover — not reached by these tests
        pass


def _make_provider(body: dict) -> _MinimalProvider:
    return _MinimalProvider(
        provider="local",
        client=_mock_httpx_client(body),
        system_prompt="SYS",
    )


def _read_rows(runs_dir: Path, run_id: str) -> List[dict]:
    usage_path = runs_dir / run_id / "llm_usage.jsonl"
    assert usage_path.exists(), f"expected usage ledger at {usage_path}"
    return [
        json.loads(line)
        for line in usage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def _runs_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs_dir))
    monkeypatch.delenv("COURSEFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_BASE_URL", raising=False)
    return runs_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dispatch_call_with_usage_appends_usage_row(
    _runs_env: Path, monkeypatch: pytest.MonkeyPatch
):
    """The base dispatch seam appends one OP2 row per call when
    ``ED4ALL_RUN_ID`` is set — the outline/rewrite-tier metering fix."""
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-BASE-TAP")
    p = _make_provider(_success_body())

    text, retries, usage = p._dispatch_call_with_usage("user prompt")
    assert text == "oa-reply"
    assert retries == 0
    assert usage["prompt_tokens"] == 42

    rows = _read_rows(_runs_env, "RUN-BASE-TAP")
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "local"
    assert row["model"] == p._model
    assert row["prompt_tokens"] == 42
    assert row["completion_tokens"] == 7
    assert row["finish_reason"] == "stop"
    assert isinstance(row["duration_ms"], (int, float))
    assert "ts" in row


def test_dispatch_call_no_row_when_run_id_unset(
    _runs_env: Path, monkeypatch: pytest.MonkeyPatch
):
    """No run id → strict no-op: no row, no runs-dir mutation."""
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    p = _make_provider(_success_body())
    text, _, _ = p._dispatch_call_with_usage("user prompt")
    assert text == "oa-reply"
    assert not _runs_env.exists()


def test_dispatch_call_meters_before_truncation_raise(
    _runs_env: Path, monkeypatch: pytest.MonkeyPatch
):
    """A ``finish_reason == "length"`` call still records its metering row
    (the tap fires BEFORE ``_extract_text`` raises the truncation guard) —
    truncation diagnosis is exactly what the ``finish_reason`` field is
    for."""
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-BASE-TAP-LEN")
    p = _make_provider(_success_body(finish_reason="length"))

    with pytest.raises(SynthesisProviderError):
        p._dispatch_call_with_usage("user prompt")

    rows = _read_rows(_runs_env, "RUN-BASE-TAP-LEN")
    assert len(rows) == 1
    assert rows[0]["finish_reason"] == "length"


def test_multiple_dispatches_append(
    _runs_env: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ED4ALL_RUN_ID", "RUN-BASE-TAP-2")
    p = _make_provider(_success_body())
    p._dispatch_call_with_usage("a")
    p._dispatch_call_with_usage("b")
    assert len(_read_rows(_runs_env, "RUN-BASE-TAP-2")) == 2
