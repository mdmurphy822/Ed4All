"""GLM-OCR lane LLM-usage metering tests (OP2 tap parity).

Covers the three lane call-site taps plus the meter helper they ride:

* ``llm_usage_meter.record_provider_usage`` — the PROVIDER-keyed row shape the
  GUI stats band / ``BuildCostAggregator`` read (``row["provider"]``), with the
  meter's ledger resolution (``ED4ALL_RUN_ID`` → run ledger; unset → sidecar).
* ``glmocr/alttext.py::OpenAICompatibleAltTextClient.describe`` — provider
  ``semantik-alttext``, tokens from the server-reported ``usage``.
* ``glmocr/sdk_client.py::SdkGlmOcrClient._parse_chunk`` — provider
  ``semantik-glmocr``, honest zero tokens (the SDK surfaces no usage) + the
  additive ``pages`` field.
* ``glmocr/heading_judge.py::_emit_usage_row`` — the IN-LANE fallback: with
  ``SEMANTIK_LLM_USAGE_PATH`` unset the tap resolves the run ledger via the
  meter (``ED4ALL_RUN_ID``); the explicit path still wins when set.

All HTTP is mocked; nothing touches a seat. The repo conftest isolates
``ED4ALL_STATE_RUNS_DIR`` into tmp, and every test that exercises the sidecar
fallback redirects ``SEMANTIK_DATA_DIR`` too.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

_SEMANTIK_ROOT = Path(__file__).resolve().parents[2]
if str(_SEMANTIK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIK_ROOT))

from semantik_structure import llm_usage_meter  # noqa: E402
from semantik_structure.glmocr import alttext as alttext_mod  # noqa: E402
from semantik_structure.glmocr import heading_judge as hj  # noqa: E402
from semantik_structure.glmocr import sdk_client as sdk_mod  # noqa: E402
from semantik_structure.glmocr.transform import GlmPage  # noqa: E402


RUN_ID = "WF-test-usage-0001"


@pytest.fixture()
def run_ledger(tmp_path, monkeypatch):
    """Point the meter at a tmp run ledger via the env chain it documents."""
    runs = tmp_path / "state_runs"
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs))
    monkeypatch.setenv("ED4ALL_RUN_ID", RUN_ID)
    # A stray SEMANTIK_LLM_USAGE_PATH would divert the judge tap.
    monkeypatch.delenv("SEMANTIK_LLM_USAGE_PATH", raising=False)
    return runs / RUN_ID / "llm_usage.jsonl"


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ── record_provider_usage (the meter helper). ────────────────────────────────


def test_record_provider_usage_writes_op2_row_to_run_ledger(run_ledger):
    llm_usage_meter.record_provider_usage(
        provider="semantik-glmocr",
        model="glm-ocr",
        usage={"prompt_tokens": 11, "completion_tokens": 7},
        duration_ms=123.456,
        finish_reason="stop",
        phase="semantik_conversion",
        extra={"pages": 3},
    )
    rows = _rows(run_ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "semantik-glmocr"
    assert row["model"] == "glm-ocr"
    assert row["prompt_tokens"] == 11
    assert row["completion_tokens"] == 7
    assert row["duration_ms"] == 123.456
    assert row["finish_reason"] == "stop"
    assert row["phase"] == "semantik_conversion"
    assert row["pages"] == 3
    assert "ts" in row


def test_record_provider_usage_absent_usage_is_zeros(run_ledger):
    llm_usage_meter.record_provider_usage(
        provider="semantik-glmocr", model="glm-ocr", usage=None,
        duration_ms=1.0,
    )
    row = _rows(run_ledger)[0]
    assert row["prompt_tokens"] == 0
    assert row["completion_tokens"] == 0
    assert "finish_reason" not in row


def test_record_provider_usage_no_run_id_falls_to_sidecar(tmp_path, monkeypatch):
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.setenv("SEMANTIK_DATA_DIR", str(tmp_path / "semantik_data"))
    llm_usage_meter.record_provider_usage(
        provider="semantik-alttext", model="m", usage=None, duration_ms=2.0,
    )
    sidecar = tmp_path / "semantik_data" / "llm_usage" / "llm_usage.jsonl"
    assert sidecar.is_file()
    assert _rows(sidecar)[0]["provider"] == "semantik-alttext"


def test_record_provider_usage_never_raises(monkeypatch):
    # Unresolvable ledger dirs + a hostile usage object → swallowed.
    monkeypatch.setenv("ED4ALL_RUN_ID", RUN_ID)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", "/proc/nope/state_runs")
    llm_usage_meter.record_provider_usage(
        provider="p", model="m", usage=object(), duration_ms=1.0,
    )  # must not raise


# ── alttext describe() tap (seat :8003, provider semantik-alttext). ──────────


def _alttext_payload():
    return {
        "choices": [{
            "message": {"content": json.dumps({
                "alt": "A number line from 0 to 10.",
                "long_desc": None,
                "caption_suggestion": None,
            })},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 321, "completion_tokens": 44},
    }


def _patch_urlopen(monkeypatch, payload):
    calls = []

    def _fake_urlopen(req, timeout=None):
        calls.append(req)
        return io.StringIO(json.dumps(payload))

    monkeypatch.setattr(
        alttext_mod.urllib.request, "urlopen", _fake_urlopen
    )
    return calls


def test_alttext_describe_emits_metering_row(run_ledger, monkeypatch):
    _patch_urlopen(monkeypatch, _alttext_payload())
    client = alttext_mod.OpenAICompatibleAltTextClient(
        base_url="http://localhost:8003/v1", model="qwen3-vl-30b",
    )
    result = client.describe("aW1n", None)
    assert result["alt"].startswith("A number line")
    rows = _rows(run_ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "semantik-alttext"
    assert row["model"] == "qwen3-vl-30b"
    assert row["prompt_tokens"] == 321
    assert row["completion_tokens"] == 44
    assert row["finish_reason"] == "stop"
    assert row["duration_ms"] >= 0
    assert row["phase"] == "semantik_conversion"


def test_alttext_describe_no_run_id_no_run_ledger(tmp_path, monkeypatch):
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SEMANTIK_DATA_DIR", str(tmp_path / "semantik_data"))
    _patch_urlopen(monkeypatch, _alttext_payload())
    client = alttext_mod.OpenAICompatibleAltTextClient(
        base_url="http://localhost:8003/v1", model="m",
    )
    client.describe("aW1n", None)
    assert not (tmp_path / "runs").exists()  # no run ledger without a run id
    sidecar = tmp_path / "semantik_data" / "llm_usage" / "llm_usage.jsonl"
    assert sidecar.is_file()  # standalone runs still meter (meter contract)


def test_alttext_metering_failure_never_breaks_the_call(run_ledger, monkeypatch):
    _patch_urlopen(monkeypatch, _alttext_payload())

    def _boom(**_kwargs):
        raise RuntimeError("tap exploded")

    monkeypatch.setattr(
        llm_usage_meter, "record_provider_usage", _boom
    )
    client = alttext_mod.OpenAICompatibleAltTextClient(
        base_url="http://localhost:8003/v1", model="m",
    )
    result = client.describe("aW1n", None)  # must not raise
    assert result["alt"]


# ── GLM-OCR SDK tap (seat :8002, provider semantik-glmocr). ──────────────────


class _FakeSdkResult:
    def __init__(self, regions):
        self.json_result = regions
        self._error = None


class _FakeParser:
    def parse(self, paths, save_layout_visualization=False):
        return [
            _FakeSdkResult([{"index": 0, "native_label": "text",
                             "bbox_2d": [0, 0, 10, 10], "content": "x"}])
            for _ in paths
        ]

    def close(self):
        pass


def test_sdk_parse_chunk_emits_metering_row(run_ledger, monkeypatch, tmp_path):
    client = sdk_mod.SdkGlmOcrClient(
        base_url="http://localhost:8002/v1", model="glm-ocr", workers=1,
    )
    monkeypatch.setattr(client, "_make_parser", lambda: _FakeParser())
    p1, p2 = tmp_path / "page-1.png", tmp_path / "page-2.png"
    pages = client.parse_pages([p1, p2])
    assert [p.page_no for p in pages] == [1, 2]
    assert all(isinstance(p, GlmPage) for p in pages)
    rows = _rows(run_ledger)
    assert len(rows) == 1  # one row per parse chunk (workers=1 → one chunk)
    row = rows[0]
    assert row["provider"] == "semantik-glmocr"
    assert row["model"] == "glm-ocr"
    # The SDK surfaces no server usage → honest zeros per the tap contract.
    assert row["prompt_tokens"] == 0
    assert row["completion_tokens"] == 0
    assert row["duration_ms"] >= 0
    assert row["pages"] == 2
    assert row["phase"] == "semantik_conversion"


def test_sdk_metering_failure_never_breaks_the_parse(run_ledger, monkeypatch, tmp_path):
    client = sdk_mod.SdkGlmOcrClient(
        base_url="http://localhost:8002/v1", model="glm-ocr", workers=1,
    )
    monkeypatch.setattr(client, "_make_parser", lambda: _FakeParser())

    def _boom(**_kwargs):
        raise RuntimeError("tap exploded")

    monkeypatch.setattr(llm_usage_meter, "record_provider_usage", _boom)
    pages = client.parse_pages([tmp_path / "page-1.png"])  # must not raise
    assert len(pages) == 1 and pages[0].error is None


# ── heading-judge IN-LANE fallback (provider semantik-heading-judge). ────────


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeRequests:
    def __init__(self, payload):
        self._payload = payload

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        return _FakeResp(self._payload)


def _judge_payload():
    return {
        "choices": [{"message": {"content": '{"levels": {"1": 4}}'},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45},
    }


def test_judge_tap_falls_back_to_run_ledger_when_path_unset(run_ledger):
    content, finish = hj._post_judge_completion(
        base_url="http://x/v1", api_key=None, model="nemotron-3-super",
        messages=[{"role": "user", "content": "hi"}], max_tokens=64,
        timeout=5.0, requests_module=_FakeRequests(_judge_payload()))
    assert finish == "stop" and content is not None
    rows = _rows(run_ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "semantik-heading-judge"
    assert row["model"] == "nemotron-3-super"
    assert row["prompt_tokens"] == 123
    assert row["completion_tokens"] == 45


def test_judge_tap_explicit_path_still_wins(run_ledger, tmp_path, monkeypatch):
    explicit = tmp_path / "explicit_usage.jsonl"
    monkeypatch.setenv("SEMANTIK_LLM_USAGE_PATH", str(explicit))
    hj._post_judge_completion(
        base_url="http://x/v1", api_key=None, model="m",
        messages=[], max_tokens=8, timeout=5.0,
        requests_module=_FakeRequests(_judge_payload()))
    assert explicit.is_file() and len(_rows(explicit)) == 1
    assert not run_ledger.exists()  # never double-writes to the run ledger


def test_judge_tap_no_path_no_run_id_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("SEMANTIK_LLM_USAGE_PATH", raising=False)
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "runs"))
    hj._post_judge_completion(
        base_url="http://x/v1", api_key=None, model="m",
        messages=[], max_tokens=8, timeout=5.0,
        requests_module=_FakeRequests(_judge_payload()))
    assert not (tmp_path / "runs").exists()
