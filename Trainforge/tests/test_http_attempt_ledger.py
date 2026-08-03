from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx

from Trainforge.generators.providers._openai_compatible_client import OpenAICompatibleClient
from Trainforge.generators.providers.http_attempt_ledger import (
    DurableHttpAttemptLedger,
    install_on_client,
)


def _ledger(tmp_path):
    ticks = iter(float(value) for value in range(10, 30))
    return DurableHttpAttemptLedger(
        tmp_path / "attempts.jsonl",
        utc_now=lambda: datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
        monotonic=lambda: next(ticks),
    )


def _rows(ledger):
    return [
        json.loads(line)
        for line in ledger.path.read_text(encoding="utf-8").splitlines()
    ]


def test_records_each_http_attempt_and_exact_backoff_without_bodies(tmp_path):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="secret response")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
            },
        })

    client = OpenAICompatibleClient(
        base_url="http://user:password@localhost:8000/v1",
        model="served-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=2,
        initial_backoff_seconds=0,
        sleep_fn=lambda _seconds: None,
    )
    ledger = _ledger(tmp_path)
    install_on_client(client, ledger)
    payload = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "private prompt"}],
    }
    with ledger.unit("unit-7"):
        body, retries = client.post_with_usage(payload, task="staged_synthesis:plan")

    assert retries == 1
    assert body["usage"]["total_tokens"] == 6
    rows = _rows(ledger)
    assert [row["event"] for row in rows] == [
        "http_attempt_started", "http_attempt_terminal",
        "retry_backoff", "http_attempt_started", "http_attempt_terminal",
    ]
    assert [rows[1]["http_status"], rows[4]["http_status"]] == [503, 200]
    assert rows[0]["attempt"] == rows[1]["attempt"] == rows[2]["attempt"] == 1
    assert rows[3]["attempt"] == rows[4]["attempt"] == 2
    assert rows[2]["backoff_seconds"] == 0
    assert rows[4]["finish_reason"] == "stop"
    assert rows[4]["usage"]["total_tokens"] == 6
    assert rows[0]["endpoint"] == "http://localhost:8000/v1/chat/completions"
    serialized = ledger.path.read_text(encoding="utf-8")
    assert "private prompt" not in serialized
    assert "secret response" not in serialized
    assert "password" not in serialized
    assert rows[0]["request_sha256"] == rows[4]["request_sha256"]


def test_records_sanitized_exception_class_and_code(tmp_path):
    class SecretTransportError(httpx.TransportError):
        code = "socket_reset"

    def handler(_request):
        raise SecretTransportError("contains-secret")

    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",
        model="served-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
    )
    ledger = _ledger(tmp_path)
    install_on_client(client, ledger)
    with ledger.unit("unit-9"):
        try:
            client.post_with_usage(
                {"model": "served-model", "messages": []},
                task="staged_synthesis:realization",
            )
        except Exception:
            pass

    rows = _rows(ledger)
    assert rows[0]["event"] == "http_attempt_started"
    row = rows[1]
    assert row["http_status"] is None
    assert row["exception_class"] == "SecretTransportError"
    assert row["exception_code"] == "socket_reset"
    assert "contains-secret" not in ledger.path.read_text(encoding="utf-8")


def test_requires_unit_binding_before_transport(tmp_path):
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",
        model="served-model",
        client=httpx.Client(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"choices": []})
        )),
        max_retries=1,
    )
    ledger = _ledger(tmp_path)
    install_on_client(client, ledger)
    try:
        client.post_with_usage(
            {"model": "served-model", "messages": []},
            task="staged_synthesis:plan",
        )
    except RuntimeError as exc:
        assert "bound unit" in str(exc)
    else:
        raise AssertionError("missing unit binding must fail loudly")


def test_attempt_numbers_continue_across_same_stage_repair_calls(tmp_path):
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",
        model="served-model",
        client=httpx.Client(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {},
            })
        )),
        max_retries=1,
    )
    ledger = _ledger(tmp_path)
    install_on_client(client, ledger)
    payload = {"model": "served-model", "messages": []}
    with ledger.unit("unit-repair"):
        client.post_with_usage(payload, task="staged_synthesis:plan")
        client.post_with_usage(payload, task="staged_synthesis:plan")
    rows = _rows(ledger)
    assert [row["attempt"] for row in rows if row["event"].endswith("started")] == [1, 2]


def test_attempt_started_is_fsynced_before_transport_dispatch(tmp_path):
    ledger = _ledger(tmp_path)

    def handler(_request):
        rows = _rows(ledger)
        assert len(rows) == 1
        assert rows[0]["event"] == "http_attempt_started"
        raise httpx.TransportError("stop after durable proof")

    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",
        model="served-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
    )
    install_on_client(client, ledger)
    with ledger.unit("unit-started"):
        try:
            client.post_with_usage(
                {"model": "served-model", "messages": []},
                task="staged_synthesis:plan",
            )
        except Exception:
            pass
    assert [row["event"] for row in _rows(ledger)] == [
        "http_attempt_started", "http_attempt_terminal",
    ]


def test_deadline_caps_actual_request_timeout_with_cleanup_reserve(tmp_path):
    now = [100.0]
    observed = []

    def handler(request):
        observed.append(request)
        return httpx.Response(200, json={"choices": [], "usage": {}})

    ledger = DurableHttpAttemptLedger(
        tmp_path / "attempts.jsonl",
        monotonic=lambda: now[0],
    )
    ledger.configure_deadline(
        hard_deadline=720.0, max_timeout_seconds=240.0, cleanup_seconds=30.0,
    )
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1", model="served",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
    )
    install_on_client(client, ledger)
    now[0] = 500.0
    with ledger.unit("deadline"):
        client.post_with_usage(
            {"model": "served", "messages": []}, task="stage",
        )
    assert observed
    rows = _rows(ledger)
    assert rows[0]["actual_timeout_seconds"] == 190.0
    assert rows[1]["actual_timeout_seconds"] == 190.0

    now[0] = 691.0
    with ledger.unit("closed"):
        try:
            client.post_with_usage(
                {"model": "served", "messages": []}, task="stage",
            )
        except Exception as exc:
            assert getattr(exc, "code", None) == "cell_request_admission_closed"
        else:
            raise AssertionError("request crossed cleanup reserve")


def test_raw_audit_refs_are_exact_and_ledger_remains_body_free(tmp_path):
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1", model="served",
        client=httpx.Client(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={
                "choices": [{"message": {"content": "private response"}}],
                "usage": {},
            })
        )),
        max_retries=1,
    )
    ledger = DurableHttpAttemptLedger(
        tmp_path / "attempts.jsonl", raw_audit_root=tmp_path / "raw",
    )
    install_on_client(client, ledger)
    with ledger.unit("raw"):
        client.post_with_usage(
            {"model": "served", "messages": [
                {"role": "user", "content": "private prompt"},
            ]}, task="stage",
        )
    rows = _rows(ledger)
    for row, field in ((rows[0], "request_raw_ref"), (rows[1], "response_raw_ref")):
        ref = row[field]
        payload = __import__("pathlib").Path(ref["path"]).read_bytes()
        assert len(payload) == ref["bytes"]
        assert __import__("hashlib").sha256(payload).hexdigest() == ref["sha256"]
    serialized = ledger.path.read_text(encoding="utf-8")
    assert "private prompt" not in serialized
    assert "private response" not in serialized
