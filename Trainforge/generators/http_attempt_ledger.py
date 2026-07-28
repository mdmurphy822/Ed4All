"""Durable, secret-free ledger for OpenAI-compatible HTTP attempts.

This module is deliberately transport-only: it records request identity and
outcomes, never request/response bodies or headers.  ``install_on_client`` is
intended for bounded diagnostic pilots and leaves the client's retry policy
unchanged.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the exact JSON request without retaining its potentially private body."""
    return hashlib.sha256(_stable(payload).encode("utf-8")).hexdigest()


def sanitized_endpoint(url: str) -> str:
    """Return endpoint identity without credentials, query parameters, or fragments."""
    split = urlsplit(str(url))
    host = split.hostname or ""
    if split.port is not None:
        host = f"{host}:{split.port}"
    return urlunsplit((split.scheme, host, split.path, "", ""))


class DurableCallIntentManifest:
    """Independent, append-only authority for admitted logical provider calls.

    Rows deliberately contain identities only.  Prompt/response bodies,
    headers, endpoints, and credentials are never retained here.  The append
    is fsynced before transport instrumentation is entered, so a missing
    capture or a wholly missing HTTP started/terminal pair remains detectable.

    ``logical_attempt`` is the call number within one unit+stage.  Attempt one
    is ``initial``; later calls are ``repair``.  The serial dialect probe uses
    kind ``dialect``.  A repair is one new logical call and, under the
    benchmark's one-transport-attempt matrix contract, expects exactly one
    HTTP started row and one terminal row.
    """

    VERSION = "ed4all.provider-call-intent.v1"
    BOUND_VERSION = "ed4all.provider-call-intent.v2"

    def __init__(
        self, path: Path, *, run_id: str, cell_id: str, utc_now: Any = None,
        synthesis_contract_sha256: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = str(run_id)
        self.cell_id = str(cell_id)
        self.synthesis_contract_sha256 = (
            str(synthesis_contract_sha256)
            if synthesis_contract_sha256 is not None else None
        )
        if (
            self.synthesis_contract_sha256 is not None
            and (
                len(self.synthesis_contract_sha256) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in self.synthesis_contract_sha256
                )
            )
        ):
            raise ValueError("synthesis contract SHA-256 is invalid")
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._counts: contextvars.ContextVar[Optional[dict[str, int]]] = (
            contextvars.ContextVar("provider_intent_stage_counts", default=None)
        )
        self._last: contextvars.ContextVar[Optional[dict[str, Any]]] = (
            contextvars.ContextVar("provider_intent_last", default=None)
        )

    @contextlib.contextmanager
    def unit(self) -> Iterator[None]:
        counts_token = self._counts.set({})
        last_token = self._last.set(None)
        try:
            yield
        finally:
            self._last.reset(last_token)
            self._counts.reset(counts_token)

    def admit(
        self, *, unit: str, stage: str, payload: Mapping[str, Any],
        response_dialect: Optional[str] = None,
    ) -> dict[str, Any]:
        counts = self._counts.get()
        if counts is None:
            raise RuntimeError("provider intent manifest requires bound unit")
        logical_attempt = int(counts.get(stage, 0)) + 1
        counts[stage] = logical_attempt
        schema = (
            ((payload.get("response_format") or {}).get("json_schema") or {}).get(
                "schema"
            )
            if isinstance(payload.get("response_format"), Mapping)
            else None
        )
        contract = {
            "stage": stage,
            "model": str(payload.get("model") or ""),
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "response_schema_sha256": hashlib.sha256(
                _stable(schema or {}).encode("utf-8")
            ).hexdigest(),
            "response_dialect": str(response_dialect or ""),
        }
        if self.synthesis_contract_sha256 is not None:
            contract["synthesis_contract_sha256"] = (
                self.synthesis_contract_sha256
            )
        kind = (
            "dialect"
            if stage == "staged_synthesis:dialect_preflight"
            else "initial" if logical_attempt == 1 else "repair"
        )
        row = {
            "intent_version": (
                self.BOUND_VERSION
                if self.synthesis_contract_sha256 is not None else self.VERSION
            ),
            "utc": self._utc_now().astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "unit": str(unit),
            "kind": kind,
            "stage": str(stage),
            "logical_attempt": logical_attempt,
            "request_sha256": request_sha256(payload),
            "model": str(payload.get("model") or ""),
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "response_schema_sha256": contract["response_schema_sha256"],
            "response_dialect": contract["response_dialect"],
            "contract_sha256": hashlib.sha256(
                _stable(contract).encode("utf-8")
            ).hexdigest(),
        }
        if self.synthesis_contract_sha256 is not None:
            row["synthesis_contract_sha256"] = (
                self.synthesis_contract_sha256
            )
        encoded = f"{_stable(row)}\n".encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        self._last.set(dict(row))
        return dict(row)

    def current(self) -> Optional[dict[str, Any]]:
        value = self._last.get()
        return dict(value) if value is not None else None


class DurableHttpAttemptLedger:
    """Append-only JSONL ledger; every event is flushed and fsynced before return."""

    VERSION = "ed4all.http-attempt-ledger.v1"

    def __init__(
        self,
        path: Path,
        *,
        raw_audit_root: Optional[Path] = None,
        utc_now: Any = None,
        monotonic: Any = None,
    ) -> None:
        import time

        self.path = Path(path)
        self.raw_audit_root = (
            Path(raw_audit_root) if raw_audit_root is not None else None
        )
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic
        self._lock = threading.Lock()
        self._unit: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
            "http_attempt_ledger_unit", default=None
        )
        self._stage: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
            "http_attempt_ledger_stage", default=None
        )
        self._attempt: contextvars.ContextVar[int] = contextvars.ContextVar(
            "http_attempt_ledger_attempt", default=0
        )
        self._last_attempt: contextvars.ContextVar[int] = contextvars.ContextVar(
            "http_attempt_ledger_last_attempt", default=0
        )
        self._last_identity: contextvars.ContextVar[Optional[dict[str, Any]]] = (
            contextvars.ContextVar("http_attempt_ledger_last_identity", default=None)
        )
        self._stage_counts: contextvars.ContextVar[Optional[dict[str, int]]] = (
            contextvars.ContextVar("http_attempt_ledger_stage_counts", default=None)
        )
        self._hard_deadline: Optional[float] = None
        self._max_timeout_seconds = 240.0
        self._cleanup_seconds = 30.0
        self._intent_manifest: Optional[DurableCallIntentManifest] = None

    def configure_deadline(
        self, *, hard_deadline: float, max_timeout_seconds: float,
        cleanup_seconds: float = 30.0,
    ) -> None:
        self._hard_deadline = float(hard_deadline)
        self._max_timeout_seconds = float(max_timeout_seconds)
        self._cleanup_seconds = float(cleanup_seconds)

    def resolve_timeout(self) -> float:
        if self._hard_deadline is None:
            return self._max_timeout_seconds
        remaining = (
            self._hard_deadline - float(self._monotonic()) - self._cleanup_seconds
        )
        if remaining <= 0:
            from Trainforge.generators._synthesis_common import (
                SynthesisProviderError,
            )
            raise SynthesisProviderError(
                "benchmark cell has no safe request budget remaining",
                code="cell_request_admission_closed",
            )
        return min(self._max_timeout_seconds, remaining)

    @contextlib.contextmanager
    def unit(self, unit: str) -> Iterator[None]:
        """Bind a stable pilot unit identity for calls made in this context."""
        token = self._unit.set(str(unit))
        counts_token = self._stage_counts.set({})
        intent_context = (
            self._intent_manifest.unit()
            if self._intent_manifest is not None
            else contextlib.nullcontext()
        )
        try:
            with intent_context:
                yield
        finally:
            self._stage_counts.reset(counts_token)
            self._unit.reset(token)

    def current_unit(self) -> str:
        return self._required(self._unit.get(), "unit")

    @contextlib.contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        """Bind a logical staged-synthesis call and reset its attempt counter."""
        counts = self._stage_counts.get()
        if counts is None:
            raise RuntimeError("HTTP attempt ledger requires bound unit")
        initial_attempt = int(counts.get(str(stage), 0))
        stage_token = self._stage.set(str(stage))
        attempt_token = self._attempt.set(initial_attempt)
        last_token = self._last_attempt.set(initial_attempt)
        identity_token = self._last_identity.set(None)
        try:
            yield
        finally:
            counts[str(stage)] = self._attempt.get()
            self._last_identity.reset(identity_token)
            self._last_attempt.reset(last_token)
            self._attempt.reset(attempt_token)
            self._stage.reset(stage_token)

    def record_response(
        self, *, url: str, payload: Mapping[str, Any], response: Any,
        actual_timeout_seconds: float,
    ) -> None:
        attempt = self._last_attempt.get()
        identity = self._required_identity()
        status = int(response.status_code)
        finish_reason = None
        usage: dict[str, int] = {}
        if status == 200:
            try:
                body = response.json()
            except (TypeError, ValueError):
                body = None
            if isinstance(body, Mapping):
                choices = body.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                    value = choices[0].get("finish_reason")
                    finish_reason = str(value) if value is not None else None
                raw_usage = body.get("usage")
                if isinstance(raw_usage, Mapping):
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = raw_usage.get(key)
                        if isinstance(value, int) and not isinstance(value, bool):
                            usage[key] = value
        self._append({
            "event": "http_attempt_terminal",
            "unit": self._required(self._unit.get(), "unit"),
            "stage": self._required(self._stage.get(), "stage"),
            "attempt": attempt,
            **identity,
            "http_status": status,
            "exception_class": None,
            "exception_code": None,
            "retryable": status in {429, 500, 502, 503, 504},
            "finish_reason": finish_reason,
            "usage": usage,
            "actual_timeout_seconds": actual_timeout_seconds,
            "response_raw_ref": self._raw_ref(
                "response", bytes(response.content),
            ),
        })

    def record_exception(
        self, *, url: str, payload: Mapping[str, Any], exception: BaseException,
        actual_timeout_seconds: float,
    ) -> None:
        attempt = self._last_attempt.get()
        identity = self._required_identity()
        code = getattr(exception, "code", None)
        if code is None:
            code = getattr(exception, "errno", None)
        self._append({
            "event": "http_attempt_terminal",
            "unit": self._required(self._unit.get(), "unit"),
            "stage": self._required(self._stage.get(), "stage"),
            "attempt": attempt,
            **identity,
            "http_status": None,
            "exception_class": type(exception).__name__,
            "exception_code": str(code) if code is not None else None,
            "retryable": True,
            "finish_reason": None,
            "usage": {},
            "actual_timeout_seconds": actual_timeout_seconds,
            "response_raw_ref": self._raw_ref("response", b""),
        })

    def record_started(
        self, *, url: str, payload: Mapping[str, Any],
        actual_timeout_seconds: float,
    ) -> None:
        """Fsync request identity before dispatch so hung work is observable."""
        attempt = self._attempt.get() + 1
        self._attempt.set(attempt)
        self._last_attempt.set(attempt)
        identity = {
            "request_sha256": request_sha256(payload),
            "endpoint": sanitized_endpoint(url),
            "model": str(payload.get("model") or ""),
            "request_raw_ref": self._raw_ref(
                "request", _stable(payload).encode("utf-8"),
            ),
        }
        self._last_identity.set(identity)
        self._append({
            "event": "http_attempt_started",
            "unit": self._required(self._unit.get(), "unit"),
            "stage": self._required(self._stage.get(), "stage"),
            "attempt": attempt,
            **identity,
            "http_status": None,
            "exception_class": None,
            "exception_code": None,
            "retryable": None,
            "finish_reason": None,
            "usage": {},
            "actual_timeout_seconds": actual_timeout_seconds,
        })

    def _required_identity(self) -> dict[str, Any]:
        identity = self._last_identity.get()
        if identity is None:
            raise RuntimeError("HTTP attempt ledger terminal has no started attempt")
        return identity

    def record_backoff(self, seconds: float) -> None:
        identity = self._last_identity.get()
        if identity is None:
            raise RuntimeError("HTTP attempt ledger backoff has no preceding attempt")
        self._append({
            "event": "retry_backoff",
            "unit": self._required(self._unit.get(), "unit"),
            "stage": self._required(self._stage.get(), "stage"),
            "attempt": self._last_attempt.get(),
            **identity,
            "backoff_seconds": float(seconds),
        })

    @staticmethod
    def _required(value: Optional[str], name: str) -> str:
        if value is None or not value.strip():
            raise RuntimeError(f"HTTP attempt ledger requires bound {name}")
        return value

    def _append(self, fields: Mapping[str, Any]) -> None:
        now = self._utc_now()
        utc = now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        row = {
            "ledger_version": self.VERSION,
            "utc": utc,
            "monotonic_seconds": float(self._monotonic()),
            **fields,
        }
        payload = f"{_stable(row)}\n".encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

    def _raw_ref(self, kind: str, payload: bytes) -> Optional[dict[str, Any]]:
        if self.raw_audit_root is None:
            return None
        digest = hashlib.sha256(payload).hexdigest()
        root = self.raw_audit_root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = root / f"{kind}-{digest}.bin"
        with self._lock:
            if not path.exists():
                with path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                path.chmod(0o600)
        return {
            "path": str(path),
            "sha256": digest,
            "bytes": len(payload),
        }


class _LedgeredHttpClient:
    def __init__(self, wrapped: Any, ledger: DurableHttpAttemptLedger) -> None:
        self._wrapped = wrapped
        self._ledger = ledger

    def post(self, url: str, *, json: Mapping[str, Any], **kwargs: Any) -> Any:
        actual_timeout = self._ledger.resolve_timeout()
        kwargs["timeout"] = actual_timeout
        self._ledger.record_started(
            url=url, payload=json, actual_timeout_seconds=actual_timeout,
        )
        try:
            response = self._wrapped.post(url, json=json, **kwargs)
        except Exception as exc:
            self._ledger.record_exception(
                url=url, payload=json, exception=exc,
                actual_timeout_seconds=actual_timeout,
            )
            raise
        self._ledger.record_response(
            url=url, payload=json, response=response,
            actual_timeout_seconds=actual_timeout,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def install_on_client(
    openai_client: Any,
    ledger: DurableHttpAttemptLedger,
    *,
    intent_manifest: Optional[DurableCallIntentManifest] = None,
) -> None:
    """Install attempt/backoff instrumentation on an OpenAICompatibleClient.

    The caller binds ``ledger.unit(...)`` around each pilot unit.  Stage names
    are derived from the existing ``post_with_usage(..., task=...)`` value.
    Installation is idempotent for one ledger and does not alter payloads,
    response handling, retry counts, or sleep durations.
    """
    if getattr(openai_client, "_http_attempt_ledger", None) is ledger:
        return
    if getattr(openai_client, "_http_attempt_ledger", None) is not None:
        raise RuntimeError("an HTTP attempt ledger is already installed")
    ledger._intent_manifest = intent_manifest

    openai_client._client = _LedgeredHttpClient(openai_client.client, ledger)
    original_post = openai_client.post_with_usage
    original_sleep = openai_client._sleep_fn

    def post_with_usage(payload: Mapping[str, Any], *, task: str) -> Any:
        effective_payload = dict(payload)
        if intent_manifest is not None:
            # ``OpenAICompatibleClient._post_with_retry_inner`` adds its JSON
            # dialect fields immediately before transport.  Bind the intent
            # to that effective payload, rather than the caller's pre-dialect
            # mapping, so the independently admitted request identity matches
            # the HTTP started/terminal evidence byte for byte.
            effective_payload = openai_client.effective_payload(payload)
            intent_manifest.admit(
                unit=ledger.current_unit(), stage=task,
                payload=effective_payload,
                response_dialect=getattr(
                    openai_client, "_response_dialect", None,
                ),
            )
        with ledger.stage(task):
            return original_post(effective_payload, task=task)

    def sleep(seconds: float) -> None:
        ledger.record_backoff(seconds)
        original_sleep(seconds)

    openai_client.post_with_usage = post_with_usage
    openai_client._sleep_fn = sleep
    openai_client._http_attempt_ledger = ledger
