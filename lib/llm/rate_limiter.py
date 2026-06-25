"""Cloud-provider admission control — RPM / TPM token buckets + concurrency.

SETUP SCAFFOLD (ships DARK / default-OFF). This module exists so the
"70B-everywhere" NVIDIA build mode (see
``plans/finegrain/nvidia-70b-everywhere-2026-06.md``) has a shared
admission gate ready to wire when the operator supplies real RPM/TPM
ceilings + pricing. **Nothing dispatches to NVIDIA from here** — the
limiter only sleeps/admits; it never makes a network call.

Three admission axes (all OFF unless ``ED4ALL_CLOUD_RATE_LIMIT`` is
truthy AND a ceiling is configured):

* **RPM bucket** — a classic token bucket refilling at ``rpm/60`` tokens
  per second; one request debits one token.
* **TPM bucket** — a token bucket refilling at ``tpm/60`` per second.
  A request debits an ESTIMATE of its token cost up front, then the
  caller may ``reconcile`` against the server-reported ``usage`` so the
  bucket tracks real consumption rather than the estimate.
* **Concurrency semaphore** — a hard ceiling on simultaneous in-flight
  requests.

Default-OFF contract: when the env flag is unset/falsey OR no ceiling is
configured, :func:`get_admission_gate` returns a :class:`_NullGate` whose
``acquire`` / ``reconcile`` are pure no-ops, so a default run is
byte-identical (no sleeps, no state, no import cost on the hot path).

Parse-with-fallback: a garbage env value resolves OFF; a garbage ceiling
falls back to "unset" (no ceiling on that axis).

Cross-process note (documented, not solved here): an in-process singleton
gate does NOT coordinate separate OS processes that share one API key.
Concurrent ``ed4all run`` sessions against the same NVIDIA key would each
get their own bucket. A real multi-session deploy needs flock/IPC
coordination — left as a documented TODO; the single-session slice test
this scaffold targets does not need it.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

__all__ = [
    "RATE_LIMIT_ENV",
    "RateLimitConfig",
    "TokenBucket",
    "AdmissionGate",
    "is_rate_limit_enabled",
    "resolve_rate_limit_config",
    "get_admission_gate",
    "reset_admission_gate",
]

# The single master switch. Default OFF — a default run never builds a
# real gate (mirrors LOCAL_DISPATCHER_ALLOW_STUB / ED4ALL_CLOUD_*).
RATE_LIMIT_ENV = "ED4ALL_CLOUD_RATE_LIMIT"

# Per-axis ceiling envs. ALL unset by default — no hardcoded RPM/TPM
# guess (the plan is explicit: ceilings are operator-supplied at run
# time, not baked into the scaffold).
_RPM_ENV = "ED4ALL_CLOUD_RPM"
_TPM_ENV = "ED4ALL_CLOUD_TPM"
_CONCURRENCY_ENV = "ED4ALL_CLOUD_MAX_CONCURRENCY"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def is_rate_limit_enabled(env: Optional[dict] = None) -> bool:
    """True iff ``ED4ALL_CLOUD_RATE_LIMIT`` is truthy (parse-with-fallback)."""
    source = env if env is not None else os.environ
    return str(source.get(RATE_LIMIT_ENV, "")).strip().lower() in _TRUTHY


def _parse_positive_float(raw: Optional[str]) -> Optional[float]:
    """Parse a positive float; garbage / non-positive / None → None."""
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value <= 0 or value != value:  # non-positive or NaN
        return None
    return value


def _parse_positive_int(raw: Optional[str]) -> Optional[int]:
    """Parse a positive int; garbage / non-positive / None → None."""
    parsed = _parse_positive_float(raw)
    if parsed is None:
        return None
    return int(parsed)


@dataclass(frozen=True)
class RateLimitConfig:
    """Resolved admission-control ceilings. All axes optional (None=off)."""

    enabled: bool = False
    rpm: Optional[float] = None
    tpm: Optional[float] = None
    max_concurrency: Optional[int] = None

    @property
    def any_ceiling(self) -> bool:
        return (
            self.rpm is not None
            or self.tpm is not None
            or self.max_concurrency is not None
        )

    @property
    def active(self) -> bool:
        """A real gate is built only when the switch is ON and ≥1 ceiling set."""
        return self.enabled and self.any_ceiling


def resolve_rate_limit_config(env: Optional[dict] = None) -> RateLimitConfig:
    """Resolve the admission config from the environment (parse-with-fallback)."""
    source = env if env is not None else os.environ
    return RateLimitConfig(
        enabled=is_rate_limit_enabled(source),
        rpm=_parse_positive_float(source.get(_RPM_ENV)),
        tpm=_parse_positive_float(source.get(_TPM_ENV)),
        max_concurrency=_parse_positive_int(source.get(_CONCURRENCY_ENV)),
    )


class TokenBucket:
    """A thread-safe token bucket with continuous refill.

    ``capacity`` tokens, refilling at ``refill_per_sec`` tokens/second.
    :meth:`acquire` blocks (sleeping via the injectable ``sleep_fn``)
    until ``amount`` tokens are available, then debits them. A
    monotonic clock is injectable for deterministic tests.
    """

    def __init__(
        self,
        capacity: float,
        refill_per_sec: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._capacity = float(capacity)
        self._refill = float(refill_per_sec)
        self._tokens = float(capacity)
        self._clock = clock
        self._sleep = sleep_fn
        self._last = clock()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        if self._refill > 0:
            self._tokens = min(
                self._capacity, self._tokens + elapsed * self._refill
            )

    def acquire(self, amount: float = 1.0) -> float:
        """Block until ``amount`` tokens are available, then debit them.

        Returns the total seconds slept (0.0 when admitted immediately).
        An ``amount`` above capacity is clamped to capacity so a single
        oversized request can never deadlock the bucket.
        """
        amount = min(float(amount), self._capacity) if self._capacity else 0.0
        slept = 0.0
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= amount or self._refill <= 0:
                    self._tokens -= amount
                    return slept
                deficit = amount - self._tokens
                wait = deficit / self._refill
            self._sleep(wait)
            slept += wait

    def reconcile(self, estimated: float, actual: float) -> None:
        """Adjust the bucket for the gap between an estimate and reality.

        Debit the extra when ``actual > estimated`` (we under-charged),
        credit back when ``actual < estimated``. Keeps the TPM bucket
        tracking server-reported usage instead of the up-front estimate.
        """
        delta = float(actual) - float(estimated)
        with self._lock:
            self._refill_locked()
            self._tokens = max(
                -self._capacity, min(self._capacity, self._tokens - delta)
            )

    @property
    def tokens(self) -> float:
        with self._lock:
            self._refill_locked()
            return self._tokens


class _Reservation:
    """Returned by :meth:`AdmissionGate.acquire`; releases the semaphore."""

    def __init__(
        self,
        gate: "AdmissionGate",
        estimated_tokens: float,
    ) -> None:
        self._gate = gate
        self._estimated = estimated_tokens
        self._reconciled = False

    def reconcile(self, actual_tokens: float) -> None:
        """Reconcile the TPM bucket against server-reported usage (once)."""
        if self._reconciled:
            return
        self._reconciled = True
        self._gate._reconcile(self._estimated, actual_tokens)

    def __enter__(self) -> "_Reservation":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        self._gate._release()


class AdmissionGate:
    """Composes RPM + TPM buckets + a concurrency semaphore.

    :meth:`acquire` debits one RPM token + ``estimated_tokens`` TPM
    tokens and takes a concurrency slot, blocking as needed. The returned
    reservation MUST be released (use it as a context manager) and may be
    reconciled with the server-reported usage.
    """

    def __init__(
        self,
        config: RateLimitConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._rpm = (
            TokenBucket(
                capacity=config.rpm,
                refill_per_sec=config.rpm / 60.0,
                clock=clock,
                sleep_fn=sleep_fn,
            )
            if config.rpm is not None
            else None
        )
        self._tpm = (
            TokenBucket(
                capacity=config.tpm,
                refill_per_sec=config.tpm / 60.0,
                clock=clock,
                sleep_fn=sleep_fn,
            )
            if config.tpm is not None
            else None
        )
        self._semaphore = (
            threading.BoundedSemaphore(config.max_concurrency)
            if config.max_concurrency is not None
            else None
        )

    def acquire(self, estimated_tokens: float = 0.0) -> _Reservation:
        """Block until admitted; debit RPM + estimated TPM; take a slot."""
        if self._semaphore is not None:
            self._semaphore.acquire()
        if self._rpm is not None:
            self._rpm.acquire(1.0)
        if self._tpm is not None and estimated_tokens > 0:
            self._tpm.acquire(float(estimated_tokens))
        return _Reservation(self, float(estimated_tokens))

    def _reconcile(self, estimated: float, actual: float) -> None:
        if self._tpm is not None:
            self._tpm.reconcile(estimated, actual)

    def _release(self) -> None:
        if self._semaphore is not None:
            try:
                self._semaphore.release()
            except ValueError:
                # Defensive: double-release is a no-op rather than a crash.
                pass


class _NullGate:
    """No-op admission gate used when rate limiting is OFF (byte-stable)."""

    def acquire(self, estimated_tokens: float = 0.0) -> "_NullReservation":
        return _NullReservation()


class _NullReservation:
    def reconcile(self, actual_tokens: float) -> None:  # noqa: D401
        return None

    def __enter__(self) -> "_NullReservation":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def release(self) -> None:
        return None


# Process-level singleton so every composed authoring seat shares ONE
# gate (the whole point of admission control). Resolved lazily from the
# environment on first use; ``reset_admission_gate`` re-resolves (tests).
_GATE_SINGLETON: Optional[object] = None
_GATE_LOCK = threading.Lock()


def get_admission_gate(
    env: Optional[dict] = None,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> object:
    """Return the shared admission gate (a no-op gate when rate-limiting is OFF).

    Default-OFF → :class:`_NullGate` (no buckets, no state). When
    ``ED4ALL_CLOUD_RATE_LIMIT`` is truthy AND ≥1 ceiling env is set, a
    real :class:`AdmissionGate` is built once and cached process-wide.
    """
    global _GATE_SINGLETON
    config = resolve_rate_limit_config(env)
    if not config.active:
        # Default path — pure no-op. Never cached, never holds state.
        return _NullGate()
    with _GATE_LOCK:
        if _GATE_SINGLETON is None or isinstance(_GATE_SINGLETON, _NullGate):
            _GATE_SINGLETON = AdmissionGate(
                config, clock=clock, sleep_fn=sleep_fn
            )
        return _GATE_SINGLETON


def reset_admission_gate() -> None:
    """Drop the cached singleton (test isolation)."""
    global _GATE_SINGLETON
    with _GATE_LOCK:
        _GATE_SINGLETON = None
