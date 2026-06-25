"""NVIDIA-70b-everywhere — rate-limiter scaffold tests (no network).

The admission gate ships DARK (default off). These tests prove:
  - default-OFF is a pure no-op (no buckets, no sleeps, byte-stable);
  - when ON, the RPM + TPM buckets + concurrency semaphore admit/block as
    expected, driven by a FAKE clock + sleep_fn (no wall-clock, no network).
"""

from __future__ import annotations

import lib.llm.rate_limiter as rl


# ---------------------------------------------------------------------------
# Default-OFF — pure no-op
# ---------------------------------------------------------------------------


def test_default_off_returns_null_gate():
    """No env set → _NullGate; acquire/reconcile/release are no-ops."""
    rl.reset_admission_gate()
    gate = rl.get_admission_gate(env={})
    assert isinstance(gate, rl._NullGate)
    res = gate.acquire(estimated_tokens=99999)
    # No exception, no state — the reservation is inert.
    res.reconcile(123456)
    res.release()
    with gate.acquire(5) as r:
        r.reconcile(5)


def test_flag_on_but_no_ceiling_is_still_off():
    """ED4ALL_CLOUD_RATE_LIMIT on but NO ceiling configured → still no-op."""
    rl.reset_admission_gate()
    config = rl.resolve_rate_limit_config(env={"ED4ALL_CLOUD_RATE_LIMIT": "true"})
    assert config.enabled is True
    assert config.active is False  # no ceiling → not active
    gate = rl.get_admission_gate(env={"ED4ALL_CLOUD_RATE_LIMIT": "true"})
    assert isinstance(gate, rl._NullGate)


def test_garbage_flag_resolves_off():
    """Garbage env value → OFF (parse-with-fallback)."""
    assert rl.is_rate_limit_enabled({"ED4ALL_CLOUD_RATE_LIMIT": "garbage"}) is False
    assert rl.is_rate_limit_enabled({"ED4ALL_CLOUD_RATE_LIMIT": ""}) is False
    assert rl.is_rate_limit_enabled({"ED4ALL_CLOUD_RATE_LIMIT": "1"}) is True


def test_garbage_ceiling_falls_back_to_unset():
    config = rl.resolve_rate_limit_config(
        env={
            "ED4ALL_CLOUD_RATE_LIMIT": "true",
            "ED4ALL_CLOUD_RPM": "notanumber",
            "ED4ALL_CLOUD_TPM": "-5",
            "ED4ALL_CLOUD_MAX_CONCURRENCY": "0",
        }
    )
    assert config.rpm is None
    assert config.tpm is None
    assert config.max_concurrency is None
    assert config.active is False


# ---------------------------------------------------------------------------
# Token bucket — fake clock
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_token_bucket_admits_until_empty_then_blocks():
    clock = _FakeClock()
    slept = []
    bucket = rl.TokenBucket(
        capacity=3, refill_per_sec=1.0, clock=clock, sleep_fn=lambda s: (slept.append(s), clock.advance(s)),
    )
    # First 3 admit instantly (capacity=3).
    assert bucket.acquire(1) == 0.0
    assert bucket.acquire(1) == 0.0
    assert bucket.acquire(1) == 0.0
    assert not slept
    # 4th must wait ~1s (refill 1/s).
    waited = bucket.acquire(1)
    assert waited == 1.0
    assert slept == [1.0]


def test_token_bucket_refills_over_time():
    clock = _FakeClock()
    bucket = rl.TokenBucket(capacity=10, refill_per_sec=2.0, clock=clock, sleep_fn=lambda s: clock.advance(s))
    bucket.acquire(10)  # drain
    assert bucket.tokens == 0.0
    clock.advance(3)  # 3s * 2/s = 6 tokens
    assert bucket.tokens == 6.0


def test_token_bucket_reconcile_debits_extra():
    clock = _FakeClock()
    bucket = rl.TokenBucket(capacity=100, refill_per_sec=0.0, clock=clock, sleep_fn=lambda s: None)
    bucket.acquire(20)  # estimate 20
    assert bucket.tokens == 80.0
    bucket.reconcile(estimated=20, actual=30)  # actual was higher → debit 10 more
    assert bucket.tokens == 70.0
    bucket.reconcile(estimated=20, actual=5)  # over-charged → credit back 15
    assert bucket.tokens == 85.0


# ---------------------------------------------------------------------------
# Admission gate — composed buckets + semaphore
# ---------------------------------------------------------------------------


def test_admission_gate_rpm_blocks():
    clock = _FakeClock()
    slept = []
    config = rl.RateLimitConfig(enabled=True, rpm=60.0)  # 1/sec
    gate = rl.AdmissionGate(
        config, clock=clock, sleep_fn=lambda s: (slept.append(s), clock.advance(s)),
    )
    gate.acquire().release()  # first instant
    gate.acquire().release()  # capacity=60 so still instant
    assert not slept


def test_admission_gate_tpm_reconcile():
    clock = _FakeClock()
    config = rl.RateLimitConfig(enabled=True, tpm=1000.0)
    gate = rl.AdmissionGate(config, clock=clock, sleep_fn=lambda s: clock.advance(s))
    res = gate.acquire(estimated_tokens=100)
    assert gate._tpm.tokens == 900.0
    res.reconcile(actual_tokens=150)  # used more than estimated
    assert gate._tpm.tokens == 850.0
    res.release()


def test_admission_gate_concurrency_semaphore():
    config = rl.RateLimitConfig(enabled=True, max_concurrency=2)
    gate = rl.AdmissionGate(config)
    r1 = gate.acquire()
    r2 = gate.acquire()
    # Semaphore is bounded at 2 — both held. Release restores capacity.
    r1.release()
    r2.release()
    # Double-release is a no-op, not a crash.
    r1.release()


def test_get_admission_gate_active_singleton():
    rl.reset_admission_gate()
    env = {"ED4ALL_CLOUD_RATE_LIMIT": "true", "ED4ALL_CLOUD_RPM": "120"}
    g1 = rl.get_admission_gate(env=env)
    g2 = rl.get_admission_gate(env=env)
    assert isinstance(g1, rl.AdmissionGate)
    assert g1 is g2  # cached process-wide
    rl.reset_admission_gate()
