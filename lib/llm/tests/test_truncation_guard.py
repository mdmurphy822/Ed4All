"""Unit tests for the provider-agnostic input-truncation tripwire."""
from __future__ import annotations

import pytest

from lib.llm.truncation_guard import check_prompt_not_truncated
from lib.retrieval.answer_backend import PromptTruncatedError


def test_trips_on_large_shortfall():
    """reported << 0.5 * estimate (above the floor) → raise."""
    with pytest.raises(PromptTruncatedError) as exc:
        check_prompt_not_truncated(
            200, 8800, model_id="qwen2.5:7b", num_ctx=8192
        )
    msg = str(exc.value)
    assert "qwen2.5:7b" in msg
    assert "num_ctx=8192" in msg
    # Names BOTH operator num_ctx envs.
    assert "ED4ALL_REWRITE_NUM_CTX" in msg
    assert "ED4ALL_ANSWER_NUM_CTX" in msg


def test_noops_when_reported_matches_estimate():
    """reported close to the estimate → no raise."""
    check_prompt_not_truncated(
        8700, 8800, model_id="m", num_ctx=8192
    )  # must not raise


def test_noops_on_absent_usage():
    """None / non-numeric reported → no signal → no-op."""
    check_prompt_not_truncated(None, 8800, model_id="m", num_ctx=8192)
    check_prompt_not_truncated("not-a-number", 8800, model_id="m", num_ctx=8192)
    check_prompt_not_truncated({}, 8800, model_id="m", num_ctx=8192)


def test_noops_on_zero_or_negative_reported():
    """server omitted / zeroed usage → no-op (not a truncation signal)."""
    check_prompt_not_truncated(0, 8800, model_id="m", num_ctx=8192)
    check_prompt_not_truncated(-5, 8800, model_id="m", num_ctx=8192)


def test_noops_below_min_estimate_floor():
    """A tiny estimate (< floor) can't be truncated → no raise even at 1."""
    check_prompt_not_truncated(
        1, 100, model_id="m", num_ctx=4096, min_estimate=256
    )


def test_no_stale_cross_provider_signal():
    """A genuine empty/absent usage from a DIFFERENT provider call never
    trips: the helper reads ONLY the value passed in (no shared mutable
    last_usage to go stale across providers)."""
    # Simulate an Anthropic call that returns no usage dict — the rewrite
    # tier passes the value through (here None / {} ) so no prior OAI count
    # leaks in.
    check_prompt_not_truncated(None, 9000, model_id="claude", num_ctx=200000)
    check_prompt_not_truncated({}, 9000, model_id="claude", num_ctx=200000)


def test_custom_fraction_and_floor():
    """The fraction / floor knobs are honoured."""
    # reported 600 vs estimate 1000: NOT < 0.5*1000=500, so default no-op.
    check_prompt_not_truncated(600, 1000, model_id="m", num_ctx=8192)
    # With a tighter 0.7 fraction, 600 < 700 → trips.
    with pytest.raises(PromptTruncatedError):
        check_prompt_not_truncated(
            600, 1000, model_id="m", num_ctx=8192, reported_fraction=0.7
        )
