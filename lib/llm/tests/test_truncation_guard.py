"""Unit tests for the provider-agnostic input-truncation tripwire."""
from __future__ import annotations

import pytest

from lib.llm.truncation_guard import (
    check_prompt_fits_window,
    check_prompt_not_truncated,
)
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


# ---------------------------------------------------------------------------
# Defect 3 — mid-size truncation arm (reported ~= window cap but > estimate/2)
# ---------------------------------------------------------------------------
def test_mid_size_arm_catches_8192_cap_on_12k_estimate():
    """The bug the /2 severe arm misses: an 8192-served window on a ~12k
    estimate reports ~8194 (> estimate/2 = 6000, so the severe arm PASSES),
    but the head WAS silently dropped. The mid-size arm — materially below
    the estimate AND sitting at the known window cap — trips it."""
    with pytest.raises(PromptTruncatedError) as exc:
        check_prompt_not_truncated(
            8194, 12000, model_id="qwen2.5:7b", num_ctx=8192
        )
    assert "8192" in str(exc.value)


def test_mid_size_arm_needs_both_conditions():
    """Materially-below alone must NOT trip (the estimate over-counts ~1.6x,
    so a NON-truncated call legitimately reports ~0.6x the estimate). The
    reported count must ALSO sit at the served-window cap."""
    # reported 7000 is materially below 12000 (< 9000) but NOWHERE near the
    # 16384 cap → not consistent with truncation → no-op.
    check_prompt_not_truncated(7000, 12000, model_id="m", num_ctx=16384)


def test_mid_size_arm_noops_when_reported_above_cap_band():
    """A reported count above the cap band is not a window-truncation signal."""
    # reported 9000 vs num_ctx 8192: outside the +/-5% (floor 64) band, and
    # not below estimate*0.75=6600 → no-op.
    check_prompt_not_truncated(9000, 8800, model_id="m", num_ctx=8192)


# ---------------------------------------------------------------------------
# Recalibration (2026-07-09) — the mid-size arm must NOT false-trip a
# legitimate near-full-window prompt, but MUST still trip a report that
# saturated the window cap. The 2.5-c/t estimate over-counts ~1.35x, so a real
# 15.5k-token prompt reports ~15.5k against a 21k UPPER-bound estimate.
# ---------------------------------------------------------------------------
def test_mid_size_arm_no_false_trip_on_legit_full_window_prompt():
    """est 21000 (upper bound), reported 15500, num_ctx 16384 → must NOT trip.
    Real tokenization ~3.38 c/t means calibrated estimate ~15532, so 15500 is
    a legitimate near-full-window prompt that FITS the 16384 window; it sits
    BELOW the cap ceiling (884 below, outside a report>=cap saturation) and is
    NOT materially below the calibrated estimate, so neither arm fires."""
    check_prompt_not_truncated(15500, 21000, model_id="qwen2.5:7b", num_ctx=16384)


def test_mid_size_arm_trips_when_reported_saturates_cap():
    """est 21000, reported 16386, num_ctx 16384 → MUST trip. reported has
    reached/exceeded the window ceiling (16386 >= 16384, within the cap band)
    — the saturation signature of a head-truncation — so it trips even though
    16386 is NOT materially below the (over-counting) 21000 upper bound."""
    with pytest.raises(PromptTruncatedError) as exc:
        check_prompt_not_truncated(16386, 21000, model_id="qwen2.5:7b", num_ctx=16384)
    assert "16384" in str(exc.value)


# ---------------------------------------------------------------------------
# Defect 3 — pre-dispatch deterministic check (CALIBRATED, 2026-07-09).
# The 2.5-c/t estimate over-counts real tokenization by ~1.35x, so the refusal
# converts the estimate with an OPTIMISTIC 3.8-c/t divisor and fires only when
# even that generous reading overflows the window.
# ---------------------------------------------------------------------------
def test_fits_window_trips_when_estimate_exceeds_num_ctx():
    """(b) chars/3.8 > num_ctx STILL refuses. est 26000 → optimistic
    26000*2.5/3.8 ≈ 17105 > 16384 → raise (a real cannot-fit prompt)."""
    with pytest.raises(PromptTruncatedError) as exc:
        check_prompt_fits_window(26000, model_id="qwen2.5:7b", num_ctx=16384)
    msg = str(exc.value)
    assert "26000" in msg and "16384" in msg


def test_fits_window_dispatches_when_upper_bound_high_but_real_fits():
    """(a) est 20000 (2.5-c/t UPPER bound) does NOT refuse: optimistic
    20000*2.5/3.8 ≈ 13157 < 16384, so the prompt whose REAL token count fits
    the 16384 window dispatches instead of being refused. This is the exact
    over-eager refusal the recalibration fixes (est 16k-22k blocks that fit)."""
    check_prompt_fits_window(20000, model_id="qwen2.5:7b", num_ctx=16384)


def test_fits_window_refusal_boundary_at_calibrated_ratio():
    """The refusal boundary is estimated * (2.5/3.8) vs num_ctx. Just below
    the boundary dispatches; just above refuses. For num_ctx=16384 the
    boundary is 16384*3.8/2.5 = 24903.68 upper-bound tokens."""
    check_prompt_fits_window(24000, model_id="m", num_ctx=16384)  # optimistic 15789 → fits
    with pytest.raises(PromptTruncatedError):
        check_prompt_fits_window(25000, model_id="m", num_ctx=16384)  # optimistic 16447 → refuse


def test_fits_window_noops_when_estimate_fits():
    check_prompt_fits_window(8000, model_id="m", num_ctx=8192)  # no raise
    check_prompt_fits_window(8192, model_id="m", num_ctx=8192)  # equal → fits (optimistic < cap)


def test_fits_window_custom_optimistic_divisor():
    """The optimistic divisor is a knob. est 21823 refuses under the raw 2.5
    upper bound (legacy behavior) when the divisor is set to 2.5, but the
    default 3.8 divisor lets it dispatch (real tokens fit)."""
    check_prompt_fits_window(21823, model_id="m", num_ctx=16384)  # default 3.8 → fits
    with pytest.raises(PromptTruncatedError):
        check_prompt_fits_window(
            21823, model_id="m", num_ctx=16384,
            optimistic_chars_per_token=2.5,  # raw upper bound → legacy refuse
        )


def test_fits_window_noops_below_floor_or_unknown_window():
    # Tiny estimate → no assertion.
    check_prompt_fits_window(100, model_id="m", num_ctx=64, min_estimate=256)
    # Unknown / non-positive window → no assertion (can't compare).
    check_prompt_fits_window(50000, model_id="m", num_ctx=0)
    check_prompt_fits_window(50000, model_id="m", num_ctx=-1)
