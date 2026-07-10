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
# Mid-size truncation arm (redesigned 2026-07-10). A mid-size clip now requires
# POSITIVE evidence, NOT coincidence with the assumed window:
#   (1) the assembled estimate overflows the assumed window (est >= 1.1*num_ctx)
#   (2) reported is pinned at the ceiling from below ([0.98, 1.02]*num_ctx)
# ---------------------------------------------------------------------------
def test_mid_size_arm_catches_8192_cap_on_12k_estimate():
    """The bug the /2 severe arm misses: an 8192-served window on a ~12k
    estimate reports ~8194 (> estimate/2 = 6000, so the severe arm PASSES),
    but the head WAS silently dropped. The mid-size arm trips it — the estimate
    (12000) overflows 1.1*8192=9011 AND 8194 is inside [0.98, 1.02]*8192."""
    with pytest.raises(PromptTruncatedError) as exc:
        check_prompt_not_truncated(
            8194, 12000, model_id="qwen2.5:7b", num_ctx=8192
        )
    assert "8192" in str(exc.value)


def test_mid_size_arm_needs_both_conditions():
    """Condition 2 alone (reported near a window) must NOT trip when the
    estimate does not overflow the assumed window — a clip is not even
    plausible."""
    # est 12000 does NOT clear 1.1*16384=18022 (condition 1 fails), so even
    # though 7000 is far from the 16384 cap the arm is off regardless → no-op.
    check_prompt_not_truncated(7000, 12000, model_id="m", num_ctx=16384)


def test_mid_size_arm_noops_when_estimate_fits_under_window():
    """A prompt whose estimate fits comfortably under num_ctx can NEVER
    mid-size-trip, no matter where `reported` lands (incl. right at the cap)."""
    # est 8000 < 1.1*8192=9011 → condition 1 fails; reported 8192 sits exactly
    # at the cap but is irrelevant because the prompt could not have overflowed.
    check_prompt_not_truncated(8192, 8000, model_id="m", num_ctx=8192)


def test_mid_size_arm_noops_when_reported_above_cap_band():
    """A reported count above the [0.98, 1.02] ceiling band is not a
    window-truncation signal even when the estimate overflows."""
    # est 12000 overflows 1.1*8192=9011 (condition 1 met) but reported 9000 is
    # above 1.02*8192=8356 → condition 2 fails → no-op.
    check_prompt_not_truncated(9000, 12000, model_id="m", num_ctx=8192)


# ---------------------------------------------------------------------------
# Live FALSE-POSITIVE kill (2026-07-10). The observed FP: a HEALTHY 16k server,
# an ASSUMED window of 4096, true prompts ~4.0-4.3k tokens whose 2.5-c/t
# estimate over-shoots 4096. Condition 1 (estimate over-window) is satisfied,
# so the ONLY separator is where `reported` falls vs the ceiling band. Reports
# in the tails of the true-size distribution (strictly below 0.98*num_ctx or
# above 1.02*num_ctx) must NOT trip. A 9637-token probe round-trips unclipped
# on the real 16k server, confirming nothing was actually truncated.
# ---------------------------------------------------------------------------
def test_live_fp_reported_below_band_does_not_trip():
    """True ~3.9k prompt, assumed window 4096, estimate ~5400 (over-window).
    reported 3900 is strictly BELOW 0.98*4096=4014.08 → NOT pinned at the
    ceiling → must NOT trip (the old rule false-fired here)."""
    check_prompt_not_truncated(3900, 5400, model_id="qwen2.5:7b", num_ctx=4096)


def test_live_fp_reported_above_band_does_not_trip():
    """True ~4.3k prompt, assumed window 4096, estimate ~5400 (over-window).
    reported 4300 is ABOVE 1.02*4096=4177.92 → outside the ceiling band →
    must NOT trip (a healthy 16k server whose true size just exceeded the
    assumed 4096 cap)."""
    check_prompt_not_truncated(4300, 5400, model_id="qwen2.5:7b", num_ctx=4096)


# ---------------------------------------------------------------------------
# True-POSITIVE detection preserved.
# ---------------------------------------------------------------------------
def test_mid_size_arm_trips_when_reported_saturates_cap():
    """est 21000, reported 16386, num_ctx 16384 → MUST trip. estimate overflows
    1.1*16384=18022 AND 16386 is inside [0.98, 1.02]*16384 (the ceiling band) —
    the signature of a head-clip."""
    with pytest.raises(PromptTruncatedError) as exc:
        check_prompt_not_truncated(16386, 21000, model_id="qwen2.5:7b", num_ctx=16384)
    assert "16384" in str(exc.value)


def test_tp_clipped_prompt_estimate_1_5x_reported_at_cap_trips():
    """The canonical genuine clip: estimate 1.5*num_ctx, reported == num_ctx →
    MUST still trip. est 6144 overflows 1.1*4096=4505.6 AND reported 4096 sits
    exactly in [0.98, 1.02]*4096."""
    with pytest.raises(PromptTruncatedError) as exc:
        check_prompt_not_truncated(4096, 6144, model_id="qwen2.5:7b", num_ctx=4096)
    assert "num_ctx=4096" in str(exc.value)


def test_tp_severe_arm_untouched():
    """The historical severe case (reported < estimate/2) must still trip,
    independent of any window/cap consideration."""
    with pytest.raises(PromptTruncatedError):
        check_prompt_not_truncated(2000, 5000, model_id="m", num_ctx=16384)


def test_residual_fp_band_is_documented_and_still_trips():
    """HONEST residual: an honestly-served prompt whose TRUE size lands exactly
    at the assumed cap is INDISTINGUISHABLE from a clip and still trips. This
    pins the documented inner-band limit so a future narrowing is a deliberate
    change, not a silent one. est 5400 over 1.1*4096, reported 4096 in band."""
    with pytest.raises(PromptTruncatedError) as exc:
        check_prompt_not_truncated(4096, 5400, model_id="m", num_ctx=4096)
    # Message surfaces the estimate/reported ratio + the larger-real-window
    # caveat so an operator can spot the ambiguity.
    msg = str(exc.value)
    assert "estimate/reported" in msg
    assert "ACTUAL window is LARGER" in msg


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
