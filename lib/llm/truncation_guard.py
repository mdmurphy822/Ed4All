"""Provider-agnostic input-prompt truncation tripwire.

A local OpenAI-compatible server (Ollama, vLLM, llama.cpp, …) serves a
fixed ``num_ctx`` window and SILENTLY truncates the prompt HEAD when the
rendered prompt exceeds it — the system prompt + leading tokens vanish
and the model authors from the surviving tail with no rules / no
allowed-id list, fabricating output that the downstream gates then
re-stamp green. The only signal is the server-reported
``usage.prompt_tokens``: a count far BELOW the local estimate means the
head was dropped.

This module hoists the tripwire originally written for the grounded-
answer path (``lib/retrieval/answer_composer.py::_check_prompt_not_truncated``)
into a provider-agnostic helper so the Courseforge rewrite tier (and any
future content-gen call site) can reuse the exact same detection without
re-deriving the constants. The answer path's wrapper now delegates here,
preserving byte-equivalent behaviour (same 0.5 / 256 constants, same
message intent).

The check is a GUARD, not a requirement:

- **Fail-OPEN on a missing signal.** When usage is absent / zero / non-
  numeric (Ollama may omit it), the helper no-ops — a missing signal must
  never block a valid call.
- **Conservative.** Only a LARGE shortfall trips it (reported <
  ``estimated * reported_fraction`` AND the estimate clears the absolute
  ``min_estimate`` floor). The estimate is a 2.5-char/token UPPER bound, so
  the helper errs toward NOT false-tripping.
"""
from __future__ import annotations

__all__ = [
    "check_prompt_not_truncated",
    "check_prompt_fits_window",
    "UPPER_BOUND_CHARS_PER_TOKEN",
    "MEASURED_CHARS_PER_TOKEN",
    "PREDISPATCH_OPTIMISTIC_CHARS_PER_TOKEN",
]


# ---------------------------------------------------------------------------
# Char-per-token calibration (2026-07-09 live measurement).
# ---------------------------------------------------------------------------
#
# ``lib.retrieval._prompts.estimate_tokens`` divides char length by 2.5 — an
# INTENTIONAL UPPER bound on the token count (a math/symbol-dense corpus packs
# ~2.5 chars/token; prose packs ~4). A 2026-07-09 live measurement of this
# project's math-heavy algebra corpus (accessible HTML / text) against
# qwen2.5-7b-16k via Ollama ``/v1`` — comparing the 2.5-c/t estimate to the
# server-reported ``usage.prompt_tokens`` — found REAL tokenization runs at
# ~3.38 chars/token, i.e. the 2.5 estimate OVERCOUNTS the real prompt by
# ~1.35x. Refusing dispatch on that raw upper bound rejected hundreds of
# blocks whose REAL token count fit the served window (397/659 rewrite blocks
# in the 2026-07-08 run were refused pre-dispatch and mass-fell-back to
# template rendering). The two guards below therefore CALIBRATE against this
# measurement instead of trusting the raw 2.5-c/t upper bound:
#
# - the PRE-dispatch refusal (:func:`check_prompt_fits_window`) converts the
#   estimate with an OPTIMISTIC divisor chosen ABOVE the measured 3.38 c/t, so
#   a refusal fires ONLY when even a generous reading of the prompt cannot
#   fit;
# - the POST-dispatch mid-size arm (:func:`check_prompt_not_truncated`)
#   converts the estimate by the measured 3.38 ratio before its "materially
#   below" comparison, so a legitimate near-full-window prompt no longer
#   false-trips.
UPPER_BOUND_CHARS_PER_TOKEN = 2.5
# Live 2026-07-09 measurement (qwen2.5-7b-16k, math-heavy algebra corpus).
MEASURED_CHARS_PER_TOKEN = 3.38
# Chosen ABOVE the measured 3.38 so the pre-dispatch refusal only fires when
# even an optimistic reading overflows the window (headroom against corpora
# that tokenize slightly denser than the measured sample).
PREDISPATCH_OPTIMISTIC_CHARS_PER_TOKEN = 3.8


def check_prompt_fits_window(
    estimated: int,
    *,
    model_id: str,
    num_ctx: int,
    min_estimate: int = 256,
    optimistic_chars_per_token: float = PREDISPATCH_OPTIMISTIC_CHARS_PER_TOKEN,
) -> None:
    """PRE-dispatch deterministic tripwire — the cheapest truncation guard.

    A local server serves a fixed ``num_ctx`` window; a rendered prompt whose
    LOCAL estimate exceeds that window WILL be head-truncated on dispatch.
    ``estimated`` is the 2.5-char/token UPPER bound (see
    :data:`UPPER_BOUND_CHARS_PER_TOKEN`), which OVERCOUNTS real tokenization by
    ~1.35x (the 2026-07-09 measurement). Refusing on that raw upper bound
    rejects blocks whose real token count fits the window, so the refusal is
    CALIBRATED: the estimate is converted to an OPTIMISTIC token count via a
    divisor chosen ABOVE the measured 3.38 c/t
    (:data:`PREDISPATCH_OPTIMISTIC_CHARS_PER_TOKEN`, default 3.8) and the
    refusal fires ONLY when even that generous reading overflows ``num_ctx``.
    Equivalently: refuse when ``estimated * (2.5 / optimistic_chars_per_token)
    > num_ctx``.

    Raise BEFORE paying the round-trip so the caller can escalate a block that
    genuinely cannot fit. Deterministic — no server usage needed — so it fires
    even when the server omits ``usage`` (the fail-OPEN blind spot of
    :func:`check_prompt_not_truncated`).

    No-op when ``estimated`` is below ``min_estimate`` (a tiny prompt can't
    overflow) or ``num_ctx`` is non-positive (unknown window → no assertion).

    Raises
    ------
    PromptTruncatedError:
        When ``estimated >= min_estimate`` and the OPTIMISTIC token count
        ``estimated * (2.5 / optimistic_chars_per_token)`` exceeds ``num_ctx``.
    """
    if num_ctx <= 0:
        return
    if estimated < min_estimate:
        return
    # Convert the 2.5-c/t UPPER bound to an OPTIMISTIC token count using a
    # divisor above the measured 3.38 c/t; refuse only when even that reading
    # cannot fit. optimistic = estimated * (2.5 / 3.8).
    optimistic = estimated * (
        UPPER_BOUND_CHARS_PER_TOKEN / float(optimistic_chars_per_token)
    )
    if optimistic <= num_ctx:
        return
    from lib.retrieval.answer_backend import PromptTruncatedError

    raise PromptTruncatedError(
        f"Prompt will be truncated by the model server for model "
        f"{model_id!r}: even an OPTIMISTIC {optimistic_chars_per_token:.1f}-"
        f"char/token reading of the prompt (~{int(optimistic)} tokens, from a "
        f"{estimated}-token 2.5-c/t upper-bound estimate) EXCEEDS the served "
        f"context window (num_ctx={num_ctx}), so the system prompt + leading "
        f"tokens would be dropped on dispatch and any output is ungrounded. "
        f"Fixes: raise the server window (OLLAMA_CONTEXT_LENGTH or a Modelfile "
        f"'PARAMETER num_ctx') AND the matching num_ctx env var "
        f"(ED4ALL_ANSWER_NUM_CTX for answers, ED4ALL_REWRITE_NUM_CTX for the "
        f"rewrite tier), or enable ED4ALL_REWRITE_FIT_WINDOW to shrink the "
        f"prompt to fit."
    )


def check_prompt_not_truncated(
    reported: object,
    estimated: int,
    *,
    model_id: str,
    num_ctx: int,
    reported_fraction: float = 0.5,
    min_estimate: int = 256,
    materially_below_fraction: float = 0.75,
    cap_margin: float = 0.05,
) -> None:
    """Raise when the server silently truncated the prompt HEAD.

    Parameters
    ----------
    reported:
        The server-reported ``usage.prompt_tokens`` for the call (the raw
        value off the response body / client usage dict). Non-numeric /
        ``None`` / ``<= 0`` is treated as "no signal" and no-ops.
    estimated:
        The local UPPER-bound estimate of the prompt's token count
        (system prompt + user prompt), via the 2.5-char/token
        ``lib.retrieval._prompts.estimate_tokens`` divisor.
    model_id:
        Resolved model id, interpolated into the raised message.
    num_ctx:
        The served context window, interpolated into the raised message.
    reported_fraction:
        Fraction of the estimate the reported count may fall below before
        the head is declared truncated on the SEVERE arm (default ``0.5`` —
        reported below HALF the estimate). Mirrors the answer path's
        ``_TRUNCATION_REPORTED_FRACTION``.
    min_estimate:
        Absolute floor below which the fractional check is skipped (tiny
        prompts carry too little signal for the ratio to be meaningful,
        and a small window can't truncate them anyway). Mirrors the
        answer path's ``_TRUNCATION_MIN_ESTIMATE_TOKENS``.
    materially_below_fraction:
        MID-SIZE arm — fraction below which ``reported`` is "materially
        below" the CALIBRATED estimate (default ``0.75``). The 2.5-char/token
        estimate is a ~1.35x UPPER bound (2026-07-09 measurement: real
        tokenization ~3.38 c/t), so this fraction is applied to the estimate
        CONVERTED by that ratio (``estimated * 2.5/3.38``) — an effective
        ~0.55x of the raw upper bound — so a NON-truncated near-full-window
        call (reporting ~0.74x the raw estimate) does not false-trip. The
        mid-size arm still requires the SECOND condition (``reported`` sitting
        right at the served-window cap); a report that has reached/exceeded
        the cap trips on the cap-saturation signal alone.
    cap_margin:
        MID-SIZE arm — relative tolerance for "reported sits at the served
        window cap" (default ``0.05`` → within +/-5% of ``num_ctx``, floor
        64 tokens). When the server truncates to its window, the reported
        prompt-token count lands ~= ``num_ctx``; combined with materially-
        below-estimate that is the signature of a mid-size head-truncation
        the ``/2`` severe arm misses (e.g. an 8192-served window on a ~12k
        estimate reports ~8194 > estimate/2 and would otherwise PASS).

    Raises
    ------
    PromptTruncatedError:
        When ``reported`` is a positive int, ``estimated`` clears the
        ``min_estimate`` floor, and EITHER (severe) ``reported < estimated
        * reported_fraction`` OR (mid-size) ``reported`` is materially below
        the estimate AND consistent with the served-window cap. The message
        names the two operator fixes (raise the server window AND the
        matching num_ctx env var).

    Notes
    -----
    ``PromptTruncatedError`` is imported lazily from
    ``lib.retrieval.answer_backend`` so this module stays a leaf with no
    import-time dependency on the retrieval package (the rewrite tier
    imports this helper without pulling the answer stack).
    """
    try:
        reported_int = int(reported)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if reported_int <= 0:
        return  # server omitted / zeroed usage — no signal.
    if estimated < min_estimate:
        return  # too small for the ratio to be meaningful.
    threshold = estimated * reported_fraction
    # SEVERE arm (legacy): reported far below the raw upper-bound estimate.
    # Kept on the raw 2.5-c/t bound — a >2x shortfall is unambiguous under
    # either tokenization, and a normal ~0.74x report never clears it.
    severe = reported_int < threshold
    # MID-SIZE arm. The server truncated the prompt to its window, so the
    # reported prompt-token count lands AT that window (~= num_ctx) even though
    # it is only modestly below the local estimate (the /2 severe arm would let
    # it pass). Recalibrated (2026-07-09): the 2.5-c/t estimate is a ~1.35x
    # UPPER bound, so a NON-truncated call legitimately reports ~0.74x the
    # estimate — the "materially below" comparison therefore runs against the
    # estimate CONVERTED by the measured 3.38 c/t ratio, not the raw upper
    # bound, so a legitimate near-full-window prompt (reporting ~0.74x the
    # estimate and sitting just under the cap) no longer false-trips.
    calibrated_estimate = estimated * (
        UPPER_BOUND_CHARS_PER_TOKEN / MEASURED_CHARS_PER_TOKEN
    )
    materially_below = reported_int < calibrated_estimate * materially_below_fraction
    cap_tolerance = max(64, int(int(num_ctx) * cap_margin)) if num_ctx > 0 else 0
    at_window_cap = (
        num_ctx > 0 and abs(reported_int - int(num_ctx)) <= cap_tolerance
    )
    # A report that has REACHED/exceeded the window ceiling (within the cap
    # band above) is the saturation signature of a head-truncation — the server
    # clipped the prompt to its window, so usage lands at ~num_ctx regardless of
    # how far below the (over-counting) estimate that is. A report BELOW the
    # ceiling only trips when the CALIBRATED estimate corroborates that the
    # untruncated prompt was materially larger than what came back.
    reached_cap = num_ctx > 0 and reported_int >= int(num_ctx)
    mid_size = at_window_cap and (reached_cap or materially_below)
    if severe or mid_size:
        # Lazy import keeps this module a leaf (no retrieval import at
        # module load). PromptTruncatedError is the canonical fail-closed
        # error type shared with the answer path.
        from lib.retrieval.answer_backend import PromptTruncatedError

        raise PromptTruncatedError(
            f"Prompt HEAD was silently truncated by the model server for "
            f"model {model_id!r}: local UPPER-bound estimate ~{estimated} "
            f"prompt tokens (~{int(calibrated_estimate)} at the measured 3.38 "
            f"char/token) but the server reported {reported_int} — a shortfall "
            f"/ at-cap saturation consistent with the served context window "
            f"(num_ctx={num_ctx}) being too small for this prompt, so the "
            f"system prompt + leading tokens were dropped and any output is "
            f"ungrounded. Fixes: raise the server "
            f"window (OLLAMA_CONTEXT_LENGTH or a Modelfile 'PARAMETER "
            f"num_ctx') AND set the matching num_ctx env var "
            f"(ED4ALL_ANSWER_NUM_CTX for answers, ED4ALL_REWRITE_NUM_CTX "
            f"for the rewrite tier) to the served window so the prompt "
            f"budget shrinks the prompt to fit."
        )
