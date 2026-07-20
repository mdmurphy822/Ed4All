"""Provider-agnostic input-prompt truncation tripwire.

A local OpenAI-compatible server (Ollama, vLLM, llama.cpp, …) serves a
fixed ``num_ctx`` window and SILENTLY truncates the prompt HEAD when the
rendered prompt exceeds it — the system prompt + leading tokens vanish
and the model authors from the surviving tail with no rules / no
allowed-id list, fabricating output that the downstream gates then
re-stamp green. The only signal is the server-reported
``usage.prompt_tokens``: a count far BELOW the local estimate means the
head was dropped.

The helper is provider-agnostic so every call site (the grounded-answer
path and the Courseforge rewrite tier) shares one detection and one set of
constants rather than re-deriving them.

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
# Char-per-token calibration
# ---------------------------------------------------------------------------
#
# ``lib.retrieval._prompts.estimate_tokens`` divides char length by 2.5 — an
# INTENTIONAL UPPER bound on the token count (a math/symbol-dense corpus packs
# ~2.5 chars/token; prose packs ~4). Real tokenization measures ~3.38
# chars/token, so that estimate OVERCOUNTS a real prompt by ~1.35x. Gating on
# the raw upper bound therefore refuses a large fraction of prompts whose REAL
# token count fits the served window (they then mass-fall-back to template
# rendering). Both guards below CALIBRATE against the measured ratio instead:
#
# - the PRE-dispatch refusal (:func:`check_prompt_fits_window`) converts the
#   estimate with an OPTIMISTIC divisor chosen ABOVE the measured 3.38 c/t, so
#   a refusal fires ONLY when even a generous reading of the prompt cannot
#   fit;
# - the POST-dispatch mid-size arm (:func:`check_prompt_not_truncated`)
#   converts the estimate by the measured 3.38 ratio before comparing, so a
#   legitimate near-full-window prompt no longer false-trips.
UPPER_BOUND_CHARS_PER_TOKEN = 2.5
# Measured real tokenization rate on math-dense accessible HTML.
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
    ~1.35x. Refusing on that raw upper bound
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
    over_window_fraction: float = 1.1,
    cap_margin: float = 0.02,
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
    over_window_fraction:
        MID-SIZE arm, condition 1 — the assembled prompt must be over the
        ASSUMED window by a real margin before a clip is even plausible
        (default ``1.1`` → ``estimated >= 1.1 * num_ctx``). This is POSITIVE
        evidence: a prompt whose (over-counting 2.5-c/t) estimate does not
        clear ``1.1 * num_ctx`` could not have overflowed the assumed window,
        so its reported count landing near ``num_ctx`` is coincidence, not a
        clip — and it must NEVER mid-size-trip. Do NOT replace this with a
        "reported materially below the estimate" rule: because the estimate
        over-counts ~1.35x, that predicate holds for virtually every honestly
        served prompt and collapses the arm to "reported ~= num_ctx", which
        false-fires on any prompt whose TRUE token count happens to land near
        the assumed cap.
    cap_margin:
        MID-SIZE arm, condition 2 — relative half-width of the "reported is
        pinned AT the window ceiling from below" band (default ``0.02`` →
        ``0.98 * num_ctx <= reported <= 1.02 * num_ctx``). When a server
        actually clips to its window, ``usage.prompt_tokens`` lands within a
        whisker of ``num_ctx``; combined with condition 1 that is the
        signature of a mid-size head-truncation the ``/2`` severe arm misses
        (e.g. an 8192-served window on a ~12k estimate reports ~8194 >
        estimate/2 and would otherwise PASS). The band is deliberately tight
        to shrink — but NOT eliminate — the residual-FP band (see Notes).

    Raises
    ------
    PromptTruncatedError:
        When ``reported`` is a positive int, ``estimated`` clears the
        ``min_estimate`` floor, and EITHER (severe) ``reported < estimated
        * reported_fraction`` OR (mid-size) BOTH the assembled estimate
        overflows the assumed window (``estimated >= over_window_fraction *
        num_ctx``) AND ``reported`` is pinned at the window ceiling from below
        (``|reported - num_ctx| <= cap_margin * num_ctx``). The message names
        the two operator fixes (raise the server window AND the matching
        num_ctx env var) and surfaces the estimate/reported ratio so an
        operator can judge a borderline call.

    Notes
    -----
    **Residual false-positive band (honest limit).** A genuine head-clip and
    an honestly-served prompt whose TRUE token count happens to land inside
    ``[0.98, 1.02] * num_ctx`` are INDISTINGUISHABLE from
    ``usage.prompt_tokens`` alone: both report a count pinned at the assumed
    ceiling. This bites hardest when ``num_ctx`` UNDERSTATES the server's real
    window: the 2.5-c/t estimate then over-shoots the assumed cap (satisfying
    condition 1) while the honestly served true size lands near it, leaving the
    cap band as the only separator. The tight ``[0.98, 1.02]`` band kills the
    tails of that distribution (reported strictly below ``0.98 * num_ctx`` or
    above ``1.02 * num_ctx`` no longer trips), but a served prompt whose true
    size lands *exactly* at the assumed cap will still trip. There is no
    deterministic separator for that inner band without a healthy-completion
    signal (``finish_reason`` / completion-token count), which this helper does
    not receive; the message therefore surfaces the estimate/reported ratio so
    an operator can spot the ambiguity, and the real fix for a chronic FP is to
    set the num_ctx env var to the server's ACTUAL window (so the assumed cap
    stops coinciding with real prompt sizes).

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
    # SEVERE arm: reported far below the raw upper-bound estimate. Stays on the
    # raw 2.5-c/t bound — a >2x shortfall is unambiguous under either
    # tokenization, and a normal ~0.74x report never clears it.
    severe = reported_int < threshold
    # MID-SIZE arm. A mid-size clip requires POSITIVE
    # evidence, not mere coincidence with the assumed window. BOTH must hold:
    #   (1) the ASSEMBLED prompt overflows the ASSUMED window by a real margin
    #       (estimated >= over_window_fraction * num_ctx) — so a clip is even
    #       plausible. A prompt whose estimate fits comfortably under num_ctx
    #       can NEVER mid-size-trip, no matter where `reported` lands.
    #   (2) `reported` is pinned at the window ceiling FROM BELOW
    #       ([0.98, 1.02] * num_ctx) — a real clip lands usage within a whisker
    #       of num_ctx.
    # Do NOT weaken (1) into "reported materially below a calibrated estimate":
    # because the 2.5-c/t estimate over-counts ~1.35x, that predicate holds for
    # virtually every honest prompt, degenerating the arm to "reported ~=
    # num_ctx" and false-firing whenever a TRUE token count happens to sit near
    # the assumed cap.
    over_window = num_ctx > 0 and estimated >= over_window_fraction * float(num_ctx)
    if num_ctx > 0:
        cap_tolerance = float(num_ctx) * cap_margin
        at_window_cap = (
            float(num_ctx) - cap_tolerance
        ) <= reported_int <= (float(num_ctx) + cap_tolerance)
    else:
        at_window_cap = False
    mid_size = over_window and at_window_cap
    # Surfaced in the message so an operator can judge a borderline call that
    # falls in the irreducible residual-FP band (see the docstring Notes).
    calibrated_estimate = estimated * (
        UPPER_BOUND_CHARS_PER_TOKEN / MEASURED_CHARS_PER_TOKEN
    )
    if severe or mid_size:
        # Lazy import keeps this module a leaf (no retrieval import at
        # module load). PromptTruncatedError is the canonical fail-closed
        # error type shared with the answer path.
        from lib.retrieval.answer_backend import PromptTruncatedError

        ratio = (float(estimated) / reported_int) if reported_int else 0.0
        raise PromptTruncatedError(
            f"Prompt HEAD was silently truncated by the model server for "
            f"model {model_id!r}: local UPPER-bound estimate ~{estimated} "
            f"prompt tokens (~{int(calibrated_estimate)} at the measured 3.38 "
            f"char/token) but the server reported {reported_int} "
            f"(estimate/reported ~{ratio:.2f}x) — the assembled prompt "
            f"overflows the assumed context window and the reported count is "
            f"pinned at that window ceiling (num_ctx={num_ctx}), the signature "
            f"of a head-clip that dropped the system prompt + leading tokens, "
            f"leaving any output ungrounded. NOTE: if the server's ACTUAL "
            f"window is LARGER than num_ctx={num_ctx}, a true prompt size that "
            f"coincidentally landed near {num_ctx} can trip this without a real "
            f"clip — set the num_ctx env var to the server's real window to "
            f"rule that out. Fixes: raise the server "
            f"window (OLLAMA_CONTEXT_LENGTH or a Modelfile 'PARAMETER "
            f"num_ctx') AND set the matching num_ctx env var "
            f"(ED4ALL_ANSWER_NUM_CTX for answers, ED4ALL_REWRITE_NUM_CTX "
            f"for the rewrite tier) to the served window so the prompt "
            f"budget shrinks the prompt to fit."
        )
