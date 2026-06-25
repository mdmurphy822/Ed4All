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

__all__ = ["check_prompt_not_truncated"]


def check_prompt_not_truncated(
    reported: object,
    estimated: int,
    *,
    model_id: str,
    num_ctx: int,
    reported_fraction: float = 0.5,
    min_estimate: int = 256,
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
        the head is declared truncated (default ``0.5`` — reported must be
        below HALF the estimate). Mirrors the answer path's
        ``_TRUNCATION_REPORTED_FRACTION``.
    min_estimate:
        Absolute floor below which the fractional check is skipped (tiny
        prompts carry too little signal for the ratio to be meaningful,
        and a small window can't truncate them anyway). Mirrors the
        answer path's ``_TRUNCATION_MIN_ESTIMATE_TOKENS``.

    Raises
    ------
    PromptTruncatedError:
        When ``reported`` is a positive int, ``estimated`` clears the
        ``min_estimate`` floor, and ``reported < estimated *
        reported_fraction``. The message names the two operator fixes
        (raise the server window AND the matching num_ctx env var).

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
    if reported_int < threshold:
        # Lazy import keeps this module a leaf (no retrieval import at
        # module load). PromptTruncatedError is the canonical fail-closed
        # error type shared with the answer path.
        from lib.retrieval.answer_backend import PromptTruncatedError

        raise PromptTruncatedError(
            f"Prompt HEAD was silently truncated by the model server for "
            f"model {model_id!r}: estimated ~{estimated} prompt tokens but "
            f"the server reported only {reported_int} (< {threshold:.0f}). "
            f"The served context window (num_ctx={num_ctx}) is too small "
            f"for this prompt, so the system prompt + leading tokens were "
            f"dropped and any output is ungrounded. Fixes: raise the server "
            f"window (OLLAMA_CONTEXT_LENGTH or a Modelfile 'PARAMETER "
            f"num_ctx') AND set the matching num_ctx env var "
            f"(ED4ALL_ANSWER_NUM_CTX for answers, ED4ALL_REWRITE_NUM_CTX "
            f"for the rewrite tier) to the served window so the prompt "
            f"budget shrinks the prompt to fit."
        )
