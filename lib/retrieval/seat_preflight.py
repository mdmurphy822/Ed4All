"""Seat-coherence PREFLIGHT — the vLLM restart-mode-collapse guard (E7b).

A vLLM seat that was ``docker start``-ed after a stop can pass its
``/v1/models`` readiness poll and yet emit degenerate soup / ``null``
content on the first real request (the mode-collapse-after-restart
failure class documented in the project memory: a restarted seat killed
44/44 QC windows while answering readiness probes cleanly). Any consumer
that trusts such a seat silently produces garbage.

This module probes an OpenAI-compatible seat with a tiny, deterministic
CONTENT prompt and validates the response is *coherent* — non-empty,
non-degenerate (a repetition / low-entropy heuristic that catches the
``"the the the …"`` and ``"aaaaaaaa"`` collapse shapes), and containing
an expected token for a question with an unambiguous answer. The verdict
gates measurement validity, so it is captured via ``DecisionCapture``.

Two surfaces:

* :func:`probe_seat_coherence` — pure verdict. Takes any duck-typed
  ``chat_completion``-capable client, returns a :class:`SeatCoherenceResult`.
  A transport failure is folded into an INCOHERENT verdict (a dead seat is
  as untrustworthy as a collapsed one) — never re-raised.
* :func:`preflight_or_raise` — the gate. Probes and raises
  :class:`SeatCoherenceError` (loud, naming the seat) when the verdict is
  incoherent; returns the result otherwise. The eval diagnostic-composer
  arm calls this before trusting a stronger local seat; other callers can
  reuse it as a standalone one-shot gate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lib.decision_capture import DecisionCapture
    from lib.retrieval.answer_backend import ResolvedAnswerBackend


# --------------------------------------------------------------------------- #
# Probe contract (deterministic, tiny — a coherent seat trivially passes)
# --------------------------------------------------------------------------- #

#: A minimal system directive — the probe measures raw generation health,
#: not instruction-following depth.
PROBE_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."
#: An unambiguous arithmetic question: a coherent model of any size emits a
#: response containing "4"/"four"; a mode-collapsed seat emits soup or null.
PROBE_USER_PROMPT = "What is 2 + 2? Reply with just the number."
#: Substring tokens (lower-cased) any correct answer must contain. Empty tuple
#: → the expected-token leg is skipped (degeneracy + non-empty still gate).
DEFAULT_EXPECTED_TOKENS: Tuple[str, ...] = ("4", "four")
#: Generation cap for the probe — small (cheap), but wide enough that a
#: reasoning seat with thinking-off still lands the answer token.
DEFAULT_PROBE_MAX_TOKENS = 64
#: Probe sampling temperature — deterministic.
DEFAULT_PROBE_TEMPERATURE = 0.0

# Degeneracy heuristics (calibrated to catch collapse, never a real answer).
#: Word-level: below this distinct/total ratio (once at least the min count of
#: tokens is present) the text is repetition soup ("the the the …").
_MIN_DISTINCT_WORD_RATIO = 0.4
_MIN_WORDS_FOR_WORD_CHECK = 6
#: Char-level: below this distinct/length ratio (once long enough) the text is a
#: single-glyph run ("aaaaaaaa") that carries no word boundaries.
_MIN_DISTINCT_CHAR_RATIO = 0.15
_MIN_CHARS_FOR_CHAR_CHECK = 8

DECISION_TYPE_SEAT_PREFLIGHT = "seat_coherence_preflight"
DECISION_PHASE = "libv2-answer"

#: Optional excerpt length recorded in the result signals (never the whole
#: potentially-degenerate blob).
_EXCERPT_CHARS = 120


class SeatCoherenceError(RuntimeError):
    """A seat failed the coherence preflight — loud, names the seat.

    Raised by :func:`preflight_or_raise` (and, transitively, the eval
    diagnostic-composer arm) so a mode-collapsed / dead seat can never
    silently back a measurement or an answer.
    """


@dataclass(frozen=True)
class SeatCoherenceResult:
    """The preflight verdict for one seat.

    ``coherent`` is the gate. ``signals`` carries the replayable evidence
    (excerpt, ratios, which legs passed) folded into the decision-capture
    rationale so a post-hoc audit can tell WHY a seat was trusted/refused
    without re-probing.
    """

    coherent: bool
    seat: str
    reason: str
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coherent": self.coherent,
            "seat": self.seat,
            "reason": self.reason,
            "signals": dict(self.signals),
        }


def _excerpt(text: str) -> str:
    text = " ".join((text or "").split())
    return text[:_EXCERPT_CHARS]


def _degeneracy(text: str) -> Tuple[bool, Dict[str, Any]]:
    """Detect repetition / low-entropy collapse. Returns (is_degenerate, signals)."""
    words = [w for w in (text or "").split() if w]
    distinct_word_ratio = (
        len({w.lower() for w in words}) / len(words) if words else 1.0
    )
    compact = "".join((text or "").split())
    distinct_char_ratio = (
        len(set(compact.lower())) / len(compact) if compact else 1.0
    )
    word_soup = (
        len(words) >= _MIN_WORDS_FOR_WORD_CHECK
        and distinct_word_ratio < _MIN_DISTINCT_WORD_RATIO
    )
    char_soup = (
        len(compact) >= _MIN_CHARS_FOR_CHAR_CHECK
        and distinct_char_ratio < _MIN_DISTINCT_CHAR_RATIO
    )
    signals = {
        "word_count": len(words),
        "distinct_word_ratio": round(distinct_word_ratio, 4),
        "distinct_char_ratio": round(distinct_char_ratio, 4),
    }
    return bool(word_soup or char_soup), signals


def probe_seat_coherence(
    client: Any,
    *,
    seat_label: str,
    capture: Optional["DecisionCapture"] = None,
    expected_tokens: Sequence[str] = DEFAULT_EXPECTED_TOKENS,
    max_tokens: int = DEFAULT_PROBE_MAX_TOKENS,
    temperature: float = DEFAULT_PROBE_TEMPERATURE,
) -> SeatCoherenceResult:
    """Probe a seat with a tiny content prompt; return a coherence verdict.

    ``client`` is any object exposing
    ``chat_completion(messages, *, max_tokens=, temperature=) -> str`` (the
    ``OpenAICompatibleClient`` contract). A transport / call failure is folded
    into an INCOHERENT verdict (reason ``probe_call_failed:<type>``) rather than
    propagated — a seat that can't answer a trivial prompt is not trustworthy.

    Coherence requires ALL of: non-empty stripped content, not degenerate
    (repetition / single-glyph soup), and — when ``expected_tokens`` is
    non-empty — at least one expected token present (case-insensitive
    substring). Emits one ``seat_coherence_preflight`` decision on ``capture``
    with a dynamic, replayable rationale.
    """
    messages = [
        {"role": "system", "content": PROBE_SYSTEM_PROMPT},
        {"role": "user", "content": PROBE_USER_PROMPT},
    ]
    try:
        raw = client.chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature
        )
    except Exception as exc:  # noqa: BLE001 - a failed probe IS an incoherent verdict
        result = SeatCoherenceResult(
            coherent=False,
            seat=seat_label,
            reason=f"probe_call_failed:{type(exc).__name__}",
            signals={"error": str(exc)[:200]},
        )
        _emit_preflight(capture, result)
        return result

    text = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
    stripped = text.strip()
    non_empty = bool(stripped)
    is_degenerate, degen_signals = _degeneracy(text)
    lowered = stripped.lower()
    want = [t.lower() for t in expected_tokens if t]
    expected_present = (not want) or any(tok in lowered for tok in want)

    if not non_empty:
        reason = "empty_response"
    elif is_degenerate:
        reason = "degenerate_repetition"
    elif not expected_present:
        reason = "expected_token_absent"
    else:
        reason = "coherent"
    coherent = non_empty and (not is_degenerate) and expected_present

    signals: Dict[str, Any] = {
        "excerpt": _excerpt(text),
        "response_chars": len(text),
        "non_empty": non_empty,
        "degenerate": is_degenerate,
        "expected_present": expected_present,
        "expected_tokens": list(want),
        **degen_signals,
    }
    result = SeatCoherenceResult(
        coherent=coherent, seat=seat_label, reason=reason, signals=signals
    )
    _emit_preflight(capture, result)
    return result


def preflight_or_raise(
    client: Any,
    *,
    seat_label: str,
    capture: Optional["DecisionCapture"] = None,
    expected_tokens: Sequence[str] = DEFAULT_EXPECTED_TOKENS,
    max_tokens: int = DEFAULT_PROBE_MAX_TOKENS,
    temperature: float = DEFAULT_PROBE_TEMPERATURE,
) -> SeatCoherenceResult:
    """Preflight a seat; raise :class:`SeatCoherenceError` when incoherent.

    The gate the eval diagnostic-composer arm calls before it trusts a
    stronger local seat. On failure the message NAMES the seat + the failure
    reason + a response excerpt so the operator can tell a mode-collapsed seat
    (restart it cold) from a dead one. Returns the (coherent) result otherwise.
    Reusable standalone by any caller that wants a one-shot seat gate.
    """
    result = probe_seat_coherence(
        client,
        seat_label=seat_label,
        capture=capture,
        expected_tokens=expected_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not result.coherent:
        excerpt = result.signals.get("excerpt", "")
        raise SeatCoherenceError(
            f"Seat {seat_label!r} FAILED the coherence preflight "
            f"(reason={result.reason}). A vLLM seat that passed its readiness "
            f"poll can still emit degenerate / null content after a restart "
            f"(mode-collapse-after-restart). Cold-stop then start the seat and "
            f"re-probe before trusting it. Probe response excerpt: "
            f"{excerpt!r}."
        )
    return result


def probe_resolved_backend(
    resolved: "ResolvedAnswerBackend",
    *,
    capture: Optional["DecisionCapture"] = None,
    raise_on_fail: bool = True,
    expected_tokens: Sequence[str] = DEFAULT_EXPECTED_TOKENS,
    max_tokens: int = DEFAULT_PROBE_MAX_TOKENS,
) -> SeatCoherenceResult:
    """Build the answer client for a resolved backend and preflight it.

    Convenience wrapper so a caller holding a :class:`ResolvedAnswerBackend`
    (the eval runner, a doctor probe) can preflight without constructing the
    client itself. ``raise_on_fail`` (default) routes through
    :func:`preflight_or_raise`; ``False`` returns the raw verdict. The heavy
    ``build_answer_client`` import is lazy to keep resolution paths cheap.
    """
    from lib.retrieval.answer_backend import build_answer_client

    seat_label = seat_label_for_backend(resolved)
    client = build_answer_client(resolved=resolved, capture=capture)
    if raise_on_fail:
        return preflight_or_raise(
            client,
            seat_label=seat_label,
            capture=capture,
            expected_tokens=expected_tokens,
            max_tokens=max_tokens,
        )
    return probe_seat_coherence(
        client,
        seat_label=seat_label,
        capture=capture,
        expected_tokens=expected_tokens,
        max_tokens=max_tokens,
    )


def seat_label_for_backend(resolved: "ResolvedAnswerBackend") -> str:
    """A stable, human-readable seat label ``provider:model@base_url``."""
    return f"{resolved.provider_name}:{resolved.model_id}@{resolved.base_url}"


def _emit_preflight(capture: Optional[Any], result: SeatCoherenceResult) -> None:
    if capture is None:
        return
    verdict = "coherent" if result.coherent else "incoherent"
    s = result.signals
    rationale = (
        f"seat coherence preflight {verdict} for seat {result.seat} "
        f"(reason={result.reason}); non_empty={s.get('non_empty')} "
        f"degenerate={s.get('degenerate')} expected_present={s.get('expected_present')} "
        f"distinct_word_ratio={s.get('distinct_word_ratio')} "
        f"distinct_char_ratio={s.get('distinct_char_ratio')} "
        f"response_chars={s.get('response_chars')} excerpt={s.get('excerpt')!r}"
    )
    try:
        capture.log_decision(
            decision_type=DECISION_TYPE_SEAT_PREFLIGHT,
            decision=f"seat_preflight:{verdict}",
            rationale=rationale,
            context=f"reason={result.reason}",
        )
    except Exception:  # pragma: no cover - capture must never break the gate
        pass


__all__ = [
    "PROBE_SYSTEM_PROMPT",
    "PROBE_USER_PROMPT",
    "DEFAULT_EXPECTED_TOKENS",
    "DEFAULT_PROBE_MAX_TOKENS",
    "DECISION_TYPE_SEAT_PREFLIGHT",
    "SeatCoherenceError",
    "SeatCoherenceResult",
    "probe_seat_coherence",
    "preflight_or_raise",
    "probe_resolved_backend",
    "seat_label_for_backend",
]
