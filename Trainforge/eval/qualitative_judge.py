"""Wave 102 - LLM-as-judge / NLI-as-judge qualitative scorer.

Adds a 1-5 quality score column to the headline ablation table that
captures things the deterministic metrics miss (style, completeness,
hedging quality). Three provider modes are supported:

* ``none`` (default): :meth:`QualitativeJudge.score` returns ``None``.
  The ablation renderer omits the qualitative column entirely. This is
  the only mode that doesn't require any extra dependencies or env
  vars.
* ``anthropic``: dispatches the (prompt, model_output, ground_truth)
  triple to the Anthropic SDK with a fixed 1-5 rubric. The rubric
  prefix is wrapped in ``cache_control: ephemeral`` so per-probe cost
  stays bounded. Model defaults to ``claude-sonnet-4-6`` (override via
  ``ED4ALL_LLM_JUDGE_MODEL``).
* ``local_nli``: lazy-imports ``transformers`` and loads
  ``cross-encoder/nli-deberta-v3-large`` (~400MB; quantizes to fit in
  4GB VRAM). The ENTAIL probability is mapped to 1-5 via a fixed
  banding scheme.

Provider selection is driven by ``ED4ALL_LLM_JUDGE_PROVIDER``; pass
``provider=`` explicitly to override.

The ablation runner only invokes :meth:`score` when provider != "none";
the renderer mirrors that gating so a none-provider run produces a
4-column table (no qualitative column) and an anthropic-provider run
produces a 5-column table.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional


logger = logging.getLogger(__name__)


_DEFAULT_PROVIDER_ENV = "ED4ALL_LLM_JUDGE_PROVIDER"
_DEFAULT_MODEL_ENV = "ED4ALL_LLM_JUDGE_MODEL"
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-large"

# Banding for NLI ENTAIL probability -> 1-5 scale. The bands are
# tight at the top so a high ENTAIL prob doesn't trivially saturate
# the column.
_NLI_BANDS = (
    (0.95, 5),
    (0.80, 4),
    (0.55, 3),
    (0.30, 2),
)

_RUBRIC_SYSTEM = (
    "You are a strict but fair grader for short-answer educational responses. "
    "Score the model's answer on a 1-5 scale where 1 is fully incorrect or "
    "off-topic and 5 is concise, fully correct, and well-grounded in the "
    "provided ground truth. Return ONLY the integer score."
)


class QualitativeJudge:
    """Routes (prompt, model_output, ground_truth) -> 1-5 quality score.

    Args:
        provider: One of ``none``, ``anthropic``, ``local_nli``. When
            None, falls back to the ``ED4ALL_LLM_JUDGE_PROVIDER`` env
            var (defaults to ``none``).
        model: Optional override for the underlying model id (Anthropic
            model name or HF NLI repo). Defaults are
            ``claude-sonnet-4-6`` and ``cross-encoder/nli-deberta-v3-large``.
        anthropic_client: Optional pre-built Anthropic client (used in
            tests so we don't lazy-import the real SDK).
        nli_pipeline: Optional pre-built HF pipeline returning
            ``[{"label": "ENTAILMENT", "score": float}, ...]`` (used in
            tests).
        capture: Optional ``DecisionCapture`` instance. When wired,
            every per-probe scoring call emits one
            ``decision_type="llm_chat_call"`` event so the audit trail
            can distinguish a real-LLM-driven score from a degenerate
            stub-response (the H2 silent-degradation regression class).
            Field names are LLM-agnostic — the provider lands as a
            VALUE in ``ml_features["provider"]``. Capture failures are
            swallowed so logging never crashes scoring; ``capture=None``
            is a silent no-op.
    """

    VALID_PROVIDERS = ("none", "anthropic", "local_nli")

    def __init__(
        self,
        provider: Optional[str] = None,
        *,
        model: Optional[str] = None,
        anthropic_client: Optional[Any] = None,
        nli_pipeline: Optional[Any] = None,
        capture: Optional[Any] = None,
        probe_id: Optional[str] = None,
    ) -> None:
        if provider is None:
            provider = os.environ.get(_DEFAULT_PROVIDER_ENV, "none").strip().lower()
        if provider not in self.VALID_PROVIDERS:
            raise ValueError(
                f"QualitativeJudge: unknown provider={provider!r}. "
                f"Valid: {self.VALID_PROVIDERS}"
            )
        self.provider = provider
        self.model = model or os.environ.get(_DEFAULT_MODEL_ENV) or (
            _DEFAULT_ANTHROPIC_MODEL if provider == "anthropic"
            else _DEFAULT_NLI_MODEL if provider == "local_nli"
            else ""
        )
        self._anthropic_client = anthropic_client
        self._nli_pipeline = nli_pipeline
        self._capture = capture
        self._probe_id = probe_id

    @property
    def enabled(self) -> bool:
        """True when this judge will produce non-None scores."""
        return self.provider != "none"

    def score(
        self,
        prompt: str,
        model_output: str,
        ground_truth: str,
        *,
        probe_id: Optional[str] = None,
    ) -> Optional[float]:
        """Return a 1-5 quality score, or None when provider=none.

        Args:
            probe_id: Optional per-call probe identifier surfaced in the
                ``llm_chat_call`` decision event (``ml_features["probe_id"]``)
                so a post-hoc audit can correlate scores back to the
                originating ablation probe. Falls back to the
                instance-level ``probe_id`` when omitted.
        """
        if self.provider == "none":
            return None
        if self.provider == "anthropic":
            return self._score_anthropic(
                prompt, model_output, ground_truth, probe_id=probe_id,
            )
        if self.provider == "local_nli":
            return self._score_local_nli(prompt, model_output, ground_truth)
        raise RuntimeError(  # pragma: no cover
            f"unreachable: unknown provider {self.provider!r}"
        )

    # ------------------------------------------------------------------ #
    # Anthropic backend                                                   #
    # ------------------------------------------------------------------ #

    def _score_anthropic(
        self,
        prompt: str,
        model_output: str,
        ground_truth: str,
        *,
        probe_id: Optional[str] = None,
    ) -> float:
        client = self._anthropic_client
        if client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "QualitativeJudge: provider='anthropic' but the "
                    "anthropic SDK is not installed. `pip install anthropic`."
                ) from exc
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "QualitativeJudge: provider='anthropic' requires "
                    "ANTHROPIC_API_KEY in the environment."
                )
            client = anthropic.Anthropic(api_key=api_key)

        # Cache-control on the rubric prefix so repeated probe scoring
        # in a single ablation run reuses the cached prompt prefix.
        # Wave 102 wires the call but does not retry; eval is one-shot.
        max_tokens = 8
        start = time.monotonic()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": _RUBRIC_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{prompt}\n\n"
                        f"Model answer:\n{model_output}\n\n"
                        f"Ground truth:\n{ground_truth}\n\n"
                        f"Score (1-5):"
                    ),
                },
            ],
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        text = _extract_anthropic_text(message)
        score = _parse_score_1_to_5(text)
        self._emit_decision(
            provider="anthropic",
            probe_id=probe_id if probe_id is not None else self._probe_id,
            max_tokens=max_tokens,
            score=score,
            latency_ms=latency_ms,
            response_text=text,
        )
        return score

    # ------------------------------------------------------------------ #
    # Decision capture                                                    #
    # ------------------------------------------------------------------ #

    def _emit_decision(
        self,
        *,
        provider: str,
        probe_id: Optional[str],
        max_tokens: int,
        score: float,
        latency_ms: int,
        response_text: str,
    ) -> None:
        """Emit one ``llm_chat_call`` event per scoring call.

        Field names are LLM-agnostic per the standing decision-capture
        contract — provider lands as a VALUE in ``ml_features["provider"]``.
        Capture failures are swallowed so observability never crashes
        scoring; capture=None is a silent no-op.
        """
        if self._capture is None:
            return
        try:
            self._capture.log_decision(
                decision_type="llm_chat_call",
                decision=(
                    f"Qualitative judge scored probe "
                    f"{probe_id or 'unknown'} as {score} via model "
                    f"{self.model} (latency_ms={latency_ms})."
                ),
                rationale=(
                    f"Per-probe LLM-as-judge scoring against model "
                    f"{self.model} with max_tokens={max_tokens}; "
                    f"parsed 1-5 score={score} from response_len="
                    f"{len(response_text or '')}; latency_ms={latency_ms}. "
                    f"Captured so a downgraded stub-response run can be "
                    f"distinguished from a real-LLM scoring pass when "
                    f"the eval gate inspects the audit trail."
                ),
                ml_features={
                    "provider": provider,
                    "model": self.model,
                    "probe_id": probe_id,
                    "max_tokens": max_tokens,
                    "score": score,
                    "latency_ms": latency_ms,
                },
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("llm_chat_call capture failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Local NLI backend                                                   #
    # ------------------------------------------------------------------ #

    def _score_local_nli(
        self, prompt: str, model_output: str, ground_truth: str,
    ) -> float:
        pipe = self._nli_pipeline
        if pipe is None:
            try:
                from transformers import pipeline  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "QualitativeJudge: provider='local_nli' but "
                    "transformers is not installed. `pip install transformers`."
                ) from exc
            pipe = pipeline(
                "text-classification",
                model=self.model,
                top_k=None,
            )

        results = pipe(
            f"{ground_truth} </s> {model_output}"
        )
        # Expected shape: list[list[{label, score}]] or list[{label, score}]
        flat = results[0] if results and isinstance(results[0], list) else results
        entail_prob = 0.0
        for r in flat:
            if str(r.get("label", "")).upper().startswith("ENTAIL"):
                entail_prob = float(r.get("score", 0.0))
                break
        return _entail_to_5_band(entail_prob)


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #


def _extract_anthropic_text(message: Any) -> str:
    """Pull the assistant text out of an Anthropic ``messages.create`` reply.

    Tolerates both the SDK message-object shape and a dict mock.
    """
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not content:
        return ""
    block = content[0]
    text = getattr(block, "text", None)
    if text is None and isinstance(block, dict):
        text = block.get("text", "")
    return str(text or "").strip()


def _parse_score_1_to_5(text: str) -> float:
    """Coerce a model reply to a clamped 1-5 float."""
    import re

    m = re.search(r"[1-5]", text)
    if not m:
        # Reply didn't contain a usable digit; default to the midpoint
        # rather than crash. A regression test asserts on this fallback.
        return 3.0
    value = float(m.group(0))
    return max(1.0, min(5.0, value))


def _entail_to_5_band(prob: float) -> float:
    """Map ENTAIL probability to a 1-5 band."""
    for threshold, band in _NLI_BANDS:
        if prob >= threshold:
            return float(band)
    return 1.0


__all__ = ["QualitativeJudge"]
