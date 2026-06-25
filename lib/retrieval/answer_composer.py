"""Grounded-answer composer — THE LLM call site (D9).

``compose_answer`` turns a question + ranked retrieved passages into a
structured answer constrained to those passages. It:

* renders the numbered passage context (``_prompts``),
* calls a duck-typed ``client.chat_completion`` (the layer-2
  ``OpenAICompatibleClient`` in production, ``FakeAnswerClient`` in CI),
* leniently parses the JSON envelope (first ``{...}`` span),
* validates every cited id is one of the provided passage ids, firing a
  single remediation retry on unknown ids,
* emits a ``grounded_answer_composition`` decision-capture event PER
  attempt with a dynamic rationale, and
* maps transport failures to :class:`AnswerBackendUnavailable` — never
  returns a fabricated envelope.

The model cites ``chunk_id`` values directly (they are the passage
labels), so there is no index-remapping layer that can drift.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from lib.llm.truncation_guard import check_prompt_not_truncated
from lib.retrieval._prompts import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM_PROMPT,
    CITATION_REMEDIATION_DIRECTIVE,
    render_answer_user_prompt_with_meta,
)

# Fraction of the local estimate the server-reported prompt_tokens may
# fall below before we declare the prompt head silently truncated. The
# estimate is a rough UPPER bound (2.5 chars/token, conservative), and
# tokenizers legitimately vary, so the margin is generous: only a LARGE
# shortfall (reported < estimate * (1 - margin) AND below an absolute
# floor) trips the tripwire. 0.5 → reported must be < half the estimate.
_TRUNCATION_REPORTED_FRACTION = 0.5

# Absolute floor below which the fractional check is not applied: tiny
# prompts (estimate < this) have too little signal for the ratio to be
# meaningful, and a 4096-window can't truncate them anyway.
_TRUNCATION_MIN_ESTIMATE_TOKENS = 256


class AnswerComposeError(RuntimeError):
    """Parse exhaustion / invalid envelope after all retries."""


class InvalidCitationError(AnswerComposeError):
    """Cited ids not in the provided passage set, post-remediation."""


@dataclass(frozen=True)
class RetrievedPassage:
    """A retrieved passage in the composer's frozen input shape (D9).

    Duck-typed from the LibV2 ``RetrievalResult`` via
    :meth:`from_retrieval_result`. ``source`` is the full ``chunk.source``
    dict, carried through so the pipeline's citation gate can resolve the
    anchor without a second retrieval.
    """

    chunk_id: str
    text: str
    score: float
    engine: str
    item_path: str
    section_heading: Optional[str]
    module_id: Optional[str]
    source: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_retrieval_result(cls, result: Any, *, engine: str) -> "RetrievedPassage":
        """Build from a LibV2 ``RetrievalResult`` (or any duck-type of it).

        Works for the lexical results that exist today and the E3 semantic
        results when they land (same dataclass per WS2 D8). ``source`` is
        the full chunk-source dict; ``item_path`` / ``section_heading`` /
        ``module_id`` are read from the result first, then from ``source``
        as a fallback.
        """
        source = _attr(result, "source", None)
        if not isinstance(source, dict):
            source = _to_dict(source)

        def _pick(name: str) -> Any:
            val = _attr(result, name, None)
            if val is None and isinstance(source, dict):
                val = source.get(name)
            return val

        return cls(
            chunk_id=str(_attr(result, "chunk_id", "") or ""),
            text=str(_attr(result, "text", "") or ""),
            score=float(_attr(result, "score", 0.0) or 0.0),
            engine=engine,
            item_path=str(_pick("item_path") or ""),
            section_heading=_opt_str(_pick("section_heading")),
            module_id=_opt_str(_pick("module_id")),
            source=source if isinstance(source, dict) else {},
        )


@dataclass(frozen=True)
class ComposedAnswer:
    """The composer's result (D9). Consumed by the pipeline (E6)."""

    answer_text: Optional[str]
    cited_chunk_ids: List[str]
    not_in_course: bool
    model_id: str
    prompt_version: str
    attempts: int
    latency_ms: float
    raw_response_len: int
    # The passage ids the model could have cited — the renderer-INCLUDED
    # (in-window) set (additive, optional). Over-budget trailing passages were
    # dropped from the prompt, so they are NOT in this list. The grounded-answer
    # attribution pass runs over exactly this gate-eligible set (it must never
    # "add" a citation for a passage the model never saw). Defaults to ``[]`` so
    # legacy constructions stay valid.
    allowed_chunk_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Envelope parsing
# ---------------------------------------------------------------------------

def _parse_answer_envelope(
    raw: str,
    allowed_ids: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Lenient extraction of the answer envelope.

    Extracts the first ``{...}`` span (mirroring ``_parse_rerank_response``
    tolerance), parses it, and validates shape. Returns the normalized
    dict ``{"answer": str, "citations": [str...], "not_in_course": bool}``
    or ``None`` when nothing parseable/shaped is recoverable (the caller
    then retries or raises — never fabricates an envelope).

    Citation validation (unknown ids) is the CALLER's concern so it can
    drive a remediation retry; this only enforces structural shape.
    """
    if not raw:
        return None
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None

    not_in_course = bool(parsed.get("not_in_course", False))
    answer = parsed.get("answer")
    answer_str = "" if answer is None else str(answer)

    citations_raw = parsed.get("citations")
    citations: List[str] = []
    if isinstance(citations_raw, list):
        for cid in citations_raw:
            if isinstance(cid, str) and cid:
                citations.append(cid)
    # A non-list citations field is treated as "no citations" (structural
    # shape is satisfied; the caller's remediation handles empties).

    return {
        "answer": answer_str,
        "citations": citations,
        "not_in_course": not_in_course,
    }


def _unknown_citation_ids(
    citations: Sequence[str],
    allowed_ids: Sequence[str],
) -> List[str]:
    allowed = set(allowed_ids)
    return [cid for cid in citations if cid not in allowed]


def _check_prompt_not_truncated(
    client: Any,
    estimated_prompt_tokens: int,
    *,
    model_id: str,
    num_ctx: int,
) -> None:
    """Fail closed when the server silently truncated the prompt HEAD.

    Reads the server-reported ``usage.prompt_tokens`` off the client's
    ``last_usage`` (set by ``OpenAICompatibleClient.chat_completion``).
    When the reported count is far BELOW the local estimate — reported
    ``<`` ``estimate * _TRUNCATION_REPORTED_FRACTION`` AND the estimate
    clears the absolute floor — the head was dropped (Ollama silently
    truncates when the prompt exceeds the served window), so we raise
    :class:`PromptTruncatedError` naming the operator fixes instead of
    letting the fabricated-citation answer through.

    No-ops when usage is unavailable (the server omitted it, or a test
    double doesn't expose ``last_usage``): the tripwire is a guard, not a
    requirement, and a missing signal must never block a valid answer.

    Delegates the actual shortfall comparison to the provider-agnostic
    ``lib.llm.truncation_guard.check_prompt_not_truncated`` (the rewrite
    tier reuses the same helper) — this wrapper only reads the reported
    ``prompt_tokens`` off the client's ``last_usage`` and forwards the
    answer-path constants (``0.5`` / ``256``) so behaviour is byte-
    equivalent to the inline check it replaced.
    """
    usage = getattr(client, "last_usage", None)
    if not isinstance(usage, dict):
        return
    check_prompt_not_truncated(
        usage.get("prompt_tokens"),
        estimated_prompt_tokens,
        model_id=model_id,
        num_ctx=num_ctx,
        reported_fraction=_TRUNCATION_REPORTED_FRACTION,
        min_estimate=_TRUNCATION_MIN_ESTIMATE_TOKENS,
    )


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def compose_answer(
    query: str,
    passages: Sequence[RetrievedPassage],
    *,
    client: Any,
    capture: Optional[Any] = None,
    course_code: str = "",
    max_passages: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_parse_retries: int = 3,
    extra_directive: Optional[str] = None,
) -> ComposedAnswer:
    """Compose a passage-constrained answer. THE LLM call site.

    Emits one ``grounded_answer_composition`` decision per attempt.
    Transport (connection/timeout) errors surface as
    :class:`AnswerBackendUnavailable`. Parse exhaustion raises
    :class:`AnswerComposeError`; post-remediation unknown citation ids
    raise :class:`InvalidCitationError`. Never returns a fabricated
    envelope.

    ``extra_directive`` (optional) is appended to the user prompt on every
    attempt — the completeness re-ask uses it to name the still-unanswered
    sub-question(s) without re-rendering the passage context.
    """
    selected = list(passages)[: max(0, max_passages)]
    model_id = _client_model_id(client)
    query_sha = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    engine = selected[0].engine if selected else ""
    top3 = [(p.chunk_id, round(float(p.score), 4)) for p in selected[:3]]

    base_user_prompt, render_meta = render_answer_user_prompt_with_meta(
        query, selected, max_tokens=max_tokens
    )
    # The allowed citation ids are ONLY the passages the renderer actually
    # INCLUDED in the prompt — over-budget trailing passages were dropped
    # (their text + id are not in-window), so crediting a citation to a
    # dropped passage would let the model "cite" something it never saw.
    allowed_ids = list(render_meta.get("included_ids", [p.chunk_id for p in selected]))
    estimated_prompt_tokens = int(render_meta.get("estimated_prompt_tokens", 0) or 0)
    num_ctx = int(render_meta.get("num_ctx", 0) or 0)

    start = time.monotonic()
    attempts = 0
    remediation_directive: Optional[str] = None
    last_raw = ""

    # Total attempts = parse retries + 1 initial. A single remediation
    # retry (unknown ids) consumes one attempt within the same budget.
    for attempt_index in range(max_parse_retries):
        attempts += 1
        # A caller-supplied ``extra_directive`` (e.g. the completeness re-ask)
        # rides on EVERY attempt; the citation-remediation directive is appended
        # on top when the model invents ids. Both compose additively so a
        # re-ask retry can still self-repair an unknown-citation slip.
        user_prompt = base_user_prompt
        directives = [d for d in (extra_directive, remediation_directive) if d]
        if directives:
            user_prompt = base_user_prompt + "\n\n" + "\n\n".join(directives)

        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = client.chat_completion(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors
            if _is_transport_failure(exc):
                raise AnswerBackendUnavailable(
                    f"Answer backend unavailable for model {model_id!r}: "
                    f"{exc}. Start your local model server or set "
                    f"ED4ALL_ANSWER_* — there is no canned-answer fallback."
                ) from exc
            raise

        # Truncation tripwire: compare the server-reported prompt_tokens
        # against the local estimate. A large shortfall means the head was
        # silently dropped (the prompt overflowed the served window), so
        # any citations are fabricated — fail closed BEFORE parsing.
        _check_prompt_not_truncated(
            client,
            estimated_prompt_tokens,
            model_id=model_id,
            num_ctx=num_ctx,
        )

        last_raw = raw if isinstance(raw, str) else ""
        envelope = _parse_answer_envelope(last_raw, allowed_ids)
        latency_ms = (time.monotonic() - start) * 1000.0

        if envelope is None:
            _emit_composition(
                capture, course_code, query_sha, engine, model_id,
                attempt_index, len(selected), top3, render_meta,
                citation_count=None, latency_ms=latency_ms,
                outcome="parse_failed",
            )
            continue

        not_in_course = envelope["not_in_course"]
        citations = envelope["citations"]
        unknown = _unknown_citation_ids(citations, allowed_ids)

        if unknown and not not_in_course:
            _emit_composition(
                capture, course_code, query_sha, engine, model_id,
                attempt_index, len(selected), top3, render_meta,
                citation_count=len(citations), latency_ms=latency_ms,
                outcome=f"unknown_citations:{','.join(unknown)}",
            )
            # Re-state the FULL allowed set (not just the offenders):
            # naming only bad ids yields format-compliance but not
            # set-compliance — the model keeps inventing fresh ids.
            remediation_directive = CITATION_REMEDIATION_DIRECTIVE.format(
                bad_ids=", ".join(unknown),
                allowed_ids=", ".join(allowed_ids) if allowed_ids else "(none)",
            )
            continue

        # Accepted envelope (well-formed, citations valid or refusal).
        _emit_composition(
            capture, course_code, query_sha, engine, model_id,
            attempt_index, len(selected), top3, render_meta,
            citation_count=len(citations), latency_ms=latency_ms,
            outcome="not_in_course" if not_in_course else "answered",
        )
        return ComposedAnswer(
            answer_text=None if not_in_course else envelope["answer"],
            cited_chunk_ids=[] if not_in_course else citations,
            not_in_course=not_in_course,
            model_id=model_id,
            prompt_version=ANSWER_PROMPT_VERSION,
            attempts=attempts,
            latency_ms=latency_ms,
            raw_response_len=len(last_raw),
            allowed_chunk_ids=list(allowed_ids),
        )

    # Budget exhausted. If the last failure was unknown-citation, raise
    # the more specific error; otherwise it was a parse failure.
    if remediation_directive is not None:
        raise InvalidCitationError(
            f"Composer exhausted {attempts} attempts; the model kept "
            f"citing passage ids not in the provided set "
            f"(allowed={allowed_ids}). No citation list was repaired or "
            f"fabricated."
        )
    raise AnswerComposeError(
        f"Composer exhausted {attempts} attempts without a parseable "
        f"answer envelope from model {model_id!r}."
    )


# ---------------------------------------------------------------------------
# Decision capture
# ---------------------------------------------------------------------------

def _emit_composition(
    capture: Optional[Any],
    course_code: str,
    query_sha: str,
    engine: str,
    model_id: str,
    attempt_index: int,
    n_passages: int,
    top3: List[Any],
    render_meta: Dict[str, Any],
    *,
    citation_count: Optional[int],
    latency_ms: float,
    outcome: str,
) -> None:
    """Emit one ``grounded_answer_composition`` event (None capture skips).

    Rationale interpolates dynamic, replayable signals: query sha,
    engine, model, prompt version, passage count + top-3 (chunk_id,
    score), dropped-context count, response citation count, latency, and
    the per-attempt parse/remediation outcome.
    """
    if capture is None:
        return
    dropped = render_meta.get("dropped_count", 0)
    cite_str = "n/a" if citation_count is None else str(citation_count)
    rationale = (
        f"composed answer attempt {attempt_index} for query {query_sha} "
        f"(course={course_code or '?'}, engine={engine}) via model "
        f"{model_id} prompt {ANSWER_PROMPT_VERSION}; n_passages="
        f"{n_passages} top3={top3} dropped_passages={dropped} "
        f"citations={cite_str} latency_ms={latency_ms:.1f} "
        f"outcome={outcome}"
    )
    try:
        capture.log_decision(
            decision_type="grounded_answer_composition",
            decision=f"answer_compose:{outcome}",
            rationale=rationale,
            context=f"attempt={attempt_index}",
        )
    except Exception:  # pragma: no cover - capture must never break the path
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attr(obj: Any, name: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _opt_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val)
    return s or None


def _client_model_id(client: Any) -> str:
    for name in ("model", "model_id", "_model"):
        val = getattr(client, name, None)
        if val:
            return str(val)
    return "unknown"


def _is_transport_failure(exc: Exception) -> bool:
    """True for connection/timeout-class failures (→ backend unavailable).

    Recognizes raw ``httpx`` transport errors (the FakeAnswerClient and
    a directly-constructed client raise these) AND the layer-2 client's
    ``SynthesisProviderError(code="max_retries_exceeded")`` transport
    wrapper. A non-transport ``SynthesisProviderError`` (e.g. a 4xx or a
    malformed body) is NOT a backend-availability problem and bubbles.
    """
    name = type(exc).__name__
    if name in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "ConnectionError",
    }:
        return True
    code = getattr(exc, "code", None)
    if code == "max_retries_exceeded":
        return True
    return False


# Re-export so callers import the typed error from one place.
from lib.retrieval.answer_backend import (  # noqa: E402
    AnswerBackendUnavailable,
    PromptTruncatedError,
)


__all__ = [
    "RetrievedPassage",
    "ComposedAnswer",
    "AnswerComposeError",
    "InvalidCitationError",
    "AnswerBackendUnavailable",
    "PromptTruncatedError",
    "compose_answer",
]
