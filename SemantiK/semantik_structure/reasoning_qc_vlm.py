"""Transport client for the Stage-9b reasoning-QC pass (``SEMANTIK_REASONING_QC``).

**Text-only reasoning client (2026-07-12 pivot).** The QC pass judges a
document's *reading order* and *structural TYPE* from the ASSEMBLED
accessible-HTML-side block sequence — the ordered, typed, page-annotated block
text the cascade already hands to :func:`reasoning_qc.run_reasoning_qc` — **NOT**
from page-image rasters. The historic design fed a re-rendered source raster +
block list to a VLM; live evidence (a dense 216-DPI scan chapter) showed the
rumination that exhausted the completion window was IMAGE-driven, so the pivot
drops the image entirely and reasons over TEXT. No image bytes can enter a QC
request after this change.

The module filename ``reasoning_qc_vlm.py`` is kept for churn-control; the name
is HISTORICAL — this is a plain OpenAI-compatible **text** chat client (no
``image_url`` part, no page render), used with a reasoning model, not a VLM.

Self-containment (hard constraint): this module imports ONLY
``semantik_structure.vlm_extract`` and ``semantik_structure.extract_shared`` — it
never touches ``lib/`` or ``Trainforge`` — so it survives the cross-venv SemantiK
conversion subprocess. The orchestrator (:mod:`reasoning_qc`) owns the
DecisionCapture wiring (a best-effort lazy ``lib.decision_capture`` import that is
allowed to fail); this transport module stays pure.

Seat resolution (2026-07-12 pivot — OWN reasoning seat, decoupled from the VLM
family). :func:`resolve_reasoning_qc_seat` resolves a TEXT reasoning endpoint via
a three-tier chain (first hit wins): (1) an explicit reasoning-QC seat
(``SEMANTIK_REASONING_QC_BASE_URL`` / ``_API_KEY`` / ``_MODEL``); (2) the
SPECIALIST text seat (``SEMANTIK_SPECIALIST_BASE_URL`` / ``_API_KEY`` + the QC
model override or the specialist review model) — the "reasoning model, not VLM"
endpoint; (3) as a last-resort COMPAT fallback, the legacy VLM seat
(``SEMANTIK_VLM_*``), with a deprecation-style warning. :meth:`VLMSeat.require_ready`
is called BEFORE any POST (fail-loud on a non-loopback hosted seat without a key —
never a silent stub).

Fail-soft contract: a parse miss or a :class:`VlmExtractError` (after the bounded
retries) degrades to an EMPTY verdict ``{}`` — this module NEVER raises into the
cascade. A broken QC seat can only fail to *improve* a document, never corrupt one.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

# Self-containment boundary: SemantiK-local imports ONLY.
from . import extract_shared
from . import vlm_extract
from .extract_shared import VLMSeat, resolve_vlm_seat
from .vlm_extract import VlmExtractError

logger = logging.getLogger(__name__)


class QCNullContentError(VlmExtractError):
    """The QC endpoint returned a message with ``content is None``.

    A distinct :class:`VlmExtractError` subclass (carries ``transient=False`` so
    the bounded transient-retry ladder in :func:`_post_with_retry` re-raises it
    immediately rather than burning the retry budget) so :func:`run_qc_judgment`
    can route the window into the **reasoning-preserving split ladder**. On a
    dense block window a reasoning seat (thinking-on) can exhaust the completion
    window on reasoning tokens — everything lands in ``reasoning_content`` and the
    JSON answer (``content``) never arrives. The fix is to SHRINK the block window
    (split-until-it-reasons) and re-judge each half **thinking-on**, NOT to fall
    back to a thinking-off verdict: reasoning is the whole point of the QC pass
    (OWNER DIRECTIVE — QC must NEVER emit a non-thinking judgment automatically).
    """


class QCTimeoutError(VlmExtractError):
    """A QC POST timed out (a ``requests`` ReadTimeout / ConnectTimeout).

    Carries ``transient=True`` so the bounded transient-retry ladder in
    :func:`_post_with_retry` still retries it once (the existing posture), but a
    DISTINCT subclass so that — after the transient retries are exhausted —
    :func:`run_qc_judgment` routes the window into the SAME reasoning-preserving
    split ladder as a null-content result. A wall-clock timeout on a dense window
    is the same over-deliberation failure that produces a null content; the cure
    is a smaller window judged thinking-on, NEVER a thinking-off retry.
    """


# The QC model-id override env (satellite of SEMANTIK_REASONING_QC). Default →
# the specialist review model (see _resolve_specialist_review_model). Selecting a
# DISTINCT model here is the only condition that adds a docs/LICENSING.md row.
_QC_MODEL_ENV = "SEMANTIK_REASONING_QC_MODEL"

# Own reasoning-QC seat envs (2026-07-12 pivot — decoupled from SEMANTIK_VLM_*).
_QC_BASE_URL_ENV = "SEMANTIK_REASONING_QC_BASE_URL"
_QC_API_KEY_ENV = "SEMANTIK_REASONING_QC_API_KEY"

# Tier-2 SPECIALIST text seat envs + the specialist review-model resolution chain.
_SPECIALIST_BASE_URL_ENV = "SEMANTIK_SPECIALIST_BASE_URL"
_SPECIALIST_API_KEY_ENV = "SEMANTIK_SPECIALIST_API_KEY"
_STRUCTURE_REVIEW_MODEL_ENV = "SEMANTIK_STRUCTURE_REVIEW_MODEL"
_SPECIALIST_MODEL_ENV = "SEMANTIK_SPECIALIST_MODEL"
_NVIDIA_LARGE_MODEL_ENV = "NVIDIA_LARGE_MODEL"
_SPECIALIST_DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

# The QC-specific thinking-mode env. INDEPENDENT of the extraction client's
# SEMANTIK_VLM_DISABLE_THINKING so QC can keep thinking ON while extraction runs
# thinking OFF on the same reasoning seat (opposite modes, one process).
_QC_DISABLE_THINKING_ENV = "SEMANTIK_REASONING_QC_DISABLE_THINKING"


def resolve_reasoning_qc_disable_thinking() -> bool:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_DISABLE_THINKING`` (default off).

    Default OFF → QC keeps thinking ON: the QC POST carries NO thinking-off
    ``chat_template_kwargs`` block, because reasoning-QC is a structural /
    reading-order JUDGMENT task and chain-of-thought is the whole point of a
    reasoning pass. This QC flag is DELIBERATELY separate from
    ``SEMANTIK_VLM_DISABLE_THINKING`` so extraction can run thinking-off (~10×
    faster page transcription) AND QC can run thinking-on in the same process.
    When truthy, the QC POST carries
    ``chat_template_kwargs={"thinking": False, "enable_thinking": False}`` so a
    REASONING model skips the ``<think>`` block for QC too. This env is the SOLE
    condition under which a non-thinking QC request is ever emitted (the split
    ladder always re-judges thinking-on — OWNER DIRECTIVE)."""
    return (
        (os.environ.get(_QC_DISABLE_THINKING_ENV) or "").strip().lower()
        in vlm_extract._VLM_THINKING_TRUTHY
    )

# Bounded transient-retry knobs — reuse the extraction client's constants so QC
# retry posture tracks extraction (fall back to literals if the names move).
_QC_MAX_RETRIES = getattr(vlm_extract, "_VLM_MAX_RETRIES", 2)
_QC_RETRY_BACKOFF_BASE_SECONDS = getattr(
    vlm_extract, "_VLM_RETRY_BACKOFF_BASE_SECONDS", 0.5
)


# ---------------------------------------------------------------------------
# Reasoning-preserving split-ladder + QC-POST timeout knobs.
# ---------------------------------------------------------------------------
# Minimum block-window size the shrink-until-it-reasons ladder will judge — a
# window at/below this size that STILL nulls/times-out is recorded qc_incomplete
# rather than split further (never a thinking-off retry).
_QC_MIN_WINDOW_BLOCKS = 4

_QC_MAX_SPLIT_DEPTH_ENV = "SEMANTIK_REASONING_QC_MAX_SPLIT_DEPTH"
_QC_TIMEOUT_ENV = "SEMANTIK_REASONING_QC_TIMEOUT_SECONDS"
_QC_DEFAULT_MAX_SPLIT_DEPTH = 2
_QC_DEFAULT_TIMEOUT_SECONDS = 1200.0

# QC SAMPLING knobs (thinking-mode aware). A reasoning model in thinking mode
# (Nemotron etc.) needs temperature ~0.6 / top_p ~0.95 — greedy (temp 0) causes
# unterminated thinking loops that exhaust the completion window (content=null /
# finish_reason=length) on a content-dependent tail of prompts. When the
# operator EXPLICITLY disables thinking, greedy (temp 0, no top_p) is correct
# (non-reasoning transcription). max_tokens is ALWAYS pinned so a runaway thinker
# can't consume the whole remaining context (a length-finish still nulls and
# rides the reasoning-preserving split ladder).
_QC_TEMPERATURE_ENV = "SEMANTIK_REASONING_QC_TEMPERATURE"
_QC_TOP_P_ENV = "SEMANTIK_REASONING_QC_TOP_P"
_QC_MAX_TOKENS_ENV = "SEMANTIK_REASONING_QC_MAX_TOKENS"
_QC_DEFAULT_TEMPERATURE = 0.6
_QC_DEFAULT_TOP_P = 0.95
_QC_DEFAULT_MAX_TOKENS = 16384

# TRAINED thinking-budget control (Nemotron-3 chat template): passing
# chat_template_kwargs={"reasoning_budget": N} makes the template append
# `{thinking token budget: N}` to the last user message — a control the model
# was TRAINED to respect, so it terminates deliberation near N instead of
# free-running to the max_tokens cap (attempt-11 live evidence: temp 0.6 alone
# left ~75% of top-level QC units hitting the 16384 cap first-shot; vLLM
# histogram 85/184 requests at 10-20k generated tokens). Thinking-ON only —
# the explicit thinking-off branch already writes its own chat_template_kwargs
# and a budget is meaningless without thinking. `0` (or a falsey token)
# disables the kwarg entirely — the escape hatch for a non-Nemotron seat whose
# template does not define it (jinja typically ignores unknown kwargs, but the
# off-switch is the documented recourse). max_tokens stays the HARD backstop:
# a budget-violating runaway still fails fast into the split ladder.
_QC_REASONING_BUDGET_ENV = "SEMANTIK_REASONING_QC_REASONING_BUDGET"
_QC_DEFAULT_REASONING_BUDGET = 4096
_QC_BUDGET_FALSEY = frozenset({"0", "false", "no", "off"})

# Head/tail char budget kept per block when rendering a very long block body —
# order/typing judgment does not need the full body, so a long block keeps its
# head + tail with a middle-omitted marker (see _render_block_text).
_QC_BLOCK_HEAD_TAIL_CHARS = 400


def resolve_reasoning_qc_max_split_depth() -> int:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_MAX_SPLIT_DEPTH`` (default 2).

    The maximum number of times the null/timeout split ladder halves a block
    window before recording ``qc_incomplete`` for the residual blocks. ``0`` is
    honoured (never split — a first null/timeout immediately records
    qc_incomplete). Blank / non-int / negative / garbage → the default 2. Read at
    CALL time (never cached at import)."""
    raw = os.environ.get(_QC_MAX_SPLIT_DEPTH_ENV)
    if not raw:
        return _QC_DEFAULT_MAX_SPLIT_DEPTH
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _QC_DEFAULT_MAX_SPLIT_DEPTH
    if val < 0:
        return _QC_DEFAULT_MAX_SPLIT_DEPTH
    return val


def resolve_reasoning_qc_timeout() -> float:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_TIMEOUT_SECONDS`` (default 1200).

    The per-request HTTP timeout (float seconds) for QC POSTs — a thinking-on
    reasoning judgment on a dense text window runs far longer than a plain chat
    call. A ReadTimeout at this ceiling is treated as the wall-clock flavor of the
    over-deliberation failure and enters the split ladder (see
    :func:`run_qc_judgment`). Blank / non-float / non-finite / non-positive /
    garbage → the default 1200.0. Read at CALL time."""
    raw = os.environ.get(_QC_TIMEOUT_ENV)
    if not raw:
        return _QC_DEFAULT_TIMEOUT_SECONDS
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _QC_DEFAULT_TIMEOUT_SECONDS
    if not math.isfinite(val) or val <= 0:
        return _QC_DEFAULT_TIMEOUT_SECONDS
    return val


def resolve_reasoning_qc_temperature() -> float:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_TEMPERATURE`` (default 0.6).

    The thinking-ON sampling temperature (a reasoning model needs ~0.6 to
    terminate its ``<think>`` block; greedy 0 loops). Only consulted when thinking
    is ON (see :func:`resolve_reasoning_qc_sampling`). Blank / non-float /
    non-finite / negative / garbage → the default 0.6 (``0`` is honoured — an
    operator can force greedy). Read at CALL time."""
    raw = os.environ.get(_QC_TEMPERATURE_ENV)
    if not raw:
        return _QC_DEFAULT_TEMPERATURE
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _QC_DEFAULT_TEMPERATURE
    if not math.isfinite(val) or val < 0:
        return _QC_DEFAULT_TEMPERATURE
    return val


def resolve_reasoning_qc_top_p() -> float:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_TOP_P`` (default 0.95).

    The thinking-ON nucleus-sampling top_p (paired with the temperature above).
    Only consulted when thinking is ON. Blank / non-float / non-finite / out of
    ``(0, 1]`` / garbage → the default 0.95. Read at CALL time."""
    raw = os.environ.get(_QC_TOP_P_ENV)
    if not raw:
        return _QC_DEFAULT_TOP_P
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _QC_DEFAULT_TOP_P
    if not math.isfinite(val) or val <= 0 or val > 1:
        return _QC_DEFAULT_TOP_P
    return val


def resolve_reasoning_qc_max_tokens() -> int:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_MAX_TOKENS`` (default 16384).

    The hard completion-token ceiling on every QC POST (thinking ON or OFF) so a
    runaway thinker cannot consume the whole remaining serving window (~30+ min).
    A length-finish at this ceiling still raises :class:`QCNullContentError` and
    rides the reasoning-preserving split ladder (intent-preserving — the ladder
    re-judges thinking-on on a smaller window). Blank / non-int / non-positive /
    garbage → the default 16384. Read at CALL time."""
    raw = os.environ.get(_QC_MAX_TOKENS_ENV)
    if not raw:
        return _QC_DEFAULT_MAX_TOKENS
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _QC_DEFAULT_MAX_TOKENS
    if val <= 0:
        return _QC_DEFAULT_MAX_TOKENS
    return val


def resolve_reasoning_qc_reasoning_budget() -> int:
    """Parse-with-fallback ``SEMANTIK_REASONING_QC_REASONING_BUDGET`` (default 4096).

    The Nemotron-3 TRAINED thinking-budget: emitted as
    ``chat_template_kwargs={"reasoning_budget": N}`` on thinking-ON QC POSTs so
    the chat template appends ``{thinking token budget: N}`` to the last user
    message and the model terminates deliberation near N. ``0`` (or an explicit
    falsey token ``false``/``no``/``off``) DISABLES the kwarg entirely (returns
    0) — the escape hatch for a non-Nemotron seat. Blank / non-int / negative /
    garbage → the default 4096. Read at CALL time."""
    raw = (os.environ.get(_QC_REASONING_BUDGET_ENV) or "").strip()
    if not raw:
        return _QC_DEFAULT_REASONING_BUDGET
    if raw.lower() in _QC_BUDGET_FALSEY:
        return 0
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _QC_DEFAULT_REASONING_BUDGET
    if val < 0:
        return _QC_DEFAULT_REASONING_BUDGET
    return val


def resolve_reasoning_qc_sampling() -> dict[str, Any]:
    """Return the EFFECTIVE QC sampling params (thinking-mode aware).

    Single source of truth for BOTH the POST body (:func:`_post_qc_completion`)
    and the per-unit resume-cache fingerprint (``reasoning_qc._qc_unit_fingerprint``
    — sampling changes the verdict, so a cache HIT across different sampling would
    be wrong). Thinking ON (default) → ``temperature`` /``top_p`` from the envs
    above (0.6 / 0.95) + the Nemotron-3 TRAINED ``reasoning_budget`` (default
    4096; ``None`` when disabled via ``0``/falsey). Thinking OFF (explicit
    ``SEMANTIK_REASONING_QC_DISABLE_THINKING``) → greedy ``temperature`` 0.0 with
    NO ``top_p`` override and NO ``reasoning_budget`` (a budget is meaningless
    without thinking). ``max_tokens`` is always pinned (the hard backstop — a
    budget-violating runaway still fails fast into the split ladder)."""
    max_tokens = resolve_reasoning_qc_max_tokens()
    if resolve_reasoning_qc_disable_thinking():
        return {
            "temperature": 0.0,
            "top_p": None,
            "max_tokens": max_tokens,
            "reasoning_budget": None,
        }
    budget = resolve_reasoning_qc_reasoning_budget()
    return {
        "temperature": resolve_reasoning_qc_temperature(),
        "top_p": resolve_reasoning_qc_top_p(),
        "max_tokens": max_tokens,
        "reasoning_budget": budget if budget > 0 else None,
    }


# ---------------------------------------------------------------------------
# System directive — a TEXT-ONLY structural-editor judgment, never a rewrite.
# ---------------------------------------------------------------------------
# QC judgment prompt-contract version — part of the per-unit resume-cache
# fingerprint (reasoning_qc._qc_unit_fingerprint). BUMP THIS whenever
# _QC_SYSTEM_DIRECTIVE or the _build_qc_user_text render shape changes in a way
# that would change the model's judgment, so a stale sidecar written under the
# old prompt is never served for the new prompt.
# v2 (2026-07-12): terse directive replacing the long defect-taxonomy prose —
# a controlled A/B on the SAME real exercise window showed the v1 directive
# ("strong reasoning ability" framing + elaborate (a)/(b) taxonomy + "reason
# step by step … under 1000 words") drove rumination (still generating at
# 11+ min / ~8k+ tokens) where a terse probe directive finished in 2.2-2.6k
# tokens / ~3-6 min; the added never-solve-math steer measured -15% tokens
# with sharper verdicts. Deliberation length is the reasoning_budget chat
# kwarg's job now — prompt-side coaching just adds rumination fuel.
QC_PROMPT_VERSION = 2

_QC_SYSTEM_DIRECTIVE = (
    "You are a document-structure QC reviewer. You get a converter's ordered "
    "list of text blocks, each labelled with index, structural TYPE, heading "
    "level, and page. Judge only whether the reading order and each block's "
    "TYPE are correct. "
    "Reading-order breaks: a thought that stops mid-sentence and resumes blocks "
    "later, or interleaved multi-column fragments. "
    "Structural errors: page furniture (running header, page number, caption, "
    "ToC entry) or exercise/answer-key apparatus typed as a heading, or a block "
    "with the wrong type. "
    "You judge STRUCTURE ONLY — never solve, verify, or re-compute any "
    "mathematics; exercise lists and answer dumps need only a structural "
    "glance. "
    "You NEVER rewrite, summarize, translate, or invent text — you emit ONLY a "
    "JSON judgment keyed by the given block indices. "
    "Respond with a single JSON object with these keys: "
    '"reading_order" (the block indices in the correct reading order, or [] if '
    'the given order is already correct); '
    '"phantom_headings" (blocks typed as a heading that are not a real section '
    'heading, each as {"index": int, "reason": str}); '
    '"apparatus_retype" (exercise/answer-key apparatus mistyped as a heading '
    'or body, each as {"index": int, "reason": str}); '
    '"misordered" (runs of indices whose emission order diverges from the '
    'correct reading order, each as {"run": [int, ...], "reason": str}); '
    '"confidence" (a float 0..1 for the overall judgment). '
    "Use ONLY the block indices you were given. If nothing is wrong, return "
    "every list empty. Output ONLY the JSON object, no prose, no code fence."
)


def _resolve_specialist_review_model() -> str:
    """The specialist TEXT reasoning model id (tier-2 seat / QC default model).

    Resolution (first non-empty wins): ``SEMANTIK_REASONING_QC_MODEL`` >
    ``SEMANTIK_STRUCTURE_REVIEW_MODEL`` > ``SEMANTIK_SPECIALIST_MODEL`` >
    ``NVIDIA_LARGE_MODEL`` > the literal specialist default. Mirrors the
    Stage-5d reviewer / Stage-6 specialist model chain WITHOUT importing
    ``qwen_specialists.runtime`` (self-containment boundary)."""
    for env in (
        _QC_MODEL_ENV,
        _STRUCTURE_REVIEW_MODEL_ENV,
        _SPECIALIST_MODEL_ENV,
        _NVIDIA_LARGE_MODEL_ENV,
    ):
        val = (os.environ.get(env) or "").strip()
        if val:
            return val
    return _SPECIALIST_DEFAULT_MODEL


def resolve_reasoning_qc_model(default_model: str | None = None) -> str:
    """Return the QC seat's model id.

    ``SEMANTIK_REASONING_QC_MODEL`` (satellite override) wins; otherwise the
    supplied ``default_model`` is used, and failing that the specialist review
    model (:func:`_resolve_specialist_review_model`). Read at CALL time.
    """
    raw = (os.environ.get(_QC_MODEL_ENV) or "").strip()
    if raw:
        return raw
    if default_model:
        return default_model
    return _resolve_specialist_review_model()


def resolve_reasoning_qc_seat() -> VLMSeat:
    """Resolve + fail-loud-check the QC TEXT reasoning seat (2026-07-12 pivot).

    Three-tier chain, FIRST HIT WINS — a reasoning TEXT endpoint, decoupled from
    the VLM extraction family:

      1. **Explicit reasoning-QC seat.** ``SEMANTIK_REASONING_QC_BASE_URL`` (+
         ``_API_KEY``, + the ``_MODEL`` override or the specialist review model).
      2. **Specialist text seat.** ``SEMANTIK_SPECIALIST_BASE_URL`` (+
         ``SEMANTIK_SPECIALIST_API_KEY``, + the QC model override or the
         specialist review model) — the "reasoning model, not VLM" endpoint the
         Stage-5d reviewer / Stage-6 specialists ride.
      3. **Legacy VLM seat (last-resort COMPAT).** ``SEMANTIK_VLM_*`` via
         :func:`extract_shared.resolve_vlm_seat`, with a deprecation-style
         ``logger.warning`` (this QC pass is a TEXT reasoning task after the pivot
         — a VLM seat is a compat fallback, not the intended path).

    :meth:`VLMSeat.require_ready` is called on the resolved seat FIRST — a
    non-local hosted seat with no credential raises
    :class:`extract_shared.VLMSeatError` here rather than fabricating a judgment.
    """
    model_override = (os.environ.get(_QC_MODEL_ENV) or "").strip()

    # (1) Explicit reasoning-QC seat.
    qc_base = (os.environ.get(_QC_BASE_URL_ENV) or "").strip()
    if qc_base:
        api_key = (os.environ.get(_QC_API_KEY_ENV) or "").strip() or None
        model = model_override or _resolve_specialist_review_model()
        return VLMSeat(
            provider="reasoning-qc", base_url=qc_base, api_key=api_key, model=model
        ).require_ready()

    # (2) Specialist TEXT seat (the reasoning endpoint, not the VLM).
    spec_base = (os.environ.get(_SPECIALIST_BASE_URL_ENV) or "").strip()
    if spec_base:
        api_key = (os.environ.get(_SPECIALIST_API_KEY_ENV) or "").strip() or None
        model = model_override or _resolve_specialist_review_model()
        return VLMSeat(
            provider="specialist", base_url=spec_base, api_key=api_key, model=model
        ).require_ready()

    # (3) Last-resort COMPAT: the legacy VLM seat.
    logger.warning(
        "reasoning-QC seat: neither %s nor %s is set — falling back to the LEGACY "
        "VLM seat (SEMANTIK_VLM_*). Reasoning-QC is a TEXT reasoning task after the "
        "2026-07-12 pivot; point it at a reasoning endpoint (%s or %s) to silence "
        "this warning.",
        _QC_BASE_URL_ENV, _SPECIALIST_BASE_URL_ENV, _QC_BASE_URL_ENV,
        _SPECIALIST_BASE_URL_ENV,
    )
    base = resolve_vlm_seat()
    model = model_override or base.model
    return VLMSeat(
        provider=base.provider, base_url=base.base_url, api_key=base.api_key, model=model
    ).require_ready()


# ---------------------------------------------------------------------------
# The TEXT-only POST — mirrors vlm_extract._post_chat_completion's HTTP posture,
# but the user turn is the rendered block list (no image_url part) + JSON mode.
# ---------------------------------------------------------------------------
def _render_block_text(text: str) -> str:
    """One block's body, whitespace-collapsed, middle-truncated when very long.

    Order/typing judgment does not need the full body — a very long block keeps
    its head + tail (``_QC_BLOCK_HEAD_TAIL_CHARS`` each) with a middle-omitted
    marker, so a giant block cannot blow the prompt budget while the reader can
    still see how the block opens and closes."""
    t = " ".join((text or "").split())
    if len(t) <= 2 * _QC_BLOCK_HEAD_TAIL_CHARS:
        return t
    head = t[:_QC_BLOCK_HEAD_TAIL_CHARS]
    tail = t[-_QC_BLOCK_HEAD_TAIL_CHARS:]
    dropped = len(t) - 2 * _QC_BLOCK_HEAD_TAIL_CHARS
    return f"{head} …[middle {dropped} chars omitted]… {tail}"


def _block_meta(block: Mapping[str, Any]) -> str:
    """The ``(TYPE[/role][ hN])`` metadata tag for one block record."""
    btype = str(block.get("type") or "?")
    role = block.get("role")
    level = block.get("level")
    meta = btype
    if role and str(role) != btype:
        meta += f"/{role}"
    if level is not None:
        meta += f" h{level}"
    return meta


def _build_qc_user_text(blocks: Sequence[Mapping[str, Any]]) -> str:
    """The user-turn text: the ordered, index/type/level/page-labelled block list.

    Each block is one line ``[i] (TYPE[/role][ hN]) [p.N] text`` where ``i`` is
    the block's LOCAL index in this judgment unit (the verdict is keyed by these
    indices; the orchestrator maps them back to capped region indices). The page
    number is purely informational per-block metadata — after the 2026-07-12
    text-only pivot there is NO page image."""
    lines = [
        "Blocks extracted from the document, in the converter's current reading "
        "order. Each line is: [index] (TYPE) [p.page] text.",
        "",
    ]
    for i, b in enumerate(blocks):
        page = b.get("page")
        page_tag = f" [p.{page}]" if page is not None else ""
        snippet = _render_block_text(b.get("text") or "")
        lines.append(f"[{i}] ({_block_meta(b)}){page_tag} {snippet}")
    lines.append("")
    lines.append(
        "Judge the reading order and per-block structural type from the text and "
        "structure above, and return the JSON judgment described in the system "
        "instructions."
    )
    return "\n".join(lines)


def _post_qc_completion(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    blocks: Sequence[Mapping[str, Any]],
    timeout: float,
    requests_module,
    directive: str = _QC_SYSTEM_DIRECTIVE,
) -> str:
    """One ``POST .../v1/chat/completions`` TEXT QC judgment call → raw JSON string.

    Mirrors :func:`vlm_extract._post_chat_completion`'s HTTP-status →
    transient/permanent classification, but:

      * the user turn is TEXT ONLY — the rendered block list, with **no**
        ``image_url`` part (2026-07-12 pivot: no page image ever enters a QC
        request);
      * ``response_format={"type": "json_object"}`` + ollama ``format="json"``
        request structured output;
      * ``chat_template_kwargs`` (thinking-off) is honored ONLY via the
        QC-specific :func:`resolve_reasoning_qc_disable_thinking` — the
        operator's EXPLICIT wholesale opt-in (default OFF → QC keeps thinking
        ON). There is NO automatic thinking-off path: the split ladder always
        re-judges thinking-on (OWNER DIRECTIVE).

    Raises :class:`QCNullContentError` (permanent) on a ``content is None``
    message and :class:`QCTimeoutError` (transient) on a request timeout so the
    caller can route the window into the reasoning-preserving split ladder.
    """
    # User content is a one-element list carrying ONLY a text part — NO
    # image_url. (Kept as a parts-list, not a bare string, so an OpenAI-compatible
    # server that inspects parts still sees a well-formed multimodal-shaped turn;
    # the load-bearing invariant is that no part is ever an image.)
    # Thinking-mode-aware sampling: a reasoning model in thinking mode needs
    # temperature ~0.6 / top_p ~0.95 to TERMINATE its <think> block (greedy 0
    # loops → content=null / finish_reason=length); an explicit thinking-off run
    # keeps greedy (temp 0, no top_p). max_tokens is always pinned so a runaway
    # thinker can't eat the whole serving window.
    sampling = resolve_reasoning_qc_sampling()
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": directive},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_qc_user_text(blocks)},
                ],
            },
        ],
        "temperature": sampling["temperature"],
        "max_tokens": sampling["max_tokens"],
        # Structured-output request — OpenAI dialect + ollama dialect. A server
        # that ignores one still sees the other; a server that honors neither is
        # handled by the tolerant JSON parse downstream.
        "response_format": {"type": "json_object"},
        "format": "json",
    }
    if sampling["top_p"] is not None:
        body["top_p"] = sampling["top_p"]
    # Thinking-off is emitted ONLY on the operator's explicit env opt-in — there
    # is NO automatic fallback (the split ladder always re-judges thinking-on).
    # Budget is thinking-ON only: the explicit thinking-off branch writes its
    # OWN chat_template_kwargs and never carries reasoning_budget.
    if resolve_reasoning_qc_disable_thinking():
        body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    elif sampling.get("reasoning_budget") is not None:
        # Nemotron-3 TRAINED thinking-budget control: the chat template appends
        # `{thinking token budget: N}` to the last user message, so the model
        # terminates deliberation near N instead of free-running to max_tokens.
        body["chat_template_kwargs"] = {"reasoning_budget": sampling["reasoning_budget"]}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = vlm_extract._chat_completions_url(base_url)

    try:
        resp = requests_module.post(url, json=body, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — timeout / conn → transient
        # A timeout is the wall-clock flavor of the over-deliberation failure —
        # tag it distinctly (QCTimeoutError) so the caller's split ladder can
        # catch it after the transient-retry ladder is exhausted.
        is_timeout = "Timeout" in type(exc).__name__
        err_cls = QCTimeoutError if is_timeout else VlmExtractError
        raise err_cls(
            f"reasoning-QC request to endpoint failed ({type(exc).__name__}): {exc}",
            transient=True,
        ) from exc

    status = int(getattr(resp, "status_code", 200))
    if status != 200:
        body_head = (getattr(resp, "text", "") or "")[:500]
        transient = status >= 500 or status == 429
        raise VlmExtractError(
            f"reasoning-QC endpoint returned HTTP {status} (model={model}). "
            f"Body head: {body_head!r}",
            transient=transient,
        )
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise VlmExtractError(
            "reasoning-QC endpoint returned a non-JSON body", transient=False
        ) from exc
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        keys = list(data) if isinstance(data, dict) else type(data)
        raise VlmExtractError(
            f"reasoning-QC endpoint returned a malformed response "
            f"(no choices[0].message.content): keys={keys}",
            transient=False,
        ) from exc
    if text is None:
        raise QCNullContentError(
            "reasoning-QC endpoint returned a null message content", transient=False
        )
    return str(text)


def _post_with_retry(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    blocks: Sequence[Mapping[str, Any]],
    timeout: float,
    requests_module,
    directive: str = _QC_SYSTEM_DIRECTIVE,
) -> str:
    """Bounded transient-retry around :func:`_post_qc_completion`.

    Mirrors :func:`vlm_extract._post_with_retry`. Module-level so tests can
    monkeypatch it to return canned JSON (no torch / ollama / weights load).
    Re-raises the last :class:`VlmExtractError` on a permanent error or after
    the retry budget; the caller (:func:`run_qc_judgment` / its split ladder)
    owns the null / timeout handling. A permanent :class:`QCNullContentError` is
    NOT retried by THIS ladder (it re-raises immediately so the split ladder
    fires); a transient :class:`QCTimeoutError` rides the transient retries here
    first, then re-raises into the split ladder once exhausted.
    """
    import time  # noqa: WPS433 — local, mirrors the lazy import in vlm_extract

    attempt = 0
    while True:
        try:
            return _post_qc_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                blocks=blocks,
                timeout=timeout,
                requests_module=requests_module,
                directive=directive,
            )
        except VlmExtractError as exc:
            if not exc.transient or attempt >= _QC_MAX_RETRIES:
                raise
            attempt += 1
            backoff = _QC_RETRY_BACKOFF_BASE_SECONDS * attempt
            logger.warning(
                "reasoning-QC transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                _QC_MAX_RETRIES,
                exc,
                backoff,
            )
            time.sleep(backoff)


# ---------------------------------------------------------------------------
# Tolerant JSON parse.
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _first_balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, or ``None``."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_qc_json(text: str | None) -> dict[str, Any]:
    """Parse the model's judgment string tolerantly → dict (``{}`` on miss).

    Tolerates a bare object, a ```json``` fenced block, or an object embedded in
    surrounding prose. NEVER raises — a parse miss is the empty verdict.
    """
    if not text or not text.strip():
        return {}
    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)
    balanced = _first_balanced_object(text)
    if balanced:
        candidates.append(balanced)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return {}


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------
def run_qc_judgment(
    seat: VLMSeat,
    pdf_path: str | Path,
    page_num: int,
    blocks: Sequence[Mapping[str, Any]],
    *,
    directive: str = _QC_SYSTEM_DIRECTIVE,
    requests_module=None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """POST the TEXT QC judgment (thinking-on) for one block window → verdict dict.

    2026-07-12 pivot: the judgment is TEXT-ONLY — ``blocks`` is the ordered list
    of block RECORDS (``{type, role, level, page, text}``) rendered into the user
    prompt; there is NO page render. ``pdf_path`` is RETAINED for provenance /
    logging only (the cascade call site still passes it) — it is never used to
    render anything. ``page_num`` is a representative page for log messages (each
    block carries its own page in the record).

    Reasoning-preserving contract (OWNER DIRECTIVE): the QC judgment ALWAYS runs
    thinking-on (unless the operator explicitly set
    ``SEMANTIK_REASONING_QC_DISABLE_THINKING`` for a wholesale thinking-off run).
    On a dense block window the reasoning seat can exhaust its completion window
    on reasoning tokens (``content is None`` → :class:`QCNullContentError`) or
    hit the QC POST timeout (:class:`QCTimeoutError`). Instead of the retired
    thinking-off fallback, the **split ladder** halves the block window
    (respecting :data:`_QC_MIN_WINDOW_BLOCKS`) and re-judges each half
    thinking-on, recursing at most :func:`resolve_reasoning_qc_max_split_depth`
    levels; a minimum-size window that still fails is recorded ``_qc_incomplete``
    (fail-soft skip for those blocks) — NEVER a non-thinking retry.

    The returned verdict is keyed by block indices INTO ``blocks`` (0-based) and,
    on a split ladder, may carry an ADDITIVE ``_qc_incomplete`` list of the block
    indices the ladder could not judge. Fail-soft: any transport / permanent
    error degrades to ``{}`` (an empty verdict = "nothing flagged"), so a broken
    seat never raises into the cascade. The caller supplies an
    already-``require_ready`` seat (see :func:`resolve_reasoning_qc_seat`).
    """
    req = requests_module or vlm_extract._lazy_requests()
    tmo = timeout if timeout is not None else resolve_reasoning_qc_timeout()
    try:
        return _judge_window_ladder(
            seat=seat,
            page_num=page_num,
            blocks=list(blocks),
            timeout=tmo,
            requests_module=req,
            directive=directive,
            depth=0,
        )
    except VlmExtractError as exc:
        # A NON-null, NON-timeout transport error (permanent 4xx, or a transient
        # connection error after the retry budget) fails soft to the empty
        # verdict — the historic contract. Null / timeout never reach here: the
        # ladder consumes them (and records qc_incomplete when exhausted).
        logger.warning("reasoning-QC call failed fail-soft (page %s): %s", page_num, exc)
        return {}


# ---------------------------------------------------------------------------
# Shrink-until-it-reasons split ladder (replaces the retired thinking-off
# fallback). EVERY judgment call is thinking-on; the ONLY lever is a SMALLER
# block window. Terminal on exhaustion = qc_incomplete, never a thinking-off
# retry (OWNER DIRECTIVE: reasoning is the point of the QC pass).
# ---------------------------------------------------------------------------
def _judge_window_ladder(
    *,
    seat: VLMSeat,
    page_num: int,
    blocks: list[Mapping[str, Any]],
    timeout: float,
    requests_module,
    directive: str,
    depth: int,
) -> dict[str, Any]:
    """POST one block window thinking-on; on null / timeout, enter the ladder.

    Returns a verdict keyed by indices INTO ``blocks`` (0-based). A NON-null,
    NON-timeout :class:`VlmExtractError` propagates (the caller fails it soft to
    ``{}`` for the whole window)."""
    try:
        raw = _post_with_retry(
            base_url=seat.base_url,
            api_key=seat.api_key,
            model=seat.model,
            blocks=blocks,
            timeout=timeout,
            requests_module=requests_module,
            directive=directive,
        )
        return parse_qc_json(raw)
    except (QCNullContentError, QCTimeoutError) as exc:
        return _split_ladder(
            seat=seat,
            page_num=page_num,
            blocks=blocks,
            timeout=timeout,
            requests_module=requests_module,
            directive=directive,
            depth=depth,
            cause=exc,
        )


def _split_ladder(
    *,
    seat: VLMSeat,
    page_num: int,
    blocks: list[Mapping[str, Any]],
    timeout: float,
    requests_module,
    directive: str,
    depth: int,
    cause: VlmExtractError,
) -> dict[str, Any]:
    """Halve the block window and re-judge each half thinking-on (bounded).

    Terminal (depth cap reached OR the window can no longer split into two
    ≥ :data:`_QC_MIN_WINDOW_BLOCKS` halves): record ``_qc_incomplete`` for these
    blocks — a HONEST fail-soft skip, NEVER a thinking-off retry."""
    n = len(blocks)
    max_depth = resolve_reasoning_qc_max_split_depth()
    if depth >= max_depth or n < 2 * _QC_MIN_WINDOW_BLOCKS:
        logger.warning(
            "reasoning-QC window (%d block(s), depth %d, page %s) still failed "
            "(%s: %s) — split ladder exhausted; recording qc_incomplete for these "
            "block(s). NOT retrying thinking-off (reasoning is the point of the "
            "QC pass).",
            n, depth, page_num, type(cause).__name__, cause,
        )
        return {"_qc_incomplete": list(range(n))}
    mid = n // 2
    logger.info(
        "reasoning-QC window (%d blocks, page %s) %s — splitting to halves "
        "[0:%d]+[%d:%d], re-judging thinking-on (depth %d/%d).",
        n, page_num, type(cause).__name__, mid, mid, n, depth + 1, max_depth,
    )
    left = _judge_window_ladder(
        seat=seat,
        page_num=page_num,
        blocks=blocks[:mid],
        timeout=timeout,
        requests_module=requests_module,
        directive=directive,
        depth=depth + 1,
    )
    right = _judge_window_ladder(
        seat=seat,
        page_num=page_num,
        blocks=blocks[mid:],
        timeout=timeout,
        requests_module=requests_module,
        directive=directive,
        depth=depth + 1,
    )
    return _merge_ladder_halves(left, right, mid)


def _merge_ladder_halves(
    left: dict[str, Any], right: dict[str, Any], mid: int
) -> dict[str, Any]:
    """Union the two DISJOINT half verdicts, shifting the right half by ``+mid``.

    The halves partition ``[0, n)`` with no shared block, so the union is
    conflict-free. ``reading_order`` is deliberately DROPPED on a split merge
    (no clean global permutation across halves — the conservative reading);
    per-block re-type findings + explicit ``misordered`` runs + ``_qc_incomplete``
    are unioned; ``confidence`` is the min (conservative)."""
    merged: dict[str, Any] = {}
    for key in ("phantom_headings", "apparatus_retype"):
        items = list(left.get(key) or ())
        for item in right.get(key) or ():
            shifted = _shift_index_item(item, mid)
            if shifted is not None:
                items.append(shifted)
        if items:
            merged[key] = items
    runs = list(left.get("misordered") or ())
    for r in right.get("misordered") or ():
        shifted = _shift_run(r, mid)
        if shifted is not None:
            runs.append(shifted)
    if runs:
        merged["misordered"] = runs
    confs = [
        c
        for c in (_as_float(left.get("confidence")), _as_float(right.get("confidence")))
        if c is not None
    ]
    if confs:
        merged["confidence"] = min(confs)
    incomplete = list(_int_list(left.get("_qc_incomplete")))
    incomplete += [mid + i for i in _int_list(right.get("_qc_incomplete"))]
    if incomplete:
        merged["_qc_incomplete"] = sorted(set(incomplete))
    return merged


# --- tiny index-shift helpers (ladder-local) --------------------------------
def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for x in value:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _shift_index_item(item: Any, shift: int) -> Any | None:
    """Shift a ``{"index": int, ...}`` (or bare int) block reference by ``+shift``."""
    if isinstance(item, dict):
        try:
            return {**item, "index": int(item.get("index")) + shift}
        except (TypeError, ValueError):
            return None
    try:
        return int(item) + shift
    except (TypeError, ValueError):
        return None


def _shift_run(run: Any, shift: int) -> Any | None:
    """Shift a ``{"run": [int, ...], ...}`` (or bare list) by ``+shift``."""
    raw = run.get("run") if isinstance(run, dict) else run
    if not isinstance(raw, (list, tuple)):
        return None
    shifted: list[int] = []
    for x in raw:
        try:
            shifted.append(int(x) + shift)
        except (TypeError, ValueError):
            return None
    if isinstance(run, dict):
        return {**run, "run": shifted}
    return {"run": shifted}


def unload_qc_model(seat: VLMSeat, *, requests_module=None) -> bool:
    """Best-effort ``keep_alive:0`` unload of the QC model. NEVER raises.

    Delegates to :func:`vlm_extract.unload_vlm_model` so a local (ollama) QC seat
    frees the card at the pass's ``finally``. A no-op for a hosted reasoning seat
    that ignores ``keep_alive`` (still best-effort, never raises).
    """
    try:
        return vlm_extract.unload_vlm_model(
            seat.base_url, seat.model, requests_module=requests_module
        )
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("reasoning-QC unload failed (non-fatal): %s", exc)
        return False
