"""Super heading-level JUDGE pass for the GLM-OCR lane (``SEMANTIK_HEADING_JUDGE``).

The deterministic transform (:mod:`.transform`) defaults every non-numbered
title to ``level=3`` and flags it ``heading_level_pending`` + a matching
``heading_level_pending`` escalation row. That is honest (the sidecar contract
records the uncertainty) but leaves the tree flat: a section's body subsection,
its objectives-list restatement, and its end-of-chapter review restatement all
sit at the same level. This pass hands the WHOLE chapter's ordered heading
skeleton to a reasoning model (Super/Nemotron-3), which assigns the correct
level (2-6) to EVERY pending heading so the tree is hierarchy-consistent, then
DETERMINISTICALLY re-stamps the pending headings (level only, never text) under
strict clamp rules that mirror the ``_anchor_declared_sections`` re-stamp +
escalation-hygiene precedent.

Design rule (the SemantiK house rule): the model PROPOSES levels; deterministic
code DECIDES. Every clamp is auditable, the fixed N.M spine / synthesized
chapter opener are immovable anchors, and any transport/parse failure fails
OPEN — the chapter keeps ALL current levels byte-identically and the pending
flags + escalations are RETAINED (no silent resolution).

DEFAULT ON (deviation — owner directive: the Super heading-level judge must not
be optional). Only an explicit falsey token (``SEMANTIK_HEADING_JUDGE`` in
``0``/``false``/``no``/``off``, case-insensitive) opts out; when explicitly off
this module is never imported by the lane, so ``region_provenance`` /
``heading_tree`` / escalations are byte-identical. A no-pending chapter is a
natural no-op (no POST) even with the gate on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import (
    resolve_heading_judge_base_url,
    resolve_heading_judge_checkpoint,
    resolve_heading_judge_model,
    resolve_heading_judge_timeout,
)
from . import region_map as rm
from .heading_judge_audit import normalize_signature

logger = logging.getLogger(__name__)

# Prompt-contract version — folded into the cache key so a prompt change moves
# every sidecar key (a stale verdict is never served for a changed prompt).
JUDGE_PROMPT_VERSION = 1

# Token-budget target for one chapter digest (the ~24k whole-book ceiling; a
# 650-heading chapter ≈ 11k with pending-only anchors). A digest over this
# drops content anchors, then splits into N.M-anchored windows. LEGACY DEFAULT
# (65k-seat) — the effective digest budget is resolved by
# `resolve_digest_budget_tokens()`: explicit env > seat-derived > this default.
_DIGEST_BUDGET_TOKENS = 24000

_HEADING_TEXT_TRUNCATE = 90
_CONTEXT_TEXT_TRUNCATE = 40
_ANCHOR_TRUNCATE = 80

_TRANSIENT_RETRIES = 2  # extra attempts after the first, linear backoff
# Completion-token CEILING. DELIBERATELY NOT raised past 30000: the Super seat's
# context window is 65,536 tokens and the prompt is windowed to the
# `_CTX_TOKENS_BUDGET` (31,500) prompt budget, so the worst-case FIRST attempt
# (a full 31,500-token prompt + a completion that hits this ceiling) must fit
# the window — and it does with margin: 31,500 + 30,000 = 61,500 < 65,536. The
# single doubled-max_tokens retry also clamps to this ceiling (see
# `_judge_window`), so it can never overflow either. Raising this would break
# that invariant. Env-tunable via SEMANTIK_HEADING_JUDGE_MAX_TOKENS (raise it
# only alongside SEMANTIK_HEADING_JUDGE_CTX_BUDGET for a longer-window seat).
_MAX_TOKENS_CEILING = 30000
# The completion budget must hold the THINKING block too — a thinking-ON judgment
# over a ~96-pending window deliberates 5-15k tokens before the JSON, so the old
# floor length-exhausted EVERY window (and its doubled retry), yielding zero
# verdicts. reasoning_budget is dead on this seat (chat template ignores it);
# max_tokens is the only real ceiling. The 20480 raise gives small-but-thinking-
# heavy windows base room: a live n_pending=3 window truncated at exactly
# 16384 + 24×3 = 16456 (finish=length) under the old 16384 floor, i.e. its base
# thinking alone needed >16384. 20480 clears that with headroom.
_MAX_TOKENS_FLOOR = 20480

#: Concurrent window POSTs. Windows are independent judgments that the seat can
#: batch. Bounded, floor 1.
_WINDOW_CONCURRENCY = 4

# OUTPUT-side window cap: the ladder above was input-token driven only, so a
# chapter whose digest FITS the input budget still packed every pending heading
# into ONE window. A large set of judgments plus reasoning can exhaust
# max_tokens (finish=length), while a doubled retry can overflow the seat's context
# window entirely (vLLM 400 → retry-ladder exhaustion). Windows are therefore
# ALSO capped by pending count. _MAX_PENDING_PER_WINDOW is the HARD ceiling;
# the EFFECTIVE cap is budget-derived (resolve_pending_window_cap below).
_MAX_PENDING_PER_WINDOW = 96

# Per-JUDGMENT completion-token estimate driving the pending-count window cap
# A thinking-ON judgment costs more than its emitted JSON line because the
# deliberation also scales with
# the pending count, so a 78-pending window granted floor + 24×78 = 18,256
# tokens exhausted (finish=length) while its 39-pending half completed within
# 17,320. Implied per-judgment cost ≈ 105-290 tokens depending on the base
# thinking share; 300 is the CONSERVATIVE estimate (err toward smaller windows
# — an over-split window costs one extra POST, an under-split one costs a
# truncation hole).
_EST_TOKENS_PER_JUDGMENT = 300

# Per-pending increment on the max_tokens BUDGET (`_max_tokens_for`). This MUST
# track `_EST_TOKENS_PER_JUDGMENT` — the two estimate the SAME quantity
# (thinking-inclusive completion tokens per judgment): window PACKING sizes a
# window so N×`_EST_TOKENS_PER_JUDGMENT` fits the completion room, and the
# BUDGET must then grant that room (floor + increment×N). Defining it AS
# `_EST_TOKENS_PER_JUDGMENT` (single source of truth) forbids the drift that
# under-budgeted a packed window and truncated it on the first attempt (the
# old 24-vs-300 split). Must be defined AFTER `_EST_TOKENS_PER_JUDGMENT`.
_MAX_TOKENS_PER_PENDING = _EST_TOKENS_PER_JUDGMENT

# ── Thinking-OFF completion budgets. ─────────────────────────────────────────
# The floor/ceiling/per-pending above (20480/30000/300) are sized to hold the
# THINKING block — a thinking-ON judgment deliberates 5-15k tokens before the
# JSON. But heading-leveling is a CLASSIFICATION task and the judge now runs
# REASONING-OFF by DEFAULT (`resolve_heading_judge_enable_thinking`, False), so a
# judgment emits compact JSON with NO `<think>` block. The thinking-sized floor
# would therefore give small windows unnecessary room for degenerate repetition.
# The thinking-OFF budgets collapse to JSON-sized values so `clamp(512 + 64·n, 512,
# 4096)` bounds the real output; the pending-window cap self-consistently becomes
# `(4096-512)/64 = 56`. CONSUMED ONLY when thinking is OFF — a thinking-ON run
# keeps the values above (or the seat-context-derived path) BYTE-IDENTICALLY.
# Env-tunable via SEMANTIK_HEADING_JUDGE_{MAX_TOKENS,TOKENS_FLOOR,EST_PER_JUDGMENT}_THINKOFF.
_MAX_TOKENS_CEILING_THINKOFF = 4096
_MAX_TOKENS_FLOOR_THINKOFF = 512
_EST_TOKENS_PER_JUDGMENT_THINKOFF = 64

# Floor on the budget-derived cap so a tiny operator ceiling never degenerates
# into per-heading windows.
_MIN_PENDING_WINDOW_CAP = 8

# Hard cap on the coverage RE-SPLIT rounds a window may burn re-judging its
# residual UNJUDGED pendings before the explicit unjudged set is surfaced.
_MAX_COVERAGE_RESPLIT_ROUNDS = 3

# ── Seat-context-adaptive budget derivation defaults. ───────────────────────
# The judge budgets (digest/prompt, completion ceiling, doubled-retry ctx
# guard) are DERIVED to fit the judge SEAT's own context window instead of the
# 65k-seat hardcoded constants above: the judge reads its seat's
# `max_model_len` (GET {base_url}/models) ONCE and right-sizes every budget so
# `digest_budget + ceiling <= usable <= seat_context` holds by construction. A
# bigger seat (250k) then widens the windows/budgets with NO hand-math and no
# code change (`SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT=auto`, the default). The
# individual `SEMANTIK_HEADING_JUDGE_{MAX_TOKENS,CTX_BUDGET,DIGEST_BUDGET}`
# envs still OVERRIDE the derived value per-budget; `=off` reverts to the
# 65k-seat legacy constants byte-identically.
_DEFAULT_CTX_MARGIN = 4096          # SEMANTIK_HEADING_JUDGE_CTX_MARGIN
_DEFAULT_COMPLETION_FRACTION = 0.7  # SEMANTIK_HEADING_JUDGE_COMPLETION_FRACTION
_COMPLETION_FRACTION_MIN = 0.4      # clamp band (sane split of prompt/completion)
_COMPLETION_FRACTION_MAX = 0.9
# FLOORS so a tiny / misread seat context can never degenerate a budget below
# the working 65k-seat defaults: the derived completion ceiling holds at least
# the thinking FLOOR plus room for a min-cap window of judgments, and the
# derived digest/prompt budget never falls below a small floor. On the large
# seats these floors never bind (the invariant holds naturally); they only
# catch a pathological small/garbage `max_model_len`.
_DERIVED_CEILING_FLOOR = _MAX_TOKENS_FLOOR + _MIN_PENDING_WINDOW_CAP * _EST_TOKENS_PER_JUDGMENT
_DERIVED_DIGEST_FLOOR = 4096
# Timeout (s) for the ONE seat `/models` probe under `SEAT_CONTEXT=auto`. Short
# — a slow/absent seat must never block or slow the judge; it fails soft to the
# legacy hardcoded budgets.
_SEAT_CONTEXT_PROBE_TIMEOUT = 5.0
#: Explicit tokens meaning "use the legacy hardcoded budgets" for the seat
#: context flag (the byte-identical revert lever).
_SEAT_CONTEXT_OFF = {"off", "0", "false", "no"}
#: Module-level cache of the per-base_url seat-context probe (an absent seat is
#: cached as None so `auto` costs at most one probe per process). Reset via
#: `_reset_seat_context_cache()` (tests).
_SEAT_CONTEXT_QUERY_CACHE: Dict[str, Optional[int]] = {}
_SEAT_CONTEXT_LOGGED: set = set()


def resolve_pending_window_cap() -> int:
    """EFFECTIVE pendings-per-window cap, keyed to the completion budget.

    The completion room a window really has for judgments is the max_tokens
    CEILING minus the thinking FLOOR; dividing by the conservative
    per-judgment estimate yields the pending count whose window can actually
    COMPLETE (the audit's truncation hole: a 78-pending window under the old
    fixed cap silently left a contiguous 19-heading band unjudged). Bounded by
    the hard `_MAX_PENDING_PER_WINDOW` ceiling; floored so a tiny operator
    ceiling never degenerates into per-heading windows. A raised
    `SEMANTIK_HEADING_JUDGE_MAX_TOKENS` widens the cap automatically.
    """
    budget_cap = (
        (resolve_max_tokens_ceiling() - resolve_max_tokens_floor())
        // resolve_heading_judge_est_per_judgment()
    )
    return min(resolve_max_pending_per_window(),
               max(resolve_min_pending_window_cap(), budget_cap))

_CACHE_BASENAME = "heading_judge_cache"

# The thinking-directive tuple folded into the cache key — this pass is
# thinking-ON by design (genuine relational reasoning), effort high.
_THINKING_DIRECTIVE = ("detailed thinking on", "reasoning effort: high")


def resolve_heading_judge_enable_thinking() -> bool:
    """Whether the heading judge runs with model REASONING on. Default OFF.

    Heading-level assignment is a CLASSIFICATION task, and NVIDIA's reasoning-model
    guidance + a live TRT-LLM measurement (thinking-off: 24 tok / 1.4 s; thinking-on:
    101 tok / 7.9 s — identical answer) both say classification should run
    reasoning-OFF. The bool is emitted as ``chat_template_kwargs={"enable_thinking":
    <bool>}`` on the seat POST — the control the Nemotron-3 ``nano-v3`` parser
    actually honors (verified on the TRT-LLM seat; the vLLM-era "dead" note is
    stale). Truthy env (``1``/``true``/``yes``/``on``) opts thinking back ON."""
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def resolve_reasoning_effort() -> Tuple[str, str]:
    """``(thinking_line, effort_line)`` for the judge/review prompts.

    When reasoning is OFF (:func:`resolve_heading_judge_enable_thinking`, the
    DEFAULT for this classification task) the system directive is "detailed
    thinking off" with no effort line — consistent with the ``enable_thinking:
    false`` chat-template kwarg the POST carries. When reasoning is ON,
    ``SEMANTIK_HEADING_JUDGE_REASONING_EFFORT`` selects ``high`` (default) /
    ``medium`` / ``low`` / ``off``. Folded into the cache key so a change
    re-rolls affected windows."""
    if not resolve_heading_judge_enable_thinking():
        return ("detailed thinking off", "")
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_REASONING_EFFORT") or "").strip().lower()
    if raw == "off":
        return ("detailed thinking off", "")
    if raw in ("low", "medium", "high"):
        return ("detailed thinking on", f"reasoning effort: {raw}")
    return _THINKING_DIRECTIVE


#: Modest anti-repetition penalty applied ONLY on the thinking-OFF judge POST
#: (`SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY`). Heading-leveling JSON never
#: wants repetition, so a small positive frequency_penalty is pure-upside for
#: the classification task — belt to the thinking-off token cap's suspenders
#: against the degenerate-repetition runaway.
_FREQUENCY_PENALTY_THINKOFF_DEFAULT = 0.3


def resolve_heading_judge_frequency_penalty() -> float:
    """``frequency_penalty`` for the judge POST — anti-repetition guard.

    Thinking ON → ``0.0`` (the key is NEVER put on the wire, so the thinking-ON
    body is BYTE-IDENTICAL). Thinking OFF → the
    ``SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY`` float if set, else
    ``0.3``; a resolved ``0.0`` OMITS the key (the operator opt-out). The
    TRT-LLM OpenAI-compatible seat accepts ``frequency_penalty``. Parse-with-
    fallback: blank / non-float / NaN / ±Inf / garbage → the thinking-off
    default (an explicit finite value, incl. ``0.0`` and a negative, is
    honoured — frequency_penalty is valid in ``[-2, 2]``)."""
    if resolve_heading_judge_enable_thinking():
        return 0.0
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY") or "").strip()
    if not raw:
        return _FREQUENCY_PENALTY_THINKOFF_DEFAULT
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return _FREQUENCY_PENALTY_THINKOFF_DEFAULT
    if f != f or f in (float("inf"), float("-inf")):
        return _FREQUENCY_PENALTY_THINKOFF_DEFAULT
    return f

_JUDGE_INSTRUCTIONS = (
    "You are a document-structure judge. You are given the complete ordered "
    "heading skeleton of one textbook chapter. Lines marked \"*\" are PENDING "
    "headings whose level was defaulted; lines marked \".\" are FIXED anchors "
    "(the chapter title and the numbered N.M section spine) that are correct "
    "and immutable. Your task: assign the correct heading level (an integer "
    "2-6) to EVERY pending heading so the tree is hierarchy-consistent — peers "
    "share a level, a child is exactly one level deeper than its parent, and "
    "every pending heading hangs under the nearest preceding fixed anchor or "
    "corrected pending parent. Typical patterns: a chapter's learning-objective "
    "topic titles repeat three times (as objectives list context, as the "
    "section-body subsection, and again inside end-of-chapter review/exercises) "
    "— the section-body occurrence is a child of its N.M section (level 3); "
    "review restatements sit under the review apparatus (usually level 4); "
    "apparatus titles like \"Learning Objectives\", \"Practice Makes Perfect\", "
    "\"Chapter Outline\" are children of their enclosing section. Never invent, "
    "drop, reorder, or rewrite headings. Never assign level 1. Judge only the "
    "ids marked \"*\".\n"
    "Respond with ONLY a JSON object, no prose, in exactly this shape:\n"
    "{\"levels\": {\"<region_id>\": <level int>, ...}}\n"
    "with one entry per PENDING region id (the integer after \"R\", quoted as "
    "a string key)."
)


# ── Skeleton model. ─────────────────────────────────────────────────────────
@dataclass
class SkeletonEntry:
    region_index: int
    level: int
    text: str
    source_page: Any
    pending: bool
    anchor: Optional[str] = None


@dataclass
class SkeletonPlan:
    entries: List[SkeletonEntry]
    digest: str  # the primary single-window digest (report + cache-key stable)
    pending_ids: List[int]
    windows: List[Tuple[str, List[int]]]  # (digest, [pending region ids]) per POST
    #: NORMALIZED-window architecture ONLY (SEMANTIK_HEADING_JUDGE_NORMALIZE) —
    #: a parallel list aligned to ``windows`` mapping each window to its
    #: chapter/slice metadata ``{chapter_id, slice_idx, n_slices, split,
    #: core_pending_ids, overlap_pending_ids, heading_ids, W}``. ``None`` on every
    #: non-normalized path (so the plan is byte-identical when the gate is off);
    #: its presence is the reduce-driver's "run the chapter reconcile" signal.
    chapter_slices: Optional[List[Dict[str, Any]]] = None


@dataclass
class ApplyResult:
    #: verdicts RECORDED on a pending heading (a level was written + the
    #: pending flag cleared). NOT the number of headings whose level moved —
    #: see ``changed`` / ``agreed``.
    applied: int = 0
    clamped: int = 0
    dropped: int = 0
    kept: int = 0
    #: kept AND judged — the verdict named the heading but the clamp rules
    #: rejected it (unknown level / out-of-range / caption guard). Distinct
    #: from ``unjudged`` by contract: ``kept == kept_judged + unjudged``.
    kept_judged: int = 0
    #: kept because NO verdict ever covered the heading (fail-open window /
    #: truncation hole) — the accounting the audit found conflated into
    #: ``kept``.
    unjudged: int = 0
    #: applied verdicts whose effective level DIFFERS from the pre-existing
    #: level — the only subset that can move a rendered ``<hN>`` tag.
    #: Contract: ``applied == changed + agreed``.
    changed: int = 0
    #: applied verdicts whose effective level EQUALS the pre-existing level —
    #: the judge AGREED (or re-confirmed an already-judged layout). Recorded,
    #: render-invisible.
    agreed: int = 0
    #: level-transition histogram over the CHANGED verdicts only,
    #: ``{"3->2": 28}`` — the human-readable shape of what actually moved.
    transitions: Dict[str, int] = field(default_factory=dict)
    corrections: Dict[int, Tuple[int, int, bool]] = field(default_factory=dict)


class _JudgeTransportError(Exception):
    """A POST transport error. ``transient`` decides whether it is retried;
    ``timeout`` marks the read-timeout flavor (the wall-clock face of
    over-deliberation) — never blind-retried, it triggers the window SPLIT;
    ``aborted`` marks the seat-death flavor (the engine ABORTED an in-flight
    request — HTTP 200 with ``finish_reason='abort'`` and a truncated body)."""

    def __init__(self, message: str, *, transient: bool,
                 timeout: bool = False, aborted: bool = False) -> None:
        super().__init__(message)
        self.transient = transient
        self.timeout = timeout
        self.aborted = aborted


# ── Failure mechanism taxonomy. ─────────────────────────────────────────────
# A chapter that ends 100% UNJUDGED is a REAL failure, and the operator must be
# told WHY. Before this taxonomy every non-verdict collapsed into an anonymous
# "truncation hole" warning. A seat stopped mid-flight can return HTTP 200 with
# ``finish_reason='abort'`` and a partial body, which must not be classified
# like an over-deliberating model. These tokens ride the report meta so the
# pipeline seam can name the mechanism in its warning.
MECHANISM_SEAT_ABORTED = "seat_aborted"
MECHANISM_SEAT_UNREACHABLE = "seat_unreachable"
MECHANISM_LENGTH_EXHAUSTED = "length_exhausted"
MECHANISM_EMPTY_CONTENT = "empty_content"
MECHANISM_PARSE_FAILURE = "parse_failure"
MECHANISM_COVERAGE_GAP = "coverage_gap"

#: ``finish_reason`` values that mean the seat KILLED the request mid-generation
#: (vLLM emits ``abort`` on engine shutdown / client-side abort). The body is a
#: truncated fragment, never a verdict — treating it as a parse failure hides a
#: dead seat behind a model-quality-shaped warning.
_ABORT_FINISH_REASONS = frozenset(
    {"abort", "aborted", "cancel", "cancelled", "canceled"}
)

_MECHANISM_PHRASES = {
    MECHANISM_SEAT_ABORTED: (
        "the judge seat ABORTED the in-flight request "
        "(finish_reason=abort) — the seat container was stopped/killed or its "
        "engine shut down mid-generation"
    ),
    MECHANISM_SEAT_UNREACHABLE: (
        "the judge seat was UNREACHABLE (transport failure — seat down, "
        "connection refused, or a non-retryable HTTP error)"
    ),
    MECHANISM_LENGTH_EXHAUSTED: (
        "the judgment exhausted its completion budget (finish_reason=length) "
        "and the split ladder could not recover it"
    ),
    MECHANISM_EMPTY_CONTENT: (
        "the seat returned EMPTY content (possible mode collapse or a "
        "reasoning block that consumed the whole completion window)"
    ),
    MECHANISM_PARSE_FAILURE: (
        "the seat's reply did not parse as the required JSON verdict"
    ),
    MECHANISM_COVERAGE_GAP: (
        "the verdict parsed but OMITTED assigned pending ids (partial "
        "coverage) and the re-split rounds could not close the gap"
    ),
}


def describe_failure_modes(modes: Sequence[str]) -> str:
    """Human phrase for the observed mechanism tokens (never empty)."""
    seen = [m for m in dict.fromkeys(modes) if m in _MECHANISM_PHRASES]
    if not seen:
        return (
            "UNKNOWN — no window reported a failure mechanism; the verdict "
            "simply never named these ids"
        )
    return "; ".join(_MECHANISM_PHRASES[m] for m in seen)


def _add_mechanism(wmeta: Dict[str, Any], token: str) -> None:
    """Append a mechanism token to a window's meta (order-preserving set)."""
    modes = wmeta.setdefault("mechanisms", [])
    if token not in modes:
        modes.append(token)


# ── Real-tokenizer token counting (SEMANTIK_HEADING_JUDGE_TOKENIZER). ────────
# The char heuristics (chars//4 sizer, chars//3 guard) can undercount dense or
# mathematical content, making normalized windows exceed their target and crowd
# the seat limit. When enabled (the `auto` DEFAULT),
# `_count_tokens` uses the JUDGE MODEL's OWN tokenizer (loaded ONCE, OFFLINE);
# any load failure falls back to a CONSERVATIVE `ceil(len/3)` divisor (never the
# chars//4 legacy — the fallback must OVER-count for safety). `off` reverts to
# the byte-identical char heuristics.
_DEFAULT_TOKENIZER_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
_TOKENIZER_OFF = {"off", "0", "false", "no"}
#: Approximate per-message chat-template role/framing overhead (tokens) folded
#: into the prompt-token guard so the seat-context check stays conservative.
_PROMPT_ROLE_OVERHEAD_TOKENS = 4
#: Process-singleton tokenizer cache: effective-id -> tokenizer-or-None (None =
#: load failed, cached so we never retry). Reset via `_reset_tokenizer_cache`.
_TOKENIZER_CACHE: Dict[str, Any] = {}
_TOKENIZER_LOAD_LOGGED: set = set()


def resolve_tokenizer_mode() -> str:
    """Token-counting mode from ``SEMANTIK_HEADING_JUDGE_TOKENIZER``.

    ``auto`` (DEFAULT — unset / blank / ``auto`` / garbage-that-is-not-a-token):
    use the real model tokenizer, fall back to a conservative divisor on any
    load/encode failure. A falsey token (``off``/``0``/``false``/``no``): the
    legacy char heuristics (byte-identical). Any other value: an explicit HF id
    / local path used verbatim as the tokenizer to load."""
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_TOKENIZER") or "").strip()
    if not raw:
        return "auto"
    if raw.lower() in _TOKENIZER_OFF:
        return "off"
    if raw.lower() == "auto":
        return "auto"
    return raw  # explicit HF id / local path (case preserved)


def resolve_tokenizer_id() -> str:
    """Default tokenizer id (``SEMANTIK_HEADING_JUDGE_TOKENIZER_ID``); used when
    the mode is ``auto`` (an explicit-id mode supplies its own id)."""
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_TOKENIZER_ID") or "").strip()
    return raw or _DEFAULT_TOKENIZER_ID


def _effective_tokenizer_id(mode: str) -> str:
    return resolve_tokenizer_id() if mode == "auto" else mode


def _reset_tokenizer_cache() -> None:
    """Clear the process-singleton tokenizer cache (tests)."""
    _TOKENIZER_CACHE.clear()
    _TOKENIZER_LOAD_LOGGED.clear()


def _load_tokenizer(tokenizer_id: str):
    """Load the real HF tokenizer OFFLINE (``local_files_only=True`` — a miss
    NEVER fetches). Raises on any failure; the caller memoizes + falls back.
    Isolated so tests can monkeypatch it with a fake tokenizer."""
    from transformers import AutoTokenizer  # heavy import, lazy

    return AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)


def _get_tokenizer(tokenizer_id: str):
    """Process-singleton: load ``tokenizer_id`` ONCE (module-level cache),
    caching the tokenizer OR ``None`` on any failure so we never retry, and
    logging the fallback ONCE. Offline only — a miss falls back, never fetches."""
    if tokenizer_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[tokenizer_id]
    tok = None
    try:
        tok = _load_tokenizer(tokenizer_id)
    except Exception as exc:  # noqa: BLE001 — any load failure → conservative fallback
        tok = None
        if tokenizer_id not in _TOKENIZER_LOAD_LOGGED:
            _TOKENIZER_LOAD_LOGGED.add(tokenizer_id)
            logger.warning(
                "heading-judge: real tokenizer %r unavailable (%s: %s) — using "
                "the conservative ceil(len/3) token estimate for the rest of "
                "this process", tokenizer_id, type(exc).__name__, exc)
    _TOKENIZER_CACHE[tokenizer_id] = tok
    return tok


def _count_tokens(text: str) -> int:
    """Token count of ``text``.

    ``off`` mode → ``max(1, len//4)`` (the legacy ``_estimate_tokens`` heuristic,
    byte-identical). ``auto`` / explicit-id → the real tokenizer's
    ``encode(add_special_tokens=False)`` length when loadable, else the
    CONSERVATIVE ``ceil(len/3)`` fallback (never the chars//4 legacy — the
    fallback deliberately slightly OVER-counts for safety)."""
    text = text or ""
    mode = resolve_tokenizer_mode()
    if mode == "off":
        return max(1, len(text) // 4)
    tok = _get_tokenizer(_effective_tokenizer_id(mode))
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception as exc:  # noqa: BLE001 — a bad encode → conservative fallback
            logger.debug("heading-judge tokenizer.encode failed (fallback): %s", exc)
    import math

    return max(1, math.ceil(len(text) / 3.0))


# ── Skeleton construction. ──────────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    # Window SIZER — routed through the real-tokenizer counter. `off` mode is
    # byte-identical to the legacy `max(1, len//4)`.
    return _count_tokens(text)


_SENT_RE = re.compile(r"^(.+?[.!?])(\s|$)", re.S)


def _first_sentence(text: str, *, limit: Optional[int] = None) -> str:
    if limit is None:
        limit = resolve_anchor_truncate()
    t = " ".join((text or "").split())
    m = _SENT_RE.match(t)
    s = m.group(1) if m else t
    return s[:limit]


def _next_prose_sentence(prov: Sequence[Dict[str, Any]], pos: int) -> Optional[str]:
    """First sentence of the next plain-prose region after ``pos`` (≤80 chars)."""
    for r in prov[pos + 1:]:
        if r.get("region_kind") == "paragraph":
            body = str(r.get("raw_text") or "").strip()
            if body:
                return _first_sentence(body) or None
    return None


def _render_line(e: SkeletonEntry) -> str:
    mark = "*" if e.pending else "."
    page = e.source_page if e.source_page is not None else "?"
    return (f"R{e.region_index} p{page} h{e.level}{mark} "
            f"{e.text[:resolve_heading_text_truncate()]}")


def _render_digest(entries: Sequence[SkeletonEntry], *, anchors: bool) -> str:
    lines: List[str] = []
    for e in entries:
        lines.append(_render_line(e))
        if anchors and e.pending and e.anchor:
            lines.append(f"    > {e.anchor}")
    return "\n".join(lines)


def _chunk_by_pending_cap(
    entries: Sequence[SkeletonEntry], cap: int
) -> List[List[SkeletonEntry]]:
    """Slice the entry list into consecutive chunks of at most ``cap`` pending
    headings each (cut just before the pending that would exceed the cap) —
    the no-spine fallback partition so a spineless chapter's pendings still
    land in windows that can COMPLETE."""
    chunks: List[List[SkeletonEntry]] = []
    cur: List[SkeletonEntry] = []
    cur_pending = 0
    for e in entries:
        if e.pending and cur and cur_pending >= cap:
            chunks.append(cur)
            cur = []
            cur_pending = 0
        cur.append(e)
        if e.pending:
            cur_pending += 1
    if cur:
        chunks.append(cur)
    return chunks


def _split_windows(entries: Sequence[SkeletonEntry]) -> List[Tuple[str, List[int]]]:
    """Overflow ladder step 2: split at fixed level-2 N.M anchors into windows,
    each carrying the chapter's full FIXED-anchor outline (marks + 40-char
    texts) as immutable context."""
    cap = resolve_pending_window_cap()
    ctx_lines = [
        f"R{e.region_index} p{e.source_page} h{e.level}. {e.text[:resolve_context_text_truncate()]}"
        for e in entries if not e.pending
    ]
    ctx = "\n".join(ctx_lines)
    segments: List[List[SkeletonEntry]] = []
    cur: List[SkeletonEntry] = []
    for e in entries:
        if (not e.pending) and e.level == 2 and cur:
            segments.append(cur)
            cur = []
        cur.append(e)
    if cur:
        segments.append(cur)
    if len(segments) <= 1:
        n_pending = sum(1 for e in entries if e.pending)
        if n_pending <= cap:
            # No level-2 spine to split on — one no-anchor window suffices.
            d = _render_digest(entries, anchors=False)
            return [(d, [e.region_index for e in entries if e.pending])]
        # No spine but MORE pendings than a completable window holds — the
        # old fallback shipped them all in ONE window, which is exactly the
        # exhaustion shape the pending-count cap exists to prevent. Slice by
        # pending count instead; each chunk gets the fixed-anchor context.
        segments = _chunk_by_pending_cap(entries, cap)
    # Coalesce consecutive segments while the merged window stays under the
    # OUTPUT-side pending cap — a chapter with ~40 tiny N.M segments becomes a
    # few well-filled windows instead of ~40 POSTs. A single segment that alone
    # exceeds the cap ships as its own (oversized) window: the per-window
    # split/re-split ladder absorbs moderate overflow, and slicing inside a
    # section would cost the judge its local peer context.
    merged: List[List[SkeletonEntry]] = []
    cur_merged: List[SkeletonEntry] = []
    cur_pending = 0
    for seg in segments:
        seg_pending = sum(1 for e in seg if e.pending)
        if cur_merged and cur_pending + seg_pending > cap:
            merged.append(cur_merged)
            cur_merged = []
            cur_pending = 0
        cur_merged.extend(seg)
        cur_pending += seg_pending
    if cur_merged:
        merged.append(cur_merged)
    windows: List[Tuple[str, List[int]]] = []
    for seg in merged:
        body = _render_digest(seg, anchors=False)
        digest = (
            "FIXED-ANCHOR OUTLINE (immutable context — the chapter spine):\n"
            f"{ctx}\n\nWINDOW HEADINGS:\n{body}"
        )
        windows.append((digest, [e.region_index for e in seg if e.pending]))
    return windows


# ── Phase A: whole-document skeleton as per-window CONTEXT. ──────────────────
# When SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT is on, EVERY judging window's
# digest is PREFIXED with the FULL-document heading skeleton as read-only
# context, so each window's level calls stay consistent with the rest of the
# document (fixes the cross-window blindness that produces inconsistent
# recurring-section levels). The window still judges ONLY its own pending
# subset (unchanged output contract). Budget-guarded: a whole-document skeleton
# that would overflow ``resolve_digest_budget_tokens()`` DEGRADES gracefully to
# the per-window-only skeleton (logged once) rather than truncating. The prefix
# is part of the window digest, so it folds into ``_window_cache_key`` for free.
_FULLDOC_CONTEXT_HEADER = (
    "DOCUMENT STRUCTURE — context only, do NOT re-judge these; judge only the "
    "pending set below:"
)
#: Module-level once-logging guard for the budget-overflow degrade.
_FULLDOC_CONTEXT_LOGGED: set = set()

#: Truthy tokens for the opt-in Phase-A / Phase-B gates (default OFF).
_TRUTHY_TOKENS = {"1", "true", "yes", "on"}


def _truthy_env(name: str) -> bool:
    """Parse-with-fallback truthy resolver for a default-OFF gate (unset /
    blank / garbage / falsey → False)."""
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY_TOKENS


def resolve_fulldoc_context_mode() -> bool:
    """Phase A master gate: prefix every judging window with the whole-document
    heading skeleton as read-only context (``SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT``,
    default OFF, truthy-set parse-with-fallback). OFF → byte-identical (no
    prefix, today's exact per-window digest)."""
    return _truthy_env("SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT")


def resolve_fulldoc_anchors_mode() -> bool:
    """Phase A sub-flag: also include each heading's content anchor in the
    full-doc context block (``SEMANTIK_HEADING_JUDGE_FULLDOC_ANCHORS``, default
    OFF, truthy-set) — helps the model tell a real section from front-matter
    furniture. No-op unless ``resolve_fulldoc_context_mode()`` is on."""
    return _truthy_env("SEMANTIK_HEADING_JUDGE_FULLDOC_ANCHORS")


def _maybe_apply_fulldoc_context(plan: SkeletonPlan) -> SkeletonPlan:
    """Phase A: PREFIX every window digest with the whole-document skeleton as
    read-only context when the gate is on and the context fits the digest
    budget; else DEGRADE (logged once) to the per-window-only skeleton.

    The judge body is always kept delimited by ``_WINDOW_HEADINGS_MARKER`` so
    the split ladder parses only the window's own pending lines (the context in
    the preamble is never re-parsed). OFF → returns ``plan`` unchanged
    (byte-identical)."""
    if not resolve_fulldoc_context_mode() or not plan.windows:
        return plan
    context = _render_digest(plan.entries, anchors=resolve_fulldoc_anchors_mode())
    budget = resolve_digest_budget_tokens()
    if _estimate_tokens(context) > budget:
        if "overflow" not in _FULLDOC_CONTEXT_LOGGED:
            _FULLDOC_CONTEXT_LOGGED.add("overflow")
            logger.warning(
                "heading-judge: whole-document skeleton (~%d tokens) exceeds the "
                "digest budget (%d) — DEGRADING to per-window-only context "
                "(Phase-A full-doc context skipped; a small seat / huge doc)",
                _estimate_tokens(context), budget)
        return plan
    prefix = f"{_FULLDOC_CONTEXT_HEADER}\n{context}\n\n"
    new_windows: List[Tuple[str, List[int]]] = []
    for digest, pend in plan.windows:
        if _WINDOW_HEADINGS_MARKER in digest:
            # Multi-window: the existing WINDOW HEADINGS marker already delimits
            # the judge body; the context lands in the preamble.
            new_digest = prefix + digest
        else:
            # Single-window plain skeleton: wrap the judge body with the marker
            # so the split ladder finds it separate from the context prefix.
            new_digest = prefix.rstrip("\n") + _WINDOW_HEADINGS_MARKER + digest
        new_windows.append((new_digest, list(pend)))
    return SkeletonPlan(plan.entries, plan.digest, plan.pending_ids, new_windows)


def _reset_fulldoc_context_log() -> None:
    """Clear the Phase-A once-log guard (tests)."""
    _FULLDOC_CONTEXT_LOGGED.clear()


# ── Part 1-2-5: CHAPTER-MODE judging (SEMANTIK_HEADING_JUDGE_CHAPTER_MODE). ───
# Feeding the whole-document skeleton into EVERY window (Phase A / FULLDOC)
# ballooned the model's thinking 3-10x because it reasoned over the entire
# document per call. CHAPTER-MODE bounds each judge call to ONE chapter's
# heading skeleton + that chapter's FULL CONTENT TEXT, so thinking stays
# bounded WHILE the judge gets real content — which (a) tells furniture from a
# real section (a "Version 1.0.4" title with no section body is not a section)
# and (b) lets an un-numbered book (no N.M spine) be judged from content rather
# than a bare heading list. A per-CHAPTER window; the final whole-document
# REVIEWER (Part 3) then reconciles cross-chapter consistency. Default OFF →
# byte-identical (today's _split_windows path unchanged).
_CHAPTER_CONTENT_HEADER = (
    "CHAPTER CONTENT (for context — use it to judge the headings' levels; a "
    "heading with substantial content under it is a major section, a short one "
    "a subsection; a title with no real content is furniture):"
)
#: Default per-content-region word cap under the content budget-guard (keep the
#: text nearest each heading, drop mid-section bulk). Env-tunable.
_CHAPTER_CONTENT_HEAD_WORDS = 60
#: Hard token cap on the derived document-schema preamble (Part 5).
_DOC_SCHEMA_MAX_TOKENS = 400
#: Max recurring section-title signatures listed in the schema preamble.
_DOC_SCHEMA_MAX_RECURRING = 12

#: Heading-level → structural-role gloss for the derived schema (domain-agnostic
#: — describes the OBSERVED level scheme, never a publisher's vocabulary).
_LEVEL_ROLE = {
    1: "chapter", 2: "section", 3: "subsection", 4: "sub-subsection",
    5: "minor heading", 6: "minor heading",
}
#: N.M section number (NOT followed by a third .K number).
_SCHEMA_NM_RE = re.compile(r"^\s*\d{1,3}\.\s?\d{1,3}(?!\s?\.\s?\d)")
#: N.M.K section number (a third level of numbering).
_SCHEMA_NMK_RE = re.compile(r"^\s*\d{1,3}\.\s?\d{1,3}\.\s?\d{1,3}\b")

_DOC_SCHEMA_HEADER = (
    "This document uses the following structure — judge each heading's level "
    "consistently with it:"
)


def resolve_chapter_mode() -> bool:
    """Part 1-2 master gate: judge ONE chapter per call (skeleton + full content)
    instead of budget/anchor windows (``SEMANTIK_HEADING_JUDGE_CHAPTER_MODE``,
    default OFF, truthy-set parse-with-fallback). OFF → byte-identical (today's
    ``_split_windows`` path)."""
    return _truthy_env("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE")


def resolve_doc_schema_mode() -> bool:
    """Part 5 gate: prepend a compact derived document-schema preamble to every
    chapter-mode judge call AND the reviewer call
    (``SEMANTIK_HEADING_JUDGE_DOC_SCHEMA``). Default ON WHEN CHAPTER-MODE is on
    (it is load-bearing for per-chapter leveling CONSISTENCY when each stream
    sees only one chapter); an explicit falsey token omits the preamble
    (byte-identical to chapter-mode-without-schema). Entirely off (returns
    False) when CHAPTER-MODE and NORMALIZE are both off."""
    if not (resolve_chapter_mode() or resolve_normalize_mode()):
        return False
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def resolve_chapter_content_head_words() -> int:
    """Per-content-region word cap the chapter-mode budget-guard keeps when the
    content overflows (``SEMANTIK_HEADING_JUDGE_CHAPTER_CONTENT_WORDS``, default
    60). Parse-with-fallback: positive int wins; blank / garbage / non-positive
    → the default."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_CHAPTER_CONTENT_WORDS",
                                _CHAPTER_CONTENT_HEAD_WORDS)


def _build_all_entries(
    region_provenance: Sequence[Dict[str, Any]], *, anchors: bool = True,
) -> List[SkeletonEntry]:
    """The ordered HEADING skeleton entries (shared by the legacy + chapter-mode
    paths). ``anchors`` mints each pending node's next-prose content anchor (the
    legacy digest carries it; chapter-mode passes ``anchors=False`` because the
    full chapter content is already present)."""
    prov = list(region_provenance)
    entries: List[SkeletonEntry] = []
    for pos, r in enumerate(prov):
        if r.get("region_kind") != "heading":
            continue
        ridx = r.get("first_raw_block_index")
        if ridx is None:
            continue
        text = str(r.get("heading_text") or "")
        try:
            level = int(r.get("level", 3) or 3)
        except (TypeError, ValueError):
            level = 3
        page = r.get("source_page")
        if page is None:
            pages = r.get("pages") or []
            page = pages[0] if pages else None
        pending = bool(r.get("heading_level_pending"))
        anchor = _next_prose_sentence(prov, pos) if (pending and anchors) else None
        entries.append(SkeletonEntry(int(ridx), level, text, page, pending, anchor))
    return entries


# ── Part 1: chapter segmentation (reuses the transform's chapter-opener rule). ─
@dataclass
class Chapter:
    """One chapter: its ordered region slice + the boundary mode that formed it
    (``chapter_opener`` / ``top_level_heading`` / ``whole_document``)."""

    regions: List[Dict[str, Any]]
    mode: str


def _is_chapter_opener(region: Dict[str, Any]) -> bool:
    """A chapter-opener heading — REUSES the transform's chapter-opener rule
    (never a new detector): a level-1 chapter root (a synthesized / reconciled
    opener) OR a ``Chapter N[: short title]`` opener shape under the transform's
    own tail-compactness + apparatus + end-matter guards."""
    if region.get("region_kind") != "heading":
        return False
    try:
        level = int(region.get("level", 3) or 3)
    except (TypeError, ValueError):
        level = 3
    if level == 1 and not region.get("heading_level_pending"):
        return True
    text = str(region.get("heading_text") or "").strip()
    if not text or "\n" in text:
        return False
    from .transform import (
        _CHAPTER_APPARATUS_TAIL_RE,
        _CHAPTER_OPENER_HEAD_RE,
        _CHAPTER_TAIL_MAX_WORDS,
    )

    m = _CHAPTER_OPENER_HEAD_RE.match(text)
    if not m:
        return False
    tail = m.group(2).strip()
    if tail and len(tail.split()) > _CHAPTER_TAIL_MAX_WORDS:
        return False
    if tail and _CHAPTER_APPARATUS_TAIL_RE.search(tail):
        return False
    if rm.is_endmatter_apparatus(text):
        return False
    return True


def _cut_into_chapters(
    prov: List[Dict[str, Any]], boundaries: Sequence[int], mode: str,
) -> List[Chapter]:
    """Cut ``prov`` at the sorted boundary positions; a run before the first
    boundary forms a leading (front-matter) chapter."""
    bounds = sorted(set(int(b) for b in boundaries))
    if not bounds or bounds[0] != 0:
        bounds = [0] + bounds
    chapters: List[Chapter] = []
    for i, start in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else len(prov)
        chapters.append(Chapter(regions=prov[start:end], mode=mode))
    return chapters


def segment_into_chapters(
    region_provenance: Sequence[Dict[str, Any]],
) -> List[Chapter]:
    """Segment the document into chapters, with a fail-safe fallback ladder
    (NEVER crashes on an unstructured doc):

    1. chapter-opener boundaries (top-level ``Chapter N`` / level-1 openers);
    2. else top-level (lowest-level ``hN``) heading boundaries (≥2 of them);
    3. else the whole document as ONE chapter.

    The mode used is stamped on every returned :class:`Chapter` and logged."""
    prov = list(region_provenance)
    opener_pos = [i for i, r in enumerate(prov) if _is_chapter_opener(r)]
    if opener_pos:
        mode = "chapter_opener"
        chapters = _cut_into_chapters(prov, opener_pos, mode)
    else:
        heading_levels = []
        for i, r in enumerate(prov):
            if r.get("region_kind") == "heading":
                try:
                    lvl = int(r.get("level", 3) or 3)
                except (TypeError, ValueError):
                    lvl = 3
                heading_levels.append((i, lvl))
        if heading_levels:
            min_level = min(lvl for _, lvl in heading_levels)
            top_pos = [i for i, lvl in heading_levels if lvl == min_level]
        else:
            top_pos = []
        if len(top_pos) >= 2:
            mode = "top_level_heading"
            chapters = _cut_into_chapters(prov, top_pos, mode)
        else:
            mode = "whole_document"
            chapters = [Chapter(regions=prov, mode=mode)]
    logger.info(
        "heading-judge chapter-mode: segmented document into %d chapter(s) "
        "via '%s' boundaries", len(chapters), mode)
    return chapters


# ── Part 5: derived compact DOCUMENT-SCHEMA preamble. ────────────────────────
def build_document_schema(
    region_provenance: Sequence[Dict[str, Any]],
) -> str:
    """A COMPACT, DETERMINISTIC per-document schema instruction derived ONCE
    from the whole-document heading skeleton, prepended to EVERY chapter-mode
    judge call and the reviewer call so each per-chapter stream (which sees only
    ONE chapter) stays CONSISTENT with the rest of the document WITHOUT the
    thinking-ballooning whole-document context.

    Everything is DERIVED from the actual skeleton (never hardcoded / publisher
    vocabulary): the observed heading-level scheme, the numbering convention if
    present, the chapter-opener pattern, and the recurring section titles
    (via the audit's ``normalize_signature``). Hard-capped small
    (``_DOC_SCHEMA_MAX_TOKENS``). Deterministic (same skeleton → same string);
    stdlib only, cross-venv-clean. Returns ``""`` on a headingless document."""
    from collections import Counter

    headings = [r for r in region_provenance if r.get("region_kind") == "heading"]
    if not headings:
        return ""

    lines: List[str] = []
    # 1. observed heading-level scheme.
    present_levels: List[int] = []
    for h in headings:
        try:
            lvl = int(h.get("level", 3) or 3)
        except (TypeError, ValueError):
            lvl = 3
        if lvl not in present_levels:
            present_levels.append(lvl)
    present_levels.sort()
    scheme = ", ".join(
        f"h{lv}={_LEVEL_ROLE.get(lv, 'heading')}" for lv in present_levels)
    if scheme:
        lines.append(f"- Heading-level scheme: {scheme}.")

    # 2. numbering convention (N.M / N.M.K), if present.
    texts = [str(h.get("heading_text") or "") for h in headings]
    has_nm = any(_SCHEMA_NM_RE.match(t) for t in texts)
    has_nmk = any(_SCHEMA_NMK_RE.match(t) for t in texts)
    if has_nm or has_nmk:
        num = "Sections are numbered N.M (e.g. 2.3)"
        if has_nmk:
            num += ("; a third number N.M.K (e.g. 2.3.1) is ONE LEVEL DEEPER "
                    "than its N.M section")
        lines.append(f"- Numbering convention: {num}.")

    # 3. chapter-opener pattern.
    if any(_is_chapter_opener(h) for h in headings):
        lines.append("- Chapters open with a top-level 'Chapter N' heading "
                     "(level 1).")

    # 4. recurring section titles.
    counts = Counter(
        sig for sig in (normalize_signature(t) for t in texts) if sig)
    recurring = sorted(
        ((sig, c) for sig, c in counts.items() if c >= 2),
        key=lambda kv: (-kv[1], kv[0]))
    if recurring:
        sample = [sig for sig, _ in recurring[:_DOC_SCHEMA_MAX_RECURRING]]
        lines.append(
            "- Recurring section titles across the document (keep each "
            "occurrence's level consistent with its structural position): "
            + ", ".join(f"'{s}'" for s in sample) + ".")

    if not lines:
        return ""
    schema = _DOC_SCHEMA_HEADER + "\n" + "\n".join(lines)
    # Hard token cap (never overflow the per-call prompt budget with schema).
    max_chars = _DOC_SCHEMA_MAX_TOKENS * 4
    if len(schema) > max_chars:
        schema = schema[:max_chars].rstrip()
    return schema


# ── Part 2: chapter-content builder + budget-guard. ──────────────────────────
def _chapter_content_text(
    regions: Sequence[Dict[str, Any]], *, per_region_word_cap: Optional[int] = None,
) -> str:
    """The ordered content text of a chapter's NON-heading regions
    (``raw_text``). ``per_region_word_cap`` keeps only the first N words of each
    region (the text nearest each heading, dropping mid-section bulk) — the
    budget-guard truncation."""
    parts: List[str] = []
    for r in regions:
        if r.get("region_kind") == "heading":
            continue
        txt = str(r.get("raw_text") or "").strip()
        if not txt:
            continue
        if per_region_word_cap is not None:
            words = txt.split()
            if len(words) > per_region_word_cap:
                txt = " ".join(words[:per_region_word_cap]) + " …"
        parts.append(txt)
    return "\n".join(parts)


def _chapter_content_budgeted(
    regions: Sequence[Dict[str, Any]], overhead_tokens: int,
) -> str:
    """Chapter content that FITS ``resolve_digest_budget_tokens()`` alongside the
    schema + skeleton overhead. Full content when it fits; else keep the text
    nearest each heading (first-N-words per region); a final hard char cap
    guarantees the prompt NEVER overflows. Truncation is logged."""
    budget = resolve_digest_budget_tokens()
    content = _chapter_content_text(regions)
    if _estimate_tokens(content) + overhead_tokens <= budget:
        return content
    cap = resolve_chapter_content_head_words()
    truncated = _chapter_content_text(regions, per_region_word_cap=cap)
    max_content_tokens = max(0, budget - overhead_tokens)
    hard_capped = False
    if _estimate_tokens(truncated) > max_content_tokens:
        truncated = truncated[: max_content_tokens * 4].rstrip()
        hard_capped = True
    logger.info(
        "heading-judge chapter-mode: chapter content truncated to fit the "
        "digest budget (%d tok budget, %d tok overhead, per-region cap %d "
        "words%s)", budget, overhead_tokens, cap,
        ", hard char-capped" if hard_capped else "")
    return truncated


def _render_chapter_window(
    regions: Sequence[Dict[str, Any]], schema: str, *, budgeted: bool = True,
) -> Tuple[str, List[int]]:
    """Render ONE chapter/slice judge WINDOW — the derived document-schema
    preamble (Part 5), that region span's content (Part 2), and its heading
    skeleton — returning ``(win_digest, pending_ids)``. SHARED by chapter-mode
    (:func:`_build_chapter_mode_plan`) and the NORMALIZED-window slicer
    (:func:`_build_normalized_plan`) so the two paths cannot drift.

    ``budgeted`` (default ``True``, chapter-mode's behavior → byte-identical)
    renders content via :func:`_chapter_content_budgeted` (TRUNCATED to
    ``resolve_digest_budget_tokens()``). NORMALIZED mode passes ``budgeted=False``
    to render the span's FULL, un-truncated content (:func:`_chapter_content_text`):
    the sizer must see the TRUE chapter size (else an oversized chapter's budgeted
    render caps at ~digest_budget, the ``size <= W`` split trigger is trivially
    true, and it ships one budget-TRUNCATED window — the exact critical failure
    this architecture removes). Slices are ≤ W ≤ digest_budget BY CONSTRUCTION, so
    full content always fits a correctly-sized slice."""
    ch_entries = _build_all_entries(regions, anchors=False)
    ch_pending = [e.region_index for e in ch_entries if e.pending]
    skeleton = _render_digest(ch_entries, anchors=False)
    if budgeted:
        overhead = (_estimate_tokens(schema) + _estimate_tokens(skeleton)
                    + _estimate_tokens(_CHAPTER_CONTENT_HEADER) + 8)
        content = _chapter_content_budgeted(regions, overhead)
    else:
        content = _chapter_content_text(regions)
    preamble_parts: List[str] = []
    if schema:
        preamble_parts.append(schema)
    if content:
        preamble_parts.append(_CHAPTER_CONTENT_HEADER + "\n" + content)
    if preamble_parts:
        win_digest = ("\n\n".join(preamble_parts)
                      + _WINDOW_HEADINGS_MARKER + skeleton)
    else:
        win_digest = skeleton
    return win_digest, ch_pending


def _build_chapter_mode_plan(
    region_provenance: Sequence[Dict[str, Any]],
) -> SkeletonPlan:
    """Part 1-2-5: one judge WINDOW per chapter — each carrying the derived
    document-schema preamble (Part 5), that chapter's FULL content text
    (budget-guarded, Part 2), and that chapter's heading skeleton, judging ONLY
    that chapter's pendings. Chapters with no pending headings are a natural
    no-op (no window / no POST). The whole-document ``entries`` / ``digest`` are
    kept for the report + the sibling-staircase post-rule."""
    entries = _build_all_entries(region_provenance, anchors=False)
    pending_ids = [e.region_index for e in entries if e.pending]
    digest = _render_digest(entries, anchors=False)

    schema = build_document_schema(region_provenance) if resolve_doc_schema_mode() else ""
    chapters = segment_into_chapters(region_provenance)
    windows: List[Tuple[str, List[int]]] = []
    for ch in chapters:
        ch_entries = _build_all_entries(ch.regions, anchors=False)
        ch_pending = [e.region_index for e in ch_entries if e.pending]
        if not ch_pending:
            continue
        win_digest, _ = _render_chapter_window(ch.regions, schema)
        windows.append((win_digest, ch_pending))
    if not windows and pending_ids:
        # Defensive: every pending must be covered by a window (segmentation
        # partitions all regions, so this only fires on an empty chapter set).
        windows = [(digest, pending_ids)]
    return SkeletonPlan(entries, digest, pending_ids, windows)


# ── NORMALIZED-WINDOW architecture (SEMANTIK_HEADING_JUDGE_NORMALIZE). ───────
# Chapter-mode makes ONE window per chapter, so a single oversized chapter
# overflows the digest budget and TRUNCATES (finish_reason=length — a CRITICAL
# failure). The normalized architecture fixes the WORK-UNIT size to an adaptive
# per-book target W and splits any chapter whose one-window size exceeds W into
# contiguous OVERLAPPING slices each sized <= W <= digest budget, so truncation
# becomes structurally impossible (the split ladder survives only as a
# thinking-overrun backstop). Boundary headings are re-included in TWO adjacent
# slices (judged twice); a skeleton-only CHAPTER reviewer (or the deterministic
# interior-slice-wins fallback) reconciles the double-judgments (the REDUCE)
# before the existing whole-doc final reviewer runs. Default OFF → byte-identical
# (chapter_slices is None on every non-normalized path).
# Percentile 100 scopes normalization to its correctness job: split a chapter
# only when its
# one-window size genuinely EXCEEDS the seat digest budget (P=100 → W =
# min(digest_budget, max_chapter)). A lower P proactively BALANCES typical-outlier
# chapters — a wall-clock cost with no measured recall gain — and stays reachable
# via SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE.
_NORMALIZE_DEFAULT_PERCENTILE = 100
_NORMALIZE_DEFAULT_WINDOW_MIN = 4096
_SLICE_DEFAULT_OVERLAP = 2


def resolve_normalize_mode() -> bool:
    """Master gate for the normalized-window architecture
    (``SEMANTIK_HEADING_JUDGE_NORMALIZE``, default OFF, truthy-set
    parse-with-fallback). OFF → byte-identical (chapter_slices None, the legacy
    window plan)."""
    return _truthy_env("SEMANTIK_HEADING_JUDGE_NORMALIZE")


def resolve_normalize_percentile() -> int:
    """Percentile P for the adaptive W gauge
    (``SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE``, default **100**, clamped to
    ``[1, 100]``; garbage → 100). P=100 → ``W = min(digest_budget,
    max_chapter)`` → a chapter splits ONLY on genuine seat-budget overflow
    (correctness). A LOWER P opts into proactive balancing of typical-outlier
    chapters — a wall-clock cost with NO measured recall gain (the NORMALIZE A/B
    was structurally neutral on heading recall)."""
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE") or "").strip()
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return _NORMALIZE_DEFAULT_PERCENTILE
    return min(100, max(1, v))


def resolve_normalize_window_min() -> int:
    """Lower clamp W_MIN for the adaptive W
    (``SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW_MIN``, default 4096; positive int
    parse-with-fallback)."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW_MIN",
                                _NORMALIZE_DEFAULT_WINDOW_MIN)


def resolve_slice_overlap() -> int:
    """Headings a backed-up slice re-includes from the slice just closed
    (``SEMANTIK_HEADING_JUDGE_SLICE_OVERLAP``, default 2; ``0`` honoured = no
    overlap; garbage / negative → 2)."""
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_SLICE_OVERLAP") or "").strip()
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return _SLICE_DEFAULT_OVERLAP
    return v if v >= 0 else _SLICE_DEFAULT_OVERLAP


def resolve_chapter_review_mode() -> bool:
    """Skeleton-only CHAPTER reviewer gate
    (``SEMANTIK_HEADING_JUDGE_CHAPTER_REVIEW``). Default ON when NORMALIZE is on
    (a falsey token opts out → the deterministic interior-slice-wins fallback);
    entirely off (returns False) when NORMALIZE is off."""
    if not resolve_normalize_mode():
        return False
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_CHAPTER_REVIEW") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _nearest_rank_percentile(values: Sequence[int], pct: int) -> int:
    """Deterministic NEAREST-RANK percentile over an int list (matches
    ``seat_profile._percentile``): ``k = ceil(pct/100 * n)`` clamped to
    ``[1, n]`` → ``sorted[k-1]`` (no interpolation)."""
    import math

    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    if pct <= 0:
        return s[0]
    k = min(max(math.ceil(pct / 100.0 * n), 1), n)
    return s[k - 1]


def resolve_normalized_window_tokens(chapter_window_sizes: Sequence[int]) -> int:
    """The adaptive per-book work-unit token target W.

    ``SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW`` (positive int) pins W (the
    fixed-schema override). Else ``W = clamp(percentile(sizes, P), W_MIN,
    digest_budget)`` where P is :func:`resolve_normalize_percentile` (default
    **100**, NEAREST-RANK), W_MIN is :func:`resolve_normalize_window_min`
    (default 4096), and the UPPER clamp is :func:`resolve_digest_budget_tokens` —
    never target a window larger than the seat can hold, so the upper clamp
    always wins. At the default P=100 W = ``min(digest_budget, max_chapter)`` so a
    chapter splits ONLY when its one-window size exceeds the seat digest budget
    (genuine overflow = correctness); a lower P proactively balances
    typical-outlier chapters (a wall-clock cost, no measured recall gain).
    Empty / one-element input → digest_budget (degenerate: no splitting)."""
    fixed = _explicit_pos_int_env("SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW")
    if fixed is not None:
        return fixed
    digest_budget = resolve_digest_budget_tokens()
    sizes = [int(s) for s in chapter_window_sizes if s is not None]
    if len(sizes) <= 1:
        return digest_budget
    pct_val = _nearest_rank_percentile(sizes, resolve_normalize_percentile())
    lower_clamped = max(resolve_normalize_window_min(), pct_val)
    return min(digest_budget, lower_clamped)  # upper clamp always wins


def _overlap_start(
    regions: Sequence[Dict[str, Any]], pos: int, overlap: int,
) -> int:
    """The region index a backed-up slice starts at: the earliest of the LAST
    ``overlap`` HEADING regions before ``pos`` (so those headings + the content
    between them are re-included). No prior heading / overlap 0 / pos 0 → ``pos``
    (no backup)."""
    if overlap <= 0 or pos <= 0:
        return pos
    heading_pos = [i for i in range(pos)
                   if regions[i].get("region_kind") == "heading"]
    if not heading_pos:
        return pos
    return heading_pos[-overlap:][0]


def _pending_ids_in(
    regions: Sequence[Dict[str, Any]], lo: int, hi: int,
) -> List[int]:
    """Pending heading region ids in ``regions[lo:hi]`` (document order)."""
    out: List[int] = []
    for r in regions[lo:hi]:
        if r.get("region_kind") != "heading" or not r.get("heading_level_pending"):
            continue
        rid = r.get("first_raw_block_index")
        if rid is not None:
            out.append(int(rid))
    return out


def _heading_ids_in(
    regions: Sequence[Dict[str, Any]], lo: int, hi: int,
) -> List[int]:
    """ALL heading region ids in ``regions[lo:hi]`` (document order) — the
    edge-distance basis for the deterministic interior-slice-wins fallback."""
    out: List[int] = []
    for r in regions[lo:hi]:
        if r.get("region_kind") != "heading":
            continue
        rid = r.get("first_raw_block_index")
        if rid is not None:
            out.append(int(rid))
    return out


def _make_window_sizer(
    regions: Sequence[Dict[str, Any]], schema: str,
) -> Callable[[int, int], int]:
    """A memoized ``size(a, b)`` giving the token size of the chapter window over
    ``regions[a:b]`` (schema + slice content + slice skeleton).

    PERFORMANCE: with the real tokenizer, re-rendering + re-tokenizing the whole
    growing window at every step makes ``_slice_chapter_regions`` O(n^2) in
    tokenizer calls (many seconds on a ~250-region chapter). So each region's
    rendered token contribution is tokenized ONCE (O(n) encode calls total) and
    a candidate window is sized as the cached PREFIX-SUM of its regions' counts +
    a cached schema/skeleton overhead — additive and O(1) per query; the
    few-token inaccuracy at region-join boundaries is acceptable for sizing.

    ``off`` mode keeps the EXACT legacy whole-window measurement (cheap char
    heuristic, no tokenizer) so the normalized slice boundaries are
    byte-identical to the pre-tokenizer path."""
    if resolve_tokenizer_mode() == "off":
        def _exact(a: int, b: int) -> int:
            # Measure the UNBUDGETED (true) render against W — budget-truncation
            # must never be the sizer in normalized mode (a slice is <= W <=
            # digest_budget by construction, so its full content always fits).
            wd, _ = _render_chapter_window(list(regions[a:b]), schema, budgeted=False)
            return _estimate_tokens(wd)
        return _exact

    # Memoized per-region token contribution — reuse `_build_all_entries` +
    # `_render_line` (heading lines) and `_chapter_content_text`'s per-region
    # `raw_text` (content) so the sizing basis cannot drift from the real render.
    head_tok_by_ridx: Dict[int, int] = {
        e.region_index: _count_tokens(_render_line(e))
        for e in _build_all_entries(regions, anchors=False)
    }
    per_region: List[int] = []
    for r in regions:
        if r.get("region_kind") == "heading":
            ridx = r.get("first_raw_block_index")
            per_region.append(
                head_tok_by_ridx.get(int(ridx), 0) if ridx is not None else 0)
        else:
            txt = str(r.get("raw_text") or "").strip()
            per_region.append(_count_tokens(txt) if txt else 0)
    prefix: List[int] = [0]
    for t in per_region:
        prefix.append(prefix[-1] + t)
    # Cached overhead (schema + the content header + marker/join slack).
    overhead = ((_count_tokens(schema) if schema else 0)
                + _count_tokens(_CHAPTER_CONTENT_HEADER) + 8)

    def _summed(a: int, b: int) -> int:
        return overhead + (prefix[b] - prefix[a])

    return _summed


def _slice_chapter_regions(
    regions: Sequence[Dict[str, Any]], schema: str, W: int, overlap: int,
) -> List[Tuple[int, int, int]]:
    """Partition a chapter's regions into contiguous OVERLAPPING slices, each
    window (schema + slice content + slice skeleton) grown greedily to <= W.

    Returns ``(s_start, core_start, s_end)`` region-index ranges: ``[s_start,
    core_start)`` is the re-included OVERLAP span (the last ``overlap`` headings
    of the prior slice + their content), ``[core_start, s_end)`` is the slice
    CORE (never re-covered). The cores partition ``[0, n)`` so the union of slice
    cores is EVERY region (the coverage invariant). Each slice covers >= 1 NEW
    region (progress → termination); a lone region larger than W ships oversized
    (the per-window split ladder is the backstop).

    Window sizing goes through :func:`_make_window_sizer`, which memoizes
    per-region token counts so this loop is O(n) tokenizer calls, not O(n^2)."""
    n = len(regions)
    size = _make_window_sizer(regions, schema)
    slices: List[Tuple[int, int, int]] = []
    pos = 0
    while pos < n:
        s_start = _overlap_start(regions, pos, overlap)
        s_end = pos + 1  # core always holds >= 1 NEW region (progress + coverage)
        while s_end < n and size(s_start, s_end + 1) <= W:
            s_end += 1
        slices.append((s_start, pos, s_end))
        pos = s_end
    return slices


def _build_normalized_plan(
    region_provenance: Sequence[Dict[str, Any]],
) -> SkeletonPlan:
    """Normalized-window plan: gauge the adaptive W over each chapter's full
    one-window size, then emit ONE window per chapter that fits W, or a run of
    overlapping <= W slices for a chapter that does not. Every emitted window
    carries the SAME chapter-mode shape (schema + budget-guarded content +
    skeleton) via the shared :func:`_render_chapter_window`. ``chapter_slices``
    is aligned to ``windows`` for the reduce driver."""
    entries = _build_all_entries(region_provenance, anchors=False)
    pending_ids = [e.region_index for e in entries if e.pending]
    digest = _render_digest(entries, anchors=False)
    schema = build_document_schema(region_provenance) if resolve_doc_schema_mode() else ""
    chapters = segment_into_chapters(region_provenance)
    overlap = resolve_slice_overlap()

    # Measure each chapter's TRUE (unbudgeted) one-window size, then resolve the
    # adaptive W — a budgeted render caps at ~digest_budget and would hide the
    # oversized chapters this architecture exists to split.
    ch_full: List[Tuple[Chapter, str, List[int], int]] = []
    sizes: List[int] = []
    for ch in chapters:
        win_digest, ch_pending = _render_chapter_window(
            ch.regions, schema, budgeted=False)
        size = _estimate_tokens(win_digest)
        sizes.append(size)
        ch_full.append((ch, win_digest, ch_pending, size))
    W = resolve_normalized_window_tokens(sizes)

    windows: List[Tuple[str, List[int]]] = []
    chapter_slices: List[Dict[str, Any]] = []
    for ci, (ch, win_digest, ch_pending, size) in enumerate(ch_full):
        if not ch_pending:
            continue
        if size <= W:
            windows.append((win_digest, ch_pending))
            chapter_slices.append({
                "chapter_id": ci, "slice_idx": 0, "n_slices": 1, "split": False,
                "core_pending_ids": list(ch_pending), "overlap_pending_ids": [],
                "heading_ids": _heading_ids_in(ch.regions, 0, len(ch.regions)),
                "W": W,
            })
            continue
        # Oversized chapter → overlapping <= W slices; a slice with no pending is
        # dropped (nothing to POST) but never loses a pending (a dropped slice's
        # core carries no pending — the cores partition all regions).
        emitted: List[Dict[str, Any]] = []
        for (s_start, core_start, s_end) in _slice_chapter_regions(
                ch.regions, schema, W, overlap):
            core_pending = _pending_ids_in(ch.regions, core_start, s_end)
            overlap_pending = _pending_ids_in(ch.regions, s_start, core_start)
            if not core_pending:
                # No NEW pending to judge (a pure-overlap / trailing-content
                # slice) — its pendings are already judged by the slices that
                # OWN them as cores, so dropping it loses nothing (the cores
                # partition all pendings) and saves a redundant POST.
                continue
            sdigest, s_pending = _render_chapter_window(
                list(ch.regions[s_start:s_end]), schema, budgeted=False)
            emitted.append({
                "digest": sdigest, "pending": s_pending,
                "core_pending_ids": core_pending,
                "overlap_pending_ids": overlap_pending,
                "heading_ids": _heading_ids_in(ch.regions, s_start, s_end),
            })
        n_slices = len(emitted)
        for si, em in enumerate(emitted):
            windows.append((em["digest"], em["pending"]))
            chapter_slices.append({
                "chapter_id": ci, "slice_idx": si, "n_slices": n_slices,
                "split": n_slices > 1,
                "core_pending_ids": list(em["core_pending_ids"]),
                "overlap_pending_ids": list(em["overlap_pending_ids"]),
                "heading_ids": em["heading_ids"], "W": W,
            })
        covered: set = set()
        for em in emitted:
            covered.update(em["core_pending_ids"])
        assert covered == set(ch_pending), (
            "normalized-window coverage invariant violated: chapter %d slice "
            "cores %r != chapter pendings %r"
            % (ci, sorted(covered), sorted(ch_pending)))

    if not windows and pending_ids:
        # Defensive: every pending must be covered by a window (segmentation
        # partitions all regions, so this only fires on an empty chapter set).
        windows = [(digest, list(pending_ids))]
        chapter_slices.append({
            "chapter_id": 0, "slice_idx": 0, "n_slices": 1, "split": False,
            "core_pending_ids": list(pending_ids), "overlap_pending_ids": [],
            "heading_ids": [e.region_index for e in entries], "W": W})
    return SkeletonPlan(entries, digest, pending_ids, windows,
                        chapter_slices=chapter_slices)


def build_heading_skeleton(
    region_provenance: Sequence[Dict[str, Any]],
) -> SkeletonPlan:
    """Build the ordered heading skeleton + compact digest + window plan.

    HEADING regions only, in document order. One line per heading; pending nodes
    optionally carry a content anchor (first sentence of the next prose region).

    When ``SEMANTIK_HEADING_JUDGE_NORMALIZE`` is on the plan is the
    NORMALIZED-window plan (adaptive W, oversized chapters split into overlapping
    <= W slices + a ``chapter_slices`` sidecar for the reduce). Else when
    ``SEMANTIK_HEADING_JUDGE_CHAPTER_MODE`` is on (Part 1-2) the plan is ONE
    window per chapter (skeleton + full content + the derived schema preamble),
    bounding the model's thinking to one chapter per call. Else, when
    ``SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT`` is on (Phase A) every window
    digest is prefixed with the whole-document skeleton as read-only context
    (budget-guarded); ALL OFF → byte-identical.
    """
    if resolve_normalize_mode():
        return _build_normalized_plan(region_provenance)
    if resolve_chapter_mode():
        return _build_chapter_mode_plan(region_provenance)
    prov = list(region_provenance)
    entries = _build_all_entries(prov, anchors=True)

    pending_ids = [e.region_index for e in entries if e.pending]

    digest_with = _render_digest(entries, anchors=True)
    digest_budget = resolve_digest_budget_tokens()
    fits_pending = len(pending_ids) <= resolve_pending_window_cap()
    if fits_pending and _estimate_tokens(digest_with) <= digest_budget:
        return _maybe_apply_fulldoc_context(
            SkeletonPlan(entries, digest_with, pending_ids,
                         [(digest_with, pending_ids)]))
    # ladder 1: drop content anchors.
    digest_without = _render_digest(entries, anchors=False)
    if fits_pending and _estimate_tokens(digest_without) <= digest_budget:
        return _maybe_apply_fulldoc_context(
            SkeletonPlan(entries, digest_without, pending_ids,
                         [(digest_without, pending_ids)]))
    # ladder 2: split into N.M-anchored windows (input OR output overflow).
    windows = _split_windows(entries)
    return _maybe_apply_fulldoc_context(
        SkeletonPlan(entries, digest_without, pending_ids, windows))


# ── Prompt + POST. ──────────────────────────────────────────────────────────
def build_judge_messages(
    digest: str, n_headings: int, n_pending: int
) -> List[Dict[str, str]]:
    thinking_line, effort_line = resolve_reasoning_effort()
    system = thinking_line + "\n" + _JUDGE_INSTRUCTIONS
    user = (
        (effort_line + "\n\n" if effort_line else "")
        + f"Chapter heading skeleton ({n_headings} headings, {n_pending} pending):\n\n"
        f"{digest}\n\n"
        "Return the JSON now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _max_tokens_for(n_pending: int) -> int:
    # Per-pending increment resolved at CALL time via
    # SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT (default _MAX_TOKENS_PER_PENDING ==
    # _EST_TOKENS_PER_JUDGMENT == 300 → byte-identical). Tracks the SAME estimate
    # the window-cap divisor uses so a packed window is never under-budgeted.
    floor = resolve_max_tokens_floor()
    return max(
        floor,
        min(resolve_max_tokens_ceiling(),
            floor + resolve_heading_judge_est_per_judgment() * n_pending),
    )


@dataclass
class HeadingJudgeSeat:
    base_url: str
    api_key: Optional[str]
    model: str
    timeout: float


def resolve_heading_judge_seat() -> HeadingJudgeSeat:
    api_key = os.environ.get("SEMANTIK_HEADING_JUDGE_API_KEY")
    api_key = api_key.strip() if api_key and api_key.strip() else None
    return HeadingJudgeSeat(
        base_url=resolve_heading_judge_base_url(),
        api_key=api_key,
        model=resolve_heading_judge_model(),
        timeout=resolve_heading_judge_timeout(),
    )


def _post_judge_completion(
    *,
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: Sequence[Dict[str, str]],
    max_tokens: int,
    timeout: float,
    requests_module,
) -> Tuple[Optional[str], Optional[str]]:
    """One thinking-ON judge POST → ``(content, finish_reason)``.

    Thinking is controlled by ``chat_template_kwargs={"enable_thinking": <bool>}``
    (:func:`resolve_heading_judge_enable_thinking`, default OFF — this is a
    classification task; verified on the TRT-LLM ``nano-v3`` seat). The system
    directive is kept consistent with it. The seat routes any reasoning to a
    separate ``reasoning_content`` channel; the ANSWER is
    ``choices[0].message.content``, which is what we parse.
    """
    from ..vlm_extract import _chat_completions_url

    body: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": resolve_heading_judge_enable_thinking()
        },
    }
    # Anti-repetition guard — thinking-OFF only (resolver returns 0.0 when
    # thinking is ON, so the key is omitted and the thinking-ON body stays
    # byte-identical). A resolved 0.0 (either mode) never reaches the wire.
    _freq_pen = resolve_heading_judge_frequency_penalty()
    if _freq_pen != 0.0:
        body["frequency_penalty"] = _freq_pen
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = _chat_completions_url(base_url)
    import time as _time

    _t0 = _time.monotonic()
    try:
        resp = requests_module.post(url, json=body, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — timeout / conn → transient
        is_timeout = "timeout" in type(exc).__name__.lower()
        raise _JudgeTransportError(
            f"heading-judge request failed ({type(exc).__name__}): {exc}",
            transient=not is_timeout,
            timeout=is_timeout,
        ) from exc
    _duration_ms = (_time.monotonic() - _t0) * 1000.0
    status = int(getattr(resp, "status_code", 200))
    if status != 200:
        transient = status >= 500 or status == 429
        head = (getattr(resp, "text", "") or "")[:300]
        raise _JudgeTransportError(
            f"heading-judge endpoint HTTP {status} (model={model}): {head!r}",
            transient=transient,
        )
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise _JudgeTransportError(
            "heading-judge endpoint returned a non-JSON body", transient=False
        ) from exc
    try:
        choice = data["choices"][0]
        content = choice["message"].get("content")
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        keys = list(data) if isinstance(data, dict) else type(data)
        raise _JudgeTransportError(
            f"heading-judge malformed response (no choices[0].message): keys={keys}",
            transient=False,
        ) from exc
    _emit_usage_row(data.get("usage"), model=model, duration_ms=_duration_ms,
                    finish_reason=finish)
    # A seat may answer an aborted request with HTTP 200,
    # ``finish_reason='abort'``, and whatever
    # partial tokens it had emitted. That body can never parse as a verdict, so
    # letting it fall through to ``parse_judge_response`` mislabels a dead seat
    # as a model-quality parse failure AND skips the retry ladder entirely.
    # Raise as a TRANSIENT transport error instead: the bounded ladder gets its
    # (previously nonexistent) retries, and the mechanism reaches the report.
    if finish is not None and str(finish).strip().lower() in _ABORT_FINISH_REASONS:
        raise _JudgeTransportError(
            f"heading-judge request ABORTED by the seat mid-generation "
            f"(finish_reason={finish!r}, model={model}, "
            f"{len(content or '')} chars of partial content after "
            f"{_duration_ms / 1000.0:.1f}s) — the seat was stopped/killed or "
            f"its engine shut down while the judgment was in flight",
            transient=True,
            aborted=True,
        )
    return (None if content is None else str(content)), finish


def _emit_usage_row(
    usage: Any,
    *,
    model: str,
    duration_ms: float,
    finish_reason: Optional[str],
) -> None:
    """Best-effort per-POST metering row (the OP2 ``llm_usage.jsonl`` shape).

    Cross-venv clean (zero Ed4All imports — the ``stop_seam`` twin posture):
    the target path is injected via ``SEMANTIK_LLM_USAGE_PATH`` (an absolute
    path to the run's ``llm_usage.jsonl``); ``SEMANTIK_LLM_USAGE_PHASE``
    (default ``heading_judge``) stamps the additive ``phase`` field so the
    stat-matrix reporter can attribute rows that fall between pipeline
    checkpoints. When the explicit path is UNSET (the IN-LANE step-3b judge
    inside ``run_glmocr_lane`` — only the separate pipeline ``heading_judge``
    phase injects the path), the tap falls back to the SemantiK-local meter's
    run-ledger resolution (``llm_usage_meter._run_ledger_path``: ``ED4ALL_RUN_ID``
    → ``<state-runs>/<run_id>/llm_usage.jsonl``) so in-conversion judge POSTs
    still meter into the same ledger. Neither resolvable → no-op
    (byte-identical). ANY failure is swallowed — metering must never perturb a
    judge call. Cache HITs never reach this function, so rows count only real
    seat POSTs.
    """
    path = os.environ.get("SEMANTIK_LLM_USAGE_PATH", "").strip()
    if not path:
        try:
            from .. import llm_usage_meter

            run_path = llm_usage_meter._run_ledger_path()
        except Exception:  # noqa: BLE001 — metering must never break judging
            return
        if run_path is None:
            return
        path = str(run_path)
    try:
        from datetime import datetime, timezone

        row: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": "semantik-heading-judge",
            "model": model,
            "phase": os.environ.get(
                "SEMANTIK_LLM_USAGE_PHASE", "heading_judge").strip()
            or "heading_judge",
            "prompt_tokens": int((usage or {}).get("prompt_tokens", 0) or 0),
            "completion_tokens": int(
                (usage or {}).get("completion_tokens", 0) or 0),
            "duration_ms": round(float(duration_ms), 3),
        }
        if finish_reason is not None:
            row["finish_reason"] = str(finish_reason)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001 — metering must never break judging
        logger.debug("heading-judge usage tap failed (ignoring): %s", exc)


def _make_default_post_fn(
    seat: HeadingJudgeSeat, *, requests_module=None
) -> Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]]:
    """A ``post_fn(messages, max_tokens) -> (content, finish)`` wrapping the
    real seat with the bounded transient-retry ladder."""
    if requests_module is None:
        from ..vlm_extract import _lazy_requests

        requests_module = _lazy_requests()

    def _fn(
        messages: Sequence[Dict[str, str]], max_tokens: int
    ) -> Tuple[Optional[str], Optional[str]]:
        import time as _time

        attempts = _TRANSIENT_RETRIES + 1
        for i in range(attempts):
            try:
                return _post_judge_completion(
                    base_url=seat.base_url,
                    api_key=seat.api_key,
                    model=seat.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=seat.timeout,
                    requests_module=requests_module,
                )
            except _JudgeTransportError as exc:
                # A read-TIMEOUT is over-deliberation, not flakiness — blind
                # re-POSTing the identical request burns another full timeout
                # window (up to 20 min each). Propagate immediately so the
                # caller's split ladder handles it.
                if exc.timeout or not exc.transient or i == attempts - 1:
                    raise
                # LOUD bounded retry: an ABORT means the seat died under us, so
                # say so (an anonymous silent retry is what let the incident
                # look like a model failure) and back off a little longer — a
                # seat mid-`docker stop` needs more than 0.5s to be honestly
                # re-probed. Still bounded by _TRANSIENT_RETRIES (never a stall).
                if exc.aborted:
                    logger.warning(
                        "heading-judge: %s — retrying (attempt %d/%d)",
                        exc, i + 2, attempts)
                    _time.sleep(1.5 * (i + 1))
                else:
                    _time.sleep(0.5 * (i + 1))
        raise _JudgeTransportError("heading-judge retry ladder exhausted", transient=True)

    return _fn


# ── Response parse. ─────────────────────────────────────────────────────────
def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # drop opening fence line and trailing fence
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_balanced_json(s: str) -> Optional[str]:
    """Return the first balanced ``{...}`` object substring (string-aware)."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_judge_response(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Strict parse → ``{"levels": {...}}`` or ``None`` on any failure."""
    if raw is None:
        return None
    obj = _extract_balanced_json(_strip_fences(raw))
    if obj is None:
        return None
    try:
        data = json.loads(obj)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    levels = data.get("levels")
    if not isinstance(levels, dict):
        return None
    return {"levels": levels}


# ── Sibling-staircase post-rule (deterministic, Fix 1). ──────────────────────
# The judge's only body-content error class on the whole-book audit (3/59
# applied) was nesting a run of same-class SIBLING headings progressively
# deeper (e.g. three parallel dashboard names judged 3 → 4 → 5; a two-member
# pair judged 3 → 4). Deterministic post-pass over the PROPOSALS (before the
# clamp walk): consecutive pending headings with PARALLEL SHAPE — short
# noun-phrase, similar token count, no ordinal progression, same parent
# section span (no intervening fixed anchor) — that the judge placed at
# STRICTLY INCREASING depths are flattened to the level of the FIRST.
# Domain-agnostic SHAPE heuristics only (no vocabulary). Anti-overreach: a
# run carrying a structural nesting signal (N.M numbers, an ordinal
# progression) is NEVER flattened — those numbers are the judge's legitimate
# evidence for hierarchy.

_STAIR_MAX_WORDS = 8          # a sibling label is a short noun phrase
_STAIR_MAX_WORD_DELTA = 2     # "similar token count" band vs the chain head
_NM_NUMBER_RE = re.compile(r"\b\d+\.\d+\b")
_LEADING_ORDINAL_RE = re.compile(r"^\s*\(?\d{1,4}[.):]\s+")
_FIRST_INT_RE = re.compile(r"\d+")


def _staircase_shape_words(text: str) -> Optional[int]:
    """Word count when the heading is a short noun-phrase LABEL shape
    (a parallel-sibling candidate), else ``None``. Shape only: bounded word
    count, no sentence-ending punctuation — never a vocabulary test."""
    t = " ".join((text or "").split())
    if not t:
        return None
    words = t.split()
    if len(words) > _STAIR_MAX_WORDS:
        return None
    if t[-1] in ".!?:;":
        return None
    return len(words)


def _has_structural_number(text: str) -> bool:
    """An N.M section number or a leading ordinal is a STRUCTURAL signal that
    can legitimately support nesting — such a heading never flattens."""
    t = text or ""
    return bool(_NM_NUMBER_RE.search(t) or _LEADING_ORDINAL_RE.match(t))


def _ordinal_progression(texts: Sequence[str]) -> bool:
    """True when EVERY text carries a number and the first numbers form a
    strictly monotonic sequence (e.g. "... 1", "... 2", "... 3") — an ordinal
    progression that may genuinely encode ordered sub-structure."""
    nums: List[int] = []
    for t in texts:
        m = _FIRST_INT_RE.search(t or "")
        if m is None:
            return False
        nums.append(int(m.group(0)))
    if len(nums) < 2:
        return False
    increasing = all(b > a for a, b in zip(nums, nums[1:]))
    decreasing = all(b < a for a, b in zip(nums, nums[1:]))
    return increasing or decreasing


def flatten_sibling_staircases(
    entries: Sequence[SkeletonEntry],
    verdict_map: Dict[int, int],
) -> Dict[int, Tuple[int, int]]:
    """Flatten strict sibling-staircase proposals IN PLACE on ``verdict_map``.

    Returns the audit map ``{region_id: (proposed, flattened_to)}`` for the
    entries that were changed. Only flattens when ≥2 CONSECUTIVE judged
    pending headings of parallel shape form a strict staircase (each proposal
    strictly deeper than the previous) with no structural signal supporting
    the nesting. Runs break at any non-pending (fixed) heading or any pending
    without a verdict, so a chain never crosses a parent-section boundary.
    """
    audit: Dict[int, Tuple[int, int]] = {}

    def _verdict(e: SkeletonEntry) -> Optional[int]:
        v = verdict_map.get(e.region_index)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # Maximal runs of consecutive judged-pending headings.
    runs: List[List[SkeletonEntry]] = []
    cur: List[SkeletonEntry] = []
    for e in entries:
        if e.pending and _verdict(e) is not None:
            cur.append(e)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)

    for run in runs:
        i = 0
        while i < len(run) - 1:
            head_words = _staircase_shape_words(run[i].text)
            if head_words is None:
                i += 1
                continue
            # Extend the chain while shape stays parallel to the HEAD and the
            # proposals climb strictly deeper.
            chain = [run[i]]
            j = i + 1
            while j < len(run):
                w = _staircase_shape_words(run[j].text)
                if (w is None
                        or abs(w - head_words) > _STAIR_MAX_WORD_DELTA
                        or _verdict(run[j]) <= _verdict(chain[-1])):
                    break
                chain.append(run[j])
                j += 1
            if len(chain) >= 2:
                texts = [c.text for c in chain]
                if (not any(_has_structural_number(t) for t in texts)
                        and not _ordinal_progression(texts)):
                    base = _verdict(chain[0])
                    for c in chain[1:]:
                        proposed = _verdict(c)
                        if proposed != base:
                            audit[c.region_index] = (proposed, base)
                            verdict_map[c.region_index] = base
                i = j
            else:
                i += 1
    return audit


# ── Cache (content-addressed sidecar; mirrors reasoning_qc_cache). ───────────
def _judge_cache_root() -> Path:
    # NB: ``paths`` is the PARENT package's module (semantik_structure.paths);
    # a same-package ``from . import paths`` raises ImportError (no
    # glmocr/paths.py), which the swallow-all cache guards turned into a
    # silent never-caches bug — pinned by
    # test_cache_root_resolves_without_monkeypatch.
    from .. import paths as _semantik_paths

    return _semantik_paths.resolve_cache(_CACHE_BASENAME)


def _window_cache_key(digest: str, model: str, max_tokens: int) -> str:
    raw = (
        f"{hashlib.sha256(digest.encode('utf-8')).hexdigest()}|{model}|"
        f"{JUDGE_PROMPT_VERSION}|{max_tokens}|0.0|{'/'.join(resolve_reasoning_effort())}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str, root: Path) -> Path:
    return root / key[:2] / f"{key}.json"


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    try:
        path = _cache_path(key, _judge_cache_root())
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception as exc:  # noqa: BLE001 — a corrupt sidecar → miss
        logger.debug("heading-judge cache read failed (non-fatal): %s", exc)
        return None


def _cache_put(key: str, verdict: Dict[str, Any]) -> None:
    try:
        path = _cache_path(key, _judge_cache_root())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(verdict, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 — a cache write must never break judging
        logger.debug("heading-judge cache write failed (non-fatal): %s", exc)


# ── Judge orchestration. ─────────────────────────────────────────────────────
# Serving-context budget for the doubled-max_tokens retry: prompt estimate +
# completion cap must stay under the seat's 32,768-token window or vLLM 400s
# ("maximum context length exceeded") — which the transport layer correctly
# classifies non-transient, silently fail-opening the window (the live A2
# defect: thinking exhausts the first POST, the doubled retry overflows, the
# window's ~93 pendings stay unjudged). Conservative chars//3 estimate;
# 31500 = the seat's 32,768 window minus a ~1.2k safety margin, so a
# small-prompt window still gets its doubled retry while a big-prompt one
# falls straight to the split. Both the budget and the completion ceiling are
# env-tunable so a seat re-pinned to a longer --max-model-len (the model's
# native max_position_embeddings is 262,144) lifts them WITHOUT a code change:
# SEMANTIK_HEADING_JUDGE_CTX_BUDGET / SEMANTIK_HEADING_JUDGE_MAX_TOKENS
# (parse-with-fallback to the 32k-seat defaults below).
_CTX_TOKENS_BUDGET = 31500

# SPLIT-ON-TRUNCATION ladder depth bound (env-tunable via
# SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH; mirrors reasoning_qc's
# resolve_reasoning_qc_max_split_depth). Default raised 2 -> 3: a window whose
# completion need exceeds even the max_tokens CEILING can ONLY be fixed by
# splitting it smaller (more tokens never helps — the model free-runs to
# whatever cap it is granted), so the ladder must be allowed to recurse deep
# enough to reach a fitting sub-window. Each split HALVES the pending set, so
# the budget per sub-window drops (fewer pendings -> smaller _max_tokens_for),
# which is exactly what lets a half FIT where the parent truncated (the live
# audit shape: a 13-pending window budgeted 20480 + 300*13 = 24380 deliberated
# PAST 24380 and hit finish=length; split into two ~6-7-pending halves each
# budgets ~22400 for a ~12200 need -> both fit). ``0`` is HONOURED as the
# explicit revert lever — no split ever, straight to the legacy
# doubled-max_tokens-retry-then-fail-open path.
_DEFAULT_MAX_SPLIT_DEPTH = 3
# Min pendings a window must hold to be splittable at all: a single-pending
# window cannot be halved, so it drops to the doubled-retry last resort. 2 is
# the smallest floor that keeps the ladder meaningful (a 2-pending window
# halves into two 1-pending sub-windows).
_MIN_PENDING_PER_SPLIT = 2


def _resolve_pos_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        v = int(raw)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _explicit_pos_int_env(name: str) -> Optional[int]:
    """The explicitly-set positive int at ``name``, else ``None`` (unset /
    blank / garbage / non-positive → ``None`` so the caller falls through to the
    seat-derived / legacy value)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


# ── Seat-context probe + adaptive budget derivation. ────────────────────────
def _reset_seat_context_cache() -> None:
    """Clear the per-base_url seat-context probe cache (tests)."""
    _SEAT_CONTEXT_QUERY_CACHE.clear()
    _SEAT_CONTEXT_LOGGED.clear()


def _query_seat_max_model_len(
    base_url: str, *, timeout: float = _SEAT_CONTEXT_PROBE_TIMEOUT,
) -> Optional[int]:
    """GET ``{base_url}/models`` ONCE and return the first entry's positive
    ``max_model_len``, else ``None``. Cached per base_url (an absent seat costs
    at most one probe). Cross-venv clean: stdlib ``urllib`` only (never the
    Ed4All client — the stop_seam/heading_judge twin posture). Fail-SOFT: any
    error / timeout / missing field → ``None`` (the legacy hardcoded path)."""
    if base_url in _SEAT_CONTEXT_QUERY_CACHE:
        return _SEAT_CONTEXT_QUERY_CACHE[base_url]
    result: Optional[int] = None
    try:
        import urllib.request

        url = base_url.rstrip("/") + "/models"
        headers = {"Accept": "application/json"}
        api_key = os.environ.get("SEMANTIK_HEADING_JUDGE_API_KEY")
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        for entry in (data.get("data") if isinstance(data, dict) else None) or []:
            if not isinstance(entry, dict):
                continue
            mml = entry.get("max_model_len")
            if isinstance(mml, int) and not isinstance(mml, bool) and mml > 0:
                result = mml
                break
    except Exception as exc:  # noqa: BLE001 — the probe must never break judging
        logger.debug("heading-judge seat-context probe failed (fail-soft to "
                     "legacy budgets): %s", exc)
        result = None
    _SEAT_CONTEXT_QUERY_CACHE[base_url] = result
    return result


def resolve_heading_judge_seat_context() -> Optional[int]:
    """Resolved judge-seat context window, or ``None`` for the legacy path.

    ``SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT`` (default ``auto``): ``auto`` → probe
    the judge seat's ``/models`` ONCE (cached) and read ``max_model_len``; a
    positive int → that value verbatim (no probe); ``off``/``0``/``false``/``no``
    → ``None`` (the byte-identical revert to the hardcoded budgets). A failed /
    timed-out / field-less ``auto`` probe FAILS SOFT to ``None`` (logged once).
    Garbage → treated as ``auto`` (parse-with-fallback to the default)."""
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT") or "").strip()
    low = raw.lower()
    if low in _SEAT_CONTEXT_OFF:
        return None
    if raw and low != "auto":
        try:
            ci = int(raw)
            if ci > 0:
                return ci
        except (TypeError, ValueError):
            pass
        # garbage → fall through to the auto probe (the default posture)
    ctx = _query_seat_max_model_len(resolve_heading_judge_base_url())
    if ctx and ctx > 0:
        if ctx not in _SEAT_CONTEXT_LOGGED:
            _SEAT_CONTEXT_LOGGED.add(ctx)
            logger.info("heading-judge: adapting budgets to seat context "
                        "max_model_len=%d", ctx)
        return ctx
    return None


def _derive_budgets(seat_context: int) -> Tuple[int, int, int]:
    """``(digest_budget, ceiling, ctx_budget)`` sized to fit ``seat_context``.

    ``usable = seat_context - margin``; the completion CEILING gets
    ``completion_fraction`` of usable, the digest/prompt budget gets the rest,
    and ctx_budget (the doubled-retry prompt+completion guard) is the whole
    usable window. FLOORS keep a tiny / misread seat from degenerating a budget
    below the working defaults. Invariant (any non-pathological seat):
    ``digest_budget + ceiling == usable(after flooring) <= usable <= seat_context``."""
    margin = resolve_ctx_margin()
    frac = resolve_completion_fraction()
    usable = max(0, seat_context - margin)
    ceiling = max(_DERIVED_CEILING_FLOOR, int(usable * frac))
    digest = max(_DERIVED_DIGEST_FLOOR, int(usable * (1.0 - frac)))
    ctx = max(usable, ceiling + digest)
    return digest, ceiling, ctx


def resolve_ctx_margin() -> int:
    """Reserved head-room (tokens) subtracted from the seat context before
    deriving budgets (``SEMANTIK_HEADING_JUDGE_CTX_MARGIN``, default 4096)."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_CTX_MARGIN",
                                _DEFAULT_CTX_MARGIN)


def resolve_completion_fraction() -> float:
    """Fraction of the usable seat window given to the completion CEILING
    (``SEMANTIK_HEADING_JUDGE_COMPLETION_FRACTION``, default 0.7, clamped to
    ``[0.4, 0.9]``); the remainder is the digest/prompt budget. Parse-with-
    fallback: blank / non-float / NaN / ±Inf / garbage → the 0.7 default."""
    raw = (os.environ.get("SEMANTIK_HEADING_JUDGE_COMPLETION_FRACTION")
           or "").strip()
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_COMPLETION_FRACTION
    if f != f or f in (float("inf"), float("-inf")):
        return _DEFAULT_COMPLETION_FRACTION
    return min(_COMPLETION_FRACTION_MAX, max(_COMPLETION_FRACTION_MIN, f))


def resolve_digest_budget_tokens() -> int:
    """Effective digest/prompt token budget before splitting into N.M windows.

    Precedence: explicit ``SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET`` >
    seat-derived (fits the resolved seat context) > the legacy
    ``_DIGEST_BUDGET_TOKENS`` (65k-seat) default."""
    explicit = _explicit_pos_int_env("SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET")
    if explicit is not None:
        return explicit
    ctx = resolve_heading_judge_seat_context()
    if ctx is not None:
        return _derive_budgets(ctx)[0]
    return _DIGEST_BUDGET_TOKENS


def resolve_ctx_tokens_budget() -> int:
    """Effective prompt+completion budget (the doubled-retry context guard).

    Precedence: explicit ``SEMANTIK_HEADING_JUDGE_CTX_BUDGET`` > seat-derived >
    the legacy ``_CTX_TOKENS_BUDGET`` (65k-seat) default."""
    explicit = _explicit_pos_int_env("SEMANTIK_HEADING_JUDGE_CTX_BUDGET")
    if explicit is not None:
        return explicit
    ctx = resolve_heading_judge_seat_context()
    if ctx is not None:
        return _derive_budgets(ctx)[2]
    return _CTX_TOKENS_BUDGET


# ── Thinking-OFF completion-budget resolvers. ────────────────────────────────
# Consumed by the thinking-aware short-circuit atop the three budget resolvers
# below. Explicit ``*_THINKOFF`` env wins → else the JSON-sized default.
def resolve_max_tokens_ceiling_thinkoff() -> int:
    """Thinking-OFF completion CEILING —
    ``SEMANTIK_HEADING_JUDGE_MAX_TOKENS_THINKOFF`` (default 4096)."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_MAX_TOKENS_THINKOFF",
                                _MAX_TOKENS_CEILING_THINKOFF)


def resolve_max_tokens_floor_thinkoff() -> int:
    """Thinking-OFF completion FLOOR —
    ``SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR_THINKOFF`` (default 512)."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR_THINKOFF",
                                _MAX_TOKENS_FLOOR_THINKOFF)


def resolve_est_per_judgment_thinkoff() -> int:
    """Thinking-OFF per-judgment completion estimate —
    ``SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT_THINKOFF`` (default 64)."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT_THINKOFF",
                                _EST_TOKENS_PER_JUDGMENT_THINKOFF)


def resolve_max_tokens_ceiling() -> int:
    """Effective completion-token CEILING (thinking-aware).

    When reasoning is OFF (:func:`resolve_heading_judge_enable_thinking`, the
    DEFAULT for this classification task) a judgment is ~24 tokens of pure JSON
    with no ``<think>`` block, so the ceiling collapses to the JSON-sized
    thinking-off value (:func:`resolve_max_tokens_ceiling_thinkoff`, explicit
    ``*_THINKOFF`` env > 4096) — bounding the degenerate-repetition runaway the
    20k thinking floor licensed. Thinking-ON is BYTE-IDENTICAL to the legacy
    path — precedence: explicit ``SEMANTIK_HEADING_JUDGE_MAX_TOKENS`` >
    seat-derived > the legacy ``_MAX_TOKENS_CEILING`` (65k-seat) default."""
    if not resolve_heading_judge_enable_thinking():
        return resolve_max_tokens_ceiling_thinkoff()
    explicit = _explicit_pos_int_env("SEMANTIK_HEADING_JUDGE_MAX_TOKENS")
    if explicit is not None:
        return explicit
    ctx = resolve_heading_judge_seat_context()
    if ctx is not None:
        return _derive_budgets(ctx)[1]
    return _MAX_TOKENS_CEILING


# ── Part-2 env-tunable sizes (default == the module constant → byte-identical
# when unset). All parse-with-fallback: positive int wins; blank / garbage /
# non-positive → the current default. ───────────────────────────────────────
def resolve_max_tokens_floor() -> int:
    """Completion-token FLOOR (thinking-aware).

    Thinking OFF → the JSON-sized thinking-off floor
    (:func:`resolve_max_tokens_floor_thinkoff`, ``*_THINKOFF`` env > 512): with
    no ``<think>`` block a 20480 floor is a 20k-token runaway license. Thinking
    ON → BYTE-IDENTICAL: ``SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR`` > the legacy
    ``_MAX_TOKENS_FLOOR`` (20480, the thinking-deliberation room)."""
    if not resolve_heading_judge_enable_thinking():
        return resolve_max_tokens_floor_thinkoff()
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR",
                                _MAX_TOKENS_FLOOR)


def resolve_max_pending_per_window() -> int:
    """HARD pending-per-window ceiling — ``SEMANTIK_HEADING_JUDGE_MAX_PENDING_PER_WINDOW``."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_MAX_PENDING_PER_WINDOW",
                                _MAX_PENDING_PER_WINDOW)


def resolve_min_pending_window_cap() -> int:
    """Floor on the budget-derived window cap — ``SEMANTIK_HEADING_JUDGE_MIN_PENDING_WINDOW_CAP``."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_MIN_PENDING_WINDOW_CAP",
                                _MIN_PENDING_WINDOW_CAP)


def resolve_window_concurrency() -> int:
    """Concurrent window-POST fan-out width — ``SEMANTIK_HEADING_JUDGE_CONCURRENCY``."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_CONCURRENCY",
                                _WINDOW_CONCURRENCY)


def resolve_max_coverage_resplit_rounds() -> int:
    """Coverage re-split round cap — ``SEMANTIK_HEADING_JUDGE_MAX_COVERAGE_RESPLIT_ROUNDS``."""
    return _resolve_pos_int_env(
        "SEMANTIK_HEADING_JUDGE_MAX_COVERAGE_RESPLIT_ROUNDS",
        _MAX_COVERAGE_RESPLIT_ROUNDS)


def resolve_min_pending_per_split() -> int:
    """Min pendings for a window to be splittable — ``SEMANTIK_HEADING_JUDGE_MIN_PENDING_PER_SPLIT``."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_MIN_PENDING_PER_SPLIT",
                                _MIN_PENDING_PER_SPLIT)


def resolve_anchor_truncate() -> int:
    """Content-anchor first-sentence char limit — ``SEMANTIK_HEADING_JUDGE_ANCHOR_TRUNCATE``."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_ANCHOR_TRUNCATE",
                                _ANCHOR_TRUNCATE)


def resolve_heading_text_truncate() -> int:
    """Per-heading digest-line char limit — ``SEMANTIK_HEADING_JUDGE_HEADING_TEXT_TRUNCATE``."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_HEADING_TEXT_TRUNCATE",
                                _HEADING_TEXT_TRUNCATE)


def resolve_context_text_truncate() -> int:
    """Fixed-anchor context-line char limit — ``SEMANTIK_HEADING_JUDGE_CONTEXT_TEXT_TRUNCATE``."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_CONTEXT_TEXT_TRUNCATE",
                                _CONTEXT_TEXT_TRUNCATE)


def resolve_heading_judge_est_per_judgment() -> int:
    """Per-JUDGMENT completion-token estimate (default ``_EST_TOKENS_PER_JUDGMENT``).

    The BINDING lever for a longer-context seat. It drives TWO knobs at once
    (they estimate the SAME quantity — thinking-inclusive completion tokens per
    judgment): the per-pending max_tokens BUDGET increment (`_max_tokens_for`)
    AND the pending-count window-cap divisor (`resolve_pending_window_cap`).
    Default 300 → byte-identical to the 65k-seat behaviour. On a 250k seat the
    run-env raises it to ~2000 so a window's budget (`floor + est*n`) actually
    covers thinking-heavy pages (measured ~1875 tok/pending), instead of the
    300-capped budget that truncated. Read at CALL time; mirrors
    `resolve_max_tokens_ceiling` / `resolve_ctx_tokens_budget`. Parse-with-
    fallback: positive int wins; blank / garbage / non-positive → the default.

    THINKING-AWARE: when reasoning is OFF (the DEFAULT) a judgment is ~24 tokens
    of JSON, so this collapses to the thinking-off estimate
    (:func:`resolve_est_per_judgment_thinkoff`, ``*_THINKOFF`` env > 64) — and,
    because BOTH the `_max_tokens_for` budget increment and the
    `resolve_pending_window_cap` divisor read THIS one resolver, the packing est
    and the budget est stay the SAME value in thinking-off mode too (the
    single-source-of-truth invariant `_MAX_TOKENS_PER_PENDING ==
    _EST_TOKENS_PER_JUDGMENT` holds by construction). Thinking ON → byte-identical."""
    if not resolve_heading_judge_enable_thinking():
        return resolve_est_per_judgment_thinkoff()
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT",
                                _EST_TOKENS_PER_JUDGMENT)


def resolve_heading_judge_max_split_depth() -> int:
    """Parse-with-fallback ``SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH`` (default 3).

    The maximum number of times the finish_reason=length split ladder halves a
    pending window before the DEEPEST rung falls to the doubled-max_tokens
    last-resort retry (then fail-open for those minimal pendings). ``0`` is
    HONOURED (never split — the explicit revert lever restoring the legacy
    doubled-retry-then-fail-open behaviour: a first truncation goes straight to
    the doubled retry). Blank / non-int / negative / garbage → the default 3.
    Read at CALL time (never cached at import). Mirrors
    ``reasoning_qc_vlm.resolve_reasoning_qc_max_split_depth``."""
    raw = os.environ.get("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH")
    if not raw:
        return _DEFAULT_MAX_SPLIT_DEPTH
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_SPLIT_DEPTH
    if val < 0:
        return _DEFAULT_MAX_SPLIT_DEPTH
    return val


def _estimate_prompt_tokens(messages: Sequence[Dict[str, str]]) -> int:
    # ctx-budget GUARD — routed through the real-tokenizer counter (auto/id),
    # plus a small fixed per-message role/framing overhead so the chat-template
    # wrapping is approximately included (conservative, slightly over). `off`
    # mode is byte-identical to the legacy chars//3 guard.
    text = "".join(str(m.get("content") or "") for m in messages)
    if resolve_tokenizer_mode() == "off":
        return len(text) // 3
    return _count_tokens(text) + _PROMPT_ROLE_OVERHEAD_TOKENS * len(messages)


_WINDOW_HEADINGS_MARKER = "\nWINDOW HEADINGS:\n"
_PENDING_LINE_RE = re.compile(r"^R(\d+) p\S+ h\d+\*")
_LINE_ID_RE = re.compile(r"^R(\d+) ")


def _split_window_digest(
    digest: str, win_pending: Sequence[int],
) -> List[Tuple[str, List[int]]]:
    """Reasoning-preserving SPLIT of one window digest into two halves at the
    midpoint of its PENDING lines (anchor continuation lines travel with their
    heading; the FIXED-ANCHOR OUTLINE preamble — when present — is kept on both
    halves so each retains the chapter spine context). Returns ``[]`` when the
    window has fewer than 2 pending lines (nothing to split)."""
    if _WINDOW_HEADINGS_MARKER in digest:
        preamble, body = digest.split(_WINDOW_HEADINGS_MARKER, 1)
        preamble = preamble + _WINDOW_HEADINGS_MARKER
    else:
        preamble, body = "", digest
    groups: List[List[str]] = []
    for ln in body.split("\n"):
        if ln.startswith("    >") and groups:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    pend_set = set(int(i) for i in win_pending)

    def _gid(g: List[str]) -> Optional[int]:
        m = _LINE_ID_RE.match(g[0])
        return int(m.group(1)) if m else None

    pending_idx = [i for i, g in enumerate(groups)
                   if _PENDING_LINE_RE.match(g[0]) and _gid(g) in pend_set]
    if len(pending_idx) < resolve_min_pending_per_split():
        return []
    cut = pending_idx[len(pending_idx) // 2]  # first half = pendings before cut
    halves = [groups[:cut], groups[cut:]]
    out: List[Tuple[str, List[int]]] = []
    for half in halves:
        ids = [_gid(g) for g in half
               if _PENDING_LINE_RE.match(g[0]) and _gid(g) in pend_set]
        body_txt = "\n".join(ln for g in half for ln in g)
        out.append((preamble + body_txt, [i for i in ids if i is not None]))
    return out


def _focus_window_digest(digest: str, keep_ids: Sequence[int]) -> str:
    """Last-resort FOCUSED re-judge digest for a residual unjudged set too
    small to halve: keep the preamble + every FIXED line (the chapter spine
    context) + ONLY the pending groups in ``keep_ids`` (anchor continuation
    lines travel with their heading), dropping every other pending group — so
    a lone unjudged pending gets a small window that can COMPLETE instead of
    re-POSTing the whole exhausting digest."""
    if _WINDOW_HEADINGS_MARKER in digest:
        preamble, body = digest.split(_WINDOW_HEADINGS_MARKER, 1)
        preamble = preamble + _WINDOW_HEADINGS_MARKER
    else:
        preamble, body = "", digest
    groups: List[List[str]] = []
    for ln in body.split("\n"):
        if ln.startswith("    >") and groups:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    keep = set(int(i) for i in keep_ids)
    out_lines: List[str] = []
    for g in groups:
        if _PENDING_LINE_RE.match(g[0]):
            m = _LINE_ID_RE.match(g[0])
            if m is None or int(m.group(1)) not in keep:
                continue
        out_lines.extend(g)
    return preamble + "\n".join(out_lines)


def _transport_wmeta(
    exc: _JudgeTransportError, *, finish: Optional[str] = None,
) -> Dict[str, Any]:
    """Window meta for a non-timeout transport failure, MECHANISM-tagged
    (seat aborted the in-flight request vs seat unreachable)."""
    return {
        "transport_failure": True,
        "finish": "abort" if exc.aborted else finish,
        "mechanisms": [MECHANISM_SEAT_ABORTED if exc.aborted
                       else MECHANISM_SEAT_UNREACHABLE],
        "failure_detail": str(exc),
    }


def _parse_failure_wmeta(
    content: Optional[str], finish: Optional[str],
) -> Dict[str, Any]:
    """Window meta for an unparseable reply. An EMPTY body is a DIFFERENT
    mechanism from a body that simply is not the required JSON — conflating
    them is what made a mode-collapsed / dying seat read as model noise."""
    body = content or ""
    empty = not body.strip()
    return {
        "parse_failure": True,
        "finish": finish,
        "mechanisms": [MECHANISM_EMPTY_CONTENT if empty
                       else MECHANISM_PARSE_FAILURE],
        "failure_detail": (
            f"finish_reason={finish!r}, {len(body)} char(s) of content"
        ),
    }


def _judge_one_window(
    digest: str,
    n_headings: int,
    n_pending: int,
    max_tokens: int,
    *,
    post_fn: Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]],
    win_pending: Optional[Sequence[int]] = None,
    depth: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """POST one window; on over-deliberation — ``finish_reason=length`` OR a
    read-TIMEOUT (its wall-clock flavor) — fall to the reasoning-preserving
    window SPLIT FIRST (halve the pendings, judge each half thinking-on at a
    fresh completion floor — the reasoning-QC split-ladder precedent; NEVER a
    thinking-off retry). The doubled-max_tokens retry is the LAST resort for
    an UNSPLITTABLE (single-pending / depth-capped) window only: measured
    live, a window that exhausts ~18.6k thinking tokens free-runs to whatever
    cap it is granted (37k also exhausted/timed out), so more rope wastes
    10-20 min of wall where the split finishes. Returns
    ``(parsed_or_None, wmeta)``."""
    messages = build_judge_messages(digest, n_headings, n_pending)
    over_deliberated = False
    content: Optional[str] = None
    finish: Optional[str] = None
    # CONTEXT-BUDGET GUARD on the FIRST POST (not only the doubled retry below):
    # under an adaptive seat context `max_tokens` can be the full derived ceiling
    # (`_derive_budgets(seat_ctx)[1]`), so a large window whose prompt already
    # fills much of the seat window makes `prompt + max_tokens > seat context`
    # and the seat 400s ("requested N output tokens + prompt M > max context") —
    # classified as a non-timeout `_JudgeTransportError`, which fail-opens the
    # WHOLE window's pendings. Cap the requested completion so
    # `prompt + requested <= ctx_budget <= seat - margin` holds on EVERY POST.
    # Use the CONSERVATIVE prompt estimator (chars//3, overcounts) for the cap
    # while windows are SIZED with `_estimate_tokens` (chars//4, undercounts). A
    # small window → `capped == max_tokens` → byte-identical to today.
    capped_max_tokens = min(max_tokens,
                            resolve_ctx_tokens_budget() - _estimate_prompt_tokens(messages))
    if capped_max_tokens >= resolve_max_tokens_floor():
        try:
            content, finish = post_fn(messages, capped_max_tokens)
        except _JudgeTransportError as exc:
            if not exc.timeout:
                return None, _transport_wmeta(exc)
            over_deliberated = True
            finish = "timeout"
        if finish == "length":
            over_deliberated = True
    else:
        # The prompt alone crowds the seat window — no viable completion fits
        # under `ctx_budget`. Do NOT fire the doomed first POST (it would 400);
        # go STRAIGHT to the split ladder, since a window whose PROMPT cannot fit
        # can only be repaired by splitting it smaller. An unsplittable
        # (single-pending / depth-capped) window then fails open below — the
        # genuinely-unsplittable edge, no longer a 42k-slice 400.
        over_deliberated = True
        finish = "length"

    if over_deliberated:
        # SPLIT FIRST (the truncation-proof ladder): a window whose completion
        # need exceeds even the token ceiling can ONLY be fixed by splitting it
        # smaller, never by more tokens — so on finish_reason=length (or its
        # wall-clock timeout flavor) HALVE the pending set and re-judge each
        # half at its OWN smaller _max_tokens_for budget. Recurse up to the
        # env-bounded depth; the doubled-max_tokens retry below is the LAST
        # resort for an UNSPLITTABLE (single-pending / depth-capped) window
        # only. depth==0 with the depth bound at 0 skips straight to it (the
        # explicit revert lever).
        halves = (_split_window_digest(digest, win_pending or [])
                  if depth < resolve_heading_judge_max_split_depth() else [])
        if len(halves) == 2:
            merged, agg = _fan_halves(halves, n_headings, post_fn, depth)
            agg["finish"] = "split"
            if merged:
                _stamp_coverage_gap(agg, merged, win_pending)
                return {"levels": merged}, agg
            agg["length_exhausted"] = True
            _add_mechanism(agg, MECHANISM_LENGTH_EXHAUSTED)
            return None, agg
        # Unsplittable: the doubled retry is the only remaining move.
        new_max = min(resolve_max_tokens_ceiling(), max_tokens * 2)
        if new_max > max_tokens and (
                _estimate_prompt_tokens(messages) + new_max
                <= resolve_ctx_tokens_budget()):
            try:
                content, finish = post_fn(messages, new_max)
            except _JudgeTransportError as exc:
                return None, _transport_wmeta(exc, finish=finish)
            if finish != "length":
                parsed = parse_judge_response(content)
                if parsed is None:
                    return None, _parse_failure_wmeta(content, finish)
                wmeta = {"finish": finish}
                _stamp_coverage_gap(wmeta, parsed.get("levels") or {}, win_pending)
                return parsed, wmeta
        return None, {"length_exhausted": True, "finish": finish,
                      "mechanisms": [MECHANISM_LENGTH_EXHAUSTED]}

    parsed = parse_judge_response(content)
    if parsed is None:
        return None, _parse_failure_wmeta(content, finish)
    # COVERAGE contract (live ch06 defect): a syntactically-valid verdict may
    # OMIT assigned pendings (102/193 returned, 91 silently dropped — no
    # failure flag fires). Heal by splitting over the MISSING ids only and
    # merging (smaller focused windows re-attend); a residual gap is stamped
    # so the caller never caches an under-judged verdict.
    levels: Dict[str, Any] = dict(parsed.get("levels") or {})
    missing = _coverage_gap_ids(levels, win_pending)
    if (len(missing) >= resolve_min_pending_per_split()
            and depth < resolve_heading_judge_max_split_depth()):
        heal_halves = _split_window_digest(digest, missing)
        if len(heal_halves) == 2:
            healed, agg = _fan_halves(heal_halves, n_headings, post_fn, depth)
            levels.update(healed)
            agg["finish"] = finish
            agg["coverage_healed"] = len(healed)
            _stamp_coverage_gap(agg, levels, win_pending)
            return {"levels": levels}, agg
    wmeta = {"finish": finish}
    _stamp_coverage_gap(wmeta, levels, win_pending)
    return {"levels": levels}, wmeta


def _fan_halves(
    halves: List[Tuple[str, List[int]]],
    n_headings: int,
    post_fn,
    depth: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Judge two half-windows CONCURRENTLY (an exhausting window costs
    first-POST + max(halves), not + sum(halves); the seat batches them).
    Bounded:
    ≤2 threads per split, depth ≤ resolve_heading_judge_max_split_depth().
    Returns
    ``(merged_levels, aggregated_flags)``."""
    from concurrent.futures import ThreadPoolExecutor

    agg: Dict[str, Any] = {"split_depth": depth + 1}
    merged: Dict[str, Any] = {}

    def _judge_half(pair):
        hd, hp = pair
        return _judge_one_window(
            hd, n_headings, len(hp), _max_tokens_for(len(hp)),
            post_fn=post_fn, win_pending=hp, depth=depth + 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_judge_half, halves))
    for parsed_h, wmeta_h in results:
        for flag in ("transport_failure", "parse_failure", "length_exhausted"):
            if wmeta_h.get(flag):
                agg[flag] = True
        for token in wmeta_h.get("mechanisms") or ():
            _add_mechanism(agg, token)
        if wmeta_h.get("failure_detail") and not agg.get("failure_detail"):
            agg["failure_detail"] = wmeta_h["failure_detail"]
        if parsed_h is not None:
            merged.update(parsed_h.get("levels") or {})
    return merged, agg


def _coverage_gap_ids(
    levels: Dict[str, Any], win_pending: Optional[Sequence[int]],
) -> List[int]:
    """Pending region ids the verdict did NOT assign a level to."""
    have = set()
    for k in levels:
        try:
            have.add(int(k))
        except (TypeError, ValueError):
            continue
    return [int(i) for i in (win_pending or []) if int(i) not in have]


def _coverage_gap_tolerance(win_pending: Optional[Sequence[int]]) -> int:
    """Gap size accepted as honest fail-open (1 id, or 2% of a big window)."""
    return max(1, len(win_pending or []) // 50)


def _stamp_coverage_gap(
    wmeta: Dict[str, Any],
    levels: Dict[str, Any],
    win_pending: Optional[Sequence[int]],
) -> None:
    gap = len(_coverage_gap_ids(levels, win_pending))
    if gap > _coverage_gap_tolerance(win_pending):
        wmeta["coverage_gap"] = gap
        # A coverage gap is only a MECHANISM in its own right when nothing
        # else already explained the window; a gap that a seat-abort /
        # transport / length failure CAUSED is a consequence, and reporting
        # both would bury the root cause the operator has to act on. (The
        # ``coverage_gap`` flag itself stays unconditional — it is what keeps
        # an under-judged verdict out of the resume cache.)
        if not wmeta.get("mechanisms"):
            _add_mechanism(wmeta, MECHANISM_COVERAGE_GAP)


def _judge_window_to_completion(
    digest: str,
    n_headings: int,
    win_pending: Sequence[int],
    max_tokens: int,
    *,
    post_fn: Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Drive ONE window to FULL pending coverage (the Fix-2 truncation-proof
    loop). The first attempt is the existing :func:`_judge_one_window` ladder;
    any residual UNJUDGED pendings — the audit's truncation hole: a split half
    exhausted its completion budget and its 19-heading band silently stayed
    "kept" — are RE-SPLIT over the MISSING ids only and re-judged, looping
    until every pending is judged, progress stops, or the hard round cap. A
    residual set is then stamped as the EXPLICIT ``unjudged_ids`` + logged
    LOUDLY (never silently folded into "kept").

    Progress-gated: when the first attempt judged NOTHING (``levels`` empty),
    the internal split ladder already exhausted its options over these exact
    pendings — an immediate identical re-split would re-POST the same halves,
    so the fail-open path stays byte-identical (no extra seat time).
    """
    parsed, wmeta = _judge_one_window(
        digest, n_headings, len(win_pending), max_tokens,
        post_fn=post_fn, win_pending=win_pending)
    levels: Dict[str, Any] = dict((parsed or {}).get("levels") or {})
    missing = _coverage_gap_ids(levels, win_pending)
    rounds = 0
    while missing and levels and rounds < resolve_max_coverage_resplit_rounds():
        rounds += 1
        halves = _split_window_digest(digest, missing)
        if len(halves) == 2:
            subs = halves
        else:
            # Too few missing pendings to halve — judge them in one small
            # FOCUSED window instead of re-POSTing the exhausting digest.
            subs = [(_focus_window_digest(digest, missing), list(missing))]
        for hd, hp in subs:
            if not hp:
                continue
            sub_parsed, sub_meta = _judge_one_window(
                hd, n_headings, len(hp), _max_tokens_for(len(hp)),
                post_fn=post_fn, win_pending=hp, depth=1)
            for flag in ("transport_failure", "parse_failure",
                         "length_exhausted"):
                if sub_meta.get(flag):
                    wmeta[flag] = True
            for token in sub_meta.get("mechanisms") or ():
                _add_mechanism(wmeta, token)
            if sub_meta.get("failure_detail") and not wmeta.get(
                    "failure_detail"):
                wmeta["failure_detail"] = sub_meta["failure_detail"]
            if sub_parsed:
                levels.update(sub_parsed.get("levels") or {})
        new_missing = _coverage_gap_ids(levels, win_pending)
        if len(new_missing) >= len(missing):
            missing = new_missing
            break  # no progress this round — stop burning seat time
        missing = new_missing
    if rounds:
        wmeta["resplit_rounds"] = rounds
    # Re-stamp coverage from the FINAL merged levels (a stale gap from the
    # first attempt must not survive a successful heal).
    wmeta.pop("coverage_gap", None)
    _stamp_coverage_gap(wmeta, levels, win_pending)
    if missing:
        wmeta["unjudged_ids"] = sorted(int(i) for i in missing)
        logger.warning(
            "heading-judge: %d of %d pending heading(s) UNJUDGED after "
            "%d re-split round(s) — MECHANISM: %s%s; explicit unjudged set: %s",
            len(missing), len(list(win_pending)), rounds,
            describe_failure_modes(wmeta.get("mechanisms") or ()),
            (f" [{wmeta['failure_detail']}]"
             if wmeta.get("failure_detail") else ""),
            wmeta["unjudged_ids"])
    if not levels:
        return None, wmeta
    return {"levels": levels}, wmeta


def judge_heading_levels(
    plan: SkeletonPlan,
    *,
    seat: Optional[HeadingJudgeSeat] = None,
    post_fn: Optional[Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]]] = None,
    use_cache: Optional[bool] = None,
    model: Optional[str] = None,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """One seat call per window → ``{region_index: proposed_level}`` (unfiltered
    levels; :func:`apply_judged_levels` owns range + clamp validation).

    Fail-soft: a window whose POST/parse fails contributes nothing (its pendings
    keep their current level). Never invents a level.
    """
    model = model or (seat.model if seat else resolve_heading_judge_model())
    if use_cache is None:
        use_cache = resolve_heading_judge_checkpoint()
    if post_fn is None:
        post_fn = _make_default_post_fn(seat or resolve_heading_judge_seat())

    verdict_map: Dict[int, int] = {}
    meta: Dict[str, Any] = {
        "windows": len(plan.windows), "posts": 0, "cache_hits": 0,
        "cache_misses": 0, "length_exhausted": 0, "parse_failures": 0,
        "transport_failures": 0, "finish": None, "cache_hit": False,
    }
    n_headings = len(plan.entries)

    def _run_window(
        digest: str, win_pending: List[int],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """One window: cache probe -> POST -> cache put. Thread-safe (the
        cache is per-window atomic-rename JSON files; wmeta is thread-local
        and aggregated by the caller)."""
        n_pending = len(win_pending)
        max_tokens = _max_tokens_for(n_pending)
        key = _window_cache_key(digest, model, max_tokens) if use_cache else None
        if key is not None:
            cached = _cache_get(key)
            if cached is not None:
                # A cached verdict written before the coverage contract may be
                # PARTIAL (the live ch06 poisoning: valid JSON omitting 91
                # assigned ids, no failure flag). Serve it only when its gap
                # is within the honest fail-open tolerance; else treat as a
                # MISS so the window re-judges + heals + re-caches.
                gap = _coverage_gap_ids(cached.get("levels") or {}, win_pending)
                if len(gap) <= _coverage_gap_tolerance(win_pending):
                    return cached, {"cache_hit": True}
                logger.warning(
                    "heading-judge cached verdict has coverage gap %d/%d — "
                    "re-judging window", len(gap), len(win_pending))
        parsed, wmeta = _judge_window_to_completion(
            digest, n_headings, win_pending, max_tokens, post_fn=post_fn)
        # Cache ONLY a fully-clean verdict: a partial split (one half judged,
        # one failed) or an under-judged verdict (coverage_gap) must NOT
        # persist, or a resume would serve the partial forever and its
        # pendings would never be re-judged.
        clean = not any(wmeta.get(f) for f in (
            "transport_failure", "parse_failure", "length_exhausted",
            "coverage_gap"))
        if key is not None and parsed is not None and parsed.get("levels") and clean:
            _cache_put(key, parsed)
        wmeta["posted"] = True
        return parsed, wmeta

    active = [(d, p) for d, p in plan.windows if p]
    results: List[Tuple[Optional[Dict[str, Any]], Dict[str, Any]]] = []
    if len(active) <= 1:
        results = [_run_window(d, p) for d, p in active]
    else:
        # Fan the independent window judgments out concurrently — the seat's
        # continuous batching runs them together (sequential windows made a
        # 4-window chapter a 30-40 min pass at single-stream tok/s).
        from concurrent.futures import ThreadPoolExecutor

        width = max(1, min(resolve_window_concurrency(), len(active)))
        with ThreadPoolExecutor(max_workers=width) as ex:
            results = list(ex.map(lambda dp: _run_window(*dp), active))

    unjudged_ids: List[int] = []
    failure_modes: List[str] = []
    # Per-window verdicts aligned to ``active`` (== plan.windows in normalized
    # mode, where every window has >= 1 pending) — the REDUCE driver needs each
    # slice's OWN verdict (an overlap heading judged by two slices must keep both
    # so the chapter reconcile can choose), which the merged verdict_map below
    # loses to last-writer-wins. Additive to meta; ignored by every legacy path.
    window_verdicts: List[Tuple[Tuple[int, ...], Dict[int, int]]] = []
    for (_wd, wp), (parsed, wmeta) in zip(active, results):
        wp_set = {int(i) for i in wp}
        wv: Dict[int, int] = {}
        if parsed:
            for k, v in (parsed.get("levels") or {}).items():
                try:
                    ik = int(str(k).strip())
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if ik in wp_set:
                    wv[ik] = iv
        window_verdicts.append((tuple(int(i) for i in wp), wv))
        for token in wmeta.get("mechanisms") or ():
            if token not in failure_modes:
                failure_modes.append(token)
        if wmeta.get("failure_detail") and not meta.get("failure_detail"):
            meta["failure_detail"] = wmeta["failure_detail"]
        if wmeta.get("cache_hit"):
            meta["cache_hits"] += 1
            meta["cache_hit"] = True
        else:
            if wmeta.get("posted"):
                meta["posts"] += 1
            meta["finish"] = wmeta.get("finish") or meta["finish"]
            if wmeta.get("length_exhausted"):
                meta["length_exhausted"] += 1
            if wmeta.get("parse_failure"):
                meta["parse_failures"] += 1
            if wmeta.get("transport_failure"):
                meta["transport_failures"] += 1
            if wmeta.get("resplit_rounds"):
                meta["resplit_rounds"] = (
                    meta.get("resplit_rounds", 0)
                    + int(wmeta["resplit_rounds"])
                )
            meta["cache_misses"] += 1
        unjudged_ids.extend(int(i) for i in wmeta.get("unjudged_ids") or [])
        if parsed:
            for k, v in (parsed.get("levels") or {}).items():
                try:
                    ik = int(str(k).strip())
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                verdict_map[ik] = iv
    # Explicit unjudged accounting (Fix 2): the ids no window's verdict ever
    # covered — the truncation hole must be a LOUD, named set, never silently
    # conflated with judged-kept.
    unjudged_final = sorted(i for i in set(unjudged_ids)
                            if i not in verdict_map)
    if failure_modes:
        meta["failure_modes"] = list(failure_modes)
    if unjudged_final:
        meta["unjudged_ids"] = unjudged_final
        meta["unjudged"] = len(unjudged_final)
        meta["unjudged_reason"] = describe_failure_modes(failure_modes)
        total_pending = len(plan.pending_ids)
        detail = (f" [{meta['failure_detail']}]"
                  if meta.get("failure_detail") else "")
        if total_pending and len(unjudged_final) >= total_pending:
            # A 100%-unjudged chapter is a real failure, never a judged keep.
            # The fail-open keeps every pre-judge level, so it must be loud.
            logger.error(
                "heading-judge: ALL %d pending heading(s) left UNJUDGED "
                "across %d window(s) — the judge produced NO usable verdict "
                "for this chapter; every pre-judge level was KEPT (fail-open) "
                "and this is a REAL FAILURE, not a judged-keep. MECHANISM: "
                "%s%s; explicit unjudged set: %s",
                total_pending, len(plan.windows),
                meta["unjudged_reason"], detail, unjudged_final)
        else:
            logger.warning(
                "heading-judge: %d of %d pending heading(s) left UNJUDGED "
                "across %d window(s) — MECHANISM: %s%s; explicit unjudged "
                "set: %s",
                len(unjudged_final), total_pending, len(plan.windows),
                meta["unjudged_reason"], detail, unjudged_final)
    # Deterministic sibling-staircase post-rule (Fix 1) over the assembled
    # proposals, BEFORE the clamp walk consumes them.
    stair_audit = flatten_sibling_staircases(plan.entries, verdict_map)
    meta["staircase_flattened"] = len(stair_audit)
    if stair_audit:
        logger.info(
            "heading-judge: flattened %d sibling-staircase proposal(s): %s",
            len(stair_audit),
            {rid: f"{fr}->{to}" for rid, (fr, to) in stair_audit.items()})
    meta["max_tokens"] = _max_tokens_for(len(plan.pending_ids))
    meta["window_verdicts"] = window_verdicts
    return verdict_map, meta


# ── Deterministic re-stamp (clamp rules). ────────────────────────────────────
def apply_judged_levels(
    region_provenance: List[Dict[str, Any]],
    heading_tree: List[Any],
    escalations: List[Dict[str, Any]],
    verdict_map: Dict[int, int],
) -> ApplyResult:
    """Re-stamp PENDING headings only, under the clamp rules. Mirrors
    ``transform._anchor_declared_sections``: level only (never text), then the
    heading_tree is rebuilt from region_provenance.

    Rule 5 fail-open: an empty ``verdict_map`` → NO mutation (byte-identical;
    pending flags + escalations retained)."""
    result = ApplyResult()

    prov = region_provenance
    # Index the writable (pending) heading regions by region id.
    pending_by_id: Dict[int, Dict[str, Any]] = {}
    for r in prov:
        if r.get("region_kind") == "heading" and bool(r.get("heading_level_pending")):
            rid = r.get("first_raw_block_index")
            if rid is not None:
                pending_by_id[int(rid)] = r

    # Pendings the verdict COVERED at all (valid or not) — the basis of the
    # judged-kept vs UNJUDGED accounting distinction (Fix 2).
    judged_pending: set = set()
    for rid in verdict_map:
        try:
            irid = int(rid)
        except (TypeError, ValueError):
            continue
        if irid in pending_by_id:
            judged_pending.add(irid)

    if not verdict_map:
        # Total fail-open: every pending is kept, and NONE was judged.
        result.kept = len(pending_by_id)
        result.unjudged = len(pending_by_id)
        return result

    # Rule 1 + 2 + per-entry validation: drop unknown / non-pending / out-of-range.
    accepted: Dict[int, int] = {}
    for rid, lvl in verdict_map.items():
        if rid not in pending_by_id:
            result.dropped += 1  # unknown id or a fixed-anchor target
            continue
        try:
            ilvl = int(lvl)
        except (TypeError, ValueError):
            result.dropped += 1
            continue
        if ilvl < 2 or ilvl > 6:  # rule 2: level 1 never assignable; range [2,6]
            result.dropped += 1
            continue
        # D3.4 defensive clamp: a numbered caption/pedagogical label
        # ("Figure/Example/Table/Try It N.M") must never be assigned a section
        # level ≤3 (it is furniture the transform should have routed off the
        # heading track). Drop the assignment, tallied like the fixed-anchor drop.
        if ilvl <= 3 and rm.caption_label_kind(
            str(pending_by_id[rid].get("heading_text") or "")
        ) is not None:
            result.dropped += 1
            continue
        accepted[rid] = ilvl

    if not accepted:
        # Everything dropped → nothing to apply (a fail-open keep-current).
        result.kept = len(pending_by_id)
        result.kept_judged = len(judged_pending)
        result.unjudged = len(pending_by_id) - len(judged_pending)
        return result

    # Effective-level walk in document order (rules 3 + 4).
    headings = [r for r in prov if r.get("region_kind") == "heading"]
    prev_effective = 1
    enclosing_anchor = 1
    corrections: Dict[int, Tuple[int, int, bool]] = {}
    for h in headings:
        rid = h.get("first_raw_block_index")
        is_pending = bool(h.get("heading_level_pending"))
        try:
            cur_level = int(h.get("level", 3) or 3)
        except (TypeError, ValueError):
            cur_level = 3
        if is_pending and rid is not None and int(rid) in accepted:
            proposed = accepted[int(rid)]
            eff = proposed
            clamped = False
            # Rule 4: NO ORPHANING — never above the enclosing fixed anchor + 1.
            lo = enclosing_anchor + 1
            if eff < lo:
                eff, clamped = lo, True
            # Rule 3: PARENT+1 MAX JUMP — never deeper than prev_effective + 1.
            hi = prev_effective + 1
            if eff > hi:
                eff, clamped = hi, True
            # Rule 2 re-clamp (lo/hi could push out of range on a pathological tree).
            bounded = max(2, min(6, eff))
            if bounded != eff:
                clamped = True
            eff = bounded
            corrections[int(rid)] = (cur_level, eff, clamped or eff != proposed)
            prev_effective = eff
            # A corrected pending node can PARENT a following pending node
            # (via prev_effective) but is NOT a fixed anchor for orphaning.
        else:
            eff = cur_level
            prev_effective = eff
            if not is_pending:
                enclosing_anchor = eff  # only fixed anchors set the orphan floor

    # Apply corrections deterministically + escalation hygiene (rule 6).
    for rid, (from_lvl, to_lvl, clamped) in corrections.items():
        region = pending_by_id[rid]
        region["level"] = to_lvl
        region.pop("heading_level_pending", None)
        region["heading_level_judged"] = {
            "from": from_lvl, "to": to_lvl, "clamped": bool(clamped)
        }
        # Remove the matching heading_level_pending escalation row.
        escalations[:] = [
            e for e in escalations
            if not (e.get("reason") == "heading_level_pending"
                    and e.get("region_index") == rid)
        ]
        escalations.append({
            "region_index": rid,
            "source_page": region.get("source_page"),
            "native_label": region.get("native_label", ""),
            "reason": "heading_level_judged",
            "detail": (f"pending heading level judged {from_lvl}->{to_lvl}"
                       f"{' (clamped)' if clamped else ''}"),
            "text": str(region.get("heading_text") or "")[:120],
        })
        result.applied += 1
        if clamped:
            result.clamped += 1
        # Distinguish a recorded verdict from an effective level change:
        # a verdict that re-confirms the level already on the region is a
        # no-op for the render, so it must never be counted as a re-levelling.
        # This is exactly how an idempotent re-judge of an ALREADY-judged
        # layout can report ``applied=N`` while emitting byte-identical HTML.
        if to_lvl != from_lvl:
            result.changed += 1
            key = f"{from_lvl}->{to_lvl}"
            result.transitions[key] = result.transitions.get(key, 0) + 1
        else:
            result.agreed += 1
    result.corrections = corrections
    result.kept = len(pending_by_id) - result.applied
    result.kept_judged = len(judged_pending) - result.applied
    result.unjudged = len(pending_by_id) - len(judged_pending)

    # Rebuild heading_tree from the final provenance (identical to
    # _anchor_declared_sections' rebuild).
    heading_tree[:] = [
        (int(p.get("level", 3) or 3), str(p.get("heading_text", "")))
        for p in prov if p.get("region_kind") == "heading"
    ]
    return result


def _format_transitions(
    transitions: Dict[str, int], *, limit: int = 6
) -> str:
    """``{"3->2": 28, "4->3": 2}`` -> ``"3->2 x28, 4->3 x2"`` (busiest first).

    Empty mapping -> ``""`` so a caller can print an explicit "none". Bounded
    so a pathological histogram cannot flood a log line.
    """
    if not transitions:
        return ""
    rows = sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))
    head = ", ".join(f"{k} x{v}" for k, v in rows[:limit])
    if len(rows) > limit:
        head += f", +{len(rows) - limit} more"
    return head


# ── DecisionCapture (NEW LLM call site). ─────────────────────────────────────
def _detect_course_code() -> str:
    raw = (
        os.environ.get("SEMANTIK_COURSE_CODE")
        or os.environ.get("ED4ALL_COURSE_CODE")
        or ""
    ).strip()
    return raw or "SEMANTIK"


def _emit_judge_capture(
    *,
    model: str,
    n_pending: int,
    result: ApplyResult,
    meta: Dict[str, Any],
    course_code: Optional[str] = None,
) -> None:
    """One ``structure_review`` DecisionCapture per chapter (best-effort).

    Rides the existing ``structure_review`` enum with a ``heading_level_judge``
    metadata discriminator (no ``decision_event`` schema change). Rationale is
    dynamic + replayable."""
    try:
        from lib.decision_capture import DecisionCapture
    except Exception as exc:  # noqa: BLE001 — the cross-venv bridge has no lib/
        logger.debug("heading-judge: DecisionCapture unavailable (non-fatal): %s", exc)
        return
    try:
        cap = DecisionCapture(
            course_code=course_code or _detect_course_code(),
            phase="semantik_conversion",
            tool="semantik",
        )
        cap.log_decision(
            decision_type="structure_review",
            decision=(
                f"heading_level_judge: applied {result.applied} verdict(s) of "
                f"{n_pending} pending heading level(s) "
                f"({result.changed} changed level, {result.agreed} agreed)"
            ),
            rationale=(
                f"Super heading-level judge (model={model}) judged {n_pending} "
                f"PENDING heading(s) across {meta.get('windows', 1)} window(s); the "
                f"deterministic clamp DECIDED — applied={result.applied} verdict(s) "
                f"of which {result.changed} CHANGED a level "
                f"({_format_transitions(result.transitions) or 'none'}) and "
                f"{result.agreed} AGREED with the pre-existing level (recorded, "
                f"render-invisible), clamped={result.clamped}, "
                f"dropped={result.dropped} "
                f"(unknown/non-pending/out-of-range), kept={result.kept} "
                f"(judged-kept={result.kept_judged}, "
                f"UNJUDGED={result.unjudged} — no verdict ever covered them); "
                f"staircase_flattened={meta.get('staircase_flattened', 0)}; "
                f"max_tokens={meta.get('max_tokens')}, finish={meta.get('finish')}, "
                f"cache={'hit' if meta.get('cache_hit') else 'miss'}, "
                f"posts={meta.get('posts', 0)}, "
                f"length_exhausted={meta.get('length_exhausted', 0)}, "
                f"parse_failures={meta.get('parse_failures', 0)}, "
                f"transport_failures={meta.get('transport_failures', 0)}."
            ),
            alternatives_considered=[
                {
                    "option": "Keep every pending heading at level 3",
                    "reason_rejected": (
                        f"Rejected because {result.applied} of {n_pending} pending "
                        "headings received usable judged levels."
                    ),
                },
                {
                    "option": "Apply every proposed level without structural clamps",
                    "reason_rejected": (
                        "Rejected because the invariant pass clamped "
                        f"{result.clamped} and dropped {result.dropped} "
                        "unsafe proposals."
                    ),
                },
            ],
            heading_level_judge=True,
            hj_applied=result.applied,
            hj_changed=result.changed,
            hj_agreed=result.agreed,
            hj_clamped=result.clamped,
            hj_dropped=result.dropped,
            hj_kept=result.kept,
            hj_model=model,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, never fatal
        logger.debug("heading-judge: decision capture failed (non-fatal): %s", exc)


# ── Phase B: whole-document consistency-reconciliation FINAL REVIEW. ─────────
# After the per-window judging completes, run ONE additional whole-document
# pass over the JUDGED tree (SEMANTIK_HEADING_JUDGE_FINAL_REVIEW, default OFF).
# The reasoning seat is asked to reconcile CROSS-DOCUMENT consistency problems
# ONLY — a recurring section title assigned inconsistent levels at the SAME
# structural position, or non-parallel chapter structure — while PRESERVING
# legitimate depth differences (the SAME title at DIFFERENT depths, e.g. a
# chapter-end "Summary" vs an in-section "Summary", must NOT be flattened). The
# model PROPOSES; the deterministic clamp DECIDES: a proposed re-level that
# would violate an invariant (level [2,6] / parent+1 / no-orphaning) is DROPPED
# individually. FAIL-OPEN + never-ship-worse: any transport/parse/length
# failure or a wholesale-invalid review keeps the pre-review judged levels
# byte-for-byte. OFF → byte-identical (no review pass, no extra POST).

_REVIEW_INSTRUCTIONS = (
    "You are a document-structure CONSISTENCY reviewer. You are given the "
    "complete ordered heading skeleton of one document with the level ALREADY "
    "ASSIGNED to every heading (the integer after \"h\"). Lines marked \"*\" are "
    "headings whose level you MAY revise; lines marked \".\" are FIXED anchors "
    "(the chapter title and the numbered N.M section spine) that are correct "
    "and IMMUTABLE. Your task is NOT to re-judge every heading — it is to find "
    "and reconcile genuine CROSS-DOCUMENT CONSISTENCY problems only: a recurring "
    "section title assigned INCONSISTENT levels at the SAME structural position, "
    "or a chapter whose internal structure is non-parallel with its siblings. "
    "CRITICAL — preserve legitimate depth differences: the SAME title at "
    "DIFFERENT structural depths is legitimately different (a chapter-end "
    "\"Summary\" that is a direct child of the chapter vs an in-section "
    "\"Summary\" nested under a subsection are DIFFERENT headings and must KEEP "
    "their different levels). Only reconcile a recurring title whose occurrences "
    "sit at the SAME depth (same parent context) yet were given different "
    "levels — never flatten a same-title-different-depth pair. Propose a level "
    "(an integer 2-6) ONLY for the \"*\" headings you want to CHANGE; omit every "
    "heading that is already correct. Never invent, drop, reorder, or rewrite "
    "headings. Never assign level 1. Only propose changes for ids marked \"*\".\n"
    "Respond with ONLY a JSON object, no prose, in exactly this shape:\n"
    "{\"levels\": {\"<region_id>\": <level int>, ...}}\n"
    "with one entry per heading you want to change (the integer after \"R\", "
    "quoted as a string key). An empty {\"levels\": {}} means the document is "
    "already consistent."
)


@dataclass
class ReviewResult:
    #: number of level changes the review PROPOSED (len of the review verdict).
    proposed: int = 0
    #: proposed changes actually applied (each MOVED a level through the clamp).
    applied: int = 0
    #: proposed changes REJECTED — a fixed-anchor / unknown id, an out-of-range
    #: or caption-label target, or a change that would violate parent+1 /
    #: no-orphaning (dropped individually, never clamped-in-place — the
    #: never-ship-worse contract).
    dropped: int = 0
    #: recurring same-title headings at >1 level the review LEFT ALONE — the
    #: same-title-DIFFERENT-depth pairs correctly preserved (not flattened).
    preserved_legitimate: int = 0
    #: applied changes (== ``applied`` — a review only records genuine moves).
    changed: int = 0
    transitions: Dict[str, int] = field(default_factory=dict)
    corrections: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    #: fail-open / skip discriminator (``digest_overflow`` / a transport /
    #: parse / length failure token) — ``None`` on a clean review.
    outcome: Optional[str] = None


def build_review_messages(
    digest: str, n_headings: int, n_relevelable: int
) -> List[Dict[str, str]]:
    thinking_line, effort_line = resolve_reasoning_effort()
    system = thinking_line + "\n" + _REVIEW_INSTRUCTIONS
    user = (
        (effort_line + "\n\n" if effort_line else "")
        + f"Whole-document judged heading skeleton ({n_headings} headings, "
        f"{n_relevelable} revisable):\n\n"
        f"{digest}\n\n"
        "Return the JSON now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _heading_is_relevelable(r: Dict[str, Any]) -> bool:
    """A heading the FINAL REVIEW may re-level: one the judge owns (judged this
    run OR still pending). A FIXED anchor (the N.M spine / synthesized chapter
    opener / doc title / section_number_recovered) carries NEITHER key, so the
    review can never touch it (a change to one is dropped)."""
    return ("heading_level_judged" in r) or bool(r.get("heading_level_pending"))


def _build_review_entries(
    region_provenance: Sequence[Dict[str, Any]],
) -> List[SkeletonEntry]:
    """Whole-document heading skeleton of the JUDGED tree — every heading with
    its now-assigned level; ``pending`` marks the re-levelable headings ("*")
    vs the fixed anchors ("."). Rendered by the SAME ``_render_digest``."""
    entries: List[SkeletonEntry] = []
    for r in region_provenance:
        if r.get("region_kind") != "heading":
            continue
        ridx = r.get("first_raw_block_index")
        if ridx is None:
            continue
        try:
            level = int(r.get("level", 3) or 3)
        except (TypeError, ValueError):
            level = 3
        page = r.get("source_page")
        if page is None:
            pages = r.get("pages") or []
            page = pages[0] if pages else None
        entries.append(SkeletonEntry(
            int(ridx), level, str(r.get("heading_text") or ""), page,
            _heading_is_relevelable(r), None))
    return entries


def _levels_to_int_map(levels: Dict[str, Any]) -> Dict[int, int]:
    """Coerce a ``{"<id>": <lvl>}`` verdict to ``{int: int}`` (skip garbage)."""
    out: Dict[int, int] = {}
    for k, v in (levels or {}).items():
        try:
            out[int(str(k).strip())] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _review_cache_key(digest: str, model: str, max_tokens: int) -> str:
    """Content-addressed key for the review verdict — distinct from a window
    key (the ``review`` salt) so a review verdict is never confused with a
    per-window one. Resume-safe (a killed / stopped review resumes free)."""
    raw = (
        f"{hashlib.sha256(digest.encode('utf-8')).hexdigest()}|{model}|"
        f"{JUDGE_PROMPT_VERSION}|{max_tokens}|0.0|review|"
        f"{'/'.join(_THINKING_DIRECTIVE)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _count_preserved_legitimate(
    headings: Sequence[Dict[str, Any]],
    relevelable_by_id: Dict[int, Dict[str, Any]],
    review_map: Dict[int, int],
) -> int:
    """Recurring same-title headings that occur at >1 distinct level AND that
    the review did NOT change — the same-title-DIFFERENT-depth pairs correctly
    left alone (the legitimate-depth preservation the review must not flatten).

    Reuses ``heading_judge_audit.normalize_signature`` (the exact recurring
    signature the Arm-C consistency detector reports)."""
    from collections import defaultdict

    sig_levels: Dict[str, set] = defaultdict(set)
    sig_members: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in headings:
        sig = normalize_signature(r.get("heading_text"))
        if not sig:
            continue
        try:
            lvl = int(r.get("level", 3) or 3)
        except (TypeError, ValueError):
            lvl = 3
        sig_levels[sig].add(lvl)
        sig_members[sig].append(r)
    changed_ids = set(review_map)
    preserved = 0
    for sig, levels in sig_levels.items():
        if len(levels) <= 1:
            continue  # not a recurring-at-multiple-levels signature
        for r in sig_members[sig]:
            rid = r.get("first_raw_block_index")
            if rid is None:
                continue
            if int(rid) in relevelable_by_id and int(rid) not in changed_ids:
                preserved += 1
    return preserved


def apply_review_levels(
    region_provenance: List[Dict[str, Any]],
    heading_tree: List[Any],
    escalations: List[Dict[str, Any]],
    review_map: Dict[int, int],
) -> ReviewResult:
    """Apply the FINAL-REVIEW verdict — a RE-LEVEL of already-judged headings —
    under the SAME clamp invariants as ``apply_judged_levels`` (level range
    [2,6], parent+1 max jump, no-orphaning) BUT allowing a previously-judged
    heading to move. Differences from the initial judge:

    * A proposed change that would VIOLATE an invariant is DROPPED individually
      (never clamped-in-place — never-ship-worse), tallied in ``dropped``.
    * Only ``heading_level_judged`` / ``heading_level_pending`` headings are
      re-levelable; a change to a FIXED anchor is dropped.
    * An empty ``review_map`` → NO mutation (byte-identical fail-open).

    Rebuilds ``heading_tree`` when any level moved."""
    result = ReviewResult()
    result.proposed = len(review_map)
    prov = region_provenance
    headings = [r for r in prov if r.get("region_kind") == "heading"]

    relevelable_by_id: Dict[int, Dict[str, Any]] = {}
    for r in headings:
        if _heading_is_relevelable(r):
            rid = r.get("first_raw_block_index")
            if rid is not None:
                relevelable_by_id[int(rid)] = r

    # preserved-legitimate is measured over the PRE-mutation tree.
    result.preserved_legitimate = _count_preserved_legitimate(
        headings, relevelable_by_id, review_map)

    if not review_map:
        return result

    # Pre-filter: unknown / fixed-anchor / out-of-range / caption-label → drop.
    candidates: Dict[int, int] = {}
    for rid, lvl in review_map.items():
        try:
            irid = int(rid)
        except (TypeError, ValueError):
            result.dropped += 1
            continue
        if irid not in relevelable_by_id:
            result.dropped += 1  # fixed anchor or unknown id
            continue
        try:
            ilvl = int(lvl)
        except (TypeError, ValueError):
            result.dropped += 1
            continue
        if ilvl < 2 or ilvl > 6:
            result.dropped += 1
            continue
        if ilvl <= 3 and rm.caption_label_kind(
                str(relevelable_by_id[irid].get("heading_text") or "")) is not None:
            result.dropped += 1
            continue
        candidates[irid] = ilvl

    if not candidates:
        return result

    # Effective-level walk in document order; a candidate that would violate
    # parent+1 / no-orphaning is DROPPED (kept at its current level), not
    # clamped — the review never ships a worse level than the judge produced.
    prev_effective = 1
    enclosing_anchor = 1
    corrections: Dict[int, Tuple[int, int]] = {}
    for h in headings:
        rid = h.get("first_raw_block_index")
        relevel = _heading_is_relevelable(h)
        try:
            cur_level = int(h.get("level", 3) or 3)
        except (TypeError, ValueError):
            cur_level = 3
        if relevel and rid is not None and int(rid) in candidates:
            proposed = candidates[int(rid)]
            lo = enclosing_anchor + 1   # rule 4: no orphaning
            hi = prev_effective + 1     # rule 3: parent+1 max jump
            if proposed < lo or proposed > hi:
                result.dropped += 1     # invariant violation → drop, keep level
                eff = cur_level
            elif proposed != cur_level:
                eff = proposed
                corrections[int(rid)] = (cur_level, eff)
            else:
                eff = cur_level          # valid but a no-op (already correct)
            prev_effective = eff
        else:
            eff = cur_level
            prev_effective = eff
            if not relevel:
                enclosing_anchor = eff   # only fixed anchors set the orphan floor

    for rid, (from_lvl, to_lvl) in corrections.items():
        region = relevelable_by_id[rid]
        region["level"] = to_lvl
        region.pop("heading_level_pending", None)
        region["heading_level_reviewed"] = {"from": from_lvl, "to": to_lvl}
        escalations.append({
            "region_index": rid,
            "source_page": region.get("source_page"),
            "native_label": region.get("native_label", ""),
            "reason": "heading_level_reviewed",
            "detail": (f"final-review reconciled heading level "
                       f"{from_lvl}->{to_lvl}"),
            "text": str(region.get("heading_text") or "")[:120],
        })
        result.applied += 1
        result.changed += 1
        key = f"{from_lvl}->{to_lvl}"
        result.transitions[key] = result.transitions.get(key, 0) + 1
    result.corrections = corrections

    if corrections:
        heading_tree[:] = [
            (int(p.get("level", 3) or 3), str(p.get("heading_text", "")))
            for p in prov if p.get("region_kind") == "heading"
        ]
    return result


def resolve_final_review_mode() -> bool:
    """Phase B master gate: run the whole-document consistency-reconciliation
    FINAL REVIEW after judging (``SEMANTIK_HEADING_JUDGE_FINAL_REVIEW``, default
    OFF, truthy-set parse-with-fallback). OFF → byte-identical (no review pass,
    no extra POST)."""
    return _truthy_env("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW")


_DEFAULT_FINAL_REVIEW_MIN_CHAPTERS = 2


def resolve_final_review_min_chapters() -> int:
    """Minimum distinct chapter count for the whole-document FINAL REVIEW to run
    on a NORMALIZED plan (``SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS``,
    default 2). In normalized mode the per-chapter reviewer already reconciles
    each chapter, so the whole-doc review is pure redundancy for a single-chapter
    document (and — worse — a big single chapter FRAGMENTS into many large review
    windows). Below this threshold the whole-doc review is skipped; the
    chapter-reviewer verdicts stand. Positive int parse-with-fallback: blank /
    garbage / non-positive → the default 2."""
    return _resolve_pos_int_env("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS",
                                _DEFAULT_FINAL_REVIEW_MIN_CHAPTERS)


_REVIEW_PER_CHAPTER_HEADER = (
    "PER-CHAPTER STRUCTURE (judge cross-chapter consistency — the SAME section "
    "title should sit at the SAME level across chapters unless its structural "
    "position genuinely differs):"
)
_REVIEW_WHOLE_DOC_HEADER = "WHOLE-DOCUMENT OUTLINE (the full structure by itself):"


def _build_review_digest(
    region_provenance: Sequence[Dict[str, Any]],
    entries: Sequence[SkeletonEntry],
) -> str:
    """The FINAL-REVIEW digest. On a MULTI-CHAPTER document (or under chapter
    mode) the reviewer receives BOTH the per-chapter judged skeletons AND the
    whole-document skeleton by itself, optionally prefixed with the derived
    document-schema preamble (Part 5), so it can reconcile cross-chapter
    consistency + whole-doc coherence. A single-chapter document with no schema
    → the legacy whole-document digest (byte-identical)."""
    whole = _render_digest(list(entries), anchors=False)
    schema = build_document_schema(region_provenance) if resolve_doc_schema_mode() else ""
    chapters = segment_into_chapters(region_provenance)
    if len(chapters) <= 1 and not schema:
        return whole
    parts: List[str] = []
    if schema:
        parts.append(schema)
    if len(chapters) > 1:
        parts.append(_REVIEW_PER_CHAPTER_HEADER)
        for i, ch in enumerate(chapters, 1):
            ch_entries = _build_review_entries(ch.regions)
            if not ch_entries:
                continue
            parts.append(f"Chapter {i}:\n" + _render_digest(ch_entries, anchors=False))
        parts.append(_REVIEW_WHOLE_DOC_HEADER + "\n" + whole)
    else:
        parts.append(whole)
    return "\n\n".join(parts)


def _judge_review(
    digest: str,
    n_headings: int,
    n_relevelable: int,
    max_tokens: int,
    *,
    post_fn: Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]],
    model: str,
    use_cache: bool,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """ONE review POST → ``{region_id: proposed_level}`` + meta. FAIL-OPEN: any
    transport / length-exhaust / parse failure returns an EMPTY map (the caller
    keeps the pre-review levels byte-for-byte)."""
    meta: Dict[str, Any] = {
        "posts": 0, "cache_hit": False, "finish": None, "outcome": None,
    }
    key = _review_cache_key(digest, model, max_tokens) if use_cache else None
    if key is not None:
        cached = _cache_get(key)
        if cached is not None:
            meta["cache_hit"] = True
            return _levels_to_int_map(cached.get("levels") or {}), meta
    messages = build_review_messages(digest, n_headings, n_relevelable)
    try:
        content, finish = post_fn(messages, max_tokens)
    except _JudgeTransportError as exc:
        meta["outcome"] = "transport_failure"
        logger.warning(
            "heading-judge final review: transport failure — keeping pre-review "
            "levels (fail-open): %s", exc)
        return {}, meta
    meta["posts"] = 1
    meta["finish"] = finish
    if finish == "length":
        meta["outcome"] = "length_exhausted"
        logger.warning(
            "heading-judge final review: completion exhausted (finish=length) — "
            "keeping pre-review levels (never-ship-worse)")
        return {}, meta
    parsed = parse_judge_response(content)
    if parsed is None:
        meta["outcome"] = "parse_failure"
        logger.warning(
            "heading-judge final review: reply did not parse as a JSON verdict "
            "— keeping pre-review levels (fail-open)")
        return {}, meta
    review_map = _levels_to_int_map(parsed.get("levels") or {})
    if key is not None:
        _cache_put(key, {"levels": parsed.get("levels") or {}})
    return review_map, meta


def _emit_review_capture(
    *,
    model: str,
    result: ReviewResult,
    meta: Dict[str, Any],
    course_code: Optional[str] = None,
) -> None:
    """One ``structure_review`` DecisionCapture for the FINAL REVIEW pass with a
    ``final_review`` metadata discriminator (reuses the existing enum, no schema
    change). Best-effort — a capture failure never breaks the review."""
    try:
        from lib.decision_capture import DecisionCapture
    except Exception as exc:  # noqa: BLE001 — the cross-venv bridge has no lib/
        logger.debug("heading-judge final review: DecisionCapture unavailable "
                     "(non-fatal): %s", exc)
        return
    try:
        cap = DecisionCapture(
            course_code=course_code or _detect_course_code(),
            phase="semantik_conversion",
            tool="semantik",
        )
        cap.log_decision(
            decision_type="structure_review",
            decision=(
                f"final_review: reconciled {result.applied} of "
                f"{result.proposed} proposed cross-document heading-level "
                f"consistency change(s) ({result.dropped} dropped, "
                f"{result.preserved_legitimate} legitimate depth difference(s) "
                f"preserved)"
            ),
            rationale=(
                f"Whole-document heading-consistency FINAL REVIEW (model={model}) "
                f"reconciled recurring-section-level inconsistencies over the "
                f"JUDGED tree; the deterministic clamp DECIDED — proposed="
                f"{result.proposed}, applied={result.applied} "
                f"({_format_transitions(result.transitions) or 'none'}), "
                f"dropped={result.dropped} (fixed-anchor / out-of-range / "
                f"caption / would-violate parent+1 or no-orphan), "
                f"preserved_legitimate={result.preserved_legitimate} "
                f"(same-title-DIFFERENT-depth pairs left alone, never "
                f"flattened); outcome={meta.get('outcome') or 'ok'}, "
                f"cache={'hit' if meta.get('cache_hit') else 'miss'}, "
                f"posts={meta.get('posts', 0)}, finish={meta.get('finish')}."
            ),
            alternatives_considered=[
                {
                    "option": "Keep all judged levels without final reconciliation",
                    "reason_rejected": (
                        f"Rejected because {result.applied} of {result.proposed} "
                        "review proposals safely reconciled recurring headings."
                    ),
                },
                {
                    "option": "Flatten every repeated heading title to one level",
                    "reason_rejected": (
                        f"Rejected because {result.preserved_legitimate} "
                        "repeated-title headings occur at legitimate different "
                        "depths."
                    ),
                },
            ],
            final_review=True,
            fr_proposed=result.proposed,
            fr_applied=result.applied,
            fr_dropped=result.dropped,
            fr_preserved_legitimate=result.preserved_legitimate,
            fr_model=model,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, never fatal
        logger.debug("heading-judge final review: decision capture failed "
                     "(non-fatal): %s", exc)


def run_final_review(
    region_provenance: List[Dict[str, Any]],
    heading_tree: List[Any],
    escalations: List[Dict[str, Any]],
    *,
    seat: Optional[HeadingJudgeSeat] = None,
    post_fn: Optional[Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]]] = None,
    use_cache: Optional[bool] = None,
    model: Optional[str] = None,
    course_code: Optional[str] = None,
    emit_capture: bool = True,
) -> Optional[Dict[str, Any]]:
    """Phase B: whole-document consistency-reconciliation FINAL REVIEW over the
    JUDGED tree, applied IN PLACE. Returns ``None`` when the gate is off
    (byte-identical) or a report dict otherwise.

    FAIL-OPEN + never-ship-worse: any transport/parse/length failure or a
    wholesale-invalid review keeps the pre-review judged levels byte-for-byte. A
    whole judged tree too big to fit ``resolve_digest_budget_tokens()`` in one
    review window is SKIPPED with a loud log (v1 is single-window review — never
    a silent under-review)."""
    if not resolve_final_review_mode():
        return None
    model = model or (seat.model if seat else resolve_heading_judge_model())
    if use_cache is None:
        use_cache = resolve_heading_judge_checkpoint()
    if post_fn is None:
        post_fn = _make_default_post_fn(seat or resolve_heading_judge_seat())

    entries = _build_review_entries(region_provenance)
    relevelable = [e for e in entries if e.pending]
    base_report: Dict[str, Any] = {
        "proposed": 0, "applied": 0, "dropped": 0,
        "preserved_legitimate": 0, "changed": 0, "transitions": {},
        "windows": 1,
    }
    if not relevelable:
        base_report["skipped"] = "no_relevelable_headings"
        return base_report

    digest = _build_review_digest(region_provenance, entries)
    budget = resolve_digest_budget_tokens()
    if _estimate_tokens(digest) > budget:
        logger.warning(
            "heading-judge final review: whole-document judged tree (~%d tokens) "
            "exceeds the single-review digest budget (%d) — SKIPPING the final "
            "review (v1 is single-window; judged levels kept as-is, NOT silently "
            "under-reviewed)",
            _estimate_tokens(digest), budget)
        base_report["skipped"] = "digest_overflow"
        return base_report

    review_map, rmeta = _judge_review(
        digest, len(entries), len(relevelable), _max_tokens_for(len(relevelable)),
        post_fn=post_fn, model=model, use_cache=use_cache)
    result = apply_review_levels(
        region_provenance, heading_tree, escalations, review_map)
    result.outcome = rmeta.get("outcome")

    if emit_capture:
        _emit_review_capture(model=model, result=result, meta=rmeta,
                             course_code=course_code)

    return {
        "proposed": result.proposed,
        "applied": result.applied,
        "dropped": result.dropped,
        "preserved_legitimate": result.preserved_legitimate,
        "changed": result.changed,
        "transitions": dict(result.transitions),
        "corrections": {str(k): v for k, v in result.corrections.items()},
        "outcome": result.outcome,
        "windows": 1,
        "meta": rmeta,
    }


# ── NORMALIZED reduce: skeleton-only chapter reviewer + deterministic fallback.
def _deterministic_consolidate(
    slice_infos: Sequence[Dict[str, Any]],
) -> Dict[int, int]:
    """Interior-slice-wins consolidation of per-slice verdicts (the reviewer-off
    / fail-open path). For a pending judged by MULTIPLE slices, keep the verdict
    from the slice where the heading sits MOST INTERIOR (max min-distance to
    either slice edge, measured in headings); ties → the EARLIER slice. A pending
    judged once keeps that verdict. ALWAYS complete over every judged pending —
    never leaves a conflict unresolved."""
    best: Dict[int, Tuple[Tuple[int, int], int]] = {}
    for si, info in enumerate(slice_infos):
        heading_ids: List[int] = list(info.get("heading_ids") or [])
        pos_of = {rid: p for p, rid in enumerate(heading_ids)}
        h = len(heading_ids)
        for rid, lvl in (info.get("verdicts") or {}).items():
            p = pos_of.get(int(rid))
            dist = min(p, h - 1 - p) if (p is not None and h > 0) else 0
            cand = (dist, -si)  # max dist wins; tie → earlier (smaller) si
            if int(rid) not in best or cand > best[int(rid)][0]:
                best[int(rid)] = (cand, int(lvl))
    return {rid: lvl for rid, (_c, lvl) in best.items()}


def run_chapter_review(
    region_provenance: Sequence[Dict[str, Any]],
    chapter_entries: Sequence[SkeletonEntry],
    slice_infos: Sequence[Dict[str, Any]],
    *,
    post_fn: Optional[Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]]] = None,
    model: Optional[str] = None,
    use_cache: Optional[bool] = None,
) -> Dict[int, int]:
    """Skeleton-only CHAPTER reviewer — the normalized-window REDUCE for ONE
    split chapter: consolidate its per-slice verdicts (including the CONFLICTING
    double-judgments of overlap headings) into ONE level per pending.

    The deterministic interior-slice-wins consolidation is computed FIRST and is
    the guaranteed-complete result. When ``SEMANTIK_HEADING_JUDGE_CHAPTER_REVIEW``
    is on (default while NORMALIZE is on) a skeleton-only (no content — cheap)
    reviewer POST may refine it; ANY reviewer-off / transport / parse / length /
    unparseable failure keeps the deterministic consolidation, so the returned
    map is ALWAYS complete and every overlap conflict is resolved."""
    det = _deterministic_consolidate(slice_infos)
    if not resolve_chapter_review_mode():
        return det
    relevelable = [e for e in chapter_entries if e.pending]
    if not relevelable:
        return det
    if post_fn is None:
        post_fn = _make_default_post_fn(resolve_heading_judge_seat())
    model = model or resolve_heading_judge_model()

    # Skeleton-only digest of the chapter with the deterministic levels applied
    # (no content — the reviewer only reconciles cross-slice level consistency).
    ents = [
        SkeletonEntry(
            e.region_index,
            det.get(e.region_index, e.level) if e.pending else e.level,
            e.text, e.source_page, e.pending, None)
        for e in chapter_entries
    ]
    digest = _render_digest(ents, anchors=False)
    messages = build_review_messages(digest, len(list(chapter_entries)),
                                     len(relevelable))
    try:
        content, finish = post_fn(messages, _max_tokens_for(len(relevelable)))
    except _JudgeTransportError as exc:
        logger.warning("heading-judge chapter review: transport failure — "
                       "keeping deterministic consolidation (fail-open): %s", exc)
        return det
    if finish == "length":
        logger.warning("heading-judge chapter review: completion exhausted "
                       "(finish=length) — keeping deterministic consolidation")
        return det
    parsed = parse_judge_response(content)
    if parsed is None:
        logger.warning("heading-judge chapter review: reply did not parse — "
                       "keeping deterministic consolidation (fail-open)")
        return det
    reviewed = _levels_to_int_map(parsed.get("levels") or {})
    # Reviewer verdicts win for the pendings it named; every other pending keeps
    # the deterministic consolidation (the map stays complete + conflict-free).
    out = dict(det)
    pending_set = {e.region_index for e in relevelable}
    for rid, lvl in reviewed.items():
        if rid in pending_set:
            out[rid] = lvl
    return out


def _consolidate_slice_verdicts(
    plan: SkeletonPlan,
    meta: Dict[str, Any],
    region_provenance: Sequence[Dict[str, Any]],
    *,
    post_fn: Optional[Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]]],
    seat: Optional[HeadingJudgeSeat],
    use_cache: Optional[bool],
) -> Dict[int, int]:
    """The normalized-window CHAPTER REDUCE: group slice-windows by chapter, pass
    NON-split chapters straight through, and reconcile each SPLIT chapter's slice
    verdicts (via :func:`run_chapter_review` — reviewer or the deterministic
    fallback) into one verdict per pending. Records ``normalized`` / ``W`` /
    ``chapter_reviews`` / per-chapter telemetry on ``meta``."""
    from collections import OrderedDict

    slices = plan.chapter_slices or []
    window_verdicts = meta.get("window_verdicts") or []
    model = seat.model if seat else resolve_heading_judge_model()

    if len(slices) != len(window_verdicts):
        # Alignment guard — every normalized window has >= 1 pending so
        # active == plan.windows; if it ever slips, fall back to the merged map.
        logger.warning("heading-judge normalized: slice/verdict misalignment "
                       "(%d != %d) — using the merged verdict map",
                       len(slices), len(window_verdicts))
        merged: Dict[int, int] = {}
        for _wp, wv in window_verdicts:
            merged.update(wv)
        meta["normalized"] = True
        return merged

    groups: "OrderedDict[int, List[int]]" = OrderedDict()
    for widx, cs in enumerate(slices):
        groups.setdefault(int(cs["chapter_id"]), []).append(widx)

    chapters = segment_into_chapters(region_provenance)
    consolidated: Dict[int, int] = {}
    n_reviews = 0
    ch_telemetry: List[Dict[str, Any]] = []
    for chid, widxs in groups.items():
        if len(widxs) == 1:
            _wp, wv = window_verdicts[widxs[0]]
            consolidated.update(wv)
            ch_telemetry.append({"chapter_id": chid, "n_slices": 1,
                                 "split": False})
            continue
        slice_infos = []
        for widx in widxs:
            cs = slices[widx]
            _wp, wv = window_verdicts[widx]
            slice_infos.append({
                "heading_ids": cs.get("heading_ids") or [],
                "verdicts": dict(wv),
                "core_pending_ids": cs.get("core_pending_ids") or [],
                "overlap_pending_ids": cs.get("overlap_pending_ids") or [],
            })
        ch_regions = (chapters[chid].regions
                      if 0 <= chid < len(chapters) else [])
        chapter_entries = _build_all_entries(ch_regions, anchors=False)
        cons = run_chapter_review(
            region_provenance, chapter_entries, slice_infos,
            post_fn=post_fn, model=model, use_cache=use_cache)
        consolidated.update(cons)
        n_reviews += 1
        ch_telemetry.append({"chapter_id": chid, "n_slices": len(widxs),
                             "split": True})

    meta["normalized"] = True
    meta["W"] = slices[0].get("W") if slices else None
    meta["chapter_reviews"] = n_reviews
    meta["chapters"] = ch_telemetry
    return consolidated


# ── The one public seam the lane calls. ──────────────────────────────────────
def run_heading_judge(
    region_provenance: List[Dict[str, Any]],
    heading_tree: List[Any],
    escalations: List[Dict[str, Any]],
    *,
    seat: Optional[HeadingJudgeSeat] = None,
    post_fn: Optional[Callable[[Sequence[Dict[str, str]], int], Tuple[Optional[str], Optional[str]]]] = None,
    use_cache: Optional[bool] = None,
    course_code: Optional[str] = None,
    emit_capture: bool = True,
) -> Dict[str, Any]:
    """Build the skeleton, judge pending levels, apply the clamp rules IN PLACE.

    Mutates ``region_provenance`` / ``heading_tree`` / ``escalations``. Returns a
    report dict. A no-pending chapter is a natural no-op (no POST)."""
    plan = build_heading_skeleton(region_provenance)
    n_pending = len(plan.pending_ids)
    if n_pending == 0:
        return {"n_pending": 0, "applied": 0, "clamped": 0, "dropped": 0,
                "kept": 0, "kept_judged": 0, "unjudged": 0,
                "changed": 0, "agreed": 0, "transitions": {},
                "windows": len(plan.windows), "digest": plan.digest}

    verdict_map, meta = judge_heading_levels(
        plan, seat=seat, post_fn=post_fn, use_cache=use_cache)
    # NORMALIZED-window REDUCE (SEMANTIK_HEADING_JUDGE_NORMALIZE): reconcile each
    # SPLIT chapter's overlapping slice verdicts (chapter reviewer / deterministic
    # interior-slice-wins fallback) into one verdict per pending BEFORE the clamp
    # apply. None on every non-normalized path → the merged map is used as-is.
    if plan.chapter_slices is not None:
        verdict_map = _consolidate_slice_verdicts(
            plan, meta, region_provenance,
            post_fn=post_fn, seat=seat, use_cache=use_cache)
    result = apply_judged_levels(
        region_provenance, heading_tree, escalations, verdict_map)

    if emit_capture:
        _emit_judge_capture(
            model=(seat.model if seat else resolve_heading_judge_model()),
            n_pending=n_pending, result=result, meta=meta, course_code=course_code)

    # Phase B (SEMANTIK_HEADING_JUDGE_FINAL_REVIEW, default OFF): whole-document
    # consistency reconciliation over the just-judged tree. Returns None when
    # the gate is off (byte-identical) — the same post_fn/seat/cache is reused.
    #
    # NORMALIZED-plan redundancy skip: the per-chapter reviewer already
    # reconciled each chapter, so the whole-doc review is pure redundancy for a
    # single-chapter document (and a big single chapter FRAGMENTS into many
    # large review windows). Skip it below the distinct-chapter threshold; this
    # never touches the non-normalized path (chapter_slices is None → no skip)
    # and is byte-identical when the FINAL_REVIEW gate is off (run_final_review
    # short-circuits to None either way).
    final_review = None
    _skip_final_review = False
    if plan.chapter_slices is not None:
        distinct_chapters = len({cs.get("chapter_id") for cs in plan.chapter_slices})
        if distinct_chapters < resolve_final_review_min_chapters():
            _skip_final_review = True
    if not _skip_final_review:
        final_review = run_final_review(
            region_provenance, heading_tree, escalations,
            seat=seat, post_fn=post_fn, use_cache=use_cache,
            model=(seat.model if seat else resolve_heading_judge_model()),
            course_code=course_code, emit_capture=emit_capture)

    return {
        "final_review": final_review,
        # NORMALIZED-window telemetry (meta carries W / chapter_reviews /
        # per-chapter split flags); False on every non-normalized path.
        "normalized": bool(meta.get("normalized")),
        "n_pending": n_pending,
        # ``applied`` = verdicts RECORDED (unchanged field name / meaning);
        # ``changed`` = the subset that actually MOVED a level (the only one
        # that can alter a rendered <hN>), ``agreed`` = the rest.
        "applied": result.applied,
        "changed": result.changed,
        "agreed": result.agreed,
        "transitions": dict(result.transitions),
        "clamped": result.clamped,
        "dropped": result.dropped,
        "kept": result.kept,
        "kept_judged": result.kept_judged,
        "unjudged": result.unjudged,
        # An unjudged chapter carries its mechanism at the top level, not only
        # inside ``meta``.
        "failure_modes": list(meta.get("failure_modes") or []),
        "unjudged_reason": meta.get("unjudged_reason"),
        "windows": len(plan.windows),
        "corrections": {str(k): v for k, v in result.corrections.items()},
        "meta": meta,
        "digest": plan.digest,
    }


__all__ = [
    "JUDGE_PROMPT_VERSION",
    "SkeletonEntry",
    "SkeletonPlan",
    "ApplyResult",
    "HeadingJudgeSeat",
    "build_heading_skeleton",
    "build_judge_messages",
    "Chapter",
    "segment_into_chapters",
    "build_document_schema",
    "resolve_chapter_mode",
    "resolve_doc_schema_mode",
    "resolve_chapter_content_head_words",
    "resolve_normalize_mode",
    "resolve_normalize_percentile",
    "resolve_normalize_window_min",
    "resolve_slice_overlap",
    "resolve_chapter_review_mode",
    "resolve_normalized_window_tokens",
    "run_chapter_review",
    "resolve_heading_judge_seat",
    "resolve_heading_judge_max_split_depth",
    "resolve_pending_window_cap",
    "resolve_max_tokens_ceiling",
    "resolve_max_tokens_floor",
    "resolve_heading_judge_est_per_judgment",
    "resolve_max_tokens_ceiling_thinkoff",
    "resolve_max_tokens_floor_thinkoff",
    "resolve_est_per_judgment_thinkoff",
    "resolve_heading_judge_frequency_penalty",
    "resolve_heading_judge_enable_thinking",
    "parse_judge_response",
    "flatten_sibling_staircases",
    "judge_heading_levels",
    "apply_judged_levels",
    "run_heading_judge",
    "resolve_fulldoc_context_mode",
    "resolve_fulldoc_anchors_mode",
    "resolve_final_review_mode",
    "resolve_final_review_min_chapters",
    "resolve_tokenizer_mode",
    "resolve_tokenizer_id",
    "ReviewResult",
    "build_review_messages",
    "apply_review_levels",
    "run_final_review",
]


if __name__ == "__main__":  # pragma: no cover — delegate to the standalone runner
    from .heading_judge_standalone import main

    raise SystemExit(main())
