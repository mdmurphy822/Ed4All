"""Prompt templates + version constant for the grounded-answer composer.

The answer model is a local 7B/14B-Q4 instruct model served over an
OpenAI-compatible endpoint. Such models attend best to a SHORT system
preamble plus a trailing JSON directive (the most-respected prompt
position); long preambles drift. The templates here follow that
established repo precedent (module-level system-prompt constants +
an explicit version string, as in the synthesis providers and the
Courseforge rewrite-form-data drafting path).

``ANSWER_PROMPT_VERSION`` is bumped MANUALLY on any wording change.
The bump is a cache/eval comparability boundary: it is recorded in every
decision-capture event AND in the composed-answer payload so a post-hoc
audit knows which prompt produced which answer.

Context-budget posture (token-aware, math-safe)
------------------------------------------------
Ollama (the common local backend) serves a fixed ``num_ctx`` window
(default 4096) and SILENTLY truncates the prompt HEAD when the rendered
prompt exceeds it. A char-based budget that assumes ~4 chars/token
under-counts on dense math/symbol corpora (~2.5-2.9 chars/token), so a
"12000-char" budget produces a 4000-4500-token prompt that overflows a
4096-window — the system prompt + leading passage IDs vanish and the
model fabricates citations. The remediation retry then APPENDS tokens,
truncating MORE each attempt.

The budget below is therefore TOKEN-based, sized off a configurable
serving window (``ED4ALL_ANSWER_NUM_CTX``, default 4096) with a
math-safe ``CHARS_PER_TOKEN`` divisor (2.5) and explicit headroom for
the system prompt, the question, the trailing/remediation directives
(which enumerate the allowed citation ids), and the generation budget
(``max_tokens``). Trailing (lowest-ranked) passages are dropped whole —
the question is never truncated. ``answer_backend`` additionally runs a
post-call truncation tripwire that compares the server-reported
``usage.prompt_tokens`` against the local estimate and fails closed when
the head was silently truncated.
"""
from __future__ import annotations

import math
import os
from typing import List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycle
    from lib.retrieval.answer_composer import RetrievedPassage


# Bump on ANY change to the system/user prompt wording below.
ANSWER_PROMPT_VERSION = "ws3.v2"

# Per-passage hard truncation (characters). A 14B-Q4 model loses the
# trailing JSON directive when the context balloons; capping each
# passage keeps the most-relevant heads in window.
PASSAGE_CHAR_CAP = 1500

# Math-safe chars-per-token divisor. Dense math/symbol corpora tokenize
# at ~2.5-2.9 chars/token (vs ~4 for prose); a too-generous divisor is
# exactly the under-count that overflows the serving window and gets the
# prompt head silently truncated. 2.5 is the conservative (low) end so
# the estimate is an UPPER bound on prompt tokens — we'd rather drop a
# trailing passage than overflow.
CHARS_PER_TOKEN = 2.5

# Tokens reserved on top of the passage context: the system prompt, the
# question, the trailing JSON directive (which enumerates the allowed
# ids), and remediation-retry headroom (the remediation directive
# re-states the full allowed set, appended on a retry). Generous because
# the estimate is rough and the head-truncation failure mode is severe.
RESERVE_TOKENS = 200

# Default Ollama serving window. Honest about the common local default:
# `ollama` ships num_ctx=4096 unless the operator overrides it (Modelfile
# PARAMETER num_ctx, or OLLAMA_CONTEXT_LENGTH). Drives the context budget.
DEFAULT_NUM_CTX = 4096

# Env override for the serving window. Set this to the model server's
# ACTUAL num_ctx (e.g. 8192 for a long-context Modelfile) so the budget
# uses the real window instead of the conservative 4096 default.
ENV_ANSWER_NUM_CTX = "ED4ALL_ANSWER_NUM_CTX"

# Legacy char budget — kept as a module symbol for back-compat with
# callers/tests that import it, but no longer the primary gate. The
# token budget below supersedes it.
MAX_CONTEXT_CHARS = 12000


# Terse per the Wave-114 local-model precedent (< 80 words; trailing
# JSON directive is the load-bearing part). Passage-constrained
# answering + explicit refusal instruction.
ANSWER_SYSTEM_PROMPT = (
    "You answer a student's question about one course using ONLY the "
    "numbered source passages provided. Do not use outside knowledge. "
    "If the question has multiple parts, address every part the passages "
    "support; for any part NOT covered by the passages, say so in one short "
    "sentence (e.g. \"The provided course material does not cover <part>.\") "
    "instead of omitting it silently, and never invent material for an "
    "uncovered part. If NO part of the question is covered, set "
    '"not_in_course" to true and leave "answer" empty. Cite the id of '
    "every passage that supports the answer. Output JSON only: "
    '{"answer": "...", "citations": ["<passage_id>", ...], '
    '"not_in_course": false}.'
)


# Appended to the user prompt after one or more bad citation ids come
# back; drives a single remediation retry. ``{bad_ids}`` is the
# comma-joined offending ids; ``{allowed_ids}`` is the FULL allowed set.
# Re-stating the full allowed set (not just the bad ids) is deliberate:
# naming only the offenders yields format-compliance but not
# set-compliance — the model keeps inventing fresh ids. The allowed-set
# enumeration is the load-bearing constraint and survives in the tail.
CITATION_REMEDIATION_DIRECTIVE = (
    "Your previous reply cited passage ids that were not in the provided "
    "list: {bad_ids}. The ONLY valid passage ids are: {allowed_ids}. "
    "Answer again citing ONLY ids from that list."
)


def _trailing_json_directive(allowed_ids: Sequence[str]) -> str:
    """Trailing directive that ENUMERATES the allowed citation ids.

    The tail is the most-respected position for small local models AND
    the position that survives a head-truncation, so the explicit
    allowed-id list lives here. ``allowed_ids`` are the ids of the
    passages actually included in the rendered prompt (over-budget
    passages are dropped, so their ids must NOT appear here).
    """
    if allowed_ids:
        ids = ", ".join(allowed_ids)
        allowed_clause = f"Valid citation ids: {ids}. "
    else:
        allowed_clause = ""
    return (
        f"{allowed_clause}"
        'Output JSON only: {"answer": "...", "citations": '
        '["<passage_id>", ...], "not_in_course": false}.'
    )


def resolve_num_ctx() -> int:
    """Resolve the serving-window token budget from the environment.

    ``ED4ALL_ANSWER_NUM_CTX`` overrides the :data:`DEFAULT_NUM_CTX`
    (4096) default. Garbage / non-positive values fall back to the
    default (a misconfigured window must never silently disable the
    budget).
    """
    raw = os.environ.get(ENV_ANSWER_NUM_CTX)
    if not raw:
        return DEFAULT_NUM_CTX
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_NUM_CTX
    return val if val > 0 else DEFAULT_NUM_CTX


def estimate_tokens(text: str) -> int:
    """Math-safe token estimate for ``text`` (upper-bound on prompt tokens).

    Divides char length by :data:`CHARS_PER_TOKEN` (2.5) and rounds up.
    Deliberately conservative (over-estimates) so the budget errs toward
    dropping a trailing passage rather than overflowing the window.
    """
    if not text:
        return 0
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


def context_token_budget(
    query: str,
    *,
    max_tokens: int,
    num_ctx: int,
    allowed_id_chars: int = 0,
) -> int:
    """Tokens available for the passage context after fixed costs.

    The serving window must hold: the system prompt, the passage
    context, the question, the trailing directive (allowed-id
    enumeration), remediation headroom, AND the generation budget
    (``max_tokens``). Everything except the passage context is fixed
    cost; this returns what remains for passages (>= 0).
    """
    fixed = (
        estimate_tokens(ANSWER_SYSTEM_PROMPT)
        + estimate_tokens(query)
        + estimate_tokens(_trailing_json_directive([]))
        # allowed-id enumeration appears in BOTH the trailing directive
        # and the remediation directive; budget it twice (worst case is a
        # remediation retry that carries both).
        + 2 * int(math.ceil(max(0, allowed_id_chars) / CHARS_PER_TOKEN))
        + RESERVE_TOKENS
        + max(0, int(max_tokens))
    )
    return max(0, int(num_ctx) - fixed)


def _passage_label(passage: "RetrievedPassage") -> str:
    """Human heading for a numbered block: section heading, else item path."""
    heading = getattr(passage, "section_heading", None)
    if heading:
        return str(heading)
    item_path = getattr(passage, "item_path", None) or ""
    return str(item_path)


def render_answer_user_prompt(
    query: str,
    passages: Sequence["RetrievedPassage"],
    *,
    max_tokens: int = 1024,
) -> str:
    """Render the numbered passage-context user prompt.

    Block shape (one per passage, ranked order)::

        [<chunk_id>] (<section_heading or item_path>)
        <text truncated to PASSAGE_CHAR_CAP>

    Total budget guarded by a TOKEN estimate sized off the serving
    window (``ED4ALL_ANSWER_NUM_CTX``) — trailing passages are dropped
    whole (never the question). The dropped-count + the actually-included
    id set are surfaced via :func:`render_answer_user_prompt_with_meta`.
    """
    text, _ = render_answer_user_prompt_with_meta(
        query, passages, max_tokens=max_tokens
    )
    return text


def render_answer_user_prompt_with_meta(
    query: str,
    passages: Sequence["RetrievedPassage"],
    *,
    max_tokens: int = 1024,
):
    """Like :func:`render_answer_user_prompt` but also returns metadata.

    Returns ``(prompt_text, meta)`` where ``meta`` carries
    ``{"included_ids": [...], "dropped_count": int, "context_chars": int,
    "context_tokens": int, "estimated_prompt_tokens": int,
    "num_ctx": int}`` so the composer can (a) interpolate the dropped
    count into its decision-capture rationale, (b) intersect the cited
    ids against the ids ACTUALLY included in the prompt, and (c) compare
    the local prompt-token estimate against the server-reported usage.

    Two-pass: pass 1 selects the included passages under the token budget
    so the allowed-id char total is known; pass 2 renders the trailing
    directive enumerating exactly those ids.
    """
    num_ctx = resolve_num_ctx()

    # ---- Pass 1: build candidate blocks, truncated per-passage. -------
    candidates: List[Tuple[str, str]] = []  # (chunk_id, block_text)
    for passage in passages:
        chunk_id = str(getattr(passage, "chunk_id", ""))
        body = str(getattr(passage, "text", "") or "")
        if len(body) > PASSAGE_CHAR_CAP:
            body = body[:PASSAGE_CHAR_CAP]
        block = f"[{chunk_id}] ({_passage_label(passage)})\n{body}"
        candidates.append((chunk_id, block))

    # The allowed-id enumeration cost depends on which ids survive, but
    # the per-id char cost is small and bounded; budget against ALL
    # candidate ids (an upper bound) so the selection is conservative and
    # the trailing/remediation directives always fit.
    allowed_id_chars = sum(len(cid) + 2 for cid, _ in candidates)
    token_budget = context_token_budget(
        query,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        allowed_id_chars=allowed_id_chars,
    )

    # ---- Select blocks under the token budget. ------------------------
    blocks: List[str] = []
    included_ids: List[str] = []
    context_chars = 0
    context_tokens = 0
    dropped = 0
    for chunk_id, block in candidates:
        block_tokens = estimate_tokens(block)
        # Always keep at least the first block (a single over-budget
        # passage is better than an empty context — the per-passage cap
        # already bounds it); drop trailing blocks once over budget.
        if context_tokens + block_tokens > token_budget and blocks:
            dropped += 1
            continue
        blocks.append(block)
        included_ids.append(chunk_id)
        context_chars += len(block)
        context_tokens += block_tokens

    # ---- Pass 2: render with the trailing directive over included ids -
    context = "\n\n".join(blocks)
    trailing = _trailing_json_directive(included_ids)
    prompt = (
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"{trailing}"
    )
    estimated_prompt_tokens = (
        estimate_tokens(ANSWER_SYSTEM_PROMPT) + estimate_tokens(prompt)
    )
    meta = {
        "included_ids": included_ids,
        "dropped_count": dropped,
        "context_chars": context_chars,
        "context_tokens": context_tokens,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "num_ctx": num_ctx,
    }
    return prompt, meta


__all__ = [
    "ANSWER_PROMPT_VERSION",
    "ANSWER_SYSTEM_PROMPT",
    "CITATION_REMEDIATION_DIRECTIVE",
    "PASSAGE_CHAR_CAP",
    "MAX_CONTEXT_CHARS",
    "CHARS_PER_TOKEN",
    "RESERVE_TOKENS",
    "DEFAULT_NUM_CTX",
    "ENV_ANSWER_NUM_CTX",
    "resolve_num_ctx",
    "estimate_tokens",
    "context_token_budget",
    "render_answer_user_prompt",
    "render_answer_user_prompt_with_meta",
]
