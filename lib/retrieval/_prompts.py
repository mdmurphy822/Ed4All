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
#
# NOTE: the grounded-answer MILESTONE_TARGETS (lib/retrieval/grounded_eval.py)
# and the refusal-policy pins were MEASURED under ws3.v2 — the ws3.v3 wording
# changed the model's answer-vs-refuse boundary (it now answers synthesis /
# applied questions the passages substantively support instead of over-refusing
# on literal-wording mismatches). The orchestrator MUST re-measure the gold +
# refusal-probe suites and re-pin MILESTONE_TARGETS / the refusal policy after
# any wording change lands; the constants are deliberately left untouched here.
#
# ws3.v3 → ws3.v4 (composer-completeness Phase A): the v3 boundary fixed
# over-refusal but the composer then UNDER-STATES and UNDER-CITES content it
# already has in context (96 expected key points — 28.8% — were retrieved but
# never stated, 4-shingle support 0.000 in 81/87 cases; 14 of 16 dropped
# multi-part clauses had their passage in context; the model cites only 1.63 of
# 8 retrieved passages). This is NOT a retrieval/budget problem — it is the
# composer prompt under-using in-context evidence. v4 adds three clauses, ALL
# scoped to SUPPORTED content so they cannot raise fabrication: (a) state EVERY
# point the passages support that is relevant to the question, not just the
# first/easiest; (b) answer EACH part/clause of a multi-part question
# explicitly; (c) cite EVERY passage that supports a stated claim, not only one
# (citation_resolution is already 1.0, so added citations resolve). Length is
# NOT the lever (r=-0.035 vs completeness) — v4 instructs completeness-of-
# supported-points, never "write longer". The MILESTONE_TARGETS pins
# (esp. unsupported_claim_rate ≤ 0.05 and answer_rate ≥ 0.95) were measured
# under ws3.v3 and MUST be re-confirmed by the orchestrator's eval re-run.
ANSWER_PROMPT_VERSION = "ws3.v4"

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


# Terse per the Wave-114 local-model precedent (trailing JSON directive is
# the load-bearing part). Passage-constrained answering + explicit refusal
# instruction.
#
# ws3.v3 answer-vs-refuse boundary: a 7B-Q4 model under ws3.v2 over-refused
# whenever the question's exact wording did not appear verbatim in a passage —
# it treated "synthesize across passages" and "apply a shown method to these
# specifics" as not-covered and set not_in_course=true. The SYNTHESIS + APPLY
# clauses tell the model the supporting material may be SPREAD across several
# passages (e.g. a week-overview plus a worked example) and that it should
# apply methods/worked examples the passages demonstrate to the question's
# specifics. Refusal is reserved for genuine absence: the passages contain no
# supporting material, NOT merely a wording mismatch.
#
# ws3.v4 completeness boundary (composer-completeness Phase A): the v3 prompt
# fixed over-refusal but the model then under-USES the evidence it has — it
# states the first/easiest supported point and stops, drops later clauses of a
# multi-part question whose passage IS in context, and cites only ONE supporter
# per claim. The COMPLETENESS, PER-PART, and CITE-EVERY-SUPPORTER clauses below
# are ALL scoped to SUPPORTED content (the grounding constraint stays primary —
# "ONLY the numbered passages", "never add anything they do not support"), so
# they pull more of the in-context evidence into the answer WITHOUT widening the
# door to fabrication. Length is not the lever: the instruction is completeness
# of the SUPPORTED points, never "be longer".
ANSWER_SYSTEM_PROMPT = (
    "You answer a student's question about one course using ONLY the "
    "numbered source passages provided. Do not use outside knowledge. "
    "The supporting material may be SPREAD ACROSS several passages — combine "
    "and synthesize the relevant passages into one answer. When a passage "
    "demonstrates a method or worked example, APPLY it to the specifics the "
    "question asks about. Answer whenever the passages substantively support "
    "the question, even if its exact wording does not appear verbatim. "
    "Be COMPLETE: state every point the passages support that is relevant to "
    "the question, not just the first or easiest one — but state ONLY what the "
    "passages support and never add anything they do not. "
    "If the question has multiple parts or clauses (for example \"X and Y\", "
    "\"a) ... b) ...\", or \"list two ...\"), answer EACH part explicitly; do "
    "not stop after the first. Address every part the passages support; for "
    "any part NOT covered by the passages, say so in one short sentence "
    "(e.g. \"The provided course material does not cover <part>.\") instead of "
    "omitting it silently, and never invent material for an uncovered part. "
    "Set \"not_in_course\" to true and leave \"answer\" empty ONLY when the "
    "passages contain no material supporting the question — not merely when "
    "its wording differs from the passages. Cite EVERY passage that supports a "
    "claim you state, not only one — if several passages support the same "
    "claim, cite all of them. Output JSON only: "
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
