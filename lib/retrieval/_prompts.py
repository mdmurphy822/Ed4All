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
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycle
    from lib.retrieval.answer_composer import RetrievedPassage


# Bump on ANY change to the system/user prompt wording below.
ANSWER_PROMPT_VERSION = "ws3.v1"

# Per-passage hard truncation (characters). A 14B-Q4 model loses the
# trailing JSON directive when the context balloons; capping each
# passage keeps the most-relevant heads in window.
PASSAGE_CHAR_CAP = 1500

# Total context budget (characters) across all numbered passage blocks.
# When exceeded, trailing (lowest-ranked) passages are DROPPED whole —
# the question is never truncated. The dropped count is recorded in the
# composer's decision-capture rationale.
MAX_CONTEXT_CHARS = 12000


# Terse per the Wave-114 local-model precedent (< 80 words; trailing
# JSON directive is the load-bearing part). Passage-constrained
# answering + explicit refusal instruction.
ANSWER_SYSTEM_PROMPT = (
    "You answer a student's question about one course using ONLY the "
    "numbered source passages provided. Do not use outside knowledge. "
    "If the passages do not contain enough information to answer, set "
    '"not_in_course" to true and leave "answer" empty. Cite the id of '
    "every passage that supports the answer. Output JSON only: "
    '{"answer": "...", "citations": ["<passage_id>", ...], '
    '"not_in_course": false}.'
)


# Appended to the user prompt after one or more bad citation ids come
# back; drives a single remediation retry. ``{bad_ids}`` is formatted
# with the comma-joined offending ids.
CITATION_REMEDIATION_DIRECTIVE = (
    "Your previous reply cited passage ids that were not in the provided "
    "list: {bad_ids}. Answer again using ONLY the provided passage ids."
)


# The trailing JSON directive — repeated at the END of the user prompt
# because that is the most-respected position for small local models.
_TRAILING_JSON_DIRECTIVE = (
    'Output JSON only: {"answer": "...", "citations": '
    '["<passage_id>", ...], "not_in_course": false}.'
)


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
) -> str:
    """Render the numbered passage-context user prompt.

    Block shape (one per passage, ranked order)::

        [<chunk_id>] (<section_heading or item_path>)
        <text truncated to PASSAGE_CHAR_CAP>

    Total budget guarded by ``MAX_CONTEXT_CHARS`` — trailing passages
    are dropped whole (never the question). The dropped-count is
    surfaced via :func:`render_answer_user_prompt_with_meta`.
    """
    text, _ = render_answer_user_prompt_with_meta(query, passages)
    return text


def render_answer_user_prompt_with_meta(
    query: str,
    passages: Sequence["RetrievedPassage"],
):
    """Like :func:`render_answer_user_prompt` but also returns metadata.

    Returns ``(prompt_text, meta)`` where ``meta`` carries
    ``{"included_ids": [...], "dropped_count": int, "context_chars": int}``
    so the composer can interpolate the dropped count into its
    decision-capture rationale.
    """
    blocks = []
    included_ids = []
    context_chars = 0
    dropped = 0

    for passage in passages:
        chunk_id = str(getattr(passage, "chunk_id", ""))
        body = str(getattr(passage, "text", "") or "")
        if len(body) > PASSAGE_CHAR_CAP:
            body = body[:PASSAGE_CHAR_CAP]
        block = f"[{chunk_id}] ({_passage_label(passage)})\n{body}"
        # Drop whole trailing blocks once the budget is exhausted.
        if context_chars + len(block) > MAX_CONTEXT_CHARS and blocks:
            dropped += 1
            continue
        blocks.append(block)
        included_ids.append(chunk_id)
        context_chars += len(block)

    context = "\n\n".join(blocks)
    prompt = (
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"{_TRAILING_JSON_DIRECTIVE}"
    )
    meta = {
        "included_ids": included_ids,
        "dropped_count": dropped,
        "context_chars": context_chars,
    }
    return prompt, meta


__all__ = [
    "ANSWER_PROMPT_VERSION",
    "ANSWER_SYSTEM_PROMPT",
    "CITATION_REMEDIATION_DIRECTIVE",
    "PASSAGE_CHAR_CAP",
    "MAX_CONTEXT_CHARS",
    "render_answer_user_prompt",
    "render_answer_user_prompt_with_meta",
]
