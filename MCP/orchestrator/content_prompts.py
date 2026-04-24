"""
Prompt builders for mailbox-brokered subagent tasks (Wave 34).

When ``LocalDispatcher`` hands a task off through the ``TaskMailbox``,
the outer Claude Code session reads the spec's ``prompt`` field and
feeds it to the ``Agent`` tool. This module builds those prompts for
the three most common task shapes in the pipeline:

  * ``build_content_generation_prompt`` — Courseforge content-generator
    for a single week. Inputs: week number, chapter DART HTML, planned
    learning objectives, target output directory.
  * ``build_alt_text_prompt`` — DART alt-text generator for a single
    figure. Inputs: figure bytes (base64), caption, surrounding context.
  * ``build_synthesize_training_prompt`` — Trainforge training-pair
    synthesis for a single chunk. Inputs: chunk text + LO refs.

Design rules
------------

* Prompts must NOT leak corpus-specific identifiers. Placeholders
  (e.g. ``PHYS_101``, ``INT_101``) are only used in tests and doc
  strings; the builders themselves interpolate whatever the caller
  provides.
* Every prompt carries a **schema contract** section enumerating the
  exact output shape the dispatcher expects on the return trip. The
  shape is deliberately small and flat so subagents can return valid
  JSON without elaborate prompt engineering.
* The builders return plain strings. They don't touch the filesystem
  or dispatch anything.

Return shapes (contract the prompts pin down)
---------------------------------------------

Content generation prompt asks for::

    {
      "status": "ok" | "fail",
      "outputs": {
        "pages": [
          {"filename": "week_{n}_overview.html", "html": "...",
           "source_ids": ["dart-block-...", ...]},
          ... (4 entries: overview / content / application / summary)
        ]
      },
      "error": "<string if status == fail>"
    }

Alt-text prompt asks for::

    {
      "status": "ok" | "fail",
      "alt_text": "<concise accessible description>",
      "decorative": false,
      "confidence": 0.0-1.0
    }

Training-synthesis prompt asks for::

    {
      "status": "ok" | "fail",
      "instruction_pair": {"prompt": "...", "completion": "..."},
      "preference_pair": {"prompt": "...", "chosen": "...",
                          "rejected": "..."}
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

# Contract text shared across all three prompts. Keeps the "return a
# single JSON object, status ok/fail" rule in one place.
_COMMON_RETURN_CONTRACT = (
    "Return exactly one JSON object on the final line of your reply. "
    "No prose before or after. If you cannot complete the task, return "
    '``{"status": "fail", "error": "<reason>"}``. Status must be either '
    '``"ok"`` or ``"fail"``.'
)


# --------------------------------------------------------------------- utils


def _format_lo_refs(lo_refs: Sequence[Any]) -> str:
    """Render a list of LO refs as ``TO-01, TO-02`` style for the prompt.

    Accepts dicts with an ``id`` field, bare strings, or any object with
    an ``id`` attribute. Non-conforming entries are skipped with a
    warning rather than crashing the builder.
    """
    tokens: List[str] = []
    for ref in lo_refs or []:
        if isinstance(ref, str):
            tokens.append(ref)
        elif isinstance(ref, dict) and "id" in ref:
            tokens.append(str(ref["id"]))
        elif hasattr(ref, "id"):
            tokens.append(str(ref.id))
    return ", ".join(tokens) if tokens else "(no LOs supplied)"


def _truncate(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` chars with an ellipsis marker.

    We don't try to preserve valid HTML — this is only for prompts where
    an LLM is expected to tolerate a ``... [truncated]`` tail.
    """
    if not isinstance(text, str):
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


# ------------------------------------------------------- content generation

_CONTENT_PAGE_SCHEMA_BLOCK = """\
Required output: exactly FOUR HTML pages per week, in this order:

  1. overview     — motivation + week roadmap + LO statement
  2. content      — core teaching content (depth proportionate to LOs)
  3. application  — worked examples / practice / discussion prompts
  4. summary      — synthesis + preview of next week

Each page MUST:
  * be valid HTML5 (one <main> per page)
  * carry data-cf-role, data-cf-objective-ids, data-cf-bloom-level,
    data-cf-bloom-verb, data-cf-cognitive-domain, and data-cf-content-type
    attributes on the <main> element (see Courseforge/CLAUDE.md)
  * include a data-cf-source-ids attribute listing every DART source
    block id used to ground the content; every id must resolve against
    the staging manifest (the source_refs validator enforces this)
  * embed one <script type="application/ld+json"> block with the
    canonical courseforge_jsonld_v1 shape
"""


def build_content_generation_prompt(
    *,
    week_n: int,
    chapter_html: str,
    planned_los: Sequence[Any],
    output_dir: Path,
    chapter_id: Optional[str] = None,
    max_chapter_chars: int = 40000,
) -> str:
    """Build a content-generation prompt for a single week.

    Args:
        week_n: Week number (1-indexed).
        chapter_html: The DART HTML for the chapter(s) this week covers.
            Truncated at ``max_chapter_chars`` to keep prompts tractable.
        planned_los: Iterable of planned learning objective refs. Each
            entry can be a dict (``{"id": "TO-01", "statement": "..."}``),
            a string, or an object with an ``id`` attribute.
        output_dir: Target directory the subagent should write its four
            HTML files into.
        chapter_id: Optional chapter id (for logging + prompt header).
        max_chapter_chars: Truncation limit for ``chapter_html``.

    Returns a prompt string ready to be stuffed into a mailbox task
    spec's ``prompt`` field.
    """
    los_rendered = _format_lo_refs(planned_los)
    los_detail_lines: List[str] = []
    for ref in planned_los or []:
        if isinstance(ref, dict):
            lo_id = ref.get("id") or "???"
            stmt = ref.get("statement") or ref.get("text") or ""
            los_detail_lines.append(f"- {lo_id}: {stmt}")
        elif isinstance(ref, str):
            los_detail_lines.append(f"- {ref}")
    los_detail = "\n".join(los_detail_lines) or "(no LO statements provided)"

    chapter_block = _truncate(chapter_html or "", max_chapter_chars)
    chapter_header = (
        f"Chapter id: {chapter_id}" if chapter_id else "Chapter id: (unspecified)"
    )

    return (
        f"# Task: Generate week {week_n} course content\n\n"
        f"You are the Courseforge content-generator subagent. Produce the "
        f"four HTML pages for week {week_n} of the course.\n\n"
        f"## Output directory\n{output_dir}\n\n"
        f"## Planned learning objectives for this week\n"
        f"Covered LOs: {los_rendered}\n\n"
        f"{los_detail}\n\n"
        f"## Source DART HTML\n"
        f"{chapter_header}\n"
        f"<chapter-html>\n{chapter_block}\n</chapter-html>\n\n"
        f"## Page contract\n{_CONTENT_PAGE_SCHEMA_BLOCK}\n"
        f"## Return contract\n"
        f"{_COMMON_RETURN_CONTRACT}\n\n"
        f"On success, the JSON object must be:\n"
        f"```\n"
        f"{{\n"
        f"  \"status\": \"ok\",\n"
        f"  \"outputs\": {{\n"
        f"    \"pages\": [\n"
        f"      {{\"filename\": \"week_{week_n}_overview.html\", "
        f"\"html\": \"...\", \"source_ids\": [\"...\"]}},\n"
        f"      {{\"filename\": \"week_{week_n}_content.html\", ...}},\n"
        f"      {{\"filename\": \"week_{week_n}_application.html\", ...}},\n"
        f"      {{\"filename\": \"week_{week_n}_summary.html\", ...}}\n"
        f"    ]\n"
        f"  }}\n"
        f"}}\n"
        f"```\n"
    )


# -------------------------------------------------------------- alt text


_ALT_TEXT_CONTRACT_BLOCK = """\
Requirements for the alt text:

  * 1-2 sentences, under 180 characters total
  * Describes the figure's informational content, not its appearance
  * Names axes, units, and trend for charts; labels for diagrams
  * No "image of" / "picture of" preamble
  * Set "decorative": true (and leave alt_text empty) only when the
    figure carries no instructional information
"""


def build_alt_text_prompt(
    *,
    figure_bytes: Optional[bytes] = None,
    figure_b64: Optional[str] = None,
    caption: Optional[str] = None,
    context: Optional[str] = None,
    figure_id: Optional[str] = None,
    max_context_chars: int = 2000,
) -> str:
    """Build an alt-text prompt for a single figure.

    Exactly one of ``figure_bytes`` or ``figure_b64`` should be
    supplied. The caller is expected to base64-encode bytes before
    embedding; this builder does NOT import large binary payloads —
    it just carries the reference + caption + context.

    Args:
        figure_bytes: Raw figure bytes (used to compute a length hint
            only; payload itself is carried in ``figure_b64``).
        figure_b64: Base64-encoded figure payload (preferred path).
        caption: Figure caption text from the PDF.
        context: Surrounding paragraph(s) that give the figure meaning.
        figure_id: Stable id (e.g. ``fig-03-02``) for traceability.
        max_context_chars: Truncation limit for ``context``.
    """
    fig_id = figure_id or "(unspecified figure id)"
    cap = (caption or "(no caption provided)").strip()
    ctx = _truncate((context or "(no surrounding context provided)").strip(), max_context_chars)

    if figure_bytes is not None and figure_b64 is None:
        # We don't auto-b64-encode here because that would mean carrying
        # the full binary in memory inside the prompt. Callers should
        # pre-encode. We still accept raw bytes to emit a length hint.
        size_hint = f"{len(figure_bytes)} bytes (supply figure_b64 for payload)"
        b64_block = f"<!-- figure payload omitted; size hint: {size_hint} -->"
    elif figure_b64:
        b64_block = f"<figure-base64>\n{figure_b64}\n</figure-base64>"
    else:
        b64_block = "<!-- no figure payload provided -->"

    return (
        f"# Task: Generate alt text for figure {fig_id}\n\n"
        f"You are the DART alt-text subagent. Produce accessible alt "
        f"text for the figure below.\n\n"
        f"## Caption\n{cap}\n\n"
        f"## Surrounding context\n{ctx}\n\n"
        f"## Figure\n{b64_block}\n\n"
        f"## Requirements\n{_ALT_TEXT_CONTRACT_BLOCK}\n"
        f"## Return contract\n"
        f"{_COMMON_RETURN_CONTRACT}\n\n"
        f"On success, return:\n"
        f"```\n"
        f"{{\n"
        f"  \"status\": \"ok\",\n"
        f"  \"alt_text\": \"<1-2 sentences>\",\n"
        f"  \"decorative\": false,\n"
        f"  \"confidence\": 0.0\n"
        f"}}\n"
        f"```\n"
    )


# ---------------------------------------------------- training synthesis

_TRAINING_PAIR_CONTRACT_BLOCK = """\
Return two training pairs grounded in the chunk:

  * instruction_pair: A teaching prompt + an exemplary completion that
    exercises one of the chunk's LOs. Prompt is an instruction to a
    student; completion is a model answer.
  * preference_pair: The same (or a closely related) prompt, with a
    ``chosen`` (correct) response and a ``rejected`` response that
    embodies a plausible misconception or reasoning error.

Both pairs MUST:
  * be fully grounded in the chunk content (no external facts)
  * cite the same LO refs supplied in the task
  * keep prompt >= 40 characters (preference_pair schema minimum)
"""


def build_synthesize_training_prompt(
    *,
    chunk_text: str,
    lo_refs: Sequence[Any],
    chunk_id: Optional[str] = None,
    content_type: Optional[str] = None,
    bloom_level: Optional[str] = None,
    max_chunk_chars: int = 8000,
) -> str:
    """Build a training-pair synthesis prompt for a single chunk.

    Args:
        chunk_text: The chunk body text (truncated at ``max_chunk_chars``).
        lo_refs: LO refs the pairs should exercise (dicts, strings, or
            objects with ``id``).
        chunk_id: Stable chunk id for traceability.
        content_type: One of the 8 canonical content_type_label values,
            if known. Passed through to help the subagent pick a
            suitable instructional frame.
        bloom_level: Target Bloom's level hint, if known.
        max_chunk_chars: Truncation limit for ``chunk_text``.
    """
    los_rendered = _format_lo_refs(lo_refs)
    chunk_block = _truncate(chunk_text or "", max_chunk_chars)
    chunk_header = (
        f"Chunk id: {chunk_id}" if chunk_id else "Chunk id: (unspecified)"
    )
    hints: List[str] = []
    if content_type:
        hints.append(f"content_type: {content_type}")
    if bloom_level:
        hints.append(f"target bloom_level: {bloom_level}")
    hints_block = ("; ".join(hints)) if hints else "(no additional hints)"

    return (
        f"# Task: Synthesize training pairs from a content chunk\n\n"
        f"You are the Trainforge training-synthesis subagent. Produce "
        f"one instruction pair and one preference pair from the chunk "
        f"below, grounded in the supplied learning objectives.\n\n"
        f"## Chunk\n{chunk_header}\n"
        f"LOs to exercise: {los_rendered}\n"
        f"Hints: {hints_block}\n\n"
        f"<chunk-text>\n{chunk_block}\n</chunk-text>\n\n"
        f"## Requirements\n{_TRAINING_PAIR_CONTRACT_BLOCK}\n"
        f"## Return contract\n"
        f"{_COMMON_RETURN_CONTRACT}\n\n"
        f"On success, return:\n"
        f"```\n"
        f"{{\n"
        f"  \"status\": \"ok\",\n"
        f"  \"instruction_pair\": {{\"prompt\": \"...\", \"completion\": \"...\"}},\n"
        f"  \"preference_pair\": {{\"prompt\": \"...\", \"chosen\": \"...\", "
        f"\"rejected\": \"...\"}}\n"
        f"}}\n"
        f"```\n"
    )


__all__ = [
    "build_content_generation_prompt",
    "build_alt_text_prompt",
    "build_synthesize_training_prompt",
]
