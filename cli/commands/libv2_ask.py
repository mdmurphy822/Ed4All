"""``ed4all libv2 ask`` (Wave 78 Worker C).

Intent-routed retrieval surface for LibV2 archives. A natural-language
query is classified into one of six canonical intent classes
(``objective_lookup``, ``prerequisite_query``, ``misconception_query``,
``assessment_query``, ``faceted_query``, ``concept_query``) and then
dispatched to the right backend instead of treating every query as
similarity search.

Examples
--------

    ed4all libv2 ask --slug demo-course-1 \\
        --query "Which chunks assess to-04?"

    ed4all libv2 ask --slug demo-course-1 \\
        --query "What is a prerequisite for SHACL validation?" \\
        --show-routing

    ed4all libv2 ask --slug demo-course-1 \\
        --query "How does sh:minCount work?" --top-k 10 --format json

The ``--show-routing`` flag prints the intent classification + entity
extraction *before* the result body so a human can audit which path
was taken; ``--format json`` emits the canonical envelope shape used
by the MCP tool wrapper (:mod:`MCP.tools.intent_dispatch_tool`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from LibV2.tools.intent_router import dispatch
from lib.retrieval.library_wide import (
    answer_library_question,
    list_library_courses,
)


# --------------------------------------------------------------------------- #
# Result formatters                                                           #
# --------------------------------------------------------------------------- #


_PREVIEW_CHARS = 200
_INTENT_TAGS = {
    "objective_lookup": "[OBJECTIVE]",
    "prerequisite_query": "[PREREQ]",
    "misconception_query": "[MISCONCEPTION]",
    "assessment_query": "[ASSESSMENT]",
    "faceted_query": "[FACETED]",
    "concept_query": "[CONCEPT]",
}


def _truncate(text: str, n: int = _PREVIEW_CHARS) -> str:
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def _format_entities_text(entities: Dict[str, Any]) -> List[str]:
    """Render the entity envelope as a compact set of indented lines."""
    lines = []
    if entities.get("objective_ids"):
        lines.append(f"    objective_ids: {', '.join(entities['objective_ids'])}")
    if entities.get("weeks"):
        lines.append(f"    weeks:         {', '.join(str(w) for w in entities['weeks'])}")
    if entities.get("bloom_verbs"):
        verbs = ", ".join(f"{v}({lvl})" for v, lvl in entities["bloom_verbs"])
        lines.append(f"    bloom_verbs:   {verbs}")
    if entities.get("chunk_types"):
        lines.append(f"    chunk_types:   {', '.join(entities['chunk_types'])}")
    markers = []
    if entities.get("has_prereq_marker"):
        markers.append("prereq")
    if entities.get("has_misconception_marker"):
        markers.append("misconception")
    if entities.get("has_assessment_marker"):
        markers.append("assessment")
    if markers:
        lines.append(f"    markers:       {', '.join(markers)}")
    if entities.get("residual_text"):
        lines.append(f"    residual:      {entities['residual_text']!r}")
    return lines


def _format_text(envelope: Dict[str, Any], show_routing: bool) -> str:
    """Human-readable rendering with intent tag + result preview."""
    lines: List[str] = []
    intent = envelope["intent_class"]
    tag = _INTENT_TAGS.get(intent, f"[{intent.upper()}]")
    lines.append(
        f"{tag} intent={intent} "
        f"confidence={envelope['confidence']:.2f} "
        f"slug={envelope['slug']}"
    )
    lines.append(f"  route: {envelope['route']}")

    if show_routing:
        ent_lines = _format_entities_text(envelope.get("entities") or {})
        if ent_lines:
            lines.append("  entities:")
            lines.extend(ent_lines)

    results = envelope.get("results") or []
    lines.append("")
    lines.append(f"Results ({len(results)}):")
    if not results:
        lines.append("  (no matches)")
        return "\n".join(lines)

    for i, r in enumerate(results, 1):
        if intent == "prerequisite_query":
            # Concept-graph edge result.
            lines.append("")
            lines.append(
                f"  [{i}] {r.get('concept')} -> "
                f"{r.get('target')} "
                f"(confidence={r.get('confidence')})"
            )
        elif intent == "misconception_query":
            lines.append("")
            lines.append(
                f"  [{i}] score={r.get('score', 0):.3f} chunk={r.get('chunk_id')}"
            )
            lines.append(f"      Misconception: {_truncate(r.get('misconception') or '')}")
            lines.append(f"      Correction:    {_truncate(r.get('correction') or '')}")
        else:
            # Chunk envelope (objective / assessment / faceted / concept).
            chunk_id = r.get("id") or "?"
            chunk_type = r.get("chunk_type") or "?"
            bloom = r.get("bloom_level") or "?"
            module = (r.get("source") or {}).get("module_id") or "?"
            score = r.get("score")
            score_str = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
            lines.append("")
            lines.append(
                f"  [{i}] {chunk_id}  "
                f"{chunk_type}/{bloom}{score_str}  module={module}"
            )
            lines.append(f"      {_truncate(r.get('text') or '')}")
    return "\n".join(lines)


def _format_json(envelope: Dict[str, Any], show_routing: bool) -> str:
    """JSON envelope. ``show_routing`` toggles whether to include the
    full ``entities`` block (always-on entity emission is fine — JSON
    consumers can ignore fields they don't need)."""
    payload = {
        "query": envelope["query"],
        "slug": envelope["slug"],
        "intent_class": envelope["intent_class"],
        "confidence": envelope["confidence"],
        "route": envelope["route"],
        "source_path": envelope["source_path"],
        "entities": envelope.get("entities", {}),
        "results": envelope.get("results", []),
    }
    if not show_routing:
        # Strip the bulky residual + cue flags to keep ``--format json``
        # default output readable; structural ID fields stay.
        ent = dict(payload["entities"])
        ent.pop("residual_text", None)
        ent.pop("has_prereq_marker", None)
        ent.pop("has_misconception_marker", None)
        ent.pop("has_assessment_marker", None)
        payload["entities"] = ent
    return json.dumps(payload, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Library-wide grounded-answer rendering (opt-in --library-wide)              #
# --------------------------------------------------------------------------- #


def _resolve_libv2_root(courses_root: Optional[Path]) -> Path:
    """Resolve the LibV2 root for a library-wide ask.

    ``--courses-root`` (a ``.../courses`` dir) → its parent is the LibV2 root.
    Otherwise defer to :func:`lib.paths.libv2_path` (which honors
    ``ED4ALL_LIBV2_ROOT`` / ``ED4ALL_HOME``). ``answer_library_question`` /
    ``list_library_courses`` both take the LibV2 root (the dir holding
    ``courses/``).
    """
    if courses_root is not None:
        return Path(courses_root).parent
    from lib.paths import libv2_path  # noqa: PLC0415 — read at call time (env seam)

    return libv2_path()


def _format_library_json(result: Any) -> str:
    """JSON rendering of a library-wide :class:`GroundedAnswer` (its ``to_dict``)."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def _format_library_text(result: Any) -> str:
    """Human-readable rendering of a library-wide grounded answer.

    Surfaces the per-course provenance stamped on every citation (the whole
    point of library-wide mode) so an operator sees which course each cited
    passage came from.
    """
    payload = result.to_dict()
    lines: List[str] = []
    status = payload.get("status", "?")
    citations = payload.get("citations") or []
    courses = sorted({c.get("course_slug") for c in citations if c.get("course_slug")})
    course_str = ", ".join(courses) if courses else "(none)"
    lines.append(
        f"[LIBRARY-WIDE] status={status} "
        f"engine={payload.get('engine')} home={payload.get('course_slug')}"
    )
    lines.append(f"  courses: {course_str}")

    answer_text = payload.get("answer_text")
    lines.append("")
    if answer_text:
        lines.append("Answer:")
        lines.append(f"  {answer_text.strip()}")
    else:
        refusal = payload.get("refusal") or {}
        reason = refusal.get("reason_code") or status
        lines.append(f"(no answer — {reason})")

    lines.append("")
    lines.append(f"Citations ({len(citations)}):")
    if not citations:
        lines.append("  (none)")
        return "\n".join(lines)
    for i, c in enumerate(citations, 1):
        course = c.get("course_slug") or "?"
        heading = c.get("section_heading") or c.get("page_label") or "?"
        lines.append("")
        lines.append(
            f"  [{i}] {c.get('chunk_id')}  course={course}  {heading}"
        )
        quote = c.get("text_quote") or c.get("supporting_excerpt")
        if quote:
            lines.append(f"      {_truncate(quote)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Click command                                                               #
# --------------------------------------------------------------------------- #


@click.command("ask")
@click.option(
    "--slug",
    required=True,
    help="LibV2 course slug (e.g. demo-course-1).",
)
@click.option(
    "--query",
    "query_text",
    required=True,
    help="Natural-language query to classify and dispatch.",
)
@click.option(
    "--top-k",
    "top_k",
    type=int,
    default=5,
    show_default=True,
    help="Maximum number of results to return.",
)
@click.option(
    "--show-routing",
    is_flag=True,
    default=False,
    help="Emit the intent classification + entity extraction along with results.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--courses-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override LibV2 courses root (tests). Defaults to LibV2/courses/.",
)
@click.option(
    "--library-wide",
    is_flag=True,
    default=False,
    help=(
        "Answer grounded across ALL catalog courses (union retrieval with "
        "per-course provenance on every citation), not just --slug. Local-only, "
        "fail-open to the single course. Default off => the byte-identical "
        "single-course intent-routed path."
    ),
)
def ask_command(
    slug: str,
    query_text: str,
    top_k: int,
    show_routing: bool,
    output_format: str,
    courses_root: Optional[Path],
    library_wide: bool,
) -> None:
    """Intent-routed natural-language query over a LibV2 archive."""
    if top_k < 0:
        raise click.UsageError("--top-k must be >= 0")

    fmt = output_format.lower()

    if library_wide:
        # Opt-in: grounded answer unioned across the resolved catalog course set
        # (reuses list_library_courses' load_master_catalog / filesystem
        # fallback). Explicit library_wide=True wins over the env flag.
        libv2_root = _resolve_libv2_root(courses_root)
        resolved_courses = list_library_courses(libv2_root, slug)
        result = answer_library_question(
            libv2_root,
            slug,
            query_text,
            limit=top_k,
            course_slugs=resolved_courses,
            library_wide=True,
        )
        if fmt == "json":
            click.echo(_format_library_json(result))
        else:
            click.echo(_format_library_text(result))
        return

    envelope = dispatch(
        query_text,
        slug,
        top_k=top_k,
        courses_root=courses_root,
    )

    if fmt == "json":
        click.echo(_format_json(envelope, show_routing))
    else:
        click.echo(_format_text(envelope, show_routing))


def register_libv2_ask_command(libv2_group: click.Group) -> None:
    """Attach ``ask`` to the ``ed4all libv2`` command group. Idempotent."""
    if "ask" in libv2_group.commands:
        return
    libv2_group.add_command(ask_command)
