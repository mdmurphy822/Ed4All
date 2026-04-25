"""``ed4all libv2 ask`` (Wave 78 Worker C).

Intent-routed retrieval surface for LibV2 archives. A natural-language
query is classified into one of six canonical intent classes
(``objective_lookup``, ``prerequisite_query``, ``misconception_query``,
``assessment_query``, ``faceted_query``, ``concept_query``) and then
dispatched to the right backend instead of treating every query as
similarity search.

Examples
--------

    ed4all libv2 ask --slug rdf-shacl-550-rdf-shacl-550 \\
        --query "Which chunks assess to-04?"

    ed4all libv2 ask --slug rdf-shacl-550-rdf-shacl-550 \\
        --query "What is a prerequisite for SHACL validation?" \\
        --show-routing

    ed4all libv2 ask --slug rdf-shacl-550-rdf-shacl-550 \\
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
# Click command                                                               #
# --------------------------------------------------------------------------- #


@click.command("ask")
@click.option(
    "--slug",
    required=True,
    help="LibV2 course slug (e.g. rdf-shacl-550-rdf-shacl-550).",
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
def ask_command(
    slug: str,
    query_text: str,
    top_k: int,
    show_routing: bool,
    output_format: str,
    courses_root: Optional[Path],
) -> None:
    """Intent-routed natural-language query over a LibV2 archive."""
    if top_k < 0:
        raise click.UsageError("--top-k must be >= 0")

    envelope = dispatch(
        query_text,
        slug,
        top_k=top_k,
        courses_root=courses_root,
    )

    fmt = output_format.lower()
    if fmt == "json":
        click.echo(_format_json(envelope, show_routing))
    else:
        click.echo(_format_text(envelope, show_routing))


def register_libv2_ask_command(libv2_group: click.Group) -> None:
    """Attach ``ask`` to the ``ed4all libv2`` command group. Idempotent."""
    if "ask" in libv2_group.commands:
        return
    libv2_group.add_command(ask_command)
