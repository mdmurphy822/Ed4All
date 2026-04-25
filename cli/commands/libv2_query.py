"""``ed4all libv2 query`` (Wave 77 Worker β).

Faceted, typed query over the chunk store of a LibV2 archive. Backed by
:mod:`LibV2.tools.chunk_query` so the same filter logic is reusable
from MCP tools later.

Filters compose with AND. Multi-value flags use comma-separated
values; OR semantics inside a single filter, AND across filters.
``--week`` accepts ``N`` or ``N-M``. ``--outcome`` rolls TO ids up to
include their child COs.

Examples
--------

    ed4all libv2 query --slug rdf-shacl-550-rdf-shacl-550 \\
        --chunk-type example --difficulty intermediate

    ed4all libv2 query --slug rdf-shacl-550-rdf-shacl-550 --week 7

    ed4all libv2 query --slug rdf-shacl-550-rdf-shacl-550 \\
        --outcome to-04 --format count

    ed4all libv2 query --slug rdf-shacl-550-rdf-shacl-550 \\
        --text "sh:minCount" --limit 5 --format md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import click

from LibV2.tools.chunk_query import (
    BLOOM_LEVELS,
    CHUNK_TYPES,
    DIFFICULTY_LEVELS,
    SORT_KEYS,
    ChunkQueryError,
    MalformedArchiveError,
    QueryFilter,
    QueryResult,
    UnknownSlugError,
    parse_csv,
    parse_week_spec,
    query_chunks,
    validate_choice,
)


_TEXT_PREVIEW_CHARS = 200


# --------------------------------------------------------------------------- #
# Output formatters                                                           #
# --------------------------------------------------------------------------- #


def _format_count(result: QueryResult) -> str:
    return str(result.total_matches)


def _chunk_week_label(chunk: Dict[str, Any]) -> str:
    module_id = (chunk.get("source") or {}).get("module_id") or ""
    if module_id.startswith("week_"):
        return module_id.split("_", 2)[1] if "_" in module_id else "?"
    return "?"


def _format_md(result: QueryResult) -> str:
    if not result.chunks:
        return "_(no matches)_"
    lines: List[str] = []
    lines.append(
        f"# Query results — slug={result.slug} "
        f"({result.returned}/{result.total_matches} shown)"
    )
    if result.expanded_outcomes:
        lines.append("")
        lines.append(
            "_Outcome rollup expanded to: "
            f"{', '.join(result.expanded_outcomes)}_"
        )
    lines.append("")
    for idx, chunk in enumerate(result.chunks, start=1):
        chunk_id = chunk.get("id") or "?"
        chunk_type = chunk.get("chunk_type") or "?"
        bloom = chunk.get("bloom_level") or "?"
        difficulty = chunk.get("difficulty") or "?"
        week = _chunk_week_label(chunk)
        module = (chunk.get("source") or {}).get("module_id") or "?"
        outcomes = chunk.get("learning_outcome_refs") or []
        text = (chunk.get("text") or "").strip().replace("\n", " ")
        if len(text) > _TEXT_PREVIEW_CHARS:
            text = text[:_TEXT_PREVIEW_CHARS].rstrip() + "…"
        lines.append(
            f"## {idx}. `{chunk_id}` — week {week} / {chunk_type}"
        )
        lines.append("")
        lines.append(
            f"- **module**: `{module}`"
        )
        lines.append(f"- **bloom**: {bloom} | **difficulty**: {difficulty}")
        lines.append(
            f"- **outcomes**: {', '.join(outcomes) if outcomes else '_(none)_'}"
        )
        lines.append("")
        lines.append(f"> {text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_table(result: QueryResult) -> str:
    """Fixed-width terminal table."""
    headers = ("ID", "WK", "TYPE", "BLOOM", "DIFF", "OUTCOMES", "MODULE")
    widths = [26, 3, 16, 11, 13, 22, 28]
    rows: List[List[str]] = []
    for chunk in result.chunks:
        outcomes = chunk.get("learning_outcome_refs") or []
        rows.append(
            [
                str(chunk.get("id") or "")[: widths[0]],
                str(_chunk_week_label(chunk))[: widths[1]],
                str(chunk.get("chunk_type") or "")[: widths[2]],
                str(chunk.get("bloom_level") or "")[: widths[3]],
                str(chunk.get("difficulty") or "")[: widths[4]],
                ",".join(outcomes)[: widths[5]],
                str((chunk.get("source") or {}).get("module_id") or "")[: widths[6]],
            ]
        )

    def _fmt_row(values: Sequence[str]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(values, widths))

    lines = [_fmt_row(headers), _fmt_row(["-" * w for w in widths])]
    for row in rows:
        lines.append(_fmt_row(row))
    if not rows:
        lines.append("(no matches)")
    lines.append("")
    lines.append(
        f"-- {result.returned} of {result.total_matches} matches "
        f"(slug={result.slug}, sort={result.sort_key})"
    )
    return "\n".join(lines)


def _format_json(result: QueryResult) -> str:
    payload = {
        "slug": result.slug,
        "total_matches": result.total_matches,
        "returned": result.returned,
        "sort_key": result.sort_key,
        "expanded_outcomes": result.expanded_outcomes,
        "chunks": result.chunks,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


_FORMATTERS = {
    "json": _format_json,
    "md": _format_md,
    "table": _format_table,
    "count": _format_count,
}


# --------------------------------------------------------------------------- #
# Click command                                                               #
# --------------------------------------------------------------------------- #


@click.command("query")
@click.option(
    "--slug",
    required=True,
    help="LibV2 course slug (e.g. rdf-shacl-550-rdf-shacl-550).",
)
@click.option(
    "--chunk-type",
    "chunk_type",
    default=None,
    help=(
        "Comma-separated chunk types. "
        f"Allowed: {', '.join(CHUNK_TYPES)}."
    ),
)
@click.option(
    "--bloom",
    default=None,
    help=(
        "Comma-separated Bloom levels. "
        f"Allowed: {', '.join(BLOOM_LEVELS)}."
    ),
)
@click.option(
    "--difficulty",
    default=None,
    help=(
        "Comma-separated difficulty levels. "
        f"Allowed: {', '.join(DIFFICULTY_LEVELS)}."
    ),
)
@click.option(
    "--week",
    default=None,
    help='Week filter: single value (e.g. "7") or inclusive range ("1-12").',
)
@click.option(
    "--module",
    default=None,
    help="Comma-separated module ids (exact match against source.module_id).",
)
@click.option(
    "--outcome",
    default=None,
    help=(
        "Comma-separated learning outcome ids (to-* or co-*). "
        "Querying a TO rolls up to its child COs."
    ),
)
@click.option(
    "--text",
    "text_substring",
    default=None,
    help="Case-insensitive substring match against chunk.text.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of chunks to return.",
)
@click.option(
    "--offset",
    type=int,
    default=0,
    show_default=True,
    help="Skip the first N matches (for pagination).",
)
@click.option(
    "--sort",
    "sort_key",
    type=click.Choice(SORT_KEYS, case_sensitive=False),
    default="week",
    show_default=True,
    help="Sort key.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "md", "table", "count"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--courses-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override LibV2 courses root (tests). Defaults to LibV2/courses/.",
)
def query_command(
    slug: str,
    chunk_type: Optional[str],
    bloom: Optional[str],
    difficulty: Optional[str],
    week: Optional[str],
    module: Optional[str],
    outcome: Optional[str],
    text_substring: Optional[str],
    limit: Optional[int],
    offset: int,
    sort_key: str,
    output_format: str,
    courses_root: Optional[Path],
) -> None:
    """Faceted query over chunk metadata in a LibV2 archive."""
    chunk_types = parse_csv(chunk_type)
    bloom_levels = parse_csv(bloom)
    difficulties = parse_csv(difficulty)
    modules = parse_csv(module)
    outcomes = parse_csv(outcome)

    # Validate enums up front so the user gets a clear error.
    try:
        validate_choice(chunk_types, CHUNK_TYPES, "--chunk-type")
        validate_choice(bloom_levels, BLOOM_LEVELS, "--bloom")
        validate_choice(difficulties, DIFFICULTY_LEVELS, "--difficulty")
    except ValueError as exc:
        raise click.UsageError(str(exc))

    week_min: Optional[int] = None
    week_max: Optional[int] = None
    if week is not None:
        try:
            week_min, week_max = parse_week_spec(week)
        except ValueError as exc:
            raise click.UsageError(f"--week: {exc}")

    if limit is not None and limit < 0:
        raise click.UsageError("--limit must be >= 0")
    if offset is not None and offset < 0:
        raise click.UsageError("--offset must be >= 0")

    query = QueryFilter(
        chunk_types=chunk_types,
        bloom_levels=bloom_levels,
        difficulties=difficulties,
        week_min=week_min,
        week_max=week_max,
        modules=modules,
        outcomes=outcomes,
        text_substring=text_substring,
        limit=limit,
        offset=offset,
        sort_key=sort_key,
    )

    try:
        result = query_chunks(slug, query, courses_root=courses_root)
    except UnknownSlugError as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(2)
    except MalformedArchiveError as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(2)
    except ChunkQueryError as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(2)

    formatter = _FORMATTERS[output_format.lower()]
    click.echo(formatter(result))


def register_libv2_query_command(libv2_group: click.Group) -> None:
    """Attach ``query`` to the ``ed4all libv2`` command group.

    Wave 77 wires this onto the same ``libv2`` group registered by
    :mod:`cli.commands.libv2_validate_packet`. Idempotent.
    """
    if "query" in libv2_group.commands:
        return
    libv2_group.add_command(query_command)
