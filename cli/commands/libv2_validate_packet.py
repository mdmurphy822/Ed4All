"""``ed4all libv2 validate-packet`` (Wave 75 Worker D + Wave 78).

Operator-facing wrapper around
:class:`lib.validators.libv2_packet_integrity.PacketIntegrityValidator`.

Runs SHACL-style integrity rules on a LibV2 archive
(``LibV2/courses/<slug>/``) and emits either a human-readable summary
table or a machine-readable JSON report.

Wave 78 promotes the validator to a real workflow gate at
``libv2_archival`` (see ``config/workflows.yaml``). This CLI remains
the on-demand operator interface. Strict-mode flags
(``--strict-coverage``, ``--strict-typing``, ``--strict``) opt into
fail-closed coverage + typing rules — without them the validator
preserves the Wave 75 warning-only behavior for the new rules.

Examples
--------

    ed4all libv2 validate-packet --slug rdf-shacl-550-rdf-shacl-550
    ed4all libv2 validate-packet --slug X --format json
    ed4all libv2 validate-packet --slug X --strict
    ed4all libv2 validate-packet --slug X --strict-coverage --format json
    ed4all libv2 validate-packet --slug X --strict-typing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from lib.paths import LIBV2_PATH
from lib.validators.libv2_packet_integrity import (
    PacketIntegrityValidator,
    ValidationResult,
)


def _resolve_slug(slug: str, courses_root: Optional[Path] = None) -> Path:
    """Resolve ``slug`` to a ``LibV2/courses/<slug>/`` path."""
    root = courses_root if courses_root else (LIBV2_PATH / "courses")
    return root / slug


def _format_text_report(result: ValidationResult) -> str:
    """Render a human-readable summary of the result."""
    lines = []
    lines.append("ed4all libv2 validate-packet")
    lines.append("-" * 60)
    lines.append(f"  Archive: {result.archive_root}")
    lines.append(
        f"  Rules: {result.rules_run} run, "
        f"{result.rules_passed} passed, {result.rules_failed} failed"
    )
    lines.append(
        f"  Issues: {len(result.issues)} total "
        f"(critical={result.critical_count}, warning={result.warning_count})"
    )
    summary = result.summary or {}
    if "chunk_count" in summary:
        lines.append("")
        lines.append("  Archive contents:")
        lines.append(f"    chunks:                    {summary.get('chunk_count')}")
        lines.append(
            f"    terminal_outcomes:         {summary.get('terminal_outcome_count')}"
        )
        lines.append(
            f"    component_outcomes:        {summary.get('component_outcome_count')}"
        )
        lines.append(
            f"    objectives_source:         {summary.get('objectives_source')}"
        )
        lines.append(
            f"    concept_graph_nodes:       {summary.get('concept_graph_node_count')}"
        )
        lines.append(
            f"    concept_graph_edges:       {summary.get('concept_graph_edge_count')}"
        )
        lines.append(
            "    concept_graph_semantic_edges: "
            f"{summary.get('concept_graph_semantic_edge_count')}"
        )
        lines.append(
            f"    pedagogy_graph_nodes:      {summary.get('pedagogy_graph_node_count')}"
        )
        lines.append(
            f"    pedagogy_graph_edges:      {summary.get('pedagogy_graph_edge_count')}"
        )
    by_code = (summary or {}).get("issues_by_code") or {}
    if by_code:
        lines.append("")
        lines.append("  Issues by code:")
        for code, count in sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"    {code:32s} {count}")
    if result.issues:
        lines.append("")
        lines.append("  First 10 issues:")
        for issue in result.issues[:10]:
            lines.append(
                f"    [{issue.severity:8s}] {issue.issue_code} "
                f"({issue.rule}): {issue.message}"
            )
    return "\n".join(lines)


def _write_quality_report(archive_root: Path, payload: dict) -> Optional[Path]:
    """Write the JSON report to ``<archive_root>/quality/graph_validation_report.json``.

    Returns the written path, or ``None`` if writing failed (we still
    succeed at the CLI level — the JSON is also emitted to stdout).
    """
    quality_dir = archive_root / "quality"
    if not archive_root.exists():
        return None
    quality_dir.mkdir(parents=True, exist_ok=True)
    target = quality_dir / "graph_validation_report.json"
    try:
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target
    except OSError:
        return None


@click.group(name="libv2")
def libv2_group() -> None:
    """LibV2 inspection commands."""


@libv2_group.command("validate-packet")
@click.option(
    "--slug",
    required=True,
    help=(
        "LibV2 course slug. Resolves to LibV2/courses/<slug>/. "
        "Example: rdf-shacl-550-rdf-shacl-550."
    ),
)
@click.option(
    "--strict",
    is_flag=True,
    help=(
        "Wave 78: imply both --strict-coverage and --strict-typing. "
        "Coverage + typing rules become critical; non-zero exit on any "
        "critical issue."
    ),
)
@click.option(
    "--strict-coverage",
    is_flag=True,
    help=(
        "Wave 78: promote coverage rules "
        "(every_objective_has_teaching, every_objective_has_assessment, "
        "to_has_teaching_and_assessment, domain_concept_has_chunk) to "
        "critical."
    ),
)
@click.option(
    "--strict-typing",
    is_flag=True,
    help=(
        "Wave 78: promote edge_endpoint_typing to critical (validates "
        "edge endpoint classes against the typed-endpoint contract)."
    ),
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help=(
        "Output format. 'json' also writes the report to "
        "<archive>/quality/graph_validation_report.json."
    ),
)
@click.option(
    "--courses-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override LibV2 courses root (tests only). Defaults to LibV2/courses/.",
)
def validate_packet_command(
    slug: str,
    strict: bool,
    strict_coverage: bool,
    strict_typing: bool,
    output_format: str,
    courses_root: Optional[Path],
) -> None:
    """Validate a LibV2 archive's internal SHACL-style integrity."""
    archive_root = _resolve_slug(slug, courses_root)

    # --strict implies both granular flags.
    effective_strict_coverage = bool(strict or strict_coverage)
    effective_strict_typing = bool(strict or strict_typing)

    validator = PacketIntegrityValidator(
        strict_coverage=effective_strict_coverage,
        strict_typing=effective_strict_typing,
    )
    result = validator.validate(archive_root)

    payload = result.to_dict()
    # Surface the active strictness mode in the report so post-hoc
    # readers can tell whether warnings were promoted.
    payload.setdefault("summary", {})
    payload["summary"]["strict_coverage"] = effective_strict_coverage
    payload["summary"]["strict_typing"] = effective_strict_typing

    if output_format == "json":
        # Emit to stdout AND persist alongside the archive.
        click.echo(json.dumps(payload, indent=2))
        _write_quality_report(archive_root, payload)
    else:
        click.echo(_format_text_report(result))

    # Wave 78 exit-code rules: any critical under the active
    # strictness mode → 1; otherwise 0. Warnings never trip a
    # non-zero exit (the strict flags are the way to escalate).
    if result.critical_count > 0:
        sys.exit(1)
    sys.exit(0)


def register_libv2_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all libv2`` command group to the top-level CLI."""
    cli_group.add_command(libv2_group)
