"""``ed4all objectives restructure`` — deterministic objectives-doc rebuild.

Restructures an already-synthesized ``synthesized_objectives.json`` WITHOUT a 7B
re-roll (Defect F). It composes the deterministic W1-W4 cores — lexical
near-restatement dedup (E), vacuity annotate/drop (B), chapter-anchored terminal
re-derivation (A), sub-objective quality (D) — and writes a fresh doc that
round-trips the ``--reuse-objectives`` plumbing, so the operator can feed it
straight back into ``ed4all run ... --reuse-objectives <out>``.

NO LLM anywhere. The whole pass is deterministic (``lib/objectives/restructure``);
the ONLY external state it reads is the course's DART chunkset (resolved exactly
like the planner's ``_load_semantik_chunkset_for_planning``) for the chapter-anchor
signal. A pending stop sentinel pauses the run at a per-CO boundary (exit code
3); because the pass is LLM-free there is no resume sidecar — just re-run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click


def _looks_libv2_archive(data: Dict[str, Any]) -> bool:
    """True when the doc is the LibV2 archive shape (out of scope for v1)."""
    has_courseforge = (
        isinstance(data.get("terminal_objectives"), list)
        or isinstance(data.get("chapter_objectives"), list)
    )
    has_libv2 = (
        isinstance(data.get("terminal_outcomes"), list)
        or isinstance(data.get("component_objectives"), list)
    )
    return has_libv2 and not has_courseforge


@click.group("objectives")
def objectives_group() -> None:
    """Deterministic operations over an existing objectives doc."""


@objectives_group.command("restructure")
@click.argument("objectives_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--course-name", required=True, help="Course code / name (drives the DART chunkset slug).")
@click.option(
    "--libv2-root",
    type=click.Path(file_okay=False),
    default=None,
    help="Override the LibV2 root (else ED4ALL_LIBV2_ROOT / repo default).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output path (default: <input>.restructured.json).",
)
@click.option(
    "--drop-vacuous/--annotate-only",
    "drop_vacuous",
    default=False,
    help="Drop V1-vacuous COs (keep-≥1) vs only annotate them (default).",
)
@click.option(
    "--weeks",
    type=int,
    default=None,
    help="Override duration_weeks (default: max(8, num_tos)).",
)
def restructure_command(
    objectives_path: str,
    course_name: str,
    libv2_root: Optional[str],
    output: Optional[str],
    drop_vacuous: bool,
    weeks: Optional[int],
) -> None:
    """Deterministically restructure OBJECTIVES_PATH (no LLM).

    Writes ``<input>.restructured.json`` (or ``--output``) + a sibling
    ``restructure_report.json``. Feed the output back via
    ``ed4all run ... --reuse-objectives <out>``.
    """
    from lib.generation.stop_control import GracefulStopRequested
    from lib.objectives.restructure import (
        RefuseFakeEmbedding,
        RestructureOptions,
        restructure_objectives_doc,
    )
    from lib.ontology.slugs import canonical_slug

    in_path = Path(objectives_path)
    try:
        doc = json.loads(in_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"cannot read objectives JSON {objectives_path!r}: {exc}")
    if not isinstance(doc, dict):
        raise click.ClickException(
            f"objectives JSON must be an object at the top level; got {type(doc).__name__}"
        )
    if _looks_libv2_archive(doc):
        raise click.ClickException(
            "objectives restructure v1 does not accept the LibV2 archive shape "
            "(terminal_outcomes / component_objectives). Feed that doc through "
            "`ed4all run ... --reuse-objectives <path>` — its normalizer handles "
            "the archive form."
        )

    # Chunkset resolution — reuse the planner's loader (one owner).
    from MCP.tools.pipeline_tools import _load_semantik_chunkset_for_planning

    course_slug = canonical_slug(course_name)
    chunks_by_id, all_chunks = _load_semantik_chunkset_for_planning(
        course_slug=course_slug,
        kwargs={"libv2_root": libv2_root} if libv2_root else {},
    )
    if not all_chunks:
        click.secho(
            "warning: no DART chunkset resolved for course "
            f"{course_name!r} (slug {course_slug!r}); the chapter-anchor signal "
            "is unavailable — falling back to a single deterministic terminal.",
            fg="yellow",
            err=True,
        )

    capture = _build_capture(course_name)

    options = RestructureOptions(
        course_name=course_name,
        drop_vacuous=drop_vacuous,
        weeks=weeks,
        generated_from=str(in_path),
        capture=capture,
    )

    try:
        new_doc, report = restructure_objectives_doc(
            doc, chunks_by_id, all_chunks, options=options
        )
    except RefuseFakeEmbedding as exc:
        raise click.ClickException(str(exc))
    except GracefulStopRequested as exc:
        click.secho(
            f"objectives restructure PAUSED at a unit boundary ({exc}); "
            "no output written. Re-run the same command to resume (LLM-free, "
            "lossless recompute).",
            fg="yellow",
            err=True,
        )
        raise SystemExit(3)

    out_path = Path(output) if output else in_path.with_suffix(".restructured.json")
    report_path = out_path.parent / "restructure_report.json"
    out_path.write_text(
        json.dumps(new_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    click.secho(f"Restructured objectives written to {out_path}", fg="green")
    click.echo(
        f"  {report['input_co_count']} CO(s) -> {report['surviving_co_count']} "
        f"(dropped {report['e_merge']['cos_dropped']} near-restatement, "
        f"{len(report['b_vacuity']['dropped_ids'])} vacuous); "
        f"{report['num_tos']} chapter-anchored TO(s); "
        f"duration_weeks={report['duration_weeks']} ({report['week_mode']})."
    )
    click.echo(f"  Report: {report_path}")
    click.echo(
        f"  Reuse it: ed4all run textbook-to-course --reuse-objectives {out_path}"
    )


def _build_capture(course_name: str) -> Optional[Any]:
    """Best-effort DecisionCapture for the restructure pass (never fatal)."""
    try:
        from lib.decision_capture import DecisionCapture

        return DecisionCapture(
            course_code=course_name,
            phase="objectives-restructure",
            tool="courseforge",
            streaming=True,
        )
    except Exception:  # noqa: BLE001 — capture is observability only
        return None


def register_objectives_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all objectives`` command group to the top-level CLI."""
    cli_group.add_command(objectives_group)
