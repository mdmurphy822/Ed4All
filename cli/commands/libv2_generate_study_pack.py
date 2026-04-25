"""``ed4all libv2 generate-study-pack`` (Wave 77 Worker δ).

Pure SQL-over-metadata generator that emits a single coherent study
pack — or, with ``--lesson-plan``, a teacher-facing plan — for one or
more weeks of a LibV2-archived course.

Design tenets:

* **No LLM dependency.** All inputs are structured metadata read from
  ``LibV2/courses/<slug>/`` (chunks + objectives.json + course.json).
* **Read-only against the archive.** The CLI never writes back into the
  archive; it only emits to stdout (or to ``--output``).
* **Reusable engine.** Heavy lifting lives in
  :mod:`LibV2.tools.study_pack_renderer`; this module is a thin Click
  shell.

Examples
--------

    ed4all libv2 generate-study-pack --slug rdf-shacl-550-rdf-shacl-550 --week 7
    ed4all libv2 generate-study-pack --slug X --week 7 --include-exercises
    ed4all libv2 generate-study-pack --slug X --week 7 --lesson-plan
    ed4all libv2 generate-study-pack --slug X --week 7 --format html
    ed4all libv2 generate-study-pack --slug X --week 1,2,3 --format json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import click

from lib.paths import LIBV2_PATH
from LibV2.tools.study_pack_renderer import (
    VALID_DIFFICULTIES,
    StudyPackError,
    render,
)


def _resolve_slug(slug: str, courses_root: Optional[Path] = None) -> Path:
    root = courses_root if courses_root else (LIBV2_PATH / "courses")
    return root / slug


def _parse_weeks(spec: str) -> List[int]:
    """Parse a ``--week`` spec into a sorted, deduped list of week ints.

    Accepts comma-separated integers (``1,2,3``) and ranges (``1-4``).
    """
    if spec is None:
        raise click.BadParameter("--week is required")
    out: List[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            try:
                lo = int(lo_s)
                hi = int(hi_s)
            except ValueError as exc:
                raise click.BadParameter(
                    f"Invalid week range {token!r}: {exc}"
                ) from exc
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(token))
            except ValueError as exc:
                raise click.BadParameter(
                    f"Invalid week value {token!r}: {exc}"
                ) from exc
    if not out:
        raise click.BadParameter("--week must include at least one week")
    return sorted(set(out))


def _parse_difficulties(spec: Optional[str]) -> Optional[List[str]]:
    if not spec:
        return None
    out: List[str] = []
    for token in str(spec).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in VALID_DIFFICULTIES:
            raise click.BadParameter(
                f"Unknown difficulty {token!r}. "
                f"Valid: {', '.join(VALID_DIFFICULTIES)}."
            )
        out.append(token)
    return out or None


@click.command("generate-study-pack")
@click.option(
    "--slug",
    required=True,
    help=(
        "LibV2 course slug. Resolves to LibV2/courses/<slug>/. "
        "Example: rdf-shacl-550-rdf-shacl-550."
    ),
)
@click.option(
    "--week",
    "week_spec",
    required=True,
    help=(
        "Week selector. Single int (e.g. '7'), comma-separated list "
        "('1,2,3'), or range ('1-4')."
    ),
)
@click.option(
    "--include-exercises",
    is_flag=True,
    default=False,
    help="Include exercise chunks (excluded by default).",
)
@click.option(
    "--include-self-check",
    is_flag=True,
    default=False,
    help="Include self-check / quiz chunks (excluded by default).",
)
@click.option(
    "--difficulty",
    "difficulty_spec",
    default=None,
    help=(
        "Comma-separated difficulty filter. "
        "Valid values: foundational, intermediate, advanced."
    ),
)
@click.option(
    "--lesson-plan",
    is_flag=True,
    default=False,
    help=(
        "Render a teacher-facing lesson plan instead of a student "
        "study pack: adds timing estimates, an objective table, and "
        "an assessment items section."
    ),
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["md", "html", "json"]),
    default="md",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file. Defaults to stdout.",
)
@click.option(
    "--courses-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override LibV2 courses root (tests only). Defaults to LibV2/courses/.",
)
def generate_study_pack_command(
    slug: str,
    week_spec: str,
    include_exercises: bool,
    include_self_check: bool,
    difficulty_spec: Optional[str],
    lesson_plan: bool,
    output_format: str,
    output: Optional[Path],
    courses_root: Optional[Path],
) -> None:
    """Generate a study pack (or lesson plan) for a LibV2 course."""
    weeks = _parse_weeks(week_spec)
    difficulties = _parse_difficulties(difficulty_spec)

    archive_root = _resolve_slug(slug, courses_root)

    try:
        _, rendered = render(
            archive_root,
            weeks=weeks,
            output_format=output_format,
            include_exercises=include_exercises,
            include_self_check=include_self_check,
            difficulties=difficulties,
            lesson_plan=lesson_plan,
        )
    except StudyPackError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        click.echo(f"Wrote {output}", err=True)
    else:
        click.echo(rendered, nl=False)


def register_generate_study_pack_command(libv2_group: click.Group) -> None:
    """Attach the ``generate-study-pack`` subcommand to ``ed4all libv2``."""
    libv2_group.add_command(generate_study_pack_command)


__all__ = [
    "generate_study_pack_command",
    "register_generate_study_pack_command",
]
