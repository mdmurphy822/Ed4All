"""``ed4all libv2 generate-quiz`` (Wave 77 Worker γ).

Bloom-balanced assessment generator. Reads existing
``assessment_item`` chunks from a LibV2 archive and emits a quiz in
``json``, ``md``, ``qti``, or ``imscc`` format.

The default deterministic path is LLM-free: it samples the
``assessment_item`` chunks with a seeded ``random.Random`` and
optionally attaches misconception statements (from the same archive)
as principled distractor seeds. An LLM transformation pass is
reserved as future scope — for now, sampled items are emitted AS-IS.

See :mod:`MCP.tools.quiz_generator` for the engine.

Examples
--------

::

    # Deterministic 10-question quiz with bloom-balanced sampling
    ed4all libv2 generate-quiz \\
        --slug rdf-shacl-550-rdf-shacl-550 \\
        --bloom-mix '{"remember":1,"understand":2,"apply":5,"analyze":1,"create":1}' \\
        --seed 42

    # Filter by outcomes + use misconceptions as distractor seeds
    ed4all libv2 generate-quiz --slug X \\
        --bloom-mix '{"apply":3}' \\
        --outcomes co-15,co-16,to-04 \\
        --use-misconceptions-as-distractors \\
        --format md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from lib.paths import LIBV2_PATH
from MCP.tools.quiz_generator import (
    ArchiveNotFoundError,
    BloomMixShortageError,
    QuizGenerator,
)

# Re-use the libv2 group from the validate-packet command so both
# subcommands appear under ``ed4all libv2 ...``.
from cli.commands.libv2_validate_packet import libv2_group


def _resolve_slug(slug: str, courses_root: Optional[Path] = None) -> Path:
    root = courses_root if courses_root else (LIBV2_PATH / "courses")
    return root / slug


def _parse_csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def _parse_bloom_mix(value: str) -> dict[str, int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(
            f"--bloom-mix must be valid JSON; got {value!r} ({exc})"
        )
    if not isinstance(parsed, dict):
        raise click.BadParameter(
            "--bloom-mix must be a JSON object mapping bloom level -> count"
        )
    out: dict[str, int] = {}
    for key, count in parsed.items():
        if not isinstance(key, str):
            raise click.BadParameter(
                f"--bloom-mix keys must be strings; got {key!r}"
            )
        try:
            n = int(count)
        except (TypeError, ValueError):
            raise click.BadParameter(
                f"--bloom-mix values must be integers; got {count!r} for {key!r}"
            )
        if n < 0:
            raise click.BadParameter(
                f"--bloom-mix counts must be >= 0; got {n} for {key!r}"
            )
        out[key.strip().lower()] = n
    if not any(out.values()):
        raise click.BadParameter(
            "--bloom-mix must include at least one positive count"
        )
    return out


@libv2_group.command("generate-quiz")
@click.option(
    "--slug",
    required=True,
    help="LibV2 course slug. Resolves to LibV2/courses/<slug>/.",
)
@click.option(
    "--bloom-mix",
    required=True,
    help=(
        'JSON object mapping bloom level -> count, e.g. '
        '\'{"remember":2,"understand":4,"apply":3,"analyze":1}\'.'
    ),
)
@click.option(
    "--outcomes",
    default=None,
    help="Comma-separated learning outcome refs (e.g. co-15,co-16,to-04).",
)
@click.option(
    "--difficulty",
    default=None,
    help="Comma-separated difficulty levels (foundational,intermediate,advanced).",
)
@click.option(
    "--use-misconceptions-as-distractors",
    is_flag=True,
    help=(
        "For each emitted item, attach misconception statements that "
        "share concept_tags or learning_outcome_refs as distractor seeds."
    ),
)
@click.option(
    "--num-distractors",
    type=int,
    default=3,
    show_default=True,
    help="Maximum misconception distractors per item.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Seed for deterministic sampling.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "md", "qti", "imscc"]),
    default="json",
    show_default=True,
    help="Output format. 'imscc' requires --output.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Write result to this path instead of stdout. Required for "
        "--format imscc."
    ),
)
@click.option(
    "--courses-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override LibV2 courses root (tests only).",
)
def generate_quiz_command(
    slug: str,
    bloom_mix: str,
    outcomes: Optional[str],
    difficulty: Optional[str],
    use_misconceptions_as_distractors: bool,
    num_distractors: int,
    seed: Optional[int],
    output_format: str,
    output: Optional[Path],
    courses_root: Optional[Path],
) -> None:
    """Generate a bloom-balanced quiz from a LibV2 archive."""
    parsed_mix = _parse_bloom_mix(bloom_mix)
    parsed_outcomes = _parse_csv(outcomes)
    parsed_difficulty = _parse_csv(difficulty)

    archive_root = _resolve_slug(slug, courses_root)

    try:
        engine = QuizGenerator.from_archive(archive_root)
    except ArchiveNotFoundError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    try:
        quiz = engine.generate(
            bloom_mix=parsed_mix,
            outcomes=parsed_outcomes,
            difficulty=parsed_difficulty,
            use_misconceptions_as_distractors=(
                use_misconceptions_as_distractors
            ),
            num_distractors=num_distractors,
            seed=seed,
        )
    except BloomMixShortageError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    if output_format == "imscc":
        if output is None:
            click.echo(
                "error: --format imscc requires --output <path>.imscc",
                err=True,
            )
            sys.exit(1)
        engine.write_imscc(quiz, output)
        click.echo(f"Wrote IMSCC quiz package to {output}")
        return

    if output_format == "json":
        rendered = engine.format_json(quiz)
    elif output_format == "md":
        rendered = engine.format_md(quiz)
    else:  # qti
        rendered = engine.format_qti(quiz)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        click.echo(f"Wrote {output_format} quiz to {output}")
    else:
        click.echo(rendered)


def register_libv2_generate_quiz_command(cli_group: click.Group) -> None:
    """No-op kept for symmetry — the subcommand is attached at import time.

    ``cli.commands.libv2_validate_packet.libv2_group`` is the shared
    Click group, and importing this module registers
    ``generate-quiz`` on it. We expose ``register_libv2_command`` from
    :mod:`cli.commands` (which already attaches the group to the root
    CLI) so callers don't need to wire anything new.
    """
    # The decorator above already attached the command. We keep this
    # function so the registration symmetry is documented.
    return None


__all__ = [
    "generate_quiz_command",
    "register_libv2_generate_quiz_command",
]
