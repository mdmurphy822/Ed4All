"""Trainforge command-line entry point for adapter training.

The training stage consumes an imported LibV2 course and writes
``models/<model_id>/`` back into the same course archive.

Wired through the canonical CLI as::

    ed4all run trainforge_train --course-name <COURSE_NAME> \\
        --base-model qwen2.5-1.5b

…via :mod:`cli.commands.run`. The product-local command is also available as::

    python -m Trainforge.cli.train_course --course-code <COURSE_CODE> \\
        --base-model qwen2.5-1.5b --dry-run

Training runs in-process through ``LocalBackend``. HF-gated bases require
``HF_TOKEN`` in the environment when actually training.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from lib.generation.stop_control import GracefulStopRequested
from lib.ontology.slugs import libv2_course_slug
from Trainforge.training import (
    BaseModelRegistry,
    ConfigOverrideError,
    LocalBackend,
    TrainingRunner,
    parse_config_overrides,
)

logger = logging.getLogger(__name__)


def _slugify(course_code: str) -> str:
    """Convert ``<COURSE_CODE>`` → ``<course-slug>`` for LibV2 storage.

    Delegates to the canonical ``lib.ontology.slugs.libv2_course_slug`` so the
    resolved slug matches the archive directory ``LibV2/tools/libv2/importer.py``
    created. ``train_course.py`` accepts either the course-code form (``<COURSE_CODE>``)
    or the slug form (``tst-101``) for ergonomics — both resolve identically,
    including inputs that contain spaces or punctuation.
    """
    return libv2_course_slug(course_code)


@click.command("train-course")
@click.option(
    "--course-code",
    required=True,
    help=(
        "Course code (<COURSE_CODE>) or LibV2 slug (<course-slug>). The runner "
        "resolves both via _slugify."
    ),
)
@click.option(
    "--base-model",
    required=True,
    type=click.Choice(BaseModelRegistry.list_supported(), case_sensitive=False),
    help=(
        "Base model short name. Resolved against "
        "Trainforge.training.base_models.BaseModelRegistry."
    ),
)
@click.option(
    "--config-overrides",
    default=None,
    help=(
        "Optional per-run TrainingConfig overrides (LR, epochs, rank, "
        "dpo_learning_rate, ...), as a YAML/JSON file path, an inline JSON "
        "object, or inline key=value[,key=value] pairs (list fields use '|' "
        "between items). Parsed by the SAME canonical parser the pipeline's "
        "training phase uses, so an unknown key or an out-of-range value "
        "fails here rather than mid-run."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Override for the models root. Defaults to "
        "LibV2/courses/<slug>/models/."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Skip the actual trainer call; emit the model card stub + "
        "decision capture only. Useful for exercising the emit path "
        "without a GPU."
    ),
)
def train_course_command(
    course_code: str,
    base_model: str,
    config_overrides: Optional[str],
    output_dir: Optional[str],
    dry_run: bool,
) -> None:
    """Train a course-pinned adapter on top of a LibV2-imported course.

    Example:

    \b
        python -m Trainforge.cli.train_course --course-code <COURSE_CODE> \\
            --base-model qwen2.5-1.5b --dry-run

    Real training runs in-process and requires a CUDA-capable GPU.
    """
    slug = _slugify(course_code)

    # Parse + validate BEFORE any backend or runner exists, so a typo'd
    # override key costs a second rather than a training run.
    try:
        overrides = parse_config_overrides(config_overrides)
    except ConfigOverrideError as exc:
        raise click.BadParameter(
            str(exc), param_hint="--config-overrides"
        ) from exc

    runner = TrainingRunner(
        course_slug=slug,
        base_model=base_model,
        output_dir=Path(output_dir) if output_dir else None,
        backend=LocalBackend(allow_no_gpu=dry_run),
        dry_run=dry_run,
        config_overrides=overrides or None,
    )
    # Let the caller translate a graceful stop into its own paused-state
    # contract. The standalone command maps it to exit code 3 in ``main``.
    result = runner.run()

    click.secho("Training run complete.", fg="green" if not dry_run else "cyan")
    click.echo(f"  Course slug: {slug}")
    click.echo(f"  Base model:  {base_model}")
    click.echo(f"  Model ID:    {result.model_id}")
    click.echo(f"  Run dir:     {result.run_dir}")
    click.echo(f"  Model card:  {result.model_card_path}")
    click.echo(f"  Decisions:   {result.decision_capture_path}")
    if result.adapter_path:
        click.echo(f"  Adapter:     {result.adapter_path}")
    if result.metrics:
        click.echo(f"  Metrics:     {result.metrics}")


def main() -> None:
    """Console-script entry point. Click owns argv parsing.

    Runs the click command in non-standalone mode so a graceful stop
    (``GracefulStopRequested``, raised by the runner when an ``ed4all stop``
    sentinel trips mid-training) surfaces as the canonical paused exit code 3
    with a resume hint, instead of a bare traceback. All other click control
    flow (usage errors, ``--help``, ``Abort``) keeps its standard behavior.
    """
    try:
        train_course_command.main(standalone_mode=False)
    except GracefulStopRequested as stop:
        click.secho(f"\nTraining paused (graceful stop): {stop}", fg="yellow")
        click.echo(
            "The trainer flushed its native checkpoint into the run dir. "
            "Re-run the same command to resume — resume_from_checkpoint "
            "auto-detects the latest checkpoint-*."
        )
        sys.exit(3)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except click.exceptions.Abort:
        click.echo("Aborted!", err=True)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
