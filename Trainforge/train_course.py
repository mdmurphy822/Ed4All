#!/usr/bin/env python3
"""Trainforge — train_course CLI (Wave 90 — slm-training-2026-04-26).

Top-level entry point for one training run, sibling of
``Trainforge/synthesize_training.py`` and ``Trainforge/process_course.py``.
The training stage is post-import: it consumes an already-imported
LibV2 course and writes ``models/<model_id>/`` back into the same slug.

Wired through the canonical CLI as::

    ed4all run trainforge_train --course-name TST_101 \\
        --base-model qwen2.5-1.5b

…via :mod:`cli.commands.run`. This module also functions as a direct
script::

    python -m Trainforge.train_course --course-code TST_101 \\
        --base-model qwen2.5-1.5b --dry-run

Wave 90 ships dry-run + LocalBackend. RunPod backend is stubbed
(``--backend runpod`` will fail loud); HF-gated bases (Llama, Phi)
require ``HF_TOKEN`` in the environment when actually training.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click

# Make project root importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trainforge.training import (  # noqa: E402
    BaseModelRegistry,
    ConfigOverrideError,
    LocalBackend,
    RunPodBackend,
    TrainingRunner,
    parse_config_overrides,
)
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402
from lib.ontology.slugs import libv2_course_slug  # noqa: E402


logger = logging.getLogger(__name__)


def _slugify(course_code: str) -> str:
    """Convert ``TST_101`` → ``tst-101`` to match the LibV2 slug convention.

    Delegates to the canonical ``lib.ontology.slugs.libv2_course_slug`` so the
    resolved slug matches the archive directory ``LibV2/tools/libv2/importer.py``
    created. ``train_course.py`` accepts either the course-code form (``TST_101``)
    or the slug form (``tst-101``) for ergonomics — both resolve identically.
    The prior local implementation only swapped ``_``→``-`` and left spaces /
    punctuation intact, so it diverged from the importer on any non-code input.
    """
    return libv2_course_slug(course_code)


@click.command("train-course")
@click.option(
    "--course-code",
    required=True,
    help=(
        "Course code (TST_101) or LibV2 slug (tst-101). The runner "
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
    "--backend",
    type=click.Choice(["local", "runpod"], case_sensitive=False),
    default="local",
    help="Compute backend. 'local' requires a CUDA GPU; 'runpod' is stubbed (Wave 90 follow-up).",
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
    backend: str,
    output_dir: Optional[str],
    dry_run: bool,
) -> None:
    """Train a course-pinned adapter on top of a LibV2-imported course.

    Example:

    \b
        python -m Trainforge.train_course --course-code TST_101 \\
            --base-model qwen2.5-1.5b --dry-run

    Modes:

    \b
      local   (default)  Runs in-process; needs a CUDA GPU.
      runpod              STUBBED — full RunPod dispatch lands in a
                          follow-up wave.
    """
    slug = _slugify(course_code)

    # Parse + validate BEFORE any backend or runner exists, so a typo'd
    # override key costs a second rather than a training run.
    try:
        overrides = parse_config_overrides(config_overrides)
    except ConfigOverrideError as exc:
        raise click.BadParameter(str(exc), param_hint="--config-overrides")

    backend_choice = (backend or "local").lower()
    backend_obj = (
        LocalBackend(allow_no_gpu=dry_run)
        if backend_choice == "local"
        else RunPodBackend()
    )

    runner = TrainingRunner(
        course_slug=slug,
        base_model=base_model,
        output_dir=Path(output_dir) if output_dir else None,
        backend=backend_obj,
        dry_run=dry_run,
        config_overrides=overrides or None,
    )
    # NB: a graceful stop (``ed4all stop`` sentinel tripping mid-training) makes
    # ``runner.run()`` raise ``GracefulStopRequested``. It is deliberately NOT
    # caught here: on the in-process ``ed4all run trainforge_train`` path the
    # executor's Wave-A carve-out catches it and marks the phase ``paused``.
    # The standalone ``python -m Trainforge.train_course`` path converts it to
    # the canonical paused exit code 3 in :func:`main`.
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
