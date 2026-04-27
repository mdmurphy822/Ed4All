#!/usr/bin/env python3
"""Trainforge — train_course CLI (Wave 90 — slm-training-2026-04-26).

Top-level entry point for one training run, sibling of
``Trainforge/synthesize_training.py`` and ``Trainforge/process_course.py``.
The training stage is post-import: it consumes an already-imported
LibV2 course and writes ``models/<model_id>/`` back into the same slug.

Wired through the canonical CLI as::

    ed4all run trainforge_train --course-code TST_101 \\
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
    LocalBackend,
    RunPodBackend,
    TrainingRunner,
)


logger = logging.getLogger(__name__)


def _slugify(course_code: str) -> str:
    """Convert ``TST_101`` → ``tst-101`` to match LibV2 slug convention.

    LibV2 imports lowercase the course code and substitute ``-`` for
    ``_``. ``train_course.py`` accepts either form for ergonomics.
    """
    return course_code.strip().lower().replace("_", "-")


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
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Optional YAML file whose top-level keys override the per-base "
        "training config defaults (LR, epochs, rank, etc.)."
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
        config_overrides_path=Path(config_overrides) if config_overrides else None,
    )
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
    """Console-script entry point. Click owns argv parsing."""
    train_course_command()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
