"""``ed4all harvest-bloom-labels`` — deterministic Bloom-label harvester.

Walks a Courseforge project/export (and, optionally, a LibV2 course dir) and
harvests every artifact-asserted Bloom label — the generator's OWN declared
``bloom_level`` on synthesized objectives, outline/rewrite blocks, and
assessment items — into a de-duplicated JSONL store
(``state/bloom_labels/labels.jsonl`` by default).

No LLM, no model load, no network — a pure read of already-emitted artifacts.
The store is the corpus behind voter 1 of the re-founded
``bloom_classifier_disagreement`` signal ("does the generator's own Bloom claim
survive independent checks").
"""
from __future__ import annotations

import click


@click.command("harvest-bloom-labels")
@click.argument(
    "project_path", type=click.Path(exists=True, file_okay=True, dir_okay=True)
)
@click.option(
    "--course-path",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Optional additional LibV2 course dir to harvest (archive-form "
    "objectives / assessments).",
)
@click.option(
    "--store",
    "store_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Override the default state/bloom_labels/labels.jsonl store.",
)
@click.option(
    "--run-id",
    default=None,
    help="Provenance run id stamped on each row (defaults to $ED4ALL_RUN_ID "
    "or 'unknown').",
)
@click.option(
    "--model-provenance",
    default=None,
    help="Force the model_provenance field (else resolved from artifact "
    "metadata, else 'unknown').",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Compute + report counts but write NOTHING to the store.",
)
def harvest_bloom_labels_command(
    project_path: str,
    course_path: str | None,
    store_path: str | None,
    run_id: str | None,
    model_provenance: str | None,
    dry_run: bool,
) -> None:
    """Harvest artifact-asserted Bloom labels from PROJECT_PATH into the store."""
    from lib.bloom_labels import harvest_bloom_labels

    try:
        result = harvest_bloom_labels(
            project_path,
            course_path=course_path,
            store_path=store_path,
            run_id=run_id,
            model_provenance=model_provenance,
            dry_run=dry_run,
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        raise click.ClickException(f"bloom-label harvest failed: {exc}")

    prefix = "[dry-run] " if result.dry_run else ""
    click.secho(
        f"{prefix}Harvested {result.rows_added} new label(s) "
        f"({result.duplicates_skipped} duplicate(s) skipped)",
        fg="green",
    )
    for kind in ("objective", "block", "assessment_item"):
        count = result.per_artifact_counts.get(kind, 0)
        click.echo(f"  {kind}: {count}")
    if result.malformed_skipped:
        click.secho(
            f"  warning: {result.malformed_skipped} malformed row(s) skipped",
            fg="yellow",
            err=True,
        )
    if result.missing_artifacts:
        click.secho(
            f"  note: no artifacts found for: "
            f"{', '.join(result.missing_artifacts)}",
            fg="yellow",
            err=True,
        )
    click.echo(f"  store: {result.store_path}" + (" (not written)" if dry_run else ""))


def register_harvest_bloom_labels_command(cli_group: click.Group) -> None:
    """Attach ``ed4all harvest-bloom-labels`` to the top-level CLI."""
    cli_group.add_command(harvest_bloom_labels_command)
