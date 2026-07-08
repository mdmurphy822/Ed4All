"""``ed4all import-docs`` — Markdown/docs-tree → clean accessible HTML corpus.

Deterministic, LLM-free importer for the corporate-docs beachhead. Walks a
Markdown / MDX documentation tree (honoring an ``mkdocs.yml`` nav for reading
order when present, degrading to a sorted walk otherwise) and writes one clean
``{slug}.html`` page per document into an output directory plus an
``import_manifest.json`` provenance record.

Pass the output directory straight to ``ed4all run textbook-to-course
--corpus <DIR>``: the ``dart_conversion`` phase classifies a directory of plain
``*.html`` as a vendor corpus and re-normalizes it into the canonical
``{stem}_accessible.html`` contract itself — no forged sidecars, no PDF/OCR
step, no LLM.
"""
from __future__ import annotations

from pathlib import Path

import click


@click.command("import-docs")
@click.argument("src_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Output directory for the emitted *.html pages + import_manifest.json.",
)
@click.option(
    "--source-name",
    default=None,
    help="Human name of the source corpus (recorded in the manifest; "
    "defaults to the source directory name).",
)
@click.option(
    "--license-note",
    default=None,
    help="Free-text license/attribution note recorded in the manifest.",
)
@click.option(
    "--provenance-tag",
    default="docs-import",
    show_default=True,
    help="Short provenance tag stamped on the manifest + section provenance.",
)
def import_docs_command(
    src_dir: str,
    output_dir: str,
    source_name: str | None,
    license_note: str | None,
    provenance_tag: str,
) -> None:
    """Import the docs tree at SRC_DIR into clean accessible HTML pages.

    Then run the pipeline over the output directory:

        ed4all run textbook-to-course --corpus <OUTPUT> --course-name <NAME>
    """
    from lib.importers import import_docs_corpus

    try:
        result = import_docs_corpus(
            src_dir,
            output_dir,
            source_name=source_name,
            license_note=license_note,
            provenance_tag=provenance_tag,
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"docs import failed: {exc}")

    if result.doc_count == 0:
        click.secho(
            f"No Markdown documents found under {src_dir!r} — nothing written.",
            fg="yellow",
            err=True,
        )
        raise SystemExit(1)

    total_leaks = sum(1 for d in result.documents if d.get("leaks"))
    click.secho(
        f"Imported {result.doc_count} document(s) to {result.output_dir}",
        fg="green",
    )
    click.echo(
        f"  ordering: {result.ordering_source}"
        + (f" ({result.orphan_count} orphan(s) swept)" if result.orphan_count else "")
    )
    click.echo(f"  manifest: {result.manifest_path}")
    if total_leaks:
        # Non-fatal: surfaces escaped-markup pages for inspection.
        click.secho(
            f"  warning: {total_leaks} page(s) carry escaped-markup leak markers "
            "(see manifest 'leaks').",
            fg="yellow",
            err=True,
        )
    out_path = Path(result.output_dir)
    click.echo(
        f"  Next: ed4all run textbook-to-course --corpus {out_path} "
        "--course-name <NAME>"
    )


def register_import_docs_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all import-docs`` command to the top-level CLI."""
    cli_group.add_command(import_docs_command)
