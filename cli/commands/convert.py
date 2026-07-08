"""``ed4all convert`` — the thin accessible-HTML remediation slice (B1).

The DSO wedge: convert a PDF (or a directory of PDFs, or a directory of
publisher HTML) into the canonical accessible-HTML contract
(``{stem}_accessible.html`` + the two sidecars) WITHOUT any course scaffolding.
No ``--course-name``, no workflow run directory, no LibV2 writes, no vector
index — just the conversion seam's output, written into a user-chosen output
directory.

Input detection reuses the pipeline's canonical
``_detect_conversion_input_type`` contract:

* a ``.pdf`` file, or a directory containing PDFs → the SemantiK cascade seam
  (one accessible-HTML document per PDF);
* a ``.html``/``.htm`` file, or a directory of publisher HTML pages (with no
  PDF present) → the vendor-ingest seam (one assembled accessible-HTML
  document);
* anything else → a fail-closed-clear error.

Exit codes: ``0`` all conversions succeeded, ``1`` total failure (nothing
converted / every unit failed / unrecognized input), ``2`` partial success
(some units converted, some failed — per-file failures are reported).

Graceful stop: the SemantiK cascade seam polls the run-scoped + GLOBAL stop
sentinels at its seam boundaries, so ``ed4all stop --all`` reaches an
in-flight PDF conversion even though ``convert`` mints no run id (the global
``STOP_ALL`` sentinel is always published to the cascade). The vendor-ingest
seam is a fast, deterministic single pass and does not poll; interrupt it with
Ctrl-C.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click


def _collect_pdfs(input_path: Path) -> List[Path]:
    """Enumerate the PDF unit(s) for a ``"pdf"``-classified input.

    A single ``.pdf`` file resolves to itself; a directory resolves to every
    ``.pdf`` under it (recursive, to mirror ``_detect_conversion_input_type``'s
    ``rglob`` scan), sorted for deterministic ordering.
    """
    from MCP.tools.pipeline_tools import _PDF_SUFFIXES

    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in _PDF_SUFFIXES
        )
    return []


def _convert_pdf_units(
    input_path: Path,
    out_dir: Path,
    *,
    figures_dir: Optional[str],
    reuse_conversion: bool,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Run the SemantiK cascade seam once per PDF unit; collect results.

    Returns a list of ``(input_pdf, result_dict)`` pairs preserving input
    order. A PDF whose enumeration yields nothing returns an empty list (the
    caller reports "no PDFs found").
    """
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    results: List[Tuple[Path, Dict[str, Any]]] = []
    for pdf in _collect_pdfs(input_path):
        html_output = out_dir / f"{pdf.stem}_accessible.html"
        result = _run_semantik_v2_conversion(
            str(pdf),
            str(html_output),
            figures_dir=figures_dir,
            reuse_conversion=reuse_conversion,
        )
        results.append((pdf, result))
    return results


def _convert_vendor_unit(
    input_path: Path,
    out_dir: Path,
    *,
    doc_title: Optional[str],
    reuse_conversion: bool,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Run the vendor-ingest seam once over the whole HTML input.

    A directory of publisher HTML is assembled into ONE document by the seam
    (matching the pipeline's vendor-ingest behavior), so this always returns a
    single ``(input, result)`` pair.
    """
    from MCP.tools.pipeline_tools import _run_vendor_ingest_conversion

    stem = input_path.name if input_path.is_dir() else input_path.stem
    stem = stem or "vendor"
    html_output = out_dir / f"{stem}_accessible.html"
    result = _run_vendor_ingest_conversion(
        str(input_path),
        str(html_output),
        doc_title=doc_title,
        reuse_conversion=reuse_conversion,
    )
    return [(input_path, result)]


@click.command("convert")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Output directory for the {stem}_accessible.html page(s) + sidecars.",
)
@click.option(
    "--doc-title",
    default=None,
    help="Document title for the vendor-ingest path (defaults to the input "
    "stem, prettified). Ignored for PDF inputs.",
)
@click.option(
    "--figures-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Optional figures directory threaded to the SemantiK cascade seam "
    "(PDF inputs only).",
)
@click.option(
    "--reuse-conversion",
    is_flag=True,
    default=False,
    help="Reuse prior conversion artifacts in the output dir when present "
    "(skips the model-nondeterministic cascade / ingest).",
)
def convert_command(
    input_path: str,
    output_dir: str,
    doc_title: Optional[str],
    figures_dir: Optional[str],
    reuse_conversion: bool,
) -> None:
    """Convert INPUT to accessible HTML in the --output directory.

    INPUT is a PDF, a directory of PDFs, or a directory of publisher HTML.
    This is the standalone remediation slice: NO course is created, NO LibV2
    archive, NO index is built. See docs/operations/convert-verb.md.
    """
    from MCP.tools.pipeline_tools import _detect_conversion_input_type

    in_path = Path(input_path)
    out_dir = Path(output_dir)

    input_type = _detect_conversion_input_type(in_path)
    if input_type == "unknown":
        click.secho(
            f"Unrecognized input {in_path} — expected a .pdf file, a "
            ".html/.htm file, or a directory containing PDFs or accessible "
            "HTML pages.",
            fg="red",
            err=True,
        )
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    if input_type == "pdf":
        results = _convert_pdf_units(
            in_path,
            out_dir,
            figures_dir=figures_dir,
            reuse_conversion=reuse_conversion,
        )
        if not results:
            click.secho(
                f"No PDF files found under {in_path} — nothing converted.",
                fg="yellow",
                err=True,
            )
            raise SystemExit(1)
    else:  # "vendor"
        results = _convert_vendor_unit(
            in_path,
            out_dir,
            doc_title=doc_title,
            reuse_conversion=reuse_conversion,
        )

    succeeded: List[Tuple[Path, Dict[str, Any]]] = []
    failed: List[Tuple[Path, Dict[str, Any]]] = []
    for src, result in results:
        if result.get("success"):
            succeeded.append((src, result))
        else:
            failed.append((src, result))

    total = len(results)
    click.secho(
        f"Converted {len(succeeded)}/{total} input(s) to {out_dir}",
        fg="green" if not failed else "yellow",
    )
    for src, result in succeeded:
        html_path = result.get("html_path") or result.get("output_path", "?")
        click.echo(f"  ok    {src} -> {html_path}")
    for src, result in failed:
        reason = result.get("error", "conversion failed")
        click.secho(f"  FAIL  {src}: {reason}", fg="red", err=True)

    if not succeeded:
        # Total failure: nothing converted.
        raise SystemExit(1)
    if failed:
        # Partial success: some units converted, some failed.
        raise SystemExit(2)


def register_convert_command(cli_group) -> None:
    """Attach ``ed4all convert`` to the top-level Click group. Idempotent."""
    cli_group.add_command(convert_command)
