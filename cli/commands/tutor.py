"""``ed4all tutor`` — misconception-aware tutoring CLI (Wave 77).

Three subcommands wired to ``MCP/tools/tutoring_tools.py``:

* ``ed4all tutor diagnose`` — match free-form student input against the
  archived misconception statements, return top-k matches with the
  matched misconception, the editorial correction, and the originating
  chunk + source_references for downstream citation.
* ``ed4all tutor inventory`` — cluster the corpus's misconceptions
  into semantic groups for human review.
* ``ed4all tutor guardrails`` — given a target concept slug, list the
  misconceptions whose ``interferes_with`` edge points at that concept
  so an LLM can be instructed to avoid them at generation time.

All three subcommands accept ``--format json|text``. JSON is canonical
machine output; ``text`` is a compact human-readable rendering with
truncated long fields. The shared loader caches per (slug, mtime) so
repeated invocations on the same archive are cheap.
"""

from __future__ import annotations

import json
from typing import List

import click

from MCP.tools.tutoring_tools import (
    cluster_misconceptions,
    match_misconception,
    preemptive_misconception_guardrails,
)


def _truncate(text: str, n: int = 160) -> str:
    """Compact text rendering for ``--format text``."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


@click.group(name="tutor")
def tutor_group() -> None:
    """Misconception-aware tutoring tools (LibV2 archives)."""


@tutor_group.command("diagnose")
@click.option("--slug", required=True, help="LibV2 course slug.")
@click.option(
    "--text", "student_text", required=True,
    help="Free-form student utterance to diagnose."
)
@click.option("--top-k", type=int, default=5, show_default=True,
              help="Maximum number of matches to return.")
@click.option(
    "--format", "output_format",
    type=click.Choice(["json", "text"]), default="text", show_default=True,
)
def diagnose_command(slug: str, student_text: str, top_k: int,
                     output_format: str) -> None:
    """Match a student utterance against the corpus's misconception
    statements and return top-k matches with the editorial correction."""
    results = match_misconception(slug, student_text, top_k=top_k)
    if output_format == "json":
        click.echo(json.dumps({"slug": slug, "results": results}, indent=2))
        return

    if not results:
        click.echo(f"No misconception matches for slug={slug!r}.")
        return

    backend = results[0].get("backend", "?")
    click.echo(f"Top {len(results)} match(es) [backend={backend}]:")
    for i, r in enumerate(results, 1):
        click.echo("")
        click.secho(f"  [{i}] score={r['score']:.3f}  chunk={r['chunk_id']}",
                    fg="cyan")
        click.echo(f"      Misconception: {_truncate(r['misconception'])}")
        click.echo(f"      Correction:    {_truncate(r['correction'])}")
        if r.get("concept_tags"):
            tags = ", ".join(r["concept_tags"][:6])
            click.echo(f"      Concept tags:  {tags}")
        srefs: List[dict] = r.get("source_references") or []
        if srefs:
            ids = ", ".join(s.get("sourceId", "?") for s in srefs[:3])
            click.echo(f"      Sources:       {ids}")


@tutor_group.command("inventory")
@click.option("--slug", required=True, help="LibV2 course slug.")
@click.option("--clusters", "n_clusters", type=int, default=8, show_default=True,
              help="Number of clusters to produce.")
@click.option(
    "--format", "output_format",
    type=click.Choice(["json", "text"]), default="text", show_default=True,
)
def inventory_command(slug: str, n_clusters: int, output_format: str) -> None:
    """Cluster the corpus's misconceptions into semantic groups."""
    clusters = cluster_misconceptions(slug, n_clusters=n_clusters)
    if output_format == "json":
        click.echo(json.dumps({"slug": slug, "clusters": clusters}, indent=2))
        return

    if not clusters:
        click.echo(f"No misconceptions found for slug={slug!r}.")
        return

    backend = clusters[0].get("backend", "?")
    total = sum(c["size"] for c in clusters)
    click.echo(
        f"Misconception inventory for {slug}: {len(clusters)} cluster(s), "
        f"{total} member(s) [backend={backend}]"
    )
    for i, c in enumerate(clusters, 1):
        click.echo("")
        click.secho(f"  Cluster {i}  size={c['size']}", fg="cyan")
        click.echo(f"    Label:      {_truncate(c['label'])}")
        click.echo(f"    Correction: {_truncate(c['canonical_correction'])}")
        click.echo("    Members:")
        for m in c["members"][:5]:
            click.echo(f"      - {_truncate(m, 120)}")
        if c["size"] > 5:
            click.echo(f"      ... and {c['size'] - 5} more")


@tutor_group.command("guardrails")
@click.option("--slug", required=True, help="LibV2 course slug.")
@click.option("--concept", required=True,
              help="Target concept slug (e.g. ``rdf-graph``).")
@click.option(
    "--format", "output_format",
    type=click.Choice(["json", "text"]), default="text", show_default=True,
)
def guardrails_command(slug: str, concept: str, output_format: str) -> None:
    """List misconceptions to guard against when explaining ``concept``."""
    results = preemptive_misconception_guardrails(slug, concept)
    if output_format == "json":
        click.echo(json.dumps(
            {"slug": slug, "concept": concept, "guardrails": results},
            indent=2,
        ))
        return

    if not results:
        click.echo(
            f"No misconceptions interfere with concept={concept!r} in slug={slug!r}."
        )
        return

    click.echo(
        f"Pre-emptive guardrails for concept={concept!r} ({len(results)} entries):"
    )
    for i, r in enumerate(results, 1):
        click.echo("")
        click.secho(f"  [{i}] chunk={r.get('chunk_id') or 'N/A'}", fg="cyan")
        click.echo(f"      Avoid: {_truncate(r['misconception'])}")
        if r.get("correction"):
            click.echo(f"      Truth: {_truncate(r['correction'])}")


def register_tutor_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all tutor`` command group to the top-level CLI group."""
    cli_group.add_command(tutor_group)
