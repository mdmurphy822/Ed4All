"""Command-line interface for LibV2."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


def get_repo_root() -> Path:
    """Find the repository root (contains courses/ and catalog/)."""
    # Start from current directory and search upwards
    current = Path.cwd()

    while current != current.parent:
        if (current / "courses").exists() and (current / "catalog").exists():
            return current
        if (current / "CLAUDE.md").exists():
            return current
        current = current.parent

    # Default to current directory
    return Path.cwd()


def print_success(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[green]{msg}[/green]")
    else:
        print(f"SUCCESS: {msg}")


def print_error(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[red]{msg}[/red]")
    else:
        print(f"ERROR: {msg}", file=sys.stderr)


def print_warning(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[yellow]{msg}[/yellow]")
    else:
        print(f"WARNING: {msg}")


@click.group()
@click.option("--repo", "-r", type=click.Path(exists=True), help="Repository root path")
@click.pass_context
def main(ctx, repo: Optional[str]):
    """LibV2 - SLM Model Graph Repository Management"""
    ctx.ensure_object(dict)
    ctx.obj["repo_root"] = Path(repo) if repo else get_repo_root()


@main.command("import")
@click.argument("source", type=click.Path(exists=True))
@click.option("--domain", "-d", required=True, help="Primary domain (e.g., physics, chemistry)")
@click.option("--division", type=click.Choice(["STEM", "ARTS"]), default="STEM", help="Division")
@click.option("--subdomain", "-s", multiple=True, help="Subdomains (can specify multiple)")
@click.option("--topic", "-t", multiple=True, help="Topics (can specify multiple)")
@click.option("--secondary", multiple=True, help="Secondary domains")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing course")
@click.option("--imscc", type=click.Path(exists=True), help="Path to source IMSCC package")
@click.option("--slm-version", help="SLM version used for processing")
@click.option("--slm-specialist", multiple=True, help="SLM specialists used (can specify multiple)")
@click.option("--pdf", type=click.Path(exists=True), help="Path to original PDF source")
@click.option("--html", type=click.Path(exists=True), help="Path to DART accessible HTML")
@click.option("--arxiv-id", help="Arxiv paper ID to load metadata from database")
@click.option("--arxiv-db", type=click.Path(exists=True), help="Path to arxiv papers.db SQLite database")
@click.pass_context
def import_course(ctx, source: str, domain: str, division: str, subdomain: tuple,
                  topic: tuple, secondary: tuple, force: bool, imscc: Optional[str],
                  slm_version: Optional[str], slm_specialist: tuple,
                  pdf: Optional[str], html: Optional[str],
                  arxiv_id: Optional[str], arxiv_db: Optional[str]):
    """Import a course from Sourceforge output."""
    from .importer import import_course as do_import

    repo_root = ctx.obj["repo_root"]
    source_path = Path(source)

    try:
        slug = do_import(
            source_dir=source_path,
            repo_root=repo_root,
            division=division,
            domain=domain,
            subdomains=list(subdomain) if subdomain else None,
            topics=list(topic) if topic else None,
            secondary_domains=list(secondary) if secondary else None,
            force=force,
            imscc_path=Path(imscc) if imscc else None,
            slm_version=slm_version,
            slm_specialists=list(slm_specialist) if slm_specialist else None,
            pdf_path=Path(pdf) if pdf else None,
            html_path=Path(html) if html else None,
            arxiv_id=arxiv_id,
            arxiv_db_path=Path(arxiv_db) if arxiv_db else None,
        )
        print_success(f"Imported course: {slug}")
        print(f"Location: {repo_root / 'courses' / slug}")

        # Offer to rebuild indexes
        if click.confirm("Rebuild indexes?", default=True):
            ctx.invoke(index_rebuild)

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except FileExistsError as e:
        print_error(f"{e} (use --force to overwrite)")
        sys.exit(1)


@main.group("validate")
def validate():
    """Validation commands."""
    pass


@validate.command("all")
@click.pass_context
def validate_all(ctx):
    """Validate all courses in the repository."""
    from .validator import validate_repository

    repo_root = ctx.obj["repo_root"]
    results = validate_repository(repo_root)

    if not results:
        print_warning("No courses found to validate")
        return

    all_valid = True
    for slug, result in results.items():
        if result.valid:
            print_success(f"{slug}: Valid")
        else:
            all_valid = False
            print_error(f"{slug}: Invalid")
            for error in result.errors:
                print(f"  - {error}")

        for warning in result.warnings:
            print_warning(f"  Warning: {warning}")

    if all_valid:
        print_success(f"\nAll {len(results)} courses are valid")
    else:
        invalid_count = sum(1 for r in results.values() if not r.valid)
        print_error(f"\n{invalid_count}/{len(results)} courses have errors")
        sys.exit(1)


@validate.command("course")
@click.argument("slug")
@click.pass_context
def validate_course(ctx, slug: str):
    """Validate a specific course."""
    from .validator import validate_course as do_validate

    repo_root = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug

    if not course_dir.exists():
        print_error(f"Course not found: {slug}")
        sys.exit(1)

    result = do_validate(course_dir, repo_root)

    if result.valid:
        print_success(f"{slug}: Valid")
    else:
        print_error(f"{slug}: Invalid")
        for error in result.errors:
            print(f"  - {error}")
        sys.exit(1)

    for warning in result.warnings:
        print_warning(f"  Warning: {warning}")


@validate.command("indexes")
@click.pass_context
def validate_indexes(ctx):
    """Validate index consistency."""
    from .validator import validate_indexes as do_validate

    repo_root = ctx.obj["repo_root"]
    result = do_validate(repo_root)

    if result.valid:
        print_success("Indexes are consistent")
    else:
        print_error("Index inconsistencies found:")
        for error in result.errors:
            print(f"  - {error}")
        sys.exit(1)

    for warning in result.warnings:
        print_warning(f"  Warning: {warning}")


@main.group("index")
def index():
    """Index management commands."""
    pass


@index.command("rebuild")
@click.pass_context
def index_rebuild(ctx):
    """Rebuild all indexes."""
    from .indexer import rebuild_all_indexes

    repo_root = ctx.obj["repo_root"]
    print("Rebuilding indexes...")

    results = rebuild_all_indexes(repo_root)

    for name, success in results.items():
        if success:
            print_success(f"  {name}: OK")
        else:
            print_error(f"  {name}: Failed")

    print_success("Index rebuild complete")


@main.group("catalog")
def catalog():
    """Catalog commands."""
    pass


@catalog.command("list")
@click.option("--division", type=click.Choice(["STEM", "ARTS"]), help="Filter by division")
@click.option("--domain", "-d", help="Filter by domain")
@click.option("--limit", "-n", type=int, default=50, help="Maximum results")
@click.pass_context
def catalog_list(ctx, division: Optional[str], domain: Optional[str], limit: int):
    """List courses in the catalog."""
    from .catalog import load_master_catalog, search_catalog

    repo_root = ctx.obj["repo_root"]
    catalog = load_master_catalog(repo_root)

    if catalog is None:
        print_warning("No catalog found. Run 'libv2 index rebuild' first.")
        return

    results = search_catalog(catalog, division=division, domain=domain)
    results = results[:limit]

    if not results:
        print("No courses found matching criteria")
        return

    if RICH_AVAILABLE:
        table = Table(title=f"Courses ({len(results)} shown)")
        table.add_column("Slug", style="cyan")
        table.add_column("Title")
        table.add_column("Division")
        table.add_column("Domain")
        table.add_column("Chunks", justify="right")

        for entry in results:
            table.add_row(
                entry.slug,
                entry.title[:40] + "..." if len(entry.title) > 40 else entry.title,
                entry.division,
                entry.primary_domain,
                str(entry.chunk_count),
            )
        console.print(table)
    else:
        for entry in results:
            print(f"{entry.slug}: {entry.title} ({entry.division}/{entry.primary_domain})")


@catalog.command("search")
@click.argument("query")
@click.option("--domain", "-d", help="Filter by domain")
@click.option("--difficulty", help="Filter by difficulty")
@click.pass_context
def catalog_search(ctx, query: str, domain: Optional[str], difficulty: Optional[str]):
    """Search courses by keyword."""
    from .catalog import load_master_catalog, search_catalog

    repo_root = ctx.obj["repo_root"]
    catalog = load_master_catalog(repo_root)

    if catalog is None:
        print_warning("No catalog found. Run 'libv2 index rebuild' first.")
        return

    results = search_catalog(catalog, query=query, domain=domain, difficulty=difficulty)

    if not results:
        print("No courses found matching query")
        return

    for entry in results:
        print(f"{entry.slug}: {entry.title}")
        print(f"  {entry.division}/{entry.primary_domain} | {entry.chunk_count} chunks")


@catalog.command("stats")
@click.pass_context
def catalog_stats(ctx):
    """Show catalog statistics."""
    from .catalog import get_catalog_statistics, load_master_catalog

    repo_root = ctx.obj["repo_root"]
    catalog = load_master_catalog(repo_root)

    if catalog is None:
        print_warning("No catalog found. Run 'libv2 index rebuild' first.")
        return

    stats = get_catalog_statistics(catalog)

    if RICH_AVAILABLE:
        console.print(Panel("[bold]LibV2 Repository Statistics[/bold]"))
        console.print(f"Total Courses: [cyan]{stats['total_courses']}[/cyan]")
        console.print(f"Total Chunks: [cyan]{stats['total_chunks']:,}[/cyan]")
        console.print(f"Total Tokens: [cyan]{stats['total_tokens']:,}[/cyan]")

        console.print("\n[bold]By Division:[/bold]")
        for div, count in stats["by_division"].items():
            console.print(f"  {div}: {count}")

        console.print("\n[bold]By Domain:[/bold]")
        for dom, count in sorted(stats["by_domain"].items(), key=lambda x: -x[1]):
            console.print(f"  {dom}: {count}")
    else:
        print(f"Total Courses: {stats['total_courses']}")
        print(f"Total Chunks: {stats['total_chunks']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print("\nBy Division:")
        for div, count in stats["by_division"].items():
            print(f"  {div}: {count}")


@main.command("retrieve")
@click.argument("query")
@click.option("--domain", "-d", help="Filter by domain")
@click.option("--division", type=click.Choice(["STEM", "ARTS"]), help="Filter by division")
@click.option("--subdomain", "-s", help="Filter by subdomain")
@click.option("--course", "-c", help="Limit to specific course slug")
@click.option("--chunk-type", "-t", help="Filter by chunk type (explanation, example, summary, etc.)")
@click.option("--difficulty", help="Filter by difficulty (foundational, intermediate, advanced)")
@click.option("--concept", multiple=True, help="Filter by concept tag (can specify multiple)")
@click.option("--limit", "-n", type=int, default=10, help="Maximum results (default: 10)")
@click.option("--sample-per-course", type=int, help="Max chunks per course for cross-course search")
@click.option("--output", "-o", type=click.Choice(["text", "json", "jsonld"]), default="text", help="Output format")
# Worker J: reference-retrieval flags
@click.option("--include-rationale", is_flag=True, help="Emit per-result rationale (matched tags/LOs, boost contributions)")
@click.option("--no-metadata-scoring", is_flag=True, help="Disable concept/LO/prereq boosts (pure BM25)")
@click.option("--no-concept-graph-boost", is_flag=True, help="Disable only the concept-graph-overlap boost")
@click.option("--no-lo-boost", is_flag=True, help="Disable only the LO-match boost")
@click.option("--prefer-self-contained", is_flag=True, help="Enable the prereq-coverage boost (off by default)")
@click.option("--lo-filter", multiple=True, help="LO id to boost (repeatable, e.g. --lo-filter co-03)")
@click.option("--week", "week_num", type=int, help="Filter by week number (parses source.module_id)")
@click.option("--teaching-role", help="Filter by teaching_role (transfer, assess, synthesize, ...)")
@click.option("--content-type", "content_type_label", help="Filter by content_type_label")
# Wave 70 RDF-aligned filters
@click.option("--cognitive-domain", help="Filter by cognitive_domain (factual, conceptual, procedural, metacognitive)")
@click.option("--hierarchy-level", type=click.Choice(["terminal", "chapter"]),
              help="Filter by LO hierarchy_level (resolved via learning_outcome_refs against course.json outcomes)")
@click.pass_context
def retrieve(ctx, query: str, domain: Optional[str], division: Optional[str],
             subdomain: Optional[str], course: Optional[str], chunk_type: Optional[str],
             difficulty: Optional[str], concept: tuple, limit: int,
             sample_per_course: Optional[int], output: str,
             include_rationale: bool, no_metadata_scoring: bool,
             no_concept_graph_boost: bool, no_lo_boost: bool,
             prefer_self_contained: bool, lo_filter: tuple,
             week_num: Optional[int], teaching_role: Optional[str],
             content_type_label: Optional[str],
             cognitive_domain: Optional[str],
             hierarchy_level: Optional[str]):
    """Search chunks by keyword with metadata filters.

    Streams chunks without loading entire corpus. Uses TF-IDF ranking.

    Examples:

        libv2 retrieve "flexbox layout" --domain web-development

        libv2 retrieve "accessibility" --course accessibility-in-digital-design

        libv2 retrieve "CSS grid" --chunk-type example --limit 5
    """
    from .retriever import retrieve_chunks

    repo_root = ctx.obj["repo_root"]
    concept_tags = list(concept) if concept else None

    results = retrieve_chunks(
        repo_root=repo_root,
        query=query,
        domain=domain,
        division=division,
        subdomain=subdomain,
        course_slug=course,
        chunk_type=chunk_type,
        difficulty=difficulty,
        concept_tags=concept_tags,
        teaching_role=teaching_role,
        content_type_label=content_type_label,
        week_num=week_num,
        cognitive_domain=cognitive_domain,
        hierarchy_level=hierarchy_level,
        limit=limit,
        sample_per_course=sample_per_course,
        include_rationale=include_rationale,
        metadata_scoring=not no_metadata_scoring,
        use_concept_graph_boost=not no_concept_graph_boost,
        use_lo_match_boost=not no_lo_boost,
        prefer_self_contained=prefer_self_contained,
        lo_filter=list(lo_filter) if lo_filter else None,
    )

    if not results:
        print("No results found.")
        return

    if output == "json":
        import json as json_module
        print(json_module.dumps([r.to_dict() for r in results], indent=2))
    elif output == "jsonld":
        import json as json_module
        # Emit as a JSON array of JSON-LD docs so piping to a JSON-LD
        # processor works. Each element carries its own @context.
        print(json_module.dumps([r.to_jsonld() for r in results], indent=2))
    else:
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {result.score:.3f}) ---")
            print(f"Course: {result.course_slug}")
            print(f"Domain: {result.domain} | Type: {result.chunk_type}")
            if result.source:
                print(f"Module: {result.source.get('module_title', 'N/A')}")
                print(f"Lesson: {result.source.get('lesson_title', 'N/A')}")
            preview = result.text[:300].replace('\n', ' ')
            if len(result.text) > 300:
                preview += "..."
            print(f"Text: {preview}")
            if include_rationale and result.rationale:
                r = result.rationale
                print(f"  bm25={r['bm25_score']:.3f} ngram={r['ngram_score']:.3f} boost={r['metadata_boost']:+.3f}")
                if r["matched_concept_tags"]:
                    print(f"  concept-tags: {', '.join(r['matched_concept_tags'][:6])}")
                if r["matched_lo_refs"]:
                    print(f"  matched LOs: {', '.join(r['matched_lo_refs'])}")

        print(f"\n{len(results)} result(s) found.")


@main.command("multi-retrieve")
@click.argument("query")
@click.option("--domain", "-d", help="Filter by domain")
@click.option("--division", type=click.Choice(["STEM", "ARTS"]), help="Filter by division")
@click.option("--chunk-type", "-t", help="Filter by chunk type")
@click.option("--difficulty", help="Filter by difficulty")
@click.option("--limit", "-n", type=int, default=10, help="Maximum results (default: 10)")
@click.option("--decompose/--no-decompose", default=True, help="Enable query decomposition")
@click.option("--explain", is_flag=True, help="Show decomposition explanation")
@click.option("--output", "-o", type=click.Choice(["text", "json", "jsonld"]), default="text", help="Output format")
# Wave 70 RDF-aligned filters
@click.option("--cognitive-domain", help="Filter by cognitive_domain (factual, conceptual, procedural, metacognitive)")
@click.option("--hierarchy-level", type=click.Choice(["terminal", "chapter"]),
              help="Filter by LO hierarchy_level (resolved via learning_outcome_refs against course.json outcomes)")
@click.pass_context
def multi_retrieve(ctx, query: str, domain: Optional[str], division: Optional[str],
                   chunk_type: Optional[str], difficulty: Optional[str], limit: int,
                   decompose: bool, explain: bool, output: str,
                   cognitive_domain: Optional[str],
                   hierarchy_level: Optional[str]):
    """Multi-query retrieval with query decomposition and RRF fusion.

    Decomposes complex queries into sub-queries, executes them in parallel,
    and fuses results using Reciprocal Rank Fusion (RRF).

    Examples:

        libv2 multi-retrieve "compare UDL and differentiated instruction"

        libv2 multi-retrieve "how does accessibility improve learning" --explain

        libv2 multi-retrieve "define cognitive load theory" --no-decompose
    """
    from .multi_retriever import MultiQueryRetriever

    repo_root = ctx.obj["repo_root"]

    retriever = MultiQueryRetriever(repo_root=repo_root)

    # Show decomposition explanation if requested
    if explain:
        explanation = retriever.explain_decomposition(query)
        if output == "json":
            print(json.dumps(explanation, indent=2))
        else:
            print("\n=== Query Decomposition ===")
            print(f"Original: {explanation['original_query']}")
            print(f"Intent: {explanation['detected_intent']}")
            print(f"Bloom Level: {explanation['detected_bloom_level'] or 'Not detected'}")
            print(f"Concepts: {', '.join(explanation['extracted_concepts']) or 'None'}")
            print(f"Domain Hints: {', '.join(explanation['domain_hints']) or 'None'}")
            print(f"\nSub-queries ({explanation['total_sub_queries']}):")
            for sq in explanation['sub_queries']:
                print(f"  - [{sq['aspect']}] {sq['text']} (weight: {sq['weight']:.2f})")
            print()

    # Execute retrieval
    results = retriever.retrieve(
        query=query,
        limit=limit,
        domain=domain,
        division=division,
        decompose=decompose,
        chunk_type=chunk_type,
        difficulty=difficulty,
        cognitive_domain=cognitive_domain,
        hierarchy_level=hierarchy_level,
    )

    if not results.results:
        print("No results found.")
        return

    if output == "json":
        print(json.dumps(results.to_dict(), indent=2))
    elif output == "jsonld":
        # JSON-LD emit for multi-retrieve: each FusedResult wraps a
        # RetrievalResult-shaped dict. We project to a JSON-LD envelope
        # so downstream consumers get the same @context / @type shape
        # as single-retrieve. Wrap each fused result in an ed4all:
        # RetrievalResult node — the fusion metadata (fused_score,
        # contributing_queries) lives on the envelope under ed4all: predicates.
        from .retriever import RetrievalResult

        jsonld_results = []
        for r in results.results:
            # Rehydrate RetrievalResult so the to_jsonld() projection works.
            rr = RetrievalResult(
                chunk_id=getattr(r, "chunk_id", ""),
                text=getattr(r, "text", ""),
                score=getattr(r, "fused_score", getattr(r, "score", 0.0)),
                course_slug=getattr(r, "course_slug", ""),
                domain=getattr(r, "domain", ""),
                chunk_type=getattr(r, "chunk_type", ""),
                difficulty=getattr(r, "difficulty", None),
                concept_tags=getattr(r, "concept_tags", []) or [],
                source=getattr(r, "source", {}) or {},
                tokens_estimate=getattr(r, "tokens_estimate", 0),
                learning_outcome_refs=getattr(r, "learning_outcome_refs", []) or [],
                bloom_level=getattr(r, "bloom_level", None),
            )
            jsonld_results.append(rr.to_jsonld())
        print(json.dumps(jsonld_results, indent=2))
    else:
        # Show fusion stats
        print(f"\n=== Multi-Query Results ({results.result_count} fused) ===")
        print(f"Method: {results.fusion_method.upper()}")
        if results.deduplication_stats:
            stats = results.deduplication_stats
            print(f"Deduplication: {stats.get('removed', 0)} duplicates removed")
        if results.coherence_metrics:
            coherence = results.coherence_metrics.get('overall', 0)
            print(f"Coherence: {coherence:.1%}")
        print()

        for i, result in enumerate(results.results, 1):
            print(f"--- Result {i} (score: {result.fused_score:.4f}) ---")
            print(f"Course: {result.course_slug}")
            print(f"Domain: {result.domain} | Type: {result.chunk_type}")
            print(f"Contributing queries: {len(result.contributing_queries)}")
            if result.source:
                print(f"Module: {result.source.get('module_title', 'N/A')}")

            # Show first 300 chars
            preview = result.text[:300].replace('\n', ' ')
            if len(result.text) > 300:
                preview += "..."
            print(f"Text: {preview}")
            print()

        print(f"{results.result_count} result(s) found.")


@main.command("info")
@click.argument("slug")
@click.pass_context
def course_info(ctx, slug: str):
    """Show detailed information about a course."""
    from .catalog import load_course_manifest

    repo_root = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug

    if not course_dir.exists():
        print_error(f"Course not found: {slug}")
        sys.exit(1)

    manifest = load_course_manifest(course_dir)
    if manifest is None:
        print_error("Could not load course manifest")
        sys.exit(1)

    if RICH_AVAILABLE:
        console.print(Panel(f"[bold]{manifest.title}[/bold]"))
        console.print(f"Slug: [cyan]{manifest.slug}[/cyan]")
        console.print(f"Division: {manifest.classification.division}")
        console.print(f"Domain: {manifest.classification.primary_domain}")
        if manifest.classification.subdomains:
            console.print(f"Subdomains: {', '.join(manifest.classification.subdomains)}")
        console.print(f"\nChunks: {manifest.content_profile.total_chunks:,}")
        console.print(f"Tokens: {manifest.content_profile.total_tokens:,}")
        console.print(f"Concepts: {manifest.content_profile.total_concepts:,}")
        console.print(f"\nImported: {manifest.import_timestamp}")

        # Source package info
        if manifest.source_package:
            console.print("\n[bold]Source Package:[/bold]")
            console.print(f"  IMSCC: [cyan]{manifest.source_package}[/cyan]")

        # SLM processing info
        if manifest.slm_processing:
            console.print("\n[bold]SLM Processing:[/bold]")
            if manifest.slm_processing.slm_version:
                console.print(f"  Version: [cyan]{manifest.slm_processing.slm_version}[/cyan]")
            console.print(f"  Generation: {manifest.slm_processing.generation}")
            if manifest.slm_processing.specialists_used:
                console.print(f"  Specialists: {', '.join(manifest.slm_processing.specialists_used)}")
            if manifest.slm_processing.processing_timestamp:
                console.print(f"  Processed: {manifest.slm_processing.processing_timestamp}")
    else:
        print(f"Title: {manifest.title}")
        print(f"Slug: {manifest.slug}")
        print(f"Division: {manifest.classification.division}")
        print(f"Domain: {manifest.classification.primary_domain}")
        print(f"Chunks: {manifest.content_profile.total_chunks}")
        print(f"Tokens: {manifest.content_profile.total_tokens}")
        if manifest.source_package:
            print(f"Source IMSCC: {manifest.source_package}")
        if manifest.slm_processing and manifest.slm_processing.slm_version:
            print(f"SLM Version: {manifest.slm_processing.slm_version}")


@main.command("link-outcomes")
@click.argument("slug")
@click.option("--objectives", "-o", type=click.Path(exists=True), required=True,
              help="Path to Courseforge learning_objectives.json")
@click.option("--threshold", "-t", type=float, default=0.15,
              help="Minimum similarity threshold for linking (default: 0.15)")
@click.pass_context
def link_outcomes(ctx, slug: str, objectives: str, threshold: float):
    """Link learning outcomes from Courseforge to course chunks.

    Uses TF-IDF similarity to match learning objectives to chunks.
    Updates course.json with learning outcomes and chunks.json with refs.

    Examples:

        libv2 link-outcomes accessibility-design --objectives /path/to/learning_objectives.json

        libv2 link-outcomes my-course -o objectives.json --threshold 0.2
    """
    from .outcome_linker import link_course_outcomes

    repo_root = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug

    if not course_dir.exists():
        print_error(f"Course not found: {slug}")
        sys.exit(1)

    objectives_path = Path(objectives)

    try:
        stats = link_course_outcomes(
            course_dir=course_dir,
            objectives_path=objectives_path,
            similarity_threshold=threshold,
        )

        print_success(f"Linked learning outcomes for: {slug}")
        print("\nStatistics:")
        print(f"  Outcomes loaded: {stats['outcomes_loaded']}")
        print(f"  Course-level outcomes: {stats['course_level_outcomes']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Chunks linked: {stats['chunks_linked']}")
        print(f"  Coverage: {stats['coverage_percent']}%")

        if stats['coverage_percent'] < 50:
            print_warning(f"\nLow coverage ({stats['coverage_percent']}%). Consider:")
            print("  - Lowering threshold with --threshold 0.1")
            print("  - Reviewing learning objective statements")

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to link outcomes: {e}")
        sys.exit(1)


@main.group("concepts")
def concepts():
    """Concept vocabulary governance commands."""
    pass


@concepts.command("analyze")
@click.argument("slug")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text",
              help="Output format")
@click.pass_context
def concepts_analyze(ctx, slug: str, output: str):
    """Analyze concept vocabulary usage in a course.

    Shows statistics on concept tags including valid/invalid counts,
    taxonomy coverage, and format violations.

    Examples:

        libv2 concepts analyze accessibility-design

        libv2 concepts analyze my-course -o json
    """
    from .concept_vocabulary import analyze_course_concepts

    repo_root = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug

    if not course_dir.exists():
        print_error(f"Course not found: {slug}")
        sys.exit(1)

    try:
        analysis = analyze_course_concepts(course_dir, repo_root)

        if output == "json":
            result = {
                "total_tags": analysis.total_tags,
                "unique_tags": analysis.unique_tags,
                "valid_tags": analysis.valid_tags,
                "invalid_tags": analysis.invalid_tags,
                "in_taxonomy": analysis.in_taxonomy,
                "not_in_taxonomy": analysis.not_in_taxonomy,
                "format_violations": [
                    {"tag": tag, "reason": reason}
                    for tag, reason in analysis.format_violations[:50]
                ],
                "top_tags": [
                    {"tag": tag, "count": count}
                    for tag, count in analysis.top_tags
                ],
                "top_invalid": [
                    {"tag": tag, "reason": reason, "count": count}
                    for tag, reason, count in analysis.top_invalid
                ],
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\nConcept Vocabulary Analysis: {slug}")
            print("=" * 50)
            print(f"Total tags: {analysis.total_tags:,}")
            print(f"Unique tags: {analysis.unique_tags:,}")
            print(f"Valid tags: {analysis.valid_tags:,}")
            print(f"Invalid tags: {analysis.invalid_tags:,}")
            print(f"In taxonomy: {analysis.in_taxonomy:,}")
            print(f"Not in taxonomy: {analysis.not_in_taxonomy:,}")

            # Governance check
            if analysis.unique_tags > 800:
                print_error(f"\nVOCABULARY EXPLOSION: {analysis.unique_tags} unique tags (max 800)")

            if analysis.invalid_tags > 0:
                print_warning(f"\n{analysis.invalid_tags} tags have format violations")

            if analysis.top_tags:
                print("\nTop 10 Tags:")
                for tag, count in analysis.top_tags[:10]:
                    marker = "*" if tag in [t for t, _ in analysis.format_violations] else ""
                    print(f"  {tag}: {count}{marker}")

            if analysis.top_invalid:
                print("\nTop Invalid Tags:")
                for tag, reason, count in analysis.top_invalid[:10]:
                    print(f"  {tag}: {reason} ({count}x)")

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        sys.exit(1)


@concepts.command("clean")
@click.argument("slug")
@click.option("--keep-invalid", is_flag=True, help="Keep invalid tags (normalize only)")
@click.option("--skip-guardrails", is_flag=True, help="Skip cleaning guardrails.json")
@click.option("--dry-run", is_flag=True, help="Show what would be cleaned without changing files")
@click.pass_context
def concepts_clean(ctx, slug: str, keep_invalid: bool, skip_guardrails: bool, dry_run: bool):
    """Clean concept tags in a course.

    Normalizes tags to lowercase-hyphenated format and optionally removes
    invalid tags. Also cleans allowed_topics in guardrails.json.

    Examples:

        libv2 concepts clean accessibility-design

        libv2 concepts clean my-course --dry-run

        libv2 concepts clean my-course --keep-invalid
    """
    from .concept_vocabulary import analyze_course_concepts, clean_course_concepts

    repo_root = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug

    if not course_dir.exists():
        print_error(f"Course not found: {slug}")
        sys.exit(1)

    try:
        # Show what will be cleaned
        analysis = analyze_course_concepts(course_dir, repo_root)

        print(f"\nConcept Cleaning: {slug}")
        print("=" * 50)
        print(f"Invalid tags to {'normalize' if keep_invalid else 'remove'}: {analysis.invalid_tags}")

        if analysis.top_invalid:
            print("\nSample invalid tags:")
            for tag, reason, count in analysis.top_invalid[:5]:
                print(f"  {tag}: {reason} ({count}x)")

        if dry_run:
            print_warning("\nDry run - no changes made")
            return

        # Perform cleaning
        stats = clean_course_concepts(
            course_dir=course_dir,
            repo_root=repo_root,
            remove_invalid=not keep_invalid,
            clean_guardrails=not skip_guardrails,
        )

        print_success("\nCleaning complete!")
        print(f"  Chunks modified: {stats['chunks_modified']}")
        print(f"  Tags removed: {stats['tags_removed']}")
        if "guardrails_topics_removed" in stats:
            print(f"  Guardrails topics removed: {stats['guardrails_topics_removed']}")

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Cleaning failed: {e}")
        sys.exit(1)


@main.group("eval")
def eval_group():
    """Retrieval evaluation commands."""
    pass


@eval_group.command("generate")
@click.argument("slug")
@click.option("--num-queries", "-n", type=int, default=50,
              help="Number of queries to generate (default: 50)")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text",
              help="Output format")
@click.pass_context
def eval_generate(ctx, slug: str, num_queries: int, output: str):
    """Generate an evaluation set for a course.

    Samples chunks and creates queries for retrieval evaluation.
    Saves to quality/eval_set.json.

    Examples:

        libv2 eval generate accessibility-design

        libv2 eval generate my-course -n 30 -o json
    """
    from .eval_generator import generate_and_save_eval_set

    repo_root = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug

    if not course_dir.exists():
        print_error(f"Course not found: {slug}")
        sys.exit(1)

    try:
        eval_set, path = generate_and_save_eval_set(course_dir, num_queries)

        if output == "json":
            print(json.dumps({
                "course_slug": eval_set.course_slug,
                "total_queries": len(eval_set.queries),
                "path": str(path),
                "sample_queries": [
                    {"id": q.query_id, "text": q.query_text}
                    for q in eval_set.queries[:5]
                ],
            }, indent=2))
        else:
            print_success(f"Generated eval set for: {slug}")
            print("\nStatistics:")
            print(f"  Total queries: {len(eval_set.queries)}")
            print(f"  Saved to: {path}")

            print("\nSample queries:")
            for q in eval_set.queries[:5]:
                print(f"  [{q.query_id}] {q.query_text}")

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Generation failed: {e}")
        sys.exit(1)


@eval_group.command("run")
@click.argument("slug")
@click.option("--output", "-o", type=click.Path(), help="Save report to file")
@click.option("--verbose", "-v", is_flag=True, help="Show progress for each query")
@click.option("--format", "-f", "fmt", type=click.Choice(["text", "json"]), default="text",
              help="Output format")
@click.pass_context
def eval_run(ctx, slug: str, output: Optional[str], verbose: bool, fmt: str):
    """Run evaluation against a course's eval set.

    Requires quality/eval_set.json to exist (use 'eval generate' first).
    Saves results to quality/eval_results/.

    Examples:

        libv2 eval run accessibility-design

        libv2 eval run my-course -v -o report.json
    """
    from .eval_harness import run_course_evaluation

    repo_root = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug

    if not course_dir.exists():
        print_error(f"Course not found: {slug}")
        sys.exit(1)

    eval_set_path = course_dir / "quality" / "eval_set.json"
    if not eval_set_path.exists():
        print_error(f"No eval set found. Run 'libv2 eval generate {slug}' first.")
        sys.exit(1)

    try:
        output_path = Path(output) if output else None
        report = run_course_evaluation(
            course_dir=course_dir,
            repo_root=repo_root,
            output_path=output_path,
            verbose=verbose,
        )

        if fmt == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"\nEvaluation Results: {slug}")
            print("=" * 50)
            print(f"Total queries: {report.total_queries}")
            print("\nRetrieval Metrics:")
            print(f"  Hit@1:  {report.hit_at_1:.1%}")
            print(f"  Hit@5:  {report.hit_at_5:.1%}")
            print(f"  Hit@10: {report.hit_at_10:.1%}")
            print(f"  MRR:    {report.mrr:.4f}")
            print(f"  MAP@10: {report.map_at_10:.4f}")
            print("\nLatency:")
            print(f"  Avg: {report.avg_latency_ms:.1f}ms")
            print(f"  Min: {report.min_latency_ms:.1f}ms")
            print(f"  Max: {report.max_latency_ms:.1f}ms")

            # Show warnings for poor metrics
            if report.hit_at_10 < 0.5:
                print_warning(f"\nLow Hit@10 ({report.hit_at_10:.1%}). Consider:")
                print("  - Reviewing chunk quality and metadata")
                print("  - Checking eval set query quality")

            if report.mrr < 0.3:
                print_warning(f"\nLow MRR ({report.mrr:.4f}). Relevant results ranking poorly.")

            # Show failed queries
            failed = [r for r in report.query_results if not r.hit_at_10]
            if failed:
                print(f"\nFailed queries ({len(failed)}):")
                for r in failed[:5]:
                    print(f"  [{r.query_id}] {r.query_text[:50]}...")

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        sys.exit(1)


@eval_group.command("compare")
@click.argument("baseline", type=click.Path(exists=True))
@click.argument("comparison", type=click.Path(exists=True))
@click.pass_context
def eval_compare(ctx, baseline: str, comparison: str):
    """Compare two evaluation reports.

    Detects regressions in retrieval quality.

    Examples:

        libv2 eval compare eval_20240101.json eval_20240115.json
    """
    from .eval_harness import compare_reports

    try:
        result = compare_reports(Path(baseline), Path(comparison))

        print("\nEvaluation Comparison")
        print("=" * 50)
        print(f"Baseline:   {result['baseline']['timestamp']}")
        print(f"Comparison: {result['comparison']['timestamp']}")

        print("\nMetric Changes:")
        for metric, values in result["changes"].items():
            delta = values["delta"]
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"

            # Color coding
            if metric == "avg_latency_ms":
                # For latency, lower is better
                indicator = "" if abs(delta) < 5 else ("" if delta < 0 else "")
            else:
                # For other metrics, higher is better
                indicator = "" if abs(delta) < 0.01 else ("" if delta > 0 else "")

            print(f"  {metric:15} {values['baseline']:.4f} -> {values['comparison']:.4f} ({delta_str}) {indicator}")

        if result["regression_detected"]:
            print_error("\nREGRESSION DETECTED: Significant drop in retrieval quality")
        else:
            print_success("\nNo significant regression detected")

    except Exception as e:
        print_error(f"Comparison failed: {e}")
        sys.exit(1)


@main.command("cross-index")
@click.option("--repo-root", type=click.Path(exists=True, file_okay=False),
              help="Repository root (auto-detected if omitted)")
@click.option("--output", "-o", type=click.Path(),
              help="Output path (default: <repo-root>/LibV2/catalog/cross_package_concepts.json)")
@click.pass_context
def cross_index(ctx, repo_root: Optional[str], output: Optional[str]):
    """Build the cross-package concept index.

    Scans every ``LibV2/courses/*/graph/concept_graph.json`` (and the
    optional Worker-F ``concept_graph_semantic.json``) and emits a catalog
    of which concepts appear across which courses.

    Examples:

        libv2 cross-index

        libv2 cross-index --repo-root /path/to/Ed4All --output catalog.json
    """
    from .cross_package_indexer import write_cross_package_index

    # Precedence: explicit --repo-root wins; otherwise fall back to whatever
    # the top-level ``libv2 --repo`` option (auto-detected by default) resolved.
    if repo_root is not None:
        root = Path(repo_root).resolve()
    else:
        root = Path(ctx.obj["repo_root"]).resolve()

    if output is not None:
        output_path = Path(output)
    else:
        output_path = root / "LibV2" / "catalog" / "cross_package_concepts.json"

    try:
        artifact = write_cross_package_index(root, output_path)
    except Exception as e:  # noqa: BLE001 - surface as CLI error
        print_error(f"Failed to build cross-package index: {e}")
        sys.exit(1)

    print_success(f"Wrote cross-package index: {output_path}")
    print(f"  Courses scanned: {artifact['course_count']}")
    print(f"  Unique concepts: {artifact['concept_count']}")

    # Surface the top concepts so the reviewer can sanity-check without
    # opening the JSON.
    top = list(artifact["concepts"].items())[:5]
    if top:
        print("\nTop concepts by total_courses:")
        for cid, entry in top:
            slugs = ", ".join(c["slug"] for c in entry["courses"])
            print(f"  {cid} ({entry['total_courses']} courses): {slugs}")


@main.command("retrieval-eval")
@click.option("--course", "-c", required=True, help="Course slug to evaluate")
@click.option("--gold-queries", type=click.Path(exists=True), help="Path to gold queries JSONL")
@click.option("--report", type=click.Path(), help="Path to write the evaluation report JSON")
@click.option("--limit", type=int, default=10, help="Retrieval limit per query (default: 10)")
@click.option("--no-rationale", is_flag=True, help="Skip rationale payload in the report")
@click.option("--no-metadata-scoring", is_flag=True, help="Disable concept/LO/prereq boosts")
@click.pass_context
def retrieval_eval(ctx, course: str, gold_queries: Optional[str], report: Optional[str],
                   limit: int, no_rationale: bool, no_metadata_scoring: bool):
    """Run hand-curated gold queries against retrieve_chunks and write a report.

    Reads LibV2/courses/<slug>/retrieval/gold_queries.jsonl by default.
    Writes LibV2/courses/<slug>/retrieval/evaluation_results.json by default.

    \b
    Example:
        libv2 retrieval-eval --course <your-course-slug>
    """
    from .eval_harness import evaluate_retrieval

    repo_root = ctx.obj["repo_root"]
    gold_path = Path(gold_queries) if gold_queries else None
    output_path = Path(report) if report else None

    try:
        rpt = evaluate_retrieval(
            course_slug=course,
            repo_root=repo_root,
            gold_queries_path=gold_path,
            include_rationale=not no_rationale,
            metadata_scoring=not no_metadata_scoring,
            retrieval_limit=limit,
            output_path=output_path,
        )
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    agg = rpt["aggregate"]
    print_success(f"Evaluated {agg['total_queries']} gold queries for {course}")
    print(f"  MRR:       {agg['mrr']:.4f}")
    print(f"  recall@1:  {agg['recall_at_1']:.4f}")
    print(f"  recall@5:  {agg['recall_at_5']:.4f}")
    print(f"  recall@10: {agg['recall_at_10']:.4f}")
    print(f"  avg latency: {agg['avg_latency_ms']:.1f}ms")
    print(f"\n  report: {rpt.get('gold_queries_path')} → evaluation_results.json")


@main.command("retrieval-compare")
@click.option("--course", "-c", required=True, help="Course slug to evaluate")
@click.option(
    "--probe",
    type=click.Path(exists=True),
    help="Path to probe JSON. Default: courses/<slug>/quality/retrieval_probe.json",
)
@click.option(
    "--methods",
    default="bm25,bm25+graph,hybrid",
    help="Comma-separated method presets. Valid: bm25, bm25+graph, bm25+intent, bm25+tag, hybrid",
)
@click.option("--limit", type=int, default=10, help="Retrieval limit per query (default: 10)")
@click.option(
    "--report",
    type=click.Path(),
    help="Path to write the comparison report JSON. Default: courses/<slug>/quality/retrieval_compare_<timestamp>.json",
)
@click.option("--no-save", is_flag=True, help="Print results to stdout only; do not write a report file.")
@click.pass_context
def retrieval_compare(
    ctx,
    course: str,
    probe: Optional[str],
    methods: str,
    limit: int,
    report: Optional[str],
    no_save: bool,
):
    """A/B compare retrieval-method presets over a probe-query set.

    \b
    Example:
        libv2 retrieval-compare --course rdf-shacl-551-2 \\
            --methods bm25,bm25+intent,hybrid

    Probe JSON shape — same as eval-set ``EvalQuery`` (query_id, query_text,
    expected_chunk_ids[], optional chunk_type/difficulty/notes).
    """
    from datetime import datetime as _dt
    from .eval_harness import compare_retrieval_methods

    repo_root = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"Course not found: {course_dir}")
        sys.exit(1)

    probe_path = Path(probe) if probe else (course_dir / "quality" / "retrieval_probe.json")
    if not probe_path.exists():
        print_error(f"Probe set not found: {probe_path}")
        sys.exit(1)

    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    if not method_list:
        print_error("No methods provided.")
        sys.exit(1)

    try:
        result = compare_retrieval_methods(
            repo_root=repo_root,
            course_slug=course,
            probe_path=probe_path,
            methods=method_list,
            retrieval_limit=limit,
        )
    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
        sys.exit(1)

    # Print summary table
    print_success(
        f"Compared {result['total_queries']} probe queries across "
        f"{len(method_list)} method(s) for {course}"
    )
    if RICH_AVAILABLE:
        table = Table(title="Aggregate Metrics", show_header=True)
        table.add_column("method")
        table.add_column("Hit@1", justify="right")
        table.add_column("Hit@5", justify="right")
        table.add_column("Hit@10", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("MAP@10", justify="right")
        table.add_column("ms/q", justify="right")
        for m in method_list:
            agg = result["aggregate"].get(m, {})
            table.add_row(
                m,
                f"{agg.get('hit_at_1', 0):.3f}",
                f"{agg.get('hit_at_5', 0):.3f}",
                f"{agg.get('hit_at_10', 0):.3f}",
                f"{agg.get('mrr', 0):.4f}",
                f"{agg.get('map_at_10', 0):.4f}",
                f"{agg.get('avg_latency_ms', 0):.1f}",
            )
        console.print(table)
    else:
        print()
        header = f"{'method':<14} {'Hit@1':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>7} {'MAP@10':>7} {'ms/q':>7}"
        print(header)
        print("-" * len(header))
        for m in method_list:
            agg = result["aggregate"].get(m, {})
            print(
                f"{m:<14} {agg.get('hit_at_1', 0):>6.3f} "
                f"{agg.get('hit_at_5', 0):>6.3f} {agg.get('hit_at_10', 0):>7.3f} "
                f"{agg.get('mrr', 0):>7.4f} {agg.get('map_at_10', 0):>7.4f} "
                f"{agg.get('avg_latency_ms', 0):>7.1f}"
            )

    # Per-query diff: rows where methods disagree at hit@1
    diff_rows = []
    for row in result["per_query"]:
        h1 = {m: row["results"].get(m, {}).get("hit_at_1", False) for m in method_list}
        if len(set(h1.values())) > 1:
            diff_rows.append((row["query_id"], h1, row["results"]))
    if diff_rows:
        print()
        print(f"Queries with disagreement at Hit@1 ({len(diff_rows)}/{result['total_queries']}):")
        for qid, h1, results in diff_rows:
            cells = " | ".join(f"{m}={'✓' if h1[m] else '✗'}" for m in method_list)
            top1s = " | ".join(
                f"{m}→{results.get(m, {}).get('top1', '?')[-25:]}" for m in method_list
            )
            print(f"  {qid}: {cells}")
            print(f"    top1:  {top1s}")

    # Write the report
    if not no_save:
        if report:
            output_path = Path(report)
        else:
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            output_path = course_dir / "quality" / f"retrieval_compare_{ts}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nReport: {output_path}")


@main.command("ask")
@click.argument("query")
@click.option("--course", "-c", help="Course slug to query (omit for cross-course)")
@click.option("--method", "-m", default="bm25+intent",
              help="Retrieval preset: bm25, bm25+graph, bm25+intent, bm25+tag, hybrid")
@click.option("--limit", "-n", type=int, default=10,
              help="Max chunks to retrieve (capped at 50 by LibV2 policy)")
@click.option("--answer", "-a", "one_shot_answer",
              help="Pre-supply Claude's answer to record in the same step")
@click.option("--force", is_flag=True,
              help="Bypass cache; force fresh retrieval and a new record")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def ask(ctx, query: str, course: Optional[str], method: str, limit: int,
        one_shot_answer: Optional[str], force: bool, output: str):
    """Ask the LibV2 corpus a question; persist Q&A alongside the source.

    Cache-first by default: if an answered record exists for the same
    query (case- and whitespace-normalized), the cached answer is
    returned without re-running retrieval. Pass ``--force`` to bypass
    the cache and create a fresh record.

    Per-course queries land in ``courses/<slug>/queries/``; cross-course
    queries (no ``--course``) land in ``catalog/queries/``. After
    reading the retrieved chunks below, attach a synthesized answer
    with ``libv2 answer <query_id> "<text>"``.

    \b
    Examples:
        libv2 ask "How do I model SHACL property paths?" --course rdf-shacl-551-2
        libv2 ask "compare UDL vs differentiated instruction" --method hybrid
        libv2 ask "How does owl:sameAs entail?" --course rdf-shacl-551-2 --force
    """
    from .retriever import retrieve_chunks
    from .query_log import (
        write_query_record,
        attach_answer,
        compact_retrieval_result,
        find_answered_query,
        load_record,
        query_path,
        resolve_storage_dir,
    )

    if limit > 50:
        print_warning("LibV2 RAG policy caps results at 50; clamping.")
        limit = 50

    repo_root: Path = ctx.obj["repo_root"]
    if course:
        course_dir = repo_root / "courses" / course
        if not course_dir.exists():
            print_error(f"Course not found: {course_dir}")
            sys.exit(1)

    # Cache lookup: a previously-answered record for the same query is
    # returned as-is unless --force is set. The synthesis is the
    # expensive part; retrieval is cheap. Without this, every re-ask
    # would invisibly bypass the stored answers.
    if not force and not one_shot_answer:
        cached = find_answered_query(repo_root, course, query)
        if cached is not None:
            cache_path = query_path(resolve_storage_dir(repo_root, course), cached["query_id"])
            if output == "json":
                cached["_cache_hit"] = True
                print(json.dumps(cached, indent=2))
                return
            print_success(f"Cache hit: {cached['query_id']}")
            print(f"  Path:     {cache_path}")
            print(f"  Asked:    {cached['asked_at']}")
            print(f"  Answered: {cached['answered_at']}")
            print(f"\nQuery: {query}\n")
            chunks = cached.get("retrieved_chunks") or []
            if chunks:
                print(f"Retrieved {len(chunks)} chunk(s) (cached):")
                for c in chunks:
                    print(f"  [{c['rank']}] {c['chunk_id']}  ({c.get('course_slug', '')})")
                    if c.get("section_heading"):
                        print(f"      heading: {c['section_heading']}")
            print(f"\nAnswer:\n{cached['answer']}")
            print("\n(Use --force to bypass cache and create a fresh record.)")
            return

    try:
        results = retrieve_chunks(
            repo_root=repo_root,
            query=query,
            course_slug=course,
            limit=limit,
            method=method,
        )
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)

    compact = [compact_retrieval_result(r, i + 1).to_dict() for i, r in enumerate(results)]
    record_path = write_query_record(
        repo_root=repo_root,
        course_slug=course,
        query_text=query,
        method=method,
        limit=limit,
        retrieved=compact,
    )
    record = load_record(repo_root, course, json.loads(record_path.read_text())["query_id"])
    qid = record["query_id"]

    if one_shot_answer:
        attach_answer(repo_root, course, qid, one_shot_answer)
        record = load_record(repo_root, course, qid)

    if output == "json":
        print(json.dumps(record, indent=2))
        return

    print_success(f"Recorded query: {qid}")
    print(f"  Path:   {record_path}")
    print(f"  Scope:  {'course=' + course if course else 'cross-course'}")
    print(f"  Method: {method} | limit={limit}")
    print(f"\nQuery: {query}\n")

    if not compact:
        print_warning("No chunks retrieved — refine the query or pick a different method.")
    else:
        print(f"Retrieved {len(compact)} chunk(s):")
        for c in compact:
            print(f"\n  [{c['rank']}] score={c['score']:.3f}  {c['chunk_id']}  ({c['course_slug']})")
            if c["section_heading"]:
                print(f"      heading: {c['section_heading']}")
            if c["concept_tags"]:
                print(f"      tags:    {', '.join(c['concept_tags'][:5])}")
            print(f"      {c['snippet']}")

    if record["status"] != "answered":
        scope_flag = f" --course {course}" if course else ""
        print(
            f"\nNext: read the chunks above, then attach your answer:\n"
            f"  libv2 answer {qid}{scope_flag} \"<your synthesized answer>\""
        )


@main.command("answer")
@click.argument("query_id")
@click.argument("answer_text")
@click.option("--course", "-c", help="Course slug the query was scoped to (omit for cross-course)")
@click.pass_context
def answer_cmd(ctx, query_id: str, answer_text: str, course: Optional[str]):
    """Attach Claude's synthesized answer to a previously-asked query."""
    from .query_log import attach_answer

    repo_root: Path = ctx.obj["repo_root"]
    try:
        path = attach_answer(repo_root, course, query_id, answer_text)
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)

    print_success(f"Answer recorded: {query_id}")
    print(f"  Path: {path}")


@main.group("queries")
def queries_group():
    """Browse the Q&A log Claude has built up against LibV2 corpora."""
    pass


@queries_group.command("list")
@click.option("--course", "-c", help="Course slug (omit for cross-course log)")
@click.option("--status", type=click.Choice(["open", "answered", "all"]), default="all")
@click.pass_context
def queries_list(ctx, course: Optional[str], status: str):
    """List queries asked against a corpus (sorted by asked_at)."""
    from .query_log import list_queries

    repo_root: Path = ctx.obj["repo_root"]
    items = list_queries(repo_root, course)
    if status != "all":
        items = [q for q in items if q.get("status") == status]

    if not items:
        scope = f"course={course}" if course else "cross-course"
        print(f"No queries found ({scope}, status={status}).")
        return

    for q in items:
        marker = "[A]" if q.get("status") == "answered" else "[ ]"
        text = (q.get("query_text") or "")[:80]
        print(f"  {marker} {q['query_id']}  {text}")


@queries_group.command("show")
@click.argument("query_id")
@click.option("--course", "-c", help="Course slug the query was scoped to")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def queries_show(ctx, query_id: str, course: Optional[str], output: str):
    """Show a stored Q&A record."""
    from .query_log import load_record

    repo_root: Path = ctx.obj["repo_root"]
    try:
        record = load_record(repo_root, course, query_id)
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)

    if output == "json":
        print(json.dumps(record, indent=2))
        return

    print(f"Query ID:   {record['query_id']}")
    print(f"Status:     {record['status']}")
    print(f"Scope:      {record['scope']}" + (f" ({record['course_slug']})" if record.get("course_slug") else ""))
    print(f"Method:     {record['method']} | limit={record['limit']}")
    print(f"Asked:      {record['asked_at']} by {record['asked_by']}")
    if record.get("answered_at"):
        print(f"Answered:   {record['answered_at']} by {record.get('answered_by') or 'claude'}")
    print(f"\nQuery:\n  {record['query_text']}")
    chunks = record.get("retrieved_chunks") or []
    if chunks:
        print(f"\nRetrieved {len(chunks)} chunk(s):")
        for c in chunks:
            print(f"  [{c['rank']}] {c['chunk_id']} ({c.get('course_slug', '')})")
            if c.get("section_heading"):
                print(f"      heading: {c['section_heading']}")
    if record.get("answer"):
        print(f"\nAnswer:\n{record['answer']}")
    else:
        print("\nAnswer: (open — no answer recorded yet)")


@main.command("export-rdf")
@click.argument("slug")
@click.option(
    "--output-dir", "-o", type=click.Path(file_okay=False),
    help="Output directory (default: courses/<slug>/rdf/)",
)
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["turtle", "trig", "nquads", "ntriples", "xml"]),
    default="turtle",
    help="RDF serialization format (default: turtle)",
)
@click.pass_context
def export_rdf(ctx, slug: str, output_dir: Optional[str], output_format: str):
    """Export a course's JSON artifacts as RDF using the Phase 1 JSON-LD contexts.

    Materializes Turtle (or TriG / N-Quads / etc.) files alongside the
    JSON artifacts so downstream RDF tooling (Protégé, SPARQL stores,
    pyshacl) can ingest the package without a JSON-LD-aware parser.

    Reads the per-artifact @context files from
    ``schemas/context/*_v1.jsonld`` and applies pyld + rdflib to
    materialize the triples.

    \b
    Example:
        libv2 export-rdf rdf-shacl-551-2
        libv2 export-rdf rdf-shacl-551-2 --format trig -o /tmp/rdf-out/
    """
    from .rdf_export import export_course

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug
    if not course_dir.exists():
        print_error(f"Course not found: {course_dir}")
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else course_dir / "rdf"

    try:
        results = export_course(
            repo_root=repo_root,
            course_slug=slug,
            output_dir=out_dir,
            output_format=output_format,
        )
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"RDF export failed: {e}")
        sys.exit(1)

    if not results:
        print_warning(f"No exportable artifacts found under {course_dir}")
        return

    print_success(f"Exported {len(results)} artifact(s) to {out_dir}")
    for r in results:
        print(f"  {r.artifact_relpath} → {r.output_path} ({r.triple_count:,} triples)")


@main.group("models")
def models_group():
    """Manage trained adapters attached to a course.

    Wave 93 — adapters trained by Trainforge land under
    ``courses/<slug>/models/<model_id>/`` alongside ``corpus/``,
    ``graph/``, etc. ``_pointers.json`` records which model_id is
    currently promoted.
    """
    pass


@models_group.command("list")
@click.argument("slug")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def models_list(ctx, slug: str, output: str):
    """List all imported models for a course; star the current one.

    \b
    Example:
        libv2 models list rdf-shacl-551-2
    """
    from .importer import list_course_models

    repo_root: Path = ctx.obj["repo_root"]
    try:
        info = list_course_models(slug, repo_root)
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)

    if output == "json":
        print(json.dumps(info, indent=2))
        return

    models = info.get("models", [])
    current = info.get("current")
    if not models:
        print(f"No models imported for course {slug}.")
        return

    print(f"Models for {slug} (current = {current or '<none>'}):")
    if RICH_AVAILABLE:
        table = Table(show_header=True)
        table.add_column("", justify="center")
        table.add_column("model_id", style="cyan")
        table.add_column("base_model")
        table.add_column("adapter_format")
        table.add_column("created_at")
        table.add_column("faithfulness", justify="right")
        for m in models:
            star = "*" if m.get("is_current") else " "
            base = (m.get("base_model") or {}).get("name", "?")
            scores = m.get("eval_scores") or {}
            faith = scores.get("faithfulness")
            faith_str = f"{faith:.3f}" if isinstance(faith, (int, float)) else "-"
            table.add_row(
                star,
                m.get("model_id", "?"),
                base,
                m.get("adapter_format") or "?",
                m.get("created_at") or "?",
                faith_str,
            )
        console.print(table)
    else:
        for m in models:
            marker = "*" if m.get("is_current") else " "
            base = (m.get("base_model") or {}).get("name", "?")
            print(f"  {marker} {m.get('model_id')}  base={base}  "
                  f"format={m.get('adapter_format')}  created={m.get('created_at')}")


@models_group.command("promote")
@click.argument("slug")
@click.argument("model_id")
@click.option("--promoted-by", help="Optional actor identifier recorded in history")
@click.pass_context
def models_promote(ctx, slug: str, model_id: str, promoted_by: Optional[str]):
    """Flip _pointers.json.current; demote the previous current.

    \b
    Example:
        libv2 models promote rdf-shacl-551-2 qwen2-5-1-5b-rdf-shacl-551-2-3a4f8c92
    """
    from .importer import promote_model

    repo_root: Path = ctx.obj["repo_root"]
    try:
        path = promote_model(slug, model_id, repo_root, promoted_by=promoted_by)
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except ValueError as e:
        print_error(f"Pointer file write rejected: {e}")
        sys.exit(1)

    print_success(f"Promoted: {model_id}")
    print(f"  Pointers: {path}")


@models_group.command("eval")
@click.argument("slug")
@click.argument("model_id")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def models_eval_cmd(ctx, slug: str, model_id: str, output: str):
    """Print the cached eval_report.json for a model.

    Surfaces the report Trainforge.eval.SLMEvalHarness wrote alongside
    the model card. Wave 93 does NOT run a fresh evaluation — the bridge
    from a saved adapter to a model_callable is left as
    ``NotImplementedError`` in the runner (see Wave 92 deferred items
    in ``plans/slm-training-2026-04-26.md``).

    \b
    Example:
        libv2 models eval rdf-shacl-551-2 qwen2-5-1-5b-rdf-shacl-551-2-3a4f8c92
    """
    from .importer import get_model_eval_report

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug
    if not course_dir.exists():
        print_error(f"Course not found: {course_dir}")
        sys.exit(1)
    model_dir = course_dir / "models" / model_id
    if not model_dir.exists():
        print_error(f"Model not found: {model_dir}")
        sys.exit(1)

    report = get_model_eval_report(slug, model_id, repo_root)
    if report is None:
        print_warning(
            f"No eval_report.json found for {model_id}. Evaluation has not "
            f"run for this model — invoke `python -m Trainforge.train_course "
            f"--course-code {slug}` to train and score together; or wire the "
            f"model-callable bridge in a follow-up wave."
        )
        return

    if output == "json":
        print(json.dumps(report, indent=2))
        return

    print(f"Eval report for {slug} / {model_id}:")
    for key in ("faithfulness", "coverage", "baseline_delta", "calibration_ece", "profile"):
        if key in report:
            print(f"  {key}: {report[key]}")
    if "per_tier" in report:
        print("  per_tier:")
        for tier, vals in (report.get("per_tier") or {}).items():
            print(f"    {tier}: {vals}")


@main.command("import-model")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--course", "-c", required=True, help="Course slug to attach the model to")
@click.option("--promote", is_flag=True, help="Promote the new model as current after import")
@click.option("--promoted-by", help="Optional actor identifier recorded in history")
@click.pass_context
def import_model_cmd(ctx, run_dir: str, course: str, promote: bool,
                     promoted_by: Optional[str]):
    """Import a TrainingRunner output dir into a LibV2 course.

    Validates ``model_card.json`` against LibV2ModelValidator (Wave 89);
    fails loud on critical issues. Optionally promotes the new model
    as current with ``--promote``.

    \b
    Example:
        libv2 import-model /path/to/run-dir --course rdf-shacl-551-2 --promote
    """
    from .importer import import_model
    from .validator import ValidationError

    repo_root: Path = ctx.obj["repo_root"]
    try:
        target = import_model(
            course_slug=course,
            run_dir=Path(run_dir),
            repo_root=repo_root,
            promote=promote,
            promoted_by=promoted_by,
        )
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except FileExistsError as e:
        print_error(str(e))
        sys.exit(1)
    except ValidationError as e:
        print_error(f"Model card validation failed: {e}")
        sys.exit(1)

    print_success(f"Imported model into: {target}")
    if promote:
        print(f"  Promoted as current model for course {course!r}.")
    else:
        print("  Run with --promote to set as current.")


if __name__ == "__main__":
    main()
