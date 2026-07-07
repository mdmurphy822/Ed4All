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


# WS2 — typed semantic-retrieval failures the CLI converts into a non-zero
# exit with operator guidance (NEVER a silent BM25 fallback). Imported at
# module load; ValueError covers engine-misuse (unknown engine, semantic
# without --course, method+engine combos).
try:  # pragma: no cover — import shape, not behavior
    from .vector_index import SemanticIndexError

    try:
        from lib.embedding.providers import EmbeddingBackendUnavailable
        _SEMANTIC_ERRORS = (SemanticIndexError, EmbeddingBackendUnavailable, ValueError)
    except Exception:
        _SEMANTIC_ERRORS = (SemanticIndexError, ValueError)
except Exception:
    _SEMANTIC_ERRORS = (ValueError,)


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


def print_error(msg: str, *, markup: bool = True) -> None:
    if RICH_AVAILABLE:
        # ``markup=False`` preserves literal square brackets (e.g. the
        # "[embedding]" extra name) that rich would otherwise interpret as
        # console markup and swallow. The colour is applied via style= so the
        # red rendering survives even with markup disabled.
        if markup:
            console.print(f"[red]{msg}[/red]")
        else:
            console.print(msg, style="red", markup=False)
    else:
        print(f"ERROR: {msg}", file=sys.stderr)


def print_warning(msg: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[yellow]{msg}[/yellow]")
    else:
        print(f"WARNING: {msg}")


def _fail_semantic_deps_missing(exc: ImportError) -> None:
    """Translate a missing-``[embedding]``-extra ImportError on the semantic
    code path into the SAME typed fail-closed guidance an operator gets for a
    missing index, then exit 1.

    The semantic/hybrid retrieval modules (``vector_index`` /
    ``semantic_retriever``) ``import numpy`` (and sentence-transformers) at
    module top; those live ONLY in the ``[embedding]`` pyproject extra. On a
    deps-slim box the bare ``ModuleNotFoundError`` would otherwise crash with an
    opaque stack trace instead of the actionable, fail-closed message the
    operator deserves. NEVER a silent BM25 fallback.
    """
    print_error(f"{type(exc).__name__}: {exc}", markup=False)
    print_error(
        "The semantic / hybrid-rrf retrieval engines require the [embedding] "
        "extra (numpy + sentence-transformers). Install it "
        "(pip install -e '.[embedding]') and run `libv2 vector-index build "
        "--course <slug>` before requesting a semantic engine. "
        "(Lexical / BM25 retrieval needs no extra.)",
        markup=False,
    )
    sys.exit(1)


@click.group()
@click.option("--repo", "-r", type=click.Path(exists=True), help="Repository root path")
@click.pass_context
def main(ctx, repo: Optional[str]):
    """LibV2 - SLM Model Graph Repository Management"""
    ctx.ensure_object(dict)
    ctx.obj["repo_root"] = Path(repo) if repo else get_repo_root()
    # Track whether the operator pinned an explicit repo root so destructive
    # subcommands (remove) can prefer the canonical ED4ALL_LIBV2_ROOT resolver
    # when --repo is absent, but still honor an explicit override.
    ctx.obj["repo_explicit"] = repo is not None


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


@catalog.command("backfill")
@click.pass_context
def catalog_backfill(ctx):
    """Enumerate every archived course dir into the master catalog (repair).

    W0.8 repair entry point: walks every ``courses/<slug>/manifest.json`` and
    upserts a catalog entry, merging with the existing catalog. Idempotent and
    lock-guarded — safe to run while archives are in flight.
    """
    from .catalog import backfill_master_catalog

    repo_root = ctx.obj["repo_root"]
    print("Backfilling master catalog from courses/ ...")
    summary = backfill_master_catalog(repo_root)
    print_success(
        f"Catalog backfill complete: discovered={summary['discovered']} "
        f"added={summary['added']} updated={summary['updated']} "
        f"total={summary['total']}"
    )


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
# WS2 — retrieval engine axis (orthogonal to boost presets).
@click.option("--engine", type=click.Choice(["lexical", "semantic", "hybrid-rrf"]),
              default="lexical",
              help="Retrieval engine: lexical BM25 (default), real semantic "
                   "nearest-neighbor over the vector index, or RRF hybrid. "
                   "semantic/hybrid-rrf require --course and a built index "
                   "(libv2 vector-index build); they fail closed (no BM25 "
                   "fallback) when the index is missing/stale.")
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
             hierarchy_level: Optional[str],
             engine: str):
    """Search chunks by keyword with metadata filters.

    Streams chunks without loading entire corpus. Uses TF-IDF ranking.

    Examples:

        libv2 retrieve "flexbox layout" --domain web-development

        libv2 retrieve "accessibility" --course accessibility-in-digital-design

        libv2 retrieve "CSS grid" --chunk-type example --limit 5

        libv2 retrieve "define a SHACL NodeShape" --course demo-course-1 --engine semantic
    """
    from .retriever import retrieve_chunks

    repo_root = ctx.obj["repo_root"]
    concept_tags = list(concept) if concept else None

    try:
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
            engine=engine,
        )
    except ImportError as exc:
        # Missing [embedding] extra on the semantic path → typed guidance.
        _fail_semantic_deps_missing(exc)
    except _SEMANTIC_ERRORS as exc:
        # Fail closed with the operator-facing guidance the typed error
        # carries; NEVER silently fall back to BM25 output.
        print_error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

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
                eng = r.get("engine", "lexical")
                if eng == "semantic":
                    print(f"  engine=semantic cosine={r['cosine']:.4f} model={r['model_id']}")
                elif eng == "hybrid-rrf":
                    print(
                        f"  engine=hybrid-rrf rrf={r['rrf_score']:.4f} "
                        f"(lex_rank={r['lexical_rank']} sem_rank={r['semantic_rank']})"
                    )
                else:
                    # Lexical rationale (BM25 + boosts).
                    print(f"  bm25={r['bm25_score']:.3f} ngram={r['ngram_score']:.3f} boost={r['metadata_boost']:+.3f}")
                    if r.get("matched_concept_tags"):
                        print(f"  concept-tags: {', '.join(r['matched_concept_tags'][:6])}")
                    if r.get("matched_lo_refs"):
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
@click.option("--course", "-c", help="Course slug scope (required for non-lexical engines)")
@click.option("--engine", type=click.Choice(["lexical", "semantic", "hybrid-rrf"]),
              default="lexical",
              help="Retrieval engine for every sub-query: lexical BM25 "
                   "(default), semantic, or hybrid-rrf. Non-lexical engines "
                   "require --course and a built vector index; the index is "
                   "pre-flighted before sub-query dispatch so a missing/stale "
                   "index fails closed (never swallowed, never BM25).")
@click.pass_context
def multi_retrieve(ctx, query: str, domain: Optional[str], division: Optional[str],
                   chunk_type: Optional[str], difficulty: Optional[str], limit: int,
                   decompose: bool, explain: bool, output: str,
                   cognitive_domain: Optional[str],
                   hierarchy_level: Optional[str],
                   course: Optional[str], engine: str):
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

    retriever = MultiQueryRetriever(repo_root=repo_root, course_slug=course)

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
    try:
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
            course_slug=course,
            engine=engine,
        )
    except ImportError as exc:
        _fail_semantic_deps_missing(exc)
    except _SEMANTIC_ERRORS as exc:
        print_error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

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


@main.command("remove")
@click.argument("slug")
@click.option("--yes", "-y", is_flag=True, help="Skip the interactive confirmation prompt.")
@click.pass_context
def remove_course_cmd(ctx, slug: str, yes: bool):
    """Permanently delete a course from the LibV2 repository.

    Resolves ``courses/<slug>`` under the LibV2 root (honoring
    ED4ALL_LIBV2_ROOT / ED4ALL_HOME), prints a summary (disk size + top-level
    contents), and requires ``--yes`` (or an interactive ``yes`` confirmation)
    before deleting. Also drops the slug from any derived catalog files so the
    catalog never references the vanished course.

    \b
    Refuses: a missing course, a slug escaping the courses root, and an
    empty/bare slug. DESTRUCTIVE — there is no undo.

    \b
    Examples:
        libv2 remove demo-course-1
        libv2 remove demo-course-1 --yes
    """
    from .remove import (
        CourseRemovalError,
        human_size,
        remove_course,
        resolve_course_dir,
        summarize_course,
    )

    # Honor the canonical resolver (ED4ALL_LIBV2_ROOT / ED4ALL_HOME) unless the
    # operator pinned an explicit --repo on the top-level group.
    if ctx.obj.get("repo_explicit"):
        libv2_root = ctx.obj["repo_root"]
    else:
        try:
            from lib.paths import libv2_path  # noqa: PLC0415

            libv2_root = libv2_path()
        except Exception:  # pragma: no cover — fall back to the auto-detected root
            libv2_root = ctx.obj["repo_root"]

    try:
        course_dir = resolve_course_dir(libv2_root, slug)
    except CourseRemovalError as exc:
        print_error(f"{exc.code}: {exc.detail}")
        sys.exit(1)

    disk_bytes, top_level = summarize_course(course_dir)

    print(f"Course:    {slug}")
    print(f"Location:  {course_dir}")
    print(f"Disk size: {human_size(disk_bytes)} ({disk_bytes:,} bytes)")
    if top_level:
        print(f"Contents:  {', '.join(top_level)}")
    print_warning("\nThis permanently deletes the course directory. There is no undo.")

    if not yes:
        if not click.confirm(f"Delete course '{slug}'?", default=False):
            print("Aborted.")
            sys.exit(1)

    try:
        result = remove_course(libv2_root, slug)
    except CourseRemovalError as exc:
        print_error(f"{exc.code}: {exc.detail}")
        sys.exit(1)

    print_success(f"Removed course: {slug}")
    if result.catalog_files_pruned:
        print(f"  Pruned catalog entries from {len(result.catalog_files_pruned)} file(s):")
        for rel in result.catalog_files_pruned:
            print(f"    - {rel}")
    else:
        print("  No catalog entries referenced this course (nothing to prune).")


def _resolve_libv2_root(ctx) -> Path:
    """Resolve the LibV2 root the same way ``remove`` does.

    Honors the canonical ``ED4ALL_LIBV2_ROOT`` / ``ED4ALL_HOME`` resolver unless
    the operator pinned an explicit ``--repo`` on the top-level group.
    """
    if ctx.obj.get("repo_explicit"):
        return ctx.obj["repo_root"]
    try:
        from lib.paths import libv2_path  # noqa: PLC0415

        return libv2_path()
    except Exception:  # pragma: no cover — fall back to the auto-detected root
        return ctx.obj["repo_root"]


@main.command("backup")
@click.option(
    "--out", "-o", "out",
    type=click.Path(),
    help="Backup destination (tarball or dir). Default: "
    "libv2_backup_<UTC-timestamp>.tar.gz in the current directory.",
)
@click.option(
    "--format", "-F", "fmt",
    type=click.Choice(["tar", "dir"]),
    default="tar",
    help="Snapshot format: 'tar' (gzip tarball, default) or 'dir' (directory).",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite an existing destination.")
@click.pass_context
def backup_cmd(ctx, out, fmt, force):
    """Snapshot the LibV2 metadata spine (catalog + per-course manifests).

    Read-only over the live store: captures the whole ``catalog/`` tree (the
    library manifest ``master_catalog.json`` + derived indexes) and each
    course's small metadata sidecars (``manifest.json`` + ``course.json``). The
    multi-MB chunk bodies / vector indexes / adapters are NEVER read — a
    metadata-only disaster-recovery snapshot.

    \b
    Examples:
        libv2 backup
        libv2 backup --out /mnt/backups/libv2-2026-07-01.tar.gz
        libv2 backup --format dir --out ./libv2_snapshot --force
    """
    from .backup import BackupError, create_backup

    libv2_root = _resolve_libv2_root(ctx)

    if out:
        dest = Path(out)
    else:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = "" if fmt == "dir" else ".tar.gz"
        dest = Path.cwd() / f"libv2_backup_{ts}{suffix}"

    try:
        result = create_backup(libv2_root, dest, fmt=fmt, force=force)
    except BackupError as exc:
        print_error(f"{exc.code}: {exc.detail}")
        sys.exit(1)

    m = result.manifest
    print_success(f"Backed up LibV2 metadata to: {result.dest}")
    print(f"  Format:  {result.fmt}")
    print(f"  Courses: {m.course_count}")
    print(f"  Files:   {m.file_count}")
    print(f"  Root:    {libv2_root}")


@main.command("restore")
@click.argument("backup_path", type=click.Path(exists=True))
@click.option(
    "--overwrite", is_flag=True,
    help="Overwrite live files that diverge from the backup (default: skip + report).",
)
@click.option(
    "--dry-run", is_flag=True,
    help="Verify + report what would be restored without writing anything.",
)
@click.option(
    "--verify-only", is_flag=True,
    help="Only re-hash the backup members against its manifest; restore nothing.",
)
@click.pass_context
def restore_cmd(ctx, backup_path, overwrite, dry_run, verify_only):
    """Verify + restore a LibV2 metadata backup into the library.

    Verify-first (every member is re-hashed against the backup manifest before
    any write), idempotent + resumable (a target already present with the
    recorded checksum is skipped), and non-clobbering (a divergent live file is
    left as-is unless ``--overwrite``).

    \b
    Examples:
        libv2 restore libv2_backup_20260701T120000Z.tar.gz --verify-only
        libv2 restore ./libv2_snapshot --dry-run
        libv2 restore libv2_backup.tar.gz --overwrite
    """
    from .backup import BackupError, restore_backup, verify_backup

    libv2_root = _resolve_libv2_root(ctx)
    src = Path(backup_path)

    if verify_only:
        try:
            v = verify_backup(src)
        except BackupError as exc:
            print_error(f"{exc.code}: {exc.detail}")
            sys.exit(1)
        print(f"Verified {len(v.ok)} member(s) OK.")
        if v.mismatched:
            print_error(f"  Checksum mismatch: {len(v.mismatched)} member(s)")
            for rel in v.mismatched:
                print(f"    - {rel}")
        if v.missing:
            print_error(f"  Missing: {len(v.missing)} member(s)")
            for rel in v.missing:
                print(f"    - {rel}")
        if v.passed:
            print_success("Backup verification passed.")
            sys.exit(0)
        print_error("Backup verification FAILED.")
        sys.exit(1)

    try:
        result = restore_backup(
            libv2_root, src, overwrite=overwrite, dry_run=dry_run
        )
    except BackupError as exc:
        print_error(f"{exc.code}: {exc.detail}")
        sys.exit(1)

    if dry_run:
        print(f"[dry-run] Would restore {len(result.planned)} file(s) into {libv2_root}")
        for rel in result.planned:
            print(f"    + {rel}")
    else:
        print_success(f"Restored {len(result.restored)} file(s) into {libv2_root}")
    if result.skipped:
        print(f"  Skipped (already up to date): {len(result.skipped)}")
    if result.conflicted:
        print_warning(
            f"  Conflicted (live file diverges, left as-is; pass --overwrite to "
            f"replace): {len(result.conflicted)}"
        )
        for rel in result.conflicted:
            print(f"    ! {rel}")


@main.command("link-outcomes")
@click.argument("slug")
@click.option("--objectives", "-o", type=click.Path(exists=True), required=True,
              help="Path to Courseforge learning_objectives.json")
@click.option("--threshold", "-t", type=float, default=0.20,
              help="Minimum similarity threshold for linking (default: 0.20)")
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
@click.argument("model_id", required=False)
@click.option("--judge", type=click.Choice(["none", "anthropic", "local_nli"]),
              default="none", help="Wave 103: qualitative judge for ED4ALL-Bench")
@click.option("--output", "-o", type=click.Path(), help="Save report to file")
@click.option("--verbose", "-v", is_flag=True, help="Show progress for each query")
@click.option("--format", "-f", "fmt", type=click.Choice(["text", "json"]), default="text",
              help="Output format")
@click.pass_context
def eval_run(ctx, slug: str, model_id: Optional[str], judge: str,
             output: Optional[str], verbose: bool, fmt: str):
    """Run evaluation against a course's eval set.

    Two modes:

    \b
    - Legacy retrieval eval (no MODEL_ID): runs the LibV2 retrieval
      harness against quality/eval_set.json. Use 'eval generate' first.
    - Wave 103 ED4ALL-Bench (with MODEL_ID): invokes AblationRunner
      against the named adapter under courses/<slug>/models/<model_id>.

    Examples:

    \b
        libv2 eval run accessibility-design
        libv2 eval run my-course -v -o report.json
        libv2 eval run demo-course-1 my-model-id --judge anthropic
    """
    repo_root = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug

    if not course_dir.exists():
        print_error(f"Course not found: {slug}")
        sys.exit(1)

    if model_id is not None:
        _run_ed4all_bench_eval(
            course_dir=course_dir,
            slug=slug,
            model_id=model_id,
            judge=judge,
            fmt=fmt,
        )
        return

    from .eval_harness import run_course_evaluation

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


# ---------------------------------------------------------------------- #
# Wave 103 - ED4ALL-Bench per-course eval directory                       #
# ---------------------------------------------------------------------- #


_ED4ALL_BENCH_REQUIRED_KEYS = (
    "benchmark", "benchmark_version", "top_k", "temperature", "top_p",
    "max_new_tokens", "seed", "prompt_template_file", "rubric_file",
)


def _schemas_eval_dir() -> Path:
    """Resolve schemas/eval/ relative to this file."""
    # cli.py lives at LibV2/tools/libv2/cli.py -> parents[3] is project root.
    return Path(__file__).resolve().parents[3] / "schemas" / "eval"


@eval_group.command("init")
@click.argument("slug")
@click.pass_context
def eval_init(ctx, slug: str):
    """Wave 103: scaffold the per-course eval/ directory.

    Copies the four ED4ALL-Bench defaults from schemas/eval/ into
    courses/<slug>/eval/ - prompt_template.txt, rubric.md,
    eval_config.yaml, and a placeholder holdout_split.json. Idempotent:
    existing files are not overwritten.

    \b
    Example:
        libv2 eval init demo-course-1
    """
    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug
    if not course_dir.exists():
        print_error(f"Course not found: {course_dir}")
        sys.exit(1)
    eval_dir = course_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    schemas_dir = _schemas_eval_dir()
    pairs = [
        (schemas_dir / "default_prompt_template.txt", eval_dir / "prompt_template.txt"),
        (schemas_dir / "default_rubric.md", eval_dir / "rubric.md"),
        (schemas_dir / "default_eval_config.yaml", eval_dir / "eval_config.yaml"),
    ]
    for src, dst in pairs:
        if dst.exists():
            print(f"  skip (exists): {dst.relative_to(repo_root)}")
            continue
        if not src.exists():
            print_error(f"Default missing: {src}")
            sys.exit(1)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print_success(f"  wrote: {dst.relative_to(repo_root)}")
    holdout = eval_dir / "holdout_split.json"
    if not holdout.exists():
        holdout.write_text(
            json.dumps({
                "_comment": (
                    "Wave 103 placeholder. Run HoldoutBuilder to populate "
                    "withheld_edges before publishing eval numbers."
                ),
                "withheld_edges": [],
                "stratification": {},
                "holdout_graph_hash": None,
            }, indent=2),
            encoding="utf-8",
        )
        print_success(f"  wrote: {holdout.relative_to(repo_root)}")
    else:
        print(f"  skip (exists): {holdout.relative_to(repo_root)}")
    print_success(f"Initialized eval/ for {slug}")


@eval_group.command("validate")
@click.argument("slug")
@click.pass_context
def eval_validate(ctx, slug: str):
    """Wave 103: assert the per-course eval/ directory is well-formed.

    Checks all four files exist, prompt_template.txt has the
    {context_section} and {question} placeholders, and eval_config.yaml
    carries every locked variable. Prints OK on success.

    \b
    Example:
        libv2 eval validate demo-course-1
    """
    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / slug
    if not course_dir.exists():
        print_error(f"Course not found: {course_dir}")
        sys.exit(1)
    eval_dir = course_dir / "eval"
    issues = []
    for fname in ("prompt_template.txt", "rubric.md", "eval_config.yaml",
                  "holdout_split.json"):
        if not (eval_dir / fname).exists():
            issues.append(f"missing file: eval/{fname}")
    template_path = eval_dir / "prompt_template.txt"
    if template_path.exists():
        text = template_path.read_text(encoding="utf-8")
        if "{context_section}" not in text:
            issues.append("prompt_template.txt missing {context_section}")
        if "{question}" not in text:
            issues.append("prompt_template.txt missing {question}")
    config_path = eval_dir / "eval_config.yaml"
    if config_path.exists():
        import yaml as _yaml
        try:
            config = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except _yaml.YAMLError as e:
            issues.append(f"eval_config.yaml YAML parse error: {e}")
            config = {}
        if isinstance(config, dict):
            for key in _ED4ALL_BENCH_REQUIRED_KEYS:
                if key not in config:
                    issues.append(f"eval_config.yaml missing key: {key}")
    if issues:
        for issue in issues:
            print_error(issue)
        sys.exit(1)
    print_success("OK")


def _run_ed4all_bench_eval(
    *,
    course_dir: Path,
    slug: str,
    model_id: str,
    judge: str,
    fmt: str,
) -> None:
    """ED4ALL-Bench dispatch — run a fresh adapter eval (judge=none).

    ``judge=none`` routes through the fresh-eval bridge
    (``model_eval_bridge.run_fresh_eval``): rebuild the saved adapter into
    an :class:`AdapterCallable` and score it with :class:`SLMEvalHarness`,
    writing a fresh ``eval_report.json`` under the model dir. Real runs
    need the ``[training]`` ML stack + the ``scripts/gpu_guard.sh`` wrap
    on a shared-GPU box.

    The qualitative-judge arms (``--judge anthropic`` / ``--judge
    local_nli``) layer an LLM/NLI scorer over the generations and are a
    separate concern; they still validate inputs + print a guided
    next-step until that scorer lands.
    """
    model_dir = course_dir / "models" / model_id
    if not model_dir.exists():
        print_error(f"Model not found: {model_dir}")
        sys.exit(1)
    eval_config_path = course_dir / "eval" / "eval_config.yaml"
    if not eval_config_path.exists():
        print_warning(
            f"No per-course eval/eval_config.yaml. Run "
            f"'libv2 eval init {slug}' first to scaffold one."
        )

    if judge == "none":
        # repo_root is <...>/courses/<slug> -> parents[1].
        repo_root = course_dir.parent.parent
        _run_fresh_model_eval(
            slug=slug,
            model_id=model_id,
            repo_root=repo_root,
            smoke=False,
            replace=False,
            fmt=fmt,
        )
        return

    payload = {
        "course": slug,
        "model_id": model_id,
        "judge": judge,
        "model_dir": str(model_dir),
        "eval_config": str(eval_config_path) if eval_config_path.exists() else None,
        "status": (
            f"Qualitative --judge {judge} scorer not yet wired. Run with "
            f"--judge none for the quantitative fresh-adapter eval, or "
            f"`libv2 models eval {slug} {model_id} --fresh`."
        ),
    }
    if fmt == "json":
        print(json.dumps(payload, indent=2))
    else:
        print_success(f"ED4ALL-Bench eval kickoff: {slug} / {model_id}")
        print(f"  judge: {judge}")
        print(f"  model_dir: {model_dir}")
        print(f"  status: {payload['status']}")


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


@main.command("cross-discover")
@click.argument("query")
@click.option("--repo-root", type=click.Path(exists=True, file_okay=False),
              help="Repository root (auto-detected if omitted)")
@click.option("--index", "index_path", type=click.Path(exists=True, dir_okay=False),
              help="Explicit path to cross_package_concepts.json (default: "
                   "<repo-root>/LibV2/catalog/cross_package_concepts.json)")
@click.option("--limit", type=int, default=10, show_default=True,
              help="Max candidate courses to list")
@click.option("--min-courses", type=int, default=1, show_default=True,
              help="Only match concepts taught in at least this many courses")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text",
              show_default=True, help="Output format")
@click.pass_context
def cross_discover(ctx, query: str, repo_root: Optional[str], index_path: Optional[str],
                   limit: int, min_courses: int, output: str):
    """Discover which library courses teach a concept (consumes the cross index).

    Reads the cross-package concept index built by ``libv2 cross-index`` and
    routes a topic QUERY to the candidate courses a library-wide ask should fan
    out over. Read-only: surfaces only associations the index already recorded
    (provenance-preserved by slug), never fabricates a course.

    \b
    Examples:
        libv2 cross-discover accessibility
        libv2 cross-discover "universal design" --limit 5 --min-courses 2 -o json
    """
    from .cross_package_discovery import (
        CrossPackageIndexError,
        discover_courses,
        load_cross_package_index,
    )

    if repo_root is not None:
        root = Path(repo_root).resolve()
    else:
        root = Path(ctx.obj["repo_root"]).resolve()

    try:
        index = load_cross_package_index(
            root, path=Path(index_path) if index_path else None
        )
    except CrossPackageIndexError as e:
        print_error(str(e))
        sys.exit(1)

    result = discover_courses(index, query, limit=limit, min_courses=min_courses)

    if output == "json":
        print(json.dumps(result, indent=2, sort_keys=False))
        return

    concepts = result["matched_concepts"]
    courses = result["courses"]
    if not concepts:
        print_error(f"No indexed concepts match: {query!r}")
        return
    print_success(
        f"Matched {len(concepts)} concept(s); {len(courses)} candidate course(s) "
        f"for: {query!r}"
    )
    print("\nCandidate courses (route a library-wide ask here):")
    for c in courses:
        print(
            f"  {c['slug']}  "
            f"(concepts={c['matched_concept_count']}, freq={c['total_frequency']})"
        )
    print("\nMatched concepts:")
    for concept in concepts:
        print(
            f"  {concept['concept_id']} "
            f"({concept['total_courses']} course(s)): {concept['label']}"
        )


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
        libv2 retrieval-compare --course demo-course-1 \\
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
@click.option("--snippet-chars", type=int, default=400,
              help="Chars of chunk body to include per retrieved chunk "
                   "(eval passes a larger value for fuller context).")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def ask(ctx, query: str, course: Optional[str], method: str, limit: int,
        one_shot_answer: Optional[str], force: bool, snippet_chars: int,
        output: str):
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
        libv2 ask "How do I model SHACL property paths?" --course demo-course-1
        libv2 ask "compare UDL vs differentiated instruction" --method hybrid
        libv2 ask "How does owl:sameAs entail?" --course demo-course-1 --force
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

    compact = [compact_retrieval_result(r, i + 1, snippet_chars=snippet_chars).to_dict() for i, r in enumerate(results)]
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
        libv2 export-rdf demo-course-1
        libv2 export-rdf demo-course-1 --format trig -o /tmp/rdf-out/
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
    ``courses/<slug>/models/<model_id>/`` alongside ``imscc_chunks/``
    (Phase 7c rename of ``corpus/``), ``graph/``, etc.
    ``_pointers.json`` records which model_id is currently promoted.
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
        libv2 models list demo-course-1
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
        libv2 models promote demo-course-1 qwen2-5-1-5b-demo-course-1-3a4f8c92
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
@click.option("--fresh", is_flag=True,
              help="Run a fresh evaluation from the saved adapter instead of "
                   "printing the cached report (loads the [training] ML stack).")
@click.option("--smoke", is_flag=True,
              help="With --fresh: run the harness in smoke mode (N=3 probes/stage).")
@click.option("--replace", is_flag=True,
              help="With --fresh: overwrite the canonical eval_report.json "
                   "(backing the prior one up to eval_report.json.bak) so "
                   "cached-report consumers pick up the fresh scores.")
@click.pass_context
def models_eval_cmd(ctx, slug: str, model_id: str, output: str,
                    fresh: bool, smoke: bool, replace: bool):
    """Print the cached eval_report.json for a model (or run a fresh eval).

    Default: surfaces the report Trainforge.eval.SLMEvalHarness wrote
    alongside the model card at training time.

    ``--fresh`` runs a NEW evaluation from the saved adapter via the
    fresh-eval bridge (``model_eval_bridge.run_fresh_eval`` — rebuilds
    :class:`AdapterCallable` from the model dir and scores it with
    :class:`SLMEvalHarness`). The fresh report lands non-destructively at
    ``models/<model_id>/eval_report.fresh-<ts>.json`` unless ``--replace``
    is passed (then it overwrites the canonical report after a ``.bak``
    backup). A fresh run needs the ``[training]`` ML stack and, on a
    shared-GPU box, the ``scripts/gpu_guard.sh`` wrap.

    \b
    Examples:
        libv2 models eval demo-course-1 qwen2-5-1-5b-demo-course-1-3a4f8c92
        libv2 models eval demo-course-1 <model_id> --fresh --smoke
        scripts/gpu_guard.sh run --task libv2-fresh-eval -- \\
            libv2 models eval demo-course-1 <model_id> --fresh --replace
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

    if fresh:
        _run_fresh_model_eval(
            slug=slug,
            model_id=model_id,
            repo_root=repo_root,
            smoke=smoke,
            replace=replace,
            fmt=output,
        )
        return
    if smoke or replace:
        print_warning("--smoke / --replace only apply with --fresh; ignoring.")

    report = get_model_eval_report(slug, model_id, repo_root)
    if report is None:
        print_warning(
            f"No eval_report.json found for {model_id}. Evaluation has not "
            f"run for this model — invoke `python -m Trainforge.train_course "
            f"--course-code {slug}` to train and score together, or run a "
            f"fresh eval from the saved adapter with "
            f"`libv2 models eval {slug} {model_id} --fresh`."
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


def _run_fresh_model_eval(
    *,
    slug: str,
    model_id: str,
    repo_root: Path,
    smoke: bool,
    replace: bool,
    fmt: str,
) -> None:
    """Run the fresh-eval bridge and print the resulting report.

    Shared by ``libv2 models eval --fresh`` and the ``eval run <slug>
    <model_id> --judge none`` ED4ALL-Bench path. Catches the heavy-deps
    ImportError and prints actionable guidance (the [training] extra + the
    gpu_guard wrap) instead of a bare stack trace.
    """
    from lib.decision_capture import DecisionCapture
    from . import model_eval_bridge

    capture = DecisionCapture(course_code=slug, phase="libv2-indexing", tool="libv2")
    try:
        report_path = model_eval_bridge.run_fresh_eval(
            course_slug=slug,
            model_id=model_id,
            repo_root=repo_root,
            smoke=smoke,
            replace=replace,
            capture=capture,
        )
    except ImportError as exc:
        print_error(
            f"Fresh eval unavailable — missing training deps: {exc}",
            markup=False,
        )
        # markup=False preserves the literal "[training]" extra name that
        # rich would otherwise interpret as console markup and swallow.
        if RICH_AVAILABLE:
            console.print(
                model_eval_bridge.TRAINING_DEPS_GUIDANCE,
                style="yellow", markup=False,
            )
        else:
            print(model_eval_bridge.TRAINING_DEPS_GUIDANCE)
        sys.exit(1)
    except model_eval_bridge.FreshEvalError as exc:
        print_error(str(exc))
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — surface a clean CLI error.
        print_error(f"Fresh eval failed: {exc}")
        sys.exit(1)

    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print_error(f"Fresh eval wrote a report but it could not be read: {exc}")
        sys.exit(1)

    if fmt == "json":
        print(json.dumps(report, indent=2))
        return
    print_success(f"Fresh eval report written to {report_path}")
    if smoke:
        print_warning(
            "smoke_mode report — not gate-worthy (EvalGatingValidator "
            "refuses smoke reports)."
        )
    for key in ("faithfulness", "coverage", "baseline_delta",
                "calibration_ece", "profile"):
        if key in report:
            print(f"  {key}: {report[key]}")


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
        libv2 import-model /path/to/run-dir --course demo-course-1 --promote
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


# ==========================================================================
# WS2 — retrieval-benchmark command (BM25 vs semantic vs hybrid-rrf).
# Wraps eval_harness.benchmark_retrieval_engines: the harness runs whatever
# index already exists (build it first with `libv2 vector-index build`, or
# pass --build-index to build the canonical index inline). Fail-closed:
# a missing/stale index, an unavailable backend, or a drifted gold set all
# exit non-zero with operator guidance — never a silent BM25-only result.
# ==========================================================================


@main.command("retrieval-benchmark")
@click.option("--course", "-c", required=True, help="Course slug to benchmark")
@click.option(
    "--engines",
    default="bm25,semantic,hybrid-rrf",
    help="Comma-separated engines (default: bm25,semantic,hybrid-rrf). "
    "Allowed: bm25, semantic, hybrid-rrf.",
)
@click.option(
    "--gold-set",
    "gold_set_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Override gold set path (retrieval_eval/gold_set.json shape, or a "
    ".jsonl legacy gold-queries file). Default resolves WS1 gold set then "
    "legacy retrieval/gold_queries.jsonl.",
)
@click.option(
    "--k",
    "k_values",
    default="1,3,5,10",
    help="Comma-separated Recall@k cutoffs (default: 1,3,5,10).",
)
@click.option(
    "--limit", type=int, default=10, help="Retrieval depth per query (default 10)."
)
@click.option(
    "--out",
    "output_path",
    type=click.Path(dir_okay=False),
    help="Report output path (default: courses/<slug>/retrieval_eval/"
    "benchmark_<ts>.json).",
)
@click.option(
    "--build-index",
    is_flag=True,
    help="Build (or rebuild) the canonical vector index before benchmarking "
    "the semantic/hybrid arms. Without this flag the harness runs whatever "
    "index already exists and fails closed if none is present.",
)
@click.option("--provider", help="Embedding provider for --build-index (default env/'st').")
@click.option("--model", "model_id", help="Embedding model id for --build-index.")
@click.option(
    "--models",
    "models_csv",
    help="Comma-separated embedding model ids to sweep (e.g. "
    "'BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5'). For each model the "
    "command builds a temp index (vector_index.bench-<key>/ alongside the "
    "canonical dir), runs the semantic/hybrid benchmark against it, and writes "
    "one report per model. The canonical vector_index/ is preserved across "
    "the sweep. Mutually exclusive with --model. Temp dirs are removed unless "
    "--keep is passed.",
)
@click.option(
    "--keep",
    is_flag=True,
    help="With --models, leave each per-model temp index dir "
    "(vector_index.bench-<key>/) on disk instead of cleaning it up.",
)
@click.pass_context
def retrieval_benchmark(
    ctx,
    course,
    engines,
    gold_set_path,
    k_values,
    limit,
    output_path,
    build_index,
    provider,
    model_id,
    models_csv,
    keep,
):
    """Benchmark BM25 vs semantic vs hybrid-rrf retrieval over a course gold set.

    Emits Recall@{1,3,5,10} (primary + any-relevant), MRR, and latency for
    each engine plus per-engine deltas vs the BM25 baseline, writes the JSON
    report under courses/<slug>/retrieval_eval/, and prints a human-readable
    comparison table. The semantic/hybrid arms require a pre-built vector
    index (`libv2 vector-index build --course <slug>` or `--build-index`);
    they fail closed — never BM25 output — when the index is missing/stale or
    the embedding backend is unavailable.

    Example:

        libv2 retrieval-benchmark --course demo-course-1 \\
            --engines bm25,semantic,hybrid-rrf
    """
    from .eval_harness import benchmark_retrieval_engines

    repo_root = ctx.obj["repo_root"]
    course_dir = Path(repo_root) / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    engine_list = [e.strip() for e in engines.split(",") if e.strip()]
    try:
        ks = tuple(int(k.strip()) for k in k_values.split(",") if k.strip())
    except ValueError:
        print_error(f"--k must be comma-separated integers; got {k_values!r}")
        sys.exit(1)
    if not ks:
        print_error("--k resolved to an empty cutoff list")
        sys.exit(1)

    # Multi-model sweep (wave-C CLI orchestration). For each model: build a
    # temp index alongside the canonical dir, swap it into place for the
    # benchmark run, restore the canonical index, write one tagged report.
    if models_csv:
        if model_id:
            print_error("--models and --model are mutually exclusive")
            sys.exit(1)
        model_list = [m.strip() for m in models_csv.split(",") if m.strip()]
        if not model_list:
            print_error("--models resolved to an empty model list")
            sys.exit(1)
        _run_model_sweep(
            ctx,
            course=course,
            course_dir=course_dir,
            repo_root=repo_root,
            engine_list=engine_list,
            gold_set_path=gold_set_path,
            ks=ks,
            limit=limit,
            output_path=output_path,
            provider=provider,
            model_list=model_list,
            keep=keep,
        )
        return

    # Optional inline build of the canonical index (D9: the harness reads
    # whatever vector_index/ exists; --build-index provisions it first).
    if build_index:
        from lib.embedding.providers import (
            EmbeddingBackendUnavailable,
            build_embedding_client,
        )

        from .vector_index import build_vector_index

        try:
            client = build_embedding_client(
                provider_name=provider, model_id=model_id, offline=False,
            )
            manifest = build_vector_index(course_dir, client=client, force=True)
        except EmbeddingBackendUnavailable as exc:
            print_error(f"EmbeddingBackendUnavailable: {exc}")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001 — surface build failure honestly
            print_error(f"{type(exc).__name__}: {exc}")
            sys.exit(1)
        print_success(
            f"Built vector index for {course}: {manifest.chunks_count} chunks, "
            f"dim={manifest.embedding_dim}, model={manifest.embedding_model_id}"
        )

    try:
        report = benchmark_retrieval_engines(
            repo_root,
            course,
            gold_set_path=Path(gold_set_path) if gold_set_path else None,
            engines=tuple(engine_list),
            k_values=ks,
            retrieval_limit=limit,
            output_path=Path(output_path) if output_path else None,
        )
    except (FileNotFoundError, NotImplementedError) as exc:
        # Missing gold set / semantic-axis-not-wired: fail closed with the
        # error's own guidance (never a degraded benchmark).
        print_error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
    except ImportError as exc:
        _fail_semantic_deps_missing(exc)
    except _SEMANTIC_ERRORS as exc:
        # Unknown engine, drifted gold set, missing/stale index, or an
        # unavailable embedding backend — fail closed, no BM25-only fallback.
        print_error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    _print_benchmark_table(report)
    print(f"\nReport: {report['output_path']}")


def _model_tag(model_id: str) -> str:
    """Filesystem-safe short tag for a model id (last path segment, lowered,
    non-alnum -> '-'). 'BAAI/bge-large-en-v1.5' -> 'bge-large-en-v1-5'."""
    leaf = model_id.rsplit("/", 1)[-1].lower()
    out = []
    for ch in leaf:
        out.append(ch if (ch.isalnum()) else "-")
    tag = "".join(out).strip("-")
    while "--" in tag:
        tag = tag.replace("--", "-")
    return tag or "model"


def _run_model_sweep(
    ctx,
    *,
    course,
    course_dir,
    repo_root,
    engine_list,
    gold_set_path,
    ks,
    limit,
    output_path,
    provider,
    model_list,
    keep,
):
    """Orchestrate a per-model benchmark sweep (the wave-C CLI scope the
    harness docstring punts on).

    For each model id: build a temp index into
    ``vector_index.bench-<tag>/`` alongside the canonical ``vector_index/``,
    swap it into the canonical location for the duration of the benchmark
    (the semantic retrieval path reads only ``vector_index/``), run the
    benchmark, then restore the original canonical index. One report is
    written per model (``benchmark_<tag>.json`` under ``retrieval_eval/``,
    or, when ``--out`` is given, ``<out-stem>_<tag><out-suffix>``). The model
    list is recorded in every report's ``config.models``. Temp dirs are
    removed unless ``--keep``.

    Fail-closed: a build/backend failure or a benchmark error on any model
    aborts the sweep non-zero AFTER restoring the canonical index — the
    canonical ``vector_index/`` is never left swapped out.
    """
    import shutil

    from lib.embedding.providers import (
        EmbeddingBackendUnavailable,
        build_embedding_client,
    )

    from .eval_harness import benchmark_retrieval_engines
    from .vector_index import VECTOR_INDEX_DIRNAME, build_vector_index

    canonical_dir = course_dir / VECTOR_INDEX_DIRNAME
    eval_dir = course_dir / "retrieval_eval"
    out_path = Path(output_path) if output_path else None

    reports: list = []
    for model in model_list:
        tag = _model_tag(model)
        bench_dir = course_dir / f"{VECTOR_INDEX_DIRNAME}.bench-{tag}"
        if bench_dir.exists():
            shutil.rmtree(bench_dir)

        # Build the per-model index into a temp dir by building into the
        # canonical name inside an isolated build, then relocating. We build
        # directly into bench_dir by temporarily redirecting the canonical
        # dir: build writes to <course>/vector_index, so swap-then-build.
        stashed = None
        if canonical_dir.exists():
            stashed = course_dir / f"{VECTOR_INDEX_DIRNAME}.canonical-stash"
            if stashed.exists():
                shutil.rmtree(stashed)
            canonical_dir.rename(stashed)
        try:
            try:
                client = build_embedding_client(
                    provider_name=provider, model_id=model, offline=False,
                )
                manifest = build_vector_index(
                    course_dir, client=client, force=True
                )
            except EmbeddingBackendUnavailable as exc:
                print_error(f"EmbeddingBackendUnavailable ({model}): {exc}")
                _restore_canonical(canonical_dir, stashed)
                sys.exit(1)
            except Exception as exc:  # noqa: BLE001
                print_error(f"{type(exc).__name__} building {model}: {exc}")
                _restore_canonical(canonical_dir, stashed)
                sys.exit(1)

            print_success(
                f"[{tag}] built index: {manifest.chunks_count} chunks, "
                f"dim={manifest.embedding_dim}, model={manifest.embedding_model_id}"
            )

            # The freshly built canonical dir IS this model's index; benchmark
            # it in place, then move it to its bench-<tag> home.
            if out_path is not None:
                report_path = out_path.with_name(
                    f"{out_path.stem}_{tag}{out_path.suffix}"
                )
            else:
                report_path = eval_dir / f"benchmark_{tag}.json"

            try:
                report = benchmark_retrieval_engines(
                    repo_root,
                    course,
                    gold_set_path=Path(gold_set_path) if gold_set_path else None,
                    engines=tuple(engine_list),
                    models=model_list,
                    k_values=ks,
                    retrieval_limit=limit,
                    output_path=report_path,
                )
            except (FileNotFoundError, NotImplementedError) as exc:
                print_error(f"{type(exc).__name__} ({model}): {exc}")
                _move_to_bench(canonical_dir, bench_dir, keep)
                _restore_canonical(canonical_dir, stashed)
                sys.exit(1)
            except _SEMANTIC_ERRORS as exc:
                print_error(f"{type(exc).__name__} ({model}): {exc}")
                _move_to_bench(canonical_dir, bench_dir, keep)
                _restore_canonical(canonical_dir, stashed)
                sys.exit(1)

            reports.append((model, tag, report))
            _print_benchmark_table(report)
            print(f"\n[{tag}] Report: {report['output_path']}")
        finally:
            # Move this model's built index out of the canonical slot into its
            # bench-<tag> home (or drop it), then restore the original.
            _move_to_bench(canonical_dir, bench_dir, keep)
            _restore_canonical(canonical_dir, stashed)

    print(
        f"\nSwept {len(reports)} model(s): "
        + ", ".join(t for _m, t, _r in reports)
        + (
            f". Temp indexes kept under {course_dir}/vector_index.bench-*/"
            if keep
            else ". Temp indexes cleaned; canonical vector_index/ preserved."
        )
    )


def _move_to_bench(canonical_dir, bench_dir, keep):
    """Move the just-built canonical index into its bench-<tag> dir (when
    ``keep``) or remove it. No-op when the canonical dir is absent."""
    import shutil

    if not canonical_dir.exists():
        return
    if keep:
        if bench_dir.exists():
            shutil.rmtree(bench_dir)
        canonical_dir.rename(bench_dir)
    else:
        shutil.rmtree(canonical_dir)


def _restore_canonical(canonical_dir, stashed):
    """Restore the stashed canonical index back into ``vector_index/``."""
    if stashed is not None and stashed.exists() and not canonical_dir.exists():
        stashed.rename(canonical_dir)


def _print_benchmark_table(report: dict) -> None:
    """Print a human-readable per-engine comparison table to stdout."""
    cfg = report.get("config", {})
    ks = cfg.get("k_values", [])
    per_engine = report.get("per_engine", {})
    winner = report.get("winner", {})

    header = (
        f"Retrieval benchmark — {report.get('course_slug')} "
        f"({cfg.get('total_questions')} questions, gold: "
        f"{report.get('gold_source')})"
    )
    print("\n" + header)
    print("=" * len(header))

    # Column layout: engine | Recall@k (primary) ... | MRR | avg latency
    k_cols = " ".join(f"R@{k}".rjust(7) for k in ks)
    print(f"{'engine':<14}{k_cols} {'MRR':>7} {'avg_ms':>8}")
    print("-" * (14 + len(k_cols) + 1 + 7 + 1 + 8))
    for engine in cfg.get("engines", []):
        stats = per_engine.get(engine, {})
        rp = stats.get("recall_at_k_primary", {})
        cells = " ".join(f"{rp.get(str(k), 0.0):7.3f}" for k in ks)
        print(
            f"{engine:<14}{cells} "
            f"{stats.get('mrr', 0.0):7.3f} "
            f"{stats.get('avg_latency_ms', 0.0):8.2f}"
        )

    if winner:
        print("\nvs BM25 baseline (delta Recall@k primary / delta MRR):")
        for engine, w in winner.items():
            deltas = w.get("delta_recall_at_k_primary", {})
            cells = " ".join(f"{deltas.get(str(k), 0.0):+7.3f}" for k in ks)
            print(f"  {engine:<12}{cells}  dMRR={w.get('delta_mrr', 0.0):+.3f}")


# WS2 — vector-index command group (build / status / verify).
# Wires build_vector_index / load_vector_index + the manifest validator.
# Fail-closed: status/verify surface the typed staleness errors verbatim.
# ==========================================================================


@main.group("vector-index")
def vector_index():
    """Manage the per-course on-device semantic vector index.

    The index backs `libv2 retrieve --engine semantic` (and hybrid-rrf).
    Builds are deterministic (same machine + venv + provider + model +
    device=cpu + batch_size => byte-identical embeddings.npy / id_map.json);
    the query path is fail-closed (a missing / stale index errors rather
    than silently degrading to BM25).
    """


@vector_index.command("build")
@click.option("--course", "-c", required=True, help="Course slug to index")
@click.option("--provider", help="Embedding provider (default: env ED4ALL_EMBEDDING_PROVIDER or 'st')")
@click.option("--model", "model_id", help="Embedding model id override")
@click.option("--chunkset", type=click.Choice(["imscc", "dart", "corpus-legacy"]),
              help="Pin a chunkset (default: imscc_chunks -> dart_chunks -> legacy corpus)")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), help="ST device override (default cpu)")
@click.option("--batch-size", type=int, help="Embedding batch size override")
@click.option("--offline", is_flag=True, help="Refuse network downloads (default for build is online)")
@click.option("--force", is_flag=True, help="Rebuild over a fresh index (source sha unchanged)")
@click.pass_context
def vector_index_build(ctx, course, provider, model_id, chunkset, device,
                       batch_size, offline, force):
    """Build (or rebuild) the vector index for a course.

    Downloads happen here (provision-time) unless --offline is passed; the
    query path is always offline. A fail-closed backend (weights/server
    absent) errors out — no partial index is written.

    Example:

        libv2 vector-index build --course demo-course-1 --provider st
    """
    import os

    from lib.embedding.providers import (
        EmbeddingBackendUnavailable,
        build_embedding_client,
    )

    from .vector_index import build_vector_index

    repo_root = ctx.obj["repo_root"]
    course_dir = Path(repo_root) / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    # Per-call device/batch overrides flow through the provider env chain.
    if device:
        os.environ["ED4ALL_EMBEDDING_DEVICE"] = device
    if batch_size:
        os.environ["ED4ALL_EMBEDDING_BATCH_SIZE"] = str(batch_size)

    try:
        client = build_embedding_client(
            provider_name=provider, model_id=model_id, offline=offline,
        )
        manifest = build_vector_index(
            course_dir, client=client, chunkset=chunkset, force=force,
        )
    except FileExistsError as exc:
        print_error(str(exc))
        sys.exit(1)
    except EmbeddingBackendUnavailable as exc:
        print_error(f"EmbeddingBackendUnavailable: {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — surface any build failure honestly
        print_error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    print_success(
        f"Built vector index for {course}: {manifest.chunks_count} chunks, "
        f"dim={manifest.embedding_dim}, provider={manifest.embedding_provider}, "
        f"model={manifest.embedding_model_id}"
    )
    print(f"  chunkset: {manifest.chunkset_kind}")
    print(f"  index dir: {course_dir / 'vector_index'}")


@vector_index.command("status")
@click.option("--course", "-c", required=True, help="Course slug")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def vector_index_status(ctx, course, output):
    """Show the vector-index manifest summary + staleness check.

    Reports manifest provenance and whether the index is fresh vs the live
    chunkset. Exit 0 regardless of staleness (this is a report, not a gate);
    `verify` is the exit-on-drift command.
    """
    from .vector_index import (
        MANIFEST_FILENAME,
        VECTOR_INDEX_DIRNAME,
        VectorIndexManifest,
        load_vector_index,
    )

    repo_root = ctx.obj["repo_root"]
    course_dir = Path(repo_root) / "courses" / course
    manifest_path = course_dir / VECTOR_INDEX_DIRNAME / MANIFEST_FILENAME
    if not manifest_path.exists():
        print_error(
            f"no vector index for {course}; run "
            f"`libv2 vector-index build --course {course}`."
        )
        sys.exit(1)

    manifest = VectorIndexManifest.from_file(manifest_path)
    fresh = True
    stale_reason = None
    try:
        # allow_fake so a fake-built index can still report status.
        load_vector_index(course_dir, allow_fake=True)
    except Exception as exc:  # noqa: BLE001 — capture staleness for the report
        fresh = False
        stale_reason = f"{type(exc).__name__}: {exc}"

    if output == "json":
        doc = manifest.to_dict()
        doc["fresh"] = fresh
        if stale_reason:
            doc["stale_reason"] = stale_reason
        print(json.dumps(doc, indent=2))
        return

    print(f"\n=== Vector index: {course} ===")
    print(f"provider: {manifest.embedding_provider} ({manifest.embedding_kind})")
    print(f"model: {manifest.embedding_model_id} (rev {manifest.embedding_model_revision})")
    print(f"dim: {manifest.embedding_dim} | chunks: {manifest.chunks_count}")
    print(f"chunkset: {manifest.chunkset_kind} | index_type: {manifest.index_type}")
    print(f"text policy: {manifest.text_field_policy} | device: {manifest.device}")
    print(f"chunker_version: {manifest.chunker_version}")
    if fresh:
        print_success("fresh: index matches the live chunkset.")
    else:
        print_warning(f"STALE: {stale_reason}")


@vector_index.command("verify")
@click.option("--course", "-c", required=True, help="Course slug")
@click.pass_context
def vector_index_verify(ctx, course):
    """Full sha re-verification of the vector index (exit 1 on drift).

    Runs the VectorIndexManifestValidator (on-disk sha + count + schema +
    live-chunkset checks). Exits non-zero on any critical issue so it can
    gate a pre-query check in scripts/CI.
    """
    from lib.validators.vector_index_manifest import VectorIndexManifestValidator

    from .vector_index import MANIFEST_FILENAME, VECTOR_INDEX_DIRNAME

    repo_root = ctx.obj["repo_root"]
    course_dir = Path(repo_root) / "courses" / course
    manifest_path = course_dir / VECTOR_INDEX_DIRNAME / MANIFEST_FILENAME
    if not manifest_path.exists():
        print_error(
            f"no vector index for {course}; run "
            f"`libv2 vector-index build --course {course}`."
        )
        sys.exit(1)

    result = VectorIndexManifestValidator().validate(
        {"vector_index_manifest_path": str(manifest_path)}
    )
    if result.passed:
        print_success(f"vector index for {course} verified clean.")
        return
    print_error(f"vector index for {course} FAILED verification:")
    for issue in result.issues:
        if issue.severity == "critical":
            print_error(f"  [{issue.code}] {issue.message}")
        else:
            print_warning(f"  [{issue.code}] {issue.message}")
    sys.exit(1)


# WS3 Wave C — grounded-answer surface.
# `answer-grounded` invokes the single entry point lib.retrieval.grounded_answer
# .answer_course_question (it owns refusal + the WS1 citation gate; bypassing it
# is the hallucination-by-construction path). The companion `answer-eval` and
# `refusal-calibrate` commands are thin delegations to the `python -m` entry
# points so the operator surface is one CLI even though the logic lives in lib/.
# Exit codes follow the established _SEMANTIC_ERRORS fail-closed pattern:
#   0  answered / answered_with_warnings
#   2  refused (low-confidence or model-side not_in_course) — an honest "no"
#   3  blocked by the citation gate or an invalid-citation contradiction
#   1  typed backend/index/compose failure (operator-actionable guidance)
# ==========================================================================


# Status -> exit-code map for `answer-grounded`. A refusal is not a failure
# (exit 2, distinct from 0 so scripts can branch); a citation-gate block is a
# safety stop (exit 3); typed errors map to 1 in the except arms below.
_ANSWER_STATUS_EXIT = {
    "answered": 0,
    "answered_with_warnings": 0,
    "refused_low_confidence": 2,
    "refused_not_in_course": 2,
    "blocked_invalid_citation": 3,
    "blocked_citation_gate": 3,
}


def _render_grounded_answer_text(result) -> None:
    """Accessible plain-text rendering of a GroundedAnswer (no rich markup
    in the body so screen readers / pipes get clean text)."""
    status = result.status
    print(f"status: {status}")
    if result.model_id:
        print(f"model: {result.model_id} (prompt {result.prompt_version})")
    print(f"engine: {result.engine}  course: {result.course_slug}")

    if status.startswith("refused"):
        refusal = result.refusal or {}
        reason = refusal.get("reason_code", "unknown")
        print(f"\nRefused ({reason}).")
        signals = refusal.get("signals") or result.confidence.get("signals") or {}
        if signals:
            sig = ", ".join(f"{k}={v}" for k, v in signals.items())
            print(f"  confidence signals: {sig}")
        print("  (No answer is emitted; this question is out of scope or low-confidence.)")
        return

    if status.startswith("blocked"):
        print(f"\nBlocked: answer withheld ({status}).")
        # The cited chunks that failed the gate are surfaced for the operator.
        if result.citations:
            print("  citations that reached the gate:")
            for c in result.citations:
                print(f"    [{c.chunk_id}] anchor_status={c.anchor_status}")
        print("  (The answer text is withheld because a citation failed the WS1 anchor gate.)")
        return

    # answered / answered_with_warnings
    if result.warnings:
        print(f"warnings: {', '.join(result.warnings)}")
    print("\nAnswer:")
    print(result.answer_text or "(empty)")
    if result.citations:
        print("\nCitations:")
        for c in result.citations:
            line = f"  [{c.chunk_id}] {c.page_label} (anchor: {c.anchor_status})"
            print(line)
            if c.text_quote:
                print(f"      “{c.text_quote}”")
    if result.groundedness is not None:
        rate = result.groundedness.get("groundedness_rate")
        avail = result.groundedness.get("available")
        if avail:
            print(f"\ngroundedness_rate: {rate}")
        else:
            print("\ngroundedness: unavailable (NLI deps absent)")


@main.command("answer-grounded")
@click.argument("query")
@click.option("--course", "-c", required=True, help="Course slug to answer over (single-course scope)")
@click.option("--engine", type=click.Choice(["auto", "lexical", "semantic", "hybrid-rrf"]),
              default="lexical",
              help="Retrieval engine. 'auto' resolves to 'semantic' when a vector "
                   "index exists for the course, else 'lexical' (BM25). "
                   "'hybrid-rrf' (BM25 fused with semantic via reciprocal-rank "
                   "fusion) is the benchmark-selected engine — pass it explicitly "
                   "for the best retrieval quality when an index is present. "
                   "semantic/hybrid-rrf fail closed against a pre-index tree — "
                   "never a silent downgrade.")
@click.option("--limit", "-n", type=int, default=8, help="Max passages to retrieve / pass to the composer")
@click.option("--with-groundedness", is_flag=True,
              help="Score the composed answer per-claim via the NLI harness "
                   "(loads DeBERTa; eval-time cost, advisory only — never blocks).")
@click.option("--json", "as_json", is_flag=True, help="Emit the full GroundedAnswer.to_dict() as JSON")
@click.option("--log/--no-log", "do_log", default=False,
              help="Persist the Q&A under the course's queries/ log "
                   "(answered_by=grounded:<model_id>); default off.")
@click.pass_context
def answer_grounded(ctx, query: str, course: str, engine: str, limit: int,
                    with_groundedness: bool, as_json: bool, do_log: bool):
    """Answer a single-course question, grounded + citation-gated.

    Invokes the fully-automated grounded-answer pipeline (retrieve -> calibrated
    refusal -> local-model compose -> WS1 citation gate). No cloud calls, ever;
    a citation that fails the anchor gate withholds the answer rather than
    emitting an ungrounded claim. The citation gate is NOT bypassable from the
    CLI by design.

    Exit codes: 0 answered, 2 refused (out of scope / low confidence), 3 blocked
    by the citation gate, 1 typed backend/index/compose failure.

    \b
    Examples:
        libv2 answer-grounded "What is a SHACL NodeShape?" --course demo-course-1
        libv2 answer-grounded "Explain RRF fusion" -c demo-course-2 --engine semantic
        libv2 answer-grounded "Define a derivative" -c demo-course-3 --json --with-groundedness
    """
    from lib.decision_capture import DecisionCapture
    from lib.retrieval.answer_backend import (
        AnswerBackendUnavailable,
        AnswerProviderNotLocal,
    )
    from lib.retrieval.answer_composer import AnswerComposeError
    from lib.retrieval.grounded_answer import answer_course_question

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    # 'auto' resolves at the CLI seam: semantic when a vector index exists,
    # else lexical. The pipeline never silently downgrades a *requested*
    # semantic engine; 'auto' makes the choice explicit and visible.
    resolved_engine = engine
    if engine == "auto":
        # Source the two string constants from the numpy-free lower layer so
        # probing for index presence (the lexical/auto path) does not import
        # the numpy-laden vector_index module on a deps-slim operator box.
        from lib.libv2_storage import (
            VECTOR_INDEX_DIRNAME,
            VECTOR_INDEX_MANIFEST_FILENAME as MANIFEST_FILENAME,
        )

        has_index = (course_dir / VECTOR_INDEX_DIRNAME / MANIFEST_FILENAME).exists()
        resolved_engine = "semantic" if has_index else "lexical"

    capture = DecisionCapture(course_code=course, phase="libv2-answer", tool="libv2")

    try:
        result = answer_course_question(
            repo_root,
            course,
            query,
            engine=resolved_engine,
            limit=limit,
            with_groundedness=with_groundedness,
            capture=capture,
        )
    except (AnswerProviderNotLocal, AnswerBackendUnavailable) as exc:
        print_error(f"{type(exc).__name__}: {exc}")
        print_error(
            "The grounded-answer backend is local-only. Start your local model "
            "server (e.g. ollama) and set ED4ALL_ANSWER_PROVIDER / "
            "ED4ALL_ANSWER_MODEL / ED4ALL_ANSWER_TIMEOUT_SECONDS as needed."
        )
        sys.exit(1)
    except AnswerComposeError as exc:
        print_error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
    except ImportError as exc:
        _fail_semantic_deps_missing(exc)
    except _SEMANTIC_ERRORS as exc:
        # Missing/stale vector index or engine misuse — fail closed with the
        # typed guidance (build the index), NEVER a silent BM25 result.
        print_error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    # Optional Q&A-log write — only for emitted answers (a withheld/blocked
    # answer or a refusal is not an "answer" to persist as content).
    if do_log and result.status.startswith("answered"):
        try:
            from .query_log import attach_answer, compact_retrieval_result, write_query_record

            record_path = write_query_record(
                repo_root, course, query,
                method=f"grounded:{resolved_engine}", limit=limit, retrieved=[],
            )
            qid = json.loads(record_path.read_text())["query_id"]
            attach_answer(
                repo_root, course, qid, result.answer_text or "",
                answered_by=f"grounded:{result.model_id}",
            )
        except Exception as exc:  # noqa: BLE001 — logging is best-effort, never blocks the answer
            print_warning(f"query-log write skipped: {type(exc).__name__}: {exc}")

    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _render_grounded_answer_text(result)

    sys.exit(_ANSWER_STATUS_EXIT.get(result.status, 1))


@main.command("answer-eval")
@click.option("--course", "-c", required=True, help="Course slug to evaluate")
@click.option("--engine", type=click.Choice(["lexical", "semantic", "hybrid-rrf"]),
              default="lexical", help="Retrieval engine for the eval pass")
@click.option("--limit", "-n", type=int, default=8, help="top-k retrieval limit per question")
@click.option("--no-groundedness", is_flag=True, help="Skip the per-claim NLI groundedness pass")
@click.option("--arms", default="grounded",
              help="comma-separated eval arms: base,retrieval,grounded "
                   "(default: grounded — byte-compatible with the legacy eval)")
@click.pass_context
def answer_eval(ctx, course: str, engine: str, limit: int, no_groundedness: bool,
                arms: str):
    """Run the grounded-answer eval harness over a course gold set.

    Thin delegation to ``python -m lib.retrieval.grounded_eval`` (same logic, one
    CLI). Emits a ``grounded_answer_eval_<ts>.json`` report under the course's
    ``retrieval_eval/``. Requires the real local answer backend — CI uses the
    module's fake-client tests, not this command. Exit codes pass through the
    module: 3 pipeline absent, 2 gold refused (critical gold-set issue), 0 ok.

    With ``--arms`` set to anything beyond the default ``grounded`` (e.g.
    ``base,retrieval,grounded``), the three-arm scorecard runs instead: BASE
    (qwen only, no retrieval), RETRIEVAL (retrieval only, no LLM), and GROUNDED
    (the full pipeline). It writes a ``retrieval_eval/eval_scorecard_<ts>.json``
    and prints an aligned side-by-side table; the grounded arm STILL writes its
    own ``grounded_answer_eval_<ts>.json`` report, so existing artifacts are
    unaffected. Delegates to ``python -m lib.retrieval.eval_arms``.

    \b
    Example:
        libv2 answer-eval --course demo-course-1 --engine lexical
        libv2 answer-eval --course demo-course-1 --arms base,retrieval,grounded
    """
    repo_root: Path = ctx.obj["repo_root"]

    # Default (grounded-only) → legacy delegation, byte-for-byte unchanged.
    if [a.strip() for a in arms.split(",") if a.strip()] == ["grounded"]:
        from lib.retrieval.grounded_eval import main as grounded_eval_main

        argv = ["--course", course, "--engine", engine, "--limit", str(limit),
                "--repo-root", str(repo_root)]
        if no_groundedness:
            argv.append("--no-groundedness")
        sys.exit(grounded_eval_main(argv))

    # Multi-arm → three-arm scorecard surface.
    from lib.retrieval.eval_arms import main as eval_arms_main

    argv = ["--course", course, "--engine", engine, "--limit", str(limit),
            "--arms", arms, "--repo-root", str(repo_root)]
    sys.exit(eval_arms_main(argv))


@main.command("refusal-calibrate")
@click.option("--course", "-c", required=True, help="Course slug to calibrate")
@click.option("--engine", type=click.Choice(["lexical", "semantic", "hybrid-rrf"]),
              default="lexical", help="Retrieval engine to calibrate the refusal threshold for")
@click.option("--limit", "-n", type=int, default=8, help="top-k retrieval limit per probe")
@click.option("--no-write", is_flag=True, help="Print the calibration JSON without writing the artifact")
@click.pass_context
def refusal_calibrate(ctx, course: str, engine: str, limit: int, no_write: bool):
    """Calibrate the grounded-answer refusal threshold for one (course, engine).

    Thin delegation to ``python -m lib.retrieval.refusal --calibrate``. Measures
    answerable (gold-set) vs unanswerable (refusal-probe) retrieval-score
    distributions and emits ``refusal_calibration.json`` under the course's
    ``retrieval_eval/`` (honest fallback: keeps v0-uncalibrated when the
    distributions overlap). Touches the live retriever read-only.

    \b
    Example:
        libv2 refusal-calibrate --course demo-course-1 --engine semantic
    """
    from lib.retrieval.refusal import main as refusal_main

    repo_root: Path = ctx.obj["repo_root"]
    argv = ["--calibrate", "--course", course, "--engine", engine,
            "--limit", str(limit), "--repo-root", str(repo_root)]
    if no_write:
        argv.append("--no-write")
    sys.exit(refusal_main(argv))


@main.command("attribution-calibrate")
@click.option("--course", "-c", required=True, help="Course slug to calibrate")
@click.option("--engine", type=click.Choice(["lexical", "semantic", "hybrid-rrf"]),
              default="lexical", help="Retrieval engine for the gold replay")
@click.option("--limit", "-n", type=int, default=8, help="top-k retrieval limit per question")
@click.option("--precision-floor", type=float, default=None,
              help="precision floor for the precision_floored min_overlap recommendation")
@click.option("--no-write", is_flag=True, help="Print the calibration JSON without writing the artifact")
@click.pass_context
def attribution_calibrate(ctx, course: str, engine: str, limit: int,
                          precision_floor: Optional[float], no_write: bool):
    """Calibrate the citation-attribution support min_overlap knob for a course.

    Replays the gold set through the retriever (retrieval only — NO LLM),
    scores each retrieved passage with the answer path's attribution machinery
    (gold expected_key_points as the claim proxy), joins against gold relevance,
    sweeps thresholds, and recommends min_overlap (precision-floored max-recall,
    else F1-max). Also ingests live citation-prune captures as a second evidence
    stream and flags whether ADD_MIN_SHINGLE warrants its own env knob. Emits
    ``attribution_calibration_<ts>.json`` under retrieval_eval/.

    \b
    Example:
        libv2 attribution-calibrate --course demo-course-1 --engine lexical
    """
    from lib.retrieval.attribution_calibrate import (
        DEFAULT_PRECISION_FLOOR,
        calibrate,
    )

    repo_root: Path = ctx.obj["repo_root"]
    result = calibrate(
        course,
        engine=engine,
        repo_root=repo_root,
        limit=limit,
        precision_floor=precision_floor if precision_floor is not None
        else DEFAULT_PRECISION_FLOOR,
        write=not no_write,
    )
    rec = result.recommended
    if rec is None:
        print_warning(f"{course}/{engine}: no gold samples — nothing to calibrate.")
    else:
        print_success(
            f"{course}/{engine}: recommended min_overlap={rec['min_overlap']} "
            f"(precision={rec['precision']}, recall={rec['recall']}, "
            f"f1={rec['f1']}; {rec['rule']})"
        )
        warranted = result.add_min_shingle_knob_warranted
        msg = (f"ADD_MIN_SHINGLE knob warranted: {warranted['warranted']} "
               f"({warranted['reason']})")
        (print_warning if warranted["warranted"] else print)(msg)
    if no_write:
        print(json.dumps(result.to_dict(), indent=2))
    sys.exit(0)


# ==========================================================================
# Retrieval gold-set authoring surface (retrieval-answer-eval-set P0/P1).
# `gold-validate` loads the gold set fail-closed + (optionally) prints the
# §1.3 coverage matrix; `gold-repin` re-anchors a gold set to a new chunkset
# via the §3 resolution ladder. Both are thin delegations mirroring the
# answer-eval pattern (logic lives in lib/retrieval/).
# ==========================================================================


@main.command("gold-validate")
@click.option("--course", "-c", required=True, help="Course slug whose gold set to validate")
@click.option("--coverage", is_flag=True,
              help="Also build + print the §1.3 coverage matrix and write "
                   "gold_coverage_<ts>.json under retrieval_eval/.")
@click.option("--no-coverage-write", is_flag=True,
              help="With --coverage: print the matrix but do not write the artifact.")
@click.pass_context
def gold_validate(ctx, course: str, coverage: bool, no_coverage_write: bool):
    """Load a course gold set fail-closed and print its issues.

    Validates against the schema matching the doc's schema_version (dual-version:
    v1.0 + v1.1), verifies the chunkset sha pin, chunk-id existence, quote +
    content-hash containment, and id uniqueness. Exit 1 on any critical issue.
    With --coverage, additionally builds the per-type/week/TO-CO/population/
    difficulty matrix (warnings, never failures).

    \b
    Example:
        libv2 gold-validate --course demo-course-1 --coverage
    """
    from lib.retrieval.gold_set import (
        critical_issues,
        doc_schema_version,
        load_gold_set,
    )

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    gold, issues = load_gold_set(course_dir, verify=True)
    crit = critical_issues(issues)
    warns = [i for i in issues if i not in crit]

    if gold:
        print(f"gold set: {course} (schema_version {doc_schema_version(gold)}, "
              f"{len(gold.get('questions', []))} questions, "
              f"frozen={gold.get('frozen')})")
    for i in crit:
        print_error(f"[{i.code}] {i.message}"
                    + (f" (q={i.question_id})" if i.question_id else ""))
    for i in warns:
        print_warning(f"[{i.code}] {i.message}"
                      + (f" (q={i.question_id})" if i.question_id else ""))

    if not crit:
        print_success(f"gold set valid ({len(warns)} warning(s)).")

    if coverage and gold:
        from lib.retrieval.gold_coverage import build_coverage_report, write_coverage_report
        from lib.retrieval.gold_set import _load_chunks_by_id

        kind = (gold.get("chunkset") or {}).get("kind")
        is_union = kind == "corpus"
        try:
            if no_coverage_write:
                chunks_rel = (gold.get("chunkset") or {}).get("chunks_path") or ""
                chunks_by_id = _load_chunks_by_id(course_dir / chunks_rel)
                report = build_coverage_report(gold, chunks_by_id, is_union=is_union)
                out_path = None
            else:
                report, out_path = write_coverage_report(
                    course_dir, gold, is_union=is_union
                )
        except FileNotFoundError as exc:
            print_error(str(exc))
            sys.exit(1)

        print("\nCoverage matrix:")
        print(f"  by_type:       {report.by_type}")
        print(f"  by_week:       {report.by_week}")
        print(f"  by_population: {report.by_population}")
        print(f"  by_difficulty: {report.by_difficulty}")
        print(f"  objectives covered: {len(report.by_objective)}; "
              f"uncoverable: {len(report.uncoverable_objectives)}")
        for w in report.warnings:
            print_warning(f"  [{w.code}] {w.message}")
        if out_path:
            print_success(f"  wrote {out_path}")

    sys.exit(1 if crit else 0)


@main.command("gold-repin")
@click.option("--course", "-c", required=True, help="Course slug whose gold set to re-pin")
@click.option("--kind", type=click.Choice(["dart", "imscc", "corpus"]), required=True,
              help="Target chunkset kind to re-anchor the gold set to.")
@click.option("--chunks-path", help="Override the kind's default course-dir-relative chunks.jsonl path.")
@click.option("--unfreeze-for-repin", is_flag=True,
              help="Required to re-pin a frozen:true gold set (fail-closed otherwise).")
@click.option("--drop-orphans", is_flag=True,
              help="Remove unresolvable (parked) questions so the re-pinned set "
                   "loads clean / eval-ready; orphans stay recorded in the report.")
@click.option("--dry-run", is_flag=True, help="Build the repin report without writing the gold set.")
@click.pass_context
def gold_repin(ctx, course: str, kind: str, chunks_path: Optional[str],
               unfreeze_for_repin: bool, drop_orphans: bool, dry_run: bool):
    """Re-anchor a course gold set to a new chunkset (§3 resolution ladder).

    Resolves every passage by content_sha256 exact match -> unique text_quote
    containment -> quote scoped to item_path/heading -> unresolved (question
    parked status:draft). Rewrites chunk_ids + the chunkset pin, backfills
    content_sha256, and writes a repin report under retrieval_eval/. Fail-closed
    on a frozen set without --unfreeze-for-repin.

    \b
    Example:
        libv2 gold-repin --course demo-course-1 --kind corpus
    """
    from lib.retrieval.gold_repin import GoldRepinError, repin_gold_set

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    try:
        report, written = repin_gold_set(
            course_dir,
            kind=kind,
            chunks_path=chunks_path,
            unfreeze_for_repin=unfreeze_for_repin,
            drop_orphans=drop_orphans,
            dry_run=dry_run,
        )
    except GoldRepinError as exc:
        print_error(f"{exc.code}: {exc.detail}")
        sys.exit(1)

    counts = report.counts
    print(f"re-pin {course} -> kind={report.kind} ({report.chunks_path})")
    print(f"  new sha: {report.new_sha256[:16]}... (was "
          f"{(report.old_sha256 or '')[:16]}...)")
    print(f"  resolved by content_sha256: {counts['content_sha256']}, "
          f"text_quote: {counts['text_quote']}, "
          f"text_quote_scoped: {counts['text_quote_scoped']}, "
          f"unresolved: {counts['unresolved']}")
    if report.parked_question_ids:
        print_warning(f"  parked (status:draft) questions: "
                      f"{', '.join(report.parked_question_ids)}")
    if dry_run:
        print_warning("  dry-run: no files written.")
    elif written:
        print_success(f"  wrote re-pinned gold set: {written}")
    sys.exit(0)


# ==========================================================================
# Retrieval gold/probe candidate authoring (retrieval-answer-eval-set P2).
# `gold-candidates` samples + drafts (local provider only) + pre-screens;
# `probe-candidates` builds the three probe categories + per-engine dry-runs;
# `gold-promote` merges operator-accepted candidates into gold_set.json. All
# drafting routes through the license-clean LOCAL provider (loopback-enforced
# via answer_backend); NEVER an Anthropic surface.
# ==========================================================================


@main.command("gold-candidates")
@click.option("--course", "-c", required=True, help="Course slug to draft gold candidates for")
@click.option("--n", type=int, default=None,
              help="Number of candidates to draft (default 100 = 2x the 50-question target).")
@click.option("--seed", type=int, default=0, help="Deterministic sampler seed.")
@click.option("--templates", default=None,
              help="Comma-separated template arms to run (default all): "
                   "stratified,definition,worked_example,both_population.")
@click.option("--no-write", is_flag=True, help="Build the doc without writing the artifact.")
@click.pass_context
def gold_candidates(ctx, course: str, n: Optional[int], seed: int,
                    templates: Optional[str], no_write: bool):
    """Draft gold-question candidates via the license-clean local provider.

    Runs the requested template arms (default all four): `stratified`
    stratified-samples the pinned union chunkset; `definition` mines glossary
    terms (one factual_recall question per term); `worked_example` mines
    Problem/Solution/Step chunks (one procedural question per chunk);
    `both_population` pairs a course chunk with a source chunk on the same
    concept (one conceptual_synthesis question per pair,
    expected_citation_population=both). Each arm
    drafts question + 2-4 key points + a verbatim >=40-char quote via the LOCAL
    provider only, pre-screens deterministically (quote containment + ambiguity
    + length + near-dup -- rejections RECORDED, not dropped), and writes
    retrieval_eval/gold_candidates.json. Operator edits that file (status:draft
    -> reviewed) then runs `gold-promote`.

    \b
    Example:
        libv2 gold-candidates --course demo-course-1 --n 100
        libv2 gold-candidates --course demo-course-1 --templates definition,worked_example
    """
    from lib.decision_capture import DecisionCapture
    from lib.retrieval.answer_backend import (
        AnswerBackendUnavailable,
        AnswerProviderNotLocal,
        build_answer_client,
    )
    from lib.retrieval.gold_authoring import generate_gold_candidates

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    template_list = (
        [t.strip() for t in templates.split(",") if t.strip()]
        if templates else None
    )

    capture = DecisionCapture(course_code=course, phase="libv2-answer", tool="libv2")
    try:
        client = build_answer_client(capture=capture)
    except AnswerProviderNotLocal as exc:
        print_error(f"drafting requires a loopback-local provider: {exc}")
        sys.exit(1)
    except AnswerBackendUnavailable as exc:
        print_error(f"local provider unavailable: {exc}")
        sys.exit(1)

    try:
        doc, out_path = generate_gold_candidates(
            course_dir, client=client, n=n, seed=seed,
            capture=capture, write=not no_write, templates=template_list,
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)

    run = doc.get("authoring_run", {})
    print(f"gold-candidates {course}: drafted {run.get('n_drafted')} "
          f"(requested {run.get('n_requested')}), "
          f"pre-screen passed {run.get('n_prescreen_passed')}.")
    if run.get("by_template"):
        print(f"  by_template: {run.get('by_template')}")
    if out_path:
        print_success(f"  wrote {out_path}")
    else:
        print_warning("  no-write: artifact not written.")
    sys.exit(0)


@main.command("probe-candidates")
@click.option("--course", "-c", required=True, help="Course slug to draft refusal-probe candidates for")
@click.option("--limit", type=int, default=8, help="top-k limit for the per-engine dry-runs.")
@click.option("--no-write", is_flag=True, help="Build the doc without writing the artifact.")
@click.pass_context
def probe_candidates(ctx, course: str, limit: int, no_write: bool):
    """Build refusal-probe candidates + per-engine read-only dry-runs.

    off_topic from a fixed cross-domain bank (no LLM); adjacent_domain drafted
    by the LOCAL provider over the course concept_tags vocabulary;
    out_of_scope_detail perturbed from gold questions. Every candidate gets a
    lexical/semantic/hybrid-rrf dry-run; top_passage_answers is left null for
    operator review. Writes retrieval_eval/refusal_probe_candidates.json.

    \b
    Example:
        libv2 probe-candidates --course demo-course-1
    """
    from lib.decision_capture import DecisionCapture
    from lib.retrieval.answer_backend import (
        AnswerBackendUnavailable,
        AnswerProviderNotLocal,
        build_answer_client,
    )
    from lib.retrieval.probe_authoring import generate_probe_candidates
    from LibV2.tools.libv2.retriever import retrieve_chunks

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    capture = DecisionCapture(course_code=course, phase="libv2-answer", tool="libv2")
    try:
        client = build_answer_client(capture=capture)
    except AnswerProviderNotLocal as exc:
        print_error(f"adjacent-domain drafting requires a loopback-local provider: {exc}")
        sys.exit(1)
    except AnswerBackendUnavailable as exc:
        print_error(f"local provider unavailable: {exc}")
        sys.exit(1)

    doc, out_path = generate_probe_candidates(
        course_dir, client=client, retrieve_fn=retrieve_chunks,
        libv2_root=repo_root, limit=limit, capture=capture, write=not no_write,
    )
    run = doc.get("authoring_run", {})
    print(f"probe-candidates {course}: {run.get('n_probes')} probe(s) "
          f"by_category={run.get('by_category')}.")
    under = run.get("underfilled_categories") or []
    if under:
        click.echo(
            f"  WARNING: underfilled probe categories {under} — the drafting "
            f"arm produced fewer than the per-category target (LLM parse "
            f"failure or thin vocabulary). Re-run or author manually before "
            f"promote.", err=True)
    if out_path:
        print_success(f"  wrote {out_path}")
        print_warning("  operator: confirm top_passage_answers per dry-run before "
                      "promoting into refusal_probes.json.")
    sys.exit(0)


@main.command("gold-promote")
@click.option("--course", "-c", required=True, help="Course slug whose accepted candidates to promote")
@click.option("--freeze", is_flag=True,
              help="Set frozen:true + pin the live chunkset sha after promotion.")
@click.option("--dry-run", is_flag=True, help="Build the promote report without writing.")
@click.pass_context
def gold_promote(ctx, course: str, freeze: bool, dry_run: bool):
    """Merge operator-accepted gold candidates into gold_set.json.

    ACCEPT CONVENTION: a candidate is promoted when its authoring.status is
    "reviewed" AND reviewed_by is a non-PENDING handle AND it passes the
    deterministic pre-screen. Renumbers gq-<slug>-NNNN continuing from the
    existing set, backfills provenance, re-validates fail-closed (refusing the
    whole promotion on any critical issue), and re-runs the coverage matrix so
    gaps are visible before --freeze.

    \b
    Example:
        libv2 gold-promote --course demo-course-1 --freeze
    """
    from lib.retrieval.gold_authoring import GoldPromoteError, promote_candidates

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    try:
        report, written = promote_candidates(
            course_dir, freeze=freeze, dry_run=dry_run
        )
    except GoldPromoteError as exc:
        print_error(f"{exc.code}: {exc.detail}")
        sys.exit(1)

    print(f"gold-promote {course}: promoted {len(report.promoted_ids)} "
          f"candidate(s); skipped {len(report.skipped)}, refused {len(report.refused)}.")
    if report.promoted_ids:
        print(f"  new ids: {', '.join(report.promoted_ids[:10])}"
              + (" ..." if len(report.promoted_ids) > 10 else ""))
    for r in report.refused:
        print_warning(f"  refused {r.get('candidate')}: {r.get('reason')} "
                      f"{r.get('details', '')}")
    if dry_run:
        print_warning("  dry-run: gold set not written.")
    elif written:
        print_success(f"  wrote {written}")
        try:
            from lib.retrieval.gold_coverage import write_coverage_report
            from lib.retrieval.gold_set import load_gold_set

            gold, _ = load_gold_set(course_dir, verify=False)
            kind = (gold.get("chunkset") or {}).get("kind")
            cov, cov_path = write_coverage_report(
                course_dir, gold, is_union=(kind == "corpus")
            )
            print("\nCoverage matrix (post-promote):")
            print(f"  by_type:       {cov.by_type}")
            print(f"  by_week:       {cov.by_week}")
            print(f"  by_population: {cov.by_population}")
            print(f"  by_difficulty: {cov.by_difficulty}")
            for w in cov.warnings:
                print_warning(f"  [{w.code}] {w.message}")
            print_success(f"  wrote {cov_path}")
        except Exception as exc:  # noqa: BLE001 -- coverage is advisory
            print_warning(f"  coverage report skipped: {exc}")
    if report.frozen:
        print_success("  gold set FROZEN.")
    sys.exit(0)


@main.command("gold-metadata-backfill")
@click.option("--course", "-c", required=True,
              help="Course slug whose gold-set metadata gaps to propose backfill for")
@click.option("--no-write", is_flag=True, help="Build the proposal without writing the artifact.")
@click.pass_context
def gold_metadata_backfill(ctx, course: str, no_write: bool):
    """Propose difficulty / expected_citation_population for gold questions
    missing them.

    Rule-based + deterministic (no LLM): expected_citation_population is
    inferred from the relevant_passages' chunk populations; difficulty from the
    §1.3 heuristic over passage count + distinct content weeks. Writes a
    PROPOSAL artifact (retrieval_eval/gold_metadata_backfill_proposal.json) for
    operator review -- it is NEVER auto-applied to gold_set.json (the frozen set
    is the canonical eval pin).

    \b
    Example:
        libv2 gold-metadata-backfill --course demo-course-1
    """
    from lib.retrieval.gold_metadata_backfill import generate_backfill_proposal

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    try:
        doc, out_path = generate_backfill_proposal(course_dir, write=not no_write)
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)

    print(f"gold-metadata-backfill {course}: {doc.get('n_questions')} "
          f"question(s) with proposed metadata.")
    for p in doc.get("proposals", []):
        bits = []
        if "proposed_difficulty" in p:
            bits.append(f"difficulty={p['proposed_difficulty']}")
        if "proposed_population" in p:
            bits.append(f"population={p['proposed_population']}")
        print(f"  {p.get('question_id')}: {', '.join(bits)}")
    if out_path:
        print_success(f"  wrote {out_path}")
        print_warning("  PROPOSAL ONLY -- not applied to gold_set.json; operator reviews.")
    else:
        print_warning("  no-write: artifact not written.")
    sys.exit(0)


@main.command("gold-key-points")
@click.option("--course", "-c", required=True,
              help="Course slug whose answerable questions missing expected_key_points to backfill")
@click.option("--no-write", is_flag=True, help="Build the proposal without writing the artifact.")
@click.pass_context
def gold_key_points(ctx, course: str, no_write: bool):
    """Draft expected_key_points for gold questions missing them (LOCAL model).

    For every non-refusal question carrying fewer than 2 expected_key_points
    (the GOLD_QUESTION_METADATA_INCOMPLETE warning), the license-clean LOCAL
    provider drafts 2-4 claim-level key points grounded ONLY in that question's
    own cited relevant-passage bodies. Writes a PROPOSAL artifact
    (retrieval_eval/gold_key_points_proposal.json) -- NEVER applied to
    gold_set.json (the frozen set is the canonical eval pin; the orchestrator
    applies after review).

    \b
    Example:
        libv2 gold-key-points --course demo-course-1
    """
    from lib.decision_capture import DecisionCapture
    from lib.retrieval.answer_backend import (
        AnswerBackendUnavailable,
        AnswerProviderNotLocal,
        build_answer_client,
    )
    from lib.retrieval.gold_key_points_backfill import generate_key_points_proposal

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    capture = DecisionCapture(course_code=course, phase="libv2-answer", tool="libv2")
    try:
        client = build_answer_client(capture=capture)
    except AnswerProviderNotLocal as exc:
        print_error(f"key-point drafting requires a loopback-local provider: {exc}")
        sys.exit(1)
    except AnswerBackendUnavailable as exc:
        print_error(f"local provider unavailable: {exc}")
        sys.exit(1)

    try:
        doc, out_path = generate_key_points_proposal(
            course_dir, client=client, capture=capture, write=not no_write
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)

    print(f"gold-key-points {course}: {doc.get('n_drafted_ok')}/{doc.get('n_questions')} "
          f"question(s) with drafted key points.")
    for p in doc.get("proposals", [])[:10]:
        kps = p.get("proposed_key_points") or []
        flag = "" if not p.get("error") else f" [NEEDS OPERATOR: {p['error']}]"
        print(f"  {p.get('question_id')}: {len(kps)} point(s){flag}")
    if out_path:
        print_success(f"  wrote {out_path}")
        print_warning("  PROPOSAL ONLY -- not applied to gold_set.json; operator reviews.")
    else:
        print_warning("  no-write: artifact not written.")
    sys.exit(0)


@main.command("gold-difficulty-regrade")
@click.option("--course", "-c", required=True,
              help="Course slug whose easy questions to re-grade against the rubric")
@click.option("--from", "regrade_from", default="easy",
              type=click.Choice(["easy", "medium", "hard"]),
              help="Difficulty band to re-grade (default easy -- the skewed band).")
@click.option("--no-write", is_flag=True, help="Build the proposal without writing the artifact.")
@click.pass_context
def gold_difficulty_regrade(ctx, course: str, regrade_from: str, no_write: bool):
    """Re-grade gold-question difficulty against the easy/medium/hard rubric (LOCAL model).

    Re-grades every question currently tagged --from (default easy -- the
    over-filled band) against the written rubric (easy=single-passage recall;
    medium=multi-step procedure or within-section synthesis; hard=cross-section
    synthesis / multi_part / error-analysis). HONEST -- no quota filling; a
    genuinely-easy question keeps easy. Writes a PROPOSAL artifact
    (retrieval_eval/gold_difficulty_regrade_proposal.json) with the proposed
    full-set distribution vs the coverage bands -- NEVER applied to
    gold_set.json (the orchestrator applies after review).

    \b
    Example:
        libv2 gold-difficulty-regrade --course demo-course-1
    """
    from lib.decision_capture import DecisionCapture
    from lib.retrieval.answer_backend import (
        AnswerBackendUnavailable,
        AnswerProviderNotLocal,
        build_answer_client,
    )
    from lib.retrieval.gold_difficulty_regrade import (
        generate_difficulty_regrade_proposal,
    )

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    capture = DecisionCapture(course_code=course, phase="libv2-answer", tool="libv2")
    try:
        client = build_answer_client(capture=capture)
    except AnswerProviderNotLocal as exc:
        print_error(f"regrade requires a loopback-local provider: {exc}")
        sys.exit(1)
    except AnswerBackendUnavailable as exc:
        print_error(f"local provider unavailable: {exc}")
        sys.exit(1)

    try:
        doc, out_path = generate_difficulty_regrade_proposal(
            course_dir, client=client, capture=capture,
            regrade_from=regrade_from, write=not no_write,
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)

    dist = doc.get("distribution", {})
    print(f"gold-difficulty-regrade {course}: regraded {doc.get('n_regraded')} "
          f"'{regrade_from}' question(s); {doc.get('n_changed')} reclassified, "
          f"{doc.get('n_failures')} failure(s).")
    print(f"  distribution before: {dist.get('before')}")
    print(f"  distribution after:  {dist.get('after')}")
    after_bands = dist.get("after_bands", {})
    for d, info in after_bands.items():
        status = "in-band" if info.get("in_band") else "OUT-OF-BAND"
        print(f"    {d}: {info.get('count')} ({info.get('fraction'):.0%}) "
              f"band {info.get('band')} -> {status}")
    for r in doc.get("rows", [])[:10]:
        if r.get("changed"):
            print(f"  {r.get('question_id')}: {r.get('current')} -> {r.get('proposed')}")
    if out_path:
        print_success(f"  wrote {out_path}")
        print_warning("  PROPOSAL ONLY -- not applied to gold_set.json; operator reviews.")
    else:
        print_warning("  no-write: artifact not written.")
    sys.exit(0)


@main.command("gold-parts")
@click.option("--course", "-c", required=True,
              help="Course slug whose multi_part questions' parts[] to author")
@click.option("--no-retrieval", is_flag=True,
              help="Bind parts from the parent passages only (skip per-part corpus retrieval).")
@click.option("--no-write", is_flag=True, help="Build the proposal without writing the artifact.")
@click.pass_context
def gold_parts(ctx, course: str, no_retrieval: bool, no_write: bool):
    """Author parts[] for multi_part gold questions (LOCAL model).

    Decomposes each multi_part question's text into its real sub-parts via the
    license-clean LOCAL provider, binds each part's relevant_passage_refs from
    the parent question's relevant_passages (4-shingle overlap, then per-part
    corpus retrieval), and marks a part covered:false + absence_note ONLY when
    both binding paths come back empty (conservative ws3.v2 completeness probe).
    Writes a PROPOSAL artifact (retrieval_eval/gold_parts_proposal.json) of
    schema-legal part objects -- NEVER applied to gold_set.json (the
    orchestrator applies after review).

    \b
    Example:
        libv2 gold-parts --course demo-course-1
    """
    from lib.decision_capture import DecisionCapture
    from lib.retrieval.answer_backend import (
        AnswerBackendUnavailable,
        AnswerProviderNotLocal,
        build_answer_client,
    )
    from lib.retrieval.gold_parts_authoring import generate_parts_proposal
    from LibV2.tools.libv2.retriever import retrieve_chunks

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    capture = DecisionCapture(course_code=course, phase="libv2-answer", tool="libv2")
    try:
        client = build_answer_client(capture=capture)
    except AnswerProviderNotLocal as exc:
        print_error(f"parts authoring requires a loopback-local provider: {exc}")
        sys.exit(1)
    except AnswerBackendUnavailable as exc:
        print_error(f"local provider unavailable: {exc}")
        sys.exit(1)

    retrieve_fn = None if no_retrieval else retrieve_chunks
    libv2_root = None if no_retrieval else repo_root
    try:
        doc, out_path = generate_parts_proposal(
            course_dir, client=client, retrieve_fn=retrieve_fn,
            libv2_root=libv2_root, capture=capture, write=not no_write,
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)

    print(f"gold-parts {course}: authored parts for {doc.get('n_authored_ok')}/"
          f"{doc.get('n_questions')} multi_part question(s).")
    for p in doc.get("proposals", [])[:12]:
        flag = "" if not p.get("error") else f" [FAILED: {p['error']}]"
        unc = p.get("n_uncovered", 0)
        unc_s = f", {unc} uncovered" if unc else ""
        print(f"  {p.get('question_id')}: {p.get('n_parts')} part(s){unc_s}{flag}")
    if out_path:
        print_success(f"  wrote {out_path}")
        print_warning("  PROPOSAL ONLY -- not applied to gold_set.json; operator reviews.")
    else:
        print_warning("  no-write: artifact not written.")
    sys.exit(0)


@main.command("gold-enrich-passages")
@click.option("--course", "-c", required=True,
              help="Course slug whose gold-set relevant_passages to enrich")
@click.option("--engine", type=click.Choice(["lexical", "semantic", "hybrid-rrf"]),
              default="lexical", help="Retrieval engine for the candidate top-k pool.")
@click.option("--top-k", type=int, default=10,
              help="Retrieval candidate pool depth per question (default 10).")
@click.option("--no-retrieval", is_flag=True,
              help="Source candidates from eval-observed citations only (skip retrieval top-k).")
@click.option("--promote", is_flag=True,
              help="PROMOTE: apply operator-accepted ('status: accepted') passages from "
                   "the proposal into gold_set.json (fail-closed validate + frozen gate).")
@click.option("--unfreeze-for-enrich", is_flag=True,
              help="Required (with --promote) to enrich a frozen:true gold set.")
@click.option("--no-write", is_flag=True, help="Build the proposal without writing the artifact.")
@click.option("--dry-run", is_flag=True,
              help="With --promote: build the promote report without writing the gold set.")
@click.pass_context
def gold_enrich_passages(ctx, course: str, engine: str, top_k: int,
                         no_retrieval: bool, promote: bool,
                         unfreeze_for_enrich: bool, no_write: bool, dry_run: bool):
    """Propose (and, with --promote, apply) ADDITIONAL relevant_passages.

    Default (proposal): for each gold question, sources candidate chunks from
    retrieval top-k UNION the chunks the tutor actually cited (when persisted),
    scores each against the question's expected_key_points with the EXACT
    citation-attribution support floors the eval already trusts (4-shingle
    >= 0.25 OR token-coverage >= 0.80; no new thresholds), and proposes the
    genuinely-supporting-but-unpinned ones as relevance:supporting passages.
    The single existing primary is NEVER touched. Writes a PROPOSAL artifact
    (retrieval_eval/gold_passage_enrichment_proposal.json); NOT applied.

    With --promote: applies the proposal's operator-accepted passages
    (status:accepted) into gold_set.json -- a SEPARATE explicit step -- behind
    the fail-closed validate_gold_set gate and a frozen unfreeze gate
    (--unfreeze-for-enrich), then re-pins the chunkset sha.

    \b
    Examples:
        libv2 gold-enrich-passages --course demo-course-1
        libv2 gold-enrich-passages --course demo-course-1 --promote --unfreeze-for-enrich
    """
    from lib.retrieval.gold_passage_enrichment import (
        GoldEnrichError,
        generate_enrichment_proposal,
        promote_enrichment,
    )
    from LibV2.tools.libv2.retriever import retrieve_chunks

    repo_root: Path = ctx.obj["repo_root"]
    course_dir = repo_root / "courses" / course
    if not course_dir.exists():
        print_error(f"course not found: {course_dir}")
        sys.exit(1)

    if promote:
        try:
            report, written = promote_enrichment(
                course_dir, unfreeze_for_enrich=unfreeze_for_enrich, dry_run=dry_run,
            )
        except GoldEnrichError as exc:
            print_error(f"{exc.code}: {exc.detail}")
            sys.exit(1)
        print(f"gold-enrich-passages {course}: applied {report.n_applied} "
              f"supporting passage(s); skipped {len(report.skipped)}.")
        for a in report.applied[:12]:
            print(f"  + {a.get('question_id')} <- {a.get('chunk_id')} (supporting)")
        if dry_run:
            print_warning("  dry-run: gold set not written.")
        elif written:
            print_success(f"  wrote {written}")
        sys.exit(0)

    retrieve_fn = None if no_retrieval else retrieve_chunks
    libv2_root = None if no_retrieval else repo_root
    try:
        doc, out_path = generate_enrichment_proposal(
            course_dir, retrieve_fn=retrieve_fn, libv2_root=libv2_root,
            engine=engine, top_k=top_k, write=not no_write,
        )
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)

    print(f"gold-enrich-passages {course}: {doc.get('n_proposed_passages')} "
          f"proposed supporting passage(s) across "
          f"{doc.get('n_questions_with_proposals')}/{doc.get('n_questions')} question(s).")
    if out_path:
        print_success(f"  wrote {out_path}")
        print_warning("  PROPOSAL ONLY -- primary untouched; set status:accepted then "
                      "re-run with --promote.")
    else:
        print_warning("  no-write: artifact not written.")
    sys.exit(0)


if __name__ == "__main__":
    main()
