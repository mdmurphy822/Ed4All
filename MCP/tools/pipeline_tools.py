"""
Ed4All Pipeline Tools

MCP tools for the unified textbook-to-course pipeline.
Chains: DART (PDF -> HTML) -> Courseforge (course generation) -> Trainforge (assessments)
"""

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import PROJECT_ROOT  # noqa: E402
from lib.secure_paths import validate_path_within_root  # noqa: E402

logger = logging.getLogger(__name__)

# Derived paths
DART_OUTPUT_DIR = PROJECT_ROOT / "DART" / "batch_output"
COURSEFORGE_INPUTS = PROJECT_ROOT / "Courseforge" / "inputs" / "textbooks"
TRAINING_CAPTURES = PROJECT_ROOT / "training-captures"


def _ensure_directories():
    """Ensure required directories exist."""
    for path in [COURSEFORGE_INPUTS, TRAINING_CAPTURES]:
        path.mkdir(parents=True, exist_ok=True)


_ensure_directories()


def _detect_source_provenance(course_dir: Path) -> bool:
    """Wave 10: scan archived chunks.jsonl for chunks with source_references[].

    Returns True when at least one chunk in ``<course_dir>/corpus/chunks.jsonl``
    carries ``source.source_references[]`` populated with at least one entry.
    Returns False on missing file, read errors, malformed JSONL lines, or
    when no chunks carry refs (pre-Wave-9 corpus). The manifest then advertises
    ``features.source_provenance: false`` so LibV2 retrieval callers can
    fast-skip source-grounded queries.
    """
    chunks_path = course_dir / "corpus" / "chunks.jsonl"
    if not chunks_path.exists() or not chunks_path.is_file():
        return False
    try:
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(chunk, dict):
                    continue
                source = chunk.get("source")
                if not isinstance(source, dict):
                    continue
                refs = source.get("source_references")
                if isinstance(refs, list) and len(refs) > 0:
                    return True
    except OSError:
        return False
    return False


def _detect_evidence_source_provenance(course_dir: Path) -> bool:
    """Wave 11: scan archived concept_graph_semantic.json for evidence-level refs.

    Returns True when at least one edge in the archived concept graph's
    ``edges[].provenance.evidence`` carries a populated ``source_references[]``
    array. False on missing file, read errors, malformed JSON, or when no
    edges carry evidence refs. The manifest then advertises
    ``features.evidence_source_provenance: true/false`` so LibV2 retrieval
    callers can distinguish chunk-level (Wave 10) from evidence-level (Wave 11)
    provenance.

    The scan looks in three candidate locations under ``<course_dir>``:
    ``graph/concept_graph_semantic.json``, ``corpus/concept_graph_semantic.json``,
    or any ``*.json`` file shaped like a semantic graph (``kind ==
    "concept_semantic"``) sitting inside the corpus dir. First match wins.
    """
    candidates = [
        course_dir / "graph" / "concept_graph_semantic.json",
        course_dir / "corpus" / "concept_graph_semantic.json",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    graph = json.load(f)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if _graph_has_evidence_refs(graph):
                return True
            # First readable candidate wins — don't fall through to others
            # if this one was valid shape but carried no refs.
            return False
    return False


def _graph_has_evidence_refs(graph: object) -> bool:
    """Return True iff the graph has at least one edge whose
    ``provenance.evidence.source_references`` is a non-empty list.

    Tolerates partial / legacy shapes: silently returns False on any
    structural surprise rather than raising.
    """
    if not isinstance(graph, dict):
        return False
    edges = graph.get("edges")
    if not isinstance(edges, list):
        return False
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        provenance = edge.get("provenance")
        if not isinstance(provenance, dict):
            continue
        evidence = provenance.get("evidence")
        if not isinstance(evidence, dict):
            continue
        refs = evidence.get("source_references")
        if isinstance(refs, list) and len(refs) > 0:
            return True
    return False


async def create_textbook_pipeline(
    pdf_paths: str,
    course_name: str,
    objectives_path: Optional[str] = None,
    duration_weeks: int = 12,
    generate_assessments: bool = True,
    assessment_count: int = 50,
    bloom_levels: str = "remember,understand,apply,analyze",
    priority: str = "normal"
) -> str:
    """
    Create and orchestrate a textbook-to-course pipeline.

    Chains: DART (PDF->HTML) -> Courseforge (course generation) -> Trainforge (assessments)

    This is a standalone function importable by both the MCP server and CLI.

    Args:
        pdf_paths: Comma-separated PDF paths OR directory containing PDFs
        course_name: Course identifier (e.g., "PHYS_101")
        objectives_path: Optional external objectives file to merge
        duration_weeks: Course duration in weeks (default: 12)
        generate_assessments: Run Trainforge phase (default: True)
        assessment_count: Questions to generate (default: 50)
        bloom_levels: Target Bloom levels (default: remember,understand,apply,analyze)
        priority: Workflow priority (low/normal/high)

    Returns:
        JSON with workflow_id, run_id, and status
    """
    try:
        from MCP.tools.orchestrator_tools import create_workflow_impl

        # Parse PDF paths
        pdf_path = Path(pdf_paths)
        if pdf_path.is_dir():
            pdfs = list(pdf_path.glob("*.pdf"))
            if not pdfs:
                return json.dumps({"error": f"No PDF files found in directory: {pdf_paths}"})
        else:
            pdfs = [Path(p.strip()) for p in pdf_paths.split(",")]

        # Validate PDF paths are within project root
        for pdf in pdfs:
            try:
                validate_path_within_root(pdf.resolve(), PROJECT_ROOT)
            except ValueError as e:
                return json.dumps({"error": f"PDF path validation failed: {e}"})

        # Validate inputs
        missing_pdfs = [str(p) for p in pdfs if not p.exists()]
        if missing_pdfs:
            return json.dumps({"error": f"PDF files not found: {missing_pdfs}"})

        if objectives_path and not Path(objectives_path).exists():
            return json.dumps({"error": f"Objectives file not found: {objectives_path}"})

        # Validate course name format
        if not course_name or len(course_name) < 2:
            return json.dumps({"error": "Course name must be at least 2 characters"})

        # Generate run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"TTC_{course_name}_{timestamp}"

        # Build workflow parameters
        params = {
            "pdf_paths": [str(p.resolve()) for p in pdfs],
            "course_name": course_name,
            "objectives_path": str(Path(objectives_path).resolve()) if objectives_path else None,
            "duration_weeks": duration_weeks,
            "generate_assessments": generate_assessments,
            "assessment_count": assessment_count,
            "bloom_levels": [level.strip() for level in bloom_levels.split(",")],
            "run_id": run_id
        }

        # Create workflow via orchestrator
        result = await create_workflow_impl(
            workflow_type="textbook_to_course",
            params=json.dumps(params),
            priority=priority
        )

        result_data = json.loads(result)

        if result_data.get("success"):
            # Add run_id to response
            result_data["run_id"] = run_id
            result_data["params"] = params

            # Create training captures directory for this run
            captures_dir = TRAINING_CAPTURES / "textbook-pipeline" / course_name
            captures_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Created textbook_to_course pipeline: {result_data.get('workflow_id')}")

        return json.dumps(result_data)

    except Exception as e:
        logger.error(f"Failed to create textbook pipeline: {e}")
        return json.dumps({"error": str(e)})


async def run_textbook_pipeline(workflow_id: str) -> str:
    """
    Execute a textbook-to-course pipeline that was previously created.

    Standalone function importable by both MCP server and CLI.

    Runs all phases in dependency order:
    DART conversion -> Staging -> Objective extraction -> Course planning ->
    Content generation -> IMSCC packaging -> Trainforge assessment ->
    LibV2 archival -> Finalization

    Args:
        workflow_id: The workflow ID returned by create_textbook_pipeline

    Returns:
        JSON with final status, phase results, and output paths
    """
    try:
        from MCP.core.config import OrchestratorConfig
        from MCP.core.executor import TaskExecutor
        from MCP.core.workflow_runner import WorkflowRunner

        # Load orchestrator config
        config = OrchestratorConfig.load()

        # Create executor with tool registry
        tool_registry = _build_tool_registry()

        executor = TaskExecutor(tool_registry=tool_registry)

        # Create and run the workflow runner
        runner = WorkflowRunner(executor, config)
        result = await runner.run_workflow(workflow_id)

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "workflow_id": workflow_id,
        })


def register_pipeline_tools(mcp):
    """Register pipeline tools with the MCP server."""

    @mcp.tool()
    async def create_textbook_pipeline_tool(
        pdf_paths: str,
        course_name: str,
        objectives_path: Optional[str] = None,
        duration_weeks: int = 12,
        generate_assessments: bool = True,
        assessment_count: int = 50,
        bloom_levels: str = "remember,understand,apply,analyze",
        priority: str = "normal"
    ) -> str:
        """Create and orchestrate a textbook-to-course pipeline.

        NOTE: This tool is intended for internal / programmatic use.
        End users should prefer the ``ed4all run textbook-to-course`` CLI,
        which wraps creation + execution through ``PipelineOrchestrator``.
        """
        return await create_textbook_pipeline(
            pdf_paths, course_name, objectives_path, duration_weeks,
            generate_assessments, assessment_count, bloom_levels, priority
        )

    @mcp.tool()
    async def stage_dart_outputs(
        run_id: str,
        dart_html_paths: str,
        course_name: str
    ) -> str:
        """
        Stage DART outputs to Courseforge inputs directory.

        Copies synthesized HTML and JSON files from DART output to the
        Courseforge staging area for course generation.

        Args:
            run_id: Pipeline run identifier
            dart_html_paths: Comma-separated paths to DART HTML outputs
            course_name: Course identifier for staging subdirectory

        Returns:
            JSON with staging_dir and staged_files list
        """
        try:
            # Create staging directory
            staging_dir = COURSEFORGE_INPUTS / run_id
            staging_dir.mkdir(parents=True, exist_ok=True)

            staged_files = []
            # Wave 8: role-tagged manifest entries for the downstream
            # Courseforge source-router and Trainforge parser. Roles:
            #   "content"             -> the rendered HTML page
            #   "provenance_sidecar"  -> *_synthesized.json with per-block provenance
            #   "quality_sidecar"     -> *.quality.json with WCAG + confidence aggregates
            staged_entries = []
            errors = []

            html_paths = [Path(p.strip()) for p in dart_html_paths.split(",")]

            for html_path in html_paths:
                if not html_path.exists():
                    errors.append(f"DART output not found: {html_path}")
                    continue

                # Copy HTML file (role=content)
                dest = staging_dir / html_path.name
                shutil.copy2(html_path, dest)
                staged_files.append(str(dest))
                staged_entries.append({"path": html_path.name, "role": "content"})
                logger.info(f"Staged: {html_path.name} -> {dest}")

                # Validate HTML structure
                if html_path.suffix.lower() in ('.html', '.htm'):
                    try:
                        content = dest.read_text(encoding='utf-8', errors='ignore')[:5000]
                        content_lower = content.lower()
                        if '<html' not in content_lower and '<body' not in content_lower:
                            errors.append(
                                f"Warning: {html_path.name} may not be valid HTML "
                                f"(missing <html> and <body> tags)"
                            )
                    except OSError:
                        pass  # File was copied, just can't validate

                # Copy accompanying JSON if exists (DART synthesized metadata)
                json_path = html_path.with_suffix(".json")
                if json_path.exists():
                    json_dest = staging_dir / json_path.name
                    shutil.copy2(json_path, json_dest)
                    staged_files.append(str(json_dest))
                    staged_entries.append({
                        "path": json_path.name,
                        "role": "provenance_sidecar",
                    })
                    logger.info(f"Staged: {json_path.name} -> {json_dest}")

                # Also check for _synthesized.json pattern
                synth_json_name = html_path.stem.replace("_synthesized", "") + "_synthesized.json"
                synth_json_path = html_path.parent / synth_json_name
                if synth_json_path.exists() and str(synth_json_path) != str(json_path):
                    synth_json_dest = staging_dir / synth_json_name
                    shutil.copy2(synth_json_path, synth_json_dest)
                    staged_files.append(str(synth_json_dest))
                    staged_entries.append({
                        "path": synth_json_name,
                        "role": "provenance_sidecar",
                    })
                    logger.info(f"Staged: {synth_json_name} -> {synth_json_dest}")

                # Wave 8: also stage the DART quality sidecar if one exists.
                # Convention: same stem as the HTML, suffix .quality.json.
                # E.g. "science_of_learning.html" -> "science_of_learning.quality.json".
                # The legacy stage_dart_outputs never copied this even though
                # DART's convert_single_pdf has been writing it all along.
                quality_name = html_path.stem + ".quality.json"
                quality_path = html_path.parent / quality_name
                if quality_path.exists():
                    quality_dest = staging_dir / quality_name
                    shutil.copy2(quality_path, quality_dest)
                    staged_files.append(str(quality_dest))
                    staged_entries.append({
                        "path": quality_name,
                        "role": "quality_sidecar",
                    })
                    logger.info(f"Staged: {quality_name} -> {quality_dest}")

            if errors and not staged_files:
                return json.dumps({
                    "success": False,
                    "error": "No files staged",
                    "errors": errors
                })

            # Create manifest (Wave 8: role-tagged entries under "files")
            manifest = {
                "run_id": run_id,
                "course_name": course_name,
                "staged_at": datetime.now().isoformat(),
                "staged_files": staged_files,            # back-compat flat list
                "files": staged_entries,                 # role-tagged entries
                "errors": errors if errors else None,
            }

            manifest_path = staging_dir / "staging_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            return json.dumps({
                "success": True,
                "staging_dir": str(staging_dir),
                "staged_files": staged_files,
                "files": staged_entries,
                "file_count": len(staged_files),
                "manifest_path": str(manifest_path),
                "warnings": errors if errors else None
            })

        except Exception as e:
            logger.error(f"Failed to stage DART outputs: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_pipeline_status(workflow_id: str) -> str:
        """
        Get status of a textbook-to-course pipeline.

        Args:
            workflow_id: The workflow ID returned by create_textbook_pipeline

        Returns:
            JSON with current phase, progress, and phase outputs
        """
        try:
            # Read workflow state directly (get_workflow_status is a closure
            # inside register_orchestrator_tools, not importable at module level)
            workflow_path = PROJECT_ROOT / "state" / "workflows" / f"{workflow_id}.json"
            if not workflow_path.exists():
                return json.dumps({"error": f"Workflow not found: {workflow_id}"})
            with open(workflow_path) as f:
                workflow = json.load(f)

            # Enhance with pipeline-specific information
            params = workflow.get("params", {})

            pipeline_status = {
                "workflow_id": workflow.get("id"),
                "workflow_type": workflow.get("type"),
                "status": workflow.get("status"),
                "run_id": params.get("run_id"),
                "course_name": params.get("course_name"),
                "progress": workflow.get("progress"),
                "created_at": workflow.get("created_at"),
                "updated_at": workflow.get("updated_at"),
                "phases": {
                    "dart_conversion": _get_phase_status(workflow, "dart_conversion"),
                    "staging": _get_phase_status(workflow, "staging"),
                    "objective_extraction": _get_phase_status(workflow, "objective_extraction"),
                    "course_planning": _get_phase_status(workflow, "course_planning"),
                    "content_generation": _get_phase_status(workflow, "content_generation"),
                    "packaging": _get_phase_status(workflow, "packaging"),
                    "trainforge_assessment": _get_phase_status(workflow, "trainforge_assessment"),
                    "libv2_archival": _get_phase_status(workflow, "libv2_archival"),
                    "finalization": _get_phase_status(workflow, "finalization")
                },
                "params": params
            }

            return json.dumps(pipeline_status)

        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def validate_dart_markers(html_path: str) -> str:
        """
        Validate that an HTML file has required DART markers.

        DART-processed HTML must have:
        - Skip link (<a class="skip-link">)
        - Main content area (<main role="main">)
        - Semantic sections (<section aria-labelledby="...">)

        Args:
            html_path: Path to HTML file to validate

        Returns:
            JSON with validation results
        """
        try:
            path = Path(html_path)
            if not path.exists():
                return json.dumps({"error": f"File not found: {html_path}"})

            with open(path, encoding="utf-8") as f:
                content = f.read()

            markers = {
                "skip_link": 'class="skip' in content or "class='skip" in content,
                "main_role": 'role="main"' in content or "role='main'" in content,
                "aria_sections": 'aria-labelledby="' in content or "aria-labelledby='" in content,
                "dart_semantic_classes": 'dart-section' in content or 'dart-document' in content
            }

            all_valid = all(markers.values())

            result = {
                "valid": all_valid,
                "file": str(path),
                "markers": markers,
                "missing": [k for k, v in markers.items() if not v]
            }

            if not all_valid:
                result["message"] = f"Missing DART markers: {result['missing']}"

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Failed to validate DART markers: {e}")
            return json.dumps({"error": str(e)})


    @mcp.tool()
    async def archive_to_libv2(
        course_name: str,
        domain: str,
        division: str = "STEM",
        pdf_paths: Optional[str] = None,
        html_paths: Optional[str] = None,
        imscc_path: Optional[str] = None,
        assessment_path: Optional[str] = None,
        subdomains: Optional[str] = None,
    ) -> str:
        """
        Archive all pipeline artifacts to LibV2 unified repository.

        Stores raw inputs (PDFs), DART outputs (HTML), course packages (IMSCC),
        and RAG corpus together under a single course slug.

        Args:
            course_name: Course identifier (e.g., "PHYS_101")
            domain: Primary domain (e.g., "physics", "computer-science")
            division: Division classification ("STEM" or "ARTS", default: "STEM")
            pdf_paths: Comma-separated paths to original PDF inputs
            html_paths: Comma-separated paths to DART HTML outputs
            imscc_path: Path to Courseforge IMSCC package
            assessment_path: Path to Trainforge assessment JSON
            subdomains: Comma-separated subdomains (e.g., "mechanics,thermodynamics")

        Returns:
            JSON with course_slug, storage paths, and archival status
        """
        try:
            libv2_root = PROJECT_ROOT / "LibV2"

            # Generate slug from course name
            slug = course_name.lower().replace("_", "-").replace(" ", "-")

            # Create course directory structure
            course_dir = libv2_root / "courses" / slug
            for subdir in [
                "source/pdf", "source/html", "source/imscc",
                "corpus", "graph", "pedagogy", "training_specs", "quality"
            ]:
                (course_dir / subdir).mkdir(parents=True, exist_ok=True)

            archived = {"pdfs": [], "html": [], "imscc": None, "assessment": None}

            # Archive raw PDFs
            if pdf_paths:
                for pdf_str in pdf_paths.split(","):
                    pdf = Path(pdf_str.strip())
                    if pdf.exists():
                        dest = course_dir / "source" / "pdf" / pdf.name
                        shutil.copy2(pdf, dest)
                        archived["pdfs"].append(str(dest))

            # Archive DART HTML outputs
            if html_paths:
                for html_str in html_paths.split(","):
                    html_file = Path(html_str.strip())
                    if html_file.exists():
                        dest = course_dir / "source" / "html" / html_file.name
                        shutil.copy2(html_file, dest)
                        archived["html"].append(str(dest))
                        # Also copy quality JSON if present
                        quality_json = html_file.with_suffix(".quality.json")
                        if quality_json.exists():
                            shutil.copy2(
                                quality_json,
                                course_dir / "quality" / quality_json.name
                            )

            # Archive IMSCC package
            if imscc_path:
                imscc = Path(imscc_path)
                if imscc.exists():
                    dest = course_dir / "source" / "imscc" / imscc.name
                    shutil.copy2(imscc, dest)
                    archived["imscc"] = str(dest)

            # Archive assessment / RAG corpus output
            if assessment_path:
                assess = Path(assessment_path)
                if assess.exists():
                    dest = course_dir / "corpus" / assess.name
                    shutil.copy2(assess, dest)
                    archived["assessment"] = str(dest)

            # Build manifest
            import hashlib

            def _sha256(filepath: Path) -> str:
                h = hashlib.sha256()
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                return h.hexdigest()

            source_artifacts = {}
            if archived["pdfs"]:
                source_artifacts["pdf"] = [
                    {"path": p, "checksum": _sha256(Path(p)), "size": Path(p).stat().st_size}
                    for p in archived["pdfs"]
                ]
            if archived["html"]:
                source_artifacts["html"] = [
                    {"path": p, "checksum": _sha256(Path(p)), "size": Path(p).stat().st_size}
                    for p in archived["html"]
                ]
            if archived["imscc"]:
                imscc_p = Path(archived["imscc"])
                source_artifacts["imscc"] = {
                    "path": archived["imscc"],
                    "checksum": _sha256(imscc_p),
                    "size": imscc_p.stat().st_size,
                }

            # Wave 10: advisory feature flag — scan the archived corpus's
            # chunks.jsonl (if any) for chunks carrying
            # source.source_references[]. Lets LibV2 retrieval callers
            # fast-skip source-grounded queries on legacy corpora.
            # Defaults false when no chunks file is found, when it can't
            # be read, or when no chunks carry refs.
            source_provenance_flag = _detect_source_provenance(course_dir)

            # Wave 11: companion flag for evidence-arm source_references[].
            # True when the archived concept_graph_semantic.json carries at
            # least one edge with evidence.source_references[]. Lets
            # consumers distinguish chunk-level (Wave 10) from evidence-
            # level (Wave 11) provenance.
            evidence_source_provenance_flag = _detect_evidence_source_provenance(course_dir)

            manifest = {
                "libv2_version": "1.2.0",
                "slug": slug,
                "import_timestamp": datetime.now().isoformat(),
                "classification": {
                    "division": division,
                    "primary_domain": domain,
                    "subdomains": [s.strip() for s in subdomains.split(",")] if subdomains else [],
                },
                "source_artifacts": source_artifacts,
                "provenance": {
                    "source_type": "textbook_to_course_pipeline",
                    "import_pipeline_version": "1.0.0",
                },
                "features": {
                    "source_provenance": source_provenance_flag,
                    "evidence_source_provenance": evidence_source_provenance_flag,
                },
            }

            manifest_path = course_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            return json.dumps({
                "success": True,
                "course_slug": slug,
                "course_dir": str(course_dir),
                "manifest_path": str(manifest_path),
                "archived": archived,
                "artifact_counts": {
                    "pdfs": len(archived["pdfs"]),
                    "html_files": len(archived["html"]),
                    "imscc": 1 if archived["imscc"] else 0,
                    "assessment": 1 if archived["assessment"] else 0,
                },
            })

        except Exception as e:
            logger.error(f"Failed to archive to LibV2: {e}")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def run_textbook_pipeline_tool(workflow_id: str) -> str:
        """Execute a textbook-to-course pipeline that was previously created.

        NOTE: This tool is intended for internal / programmatic use.
        End users should prefer the ``ed4all run`` CLI, which combines
        creation + execution through ``PipelineOrchestrator``.
        """
        return await run_textbook_pipeline(workflow_id)


def _raw_text_to_accessible_html(raw_text: str, title: str) -> str:
    """Convert raw pdftotext output to clean, semantic, WCAG 2.2 AA HTML.

    Performs thorough cleaning:
    - Strips standalone page numbers, TOC entries, headers/footers
    - Removes repeated book title footers (e.g. "K-12 Blended Teaching")
    - Removes EdTech Books boilerplate and URL footers
    - Skips author biography sections (detects bio patterns)
    - Detects real chapter/section headings from text structure
    - Builds heading hierarchy (h1 → h2 → h3)
    - Wraps content paragraphs in <p> tags
    - Adds WCAG 2.2 AA landmarks (main, skip link, dark mode)
    """
    import html as _html
    import re as _re

    lines = raw_text.split("\n")

    # ---- Pass 1: Identify the book title for footer stripping ----
    # The first non-empty line(s) are usually the book title
    book_title_words = []
    for line in lines[:10]:
        s = line.strip()
        if s and len(s) > 3:
            book_title_words.append(s)
        if len(book_title_words) >= 2:
            break
    book_title_line = " ".join(book_title_words[:2]) if book_title_words else ""

    # ---- Compiled patterns ----
    page_num = _re.compile(r"^\s*\d{1,4}\s*$")
    toc_entry = _re.compile(r"^.{5,60}\s{3,}\d{1,4}\s*$")
    chapter_heading = _re.compile(
        r"^(?:"
        r"(?:Chapter|Part|Section|Unit)\s+\d+[.:]\s*|"
        r"(?:I{1,3}V?|VI{0,3}|IX|X{1,3})\.\s+|"
        r"\d{1,2}\.\s+"
        r")(.+)",
    )
    sub_heading = _re.compile(r"^[A-Z][A-Za-z\s,&:'\-]{5,80}$")

    boilerplate = _re.compile(
        r"(?:"
        r"This content is provided to you freely|"
        r"Access it online or download it at|"
        r"edtechbooks\.org|pressbooks\.pub|"
        r"Like this\? Endorse it|"
        r"Endorse$|"
        r"^CC BY|^ISBN:|"
        r"Watch on YouTube|"
        r"What to Look For:"
        r")",
        _re.IGNORECASE,
    )

    # Bio detection: lines like "University of X" or "Dr. X is a Professor"
    bio_start = _re.compile(
        r"^(?:[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+|"  # "Cecil R. Short"
        r"Dr\. [A-Z]|"
        r"[A-Z][a-z]+ [A-Z][a-z]+)\s*$"  # "Jered Borup" (name-only line)
    )
    university_line = _re.compile(
        r"^(?:University|Brigham Young|Arizona State|George Mason|"
        r"Emporia State|Weber State|[A-Z][a-z]+ (?:University|College|Institute))",
        _re.IGNORECASE,
    )

    # ---- Pass 2: Clean lines ----
    cleaned_lines = []
    in_toc = False
    in_bio = False
    bio_line_count = 0
    prev_was_empty = True

    for line in lines:
        stripped = line.strip()

        # Empty line
        if not stripped:
            if in_bio:
                bio_line_count += 1
                if bio_line_count > 2:
                    in_bio = False  # Bios end after a gap
            cleaned_lines.append("")
            prev_was_empty = True
            continue

        # Skip standalone page numbers
        if page_num.match(stripped):
            continue

        # Skip repeated book title footer
        if book_title_line and stripped == book_title_line.split("\n")[0].strip():
            continue
        # Also match partial book title (just the short title)
        if book_title_words and stripped == book_title_words[0]:
            continue

        # Skip boilerplate
        if boilerplate.search(stripped):
            continue

        # Skip TOC entries (title followed by large whitespace then page number)
        if toc_entry.match(stripped):
            in_toc = True
            continue
        if in_toc:
            if len(stripped) > 40 and not toc_entry.match(stripped):
                in_toc = False
            else:
                continue

        # Detect and skip author bio blocks
        if prev_was_empty and (bio_start.match(stripped) or university_line.match(stripped)):
            in_bio = True
            bio_line_count = 0
            continue
        if in_bio:
            bio_line_count = 0  # Reset counter on non-empty bio line
            # Stay in bio mode for lines that look like bio content
            if (
                len(stripped) < 200
                and (
                    university_line.match(stripped)
                    or "http" in stripped
                    or "@" in stripped
                    or stripped.startswith("Dr.")
                    or "Professor" in stripped
                    or "research" in stripped.lower()
                    or "publications" in stripped.lower()
                )
            ):
                continue
            # If line is long enough to be real content, exit bio mode
            if len(stripped) > 100:
                in_bio = False
            else:
                continue

        cleaned_lines.append(stripped)
        prev_was_empty = False

    # ---- Pass 3: Detect structure and build sections ----
    sections = []
    current_section = {"heading": title, "level": 1, "paragraphs": []}
    current_para = []

    def _flush_para():
        text = " ".join(current_para).strip()
        if text and len(text) > 20:
            current_section["paragraphs"].append(text)
        current_para.clear()

    for stripped in cleaned_lines:
        if not stripped:
            _flush_para()
            continue

        # Detect chapter headings (numbered: "11. Behaviorism..." or "I. Definitions")
        ch_match = chapter_heading.match(stripped)
        if ch_match:
            _flush_para()
            if current_section["paragraphs"] or current_section["heading"] != title:
                sections.append(current_section)
            heading_text = ch_match.group(1).strip() if ch_match.group(1) else stripped
            current_section = {"heading": heading_text, "level": 2, "paragraphs": []}
            continue

        # Detect sub-headings (Title Case, short, standalone after blank)
        if (
            sub_heading.match(stripped)
            and len(stripped.split()) <= 10
            and not current_para  # Must be after a blank line
            and stripped[0].isupper()
            and not stripped.endswith(".")
            and not stripped.endswith(",")
        ):
            _flush_para()
            if current_section["paragraphs"]:
                sections.append(current_section)
                current_section = {"heading": stripped, "level": 3, "paragraphs": []}
            elif current_section["heading"] == title:
                current_section["heading"] = stripped
                current_section["level"] = 2
            else:
                current_section["heading"] = stripped
                current_section["level"] = 3
            continue

        current_para.append(stripped)

    _flush_para()
    if current_section["paragraphs"]:
        sections.append(current_section)

    # Build HTML
    safe_title = _html.escape(title.replace("-", " ").replace("_", " ").title())
    body_parts = []

    for section in sections:
        h_level = min(section["level"], 6)
        h_tag = f"h{h_level}"
        heading = _html.escape(section["heading"])
        section_id = _re.sub(r"[^a-z0-9]+", "-", section["heading"].lower()).strip("-")[:60]

        body_parts.append(
            f'<section id="{section_id}" aria-labelledby="{section_id}-heading">'
        )
        body_parts.append(f'  <{h_tag} id="{section_id}-heading">{heading}</{h_tag}>')

        for para in section["paragraphs"]:
            safe_para = _html.escape(para)
            body_parts.append(f"  <p>{safe_para}</p>")

        body_parts.append("</section>")

    body_html = "\n".join(body_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; max-width: 50em; margin: 0 auto; padding: 1em; color: #1a1a1a; }}
    .skip-link {{ position: absolute; left: -9999px; top: auto; width: 1px; height: 1px; overflow: hidden; }}
    .skip-link:focus {{ position: static; width: auto; height: auto; }}
    h1 {{ font-size: 2em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
    h2 {{ font-size: 1.5em; margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
    h3 {{ font-size: 1.25em; margin-top: 1.5em; }}
    section {{ margin-bottom: 1.5em; }}
    p {{ margin: 0.8em 0; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #1a1a1a; color: #e0e0e0; }}
      h1, h2 {{ border-color: #555; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
  </style>
</head>
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>
  <header role="banner">
    <h1>{safe_title}</h1>
  </header>
  <main id="main-content" role="main">
{body_html}
  </main>
  <footer role="contentinfo">
    <p>Converted by DART (Document Accessibility Remediation Tool)</p>
  </footer>
</body>
</html>"""


def _build_tool_registry() -> dict:
    """
    Build a tool registry mapping tool names to callable async functions.

    Imports and wraps all MCP tool functions so the TaskExecutor
    can invoke them by name.
    """
    registry = {}

    # DART tools
    async def _extract_and_convert_pdf(**kwargs):
        """Extract text from PDF and convert to clean, accessible HTML.

        Strategy:
        1. Try multi-source synthesis if combined JSON exists
        2. Extract text via pdftotext
        3. Build clean semantic HTML from the extracted text
           (strips page numbers, TOC artifacts, headers/footers)
        """
        from lib.paths import DART_PATH

        pdf_path = kwargs.get("pdf_path", "")
        course_code = kwargs.get("course_code")
        output_dir_str = kwargs.get("output_dir")

        pdf = Path(pdf_path)
        out_dir = Path(output_dir_str) if output_dir_str else DART_PATH / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        code = course_code or pdf.stem

        sys.path.insert(0, str(DART_PATH))

        # Strategy 1: If combined JSON exists, use multi-source synthesis
        combined_dir = DART_PATH / "batch_output" / "combined"
        combined_json = combined_dir / f"{code}_combined.json"

        if combined_json.exists():
            try:
                from multi_source_interpreter import convert_single_pdf
                html_output = out_dir / f"{code}_synthesized.html"
                convert_single_pdf(str(combined_json), str(html_output))
                return json.dumps({
                    "success": True,
                    "output_path": str(html_output),
                    "method": "multi_source_synthesis",
                })
            except ImportError:
                pass

        # Strategy 2: Extract text via pdftotext, then build accessible HTML
        import re as _re
        import subprocess

        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf), "-"],
                capture_output=True, text=True, timeout=120,
            )
            raw_text = result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fallback: try pdf_converter
            try:
                from pdf_converter.converter import PDFToAccessibleHTML
                converter = PDFToAccessibleHTML()
                conv_result = converter.convert(str(pdf), str(out_dir))
                return json.dumps({
                    "success": conv_result.success,
                    "output_path": conv_result.html_path,
                    "method": "pdf_converter",
                })
            except Exception as e2:
                return json.dumps({"error": f"DART conversion failed: {e2}"})

        if len(raw_text.strip()) < 100:
            return json.dumps({"error": "No meaningful text extracted from PDF"})

        # Build accessible HTML from raw extracted text
        html_output = out_dir / f"{code}_accessible.html"
        html_content = _raw_text_to_accessible_html(raw_text, code)
        html_output.write_text(html_content, encoding="utf-8")

        word_count = len(_re.findall(r"\b\w+\b", html_content))

        return json.dumps({
            "success": True,
            "output_path": str(html_output),
            "method": "pdftotext_to_html",
            "word_count": word_count,
            "html_length": len(html_content),
        })

    registry["extract_and_convert_pdf"] = _extract_and_convert_pdf

    # Pipeline tools - stage_dart_outputs
    async def _stage_dart_outputs(**kwargs):
        """Wrapper for stage_dart_outputs."""
        run_id = kwargs.get("run_id", "")
        dart_html_paths = kwargs.get("dart_html_paths", "")
        course_name = kwargs.get("course_name", "")

        staging_dir = COURSEFORGE_INPUTS / run_id
        staging_dir.mkdir(parents=True, exist_ok=True)

        staged_files = []
        errors = []
        html_paths = [Path(p.strip()) for p in dart_html_paths.split(",")]

        for html_path in html_paths:
            if not html_path.exists():
                errors.append(f"DART output not found: {html_path}")
                continue
            dest = staging_dir / html_path.name
            shutil.copy2(html_path, dest)
            staged_files.append(str(dest))

            # Copy JSON metadata if exists
            json_path = html_path.with_suffix(".json")
            if json_path.exists():
                shutil.copy2(json_path, staging_dir / json_path.name)
                staged_files.append(str(staging_dir / json_path.name))

        manifest = {
            "run_id": run_id,
            "course_name": course_name,
            "staged_at": datetime.now().isoformat(),
            "staged_files": staged_files,
        }
        manifest_path = staging_dir / "staging_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return json.dumps({
            "success": True,
            "staging_dir": str(staging_dir),
            "staged_files": staged_files,
            "file_count": len(staged_files),
            "manifest_path": str(manifest_path),
        })

    registry["stage_dart_outputs"] = _stage_dart_outputs

    # Courseforge tools
    try:
        from MCP.tools.courseforge_tools import register_courseforge_tools as _cf  # noqa: F401
        # Import the tool functions from courseforge_tools module scope
        # These are registered as MCP tools but we need direct callables

        async def _create_course_project(**kwargs):
            logger.info(f"_create_course_project called with kwargs: {list(kwargs.keys())}")
            logger.info(f"  objectives_path raw: {repr(kwargs.get('objectives_path'))}")
            course_name = kwargs.get("course_name", "")
            objectives_path = kwargs.get("objectives_path") or ""
            duration_weeks = kwargs.get("duration_weeks", 12)
            credit_hours = kwargs.get("credit_hours", 3)

            # Use the project creation logic directly
            project_id = f"PROJ-{course_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            project_path = _PROJECT_ROOT / "Courseforge" / "exports" / project_id

            project_path.mkdir(parents=True, exist_ok=True)
            for subdir in ["00_template_analysis", "01_learning_objectives",
                           "02_course_planning", "03_content_development",
                           "04_quality_validation", "05_final_package",
                           "agent_workspaces"]:
                (project_path / subdir).mkdir(exist_ok=True)

            config_path = project_path / "project_config.json"

            # If config already exists (from a prior phase), update rather than overwrite
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)
                # Only update fields that have real values
                if course_name:
                    config_data["course_name"] = course_name
                if objectives_path:
                    config_data["objectives_path"] = str(objectives_path)
                if duration_weeks:
                    config_data["duration_weeks"] = duration_weeks
            else:
                config_data = {
                    "project_id": project_id,
                    "course_name": course_name,
                    "objectives_path": str(objectives_path) if objectives_path else None,
                    "duration_weeks": duration_weeks,
                    "credit_hours": credit_hours,
                    "created_at": datetime.now().isoformat(),
                    "status": "initialized",
                }

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            # Generate default objective IDs from course name and weeks
            duration = duration_weeks if isinstance(duration_weeks, int) else 12
            objective_ids = [
                f"{course_name}_OBJ_{i}" for i in range(1, duration + 1)
            ]

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "project_path": str(project_path),
                "objective_ids": ",".join(objective_ids),
                "config": config_data,
            })

        registry["create_course_project"] = _create_course_project

        async def _generate_course_content(**kwargs):
            """Generate real course content modules from DART outputs + objectives.

            Reads staged DART HTML content and objectives, then produces
            one HTML module per week with structured educational content.
            """
            import html as _html
            import re as _re

            project_id = kwargs.get("project_id", "")
            project_path = _PROJECT_ROOT / "Courseforge" / "exports" / project_id
            content_dir = project_path / "03_content_development"
            content_dir.mkdir(parents=True, exist_ok=True)

            # Load project config to get objectives and duration
            config_path = project_path / "project_config.json"
            if not config_path.exists():
                return json.dumps({"error": f"Project config not found: {config_path}"})

            with open(config_path) as f:
                config = json.load(f)

            duration_weeks = config.get("duration_weeks", 12)
            objectives_path = config.get("objectives_path")

            # Load objectives
            objectives_data = {}
            if objectives_path and Path(objectives_path).exists():
                with open(objectives_path) as f:
                    objectives_data = json.load(f)

            chapter_objectives = objectives_data.get("chapter_objectives", [])
            terminal_objectives = objectives_data.get("terminal_objectives", [])

            # Collect staged DART content from staging directory
            source_content = ""
            staging_dir = COURSEFORGE_INPUTS
            for staging_run in sorted(staging_dir.iterdir()):
                if not staging_run.is_dir():
                    continue
                for src_file in staging_run.iterdir():
                    if src_file.suffix in (".html", ".htm", ".txt"):
                        try:
                            source_content += src_file.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                        except OSError:
                            pass

            # Parse source HTML into sections for topic-based selection
            # Split on </section> or <h2 or <h3 boundaries
            section_blocks = _re.split(r"(?=<section |<h[23])", source_content)
            source_sections = []
            for block in section_blocks:
                text = _re.sub(r"<[^>]+>", " ", block)
                text = _re.sub(r"\s+", " ", text).strip()
                if len(text) > 50:
                    source_sections.append(text)

            def _find_relevant_sections(objectives, all_sections, max_words=4000):
                """Find sections matching objective keywords."""
                # Extract keywords from objectives
                keywords = set()
                for obj in objectives:
                    stmt = obj.get("statement", "").lower()
                    # Extract significant words (skip common ones)
                    for word in _re.findall(r"\b[a-z]{4,}\b", stmt):
                        if word not in {"that", "this", "with", "from", "have", "will",
                                       "should", "able", "their", "which", "these",
                                       "more", "between", "both", "each", "such",
                                       "including", "based", "using", "through"}:
                            keywords.add(word)

                # Score each section by keyword overlap
                scored = []
                for section in all_sections:
                    section_lower = section.lower()
                    score = sum(1 for kw in keywords if kw in section_lower)
                    if score > 0:
                        scored.append((score, section))

                scored.sort(key=lambda x: -x[0])

                # Collect top sections up to word limit
                result = []
                total_words = 0
                for _, section in scored:
                    words_in_section = len(section.split())
                    if total_words + words_in_section > max_words:
                        if result:  # Already have some content
                            break
                    result.append(section)
                    total_words += words_in_section

                return result

            generated_files = []
            # Map weeks to chapter_objectives (6 entries cover 12 weeks in pairs)
            for week_num in range(1, duration_weeks + 1):
                week_dir = content_dir / f"week_{week_num:02d}"
                week_dir.mkdir(parents=True, exist_ok=True)

                # Map week number to objectives index (weeks come in pairs from 6 chapter groups)
                obj_idx = (week_num - 1) // 2
                week_objectives = []
                if obj_idx < len(chapter_objectives):
                    ch = chapter_objectives[obj_idx]
                    week_objectives = ch.get("objectives", [])
                    base_title = ch.get("chapter", f"Week {week_num}")
                    # Add "(Part 1)" or "(Part 2)" for paired weeks
                    part = "Part 1" if week_num % 2 == 1 else "Part 2"
                    week_title = f"{base_title} ({part})"
                else:
                    week_title = f"Week {week_num}: Course Integration"
                    # Use terminal objectives for overflow weeks
                    week_objectives = [
                        {"statement": to.get("statement", ""), "bloomLevel": to.get("bloomLevel", "")}
                        for to in terminal_objectives[-(duration_weeks - week_num + 1):][:3]
                    ]

                # Find topic-relevant source content
                relevant_sections = _find_relevant_sections(
                    week_objectives, source_sections, max_words=3000
                )

                # Build paragraphs from relevant sections
                paragraphs = []
                for section_text in relevant_sections:
                    # Split section into natural paragraphs (~150 words)
                    section_words = section_text.split()
                    for i in range(0, len(section_words), 150):
                        para = " ".join(section_words[i:i + 150])
                        if para.strip() and len(para) > 30:
                            paragraphs.append(_html.escape(para))

                # Build objectives HTML
                obj_html = ""
                if week_objectives:
                    obj_items = "\n".join(
                        f'      <li>{_html.escape(o.get("statement", ""))}'
                        f' <em>({o.get("bloomLevel", "")})</em></li>'
                        for o in week_objectives
                    )
                    obj_html = f"""
    <section id="objectives" aria-labelledby="objectives-heading">
      <h2 id="objectives-heading">Learning Objectives</h2>
      <p>By the end of this module, you should be able to:</p>
      <ul>
{obj_items}
      </ul>
    </section>"""

                # Build content sections from paragraphs
                content_sections = []
                section_size = max(1, len(paragraphs) // 3)
                section_titles = ["Key Concepts", "Discussion & Analysis", "Application"]

                for s_idx, s_title in enumerate(section_titles):
                    s_paras = paragraphs[s_idx * section_size:(s_idx + 1) * section_size]
                    if not s_paras:
                        continue
                    s_id = _re.sub(r"[^a-z0-9]+", "-", s_title.lower())
                    para_html = "\n".join(f"      <p>{p}</p>" for p in s_paras)
                    content_sections.append(f"""
    <section id="{s_id}" aria-labelledby="{s_id}-heading">
      <h2 id="{s_id}-heading">{_html.escape(s_title)}</h2>
{para_html}
    </section>""")

                sections_html = "\n".join(content_sections)

                # Build reflection/activity section
                activity_html = """
    <section id="activities" aria-labelledby="activities-heading">
      <h2 id="activities-heading">Reflection &amp; Activities</h2>
      <p>Consider the following questions as you review this week's material:</p>
      <ol>
        <li>How do the concepts presented this week connect to your own teaching or learning experience?</li>
        <li>Which ideas challenge your current understanding of instructional design?</li>
        <li>How might you apply these principles in designing a digital learning experience?</li>
      </ol>
    </section>"""

                safe_title = _html.escape(week_title)
                module_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DIGPED 101 - {safe_title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; max-width: 50em; margin: 0 auto; padding: 1em; color: #1a1a1a; }}
    .skip-link {{ position: absolute; left: -9999px; }} .skip-link:focus {{ position: static; }}
    h1 {{ font-size: 1.8em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
    h2 {{ font-size: 1.4em; margin-top: 1.5em; color: #2c5282; }}
    ul, ol {{ margin: 0.8em 0; padding-left: 1.5em; }} li {{ margin: 0.4em 0; }}
    section {{ margin-bottom: 1.5em; }}
    @media (prefers-color-scheme: dark) {{ body {{ background: #1a1a1a; color: #e0e0e0; }} h2 {{ color: #90cdf4; }} }}
  </style>
</head>
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>
  <main id="main-content" role="main">
    <h1>{safe_title}</h1>
{obj_html}
{sections_html}
{activity_html}
  </main>
</body>
</html>"""

                module_path = week_dir / "module.html"
                module_path.write_text(module_html, encoding="utf-8")
                generated_files.append(str(module_path))

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "weeks_prepared": duration_weeks,
                "content_paths": generated_files,
                "source_sections": len(source_sections),
                "content_selection": "topic-aligned",
            })

        registry["generate_course_content"] = _generate_course_content

        async def _package_imscc(**kwargs):
            """Build a real IMS Common Cartridge package from generated content.

            Creates a valid IMSCC ZIP with imsmanifest.xml and all HTML modules.
            Parseable by Trainforge's IMSCCParser.
            """
            import zipfile

            project_id = kwargs.get("project_id", "")
            project_path = _PROJECT_ROOT / "Courseforge" / "exports" / project_id
            content_dir = project_path / "03_content_development"
            final_dir = project_path / "05_final_package"
            final_dir.mkdir(parents=True, exist_ok=True)

            config_path = project_path / "project_config.json"
            course_name = project_id
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                    course_name = cfg.get("course_name", project_id)

            # Collect HTML module files
            html_files = sorted(content_dir.rglob("*.html"))
            if not html_files:
                return json.dumps({
                    "error": "No HTML modules found in content directory",
                    "content_dir": str(content_dir),
                })

            # Build imsmanifest.xml
            resource_items = []
            resource_defs = []
            for idx, html_file in enumerate(html_files, 1):
                rel_path = html_file.relative_to(content_dir)
                res_id = f"RES_{idx:03d}"
                item_id = f"ITEM_{idx:03d}"
                title_text = html_file.parent.name.replace("_", " ").title()

                resource_items.append(
                    f'      <item identifier="{item_id}" identifierref="{res_id}">'
                    f'\n        <title>{title_text}</title>'
                    f'\n      </item>'
                )
                resource_defs.append(
                    f'    <resource identifier="{res_id}" type="webcontent" '
                    f'href="{rel_path}">'
                    f'\n      <file href="{rel_path}"/>'
                    f'\n    </resource>'
                )

            items_xml = "\n".join(resource_items)
            resources_xml = "\n".join(resource_defs)

            manifest_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="{course_name}_manifest"
  xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
  xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/resource"
  xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.2.0</schemaversion>
    <lomimscc:lom>
      <lomimscc:general>
        <lomimscc:title>
          <lomimscc:string language="en">{course_name}</lomimscc:string>
        </lomimscc:title>
      </lomimscc:general>
    </lomimscc:lom>
  </metadata>
  <organizations>
    <organization identifier="ORG_1" structure="rooted-hierarchy">
      <item identifier="ROOT">
        <title>{course_name}</title>
{items_xml}
      </item>
    </organization>
  </organizations>
  <resources>
{resources_xml}
  </resources>
</manifest>"""

            # Create IMSCC ZIP package
            package_path = final_dir / f"{course_name}.imscc"
            with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("imsmanifest.xml", manifest_xml)
                for html_file in html_files:
                    rel_path = html_file.relative_to(content_dir)
                    zf.write(html_file, str(rel_path))

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "package_path": str(package_path),
                "libv2_package_path": str(package_path),
                "html_modules": len(html_files),
                "package_size_bytes": package_path.stat().st_size,
            })

        registry["package_imscc"] = _package_imscc
    except ImportError:
        pass

    # Trainforge tools
    try:
        async def _analyze_imscc_content(**kwargs):
            imscc_path = kwargs.get("imscc_path", "")
            return json.dumps({
                "source": imscc_path,
                "analyzed_at": datetime.now().isoformat(),
                "has_manifest": True,
                "content": {"html_modules": 0, "existing_assessments": 0, "total_word_count": 0},
                "learning_objectives": [],
                "assessment_opportunities": [],
            })

        registry["analyze_imscc_content"] = _analyze_imscc_content

        async def _generate_assessments(**kwargs):
            """Generate real content-grounded assessments using AssessmentGenerator.

            Reads course HTML modules (from IMSCC or content dir), builds
            source chunks, and generates actual questions with the
            content-grounded generator.
            """
            import re as _re

            course_id = kwargs.get("course_id", "")
            question_count = int(kwargs.get("question_count", 10))
            bloom_levels_str = kwargs.get("bloom_levels", "remember,understand,apply")
            objective_ids_str = kwargs.get("objective_ids", "")
            imscc_path = kwargs.get("imscc_path", "")

            output_dir = TRAINING_CAPTURES / "trainforge" / course_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Parse bloom levels and objectives
            if isinstance(bloom_levels_str, list):
                bloom_levels = bloom_levels_str
            else:
                bloom_levels = [b.strip() for b in bloom_levels_str.split(",") if b.strip()]

            if isinstance(objective_ids_str, list):
                objective_ids = objective_ids_str
            else:
                objective_ids = [o.strip() for o in objective_ids_str.split(",") if o.strip()]

            if not objective_ids:
                objective_ids = [f"{course_id}_OBJ_{i}" for i in range(1, 13)]

            # Build source chunks from IMSCC or HTML content
            source_chunks = []
            chunk_id_counter = 0

            # Try to read HTML modules from IMSCC
            if imscc_path and Path(imscc_path).exists() and Path(imscc_path).stat().st_size > 0:
                import zipfile
                try:
                    with zipfile.ZipFile(imscc_path, "r") as zf:
                        for name in zf.namelist():
                            if name.endswith(".html") or name.endswith(".htm"):
                                html_content = zf.read(name).decode("utf-8", errors="ignore")
                                # Strip HTML tags for text content
                                text = _re.sub(r"<[^>]+>", " ", html_content)
                                text = _re.sub(r"\s+", " ", text).strip()
                                if len(text) > 50:
                                    chunk_id_counter += 1
                                    source_chunks.append({
                                        "id": f"chunk_{chunk_id_counter:04d}",
                                        "text": html_content,  # Keep HTML for ContentExtractor
                                        "chunk_type": "explanation",
                                        "concept_tags": [],
                                        "source": {"file": name},
                                    })
                except zipfile.BadZipFile:
                    logger.warning(f"Invalid IMSCC ZIP: {imscc_path}")

            # Fallback: read from Courseforge content directories
            if not source_chunks:
                exports_dir = _PROJECT_ROOT / "Courseforge" / "exports"
                for project_dir in sorted(exports_dir.iterdir()):
                    content_dir = project_dir / "03_content_development"
                    if not content_dir.exists():
                        continue
                    for html_file in sorted(content_dir.rglob("*.html")):
                        try:
                            html_content = html_file.read_text(encoding="utf-8", errors="ignore")
                            text = _re.sub(r"<[^>]+>", " ", html_content)
                            text = _re.sub(r"\s+", " ", text).strip()
                            if len(text) > 50:
                                chunk_id_counter += 1
                                source_chunks.append({
                                    "id": f"chunk_{chunk_id_counter:04d}",
                                    "text": html_content,
                                    "chunk_type": "explanation",
                                    "concept_tags": [],
                                    "source": {"file": str(html_file.name)},
                                })
                        except OSError:
                            continue

            if not source_chunks:
                return json.dumps({
                    "error": "No source content found for assessment generation",
                    "imscc_path": imscc_path,
                })

            # Use the real AssessmentGenerator
            from Trainforge.generators.assessment_generator import AssessmentGenerator

            generator = AssessmentGenerator(capture=None, check_leaks=True)
            assessment = generator.generate(
                course_code=course_id,
                objective_ids=objective_ids,
                bloom_levels=bloom_levels,
                question_count=question_count,
                source_chunks=source_chunks,
            )

            # Write full assessment data
            assessment_dict = assessment.to_dict()
            output_path = output_dir / f"{assessment.assessment_id}.json"
            with open(output_path, "w") as f:
                json.dump(assessment_dict, f, indent=2)

            # Count content-grounded vs fallback
            grounded = sum(
                1 for q in assessment.questions
                if q.generation_rationale and "TEMPLATE_FALLBACK" not in q.generation_rationale
            )

            return json.dumps({
                "success": True,
                "assessment_id": assessment.assessment_id,
                "question_count": len(assessment.questions),
                "output_path": str(output_path),
                "rag_enabled": True,
                "source_chunks_used": len(source_chunks),
                "content_grounded": grounded,
                "template_fallback": len(assessment.questions) - grounded,
            })

        registry["generate_assessments"] = _generate_assessments
    except Exception:
        pass

    # LibV2 archival tool
    async def _archive_to_libv2(**kwargs):
        """Wrapper for archive_to_libv2."""

        course_name = kwargs.get("course_name", "")
        domain = kwargs.get("domain", "")
        division = kwargs.get("division", "STEM")
        pdf_paths_str = kwargs.get("pdf_paths", "")
        html_paths_str = kwargs.get("html_paths", "")
        imscc_path_str = kwargs.get("imscc_path", "")
        subdomains_str = kwargs.get("subdomains", "")

        slug = course_name.lower().replace("_", "-").replace(" ", "-")
        libv2_root = _PROJECT_ROOT / "LibV2"
        course_dir = libv2_root / "courses" / slug

        for subdir in [
            "source/pdf", "source/html", "source/imscc",
            "corpus", "graph", "pedagogy", "training_specs", "quality"
        ]:
            (course_dir / subdir).mkdir(parents=True, exist_ok=True)

        archived = {"pdfs": [], "html": [], "imscc": None}

        if pdf_paths_str:
            for p in pdf_paths_str.split(","):
                src = Path(p.strip())
                if src.exists():
                    dest = course_dir / "source" / "pdf" / src.name
                    shutil.copy2(src, dest)
                    archived["pdfs"].append(str(dest))

        if html_paths_str:
            for p in html_paths_str.split(","):
                src = Path(p.strip())
                if src.exists():
                    dest = course_dir / "source" / "html" / src.name
                    shutil.copy2(src, dest)
                    archived["html"].append(str(dest))

        if imscc_path_str:
            src = Path(imscc_path_str)
            if src.exists():
                dest = course_dir / "source" / "imscc" / src.name
                shutil.copy2(src, dest)
                archived["imscc"] = str(dest)

        manifest = {
            "libv2_version": "1.2.0",
            "slug": slug,
            "import_timestamp": datetime.now().isoformat(),
            "classification": {
                "division": division,
                "primary_domain": domain,
                "subdomains": [s.strip() for s in subdomains_str.split(",")]
                if subdomains_str else [],
            },
            "provenance": {
                "source_type": "textbook_to_course_pipeline",
                "import_pipeline_version": "1.0.0",
            },
        }

        manifest_path = course_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return json.dumps({
            "success": True,
            "course_slug": slug,
            "course_dir": str(course_dir),
            "manifest_path": str(manifest_path),
            "archived": archived,
        })

    registry["archive_to_libv2"] = _archive_to_libv2

    return registry


def _get_phase_status(workflow: dict, phase_name: str) -> dict:
    """Extract status for a specific phase from workflow tasks."""
    tasks = workflow.get("tasks", [])

    # Map phase names to agent types
    phase_agents = {
        "dart_conversion": ["dart-converter"],
        "staging": ["textbook-stager"],
        "objective_extraction": ["textbook-ingestor"],
        "course_planning": ["course-outliner"],
        "content_generation": ["content-generator"],
        "packaging": ["brightspace-packager"],
        "trainforge_assessment": ["assessment-generator"],
        "libv2_archival": ["libv2-archivist"],
        "finalization": ["brightspace-packager"]
    }

    agents = phase_agents.get(phase_name, [])

    phase_tasks = [t for t in tasks if t.get("agent_type") in agents]

    if not phase_tasks:
        return {"status": "PENDING", "tasks": 0}

    statuses = [t.get("status") for t in phase_tasks]

    if all(s == "COMPLETE" for s in statuses):
        phase_status = "COMPLETE"
    elif any(s == "ERROR" for s in statuses):
        phase_status = "ERROR"
    elif any(s == "IN_PROGRESS" for s in statuses):
        phase_status = "IN_PROGRESS"
    else:
        phase_status = "PENDING"

    return {
        "status": phase_status,
        "tasks": len(phase_tasks),
        "completed": sum(1 for s in statuses if s == "COMPLETE"),
        "errors": sum(1 for s in statuses if s == "ERROR")
    }
