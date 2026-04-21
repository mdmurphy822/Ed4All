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


def _raw_text_to_accessible_html(
    raw_text: str,
    title: str,
    metadata: Optional[dict] = None,
    *,
    source_pdf: Optional[str] = None,
    output_path: Optional[str] = None,
    figures_dir: Optional[str] = None,
    llm: Optional[object] = None,
) -> str:
    """Wave 15+16+17 entry point: route raw pdftotext / PDF to DART.converter.

    Flags:

    * ``DART_LEGACY_CONVERTER=true`` forces the pre-Wave-15 regex path
      (``_raw_text_to_accessible_html_legacy``) as a one-release safety
      fallback. Default — and the path exercised by end-to-end tests —
      is the Wave 12-15 ontology-aware pipeline
      (``DART.converter.convert_pdftotext_to_html``), which produces
      ``data-dart-block-role`` + Dublin Core + schema.org JSON-LD.
    * ``DART_LLM_CLASSIFICATION`` is respected transitively through
      ``DART.converter.default_classifier`` — when on AND a backend is
      provided, block classification goes through Claude.

    Wave 16: when ``source_pdf`` is provided, the converter reaches the
    full :func:`DART.converter.extractor.extract_document` path so
    pdfplumber tables, PyMuPDF figures, and Tesseract OCR text all
    survive into the HTML output. When ``source_pdf`` is ``None`` (the
    legacy raw-text-only call shape), behaviour is unchanged from Wave
    15 — the converter runs on ``raw_text`` alone.

    Wave 17: when ``output_path`` is provided and ``figures_dir`` is
    not overridden, the converter auto-derives a sibling figures
    directory (``<output_stem>_figures``) next to the output HTML, so
    persisted figure images stay relative to the HTML file and the
    bundle is portable. Explicit ``figures_dir=...`` overrides the
    sibling derivation.

    ``metadata`` carries Dublin Core fields (authors, date, language,
    rights, subject) that the new assembler emits as ``<meta>`` tags in
    ``<head>``. Legacy mode ignores this argument to preserve byte-for-
    byte parity with the pre-Wave-15 output.
    """
    import os as _os

    legacy_flag = _os.environ.get("DART_LEGACY_CONVERTER", "").strip().lower()
    if legacy_flag == "true":
        return _raw_text_to_accessible_html_legacy(raw_text, title)

    # Wave 16 enriched path: when a source PDF is available, go through
    # the dual-extraction layer so tables / figures / OCR contribute
    # structured blocks. Wrap extractor failures in a fall-through so a
    # broken optional extractor never blocks the raw-text conversion.
    if source_pdf:
        try:
            from DART.converter.block_segmenter import (
                segment_extracted_document,
            )
            from DART.converter.document_assembler import assemble_html
            from DART.converter.extractor import extract_document
            from DART.converter import default_classifier

            # Wave 17: derive a sibling figures dir from ``output_path``
            # so persisted figure bytes travel with the HTML. Explicit
            # ``figures_dir`` wins. Unset + unset → tempdir fallback
            # (plumbed through anyway so ``data.image_path`` still
            # points somewhere; the pipeline won't see the files but
            # tests / ad-hoc runs keep the full round-trip).
            resolved_figures_dir: Optional[Path] = None
            rel_figures_prefix = ""
            if figures_dir:
                resolved_figures_dir = Path(figures_dir)
                # A caller-supplied figures_dir is treated as relative
                # to output_path when output_path exists, else as an
                # absolute/cwd-relative path.
                if output_path:
                    out_parent = Path(output_path).resolve().parent
                    try:
                        rel = resolved_figures_dir.resolve().relative_to(
                            out_parent
                        )
                        rel_figures_prefix = str(rel) + "/"
                    except ValueError:
                        rel_figures_prefix = str(resolved_figures_dir) + "/"
                else:
                    rel_figures_prefix = str(resolved_figures_dir) + "/"
            elif output_path:
                out_path = Path(output_path)
                sibling_name = f"{out_path.stem}_figures"
                resolved_figures_dir = out_path.parent / sibling_name
                rel_figures_prefix = sibling_name + "/"
            else:
                # Neither output_path nor figures_dir provided. Fall
                # back to a tempdir so figures still materialise on
                # disk for downstream consumers that know how to find
                # them; ``<img src>`` references become absolute paths
                # which isn't portable but is better than empty ``src``.
                import tempfile as _tempfile

                resolved_figures_dir = Path(
                    _tempfile.mkdtemp(prefix="dart_figures_")
                )
                rel_figures_prefix = str(resolved_figures_dir) + "/"
                logger.debug(
                    "No output_path or figures_dir; using tempdir %s for figures",
                    resolved_figures_dir,
                )

            doc = extract_document(
                source_pdf,
                llm=llm,
                figures_dir=resolved_figures_dir,
            )

            # Rewrite each figure's ``image_path`` to include the
            # sibling-dir prefix so downstream blocks carry a relative
            # path that resolves from the HTML output location.
            if rel_figures_prefix:
                for fig in doc.figures:
                    if fig.image_path and "/" not in fig.image_path:
                        fig.image_path = rel_figures_prefix + fig.image_path

            blocks = segment_extracted_document(doc)
            classifier = default_classifier(llm=llm)
            from DART.converter.heuristic_classifier import HeuristicClassifier

            if isinstance(classifier, HeuristicClassifier):
                classified = classifier.classify_sync(blocks)
            else:
                # Use the same loop-safe bridge as convert_pdftotext_to_html.
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import threading

                    result: list = []
                    error: list = []

                    def _runner():
                        try:
                            result.append(asyncio.run(classifier.classify(blocks)))
                        except BaseException as exc:  # noqa: BLE001
                            error.append(exc)

                    thread = threading.Thread(target=_runner, daemon=True)
                    thread.start()
                    thread.join()
                    if error:
                        raise error[0]
                    classified = result[0]
                except RuntimeError:
                    classified = asyncio.run(classifier.classify(blocks))
            return assemble_html(classified, title, metadata or {})
        except RuntimeError as exc:
            logger.debug(
                "Wave 16 extractor failed (%s); falling back to raw-text path",
                exc,
            )
        except Exception as exc:  # noqa: BLE001 — never block on optional path
            logger.debug(
                "Wave 16 extractor raised unexpectedly (%s); falling back",
                exc,
            )

    # Wave 15 path (raw text only): delegate to the 4-phase pipeline.
    from DART.converter import convert_pdftotext_to_html as _convert

    return _convert(raw_text, title=title, metadata=metadata or {}, llm=llm)


def _raw_text_to_accessible_html_legacy(raw_text: str, title: str) -> str:
    """Legacy Wave 11 regex-driven converter (gated by DART_LEGACY_CONVERTER).

    Retained for one release cycle as a safety fallback so ops can
    revert to the pre-Wave-15 output shape with a single env flag. Do
    not extend — new behaviour lands in :mod:`DART.converter`.

    Processing pipeline (multi-pass, each isolates one concern):

    0. **Soft-hyphen rejoin** — fold pdftotext word-break artifacts
       (``...adapting pre-trained language mo-\\nrelated`` → one line).
    1. **Running-header detection** — lines that appear ≥4 times
       across the document (page headers/footers like
       ``"x Teaching in a Digital Age xi"``) flagged for drop.
    2. **Front-matter + TOC strip** — copyright/license/ISBN blocks
       move to a metadata region; dot-leader TOC entries
       (``"1.1 Foo . . . . . . 2"``) and numbered TOC lines drop.
    3. **Metadata-paragraph extraction** — the author/affiliation/
       arXiv-ID first paragraphs of arxiv papers move to a
       ``<header role="banner">`` metadata block, not ``<main>``.
    4. **Column-layout drop** — lines with 2+ runs of ≥5 spaces
       (pdftotext column alignment) drop from the paragraph stream.
    5. **Structure detection + heading hierarchy** — textbook
       chapter openers (``Chapter N:``/``Section N:``) land as h2;
       arxiv paper sections (``I. INTRODUCTION``/``1. METHOD``/
       ``Abstract``/``Introduction``/etc.) also land as h2;
       Title-Case standalone lines land as h3.
    6. **Render** — WCAG 2.2 AA landmarks, deduped section IDs,
       page ``<h1>`` emitted exactly once.
    """
    import html as _html
    import re as _re

    lines = raw_text.split("\n")

    # ------------------------------------------------------------------
    # Pass 0 — soft-hyphen rejoin (Fix D)
    # ------------------------------------------------------------------
    # pdftotext breaks hyphenated words across line endings. When a line
    # ends with ``word-`` and the next non-empty line starts lowercase,
    # rejoin them into a single line without the soft-hyphen.
    rejoined_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        stripped = line.rstrip()
        # Look for ``...word-`` at EOL with at least 2 letters before the
        # hyphen (avoids joining em-dashes / range hyphens).
        if _re.search(r"[A-Za-z]{2,}-$", stripped) and i + 1 < len(lines):
            nxt = lines[i + 1].lstrip()
            if nxt and nxt[0].islower():
                # Strip trailing hyphen + join with next line (without
                # leading space so the word reassembles cleanly).
                rejoined = stripped[:-1] + nxt
                rejoined_lines.append(rejoined)
                skip_next = True
                continue
        rejoined_lines.append(line)
    lines = rejoined_lines

    # ------------------------------------------------------------------
    # Pass 1 — running-header / footer detection
    # ------------------------------------------------------------------
    # Lines that appear ≥4 times in the document with short content
    # (≤ 60 chars) are almost always page headers/footers pulled by
    # pdftotext as standalone content. Build a frequency map and flag
    # high-count short lines as droppable.
    from collections import Counter as _Counter
    line_counts = _Counter(
        line.strip() for line in lines
        if line.strip() and len(line.strip()) <= 60
    )
    repeated_chrome = {
        text for text, count in line_counts.items()
        if count >= 4 and text  # guard against empty
    }

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
    # TOC entry shapes we want to drop from the body:
    #  - dot-leader form: ``"1.1 Foo . . . . . .     2"``
    #  - numbered-plus-page form: ``"Preface                              vii"``
    #  - simple ``"title  ....  page"``
    toc_entry = _re.compile(r"^.{5,80}\s{3,}\d{1,4}\s*$")
    toc_dot_leader = _re.compile(r"^.{5,120}(?:\s*\.\s*){3,}\d{1,4}\s*$")
    toc_roman_page = _re.compile(
        r"^[A-Za-z][A-Za-z\s\-]{3,80}\s{3,}[ivxl]{1,6}\s*$",
        _re.IGNORECASE,
    )
    # Column-layout paragraph: 2+ runs of 5+ consecutive spaces
    # anywhere in the line. Using findall for counting instead of a
    # single regex match avoids the overlapping-anchor pitfall where
    # ``"A     S     B"`` has two 5-space runs but can't be matched
    # by a single ``\\S\\s{5,}\\S.*\\S\\s{5,}\\S`` pattern.
    _col_run_re = _re.compile(r"\s{5,}")
    def _has_column_layout(text: str) -> bool:
        return len(_col_run_re.findall(text)) >= 2
    # Front-matter / copyright token heuristics (Fix B).
    front_matter_hint = _re.compile(
        r"(?:copyright|©|\bisbn\b|licensed under|creative commons|"
        r"all rights reserved|cover design by|typeset in|"
        r"this work is licensed|cover photo)",
        _re.IGNORECASE,
    )
    # Metadata-paragraph hints for arxiv papers (Fix E).
    arxiv_meta_hint = _re.compile(r"\barxiv[:.][\d.]+v?\d*\b", _re.IGNORECASE)
    email_hint = _re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    # Chapter/section opener — requires an explicit structural prefix.
    # Dropped the bare `\d{1,2}\.` branch because it was false-positive on
    # numbered list items and date-prefixed paragraphs ("19 April 2015: …").
    chapter_heading = _re.compile(
        r"^(?:"
        r"(?:Chapter|Part|Section|Unit|Block|Appendix)\s+\d+[.:]\s*|"
        r"(?:I{1,3}V?|VI{0,3}|IX|X{1,3})\.\s+"
        r")(.+)",
    )
    # Arxiv-paper section opener (Fix C): ALL-CAPS title after roman/
    # arabic numeral + dot, OR a canonical paper-section keyword on
    # its own line. Distinct from ``chapter_heading`` because the ALL-
    # CAPS requirement gives high precision on paper layouts.
    paper_section_numbered = _re.compile(
        r"^(?:(?:I{1,3}V?|VI{0,3}|IX|X{1,3})\.|\d{1,2}\.)\s+"
        r"([A-Z][A-Z\s&/\-]{2,60})$"
    )
    _PAPER_SECTION_KEYWORDS = frozenset([
        "abstract", "introduction", "related work", "background",
        "methodology", "methods", "method", "approach", "model",
        "experiments", "experimental setup", "evaluation",
        "results", "discussion", "analysis", "conclusion",
        "conclusions", "future work", "limitations",
        "references", "acknowledgements", "acknowledgments",
        "appendix", "appendices",
    ])
    # Sub-heading regex unchanged shape, but tightened post-match via
    # _is_valid_subheading below — the regex alone was letting through
    # author lists, citations, table-row fragments, and publisher chrome.
    sub_heading = _re.compile(r"^[A-Z][A-Za-z\s,&:'\-]{5,80}$")

    # Domain-neutral reject tokens for sub-headings that slipped through.
    _HEADING_REJECT_TOKENS = frozenset([
        "references", "bibliography", "index", "glossary", "appendix",
        "acknowledgements", "acknowledgments", "copyright", "isbn",
        "vancouver bc", "table of contents", "cover design",
        "this textbook", "typeset in", "overheard in",
        "for my comments", "updates and revisions", "for a working",
        "about the author", "about the authors",
    ])

    def _is_valid_subheading(text: str) -> bool:
        """Post-regex filter for sub-headings.

        Catches the residue that ``sub_heading`` regex lets through:
        citations, author bylines, table-cell fragments, publisher chrome,
        sentence fragments ending mid-word.
        """
        if not text:
            return False
        # Multi-space column layout (pdftotext emits aligned table rows
        # as "Kleur        Flower" / "Animal   Limb"). Real headings use
        # single spaces between words.
        if _re.search(r"\s{3,}", text):
            return False
        words = text.split()
        word_count = len(words)

        # Trailing hyphen = pdftotext soft-hyphen line break, mid-word.
        if text.endswith("-"):
            return False
        # Starts with digit + closing punctuation ("4). Being on contract…")
        # — sentence body that happened to begin with a numbered list opener.
        if _re.match(r"^\d+[.)\]]", text):
            return False
        # Trailing function word = truncated sentence. Also check the
        # penultimate word — pdftotext often wraps mid-phrase as
        # ``"… to specific"`` (the "to" is the signal, not the final
        # adjective).
        function_words = {
            "and", "or", "but", "of", "to", "for", "on", "at", "by",
            "in", "with", "as", "from", "into", "onto", "upon", "about",
            "against", "between", "through", "over", "under", "after",
            "before", "during",
            "the", "a", "an",
            "is", "are", "was", "were", "be", "been", "being",
            "that", "this", "these", "those",
            "my", "your", "his", "her", "its", "our", "their",
        }
        last = words[-1].lower().rstrip(",.:;")
        if last in function_words:
            return False
        if word_count >= 2:
            penult = words[-2].lower().rstrip(",.:;")
            if penult in function_words:
                return False
        # Too many periods = multi-sentence or citation
        # ("Tim Berners-Lee, James Hendler and Ora Lassila. The Semantic Web.")
        if text.count(".") >= 2:
            return False
        # Blocklist tokens
        normalized = " ".join(text.lower().split())
        for tok in _HEADING_REJECT_TOKENS:
            if tok in normalized:
                return False
        # Table-row fragment: >30% duplicate words
        # (pdftotext repeats column labels in some layouts).
        if word_count >= 3:
            lowered = [w.lower() for w in words]
            if len(set(lowered)) < word_count * 0.7:
                return False
        # Author byline: 2+ capitalized two-word tokens separated by ","
        # or "and" — "Tim Berners-Lee, James Hendler and Ora Lassila"
        if _re.search(r"\b(and|,)\s+[A-Z][a-z]+\s+[A-Z]", text):
            return False
        # Ends with ':' and contains a lowercase instruction verb —
        # inline instruction, not a title ("To calculate the exact
        # omission rate:"). (Fix G)
        if text.rstrip().endswith(":"):
            lowered_body = text.lower()
            if any(tok in lowered_body for tok in (
                " calculate", " explain", " consider", " note",
                " observe", " see ", " suppose",
            )):
                return False
        return True

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
    # Emits (cleaned_body_lines, metadata_lines): body flows into the
    # structured section-build pass; metadata flows into the
    # <header role="banner"> block above <main>.
    cleaned_lines = []
    metadata_lines: list[str] = []
    in_toc = False
    in_bio = False
    in_front_matter = False
    front_matter_seen_content = False
    bio_line_count = 0
    prev_was_empty = True
    # Front-matter capture only fires early in the document OR before we
    # hit the first real heading. Tracking both prevents ``Creative
    # Commons`` / ``©`` mentions deep in the body text from sucking up
    # legitimate content into metadata (Fix B refinement).
    total_nonempty = sum(1 for ln in lines if ln.strip())
    _front_matter_budget = max(50, total_nonempty // 10)  # first ~10% of doc
    _front_matter_closed = False
    _nonempty_seen = 0

    for line in lines:
        stripped = line.strip()
        if stripped:
            _nonempty_seen += 1
            if _nonempty_seen > _front_matter_budget:
                _front_matter_closed = True

        # Empty line
        if not stripped:
            if in_bio:
                bio_line_count += 1
                if bio_line_count > 2:
                    in_bio = False  # Bios end after a gap
            # A blank line after front-matter indicates the block is
            # probably over — but only exit if we've seen at least one
            # content line inside.
            if in_front_matter and front_matter_seen_content:
                in_front_matter = False
                front_matter_seen_content = False
            cleaned_lines.append("")
            prev_was_empty = True
            continue

        # Skip standalone page numbers
        if page_num.match(stripped):
            continue

        # Drop running headers/footers that appear ≥4 times (Fix B).
        if stripped in repeated_chrome:
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

        # Drop column-layout lines (table rows, author affiliation
        # grids, etc.) — 2+ runs of ≥5 consecutive spaces (Fix H).
        if _has_column_layout(stripped):
            continue

        # Skip TOC entries — strict shapes: dot-leader, roman-page,
        # simple-page (Fix B extension).
        if (
            toc_entry.match(stripped)
            or toc_dot_leader.match(stripped)
            or toc_roman_page.match(stripped)
        ):
            in_toc = True
            continue
        if in_toc:
            # Stay in TOC mode until we hit a line that doesn't look
            # like a TOC entry AND is long enough to be body content.
            if len(stripped) > 60 and not (
                toc_entry.match(stripped)
                or toc_dot_leader.match(stripped)
                or toc_roman_page.match(stripped)
            ):
                in_toc = False
            else:
                continue

        # Front-matter / copyright block capture (Fix B). Only fires in
        # the first ~10% of the document — later occurrences of
        # ``copyright`` / ``Creative Commons`` / ``©`` are legitimate
        # body-content mentions and should stay in <main>.
        if not _front_matter_closed and front_matter_hint.search(stripped):
            in_front_matter = True
            front_matter_seen_content = True
            metadata_lines.append(stripped)
            continue
        if in_front_matter and not _front_matter_closed:
            front_matter_seen_content = True
            metadata_lines.append(stripped)
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
    # First section is a title-seeded wrapper; we use the sentinel level 0
    # so the renderer knows to emit its paragraphs WITHOUT its own heading
    # (Fix A — prevents a duplicate <h1> inside <main>; the page <h1>
    # appears once in <header role="banner">).
    current_section = {"heading": title, "level": 0, "paragraphs": []}
    current_para = []

    def _flush_para():
        text = " ".join(current_para).strip()
        if not text or len(text) <= 20:
            current_para.clear()
            return
        # Drop assembled paragraphs that still look like column-layout
        # table rows (Fix H refinement — a single line may survive
        # per-line filtering, but when joined with another col-layout
        # line the paragraph exposes the pattern).
        if _has_column_layout(text):
            current_para.clear()
            return
        # Fix E — metadata-paragraph extraction. First paragraph of an
        # arxiv paper typically contains authors + affiliations + email
        # + arXiv ID. If the initial (title-seeded) section has no real
        # paragraphs yet and this paragraph matches metadata hints,
        # route it to metadata_lines rather than the body.
        if (
            current_section["level"] == 0
            and not current_section["paragraphs"]
            and (
                arxiv_meta_hint.search(text)
                or email_hint.search(text)
            )
        ):
            metadata_lines.append(text)
            current_para.clear()
            return
        current_section["paragraphs"].append(text)
        current_para.clear()

    def _is_paper_section_keyword(stripped: str) -> bool:
        """Standalone canonical paper-section keyword (Abstract,
        Introduction, …). Case-insensitive, but the whole line must be
        just the keyword (plus optional trailing punctuation)."""
        canon = stripped.lower().rstrip(":.")
        return canon in _PAPER_SECTION_KEYWORDS

    for stripped in cleaned_lines:
        if not stripped:
            _flush_para()
            continue

        # Fix C — arxiv-paper section detector. ALL-CAPS after a
        # numeral ("I. INTRODUCTION") OR a canonical keyword
        # ("Abstract", "Introduction"). Fires at h2 level; standalone
        # so it doesn't require an existing blank-line guard.
        paper_match = paper_section_numbered.match(stripped)
        if (paper_match or _is_paper_section_keyword(stripped)) and not current_para:
            if paper_match:
                heading_text = paper_match.group(1).strip().title()
            else:
                heading_text = stripped.rstrip(":.").strip().title()
            _flush_para()
            # Close out the current section (if any) and start a new h2.
            if current_section["paragraphs"] or current_section["level"] != 0:
                sections.append(current_section)
            elif current_section["level"] == 0 and current_section["paragraphs"]:
                sections.append(current_section)
            current_section = {"heading": heading_text, "level": 2, "paragraphs": []}
            continue

        # Detect chapter headings (structured: "Chapter 1: ...", "Section 2: ...",
        # "I. Definitions"). Require:
        #   (a) match came after a blank line (no open paragraph), otherwise
        #       we're inside a sentence ("Section 3.3, which essentially…") —
        #       not a real chapter opener;
        #   (b) the raw line is short enough to be a title (≤ 120 chars);
        #   (c) the captured text passes sub-heading validity (rejects
        #       truncated sentences, citations, table fragments, etc.).
        ch_match = chapter_heading.match(stripped)
        if (
            ch_match
            and not current_para
            and len(stripped) <= 120
        ):
            heading_text = ch_match.group(1).strip() if ch_match.group(1) else stripped
            if _is_valid_subheading(heading_text):
                _flush_para()
                if current_section["paragraphs"] or current_section["level"] != 0:
                    sections.append(current_section)
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
            and _is_valid_subheading(stripped)
        ):
            _flush_para()
            if current_section["paragraphs"]:
                sections.append(current_section)
                current_section = {"heading": stripped, "level": 3, "paragraphs": []}
            elif current_section["level"] == 0:
                current_section["heading"] = stripped
                current_section["level"] = 2
            else:
                current_section["heading"] = stripped
                current_section["level"] = 3
            continue

        current_para.append(stripped)

    _flush_para()
    if current_section["paragraphs"] or current_section["level"] != 0:
        sections.append(current_section)

    # Build HTML
    safe_title = _html.escape(title.replace("-", " ").replace("_", " ").title())
    body_parts = []
    seen_ids: set = set()  # Fix F — dedupe section IDs across the doc.

    def _unique_id(base: str) -> str:
        """Ensure every section id is unique — first occurrence keeps
        the bare slug, subsequent occurrences get ``-2``, ``-3`` …"""
        if base not in seen_ids:
            seen_ids.add(base)
            return base
        i = 2
        while f"{base}-{i}" in seen_ids:
            i += 1
        chosen = f"{base}-{i}"
        seen_ids.add(chosen)
        return chosen

    for section in sections:
        level = section["level"]
        # Fix A — level-0 (title-seeded) sections render their
        # paragraphs un-wrapped, with NO heading. The page <h1> appears
        # once in <header role="banner"> below. This prevents the
        # duplicate-<h1>-inside-<main> artifact that violates WCAG
        # heading nesting.
        if level == 0:
            for para in section["paragraphs"]:
                safe_para = _html.escape(para)
                body_parts.append(f"<p>{safe_para}</p>")
            continue

        h_level = min(level, 6)
        h_tag = f"h{h_level}"
        heading = _html.escape(section["heading"])
        base_id = _re.sub(r"[^a-z0-9]+", "-", section["heading"].lower()).strip("-")[:60] or "section"
        section_id = _unique_id(base_id)

        body_parts.append(
            f'<section id="{section_id}" aria-labelledby="{section_id}-heading">'
        )
        body_parts.append(f'  <{h_tag} id="{section_id}-heading">{heading}</{h_tag}>')

        for para in section["paragraphs"]:
            safe_para = _html.escape(para)
            body_parts.append(f"  <p>{safe_para}</p>")

        body_parts.append("</section>")

    body_html = "\n".join(body_parts)

    # Fix B / E — metadata block. Copyright/license/ISBN lines +
    # extracted arxiv-author paragraphs get their own <header> region
    # above the page <h1>. Rendered WITHOUT <h1> so the main page
    # heading stays authoritative.
    metadata_html = ""
    if metadata_lines:
        meta_items = []
        for entry in metadata_lines:
            safe = _html.escape(entry)
            meta_items.append(f"    <p>{safe}</p>")
        metadata_html = (
            '  <aside role="complementary" aria-label="Document metadata" '
            'class="document-metadata">\n'
            + "\n".join(meta_items)
            + "\n  </aside>"
        )

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
{metadata_html}
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
        # Output filename is keyed on the PDF basename so multi-PDF corpora
        # don't collide on a shared `course_code`. `code` is retained for
        # combined-JSON lookups + HTML title below.
        code = course_code or pdf.stem
        out_stem = pdf.stem

        sys.path.insert(0, str(DART_PATH))

        # Strategy 1: If combined JSON exists, use multi-source synthesis
        combined_dir = DART_PATH / "batch_output" / "combined"
        combined_json = combined_dir / f"{code}_combined.json"

        if combined_json.exists():
            try:
                from multi_source_interpreter import convert_single_pdf
                html_output = out_dir / f"{out_stem}_synthesized.html"
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

        # Build accessible HTML from raw extracted text.
        # Use the PDF stem (e.g. "keet_ontology_engineering") as the doc title
        # rather than the course_code — otherwise every PDF in a multi-PDF
        # corpus gets the same <h1>/<title> (the course code), which poisons
        # downstream objective extraction across the corpus.
        pretty_title = out_stem.replace("-", " ").replace("_", " ").strip()
        html_output = out_dir / f"{out_stem}_accessible.html"
        # Pass ``source_pdf`` so Wave 16 extraction enrichment kicks in
        # (pdfplumber tables + PyMuPDF figures + optional OCR). Pass
        # ``output_path`` so Wave 17 figure persistence auto-derives a
        # sibling ``{stem}_figures/`` directory; the caller override
        # (``kwargs["figures_dir"]``) still wins when set explicitly.
        # The extractor gracefully degrades when optional deps are
        # missing so this never regresses the raw-text-only path.
        html_content = _raw_text_to_accessible_html(
            raw_text,
            pretty_title,
            source_pdf=str(pdf),
            output_path=str(html_output),
            figures_dir=kwargs.get("figures_dir"),
        )
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
    # Registry variant now has full Wave 8 parity with the @mcp.tool() variant
    # (role-tagging, .quality.json copy, role-tagged manifest entries). The
    # MCP-tool variant at lines 316-451 remains the source of truth for the
    # copy/role logic; this wrapper just adapts kwargs into the Wave 8
    # staging pipeline.
    async def _stage_dart_outputs(**kwargs):
        """Stage DART outputs to Courseforge inputs with Wave 8 role-tagging.

        Copies HTML (role=content), *_synthesized.json provenance sidecars
        (role=provenance_sidecar), and *.quality.json confidence sidecars
        (role=quality_sidecar) to ``COURSEFORGE_INPUTS/{run_id}/`` and
        emits a role-tagged ``staging_manifest.json``. Kept byte-for-byte
        parity with the @mcp.tool() variant so pipeline-dispatch runs do
        not silently drop Wave 8 metadata (audit Q4 finding).
        """
        run_id = kwargs.get("run_id", "")
        dart_html_paths = kwargs.get("dart_html_paths", "")
        course_name = kwargs.get("course_name", "")

        try:
            staging_dir = COURSEFORGE_INPUTS / run_id
            staging_dir.mkdir(parents=True, exist_ok=True)

            staged_files: list = []
            staged_entries: list = []
            errors: list = []

            html_paths = [Path(p.strip()) for p in dart_html_paths.split(",") if p.strip()]

            for html_path in html_paths:
                if not html_path.exists():
                    errors.append(f"DART output not found: {html_path}")
                    continue

                # Copy HTML file (role=content)
                dest = staging_dir / html_path.name
                shutil.copy2(html_path, dest)
                staged_files.append(str(dest))
                staged_entries.append({"path": html_path.name, "role": "content"})

                # Copy accompanying JSON if it exists (DART synthesized metadata).
                json_path = html_path.with_suffix(".json")
                if json_path.exists():
                    json_dest = staging_dir / json_path.name
                    shutil.copy2(json_path, json_dest)
                    staged_files.append(str(json_dest))
                    staged_entries.append({
                        "path": json_path.name,
                        "role": "provenance_sidecar",
                    })

                # Also check for the _synthesized.json pattern.
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

                # Wave 8: also stage the DART quality sidecar if one exists.
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

            if errors and not staged_files:
                return json.dumps({
                    "success": False,
                    "error": "No files staged",
                    "errors": errors,
                })

            manifest = {
                "run_id": run_id,
                "course_name": course_name,
                "staged_at": datetime.now().isoformat(),
                "staged_files": staged_files,
                "files": staged_entries,
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
                "warnings": errors if errors else None,
            })
        except Exception as e:
            logger.error(f"Registry _stage_dart_outputs failed: {e}")
            return json.dumps({"error": str(e)})

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

        # ============================================================================
        # BLOCK: Worker α edits ONLY below this line through the next END marker.
        # Scope: _generate_course_content replacement. See plans/pipeline-execution-
        # fixes/contracts.md § "Courseforge content-generator contract".
        # ============================================================================
        async def _generate_course_content(**kwargs):
            """Generate 5-page weekly course modules from DART outputs + objectives.

            Replaces the legacy single-page stub with a full Courseforge
            emission: overview, content, application, self_check, and
            summary pages per week. Every emitted page carries the full
            ``data-cf-*`` attribute surface and a JSON-LD
            ``CourseModule`` body that validates against
            ``schemas/knowledge/courseforge_jsonld_v1.schema.json``.

            Delegates the actual HTML rendering to
            ``Courseforge.scripts.generate_course.generate_week`` (the
            mature multi-file emitter) — this wrapper only adapts the
            pipeline's kwargs into the ``week_data`` payload that the
            emitter consumes, plus forwards the Wave 9 source-routing
            map when one is present on disk.
            """
            from MCP.tools import _content_gen_helpers as _cgh
            from Courseforge.scripts import generate_course as _gen

            project_id = kwargs.get("project_id", "")
            if not project_id:
                return json.dumps({"error": "generate_course_content requires project_id"})

            project_path = _PROJECT_ROOT / "Courseforge" / "exports" / project_id
            content_dir = project_path / "03_content_development"
            content_dir.mkdir(parents=True, exist_ok=True)

            config_path = project_path / "project_config.json"
            if not config_path.exists():
                return json.dumps({"error": f"Project config not found: {config_path}"})
            with open(config_path) as f:
                config = json.load(f)

            course_code = config.get("course_name") or project_id
            duration_weeks = int(config.get("duration_weeks") or 12)
            objectives_path = config.get("objectives_path") or kwargs.get("objectives_path")

            # ---------------------------------------------------------- #
            # Staged DART HTML — prefer the staging_dir passed by the    #
            # workflow runner; fall back to the most-recent staging run. #
            # ---------------------------------------------------------- #
            staging_kwarg = kwargs.get("staging_dir")
            staging_dir = Path(staging_kwarg) if staging_kwarg else None
            html_files = _cgh.collect_staged_html(staging_dir, COURSEFORGE_INPUTS)
            topics = _cgh.parse_dart_html_files(html_files)

            # ---------------------------------------------------------- #
            # Objectives: honor supplied JSON; synthesize from DART otherwise.
            # ---------------------------------------------------------- #
            terminal_objectives, chapter_objectives = _cgh.load_objectives_json(
                objectives_path
            )
            if not terminal_objectives and not chapter_objectives:
                terminal_objectives, chapter_objectives = (
                    _cgh.synthesize_objectives_from_topics(topics, duration_weeks)
                )

            all_objectives = list(terminal_objectives) + list(chapter_objectives)
            topics_by_week = _cgh._group_topics_by_week(topics, duration_weeks)

            # ---------------------------------------------------------- #
            # Source-routing map (Wave 9). Empty dict or missing file =>  #
            # backward-compat path: pages emit without sourceReferences.  #
            # ---------------------------------------------------------- #
            source_module_map: Dict[str, Any] = {}
            map_path_kwarg = kwargs.get("source_module_map_path")
            if map_path_kwarg:
                map_path = Path(map_path_kwarg)
            else:
                map_path = project_path / "source_module_map.json"
            if map_path.exists():
                try:
                    source_module_map = json.loads(
                        map_path.read_text(encoding="utf-8")
                    ) or {}
                except (OSError, ValueError):
                    source_module_map = {}

            # Wave 2 prerequisite map: each page prerequisites the prior
            # page in the 5-page week sequence.
            prerequisite_map: Dict[str, list] = {}
            for week_num in range(1, duration_weeks + 1):
                w = f"{week_num:02d}"
                prerequisite_map[f"week_{w}_application"] = [
                    f"week_{w}_overview"
                ]
                prerequisite_map[f"week_{w}_self_check"] = [
                    f"week_{w}_application"
                ]
                prerequisite_map[f"week_{w}_summary"] = [
                    f"week_{w}_self_check"
                ]

            # ---------------------------------------------------------- #
            # Decision capture — content-generator phase.                 #
            # ---------------------------------------------------------- #
            capture = None
            try:
                from lib.decision_capture import DecisionCapture
                capture = DecisionCapture(
                    course_code=course_code,
                    phase="content-generator",
                    tool="courseforge",
                    streaming=True,
                )
                capture.log_decision(
                    decision_type="content_structure",
                    decision=(
                        f"Emit 5-page weekly modules (overview, content, "
                        f"application, self_check, summary) for "
                        f"{duration_weeks} weeks via Courseforge generate_week."
                    ),
                    rationale=(
                        "The 5-page structure matches the Courseforge "
                        "pipeline contract (plans/pipeline-execution-fixes/"
                        "contracts.md) and ensures each weekly module "
                        "validates under the page_objectives + "
                        "content_structure gates."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DecisionCapture init failed in content-generator: %s", exc
                )
                capture = None

            # ---------------------------------------------------------- #
            # Emit each week via generate_week.                           #
            # ---------------------------------------------------------- #
            generated_files: list = []
            weeks_prepared = 0
            for week_num in range(1, duration_weeks + 1):
                week_topics = (
                    topics_by_week[week_num - 1]
                    if (week_num - 1) < len(topics_by_week)
                    else []
                )
                # Per-week LO set: scope to this week's terminals + at most
                # two chapter objectives round-robin assigned by week.
                # Previously prepended ALL terminal_objectives to every week
                # (investigation Issue 12 — inflated derived-from-objective
                # edges from ~60 to 896 on OLSR_201). Now: each week gets
                # only the terminal slice round-robin assigned to it.
                week_chapter_cos = []
                if chapter_objectives:
                    step = max(1, len(chapter_objectives) // max(1, duration_weeks))
                    start = (week_num - 1) * step
                    week_chapter_cos = list(
                        chapter_objectives[start:start + step + 1]
                    )[:2] or [chapter_objectives[(week_num - 1) % len(chapter_objectives)]]

                # Scope terminals per week. With N terminals and D weeks,
                # each week claims ceil(N/D) terminals in source order.
                week_terminals: list = []
                if terminal_objectives:
                    t_step = max(
                        1,
                        (len(terminal_objectives) + duration_weeks - 1) // duration_weeks,
                    )
                    t_start = (week_num - 1) * t_step
                    week_terminals = list(
                        terminal_objectives[t_start:t_start + t_step]
                    )
                    # Guarantee at least one terminal per week when corpus
                    # has any terminals at all — round-robin fallback.
                    if not week_terminals:
                        week_terminals = [
                            terminal_objectives[(week_num - 1) % len(terminal_objectives)]
                        ]

                week_objectives = list(week_terminals) + week_chapter_cos
                seen: set = set()
                week_objectives_deduped = []
                for o in week_objectives:
                    if o["id"] in seen:
                        continue
                    seen.add(o["id"])
                    week_objectives_deduped.append(o)

                week_data = _cgh.build_week_data(
                    week_num=week_num,
                    duration_weeks=duration_weeks,
                    week_topics=week_topics,
                    week_objectives=week_objectives_deduped,
                    all_objectives=all_objectives,
                    course_code=course_code,
                )

                try:
                    count, files = _gen.generate_week(
                        week_data,
                        content_dir,
                        course_code,
                        canonical_objectives=None,  # week_data already has canonical ids
                        classification=None,
                        prerequisite_map=prerequisite_map,
                        source_module_map=source_module_map or None,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "generate_week failed for week %d: %s", week_num, exc
                    )
                    continue

                weeks_prepared += 1
                week_dir = content_dir / f"week_{week_num:02d}"
                for name in files:
                    page_path = week_dir / name
                    # Post-process: ensure every page carries an
                    # objectives <section>. Overview already has one
                    # from generate_week; the other four pages don't by
                    # default but the page_objectives gate + integration
                    # test require the data-cf-objective-id attribute on
                    # every page.
                    try:
                        body = page_path.read_text(encoding="utf-8")
                        updated = _cgh.ensure_objectives_on_page(
                            body, week_objectives_deduped,
                        )
                        if updated != body:
                            page_path.write_text(updated, encoding="utf-8")
                    except OSError as exc:
                        logger.warning(
                            "Failed to post-process %s: %s", page_path, exc,
                        )
                    generated_files.append(str(page_path))

                if capture is not None:
                    try:
                        source_stems = sorted({
                            t.get("source_file", "") for t in week_topics
                            if t.get("source_file")
                        })
                        primary_heading = (
                            week_topics[0]["heading"] if week_topics else "synthetic"
                        )
                        capture.log_decision(
                            decision_type="source_selection",
                            decision=(
                                f"Week {week_num}: ground content on "
                                f"{primary_heading!r} from sources "
                                f"{source_stems or ['(no DART staging found)']}."
                            ),
                            rationale=(
                                "Selected DART-derived topics whose parsed "
                                "headings align with the week's chapter "
                                "objectives; synthesized placeholder content "
                                "only when no DART topics were available."
                            ),
                        )
                    except Exception:  # noqa: BLE001
                        # Never let decision capture crash emission.
                        pass

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "weeks_prepared": weeks_prepared,
                "content_paths": generated_files,
                "source_sections": len(topics),
                "content_selection": (
                    "source-grounded" if topics else "synthesized"
                ),
            })

        registry["generate_course_content"] = _generate_course_content
        # END BLOCK: Worker α

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
            """Registry wrapper: real IMSCC analysis (parity with @mcp.tool() variant).

            Previously a zero-value stub (audit Q4). Now opens the zip,
            validates the manifest, counts HTML modules + existing
            assessments, and suggests assessment opportunities — matching
            the MCP variant at trainforge_tools.py:129.
            """
            import zipfile

            imscc_path = kwargs.get("imscc_path", "")
            try:
                imscc = Path(imscc_path)
                if not imscc.exists():
                    return json.dumps({"error": f"IMSCC not found: {imscc_path}"})

                analysis = {
                    "source": str(imscc),
                    "analyzed_at": datetime.now().isoformat(),
                    "content": {
                        "html_modules": 0,
                        "existing_assessments": 0,
                        "total_word_count": 0,
                    },
                    "learning_objectives": [],
                    "assessment_opportunities": [],
                }

                with zipfile.ZipFile(imscc, "r") as z:
                    if "imsmanifest.xml" not in z.namelist():
                        return json.dumps({
                            "error": (
                                f"Invalid IMSCC package: missing imsmanifest.xml "
                                f"in {imscc.name}"
                            ),
                            "hint": (
                                "A valid IMSCC package must contain an "
                                "imsmanifest.xml file"
                            ),
                        })
                    analysis["has_manifest"] = True

                    for name in z.namelist():
                        if name.endswith(".html"):
                            analysis["content"]["html_modules"] += 1
                            content = z.read(name).decode("utf-8", errors="ignore")
                            word_count = len(content.split())
                            analysis["content"]["total_word_count"] += word_count
                            if "objective" in content.lower():
                                analysis["learning_objectives"].append({
                                    "source_file": name,
                                    "detected": True,
                                })
                        elif name.endswith(".xml") and "assessment" in name.lower():
                            analysis["content"]["existing_assessments"] += 1

                if analysis["content"]["html_modules"] > 0:
                    analysis["assessment_opportunities"] = [
                        {
                            "type": "quiz",
                            "coverage": "per_module",
                            "estimated_questions": (
                                analysis["content"]["html_modules"] * 5
                            ),
                        },
                        {
                            "type": "exam",
                            "coverage": "comprehensive",
                            "estimated_questions": min(
                                50, analysis["content"]["html_modules"] * 3
                            ),
                        },
                    ]

                return json.dumps(analysis)
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry["analyze_imscc_content"] = _analyze_imscc_content

        # ============================================================================
        # BLOCK: Worker β edits ONLY below this line through the next END marker.
        # Scope: _generate_assessments replacement. See plans/pipeline-execution-
        # fixes/contracts.md § "Trainforge-execution contract".
        # ============================================================================
        async def _generate_assessments(**kwargs):
            """Run Trainforge's full corpus pipeline against the IMSCC and
            generate grounded assessments.

            Concrete steps:

            1. Invoke Trainforge's :class:`CourseProcessor` (the same code
               path ``python -m Trainforge.process_course`` uses) against
               the packaged IMSCC. Produces ``corpus/chunks.jsonl``,
               ``graph/concept_graph_semantic.json``, ``manifest.json``,
               and a ``quality/`` report, validating under chunk_v4 /
               typed-edge schemas when the opt-in flags are set.
            2. Aggregate inline ``chunk["misconceptions"]`` entries into
               a first-class ``graph/misconceptions.json`` document with
               content-hash IDs (``mc_[0-9a-f]{16}``), per REC-LNK-02 and
               the ``misconception.schema.json`` shape.
            3. Run :class:`AssessmentGenerator` honoring the workflow's
               ``question_count`` / ``bloom_levels`` / ``objective_ids``
               params against the generated chunks, writing a single
               well-formed ``assessments.json`` (NOT the legacy
               jsonl-then-concat pattern that produced "Extra data"
               errors).

            Output dir: ``{project_workspace}/trainforge/`` where
            ``project_workspace`` is derived from
            ``imscc_path.parent.parent`` (the Courseforge project dir)
            or, for standalone calls, from an explicit ``project_id``
            kwarg. Colocating with the Courseforge export dir keeps all
            per-run artifacts under one tree and lets the
            libv2-archival phase locate them without a cross-tree
            lookup.
            """
            import hashlib as _hashlib
            import os as _os
            import traceback as _traceback

            course_id = kwargs.get("course_id") or kwargs.get("course_code") or ""
            question_count = int(kwargs.get("question_count", 10))
            bloom_levels_str = kwargs.get("bloom_levels", "remember,understand,apply")
            objective_ids_str = kwargs.get("objective_ids", "")
            imscc_path_str = kwargs.get("imscc_path", "")
            project_id_kw = kwargs.get("project_id", "")
            domain = kwargs.get("domain") or "general"
            division = kwargs.get("division") or "STEM"

            # Normalize list-ish params.
            if isinstance(bloom_levels_str, list):
                bloom_levels = [str(b).strip() for b in bloom_levels_str if str(b).strip()]
            else:
                bloom_levels = [b.strip() for b in str(bloom_levels_str).split(",") if b.strip()]
            if not bloom_levels:
                bloom_levels = ["remember", "understand", "apply"]

            if isinstance(objective_ids_str, list):
                objective_ids = [str(o).strip() for o in objective_ids_str if str(o).strip()]
            else:
                objective_ids = [o.strip() for o in str(objective_ids_str).split(",") if o.strip()]
            if not objective_ids:
                objective_ids = [f"{course_id}_OBJ_{i}" for i in range(1, 7)]

            # Locate project workspace. Standard path: imscc is under
            # Courseforge/exports/<proj>/05_final_package/, so project_dir
            # is imscc.parent.parent. Explicit project_id kwarg wins if set.
            project_dir: Optional[Path] = None
            imscc_path = Path(imscc_path_str) if imscc_path_str else None
            if project_id_kw:
                candidate = _PROJECT_ROOT / "Courseforge" / "exports" / project_id_kw
                if candidate.exists():
                    project_dir = candidate
            if project_dir is None and imscc_path and imscc_path.exists():
                candidate = imscc_path.parent.parent
                if candidate.exists():
                    project_dir = candidate
            if project_dir is None:
                # Last-resort fallback: most recent export dir matching course_id.
                exports_dir = _PROJECT_ROOT / "Courseforge" / "exports"
                if exports_dir.exists():
                    matches = sorted(
                        (p for p in exports_dir.iterdir()
                         if p.is_dir() and course_id and course_id.lower() in p.name.lower()),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if matches:
                        project_dir = matches[0]
            if project_dir is None:
                return json.dumps({
                    "error": "Cannot locate project workspace for Trainforge output",
                    "imscc_path": imscc_path_str,
                    "course_id": course_id,
                })

            trainforge_dir = project_dir / "trainforge"
            # Wipe any prior run's output so a retry starts clean.
            if trainforge_dir.exists():
                shutil.rmtree(trainforge_dir, ignore_errors=True)
            trainforge_dir.mkdir(parents=True, exist_ok=True)

            if not imscc_path or not imscc_path.exists() or imscc_path.stat().st_size == 0:
                return json.dumps({
                    "error": "IMSCC package not found or empty; Trainforge requires the packaging phase to complete first",
                    "imscc_path": imscc_path_str,
                })

            # Invoke CourseProcessor. Writes:
            #   <trainforge_dir>/corpus/chunks.jsonl
            #   <trainforge_dir>/graph/concept_graph.json
            #   <trainforge_dir>/graph/concept_graph_semantic.json
            #   <trainforge_dir>/graph/pedagogy_graph.json
            #   <trainforge_dir>/manifest.json
            #   <trainforge_dir>/quality/quality_report.json
            try:
                from Trainforge.process_course import CourseProcessor
            except Exception as e:
                return json.dumps({
                    "error": f"Failed to import CourseProcessor: {e}",
                    "traceback": _traceback.format_exc(limit=4),
                })

            processor = CourseProcessor(
                imscc_path=str(imscc_path),
                output_dir=str(trainforge_dir),
                course_code=course_id,
                division=division,
                domain=domain,
                strict_mode=False,
            )

            # CourseProcessor's internal DecisionCapture uses
            # phase="content_extraction" (underscore), which is NOT one
            # of the 24 dash-separated phase names enumerated in
            # schemas/events/decision_event.schema.json. Under
            # DECISION_VALIDATION_STRICT=true (the flag the integration
            # test sets alongside the other opt-in shapes), strict
            # validation fires an exception mid-run and the corpus is
            # never written. Until process_course.py is patched (see
            # open-issues note in the handback), scope the strict flag
            # off for the duration of the CourseProcessor call only —
            # the downstream AssessmentGenerator capture below still
            # runs under the caller's configured strictness.
            _prior_strict = _os.environ.get("DECISION_VALIDATION_STRICT")
            _os.environ["DECISION_VALIDATION_STRICT"] = "false"
            try:
                summary = processor.process()
            except Exception as e:
                return json.dumps({
                    "error": f"CourseProcessor.process() failed: {e}",
                    "traceback": _traceback.format_exc(limit=6),
                    "output_dir": str(trainforge_dir),
                })
            finally:
                if _prior_strict is None:
                    _os.environ.pop("DECISION_VALIDATION_STRICT", None)
                else:
                    _os.environ["DECISION_VALIDATION_STRICT"] = _prior_strict

            chunks_path = trainforge_dir / "corpus" / "chunks.jsonl"
            semantic_graph_path = trainforge_dir / "graph" / "concept_graph_semantic.json"

            if not chunks_path.exists():
                return json.dumps({
                    "error": "CourseProcessor did not produce chunks.jsonl",
                    "output_dir": str(trainforge_dir),
                })

            # Aggregate first-class misconceptions.json. Pulls inline
            # misconceptions from each chunk, dedupes by content, and
            # assigns mc_<16-hex> content-hash IDs per
            # schemas/knowledge/misconception.schema.json.
            loaded_chunks: list = []
            with open(chunks_path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        loaded_chunks.append(json.loads(_line))
                    except (json.JSONDecodeError, ValueError):
                        continue

            mc_entities: list = []
            mc_seen: set = set()
            for _c in loaded_chunks:
                for _mc in _c.get("misconceptions") or []:
                    if not isinstance(_mc, dict):
                        continue
                    mtext = str(_mc.get("misconception", "")).strip()
                    ctext = str(_mc.get("correction", "")).strip()
                    if not mtext:
                        continue
                    # Correction is minLength:1 under the schema. Supply
                    # a minimal placeholder when the source didn't carry
                    # one (common with regex-extracted prose).
                    if not ctext:
                        ctext = "Correction not captured in source; review instructor materials."
                    _digest = _hashlib.sha256(
                        f"{mtext}|{ctext}".encode("utf-8")
                    ).hexdigest()[:16]
                    mc_id = f"mc_{_digest}"
                    if mc_id in mc_seen:
                        continue
                    mc_seen.add(mc_id)
                    entity: dict = {
                        "id": mc_id,
                        "misconception": mtext,
                        "correction": ctext,
                    }
                    tags = _c.get("concept_tags") or []
                    if isinstance(tags, list) and tags:
                        entity["concept_id"] = str(tags[0])
                    los = _c.get("learning_outcome_refs") or []
                    if isinstance(los, list) and los:
                        entity["lo_id"] = str(los[0])
                    mc_entities.append(entity)

            # Fallback: process_course.py surfaced zero misconceptions
            # but we have real chunks — try the regex extractor on chunk
            # text. Keeps the artifact shape honest while Courseforge
            # (Worker α) is still being brought online with JSON-LD
            # misconceptions.
            if not mc_entities and loaded_chunks:
                try:
                    from Trainforge.process_course import extract_misconceptions_from_text
                    for _c in loaded_chunks:
                        text = str(_c.get("text", ""))
                        for _mc in extract_misconceptions_from_text(text):
                            mtext = _mc.get("misconception", "").strip()
                            if not mtext:
                                continue
                            ctext = _mc.get("correction") or "Correction not captured in source; review instructor materials."
                            _digest = _hashlib.sha256(
                                f"{mtext}|{ctext}".encode("utf-8")
                            ).hexdigest()[:16]
                            mc_id = f"mc_{_digest}"
                            if mc_id in mc_seen:
                                continue
                            mc_seen.add(mc_id)
                            mc_entities.append({
                                "id": mc_id,
                                "misconception": mtext,
                                "correction": ctext,
                            })
                        if mc_entities:
                            break
                except Exception:
                    pass

            misconceptions_path = trainforge_dir / "graph" / "misconceptions.json"
            misconceptions_path.parent.mkdir(parents=True, exist_ok=True)
            with open(misconceptions_path, "w", encoding="utf-8") as _f:
                json.dump({"misconceptions": mc_entities}, _f, indent=2, ensure_ascii=False)

            # Run AssessmentGenerator on the Trainforge chunks. Every
            # field the ContentExtractor reads (text, concept_tags,
            # source, id) is already present in the canonical chunk
            # shape. Decision capture via create_trainforge_capture
            # writes the rationale stream.
            try:
                from Trainforge.generators.assessment_generator import AssessmentGenerator
            except Exception as e:
                return json.dumps({
                    "error": f"Failed to import AssessmentGenerator: {e}",
                    "traceback": _traceback.format_exc(limit=4),
                    "chunks_path": str(chunks_path),
                })

            gen_capture = None
            try:
                from lib.trainforge_capture import create_trainforge_capture
                gen_capture = create_trainforge_capture(
                    course_code=course_id or "UNKNOWN",
                    imscc_source=str(imscc_path),
                )
            except Exception:
                gen_capture = None

            generator = AssessmentGenerator(capture=gen_capture, check_leaks=True)
            try:
                assessment = generator.generate(
                    course_code=course_id,
                    objective_ids=objective_ids,
                    bloom_levels=bloom_levels,
                    question_count=question_count,
                    source_chunks=loaded_chunks,
                )
            except Exception as e:
                return json.dumps({
                    "error": f"AssessmentGenerator.generate() failed: {e}",
                    "traceback": _traceback.format_exc(limit=6),
                    "chunks_path": str(chunks_path),
                })

            assessments_path = trainforge_dir / "assessments.json"
            assessment_doc = assessment.to_dict()
            # Single write, single well-formed JSON document. The legacy
            # "Extra data" bug came from calling json.dump then appending
            # additional text to the same handle; we guard against that
            # by using a fresh open() and exactly one dump call.
            with open(assessments_path, "w", encoding="utf-8") as _f:
                json.dump(assessment_doc, _f, indent=2, ensure_ascii=False)

            if gen_capture is not None:
                try:
                    gen_capture.log_decision(
                        decision_type="content_selection",
                        decision=(
                            f"Trainforge phase wrote {len(loaded_chunks)} chunks, "
                            f"{len(mc_entities)} misconceptions, "
                            f"{len(assessment.questions)} assessment questions "
                            f"to {trainforge_dir}"
                        ),
                        rationale=(
                            "Ran CourseProcessor against the packaged IMSCC to produce the "
                            "canonical corpus + typed-edge graph, then synthesized misconception "
                            "entities with content-hash IDs. Colocated output under the Courseforge "
                            "project dir so downstream LibV2 archival can byte-copy without a "
                            "cross-tree lookup. Honored workflow params for bloom_levels "
                            f"({','.join(bloom_levels)}) and question_count ({question_count})."
                        ),
                    )
                except Exception:
                    pass

            mc_id_out = str(misconceptions_path) if mc_entities else None
            validated = (
                _os.getenv("TRAINFORGE_VALIDATE_CHUNKS", "").lower() == "true"
            )

            return json.dumps({
                "success": True,
                "assessment_id": assessment.assessment_id,
                "question_count": len(assessment.questions),
                "output_path": str(assessments_path),
                "assessments_path": str(assessments_path),
                "chunks_path": str(chunks_path),
                "concept_graph_path": (
                    str(semantic_graph_path) if semantic_graph_path.exists() else None
                ),
                "misconceptions_path": mc_id_out,
                "trainforge_dir": str(trainforge_dir),
                "chunks_count": len(loaded_chunks),
                "misconceptions_count": len(mc_entities),
                "strict_chunks_validated": validated,
                "processor_summary": {
                    "course_code": summary.get("course_code"),
                    "title": summary.get("title"),
                    "stats": summary.get("stats"),
                },
            })

        registry["generate_assessments"] = _generate_assessments
        # END BLOCK: Worker β
    except Exception:
        pass

    # LibV2 archival tool
    # ============================================================================
    # BLOCK: Worker γ edits ONLY below this line through the next END marker.
    # Scope: _archive_to_libv2 extension. See plans/pipeline-execution-fixes/
    # contracts.md § "LibV2-archival contract".
    # ============================================================================
    async def _archive_to_libv2(**kwargs):
        """Archive pipeline artifacts (sources + Trainforge outputs) to LibV2.

        Parity with the ``@mcp.tool()`` variant at ``pipeline_tools.py:556-726``
        (slug computation, source copying, manifest shape, feature-flag scans)
        plus Wave 15 Trainforge output copying into
        ``corpus/`` / ``graph/`` / ``training_specs/`` / ``quality/``.

        Trainforge output lookup order (first match wins):
          1. Explicit kwargs: ``project_workspace`` (str/Path), else
             ``project_id`` → ``Courseforge/exports/{project_id}/trainforge/``.
          2. Legacy ``assessment_path`` — when it points at a directory, used
             as the Trainforge output root; when it points at a file, copied
             into ``corpus/`` (preserves the MCP-tool variant's behavior so
             existing provenance-flag tests keep passing).
          3. Heuristic fallback — scan ``Courseforge/exports/*/trainforge/``
             and ``state/runs/*/trainforge/`` for the most recently modified
             ``chunks.jsonl``. Absence is not an error — features flags fall
             back to ``false`` with a warning.
        """
        course_name = (
            kwargs.get("course_name")
            or kwargs.get("course_id")
            or kwargs.get("id")
            or ""
        )
        domain = kwargs.get("domain") or "general"
        division = kwargs.get("division", "STEM")
        pdf_paths_str = kwargs.get("pdf_paths", "") or ""
        html_paths_str = kwargs.get("html_paths", "") or ""
        imscc_path_str = kwargs.get("imscc_path", "") or ""
        assessment_path_str = kwargs.get("assessment_path", "") or ""
        subdomains_str = kwargs.get("subdomains", "") or ""
        project_workspace_kw = kwargs.get("project_workspace") or ""
        project_id_kw = kwargs.get("project_id") or ""

        if not course_name:
            return json.dumps({"error": "archive_to_libv2 requires course_name"})

        slug = course_name.lower().replace("_", "-").replace(" ", "-")
        libv2_root = PROJECT_ROOT / "LibV2"
        course_dir = libv2_root / "courses" / slug

        for subdir in [
            "source/pdf", "source/html", "source/imscc",
            "corpus", "graph", "pedagogy", "training_specs", "quality"
        ]:
            (course_dir / subdir).mkdir(parents=True, exist_ok=True)

        archived = {
            "pdfs": [],
            "html": [],
            "imscc": None,
            "assessment": None,
            "trainforge": {
                "chunks": None,
                "graph": None,
                "misconceptions": None,
                "assessments": None,
                "quality_report": None,
            },
        }

        # --- Copy raw PDFs -------------------------------------------------
        if pdf_paths_str:
            for p in pdf_paths_str.split(","):
                src = Path(p.strip())
                if src.exists():
                    dest = course_dir / "source" / "pdf" / src.name
                    shutil.copy2(src, dest)
                    archived["pdfs"].append(str(dest))

        # --- Copy DART HTML outputs (+ adjacent .quality.json) -------------
        if html_paths_str:
            for p in html_paths_str.split(","):
                src = Path(p.strip())
                if src.exists():
                    dest = course_dir / "source" / "html" / src.name
                    shutil.copy2(src, dest)
                    archived["html"].append(str(dest))
                    quality_json = src.with_suffix(".quality.json")
                    if quality_json.exists():
                        shutil.copy2(
                            quality_json, course_dir / "quality" / quality_json.name
                        )

        # --- Copy IMSCC package -------------------------------------------
        if imscc_path_str:
            src = Path(imscc_path_str)
            if src.exists():
                dest = course_dir / "source" / "imscc" / src.name
                shutil.copy2(src, dest)
                archived["imscc"] = str(dest)

        # --- Resolve Trainforge workspace ---------------------------------
        trainforge_dir: Optional[Path] = None

        if project_workspace_kw:
            candidate = Path(project_workspace_kw)
            if candidate.name != "trainforge":
                candidate = candidate / "trainforge"
            if candidate.exists() and candidate.is_dir():
                trainforge_dir = candidate

        if trainforge_dir is None and project_id_kw:
            candidate = (
                PROJECT_ROOT / "Courseforge" / "exports" / project_id_kw / "trainforge"
            )
            if candidate.exists() and candidate.is_dir():
                trainforge_dir = candidate

        # Legacy assessment_path handling: keep parity with the MCP-tool
        # variant so existing provenance / evidence flag tests pass
        # (they pass assessment_path=<chunks.jsonl>). If the path points at
        # a directory, treat it as the trainforge workspace root.
        if assessment_path_str:
            ap = Path(assessment_path_str)
            if ap.exists():
                if ap.is_dir():
                    if trainforge_dir is None:
                        trainforge_dir = ap
                else:
                    dest = course_dir / "corpus" / ap.name
                    shutil.copy2(ap, dest)
                    archived["assessment"] = str(dest)

        # Heuristic fallback: scan well-known locations for chunks.jsonl.
        if trainforge_dir is None:
            candidates: list[Path] = []
            exports_root = PROJECT_ROOT / "Courseforge" / "exports"
            if exports_root.exists():
                for project_dir in exports_root.iterdir():
                    if not project_dir.is_dir():
                        continue
                    tf = project_dir / "trainforge"
                    if (tf / "chunks.jsonl").exists() or (tf / "corpus" / "chunks.jsonl").exists():
                        candidates.append(tf)
            runs_root = PROJECT_ROOT / "state" / "runs"
            if runs_root.exists():
                for run_dir in runs_root.iterdir():
                    if not run_dir.is_dir():
                        continue
                    tf = run_dir / "trainforge"
                    if (tf / "chunks.jsonl").exists() or (tf / "corpus" / "chunks.jsonl").exists():
                        candidates.append(tf)
            if candidates:
                def _chunks_mtime(p):
                    # Support both flat and nested (CourseProcessor-native) layouts.
                    nested = p / "corpus" / "chunks.jsonl"
                    flat = p / "chunks.jsonl"
                    if nested.exists():
                        return nested.stat().st_mtime
                    if flat.exists():
                        return flat.stat().st_mtime
                    return 0.0
                trainforge_dir = max(candidates, key=_chunks_mtime)

        # --- Copy Trainforge outputs --------------------------------------
        # Worker β writes in CourseProcessor's native nested layout
        # (trainforge/corpus/chunks.jsonl, trainforge/graph/*.json). We
        # also check the flat layout for backward-compat with any caller
        # that mirrors the older stub's expected paths.
        def _pick(*candidates):
            for c in candidates:
                if c.exists() and c.is_file():
                    return c
            return None

        if trainforge_dir is not None and trainforge_dir.exists():
            copy_map = [
                (_pick(trainforge_dir / "corpus" / "chunks.jsonl",
                       trainforge_dir / "chunks.jsonl"),
                 course_dir / "corpus" / "chunks.jsonl", "chunks"),
                (_pick(trainforge_dir / "graph" / "concept_graph_semantic.json",
                       trainforge_dir / "concept_graph_semantic.json"),
                 course_dir / "graph" / "concept_graph_semantic.json", "graph"),
                (_pick(trainforge_dir / "graph" / "misconceptions.json",
                       trainforge_dir / "misconceptions.json"),
                 course_dir / "graph" / "misconceptions.json", "misconceptions"),
                (_pick(trainforge_dir / "training_specs" / "assessments.json",
                       trainforge_dir / "assessments.json"),
                 course_dir / "training_specs" / "assessments.json", "assessments"),
                (_pick(trainforge_dir / "quality" / "quality_report.json"),
                 course_dir / "quality" / "quality_report.json", "quality_report"),
            ]
            for src, dest, label in copy_map:
                if src is not None and src.exists() and src.is_file():
                    try:
                        shutil.copy2(src, dest)
                        archived["trainforge"][label] = str(dest)
                    except OSError as exc:
                        logger.warning(
                            f"archive_to_libv2: failed to copy {src} -> {dest}: {exc}"
                        )
        else:
            logger.warning(
                "archive_to_libv2: no Trainforge output dir located for "
                f"course {course_name} — features flags will default to false."
            )

        # --- Build manifest (with source_artifacts checksums) -------------
        import hashlib

        def _sha256(filepath: Path) -> str:
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                for block in iter(lambda: f.read(8192), b""):
                    h.update(block)
            return h.hexdigest()

        source_artifacts: dict = {}
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

        # Wave 10 / Wave 11 feature flags — scan the archived files.
        source_provenance_flag = _detect_source_provenance(course_dir)
        evidence_source_provenance_flag = _detect_evidence_source_provenance(course_dir)

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
            "features": {
                "source_provenance": source_provenance_flag,
                "evidence_source_provenance": evidence_source_provenance_flag,
            },
            "trainforge_workspace": (
                str(trainforge_dir) if trainforge_dir is not None else None
            ),
            "artifact_counts": {
                "pdfs": len(archived["pdfs"]),
                "html_files": len(archived["html"]),
                "imscc": 1 if archived["imscc"] else 0,
                "assessment": 1 if archived["assessment"] else 0,
                "trainforge": sum(
                    1 for v in archived["trainforge"].values() if v is not None
                ),
            },
        })

    registry["archive_to_libv2"] = _archive_to_libv2
    # END BLOCK: Worker γ

    async def _build_source_module_map(**kwargs):
        """Source-router (Wave 9 ``source_mapping`` phase) — real heuristic.

        Previously wrote an empty ``source_module_map.json``, which left
        every Courseforge page emitted without ``sourceReferences[]`` and
        pinned the ``source_provenance`` / ``evidence_source_provenance``
        feature flags to false (investigation Issue 7). This implementation
        routes DART source blocks to Courseforge pages via keyword-overlap
        scoring:

          1. Enumerate DART block IDs by scanning ``staging_dir`` for
             ``*_synthesized.json`` sidecars — each ``sections[]`` entry
             contributes ``section_id``, ``section_title``, and any
             keyword-bearing text in ``data`` / ``sources_used``.
          2. Load the textbook structure (when available) and the
             project's objectives to enumerate per-page target topics.
          3. For each week (1..duration_weeks) and each page role
             (overview, content_0K, application, self_check, summary),
             score DART blocks by keyword overlap with the page's
             dominant topic. Blocks above a stronger threshold become
             ``primary`` refs; blocks above a weaker threshold become
             ``contributing`` refs.
          4. Emit the map in the Wave 9 shape that
             ``Courseforge.scripts.generate_course._page_refs_for``
             consumes: ``{week_key: {page_id: {primary, contributing,
             confidence}}}`` using ``dart:{slug}#{block_id}`` source IDs.

        No LLM. Pure text overlap — imperfect but deterministic and
        better than an empty map for provenance propagation.
        """
        project_id = kwargs.get("project_id", "")
        staging_dir_kw = kwargs.get("staging_dir", "") or ""
        textbook_structure_path = kwargs.get("textbook_structure_path", "") or ""

        if not project_id:
            return json.dumps({"error": "source-router requires project_id"})

        project_path = PROJECT_ROOT / "Courseforge" / "exports" / project_id
        project_path.mkdir(parents=True, exist_ok=True)
        map_path = project_path / "source_module_map.json"

        # ------------------------------------------------------------- #
        # Load project config for duration_weeks + course_name.          #
        # ------------------------------------------------------------- #
        config_path = project_path / "project_config.json"
        duration_weeks = 12
        course_name = project_id
        objectives_path: Optional[str] = None
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                duration_weeks = int(cfg.get("duration_weeks") or 12)
                course_name = cfg.get("course_name") or project_id
                objectives_path = cfg.get("objectives_path") or None
            except (OSError, ValueError):
                pass

        # ------------------------------------------------------------- #
        # Enumerate DART source blocks from staging_dir sidecars.        #
        # Each entry: {block_id, slug, keywords(set[str]), title}.       #
        # ------------------------------------------------------------- #
        dart_blocks: list = []
        staging_dir = Path(staging_dir_kw) if staging_dir_kw else None
        if staging_dir is None or not staging_dir.exists():
            # Fallback: scan Courseforge inputs for any synthesized sidecars.
            staging_dir = COURSEFORGE_INPUTS

        def _tokenize(text: str) -> set:
            """Lowercase, strip punctuation, drop stopwords + short tokens."""
            if not text:
                return set()
            import re as _re
            cleaned = _re.sub(r"[^a-z0-9\s]", " ", text.lower())
            _stopwords = {
                "the", "and", "for", "with", "from", "that", "this", "are",
                "was", "were", "has", "have", "had", "but", "not", "all",
                "any", "may", "can", "one", "two", "its", "their", "they",
                "will", "been", "you", "your", "our", "his", "her", "which",
                "what", "who", "why", "how", "when", "where", "into", "out",
                "over", "such", "more", "most", "some", "about", "there",
                "these", "those", "than", "then", "also", "only", "used",
                "use", "see", "via", "per",
            }
            return {
                t for t in cleaned.split()
                if len(t) > 3 and t not in _stopwords
            }

        if staging_dir and staging_dir.exists():
            for sidecar in sorted(staging_dir.rglob("*_synthesized.json")):
                try:
                    doc = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                slug = sidecar.stem.replace("_synthesized", "")
                sections = doc.get("sections") or []
                if not isinstance(sections, list):
                    continue
                for section in sections:
                    if not isinstance(section, dict):
                        continue
                    block_id = str(section.get("section_id") or "").strip()
                    if not block_id:
                        continue
                    title = str(section.get("section_title") or "").strip()
                    section_type = str(section.get("section_type") or "").strip()
                    # Gather text for keyword extraction: title + any
                    # paragraph text + key-value block labels + data keys.
                    text_bits: list = [title, section_type]
                    data = section.get("data")
                    if isinstance(data, dict):
                        for k, v in data.items():
                            text_bits.append(str(k))
                            if isinstance(v, str):
                                text_bits.append(v)
                            elif isinstance(v, list):
                                for item in v[:20]:
                                    if isinstance(item, str):
                                        text_bits.append(item)
                                    elif isinstance(item, dict):
                                        for sub_v in item.values():
                                            if isinstance(sub_v, str):
                                                text_bits.append(sub_v)
                    keywords = _tokenize(" ".join(text_bits))
                    if not keywords:
                        # Fall back to splitting the block id so at least
                        # the title contributes a scoring signal.
                        keywords = _tokenize(title) or _tokenize(slug)
                    dart_blocks.append({
                        "block_id": block_id,
                        "slug": slug,
                        "title": title,
                        "keywords": keywords,
                        "source_id": f"dart:{slug}#{block_id}",
                    })

        # ------------------------------------------------------------- #
        # Enumerate per-week topics. Preference order:                   #
        #   1. textbook_structure_path chapters/sections                 #
        #   2. objectives_path chapter/terminal objective statements     #
        #   3. DART block titles themselves (round-robin by week)        #
        # ------------------------------------------------------------- #
        week_topics: dict = {}  # week_num -> {page_id: set[str]}

        def _set_week_page(week_num: int, page_id: str, kw: set):
            week_topics.setdefault(week_num, {})[page_id] = kw

        structure_chapters: list = []
        if textbook_structure_path:
            sp = Path(textbook_structure_path)
            if sp.exists():
                try:
                    structure_doc = json.loads(sp.read_text(encoding="utf-8"))
                    chapters = structure_doc.get("chapters") or []
                    if isinstance(chapters, list):
                        structure_chapters = chapters
                except (OSError, ValueError):
                    pass

        objective_statements: list = []
        if objectives_path:
            op = Path(objectives_path)
            if op.exists():
                try:
                    obj_doc = json.loads(op.read_text(encoding="utf-8"))
                    for group in ("chapter_objectives", "terminal_objectives",
                                  "course_objectives"):
                        for item in obj_doc.get(group, []) or []:
                            if isinstance(item, dict):
                                text = (
                                    item.get("statement")
                                    or item.get("description")
                                    or item.get("text")
                                    or ""
                                )
                                if text:
                                    objective_statements.append(text)
                except (OSError, ValueError):
                    pass

        # Assemble per-week keyword bags.
        page_roles = ("overview", "content_01", "application",
                      "self_check", "summary")

        # Prefer chapters / objective statements when available.
        topic_pool: list = []
        for ch in structure_chapters:
            if isinstance(ch, dict):
                ch_title = str(ch.get("title") or "")
                ch_topics = [ch_title]
                for sub in ch.get("sections") or []:
                    if isinstance(sub, dict):
                        ch_topics.append(str(sub.get("title") or ""))
                    elif isinstance(sub, str):
                        ch_topics.append(sub)
                topic_pool.append(_tokenize(" ".join(ch_topics)))
        if not topic_pool and objective_statements:
            for stmt in objective_statements:
                topic_pool.append(_tokenize(stmt))
        if not topic_pool and dart_blocks:
            # Final fallback: let DART block titles drive topic bags, one
            # per block, so each week gets at least a nominal signal.
            for blk in dart_blocks:
                topic_pool.append(blk["keywords"])

        # Distribute topic_pool across weeks (round-robin).
        for week_num in range(1, duration_weeks + 1):
            if not topic_pool:
                primary_bag: set = set()
            else:
                # Pick the topic whose index matches (week_num-1) mod len.
                primary_bag = topic_pool[(week_num - 1) % len(topic_pool)]
            for page_id in page_roles:
                # Application / self_check / summary share week bag;
                # content_0N gets the same bag plus a blend across
                # neighbor weeks so content doesn't duplicate overview.
                bag = set(primary_bag)
                if page_id.startswith("content") and len(topic_pool) > 1:
                    neighbor = topic_pool[(week_num) % len(topic_pool)]
                    bag = bag.union(neighbor)
                _set_week_page(week_num, page_id, bag)

        # ------------------------------------------------------------- #
        # Score blocks per (week, page) and emit refs.                   #
        # ------------------------------------------------------------- #
        source_module_map: dict = {}
        chunk_ids: set = set()

        if dart_blocks:
            for week_num in range(1, duration_weeks + 1):
                week_key = f"week_{week_num:02d}"
                pages_for_week = week_topics.get(week_num, {})
                week_entries: dict = {}
                for page_id, target_bag in pages_for_week.items():
                    if not target_bag:
                        # Degenerate fallback: assign the nth DART block
                        # round-robin as primary.
                        fallback = dart_blocks[(week_num - 1) % len(dart_blocks)]
                        week_entries[page_id] = {
                            "primary": [fallback["source_id"]],
                            "contributing": [],
                            "confidence": 0.3,
                        }
                        chunk_ids.add(fallback["source_id"])
                        continue
                    scored: list = []
                    for blk in dart_blocks:
                        overlap = len(target_bag & blk["keywords"])
                        if overlap == 0:
                            continue
                        # Jaccard-ish score for ranking stability.
                        union = max(1, len(target_bag | blk["keywords"]))
                        score = overlap / union
                        scored.append((score, overlap, blk))
                    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
                    primary_ids: list = []
                    contributing_ids: list = []
                    top_score = scored[0][0] if scored else 0.0
                    # Strong threshold: top-K (K=1 for content pages,
                    # K=2 when multiple blocks clearly overlap).
                    for score, overlap, blk in scored:
                        if score >= max(0.15, top_score * 0.8) and len(primary_ids) < 2:
                            primary_ids.append(blk["source_id"])
                        elif score >= 0.05 and len(contributing_ids) < 3:
                            contributing_ids.append(blk["source_id"])
                    if not primary_ids and scored:
                        # Still assign the top match even when all scores
                        # are low — better than producing no provenance.
                        primary_ids.append(scored[0][2]["source_id"])
                    if not primary_ids:
                        # No overlap at all: round-robin a DART block as
                        # primary with low confidence.
                        fallback = dart_blocks[(week_num - 1) % len(dart_blocks)]
                        primary_ids.append(fallback["source_id"])
                        top_score = 0.2
                    for sid in primary_ids:
                        chunk_ids.add(sid)
                    for sid in contributing_ids:
                        chunk_ids.add(sid)
                    week_entries[page_id] = {
                        "primary": primary_ids,
                        "contributing": contributing_ids,
                        "confidence": round(max(top_score, 0.2), 2),
                    }
                if week_entries:
                    source_module_map[week_key] = week_entries

        map_path.write_text(
            json.dumps(source_module_map, indent=2),
            encoding="utf-8",
        )

        routing_mode = (
            "keyword_overlap_heuristic" if dart_blocks
            else "stub_empty_map"
        )

        return json.dumps({
            "source_module_map_path": str(map_path),
            "source_chunk_ids": sorted(chunk_ids),
            "staging_dir": str(staging_dir) if staging_dir else "",
            "textbook_structure_path": textbook_structure_path,
            "routing_mode": routing_mode,
            "dart_blocks_indexed": len(dart_blocks),
            "weeks_routed": len(source_module_map),
            "course_name": course_name,
        })

    registry["build_source_module_map"] = _build_source_module_map

    # ================================================================= #
    # Runtime registry stubs for the 7 tools that AGENT_TOOL_MAPPING     #
    # routes but _build_tool_registry previously skipped (MCP audit      #
    # Q1 critical finding). Each wrapper imports the @mcp.tool()         #
    # implementation at call time (register_* functions create closures  #
    # — we extract them into a capturing MCP stand-in the same way       #
    # test_stage_dart_outputs.py::_CapturingMCP does).                   #
    # ================================================================= #
    class _CapturingMCP:
        """Minimal stand-in for FastMCP: captures decorated tools by name."""
        def __init__(self) -> None:
            self.tools: dict = {}

        def tool(self):  # noqa: D401 - mimics FastMCP's .tool() decorator
            def _decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return _decorator

    def _capture_dart_tools() -> dict:
        try:
            from MCP.tools.dart_tools import register_dart_tools
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"DART tool capture failed: {exc}")
            return {}
        mcp_cap = _CapturingMCP()
        register_dart_tools(mcp_cap)
        return mcp_cap.tools

    def _capture_courseforge_tools() -> dict:
        try:
            from MCP.tools.courseforge_tools import register_courseforge_tools
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Courseforge tool capture failed: {exc}")
            return {}
        mcp_cap = _CapturingMCP()
        register_courseforge_tools(mcp_cap)
        return mcp_cap.tools

    def _capture_trainforge_tools() -> dict:
        try:
            from MCP.tools.trainforge_tools import register_trainforge_tools
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Trainforge tool capture failed: {exc}")
            return {}
        mcp_cap = _CapturingMCP()
        register_trainforge_tools(mcp_cap)
        return mcp_cap.tools

    async def _get_courseforge_status(**kwargs):
        """Registry wrapper: delegates to courseforge_tools.get_courseforge_status."""
        tools = _capture_courseforge_tools()
        tool = tools.get("get_courseforge_status")
        if tool is None:
            return json.dumps({"error": "get_courseforge_status tool unavailable"})
        return await tool()

    registry["get_courseforge_status"] = _get_courseforge_status

    async def _validate_wcag_compliance(**kwargs):
        """Registry wrapper: delegates to dart_tools.validate_wcag_compliance."""
        html_path = kwargs.get("html_path") or kwargs.get("path") or ""
        tools = _capture_dart_tools()
        tool = tools.get("validate_wcag_compliance")
        if tool is None:
            return json.dumps({"error": "validate_wcag_compliance tool unavailable"})
        return await tool(html_path=html_path)

    registry["validate_wcag_compliance"] = _validate_wcag_compliance

    async def _batch_convert_multi_source(**kwargs):
        """Registry wrapper: delegates to dart_tools.batch_convert_multi_source."""
        combined_dir = kwargs.get("combined_dir") or kwargs.get("input") or ""
        output_zip = kwargs.get("output_zip")
        output_dir = kwargs.get("output_dir")
        tools = _capture_dart_tools()
        tool = tools.get("batch_convert_multi_source")
        if tool is None:
            return json.dumps({"error": "batch_convert_multi_source tool unavailable"})
        return await tool(
            combined_dir=combined_dir,
            output_zip=output_zip,
            output_dir=output_dir,
        )

    registry["batch_convert_multi_source"] = _batch_convert_multi_source

    async def _convert_pdf_multi_source(**kwargs):
        """Registry wrapper: delegates to dart_tools.convert_pdf_multi_source."""
        combined_json_path = (
            kwargs.get("combined_json_path")
            or kwargs.get("combined_json")
            or kwargs.get("source")
            or ""
        )
        output_path = kwargs.get("output_path")
        course_code = kwargs.get("course_code")
        tools = _capture_dart_tools()
        tool = tools.get("convert_pdf_multi_source")
        if tool is None:
            return json.dumps({"error": "convert_pdf_multi_source tool unavailable"})
        return await tool(
            combined_json_path=combined_json_path,
            output_path=output_path,
            course_code=course_code,
        )

    registry["convert_pdf_multi_source"] = _convert_pdf_multi_source

    async def _intake_imscc_package(**kwargs):
        """Registry wrapper: delegates to courseforge_tools.intake_imscc_package."""
        imscc_path = kwargs.get("imscc_path") or kwargs.get("package") or ""
        output_dir = kwargs.get("output_dir") or kwargs.get("extract_to") or ""
        remediate = kwargs.get("remediate", True)
        tools = _capture_courseforge_tools()
        tool = tools.get("intake_imscc_package")
        if tool is None:
            return json.dumps({"error": "intake_imscc_package tool unavailable"})
        return await tool(
            imscc_path=imscc_path,
            output_dir=output_dir,
            remediate=remediate,
        )

    registry["intake_imscc_package"] = _intake_imscc_package

    async def _remediate_course_content(**kwargs):
        """Registry wrapper: delegates to courseforge_tools.remediate_course_content."""
        project_id = kwargs.get("project_id") or ""
        remediation_types = kwargs.get("remediation_types")
        tools = _capture_courseforge_tools()
        tool = tools.get("remediate_course_content")
        if tool is None:
            return json.dumps({"error": "remediate_course_content tool unavailable"})
        return await tool(
            project_id=project_id,
            remediation_types=remediation_types,
        )

    registry["remediate_course_content"] = _remediate_course_content

    async def _validate_assessment(**kwargs):
        """Registry wrapper: delegates to trainforge_tools.validate_assessment."""
        assessment_id = (
            kwargs.get("assessment_id")
            or kwargs.get("assessment")
            or kwargs.get("id")
            or ""
        )
        tools = _capture_trainforge_tools()
        tool = tools.get("validate_assessment")
        if tool is None:
            return json.dumps({"error": "validate_assessment tool unavailable"})
        return await tool(assessment_id=assessment_id)

    registry["validate_assessment"] = _validate_assessment

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
