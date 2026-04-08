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
from MCP.tools.path_validation import validate_path_within_root  # noqa: E402

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


def register_pipeline_tools(mcp):
    """Register pipeline tools with the MCP server."""

    @mcp.tool()
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
            # Import orchestrator tools
            from MCP.tools.orchestrator_tools import create_workflow

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
            result = await create_workflow(
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
            errors = []

            html_paths = [Path(p.strip()) for p in dart_html_paths.split(",")]

            for html_path in html_paths:
                if not html_path.exists():
                    errors.append(f"DART output not found: {html_path}")
                    continue

                # Copy HTML file
                dest = staging_dir / html_path.name
                shutil.copy2(html_path, dest)
                staged_files.append(str(dest))
                logger.info(f"Staged: {html_path.name} -> {dest}")

                # Copy accompanying JSON if exists (DART synthesized metadata)
                json_path = html_path.with_suffix(".json")
                if json_path.exists():
                    json_dest = staging_dir / json_path.name
                    shutil.copy2(json_path, json_dest)
                    staged_files.append(str(json_dest))
                    logger.info(f"Staged: {json_path.name} -> {json_dest}")

                # Also check for _synthesized.json pattern
                synth_json_name = html_path.stem.replace("_synthesized", "") + "_synthesized.json"
                synth_json_path = html_path.parent / synth_json_name
                if synth_json_path.exists() and str(synth_json_path) != str(json_path):
                    synth_json_dest = staging_dir / synth_json_name
                    shutil.copy2(synth_json_path, synth_json_dest)
                    staged_files.append(str(synth_json_dest))
                    logger.info(f"Staged: {synth_json_name} -> {synth_json_dest}")

            if errors and not staged_files:
                return json.dumps({
                    "success": False,
                    "error": "No files staged",
                    "errors": errors
                })

            # Create manifest
            manifest = {
                "run_id": run_id,
                "course_name": course_name,
                "staged_at": datetime.now().isoformat(),
                "staged_files": staged_files,
                "errors": errors if errors else None
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
            from MCP.tools.orchestrator_tools import get_workflow_status

            result = await get_workflow_status(workflow_id)
            workflow = json.loads(result)

            if "error" in workflow:
                return result

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
                "skip_link": 'class="skip-link"' in content or "class='skip-link'" in content,
                "main_role": 'role="main"' in content or "role='main'" in content,
                "aria_sections": 'aria-labelledby="' in content or "aria-labelledby='" in content
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
