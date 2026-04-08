"""
Courseforge MCP Tools

Tools for course generation, IMSCC packaging, and remediation.
Integrates decision capture for training data collection.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.libv2_storage import LibV2Storage  # noqa: E402
from lib.paths import COURSEFORGE_PATH  # noqa: E402
from lib.secure_paths import (  # noqa: E402
    safe_extract_zip,
    sanitize_path_component,
    validate_path_within_root,
)

# Import new telemetry system
try:
    from LibV2.telemetry import ArtifactRef, CaptureSession  # noqa: F401
    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

logger = logging.getLogger(__name__)

# Derived paths
EXPORTS_PATH = COURSEFORGE_PATH / "exports"


def _validate_courseforge_paths():
    """Validate Courseforge paths at module load."""
    if not COURSEFORGE_PATH.exists():
        logger.warning(f"Courseforge installation not found: {COURSEFORGE_PATH}")
    else:
        logger.info(f"Courseforge installation validated: {COURSEFORGE_PATH}")

    if not EXPORTS_PATH.exists():
        EXPORTS_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created exports directory: {EXPORTS_PATH}")


_validate_courseforge_paths()


def _create_capture(course_code: str, phase: str):
    """Create a capture session for Courseforge operations."""
    if HAS_TELEMETRY:
        try:
            return CaptureSession.start_run(
                tool_id="courseforge",
                component=phase,
                meta={"course_code": course_code},
                course_code=course_code,
                phase=phase,
            )
        except Exception as e:
            logger.warning(f"Failed to create telemetry session: {e}")
            return None
    # Fallback to legacy capture
    try:
        from lib.decision_capture import DecisionCapture
        return DecisionCapture(course_code, phase, "courseforge")
    except ImportError:
        return None


def _log_decision(capture, decision_type: str, decision: str, rationale: str, **kwargs):
    """Log a decision using telemetry emit() or legacy log_decision()."""
    if capture is None:
        return

    # Check if this is a CaptureSession (new telemetry)
    if HAS_TELEMETRY and isinstance(capture, CaptureSession):
        # Map decision types to event types
        event_type = "generation.completed"
        if "error" in decision_type.lower():
            event_type = "error.raised"
        elif "validation" in decision_type.lower():
            event_type = "validation.result"

        capture.emit(
            event_type,
            payload={
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **kwargs
            },
            severity="info" if "error" not in decision_type.lower() else "error"
        )
    else:
        # Legacy capture
        capture.log_decision(decision_type, decision, rationale, **kwargs)


def _finalize_capture(capture):
    """Finalize a capture session."""
    if capture is None:
        return

    if HAS_TELEMETRY and isinstance(capture, CaptureSession):
        capture.finish_run("success")
    elif hasattr(capture, 'save'):
        capture.save()


def register_courseforge_tools(mcp):
    """Register Courseforge tools with the MCP server."""

    @mcp.tool()
    async def create_course_project(
        course_name: str,
        objectives_path: str,
        duration_weeks: int = 12,
        credit_hours: int = 3
    ) -> str:
        """
        Initialize a new course generation project.

        Args:
            course_name: Unique course identifier (e.g., "MTH_301")
            objectives_path: Path to exam objectives file
            duration_weeks: Course duration (default: 12)
            credit_hours: Credit hours (default: 3)

        Returns:
            Project workspace path and configuration
        """
        capture = _create_capture(course_name, "courseforge-course-outliner")

        try:
            # Sanitize course name to prevent path traversal
            safe_course_name = sanitize_path_component(course_name)

            # Create timestamped project folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"{timestamp}_{safe_course_name}"
            project_path = validate_path_within_root(
                EXPORTS_PATH / project_name, EXPORTS_PATH
            )

            # Log approach decision
            _log_decision(
                capture,
                decision_type="approach_selection",
                decision=f"Creating course project for {course_name}",
                rationale=f"Initializing {duration_weeks}-week course with {credit_hours} credit hours using standard Courseforge structure",
                context=f"Objectives: {objectives_path}"
            )

            # Create project structure
            project_path.mkdir(parents=True, exist_ok=True)
            (project_path / "00_template_analysis").mkdir()
            (project_path / "01_learning_objectives").mkdir()
            (project_path / "02_course_planning").mkdir()
            (project_path / "03_content_development").mkdir()
            (project_path / "04_quality_validation").mkdir()
            (project_path / "05_final_package").mkdir()
            (project_path / "agent_workspaces").mkdir()

            # Log structure decision
            _log_decision(
                capture,
                decision_type="content_structure",
                decision="Created 6-phase project directory structure",
                rationale="Standard Courseforge workflow: template analysis -> learning objectives -> planning -> content -> validation -> packaging"
            )

            # Create project config
            config = {
                "course_name": course_name,
                "objectives_path": objectives_path,
                "duration_weeks": duration_weeks,
                "credit_hours": credit_hours,
                "created_at": datetime.now().isoformat(),
                "status": "initialized"
            }

            config_path = project_path / "project_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Create project log
            log_path = project_path / "project_log.md"
            with open(log_path, 'w') as f:
                f.write(f"# {course_name} Generation Log\n\n")
                f.write(f"Created: {datetime.now().isoformat()}\n\n")
                f.write("## Project Status\n\n")
                f.write("- [ ] Template Analysis\n")
                f.write("- [ ] Learning Objectives\n")
                f.write("- [ ] Course Planning\n")
                f.write("- [ ] Content Development\n")
                f.write("- [ ] Quality Validation\n")
                f.write("- [ ] Final Package\n")

            # Log file creation
            _log_decision(
                capture,
                decision_type="file_creation",
                decision=f"Created project at {project_path}",
                rationale="Project initialized with config and log files for tracking progress"
            )

            _finalize_capture(capture)

            return json.dumps({
                "success": True,
                "project_id": project_name,
                "project_path": str(project_path),
                "config": config,
                "capture_saved": capture is not None
            })

        except Exception as e:
            _log_decision(
                capture,
                decision_type="error_handling",
                decision=f"Project creation failed: {type(e).__name__}",
                rationale=f"Exception during project initialization: {str(e)[:200]}"
            )
            _finalize_capture(capture)
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def generate_course_content(
        project_id: str,
        week_range: Optional[str] = None,
        parallel: bool = True
    ) -> str:
        """
        Generate course content for specified weeks.

        Args:
            project_id: Project identifier from create_course_project
            week_range: Optional "start-end" weeks (e.g., "1-4"); None for all
            parallel: Use parallel generation (default: True)

        Returns:
            Generation status and content paths
        """
        try:
            project_path = validate_path_within_root(
                EXPORTS_PATH / project_id, EXPORTS_PATH
            )
            if not project_path.exists():
                return json.dumps({"error": f"Project not found: {project_id}"})

            # Load config
            config_path = project_path / "project_config.json"
            if not config_path.exists():
                return json.dumps({"error": f"Project config not found: {config_path}"})

            try:
                with open(config_path) as f:
                    config = json.load(f)
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid project config JSON: {e}"})

            duration = config.get("duration_weeks", 12)

            # Parse week range
            if week_range:
                try:
                    parts = week_range.split("-")
                    if len(parts) != 2:
                        return json.dumps({"error": "Invalid week_range format. Use 'start-end' (e.g., '1-4')"})
                    start, end = int(parts[0]), int(parts[1])
                    if start < 1 or end < start or end > duration:
                        return json.dumps({"error": f"Invalid week range: must be 1-{duration}"})
                except ValueError:
                    return json.dumps({"error": "Week range must contain integers"})
            else:
                start, end = 1, duration

            content_dir = project_path / "03_content_development"
            generated = []

            for week in range(start, end + 1):
                week_dir = content_dir / f"week_{week:02d}"
                week_dir.mkdir(exist_ok=True)
                generated.append({
                    "week": week,
                    "path": str(week_dir),
                    "status": "ready_for_generation"
                })

            # Update config status
            config["status"] = "content_generation"
            config["week_range"] = f"{start}-{end}"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "weeks_prepared": len(generated),
                "week_range": f"{start}-{end}",
                "parallel": parallel,
                "content_paths": generated
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def package_imscc(project_id: str, validate: bool = True) -> str:
        """
        Package course content into IMSCC format.

        Args:
            project_id: Project identifier
            validate: Run validation before packaging (default: True)

        Returns:
            IMSCC package path and validation report
        """
        try:
            project_path = validate_path_within_root(
                EXPORTS_PATH / project_id, EXPORTS_PATH
            )
            if not project_path.exists():
                return json.dumps({"error": f"Project not found: {project_id}"})

            # Load config
            config_path = project_path / "project_config.json"
            with open(config_path) as f:
                config = json.load(f)

            course_name = config.get("course_name", project_id)
            package_dir = project_path / "05_final_package"
            package_path = package_dir / f"{course_name}.imscc"

            # Create package directory if needed
            package_dir.mkdir(exist_ok=True)

            validation_results = []
            if validate:
                # Basic validation checks
                content_dir = project_path / "03_content_development"
                if content_dir.exists():
                    html_files = list(content_dir.rglob("*.html"))
                    validation_results.append({
                        "check": "html_files",
                        "count": len(html_files),
                        "passed": len(html_files) > 0
                    })

            # Update status
            config["status"] = "packaged"
            config["package_path"] = str(package_path)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Store IMSCC in LibV2 for downstream discovery
            libv2_package_path = None
            try:
                import shutil
                storage = LibV2Storage(project_id)
                libv2_dest = storage.get_package_output_path()
                libv2_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(package_path, libv2_dest)
                libv2_package_path = str(libv2_dest)
                logger.info("Stored IMSCC in LibV2: %s", libv2_dest)
            except Exception as e:
                logger.warning("Failed to store IMSCC in LibV2: %s", e)

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "package_path": str(package_path),
                "libv2_package_path": libv2_package_path,
                "validation": validation_results if validate else None,
                "status": "ready_for_packaging"
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def intake_imscc_package(
        imscc_path: str,
        output_dir: str,
        remediate: bool = True
    ) -> str:
        """
        Import and analyze an existing IMSCC package.

        Args:
            imscc_path: Path to IMSCC file
            output_dir: Extraction and analysis directory
            remediate: Automatically queue remediation (default: True)

        Returns:
            Analysis report with remediation queue
        """
        try:
            imscc = Path(imscc_path)
            if not imscc.exists():
                return json.dumps({"error": f"IMSCC not found: {imscc_path}"})

            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)

            # Extract package with Zip Slip protection
            extract_dir = output / "extracted"
            safe_extract_zip(imscc, extract_dir)

            # Analyze content
            analysis = {
                "source": str(imscc),
                "extracted_to": str(extract_dir),
                "manifest_found": (extract_dir / "imsmanifest.xml").exists(),
                "content_inventory": {},
                "remediation_queue": []
            }

            # Count content types
            for ext in [".html", ".pdf", ".docx", ".pptx", ".xml"]:
                files = list(extract_dir.rglob(f"*{ext}"))
                if files:
                    analysis["content_inventory"][ext] = len(files)
                    if ext in [".pdf", ".docx", ".pptx"] and remediate:
                        for f in files:
                            analysis["remediation_queue"].append({
                                "file": str(f.relative_to(extract_dir)),
                                "type": ext,
                                "action": "dart_conversion"
                            })

            return json.dumps({
                "success": True,
                "analysis": analysis,
                "remediation_queued": len(analysis["remediation_queue"])
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def remediate_course_content(
        project_id: str,
        remediation_types: Optional[str] = None
    ) -> str:
        """
        Execute remediation on analyzed course content.

        Args:
            project_id: Project identifier
            remediation_types: Comma-separated types or None for all
                             Options: dart_conversion, accessibility, quality, design

        Returns:
            Remediation report with before/after metrics
        """
        try:
            project_path = validate_path_within_root(
                EXPORTS_PATH / project_id, EXPORTS_PATH
            )
            if not project_path.exists():
                return json.dumps({"error": f"Project not found: {project_id}"})

            # Parse remediation types
            if remediation_types:
                types = [t.strip() for t in remediation_types.split(",")]
            else:
                types = ["dart_conversion", "accessibility", "quality", "design"]

            results = {
                "project_id": project_id,
                "remediation_types": types,
                "actions": []
            }

            for rtype in types:
                results["actions"].append({
                    "type": rtype,
                    "status": "queued",
                    "message": f"Remediation type '{rtype}' queued for execution"
                })

            return json.dumps({
                "success": True,
                "results": results
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_courseforge_status() -> str:
        """
        Get Courseforge installation status and active projects.

        Returns:
            Installation status and project list
        """
        try:
            status = {
                "installed": COURSEFORGE_PATH.exists(),
                "path": str(COURSEFORGE_PATH),
                "exports_path": str(EXPORTS_PATH),
                "agents_available": [],
                "active_projects": []
            }

            # Check for agents
            agents_dir = COURSEFORGE_PATH / "agents"
            if agents_dir.exists():
                status["agents_available"] = [
                    f.stem for f in agents_dir.glob("*.md")
                ]

            # List recent projects
            if EXPORTS_PATH.exists():
                projects = sorted(EXPORTS_PATH.iterdir(), reverse=True)[:10]
                for p in projects:
                    if p.is_dir():
                        config_path = p / "project_config.json"
                        if config_path.exists():
                            with open(config_path) as f:
                                config = json.load(f)
                            status["active_projects"].append({
                                "id": p.name,
                                "course": config.get("course_name"),
                                "status": config.get("status")
                            })

            return json.dumps(status)

        except Exception as e:
            return json.dumps({"error": str(e)})
