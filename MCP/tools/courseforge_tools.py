"""
Courseforge MCP Tools

Tools for course generation, IMSCC packaging, and remediation.
Integrates decision capture for training data collection.
"""

import json
import logging
import sys
import warnings
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
    try:
        from lib.decision_capture import DecisionCapture
        return DecisionCapture(course_code, phase, "courseforge")
    except ImportError:
        return None


def _log_decision(capture, decision_type: str, decision: str, rationale: str, **kwargs):
    """Log a decision using legacy log_decision()."""
    if capture is None:
        return

    capture.log_decision(decision_type, decision, rationale, **kwargs)


def _finalize_capture(capture):
    """Finalize a capture session."""
    if capture is None:
        return

    if hasattr(capture, 'save'):
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
        DEPRECATED (Wave 28e): Initialize a new course generation project.

        Post-Wave-24 the canonical course-initialization path runs
        through ``extract_textbook_structure`` +
        ``plan_course_structure`` (see ``MCP/core/executor.py``
        agent mappings for ``textbook-ingestor`` and
        ``course-outliner``). This tool remains a functional standalone
        project-initializer for external MCP clients, but new
        integrations should prefer the Wave 24 pair which produces
        ``textbook_structure.json`` + ``synthesized_objectives.json``
        in addition to the project scaffold.

        Args:
            course_name: Unique course identifier (e.g., "MTH_301")
            objectives_path: Path to exam objectives file
            duration_weeks: Course duration (default: 12)
            credit_hours: Credit hours (default: 3)

        Returns:
            Project workspace path and configuration
        """
        warnings.warn(
            "create_course_project is deprecated (Wave 28e). "
            "Prefer `extract_textbook_structure` + `plan_course_structure` "
            "(Wave 24) for new integrations.",
            DeprecationWarning,
            stacklevel=2,
        )
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
    async def package_imscc(
        project_id: str,
        validate: bool = True,
        objectives_path: Optional[str] = None,
        skip_validation: bool = False,
    ) -> str:
        """Build a real IMS Common Cartridge package from generated content.

        Wave 28e fold: delegates to the mature multi-file packager
        (``Courseforge.scripts.package_multifile_imscc.package_imscc``)
        rather than hand-rolling the ZIP. This mirrors the Wave 27
        registry-side fold at
        ``MCP/tools/pipeline_tools.py::_package_imscc`` so external
        MCP clients calling ``package_imscc`` directly now produce a
        real ``.imscc`` zip — pre-Wave-28e the tool flipped
        ``project_config.status`` and attempted a LibV2 copy without
        ever creating the zip.

        Consequences of the delegation:

        * Per-week ``learningObjectives`` validation runs by default
          (the mature packager refuses to build when any page's LO
          list references an out-of-week ID).
        * ``course_metadata.json`` is bundled at the zip root when
          present (the mature packager's Wave 3 REC-TAX-01 behavior).
        * Manifest uses IMS Common Cartridge v1.3 namespaces.
        * Resources are nested under per-week ``<item>`` wrappers in
          the organization tree.

        The legacy JSON envelope (``success``, ``package_path``,
        ``libv2_package_path``, ``html_modules``, ``package_size_bytes``)
        is preserved so callers see no contract change. LO-contract
        failure surfaces as ``{"success": false, "error": ...,
        "exit_code": 2}`` instead of silently falling through.

        Args:
            project_id: Project identifier from create_course_project
            validate: Retained for back-compat with the pre-fold kwarg.
                Has no effect on the mature packager's LO-contract
                validation; use ``skip_validation=True`` to bypass.
            objectives_path: Optional path to a canonical objectives
                JSON file the mature packager consults for LO-contract
                validation. Falls back to auto-discovery of
                ``content_dir/course.json``.
            skip_validation: When True, bypasses the mature packager's
                LO-contract validation. Default False.

        Returns:
            JSON envelope with ``success``, ``project_id``,
            ``package_path``, ``libv2_package_path``, ``html_modules``,
            and ``package_size_bytes`` on success; ``success: False``
            plus structured ``error`` + ``exit_code`` on LO-contract
            failure.
        """
        import sys as _sys

        try:
            project_path = validate_path_within_root(
                EXPORTS_PATH / project_id, EXPORTS_PATH
            )
            if not project_path.exists():
                return json.dumps({"error": f"Project not found: {project_id}"})

            content_dir = project_path / "03_content_development"
            final_dir = project_path / "05_final_package"
            final_dir.mkdir(parents=True, exist_ok=True)

            # Sanity: require the content dir + at least one HTML page.
            html_files = sorted(content_dir.rglob("*.html")) if content_dir.exists() else []
            if not html_files:
                return json.dumps({
                    "error": "No HTML modules found in content directory",
                    "content_dir": str(content_dir),
                })

            # Load project config for course_name + course_title.
            config_path = project_path / "project_config.json"
            course_name = project_id
            course_title = project_id
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                    course_name = cfg.get("course_name", project_id)
                    course_title = (
                        cfg.get("course_title")
                        or cfg.get("title")
                        or course_name
                    )
                except (OSError, json.JSONDecodeError):
                    pass

            package_path = final_dir / f"{course_name}.imscc"

            # Import the mature packager. The module lives under
            # ``Courseforge/scripts/`` (no ``__init__.py``) so we
            # prepend the directory to ``sys.path`` before importing.
            cf_scripts = (
                Path(__file__).resolve().parents[2]
                / "Courseforge" / "scripts"
            )
            if str(cf_scripts) not in _sys.path:
                _sys.path.insert(0, str(cf_scripts))
            try:
                import package_multifile_imscc as _pkg_mod  # noqa: E402
            except ImportError as exc:
                return json.dumps({
                    "success": False,
                    "error": f"Failed to import mature packager: {exc}",
                    "project_id": project_id,
                })

            # Resolve optional objectives path.
            objectives_path_obj = (
                Path(objectives_path) if objectives_path else None
            )

            # Call the (synchronous) mature packager. SystemExit raised
            # on LO-contract failure is converted into a structured
            # error response; any other exception is surfaced the same
            # way so the caller sees a normal JSON envelope.
            try:
                _pkg_mod.package_imscc(
                    content_dir,
                    package_path,
                    course_name,
                    course_title,
                    objectives_path=objectives_path_obj,
                    skip_validation=bool(skip_validation),
                )
            except SystemExit as exc:
                return json.dumps({
                    "success": False,
                    "error": (
                        "IMSCC packaging refused: per-week LO contract "
                        "validation failed. See logs for per-page details."
                    ),
                    "exit_code": (
                        exc.code if isinstance(exc.code, int) else 2
                    ),
                    "project_id": project_id,
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Mature packager raised for project %s: %s",
                    project_id, exc,
                )
                return json.dumps({
                    "success": False,
                    "error": f"Mature packager failed: {exc}",
                    "project_id": project_id,
                })

            # Update project status post-success.
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                    cfg["status"] = "packaged"
                    cfg["package_path"] = str(package_path)
                    with open(config_path, 'w') as f:
                        json.dump(cfg, f, indent=2)
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning(
                        "Failed to update project_config post-package: %s", e
                    )

            return json.dumps({
                "success": True,
                "project_id": project_id,
                "package_path": str(package_path),
                "libv2_package_path": str(package_path),
                "html_modules": len(html_files),
                "package_size_bytes": package_path.stat().st_size,
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
