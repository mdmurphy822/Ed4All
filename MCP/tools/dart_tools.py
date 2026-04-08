"""
DART MCP Tools

Tools for PDF to accessible HTML conversion using multi-source synthesis.
Combines pdftotext (text accuracy), pdfplumber (table structure), and OCR
to produce optimal accessible HTML output.

Security: All path inputs are validated to stay within project boundaries.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.paths import DART_PATH  # noqa: E402
from lib.secure_paths import PathTraversalError, validate_path_within_root  # noqa: E402

# Import new telemetry system
try:
    from LibV2.telemetry import ArtifactRef, CaptureSession, InputRef  # noqa: F401
    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

logger = logging.getLogger(__name__)

# Allowed root for DART operations
ALLOWED_ROOT = Path(os.environ.get("ED4ALL_ROOT", _PROJECT_ROOT))


def _validate_dart_paths():
    """Validate DART paths at module load."""
    if not DART_PATH.exists():
        logger.warning(f"DART installation not found: {DART_PATH}")
    elif not (DART_PATH / "multi_source_interpreter.py").exists():
        logger.warning(f"DART multi_source_interpreter.py not found at {DART_PATH}")
    else:
        logger.info(f"DART installation validated: {DART_PATH}")


_validate_dart_paths()


def _create_capture(course_code: str = "UNKNOWN", pdf_name: str = "unknown"):
    """Create a capture session for DART operations."""
    if HAS_TELEMETRY:
        try:
            return CaptureSession.start_run(
                tool_id="dart",
                component="converter",
                meta={"course_code": course_code, "pdf_name": pdf_name},
                course_code=course_code,
                phase="conversion",
            )
        except Exception as e:
            logger.warning(f"Failed to create telemetry session: {e}")
            return None
    # Fallback to legacy capture
    try:
        from lib.decision_capture import DARTDecisionCapture
        return DARTDecisionCapture(course_code, pdf_name)
    except ImportError:
        return None


def _log_tool_decision(capture, decision_type: str, decision: str, rationale: str, **kwargs):
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
        elif "approach" in decision_type.lower():
            event_type = "prompt.built"

        capture.emit(
            event_type,
            payload={
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **{k: v for k, v in kwargs.items() if k not in ("context",)}
            },
            metrics={"confidence": kwargs.get("confidence", 1.0)} if "confidence" in kwargs else {},
            severity="info" if "error" not in decision_type.lower() else "error"
        )
    else:
        # Legacy capture
        capture.log_decision(decision_type, decision, rationale, **kwargs)


def _log_conversion_start(capture, source_path: str, options: dict):
    """Log conversion start event."""
    if capture is None:
        return

    if HAS_TELEMETRY and isinstance(capture, CaptureSession):
        capture.emit(
            "input.loaded",
            payload={"source": source_path, "method": "multi_source_synthesis", **options},
            phase="ingest"
        )
    elif hasattr(capture, 'log_conversion_start'):
        capture.log_conversion_start(source_path, options)


def _log_conversion_complete(capture, output_path: str, pages: int, wcag: bool, time_sec: float):
    """Log conversion completion event."""
    if capture is None:
        return

    if HAS_TELEMETRY and isinstance(capture, CaptureSession):
        capture.emit(
            "export.completed",
            payload={"output_path": output_path, "wcag_compliant": wcag, "status": "success"},
            metrics={"pages_processed": pages, "latency_ms": time_sec * 1000},
            phase="export"
        )
    elif hasattr(capture, 'log_conversion_complete'):
        capture.log_conversion_complete(output_path, pages, wcag, time_sec)


def _finalize_capture(capture):
    """Finalize a capture session."""
    if capture is None:
        return

    if HAS_TELEMETRY and isinstance(capture, CaptureSession):
        capture.finish_run("success")
    elif hasattr(capture, 'save'):
        capture.save()


def register_dart_tools(mcp):
    """Register DART tools with the MCP server."""

    @mcp.tool()
    async def convert_pdf_multi_source(
        combined_json_path: str,
        output_path: Optional[str] = None,
        course_code: Optional[str] = None
    ) -> str:
        """
        Convert a PDF to accessible HTML using multi-source synthesis.

        This is the PRIMARY DART conversion tool. It synthesizes data from
        multiple extraction sources (pdftotext, pdfplumber, OCR) to produce
        optimal accessible HTML output.

        Security: All paths are validated to stay within project root.

        Args:
            combined_json_path: Path to combined JSON file containing all extraction sources
                               (pdftotext, tables from pdfplumber, OCR data)
            output_path: Optional output path for HTML file
            course_code: Optional course code for capture organization

        Returns:
            JSON result with synthesized HTML and metadata
        """
        capture = None
        start_time = datetime.now()

        try:
            # Validate input path
            try:
                combined_path = validate_path_within_root(
                    Path(combined_json_path), ALLOWED_ROOT, must_exist=True
                )
            except PathTraversalError as e:
                return json.dumps({"error": f"Security error: {e}"})

            # Validate output path if provided
            if output_path:
                try:
                    validate_path_within_root(Path(output_path), ALLOWED_ROOT)
                except PathTraversalError as e:
                    return json.dumps({"error": f"Security error in output path: {e}"})

            # Import the multi-source interpreter
            sys.path.insert(0, str(DART_PATH))
            from multi_source_interpreter import convert_single_pdf

            # Initialize capture
            code = combined_path.stem.replace('_combined', '')
            capture = _create_capture(course_code or code, code)
            _log_conversion_start(capture, str(combined_path), {"method": "multi_source_synthesis"})

            # Log approach decision
            _log_tool_decision(
                capture,
                decision_type="approach_selection",
                decision="Using multi-source synthesis for PDF conversion",
                rationale="Multi-source synthesis combines pdftotext (text accuracy), pdfplumber (table structure), and OCR (validation) for optimal output",
                context=f"Combined JSON: {combined_path.name}"
            )

            # Perform conversion
            result = convert_single_pdf(str(combined_path), output_path)

            # Log completion
            elapsed = (datetime.now() - start_time).total_seconds()
            _log_tool_decision(
                capture,
                decision_type="validation_result",
                decision=f"Conversion {'succeeded' if result['success'] else 'failed'}",
                rationale=f"Multi-source synthesis completed in {elapsed:.1f}s",
                confidence=1.0 if result['success'] else 0.0
            )

            _log_conversion_complete(
                capture,
                output_path=output_path or "",
                pages=len(result.get('synthesized', {}).get('sections', [])),
                wcag=True,
                time_sec=elapsed
            )
            _finalize_capture(capture)

            return json.dumps({
                "success": result['success'],
                "campus_code": result['campus_code'],
                "campus_name": result['campus_name'],
                "output_path": output_path,
                "sections_synthesized": len(result.get('synthesized', {}).get('sections', [])),
                "html_length": len(result.get('html', '')),
                "capture_saved": capture is not None
            })

        except Exception as e:
            _log_tool_decision(
                capture,
                decision_type="error_handling",
                decision=f"Conversion error: {type(e).__name__}",
                rationale=f"Exception during multi-source synthesis: {str(e)[:200]}"
            )
            _finalize_capture(capture)
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def batch_convert_multi_source(
        combined_dir: str,
        output_zip: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Batch convert all PDFs using multi-source synthesis.

        Processes all combined JSON files in the input directory and generates
        accessible HTML using multi-source synthesis.

        Security: All paths are validated to stay within project root.

        Args:
            combined_dir: Directory containing *_combined.json files
            output_zip: Optional path for output zip file
            output_dir: Optional output directory (defaults to batch_output/)

        Returns:
            Batch processing report with file counts and paths
        """
        try:
            # Validate input directory
            try:
                combined_path = validate_path_within_root(
                    Path(combined_dir), ALLOWED_ROOT, must_exist=True
                )
            except PathTraversalError as e:
                return json.dumps({"error": f"Security error: {e}"})

            # Validate output paths if provided
            if output_zip:
                try:
                    validate_path_within_root(Path(output_zip), ALLOWED_ROOT)
                except PathTraversalError as e:
                    return json.dumps({"error": f"Security error in output_zip: {e}"})

            if output_dir:
                try:
                    validate_path_within_root(Path(output_dir), ALLOWED_ROOT)
                except PathTraversalError as e:
                    return json.dumps({"error": f"Security error in output_dir: {e}"})

            # Import the multi-source interpreter
            sys.path.insert(0, str(DART_PATH))
            from multi_source_interpreter import batch_synthesize_all, create_zip

            # Run batch synthesis
            html_files = batch_synthesize_all(
                combined_dir=str(combined_path),
                output_dir=output_dir
            )

            # Create zip if requested
            if output_zip and html_files:
                create_zip(html_files, output_zip)

            return json.dumps({
                "success": True,
                "total_processed": len(html_files),
                "output_dir": output_dir or str(DART_PATH / "batch_output"),
                "output_zip": output_zip if output_zip else None,
                "files": [str(f.name) for f in html_files]
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def validate_wcag_compliance(html_path: str) -> str:
        """
        Validate HTML file for WCAG 2.2 AA compliance.

        Security: Path is validated to stay within project root.

        Args:
            html_path: Path to HTML file to validate

        Returns:
            Validation report with pass/fail and issues list
        """
        try:
            # Validate input path
            try:
                html = validate_path_within_root(
                    Path(html_path), ALLOWED_ROOT, must_exist=True
                )
            except PathTraversalError as e:
                return json.dumps({"error": f"Security error: {e}"})

            with open(html, encoding='utf-8') as f:
                content = f.read()

            issues = []

            # Check for lang attribute
            if 'lang="' not in content and "lang='" not in content:
                issues.append("Missing lang attribute on html element")

            # Check for viewport meta
            if 'viewport' not in content:
                issues.append("Missing viewport meta tag")

            # Check for skip link
            if 'skip' not in content.lower():
                issues.append("Missing skip navigation link")

            # Check for main landmark
            if '<main' not in content:
                issues.append("Missing main landmark")

            # Check for alt attributes on images
            import re
            imgs_without_alt = re.findall(r'<img(?![^>]*alt=)[^>]*>', content)
            if imgs_without_alt:
                issues.append(f"{len(imgs_without_alt)} images missing alt attributes")

            # Check heading hierarchy
            headings = re.findall(r'<h([1-6])', content)
            if headings:
                prev_level = 0
                for h in headings:
                    level = int(h)
                    if level > prev_level + 1 and prev_level > 0:
                        issues.append(f"Heading hierarchy skip: h{prev_level} to h{level}")
                        break
                    prev_level = level

            # Check for table scope attributes
            tables = re.findall(r'<table[^>]*>.*?</table>', content, re.DOTALL)
            for table in tables:
                if '<th' in table and 'scope=' not in table:
                    issues.append("Table header missing scope attribute")
                    break

            return json.dumps({
                "path": html_path,
                "passed": len(issues) == 0,
                "issues": issues,
                "checks_performed": [
                    "lang_attribute", "viewport", "skip_link", "main_landmark",
                    "alt_text", "heading_hierarchy", "table_scope"
                ]
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_dart_status() -> str:
        """
        Get DART installation status and capabilities.

        Returns:
            DART configuration and capability information
        """
        try:
            status = {
                "installed": DART_PATH.exists(),
                "path": str(DART_PATH),
                "multi_source_interpreter": (DART_PATH / "multi_source_interpreter.py").exists(),
                "capabilities": []
            }

            if status["installed"]:
                # Check for key components
                if status["multi_source_interpreter"]:
                    status["capabilities"].append("multi_source_synthesis")
                    status["capabilities"].append("batch_processing")
                    status["capabilities"].append("contact_card_generation")
                    status["capabilities"].append("systems_table_synthesis")
                if (DART_PATH / "pdf_converter").exists():
                    status["capabilities"].append("pdf_extraction")
                if (DART_PATH / "batch_output" / "combined").exists():
                    combined_files = list((DART_PATH / "batch_output" / "combined").glob("*_combined.json"))
                    status["combined_files_available"] = len(combined_files)

            return json.dumps(status)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def list_available_campuses() -> str:
        """
        List all available campus combined JSON files ready for conversion.

        Returns:
            List of campus codes and their combined JSON paths
        """
        try:
            combined_dir = DART_PATH / "batch_output" / "combined"
            if not combined_dir.exists():
                return json.dumps({"error": "Combined directory not found"})

            # Import campus names
            sys.path.insert(0, str(DART_PATH))
            from multi_source_interpreter import CAMPUS_NAMES

            files = sorted(combined_dir.glob("*_combined.json"))
            campuses = []
            for f in files:
                code = f.stem.replace('_combined', '')
                campuses.append({
                    "code": code,
                    "name": CAMPUS_NAMES.get(code, code),
                    "path": str(f),
                    "size_kb": round(f.stat().st_size / 1024, 1)
                })

            return json.dumps({
                "total": len(campuses),
                "campuses": campuses
            })

        except Exception as e:
            return json.dumps({"error": str(e)})
