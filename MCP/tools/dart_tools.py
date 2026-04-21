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


# Wave 22 DC4: canonical course_code pattern from
# schemas/events/decision_event.schema.json. The regex is strict
# intentionally — downstream consumers (Brightspace packager, LibV2
# archive layout, run-ID derivation) rely on the `{PREFIX}_{NNN}`
# shape. Any PDF-derived name that doesn't already match is normalised
# to a deterministic in-pattern value via ``normalize_course_code``.
import hashlib as _hashlib
import re as _re

_COURSE_CODE_PATTERN = _re.compile(r"^[A-Z]{2,8}_[0-9]{3}$")


def normalize_course_code(raw: str) -> str:
    """Coerce a PDF-derived course code into the canonical schema pattern.

    The canonical pattern is ``^[A-Z]{2,8}_[0-9]{3}$`` (2-8 uppercase
    letters, underscore, 3 digits). PDF filenames like ``"Ed4All"`` or
    ``"bates_teaching_digital_age"`` or ``"arxiv-2301.12345"`` do not
    match out of the box, so pre-Wave-22 every DART capture carried a
    ``course_id`` validation issue (556/1134 records on a recent run).

    Normalisation strategy:

    1. Uppercase + replace any non-alphanumeric with underscore.
    2. Strip leading/trailing underscores + collapse repeats.
    3. If the result already matches the pattern, return as-is.
    4. Otherwise, split on ``_`` and use the first purely-alphabetic
       chunk (truncated to 8 chars) as the prefix. If no alphabetic
       chunk exists, use ``"PDF"`` as the fallback prefix.
    5. Derive a deterministic 3-digit numeric suffix from the full raw
       name via SHA-256 modulo 1000 so the same PDF always produces the
       same course code. Zero-pad to 3 digits.

    Examples
    --------
    >>> normalize_course_code("Ed4All")
    'ED_...'  # 3-digit hash suffix, deterministic
    >>> normalize_course_code("MTH_101")
    'MTH_101'  # already canonical
    """
    raw = (raw or "").strip()
    if not raw:
        raw = "unknown"

    # Phase 1: aggressive uppercase + underscore normalisation.
    uppered = _re.sub(r"[^A-Za-z0-9]+", "_", raw).upper().strip("_")
    uppered = _re.sub(r"_+", "_", uppered)

    # Phase 2: early exit when already canonical.
    if _COURSE_CODE_PATTERN.match(uppered):
        return uppered

    # Phase 3: pick a prefix from the first alphabetic chunk (≥2 chars).
    chunks = [c for c in uppered.split("_") if c]
    prefix = ""
    for chunk in chunks:
        alpha_only = _re.sub(r"[^A-Z]", "", chunk)
        if len(alpha_only) >= 2:
            prefix = alpha_only[:8]
            break
    if not prefix:
        # No alphabetic chunk — fall back to a constant so the suffix
        # carries all the disambiguating signal.
        prefix = "PDF"
    # Enforce 2-8 length window post-hoc (safety net against single-char
    # alpha-only chunks sneaking through the len check above).
    if len(prefix) < 2:
        prefix = (prefix + "PDF")[:2]
    prefix = prefix[:8]

    # Phase 4: deterministic 3-digit suffix from a content hash.
    suffix_int = int(_hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % 1000
    suffix = f"{suffix_int:03d}"

    candidate = f"{prefix}_{suffix}"
    if not _COURSE_CODE_PATTERN.match(candidate):
        # Defensive: prefix may still be invalid (e.g. all digits after
        # filtering). Drop back to the constant-prefix form.
        candidate = f"PDF_{suffix}"
    return candidate


def _create_capture(course_code: str = "UNKNOWN", pdf_name: str = "unknown"):
    """Create a capture session for DART operations.

    Wave 22 DC4: ``course_code`` is normalised to the canonical
    ``^[A-Z]{2,8}_[0-9]{3}$`` pattern so DART captures stop carrying
    the ``course_id`` validation issue downstream consumers have been
    papering over.
    """
    try:
        from lib.decision_capture import DARTDecisionCapture
        return DARTDecisionCapture(normalize_course_code(course_code), pdf_name)
    except ImportError:
        return None


def _log_tool_decision(capture, decision_type: str, decision: str, rationale: str, **kwargs):
    """Log a decision using legacy log_decision()."""
    if capture is None:
        return

    capture.log_decision(decision_type, decision, rationale, **kwargs)


def _log_conversion_start(capture, source_path: str, options: dict):
    """Log conversion start event."""
    if capture is None:
        return

    if hasattr(capture, 'log_conversion_start'):
        capture.log_conversion_start(source_path, options)


def _log_conversion_complete(capture, output_path: str, pages: int, wcag: bool, time_sec: float):
    """Log conversion completion event."""
    if capture is None:
        return

    if hasattr(capture, 'log_conversion_complete'):
        capture.log_conversion_complete(output_path, pages, wcag, time_sec)


def _finalize_capture(capture):
    """Finalize a capture session."""
    if capture is None:
        return

    if hasattr(capture, 'save'):
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
                return json.dumps({
                    "error": "No combined DART outputs found. Run DART batch processing first.",
                    "expected_path": str(combined_dir),
                    "hint": "Use batch_convert_documents or convert_pdf_multi_source to generate combined outputs"
                })

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

    @mcp.tool()
    async def extract_and_convert_pdf(
        pdf_path: str,
        course_code: Optional[str] = None,
        output_dir: Optional[str] = None,
        figures_dir: Optional[str] = None,
    ) -> str:
        """
        Full DART pipeline: extract sources from PDF and convert to accessible HTML.

        Wave 22 F2 fix: folded this variant to route through the
        Wave-15+ ``_raw_text_to_accessible_html`` entry point (the
        same path the pipeline-registry variant at
        ``MCP/tools/pipeline_tools.py::_extract_and_convert_pdf``
        already uses). Pre-Wave-22 this tool routed through the
        legacy ``PDFToAccessibleHTML`` converter as its Strategy-2
        fallback, ignored ``figures_dir``, emitted no Wave-19
        sidecars, and routinely failed the ``dart_markers`` gate.

        The legacy converter is retained strictly behind
        ``DART_LEGACY_CONVERTER=true`` for one-release rollback
        safety.

        Args:
            pdf_path: Path to the PDF file to convert
            course_code: Optional course code for output organization
            output_dir: Optional output directory (defaults to DART/output/)
            figures_dir: Optional directory for persisted figure images
                (Wave 17). When unset and the Wave-16 dual-extraction path
                is taken, the pipeline auto-derives a sibling
                ``{stem}_figures/`` directory next to the output HTML so
                ``<img src>`` references stay portable.

        Returns:
            JSON with output_path (HTML), success status, and metadata
        """
        start_time = datetime.now()

        try:
            # Validate input path
            try:
                pdf = validate_path_within_root(
                    Path(pdf_path), ALLOWED_ROOT, must_exist=True
                )
            except PathTraversalError as e:
                return json.dumps({"error": f"Security error: {e}"})

            if pdf.suffix.lower() != ".pdf":
                return json.dumps({"error": f"Not a PDF file: {pdf_path}"})

            # Set up output directory
            out_dir = Path(output_dir) if output_dir else DART_PATH / "output"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Validate figures_dir path if provided
            if figures_dir:
                try:
                    validate_path_within_root(Path(figures_dir), ALLOWED_ROOT)
                except PathTraversalError as e:
                    return json.dumps(
                        {"error": f"Security error in figures_dir: {e}"}
                    )

            # Import DART modules
            sys.path.insert(0, str(DART_PATH))
            code = course_code or pdf.stem
            out_stem = pdf.stem

            # Strategy 1: If a pre-extracted combined JSON exists, use
            # multi-source synthesis (legacy Wave-8 path — source of
            # truth for the older batch_output workflow). Output
            # filename is keyed on the PDF basename to match the
            # pipeline-registry variant.
            combined_dir = DART_PATH / "batch_output" / "combined"
            combined_json_path = combined_dir / f"{code}_combined.json"

            if combined_json_path.exists():
                try:
                    from multi_source_interpreter import convert_single_pdf
                    html_output = out_dir / f"{out_stem}_synthesized.html"
                    result = convert_single_pdf(
                        str(combined_json_path), str(html_output)
                    )

                    elapsed = (datetime.now() - start_time).total_seconds()
                    return json.dumps({
                        "success": True,
                        "output_path": str(html_output),
                        "combined_json_path": str(combined_json_path),
                        "method": "multi_source_synthesis",
                        "campus_code": code,
                        "elapsed_seconds": round(elapsed, 2),
                        "html_length": (
                            result.get("html_length", 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                    })
                except ImportError:
                    pass  # Fall through to Strategy 2

            # Strategy 2: Extract via pdftotext and route through the
            # Wave-15+ ``_raw_text_to_accessible_html`` entry point. This
            # is the same path used by
            # ``MCP/tools/pipeline_tools.py::_extract_and_convert_pdf``
            # so both surfaces produce dart_markers-compliant HTML and
            # emit Wave-19 sidecars.
            import re as _re
            import subprocess

            raw_text = ""
            pdftotext_ok = False
            try:
                proc = subprocess.run(
                    ["pdftotext", "-layout", str(pdf), "-"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                raw_text = proc.stdout
                pdftotext_ok = bool(raw_text.strip())
            except (subprocess.SubprocessError, FileNotFoundError):
                pdftotext_ok = False

            # Legacy fallback: only when pdftotext is unavailable AND
            # the opt-in DART_LEGACY_CONVERTER flag is set. New runs
            # default to the Wave-15+ path.
            legacy_flag = os.environ.get(
                "DART_LEGACY_CONVERTER", ""
            ).strip().lower() == "true"
            if not pdftotext_ok and legacy_flag:
                try:
                    from pdf_converter.converter import PDFToAccessibleHTML

                    converter = PDFToAccessibleHTML()
                    legacy_result = converter.convert(str(pdf), str(out_dir))
                    elapsed = (datetime.now() - start_time).total_seconds()
                    return json.dumps({
                        "success": legacy_result.success,
                        "output_path": legacy_result.html_path,
                        "method": "pdf_converter_legacy",
                        "pages_processed": legacy_result.pages_processed,
                        "elapsed_seconds": round(elapsed, 2),
                        "error": (
                            legacy_result.error
                            if not legacy_result.success
                            else None
                        ),
                    })
                except ImportError:
                    return json.dumps({
                        "error": (
                            "DART modules not available. "
                            "pdftotext is unavailable and the legacy "
                            "pdf_converter could not be imported."
                        ),
                    })

            if not pdftotext_ok:
                return json.dumps({
                    "error": (
                        "pdftotext unavailable or returned empty text; "
                        "set DART_LEGACY_CONVERTER=true to opt into the "
                        "pre-Wave-15 pdf_converter path."
                    ),
                })

            # Wave-15+ path — same as the pipeline registry variant so
            # the MCP-tool surface and the registry surface stay in
            # parity (F2 audit requirement).
            from MCP.tools.pipeline_tools import (
                _raw_text_to_accessible_html,
            )

            pretty_title = (
                out_stem.replace("-", " ").replace("_", " ").strip()
            )
            html_output = out_dir / f"{out_stem}_accessible.html"
            html_content = _raw_text_to_accessible_html(
                raw_text,
                pretty_title,
                source_pdf=str(pdf),
                output_path=str(html_output),
                figures_dir=figures_dir,
            )
            html_output.write_text(html_content, encoding="utf-8")

            word_count = len(_re.findall(r"\b\w+\b", html_content))
            elapsed = (datetime.now() - start_time).total_seconds()

            return json.dumps({
                "success": True,
                "output_path": str(html_output),
                "method": "pdftotext_to_html",
                "word_count": word_count,
                "html_length": len(html_content),
                "elapsed_seconds": round(elapsed, 2),
                "campus_code": code,
            })

        except Exception as e:
            logger.error(f"extract_and_convert_pdf failed: {e}")
            return json.dumps({"error": str(e)})
