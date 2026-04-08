#!/usr/bin/env python3
"""
Decision Capture Integration for DART PDF Converter

Wraps the PDF converter to capture all Claude decisions during conversion.
Integrates with the Ed4All training-captures system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add Ed4All lib to path for capture imports
ED4ALL_ROOT = Path(__file__).resolve().parents[2]  # DART/pdf_converter/capture_integration.py → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

from lib.decision_capture import DARTDecisionCapture, InputRef  # noqa: E402

from .converter import ConversionResult, PDFToAccessibleHTML  # noqa: E402

logger = logging.getLogger(__name__)


class CaptureEnabledConverter(PDFToAccessibleHTML):
    """
    PDF converter with decision capture integration.

    All conversion decisions are logged to training-captures/dart/
    for Claude training data collection.
    """

    def __init__(
        self,
        course_code: str = "UNKNOWN",
        capture_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize converter with capture support.

        Args:
            course_code: Course code for capture organization (e.g., "MTH_101")
            capture_enabled: Whether to enable decision capture
            **kwargs: Arguments passed to PDFToAccessibleHTML
        """
        super().__init__(**kwargs)
        self.course_code = course_code
        self.capture_enabled = capture_enabled
        self._capture: Optional[DARTDecisionCapture] = None

    def convert(self, pdf_path: str, output_dir: str = None) -> ConversionResult:
        """
        Convert PDF to WCAG-compliant HTML with decision capture.

        All decisions made during conversion are logged for training.
        """
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem

        # Initialize capture if enabled
        if self.capture_enabled:
            self._capture = DARTDecisionCapture(self.course_code, pdf_name)
            self._capture.log_conversion_start(
                str(pdf_path),
                {
                    "dpi": self.dpi,
                    "lang": self.lang,
                    "enable_math": self.enable_math,
                    "extract_images": self.extract_images,
                    "use_ai_alt_text": self.use_ai_alt_text,
                    "validate_wcag": self.validate_wcag,
                }
            )

        try:
            # Perform conversion
            result = super().convert(str(pdf_path), output_dir)

            # Log decisions based on conversion result
            if self._capture:
                self._log_conversion_decisions(pdf_path, result)

            return result

        except Exception as e:
            # Log error decision
            if self._capture:
                self._capture.log_decision(
                    decision_type="error_handling",
                    decision=f"Conversion failed: {type(e).__name__}",
                    rationale=f"Exception during PDF processing: {str(e)[:200]}",
                    context=str(pdf_path)
                )
            raise

        finally:
            # Save capture
            if self._capture:
                try:
                    self._capture.save()
                except Exception as e:
                    logger.warning(f"Failed to save decision capture: {e}")

    def _log_conversion_decisions(
        self,
        pdf_path: Path,
        result: ConversionResult
    ):
        """Log decisions made during conversion."""
        if not self._capture:
            return

        # Log extraction method decision
        self._capture.log_decision(
            decision_type="approach_selection",
            decision="Selected text extraction method based on PDF content",
            rationale="pdftotext is used for born-digital PDFs as it preserves layout and reading order. Falls back to OCR if minimal text is extracted.",
            inputs_ref=[
                InputRef(
                    source_type="pdf",
                    path_or_id=str(pdf_path),
                    content_hash=""
                )
            ],
            context=f"Pages processed: {result.pages_processed}, Words extracted: {result.total_words}"
        )

        # Log image extraction decision if images were found
        if result.images_extracted > 0:
            self._capture.log_decision(
                decision_type="accessibility_measures",
                decision=f"Extracted {result.images_extracted} images from PDF",
                rationale=f"Images extracted for accessibility. Generated alt text for {result.images_with_alt_text} images using AI assistance.",
                context=f"Alt text coverage: {result.images_with_alt_text}/{result.images_extracted}"
            )

        # Log WCAG validation decision
        if self.validate_wcag:
            if result.wcag_compliant:
                self._capture.log_decision(
                    decision_type="validation_result",
                    decision="WCAG 2.2 AA validation passed",
                    rationale="Output HTML meets WCAG 2.2 AA accessibility standards without critical or blocking issues.",
                    confidence=1.0
                )
            else:
                self._capture.log_decision(
                    decision_type="validation_result",
                    decision=f"WCAG validation found {result.wcag_issues_count} issues ({result.wcag_critical_count} critical)",
                    rationale="Some WCAG 2.2 AA criteria were not fully met. Review and remediation may be needed for full compliance.",
                    confidence=0.7
                )

        # Log completion
        self._capture.log_conversion_complete(
            output_path=result.html_path,
            pages_processed=result.pages_processed,
            wcag_compliant=result.wcag_compliant,
            processing_time_seconds=0  # TODO: Add timing
        )

    def log_structure_decision(
        self,
        page_range: str,
        detected_structure: str,
        applied_headings: list
    ):
        """
        Log a document structure detection decision.

        Call this when making decisions about document structure.
        """
        if self._capture:
            self._capture.log_structure_decision(
                page_range, detected_structure, applied_headings
            )

    def log_alt_text_decision(
        self,
        image_id: str,
        generated_alt_text: str,
        method: str = "claude"
    ):
        """
        Log an alt text generation decision.

        Call this when generating alt text for images.
        """
        if self._capture:
            self._capture.log_alt_text_decision(
                image_id, generated_alt_text, method
            )

    def log_math_decision(
        self,
        expression_id: str,
        original_text: str,
        mathml_output: str
    ):
        """
        Log a math conversion decision.

        Call this when converting math expressions to MathML.
        """
        if self._capture:
            self._capture.log_math_decision(
                expression_id, original_text, mathml_output
            )

    def log_custom_decision(
        self,
        decision_type: str,
        decision: str,
        rationale: str,
        **kwargs
    ):
        """
        Log a custom decision during conversion.

        Use this for any conversion decision not covered by specialized methods.
        """
        if self._capture:
            self._capture.log_decision(
                decision_type=decision_type,
                decision=decision,
                rationale=rationale,
                **kwargs
            )


def convert_with_capture(
    pdf_path: str,
    output_dir: str = None,
    course_code: str = "UNKNOWN",
    **kwargs
) -> ConversionResult:
    """
    Convert PDF to accessible HTML with decision capture.

    This is the main entry point for capture-enabled conversion.

    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory for output HTML file
        course_code: Course code for capture organization
        **kwargs: Additional arguments for PDFToAccessibleHTML

    Returns:
        ConversionResult with conversion status and output path
    """
    converter = CaptureEnabledConverter(
        course_code=course_code,
        capture_enabled=True,
        **kwargs
    )
    return converter.convert(pdf_path, output_dir)


# Convenience exports
__all__ = [
    'CaptureEnabledConverter',
    'convert_with_capture',
]
