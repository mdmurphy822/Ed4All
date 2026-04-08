"""
PDF Accessibility Converter

A standalone toolkit for converting PDF documents to WCAG 2.2 AA compliant HTML.

Features:
- pdftotext extraction for born-digital PDFs
- OCR fallback for scanned documents
- Claude Code integration for intelligent text ordering and structure detection
- Automatic structure detection (headings, paragraphs, tables)
- MathML conversion for mathematical expressions (LaTeX, Unicode)
- Image extraction with AI-generated alt text
- WCAG 2.2 AA compliant output with:
  - Skip links for keyboard navigation
  - ARIA landmarks
  - Semantic HTML structure
  - Dark mode support
  - Responsive design
  - Print styles

Workflow:
1. Run: python convert.py document.pdf
2. Text is extracted and saved to output/document_extracted.txt
3. In Claude Code, say: "Review the extracted text and generate accessible HTML"
4. Claude Code generates gold-standard WCAG-compliant HTML
"""

from .converter import (
    PDFToAccessibleHTML,
    ConversionResult,
    TextBlock,
)

from .wcag_enhancer import (
    WCAGHTMLEnhancer,
    WCAGOptions,
    enhance_html_wcag,
    enhance_html_file,
)

from .wcag_validator import (
    WCAGValidator,
    ValidationReport,
    WCAGIssue,
    IssueSeverity,
    WCAGCriterion,
    validate_html_wcag,
    validate_html_file,
)

# Optional imports for math and image processing
try:
    from .math_processor import MathDetector, MathMLConverter, MathBlock
except ImportError:
    MathDetector = None
    MathMLConverter = None
    MathBlock = None

try:
    from .image_extractor import PDFImageExtractor, ImageProcessor, ExtractedImage
except ImportError:
    PDFImageExtractor = None
    ImageProcessor = None
    ExtractedImage = None

try:
    from .alt_text_generator import AltTextGenerator, AltTextResult
except ImportError:
    AltTextGenerator = None
    AltTextResult = None

try:
    from .embed_images import embed_images, load_metadata, create_figure_element
except ImportError:
    embed_images = None
    load_metadata = None
    create_figure_element = None

__version__ = '1.1.0'  # WCAG 2.2 AA update
__all__ = [
    # Core
    'PDFToAccessibleHTML',
    'ConversionResult',
    'TextBlock',
    # WCAG Enhancement
    'WCAGHTMLEnhancer',
    'WCAGOptions',
    'enhance_html_wcag',
    'enhance_html_file',
    # WCAG Validation
    'WCAGValidator',
    'ValidationReport',
    'WCAGIssue',
    'IssueSeverity',
    'WCAGCriterion',
    'validate_html_wcag',
    'validate_html_file',
    # Math processing (optional)
    'MathDetector',
    'MathMLConverter',
    'MathBlock',
    # Image processing (optional)
    'PDFImageExtractor',
    'ImageProcessor',
    'ExtractedImage',
    'AltTextGenerator',
    'AltTextResult',
    # Image embedding (optional)
    'embed_images',
    'load_metadata',
    'create_figure_element',
]


def convert(pdf_path: str, output_dir: str = None) -> ConversionResult:
    """
    Convert a PDF to WCAG-compliant accessible HTML.

    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory for output (default: ./output/)

    Returns:
        ConversionResult with success status and output path

    Example:
        >>> from pdf_converter import convert
        >>> result = convert('document.pdf')
        >>> if result.success:
        ...     print(f"Saved to: {result.html_path}")
    """
    converter = PDFToAccessibleHTML()
    return converter.convert(pdf_path, output_dir)
