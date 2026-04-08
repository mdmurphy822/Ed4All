#!/usr/bin/env python3
"""
PDF Accessibility Converter CLI

Command-line interface for converting PDF documents to WCAG 2.1 AA compliant HTML.

Usage:
    python -m pdf_converter input.pdf [options]
    pdf-to-html input.pdf [options]
"""

import argparse
import logging
import sys
from pathlib import Path

from . import PDFToAccessibleHTML, ConversionResult, __version__


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def parse_args(args: list = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='pdf-to-html',
        description='Convert PDF documents to WCAG 2.1 AA compliant accessible HTML',
        epilog='Example: pdf-to-html document.pdf -o ./output/'
    )

    parser.add_argument(
        'input',
        type=str,
        help='Path to input PDF file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory (default: ./output/)'
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Output filename (default: derived from PDF name)'
    )

    parser.add_argument(
        '--no-dark-mode',
        action='store_true',
        help='Disable dark mode CSS support'
    )

    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Skip OCR fallback for image-heavy PDFs'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for OCR processing (default: 300)'
    )

    parser.add_argument(
        '--lang',
        type=str,
        default='eng',
        help='Tesseract language code for OCR (default: eng)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    # Claude integration options
    parser.add_argument(
        '--claude-model',
        type=str,
        default='claude-sonnet-4-20250514',
        help='Claude model to use (default: claude-sonnet-4-20250514)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable Claude response caching'
    )

    # Math options
    parser.add_argument(
        '--no-math',
        action='store_true',
        help='Disable MathML conversion for mathematical content'
    )

    # Image extraction options
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip image extraction from PDF'
    )

    parser.add_argument(
        '--no-ai-alt-text',
        action='store_true',
        help='Use OCR only for alt text (skip Claude API for image descriptions)'
    )

    parser.add_argument(
        '--image-quality',
        type=int,
        default=85,
        help='JPEG quality for compressed images (1-100, default: 85)'
    )

    parser.add_argument(
        '--max-image-width',
        type=int,
        default=800,
        help='Maximum width for embedded images in pixels (default: 800)'
    )

    # Vector graphics options
    parser.add_argument(
        '--no-vector-graphics',
        action='store_true',
        help='Skip detection and rendering of vector diagrams'
    )

    parser.add_argument(
        '--vector-dpi',
        type=int,
        default=150,
        help='DPI for rendering vector graphics as images (default: 150)'
    )

    parser.add_argument(
        '--vector-min-drawings',
        type=int,
        default=5,
        help='Minimum drawing operations to consider a vector region (default: 5)'
    )

    return parser.parse_args(args)


def main(args: list = None) -> int:
    """
    Main entry point for CLI.

    Args:
        args: Command-line arguments (default: sys.argv)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parsed = parse_args(args)
    setup_logging(parsed.verbose)

    logger = logging.getLogger(__name__)

    # Validate input
    input_path = Path(parsed.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if not input_path.suffix.lower() == '.pdf':
        logger.warning(f"Input file may not be a PDF: {input_path}")

    # Determine output directory
    if parsed.output:
        output_dir = Path(parsed.output)
    else:
        # Default to ./output/ relative to current directory
        output_dir = Path.cwd() / 'output'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create converter
    converter = PDFToAccessibleHTML(
        dpi=parsed.dpi,
        lang=parsed.lang,
        claude_model=parsed.claude_model,
        enable_cache=not parsed.no_cache,
        enable_math=not parsed.no_math,
        extract_images=not parsed.no_images,
        use_ai_alt_text=not parsed.no_ai_alt_text,
        image_quality=parsed.image_quality,
        max_image_width=parsed.max_image_width,
        extract_vector_graphics=not parsed.no_vector_graphics,
        vector_min_drawings=parsed.vector_min_drawings,
        vector_render_dpi=parsed.vector_dpi,
    )

    # Convert
    logger.info(f"Converting: {input_path}")
    result = converter.convert(str(input_path), str(output_dir))

    if result.success:
        print(f"\nConversion successful!")
        print(f"  Output: {result.html_path}")
        print(f"  Title:  {result.title}")
        print(f"  Pages:  {result.pages_processed}")
        print(f"  Words:  {result.total_words:,}")
        if hasattr(result, 'images_extracted') and result.images_extracted > 0:
            print(f"  Images: {result.images_extracted} extracted, {result.images_with_alt_text} with alt text")
        if hasattr(result, 'math_expressions_converted') and result.math_expressions_converted > 0:
            print(f"  Math:   {result.math_expressions_converted} expressions converted to MathML")
        return 0
    else:
        logger.error(f"Conversion failed: {result.error}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
