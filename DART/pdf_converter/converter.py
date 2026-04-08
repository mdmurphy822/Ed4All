"""
PDF to WCAG 2.2 AA Accessible HTML Converter.

Uses pdftotext for born-digital PDFs (preserves layout and reading order)
with fallback to OCR for scanned documents.

Pipeline: PDF -> pdftotext (or OCR fallback) -> Claude Review -> Semantic HTML -> WCAG 2.2 Enhancement -> Validation
"""

import html
import re
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .claude_processor import ClaudeProcessor, DocumentStructure

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A block of text (paragraph, heading, etc.)."""
    text: str
    block_type: str = 'paragraph'  # paragraph, heading, list_item
    heading_level: int = 0


@dataclass
class ConversionResult:
    """Result of PDF to HTML conversion."""
    success: bool
    html_path: str = ''
    error: str = ''
    pages_processed: int = 0
    total_words: int = 0
    title: str = ''
    # Math and image stats
    math_expressions_converted: int = 0
    images_extracted: int = 0
    images_with_alt_text: int = 0
    # WCAG validation stats
    wcag_compliant: bool = True
    wcag_issues_count: int = 0
    wcag_critical_count: int = 0


class PDFToAccessibleHTML:
    """
    PDF to WCAG HTML converter using pdftotext with OCR fallback.

    Uses pdftotext for born-digital PDFs (best quality) with
    Tesseract OCR as fallback for scanned documents.
    Claude AI is used to review text ordering and structure detection.
    """

    def __init__(
        self,
        dpi: int = 300,
        lang: str = 'eng',
        min_confidence: float = 30.0,
        claude_api_key: Optional[str] = None,
        claude_model: str = "claude-sonnet-4-20250514",
        enable_cache: bool = True,
        # Math options
        enable_math: bool = True,
        # Image options
        extract_images: bool = True,
        use_ai_alt_text: bool = True,
        image_quality: int = 85,
        max_image_width: int = 800,
        # Vector graphics options
        extract_vector_graphics: bool = True,
        vector_min_drawings: int = 5,
        vector_cluster_distance: float = 50.0,
        vector_render_dpi: int = 150,
        # WCAG validation options
        validate_wcag: bool = True,
        wcag_strict: bool = False,
    ):
        """
        Initialize the converter.

        Args:
            dpi: Resolution for OCR fallback
            lang: Tesseract language code for OCR fallback
            min_confidence: Minimum OCR confidence (for fallback)
            claude_api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            claude_model: Claude model to use
            enable_cache: Whether to enable response caching
            enable_math: Whether to convert math to MathML
            extract_images: Whether to extract and embed images from PDF
            use_ai_alt_text: Whether to use Claude for alt text generation
            image_quality: JPEG quality for compressed images (1-100)
            max_image_width: Maximum width for embedded images
            extract_vector_graphics: Whether to detect and render vector diagrams
            vector_min_drawings: Minimum drawing operations to consider a vector region
            vector_cluster_distance: Distance (pixels) to cluster nearby drawings
            vector_render_dpi: DPI for rendering vector regions as images
            validate_wcag: Whether to validate output against WCAG 2.2 AA
            wcag_strict: If True, treat AA failures as blocking
        """
        self.dpi = dpi
        self.lang = lang
        self.min_confidence = min_confidence
        self.enable_math = enable_math
        self.extract_images = extract_images
        self.use_ai_alt_text = use_ai_alt_text
        self.image_quality = image_quality
        self.max_image_width = max_image_width
        self.extract_vector_graphics = extract_vector_graphics
        self.vector_min_drawings = vector_min_drawings
        self.vector_cluster_distance = vector_cluster_distance
        self.vector_render_dpi = vector_render_dpi
        self.validate_wcag = validate_wcag
        self.wcag_strict = wcag_strict

        self._claude = None
        self._claude_config = {
            'api_key': claude_api_key,
            'model': claude_model,
            'enable_cache': enable_cache,
        }

    @property
    def claude(self) -> 'ClaudeProcessor':
        """Lazy initialization of Claude processor."""
        if self._claude is None:
            from .claude_processor import ClaudeProcessor
            self._claude = ClaudeProcessor(**self._claude_config)
        return self._claude

    def convert(self, pdf_path: str, output_dir: str = None) -> ConversionResult:
        """
        Convert PDF to WCAG-compliant HTML.

        Args:
            pdf_path: Path to input PDF file
            output_dir: Directory for output HTML file (default: ./output/)

        Returns:
            ConversionResult with success status and output path
        """
        try:
            pdf_path = Path(pdf_path)

            # Default output directory
            if output_dir is None:
                output_dir = Path(__file__).parent.parent / 'output'
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Converting PDF: {pdf_path}")

            # 1. Try pdftotext first (best for born-digital PDFs)
            raw_text = self._extract_with_pdftotext(str(pdf_path))

            # Check if we got meaningful text
            if len(raw_text.strip()) < 100:
                logger.info("pdftotext produced minimal output, trying OCR...")
                raw_text = self._extract_with_ocr(str(pdf_path))

            if not raw_text.strip():
                return ConversionResult(
                    success=False,
                    error="No text extracted from PDF"
                )

            # Count words from raw text
            total_words = len(raw_text.split())

            # 2. Extract images from PDF (if enabled)
            images_extracted = 0
            images_with_alt_text = 0
            extracted_images = []

            if self.extract_images:
                extracted_images, images_extracted, images_with_alt_text = \
                    self._extract_and_process_images(str(pdf_path), raw_text[:1000])

            # 3. Extract tables using pdfplumber
            extracted_tables = self._extract_tables_with_pdfplumber(str(pdf_path))
            tables_dir = None
            if extracted_tables:
                tables_dir = self._save_tables_for_review(
                    extracted_tables, output_dir, pdf_path.stem
                )
                logger.info(f"Extracted {len(extracted_tables)} tables to: {tables_dir}")

            # 4. Structure the text - Save extracted text for Claude Code review workflow
            text_output_path = output_dir / f"{pdf_path.stem}_extracted.txt"
            text_output_path.write_text(raw_text, encoding='utf-8')
            logger.info(f"Extracted text saved to: {text_output_path}")

            # Save extracted images for incorporation during review
            images_dir = None
            if extracted_images:
                images_dir = self._save_images_for_review(
                    extracted_images, output_dir, pdf_path.stem
                )
                logger.info(f"Extracted {len(extracted_images)} images to: {images_dir}")

            logger.info("="*60)
            logger.info("TEXT EXTRACTION COMPLETE")
            logger.info("="*60)
            logger.info(f"To generate gold-standard HTML, run in Claude Code:")
            logger.info(f"  'Review {text_output_path} and generate accessible HTML'")
            if extracted_images:
                logger.info(f"  Include the {len(extracted_images)} extracted images from {images_dir}")
            if extracted_tables:
                logger.info(f"  Include the {len(extracted_tables)} extracted tables from {tables_dir}")
            logger.info("="*60)

            return ConversionResult(
                success=True,
                html_path=str(text_output_path),
                pages_processed=self._count_pages(str(pdf_path)),
                total_words=total_words,
                title=f"[EXTRACTED] {pdf_path.stem}",
                images_extracted=images_extracted,
                images_with_alt_text=images_with_alt_text,
            )

            # NOTE: The code below is kept for potential future use but is currently unreachable
            # since we always return after extracting text for Claude Code review workflow.

            # Embed extracted images into HTML
            if extracted_images:
                html_content = self._embed_images_in_html(html_content, extracted_images)

            # 5. Apply WCAG enhancements
            from .wcag_enhancer import WCAGHTMLEnhancer, WCAGOptions
            enhancer = WCAGHTMLEnhancer()
            wcag_html = enhancer.enhance(html_content, WCAGOptions(
                add_skip_link=True,
                add_aria_landmarks=True,
                use_sections=True,
                enhance_figures=True,
                detect_tables=True,
                dark_mode=True,
                # Math options
                enhance_math=self.enable_math,
                mathml_fallback_text=True,
                # Image options
                embed_images=self.extract_images,
                generate_alt_text=self.use_ai_alt_text,
                image_quality=self.image_quality,
                max_image_width=self.max_image_width,
            ))

            # 6. Validate WCAG compliance (if enabled)
            wcag_compliant = True
            wcag_issues_count = 0
            wcag_critical_count = 0

            if self.validate_wcag:
                from .wcag_validator import WCAGValidator
                validator = WCAGValidator(strict_mode=self.wcag_strict)
                validation_report = validator.validate(wcag_html)

                wcag_compliant = validation_report.wcag_aa_compliant
                wcag_issues_count = validation_report.total_issues
                wcag_critical_count = validation_report.critical_count

                if not wcag_compliant:
                    logger.warning(
                        f"WCAG 2.2 AA validation: {validation_report.critical_count} critical, "
                        f"{validation_report.high_count} high severity issues"
                    )
                else:
                    logger.info("WCAG 2.2 AA validation passed")

            # 7. Save output
            output_filename = f"{pdf_path.stem}_accessible.html"
            output_path = output_dir / output_filename
            output_path.write_text(wcag_html, encoding='utf-8')

            logger.info(f"Conversion complete: {output_path}")

            return ConversionResult(
                success=True,
                html_path=str(output_path),
                pages_processed=self._count_pages(str(pdf_path)),
                total_words=total_words,
                title=title,
                images_extracted=images_extracted,
                images_with_alt_text=images_with_alt_text,
                wcag_compliant=wcag_compliant,
                wcag_issues_count=wcag_issues_count,
                wcag_critical_count=wcag_critical_count,
            )

        except Exception as e:
            logger.error(f"Conversion failed: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=str(e)
            )

    def _extract_with_pdftotext(self, pdf_path: str) -> str:
        """Extract text using pdftotext (without -layout for proper reading order)."""
        try:
            # Note: Do NOT use -layout flag for multi-column documents
            # Without -layout, pdftotext reads columns in proper order
            result = subprocess.run(
                ['pdftotext', pdf_path, '-'],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"pdftotext failed: {e}")
            return ""

    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Fallback: Extract text using OCR."""
        try:
            from pdf2image import convert_from_path
            import pytesseract

            images = convert_from_path(pdf_path, dpi=self.dpi)
            text_parts = []

            for image in images:
                text = pytesseract.image_to_string(
                    image,
                    lang=self.lang,
                    config='--psm 1'
                )
                text_parts.append(text)

            return '\n\n'.join(text_parts)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def _count_pages(self, pdf_path: str) -> int:
        """Count pages in PDF."""
        try:
            result = subprocess.run(
                ['pdfinfo', pdf_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            for line in result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    return int(line.split(':')[1].strip())
        except Exception:
            pass
        return 0

    def _extract_and_process_images(
        self,
        pdf_path: str,
        context: str = ""
    ) -> Tuple[list, int, int]:
        """
        Extract images from PDF and generate alt text.

        Args:
            pdf_path: Path to PDF file
            context: Document context for alt text generation

        Returns:
            Tuple of (images list, count extracted, count with alt text)
        """
        images = []
        images_extracted = 0
        images_with_alt_text = 0

        try:
            from .image_extractor import PDFImageExtractor, ImageProcessor
            from .alt_text_generator import AltTextGenerator

            # Extract images (raster + vector if enabled)
            with PDFImageExtractor(
                pdf_path,
                extract_vector_graphics=self.extract_vector_graphics,
                vector_min_drawings=self.vector_min_drawings,
                vector_cluster_distance=self.vector_cluster_distance,
                vector_render_dpi=self.vector_render_dpi,
            ) as extractor:
                images = extractor.extract_all()
                images_extracted = len(images)

            if not images:
                logger.info("No images found in PDF")
                return [], 0, 0

            logger.info(f"Extracted {images_extracted} images from PDF")

            # Process images (resize/compress)
            processor = ImageProcessor(
                max_width=self.max_image_width,
                quality=self.image_quality
            )
            images = processor.process_all(images)

            # Generate alt text
            api_key = self._claude_config.get('api_key')
            alt_gen = AltTextGenerator(
                api_key=api_key,
                use_ai=self.use_ai_alt_text and api_key is not None,
                use_ocr_fallback=True
            )

            for img in images:
                result = alt_gen.generate(img, context)
                if result.success and result.alt_text:
                    img.alt_text = result.alt_text
                    img.long_description = result.long_description
                    images_with_alt_text += 1

            logger.info(f"Generated alt text for {images_with_alt_text}/{images_extracted} images")

        except ImportError as e:
            logger.warning(f"Image extraction dependencies not available: {e}")
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")

        return images, images_extracted, images_with_alt_text

    def _save_images_for_review(
        self,
        images: list,
        output_dir: Path,
        pdf_stem: str
    ) -> Path:
        """
        Save extracted images to disk for Claude Code review workflow.

        Args:
            images: List of ExtractedImage objects
            output_dir: Output directory
            pdf_stem: PDF filename stem

        Returns:
            Path to images directory
        """
        import json

        # Create images directory
        images_dir = output_dir / f"{pdf_stem}_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Save each image and build metadata
        metadata = []
        for idx, img in enumerate(images):
            # Determine file extension
            ext = img.format if img.format else 'png'
            if ext == 'jpeg':
                ext = 'jpg'

            filename = f"image_{idx + 1}_page_{img.page}.{ext}"
            filepath = images_dir / filename

            # Save image data
            filepath.write_bytes(img.data)

            # Build metadata entry
            metadata.append({
                'filename': filename,
                'page': img.page,
                'width': img.width,
                'height': img.height,
                'caption': img.nearby_caption or '',
                'alt_text': img.alt_text or '',
                'long_description': img.long_description or '',
            })

        # Save metadata JSON
        metadata_path = images_dir / 'images_metadata.json'
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding='utf-8'
        )

        return images_dir

    def _extract_tables_with_pdfplumber(self, pdf_path: str) -> list:
        """
        Extract tables from PDF using pdfplumber for structural detection.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of table dictionaries with page, headers, rows, and HTML
        """
        tables = []
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    if not page_tables:
                        continue

                    for table_idx, table in enumerate(page_tables):
                        # Skip empty or single-row tables
                        if not table or len(table) < 2:
                            continue

                        # Filter out None values and empty strings
                        cleaned_table = []
                        for row in table:
                            if row:
                                cleaned_row = [cell if cell else '' for cell in row]
                                # Only include rows that have at least some content
                                if any(cell.strip() for cell in cleaned_row):
                                    cleaned_table.append(cleaned_row)

                        if len(cleaned_table) < 2:
                            continue

                        # First row is typically headers
                        headers = cleaned_table[0]
                        rows = cleaned_table[1:]

                        # Generate WCAG-compliant HTML for this table
                        table_html = self._table_to_html(
                            headers, rows, page_num, table_idx
                        )

                        tables.append({
                            'page': page_num,
                            'index': table_idx,
                            'headers': headers,
                            'rows': rows,
                            'num_rows': len(rows),
                            'num_cols': len(headers),
                            'html': table_html,
                        })

            logger.info(f"Extracted {len(tables)} tables from PDF")

        except ImportError:
            logger.warning(
                "pdfplumber not installed. Install with: pip install pdfplumber"
            )
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")

        return tables

    def _table_to_html(
        self,
        headers: list,
        rows: list,
        page_num: int,
        table_idx: int
    ) -> str:
        """
        Convert table data to WCAG-compliant HTML.

        Args:
            headers: List of column header strings
            rows: List of row data (each row is a list of cell strings)
            page_num: Page number for ID generation
            table_idx: Table index on page for ID generation

        Returns:
            HTML string for the table
        """
        table_id = f"table-p{page_num}-{table_idx}"

        html_parts = [
            f'<table id="{table_id}" class="extracted-table">',
            f'  <caption>Table from page {page_num}</caption>',
            '  <thead>',
            '    <tr>',
        ]

        # Add header cells with scope
        for header in headers:
            escaped_header = html.escape(str(header).strip())
            html_parts.append(f'      <th scope="col">{escaped_header}</th>')

        html_parts.extend([
            '    </tr>',
            '  </thead>',
            '  <tbody>',
        ])

        # Add data rows
        for row in rows:
            html_parts.append('    <tr>')
            for cell in row:
                escaped_cell = html.escape(str(cell).strip())
                html_parts.append(f'      <td>{escaped_cell}</td>')
            html_parts.append('    </tr>')

        html_parts.extend([
            '  </tbody>',
            '</table>',
        ])

        return '\n'.join(html_parts)

    def _save_tables_for_review(
        self,
        tables: list,
        output_dir: Path,
        pdf_stem: str
    ) -> Path:
        """
        Save extracted tables metadata to disk for Claude Code review workflow.

        Args:
            tables: List of table dictionaries
            output_dir: Output directory
            pdf_stem: PDF filename stem

        Returns:
            Path to tables metadata file
        """
        import json

        # Create tables directory
        tables_dir = output_dir / f"{pdf_stem}_tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata JSON
        metadata_path = tables_dir / 'tables_metadata.json'
        metadata_path.write_text(
            json.dumps(tables, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

        # Also save individual HTML files for each table
        for table in tables:
            table_file = tables_dir / f"table_p{table['page']}_{table['index']}.html"
            table_file.write_text(table['html'], encoding='utf-8')

        return tables_dir

    def _embed_images_in_html(self, html_content: str, images: list) -> str:
        """
        Embed extracted images into HTML content.

        Args:
            html_content: HTML string
            images: List of ExtractedImage objects

        Returns:
            HTML with embedded images
        """
        if not images:
            return html_content

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, 'html.parser')
            main = soup.find('main') or soup.find('body')

            if not main:
                return html_content

            # Find or create a figures section at the end of main
            # We'll append figures there
            for idx, img in enumerate(images):
                figure = soup.new_tag('figure')
                figure['id'] = f'extracted-figure-{idx + 1}'
                figure['class'] = 'extracted-image'

                # Create img tag with base64 data
                img_tag = soup.new_tag('img')
                img_tag['src'] = img.data_uri
                img_tag['alt'] = img.alt_text or f'Figure from page {img.page}'
                img_tag['loading'] = 'lazy'
                img_tag['width'] = img.width
                img_tag['height'] = img.height

                figure.append(img_tag)

                # Create figcaption
                figcaption = soup.new_tag('figcaption')
                caption_text = img.nearby_caption or f'Figure {idx + 1}'

                # If we have a long description, add expandable details
                if img.long_description and img.long_description != img.alt_text:
                    figcaption.string = caption_text
                    details = soup.new_tag('details')
                    summary = soup.new_tag('summary')
                    summary.string = 'Image description'
                    details.append(summary)

                    desc_p = soup.new_tag('p')
                    desc_p.string = img.long_description
                    details.append(desc_p)

                    figcaption.append(details)
                else:
                    figcaption.string = caption_text

                figure.append(figcaption)
                main.append(figure)

            return str(soup)

        except Exception as e:
            logger.error(f"Failed to embed images: {e}")
            return html_content

    def _structure_text(self, raw_text: str) -> List[TextBlock]:
        """Structure raw text into paragraphs and headings."""
        blocks = []

        # Pre-process: Fix common OCR spacing issues in section headers
        # "I. I NTRODUCTION" -> "I. INTRODUCTION"
        raw_text = self._fix_section_headers(raw_text)

        # Split into paragraphs (blank lines separate paragraphs)
        paragraphs = re.split(r'\n\s*\n', raw_text)

        for para in paragraphs:
            # Clean up the paragraph
            text = self._clean_paragraph(para)

            if not text or len(text) < 5:
                continue

            # Try to split heading from paragraph if combined
            heading, remainder = self._split_heading_from_paragraph(text)

            if heading:
                blocks.append(TextBlock(
                    text=heading,
                    block_type='heading',
                    heading_level=self._get_heading_level(heading)
                ))
                if remainder and len(remainder) > 30:
                    blocks.append(TextBlock(
                        text=remainder,
                        block_type='paragraph'
                    ))
            elif self._is_heading(text):
                blocks.append(TextBlock(
                    text=text,
                    block_type='heading',
                    heading_level=self._get_heading_level(text)
                ))
            else:
                blocks.append(TextBlock(
                    text=text,
                    block_type='paragraph'
                ))

        return blocks

    def _split_heading_from_paragraph(self, text: str) -> tuple:
        """
        Split a heading from paragraph text if combined.

        Returns (heading, remainder) or (None, None) if not a combined block.
        """
        # Pattern: Roman numeral section followed by content
        # "I. INTRODUCTION Large language models..." -> ("I. INTRODUCTION", "Large language models...")
        match = re.match(
            r'^([IVX]+\.\s+[A-Z][A-Z\s]+?)(\s+[A-Z][a-z].+)$',
            text
        )
        if match:
            heading = match.group(1).strip()
            remainder = match.group(2).strip()
            # Verify the heading part looks valid
            if len(heading) < 100 and len(remainder) > 50:
                return heading, remainder

        # Pattern: Numbered section "1. Title Paragraph text..."
        match = re.match(
            r'^(\d+\.\s+[A-Z][a-z][^\n]{5,50})(\s+[A-Z][a-z].+)$',
            text
        )
        if match:
            heading = match.group(1).strip()
            remainder = match.group(2).strip()
            if len(heading) < 80 and len(remainder) > 50:
                return heading, remainder

        # Pattern: Lettered subsection "A. Title Paragraph text..."
        match = re.match(
            r'^([A-Z]\.\s+[A-Z][a-z][^\n]{5,50})(\s+[A-Z][a-z].+)$',
            text
        )
        if match:
            heading = match.group(1).strip()
            remainder = match.group(2).strip()
            if len(heading) < 80 and len(remainder) > 50:
                return heading, remainder

        return None, None

    def _fix_section_headers(self, text: str) -> str:
        """Fix common OCR spacing issues in section headers."""
        # Fix "I. I NTRODUCTION" -> "I. INTRODUCTION" pattern
        # Matches Roman numeral + period + space + split word
        section_fixes = [
            # Roman numeral sections with split words
            (r'([IVX]+\.)\s+([A-Z])\s+([A-Z]+)', r'\1 \2\3'),
            # "II. D ISCUSSION" -> "II. DISCUSSION"
            (r'([IVX]+\.)\s+([A-Z])\s+([A-Z][A-Z]+)', r'\1 \2\3'),
            # Numbered sections "1. I NTRODUCTION" -> "1. INTRODUCTION"
            (r'(\d+\.)\s+([A-Z])\s+([A-Z]+)', r'\1 \2\3'),
            # Fix "A. R elated work" -> "A. Related work"
            (r'([A-Z]\.)\s+([A-Z])\s+([a-z]+)', r'\1 \2\3'),
            # Fix split ALL CAPS words: "R EFERENCES" -> "REFERENCES"
            (r'\b([A-Z])\s+([A-Z]{3,})\b', r'\1\2'),
            # Fix "Q UANTUM" -> "QUANTUM"
            (r'\b([A-Z])\s+([A-Z][A-Z]+)\b', r'\1\2'),
        ]

        for pattern, replacement in section_fixes:
            text = re.sub(pattern, replacement, text)

        return text

    def _clean_paragraph(self, text: str) -> str:
        """Clean up a paragraph of text."""
        # Remove excessive whitespace but preserve single spaces
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Collapse multiple spaces
            line = re.sub(r'  +', ' ', line).strip()
            if line:
                cleaned_lines.append(line)

        # Join lines - if line ends with hyphen, join without space
        result = []
        for i, line in enumerate(cleaned_lines):
            if line.endswith('-') and i < len(cleaned_lines) - 1:
                # Hyphenated word at line break
                result.append(line[:-1])
            else:
                result.append(line + ' ')

        text = ''.join(result).strip()

        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text)  # Standalone page numbers

        # Remove common footnote patterns that get mixed into text
        footnote_patterns = [
            r'Contact emails follow the format [^.]+\.',
            r'Corresponding author[^.]*\.',
            r'Email:\s*[^\s]+@[^\s]+',
            r'\*[^.]*@[^.]*\.',
            r'\dagger[^.]*\.',
            # Handle split footnotes (line breaks in middle)
            r'\{first\.[^}]*\}@[^\s.]+\.',
            r'last\}@[^\s.]+\.',
            r'first\.last\}@[^\s.]+\.',
            r'@aalto\.fi\.?',
            r'@[a-z]+\.(edu|fi|com|org)\.?',
            # Clean up orphaned domain fragments
            r'\s+(fi|edu|com|org)\.\s*$',
            r'\s+(fi|edu|com|org)\.\s+',
        ]
        for pattern in footnote_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

        # Clean up any orphaned curly braces from removed patterns
        text = re.sub(r'\s*\{[^}]{0,20}\}\s*', ' ', text)

        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _is_heading(self, text: str) -> bool:
        """Detect if text is a heading."""
        text = text.strip()

        # Too long for a heading
        if len(text) > 200:
            return False

        # Too short - likely just a label or fragment
        if len(text) < 5:
            return False

        # Exclude common non-heading patterns
        # Author names (typically 2-3 words, personal names)
        # University/affiliation lines
        non_heading_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # "First Last" (author name)
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$',  # "First Middle Last"
            r'University',  # Affiliations
            r'Institute',
            r'Department',
            r'College',
            r'@',  # Email addresses
            r'^Fig\.',  # Figure references
            r'^Table',  # Table references
            r'^\d+$',  # Just a number
            r'^RY\s*\(',  # LaTeX/math notation artifacts
            r'^RX\s*\(',
            r'^RZ\s*\(',
            r'^H[A-Z]{1,3}\s*\(',  # Math notation like "HRY (1.438)"
            r'^\|[EV]\|',  # Math notation like "|E| X"
            r'^[A-Z]\s+[A-Z]$',  # Single letter pairs like "H A"
            r'^\[\d+\]',  # Reference citations
            r'^M\.\s+Drame',  # Author names in references
            r'^\d+\)\s+[A-Z]',  # Numbered list items that are too long
        ]
        for pattern in non_heading_patterns:
            if re.search(pattern, text, re.I):
                return False

        # Roman numeral section headers (I., II., III., etc.)
        if re.match(r'^[IVX]+\.\s+[A-Z]', text):
            return True

        # Numbered sections (1., 2., 1.1, etc.)
        if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', text):
            return True

        # ALL CAPS text that's not too long and has multiple words
        if text.isupper() and len(text) < 100 and len(text.split()) >= 2:
            return True

        # Common heading keywords (alone or with numbering)
        heading_keywords = [
            'abstract', 'introduction', 'conclusion', 'references',
            'acknowledgment', 'acknowledgement', 'appendix', 'bibliography',
            'methods', 'methodology', 'results', 'discussion', 'background',
            'related work', 'future work', 'evaluation', 'experiments',
            'overview', 'summary', 'implementation', 'architecture',
            'approach', 'contributions', 'limitations', 'fine-tuning pipeline'
        ]
        text_lower = text.lower().strip()

        for keyword in heading_keywords:
            # Exact match or with section number
            if text_lower == keyword:
                return True
            if re.match(rf'^[ivx\d]+\.?\s*{keyword}', text_lower):
                return True

        return False

    def _get_heading_level(self, text: str) -> int:
        """Determine heading level from text."""
        text = text.strip()

        # Roman numeral main sections -> h2
        if re.match(r'^[IVX]+\.\s+', text):
            return 2

        # ALL CAPS -> h2
        if text.isupper():
            return 2

        # Lettered subsections (A., B.) -> h3
        if re.match(r'^[A-Z]\.\s+', text):
            return 3

        # Numbered main sections (1., 2.) -> h2
        if re.match(r'^\d+\.\s+[A-Z]', text) and '.' not in text[2:6]:
            return 2

        # Numbered subsections (1.1, 2.1) -> h3
        if re.match(r'^\d+\.\d+\.?\s+', text):
            return 3

        # Sub-subsections (1.1.1) -> h4
        if re.match(r'^\d+\.\d+\.\d+', text):
            return 4

        # Default for keywords like Abstract, References
        return 2

    def _detect_title(self, blocks: List[TextBlock], fallback: str) -> str:
        """Detect document title from first blocks."""
        for block in blocks[:8]:
            text = block.text.strip()

            # Skip very short or very long text
            if len(text) < 15 or len(text) > 200:
                continue

            # Skip if it looks like author/affiliation info
            if '@' in text or 'University' in text:
                continue

            # Skip common non-title patterns
            skip_patterns = [
                r'^Abstract',
                r'^Keywords',
                r'^Introduction',
                r'^\d',
                r'^arXiv:',
                r'^Fig\.',
                r'^Table',
            ]
            if any(re.match(p, text, re.I) for p in skip_patterns):
                continue

            # Skip if it's just author names (short with only proper nouns)
            words = text.split()
            if len(words) <= 4 and all(w[0].isupper() and w[1:].islower() for w in words if len(w) > 1):
                continue

            # Good candidate - clean it up
            # Remove trailing author names if present (pattern: "Title Name Name")
            # Look for where proper title ends and author names begin
            title = text

            # If title contains common author-title separators, split there
            for sep in [' Linus ', ' Valter ', ' by ']:
                if sep in title:
                    title = title.split(sep)[0].strip()
                    break

            if len(title) >= 15 and not title.endswith(','):
                return title

        return fallback

    def _generate_semantic_html(self, blocks: List[TextBlock], title: str) -> str:
        """Generate semantic HTML with proper structure."""
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'  <title>{html.escape(title)}</title>',
            '  <style>',
            '    body {',
            '      font-family: Georgia, "Times New Roman", Times, serif;',
            '      font-size: 1.1rem;',
            '      line-height: 1.8;',
            '      max-width: 50rem;',
            '      margin: 0 auto;',
            '      padding: 2rem 1.5rem;',
            '      color: #1a1a1a;',
            '    }',
            '    h1 {',
            '      font-size: 1.75rem;',
            '      line-height: 1.3;',
            '      margin-bottom: 1.5rem;',
            '      text-align: center;',
            '    }',
            '    h2 {',
            '      font-size: 1.4rem;',
            '      margin-top: 2.5rem;',
            '      margin-bottom: 1rem;',
            '      border-bottom: 1px solid #ccc;',
            '      padding-bottom: 0.5rem;',
            '    }',
            '    h3 {',
            '      font-size: 1.2rem;',
            '      margin-top: 2rem;',
            '      margin-bottom: 0.75rem;',
            '    }',
            '    h4 {',
            '      font-size: 1.1rem;',
            '      margin-top: 1.5rem;',
            '      margin-bottom: 0.5rem;',
            '    }',
            '    p {',
            '      margin-bottom: 1rem;',
            '      text-align: justify;',
            '    }',
            '    section {',
            '      margin-bottom: 2rem;',
            '    }',
            '  </style>',
            '</head>',
            '<body>',
            f'  <h1>{html.escape(title)}</h1>',
        ]

        current_section = None

        for block in blocks:
            text = html.escape(block.text.strip())

            if not text:
                continue

            if block.block_type == 'heading':
                level = block.heading_level

                # Close previous section if open
                if current_section:
                    html_parts.append('  </section>')

                # Create section ID from heading text
                section_id = re.sub(r'[^a-z0-9]+', '-', text.lower())[:50].strip('-')

                html_parts.append(f'  <section id="{section_id}" aria-labelledby="{section_id}-heading">')
                html_parts.append(f'    <h{level} id="{section_id}-heading">{text}</h{level}>')
                current_section = section_id
            else:
                # Regular paragraph - only include if substantial
                if len(text) > 30:
                    html_parts.append(f'    <p>{text}</p>')

        # Close last section
        if current_section:
            html_parts.append('  </section>')

        html_parts.extend([
            '</body>',
            '</html>'
        ])

        return '\n'.join(html_parts)

    def _generate_html_from_structure(self, doc: 'DocumentStructure') -> str:
        """Generate semantic HTML from Claude's structured output."""
        from .claude_processor import BlockType

        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'  <title>{html.escape(doc.title)}</title>',
        ]

        # Add metadata
        if doc.authors:
            html_parts.append(f'  <meta name="author" content="{html.escape(", ".join(doc.authors))}">')

        if doc.metadata.get('keywords'):
            keywords = ', '.join(doc.metadata['keywords'])
            html_parts.append(f'  <meta name="keywords" content="{html.escape(keywords)}">')

        if doc.abstract:
            desc = doc.abstract[:200] + '...' if len(doc.abstract) > 200 else doc.abstract
            html_parts.append(f'  <meta name="description" content="{html.escape(desc)}">')

        # Basic styles (will be enhanced by WCAG enhancer)
        html_parts.extend([
            '  <style>',
            '    body {',
            '      font-family: Georgia, "Times New Roman", Times, serif;',
            '      font-size: 1.1rem;',
            '      line-height: 1.8;',
            '      max-width: 50rem;',
            '      margin: 0 auto;',
            '      padding: 2rem 1.5rem;',
            '      color: #1a1a1a;',
            '    }',
            '    h1 { font-size: 1.75rem; line-height: 1.3; margin-bottom: 1.5rem; text-align: center; }',
            '    h2 { font-size: 1.4rem; margin-top: 2.5rem; margin-bottom: 1rem; border-bottom: 1px solid #ccc; padding-bottom: 0.5rem; }',
            '    h3 { font-size: 1.2rem; margin-top: 2rem; margin-bottom: 0.75rem; }',
            '    h4 { font-size: 1.1rem; margin-top: 1.5rem; margin-bottom: 0.5rem; }',
            '    p { margin-bottom: 1rem; text-align: justify; }',
            '    section { margin-bottom: 2rem; }',
            '    .authors { text-align: center; font-size: 1.1rem; margin-bottom: 2rem; }',
            '    .abstract { background: #f9f9f9; padding: 1.5rem; border-left: 4px solid #0066cc; margin: 2rem 0; }',
            '    .abstract h2 { margin-top: 0; border-bottom: none; }',
            '    .references { font-size: 0.9rem; }',
            '    .references ol { padding-left: 2rem; }',
            '    .references li { margin-bottom: 0.75rem; }',
            '  </style>',
            '</head>',
            '<body>',
        ])

        # Header with title and authors
        html_parts.append('  <header>')
        html_parts.append(f'    <h1>{html.escape(doc.title)}</h1>')

        if doc.authors:
            authors_html = ', '.join(f'<strong>{html.escape(a)}</strong>' for a in doc.authors)
            html_parts.append(f'    <p class="authors">{authors_html}</p>')

        html_parts.append('  </header>')

        # Abstract section
        if doc.abstract:
            abstract_escaped = html.escape(doc.abstract)
            html_parts.extend([
                '  <section class="abstract" aria-labelledby="abstract-heading">',
                '    <h2 id="abstract-heading">Abstract</h2>',
                f'    <p>{abstract_escaped}</p>',
                '  </section>',
            ])

        # Process content blocks
        current_section = None
        in_reference_section = False

        for block in doc.blocks:
            content = html.escape(block.content.strip())
            if not content:
                continue

            if block.block_type == BlockType.HEADING.value:
                level = block.heading_level or 2
                level = min(max(level, 2), 4)  # Clamp to 2-4

                # Close previous section if open
                if current_section:
                    if in_reference_section:
                        html_parts.append('    </ol>')
                        in_reference_section = False
                    html_parts.append('  </section>')

                # Create section ID
                section_id = re.sub(r'[^a-z0-9]+', '-', content.lower())[:50].strip('-')
                heading_id = f"{section_id}-heading"

                # Check if this is references section
                is_references = 'reference' in content.lower()

                if is_references:
                    html_parts.extend([
                        f'  <section id="references" class="references" aria-labelledby="references-heading">',
                        f'    <h{level} id="references-heading">{content}</h{level}>',
                        '    <ol>',
                    ])
                    in_reference_section = True
                else:
                    html_parts.extend([
                        f'  <section id="{section_id}" aria-labelledby="{heading_id}">',
                        f'    <h{level} id="{heading_id}">{content}</h{level}>',
                    ])

                current_section = section_id

            elif block.block_type == BlockType.PARAGRAPH.value:
                if len(content) > 30:
                    html_parts.append(f'    <p>{content}</p>')

            elif block.block_type == BlockType.REFERENCE.value:
                ref_num = block.reference_number or ''
                if in_reference_section:
                    html_parts.append(f'      <li id="ref-{ref_num}">{content}</li>')
                else:
                    html_parts.append(f'    <p class="reference">[{ref_num}] {content}</p>')

            elif block.block_type == BlockType.LIST_ITEM.value:
                html_parts.append(f'    <li>{content}</li>')

            elif block.block_type == BlockType.FIGURE_CAPTION.value:
                html_parts.extend([
                    '    <figure>',
                    f'      <figcaption>{content}</figcaption>',
                    '    </figure>',
                ])

            elif block.block_type == BlockType.TABLE.value:
                html_parts.append(f'    <div class="table-content">{content}</div>')

            elif block.block_type == BlockType.DEFINITION.value:
                html_parts.extend([
                    '    <div class="definition" role="region">',
                    f'      <p>{content}</p>',
                    '    </div>',
                ])

            elif block.block_type == BlockType.ALGORITHM.value:
                html_parts.extend([
                    '    <div class="algorithm" role="region">',
                    f'      <pre>{content}</pre>',
                    '    </div>',
                ])

            elif block.block_type == BlockType.METADATA.value:
                html_parts.append(f'    <p class="metadata">{content}</p>')

            elif block.block_type == BlockType.AUTHOR.value:
                # Authors already handled in header
                pass

            elif block.block_type == BlockType.FOOTER.value:
                # Will be added at the end
                pass

            else:
                # Default to paragraph for unknown types
                if len(content) > 30:
                    html_parts.append(f'    <p>{content}</p>')

        # Close any open sections
        if current_section:
            if in_reference_section:
                html_parts.append('    </ol>')
            html_parts.append('  </section>')

        html_parts.extend([
            '</body>',
            '</html>'
        ])

        return '\n'.join(html_parts)


# Alias for backwards compatibility
OCRCompleteConverter = PDFToAccessibleHTML
