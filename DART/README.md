# DART (Digital Accessibility Remediation Tool)

A toolkit for converting PDF documents to WCAG 2.2 AA compliant accessible HTML.

## Features

- **pdftotext extraction** for born-digital PDFs (best quality)
- **OCR fallback** for scanned documents using Tesseract
- **Automatic structure detection** - headings, paragraphs, tables, references
- **Image extraction** - raster images and vector graphics rendered as PNG
- **AI-powered alt text** - automatic image descriptions using Claude
- **WCAG 2.2 AA validation** - built-in accessibility validation
- **WCAG 2.2 AA compliant output** including:
  - Skip links for keyboard navigation
  - ARIA landmarks and roles
  - Semantic HTML structure
  - Dark mode support (via `prefers-color-scheme`)
  - Reduced motion support (via `prefers-reduced-motion`)
  - Focus not obscured (WCAG 2.2 - 2.4.11, 2.4.12)
  - Focus appearance compliance (WCAG 2.2 - 2.4.13)
  - Target size minimums (WCAG 2.2 - 2.5.8)
  - Responsive design
  - Print styles

## Installation

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt install poppler-utils tesseract-ocr

# macOS
brew install poppler tesseract
```

### Python Package

```bash
cd DART

# Basic installation
pip install -e .

# With OCR support
pip install -e ".[ocr]"

# Full installation (all features)
pip install -e ".[full]"
```

## Usage

### Command Line

```bash
# Basic conversion
python convert.py document.pdf

# Specify output directory
python convert.py document.pdf -o ./accessible/

# With verbose output
python convert.py document.pdf -v

# For scanned PDFs with OCR
python convert.py scanned.pdf --dpi 400
```

### Python API

```python
from pdf_converter import convert, PDFToAccessibleHTML

# Simple conversion
result = convert('document.pdf')
if result.success:
    print(f"Saved to: {result.html_path}")
    print(f"Title: {result.title}")
    print(f"Pages: {result.pages_processed}")

# Custom configuration
converter = PDFToAccessibleHTML(dpi=400, lang='deu')
result = converter.convert('german_doc.pdf', output_dir='./output/')
```

### WCAG Enhancement Only

If you already have HTML and want to enhance it for accessibility:

```python
from pdf_converter import enhance_html_wcag, WCAGOptions

html_content = open('document.html').read()
enhanced = enhance_html_wcag(html_content, WCAGOptions(
    add_skip_link=True,
    dark_mode=True,
    add_aria_landmarks=True,
    wcag_version="2.2"  # Target WCAG 2.2 AA
))
```

### WCAG Validation

Validate existing HTML against WCAG 2.2 AA:

```python
from pdf_converter import validate_html_wcag, WCAGValidator

# Simple validation
report = validate_html_wcag(html_content)
print(f"Compliant: {report.wcag_aa_compliant}")
print(f"Issues: {report.total_issues}")
print(report.to_text())

# Detailed validation with file
validator = WCAGValidator(strict_mode=True)
report = validator.validate_file('document.html')
print(report.to_json())
```

## Examples

See the `examples/` directory for conversion examples:

- **NASA NESC 2024** (`examples/nasa_nesc_2024/`) - Flagship example with a 76-page multi-column NASA publication. Includes detailed workflow documentation.
- **Small Samples** (`examples/small_samples/`) - Quick reference examples including math textbooks, academic papers, and image-only documents.

## Gold Standard Template

The `templates/gold_standard.html` file demonstrates the target output format with:
- Complete semantic HTML5 structure
- ARIA roles and landmarks
- Accessible tables with proper headers
- Skip links for keyboard navigation
- Dark mode CSS support
- Responsive design
- Print-friendly styles

Use this as a reference when customizing output styles.

## Directory Structure

```
DART/
├── convert.py              # CLI entry point
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── CLAUDE.md               # Development guidance
├── examples/               # Conversion examples and workflows
├── pdf_converter/          # Core package
│   ├── __init__.py         # Public API
│   ├── converter.py        # PDF to HTML converter
│   ├── wcag_enhancer.py    # WCAG 2.2 enhancement
│   ├── wcag_validator.py   # WCAG 2.2 validation
│   ├── cli.py              # CLI implementation
│   ├── alt_text_generator.py  # AI-powered alt text
│   ├── image_extractor.py  # PDF image extraction (raster + vector)
│   ├── embed_images.py     # Image embedding utility
│   ├── math_processor.py   # Math/LaTeX processing
│   └── claude_processor.py # Claude API integration
├── templates/
│   └── gold_standard.html  # Reference template
├── output/                 # Default output directory
└── tests/
    ├── test_alt_text_generator.py
    ├── test_image_extractor.py
    └── test_math_processor.py
```

## CLI Options

```
Usage: python convert.py <input.pdf> [options]

Options:
  -o, --output DIR       Output directory (default: ./output/)
  -n, --name NAME        Output filename (default: derived from PDF)
  --no-dark-mode         Disable dark mode CSS support
  --no-ocr               Skip OCR fallback for image-heavy PDFs
  --dpi N                DPI for OCR processing (default: 300)
  --lang CODE            Tesseract language code (default: eng)
  --no-vector-graphics   Skip vector diagram detection/rendering
  --vector-dpi N         DPI for vector graphics rendering (default: 150)
  --vector-min-drawings  Min drawing ops to detect vector region (default: 5)
  -v, --verbose          Enable verbose output
  --version              Show version information
```

## WCAG 2.2 AA Compliance

The generated HTML meets WCAG 2.2 AA success criteria:

### Core WCAG 2.1 Criteria

| Criterion | Description | Implementation |
|-----------|-------------|----------------|
| 1.3.1 | Info and Relationships | Semantic HTML, ARIA landmarks |
| 1.4.1 | Use of Color | Links underlined, not color-only |
| 1.4.3 | Contrast (Minimum) | 4.5:1 text contrast ratio |
| 1.4.10 | Reflow | Responsive design, no horizontal scroll |
| 2.1.1 | Keyboard | All interactive elements keyboard-accessible |
| 2.4.1 | Bypass Blocks | Skip link to main content |
| 2.4.2 | Page Titled | Meaningful page title |
| 2.4.6 | Headings and Labels | Descriptive headings |
| 2.4.7 | Focus Visible | Clear focus indicators |
| 2.3.3 | Animation | Reduced motion support |

### New WCAG 2.2 Criteria

| Criterion | Description | Implementation |
|-----------|-------------|----------------|
| 2.4.11 | Focus Not Obscured (Minimum) | scroll-margin for fixed elements |
| 2.4.12 | Focus Not Obscured (Enhanced) | scroll-margin-top/bottom: 80px |
| 2.4.13 | Focus Appearance | 3px outline (exceeds 2px minimum), 3:1 contrast |
| 2.5.8 | Target Size (Minimum) | 24x24px minimum for interactive elements |

## License

MIT License
