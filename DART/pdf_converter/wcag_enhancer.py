"""
WCAG 2.2 AA HTML Enhancer

Post-processor to enhance HTML output for WCAG 2.2 AA accessibility compliance.
Designed to work with output from pdf_to_html_converter_hybrid.py.

Features:
- Skip link for keyboard navigation (WCAG 2.4.1)
- ARIA landmarks (banner, main, navigation, figure)
- Semantic section structure with aria-labelledby
- Figure enhancement with expandable descriptions
- Math figure detection and special handling
- Table detection and semantic markup
- Reference section detection and list conversion
- Subsection heading detection (A., B., C. patterns)
- Internal figure link creation
- WCAG 2.2 AA compliant CSS (dark mode, reduced motion, contrast)
- Focus not obscured support (WCAG 2.2 - 2.4.11, 2.4.12)
- Focus appearance compliance (WCAG 2.2 - 2.4.13)
- Target size minimums (WCAG 2.2 - 2.5.8)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup, NavigableString, Tag
import unicodedata


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WCAGOptions:
    """Configuration options for WCAG 2.2 AA enhancement."""
    add_skip_link: bool = True
    add_aria_landmarks: bool = True
    use_sections: bool = True
    enhance_figures: bool = True
    expandable_descriptions: bool = True
    detect_tables: bool = True
    detect_references: bool = True
    detect_subsections: bool = True
    create_figure_links: bool = True
    dark_mode: bool = True
    reduced_motion: bool = True
    print_styles: bool = True
    # Math options
    enhance_math: bool = True
    mathml_fallback_text: bool = True
    # Image options
    embed_images: bool = True
    generate_alt_text: bool = True
    image_quality: int = 85
    max_image_width: int = 800
    # Document metadata
    document_title: str = ""
    document_author: str = ""
    document_description: str = ""
    # WCAG 2.2 specific options
    focus_not_obscured: bool = True      # 2.4.11, 2.4.12 - scroll-margin for focus
    focus_appearance_2px: bool = True    # 2.4.13 - minimum 2px focus outline
    target_size_minimum: bool = True     # 2.5.8 - 24x24px minimum target size
    wcag_version: str = "2.2"            # WCAG version targeting
    # CSS injection control
    inject_css: bool = True              # Set to False to skip CSS injection (use external CSS)


# =============================================================================
# CSS Template
# =============================================================================

WCAG_CSS = '''
/* WCAG 2.2 AA Compliant Styles */

/* Base reset */
*, *::before, *::after { box-sizing: border-box; }

/* Color scheme with WCAG AA contrast (4.5:1 minimum) */
:root {
  --color-text: #1f1f1f;           /* Contrast: 13.5:1 on white */
  --color-text-muted: #595959;      /* Contrast: 7:1 on white */
  --color-bg: #ffffff;
  --color-bg-alt: #f7f7f7;
  --color-accent: #0055aa;          /* Contrast: 7:1 on white */
  --color-accent-dark: #003d7a;
  --color-border: #cccccc;
  --color-focus: #0066cc;
  --font-body: Georgia, "Times New Roman", Times, serif;
  --font-mono: "Courier New", Courier, monospace;
  --line-height: 1.8;
  --max-width: 45rem;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
  :root {
    --color-text: #e8e8e8;
    --color-text-muted: #b0b0b0;
    --color-bg: #121212;
    --color-bg-alt: #1e1e1e;
    --color-accent: #6db3f2;
    --color-accent-dark: #4a9de8;
    --color-border: #404040;
  }
}

/* Reduce motion for users who prefer it (WCAG 2.3.3) */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
  html {
    scroll-behavior: auto !important;
  }
}

/* Base styles */
html {
  font-size: 100%; /* Respect user font size preferences */
  scroll-behavior: smooth;
}

body {
  font-family: var(--font-body);
  font-size: 1.125rem; /* 18px base for readability */
  line-height: var(--line-height);
  color: var(--color-text);
  background: var(--color-bg);
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 2rem 1.5rem;
}

/* Skip link for keyboard navigation (WCAG 2.4.1) */
.skip-link {
  position: absolute;
  top: -100px;
  left: 1rem;
  background: var(--color-accent);
  color: #fff;
  padding: 0.75rem 1.5rem;
  text-decoration: none;
  font-weight: bold;
  z-index: 1000;
  border-radius: 0 0 4px 4px;
}
.skip-link:focus {
  top: 0;
  outline: 3px solid var(--color-focus);
  outline-offset: 2px;
}

/* Focus styles (WCAG 2.4.7, 2.4.13) */
/* WCAG 2.2 2.4.13 Focus Appearance - minimum 2px outline with 3:1 contrast */
:focus {
  outline: 3px solid var(--color-focus);  /* Exceeds 2px minimum requirement */
  outline-offset: 2px;
}
:focus:not(:focus-visible) {
  outline: none;
}
:focus-visible {
  outline: 3px solid var(--color-focus);  /* Exceeds 2px minimum requirement */
  outline-offset: 2px;
}

/* WCAG 2.2 2.4.11/2.4.12 Focus Not Obscured - ensure scroll margin */
*:focus {
  scroll-margin-top: 80px;
  scroll-margin-bottom: 80px;
}

/* Typography */
h1 {
  font-size: 1.75rem;
  line-height: 1.3;
  text-align: center;
  margin: 0 0 1rem;
  color: var(--color-text);
}

h2 {
  font-size: 1.4rem;
  margin: 2.5rem 0 1rem;
  color: var(--color-accent-dark);
  border-bottom: 2px solid var(--color-border);
  padding-bottom: 0.5rem;
  scroll-margin-top: 20px;
}

h3 {
  font-size: 1.2rem;
  margin: 2rem 0 0.75rem;
  color: var(--color-text);
  scroll-margin-top: 20px;
}

h4, h5, h6 {
  scroll-margin-top: 20px;
}

p {
  margin: 1.25rem 0;
  text-align: justify;
  hyphens: auto;
  -webkit-hyphens: auto;
}

/* Links (WCAG 1.4.1 - not color alone) */
a {
  color: var(--color-accent);
  text-decoration: underline;
}
a:hover, a:focus {
  color: var(--color-accent-dark);
  text-decoration-thickness: 2px;
}

/* WCAG 2.2 2.5.8 Target Size (Minimum) - 24x24 CSS pixels */
a:not(p a):not(li a):not(td a):not(figcaption a),
button,
[role="button"],
summary,
details > summary {
  min-height: 24px;
  min-width: 24px;
}

/* Inline links exempt per WCAG 2.5.8 exception for inline text */
p a, li a, td a, figcaption a, span a {
  min-height: auto;
  min-width: auto;
}

/* Figures and images */
figure {
  margin: 2rem 0;
  padding: 1.5rem;
  background: var(--color-bg-alt);
  border: 1px solid var(--color-border);
  border-radius: 8px;
}

figure img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
  border-radius: 4px;
}

figcaption {
  margin-top: 1rem;
  font-size: 0.95rem;
  color: var(--color-text-muted);
}

/* Expandable descriptions for images */
details {
  margin-top: 0.75rem;
}

details summary {
  cursor: pointer;
  color: var(--color-accent);
  font-weight: 500;
}

details summary:hover {
  text-decoration: underline;
}

details p {
  margin: 0.5rem 0 0;
  padding: 0.75rem;
  background: var(--color-bg);
  border-left: 3px solid var(--color-accent);
  font-size: 0.9rem;
}

/* Math figures */
.math-figure {
  display: inline-block;
  margin: 1rem;
  padding: 1rem;
  vertical-align: top;
  max-width: 280px;
}

.math-figure img {
  max-height: 70px;
  background: #fff;
  padding: 10px;
  border-radius: 4px;
}

.math-figure code {
  display: block;
  font-family: var(--font-mono);
  font-size: 0.85rem;
  margin-top: 0.5rem;
  color: var(--color-text-muted);
}

/* Header and footer */
header[role="banner"] {
  text-align: center;
  margin-bottom: 3rem;
  padding-bottom: 2rem;
  border-bottom: 1px solid var(--color-border);
}

.authors {
  font-style: italic;
  color: var(--color-text-muted);
  margin-top: 1rem;
}

footer {
  margin-top: 4rem;
  padding-top: 2rem;
  border-top: 1px solid var(--color-border);
  font-size: 0.9rem;
  color: var(--color-text-muted);
}

/* Table of contents */
nav[aria-label="Table of contents"] {
  background: var(--color-bg-alt);
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

nav[aria-label="Table of contents"] h2 {
  margin-top: 0;
  font-size: 1.1rem;
  border: none;
}

nav[aria-label="Table of contents"] ul {
  margin: 0;
  padding-left: 1.5rem;
}

nav[aria-label="Table of contents"] li {
  margin: 0.5rem 0;
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
  font-size: 0.95rem;
}

th, td {
  padding: 0.75rem;
  border: 1px solid var(--color-border);
  text-align: left;
}

th {
  background: var(--color-bg-alt);
  font-weight: bold;
}

caption {
  font-weight: bold;
  margin-bottom: 0.5rem;
  text-align: left;
}

/* Reference lists */
.references ol {
  padding-left: 2.5rem;
}

.references li {
  margin: 0.75rem 0;
  line-height: 1.6;
}

/* Sections */
section {
  margin: 2rem 0;
}

/* Responsive design (WCAG 1.4.10 - Reflow) */
@media (max-width: 600px) {
  body {
    font-size: 1rem;
    padding: 1rem;
  }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.2rem; }
  figure { padding: 1rem; }
  .math-figure {
    display: block;
    max-width: 100%;
    margin: 1rem 0;
  }
}

/* Print styles */
@media print {
  body {
    font-size: 11pt;
    line-height: 1.5;
    max-width: none;
    color: #000;
    background: #fff;
  }
  .skip-link, nav[aria-label="Table of contents"] { display: none; }
  a { color: #000; text-decoration: underline; }
  figure { break-inside: avoid; }
}
'''


# =============================================================================
# Main Enhancer Class
# =============================================================================

class WCAGHTMLEnhancer:
    """
    Post-processor to enhance HTML for WCAG 2.2 AA compliance.

    Usage:
        enhancer = WCAGHTMLEnhancer()
        enhanced_html = enhancer.enhance(html_content, WCAGOptions())
    """

    # Patterns for detection
    MATH_SYMBOLS = ['=', '+', '-', '×', '÷', '^', '_', '∑', '∫', 'Σ', '∏', '√', '∞',
                    '≤', '≥', '≠', '≈', '∈', '∉', '⊂', '⊃', '∪', '∩', 'α', 'β', 'γ',
                    'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ω', 'Δ', 'Ω']

    SUBSECTION_PATTERN = re.compile(r'^([A-Z])[.)]\s+([A-Z][a-zA-Z\s]+.*)$')
    NUMBERED_SUBSECTION_PATTERN = re.compile(r'^(\d+)\)\s+([A-Z][a-zA-Z\s]+.*)$')

    REFERENCE_PATTERN = re.compile(r'^\[(\d+)\]\s*(.+)$', re.DOTALL)

    FIGURE_REF_PATTERN = re.compile(r'\b(Figure|Fig\.?|Figures|Figs\.?)\s+(\d+(?:\s*[-–—]\s*\d+)?)\b', re.IGNORECASE)
    TABLE_REF_PATTERN = re.compile(r'\b(Table|Tables)\s+(\d+(?:\s*[-–—]\s*\d+)?)\b', re.IGNORECASE)

    def __init__(self):
        """Initialize the enhancer."""
        self.figure_counter = 0
        self.table_counter = 0
        self.heading_ids: Dict[str, int] = {}

    def enhance(self, html: str, options: WCAGOptions = None) -> str:
        """
        Apply all WCAG enhancements to HTML.

        Args:
            html: Input HTML string
            options: Configuration options

        Returns:
            Enhanced HTML string with WCAG 2.2 AA compliance
        """
        if options is None:
            options = WCAGOptions()

        self.figure_counter = 0
        self.table_counter = 0
        self.heading_ids = {}

        soup = BeautifulSoup(html, 'html.parser')

        # Phase 1: Document structure
        if options.add_skip_link:
            self._add_skip_link(soup)

        if options.add_aria_landmarks:
            self._add_landmarks(soup)

        if options.use_sections:
            self._add_section_structure(soup)

        # Phase 2: Semantic content improvements
        if options.detect_subsections:
            self._detect_and_convert_subsections(soup)

        if options.detect_references:
            self._detect_and_convert_references(soup)

        if options.detect_tables:
            self._detect_and_convert_tables(soup)

        # Phase 3: Figure enhancement
        if options.enhance_figures:
            self._enhance_all_figures(soup, options.expandable_descriptions)

        # Phase 3.5: Math enhancement (MathML conversion)
        if options.enhance_math:
            self._enhance_math_content(soup, options)

        # Phase 4: Cross-references
        if options.create_figure_links:
            self._create_internal_links(soup)

        # Phase 5: CSS injection
        self._inject_css(soup, options)

        # Phase 6: Final cleanup
        self._add_accessibility_footer(soup)

        return str(soup)

    # =========================================================================
    # Phase 1: Document Structure
    # =========================================================================

    def _add_skip_link(self, soup: BeautifulSoup) -> None:
        """Add skip link for keyboard navigation (WCAG 2.4.1)."""
        # Check if skip link already exists
        if soup.find('a', class_='skip-link'):
            return

        # Find or create body
        body = soup.find('body')
        if not body:
            return

        # Create skip link
        skip_link = soup.new_tag('a', href='#main-content')
        skip_link['class'] = 'skip-link'
        skip_link.string = 'Skip to main content'

        # Insert at beginning of body
        body.insert(0, skip_link)

    def _add_landmarks(self, soup: BeautifulSoup) -> None:
        """Add ARIA landmarks to major page sections."""
        body = soup.find('body')
        if not body:
            return

        # Find or create main element
        main = soup.find('main')
        if not main:
            # Look for content container
            content = body.find('div', class_='content') or body.find('article')
            if content:
                main = soup.new_tag('main', id='main-content')
                content.wrap(main)
            else:
                # Wrap all body content except header/footer in main
                main = soup.new_tag('main', id='main-content')

                # Collect elements to move
                elements_to_wrap = []
                for child in list(body.children):
                    if isinstance(child, Tag):
                        if child.name not in ['header', 'footer', 'nav', 'script', 'style']:
                            if not (child.name == 'a' and 'skip-link' in child.get('class', [])):
                                elements_to_wrap.append(child)

                if elements_to_wrap:
                    # Insert main after skip link or at start
                    skip_link = body.find('a', class_='skip-link')
                    if skip_link:
                        skip_link.insert_after(main)
                    else:
                        body.insert(0, main)

                    # Move elements into main
                    for elem in elements_to_wrap:
                        main.append(elem.extract())
        else:
            main['id'] = main.get('id', 'main-content')
            # Note: <main> has implicit role="main" per HTML5, no need to add explicitly

        # Add role to header
        header = soup.find('header')
        if header:
            header['role'] = 'banner'

        # Add role to footer
        footer = soup.find('footer')
        if footer:
            footer['role'] = 'contentinfo'

        # Add role to nav
        for nav in soup.find_all('nav'):
            nav['role'] = nav.get('role', 'navigation')

    def _add_section_structure(self, soup: BeautifulSoup) -> None:
        """Add section elements around heading groups with aria-labelledby."""
        main = soup.find('main')
        if not main:
            return

        # Find all h2 elements (major sections)
        h2_elements = main.find_all('h2')

        for h2 in h2_elements:
            # Skip if already in a section or inside nav/header/footer
            if h2.find_parent('section') or h2.find_parent('nav') or h2.find_parent('header') or h2.find_parent('footer'):
                continue

            # Generate ID for heading if not present
            heading_id = h2.get('id')
            if not heading_id:
                heading_id = self._generate_heading_id(h2.get_text(strip=True))
                h2['id'] = heading_id

            # Collect siblings until next h2 or end
            section_content = [h2]
            sibling = h2.next_sibling

            while sibling:
                next_sib = sibling.next_sibling
                if isinstance(sibling, Tag):
                    if sibling.name == 'h2':
                        break
                    section_content.append(sibling)
                elif isinstance(sibling, NavigableString) and sibling.strip():
                    section_content.append(sibling)
                sibling = next_sib

            # Create section wrapper
            section = soup.new_tag('section')
            section['aria-labelledby'] = heading_id
            section_id = heading_id.replace('-heading', '') if heading_id.endswith('-heading') else f"section-{heading_id}"
            section['id'] = section_id

            # Insert section before first element
            h2.insert_before(section)

            # Move elements into section
            for elem in section_content:
                section.append(elem.extract() if hasattr(elem, 'extract') else elem)

    def _generate_heading_id(self, text: str) -> str:
        """Generate a URL-friendly ID from heading text."""
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Convert to lowercase and replace non-alphanumeric with hyphens
        text = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
        text = re.sub(r'[\s_]+', '-', text)
        text = re.sub(r'-+', '-', text)
        text = text.strip('-')

        # Truncate to reasonable length
        text = text[:50]

        # Handle duplicates
        base_id = text or 'heading'
        if base_id in self.heading_ids:
            self.heading_ids[base_id] += 1
            return f"{base_id}-{self.heading_ids[base_id]}"
        else:
            self.heading_ids[base_id] = 0
            return base_id

    # =========================================================================
    # Phase 2: Semantic Content Improvements
    # =========================================================================

    def _detect_and_convert_subsections(self, soup: BeautifulSoup) -> None:
        """Convert A., B., C. or 1), 2) patterns to <h3> headings."""
        main = soup.find('main')
        if not main:
            return

        for p in main.find_all('p'):
            text = p.get_text(strip=True)

            # Check for letter subsection (A. Title)
            match = self.SUBSECTION_PATTERN.match(text)
            if match:
                self._convert_to_heading(soup, p, text, 3)
                continue

            # Check for numbered subsection (1) Title)
            match = self.NUMBERED_SUBSECTION_PATTERN.match(text)
            if match:
                self._convert_to_heading(soup, p, text, 4)

    def _convert_to_heading(self, soup: BeautifulSoup, element: Tag, text: str, level: int) -> None:
        """Convert an element to a heading."""
        heading = soup.new_tag(f'h{level}')
        heading.string = text
        heading['id'] = self._generate_heading_id(text)
        element.replace_with(heading)

    def _detect_and_convert_references(self, soup: BeautifulSoup) -> None:
        """Convert reference sections to ordered lists."""
        # Find references heading
        ref_heading = None
        for h2 in soup.find_all(['h2', 'h3']):
            text = h2.get_text(strip=True).lower()
            if text in ['references', 'bibliography', 'works cited']:
                ref_heading = h2
                break

        if not ref_heading:
            return

        # Collect reference paragraphs
        references = []
        sibling = ref_heading.next_sibling

        while sibling:
            if isinstance(sibling, Tag):
                if sibling.name in ['h2', 'h3', 'section']:
                    break

                if sibling.name == 'p':
                    text = sibling.get_text(strip=True)
                    match = self.REFERENCE_PATTERN.match(text)
                    if match:
                        references.append((sibling, match.group(1), match.group(2)))

            sibling = sibling.next_sibling

        if not references:
            return

        # Create ordered list
        ol = soup.new_tag('ol')
        ol['class'] = 'references-list'

        for p_elem, num, content in references:
            li = soup.new_tag('li')
            li['id'] = f'ref-{num}'
            li['value'] = num
            li.string = content.strip()
            ol.append(li)

        # Create wrapper section
        section = soup.new_tag('section')
        section['class'] = 'references'
        section['aria-labelledby'] = ref_heading.get('id', 'references-heading')

        # Replace references
        if references:
            first_ref = references[0][0]
            first_ref.insert_before(ol)

            # Remove old paragraphs
            for p_elem, _, _ in references:
                p_elem.decompose()

    def _detect_and_convert_tables(self, soup: BeautifulSoup) -> None:
        """
        Detect table-like content and convert to proper <table> markup.

        This handles:
        - Tab-separated content
        - Pipe-separated content
        - Consistent whitespace-aligned columns
        """
        main = soup.find('main')
        if not main:
            return

        # Look for sequences of paragraphs that look like table rows
        paragraphs = main.find_all('p')

        i = 0
        while i < len(paragraphs):
            p = paragraphs[i]

            # Check if this looks like a table row
            if self._looks_like_table_row(p.get_text()):
                # Collect consecutive table-like rows
                table_rows = [p]
                j = i + 1

                while j < len(paragraphs):
                    next_p = paragraphs[j]
                    # Check if still in table structure
                    if next_p.find_previous_sibling() != table_rows[-1]:
                        break
                    if self._looks_like_table_row(next_p.get_text()):
                        table_rows.append(next_p)
                        j += 1
                    else:
                        break

                # Only convert if we have multiple rows
                if len(table_rows) >= 2:
                    self._convert_rows_to_table(soup, table_rows)
                    i = j
                    continue

            i += 1

    def _looks_like_table_row(self, text: str) -> bool:
        """Check if text looks like a table row."""
        # Tab-separated
        if '\t' in text and text.count('\t') >= 1:
            return True

        # Pipe-separated
        if '|' in text and text.count('|') >= 2:
            return True

        # Multiple consecutive spaces (column alignment)
        if re.search(r'\s{3,}', text):
            parts = re.split(r'\s{3,}', text)
            if len(parts) >= 2:
                return True

        return False

    def _convert_rows_to_table(self, soup: BeautifulSoup, rows: List[Tag]) -> None:
        """Convert a list of paragraph elements to a table."""
        self.table_counter += 1

        # Create table
        table = soup.new_tag('table')
        table['id'] = f'table-{self.table_counter}'

        # Determine delimiter
        first_text = rows[0].get_text()
        if '\t' in first_text:
            delimiter = '\t'
        elif '|' in first_text:
            delimiter = '|'
        else:
            delimiter = r'\s{3,}'

        # Process rows
        for idx, row in enumerate(rows):
            text = row.get_text()

            if delimiter == r'\s{3,}':
                cells = re.split(delimiter, text)
            else:
                cells = text.split(delimiter)

            cells = [c.strip() for c in cells if c.strip()]

            tr = soup.new_tag('tr')

            # First row as header
            cell_tag = 'th' if idx == 0 else 'td'
            for cell in cells:
                td = soup.new_tag(cell_tag)
                if cell_tag == 'th':
                    td['scope'] = 'col'
                td.string = cell
                tr.append(td)

            if idx == 0:
                thead = soup.new_tag('thead')
                thead.append(tr)
                table.append(thead)
            else:
                tbody = table.find('tbody')
                if not tbody:
                    tbody = soup.new_tag('tbody')
                    table.append(tbody)
                tbody.append(tr)

        # Replace first row with table
        rows[0].insert_before(table)

        # Remove old paragraphs
        for row in rows:
            row.decompose()

    # =========================================================================
    # Phase 3: Figure Enhancement
    # =========================================================================

    def _enhance_all_figures(self, soup: BeautifulSoup, expandable: bool) -> None:
        """Enhance all figures and images for WCAG compliance."""
        # First, handle existing figures
        for figure in soup.find_all('figure'):
            self._enhance_figure(soup, figure, expandable)

        # Then, wrap standalone images
        for img in soup.find_all('img'):
            if not img.find_parent('figure'):
                self._wrap_image_in_figure(soup, img, expandable)

    def _enhance_figure(self, soup: BeautifulSoup, figure: Tag, expandable: bool) -> None:
        """Enhance a figure element for WCAG compliance."""
        self.figure_counter += 1

        # Add role
        figure['role'] = 'figure'

        # Get image
        img = figure.find('img')
        if not img:
            return

        # Check if math figure
        is_math = self._is_math_image(img)

        if is_math:
            figure['class'] = figure.get('class', [])
            if isinstance(figure['class'], str):
                figure['class'] = figure['class'].split()
            if 'math-figure' not in figure['class']:
                figure['class'].append('math-figure')

        # Generate figure ID
        fig_id = f'figure-{self.figure_counter}'
        figure['id'] = fig_id

        # Get or create figcaption
        figcaption = figure.find('figcaption')
        caption_id = f'{fig_id}-caption'

        if figcaption:
            figcaption['id'] = caption_id
            figure['aria-labelledby'] = caption_id

            # Add expandable description if requested
            if expandable and not figcaption.find('details'):
                self._add_expandable_description(soup, figcaption, img, is_math)
        else:
            # Create figcaption
            alt_text = img.get('alt', '')
            if alt_text:
                figcaption = soup.new_tag('figcaption')
                figcaption['id'] = caption_id
                figure['aria-labelledby'] = caption_id

                if is_math:
                    self._create_math_figcaption(soup, figcaption, alt_text, expandable)
                else:
                    figcaption.string = alt_text
                    if expandable:
                        self._add_expandable_description(soup, figcaption, img, is_math)

                figure.append(figcaption)

        # Add loading="lazy" if not present
        if not img.get('loading'):
            img['loading'] = 'lazy'

    def _wrap_image_in_figure(self, soup: BeautifulSoup, img: Tag, expandable: bool) -> None:
        """Wrap a standalone image in a figure element."""
        self.figure_counter += 1

        # Create figure
        figure = soup.new_tag('figure')
        figure['role'] = 'figure'
        figure['id'] = f'figure-{self.figure_counter}'

        # Check if math
        is_math = self._is_math_image(img)
        if is_math:
            figure['class'] = ['math-figure']

        # Wrap image
        img.wrap(figure)

        # Add loading lazy
        img['loading'] = 'lazy'

        # Create figcaption if alt text exists
        alt_text = img.get('alt', '')
        if alt_text:
            figcaption = soup.new_tag('figcaption')
            figcaption['id'] = f'figure-{self.figure_counter}-caption'
            figure['aria-labelledby'] = figcaption['id']

            if is_math:
                self._create_math_figcaption(soup, figcaption, alt_text, expandable)
            else:
                figcaption.string = alt_text

            figure.append(figcaption)

    def _is_math_image(self, img: Tag) -> bool:
        """Detect if an image is a mathematical equation."""
        src = img.get('src', '').lower()
        alt = img.get('alt', '')

        # Check path patterns
        if 'math' in src or 'equation' in src or 'formula' in src:
            return True

        # Check for math symbols in alt
        symbol_count = sum(1 for sym in self.MATH_SYMBOLS if sym in alt)
        if symbol_count >= 2:
            return True

        # Check for math patterns in alt
        math_patterns = [
            r'[a-z]_\{',  # Subscript notation
            r'\^[\{\d]',  # Superscript
            r'\\[a-z]+',  # LaTeX commands
            r'\bsum\b|\bint\b|\blim\b',  # Math keywords
        ]
        for pattern in math_patterns:
            if re.search(pattern, alt, re.IGNORECASE):
                return True

        return False

    def _create_math_figcaption(self, soup: BeautifulSoup, figcaption: Tag,
                                 alt_text: str, expandable: bool) -> None:
        """Create figcaption specifically for math equations."""
        # Add code element with equation
        code = soup.new_tag('code')
        code['aria-label'] = 'Mathematical notation'
        code.string = alt_text
        figcaption.append(code)

        # Add expandable description
        if expandable:
            details = soup.new_tag('details')
            summary = soup.new_tag('summary')
            summary.string = 'Full description'
            details.append(summary)

            desc_p = soup.new_tag('p')
            # Generate verbal description of the math
            desc_p.string = self._generate_math_description(alt_text)
            details.append(desc_p)

            figcaption.append(details)

    def _add_expandable_description(self, soup: BeautifulSoup, figcaption: Tag,
                                     img: Tag, is_math: bool) -> None:
        """Add expandable description to figcaption."""
        alt_text = img.get('alt', '')
        if not alt_text:
            return

        details = soup.new_tag('details')
        summary = soup.new_tag('summary')
        summary.string = 'Figure description (click to expand)' if not is_math else 'Full description'
        details.append(summary)

        desc_p = soup.new_tag('p')
        if is_math:
            desc_p.string = self._generate_math_description(alt_text)
        else:
            desc_p.string = f"Image: {alt_text}"
        details.append(desc_p)

        figcaption.append(details)

    def _generate_math_description(self, math_text: str) -> str:
        """Generate a verbal description of mathematical notation."""
        # Basic transformations for common symbols
        replacements = [
            (r'Σ|\\sum', 'the sum of'),
            (r'∏|\\prod', 'the product of'),
            (r'∫|\\int', 'the integral of'),
            (r'√|\\sqrt', 'the square root of'),
            (r'≤|\\leq', 'less than or equal to'),
            (r'≥|\\geq', 'greater than or equal to'),
            (r'≠|\\neq', 'not equal to'),
            (r'≈|\\approx', 'approximately equal to'),
            (r'∈|\\in', 'is an element of'),
            (r'∉|\\notin', 'is not an element of'),
            (r'⊂|\\subset', 'is a subset of'),
            (r'∪|\\cup', 'union'),
            (r'∩|\\cap', 'intersection'),
            (r'_\{([^}]+)\}', r' subscript \1'),
            (r'\^\{([^}]+)\}', r' superscript \1'),
            (r'_([a-zA-Z0-9])', r' subscript \1'),
            (r'\^([a-zA-Z0-9])', r' superscript \1'),
        ]

        description = math_text
        for pattern, replacement in replacements:
            description = re.sub(pattern, replacement, description)

        return f"Mathematical expression: {description}"

    # =========================================================================
    # Phase 3.5: Math Enhancement (MathML)
    # =========================================================================

    def _enhance_math_content(self, soup: BeautifulSoup, options: "WCAGOptions") -> None:
        """
        Detect and convert mathematical expressions to MathML.

        This enhances accessibility by providing proper semantic math markup
        that screen readers can interpret correctly.
        """
        try:
            from .math_processor import MathDetector, MathMLConverter
        except ImportError:
            # Math processor not available, skip silently
            return

        detector = MathDetector()
        converter = MathMLConverter()

        main = soup.find('main')
        if not main:
            return

        # Process all text nodes
        text_nodes = list(main.find_all(text=True))
        for text_node in text_nodes:
            if not text_node.strip():
                continue

            parent = text_node.parent
            if parent.name in ['script', 'style', 'code', 'pre', 'math']:
                continue

            text = str(text_node)

            # Detect math in this text
            math_blocks = detector.detect_in_text(text)
            if not math_blocks:
                continue

            # Convert math blocks and build replacement HTML
            new_content = self._replace_math_with_mathml(
                text, math_blocks, converter, soup, options
            )

            if new_content:
                self._replace_text_with_html(text_node, new_content, soup)

    def _replace_math_with_mathml(
        self,
        text: str,
        math_blocks: list,
        converter,
        soup: BeautifulSoup,
        options: "WCAGOptions"
    ) -> str:
        """Replace math expressions in text with MathML markup."""
        result = []
        last_end = 0

        for block in math_blocks:
            # Add text before this math block
            if block.start_pos > last_end:
                result.append(self._escape_html(text[last_end:block.start_pos]))

            # Convert to MathML
            converter.convert(block)

            if block.mathml:
                # Add aria-label for accessibility
                mathml = block.mathml
                if options.mathml_fallback_text and block.fallback_text:
                    # Insert aria-label into math tag
                    mathml = mathml.replace(
                        '<math ',
                        f'<math aria-label="{self._escape_attr(block.fallback_text)}" '
                    )
                    if '<math>' in mathml:
                        mathml = mathml.replace(
                            '<math>',
                            f'<math aria-label="{self._escape_attr(block.fallback_text)}">'
                        )
                result.append(mathml)
            else:
                # Fallback: accessible span
                result.append(
                    f'<span role="math" aria-label="{self._escape_attr(block.fallback_text)}" '
                    f'class="math-fallback">{self._escape_html(block.raw_content)}</span>'
                )

            last_end = block.end_pos

        # Add remaining text
        if last_end < len(text):
            result.append(self._escape_html(text[last_end:]))

        return ''.join(result) if result else None

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))

    def _escape_attr(self, text: str) -> str:
        """Escape HTML attribute value."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))

    # =========================================================================
    # Phase 4: Cross-references
    # =========================================================================

    def _create_internal_links(self, soup: BeautifulSoup) -> None:
        """Convert Figure X and Table X references to clickable links."""
        main = soup.find('main')
        if not main:
            return

        # Process text nodes
        for text_node in main.find_all(text=True):
            if not text_node.strip():
                continue

            parent = text_node.parent
            if parent.name in ['a', 'script', 'style', 'code']:
                continue

            text = str(text_node)

            # Check for figure references
            if self.FIGURE_REF_PATTERN.search(text):
                new_content = self._replace_refs_with_links(soup, text, 'figure')
                if new_content:
                    self._replace_text_with_html(text_node, new_content, soup)
                    continue

            # Check for table references
            if self.TABLE_REF_PATTERN.search(text):
                new_content = self._replace_refs_with_links(soup, text, 'table')
                if new_content:
                    self._replace_text_with_html(text_node, new_content, soup)

    def _replace_refs_with_links(self, soup: BeautifulSoup, text: str, ref_type: str) -> Optional[str]:
        """Replace figure/table references with links."""
        if ref_type == 'figure':
            pattern = self.FIGURE_REF_PATTERN
        else:
            pattern = self.TABLE_REF_PATTERN

        def replacer(match):
            ref_word = match.group(1)
            ref_num = match.group(2)

            # Handle ranges (e.g., "Figures 1-3")
            if '-' in ref_num or '–' in ref_num or '—' in ref_num:
                return match.group(0)  # Don't link ranges for now

            return f'<a href="#{ref_type}-{ref_num}">{ref_word} {ref_num}</a>'

        new_text = pattern.sub(replacer, text)
        if new_text != text:
            return new_text
        return None

    def _replace_text_with_html(self, text_node: NavigableString,
                                 html_content: str, soup: BeautifulSoup) -> None:
        """Replace a text node with HTML content."""
        # Parse the new HTML
        new_soup = BeautifulSoup(html_content, 'html.parser')

        # Replace the text node with the parsed content
        parent = text_node.parent
        index = list(parent.children).index(text_node)

        text_node.extract()

        for i, child in enumerate(new_soup.children):
            if hasattr(child, 'extract'):
                parent.insert(index + i, child.extract())
            else:
                parent.insert(index + i, NavigableString(str(child)))

    # =========================================================================
    # Phase 5: CSS Injection
    # =========================================================================

    def _inject_css(self, soup: BeautifulSoup, options: WCAGOptions) -> None:
        """Inject WCAG-compliant CSS styles."""
        # Skip CSS injection if option is disabled (external CSS will be used)
        if not options.inject_css:
            return

        head = soup.find('head')
        if not head:
            head = soup.new_tag('head')
            if soup.html:
                soup.html.insert(0, head)
            else:
                soup.insert(0, head)

        # Build CSS based on options
        css = WCAG_CSS

        if not options.dark_mode:
            # Remove dark mode section
            css = re.sub(r'@media \(prefers-color-scheme: dark\) \{[^}]+\}', '', css)

        if not options.reduced_motion:
            # Remove reduced motion section
            css = re.sub(r'@media \(prefers-reduced-motion: reduce\) \{[^}]+\}', '', css)

        if not options.print_styles:
            # Remove print section
            css = re.sub(r'@media print \{[^}]+\}', '', css)

        # Check for existing wcag styles
        existing_style = head.find('style', {'data-wcag': True})
        if existing_style:
            existing_style.string = css
        else:
            style = soup.new_tag('style')
            style['data-wcag'] = 'true'
            style.string = css
            head.append(style)

    # =========================================================================
    # Phase 6: Final Cleanup
    # =========================================================================

    def _add_accessibility_footer(self, soup: BeautifulSoup) -> None:
        """Add accessibility information footer."""
        body = soup.find('body')
        if not body:
            return

        # Check if footer exists
        footer = soup.find('footer')
        if not footer:
            footer = soup.new_tag('footer')
            footer['role'] = 'contentinfo'
            body.append(footer)

        # Check if accessibility info already exists
        if footer.find(class_='accessibility-info'):
            return

        # Add accessibility info
        acc_div = soup.new_tag('div')
        acc_div['class'] = 'accessibility-info'

        p = soup.new_tag('p')
        p.string = ('This document has been enhanced for accessibility following '
                   'WCAG 2.2 AA guidelines. It includes keyboard navigation, '
                   'screen reader support, and adjustable display settings.')
        acc_div.append(p)

        footer.append(acc_div)


# =============================================================================
# Convenience Functions
# =============================================================================

def enhance_html_wcag(html: str, options: WCAGOptions = None) -> str:
    """
    Convenience function to enhance HTML for WCAG 2.2 AA compliance.

    Args:
        html: Input HTML string
        options: Optional configuration

    Returns:
        Enhanced HTML string
    """
    enhancer = WCAGHTMLEnhancer()
    return enhancer.enhance(html, options)


def enhance_html_file(input_path: str, output_path: str = None,
                      options: WCAGOptions = None) -> str:
    """
    Enhance an HTML file for WCAG 2.2 AA compliance.

    Args:
        input_path: Path to input HTML file
        output_path: Path for output file (default: input_wcag.html)
        options: Optional configuration

    Returns:
        Path to output file
    """
    from pathlib import Path

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.wcag.html')
    else:
        output_path = Path(output_path)

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Enhance
    enhanced = enhance_html_wcag(html, options)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(enhanced)

    return str(output_path)


# =============================================================================
# CLI Support
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python wcag_html_enhancer.py <input.html> [output.html]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    result = enhance_html_file(input_file, output_file)
    print(f"Enhanced HTML written to: {result}")
