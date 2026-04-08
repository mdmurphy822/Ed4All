"""
Math Detection and MathML Conversion Module

Detects mathematical content from various sources (LaTeX, Unicode symbols, patterns)
and converts them to accessible MathML format.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MathBlock:
    """Represents a detected mathematical expression."""
    source_type: str      # 'latex', 'unicode', 'image'
    raw_content: str      # Original content
    mathml: str           # Converted MathML
    fallback_text: str    # Accessible text description
    start_pos: int = 0    # Position in original text
    end_pos: int = 0      # End position in original text


class MathDetector:
    """Detect mathematical content from text."""

    # LaTeX patterns
    LATEX_INLINE = re.compile(r'\$([^$]+)\$')
    LATEX_DISPLAY = re.compile(r'\$\$([^$]+)\$\$')
    LATEX_PAREN = re.compile(r'\\\((.+?)\\\)')
    LATEX_BRACKET = re.compile(r'\\\[(.+?)\\\]', re.DOTALL)

    # Currency pattern to avoid false positives
    CURRENCY_PATTERN = re.compile(r'\$\s*[\d,]+(?:\.\d{2})?\s*(?:USD|dollars?|cents?|million|billion|k|K|M|B)?')

    # Unicode math symbol ranges
    MATH_SYMBOLS = set('=+-×÷^_∑∫∏√∞≤≥≠≈∈∉⊂⊃∪∩αβγδεθλμπσφωΔΩ∀∃∂∇')

    # Common LaTeX commands that indicate math
    LATEX_COMMANDS = re.compile(
        r'\\(?:frac|sqrt|sum|int|prod|lim|infty|alpha|beta|gamma|delta|'
        r'theta|lambda|mu|pi|sigma|phi|omega|Sigma|Delta|Omega|partial|'
        r'nabla|forall|exists|in|notin|subset|supset|cup|cap|cdot|times|'
        r'leq|geq|neq|approx|equiv|rightarrow|leftarrow|vec|hat|bar|dot)'
    )

    def detect_in_text(self, text: str) -> List[MathBlock]:
        """
        Detect all mathematical expressions in text.

        Args:
            text: Input text to scan for math

        Returns:
            List of MathBlock objects with detected math
        """
        math_blocks = []

        # Detect display math first ($$...$$) to avoid conflicts
        math_blocks.extend(self._detect_display_latex(text))

        # Detect inline math ($...$) avoiding currency
        math_blocks.extend(self._detect_inline_latex(text))

        # Detect \(...\) and \[...\] patterns
        math_blocks.extend(self._detect_latex_delimiters(text))

        # Detect Unicode math expressions
        math_blocks.extend(self._detect_unicode_math(text))

        # Sort by position and remove overlaps
        math_blocks.sort(key=lambda x: x.start_pos)
        math_blocks = self._remove_overlaps(math_blocks)

        return math_blocks

    def _detect_display_latex(self, text: str) -> List[MathBlock]:
        """Detect display math ($$...$$)."""
        blocks = []
        for match in self.LATEX_DISPLAY.finditer(text):
            content = match.group(1).strip()
            if content and self._is_likely_math(content):
                blocks.append(MathBlock(
                    source_type='latex_display',
                    raw_content=content,
                    mathml='',  # Will be filled by converter
                    fallback_text=self._generate_fallback(content),
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        return blocks

    def _detect_inline_latex(self, text: str) -> List[MathBlock]:
        """Detect inline math ($...$) avoiding currency patterns."""
        blocks = []

        # First, mark currency positions to skip
        currency_ranges = set()
        for match in self.CURRENCY_PATTERN.finditer(text):
            for i in range(match.start(), match.end()):
                currency_ranges.add(i)

        for match in self.LATEX_INLINE.finditer(text):
            # Skip if overlaps with currency
            if match.start() in currency_ranges:
                continue

            content = match.group(1).strip()
            if content and self._is_likely_math(content):
                blocks.append(MathBlock(
                    source_type='latex_inline',
                    raw_content=content,
                    mathml='',
                    fallback_text=self._generate_fallback(content),
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        return blocks

    def _detect_latex_delimiters(self, text: str) -> List[MathBlock]:
        """Detect \(...\) and \[...\] patterns."""
        blocks = []

        for match in self.LATEX_PAREN.finditer(text):
            content = match.group(1).strip()
            if content:
                blocks.append(MathBlock(
                    source_type='latex_inline',
                    raw_content=content,
                    mathml='',
                    fallback_text=self._generate_fallback(content),
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        for match in self.LATEX_BRACKET.finditer(text):
            content = match.group(1).strip()
            if content:
                blocks.append(MathBlock(
                    source_type='latex_display',
                    raw_content=content,
                    mathml='',
                    fallback_text=self._generate_fallback(content),
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        return blocks

    def _detect_unicode_math(self, text: str) -> List[MathBlock]:
        """Detect mathematical expressions using Unicode symbols."""
        blocks = []

        # Pattern for expressions with math symbols
        # Look for sequences containing math symbols with surrounding context
        pattern = re.compile(r'[a-zA-Z0-9\s]*[∑∫∏√∞≤≥≠≈∈∉⊂⊃∪∩αβγδεθλμπσφωΔΩ∀∃∂∇][^.!?\n]*')

        for match in pattern.finditer(text):
            content = match.group().strip()
            # Only include if it has significant math content
            math_char_count = sum(1 for c in content if c in self.MATH_SYMBOLS)
            if math_char_count >= 1 and len(content) < 200:
                blocks.append(MathBlock(
                    source_type='unicode',
                    raw_content=content,
                    mathml='',
                    fallback_text=content,  # Unicode is already readable
                    start_pos=match.start(),
                    end_pos=match.end()
                ))

        return blocks

    def _is_likely_math(self, content: str) -> bool:
        """Check if content is likely mathematical (not just text in dollar signs)."""
        # Contains LaTeX commands
        if self.LATEX_COMMANDS.search(content):
            return True

        # Contains subscript/superscript notation
        if '_' in content or '^' in content:
            return True

        # Contains math operators with variables
        if re.search(r'[a-z]\s*[+\-*/=<>]\s*[a-z0-9]', content, re.IGNORECASE):
            return True

        # Contains Unicode math symbols
        if any(c in self.MATH_SYMBOLS for c in content):
            return True

        # Contains numeric expressions
        if re.search(r'\d+\s*[+\-*/^]\s*\d+', content):
            return True

        return False

    def _generate_fallback(self, latex: str) -> str:
        """Generate accessible fallback text from LaTeX."""
        text = latex

        # Replace common LaTeX commands with words
        replacements = [
            (r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1 over \2'),
            (r'\\sqrt\{([^}]*)\}', r'square root of \1'),
            (r'\\sum', 'sum'),
            (r'\\int', 'integral'),
            (r'\\infty', 'infinity'),
            (r'\\alpha', 'alpha'),
            (r'\\beta', 'beta'),
            (r'\\gamma', 'gamma'),
            (r'\\delta', 'delta'),
            (r'\\pi', 'pi'),
            (r'\\theta', 'theta'),
            (r'\\leq', 'less than or equal to'),
            (r'\\geq', 'greater than or equal to'),
            (r'\\neq', 'not equal to'),
            (r'\\approx', 'approximately equal to'),
            (r'\\times', 'times'),
            (r'\\cdot', 'dot'),
            (r'\\rightarrow', 'right arrow'),
            (r'\\leftarrow', 'left arrow'),
            (r'\^', ' to the power of '),
            (r'_', ' subscript '),
        ]

        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

        # Clean up remaining backslashes and braces
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'[{}]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _remove_overlaps(self, blocks: List[MathBlock]) -> List[MathBlock]:
        """Remove overlapping math blocks, keeping larger ones."""
        if not blocks:
            return blocks

        result = [blocks[0]]
        for block in blocks[1:]:
            last = result[-1]
            if block.start_pos >= last.end_pos:
                result.append(block)
            elif block.end_pos - block.start_pos > last.end_pos - last.start_pos:
                result[-1] = block

        return result


class MathMLConverter:
    """Convert mathematical content to MathML."""

    def __init__(self):
        """Initialize converter, checking for latex2mathml availability."""
        self._latex2mathml = None
        try:
            import latex2mathml.converter
            self._latex2mathml = latex2mathml.converter
            logger.debug("latex2mathml library available")
        except ImportError:
            logger.warning("latex2mathml not installed. Using fallback conversion.")

    def convert(self, math_block: MathBlock) -> MathBlock:
        """
        Convert a math block to MathML.

        Args:
            math_block: MathBlock with raw_content to convert

        Returns:
            MathBlock with mathml field populated
        """
        if math_block.source_type in ('latex_inline', 'latex_display'):
            math_block.mathml = self.latex_to_mathml(
                math_block.raw_content,
                display=math_block.source_type == 'latex_display'
            )
        elif math_block.source_type == 'unicode':
            math_block.mathml = self.unicode_to_mathml(math_block.raw_content)

        return math_block

    def latex_to_mathml(self, latex: str, display: bool = False) -> str:
        """
        Convert LaTeX to MathML.

        Args:
            latex: LaTeX expression (without delimiters)
            display: True for display math, False for inline

        Returns:
            MathML string
        """
        if self._latex2mathml:
            try:
                mathml = self._latex2mathml.convert(latex)
                # Add display attribute if needed
                if display:
                    mathml = mathml.replace('<math>', '<math display="block">')
                return mathml
            except Exception as e:
                logger.warning(f"latex2mathml conversion failed: {e}")

        # Fallback: manual conversion for common patterns
        return self._manual_latex_to_mathml(latex, display)

    def unicode_to_mathml(self, text: str) -> str:
        """
        Convert Unicode math to MathML.

        Args:
            text: Text containing Unicode math symbols

        Returns:
            MathML string
        """
        # Map Unicode symbols to MathML operators
        symbol_map = {
            '∑': '<mo>&#x2211;</mo>',
            '∫': '<mo>&#x222B;</mo>',
            '∏': '<mo>&#x220F;</mo>',
            '√': '<mo>&#x221A;</mo>',
            '∞': '<mo>&#x221E;</mo>',
            '≤': '<mo>&#x2264;</mo>',
            '≥': '<mo>&#x2265;</mo>',
            '≠': '<mo>&#x2260;</mo>',
            '≈': '<mo>&#x2248;</mo>',
            '∈': '<mo>&#x2208;</mo>',
            '∉': '<mo>&#x2209;</mo>',
            '⊂': '<mo>&#x2282;</mo>',
            '⊃': '<mo>&#x2283;</mo>',
            '∪': '<mo>&#x222A;</mo>',
            '∩': '<mo>&#x2229;</mo>',
            '∀': '<mo>&#x2200;</mo>',
            '∃': '<mo>&#x2203;</mo>',
            '∂': '<mo>&#x2202;</mo>',
            '∇': '<mo>&#x2207;</mo>',
            'α': '<mi>&#x03B1;</mi>',
            'β': '<mi>&#x03B2;</mi>',
            'γ': '<mi>&#x03B3;</mi>',
            'δ': '<mi>&#x03B4;</mi>',
            'ε': '<mi>&#x03B5;</mi>',
            'θ': '<mi>&#x03B8;</mi>',
            'λ': '<mi>&#x03BB;</mi>',
            'μ': '<mi>&#x03BC;</mi>',
            'π': '<mi>&#x03C0;</mi>',
            'σ': '<mi>&#x03C3;</mi>',
            'φ': '<mi>&#x03C6;</mi>',
            'ω': '<mi>&#x03C9;</mi>',
            'Δ': '<mi>&#x0394;</mi>',
            'Ω': '<mi>&#x03A9;</mi>',
        }

        # Build MathML content
        mrow_content = []
        for char in text:
            if char in symbol_map:
                mrow_content.append(symbol_map[char])
            elif char.isalpha():
                mrow_content.append(f'<mi>{char}</mi>')
            elif char.isdigit():
                mrow_content.append(f'<mn>{char}</mn>')
            elif char in '+-*/=<>':
                mrow_content.append(f'<mo>{char}</mo>')
            elif char not in ' \t\n':
                mrow_content.append(f'<mo>{char}</mo>')

        mathml = f'<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>{"".join(mrow_content)}</mrow></math>'
        return mathml

    def _manual_latex_to_mathml(self, latex: str, display: bool = False) -> str:
        """
        Manual fallback conversion for common LaTeX patterns.

        Args:
            latex: LaTeX expression
            display: True for display math

        Returns:
            MathML string
        """
        display_attr = ' display="block"' if display else ''

        # Start building MathML
        content = latex

        # Handle fractions: \frac{a}{b}
        content = re.sub(
            r'\\frac\{([^}]*)\}\{([^}]*)\}',
            r'<mfrac><mrow>\1</mrow><mrow>\2</mrow></mfrac>',
            content
        )

        # Handle superscripts: x^{n} and x^n
        content = re.sub(
            r'([a-zA-Z0-9])?\^\{([^}]*)\}',
            lambda m: f'<msup><mi>{m.group(1) or ""}</mi><mrow>{m.group(2)}</mrow></msup>',
            content
        )
        content = re.sub(
            r'([a-zA-Z0-9])\^([a-zA-Z0-9])',
            r'<msup><mi>\1</mi><mi>\2</mi></msup>',
            content
        )

        # Handle subscripts: x_{n} and x_n
        content = re.sub(
            r'([a-zA-Z0-9])?_\{([^}]*)\}',
            lambda m: f'<msub><mi>{m.group(1) or ""}</mi><mrow>{m.group(2)}</mrow></msub>',
            content
        )
        content = re.sub(
            r'([a-zA-Z0-9])_([a-zA-Z0-9])',
            r'<msub><mi>\1</mi><mi>\2</mi></msub>',
            content
        )

        # Handle square root: \sqrt{x}
        content = re.sub(
            r'\\sqrt\{([^}]*)\}',
            r'<msqrt><mrow>\1</mrow></msqrt>',
            content
        )

        # Handle operators
        operator_map = {
            r'\\sum': '<mo>&#x2211;</mo>',
            r'\\int': '<mo>&#x222B;</mo>',
            r'\\prod': '<mo>&#x220F;</mo>',
            r'\\times': '<mo>&#x00D7;</mo>',
            r'\\cdot': '<mo>&#x22C5;</mo>',
            r'\\leq': '<mo>&#x2264;</mo>',
            r'\\geq': '<mo>&#x2265;</mo>',
            r'\\neq': '<mo>&#x2260;</mo>',
            r'\\approx': '<mo>&#x2248;</mo>',
            r'\\rightarrow': '<mo>&#x2192;</mo>',
            r'\\leftarrow': '<mo>&#x2190;</mo>',
            r'\\infty': '<mo>&#x221E;</mo>',
            r'\\pm': '<mo>&#x00B1;</mo>',
        }
        for pattern, replacement in operator_map.items():
            content = re.sub(pattern, replacement, content)

        # Handle Greek letters
        greek_map = {
            'alpha': '&#x03B1;', 'beta': '&#x03B2;', 'gamma': '&#x03B3;',
            'delta': '&#x03B4;', 'epsilon': '&#x03B5;', 'theta': '&#x03B8;',
            'lambda': '&#x03BB;', 'mu': '&#x03BC;', 'pi': '&#x03C0;',
            'sigma': '&#x03C3;', 'phi': '&#x03C6;', 'omega': '&#x03C9;',
            'Sigma': '&#x03A3;', 'Delta': '&#x0394;', 'Omega': '&#x03A9;',
        }
        for name, entity in greek_map.items():
            content = re.sub(rf'\\{name}', f'<mi>{entity}</mi>', content)

        # Wrap remaining letters as <mi> and numbers as <mn>
        def wrap_chars(text):
            result = []
            i = 0
            while i < len(text):
                if text[i] == '<':
                    # Skip existing tags
                    end = text.find('>', i)
                    if end != -1:
                        # Find closing tag
                        tag_content = text[i:end+1]
                        result.append(tag_content)
                        # If not self-closing, find content and closing tag
                        if not tag_content.endswith('/>'):
                            close_start = text.find('</', end)
                            if close_start != -1:
                                close_end = text.find('>', close_start)
                                result.append(text[end+1:close_end+1])
                                i = close_end + 1
                                continue
                        i = end + 1
                        continue
                elif text[i].isalpha():
                    result.append(f'<mi>{text[i]}</mi>')
                elif text[i].isdigit():
                    # Collect consecutive digits
                    num = text[i]
                    while i + 1 < len(text) and text[i + 1].isdigit():
                        i += 1
                        num += text[i]
                    result.append(f'<mn>{num}</mn>')
                elif text[i] in '+-*/=':
                    result.append(f'<mo>{text[i]}</mo>')
                elif text[i] not in ' \t\n{}\\':
                    result.append(text[i])
                i += 1
            return ''.join(result)

        # Clean up remaining LaTeX commands
        content = re.sub(r'\\[a-zA-Z]+', '', content)
        content = re.sub(r'[{}]', '', content)

        # Only wrap if we haven't already fully converted
        if '<m' not in content:
            content = wrap_chars(content)

        mathml = f'<math xmlns="http://www.w3.org/1998/Math/MathML"{display_attr}><mrow>{content}</mrow></math>'
        return mathml

    def create_accessible_fallback(self, content: str, fallback_text: str) -> str:
        """
        Create accessible HTML fallback when MathML conversion fails.

        Args:
            content: Original math content
            fallback_text: Human-readable description

        Returns:
            Accessible HTML span
        """
        escaped_content = content.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        return f'<span role="math" aria-label="{fallback_text}" class="math-fallback">{escaped_content}</span>'


def process_text_for_math(text: str) -> Tuple[str, List[MathBlock], int]:
    """
    Process text to detect and convert mathematical expressions.

    Args:
        text: Input text

    Returns:
        Tuple of (processed_text_with_placeholders, math_blocks, count)
    """
    detector = MathDetector()
    converter = MathMLConverter()

    math_blocks = detector.detect_in_text(text)

    # Convert each block
    for block in math_blocks:
        converter.convert(block)

    # Replace math in text with placeholders (for later HTML insertion)
    processed_text = text
    offset = 0
    for i, block in enumerate(math_blocks):
        placeholder = f'[[MATH_PLACEHOLDER_{i}]]'
        start = block.start_pos + offset
        end = block.end_pos + offset
        processed_text = processed_text[:start] + placeholder + processed_text[end:]
        offset += len(placeholder) - (block.end_pos - block.start_pos)

    return processed_text, math_blocks, len(math_blocks)
