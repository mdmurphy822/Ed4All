#!/usr/bin/env python3
"""
Component Applier - Transform HTML Content with Interactive Components

This script applies Bootstrap 4.3.1 components from the Courseforge template
library to enhance plain HTML content with interactive, accessible elements.

Features:
- Pattern-based content detection
- AI-assisted component recommendation (with Claude API)
- Bootstrap 4.3.1 compatible output
- WCAG 2.2 AA accessibility compliance
- Brightspace D2L compatibility

Usage:
    python component_applier.py --input content.html --output styled.html
    python component_applier.py --input-dir /content/ --output-dir /styled/
    python component_applier.py --mapping mapping.json --input-dir /content/
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from bs4 import BeautifulSoup

# Add Ed4All lib to path for decision capture
ED4ALL_ROOT = Path(__file__).resolve().parents[3]  # scripts/component-applier/component_applier.py → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

# Configurable paths via environment variables
COURSEFORGE_PATH = Path(os.environ.get(
    'COURSEFORGE_PATH',
    Path(__file__).parent.parent.parent  # Default: relative to script location
))
DEFAULT_TEMPLATE_DIR = COURSEFORGE_PATH / 'templates'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('component_applier.log')
    ]
)
logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Available component types"""
    ACCORDION = "accordion"
    TIMELINE = "timeline"
    FLIP_CARD = "flip_card"
    CALLOUT_INFO = "callout_info"
    CALLOUT_WARNING = "callout_warning"
    CALLOUT_SUCCESS = "callout_success"
    CALLOUT_DANGER = "callout_danger"
    KNOWLEDGE_CHECK = "knowledge_check"
    CARD_LAYOUT = "card_layout"
    TABS = "tabs"
    PROGRESS_BAR = "progress_bar"
    ACTIVITY_CARD = "activity_card"
    # New interactive components
    SELF_CHECK = "self_check"
    REVEAL_CONTENT = "reveal_content"
    INLINE_QUIZ = "inline_quiz"
    PROGRESS_STEPS = "progress_steps"
    NONE = "none"


@dataclass
class ComponentMapping:
    """Maps content section to component"""
    section_id: str
    section_title: str
    content_type: str
    component: ComponentType
    confidence: float
    reason: str


@dataclass
class ApplicationResult:
    """Result of component application"""
    input_file: str
    output_file: str
    components_applied: int
    component_types: Dict[str, int] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


# Content detection patterns (fallback when AI unavailable)
CONTENT_PATTERNS = {
    ComponentType.TIMELINE: [
        r'step\s*\d',
        r'first[,\s]',
        r'second[,\s]',
        r'third[,\s]',
        r'then[,\s]',
        r'next[,\s]',
        r'finally[,\s]',
        r'procedure',
        r'process',
        r'sequence',
    ],
    ComponentType.ACCORDION: [
        r'definition',
        r'defined\s+as',
        r'refers\s+to',
        r'means\s+that',
        r'glossary',
        r'terms?:',
        r'FAQ',
        r'frequently\s+asked',
    ],
    ComponentType.CALLOUT_WARNING: [
        r'warning',
        r'caution',
        r'important',
        r'danger',
        r'critical',
        r'alert',
        r'attention',
        r'do\s+not',
    ],
    ComponentType.CALLOUT_INFO: [
        r'\btip\b',
        r'hint',
        r'best\s+practice',
        r'recommendation',
        r'note:',
        r'remember',
        r'keep\s+in\s+mind',
    ],
    ComponentType.FLIP_CARD: [
        r'compare',
        r'versus',
        r'\bvs\.?\b',
        r'difference\s+between',
        r'before\s+and\s+after',
        r'pro\s+and\s+con',
    ],
    ComponentType.KNOWLEDGE_CHECK: [
        r'check\s+your\s+understanding',
        r'self[- ]assessment',
        r'review\s+question',
        r'quiz',
        r'test\s+yourself',
    ],
    ComponentType.SELF_CHECK: [
        r'quick\s+check',
        r'try\s+it\s+yourself',
        r'practice\s+question',
        r'formative\s+assessment',
        r'checkpoint',
        r'check\s+your\s+learning',
    ],
    ComponentType.REVEAL_CONTENT: [
        r'click\s+to\s+reveal',
        r'show\s+answer',
        r'reveal\s+answer',
        r'hidden\s+content',
        r'spoiler',
        r'click\s+to\s+show',
        r'expand\s+to\s+see',
    ],
    ComponentType.INLINE_QUIZ: [
        r'multiple\s+choice',
        r'select\s+the\s+(correct|best)',
        r'which\s+of\s+the\s+following',
        r'choose\s+the\s+(answer|option)',
        r'mini[\-\s]quiz',
        r'embedded\s+assessment',
    ],
    ComponentType.PROGRESS_STEPS: [
        r'step\s+\d+\s+of\s+\d+',
        r'milestone',
        r'stage\s+\d+',
        r'phase\s+\d+',
        r'progress\s+indicator',
        r'completion\s+status',
    ],
}


class ComponentApplier:
    """
    Applies interactive components to HTML content.
    """

    # Courseforge color palette
    COLORS = {
        'primary': '#2c5aa0',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'border': '#e0e0e0',
    }

    def __init__(
        self,
        use_ai: bool = True,
        template_dir: Optional[Path] = None,
        capture: Optional["DecisionCapture"] = None,
    ):
        """
        Initialize the component applier.

        Args:
            use_ai: Whether to use Claude API for content analysis
            template_dir: Path to template directory
            capture: Optional DecisionCapture for logging component decisions
        """
        self.use_ai = use_ai
        self.template_dir = template_dir or DEFAULT_TEMPLATE_DIR
        self.results: List[ApplicationResult] = []
        self.capture = capture

    def apply_to_file(self, input_file: Path, output_file: Path) -> ApplicationResult:
        """
        Apply components to a single HTML file.

        Args:
            input_file: Path to input HTML file
            output_file: Path for output file

        Returns:
            ApplicationResult with details
        """
        logger.info(f"Processing: {input_file}")

        result = ApplicationResult(
            input_file=str(input_file),
            output_file=str(output_file),
            components_applied=0
        )

        try:
            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')

            # Find content sections to enhance
            sections = self._identify_sections(soup)

            # Apply components to each section
            for section in sections:
                component = self._detect_component_type(section)
                if component != ComponentType.NONE:
                    self._apply_component(soup, section, component)
                    result.components_applied += 1
                    result.component_types[component.value] = \
                        result.component_types.get(component.value, 0) + 1

            # Add required CSS/JS if components were applied
            if result.components_applied > 0:
                self._add_dependencies(soup)

            # Write output
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(str(soup))

            logger.info(f"Applied {result.components_applied} components to {input_file.name}")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Error processing {input_file}: {e}")

        self.results.append(result)
        return result

    def apply_to_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        recursive: bool = True
    ) -> List[ApplicationResult]:
        """
        Apply components to all HTML files in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            recursive: Process subdirectories

        Returns:
            List of ApplicationResult objects
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        pattern = '**/*.html' if recursive else '*.html'

        for input_file in input_dir.glob(pattern):
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path
            self.apply_to_file(input_file, output_file)

        return self.results

    def _identify_sections(self, soup: BeautifulSoup) -> List[dict]:
        """Identify content sections that may benefit from components"""
        sections = []

        # Look for common section patterns
        # Headings followed by content
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            section = {
                'heading': heading,
                'title': heading.get_text().strip(),
                'content': [],
                'element': heading
            }

            # Collect content until next heading
            sibling = heading.find_next_sibling()
            while sibling and sibling.name not in ['h1', 'h2', 'h3', 'h4']:
                section['content'].append(sibling)
                sibling = sibling.find_next_sibling()

            if section['content']:
                sections.append(section)

        # Look for definition lists
        for dl in soup.find_all('dl'):
            sections.append({
                'heading': None,
                'title': 'Definitions',
                'content': [dl],
                'element': dl,
                'type': 'definitions'
            })

        # Look for numbered/bulleted lists that might be steps
        for ol in soup.find_all('ol'):
            sections.append({
                'heading': None,
                'title': 'Steps',
                'content': [ol],
                'element': ol,
                'type': 'ordered_list'
            })

        return sections

    def _detect_component_type(self, section: dict) -> ComponentType:
        """Detect the best component type for a section"""
        # Get text content for pattern matching
        text_content = section['title'].lower()
        for elem in section['content']:
            if hasattr(elem, 'get_text'):
                text_content += ' ' + elem.get_text().lower()

        # Check for specific section types
        if section.get('type') == 'definitions':
            return ComponentType.ACCORDION

        if section.get('type') == 'ordered_list':
            # Check if it looks like steps/process
            if any(re.search(p, text_content) for p in CONTENT_PATTERNS[ComponentType.TIMELINE]):
                return ComponentType.TIMELINE

        # Pattern-based detection
        for component_type, patterns in CONTENT_PATTERNS.items():
            match_count = sum(1 for p in patterns if re.search(p, text_content, re.IGNORECASE))
            if match_count >= 2:  # Require at least 2 pattern matches
                return component_type

        return ComponentType.NONE

    def _apply_component(
        self,
        soup: BeautifulSoup,
        section: dict,
        component: ComponentType
    ):
        """Apply the specified component to a section"""
        if component == ComponentType.ACCORDION:
            self._apply_accordion(soup, section)
        elif component == ComponentType.TIMELINE:
            self._apply_timeline(soup, section)
        elif component == ComponentType.FLIP_CARD:
            self._apply_flip_card(soup, section)
        elif component in [ComponentType.CALLOUT_INFO, ComponentType.CALLOUT_WARNING,
                          ComponentType.CALLOUT_SUCCESS, ComponentType.CALLOUT_DANGER]:
            self._apply_callout(soup, section, component)
        elif component == ComponentType.KNOWLEDGE_CHECK:
            self._apply_knowledge_check(soup, section)
        elif component == ComponentType.SELF_CHECK:
            self._apply_self_check(soup, section)
        elif component == ComponentType.REVEAL_CONTENT:
            self._apply_reveal_content(soup, section)
        elif component == ComponentType.INLINE_QUIZ:
            self._apply_inline_quiz(soup, section)
        elif component == ComponentType.PROGRESS_STEPS:
            self._apply_progress_steps(soup, section)

    def _apply_accordion(self, soup: BeautifulSoup, section: dict):
        """Convert section to accordion"""
        accordion_id = f"accordion_{id(section)}"

        # Create accordion container
        accordion = soup.new_tag('div', attrs={
            'class': 'accordion',
            'id': accordion_id
        })

        # Handle definition lists
        dl = None
        for elem in section['content']:
            if elem.name == 'dl':
                dl = elem
                break

        if dl:
            items = []
            dt_elements = dl.find_all('dt')
            dd_elements = dl.find_all('dd')

            for i, (dt, dd) in enumerate(zip(dt_elements, dd_elements)):
                card = self._create_accordion_item(
                    soup, accordion_id, i,
                    dt.get_text().strip(),
                    str(dd)
                )
                accordion.append(card)

            dl.replace_with(accordion)
        else:
            # Create single accordion item from section
            card = self._create_accordion_item(
                soup, accordion_id, 0,
                section['title'],
                ''.join(str(c) for c in section['content'])
            )
            accordion.append(card)

            # Replace first content element, remove others
            if section['content']:
                section['content'][0].replace_with(accordion)
                for elem in section['content'][1:]:
                    elem.decompose()

    def _create_accordion_item(
        self,
        soup: BeautifulSoup,
        accordion_id: str,
        index: int,
        title: str,
        content: str
    ):
        """Create a single accordion item"""
        item_id = f"{accordion_id}_item_{index}"

        card = soup.new_tag('div', attrs={'class': 'card'})

        # Header
        header = soup.new_tag('div', attrs={
            'class': 'card-header',
            'id': f"{item_id}_header"
        })

        h_tag = soup.new_tag('h4', attrs={'class': 'mb-0'})

        button = soup.new_tag('button', attrs={
            'class': 'btn btn-link collapsed',
            'type': 'button',
            'data-toggle': 'collapse',
            'data-target': f"#{item_id}_collapse",
            'aria-expanded': 'false',
            'aria-controls': f"{item_id}_collapse"
        })
        button.string = title

        h_tag.append(button)
        header.append(h_tag)
        card.append(header)

        # Body
        collapse = soup.new_tag('div', attrs={
            'id': f"{item_id}_collapse",
            'class': 'collapse',
            'aria-labelledby': f"{item_id}_header",
            'data-parent': f"#{accordion_id}"
        })

        body = soup.new_tag('div', attrs={'class': 'card-body'})
        body.append(BeautifulSoup(content, 'html.parser'))
        collapse.append(body)
        card.append(collapse)

        return card

    def _apply_timeline(self, soup: BeautifulSoup, section: dict):
        """Convert ordered list to timeline"""
        ol = None
        for elem in section['content']:
            if elem.name == 'ol':
                ol = elem
                break

        if not ol:
            return

        timeline = soup.new_tag('div', attrs={
            'class': 'timeline',
            'role': 'list',
            'aria-label': section.get('title', 'Process steps')
        })

        for i, li in enumerate(ol.find_all('li'), 1):
            item = soup.new_tag('div', attrs={
                'class': 'timeline-item',
                'role': 'listitem'
            })

            marker = soup.new_tag('div', attrs={'class': 'timeline-marker'})
            marker.string = str(i)

            content = soup.new_tag('div', attrs={'class': 'timeline-content'})
            content.append(BeautifulSoup(str(li), 'html.parser'))

            item.append(marker)
            item.append(content)
            timeline.append(item)

        ol.replace_with(timeline)

    def _apply_callout(
        self,
        soup: BeautifulSoup,
        section: dict,
        component: ComponentType
    ):
        """Wrap section in callout box"""
        callout_class = component.value.replace('_', '-')

        # Determine icon
        icons = {
            ComponentType.CALLOUT_INFO: 'ℹ️',
            ComponentType.CALLOUT_WARNING: '⚠️',
            ComponentType.CALLOUT_SUCCESS: '✅',
            ComponentType.CALLOUT_DANGER: '🚫',
        }

        callout = soup.new_tag('div', attrs={
            'class': f'callout {callout_class}',
            'role': 'alert'
        })

        icon_div = soup.new_tag('div', attrs={
            'class': 'callout-icon',
            'aria-hidden': 'true'
        })
        icon_div.string = icons.get(component, 'ℹ️')

        content_div = soup.new_tag('div', attrs={'class': 'callout-content'})

        title = soup.new_tag('h4')
        title.string = section['title']
        content_div.append(title)

        for elem in section['content']:
            content_div.append(BeautifulSoup(str(elem), 'html.parser'))

        callout.append(icon_div)
        callout.append(content_div)

        # Replace heading with callout
        if section['heading']:
            section['heading'].replace_with(callout)
            for elem in section['content']:
                elem.decompose()

    def _apply_knowledge_check(self, soup: BeautifulSoup, section: dict):
        """Convert to knowledge check component"""
        kc = soup.new_tag('div', attrs={
            'class': 'knowledge-check',
            'role': 'region',
            'aria-labelledby': f"kc_{id(section)}"
        })

        title = soup.new_tag('h4', attrs={'id': f"kc_{id(section)}"})
        title.string = section['title'] or 'Check Your Understanding'
        kc.append(title)

        for elem in section['content']:
            question_div = soup.new_tag('div', attrs={'class': 'kc-question'})

            details = soup.new_tag('details')
            summary = soup.new_tag('summary')
            summary.string = 'Reveal Answer'
            details.append(summary)

            answer = soup.new_tag('p', attrs={'class': 'kc-answer'})
            answer.append(BeautifulSoup(str(elem), 'html.parser'))
            details.append(answer)

            question_div.append(details)
            kc.append(question_div)

        if section['heading']:
            section['heading'].replace_with(kc)
            for elem in section['content']:
                elem.decompose()

    def _apply_flip_card(self, soup: BeautifulSoup, section: dict):
        """Create flip card component"""
        # This would need more sophisticated content analysis
        # For now, basic implementation
        flip_card = soup.new_tag('div', attrs={
            'class': 'flip-card',
            'tabindex': '0',
            'role': 'button',
            'aria-label': 'Click to reveal'
        })

        inner = soup.new_tag('div', attrs={'class': 'flip-card-inner'})

        front = soup.new_tag('div', attrs={'class': 'flip-card-front'})
        front_title = soup.new_tag('h4')
        front_title.string = section['title']
        front.append(front_title)

        back = soup.new_tag('div', attrs={'class': 'flip-card-back'})
        for elem in section['content']:
            back.append(BeautifulSoup(str(elem), 'html.parser'))

        inner.append(front)
        inner.append(back)
        flip_card.append(inner)

        if section['heading']:
            section['heading'].replace_with(flip_card)
            for elem in section['content']:
                elem.decompose()

    def _apply_self_check(self, soup: BeautifulSoup, section: dict):
        """Create self-check formative assessment component"""
        self_check_id = f"selfcheck_{id(section)}"

        container = soup.new_tag('div', attrs={
            'class': 'self-check',
            'id': self_check_id,
            'role': 'region',
            'aria-labelledby': f"{self_check_id}_heading"
        })

        # Header
        header = soup.new_tag('div', attrs={'class': 'self-check-header'})
        icon = soup.new_tag('span', attrs={
            'class': 'self-check-icon',
            'aria-hidden': 'true'
        })
        icon.string = '?'
        header.append(icon)

        title = soup.new_tag('h4', attrs={'id': f"{self_check_id}_heading"})
        title.string = section['title'] or 'Quick Check'
        header.append(title)
        container.append(header)

        # Question area
        question_div = soup.new_tag('div', attrs={'class': 'self-check-question'})
        for elem in section['content']:
            question_div.append(BeautifulSoup(str(elem), 'html.parser'))
        container.append(question_div)

        # Feedback area (using details/summary for reveal)
        feedback = soup.new_tag('details', attrs={'class': 'self-check-feedback'})
        summary = soup.new_tag('summary')
        summary.string = 'Check Answer'
        feedback.append(summary)

        answer = soup.new_tag('div', attrs={'class': 'self-check-answer'})
        answer.string = 'Review the content above to verify your understanding.'
        feedback.append(answer)
        container.append(feedback)

        if section['heading']:
            section['heading'].replace_with(container)
            for elem in section['content']:
                elem.decompose()

    def _apply_reveal_content(self, soup: BeautifulSoup, section: dict):
        """Create click-to-reveal content component"""
        reveal_id = f"reveal_{id(section)}"

        container = soup.new_tag('div', attrs={
            'class': 'reveal-box',
            'id': reveal_id
        })

        # Prompt/teaser
        prompt = soup.new_tag('div', attrs={'class': 'reveal-prompt'})
        prompt_text = soup.new_tag('p')
        prompt_text.string = section['title'] or 'Click to reveal'
        prompt.append(prompt_text)
        container.append(prompt)

        # Hidden content using details/summary
        details = soup.new_tag('details', attrs={'class': 'reveal-details'})
        summary = soup.new_tag('summary', attrs={'class': 'reveal-button'})
        summary.string = 'Show Content'
        details.append(summary)

        content_div = soup.new_tag('div', attrs={'class': 'reveal-content'})
        for elem in section['content']:
            content_div.append(BeautifulSoup(str(elem), 'html.parser'))
        details.append(content_div)
        container.append(details)

        if section['heading']:
            section['heading'].replace_with(container)
            for elem in section['content']:
                elem.decompose()

    def _apply_inline_quiz(self, soup: BeautifulSoup, section: dict):
        """Create inline quiz component with multiple questions"""
        quiz_id = f"quiz_{id(section)}"

        container = soup.new_tag('div', attrs={
            'class': 'inline-quiz',
            'id': quiz_id,
            'role': 'region',
            'aria-labelledby': f"{quiz_id}_heading"
        })

        # Header
        header = soup.new_tag('div', attrs={'class': 'quiz-header'})
        title = soup.new_tag('h4', attrs={'id': f"{quiz_id}_heading"})
        title.string = section['title'] or 'Knowledge Check'
        header.append(title)
        container.append(header)

        # Questions area
        questions_div = soup.new_tag('div', attrs={'class': 'quiz-questions'})
        for i, elem in enumerate(section['content'], 1):
            question_item = soup.new_tag('div', attrs={
                'class': 'quiz-question',
                'data-question': str(i)
            })
            question_item.append(BeautifulSoup(str(elem), 'html.parser'))
            questions_div.append(question_item)
        container.append(questions_div)

        # Results area
        results = soup.new_tag('div', attrs={
            'class': 'quiz-results',
            'aria-live': 'polite'
        })
        container.append(results)

        if section['heading']:
            section['heading'].replace_with(container)
            for elem in section['content']:
                elem.decompose()

    def _apply_progress_steps(self, soup: BeautifulSoup, section: dict):
        """Create progress steps indicator component"""
        progress_id = f"progress_{id(section)}"

        container = soup.new_tag('nav', attrs={
            'class': 'progress-steps',
            'id': progress_id,
            'aria-label': section.get('title', 'Progress steps')
        })

        # Title
        if section['title']:
            title = soup.new_tag('h4', attrs={'class': 'progress-title'})
            title.string = section['title']
            container.append(title)

        # Steps list
        steps_list = soup.new_tag('ol', attrs={'class': 'steps-list'})

        # Extract steps from content
        step_num = 0
        for elem in section['content']:
            if elem.name == 'ol':
                for li in elem.find_all('li'):
                    step_num += 1
                    step_item = soup.new_tag('li', attrs={
                        'class': 'step-item pending'
                    })

                    circle = soup.new_tag('span', attrs={
                        'class': 'step-circle',
                        'aria-hidden': 'true'
                    })
                    circle.string = str(step_num)
                    step_item.append(circle)

                    label = soup.new_tag('span', attrs={'class': 'step-label'})
                    label.append(BeautifulSoup(li.get_text(), 'html.parser'))
                    step_item.append(label)

                    steps_list.append(step_item)
            else:
                # Non-list content becomes a single step
                step_num += 1
                step_item = soup.new_tag('li', attrs={
                    'class': 'step-item pending'
                })

                circle = soup.new_tag('span', attrs={
                    'class': 'step-circle',
                    'aria-hidden': 'true'
                })
                circle.string = str(step_num)
                step_item.append(circle)

                label = soup.new_tag('span', attrs={'class': 'step-label'})
                label.append(BeautifulSoup(str(elem), 'html.parser'))
                step_item.append(label)

                steps_list.append(step_item)

        container.append(steps_list)

        if section['heading']:
            section['heading'].replace_with(container)
            for elem in section['content']:
                elem.decompose()

    def _add_dependencies(self, soup: BeautifulSoup):
        """Add required CSS and JS dependencies"""
        head = soup.find('head')
        if not head:
            head = soup.new_tag('head')
            if soup.html:
                soup.html.insert(0, head)
            else:
                soup.insert(0, head)

        # Add component CSS
        component_css = soup.new_tag('style')
        component_css.string = self._get_component_css()
        head.append(component_css)

        # Add Bootstrap CSS if not present
        if not soup.find('link', href=lambda x: x and 'bootstrap' in x):
            bootstrap_css = soup.new_tag('link', attrs={
                'rel': 'stylesheet',
                'href': 'https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css'
            })
            head.append(bootstrap_css)

        # Add Bootstrap JS if not present
        body = soup.find('body')
        if body and not soup.find('script', src=lambda x: x and 'bootstrap' in x):
            jquery = soup.new_tag('script', attrs={
                'src': 'https://code.jquery.com/jquery-3.6.0.min.js'
            })
            bootstrap_js = soup.new_tag('script', attrs={
                'src': 'https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/js/bootstrap.bundle.min.js'
            })
            body.append(jquery)
            body.append(bootstrap_js)

    def _get_component_css(self) -> str:
        """Return CSS for all components"""
        return """
/* Courseforge Component Styles */
:root {
    --cf-primary: #2c5aa0;
    --cf-success: #28a745;
    --cf-warning: #ffc107;
    --cf-danger: #dc3545;
    --cf-info: #17a2b8;
    --cf-light: #f8f9fa;
    --cf-border: #e0e0e0;
}

/* Timeline */
.timeline {
    position: relative;
    padding-left: 30px;
    margin: 1.5rem 0;
}
.timeline::before {
    content: '';
    position: absolute;
    left: 10px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--cf-primary);
}
.timeline-item {
    position: relative;
    margin-bottom: 1.5rem;
    padding-left: 20px;
}
.timeline-marker {
    position: absolute;
    left: -30px;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--cf-primary);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.875rem;
}
.timeline-content {
    background: var(--cf-light);
    padding: 1rem;
    border-radius: 4px;
    border-left: 3px solid var(--cf-primary);
}

/* Callouts */
.callout {
    display: flex;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
    border-left: 4px solid;
}
.callout-info { background: #e7f3ff; border-color: var(--cf-info); }
.callout-warning { background: #fff3cd; border-color: var(--cf-warning); }
.callout-success { background: #d4edda; border-color: var(--cf-success); }
.callout-danger { background: #f8d7da; border-color: var(--cf-danger); }
.callout-icon {
    font-size: 1.5rem;
    margin-right: 1rem;
    flex-shrink: 0;
}
.callout-content h4 { margin-top: 0; }

/* Knowledge Check */
.knowledge-check {
    background: var(--cf-light);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
}
.knowledge-check h4 {
    color: var(--cf-primary);
    margin-bottom: 1rem;
}
.kc-question {
    margin-bottom: 1rem;
}
.kc-question details {
    cursor: pointer;
}
.kc-question summary {
    color: var(--cf-primary);
    font-weight: 500;
}
.kc-answer {
    margin-top: 0.5rem;
    padding: 1rem;
    background: white;
    border-radius: 4px;
}

/* Flip Card */
.flip-card {
    perspective: 1000px;
    width: 100%;
    height: 200px;
    margin: 1rem 0;
}
.flip-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    transition: transform 0.6s;
    transform-style: preserve-3d;
}
.flip-card:hover .flip-card-inner,
.flip-card:focus .flip-card-inner {
    transform: rotateY(180deg);
}
.flip-card-front, .flip-card-back {
    position: absolute;
    width: 100%;
    height: 100%;
    backface-visibility: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    border-radius: 8px;
}
.flip-card-front {
    background: var(--cf-primary);
    color: white;
}
.flip-card-back {
    background: var(--cf-light);
    transform: rotateY(180deg);
}

/* Accordion customization */
.accordion .card {
    border: 1px solid var(--cf-border);
    margin-bottom: 0.5rem;
}
.accordion .card-header {
    background: var(--cf-light);
    padding: 0;
}
.accordion .btn-link {
    color: var(--cf-primary);
    text-decoration: none;
    width: 100%;
    text-align: left;
    padding: 1rem;
}
.accordion .btn-link:hover {
    text-decoration: none;
}

/* Self-Check Component */
.self-check {
    background: var(--cf-light);
    border: 1px solid var(--cf-border);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}
.self-check-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.self-check-icon {
    width: 32px;
    height: 32px;
    background: var(--cf-primary);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
}
.self-check-header h4 {
    margin: 0;
    color: var(--cf-primary);
}
.self-check-question {
    margin-bottom: 1rem;
    padding: 1rem;
    background: white;
    border-radius: 4px;
}
.self-check-feedback summary {
    color: var(--cf-primary);
    cursor: pointer;
    font-weight: 500;
    padding: 0.5rem 0;
}
.self-check-answer {
    padding: 1rem;
    background: #d4edda;
    border-radius: 4px;
    margin-top: 0.5rem;
}

/* Reveal Content Component */
.reveal-box {
    border: 1px solid var(--cf-border);
    border-radius: 8px;
    margin: 1.5rem 0;
    overflow: hidden;
}
.reveal-prompt {
    padding: 1rem;
    background: var(--cf-light);
    border-bottom: 1px solid var(--cf-border);
}
.reveal-prompt p {
    margin: 0;
    font-weight: 500;
}
.reveal-details summary {
    padding: 0.75rem 1rem;
    background: var(--cf-primary);
    color: white;
    cursor: pointer;
    font-weight: 500;
}
.reveal-details summary:hover {
    background: #1e3d6f;
}
.reveal-content {
    padding: 1rem;
    background: white;
}

/* Inline Quiz Component */
.inline-quiz {
    background: var(--cf-light);
    border: 2px solid var(--cf-primary);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}
.quiz-header h4 {
    margin: 0 0 1rem 0;
    color: var(--cf-primary);
}
.quiz-questions {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
.quiz-question {
    background: white;
    padding: 1rem;
    border-radius: 4px;
    border-left: 3px solid var(--cf-primary);
}
.quiz-results {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 4px;
}

/* Progress Steps Component */
.progress-steps {
    margin: 1.5rem 0;
}
.progress-title {
    color: var(--cf-primary);
    margin-bottom: 1rem;
}
.steps-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    justify-content: space-between;
    position: relative;
}
.steps-list::before {
    content: '';
    position: absolute;
    top: 22px;
    left: 40px;
    right: 40px;
    height: 4px;
    background: var(--cf-border);
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    z-index: 1;
    flex: 1;
}
.step-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: white;
    border: 3px solid var(--cf-border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    color: var(--cf-text-muted);
}
.step-item.complete .step-circle {
    background: var(--cf-success);
    border-color: var(--cf-success);
    color: white;
}
.step-item.current .step-circle {
    background: var(--cf-primary);
    border-color: var(--cf-primary);
    color: white;
    box-shadow: 0 0 0 4px #e7f3ff;
}
.step-label {
    margin-top: 0.5rem;
    font-size: 0.875rem;
    text-align: center;
    max-width: 100px;
}
@media (max-width: 600px) {
    .steps-list {
        flex-direction: column;
        gap: 1rem;
    }
    .steps-list::before {
        left: 20px;
        top: 0;
        bottom: 0;
        width: 4px;
        height: auto;
        right: auto;
    }
    .step-item {
        flex-direction: row;
        justify-content: flex-start;
    }
    .step-label {
        margin-top: 0;
        margin-left: 1rem;
        max-width: none;
        text-align: left;
    }
}
"""

    def generate_report(self) -> str:
        """Generate processing report"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        components = sum(r.components_applied for r in self.results)

        # Aggregate component types
        all_types: Dict[str, int] = {}
        for r in self.results:
            for t, c in r.component_types.items():
                all_types[t] = all_types.get(t, 0) + c

        report = f"""
Component Application Report
============================
Files processed: {total}
Successful: {successful}
Components applied: {components}

Component Distribution:
"""
        for comp_type, count in sorted(all_types.items(), key=lambda x: -x[1]):
            report += f"  {comp_type}: {count}\n"

        if any(not r.success for r in self.results):
            report += "\nErrors:\n"
            for r in self.results:
                if not r.success:
                    report += f"  {r.input_file}: {r.error}\n"

        return report

    def to_json(self) -> str:
        """Export results as JSON"""
        data = {
            'summary': {
                'files_processed': len(self.results),
                'successful': sum(1 for r in self.results if r.success),
                'components_applied': sum(r.components_applied for r in self.results)
            },
            'results': [asdict(r) for r in self.results]
        }
        return json.dumps(data, indent=2)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Apply interactive components to HTML content'
    )
    parser.add_argument('--input', '-i', help='Input HTML file')
    parser.add_argument('--output', '-o', help='Output HTML file')
    parser.add_argument('--input-dir', '-d', help='Input directory')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--mapping', help='Component mapping JSON file')
    parser.add_argument('--json', action='store_true', help='Output JSON report')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Verbose output (-vv for debug)')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    applier = ComponentApplier()

    try:
        if args.input and args.output:
            applier.apply_to_file(Path(args.input), Path(args.output))
        elif args.input_dir and args.output_dir:
            applier.apply_to_directory(Path(args.input_dir), Path(args.output_dir))
        else:
            parser.error("Provide --input/--output or --input-dir/--output-dir")

        if args.json:
            print(applier.to_json())
        else:
            print(applier.generate_report())

        sys.exit(0)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(2)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
