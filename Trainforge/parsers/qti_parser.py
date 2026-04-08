"""
QTI (Question and Test Interoperability) Parser

Parses QTI 1.2 format assessment files from IMSCC packages.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QTIChoice:
    """A choice/option in a multiple choice question."""
    id: str
    text: str
    is_correct: bool = False


@dataclass
class QTIQuestion:
    """A parsed QTI question item."""
    id: str
    type: str  # multiple_choice, true_false, short_answer, essay, matching, fill_in_blank
    stem: str  # Question text
    choices: List[QTIChoice] = field(default_factory=list)
    correct_response: Optional[str] = None
    feedback: Optional[str] = None
    points: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QTIAssessment:
    """A parsed QTI assessment."""
    id: str
    title: str
    questions: List[QTIQuestion] = field(default_factory=list)
    time_limit: Optional[int] = None  # minutes
    max_attempts: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QTIParser:
    """
    Parser for QTI 1.2 assessment files.

    Usage:
        parser = QTIParser()
        assessment = parser.parse_file("/path/to/assessment.xml")
        for question in assessment.questions:
            print(f"{question.type}: {question.stem}")
    """

    # QTI namespaces
    NAMESPACES = {
        'qti': 'http://www.imsglobal.org/xsd/ims_qtiasiv1p2'
    }

    def parse_file(self, xml_path: str) -> QTIAssessment:
        """
        Parse a QTI XML file.

        Args:
            xml_path: Path to QTI XML file

        Returns:
            Parsed QTIAssessment

        Raises:
            FileNotFoundError: If XML file doesn't exist
            ValueError: If XML is invalid or malformed
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except FileNotFoundError:
            raise FileNotFoundError(f"QTI file not found: {xml_path}") from None
        except ET.ParseError as e:
            raise ValueError(f"Invalid QTI XML in {xml_path}: {e}") from e

        return self._parse_assessment(root)

    def parse_string(self, xml_content: str) -> QTIAssessment:
        """
        Parse QTI XML from string.

        Args:
            xml_content: QTI XML content

        Returns:
            Parsed QTIAssessment

        Raises:
            ValueError: If XML is invalid or malformed
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid QTI XML: {e}") from e

        return self._parse_assessment(root)

    def _parse_assessment(self, root: ET.Element) -> QTIAssessment:
        """Parse assessment element."""
        # Get assessment identifier and title
        assessment_id = root.get('ident', 'unknown')
        title = root.get('title', 'Untitled Assessment')

        # Look for title in metadata
        title_elem = root.find('.//title') or root.find('.//qtimetadata//fieldentry')
        if title_elem is not None and title_elem.text:
            title = title_elem.text

        # Parse questions
        questions = []
        for item in root.iter():
            if 'item' in item.tag.lower() and item.get('ident'):
                question = self._parse_item(item)
                if question:
                    questions.append(question)

        return QTIAssessment(
            id=assessment_id,
            title=title,
            questions=questions
        )

    def _parse_item(self, item: ET.Element) -> Optional[QTIQuestion]:
        """Parse a single question item."""
        item_id = item.get('ident', 'unknown')

        # Get question text (stem)
        stem = self._extract_stem(item)
        if not stem:
            return None

        # Determine question type and parse accordingly
        qtype, choices, correct = self._parse_response(item)

        return QTIQuestion(
            id=item_id,
            type=qtype,
            stem=stem,
            choices=choices,
            correct_response=correct,
            points=self._extract_points(item)
        )

    def _extract_stem(self, item: ET.Element) -> str:
        """Extract question stem/text."""
        # Look for mattext elements with null checks
        for mattext in item.iter():
            if 'mattext' in mattext.tag.lower():
                if mattext.text is not None and mattext.text.strip():
                    return mattext.text.strip()

        # Try presentation/material
        for material in item.iter():
            if 'material' in material.tag.lower():
                text_parts = []
                for child in material.iter():
                    if child.text is not None and child.text.strip():
                        text_parts.append(child.text.strip())
                if text_parts:
                    return ' '.join(text_parts)

        return ""

    def _parse_response(self, item: ET.Element) -> tuple:
        """
        Parse response type, choices, and correct answer.

        Returns:
            Tuple of (question_type, choices, correct_response)
        """
        choices = []
        correct = None
        qtype = "short_answer"  # default

        # Look for response_lid (multiple choice)
        for resp in item.iter():
            if 'response_lid' in resp.tag.lower():
                qtype = "multiple_choice"

                # Parse choices with null checks
                for label in resp.iter():
                    if 'response_label' in label.tag.lower():
                        choice_id = label.get('ident', '')
                        choice_text = ""

                        for mattext in label.iter():
                            if 'mattext' in mattext.tag.lower():
                                if mattext.text is not None:
                                    choice_text = mattext.text.strip()
                                    break

                        if choice_id and choice_text:
                            choices.append(QTIChoice(
                                id=choice_id,
                                text=choice_text
                            ))

        # Look for correct answer in resprocessing
        for resprocessing in item.iter():
            if 'resprocessing' in resprocessing.tag.lower():
                for respcondition in resprocessing.iter():
                    if 'respcondition' in respcondition.tag.lower():
                        # Check for setvar with SCORE
                        setvar = None
                        varequal = None

                        for child in respcondition.iter():
                            if 'setvar' in child.tag.lower():
                                setvar = child
                            if 'varequal' in child.tag.lower():
                                varequal = child

                        if setvar is not None and varequal is not None:
                            try:
                                # Safe text extraction with null checks
                                setvar_text = setvar.text if setvar.text is not None else "0"
                                score = float(setvar_text)
                                if score > 0:
                                    correct = varequal.text if varequal.text is not None else ""
                                    # Mark correct choice
                                    for choice in choices:
                                        if choice.id == correct:
                                            choice.is_correct = True
                            except (ValueError, TypeError):
                                pass

        # Check for true/false
        if len(choices) == 2:
            texts = {c.text.lower() for c in choices}
            if texts == {'true', 'false'} or texts == {'yes', 'no'}:
                qtype = "true_false"

        # Check for essay/extended response
        for resp in item.iter():
            if 'response_str' in resp.tag.lower():
                qtype = "essay"
            elif 'response_fib' in resp.tag.lower():
                qtype = "fill_in_blank"

        return qtype, choices, correct

    def _extract_points(self, item: ET.Element) -> float:
        """Extract point value for question."""
        # Look for point value in various locations
        for decvar in item.iter():
            if 'decvar' in decvar.tag.lower():
                maxvalue = decvar.get('maxvalue')
                if maxvalue:
                    try:
                        return float(maxvalue)
                    except ValueError:
                        pass

        return 1.0  # default
