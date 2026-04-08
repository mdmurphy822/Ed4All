#!/usr/bin/env python3
"""
QTI Assessment Validator
Validates QTI 1.2 assessment XML against IMS specifications

Checks:
- QTI 1.2 namespace declaration
- questestinterop root element structure
- assessment structure with valid identifiers
- qtimetadata fields (cc_profile, qmd_assessmenttype)
- section/item structure
- Response processing validity
- D2L/Brightspace compatibility
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    severity: IssueSeverity
    code: str
    message: str
    element: Optional[str] = None
    line: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class QuestionInfo:
    """Information about a single question/item"""
    identifier: str
    title: Optional[str] = None
    question_type: Optional[str] = None
    points: Optional[float] = None
    has_response_processing: bool = False


@dataclass
class ValidationResult:
    """Result of QTI assessment validation"""
    file_path: str
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    assessment_title: Optional[str] = None
    assessment_type: Optional[str] = None
    cc_profile: Optional[str] = None
    question_count: int = 0
    total_points: float = 0.0
    question_types: Dict[str, int] = field(default_factory=dict)
    questions: List[QuestionInfo] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.HIGH)


class QTIAssessmentValidator:
    """Validates QTI 1.2 assessment XML against IMS specifications"""

    # QTI 1.2 namespace
    QTI_NAMESPACE = 'http://www.imsglobal.org/xsd/ims_qtiasiv1p2'

    # Valid CC profiles
    VALID_CC_PROFILES = [
        'cc.exam.v0p1',
        'cc.quiz.v0p1',
        'cc.survey.v0p1',
        'cc.graded_survey.v0p1',
    ]

    # Valid assessment types for Brightspace
    VALID_ASSESSMENT_TYPES = [
        'Examination',
        'Assessment',
        'Quiz',
        'Survey',
        'Self-assessment',
        'Formative',
        'Summative',
    ]

    # Valid question types (cardinality + response type combinations)
    VALID_QUESTION_TYPES = {
        'multiple_choice': ('Single', 'Lid'),
        'multiple_response': ('Multiple', 'Lid'),
        'true_false': ('Single', 'Lid'),
        'short_answer': ('Single', 'Str'),
        'essay': ('Single', 'Str'),
        'fill_in_blank': ('Ordered', 'Str'),
        'matching': ('Multiple', 'Lid'),
        'numerical': ('Single', 'Num'),
    }

    # D2L-specific metadata fields
    D2L_METADATA_FIELDS = [
        'd2l_2p0:resource_type',
        'd2l_2p0:points_possible',
    ]

    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.questions: List[QuestionInfo] = []

    def validate_assessment(self, qti_path: Path) -> ValidationResult:
        """
        Validate a QTI 1.2 assessment XML file.

        Args:
            qti_path: Path to QTI XML file

        Returns:
            ValidationResult with findings
        """
        self.issues = []
        self.questions = []
        assessment_title = None
        assessment_type = None
        cc_profile = None
        question_count = 0
        total_points = 0.0
        question_types: Dict[str, int] = {}

        if not qti_path.exists():
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="QTI001",
                message=f"QTI file not found: {qti_path}",
            ))
            return ValidationResult(
                file_path=str(qti_path),
                valid=False,
                issues=self.issues,
            )

        try:
            # Parse XML
            tree = ET.parse(qti_path)
            root = tree.getroot()

            # Validate root element
            self._validate_root_element(root)

            # Extract and validate namespace
            ns = self._extract_namespace(root)

            # Find assessment element
            assessment = self._find_assessment(root, ns)
            if assessment is not None:
                assessment_title = assessment.get('title')

                # Validate assessment structure
                self._validate_assessment_structure(assessment, ns)

                # Extract and validate metadata
                cc_profile, assessment_type = self._validate_metadata(assessment, ns)

                # Validate sections
                question_count, total_points, question_types = self._validate_sections(
                    assessment, ns
                )

                # Validate response processing
                self._validate_response_processing(assessment, ns)

                # Check D2L compatibility
                self._check_d2l_compatibility(assessment, ns)

        except ET.ParseError as e:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="QTI002",
                message=f"XML parsing error: {e}",
                suggestion="Ensure the file is well-formed XML"
            ))
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="QTI003",
                message=f"Unexpected error: {e}",
            ))

        return ValidationResult(
            file_path=str(qti_path),
            valid=self._calculate_validity(),
            issues=self.issues,
            assessment_title=assessment_title,
            assessment_type=assessment_type,
            cc_profile=cc_profile,
            question_count=question_count,
            total_points=total_points,
            question_types=question_types,
            questions=self.questions,
        )

    def _calculate_validity(self) -> bool:
        """Determine if assessment is valid based on issues"""
        critical_high = sum(1 for i in self.issues
                          if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH])
        return critical_high == 0

    def _validate_root_element(self, root: ET.Element) -> None:
        """Validate the questestinterop root element"""
        tag_name = root.tag.split('}')[-1] if '}' in root.tag else root.tag

        if tag_name != 'questestinterop':
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="QTI010",
                message=f"Root element must be 'questestinterop', found: {tag_name}",
                suggestion="Ensure the file starts with <questestinterop>"
            ))

    def _extract_namespace(self, root: ET.Element) -> str:
        """Extract and validate the QTI namespace"""
        tag = root.tag
        ns = ''
        if tag.startswith('{'):
            ns = tag[1:tag.index('}')]

            if 'qti' not in ns.lower() and 'ims' not in ns.lower():
                self.issues.append(ValidationIssue(
                    severity=IssueSeverity.MEDIUM,
                    code="QTI011",
                    message=f"Non-standard QTI namespace: {ns}",
                    suggestion=f"Consider using: {self.QTI_NAMESPACE}"
                ))

        return ns

    def _find_assessment(self, root: ET.Element, ns: str) -> Optional[ET.Element]:
        """Find the assessment element"""
        ns_prefix = f'{{{ns}}}' if ns else ''

        # Try with namespace
        assessment = root.find(f'.//{ns_prefix}assessment')
        if assessment is not None:
            return assessment

        # Try without namespace
        assessment = root.find('.//assessment')
        if assessment is not None:
            return assessment

        # Try as direct child
        for child in root:
            tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag_name == 'assessment':
                return child

        self.issues.append(ValidationIssue(
            severity=IssueSeverity.CRITICAL,
            code="QTI020",
            message="No assessment element found",
            suggestion="Add an <assessment> element inside <questestinterop>"
        ))
        return None

    def _validate_assessment_structure(self, assessment: ET.Element, ns: str) -> None:
        """Validate basic assessment structure"""
        # Check for identifier
        ident = assessment.get('ident')
        if not ident:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="QTI021",
                message="Assessment missing 'ident' attribute",
                suggestion="Add ident attribute: <assessment ident=\"unique_id\">"
            ))
        elif not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.-]*$', ident):
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.MEDIUM,
                code="QTI022",
                message=f"Assessment identifier may cause issues: {ident}",
                element=ident,
                suggestion="Use alphanumeric characters, underscores, hyphens only"
            ))

        # Check for title
        title = assessment.get('title')
        if not title:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.MEDIUM,
                code="QTI023",
                message="Assessment missing 'title' attribute",
                suggestion="Add title attribute for better identification"
            ))

    def _validate_metadata(self, assessment: ET.Element, ns: str) -> Tuple[Optional[str], Optional[str]]:
        """Validate qtimetadata section"""
        ns_prefix = f'{{{ns}}}' if ns else ''
        cc_profile = None
        assessment_type = None

        # Find qtimetadata
        metadata = None
        for elem in assessment.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_name == 'qtimetadata':
                metadata = elem
                break

        if metadata is None:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.MEDIUM,
                code="QTI030",
                message="Missing qtimetadata section",
                suggestion="Add <qtimetadata> with cc_profile and qmd_assessmenttype"
            ))
            return None, None

        # Parse metadata fields
        for metadatafield in metadata.iter():
            tag_name = metadatafield.tag.split('}')[-1] if '}' in metadatafield.tag else metadatafield.tag
            if tag_name == 'qtimetadatafield':
                label_elem = None
                entry_elem = None

                for child in metadatafield:
                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child_tag == 'fieldlabel':
                        label_elem = child
                    elif child_tag == 'fieldentry':
                        entry_elem = child

                if label_elem is not None and entry_elem is not None:
                    label = label_elem.text or ''
                    entry = entry_elem.text or ''

                    if label == 'cc_profile':
                        cc_profile = entry
                        if entry not in self.VALID_CC_PROFILES:
                            self.issues.append(ValidationIssue(
                                severity=IssueSeverity.MEDIUM,
                                code="QTI031",
                                message=f"Non-standard cc_profile: {entry}",
                                element=entry,
                                suggestion=f"Valid profiles: {', '.join(self.VALID_CC_PROFILES)}"
                            ))

                    elif label == 'qmd_assessmenttype':
                        assessment_type = entry
                        if entry not in self.VALID_ASSESSMENT_TYPES:
                            self.issues.append(ValidationIssue(
                                severity=IssueSeverity.LOW,
                                code="QTI032",
                                message=f"Non-standard assessment type: {entry}",
                                element=entry,
                            ))

        if cc_profile is None:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.MEDIUM,
                code="QTI033",
                message="Missing cc_profile in metadata",
                suggestion="Add cc_profile field (e.g., cc.exam.v0p1)"
            ))

        return cc_profile, assessment_type

    def _validate_sections(self, assessment: ET.Element, ns: str) -> Tuple[int, float, Dict[str, int]]:
        """Validate section and item structure"""
        question_count = 0
        total_points = 0.0
        question_types: Dict[str, int] = {}

        # Find all sections
        sections = []
        for elem in assessment.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_name == 'section':
                sections.append(elem)

        if not sections:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="QTI040",
                message="No section elements found in assessment",
                suggestion="Add at least one <section> containing <item> elements"
            ))
            return 0, 0.0, {}

        # Validate each section
        for section in sections:
            section_ident = section.get('ident', 'unknown')

            # Find items in section
            items = []
            for elem in section.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag_name == 'item':
                    items.append(elem)

            if not items:
                self.issues.append(ValidationIssue(
                    severity=IssueSeverity.MEDIUM,
                    code="QTI041",
                    message=f"Section '{section_ident}' contains no items",
                    element=section_ident,
                ))

            # Validate each item
            for item in items:
                item_info = self._validate_item(item, ns)
                if item_info:
                    self.questions.append(item_info)
                    question_count += 1

                    if item_info.points:
                        total_points += item_info.points

                    if item_info.question_type:
                        question_types[item_info.question_type] = \
                            question_types.get(item_info.question_type, 0) + 1

        return question_count, total_points, question_types

    def _validate_item(self, item: ET.Element, ns: str) -> Optional[QuestionInfo]:
        """Validate a single item/question"""
        ident = item.get('ident')
        title = item.get('title')

        if not ident:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="QTI050",
                message="Item missing 'ident' attribute",
                suggestion="Add unique ident attribute to each item"
            ))
            return None

        # Determine question type from response_lid or response_str
        question_type = self._detect_question_type(item, ns)

        # Check for point value
        points = self._extract_points(item, ns)

        # Check for response processing
        has_resprocessing = False
        for elem in item.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_name == 'resprocessing':
                has_resprocessing = True
                break

        if not has_resprocessing:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.MEDIUM,
                code="QTI051",
                message=f"Item '{ident}' missing resprocessing",
                element=ident,
                suggestion="Add <resprocessing> for answer scoring"
            ))

        # Check for presentation
        has_presentation = False
        for elem in item.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_name == 'presentation':
                has_presentation = True
                break

        if not has_presentation:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="QTI052",
                message=f"Item '{ident}' missing presentation",
                element=ident,
                suggestion="Add <presentation> with question content"
            ))

        return QuestionInfo(
            identifier=ident,
            title=title,
            question_type=question_type,
            points=points,
            has_response_processing=has_resprocessing,
        )

    def _detect_question_type(self, item: ET.Element, ns: str) -> Optional[str]:
        """Detect the question type from response elements"""
        for elem in item.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if tag_name == 'response_lid':
                rcardinality = elem.get('rcardinality', 'Single')
                if rcardinality == 'Multiple':
                    return 'multiple_response'
                # Check if true/false
                response_labels = list(elem.iter())
                label_count = sum(1 for e in response_labels
                                 if (e.tag.split('}')[-1] if '}' in e.tag else e.tag) == 'response_label')
                if label_count == 2:
                    return 'true_false'
                return 'multiple_choice'

            elif tag_name == 'response_str':
                return 'short_answer'

            elif tag_name == 'response_num':
                return 'numerical'

            elif tag_name == 'response_grp':
                return 'matching'

        return None

    def _extract_points(self, item: ET.Element, ns: str) -> Optional[float]:
        """Extract point value from item"""
        for elem in item.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            # Check decvar for maxvalue
            if tag_name == 'decvar':
                maxvalue = elem.get('maxvalue')
                if maxvalue:
                    try:
                        return float(maxvalue)
                    except ValueError:
                        pass

            # Check metadata for points
            if tag_name == 'fieldlabel' and elem.text == 'cc_maxattempts':
                # Look for sibling fieldentry
                parent = elem.getparent() if hasattr(elem, 'getparent') else None
                if parent is not None:
                    for sibling in parent:
                        sib_tag = sibling.tag.split('}')[-1] if '}' in sibling.tag else sibling.tag
                        if sib_tag == 'fieldentry' and sibling.text:
                            try:
                                return float(sibling.text)
                            except ValueError:
                                pass

        return None

    def _validate_response_processing(self, assessment: ET.Element, ns: str) -> None:
        """Validate response processing elements"""
        for elem in assessment.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if tag_name == 'resprocessing':
                # Check for outcomes
                has_outcomes = False
                has_respcondition = False

                for child in elem.iter():
                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if child_tag == 'outcomes':
                        has_outcomes = True
                    elif child_tag == 'respcondition':
                        has_respcondition = True

                if not has_outcomes:
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.MEDIUM,
                        code="QTI060",
                        message="resprocessing missing outcomes element",
                        suggestion="Add <outcomes> with <decvar> for scoring"
                    ))

                if not has_respcondition:
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.MEDIUM,
                        code="QTI061",
                        message="resprocessing missing respcondition elements",
                        suggestion="Add <respcondition> for each possible response"
                    ))

    def _check_d2l_compatibility(self, assessment: ET.Element, ns: str) -> None:
        """Check for D2L/Brightspace specific compatibility"""
        # Check for overly complex structures
        nested_sections = 0
        for elem in assessment.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_name == 'section':
                # Count nested sections
                parent_sections = 0
                parent = elem
                while parent is not None:
                    parent = self._get_parent(assessment, parent)
                    if parent is not None:
                        p_tag = parent.tag.split('}')[-1] if '}' in parent.tag else parent.tag
                        if p_tag == 'section':
                            parent_sections += 1
                if parent_sections > 1:
                    nested_sections += 1

        if nested_sections > 0:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.LOW,
                code="QTI070",
                message=f"Deeply nested sections may not import correctly ({nested_sections} found)",
                suggestion="Flatten section structure for better D2L compatibility"
            ))

    def _get_parent(self, root: ET.Element, target: ET.Element) -> Optional[ET.Element]:
        """Get parent element (ElementTree doesn't have parent references)"""
        for parent in root.iter():
            for child in parent:
                if child is target:
                    return parent
        return None


def main():
    """CLI entry point"""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Validate QTI 1.2 assessment XML files'
    )
    parser.add_argument('-i', '--input', required=True, help='Path to QTI XML file')
    parser.add_argument('-j', '--json', action='store_true', help='Output as JSON')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Verbose output (-vv for debug)')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    args = parser.parse_args()

    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)

    validator = QTIAssessmentValidator()
    result = validator.validate_assessment(Path(args.input))

    if args.json:
        output = {
            'file_path': result.file_path,
            'valid': result.valid,
            'assessment_title': result.assessment_title,
            'assessment_type': result.assessment_type,
            'cc_profile': result.cc_profile,
            'question_count': result.question_count,
            'total_points': result.total_points,
            'question_types': result.question_types,
            'questions': [
                {
                    'identifier': q.identifier,
                    'title': q.title,
                    'question_type': q.question_type,
                    'points': q.points,
                    'has_response_processing': q.has_response_processing,
                }
                for q in result.questions
            ],
            'issues': [
                {
                    'severity': i.severity.value,
                    'code': i.code,
                    'message': i.message,
                    'element': i.element,
                    'suggestion': i.suggestion,
                }
                for i in result.issues
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"File: {result.file_path}")
        print(f"Valid: {result.valid}")
        print(f"Title: {result.assessment_title or 'Unknown'}")
        print(f"Type: {result.assessment_type or 'Unknown'}")
        print(f"CC Profile: {result.cc_profile or 'Unknown'}")
        print(f"Questions: {result.question_count}")
        print(f"Total Points: {result.total_points}")
        if result.question_types:
            print(f"\nQuestion Types:")
            for qtype, count in result.question_types.items():
                print(f"  {qtype}: {count}")
        print(f"\nIssues Found: {len(result.issues)}")
        for issue in result.issues:
            print(f"  [{issue.severity.value.upper()}] {issue.code}: {issue.message}")
            if issue.element:
                print(f"    Element: {issue.element}")
            if issue.suggestion:
                print(f"    Suggestion: {issue.suggestion}")

    return 0 if result.valid else 1


if __name__ == '__main__':
    exit(main())
