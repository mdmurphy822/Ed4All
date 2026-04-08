"""
QTI Assessment XML Validator

Validates IMSCC QTI 1.2 assessment XML files for correct format and Brightspace compatibility.
"""

from typing import List, Optional, Set
from .xml_validator import IMSCCValidator, ValidationResult, ValidationLevel

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    import xml.etree.ElementTree as etree


class QTIValidator(IMSCCValidator):
    """
    Validator for IMSCC QTI 1.2 assessment XML files.

    Validates against namespace: http://www.imsglobal.org/xsd/ims_qtiasiv1p2

    Supported question types (cc_profile values):
    - cc.multiple_choice.v0p1
    - cc.multiple_response.v0p1
    - cc.true_false.v0p1
    - cc.fib.v0p1
    - cc.essay.v0p1
    """

    # QTI namespace
    QTI_NAMESPACE = 'http://www.imsglobal.org/xsd/ims_qtiasiv1p2'

    # Valid assessment profiles
    VALID_ASSESSMENT_PROFILES = [
        'cc.exam.v0p1',
        'cc.quiz.v0p1',
        'cc.survey.v0p1',
        'cc.graded_survey.v0p1',
    ]

    # Valid question type profiles
    VALID_QUESTION_PROFILES = [
        'cc.multiple_choice.v0p1',
        'cc.multiple_response.v0p1',
        'cc.true_false.v0p1',
        'cc.fib.v0p1',
        'cc.essay.v0p1',
    ]

    # Valid assessment types
    VALID_ASSESSMENT_TYPES = [
        'Examination',
        'Assessment',
        'Quiz',
        'Survey',
        'Self-assessment',
        'Formative',
        'Summative',
    ]

    def validate(self, xml_content: str) -> ValidationResult:
        """
        Perform comprehensive validation of QTI assessment XML.

        Args:
            xml_content: QTI XML string

        Returns:
            ValidationResult with all validation findings
        """
        result = ValidationResult(is_valid=True)

        # Validate namespace
        ns_result = self.validate_namespace(xml_content, 'qti')
        result.merge(ns_result)

        # Validate root element
        root_result = self.validate_root_element(xml_content, 'questestinterop')
        result.merge(root_result)

        # Validate assessment structure
        struct_result = self._validate_assessment_structure(xml_content)
        result.merge(struct_result)

        # Validate question items
        items_result = self._validate_question_items(xml_content)
        result.merge(items_result)

        # Validate identifiers
        id_result = self._validate_identifiers(xml_content)
        result.merge(id_result)

        # Schema validation
        schema_result = self.validate_xml_string(xml_content, 'qti')
        result.merge(schema_result)

        return result

    def _validate_assessment_structure(self, xml_content: str) -> ValidationResult:
        """Validate assessment-level structure."""
        result = ValidationResult(is_valid=True)

        if not LXML_AVAILABLE:
            result.add_warning("lxml not available - detailed structure validation skipped")
            return result

        try:
            doc = etree.fromstring(xml_content.encode('utf-8'))
            ns = {'q': self.QTI_NAMESPACE}

            # Check for assessment element
            assessment = doc.find('q:assessment', ns)
            if assessment is None:
                result.add_error("No <assessment> element found", ValidationLevel.CRITICAL)
                return result

            # Check assessment has ident
            ident = assessment.get('ident')
            if not ident or not ident.strip():
                result.add_error("Assessment missing 'ident' attribute", ValidationLevel.HIGH)

            # Check assessment has title
            if not assessment.get('title'):
                result.add_warning("Assessment missing 'title' attribute")

            # Check qtimetadata
            metadata = assessment.find('q:qtimetadata', ns)
            if metadata is not None:
                self._validate_assessment_metadata(metadata, ns, result)
            else:
                result.add_warning("Assessment missing qtimetadata")

            # Check section exists
            sections = assessment.findall('q:section', ns)
            if not sections:
                result.add_error("Assessment has no sections", ValidationLevel.HIGH)

        except Exception as e:
            result.add_error(f"Failed to validate structure: {str(e)}", ValidationLevel.HIGH)

        return result

    def _validate_assessment_metadata(self, metadata, ns: dict, result: ValidationResult):
        """Validate assessment-level metadata fields."""
        fields = {}
        for field in metadata.findall('q:qtimetadatafield', ns):
            label = field.find('q:fieldlabel', ns)
            entry = field.find('q:fieldentry', ns)
            if label is not None and entry is not None:
                fields[label.text] = entry.text

        # Check cc_profile
        if 'cc_profile' in fields:
            if fields['cc_profile'] not in self.VALID_ASSESSMENT_PROFILES:
                result.add_warning(
                    f"Unrecognized assessment cc_profile: '{fields['cc_profile']}'"
                )

        # Check qmd_assessmenttype
        if 'qmd_assessmenttype' in fields:
            if fields['qmd_assessmenttype'] not in self.VALID_ASSESSMENT_TYPES:
                result.add_warning(
                    f"Unrecognized qmd_assessmenttype: '{fields['qmd_assessmenttype']}'"
                )

        # Check cc_maxattempts is numeric
        if 'cc_maxattempts' in fields:
            try:
                int(fields['cc_maxattempts'])
            except ValueError:
                result.add_error(
                    f"cc_maxattempts must be integer, got '{fields['cc_maxattempts']}'",
                    ValidationLevel.MEDIUM
                )

    def _validate_question_items(self, xml_content: str) -> ValidationResult:
        """Validate individual question items."""
        result = ValidationResult(is_valid=True)

        if not LXML_AVAILABLE:
            result.add_warning("lxml not available - detailed item validation skipped")
            return result

        try:
            doc = etree.fromstring(xml_content.encode('utf-8'))
            ns = {'q': self.QTI_NAMESPACE}

            items = doc.findall('.//q:item', ns)
            if not items:
                result.add_warning("No question items found in assessment")
                return result

            for i, item in enumerate(items, 1):
                item_result = self._validate_single_item(item, ns, i)
                result.merge(item_result)

        except Exception as e:
            result.add_error(f"Failed to validate items: {str(e)}", ValidationLevel.HIGH)

        return result

    def _validate_single_item(self, item, ns: dict, item_num: int) -> ValidationResult:
        """Validate a single question item."""
        result = ValidationResult(is_valid=True)
        prefix = f"Item {item_num}"

        # Check ident
        if not item.get('ident'):
            result.add_error(f"{prefix}: Missing 'ident' attribute", ValidationLevel.HIGH)

        # Check itemmetadata
        itemmetadata = item.find('q:itemmetadata', ns)
        if itemmetadata is None:
            result.add_warning(f"{prefix}: Missing itemmetadata")
        else:
            # Get cc_profile
            profile = self._get_item_profile(itemmetadata, ns)
            if profile:
                if profile not in self.VALID_QUESTION_PROFILES:
                    result.add_warning(f"{prefix}: Unrecognized cc_profile '{profile}'")
                else:
                    # Validate structure based on question type
                    self._validate_question_type_structure(item, ns, profile, prefix, result)

            # Validate cc_weighting (points) - should exist and be numeric
            weighting = self._get_item_weighting(itemmetadata, ns)
            if weighting is not None:
                try:
                    float(weighting)
                except ValueError:
                    result.add_error(
                        f"{prefix}: cc_weighting must be numeric, got '{weighting}'",
                        ValidationLevel.MEDIUM
                    )

        # Check presentation
        presentation = item.find('q:presentation', ns)
        if presentation is None:
            result.add_error(f"{prefix}: Missing presentation element", ValidationLevel.HIGH)
        else:
            # Check for material/mattext and validate content
            material = presentation.find('q:material', ns)
            if material is None:
                result.add_warning(f"{prefix}: Missing question material")
            else:
                mattext = material.find('q:mattext', ns)
                if mattext is None:
                    result.add_error(f"{prefix}: Missing mattext in material", ValidationLevel.HIGH)
                elif not mattext.text or not mattext.text.strip():
                    result.add_error(f"{prefix}: Question text is empty", ValidationLevel.HIGH)
                else:
                    # Validate texttype attribute
                    texttype = mattext.get('texttype')
                    if texttype and texttype not in ['text/plain', 'text/html']:
                        result.add_warning(f"{prefix}: mattext has unexpected texttype '{texttype}'")

        # Check resprocessing
        resprocessing = item.find('q:resprocessing', ns)
        if resprocessing is None:
            result.add_warning(f"{prefix}: Missing resprocessing (scoring) element")

        return result

    def _get_item_profile(self, itemmetadata, ns: dict) -> Optional[str]:
        """Extract cc_profile from itemmetadata."""
        qtimetadata = itemmetadata.find('q:qtimetadata', ns)
        if qtimetadata is not None:
            for field in qtimetadata.findall('q:qtimetadatafield', ns):
                label = field.find('q:fieldlabel', ns)
                entry = field.find('q:fieldentry', ns)
                if label is not None and entry is not None:
                    if label.text == 'cc_profile':
                        return entry.text
        return None

    def _get_item_weighting(self, itemmetadata, ns: dict) -> Optional[str]:
        """Extract cc_weighting (points) from itemmetadata."""
        qtimetadata = itemmetadata.find('q:qtimetadata', ns)
        if qtimetadata is not None:
            for field in qtimetadata.findall('q:qtimetadatafield', ns):
                label = field.find('q:fieldlabel', ns)
                entry = field.find('q:fieldentry', ns)
                if label is not None and entry is not None:
                    if label.text == 'cc_weighting':
                        return entry.text
        return None

    def _validate_question_type_structure(self, item, ns: dict, profile: str,
                                          prefix: str, result: ValidationResult):
        """Validate question structure based on type."""
        presentation = item.find('q:presentation', ns)
        if presentation is None:
            return

        if profile in ['cc.multiple_choice.v0p1', 'cc.true_false.v0p1', 'cc.multiple_response.v0p1']:
            # Should have response_lid
            response_lid = presentation.find('q:response_lid', ns)
            if response_lid is None:
                result.add_error(
                    f"{prefix}: {profile} requires response_lid element",
                    ValidationLevel.HIGH
                )
            else:
                # Check rcardinality
                cardinality = response_lid.get('rcardinality', 'Single')
                if profile == 'cc.multiple_response.v0p1' and cardinality != 'Multiple':
                    result.add_error(
                        f"{prefix}: multiple_response requires rcardinality='Multiple'",
                        ValidationLevel.HIGH
                    )
                elif profile in ['cc.multiple_choice.v0p1', 'cc.true_false.v0p1'] and cardinality != 'Single':
                    result.add_warning(
                        f"{prefix}: {profile} typically uses rcardinality='Single'"
                    )

                # Check for render_choice
                render_choice = response_lid.find('q:render_choice', ns)
                if render_choice is None:
                    result.add_error(
                        f"{prefix}: Missing render_choice element",
                        ValidationLevel.HIGH
                    )
                else:
                    # Check response_labels
                    labels = render_choice.findall('q:response_label', ns)
                    if not labels:
                        result.add_error(
                            f"{prefix}: No response_label (answer choices) found",
                            ValidationLevel.HIGH
                        )
                    elif profile == 'cc.true_false.v0p1' and len(labels) != 2:
                        result.add_warning(
                            f"{prefix}: true_false should have exactly 2 choices"
                        )

        elif profile in ['cc.fib.v0p1', 'cc.essay.v0p1']:
            # Should have response_str
            response_str = presentation.find('q:response_str', ns)
            if response_str is None:
                result.add_error(
                    f"{prefix}: {profile} requires response_str element",
                    ValidationLevel.HIGH
                )

    def _validate_identifiers(self, xml_content: str) -> ValidationResult:
        """Validate that all identifiers are unique."""
        result = ValidationResult(is_valid=True)

        if not LXML_AVAILABLE:
            result.add_warning("lxml not available - detailed identifier validation skipped")
            return result

        try:
            doc = etree.fromstring(xml_content.encode('utf-8'))
            ns = {'q': self.QTI_NAMESPACE}

            seen_ids: Set[str] = set()

            # Collect all ident attributes
            for elem in doc.iter():
                ident = elem.get('ident')
                if ident is not None:
                    # Check for empty or whitespace-only idents
                    if not ident.strip():
                        result.add_error(
                            f"Element '{elem.tag}' has empty or whitespace-only ident attribute",
                            ValidationLevel.HIGH
                        )
                        continue
                    if ident in seen_ids:
                        result.add_error(
                            f"Duplicate identifier found: '{ident}'",
                            ValidationLevel.HIGH
                        )
                    seen_ids.add(ident)

        except Exception as e:
            result.add_error(f"Failed to validate identifiers: {str(e)}", ValidationLevel.MEDIUM)

        return result

    def get_resource_type(self) -> str:
        """Return the correct manifest resource type for QTI assessments."""
        return "imsqti_xmlv1p2/imscc_xmlv1p3/assessment"


def validate_qti(xml_content: str) -> ValidationResult:
    """
    Convenience function to validate QTI assessment XML.

    Args:
        xml_content: QTI XML string

    Returns:
        ValidationResult with all validation findings
    """
    validator = QTIValidator()
    return validator.validate(xml_content)
