# Schema Validators Package
# IMSCC and QTI validation for Courseforge

"""
Schema validation modules for IMSCC packages:

- namespace_validator: Validates XML namespace declarations
- resource_reference_validator: Ensures all resource references resolve
- imscc_manifest_validator: Validates manifest against IMS CC specs
- qti_assessment_validator: Validates QTI 1.2 assessment XML

Usage:
    from schema_validators import IMSCCManifestValidator, QTIAssessmentValidator

    manifest_validator = IMSCCManifestValidator()
    result = manifest_validator.validate_manifest(Path('imsmanifest.xml'))

    qti_validator = QTIAssessmentValidator()
    result = qti_validator.validate_assessment(Path('quiz.xml'))
"""

from .namespace_validator import NamespaceValidator
from .resource_reference_validator import ResourceReferenceValidator
from .imscc_manifest_validator import IMSCCManifestValidator
from .qti_assessment_validator import QTIAssessmentValidator

__all__ = [
    'NamespaceValidator',
    'ResourceReferenceValidator',
    'IMSCCManifestValidator',
    'QTIAssessmentValidator',
]
