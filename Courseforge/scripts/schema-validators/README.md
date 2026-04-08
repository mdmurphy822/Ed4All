# Schema Validators

Validation modules for IMSCC packages ensuring compliance with IMS Common Cartridge specifications and Brightspace/D2L compatibility.

## Overview

This package provides four specialized validators for comprehensive IMSCC package validation:

| Validator | Purpose |
|-----------|---------|
| `NamespaceValidator` | Validates XML namespace declarations |
| `ResourceReferenceValidator` | Ensures all resource references resolve |
| `IMSCCManifestValidator` | Validates manifest against IMS CC specs |
| `QTIAssessmentValidator` | Validates QTI 1.2 assessment XML |

## Installation

The validators are pure Python with no external dependencies beyond the standard library.

```python
from schema_validators import (
    NamespaceValidator,
    ResourceReferenceValidator,
    IMSCCManifestValidator,
    QTIAssessmentValidator,
)
```

## Quick Start

### Validate a Manifest

```python
from pathlib import Path
from schema_validators import IMSCCManifestValidator

validator = IMSCCManifestValidator()
result = validator.validate_manifest(Path('imsmanifest.xml'))

if result.valid:
    print(f"Manifest valid! Version: {result.imscc_version}")
    print(f"Resources: {result.resource_count}")
else:
    for issue in result.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")
```

### Validate a QTI Assessment

```python
from pathlib import Path
from schema_validators import QTIAssessmentValidator

validator = QTIAssessmentValidator()
result = validator.validate_assessment(Path('quiz.xml'))

print(f"Assessment: {result.assessment_title}")
print(f"Questions: {result.question_count}")
print(f"Total Points: {result.total_points}")
```

### Check Resource References

```python
from pathlib import Path
from schema_validators import ResourceReferenceValidator

validator = ResourceReferenceValidator()
result = validator.validate_references(Path('./extracted_package/'))

print(f"Resources checked: {result.resources_checked}")
print(f"Broken references: {result.broken_references}")
```

### Validate Namespaces

```python
from pathlib import Path
from schema_validators import NamespaceValidator

validator = NamespaceValidator()
result = validator.validate_file(Path('imsmanifest.xml'))

print(f"IMSCC Version: {result.imscc_version}")
print(f"LMS Detected: {result.lms_detected}")
```

## CLI Usage

Each validator can be run from the command line:

### Manifest Validator

```bash
# Basic validation
python imscc_manifest_validator.py -i imsmanifest.xml

# JSON output
python imscc_manifest_validator.py -i imsmanifest.xml -j

# Verbose output
python imscc_manifest_validator.py -i imsmanifest.xml -vv
```

### QTI Assessment Validator

```bash
# Validate a quiz
python qti_assessment_validator.py -i quiz.xml

# JSON output with question details
python qti_assessment_validator.py -i quiz.xml -j
```

### Resource Reference Validator

```bash
# Validate extracted package
python resource_reference_validator.py -i ./extracted_package/

# JSON output
python resource_reference_validator.py -i ./extracted_package/ -j
```

### Namespace Validator

```bash
# Check namespaces
python namespace_validator.py -i imsmanifest.xml

# JSON output with detected LMS
python namespace_validator.py -i imsmanifest.xml -j
```

## Validators in Detail

### IMSCCManifestValidator

Validates the imsmanifest.xml file against IMS Common Cartridge specifications.

**Checks performed:**
- XML well-formedness
- Root element is `<manifest>` with identifier
- Required namespace declarations
- Metadata section with schema/schemaversion
- Organizations section with proper hierarchy
- Resources section with valid identifiers and types
- Resource type values match IMS CC specifications
- All identifiers are unique
- Organization items reference valid resources

**Issue codes:**
| Code | Severity | Description |
|------|----------|-------------|
| MF001 | CRITICAL | Manifest file not found |
| MF002 | CRITICAL | XML parsing error |
| MF010 | CRITICAL | Invalid root element |
| MF011 | HIGH | Missing manifest identifier |
| MF040 | CRITICAL | Missing organizations section |
| MF050 | CRITICAL | Missing resources section |
| MF070 | HIGH | Duplicate identifier found |
| MF080 | HIGH | Broken resource reference |

### QTIAssessmentValidator

Validates QTI 1.2 assessment XML files for IMS CC and Brightspace compatibility.

**Checks performed:**
- questestinterop root element
- Assessment element with valid identifier
- Metadata section (cc_profile, qmd_assessmenttype)
- Section and item structure
- Question types (multiple choice, true/false, short answer, etc.)
- Response processing (outcomes, respcondition)
- Presentation elements
- D2L/Brightspace compatibility

**Issue codes:**
| Code | Severity | Description |
|------|----------|-------------|
| QTI001 | CRITICAL | QTI file not found |
| QTI002 | CRITICAL | XML parsing error |
| QTI010 | CRITICAL | Invalid root element |
| QTI020 | CRITICAL | No assessment element |
| QTI040 | HIGH | No section elements |
| QTI050 | HIGH | Item missing identifier |
| QTI051 | MEDIUM | Missing response processing |
| QTI052 | HIGH | Missing presentation |

### ResourceReferenceValidator

Validates that all resource references in IMSCC packages resolve correctly.

**Checks performed:**
- All resource href attributes point to existing files
- All organization identifierref values exist in resources
- All file references use relative paths
- No Windows-style path separators
- Internal HTML links resolve

**Issue codes:**
| Code | Severity | Description |
|------|----------|-------------|
| RR001 | CRITICAL | Manifest not found |
| RR010 | HIGH | No resources section |
| RR020 | HIGH | Broken organization reference |
| RR030 | HIGH | Resource href missing file |
| RR031 | CRITICAL | File element missing file |
| RR040 | MEDIUM | Absolute path in href |
| RR050 | MEDIUM | Broken HTML link |

### NamespaceValidator

Validates XML namespace declarations for consistency and completeness.

**Checks performed:**
- All namespace prefixes are declared
- Namespaces match expected IMS CC patterns
- No conflicting namespace declarations
- Brightspace-specific extensions properly declared
- LMS source detection

**Detected LMS Sources:**
- Brightspace/D2L
- Canvas
- Blackboard
- Moodle
- Sakai

**Issue codes:**
| Code | Severity | Description |
|------|----------|-------------|
| NS001 | CRITICAL | XML parsing error |
| NS010 | CRITICAL | No namespace declarations |
| NS011 | CRITICAL | Missing IMS CC namespace |
| NS020 | HIGH | Mixed IMSCC versions |
| NS030 | HIGH | Undeclared namespace prefix |
| NS040 | MEDIUM | Malformed namespace URI |

## Validation Results

All validators return a `ValidationResult` dataclass with:

```python
@dataclass
class ValidationResult:
    file_path: str        # Path to validated file
    valid: bool           # Overall validity
    issues: List[Issue]   # List of validation issues
    # Plus validator-specific fields...
```

### Issue Severity Levels

| Level | Description |
|-------|-------------|
| CRITICAL | Package cannot be imported |
| HIGH | Major functionality affected |
| MEDIUM | May cause issues in some LMS |
| LOW | Best practice recommendations |

## Integration with Brightspace Packager

These validators are designed to be used as pre-flight checks before IMSCC packaging:

```python
from pathlib import Path
from schema_validators import (
    NamespaceValidator,
    ResourceReferenceValidator,
    IMSCCManifestValidator,
    QTIAssessmentValidator,
)

def validate_package(package_dir: Path) -> bool:
    """Run all validators before packaging."""
    manifest = package_dir / 'imsmanifest.xml'

    # 1. Namespace validation
    ns_result = NamespaceValidator().validate_file(manifest)
    if not ns_result.valid:
        return False

    # 2. Manifest validation
    mf_result = IMSCCManifestValidator().validate_manifest(manifest)
    if not mf_result.valid:
        return False

    # 3. Resource reference validation
    rr_result = ResourceReferenceValidator().validate_references(package_dir)
    if not rr_result.valid:
        return False

    # 4. QTI validation for all assessments
    for qti_file in package_dir.rglob('*.xml'):
        if 'assessment' in qti_file.name or 'quiz' in qti_file.name:
            qti_result = QTIAssessmentValidator().validate_assessment(qti_file)
            if not qti_result.valid:
                return False

    return True
```

## Supported Specifications

| Specification | Versions |
|---------------|----------|
| IMS Common Cartridge | 1.1.0, 1.2.0, 1.3.0 |
| QTI | 1.2 |
| Brightspace/D2L | d2l_2p0 extensions |

## Exit Codes

All CLI validators use consistent exit codes:

| Code | Meaning |
|------|---------|
| 0 | Validation passed |
| 1 | Validation failed |
| 2 | File not found or parse error |

## Examples

### Full Package Validation

```bash
#!/bin/bash
# Validate an extracted IMSCC package

PACKAGE_DIR="./extracted_course"

echo "Validating namespaces..."
python namespace_validator.py -i "$PACKAGE_DIR/imsmanifest.xml" || exit 1

echo "Validating manifest..."
python imscc_manifest_validator.py -i "$PACKAGE_DIR/imsmanifest.xml" || exit 1

echo "Validating resource references..."
python resource_reference_validator.py -i "$PACKAGE_DIR" || exit 1

echo "Validating assessments..."
for qti in "$PACKAGE_DIR"/**/assessment*.xml; do
    python qti_assessment_validator.py -i "$qti" || exit 1
done

echo "All validations passed!"
```

### JSON Pipeline

```bash
# Get structured validation results
python imscc_manifest_validator.py -i manifest.xml -j | jq '.issues[] | select(.severity == "critical")'
```

## Contributing

When adding new validators:

1. Follow the existing pattern with `IssueSeverity` enum and `ValidationResult` dataclass
2. Use consistent issue codes (e.g., XX001 for file not found, XX002 for parse error)
3. Include CLI with standard arguments (-i, -j, -v, --version)
4. Add comprehensive docstrings and type hints
5. Update this README with new validator documentation
