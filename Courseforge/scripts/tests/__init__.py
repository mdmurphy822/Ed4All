# Courseforge Test Suite
# Integration and unit tests for course generation and remediation workflows

"""
Test modules:
- test_accessibility_validator: WCAG 2.2 AA compliance validation tests
- test_imscc_extractor: IMSCC package extraction and LMS detection tests
- interactive_components/test_component_applier: Standalone component-applier tests
- test_remediation_validator: Course remediation validation tests

Run tests:
    pytest scripts/tests/ -v
    pytest scripts/tests/ -m unit        # Unit tests only
    pytest scripts/tests/ -m integration # Integration tests only
    pytest scripts/tests/ -m accessibility # Accessibility tests only
"""
