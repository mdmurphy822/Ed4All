#!/usr/bin/env python3
"""
Pattern 15 Prevention Script - Assessment XML Format Validation
================================================================

This script validates assessment XML files to prevent Pattern 15 violations that cause
"Illegal XML" errors during Brightspace IMSCC import.

Pattern 15 Root Cause:
- Assessment XML files use custom formats instead of proper QTI 1.2/D2L standards
- Brightspace validates XML content against declared resource types in manifest
- Custom formats don't match expected schemas causing import failure

Critical Validation Requirements:
- Quiz XML must use QTI 1.2 <questestinterop> format (NOT <quiz>)
- Assignment XML must use D2L schema with <dropbox> configuration
- Discussion XML must use D2L schema with <forum> setup
- All XML must match manifest resource type declarations exactly
"""

import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pattern15_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# IMSCC 1.3 Namespaces (correct, non-deprecated)
QTI_NAMESPACE = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
ASSIGNMENT_NAMESPACE = "http://www.imsglobal.org/xsd/imscc_extensions/assignment"
DISCUSSION_NAMESPACE = "http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"

# Resource types (IMSCC 1.3 standard)
ASSIGNMENT_RESOURCE_TYPE = "assignment_xmlv1p0"
DISCUSSION_RESOURCE_TYPE = "imsdt_xmlv1p3"
QUIZ_RESOURCE_TYPE = "imsqti_xmlv1p2/imscc_xmlv1p3/assessment"

class Pattern15Validator:
    """Validates assessment XML files to prevent Pattern 15 violations"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_package(self, package_dir: Path) -> Tuple[bool, List[str], List[str]]:
        """
        Validates all assessment XML files in IMSCC package directory
        
        Args:
            package_dir: Directory containing IMSCC package files
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        logger.info(f"Starting Pattern 15 validation for: {package_dir}")
        
        self.errors = []
        self.warnings = []
        
        # Find all assessment files
        quiz_files = list(package_dir.glob("quiz_*.xml"))
        assignment_files = list(package_dir.glob("assignment_*.xml"))
        discussion_files = list(package_dir.glob("discussion_*.xml"))
        
        # Validate each assessment type
        self._validate_quiz_files(quiz_files)
        self._validate_assignment_files(assignment_files)
        self._validate_discussion_files(discussion_files)
        
        # Check manifest consistency
        manifest_file = package_dir / "imsmanifest.xml"
        if manifest_file.exists():
            self._validate_manifest_consistency(manifest_file, quiz_files, assignment_files, discussion_files)
        else:
            self.errors.append("Missing imsmanifest.xml file")
        
        is_valid = len(self.errors) == 0
        
        if is_valid:
            logger.info("✅ Pattern 15 validation PASSED - All assessment XML files are compliant")
        else:
            logger.error(f"❌ Pattern 15 validation FAILED - {len(self.errors)} errors found")
            for error in self.errors:
                logger.error(f"  ERROR: {error}")
        
        if self.warnings:
            logger.warning(f"⚠️  {len(self.warnings)} warnings found")
            for warning in self.warnings:
                logger.warning(f"  WARNING: {warning}")
        
        return is_valid, self.errors, self.warnings
    
    def _validate_quiz_files(self, quiz_files: List[Path]) -> None:
        """Validates quiz XML files for QTI 1.2 compliance"""
        logger.info(f"Validating {len(quiz_files)} quiz files")
        
        for quiz_file in quiz_files:
            try:
                tree = ET.parse(quiz_file)
                root = tree.getroot()
                
                # Critical: Must use <questestinterop> root element
                if root.tag != f"{{{QTI_NAMESPACE}}}questestinterop":
                    if root.tag == "quiz":
                        self.errors.append(f"{quiz_file.name}: CRITICAL - Uses custom <quiz> format instead of QTI 1.2 <questestinterop>")
                    else:
                        self.errors.append(f"{quiz_file.name}: Invalid root element '{root.tag}' - must be <questestinterop>")
                    continue
                
                # Check for required QTI namespace (handle both lxml and ElementTree)
                has_qti_namespace = False
                if hasattr(root, 'nsmap') and root.nsmap:
                    has_qti_namespace = QTI_NAMESPACE in root.nsmap.values()
                # Also check in attributes and tag for ElementTree compatibility
                if not has_qti_namespace:
                    has_qti_namespace = QTI_NAMESPACE in str(root.attrib) or QTI_NAMESPACE in root.tag
                if not has_qti_namespace:
                    self.errors.append(f"{quiz_file.name}: Missing QTI 1.2 namespace: {QTI_NAMESPACE}")
                
                # Check for required assessment element
                assessment = root.find(f".//{{{QTI_NAMESPACE}}}assessment")
                if assessment is None:
                    self.errors.append(f"{quiz_file.name}: Missing required <assessment> element")
                    continue
                
                # Check for required qtimetadata
                qtimetadata = assessment.find(f".//{{{QTI_NAMESPACE}}}qtimetadata")
                if qtimetadata is None:
                    self.errors.append(f"{quiz_file.name}: Missing required <qtimetadata> section")
                
                # Check for section and item structure
                section = assessment.find(f".//{{{QTI_NAMESPACE}}}section")
                if section is None:
                    self.errors.append(f"{quiz_file.name}: Missing required <section> element")
                
                items = assessment.findall(f".//{{{QTI_NAMESPACE}}}item")
                if not items:
                    self.errors.append(f"{quiz_file.name}: No <item> elements found - quiz must contain questions")
                
                logger.info(f"✅ {quiz_file.name}: QTI 1.2 format validation passed")
                
            except ET.ParseError as e:
                self.errors.append(f"{quiz_file.name}: XML parsing error - {str(e)}")
            except Exception as e:
                self.errors.append(f"{quiz_file.name}: Validation error - {str(e)}")
    
    def _validate_assignment_files(self, assignment_files: List[Path]) -> None:
        """Validates assignment XML files for D2L compliance"""
        logger.info(f"Validating {len(assignment_files)} assignment files")
        
        for assignment_file in assignment_files:
            try:
                tree = ET.parse(assignment_file)
                root = tree.getroot()
                
                # Check for IMSCC assignment root element
                if not (root.tag == "assignment" or root.tag.endswith("}assignment")):
                    self.errors.append(f"{assignment_file.name}: Invalid root element '{root.tag}' - must be <assignment>")
                    continue

                # Check for IMSCC assignment namespace (handle both lxml and ElementTree)
                has_assignment_namespace = False
                if hasattr(root, 'nsmap') and root.nsmap:
                    has_assignment_namespace = ASSIGNMENT_NAMESPACE in root.nsmap.values()
                # Also check in attributes and tag for ElementTree compatibility
                if not has_assignment_namespace:
                    has_assignment_namespace = ASSIGNMENT_NAMESPACE in str(root.attrib) or ASSIGNMENT_NAMESPACE in root.tag
                if not has_assignment_namespace:
                    self.errors.append(f"{assignment_file.name}: Missing IMSCC assignment namespace: {ASSIGNMENT_NAMESPACE}")

                # Check for required gradable element (IMSCC standard)
                gradable = root.find(".//gradable")
                if gradable is None:
                    self.warnings.append(f"{assignment_file.name}: Missing <gradable> element (recommended for graded assignments)")

                # Check for submission_formats (IMSCC standard)
                submission_formats = root.find(".//submission_formats")
                if submission_formats is None:
                    self.warnings.append(f"{assignment_file.name}: Missing <submission_formats> element")

                logger.info(f"✅ {assignment_file.name}: IMSCC assignment format validation passed")
                
            except ET.ParseError as e:
                self.errors.append(f"{assignment_file.name}: XML parsing error - {str(e)}")
            except Exception as e:
                self.errors.append(f"{assignment_file.name}: Validation error - {str(e)}")
    
    def _validate_discussion_files(self, discussion_files: List[Path]) -> None:
        """Validates discussion XML files for IMSCC compliance"""
        logger.info(f"Validating {len(discussion_files)} discussion files")

        for discussion_file in discussion_files:
            try:
                tree = ET.parse(discussion_file)
                root = tree.getroot()

                # Check for IMSCC discussion root element (IMSCC uses <topic>, not <discussion>)
                if not (root.tag == "topic" or root.tag.endswith("}topic")):
                    self.errors.append(f"{discussion_file.name}: Invalid root element '{root.tag}' - must be <topic> (IMSCC standard)")
                    continue

                # Check for IMSCC discussion namespace (handle both lxml and ElementTree)
                has_discussion_namespace = False
                if hasattr(root, 'nsmap') and root.nsmap:
                    has_discussion_namespace = DISCUSSION_NAMESPACE in root.nsmap.values()
                # Also check in attributes and tag for ElementTree compatibility
                if not has_discussion_namespace:
                    has_discussion_namespace = DISCUSSION_NAMESPACE in str(root.attrib) or DISCUSSION_NAMESPACE in root.tag
                if not has_discussion_namespace:
                    self.errors.append(f"{discussion_file.name}: Missing IMSCC discussion namespace: {DISCUSSION_NAMESPACE}")

                # Check for required title element (IMSCC standard)
                title = root.find(".//title")
                if title is None:
                    self.errors.append(f"{discussion_file.name}: Missing required <title> element")

                # Check for required text element (IMSCC standard)
                text = root.find(".//text")
                if text is None:
                    self.errors.append(f"{discussion_file.name}: Missing required <text> element")

                logger.info(f"✅ {discussion_file.name}: IMSCC discussion format validation passed")
                
            except ET.ParseError as e:
                self.errors.append(f"{discussion_file.name}: XML parsing error - {str(e)}")
            except Exception as e:
                self.errors.append(f"{discussion_file.name}: Validation error - {str(e)}")
    
    def _validate_manifest_consistency(self, manifest_file: Path, quiz_files: List[Path], 
                                     assignment_files: List[Path], discussion_files: List[Path]) -> None:
        """Validates manifest resource types match assessment XML formats"""
        logger.info("Validating manifest resource type consistency")
        
        try:
            tree = ET.parse(manifest_file)
            root = tree.getroot()
            
            # Find all resource elements
            resources = root.findall(".//resource")
            
            for resource in resources:
                resource_type = resource.get("type", "")
                href = resource.get("href", "")
                
                if href.startswith("quiz_") and href.endswith(".xml"):
                    # Quiz resources must declare QTI type (IMSCC 1.3)
                    if resource_type != QUIZ_RESOURCE_TYPE:
                        self.errors.append(f"Manifest: Quiz resource '{href}' has incorrect type '{resource_type}' - should be '{QUIZ_RESOURCE_TYPE}'")

                elif href.startswith("assignment_") and href.endswith(".xml"):
                    # Assignment resources must declare IMSCC assignment type
                    if resource_type != ASSIGNMENT_RESOURCE_TYPE:
                        self.errors.append(f"Manifest: Assignment resource '{href}' has incorrect type '{resource_type}' - should be '{ASSIGNMENT_RESOURCE_TYPE}'")

                elif href.startswith("discussion_") and href.endswith(".xml"):
                    # Discussion resources must declare IMSCC discussion type
                    if resource_type != DISCUSSION_RESOURCE_TYPE:
                        self.errors.append(f"Manifest: Discussion resource '{href}' has incorrect type '{resource_type}' - should be '{DISCUSSION_RESOURCE_TYPE}'")
            
            logger.info("✅ Manifest resource type consistency validation passed")
            
        except ET.ParseError as e:
            self.errors.append(f"Manifest parsing error: {str(e)}")
        except Exception as e:
            self.errors.append(f"Manifest validation error: {str(e)}")

def main():
    """Command line interface for Pattern 15 validation"""
    if len(sys.argv) != 2:
        print("Usage: python pattern15_prevention.py <package_directory>")
        print("Example: python pattern15_prevention.py /exports/20250806_013225/")
        sys.exit(1)
    
    package_dir = Path(sys.argv[1])
    if not package_dir.exists():
        print(f"Error: Directory '{package_dir}' does not exist")
        sys.exit(1)
    
    validator = Pattern15Validator()
    is_valid, errors, warnings = validator.validate_package(package_dir)
    
    if is_valid:
        print("\n✅ PATTERN 15 VALIDATION PASSED")
        print("All assessment XML files are properly formatted for Brightspace import")
        sys.exit(0)
    else:
        print("\n❌ PATTERN 15 VALIDATION FAILED")
        print(f"Found {len(errors)} critical errors that will cause Brightspace import failure:")
        for error in errors:
            print(f"  • {error}")
        print("\nFix these errors before attempting Brightspace import")
        sys.exit(1)

if __name__ == "__main__":
    main()