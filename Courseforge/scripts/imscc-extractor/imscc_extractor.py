#!/usr/bin/env python3
"""
IMSCC Extractor - Universal IMSCC Package Import Foundation

This script provides the core import foundation for Courseforge's intake system,
supporting IMSCC packages from any LMS (Brightspace, Canvas, Blackboard, Moodle, etc.)

Features:
- Extract IMSCC packages (IMS CC 1.1, 1.2, 1.3)
- Detect source LMS from manifest namespaces and patterns
- Parse organization structure and resource inventory
- Map content types for remediation pipeline
- Generate structured course object for downstream processing

Usage:
    python imscc_extractor.py --input package.imscc --output /path/to/extracted/
    python imscc_extractor.py --input package.imscc --analyze-only
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from xml.etree import ElementTree as ET

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_COURSEFORGE_DIR = _SCRIPTS_DIR.parent
_PROJECT_ROOT = _COURSEFORGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.secure_paths import safe_extract_zip, validate_path_within_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('imscc_extractor.log')
    ]
)
logger = logging.getLogger(__name__)


class LMSType(Enum):
    """Supported LMS source types"""
    BRIGHTSPACE = "brightspace"
    CANVAS = "canvas"
    BLACKBOARD = "blackboard"
    MOODLE = "moodle"
    SAKAI = "sakai"
    GENERIC = "generic"
    UNKNOWN = "unknown"


class IMSCCVersion(Enum):
    """IMS Common Cartridge versions"""
    CC_1_0 = "1.0"
    CC_1_1 = "1.1"
    CC_1_2 = "1.2"
    CC_1_3 = "1.3"
    UNKNOWN = "unknown"


class ResourceType(Enum):
    """Content resource types for remediation classification"""
    HTML = "html"
    PDF = "pdf"
    OFFICE_DOC = "office_doc"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    QUIZ_QTI = "quiz_qti"
    ASSIGNMENT = "assignment"
    DISCUSSION = "discussion"
    LINK = "link"
    LTI = "lti"
    OTHER = "other"


@dataclass
class Resource:
    """Represents a single resource in the IMSCC package"""
    identifier: str
    type: ResourceType
    href: str
    title: str = ""
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_size: int = 0
    needs_remediation: bool = False
    remediation_reason: str = ""


@dataclass
class OrganizationItem:
    """Represents an item in the course organization hierarchy"""
    identifier: str
    title: str
    resource_ref: str = ""
    children: List['OrganizationItem'] = field(default_factory=list)
    depth: int = 0
    item_type: str = "module"  # module, unit, page, assessment


@dataclass
class ExtractedCourse:
    """Complete extracted course structure"""
    # Identification
    package_path: str
    extraction_path: str
    extraction_timestamp: str

    # LMS Detection
    source_lms: LMSType
    imscc_version: IMSCCVersion
    lms_detection_confidence: float
    detection_evidence: List[str] = field(default_factory=list)

    # Course Metadata
    title: str = ""
    description: str = ""
    identifier: str = ""
    language: str = "en"

    # Content Structure
    organization: List[OrganizationItem] = field(default_factory=list)
    resources: Dict[str, Resource] = field(default_factory=dict)

    # Remediation Analysis
    total_resources: int = 0
    resources_needing_remediation: int = 0
    remediation_summary: Dict[str, int] = field(default_factory=dict)

    # File Inventory
    html_files: List[str] = field(default_factory=list)
    pdf_files: List[str] = field(default_factory=list)
    office_files: List[str] = field(default_factory=list)
    image_files: List[str] = field(default_factory=list)
    media_files: List[str] = field(default_factory=list)
    assessment_files: List[str] = field(default_factory=list)
    other_files: List[str] = field(default_factory=list)

    # Errors and Warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class IMSCCExtractor:
    """
    Universal IMSCC Package Extractor

    Extracts and parses IMSCC packages from any major LMS,
    providing structured output for the remediation pipeline.
    """

    # Namespace mappings for different LMS and IMS CC versions
    NAMESPACES = {
        # IMS Common Cartridge namespaces
        'imscp': 'http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1',
        'imscp_1p2': 'http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1',
        'imscp_1p3': 'http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1',
        'imsmd': 'http://ltsc.ieee.org/xsd/LOM',
        'lom': 'http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource',

        # Brightspace/D2L specific
        'd2l': 'http://desire2learn.com/xsd/d2l_2p0',
        'd2l_cc': 'http://www.desire2learn.com/xsd/d2l_cc',

        # Canvas specific
        'canvas': 'http://canvas.instructure.com/xsd/cccv1p0',

        # Blackboard specific
        'bb': 'http://www.blackboard.com/content-packaging/',

        # Moodle specific
        'moodle': 'http://moodle.org/',

        # QTI namespaces
        'qti': 'http://www.imsglobal.org/xsd/ims_qtiasiv1p2',
        'qti_2p1': 'http://www.imsglobal.org/xsd/imsqti_v2p1',
    }

    # LMS detection patterns
    LMS_PATTERNS = {
        LMSType.BRIGHTSPACE: {
            'namespaces': ['d2l', 'desire2learn', 'd2l_2p0'],
            'file_patterns': ['d2l_', 'D2L', 'desire2learn'],
            'manifest_patterns': ['d2l:', 'D2LContentObject'],
        },
        LMSType.CANVAS: {
            'namespaces': ['canvas.instructure', 'canvas'],
            'file_patterns': ['canvas_', 'course_settings', 'assignment_groups'],
            'manifest_patterns': ['canvas:', 'instructure'],
        },
        LMSType.BLACKBOARD: {
            'namespaces': ['blackboard.com', 'bb_'],
            'file_patterns': ['bb_', 'blackboard', 'res00'],
            'manifest_patterns': ['bb:', 'blackboard'],
        },
        LMSType.MOODLE: {
            'namespaces': ['moodle.org', 'moodle'],
            'file_patterns': ['moodle_', 'backup_', 'course/'],
            'manifest_patterns': ['moodle:', 'backup_moodle'],
        },
        LMSType.SAKAI: {
            'namespaces': ['sakaiproject.org', 'sakai'],
            'file_patterns': ['sakai_', 'attachments/'],
            'manifest_patterns': ['sakai:'],
        },
    }

    # File extension mappings
    FILE_EXTENSIONS = {
        ResourceType.HTML: {'.html', '.htm', '.xhtml'},
        ResourceType.PDF: {'.pdf'},
        ResourceType.OFFICE_DOC: {'.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.odt', '.odp', '.ods'},
        ResourceType.IMAGE: {'.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp', '.ico'},
        ResourceType.VIDEO: {'.mp4', '.mov', '.avi', '.wmv', '.webm', '.mkv', '.flv'},
        ResourceType.AUDIO: {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac'},
    }

    def __init__(self, imscc_path: Path, output_path: Optional[Path] = None):
        """
        Initialize the IMSCC extractor.

        Args:
            imscc_path: Path to the IMSCC package file
            output_path: Optional path for extracted content
        """
        self.imscc_path = Path(imscc_path)
        self.output_path = Path(output_path) if output_path else None
        self.temp_dir: Optional[Path] = None
        self.manifest_tree: Optional[ET.ElementTree] = None
        self.manifest_root: Optional[ET.Element] = None
        self.extracted_course: Optional[ExtractedCourse] = None

    def extract(self) -> ExtractedCourse:
        """
        Main extraction method - processes the IMSCC package.

        Returns:
            ExtractedCourse object with all parsed content
        """
        logger.info(f"Starting extraction of: {self.imscc_path}")

        # Validate input
        self._validate_input()

        # Create temp directory for extraction
        self.temp_dir = Path(tempfile.mkdtemp(prefix='imscc_extract_'))
        logger.info(f"Extracting to temp directory: {self.temp_dir}")

        try:
            # Step 1: Unzip the package
            self._unzip_package()

            # Step 2: Parse the manifest
            self._parse_manifest()

            # Step 3: Detect source LMS
            lms_type, confidence, evidence = self._detect_lms()

            # Step 4: Detect IMSCC version
            imscc_version = self._detect_version()

            # Step 5: Extract metadata
            title, description, identifier, language = self._extract_metadata()

            # Step 6: Parse organization structure
            organization = self._parse_organization()

            # Step 7: Parse resources
            resources = self._parse_resources()

            # Step 8: Inventory files
            file_inventory = self._inventory_files()

            # Step 9: Analyze remediation needs
            remediation_summary = self._analyze_remediation_needs(resources)

            # Step 10: Build extracted course object
            extraction_path = str(self.output_path) if self.output_path else str(self.temp_dir)

            self.extracted_course = ExtractedCourse(
                package_path=str(self.imscc_path),
                extraction_path=extraction_path,
                extraction_timestamp=datetime.now().isoformat(),
                source_lms=lms_type,
                imscc_version=imscc_version,
                lms_detection_confidence=confidence,
                detection_evidence=evidence,
                title=title,
                description=description,
                identifier=identifier,
                language=language,
                organization=organization,
                resources=resources,
                total_resources=len(resources),
                resources_needing_remediation=sum(1 for r in resources.values() if r.needs_remediation),
                remediation_summary=remediation_summary,
                **file_inventory
            )

            # Step 11: Copy to output if specified
            if self.output_path:
                self._copy_to_output()

            logger.info(f"Extraction complete. Source LMS: {lms_type.value}, "
                       f"Version: {imscc_version.value}, Resources: {len(resources)}")

            return self.extracted_course

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            self._cleanup_temp()
            raise

    def _validate_input(self):
        """Validate the input IMSCC package exists and is valid"""
        if not self.imscc_path.exists():
            raise FileNotFoundError(f"IMSCC package not found: {self.imscc_path}")

        if not self.imscc_path.suffix.lower() in ('.imscc', '.zip'):
            logger.warning(f"Unexpected file extension: {self.imscc_path.suffix}")

        # Verify it's a valid ZIP file
        if not zipfile.is_zipfile(self.imscc_path):
            raise ValueError(f"Invalid IMSCC package (not a valid ZIP): {self.imscc_path}")

    def _unzip_package(self):
        """Extract the IMSCC package to temp directory"""
        logger.info("Unzipping package...")
        # Use safe extraction to prevent Zip Slip attacks
        safe_extract_zip(self.imscc_path, self.temp_dir)

        # Verify manifest exists
        manifest_path = self.temp_dir / 'imsmanifest.xml'
        if not manifest_path.exists():
            # Check for nested extraction (some packages have extra folder)
            subdirs = [d for d in self.temp_dir.iterdir() if d.is_dir()]
            for subdir in subdirs:
                if (subdir / 'imsmanifest.xml').exists():
                    manifest_path = subdir / 'imsmanifest.xml'
                    # Move content up one level with path validation
                    for item in subdir.iterdir():
                        # Validate target path stays within temp_dir
                        target = validate_path_within_root(
                            self.temp_dir / item.name, self.temp_dir
                        )
                        shutil.move(str(item), str(target))
                    subdir.rmdir()
                    break
            else:
                raise FileNotFoundError("imsmanifest.xml not found in package")

        logger.info(f"Package extracted. Found {len(list(self.temp_dir.rglob('*')))} files")

    def _parse_manifest(self):
        """Parse the imsmanifest.xml file"""
        manifest_path = self.temp_dir / 'imsmanifest.xml'

        try:
            self.manifest_tree = ET.parse(manifest_path)
            self.manifest_root = self.manifest_tree.getroot()
            logger.info("Manifest parsed successfully")
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse manifest XML: {e}")

    def _detect_lms(self) -> Tuple[LMSType, float, List[str]]:
        """
        Detect the source LMS from manifest patterns and file structure.

        Returns:
            Tuple of (LMSType, confidence score 0-1, list of detection evidence)
        """
        evidence = []
        scores = {lms: 0.0 for lms in LMSType}

        # Get manifest content as string for pattern matching
        manifest_str = ET.tostring(self.manifest_root, encoding='unicode')

        # Check namespaces in manifest
        for lms, patterns in self.LMS_PATTERNS.items():
            for ns_pattern in patterns['namespaces']:
                if ns_pattern.lower() in manifest_str.lower():
                    scores[lms] += 0.4
                    evidence.append(f"Namespace match: {ns_pattern} -> {lms.value}")

            for manifest_pattern in patterns['manifest_patterns']:
                if manifest_pattern.lower() in manifest_str.lower():
                    scores[lms] += 0.3
                    evidence.append(f"Manifest pattern: {manifest_pattern} -> {lms.value}")

        # Check file patterns
        all_files = [str(f.relative_to(self.temp_dir)) for f in self.temp_dir.rglob('*')]
        for lms, patterns in self.LMS_PATTERNS.items():
            for file_pattern in patterns['file_patterns']:
                matching = [f for f in all_files if file_pattern.lower() in f.lower()]
                if matching:
                    scores[lms] += 0.2 * min(len(matching) / 5, 1.0)
                    evidence.append(f"File pattern: {file_pattern} ({len(matching)} files) -> {lms.value}")

        # Determine best match
        best_lms = max(scores, key=scores.get)
        best_score = scores[best_lms]

        if best_score < 0.2:
            return LMSType.GENERIC, 0.5, ["No specific LMS patterns detected - treating as generic IMSCC"]

        return best_lms, min(best_score, 1.0), evidence

    def _detect_version(self) -> IMSCCVersion:
        """Detect the IMS Common Cartridge version"""
        manifest_str = ET.tostring(self.manifest_root, encoding='unicode')

        if 'imsccv1p3' in manifest_str or '1.3.0' in manifest_str:
            return IMSCCVersion.CC_1_3
        elif 'imsccv1p2' in manifest_str or '1.2.0' in manifest_str:
            return IMSCCVersion.CC_1_2
        elif 'imsccv1p1' in manifest_str or '1.1.0' in manifest_str:
            return IMSCCVersion.CC_1_1
        elif 'imsccv1p0' in manifest_str or '1.0.0' in manifest_str:
            return IMSCCVersion.CC_1_0

        return IMSCCVersion.UNKNOWN

    def _extract_metadata(self) -> Tuple[str, str, str, str]:
        """Extract course metadata from manifest"""
        title = ""
        description = ""
        identifier = self.manifest_root.get('identifier', '')
        language = "en"

        # Try to find title in various locations
        # Check common metadata paths
        metadata_paths = [
            './/title',
            './/{http://ltsc.ieee.org/xsd/LOM}title',
            './/general/title',
            './/lom:general/lom:title',
        ]

        for path in metadata_paths:
            try:
                elem = self.manifest_root.find(path)
                if elem is not None and elem.text:
                    title = elem.text.strip()
                    break
            except (AttributeError, TypeError) as e:
                logger.debug(f"Metadata extraction failed for path {path}: {e}")
                continue

        # Fallback: use first organization title
        if not title:
            org = self.manifest_root.find('.//{*}organization')
            if org is not None:
                item = org.find('.//{*}item')
                if item is not None:
                    title_elem = item.find('.//{*}title')
                    if title_elem is not None and title_elem.text:
                        title = title_elem.text.strip()

        # Extract description
        desc_paths = [
            './/description',
            './/{http://ltsc.ieee.org/xsd/LOM}description',
            './/general/description',
        ]

        for path in desc_paths:
            try:
                elem = self.manifest_root.find(path)
                if elem is not None and elem.text:
                    description = elem.text.strip()
                    break
            except (AttributeError, TypeError) as e:
                logger.debug(f"Description extraction failed for path {path}: {e}")
                continue

        # Extract language
        lang_paths = [
            './/language',
            './/{http://ltsc.ieee.org/xsd/LOM}language',
        ]

        for path in lang_paths:
            try:
                elem = self.manifest_root.find(path)
                if elem is not None and elem.text:
                    language = elem.text.strip()[:2]  # Get 2-letter code
                    break
            except (AttributeError, TypeError) as e:
                logger.debug(f"Language extraction failed for path {path}: {e}")
                continue

        logger.info(f"Metadata: title='{title}', identifier='{identifier}'")
        return title, description, identifier, language

    def _parse_organization(self) -> List[OrganizationItem]:
        """Parse the course organization structure"""
        organization = []

        # Find organizations element
        orgs = self.manifest_root.find('.//{*}organizations')
        if orgs is None:
            logger.warning("No organizations element found in manifest")
            return organization

        # Find default organization
        default_org_id = orgs.get('default', '')
        org_elem = None

        for org in orgs.findall('.//{*}organization'):
            if org.get('identifier') == default_org_id:
                org_elem = org
                break

        if org_elem is None:
            org_elem = orgs.find('.//{*}organization')

        if org_elem is None:
            logger.warning("No organization element found")
            return organization

        # Parse items recursively
        def parse_item(elem, depth=0) -> OrganizationItem:
            identifier = elem.get('identifier', '')
            identifierref = elem.get('identifierref', '')

            title_elem = elem.find('.//{*}title')
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ''

            # Determine item type based on depth and presence of resource
            item_type = "module"
            if depth == 0:
                item_type = "module"
            elif depth == 1:
                item_type = "unit"
            elif identifierref:
                item_type = "page"

            item = OrganizationItem(
                identifier=identifier,
                title=title,
                resource_ref=identifierref,
                depth=depth,
                item_type=item_type,
                children=[]
            )

            # Parse child items
            for child_elem in elem.findall('./{*}item'):
                child_item = parse_item(child_elem, depth + 1)
                item.children.append(child_item)

            return item

        # Parse top-level items
        for item_elem in org_elem.findall('./{*}item'):
            organization.append(parse_item(item_elem, 0))

        logger.info(f"Parsed {len(organization)} top-level organization items")
        return organization

    def _parse_resources(self) -> Dict[str, Resource]:
        """Parse all resources from the manifest"""
        resources = {}

        resources_elem = self.manifest_root.find('.//{*}resources')
        if resources_elem is None:
            logger.warning("No resources element found in manifest")
            return resources

        for res_elem in resources_elem.findall('.//{*}resource'):
            identifier = res_elem.get('identifier', '')
            res_type_str = res_elem.get('type', '')
            href = res_elem.get('href', '')

            # Determine resource type
            res_type = self._classify_resource(res_type_str, href)

            # Get title from metadata if available
            title = ""
            title_elem = res_elem.find('.//{*}title')
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()

            # Get dependencies
            dependencies = []
            for dep in res_elem.findall('.//{*}dependency'):
                dep_id = dep.get('identifierref', '')
                if dep_id:
                    dependencies.append(dep_id)

            # Get file size if possible
            file_size = 0
            if href:
                file_path = self.temp_dir / href
                if file_path.exists():
                    file_size = file_path.stat().st_size

            # Check if needs remediation
            needs_remediation, remediation_reason = self._check_remediation_need(res_type, href)

            resource = Resource(
                identifier=identifier,
                type=res_type,
                href=href,
                title=title,
                dependencies=dependencies,
                file_size=file_size,
                needs_remediation=needs_remediation,
                remediation_reason=remediation_reason,
                metadata={'original_type': res_type_str}
            )

            resources[identifier] = resource

        logger.info(f"Parsed {len(resources)} resources")
        return resources

    def _classify_resource(self, type_str: str, href: str) -> ResourceType:
        """Classify a resource based on its type string and file extension"""
        type_lower = type_str.lower()

        # Check for assessment types
        if 'qti' in type_lower or 'assessment' in type_lower:
            return ResourceType.QUIZ_QTI
        if 'assignment' in type_lower:
            return ResourceType.ASSIGNMENT
        if 'discussion' in type_lower or 'topic' in type_lower:
            return ResourceType.DISCUSSION
        if 'lti' in type_lower or 'basiclti' in type_lower:
            return ResourceType.LTI
        if 'weblink' in type_lower or 'imswl' in type_lower:
            return ResourceType.LINK

        # Check file extension
        if href:
            ext = Path(href).suffix.lower()
            for res_type, extensions in self.FILE_EXTENSIONS.items():
                if ext in extensions:
                    return res_type

        # Default to HTML for webcontent
        if 'webcontent' in type_lower or 'html' in type_lower:
            return ResourceType.HTML

        return ResourceType.OTHER

    def _check_remediation_need(self, res_type: ResourceType, href: str) -> Tuple[bool, str]:
        """Check if a resource needs remediation"""
        # PDFs always need DART conversion
        if res_type == ResourceType.PDF:
            return True, "PDF requires DART conversion to accessible HTML"

        # Office documents need conversion
        if res_type == ResourceType.OFFICE_DOC:
            return True, "Office document requires conversion to accessible HTML"

        # Images with text need alt text and possibly HTML conversion
        if res_type == ResourceType.IMAGE:
            return True, "Image may contain text requiring alt text generation"

        # HTML files may need accessibility remediation
        if res_type == ResourceType.HTML and href:
            file_path = self.temp_dir / href
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    # Check for basic accessibility issues
                    if '<img' in content and 'alt=' not in content:
                        return True, "HTML contains images without alt attributes"
                    if not re.search(r'<h[1-6]', content):
                        return True, "HTML lacks proper heading structure"
                except (OSError, IOError, UnicodeDecodeError) as e:
                    logger.debug(f"Failed to read HTML content for analysis: {e}")

        return False, ""

    def _inventory_files(self) -> Dict[str, List[str]]:
        """Create an inventory of all files by type"""
        inventory = {
            'html_files': [],
            'pdf_files': [],
            'office_files': [],
            'image_files': [],
            'media_files': [],
            'assessment_files': [],
            'other_files': [],
        }

        for file_path in self.temp_dir.rglob('*'):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(self.temp_dir))
                ext = file_path.suffix.lower()

                if ext in self.FILE_EXTENSIONS[ResourceType.HTML]:
                    inventory['html_files'].append(rel_path)
                elif ext in self.FILE_EXTENSIONS[ResourceType.PDF]:
                    inventory['pdf_files'].append(rel_path)
                elif ext in self.FILE_EXTENSIONS[ResourceType.OFFICE_DOC]:
                    inventory['office_files'].append(rel_path)
                elif ext in self.FILE_EXTENSIONS[ResourceType.IMAGE]:
                    inventory['image_files'].append(rel_path)
                elif ext in (self.FILE_EXTENSIONS[ResourceType.VIDEO] |
                           self.FILE_EXTENSIONS[ResourceType.AUDIO]):
                    inventory['media_files'].append(rel_path)
                elif ext in {'.xml', '.qti'}:
                    inventory['assessment_files'].append(rel_path)
                else:
                    inventory['other_files'].append(rel_path)

        logger.info(f"File inventory: {sum(len(v) for v in inventory.values())} total files")
        return inventory

    def _analyze_remediation_needs(self, resources: Dict[str, Resource]) -> Dict[str, int]:
        """Analyze and summarize remediation needs"""
        summary = {
            'pdf_conversion': 0,
            'office_conversion': 0,
            'image_alt_text': 0,
            'html_accessibility': 0,
            'total_needing_remediation': 0,
        }

        for resource in resources.values():
            if resource.needs_remediation:
                summary['total_needing_remediation'] += 1

                if resource.type == ResourceType.PDF:
                    summary['pdf_conversion'] += 1
                elif resource.type == ResourceType.OFFICE_DOC:
                    summary['office_conversion'] += 1
                elif resource.type == ResourceType.IMAGE:
                    summary['image_alt_text'] += 1
                elif resource.type == ResourceType.HTML:
                    summary['html_accessibility'] += 1

        logger.info(f"Remediation summary: {summary}")
        return summary

    def _copy_to_output(self):
        """Copy extracted content to output directory"""
        if not self.output_path:
            return

        if self.output_path.exists():
            logger.warning(f"Output path exists, will be overwritten: {self.output_path}")
            shutil.rmtree(self.output_path)

        shutil.copytree(self.temp_dir, self.output_path)
        logger.info(f"Extracted content copied to: {self.output_path}")

    def _cleanup_temp(self):
        """Clean up temporary extraction directory"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary directory cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")

    def get_extraction_summary(self) -> str:
        """Generate a human-readable extraction summary"""
        if not self.extracted_course:
            return "No extraction performed yet"

        ec = self.extracted_course

        summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    IMSCC EXTRACTION SUMMARY                       ║
╠══════════════════════════════════════════════════════════════════╣
║ Package: {ec.package_path[:55]:<55} ║
║ Title: {ec.title[:57]:<57} ║
║ Source LMS: {ec.source_lms.value:<52} ║
║ IMSCC Version: {ec.imscc_version.value:<49} ║
║ Detection Confidence: {ec.lms_detection_confidence:.0%:<42} ║
╠══════════════════════════════════════════════════════════════════╣
║ CONTENT INVENTORY                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║ Total Resources: {ec.total_resources:<47} ║
║ HTML Files: {len(ec.html_files):<52} ║
║ PDF Files: {len(ec.pdf_files):<53} ║
║ Office Documents: {len(ec.office_files):<46} ║
║ Images: {len(ec.image_files):<56} ║
║ Media (Audio/Video): {len(ec.media_files):<43} ║
║ Assessments: {len(ec.assessment_files):<51} ║
╠══════════════════════════════════════════════════════════════════╣
║ REMEDIATION ANALYSIS                                              ║
╠══════════════════════════════════════════════════════════════════╣
║ Resources Needing Remediation: {ec.resources_needing_remediation:<33} ║
║   - PDF Conversion (DART): {ec.remediation_summary.get('pdf_conversion', 0):<37} ║
║   - Office Conversion: {ec.remediation_summary.get('office_conversion', 0):<41} ║
║   - Image Alt Text: {ec.remediation_summary.get('image_alt_text', 0):<44} ║
║   - HTML Accessibility: {ec.remediation_summary.get('html_accessibility', 0):<40} ║
╚══════════════════════════════════════════════════════════════════╝
"""
        return summary

    def to_json(self) -> str:
        """Export extracted course as JSON"""
        if not self.extracted_course:
            return "{}"

        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            return str(obj)

        # Convert to dict with proper serialization
        data = asdict(self.extracted_course)

        # Fix enum values
        data['source_lms'] = self.extracted_course.source_lms.value
        data['imscc_version'] = self.extracted_course.imscc_version.value

        # Fix resource types in resources dict
        resources_serialized = {}
        for key, resource in self.extracted_course.resources.items():
            res_dict = asdict(resource)
            res_dict['type'] = resource.type.value
            resources_serialized[key] = res_dict
        data['resources'] = resources_serialized

        # Fix organization items (recursive)
        def fix_org_item(item_dict):
            fixed_children = []
            for child in item_dict.get('children', []):
                fixed_children.append(fix_org_item(child))
            item_dict['children'] = fixed_children
            return item_dict

        data['organization'] = [fix_org_item(asdict(item)) for item in self.extracted_course.organization]

        return json.dumps(data, indent=2)


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description='Extract and analyze IMSCC packages from any LMS'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to IMSCC package file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output directory for extracted content'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze package without extracting to output'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output extraction result as JSON'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print human-readable summary'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Verbose output (-vv for debug)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Validate input
    imscc_path = Path(args.input)
    if not imscc_path.exists():
        print(f"Error: Input file not found: {imscc_path}", file=sys.stderr)
        sys.exit(1)

    # Set output path
    output_path = None
    if args.output and not args.analyze_only:
        output_path = Path(args.output)

    # Extract
    try:
        extractor = IMSCCExtractor(imscc_path, output_path)
        result = extractor.extract()

        # Output results
        if args.json:
            print(extractor.to_json())
        elif args.summary or not args.json:
            print(extractor.get_extraction_summary())

        # Save JSON to output directory if extracting
        if output_path and output_path.exists():
            json_path = output_path / 'extraction_manifest.json'
            with open(json_path, 'w') as f:
                f.write(extractor.to_json())
            print(f"\nExtraction manifest saved to: {json_path}")

        # Cleanup temp if analyze-only
        if args.analyze_only:
            extractor._cleanup_temp()

        sys.exit(0)

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
