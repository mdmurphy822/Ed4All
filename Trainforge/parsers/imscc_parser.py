"""
IMSCC Package Parser

Extracts content from IMSCC (IMS Common Cartridge) packages for assessment generation.
Supports packages from Brightspace, Canvas, Blackboard, Moodle, and generic IMSCC.
"""

import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
_PARSERS_DIR = Path(__file__).resolve().parent
_TRAINFORGE_DIR = _PARSERS_DIR.parent
_PROJECT_ROOT = _TRAINFORGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.secure_paths import safe_extract_zip  # noqa: E402


@dataclass
class ContentItem:
    """Represents a content item from the IMSCC package."""
    id: str
    title: str
    type: str  # html, pdf, assessment, discussion, etc.
    path: str
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IMSCCPackage:
    """Parsed IMSCC package structure."""
    source_path: str
    source_lms: str  # brightspace, canvas, blackboard, moodle, generic
    version: str
    title: str
    items: List[ContentItem] = field(default_factory=list)
    learning_objectives: List[Dict[str, Any]] = field(default_factory=list)
    assessments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IMSCCParser:
    """
    Parser for IMSCC (IMS Common Cartridge) packages.

    Usage:
        parser = IMSCCParser()
        package = parser.parse("/path/to/course.imscc")
        for item in package.items:
            print(f"{item.title}: {item.type}")
    """

    # LMS detection namespaces
    LMS_NAMESPACES = {
        "d2l_2p0": "brightspace",
        "canvas.instructure": "canvas",
        "blackboard.com": "blackboard",
        "moodle.org": "moodle",
        "sakaiproject.org": "sakai"
    }

    def __init__(self):
        self.namespaces = {}

    def parse(self, imscc_path: str) -> IMSCCPackage:
        """
        Parse an IMSCC package.

        Args:
            imscc_path: Path to .imscc file

        Returns:
            Parsed IMSCCPackage structure

        Raises:
            FileNotFoundError: If IMSCC file doesn't exist
            ValueError: If IMSCC is corrupted or invalid
        """
        path = Path(imscc_path)
        if not path.exists():
            raise FileNotFoundError(f"IMSCC not found: {imscc_path}")

        try:
            with zipfile.ZipFile(path, 'r') as z:
                # Check for manifest before parsing
                if "imsmanifest.xml" not in z.namelist():
                    raise ValueError("IMSCC package missing imsmanifest.xml")

                # Parse manifest with error handling
                try:
                    manifest_content = z.read("imsmanifest.xml").decode('utf-8')
                except UnicodeDecodeError as e:
                    raise ValueError(f"Encoding error in manifest file: {e}") from e

                try:
                    root = ET.fromstring(manifest_content)
                except ET.ParseError as e:
                    raise ValueError(f"Invalid XML in manifest: {e}") from e

                # Detect LMS and version
                source_lms = self._detect_lms(root)
                version = self._detect_version(root)

                # Get title
                title = self._extract_title(root)

                # Parse items
                items = self._parse_items(root, z)

                # Parse assessments
                assessments = self._parse_assessments(z)

                return IMSCCPackage(
                    source_path=str(path),
                    source_lms=source_lms,
                    version=version,
                    title=title,
                    items=items,
                    assessments=assessments,
                    metadata={"namespaces": self.namespaces}
                )
        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid or corrupted IMSCC file: {e}") from e

    def _detect_lms(self, root: ET.Element) -> str:
        """Detect source LMS from namespace declarations."""
        # Get all namespaces with safe error handling
        try:
            self.namespaces = dict([
                node for _, node in ET.iterparse(
                    ET.tostring(root, encoding='unicode'),
                    events=['start-ns']
                )
            ]) if hasattr(ET, 'iterparse') else {}
        except Exception:
            # If namespace detection fails, continue with empty namespaces
            self.namespaces = {}

        # Check namespace URIs
        try:
            root_str = ET.tostring(root, encoding='unicode')
            for ns_marker, lms_name in self.LMS_NAMESPACES.items():
                if ns_marker in root_str:
                    return lms_name
        except Exception:
            # If detection fails, default to generic
            pass

        return "generic"

    def _detect_version(self, root: ET.Element) -> str:
        """Detect IMSCC version from schema references."""
        root_str = ET.tostring(root, encoding='unicode')

        if "imscc_v1p3" in root_str:
            return "1.3"
        elif "imscc_v1p2" in root_str:
            return "1.2"
        elif "imscc_v1p1" in root_str:
            return "1.1"
        else:
            return "1.1"  # Default to 1.1

    def _extract_title(self, root: ET.Element) -> str:
        """Extract course title from manifest."""
        # Try various title locations
        for path in [
            ".//{http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1}title",
            ".//{http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1}title",
            ".//{http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1}title",
            ".//title"
        ]:
            elem = root.find(path)
            if elem is not None and elem.text:
                return elem.text

        return "Untitled Course"

    def _parse_items(self, root: ET.Element, z: zipfile.ZipFile) -> List[ContentItem]:
        """Parse content items from manifest."""
        items = []

        # Find all resources
        for resource in root.iter():
            if 'resource' in resource.tag.lower():
                identifier = resource.get('identifier', '')
                res_type = resource.get('type', '')
                href = resource.get('href', '')

                if identifier and href:
                    # Determine content type
                    content_type = self._classify_resource(res_type, href)

                    # Read content if HTML
                    content = None
                    if content_type == 'html' and href in z.namelist():
                        try:
                            content = z.read(href).decode('utf-8', errors='ignore')
                        except Exception:
                            pass

                    items.append(ContentItem(
                        id=identifier,
                        title=identifier,  # Will be updated from organization
                        type=content_type,
                        path=href,
                        content=content
                    ))

        return items

    def _classify_resource(self, res_type: str, href: str) -> str:
        """Classify resource type based on type string and file extension."""
        href_lower = href.lower()
        type_lower = res_type.lower()

        if 'assessment' in type_lower or 'qti' in type_lower:
            return 'assessment'
        elif 'discussion' in type_lower:
            return 'discussion'
        elif href_lower.endswith('.html') or href_lower.endswith('.htm'):
            return 'html'
        elif href_lower.endswith('.pdf'):
            return 'pdf'
        elif href_lower.endswith(('.docx', '.doc')):
            return 'document'
        elif href_lower.endswith(('.pptx', '.ppt')):
            return 'presentation'
        elif href_lower.endswith(('.mp4', '.webm', '.mov')):
            return 'video'
        elif href_lower.endswith(('.mp3', '.wav')):
            return 'audio'
        elif href_lower.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return 'image'
        else:
            return 'other'

    def _parse_assessments(self, z: zipfile.ZipFile) -> List[Dict[str, Any]]:
        """Parse QTI assessment files."""
        assessments = []

        for name in z.namelist():
            if 'assessment' in name.lower() and name.endswith('.xml'):
                try:
                    content = z.read(name).decode('utf-8', errors='ignore')
                    assessments.append({
                        "path": name,
                        "raw_xml": content[:5000]  # First 5k chars for preview
                    })
                except Exception:
                    pass

        return assessments

    def extract_to_directory(self, imscc_path: str, output_dir: str) -> Path:
        """
        Extract IMSCC package to a directory.

        Args:
            imscc_path: Path to .imscc file
            output_dir: Directory to extract to

        Returns:
            Path to extraction directory

        Raises:
            FileNotFoundError: If IMSCC file doesn't exist
            ValueError: If IMSCC is corrupted or invalid
            PermissionError: If cannot write to output directory
        """
        path = Path(imscc_path)
        if not path.exists():
            raise FileNotFoundError(f"IMSCC not found: {imscc_path}")

        output = Path(output_dir)

        try:
            output.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"Cannot create output directory {output_dir}: {e}") from e

        try:
            # Use safe extraction to prevent Zip Slip attacks
            safe_extract_zip(path, output)
        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid or corrupted IMSCC file: {e}") from e
        except PermissionError as e:
            raise PermissionError(f"Cannot write to output directory {output_dir}: {e}") from e

        return output
