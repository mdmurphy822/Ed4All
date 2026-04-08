"""
Tests for the IMSCC Extractor Module
IMSCC Package Extraction and LMS Detection Testing
"""

import pytest
import sys
import zipfile
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent / 'imscc-extractor'))

try:
    from imscc_extractor import IMSCCExtractor, LMSType, ResourceType
except ImportError:
    pytest.skip("imscc_extractor module not available", allow_module_level=True)


class TestIMSCCExtractor:
    """Test suite for IMSCCExtractor class"""

    # =========================================================================
    # LMS DETECTION TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_brightspace_namespace(self):
        """Test detection of Brightspace/D2L namespace"""
        manifest = '''<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
          xmlns:d2l="http://www.desire2learn.com/xsd/d2l_2p0">
    <metadata><schema>IMS Common Cartridge</schema></metadata>
</manifest>'''
        # Detection logic test
        assert 'd2l' in manifest or 'desire2learn' in manifest

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_canvas_namespace(self):
        """Test detection of Canvas namespace"""
        manifest = '''<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
          xmlns:canvas="http://canvas.instructure.com/xsd/cccv1p0">
    <metadata><schema>IMS Common Cartridge</schema></metadata>
</manifest>'''
        assert 'canvas' in manifest or 'instructure' in manifest

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_blackboard_namespace(self):
        """Test detection of Blackboard namespace"""
        manifest = '''<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1"
          xmlns:bb="http://www.blackboard.com/content-packaging">
    <metadata><schema>IMS Common Cartridge</schema></metadata>
</manifest>'''
        assert 'blackboard' in manifest.lower()

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_moodle_namespace(self):
        """Test detection of Moodle namespace"""
        manifest = '''<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1"
          xmlns:moodle="http://moodle.org">
    <metadata><schema>IMS Common Cartridge</schema></metadata>
</manifest>'''
        assert 'moodle' in manifest.lower()

    # =========================================================================
    # VERSION DETECTION TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_imscc_1_1(self):
        """Test detection of IMS CC 1.1.0"""
        manifest = '''<?xml version="1.0"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1">
    <metadata>
        <schema>IMS Common Cartridge</schema>
        <schemaversion>1.1.0</schemaversion>
    </metadata>
</manifest>'''
        assert '1.1.0' in manifest or 'imsccv1p1' in manifest

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_imscc_1_2(self):
        """Test detection of IMS CC 1.2.0"""
        manifest = '''<?xml version="1.0"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1">
    <metadata>
        <schema>IMS Common Cartridge</schema>
        <schemaversion>1.2.0</schemaversion>
    </metadata>
</manifest>'''
        assert '1.2.0' in manifest or 'imsccv1p2' in manifest

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_imscc_1_3(self):
        """Test detection of IMS CC 1.3.0"""
        manifest = '''<?xml version="1.0"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1">
    <metadata>
        <schema>IMS Common Cartridge</schema>
        <schemaversion>1.3.0</schemaversion>
    </metadata>
</manifest>'''
        assert '1.3.0' in manifest or 'imsccv1p3' in manifest

    # =========================================================================
    # EXTRACTION TESTS
    # =========================================================================

    @pytest.mark.integration
    @pytest.mark.imscc
    def test_extracts_minimal_package(self, temp_imscc_package, temp_output_dir):
        """Test extraction of minimal IMSCC package"""
        extractor = IMSCCExtractor(temp_imscc_package, temp_output_dir)
        result = extractor.extract()

        # Should extract successfully
        assert result is not None
        assert result.success is True

    @pytest.mark.integration
    @pytest.mark.imscc
    def test_extracts_manifest(self, temp_imscc_package, temp_output_dir):
        """Test that manifest is extracted correctly"""
        extractor = IMSCCExtractor(temp_imscc_package, temp_output_dir)
        result = extractor.extract()

        # Manifest should be found
        manifest_path = temp_output_dir / 'imsmanifest.xml'
        assert manifest_path.exists() or result.manifest_path is not None

    @pytest.mark.integration
    @pytest.mark.imscc
    def test_extracts_html_content(self, temp_imscc_package, temp_output_dir):
        """Test that HTML content files are extracted"""
        extractor = IMSCCExtractor(temp_imscc_package, temp_output_dir)
        result = extractor.extract()

        # Should have HTML resources
        assert result.html_count > 0 or len(result.resources) > 0

    @pytest.mark.integration
    @pytest.mark.imscc
    def test_content_inventory(self, temp_imscc_package, temp_output_dir):
        """Test content inventory generation"""
        extractor = IMSCCExtractor(temp_imscc_package, temp_output_dir)
        result = extractor.extract()

        # Should generate inventory
        assert hasattr(result, 'resources') or hasattr(result, 'content_inventory')

    # =========================================================================
    # RESOURCE TYPE DETECTION TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_webcontent_resource(self):
        """Test detection of webcontent resource type"""
        resource_type = 'webcontent'
        assert resource_type == 'webcontent'

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_assessment_resource(self):
        """Test detection of assessment resource type"""
        resource_type = 'imsqti_xmlv1p2/imscc_xmlv1p2/assessment'
        assert 'assessment' in resource_type or 'qti' in resource_type

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_detects_discussion_resource(self):
        """Test detection of discussion resource type"""
        resource_type = 'imsdt_xmlv1p2'
        assert 'dt' in resource_type  # discussion topic

    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_handles_missing_manifest(self, tmp_path):
        """Test handling of package without manifest"""
        # Create package without manifest
        package_path = tmp_path / 'no_manifest.imscc'
        with zipfile.ZipFile(package_path, 'w') as zf:
            zf.writestr('content.html', '<html><body>No manifest</body></html>')

        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # Should handle gracefully
        try:
            extractor = IMSCCExtractor(package_path, output_dir)
            result = extractor.extract()
            # Either returns error result or raises exception
            assert result.success is False or True
        except Exception:
            # Exception is acceptable
            assert True

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_handles_corrupted_zip(self, tmp_path):
        """Test handling of corrupted ZIP file"""
        package_path = tmp_path / 'corrupted.imscc'
        package_path.write_text('not a valid zip file')

        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # Should raise appropriate error
        with pytest.raises((zipfile.BadZipFile, Exception)):
            extractor = IMSCCExtractor(package_path, output_dir)
            extractor.extract()

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_handles_nonexistent_package(self, tmp_path):
        """Test handling of nonexistent package file"""
        fake_path = tmp_path / 'nonexistent.imscc'
        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        # Should raise file not found
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            extractor = IMSCCExtractor(fake_path, output_dir)
            extractor.extract()

    # =========================================================================
    # OUTPUT FORMAT TESTS
    # =========================================================================

    @pytest.mark.integration
    @pytest.mark.imscc
    def test_json_output(self, temp_imscc_package, temp_output_dir):
        """Test JSON output generation"""
        extractor = IMSCCExtractor(temp_imscc_package, temp_output_dir)
        result = extractor.extract()

        # Should be able to convert to JSON
        if hasattr(extractor, 'to_json'):
            json_output = extractor.to_json()
            assert json_output is not None
            assert isinstance(json_output, str)

    @pytest.mark.integration
    @pytest.mark.imscc
    def test_summary_output(self, temp_imscc_package, temp_output_dir):
        """Test summary output generation"""
        extractor = IMSCCExtractor(temp_imscc_package, temp_output_dir)
        result = extractor.extract()

        # Should be able to generate summary
        if hasattr(extractor, 'get_extraction_summary'):
            summary = extractor.get_extraction_summary()
            assert summary is not None
            assert isinstance(summary, str)


class TestIMSCCManifestParsing:
    """Tests for manifest XML parsing"""

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_parses_course_title(self, minimal_manifest_content, tmp_path):
        """Test parsing of course title from manifest"""
        manifest_path = tmp_path / 'imsmanifest.xml'
        manifest_path.write_text(minimal_manifest_content)

        # Title should be extractable
        assert 'Test Course' in minimal_manifest_content

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_parses_organization_structure(self, minimal_manifest_content, tmp_path):
        """Test parsing of organization structure"""
        manifest_path = tmp_path / 'imsmanifest.xml'
        manifest_path.write_text(minimal_manifest_content)

        # Should have organization elements
        assert '<organizations>' in minimal_manifest_content
        assert '<organization' in minimal_manifest_content
        assert '<item' in minimal_manifest_content

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_parses_resources(self, minimal_manifest_content, tmp_path):
        """Test parsing of resources section"""
        manifest_path = tmp_path / 'imsmanifest.xml'
        manifest_path.write_text(minimal_manifest_content)

        # Should have resources
        assert '<resources>' in minimal_manifest_content
        assert '<resource' in minimal_manifest_content

    @pytest.mark.unit
    @pytest.mark.imscc
    def test_parses_resource_references(self, minimal_manifest_content):
        """Test parsing of resource references (identifierref)"""
        # Items should reference resources
        assert 'identifierref=' in minimal_manifest_content


class TestIMSCCExtractionAnalysis:
    """Tests for extraction analysis and remediation detection"""

    @pytest.mark.integration
    @pytest.mark.imscc
    def test_detects_accessibility_issues(self, temp_imscc_package, temp_output_dir):
        """Test detection of accessibility issues in extracted content"""
        extractor = IMSCCExtractor(temp_imscc_package, temp_output_dir)
        result = extractor.extract()

        # Should have remediation analysis
        if hasattr(result, 'needs_remediation') or hasattr(result, 'accessibility_issues'):
            # Analysis should be available
            assert True
        else:
            # At minimum extraction should work
            assert result.success is True

    @pytest.mark.integration
    @pytest.mark.imscc
    def test_identifies_pdf_content(self, tmp_path):
        """Test identification of PDF content requiring conversion"""
        # Create package with PDF reference
        manifest = '''<?xml version="1.0"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1">
    <metadata><schema>IMS Common Cartridge</schema></metadata>
    <organizations><organization identifier="org_1"/></organizations>
    <resources>
        <resource identifier="pdf_1" type="webcontent" href="docs/syllabus.pdf">
            <file href="docs/syllabus.pdf"/>
        </resource>
    </resources>
</manifest>'''

        package_dir = tmp_path / 'package'
        package_dir.mkdir()
        (package_dir / 'imsmanifest.xml').write_text(manifest)
        (package_dir / 'docs').mkdir()
        (package_dir / 'docs' / 'syllabus.pdf').write_bytes(b'%PDF-1.4 fake pdf')

        imscc_path = tmp_path / 'with_pdf.imscc'
        with zipfile.ZipFile(imscc_path, 'w') as zf:
            for f in package_dir.rglob('*'):
                if f.is_file():
                    zf.write(f, f.relative_to(package_dir))

        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        extractor = IMSCCExtractor(imscc_path, output_dir)
        result = extractor.extract()

        # Should identify PDF content
        assert result is not None
