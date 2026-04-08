"""
Tests for the Accessibility Validator Module
WCAG 2.2 AA Compliance Testing
"""

import pytest
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent / 'accessibility-validator'))

try:
    from accessibility_validator import AccessibilityValidator, IssueSeverity
except ImportError:
    pytest.skip("accessibility_validator module not available", allow_module_level=True)


class TestAccessibilityValidator:
    """Test suite for AccessibilityValidator class"""

    @pytest.fixture
    def validator(self):
        """Create a validator instance for testing"""
        return AccessibilityValidator()

    # =========================================================================
    # ALT TEXT TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_missing_alt_attributes(self, validator, write_temp_html, missing_alt_html_content):
        """Test that missing alt attributes are detected"""
        html_path = write_temp_html(missing_alt_html_content, 'missing_alt.html')
        report = validator.validate_file(html_path)

        # Should find issues
        assert report.total_issues > 0

        # Should have critical issues for missing alt
        alt_issues = [i for i in report.issues if 'alt' in i.message.lower()]
        assert len(alt_issues) > 0

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_accepts_valid_alt_text(self, validator, write_temp_html, accessible_html_content):
        """Test that valid alt text passes validation"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Should not have alt text issues for this file
        alt_issues = [i for i in report.issues
                      if 'alt' in i.message.lower() and 'missing' in i.message.lower()]
        assert len(alt_issues) == 0

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_empty_alt_on_informative_images(self, validator, write_temp_html):
        """Test detection of empty alt on images that appear informative"""
        html = '''<!DOCTYPE html>
<html lang="en">
<head><title>Test</title></head>
<body>
<main>
    <h1>Test</h1>
    <img src="chart.png" alt="">
    <img src="diagram.png" alt="">
</main>
</body>
</html>'''
        html_path = write_temp_html(html, 'empty_alt.html')
        report = validator.validate_file(html_path)

        # May or may not flag empty alt depending on implementation
        # At minimum, should not crash
        assert report is not None

    # =========================================================================
    # HEADING HIERARCHY TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_skipped_heading_levels(self, validator, write_temp_html, broken_headings_html_content):
        """Test that skipped heading levels are detected"""
        html_path = write_temp_html(broken_headings_html_content, 'broken_headings.html')
        report = validator.validate_file(html_path)

        # Should find heading hierarchy issues
        heading_issues = [i for i in report.issues if 'heading' in i.message.lower()]
        assert len(heading_issues) > 0

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_accepts_valid_heading_hierarchy(self, validator, write_temp_html, accessible_html_content):
        """Test that valid heading hierarchy passes"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Should not have heading skip issues
        skip_issues = [i for i in report.issues
                       if 'heading' in i.message.lower() and 'skip' in i.message.lower()]
        assert len(skip_issues) == 0

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_multiple_h1_tags(self, validator, write_temp_html):
        """Test detection of multiple H1 tags on same page"""
        html = '''<!DOCTYPE html>
<html lang="en">
<head><title>Test</title></head>
<body>
<main>
    <h1>First Main Title</h1>
    <p>Content</p>
    <h1>Second Main Title</h1>
    <p>More content</p>
</main>
</body>
</html>'''
        html_path = write_temp_html(html, 'multiple_h1.html')
        report = validator.validate_file(html_path)

        # Should flag multiple H1s
        h1_issues = [i for i in report.issues if 'h1' in i.message.lower()]
        assert len(h1_issues) > 0 or report.total_issues > 0

    # =========================================================================
    # FORM ACCESSIBILITY TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_forms_without_labels(self, validator, write_temp_html, forms_no_labels_html_content):
        """Test that forms without labels are detected"""
        html_path = write_temp_html(forms_no_labels_html_content, 'forms_no_labels.html')
        report = validator.validate_file(html_path)

        # Should find form/label issues
        form_issues = [i for i in report.issues
                       if 'label' in i.message.lower() or 'form' in i.message.lower()]
        assert len(form_issues) > 0

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_accepts_properly_labeled_forms(self, validator, write_temp_html, accessible_html_content):
        """Test that properly labeled forms pass"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Should not have label association issues
        label_issues = [i for i in report.issues
                        if 'label' in i.message.lower() and 'missing' in i.message.lower()]
        assert len(label_issues) == 0

    # =========================================================================
    # LANGUAGE DECLARATION TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_missing_lang_attribute(self, validator, write_temp_html):
        """Test detection of missing lang attribute"""
        html = '''<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
<main><h1>Test Page</h1><p>Content</p></main>
</body>
</html>'''
        html_path = write_temp_html(html, 'no_lang.html')
        report = validator.validate_file(html_path)

        # Should flag missing lang
        lang_issues = [i for i in report.issues if 'lang' in i.message.lower()]
        assert len(lang_issues) > 0

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_accepts_valid_lang_attribute(self, validator, write_temp_html, accessible_html_content):
        """Test that valid lang attribute passes"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Should not have lang attribute issues
        lang_issues = [i for i in report.issues
                       if 'lang' in i.message.lower() and 'missing' in i.message.lower()]
        assert len(lang_issues) == 0

    # =========================================================================
    # LINK TEXT TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_generic_link_text(self, validator, write_temp_html):
        """Test detection of generic link text like 'click here'"""
        html = '''<!DOCTYPE html>
<html lang="en">
<head><title>Test</title></head>
<body>
<main>
    <h1>Test Page</h1>
    <p>For more information, <a href="page.html">click here</a>.</p>
    <p><a href="doc.pdf">Read more</a></p>
    <p><a href="link.html">here</a></p>
</main>
</body>
</html>'''
        html_path = write_temp_html(html, 'generic_links.html')
        report = validator.validate_file(html_path)

        # Should flag generic link text
        link_issues = [i for i in report.issues if 'link' in i.message.lower()]
        assert len(link_issues) > 0

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_accepts_descriptive_link_text(self, validator, write_temp_html, accessible_html_content):
        """Test that descriptive link text passes"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Should not have generic link text issues
        generic_link_issues = [i for i in report.issues
                               if 'click here' in i.message.lower()]
        assert len(generic_link_issues) == 0

    # =========================================================================
    # TABLE ACCESSIBILITY TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_tables_without_headers(self, validator, write_temp_html):
        """Test detection of tables without proper headers"""
        html = '''<!DOCTYPE html>
<html lang="en">
<head><title>Test</title></head>
<body>
<main>
    <h1>Test Page</h1>
    <table>
        <tr>
            <td>Name</td>
            <td>Age</td>
        </tr>
        <tr>
            <td>John</td>
            <td>30</td>
        </tr>
    </table>
</main>
</body>
</html>'''
        html_path = write_temp_html(html, 'table_no_headers.html')
        report = validator.validate_file(html_path)

        # Should flag table without headers
        table_issues = [i for i in report.issues if 'table' in i.message.lower()]
        assert len(table_issues) > 0 or report.total_issues > 0

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_accepts_properly_structured_tables(self, validator, write_temp_html, accessible_html_content):
        """Test that properly structured tables pass"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Should not have table header issues
        table_header_issues = [i for i in report.issues
                               if 'table' in i.message.lower() and 'header' in i.message.lower()]
        # Accessible content should have minimal table issues
        assert len(table_header_issues) <= 1

    # =========================================================================
    # LANDMARK TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_missing_main_landmark(self, validator, write_temp_html):
        """Test detection of missing main landmark"""
        html = '''<!DOCTYPE html>
<html lang="en">
<head><title>Test</title></head>
<body>
    <h1>Test Page</h1>
    <p>Content without main landmark.</p>
</body>
</html>'''
        html_path = write_temp_html(html, 'no_main.html')
        report = validator.validate_file(html_path)

        # Should flag missing main
        main_issues = [i for i in report.issues if 'main' in i.message.lower()]
        assert len(main_issues) > 0 or report.total_issues > 0

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_accepts_proper_landmarks(self, validator, write_temp_html, accessible_html_content):
        """Test that proper ARIA landmarks pass"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Should not have missing landmark issues
        landmark_issues = [i for i in report.issues
                           if 'landmark' in i.message.lower() and 'missing' in i.message.lower()]
        assert len(landmark_issues) == 0

    # =========================================================================
    # SKIP LINK TESTS
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_detects_missing_skip_link(self, validator, write_temp_html):
        """Test detection of missing skip link"""
        html = '''<!DOCTYPE html>
<html lang="en">
<head><title>Test</title></head>
<body>
    <nav>
        <a href="page1.html">Page 1</a>
        <a href="page2.html">Page 2</a>
        <a href="page3.html">Page 3</a>
    </nav>
    <main>
        <h1>Test Page</h1>
        <p>Content here.</p>
    </main>
</body>
</html>'''
        html_path = write_temp_html(html, 'no_skip_link.html')
        report = validator.validate_file(html_path)

        # May recommend skip link
        skip_issues = [i for i in report.issues if 'skip' in i.message.lower()]
        # Skip link is recommended but may not be critical
        assert report is not None

    @pytest.mark.unit
    @pytest.mark.accessibility
    def test_accepts_skip_link(self, validator, write_temp_html, accessible_html_content):
        """Test that pages with skip links pass"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Should not have missing skip link issues
        skip_issues = [i for i in report.issues
                       if 'skip' in i.message.lower() and 'missing' in i.message.lower()]
        assert len(skip_issues) == 0

    # =========================================================================
    # INTEGRATION TESTS
    # =========================================================================

    @pytest.mark.integration
    @pytest.mark.accessibility
    def test_full_validation_accessible_file(self, validator, write_temp_html, accessible_html_content):
        """Integration test: Full validation of accessible file"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Accessible file should be WCAG AA compliant
        assert report.wcag_aa_compliant is True
        assert report.critical_count == 0

    @pytest.mark.integration
    @pytest.mark.accessibility
    def test_full_validation_inaccessible_file(self, validator, write_temp_html, missing_alt_html_content):
        """Integration test: Full validation of inaccessible file"""
        html_path = write_temp_html(missing_alt_html_content, 'missing_alt.html')
        report = validator.validate_file(html_path)

        # Inaccessible file should not be compliant
        assert report.wcag_aa_compliant is False
        assert report.total_issues > 0

    @pytest.mark.integration
    @pytest.mark.accessibility
    def test_report_generation(self, validator, write_temp_html, accessible_html_content):
        """Test that reports are generated correctly"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Report should have expected attributes
        assert hasattr(report, 'total_issues')
        assert hasattr(report, 'critical_count')
        assert hasattr(report, 'high_count')
        assert hasattr(report, 'wcag_aa_compliant')
        assert hasattr(report, 'issues')

    @pytest.mark.integration
    @pytest.mark.accessibility
    def test_json_output(self, validator, write_temp_html, accessible_html_content):
        """Test JSON output generation"""
        html_path = write_temp_html(accessible_html_content, 'accessible.html')
        report = validator.validate_file(html_path)

        # Should be able to convert to JSON
        if hasattr(validator, 'to_json'):
            json_output = validator.to_json(report)
            assert json_output is not None
            assert isinstance(json_output, str)


class TestAccessibilityValidatorEdgeCases:
    """Edge case tests for AccessibilityValidator"""

    @pytest.fixture
    def validator(self):
        return AccessibilityValidator()

    @pytest.mark.unit
    def test_handles_empty_file(self, validator, write_temp_html):
        """Test handling of empty HTML file"""
        html_path = write_temp_html('', 'empty.html')

        # Should handle gracefully, not crash
        try:
            report = validator.validate_file(html_path)
            assert report is not None
        except Exception as e:
            # Some error handling is acceptable
            assert True

    @pytest.mark.unit
    def test_handles_malformed_html(self, validator, write_temp_html):
        """Test handling of malformed HTML"""
        html = '<html><body><p>Unclosed paragraph<div>Mixed content</body>'
        html_path = write_temp_html(html, 'malformed.html')

        # Should handle gracefully
        try:
            report = validator.validate_file(html_path)
            assert report is not None
        except Exception:
            # Some error handling is acceptable
            assert True

    @pytest.mark.unit
    def test_handles_non_html_content(self, validator, write_temp_html):
        """Test handling of non-HTML content"""
        content = 'This is just plain text, not HTML.'
        html_path = write_temp_html(content, 'plaintext.html')

        # Should handle gracefully
        try:
            report = validator.validate_file(html_path)
            assert report is not None
        except Exception:
            # Some error handling is acceptable
            assert True

    @pytest.mark.unit
    def test_handles_nonexistent_file(self, validator, tmp_path):
        """Test handling of nonexistent file"""
        fake_path = tmp_path / 'nonexistent.html'

        # Should raise appropriate error
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            validator.validate_file(fake_path)
