"""Smoke tests for DART converter module."""

import pytest
from DART.pdf_converter.converter import ConversionResult, TextBlock


@pytest.mark.unit
@pytest.mark.dart
class TestConversionResult:
    """Test ConversionResult dataclass."""

    def test_default_values(self):
        result = ConversionResult(success=True)
        assert result.success is True
        assert result.html_path == ""
        assert result.error == ""
        assert result.pages_processed == 0
        assert result.wcag_compliant is True

    def test_failed_result(self):
        result = ConversionResult(success=False, error="File not found")
        assert result.success is False
        assert result.error == "File not found"

    def test_successful_result_with_stats(self):
        result = ConversionResult(
            success=True,
            html_path="/tmp/output.html",
            pages_processed=5,
            total_words=1200,
            title="Test Document",
            images_extracted=3,
            images_with_alt_text=2,
        )
        assert result.pages_processed == 5
        assert result.total_words == 1200
        assert result.images_extracted == 3


@pytest.mark.unit
@pytest.mark.dart
class TestTextBlock:
    """Test TextBlock dataclass."""

    def test_default_type(self):
        block = TextBlock(text="Hello")
        assert block.block_type == "paragraph"
        assert block.heading_level == 0

    def test_heading_block(self):
        block = TextBlock(text="Chapter 1", block_type="heading", heading_level=1)
        assert block.block_type == "heading"
        assert block.heading_level == 1
