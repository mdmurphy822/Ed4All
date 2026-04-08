"""Smoke tests for LibV2 catalog models and validation."""

import sys
from pathlib import Path

import pytest

# Ensure project root is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LibV2.tools.libv2.models.catalog import CatalogEntry
from LibV2.tools.libv2.validator import ValidationResult


@pytest.mark.unit
class TestCatalogEntry:
    """Test CatalogEntry model."""

    def test_construction(self):
        entry = CatalogEntry(
            slug="test-course",
            title="Test Course",
            division="STEM",
            primary_domain="computer-science",
        )
        assert entry.slug == "test-course"
        assert entry.division == "STEM"
        assert entry.chunk_count == 0
        assert entry.validation_status == "pending"

    def test_to_dict(self):
        entry = CatalogEntry(
            slug="phys-101",
            title="Physics 101",
            division="STEM",
            primary_domain="physics",
            secondary_domains=["mathematics"],
        )
        d = entry.to_dict()
        assert d["slug"] == "phys-101"
        assert d["secondary_domains"] == ["mathematics"]
        assert "primary_domain" in d

    def test_from_dict(self):
        data = {
            "slug": "math-201",
            "title": "Math 201",
            "division": "STEM",
            "primary_domain": "mathematics",
        }
        entry = CatalogEntry.from_dict(data)
        assert entry.slug == "math-201"
        assert entry.language == "en"  # default

    def test_roundtrip(self):
        entry = CatalogEntry(
            slug="art-100",
            title="Art History",
            division="ARTS",
            primary_domain="art-history",
            chunk_count=42,
        )
        d = entry.to_dict()
        restored = CatalogEntry.from_dict(d)
        assert restored.slug == entry.slug
        assert restored.chunk_count == 42


@pytest.mark.unit
class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_import(self):
        result = ValidationResult(valid=True, errors=[], warnings=[])
        assert result.valid is True
