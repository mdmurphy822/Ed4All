"""
Tests for lib/secure_paths.py - Path validation and ZIP security.
"""
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from lib.secure_paths import (
        PathTraversalError,
        ZipSlipError,
        is_safe_path,
        safe_extract_zip,
        safe_join_path,
        sanitize_path_component,
        validate_path_within_root,
    )
except ImportError:
    pytest.skip("secure_paths not available", allow_module_level=True)


# =============================================================================
# PATH VALIDATION TESTS
# =============================================================================

class TestPathValidation:
    """Test validate_path_within_root function."""

    @pytest.mark.unit
    @pytest.mark.security
    def test_valid_path_within_root(self, temp_root, temp_file_in_root):
        """Valid path within root should return resolved path."""
        result = validate_path_within_root(temp_file_in_root, temp_root)
        assert result == temp_file_in_root.resolve()

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_path_traversal_dotdot(self, temp_root):
        """Path with .. escaping root should raise PathTraversalError."""
        escape_path = temp_root / ".." / "escape.txt"
        with pytest.raises(PathTraversalError):
            validate_path_within_root(escape_path, temp_root)

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_absolute_path_escape(self, temp_root):
        """Absolute path outside root should raise PathTraversalError."""
        external_path = Path("/etc/passwd")
        with pytest.raises(PathTraversalError):
            validate_path_within_root(external_path, temp_root)

    @pytest.mark.unit
    @pytest.mark.security
    def test_handles_symlink_in_root(self, temp_root, temp_file_in_root):
        """Symlink pointing within root should be accepted."""
        link_path = temp_root / "link.txt"
        link_path.symlink_to(temp_file_in_root)

        result = validate_path_within_root(link_path, temp_root)
        assert result.exists()

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_symlink_escape(self, temp_root, tmp_path):
        """Symlink pointing outside root should raise PathTraversalError."""
        external_file = tmp_path / "external.txt"
        external_file.write_text("external")

        link_path = temp_root / "escape_link.txt"
        link_path.symlink_to(external_file)

        with pytest.raises(PathTraversalError):
            validate_path_within_root(link_path, temp_root)

    @pytest.mark.unit
    @pytest.mark.security
    def test_must_exist_raises_for_missing(self, temp_root):
        """must_exist=True should raise FileNotFoundError for missing path."""
        missing_path = temp_root / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            validate_path_within_root(missing_path, temp_root, must_exist=True)

    @pytest.mark.unit
    @pytest.mark.security
    def test_resolves_relative_paths(self, temp_root, temp_file_in_root):
        """Relative paths should be resolved correctly."""
        # Change to temp_root and use relative path
        original_dir = os.getcwd()
        try:
            os.chdir(temp_root)
            relative_path = Path("test_file.txt")
            result = validate_path_within_root(relative_path, temp_root)
            assert result.exists()
        finally:
            os.chdir(original_dir)


# =============================================================================
# PATH SANITIZATION TESTS
# =============================================================================

class TestPathSanitization:
    """Test sanitize_path_component function."""

    @pytest.mark.unit
    @pytest.mark.security
    def test_sanitizes_alphanumeric(self):
        """Alphanumeric names should pass through."""
        result = sanitize_path_component("valid_name123")
        assert result == "valid_name123"

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_dotdot_pattern(self):
        """Names with .. should raise ValueError."""
        with pytest.raises(ValueError, match="traversal"):
            sanitize_path_component("../escape")

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_path_separators(self):
        """Names with / or \\ should raise ValueError."""
        with pytest.raises(ValueError, match="separators"):
            sanitize_path_component("path/with/slashes")

        with pytest.raises(ValueError, match="separators"):
            sanitize_path_component("path\\with\\backslashes")

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_hidden_files_by_default(self):
        """Names starting with . should raise unless allow_dots=True."""
        with pytest.raises(ValueError, match="Hidden"):
            sanitize_path_component(".hidden")

        # Should work with allow_dots=True
        result = sanitize_path_component(".hidden", allow_dots=True)
        assert "hidden" in result

    @pytest.mark.unit
    @pytest.mark.security
    def test_replaces_dangerous_chars(self):
        """Dangerous characters should be replaced with underscores."""
        result = sanitize_path_component("name!@#$%test")
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result

    @pytest.mark.unit
    @pytest.mark.security
    def test_enforces_length_limit(self):
        """Names exceeding max_length should be truncated."""
        long_name = "a" * 300
        result = sanitize_path_component(long_name, max_length=100)
        assert len(result) <= 100

    @pytest.mark.unit
    @pytest.mark.security
    def test_handles_empty_result(self):
        """Names that sanitize to empty should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            sanitize_path_component("   ")

    @pytest.mark.unit
    @pytest.mark.security
    def test_collapses_multiple_underscores(self):
        """Multiple underscores should be collapsed."""
        result = sanitize_path_component("name___test")
        assert "___" not in result


# =============================================================================
# ZIP EXTRACTION TESTS
# =============================================================================

class TestZipExtraction:
    """Test safe_extract_zip function."""

    @pytest.mark.unit
    @pytest.mark.security
    def test_extracts_valid_zip(self, valid_zip, tmp_path):
        """Valid ZIP should extract successfully."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        count = safe_extract_zip(valid_zip, extract_dir)

        assert count == 3
        assert (extract_dir / "file1.txt").exists()
        assert (extract_dir / "subdir" / "file2.txt").exists()

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_zip_slip_attack(self, zip_with_traversal, tmp_path):
        """ZIP with path traversal should raise ZipSlipError."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with pytest.raises(ZipSlipError):
            safe_extract_zip(zip_with_traversal, extract_dir)

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_absolute_paths_in_zip(self, zip_with_absolute_path, tmp_path):
        """ZIP with absolute paths should raise ZipSlipError."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with pytest.raises(ZipSlipError):
            safe_extract_zip(zip_with_absolute_path, extract_dir)

    @pytest.mark.unit
    @pytest.mark.security
    def test_enforces_file_count_limit(self, large_zip, tmp_path):
        """ZIP exceeding file count limit should raise ValueError."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with pytest.raises(ValueError, match="too many files"):
            safe_extract_zip(large_zip, extract_dir, max_file_count=10)

    @pytest.mark.unit
    @pytest.mark.security
    def test_enforces_size_limit(self, tmp_path):
        """ZIP exceeding size limit should raise ValueError."""
        # Create ZIP with large content
        zip_path = tmp_path / "big.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("large.txt", "x" * (2 * 1024 * 1024))  # 2MB

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with pytest.raises(ValueError, match="exceeds maximum size"):
            safe_extract_zip(zip_path, extract_dir, max_total_size_mb=1)

    @pytest.mark.unit
    @pytest.mark.security
    def test_filters_by_extension(self, tmp_path):
        """ZIP with blocked extensions should raise ValueError."""
        zip_path = tmp_path / "mixed.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("good.html", "<html></html>")
            zf.writestr("bad.exe", "executable")

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with pytest.raises(ValueError, match="Blocked file extension"):
            safe_extract_zip(
                zip_path, extract_dir,
                allowed_extensions={'.html', '.txt'}
            )

    @pytest.mark.unit
    @pytest.mark.security
    def test_missing_zip_raises_error(self, tmp_path):
        """Non-existent ZIP should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            safe_extract_zip(
                tmp_path / "missing.zip",
                tmp_path / "extracted"
            )


# =============================================================================
# SAFE JOIN PATH TESTS
# =============================================================================

class TestSafeJoinPath:
    """Test safe_join_path function."""

    @pytest.mark.unit
    @pytest.mark.security
    def test_joins_valid_components(self, temp_root):
        """Valid components should be joined correctly."""
        result = safe_join_path(temp_root, "subdir", "file.txt")
        assert result == temp_root / "subdir" / "file.txt"

    @pytest.mark.unit
    @pytest.mark.security
    def test_applies_sanitization(self, temp_root):
        """Components should be sanitized during join."""
        result = safe_join_path(temp_root, "name!@#test", "file.txt")
        # Special chars should be replaced
        assert "!" not in str(result)
        assert "@" not in str(result)

    @pytest.mark.unit
    @pytest.mark.security
    def test_validates_against_root(self, temp_root):
        """Result should be validated against allowed_root."""
        # This should work - stays within root
        result = safe_join_path(
            temp_root, "subdir", "file.txt",
            allowed_root=temp_root
        )
        assert result.is_relative_to(temp_root) or str(temp_root) in str(result)


# =============================================================================
# IS SAFE PATH TESTS
# =============================================================================

class TestIsSafePath:
    """Test is_safe_path function."""

    @pytest.mark.unit
    @pytest.mark.security
    def test_returns_true_for_safe_paths(self, temp_root, temp_file_in_root):
        """Safe paths should return True."""
        assert is_safe_path(temp_file_in_root, temp_root) is True

    @pytest.mark.unit
    @pytest.mark.security
    def test_returns_false_for_unsafe_paths(self, temp_root):
        """Unsafe paths should return False."""
        escape_path = temp_root / ".." / "escape.txt"
        assert is_safe_path(escape_path, temp_root) is False

    @pytest.mark.unit
    @pytest.mark.security
    def test_no_exception_raised(self, temp_root):
        """Function should not raise, only return bool."""
        # External path
        result = is_safe_path(Path("/etc/passwd"), temp_root)
        assert result is False  # No exception raised
