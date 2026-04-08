"""
Tests for lib/write_facade.py - Controlled file writes with transactions.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from lib.write_facade import (
        PathSecurityError,
        TransactionError,
        TransactionResult,  # noqa: F401
        WriteFacade,
        WriteResult,
        WriteTracker,
        atomic_write,
        create_run_write_facade,
    )
except ImportError:
    pytest.skip("write_facade not available", allow_module_level=True)


# =============================================================================
# PATH VALIDATION TESTS
# =============================================================================

class TestPathValidation:
    """Test path validation in WriteFacade."""

    @pytest.fixture
    def facade(self, tmp_path):
        """Create facade with tmp_path as allowed."""
        return WriteFacade(allowed_paths=[tmp_path])

    @pytest.mark.unit
    @pytest.mark.security
    def test_validates_path_in_root(self, facade, tmp_path):
        """Paths within allowed root should pass validation."""
        valid_path = tmp_path / "test.txt"
        # Should not raise
        facade.validate_path(valid_path)

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_path_outside_root(self, facade, tmp_path):
        """Paths outside allowed root should raise PathSecurityError."""
        external_path = Path("/etc/passwd")

        with pytest.raises(PathSecurityError):
            facade.validate_path(external_path)

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_parent_traversal(self, facade, tmp_path):
        """Paths with .. should raise PathSecurityError."""
        traversal_path = tmp_path / ".." / "escape.txt"

        with pytest.raises(PathSecurityError):
            facade.validate_path(traversal_path)

    @pytest.mark.unit
    @pytest.mark.security
    def test_rejects_long_paths(self, tmp_path):
        """Paths exceeding max length should raise PathSecurityError."""
        facade = WriteFacade(
            allowed_paths=[tmp_path],
            max_path_length=50
        )
        long_path = tmp_path / ("a" * 100 + ".txt")

        with pytest.raises(PathSecurityError):
            facade.validate_path(long_path)

    @pytest.mark.unit
    @pytest.mark.security
    def test_enforcement_can_be_disabled(self, tmp_path):
        """With enforcement disabled, all paths allowed."""
        facade = WriteFacade(
            allowed_paths=[tmp_path],
            enforce_allowed_paths=False
        )

        # This would normally fail
        external_path = Path("/tmp/anywhere.txt")
        facade.validate_path(external_path)  # Should not raise


# =============================================================================
# WRITE OPERATIONS TESTS
# =============================================================================

class TestWriteOperations:
    """Test file write operations."""

    @pytest.fixture
    def facade(self, tmp_path):
        """Create facade with tmp_path as allowed."""
        return WriteFacade(allowed_paths=[tmp_path])

    @pytest.mark.unit
    def test_write_creates_file(self, facade, tmp_path):
        """Should create file with content."""
        file_path = tmp_path / "test.txt"
        result = facade.write(file_path, "Hello, World!")

        assert result.success is True
        assert file_path.exists()
        assert file_path.read_text() == "Hello, World!"

    @pytest.mark.unit
    def test_write_creates_parent_dirs(self, facade, tmp_path):
        """Should create parent directories if needed."""
        file_path = tmp_path / "nested" / "dir" / "test.txt"
        result = facade.write(file_path, "Content")

        assert result.success is True
        assert file_path.exists()
        assert file_path.parent.exists()

    @pytest.mark.unit
    def test_write_bytes(self, facade, tmp_path):
        """Should write bytes content."""
        file_path = tmp_path / "binary.dat"
        content = b"\x00\x01\x02\x03"
        result = facade.write(file_path, content)

        assert result.success is True
        assert file_path.read_bytes() == content

    @pytest.mark.unit
    def test_write_result_includes_hash(self, facade, tmp_path):
        """WriteResult should include content hash."""
        file_path = tmp_path / "test.txt"
        result = facade.write(file_path, "Content")

        assert result.content_hash is not None
        assert result.content_hash.startswith("sha256:")

    @pytest.mark.unit
    def test_write_result_includes_bytes_written(self, facade, tmp_path):
        """WriteResult should include bytes written count."""
        file_path = tmp_path / "test.txt"
        content = "Hello, World!"
        result = facade.write(file_path, content)

        assert result.bytes_written == len(content.encode())

    @pytest.mark.unit
    def test_write_outside_allowed_returns_error(self, facade):
        """Writing outside allowed paths returns error result."""
        result = facade.write("/etc/test.txt", "Content")

        assert result.success is False
        assert result.error is not None


# =============================================================================
# WRITE JSON TESTS
# =============================================================================

class TestWriteJson:
    """Test JSON write operations."""

    @pytest.fixture
    def facade(self, tmp_path):
        return WriteFacade(allowed_paths=[tmp_path])

    @pytest.mark.unit
    def test_write_json_serializes(self, facade, tmp_path):
        """Should serialize and write JSON data."""
        file_path = tmp_path / "data.json"
        data = {"key": "value", "count": 42}

        result = facade.write_json(file_path, data)

        assert result.success is True
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    @pytest.mark.unit
    def test_write_json_handles_non_serializable(self, facade, tmp_path):
        """Non-serializable data should return error."""
        file_path = tmp_path / "data.json"
        data = {"func": lambda x: x}  # Can't serialize

        result = facade.write_json(file_path, data)

        assert result.success is False
        assert "serialization" in result.error.lower()


# =============================================================================
# TRANSACTION TESTS
# =============================================================================

class TestTransactions:
    """Test transaction support."""

    @pytest.fixture
    def facade(self, tmp_path):
        return WriteFacade(allowed_paths=[tmp_path])

    @pytest.mark.unit
    def test_begin_transaction(self, facade):
        """Should begin a transaction."""
        facade.begin_transaction()
        assert facade._in_transaction is True

    @pytest.mark.unit
    def test_begin_transaction_twice_raises(self, facade):
        """Beginning transaction twice should raise."""
        facade.begin_transaction()
        with pytest.raises(TransactionError):
            facade.begin_transaction()

    @pytest.mark.unit
    def test_commit_transaction(self, facade, tmp_path):
        """Commit should finalize writes."""
        file_path = tmp_path / "test.txt"

        facade.begin_transaction()
        facade.write(file_path, "Content")
        result = facade.commit_transaction()

        assert result.success is True
        assert result.writes_completed == 1
        assert file_path.exists()

    @pytest.mark.unit
    def test_rollback_restores_backup(self, facade, tmp_path):
        """Rollback should restore original content."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Original")

        facade.begin_transaction()
        facade.write(file_path, "Modified")
        facade.rollback_transaction()

        assert file_path.read_text() == "Original"

    @pytest.mark.unit
    def test_rollback_deletes_new_files(self, facade, tmp_path):
        """Rollback should delete files that didn't exist before."""
        file_path = tmp_path / "new.txt"

        facade.begin_transaction()
        facade.write(file_path, "New content")
        assert file_path.exists()

        facade.rollback_transaction()
        assert not file_path.exists()

    @pytest.mark.unit
    def test_context_manager_commits(self, facade, tmp_path):
        """Context manager should commit on success."""
        file_path = tmp_path / "test.txt"

        with facade:
            facade.write(file_path, "Content")

        assert file_path.exists()
        assert facade._in_transaction is False

    @pytest.mark.unit
    def test_context_manager_rollback_on_error(self, facade, tmp_path):
        """Context manager should rollback on exception."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Original")

        try:
            with facade:
                facade.write(file_path, "Modified")
                raise ValueError("Simulated error")
        except ValueError:
            pass

        assert file_path.read_text() == "Original"


# =============================================================================
# WRITE TRACKER TESTS
# =============================================================================

class TestWriteTracker:
    """Test WriteTracker functionality."""

    @pytest.mark.unit
    def test_tracks_write_operations(self):
        """Should track write results."""
        tracker = WriteTracker()

        tracker.track(WriteResult(
            success=True, path="/path/file1.txt", bytes_written=100
        ))
        tracker.track(WriteResult(
            success=True, path="/path/file2.txt", bytes_written=200
        ))

        assert len(tracker.writes) == 2

    @pytest.mark.unit
    def test_get_summary(self):
        """Should return summary statistics."""
        tracker = WriteTracker()

        tracker.track(WriteResult(
            success=True, path="/path/file1.txt", bytes_written=100
        ))
        tracker.track(WriteResult(
            success=False, path="/path/file2.txt", error="Failed"
        ))

        summary = tracker.get_summary()

        assert summary["total_writes"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert summary["total_bytes"] == 100

    @pytest.mark.unit
    def test_to_audit_log(self):
        """Should convert to audit log format."""
        tracker = WriteTracker()

        tracker.track(WriteResult(
            success=True, path="/path/file.txt", bytes_written=100
        ))

        log = tracker.to_audit_log()

        assert len(log) == 1
        assert log[0]["path"] == "/path/file.txt"
        assert log[0]["success"] is True


# =============================================================================
# AUDIT CALLBACK TESTS
# =============================================================================

class TestAuditCallback:
    """Test audit callback functionality."""

    @pytest.mark.unit
    def test_audit_callback_called(self, tmp_path):
        """Audit callback should be called on successful write."""
        results = []

        def callback(result):
            results.append(result)

        facade = WriteFacade(
            allowed_paths=[tmp_path],
            audit_callback=callback
        )

        facade.write(tmp_path / "test.txt", "Content")

        assert len(results) == 1
        assert results[0].success is True


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.unit
    def test_atomic_write(self, tmp_path):
        """atomic_write should write file atomically."""
        file_path = tmp_path / "atomic.txt"

        result = atomic_write(file_path, "Content")

        assert result.success is True
        assert file_path.exists()
        assert file_path.read_text() == "Content"

    @pytest.mark.unit
    def test_create_run_write_facade(self, tmp_path):
        """Should create facade for run directory."""
        run_path = tmp_path / "run_001"
        run_path.mkdir()

        facade = create_run_write_facade(run_path)

        # Should allow writes to run subdirectories
        decisions_dir = run_path / "decisions"
        decisions_dir.mkdir()

        result = facade.write(decisions_dir / "test.json", "{}")
        assert result.success is True
