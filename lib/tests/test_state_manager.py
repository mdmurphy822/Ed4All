"""
Tests for lib/state_manager.py - Atomic JSON operations.
"""
import pytest
import sys
import json
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from lib.state_manager import (
        atomic_write_json,
        atomic_read_json,
        atomic_update_json,
    )
except ImportError:
    pytest.skip("state_manager not available", allow_module_level=True)


# =============================================================================
# ATOMIC WRITE JSON TESTS
# =============================================================================

class TestAtomicWriteJson:
    """Test atomic_write_json function."""

    @pytest.mark.unit
    def test_writes_valid_json(self, tmp_path, sample_json_data):
        """Should write valid JSON file."""
        json_path = tmp_path / "test.json"

        atomic_write_json(json_path, sample_json_data)

        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded == sample_json_data

    @pytest.mark.unit
    def test_creates_parent_directory(self, tmp_path, sample_json_data):
        """Should create parent directories if they don't exist."""
        json_path = tmp_path / "nested" / "dir" / "test.json"

        atomic_write_json(json_path, sample_json_data)

        assert json_path.exists()
        assert json_path.parent.exists()

    @pytest.mark.unit
    def test_atomic_rename_on_success(self, tmp_path, sample_json_data):
        """Temp file should be removed after atomic rename."""
        json_path = tmp_path / "test.json"

        atomic_write_json(json_path, sample_json_data)

        temp_path = json_path.with_suffix('.tmp')
        assert not temp_path.exists()
        assert json_path.exists()

    @pytest.mark.unit
    def test_handles_non_serializable_error(self, tmp_path):
        """Should raise TypeError for non-JSON-serializable data."""
        json_path = tmp_path / "test.json"
        bad_data = {"func": lambda x: x}  # Functions aren't serializable

        with pytest.raises(TypeError):
            atomic_write_json(json_path, bad_data)

    @pytest.mark.unit
    def test_overwrites_existing_file(self, tmp_path):
        """Should overwrite existing file."""
        json_path = tmp_path / "test.json"

        # Write initial data
        atomic_write_json(json_path, {"version": 1})

        # Overwrite
        atomic_write_json(json_path, {"version": 2})

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["version"] == 2

    @pytest.mark.unit
    def test_respects_indent_parameter(self, tmp_path, sample_json_data):
        """Should respect indent parameter for formatting."""
        json_path = tmp_path / "test.json"

        atomic_write_json(json_path, sample_json_data, indent=4)

        content = json_path.read_text()
        # With indent=4, there should be 4-space indentation
        assert "    " in content


# =============================================================================
# ATOMIC READ JSON TESTS
# =============================================================================

class TestAtomicReadJson:
    """Test atomic_read_json function."""

    @pytest.mark.unit
    def test_reads_valid_json(self, temp_json_file, sample_json_data):
        """Should read and parse valid JSON file."""
        result = atomic_read_json(temp_json_file)

        assert result == sample_json_data

    @pytest.mark.unit
    def test_returns_default_if_missing(self, tmp_path):
        """Should return default value if file doesn't exist."""
        missing_path = tmp_path / "missing.json"
        default = {"status": "new"}

        result = atomic_read_json(missing_path, default=default)

        assert result == default

    @pytest.mark.unit
    def test_raises_if_missing_without_default(self, tmp_path):
        """Should raise FileNotFoundError if no default provided."""
        missing_path = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            atomic_read_json(missing_path)

    @pytest.mark.unit
    def test_raises_on_invalid_json(self, tmp_path):
        """Should raise JSONDecodeError for invalid JSON."""
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("{ not valid json")

        with pytest.raises(json.JSONDecodeError):
            atomic_read_json(invalid_path)

    @pytest.mark.unit
    def test_reads_empty_object(self, tmp_path):
        """Should handle empty JSON object."""
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("{}")

        result = atomic_read_json(empty_path)

        assert result == {}


# =============================================================================
# ATOMIC UPDATE JSON TESTS
# =============================================================================

class TestAtomicUpdateJson:
    """Test atomic_update_json function."""

    @pytest.mark.unit
    def test_updates_existing_file(self, temp_json_file, sample_json_data):
        """Should read, update, and write existing file."""
        def add_field(data):
            data["new_field"] = "new_value"
            return data

        result = atomic_update_json(temp_json_file, add_field)

        assert result["new_field"] == "new_value"
        # Original data should be preserved
        assert result["name"] == sample_json_data["name"]

        # Verify file was updated
        with open(temp_json_file) as f:
            loaded = json.load(f)
        assert loaded["new_field"] == "new_value"

    @pytest.mark.unit
    def test_creates_with_default(self, tmp_path):
        """Should create file with default if it doesn't exist."""
        new_path = tmp_path / "new.json"

        def set_status(data):
            data["status"] = "initialized"
            return data

        result = atomic_update_json(new_path, set_status, default={"version": 1})

        assert result["version"] == 1
        assert result["status"] == "initialized"
        assert new_path.exists()

    @pytest.mark.unit
    def test_applies_update_function(self, tmp_path):
        """Update function should be applied correctly."""
        json_path = tmp_path / "counter.json"
        atomic_write_json(json_path, {"counter": 0})

        def increment_counter(data):
            data["counter"] = data.get("counter", 0) + 1
            return data

        result = atomic_update_json(json_path, increment_counter)

        assert result["counter"] == 1

    @pytest.mark.unit
    def test_returns_updated_data(self, tmp_path):
        """Should return the updated dictionary."""
        json_path = tmp_path / "test.json"
        atomic_write_json(json_path, {"key": "old"})

        def update_key(data):
            data["key"] = "new"
            return data

        result = atomic_update_json(json_path, update_key)

        assert result["key"] == "new"

    @pytest.mark.unit
    def test_empty_default_when_missing(self, tmp_path):
        """Should use empty dict if file missing and no default."""
        new_path = tmp_path / "new.json"

        def add_key(data):
            data["added"] = True
            return data

        result = atomic_update_json(new_path, add_key)

        assert result["added"] is True
        assert new_path.exists()


# =============================================================================
# CONCURRENCY TESTS
# =============================================================================

class TestConcurrency:
    """Test concurrent access behavior."""

    @pytest.mark.integration
    def test_concurrent_writes_dont_corrupt(self, tmp_path):
        """Multiple concurrent writes should not corrupt file."""
        json_path = tmp_path / "concurrent.json"
        atomic_write_json(json_path, {"counter": 0})

        results = []
        errors = []

        def increment():
            try:
                for _ in range(10):
                    def inc(data):
                        data["counter"] = data.get("counter", 0) + 1
                        return data
                    atomic_update_json(json_path, inc)
                    time.sleep(0.001)
                results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0

        # File should still be valid JSON
        final = atomic_read_json(json_path)
        assert "counter" in final
        # Counter should have been incremented (exact value depends on race conditions)
        assert final["counter"] > 0

    @pytest.mark.integration
    def test_concurrent_reads_work(self, temp_json_file, sample_json_data):
        """Multiple concurrent reads should work correctly."""
        results = []
        errors = []

        def read_file():
            try:
                for _ in range(10):
                    data = atomic_read_json(temp_json_file)
                    results.append(data)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_file) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors
        assert len(errors) == 0

        # All reads should return same data
        assert all(r == sample_json_data for r in results)
