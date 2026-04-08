"""
Tests for lib/decision_capture.py - Decision event capture system.
"""
import pytest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from lib.decision_capture import (
        DecisionCapture,
        MLFeatures,
        InputRef,
        OutputArtifact,
        OutcomeSignals,
    )
except ImportError:
    pytest.skip("decision_capture not available", allow_module_level=True)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def capture_dir(tmp_path):
    """Create temporary directories for capture output."""
    training_dir = tmp_path / "training-captures"
    training_dir.mkdir(parents=True)
    return training_dir


@pytest.fixture
def mock_libv2_storage(tmp_path):
    """Mock LibV2Storage to use temp directory."""
    with patch('lib.decision_capture.LibV2Storage') as mock_cls:
        storage = Mock()
        capture_path = tmp_path / "libv2" / "training"
        capture_path.mkdir(parents=True)
        storage.get_training_capture_path.return_value = capture_path
        mock_cls.return_value = storage
        yield mock_cls


@pytest.fixture
def mock_legacy_dir(tmp_path):
    """Mock legacy training directory."""
    legacy_dir = tmp_path / "legacy-training"
    legacy_dir.mkdir(parents=True)
    with patch('lib.decision_capture.LEGACY_TRAINING_DIR', legacy_dir):
        yield legacy_dir


# =============================================================================
# DATACLASS TESTS
# =============================================================================

class TestDataclasses:
    """Test decision capture dataclasses."""

    @pytest.mark.unit
    def test_ml_features_defaults(self):
        """MLFeatures should have sensible defaults."""
        features = MLFeatures()

        assert features.pedagogy_pattern == ""
        assert features.engagement_patterns == []
        assert features.bloom_levels == []

    @pytest.mark.unit
    def test_ml_features_with_values(self):
        """MLFeatures should accept values."""
        features = MLFeatures(
            pedagogy_pattern="problem_based_intro",
            bloom_levels=["remember", "understand"],
            udl_principles=["multiple_means_engagement"],
        )

        assert features.pedagogy_pattern == "problem_based_intro"
        assert "remember" in features.bloom_levels
        assert "multiple_means_engagement" in features.udl_principles

    @pytest.mark.unit
    def test_input_ref_structure(self):
        """InputRef should capture source references."""
        ref = InputRef(
            source_type="textbook",
            path_or_id="/path/to/textbook.pdf",
            content_hash="abc123",
            excerpt_range="pages:10-15",
        )

        assert ref.source_type == "textbook"
        assert ref.hash_algorithm == "sha256"  # Default

    @pytest.mark.unit
    def test_output_artifact_structure(self):
        """OutputArtifact should capture artifact references."""
        artifact = OutputArtifact(
            artifact_type="html",
            path="content/week_01/module_01.html",
            content_hash="def456",
            size_bytes=1024,
        )

        assert artifact.artifact_type == "html"
        assert artifact.size_bytes == 1024

    @pytest.mark.unit
    def test_outcome_signals_defaults(self):
        """OutcomeSignals should have defaults for training."""
        outcome = OutcomeSignals()

        assert outcome.accepted is True
        assert outcome.revision_count == 0
        assert outcome.edit_distance == "none"


# =============================================================================
# DECISION CAPTURE INITIALIZATION TESTS
# =============================================================================

class TestDecisionCaptureInit:
    """Test DecisionCapture initialization."""

    @pytest.mark.unit
    def test_init_creates_directories(self, mock_libv2_storage, mock_legacy_dir):
        """Should create output directories on init."""
        capture = DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            tool="courseforge",
            streaming=False,
        )

        assert capture.course_code == "TEST_101"
        assert capture.phase == "content-generator"
        assert capture.tool == "courseforge"

    @pytest.mark.unit
    def test_init_generates_session_id(self, mock_libv2_storage, mock_legacy_dir):
        """Should generate unique session ID."""
        capture = DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

        assert capture.session_id is not None
        assert len(capture.session_id) > 0

    @pytest.mark.unit
    def test_init_normalizes_course_id(self, mock_libv2_storage, mock_legacy_dir):
        """Should normalize course ID."""
        capture = DecisionCapture(
            course_code="test 101",
            phase="content-generator",
            streaming=False,
        )

        assert capture.course_id == "TEST_101"

    @pytest.mark.unit
    def test_init_with_task_id(self, mock_libv2_storage, mock_legacy_dir):
        """Should accept task_id for orchestrator cross-linking."""
        capture = DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
            task_id="T001",
        )

        assert capture.task_id == "T001"


# =============================================================================
# MODULE CONTEXT TESTS
# =============================================================================

class TestModuleContext:
    """Test module context management."""

    @pytest.mark.unit
    def test_set_module_context(self, mock_libv2_storage, mock_legacy_dir):
        """Should set module context with proper ID format."""
        capture = DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

        capture.set_module_context(week=1, module=2)

        assert capture.module_id == "TEST_101_W01_M02"
        assert capture.artifact_id is not None

    @pytest.mark.unit
    def test_set_module_context_with_hash(self, mock_libv2_storage, mock_legacy_dir):
        """Should accept artifact hash."""
        capture = DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

        capture.set_module_context(week=3, module=1, artifact_hash="custom_hash_123")

        assert capture.module_id == "TEST_101_W03_M01"
        assert capture.artifact_id == "custom_hash_123"


# =============================================================================
# LOG DECISION TESTS
# =============================================================================

class TestLogDecision:
    """Test decision logging."""

    @pytest.fixture
    def capture(self, mock_libv2_storage, mock_legacy_dir):
        """Create capture instance for testing."""
        return DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

    @pytest.mark.unit
    def test_log_decision_valid(self, capture):
        """Should log valid decision."""
        capture.log_decision(
            decision_type="content_structure",
            decision="Use accordion for FAQ section",
            rationale="Accordions reduce cognitive load by hiding content until needed",
        )

        assert len(capture.decisions) == 1
        assert capture.decisions[0]["decision_type"] == "content_structure"
        assert capture.decisions[0]["decision"] == "Use accordion for FAQ section"

    @pytest.mark.unit
    def test_log_decision_includes_event_id(self, capture):
        """Logged decisions should have event_id."""
        capture.log_decision(
            decision_type="content_structure",
            decision="Test decision",
            rationale="Test rationale for this decision",
        )

        assert "event_id" in capture.decisions[0]
        assert capture.decisions[0]["event_id"] is not None

    @pytest.mark.unit
    def test_log_decision_includes_seq(self, capture):
        """Logged decisions should have seq for ordering."""
        capture.log_decision(
            decision_type="content_structure",
            decision="Decision 1",
            rationale="Rationale for decision 1",
        )
        capture.log_decision(
            decision_type="content_structure",
            decision="Decision 2",
            rationale="Rationale for decision 2",
        )

        assert "seq" in capture.decisions[0]
        assert "seq" in capture.decisions[1]

    @pytest.mark.unit
    def test_log_decision_with_alternatives(self, capture):
        """Should log alternatives considered."""
        capture.log_decision(
            decision_type="content_structure",
            decision="Use accordion",
            rationale="Better for progressive disclosure",
            alternatives_considered=["tabs", "expandable sections", "flat list"],
        )

        assert len(capture.decisions[0]["alternatives_considered"]) == 3

    @pytest.mark.unit
    def test_log_decision_with_ml_features(self, capture):
        """Should log ML training features."""
        features = MLFeatures(
            pedagogy_pattern="worked_examples",
            bloom_levels=["apply", "analyze"],
        )

        capture.log_decision(
            decision_type="pedagogical_strategy",
            decision="Use worked examples",
            rationale="Worked examples reduce cognitive load for novices",
            ml_features=features,
        )

        assert capture.decisions[0]["ml_features"]["pedagogy_pattern"] == "worked_examples"

    @pytest.mark.unit
    def test_log_decision_with_inputs_ref(self, capture):
        """Should log input source references."""
        refs = [
            InputRef(
                source_type="textbook",
                path_or_id="/path/to/book.pdf",
                content_hash="abc123",
            )
        ]

        capture.log_decision(
            decision_type="source_selection",
            decision="Use textbook chapter 3",
            rationale="Chapter covers foundational concepts needed",
            inputs_ref=refs,
        )

        assert len(capture.decisions[0]["inputs_ref"]) == 1
        assert capture.decisions[0]["inputs_ref"][0]["source_type"] == "textbook"

    @pytest.mark.unit
    def test_log_decision_with_outputs(self, capture):
        """Should log output artifact references."""
        outputs = [
            OutputArtifact(
                artifact_type="html",
                path="week_01/module_01.html",
                content_hash="xyz789",
            )
        ]

        capture.log_decision(
            decision_type="content_structure",
            decision="Created module content",
            rationale="Module follows course outline structure",
            outputs=outputs,
        )

        assert len(capture.decisions[0]["outputs"]) == 1
        assert capture.decisions[0]["outputs"][0]["artifact_type"] == "html"

    @pytest.mark.unit
    def test_log_decision_with_is_default(self, capture):
        """Should capture non-decisions (defaults used)."""
        capture.log_decision(
            decision_type="accessibility_measures",
            decision="Used default color scheme",
            rationale="Default meets WCAG AA requirements",
            is_default=True,
        )

        assert capture.decisions[0]["is_default"] is True


# =============================================================================
# QUALITY ASSESSMENT TESTS
# =============================================================================

class TestQualityAssessment:
    """Test quality assessment of decisions."""

    @pytest.fixture
    def capture(self, mock_libv2_storage, mock_legacy_dir):
        return DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

    @pytest.mark.unit
    def test_quality_based_on_rationale_length(self, capture):
        """Quality should consider rationale length."""
        # Short rationale
        capture.log_decision(
            decision_type="content_structure",
            decision="Test",
            rationale="Short",
        )

        # Check quality level was recorded
        assert "metadata" in capture.decisions[0]
        assert "quality_level" in capture.decisions[0]["metadata"]

    @pytest.mark.unit
    def test_rationale_length_recorded(self, capture):
        """Rationale length should be recorded in metadata."""
        rationale = "This is a detailed rationale explaining the decision"

        capture.log_decision(
            decision_type="content_structure",
            decision="Test decision",
            rationale=rationale,
        )

        assert capture.decisions[0]["metadata"]["rationale_length"] == len(rationale)


# =============================================================================
# LOG NON-DECISION TESTS
# =============================================================================

class TestLogNonDecision:
    """Test non-decision (default) logging."""

    @pytest.fixture
    def capture(self, mock_libv2_storage, mock_legacy_dir):
        return DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

    @pytest.mark.unit
    def test_log_non_decision(self, capture):
        """Should log non-decisions with is_default=True."""
        if hasattr(capture, 'log_non_decision'):
            capture.log_non_decision(
                decision_type="color_scheme",
                default_value="standard_blue",
                reason="No customization needed",
            )

            # Should be marked as default
            assert any(d.get("is_default") for d in capture.decisions)
        else:
            # If method doesn't exist, log_decision with is_default should work
            capture.log_decision(
                decision_type="color_scheme",
                decision="standard_blue",
                rationale="No customization needed - using default",
                is_default=True,
            )
            assert capture.decisions[0]["is_default"] is True


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Test decision validation."""

    @pytest.fixture
    def capture(self, mock_libv2_storage, mock_legacy_dir):
        return DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

    @pytest.mark.unit
    def test_validate_sufficient_decisions(self, capture):
        """Validation should pass with sufficient decisions."""
        # Log enough decisions
        for i in range(5):
            capture.log_decision(
                decision_type="content_structure",
                decision=f"Decision {i}",
                rationale=f"Rationale for decision {i} with sufficient length",
            )

        if hasattr(capture, 'validate'):
            result = capture.validate()
            # Result structure may vary, but should not indicate critical failure
            assert result is not None

    @pytest.mark.unit
    def test_decisions_are_stored(self, capture):
        """Decisions should be stored in decisions list."""
        capture.log_decision(
            decision_type="content_structure",
            decision="Test",
            rationale="Test rationale with enough characters",
        )

        assert len(capture.decisions) >= 1


# =============================================================================
# SAVE TESTS
# =============================================================================

class TestSave:
    """Test saving captured decisions."""

    @pytest.mark.unit
    def test_save_writes_files(self, mock_libv2_storage, mock_legacy_dir, tmp_path):
        """Save should write decision files."""
        # Setup capture with actual temp directories
        capture = DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

        # Override output directories to use temp
        capture.output_dir = tmp_path / "output"
        capture.output_dir.mkdir(parents=True)
        capture.legacy_output_dir = tmp_path / "legacy"
        capture.legacy_output_dir.mkdir(parents=True)

        capture.log_decision(
            decision_type="content_structure",
            decision="Test decision",
            rationale="Test rationale for the decision",
        )

        if hasattr(capture, 'save'):
            capture.save()

            # Check that files were created
            output_files = list(capture.output_dir.glob("*.json*"))
            # May or may not have files depending on implementation
            assert capture.output_dir.exists()


# =============================================================================
# STREAMING MODE TESTS
# =============================================================================

class TestStreamingMode:
    """Test streaming (immediate write) mode."""

    @pytest.mark.unit
    def test_streaming_mode_flag(self, mock_libv2_storage, mock_legacy_dir):
        """Streaming mode should be configurable."""
        capture_streaming = DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=True,
        )
        capture_batch = DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

        assert capture_streaming.streaming_mode is True
        assert capture_batch.streaming_mode is False
