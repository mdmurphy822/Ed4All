"""
Controlled Vocabulary Validation

Enforces consistent labeling for ML training data.
Provides built-in vocabularies for Bloom's taxonomy, UDL, and cognitive load.

Phase 0 Hardening - Requirement 8: Training Capture Quality Controls
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class BloomLevel(Enum):
    """Bloom's Taxonomy levels."""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class UDLPrinciple(Enum):
    """Universal Design for Learning principles."""
    ENGAGEMENT = "engagement"
    REPRESENTATION = "representation"
    ACTION_EXPRESSION = "action_expression"


class CognitiveLoadStrategy(Enum):
    """Cognitive load management strategies."""
    CHUNKING = "chunking"
    SCAFFOLDING = "scaffolding"
    WORKED_EXAMPLES = "worked_examples"
    MULTIMEDIA = "multimedia"
    SEGMENTING = "segmenting"
    PRETRAINING = "pretraining"
    SIGNALING = "signaling"
    REDUNDANCY_REDUCTION = "redundancy_reduction"


class AssessmentType(Enum):
    """Assessment types."""
    FORMATIVE = "formative"
    SUMMATIVE = "summative"
    DIAGNOSTIC = "diagnostic"
    PERFORMANCE = "performance"
    SELF_ASSESSMENT = "self_assessment"
    PEER_ASSESSMENT = "peer_assessment"


class QuestionFormat(Enum):
    """Question format types."""
    MULTIPLE_CHOICE = "multiple_choice"
    MULTIPLE_SELECT = "multiple_select"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    MATCHING = "matching"
    ORDERING = "ordering"
    FILL_BLANK = "fill_blank"
    NUMERIC = "numeric"


@dataclass
class VocabularyViolation:
    """Single vocabulary violation."""
    field: str
    value: str
    expected: List[str]
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "value": self.value,
            "expected": self.expected,
            "suggestion": self.suggestion
        }


class ControlledVocabulary:
    """Validates values against controlled vocabularies."""

    def __init__(self, vocab_dir: Optional[Path] = None):
        """
        Initialize controlled vocabulary validator.

        Args:
            vocab_dir: Optional directory with custom vocabulary JSON files
        """
        self.vocab_dir = vocab_dir
        self._vocabularies: Dict[str, Set[str]] = {}

        # Built-in vocabularies from enums
        self._vocabularies['bloom_levels'] = {e.value for e in BloomLevel}
        self._vocabularies['udl_principles'] = {e.value for e in UDLPrinciple}
        self._vocabularies['cognitive_load'] = {e.value for e in CognitiveLoadStrategy}
        self._vocabularies['assessment_types'] = {e.value for e in AssessmentType}
        self._vocabularies['question_formats'] = {e.value for e in QuestionFormat}

        # Decision types from schema
        self._vocabularies['decision_types'] = {
            "approach_selection", "strategy_decision", "source_selection",
            "source_interpretation", "textbook_integration", "existing_content_usage",
            "content_structure", "content_depth", "content_adaptation",
            "example_selection", "pedagogical_strategy", "assessment_design",
            "bloom_level_assignment", "learning_objective_mapping",
            "accessibility_measures", "format_decision", "component_selection",
            "quality_judgment", "validation_result", "error_handling",
            "prompt_response", "file_creation", "outcome_signal",
            "chunk_selection", "question_generation", "distractor_generation",
            "revision_decision"
        }

        # Quality levels
        self._vocabularies['quality_levels'] = {
            "exemplary", "proficient", "developing", "inadequate"
        }

        # Edit distance categories
        self._vocabularies['edit_distance'] = {
            "none", "low", "medium", "high"
        }

        # Workflow phases
        self._vocabularies['workflow_phases'] = {
            "input-research", "exam-research", "course-outliner",
            "content-generator", "brightspace-packager", "dart-conversion",
            "trainforge-assessment", "validation"
        }

        # Load custom vocabularies
        if vocab_dir and vocab_dir.exists():
            self._load_custom_vocabularies(vocab_dir)

    def _load_custom_vocabularies(self, vocab_dir: Path) -> None:
        """Load custom vocabulary files from directory."""
        for vocab_file in vocab_dir.glob("*.json"):
            vocab_name = vocab_file.stem
            try:
                with open(vocab_file) as f:
                    values = json.load(f)
                if isinstance(values, list):
                    self._vocabularies[vocab_name] = set(values)
                    logger.debug(f"Loaded vocabulary: {vocab_name} ({len(values)} values)")
                elif isinstance(values, dict) and 'values' in values:
                    self._vocabularies[vocab_name] = set(values['values'])
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load vocabulary {vocab_file}: {e}")

    def validate(
        self,
        vocab_name: str,
        value: str,
        strict: bool = False
    ) -> Optional[VocabularyViolation]:
        """
        Validate a single value against vocabulary.

        Args:
            vocab_name: Name of vocabulary to check against
            value: Value to validate
            strict: If True, error on unknown vocabulary

        Returns:
            VocabularyViolation if invalid, None if valid
        """
        if vocab_name not in self._vocabularies:
            if strict:
                return VocabularyViolation(
                    field=vocab_name,
                    value=value,
                    expected=["(unknown vocabulary)"]
                )
            return None

        valid_values = self._vocabularies[vocab_name]

        # Case-insensitive comparison
        if value.lower() not in {v.lower() for v in valid_values}:
            suggestion = self._find_closest(value, valid_values)
            return VocabularyViolation(
                field=vocab_name,
                value=value,
                expected=sorted(valid_values),
                suggestion=suggestion
            )

        return None

    def validate_many(
        self,
        vocab_name: str,
        values: List[str],
        strict: bool = False
    ) -> List[VocabularyViolation]:
        """
        Validate multiple values against vocabulary.

        Args:
            vocab_name: Name of vocabulary
            values: List of values to validate
            strict: If True, error on unknown vocabulary

        Returns:
            List of violations (empty if all valid)
        """
        violations = []
        for value in values:
            v = self.validate(vocab_name, value, strict)
            if v:
                violations.append(v)
        return violations

    def validate_ml_features(
        self,
        ml_features: Dict[str, Any]
    ) -> List[VocabularyViolation]:
        """
        Validate ML features against controlled vocabularies.

        Args:
            ml_features: Dictionary of ML feature fields

        Returns:
            List of violations
        """
        violations = []

        # Check bloom_levels
        if 'bloom_levels' in ml_features:
            for level in ml_features['bloom_levels']:
                v = self.validate('bloom_levels', level)
                if v:
                    violations.append(v)

        # Check udl_principles
        if 'udl_principles' in ml_features:
            for principle in ml_features['udl_principles']:
                v = self.validate('udl_principles', principle)
                if v:
                    violations.append(v)

        # Check cognitive_load_strategy
        if 'cognitive_load_strategy' in ml_features:
            for strategy in ml_features['cognitive_load_strategy']:
                v = self.validate('cognitive_load', strategy)
                if v:
                    violations.append(v)

        return violations

    def validate_decision_event(
        self,
        event: Dict[str, Any]
    ) -> List[VocabularyViolation]:
        """
        Validate a decision event against all relevant vocabularies.

        Args:
            event: Decision event dictionary

        Returns:
            List of violations
        """
        violations = []

        # Check decision_type
        if 'decision_type' in event:
            v = self.validate('decision_types', event['decision_type'])
            if v:
                violations.append(v)

        # Check phase
        if 'phase' in event and event['phase']:
            v = self.validate('workflow_phases', event['phase'])
            if v:
                violations.append(v)

        # Check ML features
        if 'ml_features' in event and event['ml_features']:
            violations.extend(self.validate_ml_features(event['ml_features']))

        # Check quality_level in metadata
        if 'metadata' in event and event['metadata']:
            if 'quality_level' in event['metadata']:
                v = self.validate('quality_levels', event['metadata']['quality_level'])
                if v:
                    violations.append(v)

        # Check outcome edit_distance
        if 'outcome' in event and event['outcome']:
            if 'edit_distance' in event['outcome']:
                v = self.validate('edit_distance', event['outcome']['edit_distance'])
                if v:
                    violations.append(v)

        return violations

    def _find_closest(self, value: str, valid_values: Set[str]) -> Optional[str]:
        """Find closest match using Levenshtein distance."""
        value_lower = value.lower()
        best_match = None
        best_distance = float('inf')

        for valid in valid_values:
            distance = self._levenshtein(value_lower, valid.lower())
            if distance < best_distance:
                best_distance = distance
                best_match = valid

        # Only suggest if reasonably close
        if best_distance <= max(len(value) // 2, 3):
            return best_match
        return None

    def _levenshtein(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_vocabulary(self, name: str) -> Optional[Set[str]]:
        """Get valid values for a vocabulary."""
        return self._vocabularies.get(name)

    def list_vocabularies(self) -> List[str]:
        """List all available vocabulary names."""
        return sorted(self._vocabularies.keys())

    def add_vocabulary(self, name: str, values: Set[str]) -> None:
        """Add or update a vocabulary."""
        self._vocabularies[name] = values

    def normalize_value(self, vocab_name: str, value: str) -> str:
        """
        Normalize a value to match vocabulary casing.

        Args:
            vocab_name: Vocabulary name
            value: Value to normalize

        Returns:
            Normalized value (or original if not found)
        """
        if vocab_name not in self._vocabularies:
            return value

        value_lower = value.lower()
        for valid in self._vocabularies[vocab_name]:
            if valid.lower() == value_lower:
                return valid
        return value


# Global validator instance
_global_validator: Optional[ControlledVocabulary] = None


def get_vocabulary_validator() -> ControlledVocabulary:
    """Get global vocabulary validator."""
    global _global_validator
    if _global_validator is None:
        _global_validator = ControlledVocabulary()
    return _global_validator


def validate_bloom_level(level: str) -> Optional[VocabularyViolation]:
    """Convenience function to validate Bloom's level."""
    return get_vocabulary_validator().validate('bloom_levels', level)


def validate_decision_type(decision_type: str) -> Optional[VocabularyViolation]:
    """Convenience function to validate decision type."""
    return get_vocabulary_validator().validate('decision_types', decision_type)


def is_valid_bloom_level(level: str) -> bool:
    """Check if Bloom's level is valid."""
    return validate_bloom_level(level) is None


def get_bloom_levels() -> List[str]:
    """Get list of valid Bloom's levels."""
    return sorted(get_vocabulary_validator().get_vocabulary('bloom_levels') or [])
