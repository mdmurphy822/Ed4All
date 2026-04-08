#!/usr/bin/env python3
"""
Trainforge Decision Logger

Centralized decision logging for Trainforge assessment generation.
Captures all decisions during RAG retrieval, question generation,
distractor creation, and validation for Claude training data.
"""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Add Ed4All lib to path
ED4ALL_ROOT = Path(__file__).resolve().parents[2]  # decision_capture/decision_logger.py → Trainforge/ → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

from lib.trainforge_capture import (  # noqa: E402
    AlignmentCheck,
    QuestionData,
    RAGMetrics,
    TrainforgeDecisionCapture,
)


class SessionError(Exception):
    """Raised when there's a session management error."""
    pass


class TrainforgeDecisionLogger:
    """
    High-level decision logging interface for Trainforge operations.

    Provides structured methods for logging decisions at each stage
    of the assessment generation pipeline.
    """

    def __init__(
        self,
        course_code: str,
        imscc_path: str,
        auto_save: bool = True
    ):
        """
        Initialize the decision logger.

        Args:
            course_code: Course code (e.g., "INT_101")
            imscc_path: Path to source IMSCC package
            auto_save: Whether to auto-save on context exit
        """
        self.course_code = course_code
        self.imscc_path = imscc_path
        self.auto_save = auto_save
        self._capture: Optional[TrainforgeDecisionCapture] = None
        self._current_phase: Optional[str] = None

    def start_session(self, phase: str = "question-generation", force: bool = False) -> 'TrainforgeDecisionLogger':
        """
        Start a new capture session for a phase.

        Args:
            phase: Phase name (e.g., "content-analysis", "question-generation")
            force: If True, forcefully end existing session first

        Returns:
            Self for method chaining

        Raises:
            SessionError: If a session already exists and force=False
        """
        # Check for existing session
        if self._capture is not None:
            if force:
                logger.warning("Forcing new session - auto-saving previous session")
                self.end_session()
            else:
                raise SessionError(
                    "Session already active. Call end_session() first or use force=True"
                )

        self._current_phase = phase
        self._capture = TrainforgeDecisionCapture(
            self.course_code,
            self.imscc_path
        )
        # Override phase if needed
        self._capture.phase = f"trainforge-{phase}"
        return self

    def end_session(self) -> None:
        """
        End the current session and finalize captures.
        """
        if self._capture:
            try:
                self._capture.finalize()
                return None
            finally:
                self._capture = None
        return None

    def __enter__(self):
        """Context manager entry."""
        if self._capture is not None:
            logger.warning("Entering context with existing session - auto-saving")
            self.end_session()
        self.start_session(force=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with auto-save."""
        if self.auto_save:
            self.end_session()
        return False

    # ========== Learning Objective Context ==========

    def set_objective_context(
        self,
        objective_id: str,
        objective_text: str,
        bloom_target: str,
        module_context: Optional[str] = None
    ):
        """
        Set the current learning objective context.

        All subsequent decisions will be tagged with this context.

        Args:
            objective_id: Unique ID for the learning objective (e.g., "LO-001")
            objective_text: Full text of the learning objective
            bloom_target: Target Bloom's taxonomy level
            module_context: Optional module/week context
        """
        if self._capture:
            self._capture.set_learning_objective_context(objective_id, bloom_target)
            self._capture.log_decision(
                decision_type="learning_objective_mapping",
                decision=f"Set context for objective {objective_id}",
                rationale=f"Targeting {bloom_target} level: {objective_text[:100]}",
                context=module_context
            )

    # ========== RAG Retrieval Decisions ==========

    def log_retrieval(
        self,
        query: str,
        chunks_retrieved: List[Dict[str, Any]],
        chunks_selected: List[Dict[str, Any]],
        latency_ms: float,
        selection_rationale: str
    ):
        """
        Log a RAG retrieval decision.

        Args:
            query: The query used for retrieval
            chunks_retrieved: All chunks returned by retrieval
            chunks_selected: Chunks selected for use
            latency_ms: Retrieval latency in milliseconds
            selection_rationale: Why these chunks were selected
        """
        if not self._capture:
            return

        # Pass full chunk dictionaries (required by log_chunk_retrieval)
        self._capture.log_chunk_retrieval(
            query=query,
            chunks_retrieved=chunks_retrieved,
            chunks_used=chunks_selected,
            retrieval_latency_ms=latency_ms
        )

        self._capture.log_decision(
            decision_type="chunk_selection",
            decision=f"Selected {len(chunks_selected)} of {len(chunks_retrieved)} chunks",
            rationale=selection_rationale,
            context=f"Query: {query[:100]}",
            confidence=0.8 if len(chunks_selected) > 0 else 0.3
        )

    def log_retrieval_rejection(
        self,
        chunk_id: str,
        rejection_reason: str,
        relevance_score: float
    ):
        """
        Log why a specific chunk was rejected.

        Args:
            chunk_id: ID of the rejected chunk
            rejection_reason: Why it was not selected
            relevance_score: Similarity/relevance score
        """
        if self._capture:
            self._capture.log_decision(
                decision_type="chunk_selection",
                decision=f"Rejected chunk {chunk_id}",
                rationale=f"Score {relevance_score:.2f}: {rejection_reason}",
                context="Chunk did not meet selection criteria"
            )

    # ========== Question Generation Decisions ==========

    def log_question_type_selection(
        self,
        selected_type: str,
        alternatives: List[str],
        selection_rationale: str,
        bloom_alignment: str
    ):
        """
        Log the decision to use a specific question type.

        Args:
            selected_type: The question type chosen (e.g., "multiple_choice")
            alternatives: Other types that were considered
            selection_rationale: Why this type was chosen
            bloom_alignment: How it aligns with target Bloom's level
        """
        if not self._capture:
            return

        self._capture.log_decision(
            decision_type="question_generation",
            decision=f"Selected question type: {selected_type}",
            rationale=f"{selection_rationale}. Bloom alignment: {bloom_alignment}",
            alternatives_considered=[
                {"option": alt, "reason_rejected": "Less suitable for objective"}
                for alt in alternatives
            ]
        )

    def log_question_generated(
        self,
        question: QuestionData,
        source_chunks: List[str],
        generation_rationale: str,
        confidence: float = 0.8
    ):
        """
        Log a generated question.

        Args:
            question: The generated question data
            source_chunks: Chunk IDs used to generate the question
            generation_rationale: Why the question was formulated this way
            confidence: Confidence in the question quality (0-1)
        """
        if self._capture:
            self._capture.log_question_generation(
                question=question,
                source_chunks=source_chunks,
                generation_rationale=generation_rationale
            )

    def log_stem_formulation(
        self,
        stem: str,
        alternatives_considered: List[str],
        selection_rationale: str
    ):
        """
        Log the decision for question stem wording.

        Args:
            stem: The final stem text
            alternatives_considered: Other stem formulations considered
            selection_rationale: Why this wording was chosen
        """
        if self._capture:
            self._capture.log_decision(
                decision_type="question_generation",
                decision=f"Stem: {stem[:100]}",
                rationale=selection_rationale,
                alternatives_considered=[
                    {"option": alt[:80], "reason_rejected": "Less clear"}
                    for alt in alternatives_considered[:3]
                ]
            )

    # ========== Distractor Decisions ==========

    def log_distractor(
        self,
        question_id: str,
        distractor_text: str,
        misconception_targeted: str,
        rationale: str,
        plausibility_score: float = 0.7
    ):
        """
        Log a distractor generation decision.

        Args:
            question_id: ID of the parent question
            distractor_text: The distractor text
            misconception_targeted: The misconception this targets
            rationale: Why this distractor was created
            plausibility_score: How plausible this distractor is (0-1)
        """
        if self._capture:
            self._capture.log_distractor_rationale(
                question_id=question_id,
                distractor_text=distractor_text,
                misconception_targeted=misconception_targeted,
                rationale=rationale
            )

    def log_distractor_rejection(
        self,
        question_id: str,
        rejected_distractor: str,
        rejection_reason: str
    ):
        """
        Log why a potential distractor was rejected.

        Args:
            question_id: ID of the parent question
            rejected_distractor: The rejected distractor text
            rejection_reason: Why it was not used
        """
        if self._capture:
            self._capture.log_decision(
                decision_type="distractor_generation",
                decision=f"Rejected distractor for {question_id}",
                rationale=rejection_reason,
                context=rejected_distractor[:100]
            )

    # ========== Alignment & Validation Decisions ==========

    def log_alignment_check(
        self,
        question_id: str,
        alignment_result: AlignmentCheck,
        pass_fail: bool,
        issues: Optional[List[str]] = None
    ):
        """
        Log an alignment check result.

        Args:
            question_id: ID of the question being validated
            alignment_result: The alignment check results
            pass_fail: Whether alignment check passed
            issues: List of alignment issues found
        """
        if self._capture:
            # Provide all required arguments for log_alignment_check
            self._capture.log_alignment_check(
                assessment_id=question_id,
                lo_coverage={question_id: alignment_result.lo_coverage_score},
                bloom_distribution={},  # Caller should provide if available
                alignment=alignment_result
            )
            self._capture.log_decision(
                decision_type="validation_result",
                decision=f"Alignment check {'passed' if pass_fail else 'failed'} for {question_id}",
                rationale=f"LO coverage: {alignment_result.lo_coverage_score:.0%}, "
                         f"Bloom alignment: {alignment_result.bloom_alignment_score:.0%}",
                context="; ".join(issues) if issues else "No issues",
                confidence=1.0 if pass_fail else 0.5
            )

    def log_quality_validation(
        self,
        question_id: str,
        quality_score: float,
        criteria_results: Dict[str, bool],
        feedback: Optional[str] = None
    ):
        """
        Log a quality validation decision.

        Args:
            question_id: ID of the question
            quality_score: Overall quality score (0-1)
            criteria_results: Pass/fail for each criterion
            feedback: Optional feedback for improvement
        """
        if not self._capture:
            return

        passed_criteria = sum(1 for v in criteria_results.values() if v)
        total_criteria = len(criteria_results)

        self._capture.log_decision(
            decision_type="quality_judgment",
            decision=f"Quality validation for {question_id}: {quality_score:.0%}",
            rationale=f"Passed {passed_criteria}/{total_criteria} criteria. {feedback or ''}",
            confidence=quality_score
        )

    # ========== Revision Decisions ==========

    def log_revision_decision(
        self,
        question_id: str,
        revision_needed: bool,
        issues_to_fix: List[str],
        revision_strategy: str,
        revision_number: int = 1
    ):
        """
        Log a revision decision.

        Args:
            question_id: ID of the question
            revision_needed: Whether revision is required
            issues_to_fix: List of issues that need fixing
            revision_strategy: How the revision will be approached
            revision_number: Which revision this is (1, 2, 3)
        """
        if self._capture and revision_needed:
            # Map to underlying log_revision_decision signature
            self._capture.log_revision_decision(
                question_id=question_id,
                revision_number=revision_number,
                reason=f"Revision needed: {revision_strategy}",
                changes_made=issues_to_fix,
                validator_feedback="; ".join(issues_to_fix) if issues_to_fix else "No specific issues"
            )

    def log_revision_applied(
        self,
        question_id: str,
        original_version: str,
        revised_version: str,
        changes_made: str
    ):
        """
        Log a revision that was applied.

        Args:
            question_id: ID of the question
            original_version: The original question/distractor text
            revised_version: The revised text
            changes_made: Description of what was changed
        """
        if self._capture:
            self._capture.log_decision(
                decision_type="revision_decision",
                decision=f"Applied revision to {question_id}",
                rationale=changes_made,
                context=f"Original: {original_version[:100]}... -> Revised: {revised_version[:100]}..."
            )

    # ========== Utility Methods ==========

    def log_custom(
        self,
        decision_type: str,
        decision: str,
        rationale: str,
        **kwargs
    ):
        """
        Log a custom decision not covered by specialized methods.

        Args:
            decision_type: Type of decision
            decision: The decision made
            rationale: Why this decision was made
            **kwargs: Additional fields
        """
        if self._capture:
            self._capture.log_decision(
                decision_type=decision_type,
                decision=decision,
                rationale=rationale,
                **kwargs
            )

    def get_decision_count(self) -> int:
        """Get the number of decisions logged in this session."""
        return self._capture._decision_count if self._capture else 0

    def validate_session(self) -> Dict[str, Any]:
        """Validate the current session has sufficient decisions."""
        if self._capture:
            count = self._capture._decision_count
            valid = count > 0
            return {"valid": valid, "decision_count": count, "issues": [] if valid else ["No decisions logged"]}
        return {"valid": False, "decision_count": 0, "issues": ["No active session"]}


@contextmanager
def trainforge_capture_session(
    course_code: str,
    imscc_path: str,
    phase: str = "question-generation"
):
    """
    Context manager for Trainforge decision capture sessions.

    Example:
        with trainforge_capture_session("INT_101", "/path/to/course.imscc") as logger:
            logger.set_objective_context("LO-001", "Understand...", "understand")
            logger.log_question_type_selection("multiple_choice", ["true_false"], "...")
            # ... more logging
    """
    logger = TrainforgeDecisionLogger(course_code, imscc_path)
    logger.start_session(phase)
    try:
        yield logger
    finally:
        logger.end_session()


# Convenience exports
__all__ = [
    'TrainforgeDecisionLogger',
    'trainforge_capture_session',
    'SessionError',
    'QuestionData',
    'RAGMetrics',
    'AlignmentCheck',
]
