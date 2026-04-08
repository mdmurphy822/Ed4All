#!/usr/bin/env python3
"""
Trainforge Decision Capture

Specialized capture for assessment-based RAG training on IMSCC packages.
Extends StreamingDecisionCapture with assessment-specific fields.

Captures:
- RAG retrieval decisions
- Question generation decisions
- Distractor rationale
- Alignment to learning objectives
- Validation feedback loops
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .streaming_capture import StreamingDecisionCapture


@dataclass
class QuestionData:
    """Data structure for assessment questions."""
    question_id: str
    question_type: str  # multiple_choice, true_false, short_answer, essay, matching, fill_in_blank
    question_stem: str
    correct_answer: str
    distractors: List[Dict[str, str]] = field(default_factory=list)  # [{text, misconception_targeted}]
    explanation: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    bloom_level: str = "understand"
    points: int = 1
    time_estimate_seconds: int = 60


@dataclass
class RAGMetrics:
    """Metrics for RAG retrieval operations."""
    chunks_retrieved: int = 0
    chunks_used: int = 0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    context_token_count: int = 0
    embedding_model: str = ""
    similarity_threshold: float = 0.0


@dataclass
class AlignmentCheck:
    """Results of learning objective alignment validation."""
    lo_coverage_score: float = 0.0
    bloom_alignment_score: float = 0.0
    content_alignment_score: float = 0.0
    passed: bool = False
    issues: List[str] = field(default_factory=list)


class TrainforgeDecisionCapture(StreamingDecisionCapture):
    """
    Specialized capture for Trainforge assessment generation.

    Captures:
    - RAG retrieval decisions
    - Question generation decisions
    - Distractor rationale
    - Alignment to learning objectives
    """

    def __init__(
        self,
        course_code: str,
        imscc_source: str,
        phase: str = "question-generation",
        session_id: Optional[str] = None
    ):
        """
        Initialize Trainforge capture.

        Args:
            course_code: Course code being assessed
            imscc_source: Path to source IMSCC package
            phase: Trainforge phase (content-analysis, question-generation, assessment-assembly, validation)
            session_id: Optional session ID
        """
        super().__init__(course_code, phase, "trainforge", session_id)
        self.imscc_source = imscc_source
        self.questions_generated = 0
        self.current_lo: Optional[str] = None
        self.current_bloom_target: Optional[str] = None

    def set_learning_objective_context(
        self,
        lo_id: str,
        bloom_target: str,
        domain: Optional[str] = None,
        domain_weight: Optional[int] = None
    ):
        """
        Set current learning objective being assessed.

        Args:
            lo_id: Learning objective ID (e.g., INT101_D1_1.1)
            bloom_target: Target Bloom's taxonomy level
            domain: Domain name from learning objective
            domain_weight: Weight of domain in assessment
        """
        self.current_lo = lo_id
        self.current_bloom_target = bloom_target

        self.log_decision(
            decision_type="learning_objective_mapping",
            decision=f"Targeting learning objective: {lo_id}",
            rationale=f"Generating {bloom_target}-level assessment items for this objective",
            context=f"Domain: {domain}, Weight: {domain_weight}%" if domain else None,
            confidence=0.9,
            lo_id=lo_id,
            bloom_target=bloom_target,
            domain=domain,
            domain_weight=domain_weight
        )

    def log_chunk_retrieval(
        self,
        query: str,
        chunks_retrieved: List[Dict[str, Any]],
        chunks_used: List[Dict[str, Any]],
        retrieval_latency_ms: float
    ):
        """
        Log RAG retrieval decision.

        Args:
            query: The retrieval query used
            chunks_retrieved: All chunks retrieved from index
            chunks_used: Chunks selected for generation context
            retrieval_latency_ms: Time taken for retrieval
        """
        rag_metrics = RAGMetrics(
            chunks_retrieved=len(chunks_retrieved),
            chunks_used=len(chunks_used),
            retrieval_latency_ms=retrieval_latency_ms,
            context_token_count=sum(c.get('token_count', 0) for c in chunks_used)
        )

        self.log_decision(
            decision_type="chunk_selection",
            decision=f"Retrieved {len(chunks_retrieved)} chunks, using {len(chunks_used)}",
            rationale=f"Query: {query[:100]}... Selected chunks with highest relevance to learning objective",
            context=f"Top chunk relevance: {chunks_used[0].get('relevance_score', 'N/A') if chunks_used else 'N/A'}",
            confidence=0.85,
            rag_metrics=asdict(rag_metrics),
            query=query,
            chunks_used_ids=[c.get('chunk_id') for c in chunks_used],
            lo_id=self.current_lo
        )

    def log_question_generation(
        self,
        question: QuestionData,
        source_chunks: List[str],
        generation_rationale: str,
        generation_latency_ms: float = 0.0
    ):
        """
        Log question generation decision.

        Args:
            question: The generated question data
            source_chunks: IDs of chunks used for generation
            generation_rationale: Reasoning for question design
            generation_latency_ms: Time taken for generation
        """
        self.questions_generated += 1

        self.log_decision(
            decision_type="question_generation",
            decision=f"Generated {question.question_type} question at {question.difficulty} difficulty",
            rationale=generation_rationale,
            context=question.question_stem[:200],
            confidence=0.8,
            question_data={
                "question_id": question.question_id,
                "question_type": question.question_type,
                "bloom_level": question.bloom_level,
                "difficulty": question.difficulty,
                "points": question.points,
                "time_estimate_seconds": question.time_estimate_seconds
            },
            lo_id=self.current_lo,
            bloom_target=self.current_bloom_target,
            source_chunks=source_chunks,
            generation_latency_ms=generation_latency_ms
        )

    def log_distractor_rationale(
        self,
        question_id: str,
        distractor_text: str,
        misconception_targeted: str,
        rationale: str,
        plausibility_score: float = 0.0
    ):
        """
        Log distractor creation decision.

        Args:
            question_id: Parent question ID
            distractor_text: The distractor option text
            misconception_targeted: The misconception this targets
            rationale: Why this distractor was chosen
            plausibility_score: How plausible this distractor is (0-1)
        """
        self.log_decision(
            decision_type="distractor_generation",
            decision=f"Created distractor targeting: {misconception_targeted}",
            rationale=rationale,
            context=distractor_text[:200],
            confidence=plausibility_score,
            question_id=question_id,
            misconception_targeted=misconception_targeted,
            plausibility_score=plausibility_score
        )

    def log_alignment_check(
        self,
        assessment_id: str,
        lo_coverage: Dict[str, float],
        bloom_distribution: Dict[str, int],
        alignment: AlignmentCheck
    ):
        """
        Log assessment alignment validation.

        Args:
            assessment_id: The assessment being validated
            lo_coverage: Coverage scores per learning objective
            bloom_distribution: Distribution of questions by Bloom level
            alignment: Alignment check results
        """
        self.log_decision(
            decision_type="validation_result",
            decision=f"Alignment check: {'PASSED' if alignment.passed else 'FAILED'}",
            rationale=f"LO coverage: {sum(lo_coverage.values())/len(lo_coverage)*100:.1f}% avg, "
                     f"Bloom alignment: {alignment.bloom_alignment_score*100:.1f}%",
            context=f"Issues: {', '.join(alignment.issues) if alignment.issues else 'None'}",
            confidence=alignment.lo_coverage_score,
            assessment_id=assessment_id,
            lo_coverage=lo_coverage,
            bloom_distribution=bloom_distribution,
            alignment_scores={
                "lo_coverage_score": alignment.lo_coverage_score,
                "bloom_alignment_score": alignment.bloom_alignment_score,
                "content_alignment_score": alignment.content_alignment_score
            },
            passed=alignment.passed,
            issues=alignment.issues
        )

    def log_revision_decision(
        self,
        question_id: str,
        revision_number: int,
        reason: str,
        changes_made: List[str],
        validator_feedback: str
    ):
        """
        Log a question revision decision.

        Args:
            question_id: The question being revised
            revision_number: Which revision this is (1, 2, 3)
            reason: Why revision was needed
            changes_made: List of changes applied
            validator_feedback: Feedback from validator that triggered revision
        """
        self.log_decision(
            decision_type="revision_decision",
            decision=f"Revision {revision_number} for question {question_id}",
            rationale=reason,
            context=f"Validator feedback: {validator_feedback[:200]}",
            confidence=0.7 + (0.1 * revision_number),  # Confidence increases with revisions
            question_id=question_id,
            revision_number=revision_number,
            changes_made=changes_made,
            validator_feedback=validator_feedback
        )

    def log_assessment_assembly(
        self,
        assessment_id: str,
        question_ids: List[str],
        total_points: int,
        time_limit_minutes: int,
        assembly_rationale: str
    ):
        """
        Log assessment assembly decision.

        Args:
            assessment_id: The assessment being assembled
            question_ids: Questions included in this assessment
            total_points: Total point value
            time_limit_minutes: Time limit for assessment
            assembly_rationale: Reasoning for question selection and ordering
        """
        self.log_decision(
            decision_type="assessment_design",
            decision=f"Assembled assessment {assessment_id} with {len(question_ids)} questions",
            rationale=assembly_rationale,
            context=f"Total points: {total_points}, Time limit: {time_limit_minutes} minutes",
            confidence=0.85,
            assessment_id=assessment_id,
            question_count=len(question_ids),
            question_ids=question_ids,
            total_points=total_points,
            time_limit_minutes=time_limit_minutes
        )

    def log_quality_score(
        self,
        artifact_id: str,
        dimension_scores: Dict[str, float],
        overall_score: float,
        scoring_rationale: str
    ):
        """
        Log quality scoring decision for DPO/RLHF training.

        Args:
            artifact_id: ID of artifact being scored
            dimension_scores: Scores by dimension (clarity, alignment, difficulty, etc.)
            overall_score: Aggregate quality score (0-1)
            scoring_rationale: Reasoning for the scores
        """
        self.log_decision(
            decision_type="quality_judgment",
            decision=f"Quality score for {artifact_id}: {overall_score:.2f}",
            rationale=scoring_rationale,
            context=f"Dimension scores: {json.dumps(dimension_scores)}",
            confidence=overall_score,
            artifact_id=artifact_id,
            dimension_scores=dimension_scores,
            overall_score=overall_score
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this capture session."""
        return {
            "course_code": self.course_code,
            "imscc_source": self.imscc_source,
            "phase": self.phase,
            "session_id": self.session_id,
            "questions_generated": self.questions_generated,
            "decision_count": self._decision_count,
            "stream_path": str(self.stream_path),
            "meta_path": str(self.meta_path)
        }


def create_trainforge_capture(
    course_code: str,
    imscc_source: str,
    phase: str = "question-generation",
    session_id: Optional[str] = None
) -> TrainforgeDecisionCapture:
    """
    Factory function to create a Trainforge decision capture instance.

    Args:
        course_code: Course code being assessed
        imscc_source: Path to source IMSCC package
        phase: Trainforge phase
        session_id: Optional session ID

    Returns:
        TrainforgeDecisionCapture instance
    """
    return TrainforgeDecisionCapture(course_code, imscc_source, phase, session_id)


# Example usage
if __name__ == "__main__":
    print("Testing TrainforgeDecisionCapture...")

    with create_trainforge_capture(
        "INT_101",
        "/path/to/INT_101.imscc"
    ) as capture:
        # Set learning objective context
        capture.set_learning_objective_context(
            lo_id="INT101_D1_1.1",
            bloom_target="understand",
            domain="Integrator Philosophy",
            domain_weight=25
        )

        # Log chunk retrieval
        capture.log_chunk_retrieval(
            query="Define the purpose of an integrator",
            chunks_retrieved=[
                {"chunk_id": "c1", "relevance_score": 0.92, "token_count": 450},
                {"chunk_id": "c2", "relevance_score": 0.85, "token_count": 380}
            ],
            chunks_used=[
                {"chunk_id": "c1", "relevance_score": 0.92, "token_count": 450}
            ],
            retrieval_latency_ms=45.2
        )

        # Log question generation
        question = QuestionData(
            question_id="Q001",
            question_type="multiple_choice",
            question_stem="What is the primary role of an integrator in system design?",
            correct_answer="To synthesize diverse components into a coherent whole",
            distractors=[
                {"text": "To write code", "misconception_targeted": "technical_only"},
                {"text": "To manage projects", "misconception_targeted": "management_only"}
            ],
            explanation="Integrators synthesize components...",
            difficulty="medium",
            bloom_level="understand",
            points=2
        )

        capture.log_question_generation(
            question=question,
            source_chunks=["c1"],
            generation_rationale="Question targets the core definition of integrator role at understand level",
            generation_latency_ms=1250.5
        )

        # Log distractor rationale
        capture.log_distractor_rationale(
            question_id="Q001",
            distractor_text="To write code",
            misconception_targeted="technical_only",
            rationale="Students often conflate integration with pure coding tasks",
            plausibility_score=0.75
        )

        # Log alignment check
        alignment = AlignmentCheck(
            lo_coverage_score=0.92,
            bloom_alignment_score=0.88,
            content_alignment_score=0.90,
            passed=True,
            issues=[]
        )

        capture.log_alignment_check(
            assessment_id="A001",
            lo_coverage={"INT101_D1_1.1": 0.92, "INT101_D1_1.2": 0.85},
            bloom_distribution={"remember": 2, "understand": 5, "apply": 3},
            alignment=alignment
        )

    print(f"\nSession summary: {capture.get_session_summary()}")
    print("Trainforge capture test complete!")
