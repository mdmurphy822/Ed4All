"""
Trainforge Assessment Generators

Assessment generation components for Trainforge RAG training.

Classes:
    AssessmentGenerator: Main orchestrator for assessment generation
    QuestionFactory: Factory for creating different question types

Functions:
    generate_assessment: Convenience function for assessment generation
    create_mcq: Create multiple choice question
    create_tf: Create true/false question
    create_fib: Create fill-in-blank question
    create_essay: Create essay question
"""

from .assessment_generator import (
    AssessmentGenerator,
    AssessmentData,
    QuestionData,
    BLOOM_LEVELS,
    generate_assessment,
)

from .question_factory import (
    QuestionFactory,
    Question,
    QuestionChoice,
    create_mcq,
    create_tf,
    create_fib,
    create_essay,
)

__all__ = [
    # Classes
    "AssessmentGenerator",
    "AssessmentData",
    "QuestionData",
    "QuestionFactory",
    "Question",
    "QuestionChoice",
    # Functions
    "generate_assessment",
    "create_mcq",
    "create_tf",
    "create_fib",
    "create_essay",
    # Constants
    "BLOOM_LEVELS",
]
