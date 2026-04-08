"""
IMSCC Generators Package

Provides XML generation utilities for IMSCC package components:
- Assignment XML generation
- Discussion topic XML generation
- QTI assessment XML generation
- Manifest XML generation
"""

from .base_generator import BaseGenerator, generate_brightspace_id, escape_xml_attribute
from .assignment_generator import AssignmentGenerator
from .discussion_generator import DiscussionGenerator
from .quiz_generator import QuizGenerator, QuestionType, QuizQuestion, Choice
from .manifest_generator import ManifestGenerator, ResourceEntry
from .constants import (
    NAMESPACES,
    SCHEMA_LOCATIONS,
    RESOURCE_TYPES,
    QTI_QUESTION_PROFILES,
    QTI_ASSESSMENT_PROFILE,
    MAX_POINTS,
    MIN_POINTS,
    MAX_TITLE_LENGTH,
    VALID_SUBMISSION_TYPES,
    DEPRECATED_NAMESPACES,
)

__all__ = [
    'BaseGenerator',
    'generate_brightspace_id',
    'escape_xml_attribute',
    'AssignmentGenerator',
    'DiscussionGenerator',
    'QuizGenerator',
    'QuestionType',
    'QuizQuestion',
    'Choice',
    'ManifestGenerator',
    'ResourceEntry',
    'NAMESPACES',
    'SCHEMA_LOCATIONS',
    'RESOURCE_TYPES',
    'QTI_QUESTION_PROFILES',
    'QTI_ASSESSMENT_PROFILE',
    'MAX_POINTS',
    'MIN_POINTS',
    'MAX_TITLE_LENGTH',
    'VALID_SUBMISSION_TYPES',
    'DEPRECATED_NAMESPACES',
]
