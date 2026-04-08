"""
IMSCC Generators Package

Provides XML generation utilities for IMSCC package components:
- Assignment XML generation
- Discussion topic XML generation
- QTI assessment XML generation
- Manifest XML generation
"""

from .assignment_generator import AssignmentGenerator
from .base_generator import BaseGenerator, escape_xml_attribute, generate_brightspace_id
from .constants import (
    DEPRECATED_NAMESPACES,
    MAX_POINTS,
    MAX_TITLE_LENGTH,
    MIN_POINTS,
    NAMESPACES,
    QTI_ASSESSMENT_PROFILE,
    QTI_QUESTION_PROFILES,
    RESOURCE_TYPES,
    SCHEMA_LOCATIONS,
    VALID_SUBMISSION_TYPES,
)
from .discussion_generator import DiscussionGenerator
from .manifest_generator import ManifestGenerator, ResourceEntry
from .quiz_generator import Choice, QuestionType, QuizGenerator, QuizQuestion

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
