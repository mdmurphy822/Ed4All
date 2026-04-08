"""
Shared constants for Ed4All decision capture system.

This module centralizes all configuration constants to prevent circular imports
and provide a single source of truth for project configuration.

Note: Project paths are now defined in lib/paths.py - this module re-exports them
for backward compatibility.
"""

import os
from pathlib import Path
from typing import Dict, Set

# Import paths from centralized module
from .paths import (
    PROJECT_ROOT,
    SCHEMAS_PATH,
    LIBV2_PATH,
    LIBV2_CATALOG,
    LIBV2_COURSES,
    LIBV2_ONTOLOGY,
    LIBV2_SCHEMA,
    TRAINING_DIR,
    TRAINING_DIR_LEGACY,
)

# Re-export for backward compatibility
PROJECT_DIR = PROJECT_ROOT
SCHEMAS_DIR = SCHEMAS_PATH
LIBV2_ROOT = LIBV2_PATH

# Minimum decisions required per phase for quality assurance
MIN_DECISIONS_PER_PHASE: Dict[str, int] = {
    # DART phases
    "dart-conversion": 3,
    "dart-validation": 2,
    # Courseforge phases
    "courseforge-input-research": 3,
    "courseforge-exam-research": 2,
    "courseforge-course-outliner": 5,
    "courseforge-content-generator": 10,
    "courseforge-brightspace-packager": 2,
    # Trainforge phases
    "trainforge-content-analysis": 3,
    "trainforge-question-generation": 20,
    "trainforge-assessment-assembly": 5,
    "trainforge-validation": 3,
    # LibV2 phases
    "libv2-retrieval": 2,
    "libv2-indexing": 3,
    "libv2-fusion": 2,
    # Legacy phase names (for backward compatibility)
    "input-research": 3,
    "exam-research": 2,
    "course-outliner": 5,
    "content-generator": 10,
    "brightspace-packager": 2,
    "content-analysis": 3,
    "question-generation": 20,
    "assessment-assembly": 5,
    "validation": 3,
}

# Operation mapping for ML labeling
OPERATION_MAP: Dict[str, str] = {
    "content_structure": "plan_structure",
    "source_selection": "select_sources",
    "file_creation": "generate_module_html",
    "accessibility_measures": "apply_accessibility",
    "pedagogical_strategy": "design_pedagogy",
    "prompt_response": "generate_content",
    "research_approach": "conduct_research",
    "textbook_integration": "integrate_textbook",
    "existing_content_usage": "reuse_content",
    "content_depth": "set_depth",
    "assessment_design": "design_assessment",
    "question_generation": "generate_question",
    "distractor_generation": "generate_distractor",
    "chunk_selection": "select_chunks",
    "bloom_level_assignment": "assign_bloom",
    "validation_result": "validate_output",
    "revision_decision": "decide_revision",
    "approach_selection": "select_approach",
    "content_adaptation": "adapt_content",
    "error_handling": "handle_error",
    "source_usage": "use_source",
    "outcome_signal": "record_outcome",
    "alignment_check": "check_alignment",
    # LibV2 RAG operations
    "query_decomposition": "decompose_query",
    "retrieval_ranking": "rank_results",
    "result_fusion": "fuse_results",
    "chunk_deduplication": "deduplicate_chunks",
    "index_strategy": "build_index",
}

# Valid decision types for schema validation
VALID_DECISION_TYPES: Set[str] = {
    "approach_selection",
    "strategy_decision",
    "content_structure",
    "source_selection",
    "source_interpretation",
    "content_adaptation",
    "validation_result",
    "error_handling",
    "file_creation",
    "prompt_response",
    "source_usage",
    "outcome_signal",
    "chunk_selection",
    "question_generation",
    "distractor_generation",
    "alignment_check",
    "structure_detection",
    "heading_assignment",
    "alt_text_generation",
    "accessibility_measures",
    "format_decision",
    "textbook_integration",
    "existing_content_usage",
    "content_depth",
    "pedagogical_strategy",
    "assessment_design",
    "bloom_level_assignment",
    "learning_objective_mapping",
    "component_selection",
    "quality_judgment",
    "revision_decision",
    "math_conversion",
    "example_selection",
    "research_approach",
    # LibV2 RAG decision types
    "query_decomposition",
    "retrieval_ranking",
    "result_fusion",
    "chunk_deduplication",
    "index_strategy",
}

# Quality assessment thresholds
QUALITY_THRESHOLDS = {
    "exemplary": {
        "rationale_min_length": 100,
        "requires_inputs_ref": True,
        "requires_alternatives": True,
    },
    "proficient": {
        "rationale_min_length": 50,
        "requires_inputs_ref": False,  # OR alternatives
        "requires_alternatives": False,
    },
    "developing": {
        "rationale_min_length": 20,
        "requires_inputs_ref": False,
        "requires_alternatives": False,
    },
    "inadequate": {
        "rationale_min_length": 0,
        "requires_inputs_ref": False,
        "requires_alternatives": False,
    },
}

# Environment variable to enable/disable validation
VALIDATE_DECISIONS = os.environ.get("VALIDATE_DECISIONS", "true").lower() == "true"


def validate_project_paths() -> Dict[str, bool]:
    """Validate that required project paths exist.

    Returns:
        Dict mapping path names to existence status
    """
    return {
        "PROJECT_DIR": PROJECT_DIR.exists(),
        "SCHEMAS_DIR": SCHEMAS_DIR.exists(),
        "LIBV2_ROOT": LIBV2_ROOT.exists(),
        "LIBV2_CATALOG": LIBV2_CATALOG.exists(),
        "LIBV2_COURSES": LIBV2_COURSES.exists(),
        # Legacy paths (deprecated)
        "TRAINING_DIR_LEGACY": TRAINING_DIR.exists(),
    }


def ensure_training_dir(tool: str = "") -> Path:
    """Ensure training directory exists and return path.

    Args:
        tool: Optional tool name for subdirectory (dart, courseforge, trainforge)

    Returns:
        Path to training directory (created if needed)
    """
    target = TRAINING_DIR / tool if tool else TRAINING_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target
