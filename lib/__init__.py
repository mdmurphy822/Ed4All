"""
Ed4All Decision Capture Library

Provides decision capture utilities for DART, Courseforge, and Trainforge operations.
All Claude decisions are logged for training data collection.
"""

from .decision_capture import (
    DecisionCapture,
    DARTDecisionCapture,
    MLFeatures,
    InputRef,
    OutcomeSignals,
    create_capture,
    create_dart_capture
)

from .streaming_capture import (
    StreamingDecisionCapture,
    CaptureValidator,
    create_streaming_capture,
    validate_phase_capture,
    get_capture_stats
)

from .trainforge_capture import (
    TrainforgeDecisionCapture,
    QuestionData,
    RAGMetrics,
    AlignmentCheck,
    create_trainforge_capture
)

from .validation import (
    validate_decision,
    validate_phase_completeness,
    validate_capture_file,
    CaptureValidator as SchemaValidator,
    load_schema,
    check_jsonschema_available
)

from .constants import (
    PROJECT_DIR,
    TRAINING_DIR,
    TRAINING_DIR_LEGACY,
    SCHEMAS_DIR,
    LIBV2_ROOT,
    LIBV2_CATALOG,
    LIBV2_COURSES,
    LIBV2_ONTOLOGY,
    LIBV2_SCHEMA,
    MIN_DECISIONS_PER_PHASE,
    OPERATION_MAP,
    VALID_DECISION_TYPES,
    QUALITY_THRESHOLDS,
    VALIDATE_DECISIONS,
    validate_project_paths,
    ensure_training_dir,
)

from .libv2_storage import (
    LibV2Storage,
    LibV2StorageError,
    get_course_storage,
    list_all_courses,
    validate_libv2_structure,
)

from .quality import (
    assess_decision_quality,
    assess_from_inputs_ref,
    calculate_quality_breakdown,
    check_quality_acceptable,
)

__all__ = [
    # Core capture
    'DecisionCapture',
    'DARTDecisionCapture',
    'MLFeatures',
    'InputRef',
    'OutcomeSignals',
    'create_capture',
    'create_dart_capture',
    # Streaming
    'StreamingDecisionCapture',
    'CaptureValidator',
    'create_streaming_capture',
    'validate_phase_capture',
    'get_capture_stats',
    # Trainforge
    'TrainforgeDecisionCapture',
    'QuestionData',
    'RAGMetrics',
    'AlignmentCheck',
    'create_trainforge_capture',
    # Validation
    'validate_decision',
    'validate_phase_completeness',
    'validate_capture_file',
    'SchemaValidator',
    'load_schema',
    'check_jsonschema_available',
    # Constants
    'PROJECT_DIR',
    'TRAINING_DIR',
    'TRAINING_DIR_LEGACY',
    'SCHEMAS_DIR',
    'MIN_DECISIONS_PER_PHASE',
    'OPERATION_MAP',
    'VALID_DECISION_TYPES',
    'QUALITY_THRESHOLDS',
    'VALIDATE_DECISIONS',
    'validate_project_paths',
    'ensure_training_dir',
    # LibV2 Storage
    'LIBV2_ROOT',
    'LIBV2_CATALOG',
    'LIBV2_COURSES',
    'LIBV2_ONTOLOGY',
    'LIBV2_SCHEMA',
    'LibV2Storage',
    'LibV2StorageError',
    'get_course_storage',
    'list_all_courses',
    'validate_libv2_structure',
    # Quality
    'assess_decision_quality',
    'assess_from_inputs_ref',
    'calculate_quality_breakdown',
    'check_quality_acceptable',
]
