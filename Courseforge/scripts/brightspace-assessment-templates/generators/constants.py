"""
IMSCC Constants and Namespace Definitions

Single source of truth for all IMSCC namespaces, resource types,
and other constants used across all generators.

CRITICAL: Do not hardcode these values elsewhere. Always import from here.
"""

# ============================================================================
# IMSCC NAMESPACES
# ============================================================================
# Verified from actual Brightspace exports

NAMESPACES = {
    # Assignment namespace - IMSCC extensions
    'assignment': 'http://www.imsglobal.org/xsd/imscc_extensions/assignment',

    # Discussion namespace - IMSCC 1.3 discussion topic
    'discussion': 'http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3',

    # QTI namespace - QTI ASI v1.2
    'qti': 'http://www.imsglobal.org/xsd/ims_qtiasiv1p2',

    # Manifest namespace - IMSCC 1.3 content packaging
    'manifest': 'http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1',

    # Common namespaces
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    'lom': 'http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource',
    'lomimscc': 'http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest',
}

# ============================================================================
# SCHEMA LOCATIONS
# ============================================================================

SCHEMA_LOCATIONS = {
    'assignment': (
        'http://www.imsglobal.org/xsd/imscc_extensions/assignment '
        'http://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd'
    ),
    'discussion': (
        'http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3 '
        'http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3.xsd'
    ),
    'qti': (
        'http://www.imsglobal.org/xsd/ims_qtiasiv1p2 '
        'http://www.imsglobal.org/xsd/ims_qtiasiv1p2p1.xsd'
    ),
    'manifest': (
        'http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1 '
        'http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1.xsd'
    ),
}

# ============================================================================
# RESOURCE TYPES
# ============================================================================
# Exact resource types for manifest <resource type="...">

RESOURCE_TYPES = {
    'assignment': 'assignment_xmlv1p0',
    'discussion': 'imsdt_xmlv1p3',
    'quiz': 'imsqti_xmlv1p2/imscc_xmlv1p3/assessment',
    'webcontent': 'webcontent',
    'weblink': 'imswl_xmlv1p3',
}

# ============================================================================
# QTI QUESTION PROFILES
# ============================================================================
# Valid cc_profile values for QTI question types

QTI_QUESTION_PROFILES = {
    'multiple_choice': 'cc.multiple_choice.v0p1',
    'multiple_response': 'cc.multiple_response.v0p1',
    'true_false': 'cc.true_false.v0p1',
    'fill_in_blank': 'cc.fib.v0p1',
    'essay': 'cc.essay.v0p1',
}

# QTI assessment profile
QTI_ASSESSMENT_PROFILE = 'cc.exam.v0p1'

# ============================================================================
# VALIDATION CONSTRAINTS
# ============================================================================

# Maximum points value allowed (prevents overflow/display issues)
MAX_POINTS = 10000.0

# Minimum points value (must be non-negative)
MIN_POINTS = 0.0

# Maximum title length (characters)
MAX_TITLE_LENGTH = 500

# Maximum content length (characters) - prevents massive XML files
MAX_CONTENT_LENGTH = 1000000  # 1MB of text

# Valid submission types for assignments
VALID_SUBMISSION_TYPES = frozenset(['file', 'text', 'url', 'online'])

# ============================================================================
# DEPRECATED NAMESPACES (for validation/detection)
# ============================================================================
# These should trigger warnings or errors if detected

DEPRECATED_NAMESPACES = {
    'd2l_2p0': 'http://www.desire2learn.com/xsd/d2l_2p0',
    'desire2learn': 'http://www.desire2learn.com',
}
