"""
Tool Parameter Schemas for MCP Tools

Defines the expected parameters for each tool and mappings from
generic task parameters to tool-specific parameter names.

This enables the TaskExecutor to call tools with the correct signatures
instead of the broken prompt-first calling convention.
"""

from typing import Any, Dict, List, Optional

# =============================================================================
# TOOL SCHEMAS
# =============================================================================
# Each schema defines:
#   - required: List of required parameter names
#   - optional: List of optional parameter names with defaults
#   - param_mapping: Maps generic task param names to tool-specific names
#   - defaults: Default values for optional parameters
# =============================================================================

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # DART TOOLS - Multi-Source Synthesis (5)
    # =========================================================================
    "convert_pdf_multi_source": {
        "required": ["combined_json_path"],
        "optional": ["output_path", "course_code"],
        "defaults": {
            "output_path": None,
            "course_code": None,
        },
        "param_mapping": {
            "input": "combined_json_path",
            "combined_json": "combined_json_path",
            "source": "combined_json_path",
            "output": "output_path",
            "output_path": "output_path",
            "course": "course_code",
        },
        "description": "Convert PDF using multi-source synthesis (pdftotext + pdfplumber + OCR)",
    },

    "batch_convert_multi_source": {
        "required": ["combined_dir"],
        "optional": ["output_zip", "output_dir"],
        "defaults": {
            "output_zip": None,
            "output_dir": None,
        },
        "param_mapping": {
            "input": "combined_dir",
            "source_dir": "combined_dir",
            "combined": "combined_dir",
            "output": "output_dir",
            "zip": "output_zip",
            "archive": "output_zip",
        },
        "description": "Batch convert using multi-source synthesis",
    },

    "list_available_campuses": {
        "required": [],
        "optional": [],
        "defaults": {},
        "param_mapping": {},
        "description": "List available campus combined JSON files for conversion",
    },

    "validate_wcag_compliance": {
        "required": ["html_path"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "input": "html_path",
            "file": "html_path",
            "path": "html_path",
        },
        "description": "Validate HTML file for WCAG 2.2 AA compliance",
    },

    "get_dart_status": {
        "required": [],
        "optional": [],
        "defaults": {},
        "param_mapping": {},
        "description": "Get DART installation status and capabilities",
    },

    # =========================================================================
    # PIPELINE TOOLS (Textbook-to-Course)
    # =========================================================================
    "stage_dart_outputs": {
        "required": ["run_id", "dart_html_paths", "course_name"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "html_paths": "dart_html_paths",
            "paths": "dart_html_paths",
            "course": "course_name",
        },
        "description": "Stage DART HTML outputs to Courseforge inputs directory",
    },

    "extract_and_convert_pdf": {
        "required": ["pdf_path"],
        "optional": ["course_code", "output_dir", "figures_dir"],
        "defaults": {
            "course_code": None,
            "output_dir": None,
            "figures_dir": None,
        },
        "param_mapping": {
            "input": "pdf_path",
            "source": "pdf_path",
            "path": "pdf_path",
            "pdf": "pdf_path",
            "course": "course_code",
            "output": "output_dir",
            "figures": "figures_dir",
        },
        "description": "Extract sources from PDF and convert to accessible HTML via DART",
    },

    # =========================================================================
    # COURSEFORGE TOOLS (6)
    # =========================================================================
    "create_course_project": {
        "required": ["course_name"],
        "optional": ["objectives_path", "duration_weeks", "credit_hours"],
        "defaults": {
            "objectives_path": None,
            "duration_weeks": 12,
            "credit_hours": 3,
        },
        "param_mapping": {
            "course": "course_name",
            "name": "course_name",
            "course_code": "course_name",
            "objectives": "objectives_path",
            "objectives_file": "objectives_path",
            "duration": "duration_weeks",
            "weeks": "duration_weeks",
            "credits": "credit_hours",
        },
        "description": "[DEPRECATED — use extract_textbook_structure + plan_course_structure from Wave 24] Initialize a new course generation project",
    },

    "generate_course_content": {
        "required": ["project_id"],
        "optional": ["week_range", "parallel"],
        "defaults": {
            "week_range": None,
            "parallel": True,
        },
        "param_mapping": {
            "project": "project_id",
            "id": "project_id",
            "weeks": "week_range",
            "range": "week_range",
        },
        "description": "Generate course content for specified weeks",
    },

    "package_imscc": {
        "required": ["project_id"],
        "optional": ["validate"],
        "defaults": {
            "validate": True,
        },
        "param_mapping": {
            "project": "project_id",
            "id": "project_id",
            "run_validation": "validate",
        },
        "description": "Package course content into IMSCC format",
    },

    "intake_imscc_package": {
        "required": ["imscc_path", "output_dir"],
        "optional": ["remediate"],
        "defaults": {
            "remediate": True,
        },
        "param_mapping": {
            "input": "imscc_path",
            "package": "imscc_path",
            "source": "imscc_path",
            "output": "output_dir",
            "extract_to": "output_dir",
            "auto_remediate": "remediate",
        },
        "description": "Import and analyze an existing IMSCC package",
    },

    "remediate_course_content": {
        "required": ["project_id"],
        "optional": ["remediation_types"],
        "defaults": {
            "remediation_types": None,
        },
        "param_mapping": {
            "project": "project_id",
            "id": "project_id",
            "types": "remediation_types",
            "remediation": "remediation_types",
        },
        "description": "Execute remediation on analyzed course content",
    },

    "get_courseforge_status": {
        "required": [],
        "optional": [],
        "defaults": {},
        "param_mapping": {},
        "description": "Get Courseforge installation status and active projects",
    },

    # =========================================================================
    # TRAINFORGE TOOLS (5)
    # =========================================================================
    "analyze_imscc_content": {
        "required": ["imscc_path"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "input": "imscc_path",
            "package": "imscc_path",
            "source": "imscc_path",
        },
        "description": "Analyze IMSCC package content for assessment generation",
    },

    "generate_assessments": {
        "required": ["course_id", "objective_ids", "bloom_levels"],
        "optional": ["question_count", "course_slug", "imscc_path"],
        "defaults": {
            "question_count": 10,
        },
        "param_mapping": {
            "course": "course_id",
            "objectives": "objective_ids",
            "blooms": "bloom_levels",
            "levels": "bloom_levels",
            "count": "question_count",
            "num_questions": "question_count",
            "slug": "course_slug",
            "package": "imscc_path",
            "imscc": "imscc_path",
        },
        "description": "Generate assessments from course content",
    },

    "validate_assessment": {
        "required": ["assessment_id"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "assessment": "assessment_id",
            "id": "assessment_id",
        },
        "description": "Validate generated assessment for quality and alignment",
    },

    "export_training_data": {
        "required": [],
        "optional": ["format_type", "date_range"],
        "defaults": {
            "format_type": "jsonl",
            "date_range": None,
        },
        "param_mapping": {
            "format": "format_type",
            "output_format": "format_type",
            "dates": "date_range",
            "range": "date_range",
        },
        "description": "Export captured training data in specified format",
    },

    "get_trainforge_status": {
        "required": [],
        "optional": [],
        "defaults": {},
        "param_mapping": {},
        "description": "Get Trainforge installation status and training data statistics",
    },

    # =========================================================================
    # ORCHESTRATOR TOOLS (9)
    # =========================================================================
    "create_workflow": {
        "required": ["workflow_type", "params"],
        "optional": ["priority"],
        "defaults": {
            "priority": "normal",
        },
        "param_mapping": {
            "type": "workflow_type",
            "workflow": "workflow_type",
            "parameters": "params",
            "config": "params",
        },
        "description": "Create a new workflow execution",
    },

    "get_workflow_status": {
        "required": ["workflow_id"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "workflow": "workflow_id",
            "id": "workflow_id",
        },
        "description": "Get current status of a workflow",
    },

    "dispatch_agent_task": {
        "required": ["workflow_id", "agent_type", "task_prompt"],
        "optional": ["dependencies"],
        "defaults": {
            "dependencies": None,
        },
        "param_mapping": {
            "workflow": "workflow_id",
            "agent": "agent_type",
            "prompt": "task_prompt",
            "task": "task_prompt",
            "deps": "dependencies",
        },
        "description": "Dispatch a task to a specific agent",
    },

    "poll_task_completions": {
        "required": [],
        "optional": ["workflow_id"],
        "defaults": {
            "workflow_id": None,
        },
        "param_mapping": {
            "workflow": "workflow_id",
            "id": "workflow_id",
        },
        "description": "Poll for completed or errored tasks",
    },

    "update_generation_progress": {
        "required": ["component", "status"],
        "optional": ["details"],
        "defaults": {
            "details": None,
        },
        "param_mapping": {
            "name": "component",
            "state": "status",
            "info": "details",
        },
        "description": "Update GENERATION_PROGRESS.md shared state",
    },

    "acquire_batch_lock": {
        "required": ["resource", "owner"],
        "optional": ["ttl_seconds"],
        "defaults": {
            "ttl_seconds": 3600,
        },
        "param_mapping": {
            "lock_name": "resource",
            "lock_owner": "owner",
            "ttl": "ttl_seconds",
            "timeout": "ttl_seconds",
        },
        "description": "Acquire exclusive lock on a resource for batch processing",
    },

    "release_batch_lock": {
        "required": ["resource", "owner"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "lock_name": "resource",
            "lock_owner": "owner",
        },
        "description": "Release a batch lock",
    },

    "execute_workflow_task": {
        "required": ["workflow_id", "task_id"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "workflow": "workflow_id",
            "task": "task_id",
        },
        "description": "Execute a pending task by invoking its mapped agent tool",
    },

    "complete_workflow_task": {
        "required": ["workflow_id", "task_id", "status"],
        "optional": ["result", "error"],
        "defaults": {
            "result": None,
            "error": None,
        },
        "param_mapping": {
            "workflow": "workflow_id",
            "task": "task_id",
            "state": "status",
            "output": "result",
            "err": "error",
        },
        "description": "Mark a workflow task as complete or failed",
    },

    "archive_to_libv2": {
        "required": ["course_name"],
        "optional": ["domain", "division", "pdf_paths", "html_paths", "imscc_path", "assessment_path", "subdomains"],
        "defaults": {
            "division": "STEM",
            "domain": "general",
        },
        "param_mapping": {
            "course_id": "course_name",
            "name": "course_name",
        },
        "description": "Archive pipeline artifacts to LibV2 repository",
    },

    "build_source_module_map": {
        "required": ["project_id"],
        "optional": ["staging_dir", "textbook_structure_path", "course_name"],
        "defaults": {},
        "param_mapping": {},
        "description": "Wave 9 source_mapping phase stub: writes an empty source_module_map.json so content-generator falls through to the LO-only backward-compat path.",
    },

    # Wave 24: Replace textbook-ingestor's create_course_project dispatch
    # with a real SemanticStructureExtractor call.
    "extract_textbook_structure": {
        "required": ["course_name"],
        "optional": [
            "staging_dir", "duration_weeks", "duration_weeks_explicit",
            "objectives_path", "credit_hours",
        ],
        "defaults": {
            "duration_weeks": 12,
            "duration_weeks_explicit": True,
            "credit_hours": 3,
        },
        "param_mapping": {
            "course": "course_name",
            "name": "course_name",
            "course_code": "course_name",
            "objectives": "objectives_path",
            "objectives_file": "objectives_path",
            "weeks": "duration_weeks",
            "duration": "duration_weeks",
        },
        "description": "Wave 24: Extract semantic structure from staged DART HTML into textbook_structure.json.",
    },

    # Wave 24: Synthesize + persist real TO-NN/CO-NN objectives from
    # the textbook_structure (or supplied objectives_path).
    "plan_course_structure": {
        "required": [],
        "optional": [
            "project_id", "course_name", "duration_weeks",
            "objectives_path", "staging_dir", "source_module_map_path",
        ],
        "defaults": {
            "duration_weeks": 12,
        },
        "param_mapping": {
            "project": "project_id",
            "course": "course_name",
            "name": "course_name",
            "course_code": "course_name",
            "objectives": "objectives_path",
            "objectives_file": "objectives_path",
            "weeks": "duration_weeks",
            "duration": "duration_weeks",
        },
        "description": "Wave 24: Plan course structure — synthesize TO/CO objectives from textbook structure and persist synthesized_objectives.json.",
    },

    # =========================================================================
    # PIPELINE TOOLS - Additional (create/run/status/markers)
    # Added by pipeline-plumbing remediation to close MCP audit Q1 (latent
    # PR #45 failure mode for tools present as @mcp.tool() + reachable from
    # agent mappings but missing TOOL_SCHEMAS entries). Required-param lists
    # match each tool's @mcp.tool() decorator signature in MCP/tools/*.py.
    # create_textbook_pipeline_tool + run_textbook_pipeline_tool are
    # DEPRECATED (Wave 7) but remain wired for one cycle.
    # =========================================================================
    "create_textbook_pipeline_tool": {
        "required": ["pdf_paths", "course_name"],
        "optional": [
            "objectives_path", "duration_weeks", "generate_assessments",
            "assessment_count", "bloom_levels", "priority",
        ],
        "defaults": {
            "objectives_path": None,
            "duration_weeks": 12,
            "generate_assessments": True,
            "assessment_count": 50,
            "bloom_levels": "remember,understand,apply,analyze",
            "priority": "normal",
        },
        "param_mapping": {
            "pdfs": "pdf_paths",
            "input": "pdf_paths",
            "sources": "pdf_paths",
            "course": "course_name",
            "name": "course_name",
            "objectives": "objectives_path",
            "duration": "duration_weeks",
            "weeks": "duration_weeks",
            "blooms": "bloom_levels",
            "count": "assessment_count",
        },
        "description": "Create and orchestrate a textbook-to-course pipeline (DEPRECATED Wave 7; prefer `ed4all run textbook-to-course`)",
    },

    "run_textbook_pipeline_tool": {
        "required": ["workflow_id"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "workflow": "workflow_id",
            "id": "workflow_id",
        },
        "description": "Execute a previously created textbook-to-course pipeline (DEPRECATED Wave 7)",
    },

    "get_pipeline_status": {
        "required": ["workflow_id"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "workflow": "workflow_id",
            "id": "workflow_id",
        },
        "description": "Get status of a textbook-to-course pipeline",
    },

    "validate_dart_markers": {
        "required": ["html_path"],
        "optional": [],
        "defaults": {},
        "param_mapping": {
            "input": "html_path",
            "file": "html_path",
            "path": "html_path",
        },
        "description": "Validate that an HTML file has required DART accessibility markers",
    },

    # =========================================================================
    # ANALYSIS TOOLS (3)
    # =========================================================================
    "analyze_training_data": {
        "required": [],
        "optional": [],
        "defaults": {},
        "param_mapping": {},
        "description": "Analyze captured training data quality and distribution",
    },

    "get_quality_distribution": {
        "required": [],
        "optional": ["min_quality"],
        "defaults": {
            "min_quality": "developing",
        },
        "param_mapping": {
            "quality": "min_quality",
            "threshold": "min_quality",
        },
        "description": "Get quality distribution with filtering preview",
    },

    "preview_export_filter": {
        "required": [],
        "optional": ["min_quality", "min_confidence", "require_accepted", "deduplicate"],
        "defaults": {
            "min_quality": "developing",
            "min_confidence": 0.0,
            "require_accepted": False,
            "deduplicate": True,
        },
        "param_mapping": {
            "quality": "min_quality",
            "confidence": "min_confidence",
            "accepted_only": "require_accepted",
            "dedupe": "deduplicate",
        },
        "description": "Preview how many records would be exported with given filters",
    },

}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the schema for a specific tool.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool schema dict or None if not found
    """
    return TOOL_SCHEMAS.get(tool_name)


def get_required_params(tool_name: str) -> List[str]:
    """
    Get required parameters for a tool.

    Args:
        tool_name: Name of the tool

    Returns:
        List of required parameter names
    """
    schema = TOOL_SCHEMAS.get(tool_name, {})
    return schema.get("required", [])


def get_optional_params(tool_name: str) -> List[str]:
    """
    Get optional parameters for a tool.

    Args:
        tool_name: Name of the tool

    Returns:
        List of optional parameter names
    """
    schema = TOOL_SCHEMAS.get(tool_name, {})
    return schema.get("optional", [])


def get_param_mapping(tool_name: str) -> Dict[str, str]:
    """
    Get parameter name mapping for a tool.

    Args:
        tool_name: Name of the tool

    Returns:
        Dict mapping generic names to tool-specific names
    """
    schema = TOOL_SCHEMAS.get(tool_name, {})
    return schema.get("param_mapping", {})


def get_defaults(tool_name: str) -> Dict[str, Any]:
    """
    Get default values for optional parameters.

    Args:
        tool_name: Name of the tool

    Returns:
        Dict of parameter defaults
    """
    schema = TOOL_SCHEMAS.get(tool_name, {})
    return schema.get("defaults", {})


def list_all_tools() -> List[str]:
    """
    Get list of all tool names.

    Returns:
        List of tool names
    """
    return list(TOOL_SCHEMAS.keys())


def validate_tool_params(tool_name: str, params: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate that all required parameters are present.

    Args:
        tool_name: Name of the tool
        params: Parameters to validate

    Returns:
        Tuple of (is_valid, list of missing required params)
    """
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return False, [f"Unknown tool: {tool_name}"]

    required = schema.get("required", [])
    mapping = schema.get("param_mapping", {})

    missing = []
    for req_param in required:
        if req_param in params:
            continue
        # Check if any mapped name is present
        found = False
        for mapped_name, target in mapping.items():
            if target == req_param and mapped_name in params:
                found = True
                break
        if not found:
            missing.append(req_param)

    return len(missing) == 0, missing


# =============================================================================
# TOOL CATEGORIES (for documentation/filtering)
# =============================================================================

TOOL_CATEGORIES = {
    "dart": [
        "convert_pdf_multi_source",
        "batch_convert_multi_source",
        "list_available_campuses",
        "validate_wcag_compliance",
        "get_dart_status",
    ],
    "courseforge": [
        "create_course_project",
        "generate_course_content",
        "package_imscc",
        "intake_imscc_package",
        "remediate_course_content",
        "get_courseforge_status",
    ],
    "trainforge": [
        "analyze_imscc_content",
        "generate_assessments",
        "validate_assessment",
        "export_training_data",
        "get_trainforge_status",
    ],
    "orchestrator": [
        "create_workflow",
        "get_workflow_status",
        "dispatch_agent_task",
        "poll_task_completions",
        "update_generation_progress",
        "acquire_batch_lock",
        "release_batch_lock",
        "execute_workflow_task",
        "complete_workflow_task",
    ],
    "pipeline": [
        "stage_dart_outputs",
        "extract_and_convert_pdf",
        "archive_to_libv2",
    ],
}


def get_tools_by_category(category: str) -> List[str]:
    """
    Get tools belonging to a specific category.

    Args:
        category: Category name (dart, courseforge, trainforge, orchestrator)

    Returns:
        List of tool names in that category
    """
    return TOOL_CATEGORIES.get(category, [])
