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
        "optional": ["stage_mode"],
        "defaults": {"stage_mode": None},
        "param_mapping": {
            "html_paths": "dart_html_paths",
            "paths": "dart_html_paths",
            "course": "course_name",
            "mode": "stage_mode",
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
        # Wave 37: machine-readable deprecation flag so operators /
        # audit tooling can surface the status without string-matching
        # the description. New integrations should route through
        # ``extract_textbook_structure`` + ``plan_course_structure``
        # (pipeline-internal) or ``textbook_to_course`` via the unified
        # CLI; this entry remains registered for external MCP clients
        # that already depend on it.
        "deprecated": True,
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

    # Wave 30 Gap 3 / Wave 32 Deliverable A: register the
    # synthesize_training schema so
    # ``param_mapper.get_tool_schema("synthesize_training")`` stops
    # returning None. Pre-Wave-32 the Wave-30 PR wired the tool into
    # ``_build_tool_registry`` + ``AGENT_TOOL_MAPPING`` but missed this
    # third location of the three-location wiring invariant, so at
    # runtime the param mapper raised ``ParameterMappingError("Unknown
    # tool: synthesize_training")`` on every dispatch, tripped the
    # poison-pill detector, and the ``training_synthesis`` phase never
    # produced ``instruction_pairs.jsonl`` / ``preference_pairs.jsonl``
    # in real runs.
    #
    # Signature mirrors both callers:
    #   * ``MCP/tools/pipeline_tools.py::synthesize_training`` (@mcp.tool
    #     variant at L573) → (corpus_dir, course_code, provider, seed).
    #   * ``MCP/tools/pipeline_tools.py::_synthesize_training`` (pipeline
    #     registry variant at L2993) — accepts a wider alias surface
    #     (``trainforge_dir`` / ``course_name`` / ``course_id`` /
    #     ``assessments_path`` / ``chunks_path``) so the param mapping
    #     block below also registers those as aliases.
    "synthesize_training": {
        # Wave 33 Bug A: Pre-Wave-33 ``corpus_dir`` was listed as
        # required and ``assessments_path`` / ``chunks_path`` weren't
        # recognised at all, so the live dispatcher (which routes
        # ``assessments_path`` + ``chunks_path`` from the
        # ``trainforge_assessment`` phase outputs — see
        # ``config/workflows.yaml::training_synthesis.inputs_from``)
        # triggered ``ParameterMappingError("Missing required
        # parameters: ['corpus_dir']")`` on every run. The tool
        # function already accepts + derives ``corpus_dir`` from any of
        # ``corpus_dir`` / ``trainforge_dir`` / ``output_dir`` /
        # ``assessments_path`` (parent) / ``chunks_path`` (grandparent)
        # and returns a structured error envelope when none are given,
        # so the schema's contribution is limited to: enforce
        # ``course_code`` (the one kwarg the tool genuinely can't
        # derive) and surface the rest as optional pass-through kwargs.
        # ``param_mapping`` keeps the ``trainforge_dir`` → ``corpus_dir``
        # alias for legacy callers, but ``assessments_path`` and
        # ``chunks_path`` are deliberately NOT mapped — renaming them
        # to ``corpus_dir`` would hand the tool a file path masquerading
        # as a directory (chunks.jsonl vs. its grandparent), breaking
        # the corpus/chunks.jsonl lookup.
        "required": ["course_code"],
        "optional": [
            "corpus_dir",
            "trainforge_dir",
            "assessments_path",
            "chunks_path",
            "provider",
            "seed",
            # Wave 129: deterministic-generator pass-throughs (Wave 124-127).
            # Mirrors run_synthesis() kwargs at
            # Trainforge/synthesize_training.py:677-685 so the workflow-phase
            # dispatch + external MCP clients can trigger kg_metadata /
            # violation_detection / abstention / schema_translation without
            # routing through the CLI.
            "with_kg_metadata",
            "kg_metadata_max_pairs",
            "with_violation_detection",
            "violation_detection_max_pairs",
            "with_abstention",
            "abstention_max_pairs",
            "with_schema_translation",
            "schema_translation_max_pairs",
        ],
        "defaults": {
            "provider": "mock",
            "seed": None,
            "with_kg_metadata": False,
            "kg_metadata_max_pairs": 2000,
            "with_violation_detection": False,
            # violation_detection_max_pairs intentionally unset =
            # unlimited (family-balanced round-robin trim only when
            # caller passes an explicit cap).
            "with_abstention": False,
            "abstention_max_pairs": 1000,
            "with_schema_translation": False,
            "schema_translation_max_pairs": 50,
        },
        "param_mapping": {
            # Corpus dir aliases — registry variant accepts any of these
            # and derives the corpus_dir when one isn't passed directly.
            # NOTE: assessments_path / chunks_path are pass-through
            # (see header comment) — the tool derives corpus_dir from
            # them internally.
            "trainforge_dir": "corpus_dir",
            "output_dir": "corpus_dir",
            "workspace": "corpus_dir",
            # Course code aliases — registry variant maps course_name /
            # course_id onto course_code for decision capture.
            "course_name": "course_code",
            "course_id": "course_code",
            "course": "course_code",
            "name": "course_code",
        },
        "description": (
            "Wave 30 Gap 3 (+ Wave 33 Bug A dispatch-shape fix): "
            "synthesize SFT + DPO training pairs from a Trainforge "
            "corpus (reads corpus/chunks.jsonl, writes "
            "training_specs/instruction_pairs.jsonl + preference_pairs.jsonl)."
        ),
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
    # PIPELINE TOOLS - Additional (status/markers)
    # Added by pipeline-plumbing remediation to close MCP audit Q1 (latent
    # PR #45 failure mode for tools present as @mcp.tool() + reachable from
    # agent mappings but missing TOOL_SCHEMAS entries). Required-param lists
    # match each tool's @mcp.tool() decorator signature in MCP/tools/*.py.
    # Wave 28f: create_textbook_pipeline_tool + run_textbook_pipeline_tool
    # (Wave 7 deprecated wrappers) were removed entirely — external MCP
    # clients now route through the workflow API.
    # =========================================================================
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

    # =========================================================================
    # PHASE 4 SUBTASK 4 — TWO-PASS ROUTER PHASE-HANDLER SCHEMAS
    # =========================================================================
    # Phase 3.5 wired the four ``_run_*`` phase helpers into
    # ``_build_tool_registry`` (pipeline_tools.py:4326-4335) and into
    # ``_PHASE_TOOL_MAPPING`` (executor.py:206) but missed the third
    # leg of the three-location wiring invariant: ``TOOL_SCHEMAS`` here.
    # Without these entries, ``param_mapper.map_task_to_tool_params``
    # raises ``ParameterMappingError("Unknown tool: ...")`` whenever
    # the executor tries to invoke them via the synthetic task that
    # Phase 4 Subtask 1 introduced — the same failure mode the Wave 30
    # comment at the synthesize_training schema documents.
    #
    # All four take ``project_id`` as the only required input plus a
    # phase-specific ``blocks_*_path`` to chain between phases. The
    # pipeline_tools handlers tolerate missing optional kwargs by
    # returning a structured error envelope, so the schemas keep
    # ``required`` minimal and let the handler enforce the contract.
    # =========================================================================
    "run_content_generation_outline": {
        "required": ["project_id"],
        "optional": [
            "source_module_map_path",
            "staging_dir",
            "duration_weeks_explicit",
        ],
        "defaults": {},
        "param_mapping": {},
        "description": (
            "Phase 3 two-pass router outline tier — emits "
            "blocks_outline_path JSONL of outline-tier Blocks."
        ),
    },

    "run_inter_tier_validation": {
        "required": ["blocks_outline_path", "project_id"],
        "optional": [],
        "defaults": {},
        "param_mapping": {},
        "description": (
            "Phase 3 inter-tier validation — runs Block-input "
            "validators on outline-tier blocks and emits "
            "blocks_validated_path / blocks_failed_path JSONL."
        ),
    },

    "run_content_generation_rewrite": {
        "required": ["project_id"],
        "optional": ["blocks_validated_path"],
        "defaults": {},
        "param_mapping": {},
        "description": (
            "Phase 3 two-pass router rewrite tier — emits final "
            "HTML pages plus blocks_final_path JSONL."
        ),
    },

    "run_post_rewrite_validation": {
        "required": ["blocks_final_path", "project_id"],
        "optional": [],
        "defaults": {},
        "param_mapping": {},
        "description": (
            "Phase 3.5 post-rewrite validation — re-runs the four "
            "Block-input validators against rewrite-tier blocks and "
            "emits blocks_validated_path / blocks_failed_path JSONL."
        ),
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
        "synthesize_training",
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
