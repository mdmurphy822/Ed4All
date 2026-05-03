"""
Workflow Runner - Executes multi-phase workflows end-to-end.

This module provides the missing orchestration layer that chains
workflow phases together, routing outputs from each phase into
the next phase's inputs.

Usage:
    runner = WorkflowRunner(executor, config)
    result = await runner.run_workflow(workflow_id)
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_LO_ID_RE = re.compile(r"^[a-zA-Z]{2,}-\d{2,}$")

from .config import OrchestratorConfig, WorkflowPhase
from .executor import _PHASE_TOOL_MAPPING, ExecutionResult, TaskExecutor

logger = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parent.parent.parent / "state"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_YAML_PATH = PROJECT_ROOT / "config" / "workflows.yaml"
WORKFLOWS_META_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "config" / "workflows_meta.schema.json"


# =============================================================================
# INTER-PHASE DATA ROUTING
# =============================================================================
# Defines how outputs from one phase become inputs to the next.
# Format: {phase_name: {param_name: (source_type, *source_path)}}
#   - ("workflow_params", key) => from workflow creation params
#   - ("phase_outputs", phase_name, key) => from a prior phase's extracted outputs
#   - ("literal", value) => hardcoded value
#
# REC-CTR-05 (Wave 6): Routing is now primarily defined in config/workflows.yaml
# via per-phase `inputs_from:` and `outputs:` blocks. The legacy dicts below
# act as backwards-compat fallbacks for phases whose YAML entries have not yet
# been annotated. `_load_workflows_config()` validates the YAML against
# schemas/config/workflows_meta.schema.json at module load time, so typos in
# gate IDs, phase names, severities, or inter-phase references are caught
# pre-flight.
# =============================================================================

_LEGACY_PHASE_PARAM_ROUTING: Dict[str, Dict[str, Tuple]] = {
    "dart_conversion": {
        # Task creation handled specially in _create_phase_tasks (one task per PDF)
        "course_code": ("workflow_params", "course_name"),
    },
    "staging": {
        "run_id": ("workflow_params", "run_id"),
        "dart_html_paths": ("phase_outputs", "dart_conversion", "output_paths"),
        "course_name": ("workflow_params", "course_name"),
    },
    "objective_extraction": {
        "course_name": ("workflow_params", "course_name"),
        "objectives_path": ("workflow_params", "objectives_path"),
        "duration_weeks": ("workflow_params", "duration_weeks"),
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
        # Wave 24: textbook-ingestor needs staging_dir so
        # extract_textbook_structure can walk the staged DART HTML.
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
    },
    "source_mapping": {
        # Wave 9: DART source-block -> Courseforge page routing.
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "textbook_structure_path": (
            "phase_outputs", "objective_extraction", "textbook_structure_path",
        ),
    },
    # Phase 7b ST 11: chunking phase — DART chunkset emit. Mirrors the
    # YAML routing at config/workflows.yaml::chunking. Phase 8 ST 3
    # adds the optional ``libv2_root`` workflow param so ops topologies
    # that mount LibV2 at a non-default location can override the
    # in-tree default via ``--libv2-root`` / ``ED4ALL_LIBV2_ROOT``.
    "chunking": {
        "course_name": ("workflow_params", "course_name"),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "libv2_root": ("workflow_params", "libv2_root"),
    },
    # Phase 6 ST 11: concept_extraction phase — pedagogy-graph builder.
    # Mirrors the YAML routing at config/workflows.yaml::concept_extraction.
    # Phase 7b ST 14.5 added the upstream dart_chunks_path consumption;
    # Phase 8 ST 3 adds the optional ``libv2_root`` workflow param.
    "concept_extraction": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "course_name": ("workflow_params", "course_name"),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "dart_chunks_path": (
            "phase_outputs", "chunking", "dart_chunks_path",
        ),
        "libv2_root": ("workflow_params", "libv2_root"),
    },
    "course_planning": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "course_name": ("workflow_params", "course_name"),
        "objectives_path": ("workflow_params", "objectives_path"),
        "duration_weeks": ("workflow_params", "duration_weeks"),
        # Wave 40: route duration_weeks_explicit so _plan_course_structure's
        # config-over-kwargs precedence check activates on real runs.
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
        "source_module_map_path": (
            "phase_outputs", "source_mapping", "source_module_map_path",
        ),
        # Phase 6 ST 16 / Phase 8 ST 5: route the concept-graph path
        # emitted by ``concept_extraction`` so the planner's two-stage
        # linker populates ``LearningObjective.keyConcepts[]`` from
        # ``concept_graph_semantic.json`` before persistence. Mirrors
        # the YAML routing at config/workflows.yaml::course_planning;
        # the legacy dict is consulted as a fallback when YAML lookup
        # misses (see ``_get_phase_param_routing``).
        "concept_graph_path": (
            "phase_outputs", "concept_extraction", "concept_graph_path",
        ),
    },
    "content_generation": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "source_module_map_path": (
            "phase_outputs", "source_mapping", "source_module_map_path",
        ),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        # Wave 40: same rationale as course_planning —
        # _generate_course_content's precedence check needs the flag.
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
    },
    # Phase 3 Subtask 5: input routing for the two-pass router phases.
    # Mirrors the legacy ``content_generation`` routing for the outline
    # tier; the rewrite tier additionally consumes
    # ``blocks_validated_path`` from the inter-tier validation phase.
    "content_generation_outline": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "source_module_map_path": (
            "phase_outputs", "source_mapping", "source_module_map_path",
        ),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
    },
    "inter_tier_validation": {
        "blocks_outline_path": (
            "phase_outputs", "content_generation_outline",
            "blocks_outline_path",
        ),
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
    },
    "content_generation_rewrite": {
        "blocks_validated_path": (
            "phase_outputs", "inter_tier_validation",
            "blocks_validated_path",
        ),
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "source_module_map_path": (
            "phase_outputs", "source_mapping", "source_module_map_path",
        ),
        "staging_dir": ("phase_outputs", "staging", "staging_dir"),
        "duration_weeks_explicit": (
            "workflow_params", "duration_weeks_explicit",
        ),
    },
    "packaging": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
    },
    # Phase 7c ST 16: imscc_chunking phase — IMSCC chunkset emit
    # post-packaging. Mirrors the YAML routing at
    # config/workflows.yaml::imscc_chunking. Phase 8 ST 3 adds the
    # optional ``libv2_root`` workflow param so ops topologies that
    # mount LibV2 at a non-default location can override the in-tree
    # default via ``--libv2-root`` / ``ED4ALL_LIBV2_ROOT``.
    "imscc_chunking": {
        "course_name": ("workflow_params", "course_name"),
        "imscc_path": ("phase_outputs", "packaging", "package_path"),
        "libv2_root": ("workflow_params", "libv2_root"),
    },
    "trainforge_assessment": {
        "course_id": ("workflow_params", "course_name"),
        "imscc_path": ("phase_outputs", "packaging", "package_path"),
        "bloom_levels": ("workflow_params", "bloom_levels"),
        "question_count": ("workflow_params", "assessment_count"),
        # Wave 24: real TO/CO objective_ids come from course_planning
        # (was objective_extraction with phantom {COURSE}_OBJ_N IDs).
        "objective_ids": ("phase_outputs", "course_planning", "objective_ids"),
        # Phase 8 ST 2: route the upstream IMSCC chunkset path
        # written by ``imscc_chunking`` (Phase 7c ST 16) so
        # ``_run_trainforge_assessment`` can pass it to
        # ``CourseProcessor`` and short-circuit the in-process
        # ``_chunk_content`` rebuild. Mirrors the equivalent YAML
        # routing at ``config/workflows.yaml::trainforge_assessment``;
        # the legacy dict is consulted as a fallback when YAML
        # lookup misses (see ``_get_phase_param_routing``).
        "imscc_chunks_path": (
            "phase_outputs", "imscc_chunking", "imscc_chunks_path",
        ),
    },
    "libv2_archival": {
        "course_name": ("workflow_params", "course_name"),
        "domain": ("workflow_params", "domain"),
        "division": ("workflow_params", "division"),
        "pdf_paths": ("workflow_params", "pdf_paths"),
        "html_paths": ("phase_outputs", "dart_conversion", "output_paths"),
        "imscc_path": ("phase_outputs", "packaging", "package_path"),
        # Phase 6 ST 18 / Phase 7c.5 / Phase 8 ST 5: thread the three
        # chunkset SHA-256s (concept graph from ``concept_extraction``,
        # DART chunkset from ``chunking``, IMSCC chunkset from
        # ``imscc_chunking``) so the LibV2 manifest carries each hash
        # and the ``libv2_manifest`` gate can cross-check on-disk
        # artifacts. Mirrors the YAML routing at
        # config/workflows.yaml::libv2_archival; the legacy dict is
        # consulted as a fallback when YAML lookup misses (see
        # ``_get_phase_param_routing``).
        "concept_graph_sha256": (
            "phase_outputs", "concept_extraction", "concept_graph_sha256",
        ),
        "dart_chunks_sha256": (
            "phase_outputs", "chunking", "dart_chunks_sha256",
        ),
        "imscc_chunks_sha256": (
            "phase_outputs", "imscc_chunking", "imscc_chunks_sha256",
        ),
    },
    "finalization": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
        "course_slug": ("phase_outputs", "libv2_archival", "course_slug"),
    },
}

# Maps phase names to the keys extracted from their task results.
# After a phase completes, these fields are pulled from the result
# and stored in workflow state under phase_outputs[phase_name].
_LEGACY_PHASE_OUTPUT_KEYS: Dict[str, List[str]] = {
    # Wave 32 Deliverable B: surface html_path + html_paths (router
    # canonical keys) alongside the legacy output_path / output_paths
    # aliases so the DartMarkersValidator gate builder picks them up
    # without a router change. Pre-Wave-32 runs reported
    # ``dart_markers skipped — missing inputs: html_path`` because
    # ``_build_dart_markers`` looked for html_path but the phase only
    # surfaced output_path.
    "dart_conversion": [
        "output_path", "output_paths",
        "html_path", "html_paths",
        "success", "html_length",
    ],
    "staging": ["staging_dir", "staged_files", "file_count"],
    # Wave 24: objective_extraction no longer emits objective_ids; it
    # now emits textbook_structure_path + chapter_count + source_file_count
    # + duration_weeks (autoscaled when --weeks unset).
    # Real objective_ids surface from course_planning's synthesize step.
    "objective_extraction": [
        "project_id", "project_path", "textbook_structure_path",
        "chapter_count", "duration_weeks", "source_file_count",
    ],
    "source_mapping": ["source_module_map_path", "source_chunk_ids"],
    "course_planning": [
        "project_id", "synthesized_objectives_path",
        "objective_ids", "terminal_count", "chapter_count",
    ],
    # Wave 32 Deliverable B: add page_paths + content_dir so the
    # ContentGroundingValidator + PageObjectivesValidator builders
    # can resolve inputs (pre-Wave-32 both gates silently skipped).
    "content_generation": [
        "project_id", "content_paths", "page_paths", "content_dir",
        "weeks_prepared",
    ],
    # Phase 3 Subtask 5: two-pass router phase output declarations.
    # The outline tier emits a Block-list JSON sidecar (no HTML body);
    # the validation tier filters into pass/fail Block lists; the
    # rewrite tier emits the final HTML pages plus a final Block JSON
    # for downstream consumers (Trainforge ingest reads from the
    # rewrite-tier blocks_final_path when COURSEFORGE_TWO_PASS=true).
    "content_generation_outline": [
        "blocks_outline_path", "project_id", "weeks_prepared",
    ],
    "inter_tier_validation": [
        "blocks_validated_path", "blocks_failed_path",
    ],
    "content_generation_rewrite": [
        "content_paths", "page_paths", "content_dir",
        "blocks_final_path",
    ],
    # Phase 3.5 Subtask 12: post-rewrite validation phase output keys.
    # Mirrors inter_tier_validation's shape — emits
    # ``blocks_validated_path`` (rewrite-tier blocks that passed every
    # gate) and ``blocks_failed_path`` (rewrite-tier blocks that
    # tripped at least one gate). Packaging consumes blocks_validated_path
    # via the post_rewrite_validation -> packaging dependency chain
    # introduced in Subtask 10 + 11.
    "post_rewrite_validation": [
        "blocks_validated_path", "blocks_failed_path",
    ],
    # Wave 32 Deliverable B: surface imscc_path + content_dir so
    # IMSCCValidator + PageObjectivesValidator builders pick them up.
    "packaging": [
        "package_path", "libv2_package_path", "imscc_path",
        "content_dir", "project_id",
    ],
    # Wave 24: surface chunks_path + assessments_path for the
    # assessment_objective_alignment gate input builder.
    "trainforge_assessment": [
        "output_path", "assessments_path", "assessment_id",
        "question_count", "chunks_path",
    ],
    "libv2_archival": ["course_slug", "course_dir", "manifest_path"],
    "finalization": ["project_id", "package_path", "course_slug"],
}


# Backwards-compat: expose the legacy aliases. Callers outside this module
# historically imported these names directly. New code should call
# _get_phase_param_routing() / _get_phase_output_keys() or the YAML-first
# accessors, which respect per-phase YAML overrides.
PHASE_PARAM_ROUTING = _LEGACY_PHASE_PARAM_ROUTING
PHASE_OUTPUT_KEYS = _LEGACY_PHASE_OUTPUT_KEYS


# =============================================================================
# YAML-BASED PHASE ROUTING LOADER (REC-CTR-05)
# =============================================================================

# Module-level cache for loaded + validated workflows.yaml. Populated lazily
# by _load_workflows_config(). Reset for tests via _reset_workflows_cache().
_WORKFLOWS_CONFIG_CACHE: Optional[Dict[str, Any]] = None

# Track phases we've already warn-logged for fall-through to legacy defaults,
# to avoid log spam when the same phase fires repeatedly across a workflow.
_FALLBACK_LOGGED: set = set()


def _reset_workflows_cache() -> None:
    """Clear the cached workflows config and fallback-log tracker.

    Primarily used by tests to force a reload after modifying the underlying
    YAML or schema on disk.
    """
    global _WORKFLOWS_CONFIG_CACHE
    _WORKFLOWS_CONFIG_CACHE = None
    _FALLBACK_LOGGED.clear()


def _load_workflows_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load and validate config/workflows.yaml against the meta-schema.

    Validates against schemas/config/workflows_meta.schema.json plus a
    cross-reference integrity check: any `inputs_from` entry with
    source=phase_outputs must reference a prior-phase output declared in
    that phase's `outputs:` list.

    Raises:
        ValueError: If workflows.yaml is missing, malformed, or fails
            meta-schema/cross-ref validation.

    Returns:
        The raw parsed YAML dict (already validated).
    """
    global _WORKFLOWS_CONFIG_CACHE
    if _WORKFLOWS_CONFIG_CACHE is not None and not force_reload:
        return _WORKFLOWS_CONFIG_CACHE

    if not WORKFLOWS_YAML_PATH.exists():
        raise ValueError(
            f"Workflows config not found: {WORKFLOWS_YAML_PATH}. "
            "workflow_runner requires config/workflows.yaml to load phase routing."
        )

    try:
        with open(WORKFLOWS_YAML_PATH) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {WORKFLOWS_YAML_PATH}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"workflows.yaml must be a mapping at the top level, got {type(data).__name__}"
        )

    # Meta-schema validation (REC-CTR-05). If jsonschema is not installed or
    # the schema file is missing, log a warning and skip — don't block
    # execution purely on meta-schema tooling availability.
    if WORKFLOWS_META_SCHEMA_PATH.exists():
        try:
            import jsonschema
            with open(WORKFLOWS_META_SCHEMA_PATH) as f:
                meta_schema = json.load(f)
            try:
                jsonschema.validate(data, meta_schema)
            except jsonschema.ValidationError as e:
                path = ".".join(str(p) for p in e.absolute_path)
                raise ValueError(
                    f"config/workflows.yaml failed meta-schema validation at '{path}': "
                    f"{e.message}"
                ) from e
        except ImportError:
            logger.warning(
                "jsonschema not installed; skipping workflows.yaml meta-schema validation. "
                "Install jsonschema to catch config typos pre-flight."
            )
    else:
        logger.warning(
            "Meta-schema not found at %s; skipping structural validation of workflows.yaml.",
            WORKFLOWS_META_SCHEMA_PATH,
        )

    # Cross-reference integrity: every phase_outputs input must resolve
    # to a prior phase's declared outputs.
    _validate_inputs_from_references(data)

    _WORKFLOWS_CONFIG_CACHE = data
    return data


def _validate_inputs_from_references(workflows_data: Dict[str, Any]) -> None:
    """Ensure `inputs_from: {source: phase_outputs,...}` references resolve.

    For each workflow, iterates phases in declared order and checks that any
    phase_outputs-sourced input refers to (phase, output) that was declared
    in a prior phase's `outputs:` list. Phases without an explicit `outputs:`
    block are treated as exposing the legacy output keys for that phase,
    preserving backwards compatibility.

    Raises:
        ValueError: On the first unresolved reference, with a clear message.
    """
    for wf_name, wf in (workflows_data.get("workflows") or {}).items():
        if not isinstance(wf, dict):
            continue
        seen_outputs: Dict[str, set] = {}
        for phase in wf.get("phases", []) or []:
            if not isinstance(phase, dict):
                continue
            phase_name = phase.get("name", "<unnamed>")
            for route in phase.get("inputs_from") or []:
                if not isinstance(route, dict):
                    continue
                if route.get("source") != "phase_outputs":
                    continue
                ref_phase = route.get("phase")
                ref_output = route.get("output")
                if ref_phase not in seen_outputs:
                    raise ValueError(
                        f"Workflow '{wf_name}' phase '{phase_name}' inputs_from "
                        f"references unknown or not-yet-declared phase '{ref_phase}'."
                    )
                if ref_output not in seen_outputs[ref_phase]:
                    raise ValueError(
                        f"Workflow '{wf_name}' phase '{phase_name}' inputs_from "
                        f"references '{ref_phase}.{ref_output}' but '{ref_phase}' does "
                        f"not declare '{ref_output}' in its outputs. "
                        f"Declared outputs: {sorted(seen_outputs[ref_phase])}"
                    )
            # Record this phase's declared outputs, falling back to legacy
            # keys so legacy phases still satisfy downstream references.
            declared = phase.get("outputs")
            if declared is None:
                declared = _LEGACY_PHASE_OUTPUT_KEYS.get(phase_name, [])
            seen_outputs[phase_name] = set(declared or [])


def _phase_yaml_block(phase_name: str) -> Optional[Dict[str, Any]]:
    """Locate the first phase entry matching `phase_name` across all workflows.

    Phase names are used as dict keys in the legacy dicts, so callers only
    have a phase name (not workflow+phase). If the same phase name appears in
    multiple workflows (e.g. `dart_conversion` in `batch_dart`-siblings),
    the first YAML block with an `inputs_from:` or `outputs:` annotation
    wins. This preserves the prior implicit behavior where a phase had a
    single global routing signature.
    """
    try:
        data = _load_workflows_config()
    except ValueError:
        # Propagate to caller at first use; logged there.
        raise

    fallback: Optional[Dict[str, Any]] = None
    for wf in (data.get("workflows") or {}).values():
        if not isinstance(wf, dict):
            continue
        for phase in wf.get("phases", []) or []:
            if not isinstance(phase, dict):
                continue
            if phase.get("name") == phase_name:
                if phase.get("inputs_from") or phase.get("outputs"):
                    return phase
                if fallback is None:
                    fallback = phase
    return fallback


def _get_phase_param_routing(phase_name: str) -> Dict[str, Tuple]:
    """Return {param: (source_type, *path)} routing for a phase.

    Preference order:
      1. YAML `inputs_from:` block for this phase (REC-CTR-05).
      2. Legacy in-memory `_LEGACY_PHASE_PARAM_ROUTING` entry (warn once).
      3. Empty dict.
    """
    try:
        block = _phase_yaml_block(phase_name)
    except ValueError as e:
        logger.error("Failed to load workflows.yaml for phase routing: %s", e)
        block = None

    if block and block.get("inputs_from"):
        routing: Dict[str, Tuple] = {}
        for route in block["inputs_from"]:
            if not isinstance(route, dict):
                continue
            param = route.get("param")
            source = route.get("source")
            if not param or not source:
                continue
            if source == "workflow_params":
                routing[param] = ("workflow_params", route.get("key"))
            elif source == "phase_outputs":
                routing[param] = (
                    "phase_outputs",
                    route.get("phase"),
                    route.get("output"),
                )
            elif source == "literal":
                routing[param] = ("literal", route.get("value"))
        return routing

    # Fallback to legacy in-memory dict
    if phase_name in _LEGACY_PHASE_PARAM_ROUTING:
        if phase_name not in _FALLBACK_LOGGED:
            logger.warning(
                "Phase '%s' has no `inputs_from:` block in config/workflows.yaml; "
                "falling back to legacy in-memory routing. Annotate the phase to "
                "silence this warning.",
                phase_name,
            )
            _FALLBACK_LOGGED.add(phase_name)
        return _LEGACY_PHASE_PARAM_ROUTING[phase_name]

    return {}


def _get_phase_output_keys(phase_name: str) -> List[str]:
    """Return the list of output keys to extract from a phase's task results.

    Preference order:
      1. YAML `outputs:` block for this phase.
      2. Legacy in-memory `_LEGACY_PHASE_OUTPUT_KEYS` entry (warn once).
      3. Empty list.
    """
    try:
        block = _phase_yaml_block(phase_name)
    except ValueError as e:
        logger.error("Failed to load workflows.yaml for phase outputs: %s", e)
        block = None

    if block and block.get("outputs"):
        return list(block["outputs"])

    if phase_name in _LEGACY_PHASE_OUTPUT_KEYS:
        key = f"outputs:{phase_name}"
        if key not in _FALLBACK_LOGGED:
            logger.warning(
                "Phase '%s' has no `outputs:` block in config/workflows.yaml; "
                "falling back to legacy in-memory output keys.",
                phase_name,
            )
            _FALLBACK_LOGGED.add(key)
        return list(_LEGACY_PHASE_OUTPUT_KEYS[phase_name])

    return []


# Eager load + validate workflows.yaml at module import so typos surface
# before any workflow attempts to run. Tests that want a pristine config
# should call _reset_workflows_cache() after patching.
try:
    _load_workflows_config()
except ValueError as _e:
    # Log and re-raise so downstream imports see the error immediately.
    logger.error("workflows.yaml failed pre-flight validation: %s", _e)
    raise


# =============================================================================
# Wave 80 Worker A: --reuse-objectives helpers
# =============================================================================


def _normalize_to_courseforge_form(
    data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Normalize an objectives JSON into Courseforge synthesized form.

    Wave 80 Worker A. Accepts:

      * Courseforge synthesized form: ``terminal_objectives[]`` +
        ``chapter_objectives[]``. ``chapter_objectives`` may be a flat
        list of objective dicts OR the canonical
        ``[{"chapter": ..., "objectives": [...]}]`` group shape.
      * Wave 75 LibV2 archive form: ``terminal_outcomes[]`` +
        ``component_objectives[]`` (flat list with optional
        ``parent_terminal`` back-pointer).

    Returns a dict carrying:
      * ``terminal_objectives`` — list of terminal LO dicts (Courseforge
        shape: ``id``, ``statement``, etc.).
      * ``chapter_objectives`` — list of ``{"chapter": str,
        "objectives": [...]}`` groups (Courseforge shape).
      * ``course_name`` (best-effort, may be missing) and
        ``duration_weeks`` (best-effort, may be missing).

    Returns ``None`` when neither shape is present.
    """
    has_courseforge = (
        isinstance(data.get("terminal_objectives"), list)
        or isinstance(data.get("chapter_objectives"), list)
    )
    has_libv2 = (
        isinstance(data.get("terminal_outcomes"), list)
        or isinstance(data.get("component_objectives"), list)
    )
    if not (has_courseforge or has_libv2):
        return None

    if has_courseforge and not has_libv2:
        # Already in target form. Ensure chapter_objectives is in the
        # group shape ([{chapter, objectives}], not a flat list).
        terminal = list(data.get("terminal_objectives") or [])
        chapter_raw = list(data.get("chapter_objectives") or [])
        chapter_groups = _coerce_chapter_groups(chapter_raw)
        return {
            "terminal_objectives": terminal,
            "chapter_objectives": chapter_groups,
            "course_name": data.get("course_name"),
            "duration_weeks": data.get("duration_weeks"),
        }

    # LibV2 archive form (or mixed — we prefer libv2 keys when both
    # present, since the user explicitly handed us the archive shape).
    terminal_raw = list(data.get("terminal_outcomes") or [])
    components_raw = list(data.get("component_objectives") or [])

    # Map to Courseforge shape. LibV2 IDs are lowercase by default; we
    # preserve them verbatim — the LO ID regex accepts both cases.
    terminal_objectives: List[Dict[str, Any]] = []
    for to in terminal_raw:
        if not isinstance(to, dict) or "id" not in to:
            continue
        entry: Dict[str, Any] = {"id": to["id"]}
        for key in (
            "statement", "bloom_level", "bloom_verb",
            "cognitive_domain", "weeks",
        ):
            if to.get(key) is not None:
                entry[key] = to[key]
        terminal_objectives.append(entry)

    # Group component objectives by parent_terminal -> a chapter group
    # (one group per terminal). LibV2 stores the parent reverse-link as
    # ``parent_terminal``; Courseforge's content-generator only needs
    # the flat per-week shape, so we emit one group per CO with the
    # parent's id as the chapter label fallback.
    chapter_groups: List[Dict[str, Any]] = []
    for co in components_raw:
        if not isinstance(co, dict) or "id" not in co:
            continue
        obj: Dict[str, Any] = {"id": co["id"]}
        for key in (
            "statement", "bloom_level", "bloom_verb",
            "cognitive_domain", "week", "source_refs",
        ):
            if co.get(key) is not None:
                obj[key] = co[key]
        # Preserve the parent_terminal back-pointer so downstream
        # consumers (and our cross-validation below) can verify the
        # hierarchy.
        if co.get("parent_terminal"):
            obj["parent_terminal"] = co["parent_terminal"]
        # Emit as a per-CO group. Use ``Week N`` style label by index
        # to match _plan_course_structure's convention.
        chapter_groups.append({
            "chapter": f"Week {len(chapter_groups) + 1}",
            "objectives": [obj],
        })

    return {
        "terminal_objectives": terminal_objectives,
        "chapter_objectives": chapter_groups,
        "course_name": data.get("course_code") or data.get("course_name"),
        "duration_weeks": data.get("duration_weeks"),
    }


def _coerce_chapter_groups(
    chapter_raw: List[Any],
) -> List[Dict[str, Any]]:
    """Coerce a chapter_objectives list to the canonical group shape.

    Accepts the dual shapes already supported by
    ``_content_gen_helpers.load_objectives_json``:

      * Group shape: ``[{"chapter": str, "objectives": [...]}, ...]``.
      * Flat shape: ``[{"id": "co-01", ...}, ...]``.

    Always returns the group shape, one group per CO when the input
    was flat (so the hierarchy is preserved 1:1 without forcing a
    chapter assignment).
    """
    groups: List[Dict[str, Any]] = []
    flat_buffer: List[Dict[str, Any]] = []
    for entry in chapter_raw:
        if not isinstance(entry, dict):
            continue
        if "objectives" in entry and isinstance(entry["objectives"], list):
            groups.append({
                "chapter": entry.get("chapter") or f"Week {len(groups) + 1}",
                "objectives": list(entry["objectives"]),
            })
        else:
            flat_buffer.append(entry)
    # If we accumulated flat-shape entries (or the input had nothing
    # but flat entries), emit one group per CO so the hierarchy stays
    # 1:1 with the input.
    for flat in flat_buffer:
        groups.append({
            "chapter": f"Week {len(groups) + 1}",
            "objectives": [flat],
        })
    return groups


def _validate_reused_lo_coherence(
    terminal: List[Dict[str, Any]],
    chapter_flat: List[Dict[str, Any]],
) -> Optional[str]:
    """Return None on success; an error string on failure.

    Wave 80 Worker A. Cross-validates a reused objectives file:

    * Every LO ID matches ``^[a-zA-Z]{2,}-\\d{2,}$`` (mirrors
      ``schemas/knowledge/courseforge_jsonld_v1.schema.json`` and
      ``lib/ontology/learning_objectives.py::validate_lo_id``).
    * No duplicate IDs (across terminal + chapter combined).
    * Every CO ``parent_terminal`` (or ``parent_to``) reference, when
      present, resolves to an existing TO ID.
    """
    seen_ids: set = set()
    terminal_ids: set = set()
    for to in terminal:
        to_id = (to or {}).get("id")
        if not to_id:
            return "terminal entry missing 'id' field"
        if not _LO_ID_RE.match(str(to_id)):
            return (
                f"terminal id {to_id!r} does not match LO id regex "
                f"^[a-zA-Z]{{2,}}-\\d{{2,}}$"
            )
        if to_id in seen_ids:
            return f"duplicate LO id {to_id!r}"
        seen_ids.add(to_id)
        terminal_ids.add(to_id)
    for co in chapter_flat:
        co_id = (co or {}).get("id")
        if not co_id:
            return "chapter/component entry missing 'id' field"
        if not _LO_ID_RE.match(str(co_id)):
            return (
                f"chapter id {co_id!r} does not match LO id regex "
                f"^[a-zA-Z]{{2,}}-\\d{{2,}}$"
            )
        if co_id in seen_ids:
            return f"duplicate LO id {co_id!r}"
        seen_ids.add(co_id)
        # Hierarchy back-pointer (when present) must resolve.
        parent = co.get("parent_terminal") or co.get("parent_to")
        if parent and parent not in terminal_ids:
            return (
                f"chapter {co_id!r} parent_terminal={parent!r} "
                f"does not reference a known TO id "
                f"(known: {sorted(terminal_ids)})"
            )
    return None


def _warn_on_source_map_mismatch(
    source_map_path: str,
    terminal: List[Dict[str, Any]],
    chapter_flat: List[Dict[str, Any]],
) -> None:
    """Best-effort warning when the reused LOs miss ids referenced in
    the source_module_map. Pure logging — never raises.

    This catches the case where a user supplies an objectives file
    that's been heavily edited (e.g. removed half the COs) while the
    upstream source_module_map still references the original IDs.
    Downstream content_generation will then emit pages referencing
    objective ids that don't resolve.
    """
    try:
        path = Path(source_map_path)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return

    # Collect all LO ids referenced by the source map (best-effort —
    # the schema is a router output we don't want to over-couple to).
    referenced: set = set()
    if isinstance(data, dict):
        for entries in data.values():
            if isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict):
                        for key in ("objective_ids", "lo_ids", "objectives"):
                            val = e.get(key)
                            if isinstance(val, list):
                                for v in val:
                                    if isinstance(v, str) and _LO_ID_RE.match(v):
                                        referenced.add(v)
                            elif isinstance(val, str) and _LO_ID_RE.match(val):
                                referenced.add(val)

    if not referenced:
        return

    available = {str(t.get("id")) for t in terminal if t.get("id")}
    available |= {str(c.get("id")) for c in chapter_flat if c.get("id")}
    # Case-insensitive comparison since the LO id regex allows mixed.
    available_lc = {x.lower() for x in available}
    missing = [
        rid for rid in referenced
        if rid.lower() not in available_lc
    ]
    if missing:
        logger.warning(
            "reuse_objectives: source_module_map references %d LO id(s) "
            "absent from the supplied objectives file. content_generation "
            "may emit pages with unresolved objective references. "
            "missing=%s",
            len(missing),
            sorted(missing)[:10],
        )


class WorkflowRunner:
    """
    Executes a multi-phase workflow end-to-end with inter-phase data routing.

    Bridges the gap between the workflow YAML definitions and the
    TaskExecutor's phase-level execution. Handles:
    - Phase dependency ordering (topological sort via depends_on)
    - Task creation for each phase from config + routed params
    - Inter-phase output-to-input data routing
    - Optional phase skipping
    - Workflow state persistence for crash recovery
    """

    def __init__(self, executor: TaskExecutor, config: OrchestratorConfig):
        self.executor = executor
        self.config = config

    async def run_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute all phases of a workflow in dependency order.

        Args:
            workflow_id: ID of the workflow to execute

        Returns:
            Dict with workflow_id, status, phase_results, and phase_outputs
        """
        # Load workflow state
        workflow_path = STATE_PATH / "workflows" / f"{workflow_id}.json"
        if not workflow_path.exists():
            return {"error": f"Workflow not found: {workflow_id}"}

        with open(workflow_path) as f:
            workflow_state = json.load(f)

        workflow_type = workflow_state.get("type", "")
        workflow_params = workflow_state.get("params", {})
        if isinstance(workflow_params, str):
            workflow_params = json.loads(workflow_params)

        # Load workflow config from YAML
        wf_config = self.config.get_workflow(workflow_type)
        if not wf_config:
            return {"error": f"Unknown workflow type: {workflow_type}"}

        # Initialize phase outputs (may already exist from partial run)
        phase_outputs: Dict[str, Dict] = workflow_state.get("phase_outputs", {})

        # Wave 74 Session 3: honour --skip-dart by synthesising the
        # dart_conversion phase_output from an existing DART/output/
        # directory before the phase loop runs. Downstream phases
        # (staging, libv2_archival) then resolve their inputs_from
        # without dart_conversion actually executing.
        if workflow_params.get("skip_dart") and "dart_conversion" not in phase_outputs:
            synthesized = self._synthesize_dart_skip_output(workflow_params)
            if synthesized is not None:
                phase_outputs["dart_conversion"] = synthesized

        # Phase 5 Subtask 2: honour --outline / courseforge-* stage
        # subcommands by synthesising every upstream phase's
        # phase_output from the on-disk artifacts under the project
        # export + LibV2 course dir. The phase loop's _completed skip
        # check then short-circuits every upstream phase, so the
        # downstream target phase (typically content_generation_rewrite)
        # runs without re-dispatching the upstream chain.
        #
        # Resolution chain for the OUTLINE_DIR:
        #   1. Explicit ``outline_dir`` workflow param (Worker WA's
        #      forthcoming --outline CLI flag).
        #   2. ``courseforge_stage`` set => walk
        #      Courseforge/exports/PROJ-{COURSE_CODE}-* and pick the
        #      most-recently-modified project dir.
        #
        # Honours --force (Worker WA's ``force_rerun`` workflow param,
        # commit 96e1bde) by flipping _completed to False on every
        # synthesised entry so the phase loop re-runs them.
        outline_dir_resolved = self._resolve_outline_dir(workflow_params)
        if outline_dir_resolved is not None:
            try:
                outline_synth = self._synthesize_outline_output(
                    outline_dir_resolved
                )
            except Exception as e:  # noqa: BLE001 — defensive
                logger.error(
                    "outline reuse: synthesis raised %s; falling through",
                    e,
                )
                outline_synth = {}
            force_rerun = bool(workflow_params.get("force_rerun"))
            for phase_name, phase_out in outline_synth.items():
                if phase_name in phase_outputs:
                    continue
                if force_rerun:
                    phase_out = {**phase_out, "_completed": False}
                phase_outputs[phase_name] = phase_out

        # Update workflow status
        workflow_state["status"] = "RUNNING"
        workflow_state["started_at"] = datetime.now().isoformat()
        self._save_workflow_state(workflow_path, workflow_state)

        # Sort phases by dependency order
        sorted_phases = self._topological_sort(wf_config.phases)

        # Execute each phase
        all_results: Dict[str, Dict] = {}
        final_status = "COMPLETE"

        for phase_idx, phase in enumerate(sorted_phases):
            phase_name = phase.name

            # Skip already-completed phases (crash recovery)
            if phase_name in phase_outputs and phase_outputs[phase_name].get("_completed"):
                logger.info(f"Skipping already-completed phase: {phase_name}")
                continue

            # Check if this optional phase should be skipped
            if self._should_skip_phase(phase, workflow_params):
                logger.info(f"Skipping optional phase: {phase_name}")
                phase_outputs[phase_name] = {"_skipped": True, "_completed": True}
                workflow_state["phase_outputs"] = phase_outputs
                self._save_workflow_state(workflow_path, workflow_state)
                continue

            # Check that all dependencies completed
            if not self._dependencies_met(phase, phase_outputs):
                logger.error(
                    f"Phase {phase_name} dependencies not met: {phase.depends_on}"
                )
                final_status = "FAILED"
                break

            # Wave 80 Worker A: honour --reuse-objectives by synthesising
            # the course_planning phase_output from the user-supplied
            # objectives JSON instead of dispatching the course-outliner
            # subagent. Stable across re-runs (no LLM nondeterminism),
            # preserving chunk learning_outcome_refs continuity. We do
            # this just-in-time (inside the phase loop) rather than
            # pre-loop because the synthesised output needs project_id
            # from objective_extraction, which hasn't run pre-loop.
            if (
                phase_name == "course_planning"
                and workflow_params.get("reuse_objectives_path")
            ):
                synthesized_planning = (
                    self._synthesize_course_planning_reuse_output(
                        workflow_params, phase_outputs,
                    )
                )
                if synthesized_planning is not None:
                    logger.info(
                        "course_planning: reusing user-supplied objectives "
                        "from %s; skipping course-outliner dispatch",
                        workflow_params.get("reuse_objectives_path"),
                    )
                    phase_outputs[phase_name] = synthesized_planning
                    workflow_state["phase_outputs"] = phase_outputs
                    self._save_workflow_state(workflow_path, workflow_state)
                    all_results[phase_name] = {
                        "task_count": 0,
                        "completed": 0,
                        "failed": 0,
                        "gates_passed": True,
                    }
                    continue
                else:
                    # Synthesis failed (e.g. project dir not yet created
                    # or objectives file unreadable). Surface as a hard
                    # failure: the user explicitly opted in to reuse, so
                    # silently falling back to a fresh LO mint would
                    # defeat the purpose.
                    logger.error(
                        "course_planning: --reuse-objectives synthesis "
                        "failed; aborting workflow"
                    )
                    final_status = "FAILED"
                    break

            logger.info(f"Starting phase {phase_idx + 1}/{len(sorted_phases)}: {phase_name}")

            # Route parameters from workflow params + prior phase outputs
            routed_params = self._route_params(phase_name, workflow_params, phase_outputs)

            # Create tasks for this phase
            tasks = self._create_phase_tasks(
                workflow_id, phase, routed_params, workflow_params
            )

            # Add tasks to workflow state
            workflow_state.setdefault("tasks", []).extend(tasks)
            self._save_workflow_state(workflow_path, workflow_state)

            # Get validation gate configs from phase
            gate_configs = getattr(phase, "validation_gates", None)

            # Execute the phase.
            #
            # Wave 23 Sub-task A: thread accumulated phase_outputs +
            # workflow_params through to the executor so the per-gate
            # input router can build validator-specific inputs. Without
            # these, every gate received a generic artifacts blob and
            # silently failed / skipped.
            # Wave 33 Bug B: hand the executor a way to extract the
            # current phase's outputs BEFORE the gate router runs.
            # Pre-Wave-33 extraction happened here (post-execute_phase)
            # so gate builders never saw the current phase's keys and
            # six gates silently skipped with "missing inputs: *".
            results, gates_passed, gate_results = await self.executor.execute_phase(
                workflow_id=workflow_id,
                phase_name=phase_name,
                phase_index=phase_idx,
                tasks=tasks,
                gate_configs=gate_configs,
                max_concurrent=getattr(phase, "max_concurrent", 5),
                phase_outputs=phase_outputs,
                workflow_params=workflow_params,
                extract_phase_outputs_fn=self._extract_phase_outputs,
            )

            # Extract outputs from results
            extracted = self._extract_phase_outputs(phase_name, results)
            extracted["_completed"] = True
            extracted["_gates_passed"] = gates_passed
            phase_outputs[phase_name] = extracted

            # Phase 5 Subtask 4: write the operator-facing
            # ``02_validation_report/report.json`` aggregation after
            # the ``inter_tier_validation`` and
            # ``post_rewrite_validation`` phases complete. The shipped
            # ``_run_inter_tier_validation`` helper writes JSONL only
            # (``blocks_validated_path`` + ``blocks_failed_path``); the
            # operator-facing structured per-block summary is a Phase 5
            # deliverable. Best-effort — failure to write the report
            # does NOT abort the workflow (it's an aggregation; the
            # raw JSONL is the source of truth).
            if phase_name in ("inter_tier_validation", "post_rewrite_validation"):
                try:
                    self._write_validation_report(
                        workflow_id=workflow_id,
                        phase_name=phase_name,
                        phase_output=extracted,
                        gate_results_list=gate_results,
                    )
                except Exception as exc:  # noqa: BLE001 — defensive
                    logger.warning(
                        "Phase 5 validation_report writer failed for "
                        "%s (non-fatal): %s",
                        phase_name, exc,
                    )

            # Persist phase outputs
            workflow_state["phase_outputs"] = phase_outputs
            self._save_workflow_state(workflow_path, workflow_state)

            all_results[phase_name] = {
                "task_count": len(tasks),
                "completed": sum(1 for r in results.values() if r.status == "COMPLETE"),
                # Wave 33 Bug C: count "FAILED" alongside "ERROR" and
                # "TIMEOUT" so tool envelopes with ``success=False``
                # surface in the phase summary instead of being
                # silently counted as completed.
                "failed": sum(
                    1 for r in results.values()
                    if r.status in ("ERROR", "TIMEOUT", "FAILED")
                ),
                "gates_passed": gates_passed,
            }

            # Check if phase failed
            # Wave 33 Bug C: include "FAILED" status so phases that had
            # every task return ``success=False`` envelopes stop the
            # workflow instead of advancing with a stale "12/12
            # complete" count.
            phase_failed = any(
                r.status in ("ERROR", "TIMEOUT", "FAILED")
                for r in results.values()
            )
            if phase_failed and not getattr(phase, "optional", False):
                logger.error(f"Phase {phase_name} failed, stopping workflow")
                final_status = "FAILED"
                break

            if not gates_passed and not getattr(phase, "optional", False):
                logger.error(f"Phase {phase_name} failed validation gates, stopping workflow")
                final_status = "FAILED"
                break

        # Finalize workflow state
        workflow_state["status"] = final_status
        workflow_state["completed_at"] = datetime.now().isoformat()
        self._save_workflow_state(workflow_path, workflow_state)

        return {
            "workflow_id": workflow_id,
            "status": final_status,
            "phase_results": all_results,
            "phase_outputs": {
                k: {pk: pv for pk, pv in v.items() if not pk.startswith("_")}
                for k, v in phase_outputs.items()
            },
        }

    def _route_params(
        self,
        phase_name: str,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """
        Build task params for a phase by resolving the routing table.

        Args:
            phase_name: Name of the phase to build params for
            workflow_params: Original workflow creation params
            phase_outputs: Accumulated outputs from prior phases

        Returns:
            Dict of resolved parameter values
        """
        routing = _get_phase_param_routing(phase_name)
        params = {}

        for param_name, source_spec in routing.items():
            source_type = source_spec[0]

            if source_type == "workflow_params":
                key = source_spec[1]
                value = workflow_params.get(key)
                if value is not None:
                    # Handle list values that need comma-joining for tool params
                    if isinstance(value, list):
                        value = ",".join(str(v) for v in value)
                    params[param_name] = value

            elif source_type == "phase_outputs":
                phase_key = source_spec[1]
                output_key = source_spec[2]
                phase_data = phase_outputs.get(phase_key, {})
                value = phase_data.get(output_key)
                if value is not None:
                    if isinstance(value, list):
                        value = ",".join(str(v) for v in value)
                    params[param_name] = value

            elif source_type == "literal":
                params[param_name] = source_spec[1]

        return params

    def _create_phase_tasks(
        self,
        workflow_id: str,
        phase: WorkflowPhase,
        routed_params: Dict[str, Any],
        workflow_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create task dicts for a phase.

        Handles special cases:
        - dart_conversion: one task per PDF file
        - content_generation with batch_by=week: one task per week
        - Default: one task per agent in the phase

        Args:
            workflow_id: Parent workflow ID
            phase: Phase configuration
            routed_params: Parameters resolved from routing table
            workflow_params: Original workflow creation params

        Returns:
            List of task dicts ready for execute_phase()
        """
        tasks = []
        timestamp = datetime.now().strftime("%H%M%S")
        workflow_params = workflow_params or {}

        # Special case: dart_conversion creates one task per PDF
        if phase.name == "dart_conversion":
            pdf_paths = workflow_params.get("pdf_paths", [])
            if isinstance(pdf_paths, str):
                pdf_paths = [p.strip() for p in pdf_paths.split(",")]
            for i, pdf_path in enumerate(pdf_paths):
                task_id = f"T-{phase.name}-{i}-{timestamp}"
                task = {
                    "id": task_id,
                    "agent_type": phase.agents[0],
                    "phase": phase.name,
                    "status": "PENDING",
                    "params": {
                        "pdf_path": pdf_path,
                        "course_code": workflow_params.get("course_name", ""),
                    },
                    "created_at": datetime.now().isoformat(),
                    "dependencies": [],
                }
                tasks.append(task)
            return tasks

        # Special case: batch_by week creates one task per week
        if phase.batch_by == "week":
            duration = workflow_params.get("duration_weeks", 12)
            for week in range(1, duration + 1):
                task_id = f"T-{phase.name}-w{week}-{timestamp}"
                task = {
                    "id": task_id,
                    "agent_type": phase.agents[0],
                    "phase": phase.name,
                    "status": "PENDING",
                    "params": {
                        **routed_params,
                        "week_range": f"{week}-{week}",
                    },
                    "created_at": datetime.now().isoformat(),
                    "dependencies": [],
                }
                tasks.append(task)
            return tasks

        # Default: one task per agent
        for agent_name in phase.agents:
            task_id = f"T-{phase.name}-{agent_name}-{timestamp}"
            task = {
                "id": task_id,
                "agent_type": agent_name,
                "phase": phase.name,
                "status": "PENDING",
                "params": routed_params.copy(),
                "created_at": datetime.now().isoformat(),
                "dependencies": [],
            }
            tasks.append(task)

        # Phase 4 Subtask 1 — synthesize a single virtual task for
        # phases that declare ``agents: []`` but ARE registered in
        # ``_PHASE_TOOL_MAPPING`` (e.g. ``inter_tier_validation``,
        # ``post_rewrite_validation``, plus the two-pass
        # outline/rewrite phases when wired without an explicit
        # agent). Without this fallback, the per-agent loop above
        # yields zero tasks, ``execute_phase`` runs the validation
        # gate chain only, and the dedicated phase-handler
        # (``run_inter_tier_validation`` / ``run_post_rewrite_validation``
        # / etc.) never lands its blocks-validated-and-persist
        # work to disk. The executor's ``_PHASE_TOOL_MAPPING.get``
        # path keys off ``phase.name`` so the placeholder
        # ``agent_type="phase-handler"`` is intentional — the agent
        # name is irrelevant on this routing path.
        if not tasks and _PHASE_TOOL_MAPPING.get(phase.name):
            task_id = f"T-{phase.name}-phase-handler-{timestamp}"
            tasks.append({
                "id": task_id,
                "agent_type": "phase-handler",
                "phase": phase.name,
                "status": "PENDING",
                "params": routed_params.copy(),
                "created_at": datetime.now().isoformat(),
                "dependencies": [],
            })

        return tasks

    def _extract_phase_outputs(
        self,
        phase_name: str,
        results: Dict[str, ExecutionResult],
    ) -> Dict[str, Any]:
        """
        Extract key output values from phase results for downstream routing.

        Args:
            phase_name: Name of the completed phase
            results: Dict of task_id -> ExecutionResult

        Returns:
            Dict of extracted output values
        """
        output_keys = _get_phase_output_keys(phase_name)
        extracted = {}

        for result in results.values():
            if result.status != "COMPLETE":
                continue

            result_data = result.result
            if not isinstance(result_data, dict):
                continue

            for key in output_keys:
                if key in result_data and key not in extracted:
                    extracted[key] = result_data[key]

        # Special handling: collect multiple output_paths into output_paths list
        if phase_name == "dart_conversion":
            paths = []
            for result in results.values():
                if result.status == "COMPLETE" and isinstance(result.result, dict):
                    path = (
                        result.result.get("output_path")
                        or result.result.get("html_path")
                    )
                    if path:
                        paths.append(path)
            if paths:
                joined = ",".join(paths)
                extracted["output_paths"] = joined
                # Wave 32 Deliverable B: alias as html_paths (router
                # canonical key) so DartMarkersValidator gate builder
                # picks it up without a router change.
                extracted["html_paths"] = joined
                # And surface a single representative html_path for
                # validators that only accept the scalar form.
                extracted.setdefault("html_path", paths[0])

        return extracted

    def _should_skip_phase(
        self, phase: WorkflowPhase, workflow_params: Dict[str, Any]
    ) -> bool:
        """Check if an optional phase should be skipped based on workflow params.

        Wave 74 Session 3: dart_conversion's --skip-dart path is
        handled upstream by pre-populating ``phase_outputs`` in
        ``run_workflow`` before the loop runs. The already-completed
        guard then skips execution naturally, preserving the
        synthesised output dict (this method would have overwritten it
        with a bare ``{"_skipped": True, "_completed": True}``).

        Phase 3 Subtask 1: phases may carry an ``enabled_when_env``
        predicate (``"VAR=value"`` or ``"VAR!=value"``); when present,
        the predicate is evaluated against the live environment and
        the phase skips when unsatisfied. This gate runs BEFORE the
        legacy optional-phase logic so a non-optional phase can still
        skip via the env predicate (e.g. the legacy
        ``content_generation`` phase carries
        ``enabled_when_env: "COURSEFORGE_TWO_PASS!=true"`` to disable
        itself when the new two-pass router is engaged).

        Phase 5 Subtask 4: when ``workflow_params['courseforge_stage']``
        is set (CLI plumbed by ``cli/commands/run.py`` for the four
        Phase 5 ``courseforge-*`` subcommands), phases NOT in the
        active-phase whitelist for that stage are skipped. Whitelist
        per stage:

        * ``courseforge_outline``: ``[content_generation_outline]``.
          Validate + rewrite + post_rewrite_validation skip.
        * ``courseforge_validate``: ``[inter_tier_validation,
          post_rewrite_validation]``. The two read-only validator
          phases run; outline + rewrite skip.
        * ``courseforge_rewrite``: ``[content_generation_rewrite,
          post_rewrite_validation]``. Outline + inter_tier_validation
          skip (the rewrite tier consumes the synthesizer-reconstructed
          ``inter_tier_validation`` output from disk via
          ``_synthesize_outline_output``).
        * ``courseforge`` / ``full`` (or absent — falls through to
          existing behaviour): all four two-pass phases run; nothing
          skipped from the courseforge whitelist.

        Upstream phases (dart_conversion, staging, chunking,
        objective_extraction, source_mapping, concept_extraction,
        course_planning) are also skipped because the runner
        pre-populates them via ``_synthesize_outline_output`` before
        the phase loop runs (their ``_completed=True`` guard at
        ``run_workflow:897`` already short-circuits them; this gate
        catches the case where the synthesizer didn't fire — e.g.
        operator passed only ``--course-name`` without setting up a
        prior project export). Downstream phases (packaging,
        imscc_chunking, trainforge_assessment, training_synthesis,
        libv2_archival, finalization) skip because the Phase 5 stage
        subcommands are scoped to the Courseforge two-pass surface
        only — operators who want post-rewrite phases should run the
        full ``ed4all run textbook-to-course`` pipeline.
        """
        predicate = getattr(phase, "enabled_when_env", None)
        if predicate:
            if not self._eval_enabled_when_env(predicate):
                return True

        # Phase 5 Subtask 4: courseforge_stage whitelist gate. Runs
        # BEFORE the optional-phase early-return below so non-optional
        # phases (e.g. dart_conversion, packaging) can still be
        # skipped when the operator scoped a stage subcommand to a
        # subset of the Courseforge surface.
        stage = workflow_params.get("courseforge_stage")
        if stage and self._should_skip_for_courseforge_stage(phase.name, stage):
            return True

        if not getattr(phase, "optional", False):
            return False

        # Skip trainforge_assessment if generate_assessments is False
        if phase.name == "trainforge_assessment":
            return not workflow_params.get("generate_assessments", True)

        return False

    # Phase 5 Subtask 4: per-stage active-phase whitelist. Source of
    # truth for the four ``courseforge-*`` subcommand handlers in
    # ``cli/commands/run.py``. Stage names accept both hyphenated
    # (``courseforge-rewrite``) and underscored (``courseforge_rewrite``)
    # spellings — ``run.py::_normalize_workflow`` already collapses
    # hyphens to underscores before passing the stage through, but
    # we accept both here for defence-in-depth.
    _COURSEFORGE_STAGE_ACTIVE_PHASES: Dict[str, frozenset] = {
        "courseforge_outline": frozenset({"content_generation_outline"}),
        "courseforge_validate": frozenset({
            "inter_tier_validation",
            "post_rewrite_validation",
        }),
        "courseforge_rewrite": frozenset({
            "content_generation_rewrite",
            "post_rewrite_validation",
        }),
        "courseforge": frozenset({
            "content_generation_outline",
            "inter_tier_validation",
            "content_generation_rewrite",
            "post_rewrite_validation",
        }),
    }

    @classmethod
    def _resolve_courseforge_stage_active_phases(
        cls, stage: str
    ) -> Optional[frozenset]:
        """Resolve a courseforge_stage name to its active-phase whitelist.

        Returns ``None`` when ``stage`` is unrecognised so the caller
        treats that as "no whitelist applied" and falls through to
        normal phase-loop semantics.
        """
        if not stage:
            return None
        normalized = stage.replace("-", "_").strip().lower()
        return cls._COURSEFORGE_STAGE_ACTIVE_PHASES.get(normalized)

    def _should_skip_for_courseforge_stage(
        self, phase_name: str, stage: str
    ) -> bool:
        """Return True if ``phase_name`` is NOT in the stage whitelist.

        Phase 5 Subtask 4: phases outside the four-phase Courseforge
        two-pass surface (``content_generation_outline``,
        ``inter_tier_validation``, ``content_generation_rewrite``,
        ``post_rewrite_validation``) are ALSO skipped when a stage is
        active because Phase 5 stage subcommands are scoped to the
        Courseforge surface only — pre-Courseforge phases pre-populate
        via ``_synthesize_outline_output``, post-Courseforge phases
        belong to the full ``textbook_to_course`` workflow.
        """
        active = self._resolve_courseforge_stage_active_phases(stage)
        if active is None:
            # Unknown stage — don't skip on behalf of a typo.
            return False
        # Phases inside the two-pass surface but outside the stage's
        # whitelist => skip.
        two_pass_surface = self._COURSEFORGE_STAGE_ACTIVE_PHASES["courseforge"]
        if phase_name in two_pass_surface:
            return phase_name not in active
        # Phases outside the two-pass surface entirely — pre-Courseforge
        # (synthesized via _synthesize_outline_output) and
        # post-Courseforge (out-of-scope for stage subcommands) — skip.
        return True

    @staticmethod
    def _eval_enabled_when_env(predicate: str) -> bool:
        """Evaluate an ``enabled_when_env`` predicate against ``os.environ``.

        Grammar (Phase 3 Subtask 1):
            "<NAME>=<value>"   -> True when ``os.environ[NAME] == value`` (case-insensitive)
            "<NAME>!=<value>"  -> True when ``os.environ[NAME] != value`` (case-insensitive)

        The literal ``true`` matches any of ``1`` / ``true`` / ``yes`` /
        ``on`` (case-insensitive), mirroring
        ``Courseforge/scripts/blocks.py::_EMIT_BLOCKS_TRUTHY`` at ``:40``
        so the two-pass-router gate is consistent with the Phase 2
        emit-blocks gate.

        Malformed predicates (no operator, empty NAME, etc.) return
        ``True`` so a typo doesn't silently skip a phase — the
        predicate is treated as "enabled by default" and surfaces the
        bug at YAML-load review time instead.
        """
        if not predicate or not isinstance(predicate, str):
            return True

        truthy = {"1", "true", "yes", "on"}

        # Order matters: check ``!=`` before ``=`` so the longer
        # operator wins.
        if "!=" in predicate:
            name, _, value = predicate.partition("!=")
            negate = True
        elif "=" in predicate:
            name, _, value = predicate.partition("=")
            negate = False
        else:
            return True

        name = name.strip()
        value = value.strip()
        if not name:
            return True

        env_value = os.environ.get(name, "").strip().lower()
        target = value.lower()

        if target == "true":
            matched = env_value in truthy
        else:
            matched = env_value == target

        return (not matched) if negate else matched

    def _synthesize_dart_skip_output(
        self, workflow_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build a dart_conversion phase_output from existing DART HTMLs.

        Walks ``workflow_params['dart_output_dir']`` for
        ``*_accessible.html`` files and returns a dict mirroring what
        ``_extract_phase_outputs`` would have produced on a live run:
        ``output_path``, ``output_paths``, ``html_path``, ``html_paths``,
        plus the ``_completed``/``_skipped``/``_gates_passed`` markers
        the phase loop expects.

        When the corpus params include explicit ``pdf_paths``, we emit
        one entry per PDF in corpus order so downstream staging's
        ``{stem}_accessible.html`` lookup matches the PDF ordering. If a
        PDF has no matching HTML we skip it silently — the CLI already
        warned at --skip-dart validation time.
        """
        from pathlib import Path as _Path

        dart_dir_str = workflow_params.get("dart_output_dir") or "DART/output"
        dart_dir = _Path(dart_dir_str)
        if not dart_dir.is_absolute():
            dart_dir = (PROJECT_ROOT / dart_dir_str).resolve()
        if not dart_dir.is_dir():
            logger.error(
                "skip_dart set but dart_output_dir is not a directory: %s",
                dart_dir,
            )
            return None

        # Order htmls by corpus PDF order when available; fall back to
        # a stable sort over the directory listing.
        pdf_paths = workflow_params.get("pdf_paths") or []
        if isinstance(pdf_paths, str):
            pdf_paths = [p.strip() for p in pdf_paths.split(",") if p.strip()]
        ordered_htmls: List[_Path] = []
        if pdf_paths:
            for pdf in pdf_paths:
                stem = _Path(pdf).stem
                candidate = dart_dir / f"{stem}_accessible.html"
                if candidate.exists():
                    ordered_htmls.append(candidate)
        if not ordered_htmls:
            ordered_htmls = sorted(dart_dir.glob("*_accessible.html"))

        if not ordered_htmls:
            logger.error(
                "skip_dart set but no ``*_accessible.html`` files found in %s",
                dart_dir,
            )
            return None

        path_strs = [str(p) for p in ordered_htmls]
        joined = ",".join(path_strs)
        logger.info(
            "skip_dart: synthesised dart_conversion phase_output "
            "from %d HTML(s) in %s",
            len(path_strs),
            dart_dir,
        )
        return {
            "output_path": path_strs[0],
            "output_paths": joined,
            "html_path": path_strs[0],
            "html_paths": joined,
            "success": True,
            "html_length": sum(
                (p.stat().st_size if p.exists() else 0) for p in ordered_htmls
            ),
            "_completed": True,
            "_skipped": True,
            "_gates_passed": True,
            "_skip_reason": "skip_dart=True; reused existing DART HTMLs",
        }

    def _synthesize_course_planning_reuse_output(
        self,
        workflow_params: Dict[str, Any],
        phase_outputs: Dict[str, Dict],
    ) -> Optional[Dict[str, Any]]:
        """Build a ``course_planning`` phase_output from a reused LO file.

        Wave 80 Worker A. Loads the user-supplied objectives JSON
        (Courseforge synthesized form OR Wave 75 LibV2 archive form),
        normalizes to the Courseforge form expected by downstream
        consumers (content-generator, Trainforge CourseProcessor),
        cross-validates LO ID hierarchy / format / uniqueness, and
        writes the result to
        ``{project_path}/01_learning_objectives/synthesized_objectives.json``.

        Returns a dict mirroring what ``_extract_phase_outputs`` would
        have produced on a live run:

        * ``project_id`` — from the upstream ``objective_extraction``
          phase output (pre-conditions: that phase completed).
        * ``synthesized_objectives_path`` — absolute path to the file
          written into the project's
          ``01_learning_objectives/synthesized_objectives.json``.
        * ``objective_ids`` — comma-joined LO IDs (TO-NN + CO-NN).
        * ``terminal_count`` / ``chapter_count``.
        * ``_completed`` / ``_skipped`` / ``_gates_passed`` markers.

        Returns ``None`` (and logs at error level) when:

        * The reuse file is missing/unreadable/malformed (CLI already
          validates at parse time, but a race or manual workflow-state
          edit could still trip this).
        * The upstream ``objective_extraction`` phase did NOT produce a
          ``project_path`` / ``project_id`` we can resolve. Without a
          project to write into, the content-generator cannot pick up
          the objectives via ``project_config.json``.
        * Cross-validation fails (orphan parent_terminal references,
          malformed IDs, or duplicates).
        """
        from pathlib import Path as _Path

        reuse_path_str = workflow_params.get("reuse_objectives_path")
        if not reuse_path_str:
            return None

        reuse_path = _Path(reuse_path_str)
        if not reuse_path.is_file():
            logger.error(
                "reuse_objectives: file not found: %s", reuse_path,
            )
            return None
        try:
            raw = reuse_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError) as e:
            logger.error(
                "reuse_objectives: failed to parse %s: %s", reuse_path, e,
            )
            return None
        if not isinstance(data, dict):
            logger.error(
                "reuse_objectives: top-level JSON must be an object; "
                "got %s",
                type(data).__name__,
            )
            return None

        # Normalize into Courseforge synthesized form. Accept either
        # shape on input.
        normalized = _normalize_to_courseforge_form(data)
        if normalized is None:
            logger.error(
                "reuse_objectives: file does not match a recognised "
                "shape (Courseforge synthesized OR LibV2 archive). "
                "path=%s",
                reuse_path,
            )
            return None

        terminal: List[Dict[str, Any]] = normalized["terminal_objectives"]
        chapter_groups: List[Dict[str, Any]] = normalized["chapter_objectives"]

        if not terminal:
            logger.error(
                "reuse_objectives: zero terminal objectives in %s",
                reuse_path,
            )
            return None

        # Cross-validation. Flatten chapter groups for ID checks.
        chapter_flat: List[Dict[str, Any]] = []
        for group in chapter_groups:
            inner = group.get("objectives") or []
            for obj in inner:
                if isinstance(obj, dict):
                    chapter_flat.append(obj)

        validation_err = _validate_reused_lo_coherence(terminal, chapter_flat)
        if validation_err:
            logger.error(
                "reuse_objectives: cross-validation failed: %s",
                validation_err,
            )
            return None

        # Optional warning: compare against source_module_map if available.
        source_map_data = phase_outputs.get("source_mapping") or {}
        source_map_path = source_map_data.get("source_module_map_path")
        if source_map_path:
            _warn_on_source_map_mismatch(
                source_map_path, terminal, chapter_flat,
            )

        # Resolve project path / id from upstream objective_extraction.
        objective_extraction_out = phase_outputs.get(
            "objective_extraction"
        ) or {}
        project_id = objective_extraction_out.get("project_id")
        project_path_str = objective_extraction_out.get("project_path")
        if not project_path_str and project_id:
            project_path_str = str(
                PROJECT_ROOT / "Courseforge" / "exports" / project_id
            )
        if not project_path_str:
            logger.error(
                "reuse_objectives: cannot resolve project_path from "
                "upstream objective_extraction output. Did the phase "
                "complete? extracted=%s",
                objective_extraction_out,
            )
            return None

        project_path = _Path(project_path_str)
        if not project_path.is_dir():
            logger.error(
                "reuse_objectives: resolved project_path is not a "
                "directory: %s",
                project_path,
            )
            return None

        # Build the canonical synthesized JSON.
        course_name = (
            workflow_params.get("course_name")
            or normalized.get("course_name")
            or project_id
            or ""
        )
        duration_weeks = normalized.get("duration_weeks") or workflow_params.get(
            "duration_weeks",
        )

        lo_entries: List[Dict[str, Any]] = []
        for to in terminal:
            entry = dict(to)
            entry["hierarchy_level"] = "terminal"
            lo_entries.append(entry)
        for co in chapter_flat:
            entry = dict(co)
            entry["hierarchy_level"] = "chapter"
            lo_entries.append(entry)

        synthesized = {
            "course_name": course_name,
            "generated_from": str(reuse_path),
            "mint_method": "reuse_objectives",
            "duration_weeks": duration_weeks,
            "learning_outcomes": lo_entries,
            "terminal_objectives": [dict(t) for t in terminal],
            "chapter_objectives": chapter_groups,
            "synthesized_at": datetime.now().isoformat(),
        }

        # Write into the project directory. Use the canonical filename.
        objectives_out_dir = project_path / "01_learning_objectives"
        objectives_out_dir.mkdir(parents=True, exist_ok=True)
        objectives_out_path = (
            objectives_out_dir / "synthesized_objectives.json"
        )
        try:
            objectives_out_path.write_text(
                json.dumps(synthesized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(
                "reuse_objectives: failed to write %s: %s",
                objectives_out_path, e,
            )
            return None

        # Update project_config so downstream phases pick it up.
        config_path = project_path / "project_config.json"
        config_data: Dict[str, Any] = {}
        if config_path.exists():
            try:
                config_data = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                config_data = {}
        config_data["objectives_path"] = str(objectives_out_path)
        config_data["synthesized_objectives_path"] = str(objectives_out_path)
        config_data["course_name"] = course_name
        config_data["project_id"] = project_id or project_path.name
        config_data["status"] = "planned"
        if duration_weeks is not None:
            config_data["duration_weeks"] = duration_weeks
        try:
            config_path.write_text(
                json.dumps(config_data, indent=2), encoding="utf-8",
            )
        except OSError as e:
            logger.warning(
                "reuse_objectives: failed to update project_config.json "
                "(non-fatal): %s",
                e,
            )

        objective_ids = [str(e["id"]) for e in lo_entries if e.get("id")]
        joined_ids = ",".join(objective_ids)

        logger.info(
            "reuse_objectives: synthesised course_planning phase_output "
            "with %d terminal + %d chapter objectives from %s",
            len(terminal), len(chapter_flat), reuse_path,
        )

        return {
            "project_id": project_id or project_path.name,
            "synthesized_objectives_path": str(objectives_out_path),
            "objective_ids": joined_ids,
            "terminal_count": len(terminal),
            "chapter_count": len(chapter_flat),
            "_completed": True,
            "_skipped": True,
            "_gates_passed": True,
            "_skip_reason": (
                "reuse_objectives=True; reused user-supplied "
                "objectives JSON"
            ),
        }

    def _resolve_outline_dir(
        self, workflow_params: Dict[str, Any]
    ) -> Optional[Path]:
        """Resolve the OUTLINE_DIR for ``_synthesize_outline_output``.

        Phase 5 Subtask 2. Resolution chain:

        * ``workflow_params["outline_dir"]`` — explicit operator-supplied
          project export path (Worker WA's forthcoming --outline flag).
        * ``workflow_params["courseforge_stage"]`` set (commit 96e1bde) =>
          walk ``Courseforge/exports/PROJ-{COURSE_NAME}-*`` and pick
          the most-recently-modified candidate.

        Returns ``None`` when neither route resolves a directory; the
        caller treats that as "not a stage subcommand run, fall
        through to normal full-pipeline execution."
        """
        explicit = workflow_params.get("outline_dir")
        if explicit:
            cand = Path(explicit)
            if cand.is_dir():
                return cand
            logger.warning(
                "outline reuse: outline_dir param=%r not a directory; "
                "falling through to courseforge_stage resolution",
                explicit,
            )

        stage = workflow_params.get("courseforge_stage")
        if not stage:
            return None
        course_name = workflow_params.get("course_name") or ""
        if not course_name:
            logger.warning(
                "outline reuse: courseforge_stage=%r set but course_name "
                "is empty; cannot resolve project dir",
                stage,
            )
            return None
        exports_root = PROJECT_ROOT / "Courseforge" / "exports"
        if not exports_root.is_dir():
            logger.warning(
                "outline reuse: %s not a directory; no project to "
                "resume from",
                exports_root,
            )
            return None
        prefix = f"PROJ-{course_name}-"
        candidates: List[Tuple[float, Path]] = []
        for cand in exports_root.iterdir():
            if not cand.is_dir():
                continue
            if not cand.name.startswith(prefix):
                continue
            candidates.append((cand.stat().st_mtime, cand))
        if not candidates:
            logger.warning(
                "outline reuse: no project dir under %s matches "
                "course_name=%r (prefix=%r)",
                exports_root, course_name, prefix,
            )
            return None
        candidates.sort(reverse=True)
        resolved = candidates[0][1]
        logger.info(
            "outline reuse: resolved courseforge_stage=%r project to %s "
            "(most recent of %d candidates)",
            stage, resolved, len(candidates),
        )
        return resolved

    def _synthesize_outline_output(
        self,
        outline_dir: Path,
        target_phases: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Reconstruct phase_outputs for upstream phases from disk.

        Phase 5 Subtask 2. When an operator runs ``ed4all run
        courseforge-rewrite`` (or any of the new courseforge-* stage
        subcommands), the upstream phases — dart_conversion, staging,
        chunking, objective_extraction, source_mapping,
        concept_extraction, course_planning,
        content_generation_outline, inter_tier_validation — must already
        have run; their output artifacts live under the project export
        directory + the course's LibV2 directory. This synthesizer walks
        those locations and reconstructs the per-phase ``phase_outputs``
        dicts (matching the keys ``inputs_from`` references in
        ``config/workflows.yaml``) so the workflow runner's ``_completed``
        skip check at line 860 fires for every upstream phase. The
        rewrite tier (or any single-tier phase that depends on these
        upstream outputs) then runs without re-dispatching the upstream
        phases.

        ``outline_dir`` accepts either:

        * The Courseforge project export root, e.g.
          ``Courseforge/exports/PROJ-PHYS_101-20260502/``.
        * The ``01_outline/`` subdirectory inside that project, e.g.
          ``Courseforge/exports/PROJ-PHYS_101-20260502/01_outline``.

        In either case we resolve to the project_path. ``project_config.json``
        at the project root supplies course_name + staging_dir.

        Returns a dict keyed by phase_name; each value is a phase_outputs
        dict carrying ``_completed: True`` plus the canonical output
        keys that ``inputs_from`` for downstream phases pulls. When an
        upstream artifact is absent / unreadable, that phase is omitted
        from the returned dict (warning-logged) so the workflow runner's
        ``_dependencies_met`` check at line 1643 surfaces the gap as a
        normal dependency failure rather than a silent inconsistency.

        Recognized phase names (plan §5):

        * ``dart_conversion`` — synthesises ``output_paths`` from the
          staging manifest's HTML inputs (each staged ``*_accessible.html``
          maps back to a DART output).
        * ``staging`` — ``staging_dir`` from project_config.json.
        * ``chunking`` — reads ``LibV2/courses/<slug>/dart_chunks/
          manifest.json`` for ``dart_chunks_sha256`` + ``chunks.jsonl``
          path.
        * ``objective_extraction`` — reads
          ``<project>/01_learning_objectives/textbook_structure.json``.
        * ``source_mapping`` — reads ``<project>/source_module_map.json``.
        * ``concept_extraction`` — reads
          ``LibV2/courses/<slug>/concept_graph/manifest.json``.
        * ``course_planning`` — reads
          ``<project>/01_learning_objectives/synthesized_objectives.json``.
        * ``content_generation_outline`` — reads
          ``<project>/01_outline/blocks_outline.jsonl``.
        * ``inter_tier_validation`` — reads
          ``<project>/01_outline/blocks_validated.jsonl`` (+
          ``blocks_failed.jsonl``).

        ``target_phases`` filters which upstream phases to reconstruct;
        defaults to the full canonical list above. Unknown names are
        silently dropped (not an error).
        """
        from pathlib import Path as _Path

        canonical_phases = [
            "dart_conversion",
            "staging",
            "chunking",
            "objective_extraction",
            "source_mapping",
            "concept_extraction",
            "course_planning",
            "content_generation_outline",
            "inter_tier_validation",
        ]
        if target_phases is None:
            phases = list(canonical_phases)
        else:
            phases = [p for p in target_phases if p in canonical_phases]

        outline_dir = _Path(outline_dir)
        if outline_dir.name == "01_outline":
            project_path = outline_dir.parent
        else:
            project_path = outline_dir

        if not project_path.is_dir():
            logger.error(
                "outline reuse: project_path is not a directory: %s",
                project_path,
            )
            return {}

        # Load project_config.json — supplies course_name +
        # staging_dir + project_id.
        config_path = project_path / "project_config.json"
        config_data: Dict[str, Any] = {}
        if config_path.exists():
            try:
                config_data = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as e:
                logger.warning(
                    "outline reuse: project_config.json unreadable at %s: %s",
                    config_path, e,
                )

        course_name = (
            config_data.get("course_name")
            or project_path.name.split("-")[1] if "-" in project_path.name else ""
        )
        project_id = config_data.get("project_id") or project_path.name
        course_slug = (
            (course_name or "").lower().replace("_", "-").replace(" ", "-")
        )
        libv2_course_dir = (
            PROJECT_ROOT / "LibV2" / "courses" / course_slug
            if course_slug else None
        )

        synthesized: Dict[str, Dict[str, Any]] = {}

        # ----- staging -----------------------------------------------
        if "staging" in phases:
            staging_dir_str = config_data.get("staging_dir")
            staging_dir: Optional[Path] = None
            if staging_dir_str:
                cand = _Path(staging_dir_str)
                if cand.is_dir():
                    staging_dir = cand
            # Fall back: walk Courseforge/inputs/textbooks/ for the
            # most-recent staging dir whose manifest carries
            # course_name == course_name.
            if staging_dir is None:
                inputs_root = (
                    PROJECT_ROOT / "Courseforge" / "inputs" / "textbooks"
                )
                if inputs_root.is_dir() and course_name:
                    candidates = []
                    for cand in inputs_root.iterdir():
                        if not cand.is_dir():
                            continue
                        manifest = cand / "staging_manifest.json"
                        if not manifest.exists():
                            continue
                        try:
                            mdata = json.loads(
                                manifest.read_text(encoding="utf-8")
                            )
                        except (OSError, ValueError):
                            continue
                        if mdata.get("course_name") == course_name:
                            candidates.append((manifest.stat().st_mtime, cand))
                    if candidates:
                        candidates.sort(reverse=True)
                        staging_dir = candidates[0][1]

            if staging_dir is not None and staging_dir.is_dir():
                staged_files = sorted(
                    str(p) for p in staging_dir.glob("*.html")
                )
                synthesized["staging"] = {
                    "staging_dir": str(staging_dir),
                    "staged_files": staged_files,
                    "file_count": len(staged_files),
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": "outline reuse: staging_dir from project_config",
                }
            else:
                logger.warning(
                    "outline reuse: staging_dir not found for project %s "
                    "(config_data.staging_dir=%r); skipping staging "
                    "phase pre-population",
                    project_id, staging_dir_str,
                )

        # Resolve dart_html_paths from staging if available.
        # ----- dart_conversion ---------------------------------------
        if "dart_conversion" in phases and "staging" in synthesized:
            staged_files = synthesized["staging"].get("staged_files") or []
            if staged_files:
                synthesized["dart_conversion"] = {
                    "output_path": staged_files[0],
                    "output_paths": ",".join(staged_files),
                    "html_path": staged_files[0],
                    "html_paths": ",".join(staged_files),
                    "success": True,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: derived from staging manifest"
                    ),
                }

        # ----- chunking ----------------------------------------------
        if "chunking" in phases and libv2_course_dir is not None:
            chunks_dir = libv2_course_dir / "dart_chunks"
            chunks_path = chunks_dir / "chunks.jsonl"
            manifest_path = chunks_dir / "manifest.json"
            if chunks_path.exists() and manifest_path.exists():
                try:
                    cmanifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    sha256 = cmanifest.get("chunks_sha256") or ""
                    synthesized["chunking"] = {
                        "dart_chunks_path": str(chunks_path),
                        "dart_chunks_sha256": sha256,
                        "manifest_path": str(manifest_path),
                        "course_slug": course_slug,
                        "chunks_count": cmanifest.get("chunks_count", 0),
                        "_completed": True,
                        "_skipped": True,
                        "_gates_passed": True,
                        "_skip_reason": (
                            "outline reuse: read dart_chunks/manifest.json"
                        ),
                    }
                except (OSError, ValueError) as e:
                    logger.warning(
                        "outline reuse: chunking manifest unreadable at "
                        "%s: %s",
                        manifest_path, e,
                    )
            else:
                logger.warning(
                    "outline reuse: chunking artifacts missing under "
                    "%s; skipping chunking phase pre-population",
                    chunks_dir,
                )

        # ----- objective_extraction ----------------------------------
        if "objective_extraction" in phases:
            structure_path = (
                project_path / "01_learning_objectives"
                / "textbook_structure.json"
            )
            if structure_path.exists():
                chapter_count = 0
                duration_weeks = config_data.get("duration_weeks")
                try:
                    structure_data = json.loads(
                        structure_path.read_text(encoding="utf-8")
                    )
                    chapter_count = len(
                        structure_data.get("chapters") or []
                    )
                    if duration_weeks is None:
                        duration_weeks = structure_data.get("duration_weeks")
                except (OSError, ValueError) as e:
                    logger.warning(
                        "outline reuse: textbook_structure.json "
                        "unreadable: %s",
                        e,
                    )
                synthesized["objective_extraction"] = {
                    "project_id": project_id,
                    "project_path": str(project_path),
                    "textbook_structure_path": str(structure_path),
                    "chapter_count": chapter_count,
                    "duration_weeks": duration_weeks,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read textbook_structure.json"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: textbook_structure.json missing at "
                    "%s; skipping objective_extraction pre-population",
                    structure_path,
                )

        # ----- source_mapping ----------------------------------------
        if "source_mapping" in phases:
            map_path = project_path / "source_module_map.json"
            if map_path.exists():
                source_chunk_ids: List[str] = []
                try:
                    map_data = json.loads(
                        map_path.read_text(encoding="utf-8")
                    )
                    if isinstance(map_data, dict):
                        for week_entries in map_data.values():
                            if not isinstance(week_entries, list):
                                continue
                            for entry in week_entries:
                                if isinstance(entry, dict):
                                    cid = entry.get("chunk_id")
                                    if cid:
                                        source_chunk_ids.append(str(cid))
                except (OSError, ValueError) as e:
                    logger.warning(
                        "outline reuse: source_module_map.json "
                        "unreadable: %s",
                        e,
                    )
                staging_dir_str = (
                    synthesized.get("staging", {}).get("staging_dir") or ""
                )
                synthesized["source_mapping"] = {
                    "source_module_map_path": str(map_path),
                    "source_chunk_ids": sorted(set(source_chunk_ids)),
                    "staging_dir": staging_dir_str,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read source_module_map.json"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: source_module_map.json missing at "
                    "%s; skipping source_mapping pre-population",
                    map_path,
                )

        # ----- concept_extraction ------------------------------------
        if "concept_extraction" in phases and libv2_course_dir is not None:
            graph_dir = libv2_course_dir / "concept_graph"
            graph_path = graph_dir / "concept_graph_semantic.json"
            cmanifest_path = graph_dir / "manifest.json"
            if graph_path.exists():
                sha256 = ""
                if cmanifest_path.exists():
                    try:
                        cmanifest = json.loads(
                            cmanifest_path.read_text(encoding="utf-8")
                        )
                        sha256 = cmanifest.get("concept_graph_sha256") or ""
                    except (OSError, ValueError):
                        sha256 = ""
                synthesized["concept_extraction"] = {
                    "concept_graph_path": str(graph_path),
                    "concept_graph_sha256": sha256,
                    "course_slug": course_slug,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read concept_graph_semantic.json"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: concept_graph_semantic.json missing "
                    "at %s; skipping concept_extraction pre-population",
                    graph_path,
                )

        # ----- course_planning ---------------------------------------
        if "course_planning" in phases:
            objectives_path = (
                project_path / "01_learning_objectives"
                / "synthesized_objectives.json"
            )
            if objectives_path.exists():
                terminal_count = 0
                chapter_count = 0
                objective_ids: List[str] = []
                try:
                    odata = json.loads(
                        objectives_path.read_text(encoding="utf-8")
                    )
                    terminal = odata.get("terminal_objectives") or []
                    chapter_groups = odata.get("chapter_objectives") or []
                    terminal_count = len(terminal)
                    for to in terminal:
                        if isinstance(to, dict) and to.get("id"):
                            objective_ids.append(str(to["id"]))
                    for group in chapter_groups:
                        if not isinstance(group, dict):
                            continue
                        inner = group.get("objectives") or []
                        for co in inner:
                            if isinstance(co, dict) and co.get("id"):
                                objective_ids.append(str(co["id"]))
                                chapter_count += 1
                except (OSError, ValueError) as e:
                    logger.warning(
                        "outline reuse: synthesized_objectives.json "
                        "unreadable: %s",
                        e,
                    )
                synthesized["course_planning"] = {
                    "project_id": project_id,
                    "synthesized_objectives_path": str(objectives_path),
                    "objective_ids": ",".join(objective_ids),
                    "terminal_count": terminal_count,
                    "chapter_count": chapter_count,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read synthesized_objectives.json"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: synthesized_objectives.json missing "
                    "at %s; skipping course_planning pre-population",
                    objectives_path,
                )

        # ----- content_generation_outline ----------------------------
        outline_subdir = project_path / "01_outline"
        if "content_generation_outline" in phases:
            blocks_outline_path = outline_subdir / "blocks_outline.jsonl"
            if blocks_outline_path.exists():
                # Count weeks via a one-pass scan of the JSONL.
                weeks_seen: set = set()
                block_count = 0
                try:
                    with blocks_outline_path.open(
                        "r", encoding="utf-8"
                    ) as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            block_count += 1
                            try:
                                entry = json.loads(line)
                            except ValueError:
                                continue
                            wk = entry.get("week")
                            if wk is not None:
                                weeks_seen.add(wk)
                except OSError as e:
                    logger.warning(
                        "outline reuse: blocks_outline.jsonl unreadable: %s",
                        e,
                    )
                synthesized["content_generation_outline"] = {
                    "blocks_outline_path": str(blocks_outline_path),
                    "project_id": project_id,
                    "weeks_prepared": len(weeks_seen),
                    "block_count": block_count,
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read blocks_outline.jsonl"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: blocks_outline.jsonl missing at %s; "
                    "skipping content_generation_outline pre-population",
                    blocks_outline_path,
                )

        # ----- inter_tier_validation ---------------------------------
        if "inter_tier_validation" in phases:
            validated_path = outline_subdir / "blocks_validated.jsonl"
            failed_path = outline_subdir / "blocks_failed.jsonl"
            if validated_path.exists():
                synthesized["inter_tier_validation"] = {
                    "blocks_validated_path": str(validated_path),
                    "blocks_failed_path": str(failed_path)
                    if failed_path.exists()
                    else "",
                    "_completed": True,
                    "_skipped": True,
                    "_gates_passed": True,
                    "_skip_reason": (
                        "outline reuse: read blocks_validated.jsonl"
                    ),
                }
            else:
                logger.warning(
                    "outline reuse: blocks_validated.jsonl missing at "
                    "%s; skipping inter_tier_validation pre-population",
                    validated_path,
                )

        logger.info(
            "outline reuse: synthesised phase_outputs for %d phase(s) "
            "from %s: %s",
            len(synthesized), project_path, sorted(synthesized.keys()),
        )
        return synthesized

    # Phase 5 Subtask 4: validation-report writer schema version. Bumped
    # alongside any breaking change to the per-block summary shape;
    # consumers (operator-facing dashboards, dry-run preview tooling,
    # the Phase 6 ABCD concept-extractor's validator surface) should
    # gate on this field when reading the report.
    _VALIDATION_REPORT_SCHEMA_VERSION = "v1"

    def _write_validation_report(
        self,
        *,
        workflow_id: str,
        phase_name: str,
        phase_output: Dict[str, Any],
        gate_results_list: Optional[List[Dict[str, Any]]],
    ) -> Optional[Path]:
        """Aggregate inter-tier / post-rewrite gate results into ``report.json``.

        Phase 5 Subtask 4. The shipped phase helpers
        (``_run_inter_tier_validation``,
        ``_run_post_rewrite_validation``) emit JSONL only —
        ``blocks_validated.jsonl`` + ``blocks_failed.jsonl`` next to the
        consumed blocks file. The operator-facing structured summary
        (passed / failed / escalated counts plus a ``per_block`` array
        keyed by ``block_id``) is a Phase 5 deliverable that lives at:

        * ``{project_root}/02_validation_report/report.json`` for
          ``inter_tier_validation``.
        * ``{project_root}/04_rewrite/02_validation_report/report.json``
          for ``post_rewrite_validation``.

        Where ``project_root`` is derived from the
        ``blocks_validated_path`` extracted output (which lives at
        ``{project_root}/01_outline/blocks_validated.jsonl`` for the
        outline-tier inter_tier_validation phase, and at
        ``{project_root}/04_rewrite/blocks_validated.jsonl`` for the
        rewrite-tier post_rewrite_validation phase — matching how the
        rewrite tier writes its blocks JSONL into ``04_rewrite/``).

        Returns the report path on successful write, or ``None`` when
        the report could not be written (no ``blocks_validated_path``
        in the phase output, or filesystem error).

        Schema (matches plan §6 ``report.json``):

        ::

            {
              "run_id": "<workflow_id>",
              "phase": "<phase_name>",
              "schema_version": "v1",
              "total_blocks": <int>,
              "passed": <int>,
              "failed": <int>,
              "escalated": <int>,
              "per_block": [
                {
                  "block_id": "<id>",
                  "block_type": "<type>",
                  "page": "<page_id|null>",
                  "week": <int|null>,
                  "status": "passed|failed|escalated",
                  "gate_results": [...],
                  "escalation_marker": "<marker|null>"
                },
                ...
              ]
            }
        """
        validated_path_raw = (phase_output or {}).get(
            "blocks_validated_path"
        )
        if not validated_path_raw:
            logger.debug(
                "Phase 5 validation_report: no blocks_validated_path "
                "in %s phase_output; nothing to aggregate",
                phase_name,
            )
            return None

        validated_path = Path(validated_path_raw)
        # Project root is two levels up from blocks_validated.jsonl:
        # ``<project_root>/<stage_dir>/blocks_validated.jsonl``.
        if not validated_path.is_absolute():
            validated_path = Path(validated_path)

        # The blocks JSONL lives in either ``01_outline/`` (outline-tier
        # inter_tier_validation) or ``04_rewrite/`` (rewrite-tier
        # post_rewrite_validation). The report dir is sibling to the
        # blocks file's stage dir for inter_tier_validation, and lives
        # INSIDE the stage dir for post_rewrite_validation per plan §6
        # ("rewrite writes its own equivalent under
        # 04_rewrite/02_validation_report/report.json").
        stage_dir = validated_path.parent
        if phase_name == "inter_tier_validation":
            report_dir = stage_dir.parent / "02_validation_report"
        else:  # post_rewrite_validation
            report_dir = stage_dir / "02_validation_report"

        try:
            report_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Phase 5 validation_report: cannot create %s: %s",
                report_dir, exc,
            )
            return None

        # Load the validated + failed blocks JSONL to build per-block
        # records. Failed blocks set status='failed'; blocks with
        # ``escalation_marker`` set are reclassified as 'escalated'.
        validated_blocks: List[Dict[str, Any]] = []
        failed_blocks: List[Dict[str, Any]] = []

        if validated_path.exists():
            try:
                for line in validated_path.read_text(
                    encoding="utf-8"
                ).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        validated_blocks.append(json.loads(line))
                    except ValueError:
                        continue
            except OSError as exc:
                logger.warning(
                    "Phase 5 validation_report: blocks_validated.jsonl "
                    "unreadable at %s: %s",
                    validated_path, exc,
                )

        failed_path_raw = (phase_output or {}).get("blocks_failed_path")
        if failed_path_raw:
            failed_path = Path(failed_path_raw)
            if failed_path.exists():
                try:
                    for line in failed_path.read_text(
                        encoding="utf-8"
                    ).splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            failed_blocks.append(json.loads(line))
                        except ValueError:
                            continue
                except OSError as exc:
                    logger.warning(
                        "Phase 5 validation_report: blocks_failed.jsonl "
                        "unreadable at %s: %s",
                        failed_path, exc,
                    )

        # Aggregate counts. Escalated == failed-with-non-null
        # escalation_marker (plan §3 escalated_only path); plain failed
        # blocks have ``escalation_marker is None`` or absent.
        per_block: List[Dict[str, Any]] = []
        passed_count = 0
        failed_count = 0
        escalated_count = 0

        # gate_results_list is the executor's emit; we attach the
        # full chain to every block's ``gate_results`` in the report so
        # the operator can introspect each gate's per-block findings
        # without re-running the validators. Down-shape any
        # ``GateResult.to_dict()`` payloads to a stable shape per plan
        # §6 (gate_id / action / passed / issues).
        gate_chain_summary: List[Dict[str, Any]] = []
        for gr in gate_results_list or []:
            if not isinstance(gr, dict):
                continue
            gate_chain_summary.append({
                "gate_id": gr.get("gate_id"),
                "action": gr.get("action"),
                "passed": gr.get("passed"),
                "issue_count": len(gr.get("issues") or []),
            })

        def _record_block(entry: Dict[str, Any], status: str) -> None:
            nonlocal passed_count, failed_count, escalated_count
            esc = entry.get("escalation_marker")
            if status == "failed" and esc:
                status = "escalated"
            if status == "passed":
                passed_count += 1
            elif status == "escalated":
                escalated_count += 1
            else:
                failed_count += 1
            per_block.append({
                "block_id": entry.get("block_id"),
                "block_type": entry.get("block_type"),
                "page": entry.get("page_id"),
                "week": entry.get("week"),
                "status": status,
                "gate_results": gate_chain_summary,
                "escalation_marker": esc,
            })

        for entry in validated_blocks:
            _record_block(entry, "passed")
        for entry in failed_blocks:
            _record_block(entry, "failed")

        report = {
            "run_id": workflow_id,
            "phase": phase_name,
            "schema_version": self._VALIDATION_REPORT_SCHEMA_VERSION,
            "total_blocks": passed_count + failed_count + escalated_count,
            "passed": passed_count,
            "failed": failed_count,
            "escalated": escalated_count,
            "per_block": per_block,
        }

        report_path = report_dir / "report.json"
        try:
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Phase 5 validation_report: cannot write %s: %s",
                report_path, exc,
            )
            return None

        logger.info(
            "Phase 5 validation_report: wrote %s "
            "(total=%d passed=%d failed=%d escalated=%d)",
            report_path, report["total_blocks"], passed_count,
            failed_count, escalated_count,
        )
        return report_path

    def _dependencies_met(
        self, phase: WorkflowPhase, phase_outputs: Dict[str, Dict]
    ) -> bool:
        """Check that all phase dependencies have completed.

        Phase 3 Subtask 4: when a phase declares
        ``depends_on_when_env`` paired with
        ``depends_on_when_env_value`` and the predicate is satisfied,
        the alt list replaces ``depends_on`` for this check. Used by
        ``course_generation::packaging`` to switch from depending on
        the legacy ``content_generation`` to the rewrite tier
        ``content_generation_rewrite`` when ``COURSEFORGE_TWO_PASS=true``.
        """
        deps = self._effective_depends_on(phase)
        for dep in deps:
            dep_output = phase_outputs.get(dep, {})
            if not dep_output.get("_completed"):
                return False
        return True

    def _effective_depends_on(self, phase: WorkflowPhase) -> List[str]:
        """Resolve a phase's effective ``depends_on`` for the current env.

        Mirrors the env-aware switch in ``_dependencies_met``: when a
        phase declares ``depends_on_when_env`` paired with
        ``depends_on_when_env_value`` and the predicate is satisfied
        against the live environment, the alt list replaces the static
        ``depends_on``. Used by ``_topological_sort`` so the dispatch
        order matches the dependency check (Phase 3.5: packaging
        switches from depending on the legacy ``content_generation`` to
        the rewrite-tier ``post_rewrite_validation`` when
        ``COURSEFORGE_TWO_PASS=true``).
        """
        alt_pred = getattr(phase, "depends_on_when_env", None)
        alt_value = getattr(phase, "depends_on_when_env_value", None)
        if alt_pred and alt_value and self._eval_enabled_when_env(alt_pred):
            return list(alt_value)
        return list(phase.depends_on or [])

    def _topological_sort(self, phases: List[WorkflowPhase]) -> List[WorkflowPhase]:
        """
        Sort phases respecting depends_on ordering.

        Uses Kahn's algorithm for topological sort. Honors the same
        env-aware ``depends_on_when_env`` switch as ``_dependencies_met``
        so the queue order matches the dependency check at runtime.
        """
        phase_map = {p.name: p for p in phases}
        effective_deps = {p.name: self._effective_depends_on(p) for p in phases}
        in_degree = {p.name: 0 for p in phases}

        for name, deps in effective_deps.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_names = []

        while queue:
            # Pick the first available (stable sort)
            name = queue.pop(0)
            sorted_names.append(name)

            for other_name, deps in effective_deps.items():
                if name in deps:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0:
                        queue.append(other_name)

        # Detect circular dependencies
        if len(sorted_names) < len(phases):
            unresolved = {p.name for p in phases} - set(sorted_names)
            logger.error(f"Circular dependencies detected in phases: {unresolved}")
            raise ValueError(f"Circular dependencies detected: {unresolved}")

        return [phase_map[name] for name in sorted_names if name in phase_map]

    def _save_workflow_state(self, path: Path, state: Dict[str, Any]) -> None:
        """Persist workflow state to disk."""
        state["updated_at"] = datetime.now().isoformat()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to save workflow state: {e}")
