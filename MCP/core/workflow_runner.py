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
from .executor import ExecutionResult, TaskExecutor

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
    "packaging": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
    },
    "trainforge_assessment": {
        "course_id": ("workflow_params", "course_name"),
        "imscc_path": ("phase_outputs", "packaging", "package_path"),
        "bloom_levels": ("workflow_params", "bloom_levels"),
        "question_count": ("workflow_params", "assessment_count"),
        # Wave 24: real TO/CO objective_ids come from course_planning
        # (was objective_extraction with phantom {COURSE}_OBJ_N IDs).
        "objective_ids": ("phase_outputs", "course_planning", "objective_ids"),
    },
    "libv2_archival": {
        "course_name": ("workflow_params", "course_name"),
        "domain": ("workflow_params", "domain"),
        "division": ("workflow_params", "division"),
        "pdf_paths": ("workflow_params", "pdf_paths"),
        "html_paths": ("phase_outputs", "dart_conversion", "output_paths"),
        "imscc_path": ("phase_outputs", "packaging", "package_path"),
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
        """
        predicate = getattr(phase, "enabled_when_env", None)
        if predicate:
            if not self._eval_enabled_when_env(predicate):
                return True

        if not getattr(phase, "optional", False):
            return False

        # Skip trainforge_assessment if generate_assessments is False
        if phase.name == "trainforge_assessment":
            return not workflow_params.get("generate_assessments", True)

        return False

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

    def _dependencies_met(
        self, phase: WorkflowPhase, phase_outputs: Dict[str, Dict]
    ) -> bool:
        """Check that all phase dependencies have completed."""
        for dep in (phase.depends_on or []):
            dep_output = phase_outputs.get(dep, {})
            if not dep_output.get("_completed"):
                return False
        return True

    def _topological_sort(self, phases: List[WorkflowPhase]) -> List[WorkflowPhase]:
        """
        Sort phases respecting depends_on ordering.

        Uses Kahn's algorithm for topological sort.
        """
        phase_map = {p.name: p for p in phases}
        in_degree = {p.name: 0 for p in phases}

        for phase in phases:
            for dep in (phase.depends_on or []):
                if dep in in_degree:
                    in_degree[phase.name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_names = []

        while queue:
            # Pick the first available (stable sort)
            name = queue.pop(0)
            sorted_names.append(name)

            for phase in phases:
                if name in (phase.depends_on or []):
                    in_degree[phase.name] -= 1
                    if in_degree[phase.name] == 0:
                        queue.append(phase.name)

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
