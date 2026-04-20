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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

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
    },
    "packaging": {
        "project_id": ("phase_outputs", "objective_extraction", "project_id"),
    },
    "trainforge_assessment": {
        "course_id": ("workflow_params", "course_name"),
        "imscc_path": ("phase_outputs", "packaging", "package_path"),
        "bloom_levels": ("workflow_params", "bloom_levels"),
        "question_count": ("workflow_params", "assessment_count"),
        "objective_ids": ("phase_outputs", "objective_extraction", "objective_ids"),
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
    "dart_conversion": ["output_path", "output_paths", "success", "html_length"],
    "staging": ["staging_dir", "staged_files", "file_count"],
    "objective_extraction": [
        "project_id", "project_path", "objective_ids", "textbook_structure_path",
    ],
    "source_mapping": ["source_module_map_path", "source_chunk_ids"],
    "course_planning": ["project_id"],
    "content_generation": ["project_id", "content_paths", "weeks_prepared"],
    "packaging": ["package_path", "libv2_package_path", "project_id"],
    "trainforge_assessment": ["output_path", "assessment_id", "question_count"],
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

            # Execute the phase
            results, gates_passed, gate_results = await self.executor.execute_phase(
                workflow_id=workflow_id,
                phase_name=phase_name,
                phase_index=phase_idx,
                tasks=tasks,
                gate_configs=gate_configs,
                max_concurrent=getattr(phase, "max_concurrent", 5),
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
                "failed": sum(1 for r in results.values() if r.status in ("ERROR", "TIMEOUT")),
                "gates_passed": gates_passed,
            }

            # Check if phase failed
            phase_failed = any(r.status in ("ERROR", "TIMEOUT") for r in results.values())
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
                    path = result.result.get("output_path")
                    if path:
                        paths.append(path)
            if paths:
                extracted["output_paths"] = ",".join(paths)

        return extracted

    def _should_skip_phase(
        self, phase: WorkflowPhase, workflow_params: Dict[str, Any]
    ) -> bool:
        """Check if an optional phase should be skipped based on workflow params."""
        if not getattr(phase, "optional", False):
            return False

        # Skip trainforge_assessment if generate_assessments is False
        if phase.name == "trainforge_assessment":
            return not workflow_params.get("generate_assessments", True)

        return False

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
