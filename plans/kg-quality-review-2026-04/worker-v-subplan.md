# Worker V Sub-plan — REC-CTR-05 + REC-CTR-06 (Wave 6)

**Branch:** `worker-v/wave6-workflow-governance`
**Target:** `dev-v0.2.0`

## 1. Current state (read)

### `MCP/core/workflow_runner.py`
- L37 `PHASE_PARAM_ROUTING: Dict[str, Dict[str, Tuple]]` — 9 phases, each a dict of `{param_name: (source_type, *source_path)}` where `source_type ∈ {"workflow_params", "phase_outputs", "literal"}`.
- L87 `PHASE_OUTPUT_KEYS: Dict[str, List[str]]` — 9 phases, each a list of output-key strings extracted from task results.
- L272 consumer: `_route_params()` looks up `PHASE_PARAM_ROUTING.get(phase_name, {})` and resolves values.
- L403 consumer: `_extract_phase_outputs()` looks up `PHASE_OUTPUT_KEYS.get(phase_name, [])` and extracts from `ExecutionResult.result` dicts.
- Additional special handling at L419 — `dart_conversion` joins multiple `output_path`s into a CSV `output_paths`.

### `config/workflows.yaml`
- 583 lines. Top-level keys: `workflows`, `defaults`, `dependency_resolution`, `hardening`.
- Each workflow: `description`, `retry_policy`, `poison_pill`, `phases[]`.
- Each phase: `name`, `agents`, `parallel`, `max_concurrent`, `batch_by`, `depends_on`, `timeout_minutes`, `batch_timeout_minutes`, optional `validation_gates[]`, `description`, `optional`.
- Gate entry: `gate_id`, `validator`, `severity`, `threshold`, `behavior.{on_fail,on_error}`.
- `dart_conversion` phase exists in both `batch_dart` (as `multi_source_synthesis` — note: differs from Python dict key!) and `textbook_to_course`.

**CRITICAL FINDING:** `batch_dart.phases[0].name == "multi_source_synthesis"`, NOT `dart_conversion`. The DART marker gate needs to live under `multi_source_synthesis` for `batch_dart`, and under `dart_conversion` for `textbook_to_course`.

### `MCP/tools/pipeline_tools.py`
- L368 `validate_dart_markers(html_path: str) -> str` — decorated `@mcp.tool()` inside the `register_pipeline_tools(mcp)` function. Takes a path, returns JSON string with `{valid, file, markers, missing, message?}`.
- Does NOT match the gate Validator protocol (`validate(inputs: Dict) -> GateResult`).
- Need a thin wrapper class `DartMarkersValidator` with `name`, `version`, `validate(inputs)` that reuses the marker-detection logic. Expected `inputs` keys: `html_path` or `html_content`. Returns `GateResult` with critical issues per missing marker.

### `MCP/hardening/validation_gates.py`
- `ALLOWED_VALIDATOR_PREFIXES = ("lib.validators.", "lib.leak_checker", "DART.pdf_converter.",)` — **does not include `MCP.tools.`**. Two options:
  1. Add `MCP.tools.` to the allowlist.
  2. Place the `DartMarkersValidator` wrapper in `lib/validators/dart_markers.py` (preferred — keeps validators together and respects the existing allowlist design).

**Decision:** place wrapper at `lib/validators/dart_markers.py`. That's consistent with how all existing validators live under `lib/validators/`.

## 2. Meta-schema design (`schemas/config/workflows_meta.schema.json`)

Draft 2020-12. Validates `config/workflows.yaml` once loaded into a dict.

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://ed4all.local/schemas/config/workflows_meta.schema.json",
  "type": "object",
  "required": ["workflows"],
  "properties": {
    "workflows": {
      "type": "object",
      "additionalProperties": { "$ref": "#/$defs/Workflow" }
    },
    "defaults": { "type": "object" },
    "dependency_resolution": { "type": "object" },
    "hardening": { "type": "object" }
  },
  "$defs": {
    "Workflow": {
      "type": "object",
      "required": ["phases"],
      "properties": {
        "description": { "type": "string" },
        "mcp_tools": { "type": "array", "items": { "type": "string" } },
        "retry_policy": { "type": "object" },
        "poison_pill": { "type": "object" },
        "phases": {
          "type": "array",
          "items": { "$ref": "#/$defs/Phase" }
        }
      }
    },
    "Phase": {
      "type": "object",
      "required": ["name"],
      "properties": {
        "name": { "type": "string", "pattern": "^[a-z_][a-z0-9_]*$" },
        "agents": { "type": "array", "items": { "type": "string" } },
        "parallel": { "type": "boolean" },
        "max_concurrent": { "type": "integer", "minimum": 1 },
        "batch_by": { "type": ["string", "null"] },
        "depends_on": { "type": "array", "items": { "type": "string" } },
        "optional": { "type": "boolean" },
        "timeout_minutes": { "type": "integer", "minimum": 1 },
        "batch_timeout_minutes": { "type": "integer" },
        "description": { "type": "string" },
        "inputs_from": {
          "type": "array",
          "items": { "$ref": "#/$defs/InputRoute" }
        },
        "outputs": {
          "type": "array",
          "items": { "type": "string" }
        },
        "validation_gates": {
          "type": "array",
          "items": { "$ref": "#/$defs/Gate" }
        }
      }
    },
    "InputRoute": {
      "type": "object",
      "required": ["param", "source"],
      "properties": {
        "param": { "type": "string" },
        "source": { "enum": ["workflow_params", "phase_outputs", "literal"] },
        "key":   { "type": "string" },   // required for workflow_params + literal
        "phase": { "type": "string" },   // required for phase_outputs
        "output": { "type": "string" },  // required for phase_outputs
        "value": {}                      // for literal
      },
      "allOf": [
        { "if": { "properties": { "source": { "const": "workflow_params" } } },
          "then": { "required": ["key"] } },
        { "if": { "properties": { "source": { "const": "phase_outputs" } } },
          "then": { "required": ["phase", "output"] } },
        { "if": { "properties": { "source": { "const": "literal" } } },
          "then": { "required": ["value"] } }
      ]
    },
    "Gate": {
      "type": "object",
      "required": ["gate_id", "validator", "severity"],
      "properties": {
        "gate_id": { "type": "string" },
        "validator": { "type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_.]*$" },
        "severity": { "enum": ["critical", "warning", "info"] },
        "threshold": { "type": "object" },
        "behavior": {
          "type": "object",
          "properties": {
            "on_fail": { "enum": ["block", "warn", "fail_closed"] },
            "on_error": { "enum": ["block", "warn", "fail_closed"] }
          }
        },
        "description": { "type": "string" },
        "enabled": { "type": "boolean" }
      }
    }
  }
}
```

### Cross-reference integrity

Strict schema alone can't enforce "`inputs_from.phase` must reference a prior phase whose `outputs` lists the named `output`". This is a graph-level check. Implement as a Python post-validation pass in `_load_workflows_config()`:

- For each workflow, iterate phases in order.
- Build a `set[str]` of outputs-seen-so-far; start empty.
- For each phase's `inputs_from` where `source == "phase_outputs"`: assert `(phase, output)` pair is in seen-map.
- After processing the phase, add its `outputs[]` to the seen-map under its name.
- Raise `ValueError` with a clear message listing the unresolved reference on failure.

## 3. Phase routing translation (YAML blocks)

Translate each entry from `PHASE_PARAM_ROUTING` + `PHASE_OUTPUT_KEYS` into `inputs_from:` + `outputs:` YAML blocks. Table:

| Phase | inputs_from | outputs |
|-------|-------------|---------|
| `dart_conversion` | `course_code<-workflow_params.course_name` | `output_path`, `output_paths`, `success`, `html_length` |
| `staging` | `run_id<-workflow_params.run_id`, `dart_html_paths<-phase_outputs.dart_conversion.output_paths`, `course_name<-workflow_params.course_name` | `staging_dir`, `staged_files`, `file_count` |
| `objective_extraction` | `course_name<-workflow_params.course_name`, `objectives_path<-workflow_params.objectives_path`, `duration_weeks<-workflow_params.duration_weeks` | `project_id`, `project_path`, `objective_ids` |
| `course_planning` | `project_id<-phase_outputs.objective_extraction.project_id`, `course_name<-workflow_params.course_name`, `objectives_path<-workflow_params.objectives_path` | `project_id` |
| `content_generation` | `project_id<-phase_outputs.objective_extraction.project_id` | `project_id`, `content_paths`, `weeks_prepared` |
| `packaging` | `project_id<-phase_outputs.objective_extraction.project_id` | `package_path`, `libv2_package_path`, `project_id` |
| `trainforge_assessment` | `course_id<-workflow_params.course_name`, `imscc_path<-phase_outputs.packaging.package_path`, `bloom_levels<-workflow_params.bloom_levels`, `question_count<-workflow_params.assessment_count`, `objective_ids<-phase_outputs.objective_extraction.objective_ids` | `output_path`, `assessment_id`, `question_count` |
| `libv2_archival` | 6 routes — `course_name`, `domain`, `division`, `pdf_paths` from params; `html_paths`, `imscc_path` from phase outputs | `course_slug`, `course_dir`, `manifest_path` |
| `finalization` | `project_id<-phase_outputs.objective_extraction.project_id`, `course_slug<-phase_outputs.libv2_archival.course_slug` | `project_id`, `package_path`, `course_slug` |

These blocks added to the phases appearing in `textbook_to_course`. Phases like `multi_source_synthesis` in `batch_dart` do NOT have corresponding entries in the old Python dicts and will remain un-annotated → fall through to empty defaults (warn-log once if the phase name matches the Python-dict list).

## 4. DART markers validator

### `lib/validators/dart_markers.py` (new)
```python
from pathlib import Path
from typing import Any, Dict, List
from MCP.hardening.validation_gates import GateIssue, GateResult

class DartMarkersValidator:
    """Validates DART-processed HTML contains required accessibility markers."""
    name = "dart_markers"
    version = "1.0.0"

    REQUIRED_MARKERS = {
        "skip_link": ('class="skip', "class='skip"),
        "main_role": ('role="main"', "role='main'"),
        "aria_sections": ('aria-labelledby="', "aria-labelledby='"),
        "dart_semantic_classes": ("dart-section", "dart-document"),
    }

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "dart_markers")
        content = inputs.get("html_content", "")
        if not content and inputs.get("html_path"):
            p = Path(inputs["html_path"])
            if not p.exists():
                return GateResult(gate_id=gate_id, validator_name=self.name,
                                  validator_version=self.version, passed=False,
                                  issues=[GateIssue(severity="critical", code="FILE_NOT_FOUND",
                                                    message=f"File not found: {p}")])
            content = p.read_text(encoding="utf-8")

        issues: List[GateIssue] = []
        for name, needles in self.REQUIRED_MARKERS.items():
            if not any(n in content for n in needles):
                issues.append(GateIssue(severity="critical", code=f"MISSING_{name.upper()}",
                                        message=f"Required DART marker missing: {name}",
                                        suggestion=f"Ensure HTML emits one of: {needles}"))
        return GateResult(gate_id=gate_id, validator_name=self.name,
                          validator_version=self.version,
                          passed=len(issues)==0, issues=issues)
```

- Reuses marker logic from `MCP/tools/pipeline_tools.py:392–397`.
- Leaves the MCP tool untouched (keeps backward compat; the tool call pattern still works).

### `config/workflows.yaml` additions
- Under `batch_dart.phases[name==multi_source_synthesis]` — add `validation_gates` entry for `dart_markers` (the phase name in YAML is `multi_source_synthesis`, not `dart_conversion`).
- Under `textbook_to_course.phases[name==dart_conversion]` — add `validation_gates` entry for `dart_markers`.

Validator path: `lib.validators.dart_markers.DartMarkersValidator`.

## 5. `workflow_runner.py` rewrite

- Add `_load_workflows_config()` module-level function: reads YAML, validates against meta-schema, raises `ValueError` on failure. Cached (`_WORKFLOWS_CONFIG = None` module singleton with lazy init).
- Add `_get_phase_param_routing(phase_name: str) -> Dict[str, Tuple]` — looks up YAML `inputs_from`, translates to legacy tuple format. Falls back to in-memory `_LEGACY_PHASE_PARAM_ROUTING` dict if YAML block missing.
- Add `_get_phase_output_keys(phase_name: str) -> List[str]` — looks up YAML `outputs`. Falls back to `_LEGACY_PHASE_OUTPUT_KEYS`.
- Consumers at L272, L403 call the new accessor functions.
- Keep the old dicts renamed with `_LEGACY_` prefix to document the fallback.
- `warn_log_once` pattern for fall-through cases.

## 6. Tests (`lib/tests/test_workflow_runner_meta_schema.py`)

1. `test_meta_schema_accepts_current_workflows_yaml` — load actual YAML, validate.
2. `test_meta_schema_rejects_invalid_severity` — mutate gate severity to bogus value, assert validation failure.
3. `test_meta_schema_rejects_missing_gate_id` — mutate to drop `gate_id`, expect failure.
4. `test_meta_schema_rejects_unresolved_inputs_from` — inject `inputs_from` that references a non-existent output.
5. `test_workflow_runner_loads_yaml_config` — call `_load_workflows_config()`, assert returns expected workflow count.
6. `test_get_phase_param_routing_from_yaml` — after loading, assert a known phase returns expected routing tuple equivalent.
7. `test_get_phase_output_keys_from_yaml` — same for outputs.
8. `test_dart_markers_validator_passes_on_compliant_html` — feed a known-good HTML string with all markers; assert `passed==True`.
9. `test_dart_markers_validator_fails_on_missing_main_role` — strip `role="main"`; assert critical issue.

All tests placed in `lib/tests/test_workflow_runner_meta_schema.py` (one file, per task instructions). DartMarkersValidator tests colocated.

## 7. Verification checklist

- `python3 -m ci.integrity_check` passes.
- New tests pass.
- Full suite still passes (962 baseline + ~9 new ≈ 971).
- Manual smoke: temporarily insert `severity: bogus` in workflows.yaml → loading `OrchestratorConfig` via new meta-schema raises.

## 8. Risks & mitigations

- **Risk:** Validator path allowlist excludes `lib.validators.dart_markers`. → Check: allowlist already has `lib.validators.` prefix — dart_markers is covered.
- **Risk:** Tests that currently load `workflows.yaml` might fail if the meta-schema rejects a shape we didn't anticipate. → Mitigation: run `test_meta_schema_accepts_current_workflows_yaml` first as a contract; tighten schema until it passes against real config.
- **Risk:** `batch_dart` phase naming (`multi_source_synthesis` vs. `dart_conversion`) causing confusion. → Document in YAML comment; apply the gate to the correct actual phase name.

## 9. Commit + PR

Branch `worker-v/wave6-workflow-governance` from `dev-v0.2.0`. Single commit. PR to `dev-v0.2.0`.
