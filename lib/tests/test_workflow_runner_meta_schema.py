"""Regression tests for REC-CTR-05 + REC-CTR-06 (Wave 6).

Covers:
- Meta-schema validation of config/workflows.yaml
  (schemas/config/workflows_meta.schema.json)
- YAML-backed phase routing accessors in MCP.core.workflow_runner
- DartMarkersValidator gate wrapper for the orphaned `validate_dart_markers`
  MCP tool (REC-CTR-06)
"""

import copy
import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_YAML = PROJECT_ROOT / "config" / "workflows.yaml"
META_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "config" / "workflows_meta.schema.json"


# ---------------------------------------------------------------------------
# Meta-schema fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def meta_schema():
    jsonschema = pytest.importorskip("jsonschema")  # noqa: F841
    with open(META_SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def workflows_yaml_data():
    with open(WORKFLOWS_YAML) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Meta-schema happy / sad paths
# ---------------------------------------------------------------------------


def test_meta_schema_accepts_current_workflows_yaml(meta_schema, workflows_yaml_data):
    """REC-CTR-05: The real config/workflows.yaml must validate clean."""
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(workflows_yaml_data, meta_schema)


def test_meta_schema_rejects_unknown_severity(meta_schema, workflows_yaml_data):
    """Severity outside the enum must fail schema validation."""
    jsonschema = pytest.importorskip("jsonschema")
    mutated = copy.deepcopy(workflows_yaml_data)
    # Pick any workflow with at least one gate.
    for wf in mutated["workflows"].values():
        for phase in wf["phases"]:
            if phase.get("validation_gates"):
                phase["validation_gates"][0]["severity"] = "catastrophic"
                break
        else:
            continue
        break
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated, meta_schema)


def test_meta_schema_rejects_missing_gate_id(meta_schema, workflows_yaml_data):
    """Gate entries must carry a `gate_id`."""
    jsonschema = pytest.importorskip("jsonschema")
    mutated = copy.deepcopy(workflows_yaml_data)
    for wf in mutated["workflows"].values():
        for phase in wf["phases"]:
            if phase.get("validation_gates"):
                del phase["validation_gates"][0]["gate_id"]
                break
        else:
            continue
        break
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated, meta_schema)


def test_meta_schema_rejects_bad_validator_path(meta_schema, workflows_yaml_data):
    """Validator paths without a module separator must fail the regex."""
    jsonschema = pytest.importorskip("jsonschema")
    mutated = copy.deepcopy(workflows_yaml_data)
    for wf in mutated["workflows"].values():
        for phase in wf["phases"]:
            if phase.get("validation_gates"):
                phase["validation_gates"][0]["validator"] = "nodot"
                break
        else:
            continue
        break
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated, meta_schema)


def test_meta_schema_rejects_invalid_phase_name(meta_schema, workflows_yaml_data):
    """Phase names must be snake_case (lowercase + underscores)."""
    jsonschema = pytest.importorskip("jsonschema")
    mutated = copy.deepcopy(workflows_yaml_data)
    first_wf = next(iter(mutated["workflows"].values()))
    first_wf["phases"][0]["name"] = "NotSnake-Case"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated, meta_schema)


def test_meta_schema_rejects_bad_inputs_from_source(meta_schema, workflows_yaml_data):
    """`source:` must be one of workflow_params, phase_outputs, literal."""
    jsonschema = pytest.importorskip("jsonschema")
    mutated = copy.deepcopy(workflows_yaml_data)
    # Attach an invalid inputs_from entry to the first phase we see.
    first_wf = next(iter(mutated["workflows"].values()))
    first_wf["phases"][0]["inputs_from"] = [
        {"param": "foo", "source": "magic", "key": "bar"}
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated, meta_schema)


# ---------------------------------------------------------------------------
# Cross-reference integrity (graph check)
# ---------------------------------------------------------------------------


def test_inputs_from_cross_reference_rejects_unresolved():
    """REC-CTR-05: phase_outputs references must resolve to a prior-phase output.

    Exercises `_validate_inputs_from_references` directly with a synthetic
    workflow that references a non-existent output.
    """
    from MCP.core.workflow_runner import _validate_inputs_from_references

    bad_cfg = {
        "workflows": {
            "demo": {
                "phases": [
                    {"name": "first", "outputs": ["artifact_a"]},
                    {
                        "name": "second",
                        "inputs_from": [
                            {
                                "param": "need",
                                "source": "phase_outputs",
                                "phase": "first",
                                "output": "artifact_missing",
                            }
                        ],
                    },
                ]
            }
        }
    }

    with pytest.raises(ValueError, match="artifact_missing"):
        _validate_inputs_from_references(bad_cfg)


def test_inputs_from_cross_reference_rejects_unknown_phase():
    """References to a not-yet-declared (or non-existent) phase must fail."""
    from MCP.core.workflow_runner import _validate_inputs_from_references

    bad_cfg = {
        "workflows": {
            "demo": {
                "phases": [
                    {
                        "name": "first",
                        "inputs_from": [
                            {
                                "param": "x",
                                "source": "phase_outputs",
                                "phase": "ghost",
                                "output": "y",
                            }
                        ],
                    }
                ]
            }
        }
    }

    with pytest.raises(ValueError, match="ghost"):
        _validate_inputs_from_references(bad_cfg)


def test_inputs_from_cross_reference_allows_valid_dag():
    """A well-formed workflow must pass the cross-ref check."""
    from MCP.core.workflow_runner import _validate_inputs_from_references

    good_cfg = {
        "workflows": {
            "demo": {
                "phases": [
                    {"name": "first", "outputs": ["a", "b"]},
                    {
                        "name": "second",
                        "inputs_from": [
                            {
                                "param": "x",
                                "source": "phase_outputs",
                                "phase": "first",
                                "output": "a",
                            }
                        ],
                        "outputs": ["c"],
                    },
                ]
            }
        }
    }

    # Should not raise.
    _validate_inputs_from_references(good_cfg)


# ---------------------------------------------------------------------------
# workflow_runner loader + accessors
# ---------------------------------------------------------------------------


def test_workflow_runner_loads_yaml_config():
    """Module-load validation must produce a cached config with all workflows."""
    from MCP.core.workflow_runner import _load_workflows_config

    cfg = _load_workflows_config()
    assert "workflows" in cfg
    expected = {
        "course_generation",
        "intake_remediation",
        "batch_dart",
        "rag_training",
        "textbook_to_course",
    }
    assert expected.issubset(set(cfg["workflows"].keys()))


def test_get_phase_param_routing_matches_legacy_structure():
    """YAML-sourced routing must produce the same tuple shape as legacy dict."""
    from MCP.core.workflow_runner import (
        _LEGACY_PHASE_PARAM_ROUTING,
        _get_phase_param_routing,
    )

    # Phase with a YAML `inputs_from:` block — must yield tuples identical to
    # the legacy dict for textbook_to_course phases.
    for phase_name, legacy_routing in _LEGACY_PHASE_PARAM_ROUTING.items():
        yaml_routing = _get_phase_param_routing(phase_name)
        assert yaml_routing == legacy_routing, (
            f"Mismatch on phase {phase_name}: legacy={legacy_routing}, "
            f"yaml={yaml_routing}"
        )


def test_get_phase_output_keys_matches_legacy():
    """YAML-sourced output keys must match legacy for annotated phases."""
    from MCP.core.workflow_runner import (
        _LEGACY_PHASE_OUTPUT_KEYS,
        _get_phase_output_keys,
    )

    for phase_name, legacy_keys in _LEGACY_PHASE_OUTPUT_KEYS.items():
        yaml_keys = _get_phase_output_keys(phase_name)
        assert list(yaml_keys) == list(legacy_keys), (
            f"Mismatch on phase {phase_name}: legacy={legacy_keys}, "
            f"yaml={yaml_keys}"
        )


def test_get_phase_param_routing_fallback_to_legacy_on_missing_yaml():
    """Unannotated phases must fall through to the legacy in-memory dict."""
    from MCP.core.workflow_runner import _get_phase_param_routing

    # `multi_source_synthesis` is in batch_dart but has no legacy routing
    # and no `inputs_from:` -> expect an empty dict.
    assert _get_phase_param_routing("multi_source_synthesis") == {}


def test_get_phase_output_keys_unknown_phase_returns_empty():
    from MCP.core.workflow_runner import _get_phase_output_keys

    assert _get_phase_output_keys("phase_that_does_not_exist") == []


# ---------------------------------------------------------------------------
# DartMarkersValidator (REC-CTR-06)
# ---------------------------------------------------------------------------


GOOD_DART_HTML = """<!DOCTYPE html>
<html lang="en">
<head><title>DART doc</title></head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>
  <main role="main">
    <section class="dart-section" aria-labelledby="sec-1">
      <h2 id="sec-1">Section 1</h2>
      <p>Body.</p>
    </section>
  </main>
</body>
</html>
"""


def test_dart_markers_validator_passes_on_compliant_html():
    from lib.validators.dart_markers import DartMarkersValidator

    result = DartMarkersValidator().validate({"html_content": GOOD_DART_HTML})
    # Legacy critical markers are present -> gate passes.
    assert result.passed is True
    # Wave 8 added warning-level provenance checks. GOOD_DART_HTML has a
    # <section> without data-dart-source / data-dart-block-id, so warnings
    # are expected. No *critical* issues should be raised.
    critical = [i for i in result.issues if i.severity == "critical"]
    assert critical == []
    # Score reflects critical markers only; warnings do not deduct.
    assert result.score == 1.0


def test_dart_markers_validator_fails_on_missing_main_role():
    from lib.validators.dart_markers import DartMarkersValidator

    stripped = GOOD_DART_HTML.replace('role="main"', "")
    result = DartMarkersValidator().validate({"html_content": stripped})
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "MISSING_MAIN_ROLE" in codes


def test_dart_markers_validator_fails_on_empty_input():
    from lib.validators.dart_markers import DartMarkersValidator

    result = DartMarkersValidator().validate({})
    assert result.passed is False
    assert any(i.code == "EMPTY_CONTENT" for i in result.issues)


def test_dart_markers_validator_reports_missing_file(tmp_path):
    from lib.validators.dart_markers import DartMarkersValidator

    missing = tmp_path / "nope.html"
    result = DartMarkersValidator().validate({"html_path": str(missing)})
    assert result.passed is False
    assert any(i.code == "FILE_NOT_FOUND" for i in result.issues)


def test_dart_markers_validator_reads_file(tmp_path):
    from lib.validators.dart_markers import DartMarkersValidator

    html_file = tmp_path / "good.html"
    html_file.write_text(GOOD_DART_HTML, encoding="utf-8")
    result = DartMarkersValidator().validate({"html_path": str(html_file)})
    assert result.passed is True


def test_dart_markers_validator_path_is_allowlisted():
    """REC-CTR-06: Validator must be importable via ValidationGateManager.

    The gate manager has an allowlist of module prefixes. The new validator
    lives under `lib.validators.` which is already allowed, so the gate
    manager should accept it without modification.
    """
    from MCP.hardening.validation_gates import ValidationGateManager

    mgr = ValidationGateManager()
    validator = mgr.load_validator("lib.validators.dart_markers.DartMarkersValidator")
    # Validate smoke
    result = validator.validate({"html_content": GOOD_DART_HTML})
    assert result.passed is True


# ---------------------------------------------------------------------------
# Gate wiring in workflows.yaml (REC-CTR-06)
# ---------------------------------------------------------------------------


def test_dart_markers_gate_wired_to_batch_dart_and_textbook_pipeline(
    workflows_yaml_data,
):
    """The dart_markers gate must appear in both workflows per REC-CTR-06."""
    wf = workflows_yaml_data["workflows"]

    # batch_dart: gate is on `multi_source_synthesis` (the DART-producing phase)
    batch_phases = {p["name"]: p for p in wf["batch_dart"]["phases"]}
    batch_gates = [
        g["gate_id"] for g in (batch_phases["multi_source_synthesis"].get("validation_gates") or [])
    ]
    assert "dart_markers" in batch_gates

    # textbook_to_course: gate is on `dart_conversion`
    tbc_phases = {p["name"]: p for p in wf["textbook_to_course"]["phases"]}
    tbc_gates = [
        g["gate_id"] for g in (tbc_phases["dart_conversion"].get("validation_gates") or [])
    ]
    assert "dart_markers" in tbc_gates
