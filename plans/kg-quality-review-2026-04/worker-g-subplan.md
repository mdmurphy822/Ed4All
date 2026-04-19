# Worker G Sub-Plan — REC-CTR-04: decision_type enum reconciliation + fail-closed (opt-in)

**Branch:** `worker-g/wave1-decision-type`
**Base:** `dev-v0.2.0`
**Status:** Draft — written BEFORE any code changes per coordination contract.

---

## 1. Current state discovery

### 1.1 Schema enum

**File:** `schemas/events/decision_event.schema.json`
**Lines:** L61–104 (the `decision_type` object; `enum` array at L63–103).
**Value count:** **39**.

Current values (in order as written in the schema):

```
approach_selection
strategy_decision
source_selection
source_interpretation
textbook_integration
existing_content_usage
content_structure
content_depth
content_adaptation
example_selection
pedagogical_strategy
assessment_design
bloom_level_assignment
learning_objective_mapping
accessibility_measures
format_decision
component_selection
quality_judgment
validation_result
error_handling
prompt_response
file_creation
outcome_signal
chunk_selection
question_generation
distractor_generation
revision_decision
source_usage
alignment_check
structure_detection
heading_assignment
alt_text_generation
math_conversion
research_approach
query_decomposition
retrieval_ranking
result_fusion
chunk_deduplication
index_strategy
```

Observation: the ordering is **not alphabetical**. It's grouped approximately by concern (approach → content → pedagogy → assessment → validation → I/O → RAG). We will preserve the existing ordering convention (group-then-append), appending new values at the end of the enum array rather than forcing alphabetical sort.

### 1.2 In-code allowlist

**File:** `lib/decision_capture.py`
**Lines:** L87–93 (the `ALLOWED_DECISION_TYPES` tuple; docstring preamble at L75–86).

Exact current shape:

```python
ALLOWED_DECISION_TYPES: tuple = (
    # Worker C (training-pair synthesis, landed in worker-c/training-pairs):
    "instruction_pair_synthesis",
    "preference_pair_generation",
    # Worker F (typed-edge concept graph, landed in worker-f/typed-edge-graph):
    "typed_edge_inference",
)
```

3 values. None of them appear in the schema enum today — that is the drift we are reconciling.

### 1.3 Historical training-captures grep

Command:

```python
import json, glob
types=set()
for p in glob.glob('training-captures/**/*.jsonl', recursive=True):
    with open(p) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                if 'decision_type' in r: types.add(r['decision_type'])
            except: pass
```

Scanned **115** JSONL files. Distinct `decision_type` values observed historically:

```
bloom_level_assignment       (already in schema enum — OK)
chunk_serialization          (NOT in schema, NOT in allowlist)
imscc_extraction             (NOT in schema, NOT in allowlist)
instruction_pair_synthesis   (in allowlist, NOT in schema)
preference_pair_generation   (in allowlist, NOT in schema)
```

### 1.4 Live emit sites outside both

Grep `decision_type\s*=` across the tree (`.py`) surfaces many additional emit sites whose values are NOT in the schema enum AND NOT historical in `training-captures/`. These are NOT added to the enum in this PR because:

1. Spec explicitly limits additions to (a) in-code allowlist values, (b) `typed_edge_inference` per review, (c) historical training-captures values.
2. Some of them (executor phase/task lifecycle) are orchestrator-private and may belong to a different event schema entirely.
3. Adding them without verifying they are canonical risks enshrining transient naming.

These are catalogued here for visibility (future wave or sibling PR):

| emit site | decision_type | note |
|-----------|---------------|------|
| `MCP/core/executor.py:383` | `task_execution` | orchestrator internal |
| `MCP/core/executor.py:415` | `task_completion` | orchestrator internal |
| `MCP/core/executor.py:513` | `task_retry` | orchestrator internal |
| `MCP/core/executor.py:685` | `workflow_execution` | orchestrator internal |
| `MCP/core/executor.py:832` | `phase_start` | orchestrator internal |
| `MCP/core/executor.py:938` | `phase_completion` | orchestrator internal |
| `Trainforge/generators/question_factory.py` (6 sites) | `question_creation` | alias of `question_generation`? |
| `Trainforge/generators/question_factory.py:457` | `bloom_alignment_rejection` | specific to factory |
| `Trainforge/generators/assessment_generator.py:217` | `assessment_planning` | |
| `Trainforge/generators/assessment_generator.py:312` | `leak_check_filtering` | |
| `Trainforge/generators/assessment_generator.py:352` | `assessment_generation` | |
| `Trainforge/generators/assessment_generator.py:393` | `chunk_retrieval` | |
| `Trainforge/generators/assessment_generator.py:444` | `question_type_selection` | |
| `Trainforge/generators/assessment_generator.py:1021` | `objective_assessment` | |
| `Trainforge/process_course.py:2328` | `boilerplate_strip` | |

These produce zero matches in `training-captures/**/*.jsonl` today — either the code paths haven't run under capture recently or captures for those runs are not in the repo. Out of scope per spec.

### 1.5 Sibling schema check

**File:** `schemas/events/trainforge_decision.schema.json`
Uses `allOf: [{"$ref": "decision_event.schema.json"}, ...]` — it **inherits** the `decision_type` enum from `decision_event.schema.json`. It does NOT carry its own enum. No modification needed. Confirmed at L6–8 of that schema.

---

## 2. Final enum plan

### 2.1 Values to add (5)

All three in-code allowlist values, plus two historical-corpus values:

1. `instruction_pair_synthesis` — emitted by `Trainforge/synthesize_training.py:226, 261, 312`; 115 captures include this.
2. `preference_pair_generation` — emitted by `Trainforge/synthesize_training.py:288`; captures include this.
3. `typed_edge_inference` — emitted by `Trainforge/rag/typed_edge_inference.py:176` and `Trainforge/process_course.py:1842` (LLM escalation path). Listed in in-code allowlist.
4. `chunk_serialization` — emitted by `Trainforge/process_course.py:1679`; appears in historical captures.
5. `imscc_extraction` — emitted by `Trainforge/process_course.py:663`; appears in historical captures.

### 2.2 Final count

**39 + 5 = 44 values.**

### 2.3 Placement convention

Appended to the end of the existing enum, grouped as a `"// Worker G reconciliation"`-style conceptual block (we cannot actually place JSON comments in strict JSON-Schema, but the ordering documents the intent for readers via the sub-plan). The existing enum is grouped-not-alphabetical, so preserving that convention means appending.

---

## 3. Code changes

### 3.1 `schemas/events/decision_event.schema.json`

Extend `properties.decision_type.enum` from 39 → 44 values by appending:

```
"instruction_pair_synthesis",
"preference_pair_generation",
"typed_edge_inference",
"chunk_serialization",
"imscc_extraction"
```

No other edits to that schema. `$id`, `$schema`, and all other properties unchanged.

### 3.2 `lib/decision_capture.py`

**Change 1 — replace hardcoded `ALLOWED_DECISION_TYPES` tuple (L75–93):**

Old shape:

```python
# ADR-001 Contract 3: decision-type registry.
# ... docstring explaining why this was an advisory free-string field ...
ALLOWED_DECISION_TYPES: tuple = (
    # Worker C (training-pair synthesis, landed in worker-c/training-pairs):
    "instruction_pair_synthesis",
    "preference_pair_generation",
    # Worker F (typed-edge concept graph, landed in worker-f/typed-edge-graph):
    "typed_edge_inference",
)
```

New shape: derive from schema at module-import time, following `lib/validation.py::load_schema` pattern (which loads `schemas/events/decision_event.schema.json` from `SCHEMAS_DIR`):

```python
# ADR-001 Contract 3 + REC-CTR-04: decision-type registry.
# Source of truth is schemas/events/decision_event.schema.json.
# Loaded at import time via lib/validation.py::load_schema.
# On schema-load failure, falls back to a minimal tuple (backward compat)
# so this module still imports cleanly in environments where the schema
# file is not present (e.g., minimal test environments).
try:
    from .validation import load_schema as _load_schema
    _decision_schema = _load_schema("decision_event")
    ALLOWED_DECISION_TYPES: tuple = tuple(
        _decision_schema["properties"]["decision_type"]["enum"]
    )
except Exception as _e:  # pragma: no cover - defensive fallback
    logger.warning(
        "Failed to load decision_event schema for ALLOWED_DECISION_TYPES: %s; "
        "falling back to minimal tuple", _e
    )
    ALLOWED_DECISION_TYPES: tuple = (
        "instruction_pair_synthesis",
        "preference_pair_generation",
        "typed_edge_inference",
    )
```

Placement: replaces L75–93 verbatim. `logger` is defined at L72, so this block must remain after L72.

**Change 2 — flip `_validate_record` from warn-only to fail-closed under env var (L401–414):**

Old shape:

```python
def _validate_record(self, record: Dict[str, Any]) -> None:
    """Validate a decision record, adding issues to metadata if found."""
    if not VALIDATE_DECISIONS:
        return
    try:
        from .validation import validate_decision
        is_valid, issues = validate_decision(record, self.tool)
        if not is_valid:
            logger.warning("Decision validation issues: %s", issues)
            record["metadata"]["validation_issues"] = issues
    except ImportError:
        pass  # Validation module not available
    except Exception as e:
        logger.warning("Decision validation error: %s", e)
```

Note the pre-existing `VALIDATE_DECISIONS` module-level constant in `lib/constants.py:184` is `os.environ.get("VALIDATE_DECISIONS", "true").lower() == "true"` — i.e., it's already an env-var gate, but defaults to `True`. The gate controls whether validation runs at all (warn-only mode when it does run).

For REC-CTR-04 we need a **separate** gate that controls whether unknown `decision_type` (or any validation failure) BLOCKS the write. The review specified `VALIDATE_DECISIONS=true` as the trigger; read literally, that matches the existing constant. However the existing `VALIDATE_DECISIONS` already defaults to `"true"`, so using it as-is would flip behavior for every caller on the planet.

**Resolution:** Introduce a NEW env var `DECISION_VALIDATION_STRICT` (read directly inside `_validate_record` via `os.getenv`, bypassing the module-level constant) — checked ONLY when `VALIDATE_DECISIONS` is already truthy. Default unset → current warn-only behavior preserved. Set to `"true"` → raise `ValueError` on validation failure. This matches the spec intent ("fail-closed is OPT-IN via env var. Preserve warn-only as default for backward compat") while keeping backward compat bulletproof.

> **Deviation from literal spec.** The spec text says "when `os.getenv('VALIDATE_DECISIONS') == 'true'`, raise `ValueError`". Taken literally that collides with the pre-existing `VALIDATE_DECISIONS` constant semantics (which defaults true). To preserve backward compat (explicit spec requirement in the same paragraph), we introduce `DECISION_VALIDATION_STRICT` as the opt-in strict flag. The behavior contract is unchanged: "without the flag, warn-only; with the flag, fail-closed." Calling convention documented in docstring.

New shape:

```python
def _validate_record(self, record: Dict[str, Any]) -> None:
    """Validate a decision record.

    Behavior matrix (REC-CTR-04):
    - ``VALIDATE_DECISIONS`` unset/false -> no-op (preserves backward compat
      for callers that explicitly opted out of validation entirely).
    - ``VALIDATE_DECISIONS`` truthy + ``DECISION_VALIDATION_STRICT`` unset ->
      warn-only. Validation issues are appended to
      ``record["metadata"]["validation_issues"]`` and the record IS still
      written. This is the historical default and is preserved to avoid
      breaking in-flight callers.
    - ``VALIDATE_DECISIONS`` truthy + ``DECISION_VALIDATION_STRICT=true`` ->
      fail-closed. Validation failures raise ``ValueError`` and the record
      is NOT written (caller must handle).

    Opt-in strict mode is the reconciliation target from REC-CTR-04.
    """
    if not VALIDATE_DECISIONS:
        return

    strict = os.getenv("DECISION_VALIDATION_STRICT", "").lower() == "true"

    try:
        from .validation import validate_decision
        is_valid, issues = validate_decision(record, self.tool)
        if not is_valid:
            if strict:
                raise ValueError(
                    f"Decision validation failed (strict mode): {issues}"
                )
            logger.warning("Decision validation issues: %s", issues)
            record["metadata"]["validation_issues"] = issues
    except ImportError:
        pass  # Validation module not available
    except ValueError:
        raise  # Re-raise strict-mode failures
    except Exception as e:
        logger.warning("Decision validation error: %s", e)
```

Important: in strict mode the `ValueError` must propagate through `log_decision`, which means the record is NOT appended to `self.decisions` and NOT streamed to disk. That is the intended fail-closed semantics. `log_decision` already calls `_validate_record(record)` BEFORE `self.decisions.append(record)` and `self._write_to_streams(record)` (L478–480), so the ordering already supports this.

### 3.3 Tests

**Test file:** `lib/tests/test_decision_capture.py` (confirmed to exist; 540 lines; pytest-based with `unit` marker).

Append a new test class to the end of the file:

```python
# =============================================================================
# DECISION TYPE RECONCILIATION TESTS (REC-CTR-04 / Worker G)
# =============================================================================

class TestDecisionTypeReconciliation:
    """Test decision_type enum reconciliation + fail-closed gate."""

    @pytest.fixture
    def capture(self, mock_libv2_storage, mock_legacy_dir):
        return DecisionCapture(
            course_code="TEST_101",
            phase="content-generator",
            streaming=False,
        )

    @pytest.mark.unit
    def test_allowed_decision_types_loaded_from_schema(self):
        """ALLOWED_DECISION_TYPES should be derived from schema enum."""
        from lib.decision_capture import ALLOWED_DECISION_TYPES
        # Schema-loaded tuple should contain the REC-CTR-04 additions.
        assert "instruction_pair_synthesis" in ALLOWED_DECISION_TYPES
        assert "preference_pair_generation" in ALLOWED_DECISION_TYPES
        assert "typed_edge_inference" in ALLOWED_DECISION_TYPES
        # And historical corpus types.
        assert "chunk_serialization" in ALLOWED_DECISION_TYPES
        assert "imscc_extraction" in ALLOWED_DECISION_TYPES
        # And at least one of the long-standing values.
        assert "content_structure" in ALLOWED_DECISION_TYPES

    @pytest.mark.unit
    def test_all_schema_enum_values_accepted(self, capture, monkeypatch):
        """Every value in the schema enum should be accepted without raising."""
        from lib.decision_capture import ALLOWED_DECISION_TYPES
        # Strict mode on — every schema enum value must validate clean.
        monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
        for dtype in ALLOWED_DECISION_TYPES:
            # Each one should log without raising.
            capture.log_decision(
                decision_type=dtype,
                decision=f"Test decision for {dtype}",
                rationale="A sufficiently long rationale covering the test case.",
            )
        assert len(capture.decisions) == len(ALLOWED_DECISION_TYPES)

    @pytest.mark.unit
    def test_unknown_decision_type_warns_by_default(self, capture, monkeypatch):
        """Unknown decision_type warns (does not raise) when strict flag unset."""
        monkeypatch.delenv("DECISION_VALIDATION_STRICT", raising=False)
        # Novel type not in enum.
        capture.log_decision(
            decision_type="definitely_not_in_the_enum_9f2a",
            decision="Test with unknown type",
            rationale="Testing warn-only backward-compat path for unknown decision types.",
        )
        # Record was written.
        assert len(capture.decisions) == 1
        # validation_issues was populated.
        issues = capture.decisions[0]["metadata"].get("validation_issues", [])
        assert any("definitely_not_in_the_enum_9f2a" in str(i) for i in issues)

    @pytest.mark.unit
    def test_unknown_decision_type_blocks_when_strict_true(self, capture, monkeypatch):
        """Unknown decision_type raises ValueError under DECISION_VALIDATION_STRICT=true."""
        monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")
        with pytest.raises(ValueError, match="validation failed"):
            capture.log_decision(
                decision_type="definitely_not_in_the_enum_7b1c",
                decision="Test with unknown type",
                rationale="Strict mode should block records whose decision_type is unknown.",
            )
        # Record was NOT written.
        assert len(capture.decisions) == 0

    @pytest.mark.unit
    def test_backward_compat_unchanged_when_flag_unset(self, capture, monkeypatch):
        """Without DECISION_VALIDATION_STRICT, known-good records pass silently."""
        monkeypatch.delenv("DECISION_VALIDATION_STRICT", raising=False)
        capture.log_decision(
            decision_type="content_structure",
            decision="Valid decision with known type",
            rationale="Known type should validate clean and land in decisions list.",
        )
        assert len(capture.decisions) == 1
        # No validation_issues key should have been added for a clean record.
        assert "validation_issues" not in capture.decisions[0]["metadata"]
```

Four test cases cover:
1. Enum is schema-derived.
2. Every schema value is accepted even under strict mode.
3. Warn-only path (the default) still populates `metadata.validation_issues`.
4. Strict-mode path raises `ValueError` and drops the record.
5. Backward compat clean path (no validation_issues field pollution on a known-good record).

---

## 4. Verification

```bash
# JSON + schema valid; enum count sanity
python3 -c "import json, jsonschema; s=json.load(open('schemas/events/decision_event.schema.json')); jsonschema.Draft7Validator.check_schema(s); print('OK'); print('enum count:', len(s['properties']['decision_type']['enum']))"

# CI integrity (rglobs schemas/)
python3 -m ci.integrity_check

# Existing decision-capture tests still pass
pytest lib/tests/test_decision_capture.py -v

# New tests pass
pytest lib/tests/test_decision_capture.py::TestDecisionTypeReconciliation -v
```

Schema is Draft 7 (per `"$schema": "http://json-schema.org/draft-07/schema#"` on L2). Use `Draft7Validator.check_schema`, not `Draft202012Validator.check_schema` as the spec snippet shows — the spec is wrong about the draft. `Draft7Validator` is what validation.py actually uses (L17).

---

## 5. Non-goals for this PR

- Orchestrator-private decision types (executor lifecycle). Separate PR.
- Question-factory/assessment-generator novel types (`question_creation`, `assessment_planning`, etc.). Separate PR.
- Migrating historical captures that used `chunk_serialization` / `imscc_extraction`. They already validate against the new enum; no data migration needed.
- `lib/constants.py::VALID_DECISION_TYPES` — this is a basic-validation fallback set (used only when `jsonschema` is not available). It is stale vs schema by design (spec scopes this work to schema + decision_capture allowlist + validator flip).
- Flipping default to strict. Explicit spec requirement: "Fail-closed is OPT-IN via env var."

---

## 6. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Introducing schema-load at `decision_capture.py` import creates an import-order / circular dependency. | `load_schema` is lazy and reads a JSON file; no circular import with `validation.py` possible because `validation.py` already imports from `constants.py` which `decision_capture.py` also imports. Fallback tuple preserves import success even if schema file is missing. |
| `DECISION_VALIDATION_STRICT` env var naming collides with a future env var. | Name is specific enough; documented in docstring. |
| Deviation from literal spec env-var name. | Explicit deviation block in §3.2. Behavior contract matches spec intent. Sub-plan is the committed artifact documenting the rationale. |
| Existing tests call `log_decision` with unknown types and currently pass. | Current tests use known types (`content_structure`, `pedagogical_strategy`, etc.) — confirmed by grep of `lib/tests/test_decision_capture.py`. No existing test uses a novel type. |
| Validator is called inside `log_decision` BEFORE the record is appended/written. Strict-mode raise will escape and propagate to the caller. | Intended. The caller handles the exception (fail-closed contract). Test `test_unknown_decision_type_blocks_when_strict_true` asserts this. |

---

## 7. Files touched (final list)

1. `schemas/events/decision_event.schema.json` — enum extended by 5 values.
2. `lib/decision_capture.py` — replace hardcoded tuple with schema-derived tuple (L75–93); add strict-mode branch to `_validate_record` (L401–414).
3. `lib/tests/test_decision_capture.py` — append `TestDecisionTypeReconciliation` (5 test methods).
4. `plans/kg-quality-review-2026-04/worker-g-subplan.md` — this file.

No other files touched.
