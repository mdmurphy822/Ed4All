# Worker L Sub-Plan — REC-CTR-03 (packager default-on + workflow gate)

**Branch:** `worker-l/wave2-objectives-default`
**Base:** `dev-v0.2.0`
**Scope:** Flip `validate_content_objectives` default in `Courseforge/scripts/package_multifile_imscc.py`; promote LO validation to a workflow-level gate; keep `--skip-validation` as explicit opt-out.

---

## 1. Current-state audit

### `Courseforge/scripts/package_multifile_imscc.py`

Already present on `dev-v0.2.0` (Worker I landed the opt-in gate at commit 4159576):

- Lines 127–153: `validate_content_objectives(content_dir, objectives_path) -> Tuple[bool, List[str]]`
  - Imports `discover_html_pages`, `load_canonical_objectives`, `validate_page` from sibling `validate_page_objectives.py`.
  - Walks `week_*` subdirs; only validates pages whose path contains a `week_*` segment.
  - Returns `(True, [])` on pass or `(False, [msg, ...])` on failures.
- Lines 156–196: `package_imscc(content_dir, output_path, course_code, course_title, *, objectives_path=None, skip_validation=False)`
  - Current gating logic:
    ```python
    if objectives_path and not skip_validation:
        # run validation, SystemExit(2) on failure
    elif skip_validation:
        # print SKIPPED
    ```
  - Default is OPT-IN: validation runs only when `objectives_path` is explicitly passed.
- Lines 199–221: `build_parser()` + `__main__` block.
  - `--objectives` optional (default `None`).
  - `--skip-validation` escape hatch (default `False`).

### `Courseforge/scripts/validate_page_objectives.py`

Companion module. Exposes:
- `extract_json_ld_blocks(html)`
- `extract_lo_ids(html)`
- `infer_week_from_path(page_path)`
- `validate_page(page_path, canonical, week_num=None) -> Tuple[bool, str]`
- `discover_html_pages(root) -> List[Path]`

Re-imports `load_canonical_objectives`, `resolve_week_objectives` from `generate_course.py`.

### `config/workflows.yaml`

Existing gate pattern example from `packaging` phase of `course_generation` (lines 67–75):

```yaml
validation_gates:
  - gate_id: imscc_structure
    validator: lib.validators.imscc.IMSCCValidator
    severity: critical
    threshold:
      max_critical_issues: 0
    behavior:
      on_fail: block
      on_error: fail_closed
```

`textbook_to_course` has a `packaging` phase too (lines 412–428) with a `warning`-severity `imscc_structure` gate — Worker L adds a sibling `page_objectives` gate to BOTH.

### `lib/validators/`

Existing validator file shape (read `lib/validators/content.py` + `lib/validators/bloom.py`). Every validator:

1. Has a `name` class attribute and `version` class attribute.
2. Implements `validate(self, inputs: Dict[str, Any]) -> GateResult`.
3. Returns a `GateResult` from `MCP.hardening.validation_gates` with fields: `gate_id`, `validator_name`, `validator_version`, `passed: bool`, `score: Optional[float]`, `issues: List[GateIssue]`.
4. Issues have `severity` ("error"/"warning"/"info"), `code` (machine-readable), `message`, `suggestion` (optional).
5. `lib/validators/__init__.py` re-exports validator classes.

### Test location pattern

Tests for packager + related scripts live in `Courseforge/scripts/tests/`:
- `test_packager_validation_gate.py` (Worker I — pre-existing coverage of the gate behavior).
- `test_generate_course_lo_specificity.py`.

Worker L adds `test_packager_default.py` in the same directory — co-located with the packager test it complements.

---

## 2. Behavior contract for this change

### Before Worker L

- `--skip-validation` unset, `--objectives PATH`: validation runs.
- `--skip-validation` unset, no `--objectives`: **validation silently skipped** (opt-in default).
- `--skip-validation` set: validation skipped regardless.

### After Worker L

- `--skip-validation` unset, `--objectives PATH`: validation runs (unchanged).
- `--skip-validation` unset, no `--objectives`: **auto-discover** `content_dir / "course.json"`.
  - If found: validation runs against it.
  - If absent: print a WARNING and skip. Do NOT hard-fail — backward-compat for callers that never passed `--objectives`.
- `--skip-validation` set: validation skipped regardless (unchanged; explicit opt-out preserved).

### Auto-discovery path

Master plan specifies `content_dir / "course.json"` as the canonical location. In current Courseforge generation, no such file is emitted at the content-dir root by default — this is the forward-compat path. Callers (Worker J's upcoming `course_metadata.json` emit is SEPARATE; callers that want auto-discovery today can copy their exam-objectives JSON to `<content_dir>/course.json`).

### Hard-fail contract

Preserved: validation FAILURE raises `SystemExit(2)`. Missing objectives file with no `--skip-validation` and no auto-discovery hit is a WARNING (not a fail). This matches the master plan bullet: "never hard-fail on missing objectives file alone. Hard-fail only on VALIDATION FAILURE."

---

## 3. Old/new for `package_imscc` call site

### Old (lines 165–180)

```python
def package_imscc(
    content_dir: Path,
    output_path: Path,
    course_code: str,
    course_title: str,
    *,
    objectives_path: Optional[Path] = None,
    skip_validation: bool = False,
):
    """Create the IMSCC zip package. Refuses to build when per-week LO
    validation fails (unless skip_validation is explicitly set)."""
    if objectives_path and not skip_validation:
        print(f"[validate] Checking per-week learningObjectives against {objectives_path.name}...")
        ok, failures = validate_content_objectives(content_dir, objectives_path)
        if not ok:
            print(f"[validate] REFUSING TO PACKAGE — {len(failures)} page(s) violate per-week LO contract:")
            for msg in failures:
                print(f"  - {msg}")
            print("Fix the offending pages (or re-run generate_course.py with --objectives) then retry.")
            print("Override with --skip-validation if you really know what you're doing.")
            raise SystemExit(2)
        print(f"[validate] All week pages pass per-week LO contract.")
    elif skip_validation:
        print("[validate] SKIPPED (per --skip-validation) — build will not be gated on LO correctness.")
```

### New

```python
def package_imscc(
    content_dir: Path,
    output_path: Path,
    course_code: str,
    course_title: str,
    *,
    objectives_path: Optional[Path] = None,
    skip_validation: bool = False,
):
    """Create the IMSCC zip package. Refuses to build when per-week LO
    validation fails (unless skip_validation is explicitly set).

    Default behavior (Wave 2, Worker L — REC-CTR-03):
      - If ``skip_validation`` is unset and no ``objectives_path`` was passed,
        auto-discover ``content_dir / "course.json"`` and validate against it
        when present.
      - If no objectives file is available (neither explicit nor auto-discovered)
        and ``skip_validation`` is unset, log a warning and skip validation
        (backward-compat for callers that never wired the flag). Hard-fail
        only occurs on actual VALIDATION FAILURE.
      - ``skip_validation=True`` remains the explicit opt-out.
    """
    # Auto-discover objectives if not explicitly provided (default-on behavior).
    if objectives_path is None and not skip_validation:
        candidate = content_dir / "course.json"
        if candidate.exists():
            objectives_path = candidate
            print(f"[validate] Auto-discovered objectives at {candidate}")

    if skip_validation:
        print("[validate] SKIPPED (per --skip-validation) — build will not be gated on LO correctness.")
    elif objectives_path is None:
        print(
            "[validate] WARNING: no objectives file found; skipping LO validation. "
            "Pass --objectives or place course.json at content root to enable."
        )
    else:
        print(f"[validate] Checking per-week learningObjectives against {objectives_path.name}...")
        ok, failures = validate_content_objectives(content_dir, objectives_path)
        if not ok:
            print(f"[validate] REFUSING TO PACKAGE — {len(failures)} page(s) violate per-week LO contract:")
            for msg in failures:
                print(f"  - {msg}")
            print("Fix the offending pages (or re-run generate_course.py with --objectives) then retry.")
            print("Override with --skip-validation if you really know what you're doing.")
            raise SystemExit(2)
        print("[validate] All week pages pass per-week LO contract.")
```

### CLI help text refresh

Pre-Worker-L `--skip-validation` help reads: "Bypass objectives validation even when --objectives is given (escape hatch)". New text reflects that validation now runs by default (not opt-in):

```
--skip-validation: "Opt out of per-week LO validation (not recommended for production builds)."
--objectives: "Canonical objectives JSON to validate per-week LO specificity before packaging.
               If omitted, auto-discovered at <content_dir>/course.json when present."
```

---

## 4. `lib/validators/page_objectives.py` (NEW)

### Import strategy

`validate_content_objectives` lives in `Courseforge/scripts/package_multifile_imscc.py` which is a SCRIPT (not a module reachable via normal import). Two options:

1. **Refactor**: move `validate_content_objectives` to `lib/validators/page_objectives.py` and have the script call back into `lib/`. Cleanest but wider blast radius.
2. **importlib adapter**: load the script module via `importlib.util.spec_from_file_location` from inside the validator. Narrower, doesn't disturb the existing test.

Choosing **(2)**: narrower diff, doesn't require changing the packager script's internal structure or existing tests. The validator resolves `Courseforge/scripts/package_multifile_imscc.py` relative to repo root, imports `validate_content_objectives` once, and caches the callable. Falls back to a clear error message if the script can't be loaded.

Actually, on reflection: `validate_content_objectives` already imports from `validate_page_objectives` at call time (lazy), which itself imports `load_canonical_objectives` and `resolve_week_objectives` from `generate_course.py`. The cleanest path: have the validator import directly from `validate_page_objectives.py` (which exposes the same helpers) and reproduce the 12-line walk inline, avoiding the script-module dance entirely. This matches the Worker I gate's lazy-import pattern.

### File shape

```python
"""Page Objectives Validator (Worker L — REC-CTR-03).

Wraps the per-week learningObjectives gate that already lives in
``Courseforge/scripts/package_multifile_imscc.py::validate_content_objectives``
so the orchestrator's validation-gate framework can invoke it as a
first-class workflow gate (configured in ``config/workflows.yaml``).

Referenced by: config/workflows.yaml (course_generation, textbook_to_course)
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult


def _load_page_objectives_helpers():
    """Lazily load validate_page_objectives from Courseforge/scripts/.

    The helpers live in a script dir, not a python package. Use importlib
    to load them once and cache the module. Falls back to a clear error
    at validation time if the script layout changes.
    """
    scripts_dir = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
    module_path = scripts_dir / "validate_page_objectives.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Expected validate_page_objectives.py at {module_path}; "
            "PageObjectivesValidator cannot run without it."
        )
    # Register scripts dir on sys.path so validate_page_objectives can import
    # load_canonical_objectives and resolve_week_objectives from generate_course.
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "cf_validate_page_objectives", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PageObjectivesValidator:
    """Validates per-week learningObjectives JSON-LD against canonical objectives."""

    name = "page_objectives"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate per-week LO specificity for generated HTML pages.

        Expected inputs:
            content_dir: Path to course content root (with week_* subdirs).
            objectives_path: Path to canonical objectives JSON (optional).
                If absent, auto-discover ``content_dir / "course.json"``.
        """
        gate_id = inputs.get("gate_id", "page_objectives")

        content_dir_raw = inputs.get("content_dir")
        if not content_dir_raw:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="MISSING_CONTENT_DIR",
                        message="content_dir is required for PageObjectivesValidator",
                    )
                ],
            )

        content_dir = Path(content_dir_raw)
        if not content_dir.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="CONTENT_DIR_NOT_FOUND",
                        message=f"content_dir does not exist: {content_dir}",
                    )
                ],
            )

        # Resolve objectives_path (explicit → auto-discover → skip with warning)
        objectives_path_raw = inputs.get("objectives_path")
        objectives_path: Optional[Path] = (
            Path(objectives_path_raw) if objectives_path_raw else None
        )
        if objectives_path is None:
            candidate = content_dir / "course.json"
            if candidate.exists():
                objectives_path = candidate

        if objectives_path is None:
            # Backward-compat: no objectives available → warn, do not fail.
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="NO_OBJECTIVES_FILE",
                        message=(
                            "No objectives file provided and no course.json "
                            f"at {content_dir}. Per-week LO validation skipped."
                        ),
                        suggestion=(
                            "Pass --objectives PATH or emit course.json "
                            "at the content dir root."
                        ),
                    )
                ],
            )

        if not objectives_path.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="OBJECTIVES_FILE_NOT_FOUND",
                        message=f"Objectives file does not exist: {objectives_path}",
                    )
                ],
            )

        # Run the validation via the Courseforge helpers.
        try:
            helpers = _load_page_objectives_helpers()
        except FileNotFoundError as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="VALIDATOR_DEPENDENCIES_MISSING",
                        message=str(exc),
                    )
                ],
            )

        canonical = helpers.load_canonical_objectives(objectives_path)
        pages = helpers.discover_html_pages(content_dir)
        failures: List[Tuple[Path, str]] = []
        for page in pages:
            if not any(part.startswith("week_") for part in page.parts):
                continue
            ok, msg = helpers.validate_page(page, canonical)
            if not ok:
                failures.append((page, msg))

        issues: List[GateIssue] = []
        for page, msg in failures:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="LO_SPECIFICITY_VIOLATION",
                    message=msg,
                    location=str(page),
                    suggestion=(
                        "Re-run generate_course.py with --objectives pointing "
                        "to the canonical exam-objectives JSON."
                    ),
                )
            )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=not failures,
            score=1.0 if not failures else max(0.0, 1.0 - len(failures) * 0.1),
            issues=issues,
        )
```

`lib/validators/__init__.py` gets the new validator added to its export list.

---

## 5. Workflow gate YAML entries

Add to `config/workflows.yaml`:

### Under `course_generation` → `packaging` phase `validation_gates:`

```yaml
      - name: packaging
        ...
        validation_gates:
          - gate_id: imscc_structure
            validator: lib.validators.imscc.IMSCCValidator
            severity: critical
            threshold:
              max_critical_issues: 0
            behavior:
              on_fail: block
              on_error: fail_closed

          - gate_id: page_objectives
            validator: lib.validators.page_objectives.PageObjectivesValidator
            severity: critical
            threshold:
              max_critical_issues: 0
            behavior:
              on_fail: block
              on_error: fail_closed
```

### Under `textbook_to_course` → `packaging` phase `validation_gates:`

Same entry. (textbook_to_course uses `severity: warning` for the existing imscc_structure gate — new `page_objectives` entry is still `severity: critical` because LO fanout silently caps Trainforge quality metrics; we want to block.)

Both entries match the format/indentation of existing sibling gates exactly (2-space nested under `validation_gates:`, keys in the same order).

---

## 6. Test design — `Courseforge/scripts/tests/test_packager_default.py`

Five tests, all using the existing fixture shape from `test_packager_validation_gate.py` (same `_page_html`, similar `content_dir` and `objectives_path` fixtures but adapted for the new defaults).

1. **`test_validation_runs_by_default_with_auto_discovery`**
   - Create content dir with valid week_01 page + `course.json` at content root (copy of the objectives fixture).
   - Call `package_imscc(...)` WITHOUT passing `objectives_path`.
   - Assert auto-discovery message printed, validation ran, package produced.

2. **`test_skip_validation_bypasses`**
   - Content dir with a VIOLATING page + `course.json` at root.
   - Call with `skip_validation=True`.
   - Assert validation skipped (SKIPPED message), package produced successfully.

3. **`test_validation_fails_on_broken_lo`**
   - Content dir with a violating page (canonical LOs from a different week).
   - Call WITHOUT `objectives_path` but WITH `course.json` auto-discoverable.
   - Assert `SystemExit(2)` raised, package NOT produced.

4. **`test_no_objectives_no_autodiscovery_warns`**
   - Content dir WITHOUT `course.json` and WITHOUT `objectives_path` argument.
   - Call `package_imscc(...)`.
   - Assert WARNING message printed, package produced successfully (no raise).
   - This preserves backward-compat for callers that never used the flag.

5. **`test_page_objectives_validator_returns_validation_result`**
   - Direct test of `PageObjectivesValidator().validate({...})` from `lib.validators.page_objectives`.
   - Clean content: assert `result.passed is True`, `result.critical_count == 0`.
   - Violating content: assert `result.passed is False`, at least one issue with code `LO_SPECIFICITY_VIOLATION`.
   - No-objectives content: assert `result.passed is True` with a `NO_OBJECTIVES_FILE` warning.

---

## 7. Verification steps

1. `python3 -m ci.integrity_check` passes 8/8.
2. `pytest Courseforge/scripts/tests/test_packager_default.py -x` green (5 new tests).
3. `pytest Courseforge/scripts/tests/test_packager_validation_gate.py -x` still green (existing test coverage — including `test_no_objectives_arg_skips_validation_entirely` which is now OBSOLETE and must be updated to reflect auto-discovery OR become the "no course.json and no objectives arg" path that warns + proceeds. This test needs edit to match new behavior.).
4. `python3 -c "import yaml; yaml.safe_load(open('config/workflows.yaml'))"` parses cleanly.
5. `python3 -c "from lib.validators.page_objectives import PageObjectivesValidator; print(PageObjectivesValidator().name, PageObjectivesValidator().version)"` imports and instantiates.

### Note on existing test `test_no_objectives_arg_skips_validation_entirely`

Pre-Worker-L behavior this test locks in: "callers that don't pass `--objectives` get NO validation." After Worker L:
- If `content_dir/course.json` exists, validation RUNS.
- If not, a warning is printed and packaging proceeds.

The test as written creates a content dir with NO `course.json` at root, so its behavior is still "validation skipped, packaging succeeds" — but with an added warning line. The test should continue to pass (it only asserts `output.exists()`), no edit needed. Verify this holds during implementation.

---

## 8. Files touched

**New:**
- `lib/validators/page_objectives.py`
- `Courseforge/scripts/tests/test_packager_default.py`
- `plans/kg-quality-review-2026-04/worker-l-subplan.md` (this file)

**Modified:**
- `Courseforge/scripts/package_multifile_imscc.py` (package_imscc default flip + CLI help text)
- `config/workflows.yaml` (2 × page_objectives gate entries)
- `lib/validators/__init__.py` (add `PageObjectivesValidator` export)

**Not touched (per master plan constraints):**
- `Courseforge/scripts/generate_course.py` (Workers J + K)
- `Trainforge/*` (Worker J)
- `schemas/taxonomies/*` (Worker F already published)
- existing `lib/validators/*.py` files
