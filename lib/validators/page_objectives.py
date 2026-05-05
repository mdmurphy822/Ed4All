"""
Page Objectives Validator (Worker L — REC-CTR-03)

Wraps the per-week learningObjectives gate that lives in
``Courseforge/scripts/validate_page_objectives.py`` so the orchestrator's
validation-gate framework can invoke it as a first-class workflow gate
configured in ``config/workflows.yaml``.

Behavior contract:
    - If ``objectives_path`` is provided, validate against it.
    - Otherwise auto-discover ``content_dir / "course.json"``.
    - If no objectives source is available, return ``passed=True`` with a
      single warning issue — backward-compat for callers that never wired
      an objectives file. Hard-fail (``passed=False`` with critical issues)
      only occurs on a genuine LO-specificity violation.

Referenced by: config/workflows.yaml (course_generation, textbook_to_course)
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult


def _load_page_objectives_helpers():
    """Lazily load ``validate_page_objectives`` from ``Courseforge/scripts/``.

    The helpers live in a script directory that is not a normal python
    package. Load the module via ``importlib`` so the validator doesn't
    require sys.path surgery at import time.

    Raises:
        FileNotFoundError: if the expected script layout has moved. The
            error is wrapped into a ``GateResult`` by the caller.
    """
    scripts_dir = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
    module_path = scripts_dir / "validate_page_objectives.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Expected validate_page_objectives.py at {module_path}; "
            "PageObjectivesValidator cannot run without it."
        )
    # validate_page_objectives imports load_canonical_objectives +
    # resolve_week_objectives from generate_course at module-load time, so
    # the scripts dir must be importable before the module executes.
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "cf_validate_page_objectives", module_path
    )
    if spec is None or spec.loader is None:
        raise FileNotFoundError(
            f"Could not load spec for {module_path}."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PageObjectivesValidator:
    """Validates per-week ``learningObjectives`` JSON-LD against canonical objectives."""

    name = "page_objectives"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate per-week LO specificity for generated HTML pages.

        Expected inputs:
            content_dir: Path (or str) to course content root with
                ``week_*`` subdirectories.
            objectives_path: Optional path to canonical objectives JSON.
                If absent, auto-discover ``content_dir / "course.json"``.
            gate_id: Optional override for result ``gate_id`` (defaults to
                ``"page_objectives"``).
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

        # Resolve objectives_path (explicit -> auto-discover -> skip-with-warning)
        objectives_path_raw = inputs.get("objectives_path")
        objectives_path: Optional[Path] = (
            Path(objectives_path_raw) if objectives_path_raw else None
        )
        if objectives_path is None:
            candidate = content_dir / "course.json"
            if candidate.exists():
                objectives_path = candidate

        if objectives_path is None:
            # Silent-degradation guard (page_objectives audit fix): the
            # gate is wired ``critical`` on both ``course_generation`` and
            # ``textbook_to_course`` packaging phases (see
            # ``config/workflows.yaml::validation_gates``), so a run that
            # reaches this validator without either an explicit
            # ``objectives_path`` OR an auto-discoverable
            # ``content_dir / course.json`` represents an upstream
            # contract failure (course-planning didn't emit objectives,
            # or packaging didn't surface course.json). The pre-fix
            # branch returned ``passed=True`` with a "backward-compat"
            # warning, which silently skipped per-week LO validation on
            # a critical-severity gate. Inverting to fail-closed surfaces
            # the failure at the gate that's already declared as
            # blocking, rather than letting the run ship a course whose
            # per-page LO specificity was never checked.
            candidate_path = str(content_dir / "course.json")
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="PAGE_OBJECTIVES_PATH_MISSING",
                        message=(
                            "No objectives file provided and no course.json "
                            f"found at {candidate_path}. Did the upstream "
                            "course_planning / packaging phase run and emit "
                            "course.json? Cannot validate per-week LO "
                            "specificity without a canonical objectives "
                            "source."
                        ),
                        suggestion=(
                            "Pass objectives_path explicitly or ensure "
                            "course.json is emitted at the content dir "
                            "root by the packaging phase."
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

        # Lazy import the Courseforge helpers.
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

        issues: List[GateIssue] = []
        fail_count = 0
        for page in pages:
            # Only validate week_* pages; project docs and non-week HTML aren't
            # expected to carry LO metadata.
            if not any(part.startswith("week_") for part in page.parts):
                continue
            ok, msg = helpers.validate_page(page, canonical)
            if not ok:
                fail_count += 1
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="LO_SPECIFICITY_VIOLATION",
                        message=msg,
                        location=str(page),
                        suggestion=(
                            "Re-run generate_course.py with --objectives "
                            "pointing to the canonical exam-objectives JSON."
                        ),
                    )
                )

        passed = fail_count == 0
        score = 1.0 if passed else max(0.0, 1.0 - fail_count * 0.1)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )
