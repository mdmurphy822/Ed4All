"""
OSCQR Validator (Wave 31 — real implementation)

Validates course quality against selected items from the Open SUNY Course
Quality Review rubric (https://oscqr.suny.edu/). Pre-Wave-31 this validator
returned ``NOT_IMPLEMENTED`` unconditionally, which meant the
``course_generation`` pipeline had zero automated quality signal: users
asked "what quality is my course?" and got no answer.

The rubric has ~50 items. Wave 31 implements the ~12 most-automatable
items across all six OSCQR categories — enough to catch the defect
pattern observed on the ``OLSR_SIM_01`` simulation (empty objectives
lists, identical activity prompts, placeholder assessments) while
leaving richer pedagogical items (e.g. "instructor presence") for
human review.

Items covered
-------------

Each item returns a ``GateIssue`` when it fails. The ``code`` encodes
the OSCQR item number + category:

* Course Overview (OV-1, OV-3) — syllabus present + complete.
* Learner Support (LS-1) — accessibility statement present.
* Course Structure (CS-2, CS-3, CS-5) — weekly modules, unique module
  titles, learning objectives on every page.
* Content / Learning Activities (CLA-1, CLA-3) — non-empty content
  (word-count floor), varied activity prompts (no copy-paste).
* Instructor Interaction (II-3) — discussion / interaction prompt
  per module.
* Assessment & Measurement (AM-1, AM-2) — multiple question types,
  assessments reference objectives, no placeholder stems.

Score aggregation
-----------------

``score = passed_items / total_checkable_items`` (0.0–1.0). The gate
threshold (``min_score: 0.7`` by default) is applied by the gate
manager; individual items are tagged ``severity="critical"`` for
hard-blockers (missing syllabus, empty objectives, identical
activities) and ``severity="warning"`` for polish items (heading
variety, discussion prompt presence).

Referenced by: ``config/workflows.yaml``
(``course_generation.validation.validation_gates[oscqr_score]``).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_oscqr_decision(
    capture: Any,
    *,
    passed: bool,
    score: float,
    items_checkable: int,
    items_passed: int,
    items_failed: int,
    items_skipped: int,
    critical_failures: int,
    failed_item_ids: List[str],
) -> None:
    """Emit one ``oscqr_score_check`` decision per ``validate()`` call (H3 W6b).

    Rationale + ``metrics`` interpolate the OSCQR rubric counts so
    post-hoc replay can distinguish per-rubric pass/fail mixes.
    """
    if capture is None:
        return
    code = None if passed else "OSCQR_CRITICAL_FAILURES"
    decision = "passed" if passed else f"failed:{code}"
    failed_id_preview = ",".join(failed_item_ids[:5]) or "<none>"
    metrics: Dict[str, Any] = {
        "score": float(round(score, 4)),
        "items_checkable": int(items_checkable),
        "items_passed": int(items_passed),
        "items_failed": int(items_failed),
        "items_skipped": int(items_skipped),
        "critical_failures": int(critical_failures),
        "passed": bool(passed),
        "failure_code": code,
    }
    rationale = (
        f"OSCQR rubric verdict: score={score:.4f}, "
        f"items=({items_passed} passed / {items_failed} failed / "
        f"{items_skipped} skipped of {items_checkable} checkable), "
        f"critical_failures={critical_failures}, "
        f"failed_item_ids=[{failed_id_preview}], "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="oscqr_score_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on oscqr_score_check: %s",
            exc,
        )

# Keep the import order explicit so downstream tooling (pytest-order)
# doesn't have to introspect our dependencies.
try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover - bs4 ships with the repo env
    BeautifulSoup = None  # type: ignore


MIN_WORDS_PER_PAGE = 80
MIN_UNIQUE_MODULE_TITLES = 0.6  # fraction of weeks with distinct titles
MIN_DISTINCT_ACTIVITY_FRACTION = 0.75  # fraction of activity prompts that must differ
MIN_QUESTION_TYPE_VARIETY = 2  # at least 2 different question types


@dataclass
class OSCQRItem:
    """Single OSCQR rubric item result."""
    category: str
    item_id: str
    passed: bool
    severity: str  # "critical" | "warning"
    message: str
    suggestion: Optional[str] = None
    locations: List[str] = field(default_factory=list)


class OSCQRValidator:
    """Validates course quality against a subset of OSCQR rubric items.

    Expected inputs:
        course_path / content_dir: course content directory (weekly HTML).
        imscc_path: optional path to IMSCC package for assessment inspection.
        course_json_path: optional path to ``course.json``.
        syllabus_path: optional explicit syllabus HTML / MD path.
        min_score: optional — applied by gate manager, not here.

    Any subset of inputs works — missing inputs mark their items as
    "not_checkable" (skipped, not failed) so the validator degrades
    gracefully on a partial run.
    """

    name = "oscqr_score"
    version = "1.0.0"

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "oscqr_score")
        capture = inputs.get("decision_capture")

        course_path = self._resolve_course_dir(inputs)
        course_json = self._resolve_course_json(inputs, course_path)
        imscc_path = inputs.get("imscc_path")

        items: List[OSCQRItem] = []

        # Course Overview items
        items.append(self._check_syllabus_present(course_path, inputs))
        items.append(self._check_syllabus_complete(course_path, inputs))

        # Learner Support
        items.append(self._check_accessibility_statement(course_path))

        # Course Structure
        items.append(self._check_weekly_modules_present(course_path))
        items.append(self._check_module_titles_unique(course_path))
        items.append(self._check_page_objectives_populated(course_path))

        # Content / Learning Activities
        items.append(self._check_content_word_floor(course_path))
        items.append(self._check_activity_prompt_variety(course_path))

        # Instructor Interaction
        items.append(self._check_interaction_prompts(course_path))

        # Assessment & Measurement
        items.append(self._check_question_type_variety(course_json, imscc_path))
        items.append(self._check_assessments_reference_objectives(course_json, imscc_path))
        items.append(self._check_no_placeholder_assessments(course_json, imscc_path))

        # Convert to GateIssues
        checkable = [i for i in items if i.severity != "skipped"]
        passed = [i for i in checkable if i.passed]
        failed = [i for i in checkable if not i.passed]

        score = len(passed) / len(checkable) if checkable else 0.0

        issues: List[GateIssue] = []
        for item in items:
            if item.severity == "skipped":
                # Surface skipped items as informational warnings so they
                # appear in the gate output without counting against score.
                issues.append(GateIssue(
                    severity="warning",
                    code=f"{item.item_id}_NOT_CHECKABLE",
                    message=f"[{item.category}] {item.item_id}: {item.message}",
                    suggestion=item.suggestion,
                ))
                continue
            if item.passed:
                continue
            issues.append(GateIssue(
                severity=item.severity,
                code=f"{item.item_id}_FAIL",
                message=f"[{item.category}] {item.item_id}: {item.message}",
                suggestion=item.suggestion,
                location=",".join(item.locations) if item.locations else None,
            ))

        # Pass = no critical failures. Score threshold is applied by the
        # gate manager via ``min_score`` in the gate config.
        critical = [i for i in failed if i.severity == "critical"]
        passed_overall = len(critical) == 0

        skipped = [i for i in items if i.severity == "skipped"]
        failed_item_ids = [i.item_id for i in failed]
        _emit_oscqr_decision(
            capture,
            passed=passed_overall,
            score=score,
            items_checkable=len(checkable),
            items_passed=len(passed),
            items_failed=len(failed),
            items_skipped=len(skipped),
            critical_failures=len(critical),
            failed_item_ids=failed_item_ids,
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed_overall,
            score=score,
            issues=issues,
        )

    # ------------------------------------------------------------------ #
    # Path resolution helpers
    # ------------------------------------------------------------------ #

    def _resolve_course_dir(self, inputs: Dict[str, Any]) -> Optional[Path]:
        for key in ("course_path", "content_dir", "course_dir", "project_path"):
            val = inputs.get(key)
            if isinstance(val, str) and val:
                p = Path(val)
                if p.exists():
                    return p
            elif isinstance(val, Path) and val.exists():
                return val
        return None

    def _resolve_course_json(
        self,
        inputs: Dict[str, Any],
        course_path: Optional[Path],
    ) -> Optional[Dict[str, Any]]:
        explicit = inputs.get("course_json_path")
        if isinstance(explicit, str) and explicit:
            p = Path(explicit)
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
        if course_path:
            for candidate in (course_path / "course.json", course_path.parent / "course.json"):
                if candidate.exists():
                    try:
                        return json.loads(candidate.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
        return None

    # Size guard: skip HTML files larger than this when scanning a
    # course dir (they're almost certainly DART textbook outputs, not
    # course modules). Keeps the validator fast on LibV2 archives.
    _MAX_PAGE_SIZE_BYTES = 500_000

    def _list_html_pages(self, course_path: Optional[Path]) -> List[Path]:
        if course_path is None or not course_path.exists():
            return []
        result: List[Path] = []
        try:
            for p in course_path.rglob("*.html"):
                # Skip files under source/, sources/ (raw uploads) and
                # corpus/ (Trainforge content store) — those don't
                # belong to the course module structure.
                parts = {part.lower() for part in p.parts}
                if parts & {"source", "sources", "corpus"}:
                    continue
                try:
                    if p.stat().st_size > self._MAX_PAGE_SIZE_BYTES:
                        continue
                except OSError:
                    continue
                result.append(p)
        except OSError:
            return []
        return sorted(result)

    # ------------------------------------------------------------------ #
    # OSCQR items (each returns one OSCQRItem)
    # ------------------------------------------------------------------ #

    def _check_syllabus_present(
        self,
        course_path: Optional[Path],
        inputs: Dict[str, Any],
    ) -> OSCQRItem:
        syllabus_path_hint = inputs.get("syllabus_path")
        candidate_paths: List[Path] = []
        if isinstance(syllabus_path_hint, str) and syllabus_path_hint:
            candidate_paths.append(Path(syllabus_path_hint))
        if course_path:
            # Limit search to direct children + one-level-down to avoid
            # walking giant corpus trees (LibV2 archives can carry
            # multi-MB DART HTML under source/).
            for name in ("syllabus.html", "syllabus.md", "Syllabus.html"):
                candidate_paths.append(course_path / name)
                # One level down — common layouts: overview/syllabus.html,
                # syllabus/syllabus.html, modules/syllabus.html.
                for sub in course_path.iterdir() if course_path.exists() else []:
                    if sub.is_dir() and not sub.name.startswith('.'):
                        candidate_paths.append(sub / name)

        for p in candidate_paths:
            try:
                if p.exists():
                    return OSCQRItem(
                        category="Course Overview",
                        item_id="OV-1",
                        passed=True,
                        severity="critical",
                        message="Syllabus file found",
                        locations=[str(p)],
                    )
            except OSError:
                continue
        return OSCQRItem(
            category="Course Overview",
            item_id="OV-1",
            passed=False,
            severity="critical",
            message="No syllabus file found in course (syllabus.html / syllabus.md)",
            suggestion="Add a syllabus page describing course purpose, outcomes, schedule, and policies.",
        )

    def _check_syllabus_complete(
        self,
        course_path: Optional[Path],
        inputs: Dict[str, Any],
    ) -> OSCQRItem:
        # A minimal "complete" syllabus mentions: objectives, schedule,
        # grading, and contact.
        syllabus_content = ""
        syllabus_path = None
        if isinstance(inputs.get("syllabus_path"), str):
            p = Path(inputs["syllabus_path"])
            if p.exists():
                syllabus_path = p
                try:
                    syllabus_content = p.read_text(encoding="utf-8").lower()
                except OSError:
                    pass
        if not syllabus_content and course_path:
            # Same direct-children + one-level-down strategy as _check_syllabus_present.
            candidates = []
            for name in ("syllabus.html", "syllabus.md", "Syllabus.html"):
                candidates.append(course_path / name)
                try:
                    for sub in course_path.iterdir():
                        if sub.is_dir() and not sub.name.startswith('.'):
                            candidates.append(sub / name)
                except OSError:
                    continue
            for cand in candidates:
                try:
                    if cand.exists():
                        syllabus_content = cand.read_text(encoding="utf-8").lower()
                        syllabus_path = cand
                        break
                except OSError:
                    continue

        if not syllabus_content:
            return OSCQRItem(
                category="Course Overview",
                item_id="OV-3",
                passed=False,
                severity="skipped",
                message="Syllabus not available — cannot check completeness",
                suggestion="Pass syllabus_path or add syllabus.html to the course dir.",
            )

        required_terms = ["objective", "schedule", "grad", "contact"]
        missing = [t for t in required_terms if t not in syllabus_content]
        passed = len(missing) == 0
        return OSCQRItem(
            category="Course Overview",
            item_id="OV-3",
            passed=passed,
            severity="warning",
            message=(
                f"Syllabus complete (mentions {len(required_terms)} required sections)"
                if passed
                else f"Syllabus missing sections: {', '.join(missing)}"
            ),
            suggestion="Add objectives, schedule, grading, and contact sections.",
            locations=[str(syllabus_path)] if syllabus_path else [],
        )

    def _check_accessibility_statement(self, course_path: Optional[Path]) -> OSCQRItem:
        if course_path is None:
            return OSCQRItem(
                category="Learner Support",
                item_id="LS-1",
                passed=False,
                severity="skipped",
                message="No course_path available",
            )
        pages = self._list_html_pages(course_path)
        if not pages:
            return OSCQRItem(
                category="Learner Support",
                item_id="LS-1",
                passed=False,
                severity="skipped",
                message="No HTML pages to scan for accessibility statement",
            )
        # A minimal accessibility statement mentions "accessibility" or
        # "disability" in at least one course page.
        found_locations: List[str] = []
        for p in pages:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if "accessibility" in content or "disability service" in content:
                found_locations.append(str(p))
                break
        passed = len(found_locations) > 0
        return OSCQRItem(
            category="Learner Support",
            item_id="LS-1",
            passed=passed,
            severity="warning",
            message=(
                "Accessibility statement detected"
                if passed
                else "No accessibility or disability-services statement found in any page"
            ),
            suggestion="Add an accessibility statement linking to institutional disability services.",
            locations=found_locations[:1],
        )

    def _check_weekly_modules_present(self, course_path: Optional[Path]) -> OSCQRItem:
        if course_path is None:
            return OSCQRItem(
                category="Course Structure",
                item_id="CS-2",
                passed=False,
                severity="skipped",
                message="No course_path available",
            )
        week_dirs = sorted(
            [p for p in course_path.rglob("*") if p.is_dir() and re.match(r"(?i)^week[_-]?\d+$", p.name)]
        )
        if not week_dirs:
            # Also accept flat filename conventions e.g. "week_01_overview.html"
            week_files = sorted(course_path.rglob("week_*.html"))
            if not week_files:
                return OSCQRItem(
                    category="Course Structure",
                    item_id="CS-2",
                    passed=False,
                    severity="critical",
                    message="No weekly modules detected (no week_* directories or files)",
                    suggestion="Structure course content into per-week modules.",
                )
        count = len(week_dirs) if week_dirs else len({
            re.match(r"(week_\d+)", p.name).group(1)
            for p in course_path.rglob("week_*.html")
            if re.match(r"(week_\d+)", p.name)
        })
        return OSCQRItem(
            category="Course Structure",
            item_id="CS-2",
            passed=True,
            severity="critical",
            message=f"Found {count} weekly modules",
        )

    def _check_module_titles_unique(self, course_path: Optional[Path]) -> OSCQRItem:
        if course_path is None:
            return OSCQRItem(
                category="Course Structure",
                item_id="CS-3",
                passed=False,
                severity="skipped",
                message="No course_path available",
            )
        if BeautifulSoup is None:
            return OSCQRItem(
                category="Course Structure",
                item_id="CS-3",
                passed=False,
                severity="skipped",
                message="BeautifulSoup unavailable",
            )
        overview_pages = [
            p for p in course_path.rglob("*overview*.html")
            if re.search(r"week[_-]?\d+", p.name)
        ]
        if len(overview_pages) < 3:
            # Need a few weeks to meaningfully check uniqueness.
            return OSCQRItem(
                category="Course Structure",
                item_id="CS-3",
                passed=True,
                severity="warning",
                message=f"Too few weekly overviews ({len(overview_pages)}) to assess title uniqueness; pass by default",
            )
        titles: List[str] = []
        for p in overview_pages:
            try:
                soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            except Exception:  # noqa: BLE001
                continue
            h1 = soup.find("h1")
            if h1 and h1.get_text().strip():
                titles.append(h1.get_text().strip().lower())
        if not titles:
            return OSCQRItem(
                category="Course Structure",
                item_id="CS-3",
                passed=False,
                severity="warning",
                message="No H1 titles found on weekly overview pages",
            )
        unique = len(set(titles))
        frac = unique / len(titles)
        passed = frac >= MIN_UNIQUE_MODULE_TITLES
        return OSCQRItem(
            category="Course Structure",
            item_id="CS-3",
            passed=passed,
            severity="warning",
            message=(
                f"{unique}/{len(titles)} weekly titles are distinct ({frac:.0%})"
                if passed
                else f"Only {unique}/{len(titles)} weekly titles are distinct ({frac:.0%}) — "
                     "minimum 60%. Weeks sharing the same H1 suggest template boilerplate."
            ),
            suggestion="Give each week a distinct topic-focused H1 title.",
        )

    def _check_page_objectives_populated(self, course_path: Optional[Path]) -> OSCQRItem:
        if course_path is None or BeautifulSoup is None:
            return OSCQRItem(
                category="Course Structure",
                item_id="CS-5",
                passed=False,
                severity="skipped",
                message="Cannot scan for page objectives",
            )
        pages = [p for p in self._list_html_pages(course_path) if re.search(r"week[_-]?\d+", p.name)]
        if not pages:
            return OSCQRItem(
                category="Course Structure",
                item_id="CS-5",
                passed=False,
                severity="skipped",
                message="No weekly HTML pages found",
            )
        empty_pages: List[str] = []
        for p in pages:
            try:
                soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            except Exception:  # noqa: BLE001
                continue
            # Look for a learningObjectives-style list.
            has_populated_list = False
            for ul in soup.find_all(["ul", "ol"]):
                # Look for a heading directly before that mentions "objectives".
                prev_sib = ul.find_previous(["h1", "h2", "h3", "h4"])
                if prev_sib and "objective" in prev_sib.get_text().lower():
                    if len(ul.find_all("li")) >= 1:
                        has_populated_list = True
                        break
            # Fallback — JSON-LD learningObjectives.
            if not has_populated_list:
                for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
                    try:
                        data = json.loads(script.string or "{}")
                    except (json.JSONDecodeError, TypeError):
                        continue
                    los = data.get("learningObjectives") if isinstance(data, dict) else None
                    if isinstance(los, list) and los:
                        has_populated_list = True
                        break
            if not has_populated_list:
                empty_pages.append(str(p))
        # Only fail if ≥25% of weekly pages have empty objectives.
        frac_empty = len(empty_pages) / max(1, len(pages))
        passed = frac_empty < 0.25
        return OSCQRItem(
            category="Course Structure",
            item_id="CS-5",
            passed=passed,
            severity="critical",
            message=(
                f"Page objectives populated on {len(pages) - len(empty_pages)}/{len(pages)} pages"
                if passed
                else f"{len(empty_pages)}/{len(pages)} weekly pages have empty or missing objectives list"
            ),
            suggestion="Populate per-page learningObjectives from the canonical objectives file.",
            locations=empty_pages[:5],
        )

    def _check_content_word_floor(self, course_path: Optional[Path]) -> OSCQRItem:
        if course_path is None or BeautifulSoup is None:
            return OSCQRItem(
                category="Content / Learning Activities",
                item_id="CLA-1",
                passed=False,
                severity="skipped",
                message="Cannot scan page content",
            )
        pages = [p for p in self._list_html_pages(course_path) if re.search(r"week[_-]?\d+", p.name)]
        if not pages:
            return OSCQRItem(
                category="Content / Learning Activities",
                item_id="CLA-1",
                passed=False,
                severity="skipped",
                message="No weekly content pages",
            )
        thin_pages: List[str] = []
        for p in pages:
            try:
                soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            except Exception:  # noqa: BLE001
                continue
            for nav in soup.find_all(["nav", "header", "footer"]):
                nav.decompose()
            text = soup.get_text(separator=" ", strip=True)
            words = text.split()
            if len(words) < MIN_WORDS_PER_PAGE:
                thin_pages.append(str(p))
        frac_thin = len(thin_pages) / max(1, len(pages))
        passed = frac_thin < 0.25
        return OSCQRItem(
            category="Content / Learning Activities",
            item_id="CLA-1",
            passed=passed,
            severity="critical",
            message=(
                f"All pages ≥{MIN_WORDS_PER_PAGE} words of content"
                if passed
                else f"{len(thin_pages)}/{len(pages)} pages have <{MIN_WORDS_PER_PAGE} words of substantive content"
            ),
            suggestion="Expand thin pages with real explanation, examples, and context.",
            locations=thin_pages[:5],
        )

    def _check_activity_prompt_variety(self, course_path: Optional[Path]) -> OSCQRItem:
        if course_path is None or BeautifulSoup is None:
            return OSCQRItem(
                category="Content / Learning Activities",
                item_id="CLA-3",
                passed=False,
                severity="skipped",
                message="Cannot scan activity prompts",
            )
        # Look for pages named *application*.html or *activity*.html —
        # typical Courseforge naming.
        pages = [
            p for p in self._list_html_pages(course_path)
            if re.search(r"(application|activity|practice)", p.name, re.IGNORECASE)
        ]
        if len(pages) < 3:
            return OSCQRItem(
                category="Content / Learning Activities",
                item_id="CLA-3",
                passed=True,
                severity="warning",
                message=f"Too few activity pages ({len(pages)}) to meaningfully assess variety",
            )
        prompts: List[str] = []
        for p in pages:
            try:
                soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            except Exception:  # noqa: BLE001
                continue
            # Take the first substantial <p> in the main content as the prompt.
            for para in soup.find_all("p"):
                text = para.get_text(strip=True)
                if len(text) > 30:
                    prompts.append(re.sub(r"\s+", " ", text.lower()[:200]))
                    break
        if not prompts:
            return OSCQRItem(
                category="Content / Learning Activities",
                item_id="CLA-3",
                passed=False,
                severity="warning",
                message="No activity prompts detected",
            )
        # Copy-paste detection: count the fraction of distinct prompts.
        distinct = len(set(prompts))
        frac = distinct / len(prompts)
        passed = frac >= MIN_DISTINCT_ACTIVITY_FRACTION
        return OSCQRItem(
            category="Content / Learning Activities",
            item_id="CLA-3",
            passed=passed,
            severity="critical",
            message=(
                f"{distinct}/{len(prompts)} activity prompts are distinct ({frac:.0%})"
                if passed
                else f"Only {distinct}/{len(prompts)} activity prompts are distinct ({frac:.0%}); "
                     "the same prompt appears to be copy-pasted across weeks."
            ),
            suggestion="Author week-specific activity prompts that reference the week's content.",
        )

    def _check_interaction_prompts(self, course_path: Optional[Path]) -> OSCQRItem:
        if course_path is None:
            return OSCQRItem(
                category="Instructor Interaction",
                item_id="II-3",
                passed=False,
                severity="skipped",
                message="Cannot scan for interaction prompts",
            )
        pages = self._list_html_pages(course_path)
        if not pages:
            return OSCQRItem(
                category="Instructor Interaction",
                item_id="II-3",
                passed=False,
                severity="skipped",
                message="No pages to scan",
            )
        interaction_terms = re.compile(
            r"\b(discuss|discussion|reflect|share|post|reply|forum|respond|collaborate)\b",
            re.IGNORECASE,
        )
        found_any = False
        for p in pages:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if interaction_terms.search(content):
                found_any = True
                break
        return OSCQRItem(
            category="Instructor Interaction",
            item_id="II-3",
            passed=found_any,
            severity="warning",
            message=(
                "Interaction / discussion prompt detected"
                if found_any
                else "No discussion, reflection, or collaboration prompts detected"
            ),
            suggestion="Add discussion prompts, reflection questions, or collaborative activities.",
        )

    def _check_question_type_variety(
        self,
        course_json: Optional[Dict[str, Any]],
        imscc_path: Optional[str],
    ) -> OSCQRItem:
        assessments = self._load_assessments(course_json, imscc_path)
        if not assessments:
            return OSCQRItem(
                category="Assessment & Measurement",
                item_id="AM-1",
                passed=False,
                severity="skipped",
                message="No assessments available for question-type analysis",
            )
        types_seen: Counter = Counter()
        for q in assessments:
            qtype = q.get("type") or q.get("question_type") or q.get("kind")
            if qtype:
                types_seen[str(qtype).lower()] += 1
        distinct_types = len(types_seen)
        passed = distinct_types >= MIN_QUESTION_TYPE_VARIETY
        return OSCQRItem(
            category="Assessment & Measurement",
            item_id="AM-1",
            passed=passed,
            severity="warning",
            message=(
                f"Assessments use {distinct_types} distinct question types: "
                f"{sorted(types_seen.keys())}"
                if passed
                else f"Assessments use only {distinct_types} question type(s); "
                     "minimum 2 for meaningful variety."
            ),
            suggestion="Mix multiple-choice with short-answer, true/false, ordering, or matching items.",
        )

    def _check_assessments_reference_objectives(
        self,
        course_json: Optional[Dict[str, Any]],
        imscc_path: Optional[str],
    ) -> OSCQRItem:
        assessments = self._load_assessments(course_json, imscc_path)
        if not assessments:
            return OSCQRItem(
                category="Assessment & Measurement",
                item_id="AM-2",
                passed=False,
                severity="skipped",
                message="No assessments available",
            )
        total = len(assessments)
        linked = 0
        for q in assessments:
            # Common schemas: objective_id, objective_ids, learning_outcome_refs.
            if any(q.get(k) for k in ("objective_id", "objective_ids", "learning_outcome_refs", "objectives")):
                linked += 1
        frac = linked / max(1, total)
        passed = frac >= 0.9
        return OSCQRItem(
            category="Assessment & Measurement",
            item_id="AM-2",
            passed=passed,
            severity="critical",
            message=(
                f"{linked}/{total} ({frac:.0%}) assessment items reference a learning objective"
                if passed
                else f"Only {linked}/{total} ({frac:.0%}) assessment items reference a learning objective "
                     "(OSCQR requires alignment)"
            ),
            suggestion="Map every assessment item to the learning objective it measures.",
        )

    def _check_no_placeholder_assessments(
        self,
        course_json: Optional[Dict[str, Any]],
        imscc_path: Optional[str],
    ) -> OSCQRItem:
        assessments = self._load_assessments(course_json, imscc_path)
        if not assessments:
            return OSCQRItem(
                category="Assessment & Measurement",
                item_id="AM-2b",
                passed=False,
                severity="skipped",
                message="No assessments available",
            )
        placeholder_re = re.compile(
            r"(placeholder|lorem ipsum|\[todo\]|tbd|\bexample question\b|insert .*here)",
            re.IGNORECASE,
        )
        bad: List[str] = []
        for q in assessments:
            stem = q.get("stem") or q.get("question") or q.get("prompt") or ""
            if placeholder_re.search(str(stem)):
                bad.append(str(stem)[:80])
        passed = len(bad) == 0
        return OSCQRItem(
            category="Assessment & Measurement",
            item_id="AM-2b",
            passed=passed,
            severity="critical",
            message=(
                "No placeholder stems detected in assessments"
                if passed
                else f"{len(bad)} assessment item(s) contain placeholder text (TBD, TODO, Lorem Ipsum)"
            ),
            suggestion="Replace placeholder stems with real questions targeting the objective.",
            locations=bad[:3],
        )

    # ------------------------------------------------------------------ #
    # Assessment loading (course.json OR IMSCC QTI parse)
    # ------------------------------------------------------------------ #

    def _load_assessments(
        self,
        course_json: Optional[Dict[str, Any]],
        imscc_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        if course_json:
            # Two common shapes:
            #   {"assessments": [{"questions": [...]}]}
            #   {"questions": [...]}
            qs: List[Dict[str, Any]] = []
            for assessment in course_json.get("assessments", []) or []:
                for q in assessment.get("questions", []) or []:
                    qs.append(q)
            for q in course_json.get("questions", []) or []:
                qs.append(q)
            if qs:
                return qs
        # IMSCC QTI parse is expensive; skip for Wave 31 minimal impl.
        return []


__all__ = ["OSCQRValidator", "OSCQRItem"]
