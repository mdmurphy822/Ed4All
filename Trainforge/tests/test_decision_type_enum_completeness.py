"""Wave 22 DC2 regression — Trainforge decision_type emits must match the schema enum.

Pre-Wave-22, Trainforge emitted five ``decision_type`` values that
were not in the canonical enum at
``schemas/events/decision_event.schema.json`` — ``assessment_planning``,
``question_type_selection``, ``assessment_generation``,
``content_selection``, and ``boilerplate_strip``. 49% of decision
records from a recent run carried ``metadata.validation_issues`` as a
result, and the orchestrator papered over the landmine by force-
disabling ``DECISION_VALIDATION_STRICT`` for the duration of the
Trainforge call (since removed in Wave 22).

This test scrapes the actual string literals that follow
``decision_type=`` in the top emit sites (assessment_generator,
process_course, and the Courseforge→Trainforge stage emitting
``content_selection`` from ``MCP/tools/pipeline_tools.py``) and asserts
each is in the canonical enum. When a new emit site appears, this test
fails fast so the schema can be updated in the same PR.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "events"
    / "decision_event.schema.json"
)


def _load_enum() -> set:
    """Return the current ``decision_type`` enum as a set."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    return set(schema["properties"]["decision_type"]["enum"])


# Regex matches ``decision_type="..."`` or ``decision_type='...'`` even
# when split across lines by a trailing backslash or standalone newline.
_DECISION_TYPE_RE = re.compile(
    r"decision_type\s*=\s*[\"']([A-Za-z0-9_-]+)[\"']"
)


def _scrape_decision_types(source: Path) -> set:
    """Return every literal string bound to ``decision_type=`` in ``source``."""
    if not source.exists():
        return set()
    text = source.read_text(encoding="utf-8", errors="replace")
    return set(_DECISION_TYPE_RE.findall(text))


# ---------------------------------------------------------------------------
# Top-5 Trainforge emit sites (audit DC2). Paths are absolute-from-root.
# ---------------------------------------------------------------------------

_EMIT_SITES = [
    PROJECT_ROOT / "Trainforge" / "generators" / "assessment_generator.py",
    PROJECT_ROOT / "Trainforge" / "process_course.py",
    # The ``content_selection`` emit lives in the pipeline-tool boundary.
    PROJECT_ROOT / "MCP" / "tools" / "pipeline_tools.py",
]


@pytest.mark.unit
@pytest.mark.parametrize("source", _EMIT_SITES, ids=lambda p: p.name)
def test_emit_site_decision_types_are_in_schema_enum(source):
    """Every ``decision_type=...`` literal in the emit site must be in the enum.

    The per-site parametrisation means a drift at any one of the five
    tracked sites fires an isolated failure, pinpointing where to
    update the schema enum.
    """
    allowed = _load_enum()
    found = _scrape_decision_types(source)
    missing = sorted(found - allowed)
    assert not missing, (
        f"{source.name} emits decision_type values that are not in the "
        f"canonical enum at {SCHEMA_PATH.relative_to(PROJECT_ROOT)}: "
        f"{missing}. Add these to "
        f"properties.decision_type.enum (alphabetised)."
    )


@pytest.mark.unit
def test_enum_covers_known_trainforge_values():
    """The six Wave 22 DC2 additions must be present in the canonical enum.

    A separate assertion (independent of source scraping) so the schema
    can't be silently regressed by reverting the enum additions even
    when emit sites are deleted.
    """
    allowed = _load_enum()
    required = {
        "assessment_planning",
        "assessment_generation",
        "boilerplate_strip",
        "content_selection",
        "question_type_selection",
        # DC3 add: the pipeline_run_attribution capture emitted at the
        # top of ``_raw_text_to_accessible_html`` in MCP/tools/
        # pipeline_tools.py.
        "pipeline_run_attribution",
    }
    missing = sorted(required - allowed)
    assert not missing, (
        f"decision_event.schema.json is missing Wave 22 enum values: "
        f"{missing}"
    )


@pytest.mark.unit
def test_enum_covers_wave89_slm_training_values():
    """Wave 89 (slm-training) adds five decision_type values for the
    upcoming Wave 90 trainforge-training pipeline. Pre-staging them in
    the canonical enum unblocks Wave 90's runner from running afoul
    of DECISION_VALIDATION_STRICT=true on first emit.
    """
    allowed = _load_enum()
    required = {
        "base_model_selection",
        "eval_run_decision",
        "hyperparameter_selection",
        "model_promotion_decision",
        "training_run_planning",
    }
    missing = sorted(required - allowed)
    assert not missing, (
        f"decision_event.schema.json is missing Wave 89 SLM-training enum "
        f"values: {missing}. Add to properties.decision_type.enum "
        f"(alphabetised)."
    )


@pytest.mark.unit
def test_phase_enum_covers_wave89_trainforge_training():
    """Wave 89 also adds a new phase value: ``trainforge-training``.
    Without this, the Wave 90 runner's DecisionCapture will fail-close
    under DECISION_VALIDATION_STRICT=true when it tries to log a
    decision under that phase.
    """
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    phase_enum = {v for v in schema["properties"]["phase"]["enum"] if isinstance(v, str)}
    assert "trainforge-training" in phase_enum, (
        "decision_event.schema.json phase enum is missing 'trainforge-training' "
        "(Wave 89 SLM-training pipeline phase value)."
    )


@pytest.mark.unit
def test_paraphrase_used_deterministic_draft_in_enum():
    """Wave 135d renames ``surface_form_preservation_fallback`` to
    ``paraphrase_used_deterministic_draft``. The new name describes
    what the synthesis pipeline actually does (the paraphrase emit
    chose the deterministic draft path) without asserting the
    pre-Wave-135 contract that preservation was required.
    """
    allowed = _load_enum()
    assert "paraphrase_used_deterministic_draft" in allowed, (
        "decision_event.schema.json must include "
        "'paraphrase_used_deterministic_draft' (Wave 135d rename of "
        "'surface_form_preservation_fallback'). Add to "
        "properties.decision_type.enum (alphabetised)."
    )
    assert "surface_form_preservation_fallback" not in allowed, (
        "Wave 135d removed 'surface_form_preservation_fallback' from "
        "the canonical enum. The capture rename is complete; the old "
        "string must not reappear."
    )


@pytest.mark.unit
def test_enum_is_alphabetised_and_unique():
    """Enum values must be alphabetised and unique (maintenance guard)."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    enum_list = schema["properties"]["decision_type"]["enum"]
    assert len(enum_list) == len(set(enum_list)), (
        "decision_type enum contains duplicates"
    )
    assert enum_list == sorted(enum_list), (
        "decision_type enum is not alphabetised — re-sort for "
        "maintainability (makes schema drift diffs clean)."
    )
