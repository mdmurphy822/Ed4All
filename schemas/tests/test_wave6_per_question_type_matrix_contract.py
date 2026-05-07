"""GPT Feedback v2 — Wave 6 end-of-wave per-question-type matrix
contract gate.

Authored 2026-05-07 against the closing 6-test gate enumerated in
``plans/wave6-per-question-type-matrix-2026-05.md`` § 4.

Predecessors landing:

* W6.A — :class:`lib.validators.assessment.AssessmentQualityValidator`
  carries the per-question-type quality matrix
  (:data:`_PER_QUESTION_TYPE_THRESHOLDS`,
  :func:`_normalize_question_type`, per-bucket diversity sub-checks
  emitting type-suffixed warnings like ``LOW_STEM_DIVERSITY_ESSAY``).
* W6.B — :func:`Trainforge.generators.assessment_quality_report.
  build_assessment_dimension` emits the ``per_question_type_summary``
  bucket dict and stamps each ``per_question_issues[]`` entry with
  ``question_type``.
* W6.C — ``config/workflows.yaml`` carries the
  ``per_question_type_thresholds`` operator-overlay block at both
  ``assessment_quality`` gate sites
  (``rag_training::assessment_generation`` and
  ``textbook_to_course::trainforge_assessment``); the config flows
  into validator inputs via the Wave 78 setdefault-merge at
  ``MCP/hardening/validation_gates.py:266-271``.

Predecessor wave-end gates this file mirrors:

* ``test_wave5_trainforge_ingestion_contract.py`` (Wave 5) — closest
  precedent for the wave-end-gate file structure (project-root
  ``sys.path`` injection, inlined fixtures, inlined helpers).
* ``test_wave4_pair_validation_contract.py`` (Wave 4 TIGHT) — THE
  TEMPLATE for the gate-file conventions across waves.
* ``test_wave4_medium_pair_objective_delivery_contract.py``
  (Wave 4 MEDIUM) — closest precedent for the per-type / per-bucket
  threshold-table assertion patterns.

This file is a SELF-CONTAINED contract test mirroring the prior
wave-end gates. Every fixture is inlined locally so a future rename
of a per-worker test file cannot silently disable this wave-end gate.

Tests:

1. ``test_per_question_type_threshold_table_has_5_entries`` — the
   W6.A canonical table contract: exactly five ``question_type``
   keys (``multiple_choice``, ``true_false``, ``short_answer``,
   ``essay``, ``fill_in_blank``) and each per-type sub-table carries
   the four required axes (``stem_diversity``,
   ``correct_answer_diversity``, ``distractor_template_max_ratio``,
   ``min_stem_chars``).
2. ``test_essay_distinct_stem_floor_below_corpus_wide`` — the W6.A
   calibration contract: essay's per-type ``stem_diversity`` floor is
   STRICTLY BELOW the legacy corpus-wide :data:`STEM_DIVERSITY_THRESHOLD`
   (small-N essay buckets need a relaxed floor since N=2-3 essay
   questions trip a 0.7 ratio trivially); MC's per-type floor is at
   or above the corpus-wide floor (MC is the volume class and demands
   stricter diversity).
3. ``test_per_type_diversity_emits_warning_not_critical`` — the W6.A
   severity-flip contract: a 3-essay bucket with identical stems
   trips the per-type ``LOW_STEM_DIVERSITY_ESSAY`` issue at
   ``severity="warning"``. Calibration-deferred per plan §5; severity
   flips to critical in a follow-up micro-wave once corpus-wide
   per-type warning-fire rates stabilise. Per the W6.A worker note
   (additive contract): the legacy corpus-wide ``LOW_STEM_DIVERSITY``
   critical may also fire on the same fixture and that's the
   back-compat contract — the test does not assert legacy absence,
   it just asserts the new per-type warning fires alongside.
4. ``test_qti_field_name_normalization`` — the W4.B Lesson 1 (pair-
   schema field-name divergence) mirror on the assessment surface:
   :func:`_normalize_question_type` MUST resolve a question dict
   carrying ``"type":"multiple_choice"`` (QTI parser shape, NOT
   generator shape) to the canonical ``"multiple_choice"`` bucket,
   AND the validator's per-type dispatch MUST tag the resulting
   issue codes with the ``MULTIPLE_CHOICE`` suffix.
5. ``test_per_question_type_summary_in_quality_report`` — the W6.B
   ``per_question_type_summary`` emit contract: a 6-question
   assessment (3 MC + 2 essay + 1 T/F) buckets cleanly, each
   non-singleton bucket carries the four diversity fields plus
   ``question_count`` / ``issue_count`` / ``top_codes``, and every
   ``per_question_issues[]`` entry carries the ``question_type``
   stamp.
6. ``test_workflow_yaml_carries_per_type_config_at_both_gate_sites``
   — the W6.C YAML round-trip contract: both gate sites
   (``rag_training::assessment_generation`` and
   ``textbook_to_course::trainforge_assessment``) carry
   ``gate.config.per_question_type_thresholds`` with all five
   canonical question_type keys.

No optional 7th test — Wave 6 surface is single-validator, no
merge-collapse risk class (unlike W5.F's chunker merge boundary).

Drift notes vs plan §4:

* Test 3 fixture mixes the 3 identical-stem essay questions with
  three additional distinct non-essay stems so the corpus-wide
  ``distinct_stem_ratio`` stays above the legacy 0.7 floor and the
  legacy critical ``LOW_STEM_DIVERSITY`` does NOT fire. This lets
  the test assert ``result.passed == True`` (per plan §4 Test 3
  sub-clause 1) WHILE still firing the per-type warning (sub-clause
  2 + 3). The original plan text "No ``LOW_STEM_DIVERSITY``
  (corpus-wide variant) on the same fixture" and the W6.A worker
  note ("the legacy critical may also fire") are reconciled by
  scoping the duplicate stems to the essay bucket only — the legacy
  corpus-wide check stays green, the per-type check fires as
  designed.
* Test 5 fixture omits ``short_answer`` + ``fill_in_blank`` per the
  plan text ("3 MC + 2 essay + 1 T/F"); the resulting summary has
  exactly 3 buckets (not 5) so the ``>=`` assertion in sub-clause 2
  reflects the actual fixture.
* Test 6 plan §4 snippet keys phase rows by ``phase_id`` but the
  on-disk ``config/workflows.yaml`` shape uses ``name:`` for phase
  rows (``MCP/core/workflow_runner.py`` canonicalises onto a
  ``phase_id`` attribute at load time). The YAML shape is the
  contract under test here, so the assertion targets ``name``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Test 1 — Per-question-type threshold table has exactly 5 entries
# --------------------------------------------------------------------------- #


def test_per_question_type_threshold_table_has_5_entries() -> None:
    """Plan §4 Test 1 — :data:`_PER_QUESTION_TYPE_THRESHOLDS` MUST
    enumerate the canonical 5 question types and each per-type
    sub-table MUST carry the 4 required threshold axes.

    Pins the W6.A table-shape contract: any future refactor that drops
    a question_type key OR a per-type axis breaks this test, so a
    silent-degradation regression on the per-type matrix can't slip
    past CI.
    """
    from lib.validators.assessment import _PER_QUESTION_TYPE_THRESHOLDS

    expected_types = {
        "multiple_choice",
        "true_false",
        "short_answer",
        "essay",
        "fill_in_blank",
    }
    actual_types = set(_PER_QUESTION_TYPE_THRESHOLDS.keys())
    assert actual_types == expected_types, (
        f"_PER_QUESTION_TYPE_THRESHOLDS must enumerate exactly "
        f"{sorted(expected_types)!r}; got {sorted(actual_types)!r}"
    )

    required_axes = {
        "stem_diversity",
        "correct_answer_diversity",
        "distractor_template_max_ratio",
        "min_stem_chars",
    }
    for qt, type_table in _PER_QUESTION_TYPE_THRESHOLDS.items():
        missing = required_axes - set(type_table.keys())
        assert not missing, (
            f"_PER_QUESTION_TYPE_THRESHOLDS[{qt!r}] is missing required "
            f"axes {sorted(missing)!r}; got keys "
            f"{sorted(type_table.keys())!r}"
        )


# --------------------------------------------------------------------------- #
# Test 2 — Essay distinct-stem floor strictly below the corpus-wide floor
# --------------------------------------------------------------------------- #


def test_essay_distinct_stem_floor_below_corpus_wide() -> None:
    """Plan §4 Test 2 — essay's per-type ``stem_diversity`` floor MUST
    be STRICTLY BELOW the legacy corpus-wide
    :data:`STEM_DIVERSITY_THRESHOLD` (essay buckets are typically
    small-N, 1-3 questions per assessment, where the 0.7 corpus-wide
    floor would trip trivially on legitimate variant question stems).
    Multiple-choice's per-type floor MUST be at or above the corpus-
    wide floor — MC is the volume class and demands the strictest
    diversity contract.

    Pins the W6.A calibration contract: any future refactor that
    relaxes MC's floor below the corpus-wide threshold OR tightens
    essay's floor at/above it inverts the calibration premise and
    breaks this test.
    """
    from lib.validators.assessment import (
        _PER_QUESTION_TYPE_THRESHOLDS,
        STEM_DIVERSITY_THRESHOLD,
    )

    essay_floor = _PER_QUESTION_TYPE_THRESHOLDS["essay"]["stem_diversity"]
    mc_floor = _PER_QUESTION_TYPE_THRESHOLDS["multiple_choice"]["stem_diversity"]

    assert essay_floor < STEM_DIVERSITY_THRESHOLD, (
        f"essay per-type stem_diversity floor ({essay_floor!r}) MUST be "
        f"STRICTLY BELOW corpus-wide STEM_DIVERSITY_THRESHOLD "
        f"({STEM_DIVERSITY_THRESHOLD!r}) — small-N essay buckets need a "
        "relaxed floor. Inverting this calibration breaks the W6.A premise."
    )
    assert mc_floor >= STEM_DIVERSITY_THRESHOLD, (
        f"multiple_choice per-type stem_diversity floor ({mc_floor!r}) "
        f"MUST be at or above corpus-wide STEM_DIVERSITY_THRESHOLD "
        f"({STEM_DIVERSITY_THRESHOLD!r}) — MC is the volume class and "
        "demands the strictest diversity contract."
    )


# --------------------------------------------------------------------------- #
# Test 3 — Per-type diversity check emits warning, not critical
# --------------------------------------------------------------------------- #


def test_per_type_diversity_emits_warning_not_critical() -> None:
    """Plan §4 Test 3 — a 3-essay bucket with identical stems MUST
    trip the W6.A per-type diversity warning at
    ``severity="warning"`` (calibration-deferred per plan §5; severity
    flips to critical in a follow-up micro-wave once corpus-wide
    per-type warning-fire rates stabilise).

    The fixture additionally carries 3 distinct non-essay questions
    (1 MC + 1 T/F + 1 short_answer) so the corpus-wide stem-diversity
    check (legacy ``LOW_STEM_DIVERSITY`` at ``severity="critical"``)
    does NOT fire — the resulting validator pass keeps
    ``result.passed == True``.

    Per the W6.A worker note (additive contract): no need to assert
    legacy ``LOW_STEM_DIVERSITY`` doesn't fire — the legacy critical
    may also fire on the same fixture and that's the back-compat
    contract. We just assert the new per-type warning fires alongside.
    The drift note in the module docstring documents the fixture
    design that reconciles the two.
    """
    from lib.validators.assessment import AssessmentQualityValidator

    # 3 essay questions with stems all starting "Discuss the role of X".
    # Identical stems trip the essay-bucket distinct-stem ratio
    # (3/3 unique would be 1.0; 1/3 unique = 0.33 < 0.55 essay floor).
    # "Discuss" is a Bloom verb (understand-level) per
    # ``schemas/taxonomies/bloom_verbs.json``, so no VERB_LESS_STEM
    # warning fires per question.
    essay_stem = "Discuss the role of X in the system architecture."
    essay_questions = [
        {
            "question_id": f"e-{i}",
            "question_type": "essay",
            "stem": essay_stem,
            "bloom_level": "understand",
        }
        for i in range(1, 4)
    ]

    # Three distinct non-essay questions to keep the corpus-wide
    # distinct-stem ratio above the legacy 0.7 floor. Corpus-wide
    # stem set = {essay_stem (×3 collapses to 1), mc_stem, tf_stem,
    # sa_stem} = 4 unique / 6 total = 0.667. That's still BELOW the
    # 0.7 corpus-wide floor — bumping to 4 distinct non-essay
    # stems gives 5 unique / 7 total = 0.714, above the 0.7 floor.
    distinct_questions = [
        {
            "question_id": "mc-1",
            "question_type": "multiple_choice",
            "stem": "Which property describes the cardinality constraint?",
            "bloom_level": "remember",
            "choices": [
                {"text": "sh:minCount", "is_correct": True},
                {"text": "sh:datatype", "is_correct": False},
                {"text": "sh:class", "is_correct": False},
            ],
        },
        {
            "question_id": "tf-1",
            "question_type": "true_false",
            "stem": "Identify whether SHACL validates RDF graphs.",
            "bloom_level": "remember",
            "correct_answer": "true",
        },
        {
            "question_id": "sa-1",
            "question_type": "short_answer",
            "stem": "Define the term 'graph closure' in your own words.",
            "bloom_level": "understand",
            "correct_answer": "Closure under entailment.",
        },
        {
            "question_id": "sa-2",
            "question_type": "short_answer",
            "stem": "Explain how datatype constraints differ from class constraints.",
            "bloom_level": "understand",
            "correct_answer": "Datatype constrains literals; class constrains IRIs.",
        },
    ]

    assessment = {
        "questions": essay_questions + distinct_questions,
    }

    validator = AssessmentQualityValidator()
    result = validator.validate({"assessment_data": assessment})

    # ---- Sub-clause 1 — gate passes (warning-only severity day-1). ---- #
    # The fixture is designed so no critical issues fire. If a critical
    # surfaces, dump it for diagnosis.
    critical_codes = [
        i.code for i in (result.issues or []) if i.severity == "critical"
    ]
    assert result.passed, (
        f"validator MUST pass on the warning-only fixture; got "
        f"passed={result.passed}, critical_codes={critical_codes!r}, "
        f"all_codes={[(i.code, i.severity) for i in (result.issues or [])]!r}"
    )

    # ---- Sub-clause 2 — at least one LOW_STEM_DIVERSITY_ESSAY fires. ---- #
    essay_diversity_issues = [
        i for i in (result.issues or [])
        if i.code == "LOW_STEM_DIVERSITY_ESSAY"
    ]
    assert essay_diversity_issues, (
        "expected at least one LOW_STEM_DIVERSITY_ESSAY issue from the "
        "3-identical-stem essay bucket; got codes="
        f"{[(i.code, i.severity) for i in (result.issues or [])]!r}"
    )

    # ---- Sub-clause 3 — that issue's severity is "warning". ---- #
    for issue in essay_diversity_issues:
        assert issue.severity == "warning", (
            f"LOW_STEM_DIVERSITY_ESSAY MUST be severity='warning' day-1 "
            f"(calibration-deferred per plan §5); got severity={issue.severity!r}"
        )


# --------------------------------------------------------------------------- #
# Test 4 — QTI ``type`` field-name normalization + per-type dispatch
# --------------------------------------------------------------------------- #


def test_qti_field_name_normalization() -> None:
    """Plan §4 Test 4 — :func:`_normalize_question_type` MUST resolve
    a question dict carrying ``"type":"multiple_choice"`` (QTI parser
    shape) to the canonical ``"multiple_choice"`` bucket, mirroring
    the W4.B Lesson 1 (pair-schema field-name divergence) resolution
    pattern. AND the validator's per-type dispatch MUST tag the
    resulting bucket-diversity issue codes with the
    ``MULTIPLE_CHOICE`` suffix when the bucket trips its per-type
    floor.

    Pins the QTI vs generator field-name resolution contract: a
    future refactor that drops the QTI ``type`` arm of the resolution
    chain breaks this test, so QTI-parsed assessments routed
    directly into the validator can't silently fall through to the
    unknown-type default bucket.
    """
    from lib.validators.assessment import (
        AssessmentQualityValidator,
        _normalize_question_type,
    )

    # ---- Sub-clause 1 — _normalize_question_type resolves QTI 'type' ---- #
    qti_question = {
        "question_id": "qti-1",
        "type": "multiple_choice",  # QTI parser shape (NOT generator shape)
        "stem": "Which property describes the cardinality constraint?",
        "choices": [
            {"text": "sh:minCount", "is_correct": True},
            {"text": "sh:datatype", "is_correct": False},
            {"text": "sh:class", "is_correct": False},
        ],
    }
    assert _normalize_question_type(qti_question) == "multiple_choice", (
        "_normalize_question_type MUST resolve QTI-shape "
        "{'type': 'multiple_choice'} to 'multiple_choice'; got "
        f"{_normalize_question_type(qti_question)!r}"
    )

    # ---- Sub-clause 2 — validator dispatches MC bucket via QTI shape. ---- #
    # 3 MC questions (QTI ``type`` shape) with identical stems trip
    # the MC distinct-stem ratio (1/3 = 0.33 < 0.75 MC floor).
    # The "Identify" prefix is a Bloom verb (remember-level) so no
    # VERB_LESS_STEM warning surfaces.
    duplicate_mc_stem = "Identify the SHACL constraint for cardinality."
    duplicate_mc_questions = [
        {
            "question_id": f"qti-mc-{i}",
            "type": "multiple_choice",  # QTI parser shape
            "stem": duplicate_mc_stem,
            "choices": [
                {"text": "sh:minCount", "is_correct": True},
                {"text": "sh:datatype", "is_correct": False},
                {"text": "sh:class", "is_correct": False},
            ],
        }
        for i in range(1, 4)
    ]

    validator = AssessmentQualityValidator()
    result = validator.validate(
        {"assessment_data": {"questions": duplicate_mc_questions}}
    )

    # MC bucket per-type warning carries the MULTIPLE_CHOICE suffix.
    mc_diversity_issues = [
        i for i in (result.issues or [])
        if i.code == "LOW_STEM_DIVERSITY_MULTIPLE_CHOICE"
    ]
    assert mc_diversity_issues, (
        "validator MUST dispatch QTI-shape MC questions to the "
        "MULTIPLE_CHOICE bucket; expected at least one "
        "LOW_STEM_DIVERSITY_MULTIPLE_CHOICE issue, got codes="
        f"{sorted({i.code for i in (result.issues or [])})!r}"
    )

    # And each issue is severity="warning" (per W6.A calibration-deferred
    # severity contract).
    for issue in mc_diversity_issues:
        assert issue.severity == "warning", (
            f"LOW_STEM_DIVERSITY_MULTIPLE_CHOICE MUST be severity='warning'; "
            f"got severity={issue.severity!r}"
        )


# --------------------------------------------------------------------------- #
# Test 5 — per_question_type_summary in quality report
# --------------------------------------------------------------------------- #


def test_per_question_type_summary_in_quality_report() -> None:
    """Plan §4 Test 5 — :func:`build_assessment_dimension` MUST emit a
    ``per_question_type_summary`` block keyed by canonical
    ``question_type`` AND every entry in ``per_question_issues[]``
    MUST carry the ``question_type`` stamp.

    Fixture: 3 MC + 2 essay + 1 T/F questions (6 total). The summary
    bucket dict has exactly 3 entries (one per present question_type;
    short_answer + fill_in_blank are absent from the fixture so they
    don't emit empty bucket dicts).

    Pins the W6.B emit contract on both surfaces:
      * the ``per_question_type_summary`` shape (4 diversity / count
        fields per bucket — ``question_count``, ``issue_count``,
        ``top_codes``, ``distinct_stem_ratio``,
        ``distinct_correct_answer_ratio``).
      * the ``per_question_issues[]`` per-entry ``question_type`` stamp.
    """
    from Trainforge.generators.assessment_quality_report import (
        build_assessment_dimension,
    )

    # 3 MC questions — distinct stems + answers (no diversity warnings).
    mc_questions = [
        {
            "question_id": f"mc-{i}",
            "question_type": "multiple_choice",
            "stem": f"Which SHACL property handles {topic}?",
            "bloom_level": "remember",
            "choices": [
                {"text": correct, "is_correct": True},
                {"text": "sh:nodeKind", "is_correct": False},
                {"text": "sh:pattern", "is_correct": False},
            ],
        }
        for i, (topic, correct) in enumerate(
            [
                ("cardinality lower bound", "sh:minCount"),
                ("cardinality upper bound", "sh:maxCount"),
                ("literal datatype", "sh:datatype"),
            ],
            start=1,
        )
    ]

    # 2 essay questions — distinct stems.
    essay_questions = [
        {
            "question_id": "e-1",
            "question_type": "essay",
            "stem": "Analyse how SHACL closes the open-world axiom of RDF.",
            "bloom_level": "analyze",
        },
        {
            "question_id": "e-2",
            "question_type": "essay",
            "stem": "Compare RDFS reasoning with OWL reasoning across use cases.",
            "bloom_level": "analyze",
        },
    ]

    # 1 T/F question — single-bucket case.
    tf_questions = [
        {
            "question_id": "tf-1",
            "question_type": "true_false",
            "stem": "Identify whether OWL extends RDFS expressivity.",
            "bloom_level": "remember",
            "correct_answer": "true",
        },
    ]

    assessment = {
        "questions": mc_questions + essay_questions + tf_questions,
    }

    dimension = build_assessment_dimension(assessment)
    assert dimension is not None, (
        "build_assessment_dimension MUST emit a dimension dict for a "
        "non-empty assessment; got None"
    )

    # ---- Sub-clause 1 — per_question_type_summary present. ---- #
    summary = dimension.get("per_question_type_summary")
    assert isinstance(summary, dict), (
        f"dimension MUST carry per_question_type_summary (dict); got "
        f"{type(summary).__name__}: {summary!r}"
    )

    # ---- Sub-clause 2 — three buckets present (one per question_type). ---- #
    expected_buckets = {"multiple_choice", "essay", "true_false"}
    actual_buckets = set(summary.keys())
    assert expected_buckets <= actual_buckets, (
        f"per_question_type_summary MUST carry buckets for all present "
        f"question_types {sorted(expected_buckets)!r}; got "
        f"{sorted(actual_buckets)!r}"
    )

    # ---- Sub-clause 3 — each bucket carries the required fields. ---- #
    # The diversity ratio fields (distinct_stem_ratio,
    # distinct_correct_answer_ratio) are skipped on N=1 buckets per
    # the W6.B "ratios are noisy at bucket size 1" contract — so the
    # ``true_false`` bucket (N=1) carries only count + issue fields,
    # while the multiple_choice (N=3) and essay (N=2) buckets carry
    # the full 5-field shape.
    multi_question_required = {
        "question_count",
        "issue_count",
        "top_codes",
        "distinct_stem_ratio",
        "distinct_correct_answer_ratio",
    }
    single_question_required = {"question_count", "issue_count", "top_codes"}

    for bucket_name in ("multiple_choice", "essay"):
        bucket = summary[bucket_name]
        assert isinstance(bucket, dict), (
            f"summary[{bucket_name!r}] MUST be a dict; got "
            f"{type(bucket).__name__}"
        )
        missing = multi_question_required - set(bucket.keys())
        assert not missing, (
            f"summary[{bucket_name!r}] (N>=2) MUST carry "
            f"{sorted(multi_question_required)!r}; missing "
            f"{sorted(missing)!r}; got keys {sorted(bucket.keys())!r}"
        )

    tf_bucket = summary["true_false"]
    missing = single_question_required - set(tf_bucket.keys())
    assert not missing, (
        f"summary['true_false'] (N=1) MUST carry "
        f"{sorted(single_question_required)!r}; missing "
        f"{sorted(missing)!r}; got keys {sorted(tf_bucket.keys())!r}"
    )

    # ---- Sub-clause 4 — every per_question_issues entry carries
    #     question_type. The fixture is benign (distinct stems + no
    #     placeholder content) so per_question_issues may be empty;
    #     when populated, every entry MUST carry the stamp. ---- #
    per_question_issues = dimension.get("per_question_issues")
    assert isinstance(per_question_issues, list), (
        f"dimension MUST carry per_question_issues (list); got "
        f"{type(per_question_issues).__name__}"
    )
    for entry in per_question_issues:
        assert "question_type" in entry, (
            f"every per_question_issues entry MUST carry the W6.B "
            f"question_type stamp; got entry without it: {entry!r}"
        )


# --------------------------------------------------------------------------- #
# Test 6 — Workflow YAML carries per-type config at both gate sites
# --------------------------------------------------------------------------- #


def test_workflow_yaml_carries_per_type_config_at_both_gate_sites() -> None:
    """Plan §4 Test 6 — ``config/workflows.yaml`` MUST carry the W6.C
    ``per_question_type_thresholds`` operator-overlay block at both
    ``assessment_quality`` gate sites:

      * ``rag_training::assessment_generation``
      * ``textbook_to_course::trainforge_assessment``

    Each site MUST enumerate all five canonical question_type keys
    (``multiple_choice``, ``true_false``, ``short_answer``, ``essay``,
    ``fill_in_blank``).

    Pins the W6.C YAML round-trip contract: a future refactor that
    drops the operator overlay at either gate site (or drops a
    question_type key) breaks this test, so the operator surface
    can't silently degrade to the validator's hardcoded defaults.
    """
    yaml = pytest.importorskip("yaml")

    workflows_path = PROJECT_ROOT / "config" / "workflows.yaml"
    cfg = yaml.safe_load(workflows_path.read_text(encoding="utf-8"))

    expected_question_types = {
        "multiple_choice",
        "true_false",
        "short_answer",
        "essay",
        "fill_in_blank",
    }

    gate_sites: List[Dict[str, str]] = [
        {"workflow_id": "rag_training", "phase_id": "assessment_generation"},
        {
            "workflow_id": "textbook_to_course",
            "phase_id": "trainforge_assessment",
        },
    ]

    workflows = cfg.get("workflows") or {}

    for site in gate_sites:
        wf_id = site["workflow_id"]
        phase_id = site["phase_id"]

        wf = workflows.get(wf_id)
        assert wf is not None, (
            f"workflow {wf_id!r} not found in workflows.yaml; got "
            f"workflow keys {sorted(workflows.keys())!r}"
        )

        # Phase rows in config/workflows.yaml use ``name:`` (not
        # ``phase_id:``). The plan §4 spec named ``phase_id`` because
        # the orchestrator's internal type maps the YAML ``name`` field
        # onto a ``phase_id`` attribute (``MCP/core/workflow_runner.py``
        # canonicalises both at load time). The on-disk YAML shape is
        # the contract under test here, so we assert against ``name``.
        phases = wf.get("phases") or []
        phase = next(
            (p for p in phases if p.get("name") == phase_id),
            None,
        )
        assert phase is not None, (
            f"phase {phase_id!r} not found in workflow {wf_id!r}; got "
            f"phase names {[p.get('name') for p in phases]!r}"
        )

        validation_gates = phase.get("validation_gates") or []
        gate = next(
            (g for g in validation_gates if g.get("gate_id") == "assessment_quality"),
            None,
        )
        assert gate is not None, (
            f"assessment_quality gate not found in {wf_id}::{phase_id}; "
            f"got gate_ids {[g.get('gate_id') for g in validation_gates]!r}"
        )

        gate_config = gate.get("config") or {}
        per_type = gate_config.get("per_question_type_thresholds") or {}
        actual_types = set(per_type.keys())

        assert expected_question_types <= actual_types, (
            f"{wf_id}::{phase_id}::assessment_quality::config."
            f"per_question_type_thresholds MUST enumerate all five "
            f"canonical question_type keys "
            f"{sorted(expected_question_types)!r}; got "
            f"{sorted(actual_types)!r}"
        )
