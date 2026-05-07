"""GPT Feedback v2 — Wave 7 end-of-wave per-question-type eval
segmentation contract gate.

Authored 2026-05-07 against the closing 6-test gate enumerated in
``plans/wave7-per-type-eval-segmentation-2026-05.md`` § 4.

Predecessors landing:

* W7.A — :mod:`Trainforge.eval._per_type_helpers` ships
  :data:`RELEVANT_QUESTION_TYPES`, :func:`normalize_question_type`,
  :func:`bucket_per_question_records`, :func:`attach_relevance` —
  the shared per-question-type bucketing surface consumed by every
  W3.F evaluator. ``_load_prompts_for_metrics`` patched to thread
  ``question_type`` through the prompt-dict shape. Per-type
  ``answerable_rate`` + ``placeholder_rate`` emit added.
* W7.B — Per-type ``single_correct_rate`` + ``distractor_entropy`` +
  ``bloom_alignment_rate`` + ``source_support_rate`` emit landed.
  ``_load_rendered_html_for_metrics`` boundary contract changed to
  ``List[Tuple[str, str]]`` so the single_correct_rate evaluator can
  bucket by question_type alongside the rendered HTML.
* W7.C — Eval-report shape contract documented in the canonical
  Trainforge CLAUDE.md "Per-question-type segmentation" section, and
  per-component ``Trainforge/eval/tests/test_*_per_type.py`` tests
  pin the per-evaluator emit shape.
* W7.D — :data:`lib.validators.eval_gating.PER_TYPE_METRIC_CONFIG`
  + 6 per-type warning codes. Six warning-severity GateIssue codes
  fire per below-floor bucket; calibration-deferred severity flip per
  plan §5.

Predecessor wave-end gates this file mirrors:

* ``test_wave6_per_question_type_matrix_contract.py`` (Wave 6) —
  closest precedent for the wave-end-gate file structure (project-
  root ``sys.path`` injection, inlined fixtures, inlined helpers).
* ``test_wave5_trainforge_ingestion_contract.py`` (Wave 5) — closer
  precedent for the ``schemas/tests/`` path discipline (Lesson 2 from
  Wave 5 §7: the wave-end gate MUST land at ``schemas/tests/``, NOT
  ``lib/validators/tests/``).
* ``test_wave4_pair_validation_contract.py`` (Wave 4 TIGHT) — THE
  TEMPLATE for the gate-file conventions across waves.

This file is a SELF-CONTAINED contract test mirroring the prior
wave-end gates. Every fixture is inlined locally so a future rename
of a per-component test file cannot silently disable this wave-end gate.

Tests:

1. ``test_per_type_helpers_canonical_keys_match_w6_validator`` — the
   W7.A relevance-table contract: every metric's ``relevant_set`` is
   a subset of the W6.A canonical 5-type matrix
   (:data:`lib.validators.assessment._PER_QUESTION_TYPE_THRESHOLDS`).
   The eval surface CANNOT introduce novel question types — single
   source of truth stays the W6.A table. AND every metric's
   ``relevant_set`` is non-empty.
2. ``test_distractor_entropy_relevant_only_for_multiple_choice`` —
   the W7.A discriminator contract: distractors are MC-specific.
   T/F has 1 distractor (entropy=0 by construction), SA/essay/
   fill_in_blank have empty distractor lists, so per-type emit MUST
   mark them not-relevant. The W7.D gate skips them via the
   ``relevant=False`` short-circuit; otherwise every essay bucket
   would warning-fire on a structural zero.
3. ``test_load_prompts_for_metrics_threads_question_type`` — the
   W7.A loader-boundary contract:
   :meth:`Trainforge.eval.slm_eval_harness.SLMEvalHarness._load_prompts_for_metrics`
   threads ``question_type`` from ``assessments.json`` into the
   prompt-dict shape, including the QTI ``type`` fallback. Bypasses
   ``__init__`` via ``__new__`` per the W7.C precedent at
   ``Trainforge/eval/tests/test_slm_eval_harness_per_type.py:195``.
4. ``test_eval_report_per_metric_carries_per_question_type_block`` —
   the W7.A/B emit contract: every W3.F result dict in
   ``eval_report.json`` carries a ``per_question_type`` dict (or
   ``None`` for deps-missing metrics). Mirrors the W6.B
   ``per_question_type_summary`` shape contract from the Wave 6
   wave-end gate.
5. ``test_eval_gating_per_type_warning_does_not_block_promotion`` —
   the W7.D day-1 contract: per-type warnings emit at
   ``severity="warning"`` and never flip ``passed=False``.
   Calibration-deferred per plan §5; severity flips to critical in a
   follow-up micro-wave.
6. ``test_eval_gating_skips_per_type_when_irrelevant_or_deps_missing``
   — the W7.D defense-in-depth contract: structural zeros on
   irrelevant buckets MUST NOT trip warnings.
   ``distractor_entropy.essay=0.0`` (structural — essay has no
   distractors) MUST be skipped because ``relevant=False``. Same for
   deps-missing branches (``per_question_type=None``).

No optional 7th test — Wave 7 surface is single-evaluator-family +
single-gate; no merge-collapse risk class to assert.

Drift notes vs plan §4:

* Test 3 plan snippet shows ``SLMEvalHarness(course_path=course_path,
  ...)`` direct instantiation with placeholder ``...``; the harness
  has hard ``__init__`` deps (real adapter callable / holdout split /
  pyproject training extras). We follow the W7.C precedent at
  ``Trainforge/eval/tests/test_slm_eval_harness_per_type.py`` and
  bypass ``__init__`` via ``__new__`` + manual ``course_path``
  attribute assignment. The ``_load_prompts_for_metrics`` method
  reads only ``self.course_path`` so the bypass is sufficient.
* Test 6 plan snippet uses ``"mean_distractor_entropy": 0.0`` at the
  per-bucket level. The actual per-bucket emit field key is
  ``"distractor_entropy_mean"`` per ``Trainforge/eval/distractor_entropy.py:143``
  (note: corpus-wide top-level field IS ``mean_distractor_entropy``,
  but the per-bucket field reverses the word order). This matches
  the W7.D ``PER_TYPE_METRIC_CONFIG`` ``value_field`` at
  ``lib/validators/eval_gating.py:91``. The test fixture uses
  ``distractor_entropy_mean`` to align with the actual emit + gate
  contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Test 1 — W7.A relevance-table is a subset of W6.A canonical 5-type matrix
# --------------------------------------------------------------------------- #


def test_per_type_helpers_canonical_keys_match_w6_validator() -> None:
    """Plan §4 Test 1 — W7.A's :data:`RELEVANT_QUESTION_TYPES` keys
    MUST be a subset of W6.A's canonical 5-type matrix; the eval
    surface CANNOT introduce novel question types — single source of
    truth stays the W6.A table.

    Pins the cross-wave canonical-type contract: any future refactor
    that lets the W7.A table drift away from the W6.A matrix breaks
    this test. Also asserts every metric's relevant set is non-empty
    (an empty set would silently disable the gate for that metric).
    """
    from Trainforge.eval._per_type_helpers import RELEVANT_QUESTION_TYPES
    from lib.validators.assessment import _PER_QUESTION_TYPE_THRESHOLDS

    canonical = set(_PER_QUESTION_TYPE_THRESHOLDS.keys())
    for metric_name, relevant_set in RELEVANT_QUESTION_TYPES.items():
        assert relevant_set.issubset(canonical), (
            f"{metric_name} relevant_set {relevant_set!r} introduces "
            f"non-canonical question types beyond {sorted(canonical)!r}"
        )

    # Every metric must declare a non-empty relevant set — an empty
    # set would silently disable per-type gating for the metric.
    for metric_name, relevant_set in RELEVANT_QUESTION_TYPES.items():
        assert relevant_set, (
            f"{metric_name} declares an empty relevant_set — would "
            "silently disable per-type gating for the metric."
        )


# --------------------------------------------------------------------------- #
# Test 2 — distractor_entropy is MC-only per the relevance table
# --------------------------------------------------------------------------- #


def test_distractor_entropy_relevant_only_for_multiple_choice() -> None:
    """Plan §4 Test 2 — distractors are MC-specific. T/F has 1
    distractor (entropy=0 by construction), SA/essay/fill_in_blank
    have empty distractor lists. Per-type bucket emit MUST mark them
    not-relevant so the W7.D gate skips them; otherwise every essay
    bucket would warning-fire on a structural zero.

    Pins the W7.A discriminator contract for the most narrowly-scoped
    metric on the relevance table. A future refactor that widens
    distractor_entropy beyond multiple_choice would silently re-
    introduce the structural-zero false-positive class on every non-MC
    bucket.
    """
    from Trainforge.eval._per_type_helpers import RELEVANT_QUESTION_TYPES

    assert RELEVANT_QUESTION_TYPES["distractor_entropy"] == {
        "multiple_choice"
    }, (
        f"distractor_entropy relevant set MUST be exactly "
        f"{{'multiple_choice'}}; got "
        f"{RELEVANT_QUESTION_TYPES['distractor_entropy']!r}"
    )
    assert "essay" not in RELEVANT_QUESTION_TYPES["distractor_entropy"], (
        "essay MUST NOT be in distractor_entropy relevant set — "
        "essay questions have empty distractor lists; including them "
        "would re-introduce the structural-zero false-positive class."
    )
    assert (
        "true_false" not in RELEVANT_QUESTION_TYPES["distractor_entropy"]
    ), (
        "true_false MUST NOT be in distractor_entropy relevant set — "
        "T/F has 1 distractor so entropy=0 by construction; including "
        "it would re-introduce the structural-zero false-positive class."
    )


# --------------------------------------------------------------------------- #
# Test 3 — _load_prompts_for_metrics threads question_type (incl. QTI)
# --------------------------------------------------------------------------- #


def test_load_prompts_for_metrics_threads_question_type(
    tmp_path: Path,
) -> None:
    """Plan §4 Test 3 — Finding A: ``assessments.json`` carries
    ``question_type`` but the eval harness loader was dropping it.
    Confirm it's now threaded through, including the QTI ``type``
    fallback.

    Bypasses ``SLMEvalHarness.__init__`` via ``__new__`` (W7.C
    precedent at ``Trainforge/eval/tests/test_slm_eval_harness_per_type.py:195``)
    because the harness ``__init__`` has hard deps (real adapter
    callable / holdout split / pyproject training extras) that are
    irrelevant to the loader-boundary contract under test.
    ``_load_prompts_for_metrics`` reads only ``self.course_path`` so
    the bypass is sufficient.
    """
    from Trainforge.eval.slm_eval_harness import SLMEvalHarness

    course_path = tmp_path / "course"
    course_path.mkdir()
    (course_path / "assessments.json").write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": "q1",
                        "question_type": "multiple_choice",
                        "stem": "What is X?",
                        "correct_answer": "A",
                    },
                    {
                        "question_id": "q2",
                        "question_type": "essay",
                        "stem": "Discuss Y.",
                        "correct_answer": "...",
                    },
                    {
                        "question_id": "q3",
                        # QTI parser shape — ``type`` instead of
                        # ``question_type``. The W6 / W7 normalisation
                        # chain falls back to this key.
                        "type": "true_false",
                        "stem": "Z is foo.",
                        "correct_answer": "true",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    harness = SLMEvalHarness.__new__(SLMEvalHarness)
    harness.course_path = course_path  # type: ignore[attr-defined]

    prompts = harness._load_prompts_for_metrics()

    assert len(prompts) == 3, (
        f"loader MUST emit one prompt per question; got {len(prompts)}"
    )
    assert prompts[0]["question_type"] == "multiple_choice", (
        f"prompts[0] question_type MUST be 'multiple_choice'; got "
        f"{prompts[0].get('question_type')!r}"
    )
    assert prompts[1]["question_type"] == "essay", (
        f"prompts[1] question_type MUST be 'essay'; got "
        f"{prompts[1].get('question_type')!r}"
    )
    # QTI fallback — the loader resolves ``type`` when ``question_type``
    # is absent. Mirrors the W6.A ``_normalize_question_type``
    # resolution chain.
    assert prompts[2]["question_type"] == "true_false", (
        f"prompts[2] question_type MUST resolve QTI 'type' fallback to "
        f"'true_false'; got {prompts[2].get('question_type')!r}"
    )


# --------------------------------------------------------------------------- #
# Test 4 — Each W3.F metric block carries a per_question_type dict
# --------------------------------------------------------------------------- #


def test_eval_report_per_metric_carries_per_question_type_block() -> None:
    """Plan §4 Test 4 — End-to-end shape contract: every W3.F result
    dict carries a ``per_question_type`` dict (success path) or
    ``None`` (deps-missing path). Mirrors the W6.B
    ``per_question_type_summary`` shape contract from the Wave 6
    wave-end gate.

    Pins the W7.A/B emit contract on the two evaluators with no
    external deps (``answerable_rate`` is pure Jaccard, ``placeholder_rate``
    is pure regex) so this test runs cleanly on a bare CI checkout.
    """
    from Trainforge.eval.answerable_rate import AnswerableRateEvaluator
    from Trainforge.eval.placeholder_rate import PlaceholderRateEvaluator

    prompts = [
        {
            "question_id": "q1",
            "question_type": "multiple_choice",
            "stem": "X?",
            "correct_answer": "A is foo",
            "source_chunks": ["c1"],
        },
        {
            "question_id": "q2",
            "question_type": "essay",
            "stem": "Discuss.",
            "correct_answer": "Y discusses Z thoroughly",
            "source_chunks": ["c1"],
        },
    ]
    chunk_corpus = {"c1": "A is foo. Y discusses Z thoroughly."}

    # ---- AnswerableRateEvaluator ---- #
    ar = AnswerableRateEvaluator().evaluate(prompts, chunk_corpus)
    assert "per_question_type" in ar, (
        f"AnswerableRateEvaluator result MUST carry per_question_type; "
        f"got keys {sorted(ar.keys())!r}"
    )
    pqt_ar = ar["per_question_type"]
    assert isinstance(pqt_ar, dict), (
        f"answerable_rate.per_question_type MUST be a dict; got "
        f"{type(pqt_ar).__name__}"
    )
    assert "multiple_choice" in pqt_ar, (
        f"answerable_rate.per_question_type MUST carry "
        f"multiple_choice bucket; got keys {sorted(pqt_ar.keys())!r}"
    )
    assert "essay" in pqt_ar, (
        f"answerable_rate.per_question_type MUST carry essay bucket; "
        f"got keys {sorted(pqt_ar.keys())!r}"
    )

    # ---- PlaceholderRateEvaluator ---- #
    pr = PlaceholderRateEvaluator().evaluate(prompts)
    assert "per_question_type" in pr, (
        f"PlaceholderRateEvaluator result MUST carry per_question_type; "
        f"got keys {sorted(pr.keys())!r}"
    )
    pqt_pr = pr["per_question_type"]
    assert isinstance(pqt_pr, dict), (
        f"placeholder_rate.per_question_type MUST be a dict; got "
        f"{type(pqt_pr).__name__}"
    )


# --------------------------------------------------------------------------- #
# Test 5 — per-type warning fires but does NOT block promotion
# --------------------------------------------------------------------------- #


def test_eval_gating_per_type_warning_does_not_block_promotion(
    tmp_path: Path,
) -> None:
    """Plan §4 Test 5 — W7.D day-1 contract: per-type warnings emit
    at ``severity="warning"`` and never flip ``passed=False``.
    Calibration-deferred per plan §5; severity flips to critical in
    a follow-up micro-wave.

    Fixture: corpus-wide thresholds pass (faithfulness=0.85, no
    yes-bias, no negative-grounding probe, no source_match probe), so
    the only thing that COULD flip ``passed=False`` is a per-type
    bucket — which by W7.D contract MUST emit at warning severity
    only.
    """
    from lib.validators.eval_gating import EvalGatingValidator

    model_dir = tmp_path / "model"
    eval_dir = model_dir / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "eval_report.json").write_text(
        json.dumps(
            {
                "faithfulness": 0.85,  # passes critical
                "coverage": 0.7,
                "profile": "rdf_shacl",
                "per_tier": {},
                "per_invariant": {},
                "answerable_rate": {
                    "answerable_rate": 0.85,
                    "total_questions": 12,
                    "answerable_count": 10,
                    "per_question_type": {
                        "multiple_choice": {
                            "answerable_rate": 0.95,
                            "total_questions": 10,
                            "answerable_count": 9,
                            "relevant": True,
                        },
                        "essay": {
                            # WAY below the 0.30 default per-type
                            # threshold — guaranteed to fire.
                            "answerable_rate": 0.10,
                            "total_questions": 2,
                            "answerable_count": 0,
                            "relevant": True,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = EvalGatingValidator().validate(
        {"model_dir": str(model_dir)}
    )

    # ---- Sub-clause 1 — gate passes (warning never blocks). ---- #
    critical_codes = [
        i.code for i in result.issues if i.severity == "critical"
    ]
    assert result.passed is True, (
        f"per-type warning MUST NOT block promotion (W7.D day-1 "
        f"calibration-deferred contract); got passed={result.passed}, "
        f"critical_codes={critical_codes!r}, all_codes="
        f"{[(i.code, i.severity) for i in result.issues]!r}"
    )

    # ---- Sub-clause 2 — the per-type warning code is present. ---- #
    warning_codes = [
        i.code for i in result.issues if i.severity == "warning"
    ]
    assert "EVAL_ANSWERABLE_PER_TYPE_BELOW_THRESHOLD" in warning_codes, (
        f"essay bucket below threshold MUST emit "
        f"EVAL_ANSWERABLE_PER_TYPE_BELOW_THRESHOLD warning; got "
        f"warning_codes={warning_codes!r}"
    )


# --------------------------------------------------------------------------- #
# Test 6 — irrelevant + deps-missing buckets are skipped cleanly
# --------------------------------------------------------------------------- #


def test_eval_gating_skips_per_type_when_irrelevant_or_deps_missing(
    tmp_path: Path,
) -> None:
    """Plan §4 Test 6 — Defense-in-depth: structural zeros on
    irrelevant buckets MUST NOT trip warnings.
    ``distractor_entropy.essay=0.0`` (structural — essay has no
    distractors) MUST be skipped because ``relevant=False``. Same for
    deps-missing branches (``per_question_type=None``).

    Pins the W7.D short-circuit contract via two fixtures:
      * ``distractor_entropy.per_question_type[essay].relevant=False``
        — even with the bucket scalar at 0.0, the gate MUST skip the
        bucket (irrelevant short-circuit at
        ``lib/validators/eval_gating.py:326``).
      * ``bloom_alignment_rate.per_question_type=None,
        deps_missing=True`` — the gate MUST skip cleanly without
        crashing on the ``None`` per_question_type (deps-missing
        short-circuit at ``lib/validators/eval_gating.py:316``).

    Note on the per-bucket field name: the actual W7.B emit at
    ``Trainforge/eval/distractor_entropy.py:143`` uses
    ``distractor_entropy_mean`` per bucket (corpus-wide top-level
    field is ``mean_distractor_entropy`` — word order reversed). The
    W7.D ``PER_TYPE_METRIC_CONFIG`` ``value_field`` at
    ``lib/validators/eval_gating.py:91`` matches this. Plan §4 Test 6
    snippet showed ``mean_distractor_entropy`` at the per-bucket
    level; we use the actual emit key here.
    """
    from lib.validators.eval_gating import EvalGatingValidator

    model_dir = tmp_path / "model"
    (model_dir / "eval").mkdir(parents=True)
    (model_dir / "eval" / "eval_report.json").write_text(
        json.dumps(
            {
                "faithfulness": 0.85,
                "coverage": 0.7,
                "profile": "rdf_shacl",
                "per_tier": {},
                "per_invariant": {},
                "distractor_entropy": {
                    "mean_distractor_entropy": 0.6,
                    "per_question_type": {
                        "essay": {
                            # Structural zero on irrelevant bucket;
                            # MUST be skipped via relevant=False.
                            "distractor_entropy_mean": 0.0,
                            "relevant": False,
                        },
                        "multiple_choice": {
                            "distractor_entropy_mean": 0.7,
                            "relevant": True,
                        },
                    },
                },
                "bloom_alignment_rate": {
                    "bloom_alignment_rate": None,
                    "per_question_type": None,
                    "deps_missing": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = EvalGatingValidator().validate(
        {"model_dir": str(model_dir)}
    )

    # ---- Sub-clause 1 — distractor_entropy.essay=0.0 with
    #     relevant=False MUST NOT fire a warning (irrelevant
    #     short-circuit). MC bucket=0.7 above threshold so it MUST
    #     NOT fire either. Net: zero distractor_entropy warnings. ---- #
    distractor_warnings = [
        i
        for i in result.issues
        if i.code == "EVAL_DISTRACTOR_ENTROPY_PER_TYPE_BELOW_THRESHOLD"
    ]
    assert distractor_warnings == [], (
        f"distractor_entropy.essay=0.0 with relevant=False MUST be "
        f"skipped via the W7.D irrelevant short-circuit; got "
        f"warnings={[(i.code, i.message) for i in distractor_warnings]!r}"
    )

    # ---- Sub-clause 2 — bloom_alignment_rate.per_question_type=None
    #     (deps-missing) MUST be skipped cleanly (no warnings, no
    #     crash). ---- #
    bloom_warnings = [
        i
        for i in result.issues
        if i.code == "EVAL_BLOOM_ALIGNMENT_PER_TYPE_BELOW_THRESHOLD"
    ]
    assert bloom_warnings == [], (
        f"bloom_alignment_rate.per_question_type=None (deps_missing=True) "
        f"MUST be skipped via the W7.D deps-missing short-circuit; got "
        f"warnings={[(i.code, i.message) for i in bloom_warnings]!r}"
    )
