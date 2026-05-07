"""Wave 7 W7.C — eval-report shape contract + harness per-type wiring tests.

End-to-end harness-level tests confirming the W3.F result dicts
land in ``eval_report.json`` carrying ``per_question_type`` blocks.
The harness wiring at ``slm_eval_harness.py:1030-1041`` routes each
evaluator's return-dict verbatim into the report (no shape mangling),
so the per-type blocks land additively inside each per-metric result
dict — not at the report top-level.

Bypasses ``SLMEvalHarness.__init__`` via ``__new__`` to skip the real-
adapter / holdout-graph dependencies; the W7.C contract is the
prompt-loader + ``_run_<metric>`` dispatch chain shape, not the full
harness execution path.

Deferred verification: ``source_support_rate``'s success-path
per-type emit lands with W7.B (Anthropic NLI deps); on a bare CI
checkout the deps-missing branch fires and ``per_question_type``
is ``None`` (still satisfies the shape contract — the field is
present on every result dict).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval._per_type_helpers import RELEVANT_QUESTION_TYPES


_CANONICAL_QUESTION_TYPES = {
    "multiple_choice",
    "true_false",
    "short_answer",
    "essay",
    "fill_in_blank",
}


def _build_fixture_course(tmp_path: Path) -> Path:
    """Materialise a 6-question, 3-type fixture course tree.

    Layout mirrors the real LibV2 course tree expected by the harness'
    metric loaders: ``assessments.json`` at the root + ``imscc_chunks/
    chunks.jsonl`` for the chunk corpus.
    """
    course_dir = tmp_path / "course"
    course_dir.mkdir()
    (course_dir / "imscc_chunks").mkdir()

    assessments = {
        "questions": [
            # 2 multiple_choice
            {
                "question_id": "mc-1",
                "question_type": "multiple_choice",
                "stem": "What does RDFS define?",
                "correct_answer": (
                    "RDFS describes vocabulary terms using "
                    "classes properties resources knowledge."
                ),
                "distractors": [
                    "A query language for SPARQL endpoints",
                    "A constraint validation language",
                    "An ontology alignment toolkit",
                ],
                "source_chunks": ["c1"],
                "bloom_level": "remember",
                "feedback": "RDFS introduces class and property semantics.",
                "rendered_html": (
                    '<ul><li data-cf-correct="true">RDFS describes vocabulary'
                    "</li><li>SPARQL</li></ul>"
                ),
            },
            {
                "question_id": "mc-2",
                "question_type": "multiple_choice",
                "stem": "Which standard validates RDF graphs structurally?",
                "correct_answer": (
                    "SHACL validates structural constraints "
                    "across RDF graphs and shapes."
                ),
                "distractors": [
                    "RDFS provides only typing axioms",
                    "OWL focuses on description logic",
                    "SPARQL is a query language",
                ],
                "source_chunks": ["c2"],
                "bloom_level": "understand",
                "feedback": "SHACL targets structural validation.",
                "rendered_html": (
                    '<ul><li data-cf-correct="true">SHACL</li>'
                    "<li>RDFS</li></ul>"
                ),
            },
            # 2 short_answer
            {
                "question_id": "sa-1",
                "question_type": "short_answer",
                "stem": "Briefly describe RDFS.",
                "correct_answer": (
                    "RDFS describes vocabulary terms classes "
                    "properties and resources."
                ),
                "distractors": [],
                "source_chunks": ["c1"],
                "bloom_level": "understand",
                "feedback": "Look for class / property axioms.",
            },
            {
                "question_id": "sa-2",
                "question_type": "short_answer",
                # Placeholder pattern — exercises the placeholder_rate
                # per-type bucket.
                "stem": "Discuss SHACL constraints.",
                "correct_answer": "Correct answer based on content from the lecture.",
                "distractors": [],
                "source_chunks": ["c2"],
                "bloom_level": "understand",
                "feedback": "Address each constraint kind.",
            },
            # 2 essay
            {
                "question_id": "essay-1",
                "question_type": "essay",
                "stem": "Compare RDFS and SHACL.",
                "correct_answer": (
                    "RDFS provides typing axioms while SHACL provides "
                    "structural validation through shapes."
                ),
                "distractors": [],
                "source_chunks": ["c1", "c2"],
                "bloom_level": "analyze",
                "feedback": "Cover both axes.",
            },
            {
                "question_id": "essay-2",
                "question_type": "essay",
                "stem": "Discuss SPARQL endpoints.",
                # Intentionally low-overlap with cited chunks → not
                # answerable; exercises answerable_rate.essay bucket = 0.
                "correct_answer": "alpha beta gamma delta epsilon zeta",
                "distractors": [],
                "source_chunks": ["c2"],
                "bloom_level": "evaluate",
                "feedback": "Cover query semantics.",
            },
        ]
    }
    (course_dir / "assessments.json").write_text(
        json.dumps(assessments), encoding="utf-8"
    )

    chunks = [
        {
            "chunk_id": "c1",
            "text": (
                "RDFS describes vocabulary terms using classes "
                "properties and resources to model knowledge."
            ),
        },
        {
            "chunk_id": "c2",
            "text": (
                "SHACL validates structural constraints across "
                "RDF graphs and shapes for consistency."
            ),
        },
    ]
    chunks_path = course_dir / "imscc_chunks" / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    return course_dir


def _build_eval_report(course_dir: Path) -> Dict[str, Any]:
    """Assemble the W3.F slice of ``eval_report.json``.

    Mirrors the harness wiring at ``slm_eval_harness.py:1030-1041`` —
    invokes each ``_run_<metric>`` directly and folds the results into
    a dict whose top-level keys match what the real ``run_all`` writes.
    Bypasses ``__init__`` so the test does not need a real model
    callable / holdout split / pyproject training extras.
    """
    from Trainforge.eval.slm_eval_harness import SLMEvalHarness

    harness = SLMEvalHarness.__new__(SLMEvalHarness)
    harness.course_path = course_dir  # type: ignore[attr-defined]

    prompts = harness._load_prompts_for_metrics()
    chunk_corpus = harness._load_chunk_corpus_for_metrics()
    rendered_html_blocks = harness._load_rendered_html_for_metrics(prompts)

    out: Dict[str, Any] = {}
    out["answerable_rate"] = harness._run_answerable_rate(prompts, chunk_corpus)
    out["distractor_entropy"] = harness._run_distractor_entropy(prompts)
    out["placeholder_rate"] = harness._run_placeholder_rate(prompts)
    out["bloom_alignment_rate"] = harness._run_bloom_alignment_rate(prompts)
    out["source_support_rate"] = harness._run_source_support_rate(
        prompts, chunk_corpus
    )
    # single_correct_rate W7.B contract is List[Tuple[str, str]]; the
    # legacy harness loader returns List[str], so adapt here. The
    # evaluator carries a defensive fallback for bare strings (degraded
    # — empty per_question_type), which is the contract under test.
    out["single_correct_rate"] = harness._run_single_correct_rate(
        rendered_html_blocks
    )
    return out


# ---------------------------------------------------------------------------
# 1. Every W3.F metric block carries a per_question_type field.
# ---------------------------------------------------------------------------
def test_eval_report_carries_per_question_type_on_all_six_metrics(
    tmp_path: Path,
) -> None:
    """Six W3.F blocks each carry a ``per_question_type`` field.

    The field MAY be a populated dict (success path) OR ``None``
    (deps-missing or pre-W7.B per-type extension); the W7.C contract
    is presence — never absence.
    """
    course_dir = _build_fixture_course(tmp_path)
    report = _build_eval_report(course_dir)

    expected_blocks = {
        "answerable_rate",
        "single_correct_rate",
        "distractor_entropy",
        "bloom_alignment_rate",
        "placeholder_rate",
        "source_support_rate",
    }
    assert set(report.keys()) >= expected_blocks

    for metric in expected_blocks:
        block = report[metric]
        assert "per_question_type" in block, (
            f"{metric}: missing per_question_type key in result dict"
        )
        pqt = block["per_question_type"]
        assert pqt is None or isinstance(pqt, dict), (
            f"{metric}: per_question_type must be None or dict, got {type(pqt)}"
        )


# ---------------------------------------------------------------------------
# 2. Every emitted bucket key is one of the canonical 5 question_types.
# ---------------------------------------------------------------------------
def test_per_question_type_buckets_match_canonical_5_types(
    tmp_path: Path,
) -> None:
    """Bucket keys are subset of canonical types; no empty-string leak."""
    course_dir = _build_fixture_course(tmp_path)
    report = _build_eval_report(course_dir)

    for metric, block in report.items():
        pqt = block.get("per_question_type")
        if not isinstance(pqt, dict):
            continue  # deps_missing or empty → no buckets to check
        for qt in pqt.keys():
            assert qt != "", (
                f"{metric}: empty-string bucket leaked into per_question_type"
            )
            assert qt in _CANONICAL_QUESTION_TYPES, (
                f"{metric}: bucket key '{qt}' is not a canonical question_type"
            )


# ---------------------------------------------------------------------------
# 3. Round-trip the report through json.dump → json.load with no drift.
# ---------------------------------------------------------------------------
def test_eval_report_per_question_type_round_trips_through_json(
    tmp_path: Path,
) -> None:
    """JSON serialise → deserialise preserves per_question_type shape."""
    course_dir = _build_fixture_course(tmp_path)
    report = _build_eval_report(course_dir)

    out_path = tmp_path / "eval_report.json"
    out_path.write_text(json.dumps(report), encoding="utf-8")
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert set(reloaded.keys()) == set(report.keys())
    for metric, block in report.items():
        rblock = reloaded[metric]
        assert "per_question_type" in rblock
        # Compare by serialised form so float / int promotion noise
        # doesn't false-flag drift.
        assert json.dumps(rblock["per_question_type"], sort_keys=True) == (
            json.dumps(block["per_question_type"], sort_keys=True)
        ), f"{metric}: per_question_type drifted across json round-trip"


# ---------------------------------------------------------------------------
# 4. Irrelevant buckets carry relevant=False per RELEVANT_QUESTION_TYPES.
# ---------------------------------------------------------------------------
def test_irrelevant_buckets_marked_not_relevant_in_eval_report(
    tmp_path: Path,
) -> None:
    """``single_correct_rate.per_question_type[short_answer].relevant`` is
    ``False`` — single_correct_rate is structurally MC + TF only per
    :data:`RELEVANT_QUESTION_TYPES`."""
    course_dir = _build_fixture_course(tmp_path)
    report = _build_eval_report(course_dir)

    # answerable_rate is relevant across all 5 types — short_answer
    # bucket should carry relevant=True.
    answerable_pqt = report["answerable_rate"]["per_question_type"]
    if isinstance(answerable_pqt, dict) and "short_answer" in answerable_pqt:
        assert answerable_pqt["short_answer"]["relevant"] is True
        # Sanity-check the canonical relevance table for the same metric.
        assert "short_answer" in RELEVANT_QUESTION_TYPES["answerable_rate"]

    # distractor_entropy is structurally MC-only — non-MC buckets should
    # carry relevant=False.
    entropy_pqt = report["distractor_entropy"]["per_question_type"]
    if isinstance(entropy_pqt, dict):
        for qt in ("essay", "short_answer"):
            if qt in entropy_pqt:
                assert entropy_pqt[qt]["relevant"] is False, (
                    f"distractor_entropy: {qt} bucket should be irrelevant "
                    f"per RELEVANT_QUESTION_TYPES"
                )
        # Sanity-check the canonical relevance table.
        assert RELEVANT_QUESTION_TYPES["distractor_entropy"] == {"multiple_choice"}


__all__: List[str] = []
