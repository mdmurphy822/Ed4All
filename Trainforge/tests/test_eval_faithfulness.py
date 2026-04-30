"""Wave 92 — FaithfulnessEvaluator tests.

All model_callables are mocked. The harness treats held-out edges as
ground-truth True statements; it scores responses on whether they
affirm the statement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.faithfulness import FaithfulnessEvaluator  # noqa: E402
from Trainforge.eval.holdout_builder import HoldoutBuilder  # noqa: E402


def _build_course(tmp_path: Path) -> Path:
    course = tmp_path / "tst-101"
    (course / "graph").mkdir(parents=True)
    edges = [
        {"source": "concept_a", "target": "concept_b", "relation_type": "prerequisite_of"},
        {"source": "concept_b", "target": "concept_c", "relation_type": "prerequisite_of"},
        {"source": "concept_c", "target": "concept_d", "relation_type": "prerequisite_of"},
        {"source": "concept_d", "target": "concept_e", "relation_type": "prerequisite_of"},
        {"source": "concept_e", "target": "concept_f", "relation_type": "prerequisite_of"},
        {"source": "concept_f", "target": "concept_g", "relation_type": "prerequisite_of"},
        {"source": "concept_g", "target": "concept_h", "relation_type": "prerequisite_of"},
        {"source": "concept_h", "target": "concept_i", "relation_type": "prerequisite_of"},
        {"source": "concept_i", "target": "concept_j", "relation_type": "prerequisite_of"},
        {"source": "concept_j", "target": "concept_k", "relation_type": "prerequisite_of"},
    ]
    (course / "graph" / "pedagogy_graph.json").write_text(
        json.dumps({"nodes": [], "edges": edges}), encoding="utf-8",
    )
    HoldoutBuilder(course, holdout_pct=0.5, seed=42).build()
    return course


def test_perfect_model_scores_one(tmp_path):
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"

    def perfect(prompt: str) -> str:
        return "Yes, that holds."

    fr = FaithfulnessEvaluator(holdout, perfect).evaluate()
    assert fr["accuracy"] == 1.0
    assert fr["correct"] == fr["total_questions"]


def test_always_no_model_scores_zero(tmp_path):
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"

    def negative(prompt: str) -> str:
        return "No, that is not correct."

    fr = FaithfulnessEvaluator(holdout, negative).evaluate()
    assert fr["accuracy"] == 0.0


def test_ambiguous_model_does_not_count_correct(tmp_path):
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"

    def vague(prompt: str) -> str:
        return "Hmm, perhaps. It depends."

    fr = FaithfulnessEvaluator(holdout, vague).evaluate()
    assert fr["accuracy"] == 0.0
    assert fr["ambiguous"] >= 1


def test_max_questions_caps_probe_count(tmp_path):
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"
    fr = FaithfulnessEvaluator(holdout, lambda p: "yes", max_questions=2).evaluate()
    assert fr["total_questions"] <= 2


def test_callable_exception_does_not_crash_evaluation(tmp_path):
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"

    def boom(prompt: str) -> str:
        raise RuntimeError("model failed")

    fr = FaithfulnessEvaluator(holdout, boom).evaluate()
    assert fr["errors"]
    assert fr["accuracy"] == 0.0


def test_per_question_results_have_probe_text(tmp_path):
    course = _build_course(tmp_path)
    holdout = course / "eval" / "holdout_split.json"
    fr = FaithfulnessEvaluator(holdout, lambda p: "yes").evaluate()
    for entry in fr["per_question_results"]:
        assert "probe" in entry
        assert "edge" in entry
        assert entry["edge"]["relation_type"] == "prerequisite_of"


def test_chunk_at_difficulty_template_dropped() -> None:
    """Wave 108 / Phase B: trivially-true difficulty probes were padding
    faithfulness scores. The template lookup must NOT carry a
    chunk_at_difficulty entry; held-out edges of that type fall through
    to the generic template."""
    from Trainforge.eval.faithfulness import _RELATION_TEMPLATES
    assert "chunk_at_difficulty" not in _RELATION_TEMPLATES, (
        f"chunk_at_difficulty template must be dropped (Phase B); "
        f"current templates: {sorted(_RELATION_TEMPLATES)}"
    )


def test_evaluate_emits_yes_rate(tmp_path) -> None:
    """yes_rate must surface alongside accuracy so the gating validator
    can detect a yes-biased model even when accuracy looks high (every
    edge in the holdout split is a TRUE statement, so a 'yes always'
    model trivially scores 1.0 on faithfulness)."""
    import json
    from Trainforge.eval.faithfulness import FaithfulnessEvaluator

    split_path = tmp_path / "holdout_split.json"
    split_path.write_text(json.dumps({
        "withheld_edges": [
            {"source": "concept_a", "target": "concept_b", "relation_type": "prerequisite_of"},
            {"source": "concept_a", "target": "concept_c", "relation_type": "prerequisite_of"},
            {"source": "concept_d", "target": "concept_e", "relation_type": "prerequisite_of"},
        ],
        "probes": [],
    }), encoding="utf-8")

    yes_always = lambda _prompt: "Yes."
    result = FaithfulnessEvaluator(split_path, yes_always).evaluate()
    assert result["accuracy"] == 1.0
    assert "yes_rate" in result
    assert result["yes_rate"] == 1.0


def test_format_probe_scrubs_chunk_id_via_label_resolver() -> None:
    """Audit 2026-04-30 fix: when an edge's source is a chunk-ID
    literal, the probe must carry the chunk's human-readable label,
    not the raw `shacl_551_chunk_NNNNN` string. This is what zeroed
    adapter+RAG faithfulness on the cc07cc76 run.
    """
    from Trainforge.eval.chunk_labels import ChunkLabelResolver
    from Trainforge.eval.faithfulness import _format_probe

    resolver = ChunkLabelResolver(labels={
        "rdf_shacl_551_chunk_00270": "Validating SHACL property shapes",
    })
    edge = {
        "source": "rdf_shacl_551_chunk_00270",
        "target": "CO-18",
        "relation_type": "assesses",
    }
    probe = _format_probe(edge, resolver)
    assert "rdf_shacl_551_chunk_00270" not in probe
    assert "Validating SHACL property shapes" in probe


def test_relation_template_path_scrubs_chunk_id_with_resolver_label() -> None:
    """Audit 2026-04-30 Fix 3 explicit regression: a `teaches` edge
    whose source is a chunk-ID literal must render with the resolver's
    label, not the raw `<corpus>_chunk_NNNNN` string. The
    `_RELATION_TEMPLATES` dict has a `teaches` entry, so this exercises
    the relation-template branch (not the generic fallback)."""
    from Trainforge.eval.chunk_labels import ChunkLabelResolver
    from Trainforge.eval.faithfulness import _RELATION_TEMPLATES, _format_probe

    assert "teaches" in _RELATION_TEMPLATES, (
        "regression test guards the relation-template path; if "
        "`teaches` is removed from _RELATION_TEMPLATES, update this "
        "test to use a still-supported chunk-source relation."
    )

    resolver = ChunkLabelResolver(labels={
        "rdf_shacl_551_chunk_00270": "the chunk teaching SHACL property paths",
    })
    edge = {
        "source": "rdf_shacl_551_chunk_00270",
        "target": "concept:property_path",
        "relation_type": "teaches",
    }
    probe = _format_probe(edge, resolver)
    assert "the chunk teaching SHACL property paths" in probe
    assert "rdf_shacl_551_chunk_00270" not in probe
    # Sanity: the relation-template wording is preserved.
    assert "teach" in probe


@pytest.mark.parametrize("source,target,relation_type", [
    ("CO-18", "TO-01", "supports_outcome"),  # canonical course/terminal IDs
    ("concept_alpha", "concept_beta", "prerequisite_of"),  # raw concept IDs
    ("chunk_001", "bloom:remember", "at_bloom_level"),  # bloom-level target
    ("mc_a1b2c3d4e5f6789a", "concept_x", "interferes_with"),  # misconception ID
    ("CO-18", "concept_alpha", "made_up_relation"),  # generic-fallback template
])
def test_format_probe_passes_through_non_chunk_sources(
    source, target, relation_type,
) -> None:
    """Non-chunk node classes — concept IDs (CO-NN, TO-NN, concept_X),
    bloom-level targets (bloom:remember), and misconception IDs
    (mc_<16hex>) — must flow through ``_format_probe`` unmodified. Only
    raw chunk-ID literals (``chunk_NNN``, ``<corpus>_chunk_NNN``) get
    scrubbed by the resolver. Also covers the generic-template fallback
    path so an unknown relation_type doesn't silently mangle non-chunk
    node IDs either.
    """
    from Trainforge.eval.chunk_labels import ChunkLabelResolver
    from Trainforge.eval.faithfulness import _format_probe

    resolver = ChunkLabelResolver(labels={})
    edge = {
        "source": source,
        "target": target,
        "relation_type": relation_type,
    }
    probe = _format_probe(edge, resolver)
    # The non-chunk source/target appear verbatim in the rendered probe.
    if not resolver.is_chunk_id(source):
        assert source in probe, f"source {source!r} was unexpectedly scrubbed: {probe!r}"
    if not resolver.is_chunk_id(target):
        assert target in probe, f"target {target!r} was unexpectedly scrubbed: {probe!r}"
