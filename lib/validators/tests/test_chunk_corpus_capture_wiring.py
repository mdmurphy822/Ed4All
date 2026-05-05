"""H3 Wave W4 — DecisionCapture wiring for the six chunk-corpus validators.

Asserts that each validator emits exactly one capture event per
``validate()`` call, with the expected ``decision_type`` + the
computed-metric payload required by the H3 plan W4 § "Per-call dynamic
signals" contract.

Mirrors the kg_quality test pattern (test_kg_quality_validator.py
``_MockCapture`` + ``test_validator_emits_decision_capture_*``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.curie_anchoring import CurieAnchoringValidator  # noqa: E402
from lib.validators.min_edge_count import (  # noqa: E402
    DEFAULT_MIN_CONCEPT_NODES,
    DEFAULT_MIN_EDGE_TYPES,
    DEFAULT_MIN_EDGES,
    MinEdgeCountValidator,
)
from lib.validators.property_coverage import PropertyCoverageValidator  # noqa: E402
from lib.validators.synthesis_diversity import SynthesisDiversityValidator  # noqa: E402
from lib.validators.synthesis_leakage import SynthesisLeakageValidator  # noqa: E402
from lib.validators.synthesis_quota import SynthesisQuotaValidator  # noqa: E402


class _MockCapture:
    """Minimal DecisionCapture stub — records every log_decision call.

    Mirrors ``test_kg_quality_validator._MockCapture``; the H3 W4 plan
    explicitly cites that test as the canonical exemplar.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


# ---------------------------------------------------------------------------
# MinEdgeCountValidator
# ---------------------------------------------------------------------------


def _write_pedagogy_concept(
    tmp_path: Path,
    *,
    n_edges: int,
    n_edge_types: int,
    n_concept_nodes: int,
) -> Dict[str, Path]:
    edge_types = [f"rel_{i}" for i in range(max(1, n_edge_types))]
    edges = [
        {
            "source": f"c{i}", "target": f"c{i+1}",
            "relation_type": edge_types[i % len(edge_types)],
        }
        for i in range(n_edges)
    ]
    nodes = [{"id": f"c{i}", "label": f"concept-{i}"} for i in range(n_concept_nodes)]
    pedagogy = tmp_path / "pedagogy_graph.json"
    concept = tmp_path / "concept_graph.json"
    pedagogy.write_text(json.dumps({"edges": edges}), encoding="utf-8")
    concept.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")
    return {"pedagogy_graph_path": pedagogy, "concept_graph_path": concept}


def test_min_edge_count_emits_capture_on_pass(tmp_path: Path) -> None:
    capture = _MockCapture()
    inputs = _write_pedagogy_concept(
        tmp_path, n_edges=200, n_edge_types=5, n_concept_nodes=80,
    )
    inputs["decision_capture"] = capture
    result = MinEdgeCountValidator().validate(inputs)
    assert result.passed is True
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "min_edge_count_check"
    assert call["decision"] == "passed"
    metrics = call["metrics"]
    assert metrics["edge_count"] == 200
    assert metrics["distinct_edge_types"] == 5
    assert metrics["concept_node_count"] == 80
    assert metrics["min_edges"] == DEFAULT_MIN_EDGES
    assert metrics["min_edge_types"] == DEFAULT_MIN_EDGE_TYPES
    assert metrics["min_concept_nodes"] == DEFAULT_MIN_CONCEPT_NODES
    assert metrics["passed"] is True
    assert metrics["failure_code"] is None


def test_min_edge_count_emits_capture_on_below_floor(tmp_path: Path) -> None:
    capture = _MockCapture()
    inputs = _write_pedagogy_concept(
        tmp_path, n_edges=10, n_edge_types=2, n_concept_nodes=10,
    )
    inputs["decision_capture"] = capture
    result = MinEdgeCountValidator().validate(inputs)
    assert result.passed is False
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "min_edge_count_check"
    assert call["decision"].startswith("failed:")
    assert call["metrics"]["passed"] is False
    assert call["metrics"]["failure_code"] in {
        "PEDAGOGY_EDGES_BELOW_FLOOR",
        "PEDAGOGY_EDGE_TYPES_BELOW_FLOOR",
        "CONCEPT_NODES_BELOW_FLOOR",
    }


def test_min_edge_count_emits_capture_on_missing_inputs(tmp_path: Path) -> None:
    capture = _MockCapture()
    MinEdgeCountValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision_type"] == "min_edge_count_check"
    assert capture.calls[0]["metrics"]["failure_code"] == "MISSING_INPUTS"


# ---------------------------------------------------------------------------
# SynthesisDiversityValidator
# ---------------------------------------------------------------------------


def _write_instruction_pairs(
    path: Path, rows: List[Dict[str, Any]],
) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )
    return path


_DIVERSE_PREFIXES = [
    "Alpha begins", "Beta opens", "Gamma starts", "Delta launches",
    "Epsilon kicks", "Zeta proceeds", "Eta unfolds", "Theta initiates",
    "Iota leads", "Kappa heads", "Lambda commences", "Mu reveals",
    "Nu tackles", "Xi addresses", "Omicron probes", "Pi explores",
    "Rho analyses", "Sigma reviews", "Tau covers", "Upsilon studies",
]


def _diverse_pairs(n: int = 120) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        prefix = _DIVERSE_PREFIXES[i % len(_DIVERSE_PREFIXES)]
        rows.append({
            "template_id": f"template_{i % 12}",
            "completion": f"{prefix} body text item {i}.",
        })
    return rows


def test_synthesis_diversity_emits_capture_on_pass(tmp_path: Path) -> None:
    capture = _MockCapture()
    inst = _write_instruction_pairs(
        tmp_path / "instruction_pairs.jsonl", _diverse_pairs(200),
    )
    SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(inst),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "synthesis_diversity_check"
    metrics = call["metrics"]
    assert metrics["total_pairs"] == 200
    assert metrics["n_unique_templates"] == 12
    assert "top1_share" in metrics
    assert "top3_share" in metrics
    assert metrics["passed"] is True


def test_synthesis_diversity_emits_capture_on_collapse(tmp_path: Path) -> None:
    capture = _MockCapture()
    rows = [
        {"template_id": "x", "completion": f"Same prefix body {i}"}
        for i in range(100)
    ]
    inst = _write_instruction_pairs(tmp_path / "ip.jsonl", rows)
    result = SynthesisDiversityValidator().validate({
        "instruction_pairs_path": str(inst),
        "decision_capture": capture,
    })
    assert result.passed is False
    assert len(capture.calls) == 1
    metrics = capture.calls[0]["metrics"]
    assert metrics["n_unique_templates"] == 1
    assert metrics["passed"] is False


def test_synthesis_diversity_emits_capture_on_missing_inputs() -> None:
    capture = _MockCapture()
    SynthesisDiversityValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    assert capture.calls[0]["metrics"]["failure_code"] == "MISSING_INPUTS"


# ---------------------------------------------------------------------------
# SynthesisLeakageValidator
# ---------------------------------------------------------------------------


def _setup_leakage_corpus(
    tmp_path: Path,
    *,
    pair_rows: List[Dict[str, Any]],
    chunk_rows: List[Dict[str, Any]],
) -> Path:
    course_dir = tmp_path / "course"
    (course_dir / "training_specs").mkdir(parents=True)
    (course_dir / "imscc_chunks").mkdir(parents=True)
    inst = course_dir / "training_specs" / "instruction_pairs.jsonl"
    chunks = course_dir / "imscc_chunks" / "chunks.jsonl"
    inst.write_text(
        "\n".join(json.dumps(r) for r in pair_rows) + "\n", encoding="utf-8",
    )
    chunks.write_text(
        "\n".join(json.dumps(r) for r in chunk_rows) + "\n", encoding="utf-8",
    )
    return course_dir


def test_synthesis_leakage_emits_capture_on_pass(tmp_path: Path) -> None:
    capture = _MockCapture()
    course_dir = _setup_leakage_corpus(
        tmp_path,
        pair_rows=[
            {"chunk_id": "c1", "prompt": "p", "completion": "very different answer"},
            {"chunk_id": "c2", "prompt": "p", "completion": "another different answer"},
        ],
        chunk_rows=[
            {"id": "c1", "text": "Some chunk source text describing things."},
            {"id": "c2", "text": "Different chunk content about other ideas."},
        ],
    )
    result = SynthesisLeakageValidator().validate({
        "course_dir": str(course_dir),
        "decision_capture": capture,
    })
    assert result.passed is True
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "synthesis_leakage_check"
    metrics = call["metrics"]
    assert metrics["n_pairs_audited"] == 2
    assert metrics["verbatim_leak_count"] == 0
    assert metrics["assessment_scaffold_count"] == 0
    assert metrics["passed"] is True


def test_synthesis_leakage_emits_capture_on_verbatim_leak(tmp_path: Path) -> None:
    capture = _MockCapture()
    leaked_text = "A" * 80  # exceeds 50-char span threshold
    course_dir = _setup_leakage_corpus(
        tmp_path,
        pair_rows=[
            {"chunk_id": f"c{i}", "prompt": "p", "completion": leaked_text}
            for i in range(10)
        ],
        chunk_rows=[
            {"id": f"c{i}", "text": leaked_text + " trailing"} for i in range(10)
        ],
    )
    result = SynthesisLeakageValidator().validate({
        "course_dir": str(course_dir),
        "decision_capture": capture,
    })
    assert result.passed is False
    assert len(capture.calls) == 1
    metrics = capture.calls[0]["metrics"]
    assert metrics["verbatim_leak_count"] == 10
    assert metrics["verbatim_leak_rate"] == 1.0
    assert metrics["passed"] is False
    assert metrics["failure_code"] == "VERBATIM_LEAKAGE_ABOVE_THRESHOLD"


def test_synthesis_leakage_emits_capture_on_missing_inputs() -> None:
    capture = _MockCapture()
    SynthesisLeakageValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    assert capture.calls[0]["metrics"]["failure_code"] == "MISSING_INPUTS"


# ---------------------------------------------------------------------------
# SynthesisQuotaValidator
# ---------------------------------------------------------------------------


def test_synthesis_quota_emits_capture_on_pass(tmp_path: Path) -> None:
    capture = _MockCapture()
    course_dir = tmp_path / "course"
    (course_dir / "imscc_chunks").mkdir(parents=True)
    chunks = course_dir / "imscc_chunks" / "chunks.jsonl"
    chunks.write_text(
        "\n".join(
            json.dumps({"id": f"c{i}", "learning_outcome_refs": ["TO-01"]})
            for i in range(50)
        ) + "\n",
        encoding="utf-8",
    )
    SynthesisQuotaValidator().validate({
        "course_dir": str(course_dir),
        "instruction_variants_per_chunk": 2,
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "synthesis_quota_check"
    metrics = call["metrics"]
    assert metrics["eligible_chunks"] == 50
    assert metrics["instruction_variants"] == 2
    assert metrics["estimated_dispatches"] == 50 * 3
    assert metrics["ceiling"] == 1500
    assert metrics["passed"] is True


def test_synthesis_quota_emits_capture_on_skip_no_course_dir() -> None:
    capture = _MockCapture()
    SynthesisQuotaValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    metrics = capture.calls[0]["metrics"]
    assert metrics["skip_reason"] == "course_dir_missing"
    assert metrics["passed"] is True


def test_synthesis_quota_emits_capture_on_over_ceiling(tmp_path: Path) -> None:
    capture = _MockCapture()
    course_dir = tmp_path / "course"
    (course_dir / "imscc_chunks").mkdir(parents=True)
    chunks = course_dir / "imscc_chunks" / "chunks.jsonl"
    chunks.write_text(
        "\n".join(
            json.dumps({"id": f"c{i}", "learning_outcome_refs": ["TO-01"]})
            for i in range(2000)
        ) + "\n",
        encoding="utf-8",
    )
    SynthesisQuotaValidator().validate({
        "course_dir": str(course_dir),
        "instruction_variants_per_chunk": 1,
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    metrics = capture.calls[0]["metrics"]
    assert metrics["estimated_dispatches"] == 4000  # 2000 * (1+1)
    assert metrics["failure_code"] == "SYNTHESIS_QUOTA_OVER_CEILING"


# ---------------------------------------------------------------------------
# PropertyCoverageValidator
# ---------------------------------------------------------------------------


def _setup_property_corpus(
    tmp_path: Path, pairs: List[Dict[str, str]],
) -> Path:
    course_dir = tmp_path / "course"
    (course_dir / "training_specs").mkdir(parents=True)
    inst = course_dir / "training_specs" / "instruction_pairs.jsonl"
    inst.write_text(
        "\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8",
    )
    return course_dir


class _StubProperty:
    def __init__(self, pid: str, surface_forms: List[str], min_pairs: int):
        self.id = pid
        self.surface_forms = surface_forms
        self.min_pairs = min_pairs

    def matches(self, text: str) -> bool:
        if not text:
            return False
        return any(sf in text for sf in self.surface_forms)


class _StubManifest:
    def __init__(self, properties: List[_StubProperty]) -> None:
        self.properties = properties


def test_property_coverage_emits_capture_on_pass(tmp_path: Path) -> None:
    capture = _MockCapture()
    course_dir = _setup_property_corpus(tmp_path, [
        {"prompt": "what is owl:sameAs", "completion": "owl:sameAs is a property"}
        for _ in range(3)
    ])
    manifest = _StubManifest([
        _StubProperty("owl_sameAs", ["owl:sameAs"], min_pairs=2),
    ])
    with patch(
        "lib.ontology.property_manifest.load_property_manifest",
        return_value=manifest,
    ):
        PropertyCoverageValidator().validate({
            "course_dir": str(course_dir),
            "course_slug": "rdf-shacl-test",
            "decision_capture": capture,
        })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "property_coverage_check"
    metrics = call["metrics"]
    assert metrics["properties_declared"] == 1
    assert metrics["properties_covered"] == 1
    assert metrics["properties_below_floor"] == 0
    assert metrics["coverage_rate"] == 1.0
    assert metrics["per_property_counts"] == {"owl_sameAs": 3}
    assert metrics["passed"] is True


def test_property_coverage_emits_capture_on_below_floor(tmp_path: Path) -> None:
    capture = _MockCapture()
    course_dir = _setup_property_corpus(tmp_path, [
        {"prompt": "what is owl:sameAs", "completion": "owl:sameAs is a property"},
    ])
    manifest = _StubManifest([
        _StubProperty("owl_sameAs", ["owl:sameAs"], min_pairs=5),
        _StubProperty("rdfs_subClassOf", ["rdfs:subClassOf"], min_pairs=3),
    ])
    with patch(
        "lib.ontology.property_manifest.load_property_manifest",
        return_value=manifest,
    ):
        result = PropertyCoverageValidator().validate({
            "course_dir": str(course_dir),
            "course_slug": "rdf-shacl-test",
            "decision_capture": capture,
        })
    assert result.passed is False
    assert len(capture.calls) == 1
    metrics = capture.calls[0]["metrics"]
    assert metrics["properties_declared"] == 2
    assert metrics["properties_covered"] == 0
    assert metrics["properties_below_floor"] == 2
    assert metrics["coverage_rate"] == 0.0
    assert metrics["failure_code"] == "PROPERTY_COVERAGE_BELOW_FLOOR"


def test_property_coverage_emits_capture_on_missing_inputs() -> None:
    capture = _MockCapture()
    PropertyCoverageValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    assert capture.calls[0]["metrics"]["failure_code"] == "MISSING_INPUTS"


# ---------------------------------------------------------------------------
# CurieAnchoringValidator
# ---------------------------------------------------------------------------


def _setup_anchoring_corpus(
    tmp_path: Path,
    *,
    pair_rows: List[Dict[str, Any]],
    chunk_rows: List[Dict[str, Any]],
) -> Path:
    course_dir = tmp_path / "course"
    (course_dir / "training_specs").mkdir(parents=True)
    (course_dir / "imscc_chunks").mkdir(parents=True)
    inst = course_dir / "training_specs" / "instruction_pairs.jsonl"
    chunks = course_dir / "imscc_chunks" / "chunks.jsonl"
    inst.write_text(
        "\n".join(json.dumps(r) for r in pair_rows) + "\n", encoding="utf-8",
    )
    chunks.write_text(
        "\n".join(json.dumps(r) for r in chunk_rows) + "\n", encoding="utf-8",
    )
    return course_dir


def test_curie_anchoring_emits_capture_on_pass(tmp_path: Path) -> None:
    capture = _MockCapture()
    course_dir = _setup_anchoring_corpus(
        tmp_path,
        pair_rows=[
            {
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.understand",
                "prompt": "what is owl:sameAs",
                "completion": f"answer mentioning owl:sameAs in body {i}",
            }
            for i in range(20)
        ],
        chunk_rows=[
            {"id": f"c{i}", "text": "Discusses owl:sameAs and similar."}
            for i in range(20)
        ],
    )
    result = CurieAnchoringValidator().validate({
        "course_dir": str(course_dir),
        "decision_capture": capture,
    })
    assert result.passed is True
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "curie_anchoring_check"
    metrics = call["metrics"]
    assert metrics["mode"] == "pairs"
    assert metrics["n_pairs_audited"] == 20
    assert metrics["n_anchored_pairs"] == 20
    assert metrics["actual_pair_anchoring_rate"] == 1.0
    assert metrics["min_pair_anchoring_rate"] == 0.95
    assert metrics["passed"] is True


def test_curie_anchoring_emits_capture_on_below_floor(tmp_path: Path) -> None:
    capture = _MockCapture()
    course_dir = _setup_anchoring_corpus(
        tmp_path,
        pair_rows=[
            {
                "chunk_id": f"c{i}",
                "template_id": "paraphrase.understand",
                "prompt": "what is the concept",
                "completion": "no curies anywhere here at all",
            }
            for i in range(10)
        ],
        chunk_rows=[
            {"id": f"c{i}", "text": "Uses owl:sameAs in source."} for i in range(10)
        ],
    )
    result = CurieAnchoringValidator().validate({
        "course_dir": str(course_dir),
        "decision_capture": capture,
    })
    assert result.passed is False
    assert len(capture.calls) == 1
    metrics = capture.calls[0]["metrics"]
    assert metrics["mode"] == "pairs"
    assert metrics["n_pairs_audited"] == 10
    assert metrics["n_anchored_pairs"] == 0
    assert metrics["actual_pair_anchoring_rate"] == 0.0
    assert metrics["failure_code"] == "PAIR_ANCHORING_BELOW_THRESHOLD"


def test_curie_anchoring_emits_capture_on_missing_inputs() -> None:
    capture = _MockCapture()
    CurieAnchoringValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    assert capture.calls[0]["metrics"]["failure_code"] == "MISSING_INPUTS"


def test_curie_anchoring_emits_capture_on_blocks_path() -> None:
    """Block-list seam (Phase 3 Subtask 51) emits with mode='blocks'."""
    capture = _MockCapture()

    class _Block:
        def __init__(self, bid: str, content: Dict[str, Any]):
            self.block_id = bid
            self.content = content

    blocks = [
        _Block("b1", {
            "curies": ["owl:sameAs"],
            "key_claims": ["mentions owl:sameAs explicitly"],
        }),
        _Block("b2", {
            "curies": ["rdfs:subClassOf"],
            "key_claims": ["mentions rdfs:subClassOf explicitly"],
        }),
    ]
    result = CurieAnchoringValidator().validate({
        "blocks": blocks,
        "decision_capture": capture,
    })
    assert result.passed is True
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "curie_anchoring_check"
    assert call["metrics"]["mode"] == "blocks"
    assert call["metrics"]["n_pairs_audited"] == 2
    assert call["metrics"]["n_anchored_pairs"] == 2


# ---------------------------------------------------------------------------
# Per-call inputs override constructor-injected capture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "validator_factory,inputs_factory",
    [
        (
            MinEdgeCountValidator,
            lambda tmp_path: _write_pedagogy_concept(
                tmp_path, n_edges=200, n_edge_types=5, n_concept_nodes=80,
            ),
        ),
        (
            SynthesisDiversityValidator,
            lambda tmp_path: {
                "instruction_pairs_path": str(
                    _write_instruction_pairs(
                        tmp_path / "ip.jsonl", _diverse_pairs(120),
                    )
                ),
            },
        ),
        (
            SynthesisQuotaValidator,
            lambda tmp_path: {},  # course_dir unset → soft-skip path emits
        ),
    ],
)
def test_per_call_capture_overrides_constructor(
    tmp_path: Path, validator_factory: Any, inputs_factory: Any,
) -> None:
    """Per-call ``inputs['decision_capture']`` wins over the
    constructor-injected one (workflow-runner dispatch path; matches
    the kg_quality precedent)."""
    constructor_capture = _MockCapture()
    per_call_capture = _MockCapture()
    validator = validator_factory(decision_capture=constructor_capture)
    inputs = inputs_factory(tmp_path)
    inputs["decision_capture"] = per_call_capture
    validator.validate(inputs)
    assert len(per_call_capture.calls) == 1
    assert len(constructor_capture.calls) == 0
