"""Wave 92 — Behavioral-invariant tests.

Synthetic pedagogy graphs; mocked model_callables. The three
invariant classes share a fixture that builds a small graph with
prereq edges, at_bloom_level edges, and Misconception nodes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.eval.invariants import (  # noqa: E402
    BloomLevelInvariant,
    MisconceptionRejectionInvariant,
    PrerequisiteOrderInvariant,
)


def _build_course(tmp_path: Path) -> Path:
    course = tmp_path / "tst-101"
    (course / "graph").mkdir(parents=True)
    nodes = [
        {"id": "bloom:remember", "class": "BloomLevel", "level": "remember"},
        {"id": "bloom:apply", "class": "BloomLevel", "level": "apply"},
        {
            "id": "mc_001", "class": "Misconception",
            "label": "RDF triples are like SQL rows",
            "statement": "An RDF triple is the same as a row in a relational table.",
        },
        {
            "id": "mc_002", "class": "Misconception",
            "label": "SHACL is RDF Schema",
            "statement": "SHACL is just an alternative spelling of RDFS.",
        },
        {"id": "concept_rdf", "class": "Concept", "label": "RDF"},
        {"id": "concept_shacl", "class": "Concept", "label": "SHACL"},
    ]
    edges = [
        {"source": "concept_rdf", "target": "concept_shacl", "relation_type": "prerequisite_of"},
        {"source": "concept_shacl", "target": "concept_rdf", "relation_type": "prerequisite_of"},
        {"source": "chunk_01", "target": "bloom:remember", "relation_type": "at_bloom_level"},
        {"source": "chunk_02", "target": "bloom:apply", "relation_type": "at_bloom_level"},
        {"source": "mc_001", "target": "concept_rdf", "relation_type": "interferes_with"},
        {"source": "mc_002", "target": "concept_shacl", "relation_type": "interferes_with"},
    ]
    (course / "graph" / "pedagogy_graph.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges}), encoding="utf-8",
    )
    return course


# ---------------------------------------------------------------------- #
# PrerequisiteOrderInvariant                                              #
# ---------------------------------------------------------------------- #


def test_prerequisite_pass_when_response_mentions_prereq(tmp_path):
    course = _build_course(tmp_path)
    inv = PrerequisiteOrderInvariant(course, max_prompts=10)

    def model(prompt: str) -> str:
        # Always mentions both concepts as prerequisites — pass.
        return "First learn about concept_rdf, then concept_shacl."

    out = inv.evaluate(model)
    assert out["passed"] >= 1
    assert out["pass_rate"] > 0.0


def test_prerequisite_pass_when_response_refuses(tmp_path):
    course = _build_course(tmp_path)
    inv = PrerequisiteOrderInvariant(course, max_prompts=10)

    def refuser(prompt: str) -> str:
        return "I cannot answer without prerequisite background knowledge."

    out = inv.evaluate(refuser)
    # Each prompt should pass via the refusal pattern
    assert out["pass_rate"] == 1.0


def test_prerequisite_fail_when_response_skips_prereq(tmp_path):
    course = _build_course(tmp_path)
    inv = PrerequisiteOrderInvariant(course, max_prompts=10)

    def confident(prompt: str) -> str:
        # Confidently answers without naming any prerequisite.
        return "It is a standard formalism. Nothing else needed."

    out = inv.evaluate(confident)
    assert out["pass_rate"] == 0.0


# ---------------------------------------------------------------------- #
# BloomLevelInvariant                                                     #
# ---------------------------------------------------------------------- #


def test_bloom_remember_pass_with_definition(tmp_path):
    course = _build_course(tmp_path)
    inv = BloomLevelInvariant(course, max_per_level=2)

    def definitional(prompt: str) -> str:
        # Always responds with definition language; passes
        # only when the prompt's bloom level is "remember".
        return "It is defined as a fundamental construct of the language."

    out = inv.evaluate(definitional)
    # We synthesised at_bloom_level edges for remember + apply.
    # Definitional response only matches the remember pattern;
    # apply prompts will fail because no procedure language.
    pass_rate = out["pass_rate"]
    # At least the remember bucket passes.
    assert any(p["outcome"] == "pass" for p in out["per_prompt"])


def test_bloom_apply_pass_with_procedure(tmp_path):
    course = _build_course(tmp_path)
    inv = BloomLevelInvariant(course, max_per_level=2)

    def procedural(prompt: str) -> str:
        return "First, apply the operation. Then run the next step. Finally, write the result."

    out = inv.evaluate(procedural)
    apply_results = [p for p in out["per_prompt"] if p["bloom_level"] == "apply"]
    if apply_results:
        assert all(p["outcome"] == "pass" for p in apply_results)


# ---------------------------------------------------------------------- #
# MisconceptionRejectionInvariant                                         #
# ---------------------------------------------------------------------- #


def test_misconception_reject_passes(tmp_path):
    course = _build_course(tmp_path)
    inv = MisconceptionRejectionInvariant(course)

    def rejector(prompt: str) -> str:
        return "False — this is a common misconception. Actually, the correct view is..."

    out = inv.evaluate(rejector)
    assert out["pass_rate"] == 1.0
    assert out["passed"] == out["total"] == 2


def test_misconception_accept_fails(tmp_path):
    course = _build_course(tmp_path)
    inv = MisconceptionRejectionInvariant(course)

    def naive(prompt: str) -> str:
        return "Yes, that's exactly right."

    out = inv.evaluate(naive)
    assert out["pass_rate"] == 0.0


def test_misconception_target_concepts_in_results(tmp_path):
    course = _build_course(tmp_path)
    inv = MisconceptionRejectionInvariant(course)
    out = inv.evaluate(lambda p: "false")
    for entry in out["per_prompt"]:
        # Each misconception was wired with at least one
        # interferes_with edge in the fixture.
        assert entry["target_concepts"]
