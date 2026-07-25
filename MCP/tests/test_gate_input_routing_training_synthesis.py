"""Gate-input coverage for the ``training_synthesis`` phase.

Every gate on that phase resolved to ``__no_builder_registered__``, so the
executor skipped all ten -- five of them critical -- and the training corpus
shipped with no validation at all. A real run emitted 12,397 pairs of which
86% carried raw internal chunk identifiers ("explain how 'Complete
Factorization' relates to '<course>_chunk_00633'") and not one gate fired.

These tests pin the wiring so the phase cannot go dark again.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from MCP.hardening.gate_input_routing import (
    _build_training_synthesis,
    default_router,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = PROJECT_ROOT / "config" / "workflows.yaml"


def _training_synthesis_gates():
    cfg = yaml.safe_load(WORKFLOWS.read_text(encoding="utf-8"))
    for phase in cfg["workflows"]["textbook_to_course"]["phases"]:
        if phase["name"] == "training_synthesis":
            return phase.get("validation_gates") or []
    raise AssertionError("textbook_to_course has no training_synthesis phase")


def _corpus(tmp_path: Path) -> Path:
    """Minimal corpus tree shaped like the Courseforge trainforge export."""
    specs = tmp_path / "trainforge" / "training_specs"
    specs.mkdir(parents=True)
    (specs / "instruction_pairs.jsonl").write_text("", encoding="utf-8")
    (specs / "preference_pairs.jsonl").write_text("", encoding="utf-8")
    graph = tmp_path / "trainforge" / "graph"
    graph.mkdir(parents=True)
    (graph / "concept_graph_semantic.json").write_text("{}", encoding="utf-8")
    (graph / "pedagogy_graph.json").write_text("{}", encoding="utf-8")
    return specs


def _outputs(specs: Path) -> dict:
    return {
        "training_synthesis": {
            "instruction_pairs_path": str(specs / "instruction_pairs.jsonl"),
            "preference_pairs_path": str(specs / "preference_pairs.jsonl"),
        },
    }


def test_no_training_synthesis_gate_is_unrouted(tmp_path: Path):
    """The regression that let a poisoned corpus through: not one gate on
    this phase may resolve to __no_builder_registered__."""
    router = default_router()
    outputs = _outputs(_corpus(tmp_path))
    params = {"course_name": "demo-course"}

    unrouted = []
    for gate in _training_synthesis_gates():
        _inputs, missing = router.build(gate["validator"], outputs, params)
        if "__no_builder_registered__" in missing:
            unrouted.append((gate["gate_id"], gate["validator"]))

    assert not unrouted, (
        "gates on training_synthesis have no input builder, so the executor "
        f"silently skips them: {unrouted}"
    )


def test_every_critical_training_synthesis_gate_resolves(tmp_path: Path):
    """A critical gate that skips is indistinguishable from one that passed."""
    router = default_router()
    outputs = _outputs(_corpus(tmp_path))
    params = {"course_name": "demo-course"}

    skipped = []
    for gate in _training_synthesis_gates():
        if gate.get("severity") != "critical":
            continue
        _inputs, missing = router.build(gate["validator"], outputs, params)
        if missing:
            skipped.append((gate["gate_id"], missing))

    assert not skipped, f"critical gates skipped for missing inputs: {skipped}"


def test_builder_derives_corpus_dirs_from_pairs_path(tmp_path: Path):
    """course_dir / training_specs_dir come off instruction_pairs_path,
    because libv2_archival (which mints the LibV2 course dir) has not run
    yet when this phase is gated."""
    specs = _corpus(tmp_path)
    inputs, missing = _build_training_synthesis(
        _outputs(specs), {"course_name": "demo-course"}
    )

    assert not missing
    assert inputs["training_specs_dir"] == str(specs)
    assert inputs["course_dir"] == str(specs.parent)
    assert inputs["course_slug"] == "demo-course"


def test_builder_falls_back_to_on_disk_graphs(tmp_path: Path):
    """min_edge_count needs BOTH graph paths, and pedagogy_graph.json is
    not a phase output here -- it must be found beside the corpus."""
    specs = _corpus(tmp_path)
    inputs, _missing = _build_training_synthesis(
        _outputs(specs), {"course_name": "demo-course"}
    )

    graph_dir = specs.parent / "graph"
    assert inputs["concept_graph_path"] == str(
        graph_dir / "concept_graph_semantic.json"
    )
    assert inputs["pedagogy_graph_path"] == str(graph_dir / "pedagogy_graph.json")


def test_builder_skips_when_no_pairs_exist(tmp_path: Path):
    """With no corpus there is nothing to audit, so the gate skips rather
    than passing vacuously."""
    inputs, missing = _build_training_synthesis({}, {"course_name": "demo"})

    assert missing == ["instruction_pairs_path"]
    assert "instruction_pairs_path" not in inputs


def test_curie_anchoring_keeps_phase3_faildoud_marker(tmp_path: Path):
    """CurieAnchoringValidator is wired at two placements. The Phase 3
    outline placement is a known YAML misnomer and must keep surfacing as a
    structured skip; only the training_synthesis placement builds inputs."""
    router = default_router()
    dotted = "lib.validators.curie_anchoring.CurieAnchoringValidator"

    _inputs, missing = router.build(dotted, {"content_generation_outline": {}}, {})
    assert missing == ["wrong_validator_class"]

    inputs, missing = router.build(dotted, _outputs(_corpus(tmp_path)), {})
    assert not missing
    assert "instruction_pairs_path" in inputs
