"""Wave 23 Sub-task A tests — per-gate input routing.

Before Wave 23, ``TaskExecutor.execute_phase`` invoked
``ValidationGateManager.run_phase_gates`` with a generic
``{'artifacts': ..., 'results': ...}`` blob regardless of the
validator's input shape. ``PageObjectivesValidator``,
``ContentStructureValidator``, and friends silently returned
MISSING_INPUT issues that the ``on_fail: warn`` severity swallowed —
every gate either skipped unnoticed or returned VALIDATOR_ERROR.

This suite locks in the per-validator input-builder registry so
adding a new validator is a one-line registry edit, not an executor
hack.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from MCP.hardening.gate_input_routing import (
    GateInputRouter,
    default_router,
)

# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _make_phase_outputs(**kwargs) -> Dict[str, Dict[str, Any]]:
    """Build a minimal phase_outputs dict with explicit keys."""
    return {k: v for k, v in kwargs.items()}


# ---------------------------------------------------------------------- #
# Registry smoke
# ---------------------------------------------------------------------- #


def test_default_router_registers_every_shipping_validator():
    """Every validator in config/workflows.yaml should have a builder."""
    r = default_router()
    # Spot-check each validator dotted path we know ships today.
    expected = {
        "lib.validators.content.ContentStructureValidator",
        "lib.validators.page_objectives.PageObjectivesValidator",
        "lib.validators.source_refs.PageSourceRefValidator",
        "lib.validators.imscc.IMSCCValidator",
        "DART.pdf_converter.wcag_validator.WCAGValidator",
        "lib.validators.oscqr.OSCQRValidator",
        "lib.validators.dart_markers.DartMarkersValidator",
        "lib.validators.assessment.AssessmentQualityValidator",
        "lib.validators.assessment.FinalQualityValidator",
        "lib.validators.bloom.BloomAlignmentValidator",
        "lib.validators.leak_check.LeakCheckValidator",
        "lib.validators.content_facts.ContentFactValidator",
        "lib.validators.question_quality.QuestionQualityValidator",
        "lib.validators.libv2_manifest.LibV2ManifestValidator",
        # Activated dormant gate — must have a builder so it actually runs.
        "lib.validators.kg_quality.KGQualityValidator",
    }
    assert expected.issubset(set(r.builders.keys())), (
        f"Missing registrations: {expected - set(r.builders.keys())}"
    )


# ---------------------------------------------------------------------- #
# Per-validator builders
# ---------------------------------------------------------------------- #


def test_page_objectives_builder_gets_content_dir(tmp_path: Path):
    """PageObjectivesValidator expects a content_dir kwarg."""
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "index.html").write_text("<html></html>", encoding="utf-8")

    phase_outputs = _make_phase_outputs(
        content_generation={
            "content_paths": str(content_dir / "index.html"),
            "_completed": True,
        },
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.page_objectives.PageObjectivesValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert "content_dir" in inputs
    assert Path(inputs["content_dir"]).exists()


def test_page_objectives_builder_skips_when_content_dir_missing():
    """Required input absent → missing list non-empty (skip, not pass)."""
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.page_objectives.PageObjectivesValidator",
        {},
        {},
    )
    assert missing == ["content_dir"], (
        "PageObjectives should skip when content_dir can't be resolved, "
        "not silently pass."
    )


def test_content_structure_builder_resolves_html_path(tmp_path: Path):
    """ContentStructureValidator needs html_path or html_content."""
    html = tmp_path / "out.html"
    html.write_text("<h1>hi</h1>", encoding="utf-8")

    phase_outputs = _make_phase_outputs(
        dart_conversion={"output_path": str(html)},
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.content.ContentStructureValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert inputs["html_path"] == str(html)


def test_source_refs_builder_composes_page_paths_and_staging(tmp_path: Path):
    """PageSourceRefValidator needs page_paths + staging_dir + smm path."""
    html = tmp_path / "week_1" / "page.html"
    html.parent.mkdir(parents=True)
    html.write_text("<html></html>", encoding="utf-8")
    smm = tmp_path / "smm.json"
    smm.write_text("{}", encoding="utf-8")

    phase_outputs = _make_phase_outputs(
        dart_conversion={"output_paths": str(html)},
        staging={"staging_dir": str(tmp_path / "staging")},
        source_mapping={"source_module_map_path": str(smm)},
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.source_refs.PageSourceRefValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert inputs["page_paths"] == [str(html)]
    assert inputs["staging_dir"] == str(tmp_path / "staging")
    assert inputs["source_module_map_path"] == str(smm)


def test_imscc_builder_prefers_package_path():
    """IMSCCValidator needs imscc_path."""
    phase_outputs = _make_phase_outputs(
        packaging={"package_path": "/tmp/course.imscc"},
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.imscc.IMSCCValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert inputs["imscc_path"] == "/tmp/course.imscc"


def test_oscqr_builder_runs_without_any_required_inputs():
    """OSCQRValidator is a stub — never skip it, just forward what we have."""
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.oscqr.OSCQRValidator",
        {},
        {},
    )
    # OSCQR has no required inputs — it's a stub validator. Building
    # empty inputs is valid.
    assert missing == []


def test_unknown_validator_falls_through_with_warning(caplog):
    """Unknown validator dotted path → mark as missing, log warning."""
    r = default_router()
    with caplog.at_level(logging.WARNING):
        inputs, missing = r.build(
            "lib.validators.not_a_real.NotARealValidator",
            {},
            {},
        )
    assert missing == ["__no_builder_registered__"]
    assert any(
        "No gate-input builder registered" in rec.getMessage()
        for rec in caplog.records
    )


def test_libv2_manifest_builder_resolves_from_archival_phase():
    """LibV2ManifestValidator needs manifest_path + course_dir."""
    phase_outputs = _make_phase_outputs(
        libv2_archival={
            "manifest_path": "/tmp/course/manifest.json",
            "course_dir": "/tmp/course",
        },
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.libv2_manifest.LibV2ManifestValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert inputs["manifest_path"] == "/tmp/course/manifest.json"
    assert inputs["course_dir"] == "/tmp/course"


def test_libv2_manifest_builder_skips_when_no_manifest():
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.libv2_manifest.LibV2ManifestValidator",
        {},
        {},
    )
    assert "manifest_path" in missing


def test_register_new_validator_does_not_require_executor_edits():
    """Registry is data-driven — new validator = one register() call."""
    def _my_builder(outputs, params):
        return {"custom_key": "yes"}, []

    r = GateInputRouter()
    r.register("my.new.Validator", _my_builder)
    inputs, missing = r.build("my.new.Validator", {}, {})
    assert missing == []
    assert inputs == {"custom_key": "yes"}


def test_builder_exception_marks_gate_as_skipped(caplog):
    """A builder that raises must not crash the executor."""
    def _bad_builder(outputs, params):
        raise RuntimeError("oops")

    r = GateInputRouter()
    r.register("my.broken.Validator", _bad_builder)
    with caplog.at_level(logging.WARNING):
        inputs, missing = r.build("my.broken.Validator", {}, {})
    assert missing == ["__builder_error__"]
    assert any("raised:" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------- #
# W1 — Phase 3 / 3.5 / 4 Courseforge two-pass validator wiring.
# Closes the no-builder fallthrough that stamped these gates passed=True
# via waiver_info["skipped"]="true".
# ---------------------------------------------------------------------- #


W1_VALIDATOR_DOTTED_PATHS = [
    # Group A — Block-input validators (rewrite_*).
    "Courseforge.router.inter_tier_gates.BlockCurieAnchoringValidator",
    "Courseforge.router.inter_tier_gates.BlockContentTypeValidator",
    "Courseforge.router.inter_tier_gates.BlockPageObjectivesValidator",
    "Courseforge.router.inter_tier_gates.BlockSourceRefValidator",
    # Group B — Rewrite-emit shape + sentence-grounding.
    "lib.validators.rewrite_html_shape.RewriteHtmlShapeValidator",
    "lib.validators.rewrite_source_grounding.RewriteSourceGroundingValidator",
    # Group C — Block-only SHACL.
    "lib.validators.courseforge_outline_shacl.CourseforgeOutlineShaclValidator",
    # Group D — Phase-4 statistical-tier validators.
    "lib.validators.objective_assessment_similarity.ObjectiveAssessmentSimilarityValidator",
    "lib.validators.concept_example_similarity.ConceptExampleSimilarityValidator",
    "lib.validators.objective_roundtrip_similarity.ObjectiveRoundtripSimilarityValidator",
    "lib.validators.bloom_classifier_disagreement.BloomClassifierDisagreementValidator",
    # Group E — degraded fail-loud entries (chunk-shape; YAML mis-points).
    "lib.validators.curie_anchoring.CurieAnchoringValidator",
    "lib.validators.content_type.ContentTypeValidator",
]


@pytest.mark.parametrize("validator_path", W1_VALIDATOR_DOTTED_PATHS)
def test_thirteen_courseforge_two_pass_validators_have_builders(
    validator_path: str,
) -> None:
    """Every W1 validator dotted path must have a builder registered.

    Before W1, these 13 gates short-circuited via the no-builder
    fallthrough (``__no_builder_registered__``) and the executor
    stamped them ``passed=True, waiver_info["skipped"]="true"``.
    Registering a builder forces the gate through the structured-skip
    path (or runs it for real when inputs resolve).
    """
    r = default_router()
    assert validator_path in r.builders, (
        f"W1 regression: {validator_path} has no builder registered "
        "in default_router() — gate will silently pass via the "
        "no-builder skip path. Add r.register(...) in "
        "MCP/hardening/gate_input_routing.py::default_router."
    )


# ---------------------------------------------------------------------- #
# Per-builder happy-path fixtures + tests
# ---------------------------------------------------------------------- #


def _write_blocks_jsonl(path: Path, entries: List[Dict[str, Any]]) -> Path:
    """Write a minimal blocks JSONL file the hydrator can consume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


def _minimal_block_entry(
    block_id: str = "p1#concept_x_0",
    block_type: str = "concept",
) -> Dict[str, Any]:
    """Minimal Block JSONL entry the hydrator accepts.

    ``block_id`` + ``block_type`` are the only required keys after the
    hydrator's defaults fill in. ``page_id`` defaults to ``block_id``.
    """
    return {
        "block_id": block_id,
        "block_type": block_type,
        "page_id": "page_1",
        "sequence": 0,
        "content": "<p>concept body</p>",
    }


def _make_outline_phase_outputs(
    blocks_path: Path,
    objectives_path: str = "",
    manifest_path: str = "",
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {
        "content_generation_outline": {
            "blocks_outline_path": str(blocks_path),
            "_completed": True,
        },
    }
    if objectives_path:
        out["course_planning"] = {"objectives_path": objectives_path}
    if manifest_path:
        out["staging"] = {"manifest_path": manifest_path}
    return out


def _make_rewrite_phase_outputs(
    blocks_path: Path,
    objectives_path: str = "",
    manifest_path: str = "",
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {
        "content_generation_rewrite": {
            "blocks_final_path": str(blocks_path),
            "_completed": True,
        },
    }
    if objectives_path:
        out["course_planning"] = {"objectives_path": objectives_path}
    if manifest_path:
        out["staging"] = {"manifest_path": manifest_path}
    return out


def test_group_a_block_input_builder_hydrates_blocks(tmp_path: Path) -> None:
    """Group A: Block-input builder produces ``blocks`` + path fixtures."""
    blocks_path = _write_blocks_jsonl(
        tmp_path / "blocks_final.jsonl",
        [_minimal_block_entry()],
    )
    objectives = tmp_path / "synthesized_objectives.json"
    objectives.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "staging_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    r = default_router()
    inputs, missing = r.build(
        "Courseforge.router.inter_tier_gates.BlockCurieAnchoringValidator",
        _make_rewrite_phase_outputs(
            blocks_path,
            objectives_path=str(objectives),
            manifest_path=str(manifest),
        ),
        {},
    )
    assert missing == []
    assert "blocks" in inputs and len(inputs["blocks"]) == 1
    assert inputs["blocks"][0].block_id == "p1#concept_x_0"
    assert inputs["objectives_path"] == str(objectives)
    assert inputs["manifest_path"] == str(manifest)


def test_group_a_outline_seam_pulls_blocks_outline_path(
    tmp_path: Path,
) -> None:
    """outline_* gates must read blocks_outline_path, not blocks_final_path.

    The four Block validators are wired via the rewrite-tier shim by
    default (rewrite_* gates are the canonical post-rewrite seam), but
    the inter-tier seam re-uses the same dotted paths. Confirm fallback
    resolution picks up the outline emit when only that's present.
    """
    blocks_path = _write_blocks_jsonl(
        tmp_path / "blocks_outline.jsonl",
        [_minimal_block_entry()],
    )

    r = default_router()
    inputs, missing = r.build(
        "Courseforge.router.inter_tier_gates.BlockPageObjectivesValidator",
        _make_outline_phase_outputs(blocks_path),
        {},
    )
    # The default-registered shim is rewrite-tier; when the rewrite
    # phase is absent it falls back to the outline emit.
    assert missing == []
    assert "blocks" in inputs and len(inputs["blocks"]) == 1


def test_group_a_skips_when_no_blocks_path() -> None:
    """No outline / rewrite phase output → structured skip, not silent pass."""
    r = default_router()
    inputs, missing = r.build(
        "Courseforge.router.inter_tier_gates.BlockSourceRefValidator",
        {},
        {},
    )
    assert missing  # non-empty
    assert "blocks_outline_path|blocks_final_path" in missing[0] or missing


def test_group_b_rewrite_html_shape_uses_block_input(tmp_path: Path) -> None:
    """RewriteHtmlShapeValidator wires through the rewrite-tier shim."""
    blocks_path = _write_blocks_jsonl(
        tmp_path / "blocks_final.jsonl",
        [_minimal_block_entry()],
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.rewrite_html_shape.RewriteHtmlShapeValidator",
        _make_rewrite_phase_outputs(blocks_path),
        {},
    )
    assert missing == []
    assert "blocks" in inputs


def test_group_b_rewrite_source_grounding_surfaces_chunks(
    tmp_path: Path,
) -> None:
    """RewriteSourceGroundingValidator gets ``source_chunks`` from manifest."""
    blocks_path = _write_blocks_jsonl(
        tmp_path / "blocks_final.jsonl",
        [_minimal_block_entry()],
    )
    manifest_path = tmp_path / "staging_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "files": [
                {"source_id": "dart:foo#b1", "text": "chunk text alpha"},
                {"sourceId": "dart:foo#b2", "plain_text": "chunk text beta"},
            ],
        }),
        encoding="utf-8",
    )

    r = default_router()
    inputs, missing = r.build(
        "lib.validators.rewrite_source_grounding.RewriteSourceGroundingValidator",
        _make_rewrite_phase_outputs(
            blocks_path, manifest_path=str(manifest_path),
        ),
        {},
    )
    assert missing == []
    assert "blocks" in inputs
    assert inputs.get("source_chunks") == {
        "dart:foo#b1": "chunk text alpha",
        "dart:foo#b2": "chunk text beta",
    }


def test_group_c_shacl_returns_blocks_only(tmp_path: Path) -> None:
    """CourseforgeOutlineShaclValidator gets ``blocks`` only — no extra keys."""
    blocks_path = _write_blocks_jsonl(
        tmp_path / "blocks_final.jsonl",
        [_minimal_block_entry()],
    )
    objectives = tmp_path / "synthesized_objectives.json"
    objectives.write_text("{}", encoding="utf-8")

    r = default_router()
    inputs, missing = r.build(
        "lib.validators.courseforge_outline_shacl.CourseforgeOutlineShaclValidator",
        _make_rewrite_phase_outputs(
            blocks_path, objectives_path=str(objectives),
        ),
        {},
    )
    assert missing == []
    assert set(inputs.keys()) == {"blocks"}


def test_group_d_statistical_input_keeps_blocks_and_objectives(
    tmp_path: Path,
) -> None:
    """Statistical-tier builder surfaces ``blocks`` + ``objectives_path``."""
    blocks_path = _write_blocks_jsonl(
        tmp_path / "blocks_final.jsonl",
        [_minimal_block_entry()],
    )
    objectives = tmp_path / "synthesized_objectives.json"
    objectives.write_text("{}", encoding="utf-8")

    r = default_router()
    for dotted in (
        "lib.validators.objective_assessment_similarity.ObjectiveAssessmentSimilarityValidator",
        "lib.validators.concept_example_similarity.ConceptExampleSimilarityValidator",
        "lib.validators.objective_roundtrip_similarity.ObjectiveRoundtripSimilarityValidator",
        "lib.validators.bloom_classifier_disagreement.BloomClassifierDisagreementValidator",
        # GPT Feedback v2 Wave 1.7 W1.7.C — same statistical-tier surface.
        "lib.validators.block_objective_delivery.BlockObjectiveDeliveryValidator",
    ):
        inputs, missing = r.build(
            dotted,
            _make_rewrite_phase_outputs(
                blocks_path, objectives_path=str(objectives),
            ),
            {},
        )
        assert missing == [], f"{dotted} should resolve when blocks + objectives are present"
        assert "blocks" in inputs
        assert inputs.get("objectives_path") == str(objectives)
        # Statistical-tier surface is intentionally pruned: no
        # manifest_path / staging_dir / valid_*_ids leakage. Wave 1.7
        # W1.7.C Drift B fix may add ``objective_statements`` +
        # ``objectives`` when the JSON has LO entries; the empty ``{}``
        # fixture flattens to zero entries so the legacy two-key shape
        # still holds here.
        assert set(inputs.keys()) <= {
            "blocks", "objectives_path",
            "objective_statements", "objectives",
        }


def test_group_d_statistical_input_drift_b_fix_populates_objective_maps(
    tmp_path: Path,
) -> None:
    """Wave 1.7 W1.7.C — Drift B: builder loads + flattens
    ``synthesized_objectives.json`` and surfaces ``objective_statements``
    + ``objectives`` for the statistical-tier validators.

    Pre-Wave-1.7 the builder only emitted ``{blocks, objectives_path}``,
    so ``ObjectiveAssessmentSimilarityValidator`` silently degraded to
    ``OBJECTIVE_STATEMENT_UNRESOLVED`` warnings on every block. This
    test pins the fix.
    """
    import json as _json

    blocks_path = _write_blocks_jsonl(
        tmp_path / "blocks_final.jsonl",
        [_minimal_block_entry()],
    )
    objectives = tmp_path / "synthesized_objectives.json"
    payload = {
        "terminal_objectives": [
            {
                "id": "TO-01",
                "bloom_level": "analyze",
                "bloom_verb": "analyze",
                "statement": "Analyze RDF data models.",
            },
        ],
        "chapter_objectives": [
            {
                "id": "CO-08",
                "bloom_level": "create",
                "bloom_verb": "construct",
                "statement": "Construct subclass hierarchies in Turtle.",
            },
        ],
    }
    objectives.write_text(_json.dumps(payload), encoding="utf-8")

    r = default_router()
    inputs, missing = r.build(
        "lib.validators.objective_assessment_similarity.ObjectiveAssessmentSimilarityValidator",
        _make_rewrite_phase_outputs(
            blocks_path, objectives_path=str(objectives),
        ),
        {},
    )
    assert missing == []
    assert inputs.get("objectives_path") == str(objectives)
    # Drift B fix lands these two new keys.
    statements = inputs.get("objective_statements")
    assert isinstance(statements, dict)
    assert statements.get("TO-01") == "Analyze RDF data models."
    assert statements.get("CO-08") == (
        "Construct subclass hierarchies in Turtle."
    )

    full = inputs.get("objectives")
    assert isinstance(full, dict)
    assert full.get("TO-01", {}).get("bloom_level") == "analyze"
    assert full.get("CO-08", {}).get("bloom_verb") == "construct"


def test_group_e_degraded_chunk_input_returns_wrong_validator_class() -> None:
    """Group E: chunk-shape CurieAnchoring / ContentType always skip.

    The Phase 3 outline gates wire these chunk-shape validators by a
    YAML misnomer (the Block-shape variants live under
    ``Courseforge.router.inter_tier_gates``). The router emits a
    structured ``wrong_validator_class`` skip rather than a silent
    no-builder pass. W4 corrects the YAML; until then this builder is
    fail-loud safety against drift.
    """
    r = default_router()
    for dotted in (
        "lib.validators.curie_anchoring.CurieAnchoringValidator",
        "lib.validators.content_type.ContentTypeValidator",
    ):
        inputs, missing = r.build(dotted, {}, {})
        assert missing == ["wrong_validator_class"], (
            f"{dotted} must emit a degraded structured skip "
            "(W4 corrects the YAML mis-pointer)."
        )
        assert inputs == {}


# ---------------------------------------------------------------------- #
# W4 — outline-tier inter_tier_validation gates point at Block-shape
# validators only. The chunk-shape validators (CurieAnchoringValidator /
# ContentTypeValidator under ``lib.validators.*``) misfit the Block-input
# inter-tier seam — W1 registered them as fail-loud safety-net entries
# (`wrong_validator_class` skip), W4 corrects the YAML so the seam wires
# the correct Block-shape variants. Any future YAML drift back to the
# chunk-shape paths trips this gate's allow-list.
# ---------------------------------------------------------------------- #


# Six approved ``lib.validators.*`` paths used at the outline seam (in
# course_generation::inter_tier_validation; textbook_to_course's seam
# uses a strict subset). Anything else under ``lib.validators.*`` for an
# ``outline_*`` gate trips the regression assertion.
_W4_OUTLINE_LIB_VALIDATORS_ALLOWLIST = frozenset({
    "lib.validators.objective_assessment_similarity.ObjectiveAssessmentSimilarityValidator",
    "lib.validators.concept_example_similarity.ConceptExampleSimilarityValidator",
    "lib.validators.objective_roundtrip_similarity.ObjectiveRoundtripSimilarityValidator",
    "lib.validators.bloom_classifier_disagreement.BloomClassifierDisagreementValidator",
    # W-D10 T10.1: subpackage-canonical alias for the path above.
    "lib.validators.bloom.classifier_disagreement.BloomClassifierDisagreementValidator",
    "lib.validators.courseforge_outline_shacl.CourseforgeOutlineShaclValidator",
    # Worker W7: Block-input payload validator gating assessment_item
    # distractors[] / correct_answer_index. Same Block-input shape as
    # the four Courseforge.router.inter_tier_gates.Block* siblings, but
    # lives under lib.validators.* because it's a payload-shape gate
    # rather than a structural-reference gate.
    "lib.validators.assessment_item_payload.BlockAssessmentItemPayloadValidator",
    # GPT Feedback v2 Wave 1.7 W1.7.C: tri-axis per-block-per-objective
    # delivery gate (NLI entailment / Bloom-gap / verb synonym). Same
    # statistical-tier surface as the four Phase 4 PoC validators above
    # — wired symmetrically at outline + rewrite seams.
    "lib.validators.block_objective_delivery.BlockObjectiveDeliveryValidator",
    # Pre-Wave-1.5 statistical-tier outline-seam validators — emit
    # warning-severity GateIssues against the assessment_item /
    # distractor / instructional-depth / Bloom-structural surfaces.
    # Wired in textbook_to_course::inter_tier_validation only;
    # course_generation::inter_tier_validation does not carry these.
    # Allow-list extended (silent-drift audit 2026-05-06, Class 7).
    "lib.validators.assessment_retrieval_grounding.AssessmentRetrievalGroundingValidator",
    "lib.validators.distractor_plausibility.DistractorPlausibilityValidator",
    "lib.validators.distractor_misconception_alignment.DistractorMisconceptionAlignmentValidator",
    "lib.validators.instructional_depth.InstructionalDepthValidator",
    "lib.validators.bloom_structural_enforcement.BloomStructuralEnforcementValidator",
    # W-D10 T10.1: subpackage-canonical alias for the path above.
    "lib.validators.bloom.structural_enforcement.BloomStructuralEnforcementValidator",
    # Wave 2 W2.C: structural distractor sanity gate (length / repetition
    # / negation). Wired symmetrically at outline + rewrite seams.
    "lib.validators.distractor_structural.DistractorStructuralValidator",
    # Wave 2 W2.D: critical-severity padded-distractor gate. Wired
    # symmetrically at outline + rewrite + trainforge_assessment seams.
    "lib.validators.padded_distractor.PaddedDistractorValidator",
})


def test_outline_seam_uses_block_validators() -> None:
    """W4: every ``outline_*`` gate at the inter-tier seam wires a
    Block-shape validator OR an approved statistical-tier
    ``lib.validators.*`` path.

    Pre-W4, ``outline_curie_anchoring`` and ``outline_content_type`` in
    ``textbook_to_course::inter_tier_validation`` referenced the
    chunk-shape validators (``lib.validators.curie_anchoring.*`` /
    ``lib.validators.content_type.*``) which misfit the Block-input
    seam. W1 added a fail-loud safety net in ``default_router`` so the
    misfit chunk-shape entries return ``wrong_validator_class``. W4
    repoints the YAML at the matching ``Courseforge.router.inter_tier_gates.Block*``
    classes. This test guards against drift back to the chunk-shape
    paths in either workflow.
    """
    import yaml

    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "workflows.yaml"
    )
    with config_path.open() as fh:
        workflows = yaml.safe_load(fh)

    workflows_with_inter_tier = []
    for workflow_name in ("textbook_to_course", "course_generation"):
        wf = workflows["workflows"][workflow_name]
        for phase in wf["phases"]:
            if phase["name"] != "inter_tier_validation":
                continue
            workflows_with_inter_tier.append(workflow_name)
            gates = phase.get("validation_gates", [])
            outline_gates = [
                g for g in gates if g["gate_id"].startswith("outline_")
            ]
            assert outline_gates, (
                f"{workflow_name}::inter_tier_validation has no "
                f"outline_* gates — sanity-check failed before assertions."
            )
            for gate in outline_gates:
                validator = gate["validator"]
                gate_id = gate["gate_id"]
                allowed = (
                    validator.startswith(
                        "Courseforge.router.inter_tier_gates.Block"
                    )
                    or validator in _W4_OUTLINE_LIB_VALIDATORS_ALLOWLIST
                )
                assert allowed, (
                    f"W4 regression: {workflow_name}::"
                    f"inter_tier_validation::{gate_id} points at "
                    f"{validator!r}. Outline-seam gates must wire a "
                    f"Block-shape validator (Courseforge.router."
                    f"inter_tier_gates.Block*) or one of the approved "
                    f"statistical-tier paths "
                    f"({sorted(_W4_OUTLINE_LIB_VALIDATORS_ALLOWLIST)})."
                )

    # Sanity-check: BOTH workflows have an inter_tier_validation phase.
    # If a future refactor renames or drops the phase, this assertion
    # surfaces it loudly instead of silently skipping the loop above.
    assert set(workflows_with_inter_tier) == {
        "textbook_to_course",
        "course_generation",
    }, (
        "Expected inter_tier_validation in both textbook_to_course and "
        f"course_generation; saw it in {workflows_with_inter_tier}."
    )


# ---------------------------------------------------------------------- #
# Wave2-I9 — Finding 5: chunkset_manifest / concept_graph / abcd_objective
# builders. Pre-Wave2-I9 the gate router silently skipped these validators
# via ``__no_builder_registered__``, downgrading three critical-shape
# gates to passed=True no-ops.
# ---------------------------------------------------------------------- #


WAVE2_I9_VALIDATOR_PATHS = [
    "lib.validators.chunkset_manifest.ChunksetManifestValidator",
    "lib.validators.concept_graph.ConceptGraphValidator",
    "lib.validators.abcd_objective.AbcdObjectiveValidator",
]


@pytest.mark.parametrize("validator_path", WAVE2_I9_VALIDATOR_PATHS)
def test_wave2_i9_validators_have_builders_registered(validator_path: str):
    """Every Wave2-I9 validator must resolve to a real builder.

    Regression guard against silent ``__no_builder_registered__`` skips.
    """
    r = default_router()
    assert validator_path in r.builders, (
        f"Wave2-I9 regression: no builder registered for {validator_path}; "
        f"gate will silently skip via __no_builder_registered__."
    )


# ---- ChunksetManifestValidator ---------------------------------------- #


def test_chunkset_manifest_builder_resolves_dart_manifest_from_chunks_path(
    tmp_path: Path,
):
    """DART chunking emits dart_chunks_path; manifest sits beside it."""
    chunks_dir = tmp_path / "dart_chunks"
    chunks_dir.mkdir()
    chunks_path = chunks_dir / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    phase_outputs = _make_phase_outputs(
        chunking={"dart_chunks_path": str(chunks_path)},
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.chunkset_manifest.ChunksetManifestValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert inputs["chunkset_manifest_path"] == str(chunks_dir / "manifest.json")


def test_chunkset_manifest_builder_resolves_imscc_manifest_from_chunks_path(
    tmp_path: Path,
):
    """IMSCC chunking emits imscc_chunks_path; manifest sits beside it."""
    chunks_dir = tmp_path / "imscc_chunks"
    chunks_dir.mkdir()
    chunks_path = chunks_dir / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    phase_outputs = _make_phase_outputs(
        imscc_chunking={"imscc_chunks_path": str(chunks_path)},
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.chunkset_manifest.ChunksetManifestValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert inputs["chunkset_manifest_path"] == str(chunks_dir / "manifest.json")


def test_chunkset_manifest_builder_prefers_explicit_manifest_path():
    """Explicit manifest_path on the chunking phase output wins."""
    phase_outputs = _make_phase_outputs(
        chunking={
            "dart_chunks_path": "/tmp/dart_chunks/chunks.jsonl",
            "manifest_path": "/tmp/dart_chunks/explicit_manifest.json",
        },
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.chunkset_manifest.ChunksetManifestValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert inputs["chunkset_manifest_path"] == "/tmp/dart_chunks/explicit_manifest.json"


def test_chunkset_manifest_builder_skips_when_no_chunks_path():
    """No chunking phase output → skip with structured reason."""
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.chunkset_manifest.ChunksetManifestValidator",
        {},
        {},
    )
    assert missing == ["chunkset_manifest_path"]


# ---- ConceptGraphValidator -------------------------------------------- #


def test_concept_graph_builder_resolves_from_concept_extraction_phase():
    """ConceptGraphValidator needs concept_graph_path."""
    phase_outputs = _make_phase_outputs(
        concept_extraction={
            "concept_graph_path": "/tmp/course/concept_graph/concept_graph_semantic.json",
        },
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.concept_graph.ConceptGraphValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert (
        inputs["concept_graph_path"]
        == "/tmp/course/concept_graph/concept_graph_semantic.json"
    )


def test_concept_graph_builder_falls_back_to_locate_scan():
    """Missing concept_extraction phase but the key surfaces elsewhere."""
    phase_outputs = _make_phase_outputs(
        course_planning={
            "concept_graph_path": "/tmp/course/concept_graph/concept_graph_semantic.json",
        },
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.concept_graph.ConceptGraphValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert (
        inputs["concept_graph_path"]
        == "/tmp/course/concept_graph/concept_graph_semantic.json"
    )


def test_concept_graph_builder_skips_when_path_missing():
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.concept_graph.ConceptGraphValidator",
        {},
        {},
    )
    assert missing == ["concept_graph_path"]


# ---- AbcdObjectiveValidator ------------------------------------------- #


def test_abcd_objective_builder_resolves_from_course_planning_phase():
    """AbcdObjectiveValidator needs synthesized_objectives_path."""
    phase_outputs = _make_phase_outputs(
        course_planning={
            "synthesized_objectives_path": "/tmp/course/synthesized_objectives.json",
        },
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.abcd_objective.AbcdObjectiveValidator",
        phase_outputs,
        {},
    )
    assert missing == []
    assert (
        inputs["synthesized_objectives_path"]
        == "/tmp/course/synthesized_objectives.json"
    )


def test_abcd_objective_builder_honors_workflow_params_override():
    """workflow_params.objectives_path is the canonical override."""
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.abcd_objective.AbcdObjectiveValidator",
        {},
        {"objectives_path": "/tmp/reuse/synthesized_objectives.json"},
    )
    assert missing == []
    assert (
        inputs["synthesized_objectives_path"]
        == "/tmp/reuse/synthesized_objectives.json"
    )


def test_abcd_objective_builder_skips_when_path_unresolvable():
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.abcd_objective.AbcdObjectiveValidator",
        {},
        {},
    )
    assert missing == ["synthesized_objectives_path"]


# ---- Integration: routing dispatches to the right builder ------------- #


def test_wave2_i9_router_dispatches_to_correct_builder(tmp_path: Path):
    """Top-level routing function should pick the right builder per name.

    Integration-shaped guard: a single phase_outputs dict carrying
    inputs for all three Wave2-I9 validators routes each validator
    through its own builder, producing the validator-specific shape.
    """
    chunks_dir = tmp_path / "dart_chunks"
    chunks_dir.mkdir()
    chunks_path = chunks_dir / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    phase_outputs = _make_phase_outputs(
        chunking={"dart_chunks_path": str(chunks_path)},
        concept_extraction={
            "concept_graph_path": "/tmp/concept_graph_semantic.json",
        },
        course_planning={
            "synthesized_objectives_path": "/tmp/synthesized_objectives.json",
        },
    )
    r = default_router()

    chunkset_in, chunkset_missing = r.build(
        "lib.validators.chunkset_manifest.ChunksetManifestValidator",
        phase_outputs,
        {},
    )
    assert chunkset_missing == []
    assert "chunkset_manifest_path" in chunkset_in
    assert "concept_graph_path" not in chunkset_in
    assert "synthesized_objectives_path" not in chunkset_in

    concept_in, concept_missing = r.build(
        "lib.validators.concept_graph.ConceptGraphValidator",
        phase_outputs,
        {},
    )
    assert concept_missing == []
    assert "concept_graph_path" in concept_in
    assert "chunkset_manifest_path" not in concept_in
    assert "synthesized_objectives_path" not in concept_in

    abcd_in, abcd_missing = r.build(
        "lib.validators.abcd_objective.AbcdObjectiveValidator",
        phase_outputs,
        {},
    )
    assert abcd_missing == []
    assert "synthesized_objectives_path" in abcd_in
    assert "concept_graph_path" not in abcd_in
    assert "chunkset_manifest_path" not in abcd_in


# ---------------------------------------------------------------------- #
# Disk-glob content-dir fallback (textbook_to_course robustness fix)
#
# For textbook_to_course, generated pages live at
# ``<project_export>/03_content_development/week_NN/*.html`` but the
# content-generation phase output may carry no content_paths (subagent
# dispatch). The legacy resolution arms in ``_find_content_dir`` miss
# this layout, so source_refs / page_objectives / content_structure all
# scanned the wrong (empty) directory. The disk-glob fallback derives
# the export root from a project_path signal and globs the canonical
# content subdirs. These tests lock in: (a) the new layout resolves,
# (b) the legacy ``content/`` layout still resolves, (c) the explicit
# content_paths path is byte-identical (fallback never fires).
# ---------------------------------------------------------------------- #


def _make_textbook_export(tmp_path: Path, *, n_pages: int = 3) -> Path:
    """Build a textbook_to_course project export with weekly pages.

    Layout: ``<export>/03_content_development/week_NN/page.html`` with
    a ``data-cf-source-ids`` attr so the pages mimic the real emit.
    Returns the export root.
    """
    export = tmp_path / "PROJ-PHYS_101-abc12345"
    for i in range(1, n_pages + 1):
        wk = export / "03_content_development" / f"week_{i:02d}"
        wk.mkdir(parents=True)
        (wk / "overview.html").write_text(
            f'<html><body data-cf-source-ids="src-{i}">'
            f"<h1>Week {i}</h1></body></html>",
            encoding="utf-8",
        )
    # A sibling stage dir that should NOT be picked up as content.
    (export / "01_learning_objectives").mkdir(parents=True)
    return export


def test_all_html_paths_globs_textbook_03_content_development(tmp_path: Path):
    """Scenario (a): no content_paths/content_dir, but the
    03_content_development/week_NN/*.html layout exists on disk and is
    reachable via objective_extraction.project_path."""
    from MCP.hardening.gate_input_routing import (
        _all_html_paths,
        _find_content_dir,
    )

    export = _make_textbook_export(tmp_path, n_pages=3)
    phase_outputs = _make_phase_outputs(
        objective_extraction={"project_path": str(export)},
        # content_generation deliberately carries NO content_paths
        # (subagent-dispatched emit) — this is the failing case.
        content_generation={"_completed": True},
    )

    cd = _find_content_dir(phase_outputs, {})
    assert cd is not None
    assert cd == export / "03_content_development"

    pages = _all_html_paths(phase_outputs, {})
    assert len(pages) == 3
    assert all(p.endswith("overview.html") for p in pages)
    assert all("03_content_development" in p for p in pages)


def test_source_refs_builder_resolves_via_disk_glob_fallback(tmp_path: Path):
    """The source_refs gate (PageSourceRefValidator) finds the generated
    pages via the disk-glob fallback when content_paths is absent."""
    export = _make_textbook_export(tmp_path, n_pages=2)
    phase_outputs = _make_phase_outputs(
        objective_extraction={"project_path": str(export)},
        content_generation={"_completed": True},
    )
    r = default_router()
    inputs, missing = r.build(
        "lib.validators.source_refs.PageSourceRefValidator",
        phase_outputs,
        {},
    )
    assert missing == [], "source_refs must not skip — pages exist on disk"
    assert len(inputs["page_paths"]) == 2


def test_disk_glob_fallback_resolves_export_from_workflow_params(
    tmp_path: Path,
):
    """The export root can also be resolved from workflow_params when no
    phase output surfaces a project_path."""
    from MCP.hardening.gate_input_routing import _all_html_paths

    export = _make_textbook_export(tmp_path, n_pages=2)
    phase_outputs = _make_phase_outputs(
        content_generation={"_completed": True},
    )
    pages = _all_html_paths(phase_outputs, {"project_path": str(export)})
    assert len(pages) == 2


def test_disk_glob_fallback_resolves_legacy_content_layout(tmp_path: Path):
    """Scenario (b): the legacy ``<export>/content/*.html`` flat layout
    still resolves via the disk-glob fallback."""
    from MCP.hardening.gate_input_routing import (
        _all_html_paths,
        _find_content_dir,
    )

    export = tmp_path / "PROJ-BIO_201-legacy"
    content = export / "content"
    content.mkdir(parents=True)
    (content / "module_1.html").write_text("<html></html>", encoding="utf-8")
    (content / "module_2.html").write_text("<html></html>", encoding="utf-8")

    phase_outputs = _make_phase_outputs(
        objective_extraction={"project_path": str(export)},
        content_generation={"_completed": True},
    )
    # The existing project_path/"content" arm already handles this, but
    # assert it stays green so the fallback ordering doesn't regress it.
    cd = _find_content_dir(phase_outputs, {})
    assert cd == content
    pages = _all_html_paths(phase_outputs, {})
    assert len(pages) == 2


def test_disk_glob_prefers_03_content_development_within_fallback(
    tmp_path: Path,
):
    """Within the disk-glob fallback proper, 03_content_development is
    tried before legacy content/. (When the legacy project_path/"content"
    arm fires first — i.e. a content/ dir exists — that arm wins for
    back-compat; this test isolates the fallback's own subdir priority by
    routing the export via workflow_params, which only the fallback
    consults.)"""
    from MCP.hardening.gate_input_routing import (
        _find_content_dir,
        _glob_content_dir_from_export,
    )

    export = _make_textbook_export(tmp_path, n_pages=1)
    legacy = export / "content"
    legacy.mkdir(parents=True)
    (legacy / "stale.html").write_text("<html></html>", encoding="utf-8")

    # Direct fallback helper: 03_content_development wins over content/.
    assert (
        _glob_content_dir_from_export(export)
        == export / "03_content_development"
    )

    # End-to-end via workflow_params only (no project_path phase output),
    # so the legacy project_path/"content" arm is never reached and the
    # fallback's subdir priority is what's exercised.
    phase_outputs = _make_phase_outputs(
        content_generation={"_completed": True},
    )
    cd = _find_content_dir(phase_outputs, {"project_path": str(export)})
    assert cd == export / "03_content_development"


def test_content_paths_present_is_unchanged_no_fallback(tmp_path: Path):
    """Scenario (c): when content_paths IS present, behaviour is
    byte-identical — the fallback never fires even if an export root is
    also discoverable on disk."""
    from MCP.hardening.gate_input_routing import (
        _all_html_paths,
        _find_content_dir,
    )

    # An on-disk export that WOULD be globbed if the fallback fired.
    export = _make_textbook_export(tmp_path, n_pages=5)

    # But content_generation carries explicit content_paths pointing at a
    # totally separate content/ dir.
    real_content = tmp_path / "explicit" / "content"
    real_content.mkdir(parents=True)
    page = real_content / "index.html"
    page.write_text("<html></html>", encoding="utf-8")

    phase_outputs = _make_phase_outputs(
        objective_extraction={"project_path": str(export)},
        content_generation={"content_paths": str(page)},
    )

    cd = _find_content_dir(phase_outputs, {})
    assert cd == real_content, "explicit content_paths must win, not the glob"

    pages = _all_html_paths(phase_outputs, {})
    assert pages == [str(page)]
    assert not any("03_content_development" in p for p in pages)


def test_explicit_content_dir_key_still_wins(tmp_path: Path):
    """An explicit content_dir key short-circuits everything, including
    the new fallback (byte-identical legacy behaviour)."""
    from MCP.hardening.gate_input_routing import _find_content_dir

    export = _make_textbook_export(tmp_path, n_pages=2)
    explicit = tmp_path / "explicit_dir"
    explicit.mkdir()

    phase_outputs = _make_phase_outputs(
        objective_extraction={"project_path": str(export)},
        packaging={"content_dir": str(explicit)},
    )
    cd = _find_content_dir(phase_outputs, {})
    assert cd == explicit


def test_disk_glob_fallback_returns_none_when_export_has_no_html(
    tmp_path: Path,
):
    """No HTML anywhere under the export → fallback resolves nothing
    (gate skips with structured reason, not a false pass)."""
    from MCP.hardening.gate_input_routing import (
        _all_html_paths,
        _find_content_dir,
    )

    export = tmp_path / "empty_export"
    (export / "03_content_development").mkdir(parents=True)  # no .html
    phase_outputs = _make_phase_outputs(
        objective_extraction={"project_path": str(export)},
        content_generation={"_completed": True},
    )
    assert _find_content_dir(phase_outputs, {}) is None
    assert _all_html_paths(phase_outputs, {}) == []


def test_find_content_dir_optional_workflow_params_default(tmp_path: Path):
    """Back-compat: _find_content_dir is still callable with a single
    positional arg (workflow_params defaults to None)."""
    from MCP.hardening.gate_input_routing import _find_content_dir

    content = tmp_path / "content"
    content.mkdir()
    (content / "i.html").write_text("<html></html>", encoding="utf-8")
    phase_outputs = _make_phase_outputs(
        packaging={"content_dir": str(content)},
    )
    # Single-arg call must still resolve (no workflow_params).
    assert _find_content_dir(phase_outputs) == content


# ---------------------------------------------------------------------- #
# kg_quality gate activation (dormant-gate fail-closed wiring)
# ---------------------------------------------------------------------- #
#
# The kg_quality_report gate (config/workflows.yaml ::
# textbook_to_course::libv2_archival; validator
# lib.validators.kg_quality.KGQualityValidator; critical / block /
# fail_closed) had NO registered builder, so default_router().build()
# returned ({}, ["__no_builder_registered__"]) and the executor stamped
# GATE_SKIPPED_MISSING_INPUTS (passed=True) — the gate never ran. These
# tests lock in the builder + the fail-closed-on-missing-graph contract.

_KG_VALIDATOR = "lib.validators.kg_quality.KGQualityValidator"


def _write_minimal_semantic_graph(path: Path) -> None:
    """Write a tiny but valid concept_graph_semantic.json the reporter +
    EdgeConsensusAggregator can both consume without raising."""
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = {
        "nodes": [
            {"id": "concept-a", "label": "Concept A", "type": "DomainConcept"},
            {"id": "concept-b", "label": "Concept B", "type": "DomainConcept"},
        ],
        "edges": [
            {
                "source": "concept-a",
                "target": "concept-b",
                "type": "related-to",
                "confidence": 0.9,
                "provenance": {"rule": "cooccurrence", "rule_version": 1},
            }
        ],
        "rule_versions": {"cooccurrence": 1},
    }
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")


def test_kg_quality_builder_is_registered():
    """default_router() must register the kg_quality builder so the gate
    actually runs (not __no_builder_registered__)."""
    r = default_router()
    assert _KG_VALIDATOR in r.builders


def test_kg_quality_builder_routes_concept_extraction_semantic_graph(
    tmp_path: Path,
):
    """concept_extraction.concept_graph_path (the SEMANTIC graph) routes to
    semantic_graph_path; course_slug / run_id / output_dir resolve from the
    libv2_archival course_dir + workflow params."""
    course_dir = tmp_path / "course"
    semantic = course_dir / "concept_graph" / "concept_graph_semantic.json"
    _write_minimal_semantic_graph(semantic)

    phase_outputs = _make_phase_outputs(
        concept_extraction={
            "concept_graph_path": str(semantic),
            "course_slug": "phys-101",
            "concept_graph_sha256": "deadbeef",
        },
        libv2_archival={
            "course_slug": "phys-101",
            "course_dir": str(course_dir),
        },
    )
    r = default_router()
    inputs, missing = r.build(
        _KG_VALIDATOR, phase_outputs, {"course_name": "phys-101", "run_id": "R"}
    )
    # No router-skip: the builder NEVER short-circuits to a structured
    # missing-list, so the validator's own fail-closed arm governs.
    assert missing == []
    assert inputs["semantic_graph_path"] == str(semantic)
    assert inputs["course_slug"] == "phys-101"
    assert inputs["run_id"] == "R"
    # output_dir is the canonical LibV2 quality/ home of the report.
    assert inputs["output_dir"] == str(course_dir / "quality")
    # asserted concept_graph.json is surfaced as the sibling (reporter
    # tolerates its absence).
    assert inputs["concept_graph_path"] == str(
        semantic.parent / "concept_graph.json"
    )


def test_kg_quality_gate_runs_and_passes_on_real_graph(tmp_path: Path):
    """Given a phase_outputs dict with a real semantic graph, the validator
    runs end-to-end and passes (no missing-graph fail-closed)."""
    from lib.validators.kg_quality import KGQualityValidator

    course_dir = tmp_path / "course"
    semantic = course_dir / "concept_graph" / "concept_graph_semantic.json"
    _write_minimal_semantic_graph(semantic)

    phase_outputs = _make_phase_outputs(
        concept_extraction={"concept_graph_path": str(semantic)},
        libv2_archival={
            "course_slug": "phys-101",
            "course_dir": str(course_dir),
        },
    )
    r = default_router()
    inputs, missing = r.build(
        _KG_VALIDATOR, phase_outputs, {"course_name": "phys-101", "run_id": "R"}
    )
    assert missing == []

    result = KGQualityValidator().validate(dict(inputs, gate_id="kg_quality_report"))
    # Graph present → gate runs; warning-only validator returns passed=True.
    assert result.passed is True
    assert result.score is not None
    # The report landed under the routed output_dir.
    assert (course_dir / "quality" / "kg_quality_report.json").exists()


def test_kg_quality_gate_fails_closed_when_graph_missing(tmp_path: Path):
    """A libv2_archival run with NO concept / semantic graph must FAIL
    CLOSED (critical block), not skip with passed=True. This is the point
    of activation: refuse to ship an empty KG to LibV2."""
    from lib.validators.kg_quality import KGQualityValidator

    course_dir = tmp_path / "course"
    # course_dir exists but carries NO concept_graph_semantic.json.
    (course_dir / "concept_graph").mkdir(parents=True)

    phase_outputs = _make_phase_outputs(
        libv2_archival={
            "course_slug": "phys-101",
            "course_dir": str(course_dir),
        },
    )
    r = default_router()
    inputs, missing = r.build(
        _KG_VALIDATOR, phase_outputs, {"course_name": "phys-101", "run_id": "R"}
    )
    # Builder NEVER routes a fabricated path for a truly-absent graph, so
    # semantic_graph_path is absent. It returns an EMPTY missing-list so
    # the gate is NOT marked GATE_SKIPPED_MISSING_INPUTS — the validator
    # adjudicates the fail-closed verdict itself.
    assert missing == []
    assert "semantic_graph_path" not in inputs

    result = KGQualityValidator().validate(dict(inputs, gate_id="kg_quality_report"))
    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "KG_QUALITY_PEDAGOGY_GRAPH_MISSING" and i.severity == "critical"
        for i in result.issues
    )


def test_kg_quality_builder_skips_when_no_course_dir():
    """No course_dir + no graph anywhere → output_dir / graph paths can't
    resolve. The validator then fails closed on the missing context (this
    is NOT a router-skip; the builder still returns an empty missing-list
    and lets the validator block)."""
    from lib.validators.kg_quality import KGQualityValidator

    r = default_router()
    inputs, missing = r.build(
        _KG_VALIDATOR, {}, {"course_name": "phys-101", "run_id": "R"}
    )
    assert missing == []
    assert "semantic_graph_path" not in inputs
    assert "output_dir" not in inputs

    result = KGQualityValidator().validate(dict(inputs, gate_id="kg_quality_report"))
    assert result.passed is False
    assert result.action == "block"


def test_kg_quality_does_not_overwrite_canonical_consensus_sibling(
    tmp_path: Path,
):
    """The validator re-runs EdgeConsensusAggregator to attenuate the
    consistency axis. EdgeConsensusAggregator.build() is deterministic, so
    the re-run is idempotent and must NOT clobber the authoring-time
    canonical sibling (concept_graph/edge_consensus_report.json). The
    validator writes its own copy under output_dir (quality/); the
    canonical sibling stays byte-stable."""
    from lib.validators.kg_quality import KGQualityValidator

    course_dir = tmp_path / "course"
    semantic = course_dir / "concept_graph" / "concept_graph_semantic.json"
    _write_minimal_semantic_graph(semantic)

    # Pre-existing canonical consensus sibling authored at
    # concept_extraction time (sentinel content we assert stays put).
    canonical_sibling = semantic.parent / "edge_consensus_report.json"
    sentinel = {"summary": {"contradiction_rate": 0.0}, "_sentinel": "do-not-touch"}
    canonical_sibling.write_text(json.dumps(sentinel), encoding="utf-8")
    sentinel_bytes = canonical_sibling.read_bytes()

    phase_outputs = _make_phase_outputs(
        concept_extraction={"concept_graph_path": str(semantic)},
        libv2_archival={
            "course_slug": "phys-101",
            "course_dir": str(course_dir),
        },
    )
    r = default_router()
    inputs, _ = r.build(
        _KG_VALIDATOR, phase_outputs, {"course_name": "phys-101", "run_id": "R"}
    )
    KGQualityValidator().validate(dict(inputs, gate_id="kg_quality_report"))

    # The canonical sibling is untouched — no second DIVERGENT report.
    assert canonical_sibling.read_bytes() == sentinel_bytes
