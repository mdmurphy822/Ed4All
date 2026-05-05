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
        # manifest_path / staging_dir / valid_*_ids leakage.
        assert set(inputs.keys()) <= {"blocks", "objectives_path"}


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
    "lib.validators.courseforge_outline_shacl.CourseforgeOutlineShaclValidator",
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
