"""Wave 29 gate input router coverage tests (Defect 2).

The 5-persona SIM_RUN_01 run surfaced four gates with runtime issues:

* ``libv2_manifest`` → skipped: missing inputs: manifest_path
* ``assessment_objective_alignment`` → skipped: missing inputs: chunks_path
* ``dart_markers`` → skipped: missing inputs: html_path
* ``assessment_quality`` → CRASH on ``json.loads`` of empty/absent file

Wave 29 closes the coverage gaps:

* ``libv2_manifest`` derives ``manifest_path`` from ``course_dir``.
* ``assessment_objective_alignment`` falls back to the LibV2-archived
  ``course_dir/corpus/chunks.jsonl``.
* ``dart_markers`` picks up batch outputs from ``output_paths`` and
  surfaces ``html_paths[]`` alongside a representative ``html_path``.
* ``assessment_quality`` checks existence + non-empty before handing
  the path off, so a missing / truncated file yields a structured
  skip rather than a JSON-decode crash.
"""

from __future__ import annotations

from pathlib import Path

from MCP.hardening.gate_input_routing import (
    _build_assessment_objective_alignment,
    _build_assessment_quality,
    _build_dart_markers,
    _build_libv2_manifest,
    default_router,
)

# --------------------------------------------------------------------- #
# libv2_manifest — derive from course_dir
# --------------------------------------------------------------------- #


def test_libv2_manifest_explicit_path(tmp_path: Path):
    """Explicit manifest_path in phase outputs short-circuits the
    course_dir derivation."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    outputs = {"libv2_archival": {"manifest_path": str(manifest)}}
    inputs, missing = _build_libv2_manifest(outputs, {})
    assert missing == []
    assert inputs["manifest_path"] == str(manifest)


def test_libv2_manifest_derives_from_course_dir(tmp_path: Path):
    """When ``manifest_path`` isn't surfaced but ``course_dir`` is, the
    builder derives ``course_dir/manifest.json`` if it exists."""
    course_dir = tmp_path / "MY_COURSE"
    course_dir.mkdir()
    (course_dir / "manifest.json").write_text('{"course_id": "X"}', encoding="utf-8")

    outputs = {"libv2_archival": {"course_dir": str(course_dir)}}
    inputs, missing = _build_libv2_manifest(outputs, {})
    assert missing == []
    assert inputs["manifest_path"] == str(course_dir / "manifest.json")
    assert inputs["course_dir"] == str(course_dir)


def test_libv2_manifest_skipped_when_no_signals():
    """No manifest_path, no course_dir → structured skip."""
    inputs, missing = _build_libv2_manifest({}, {})
    assert missing == ["manifest_path"]


# --------------------------------------------------------------------- #
# assessment_objective_alignment — chunks fallback
# --------------------------------------------------------------------- #


def test_assessment_objective_alignment_explicit_chunks(tmp_path: Path):
    """Explicit chunks_path short-circuits the LibV2 fallback."""
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    assessments = tmp_path / "assessments.json"
    assessments.write_text("{}", encoding="utf-8")

    outputs = {
        "trainforge_assessment": {
            "output_path": str(assessments),
            "chunks_path": str(chunks),
        },
    }
    inputs, missing = _build_assessment_objective_alignment(outputs, {})
    assert missing == []
    assert inputs["assessments_path"] == str(assessments)
    assert inputs["chunks_path"] == str(chunks)


def test_assessment_objective_alignment_falls_back_to_libv2_archive(tmp_path: Path):
    """When chunks_path isn't surfaced and the assessments path has no
    sibling corpus/chunks.jsonl, the builder pulls from the archived
    ``course_dir/corpus/chunks.jsonl``."""
    # Assessments in a lonely dir with no corpus sibling.
    tf_dir = tmp_path / "trainforge_run"
    tf_dir.mkdir()
    assessments = tf_dir / "assessments.json"
    assessments.write_text("{}", encoding="utf-8")

    # LibV2-archived course dir with the chunks.
    archive_dir = tmp_path / "LibV2" / "courses" / "archived_course"
    (archive_dir / "corpus").mkdir(parents=True)
    chunks = archive_dir / "corpus" / "chunks.jsonl"
    chunks.write_text('{"chunk_id": "c1"}\n', encoding="utf-8")

    outputs = {
        "trainforge_assessment": {"output_path": str(assessments)},
        "libv2_archival": {"course_dir": str(archive_dir)},
    }
    inputs, missing = _build_assessment_objective_alignment(outputs, {})
    assert missing == []
    assert inputs["chunks_path"] == str(chunks)


def test_assessment_objective_alignment_skipped_without_chunks(tmp_path: Path):
    """Assessments found but chunks nowhere → structured skip."""
    tf_dir = tmp_path / "tf"
    tf_dir.mkdir()
    assessments = tf_dir / "assessments.json"
    assessments.write_text("{}", encoding="utf-8")

    outputs = {"trainforge_assessment": {"output_path": str(assessments)}}
    inputs, missing = _build_assessment_objective_alignment(outputs, {})
    assert missing == ["chunks_path"]
    assert inputs["assessments_path"] == str(assessments)


def test_assessment_objective_alignment_surfaces_synthesized_objectives_path(
    tmp_path: Path,
):
    """W5.E: builder routes course_planning.synthesized_objectives_path
    into the validator inputs so the validator can union the
    objectives' id set into the chunks-side resolution surface."""
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    assessments = tmp_path / "assessments.json"
    assessments.write_text("{}", encoding="utf-8")
    synthesized = tmp_path / "synthesized_objectives.json"
    synthesized.write_text(
        '{"terminal_objectives": [{"id": "TO-01"}], "chapter_objectives": []}',
        encoding="utf-8",
    )

    outputs = {
        "trainforge_assessment": {
            "output_path": str(assessments),
            "chunks_path": str(chunks),
        },
        "course_planning": {
            "synthesized_objectives_path": str(synthesized),
        },
    }
    inputs, missing = _build_assessment_objective_alignment(outputs, {})
    assert missing == []
    assert inputs["assessments_path"] == str(assessments)
    assert inputs["chunks_path"] == str(chunks)
    assert inputs["synthesized_objectives_path"] == str(synthesized)


def test_assessment_objective_alignment_omits_synthesized_when_unset(
    tmp_path: Path,
):
    """W5.E back-compat: when the course_planning phase didn't emit
    synthesized_objectives_path (e.g. ``rag_training`` legacy
    workflow), the input is omitted — validator falls back to the
    chunks-only resolution surface byte-identically."""
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    assessments = tmp_path / "assessments.json"
    assessments.write_text("{}", encoding="utf-8")

    outputs = {
        "trainforge_assessment": {
            "output_path": str(assessments),
            "chunks_path": str(chunks),
        },
    }
    inputs, missing = _build_assessment_objective_alignment(outputs, {})
    assert missing == []
    assert "synthesized_objectives_path" not in inputs


# --------------------------------------------------------------------- #
# dart_markers — batch-aware html_paths
# --------------------------------------------------------------------- #


def test_dart_markers_single_html(tmp_path: Path):
    """Single-file DART output still returns a representative html_path."""
    html = tmp_path / "out.html"
    html.write_text("<html></html>", encoding="utf-8")

    outputs = {"dart_conversion": {"output_path": str(html)}}
    inputs, missing = _build_dart_markers(outputs, {})
    assert missing == []
    assert inputs["html_path"] == str(html)
    assert inputs["html_paths"] == [str(html)]


def test_dart_markers_batch_html(tmp_path: Path):
    """Comma-joined ``output_paths`` surfaces as a list under
    ``html_paths``."""
    a = tmp_path / "chapter1.html"
    b = tmp_path / "chapter2.html"
    c = tmp_path / "chapter3.html"
    for p in (a, b, c):
        p.write_text("<html></html>", encoding="utf-8")

    outputs = {
        "dart_conversion": {"output_paths": f"{a},{b},{c}"},
    }
    inputs, missing = _build_dart_markers(outputs, {})
    assert missing == []
    assert inputs["html_path"] == str(a)
    assert inputs["html_paths"] == [str(a), str(b), str(c)]


def test_dart_markers_skipped_on_no_html():
    """No DART outputs anywhere → structured skip, not a crash."""
    inputs, missing = _build_dart_markers({}, {})
    assert missing == ["html_path"]


# --------------------------------------------------------------------- #
# assessment_quality — missing / empty file handling
# --------------------------------------------------------------------- #


def test_assessment_quality_valid_nonempty(tmp_path: Path):
    """Well-formed non-empty path short-circuits to the happy path."""
    p = tmp_path / "assessments.json"
    p.write_text('{"questions": []}', encoding="utf-8")

    outputs = {"trainforge_assessment": {"output_path": str(p)}}
    inputs, missing = _build_assessment_quality(outputs, {})
    assert missing == []
    assert inputs["assessment_path"] == str(p)


def test_assessment_quality_missing_path_returns_skip():
    """No candidate path at all → structured skip."""
    inputs, missing = _build_assessment_quality({}, {})
    assert missing == ["ASSESSMENTS_FILE_MISSING"]


def test_assessment_quality_nonexistent_file_returns_skip(tmp_path: Path):
    """Path surfaced but file doesn't exist → skip (no crash)."""
    fake = tmp_path / "never_created.json"
    outputs = {"trainforge_assessment": {"output_path": str(fake)}}
    inputs, missing = _build_assessment_quality(outputs, {})
    assert missing == ["ASSESSMENTS_FILE_MISSING"]


def test_assessment_quality_empty_file_returns_skip(tmp_path: Path):
    """Path exists but the file is empty → skip rather than
    ``json.JSONDecodeError``.

    This is the exact crash Defect 6 surfaced: pre-Wave-29 the
    validator was handed an empty path and crashed on
    ``json.loads``. Under Wave 29 the builder intercepts and emits
    a structured skip reason.
    """
    empty = tmp_path / "empty_assessments.json"
    empty.write_text("", encoding="utf-8")

    outputs = {"trainforge_assessment": {"output_path": str(empty)}}
    inputs, missing = _build_assessment_quality(outputs, {})
    assert missing == ["ASSESSMENTS_FILE_MISSING"]


# --------------------------------------------------------------------- #
# Registry integrity — all four gates reachable through default_router
# --------------------------------------------------------------------- #


def test_default_router_covers_all_defect2_gates():
    """The default router must include builders for every gate Defect 2
    called out."""
    r = default_router()
    must_have = {
        "lib.validators.libv2_manifest.LibV2ManifestValidator",
        "lib.validators.assessment_objective_alignment.AssessmentObjectiveAlignmentValidator",
        "lib.validators.dart_markers.DartMarkersValidator",
        "lib.validators.assessment.AssessmentQualityValidator",
    }
    missing = must_have - set(r.builders.keys())
    assert not missing, f"Missing builders: {missing}"


# --------------------------------------------------------------------- #
# IB6 / IB7 keystone block-quality gates — latent no-builder skip.
#
# These validators are wired at inter_tier_validation +
# post_rewrite_validation (and the assessment seams) in
# config/workflows.yaml but shipped with NO builder registered, so the
# executor returned ``__no_builder_registered__`` and every one of them
# SKIPPED silently on a real run. The IB6 keystone rubric, anatomy
# slot-presence, interaction-feedback, QA-checklist, Bloom type-range,
# cognitive-load, retrieval-presence, instructional-depth, Bloom
# structural-enforcement, and assessment-retrieval-grounding gates never
# fired. These tests pin the registration so the regression can't recur.
# --------------------------------------------------------------------- #

# Every block-quality / IB6 / IB7 gate validator that MUST resolve a
# builder (i.e. NOT fall through to ``__no_builder_registered__``).
_IB6_IB7_BLOCK_QUALITY_VALIDATORS = {
    "lib.validators.content.BlockCognitiveLoadValidator",
    "lib.validators.anatomy_slot_presence.AnatomySlotPresenceValidator",
    "lib.validators.interaction_feedback.InteractionFeedbackValidator",
    "lib.validators.block_quality_rubric.BlockQualityRubricValidator",
    "lib.validators.qa_checklist.QaChecklistValidator",
    "lib.validators.retrieval_presence.RetrievalPresenceValidator",
    "lib.validators.bloom_type_range.BloomTypeRangeValidator",
    "lib.validators.instructional_depth.InstructionalDepthValidator",
    "lib.validators.bloom.structural_enforcement.BloomStructuralEnforcementValidator",
    "lib.validators.assessment_retrieval_grounding.AssessmentRetrievalGroundingValidator",
}


def test_default_router_covers_ib6_ib7_block_quality_gates():
    """The default router must include builders for every IB6/IB7
    block-quality gate. Pre-fix these all skipped via
    ``__no_builder_registered__`` because no builder was wired."""
    r = default_router()
    missing = _IB6_IB7_BLOCK_QUALITY_VALIDATORS - set(r.builders.keys())
    assert not missing, f"Missing builders for block-quality gates: {missing}"


def test_ib6_ib7_gates_resolve_a_builder_not_no_builder():
    """Each IB6/IB7 gate must resolve a builder through ``router.build``,
    NOT the ``__no_builder_registered__`` fallthrough that made the gate
    skip silently on every real post_rewrite run."""
    r = default_router()
    for validator_path in _IB6_IB7_BLOCK_QUALITY_VALIDATORS:
        # No phase outputs / params → the builder runs and reports a
        # structured missing-blocks skip, NOT the no-builder sentinel.
        _inputs, missing = r.build(validator_path, {}, {})
        assert "__no_builder_registered__" not in missing, (
            f"{validator_path} fell through to the no-builder fallback "
            f"(gate would skip silently); missing={missing}"
        )


def _gate_validators_for_phase(phase_name: str) -> set:
    """Collect every validation-gate validator dotted path declared on a
    workflow phase across config/workflows.yaml. Returns the union over
    all workflows that define the phase."""
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((repo_root / "config" / "workflows.yaml").read_text())
    found: set = set()
    for wf in (cfg.get("workflows") or {}).values():
        for phase in wf.get("phases") or []:
            if phase.get("name") != phase_name:
                continue
            for gate in phase.get("validation_gates") or []:
                vp = gate.get("validator")
                if isinstance(vp, str) and vp:
                    found.add(vp)
    return found


def test_every_post_rewrite_and_inter_tier_gate_has_a_builder():
    """Anti-recurrence guard: every validator wired at the
    ``post_rewrite_validation`` + ``inter_tier_validation`` phases in
    config/workflows.yaml must have a registered gate-input builder.

    This is the test that would have caught the IB6/IB7 keystone gates
    skipping on a real run. Derives the gate set from the YAML so a
    newly-wired gate without a builder fails here instead of silently
    skipping in production."""
    r = default_router()
    registered = set(r.builders.keys())
    wired = (
        _gate_validators_for_phase("post_rewrite_validation")
        | _gate_validators_for_phase("inter_tier_validation")
    )
    assert wired, "expected to discover gate validators from workflows.yaml"
    missing = wired - registered
    assert not missing, (
        "post_rewrite/inter_tier gate validators with NO registered "
        f"builder (these skip silently in production): {sorted(missing)}"
    )


# --------------------------------------------------------------------- #
# Render-dependent gate placement — anti-recurrence guard.
#
# udl_coverage / anatomy_slot_presence / interaction_feedback /
# block_quality_rubric / qa_checklist measure rewrite/render-tier fields
# (UDL n_representations from rendered HTML tags, the IB1 anatomy
# interaction/feedback/transition slots, the Bloom chip, rendered
# interaction/feedback HTML) that do NOT exist on outline-tier dict
# blocks. Wired at inter_tier_validation they blanket-fail every block as
# a false positive (course-a-calib calibration, 2026-06-24). They must
# live at post_rewrite_validation ONLY. These tests pin that placement so
# a re-wire at inter_tier can't silently reintroduce the FP storm.
# --------------------------------------------------------------------- #

_RENDER_DEPENDENT_GATE_VALIDATORS = {
    "lib.validators.udl_coverage.UdlCoverageValidator",
    "lib.validators.anatomy_slot_presence.AnatomySlotPresenceValidator",
    "lib.validators.interaction_feedback.InteractionFeedbackValidator",
    "lib.validators.block_quality_rubric.BlockQualityRubricValidator",
    "lib.validators.qa_checklist.QaChecklistValidator",
}


def test_render_dependent_gates_not_wired_at_inter_tier():
    """Render-dependent IB4.5/IB6 gates must NOT be wired at the
    outline-tier inter_tier_validation phase (they blanket-FP there)."""
    inter_tier = _gate_validators_for_phase("inter_tier_validation")
    leaked = _RENDER_DEPENDENT_GATE_VALIDATORS & inter_tier
    assert not leaked, (
        "render-dependent gates wired at inter_tier_validation (they "
        f"blanket-fail outline-tier dict blocks as false positives): {sorted(leaked)}"
    )


def test_render_dependent_gates_remain_at_post_rewrite():
    """The same gates MUST stay wired at post_rewrite_validation, where
    the rendered fields exist and the signal is real (no signal lost)."""
    post_rewrite = _gate_validators_for_phase("post_rewrite_validation")
    missing = _RENDER_DEPENDENT_GATE_VALIDATORS - post_rewrite
    assert not missing, (
        "render-dependent gates dropped from post_rewrite_validation "
        f"(real signal lost): {sorted(missing)}"
    )
