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
from typing import Any, Dict

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
