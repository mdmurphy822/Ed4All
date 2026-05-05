"""H3 Wave W6b — DecisionCapture wiring regression tests.

Per the H3 plan (`plans/h3-validator-capture-wiring-2026-05.md` §3 W6b)
every manifest / packaging / accessibility validator in this wave's
scope MUST emit exactly one `decision_type`-matching event per
`validate()` call. The tests below assert:

1. Capture is threaded via `inputs["decision_capture"]` (Pattern A —
   matches `MCP/hardening/validation_gates.py` Worker S0.5 injection).
2. Exactly one event fires per call (corpus-wide validators —
   block-level cardinality is W1's territory).
3. The emitted `decision_type` equals the canonical enum value.
4. Rationale interpolates dynamic signals (issue counts / file paths
   / score) — proxied here by asserting `metrics` is non-empty + has
   the expected keys.

Validators covered (8 files / 9 classes):
- `lib.validators.imscc.IMSCCValidator` → `imscc_structure_check`
- `lib.validators.imscc.IMSCCParseValidator` → `imscc_parse_check`
- `lib.validators.oscqr.OSCQRValidator` → `oscqr_score_check`
- `lib.validators.chunkset_manifest.ChunksetManifestValidator` → `chunkset_manifest_check`
- `lib.validators.libv2_manifest.LibV2ManifestValidator` → `libv2_manifest_check`
- `lib.validators.libv2_model.LibV2ModelValidator` → `libv2_model_check`
- `lib.validators.libv2_packet_integrity.PacketIntegrityValidator` → `libv2_packet_integrity_check`
- `lib.validators.concept_graph.ConceptGraphValidator` → `concept_graph_check`

The DART `WCAGValidator.wcag_compliance_check` test lives under
`DART/pdf_converter/tests/test_wcag_capture_wiring.py` (per H3 plan
§4.1 Correction 1.2 — that directory is created by this wave).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.chunkset_manifest import ChunksetManifestValidator  # noqa: E402
from lib.validators.concept_graph import ConceptGraphValidator  # noqa: E402
from lib.validators.imscc import IMSCCParseValidator, IMSCCValidator  # noqa: E402
from lib.validators.libv2_manifest import LibV2ManifestValidator  # noqa: E402
from lib.validators.libv2_model import LibV2ModelValidator  # noqa: E402
from lib.validators.libv2_packet_integrity import PacketIntegrityValidator  # noqa: E402
from lib.validators.oscqr import OSCQRValidator  # noqa: E402


class _MockCapture:
    """Minimal DecisionCapture stub — records every log_decision call.

    Mirrors `lib/validators/tests/test_kg_quality_validator.py::_MockCapture`.
    """

    def __init__(self) -> None:
        self.calls: List[dict] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


# ---------------------------------------------------------------------------
# IMSCCValidator → imscc_structure_check
# ---------------------------------------------------------------------------


def test_imscc_structure_emits_on_missing_path():
    capture = _MockCapture()
    IMSCCValidator().validate({
        "imscc_path": "/nonexistent/never-here.imscc",
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "imscc_structure_check"
    assert call["decision"].startswith("failed:")
    assert call["metrics"]["exists"] is False
    assert "issue_count" in call["metrics"]


def test_imscc_structure_emits_on_zip_path(tmp_path: Path):
    capture = _MockCapture()
    zip_path = tmp_path / "package.imscc"
    zip_path.write_bytes(b"PK\x03\x04stub")
    IMSCCValidator().validate({
        "imscc_path": str(zip_path),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "imscc_structure_check"
    assert call["metrics"]["is_zip"] is True


def test_imscc_structure_emits_on_extracted_dir(tmp_path: Path):
    capture = _MockCapture()
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "imsmanifest.xml").write_text(
        '<?xml version="1.0"?>\n'
        '<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1">\n'
        '  <organizations><organization identifier="o1"/></organizations>\n'
        '  <resources><resource identifier="r1" type="webcontent" href="x.html"/></resources>\n'
        '</manifest>\n',
        encoding="utf-8",
    )
    IMSCCValidator().validate({
        "imscc_path": str(extracted),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "imscc_structure_check"
    assert "manifest_path_str" in call["metrics"]


# ---------------------------------------------------------------------------
# IMSCCParseValidator → imscc_parse_check
# ---------------------------------------------------------------------------


def test_imscc_parse_emits_on_missing_path():
    capture = _MockCapture()
    IMSCCParseValidator().validate({
        "imscc_path": "/nonexistent/missing.imscc",
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision_type"] == "imscc_parse_check"
    assert capture.calls[0]["metrics"]["exists"] is False


def test_imscc_parse_emits_on_bad_zip(tmp_path: Path):
    capture = _MockCapture()
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"not a real zip archive")
    IMSCCParseValidator().validate({
        "imscc_path": str(bad_zip),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "imscc_parse_check"
    assert call["metrics"]["suffix"] == ".zip"


# ---------------------------------------------------------------------------
# OSCQRValidator → oscqr_score_check
# ---------------------------------------------------------------------------


def test_oscqr_emits_decision_capture(tmp_path: Path):
    capture = _MockCapture()
    course_dir = tmp_path / "course"
    course_dir.mkdir()
    OSCQRValidator().validate({
        "course_path": str(course_dir),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "oscqr_score_check"
    metrics = call["metrics"]
    assert "score" in metrics
    assert "items_checkable" in metrics
    assert "items_failed" in metrics
    assert "critical_failures" in metrics


# ---------------------------------------------------------------------------
# ChunksetManifestValidator → chunkset_manifest_check
# ---------------------------------------------------------------------------


def test_chunkset_manifest_emits_on_missing_input():
    capture = _MockCapture()
    ChunksetManifestValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "chunkset_manifest_check"
    assert call["decision"] == "failed:CHUNKSET_MANIFEST_MISSING_INPUT"


def test_chunkset_manifest_emits_on_missing_file(tmp_path: Path):
    capture = _MockCapture()
    ChunksetManifestValidator().validate({
        "chunkset_manifest_path": str(tmp_path / "nope.json"),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"] == "failed:CHUNKSET_MANIFEST_NOT_FOUND"


def test_chunkset_manifest_emits_on_bad_json(tmp_path: Path):
    capture = _MockCapture()
    bad = tmp_path / "manifest.json"
    bad.write_text("not json", encoding="utf-8")
    ChunksetManifestValidator().validate({
        "chunkset_manifest_path": str(bad),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"] == "failed:CHUNKSET_MANIFEST_INVALID_JSON"


# ---------------------------------------------------------------------------
# LibV2ManifestValidator → libv2_manifest_check
# ---------------------------------------------------------------------------


def test_libv2_manifest_emits_on_missing_input():
    capture = _MockCapture()
    LibV2ManifestValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "libv2_manifest_check"
    assert call["decision"] == "failed:MISSING_MANIFEST_PATH"


def test_libv2_manifest_emits_on_missing_file(tmp_path: Path):
    capture = _MockCapture()
    LibV2ManifestValidator().validate({
        "manifest_path": str(tmp_path / "missing.json"),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"] == "failed:MANIFEST_NOT_FOUND"


def test_libv2_manifest_emits_on_invalid_json(tmp_path: Path):
    capture = _MockCapture()
    bad = tmp_path / "manifest.json"
    bad.write_text("{not json", encoding="utf-8")
    LibV2ManifestValidator().validate({
        "manifest_path": str(bad),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"] == "failed:INVALID_JSON"


# ---------------------------------------------------------------------------
# LibV2ModelValidator → libv2_model_check
# ---------------------------------------------------------------------------


def test_libv2_model_emits_on_missing_input():
    capture = _MockCapture()
    LibV2ModelValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "libv2_model_check"
    assert call["decision"] == "failed:MISSING_MODEL_CARD_PATH"


def test_libv2_model_emits_on_missing_file(tmp_path: Path):
    capture = _MockCapture()
    LibV2ModelValidator().validate({
        "model_card_path": str(tmp_path / "no.json"),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"] == "failed:MODEL_CARD_NOT_FOUND"


def test_libv2_model_emits_on_invalid_json(tmp_path: Path):
    capture = _MockCapture()
    bad = tmp_path / "model_card.json"
    bad.write_text("not-json", encoding="utf-8")
    LibV2ModelValidator().validate({
        "model_card_path": str(bad),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"] == "failed:INVALID_JSON"


# ---------------------------------------------------------------------------
# PacketIntegrityValidator → libv2_packet_integrity_check
# ---------------------------------------------------------------------------


def test_libv2_packet_integrity_emits_on_missing_inputs():
    capture = _MockCapture()
    PacketIntegrityValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "libv2_packet_integrity_check"
    assert call["decision"] == "failed:MISSING_ARCHIVE_INPUTS"


def test_libv2_packet_integrity_emits_on_archive_root(tmp_path: Path):
    capture = _MockCapture()
    PacketIntegrityValidator().validate({
        "course_dir": str(tmp_path / "missing-archive"),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "libv2_packet_integrity_check"
    assert "rules_run" in call["metrics"]


# ---------------------------------------------------------------------------
# ConceptGraphValidator → concept_graph_check
# ---------------------------------------------------------------------------


def test_concept_graph_emits_on_missing_input():
    capture = _MockCapture()
    ConceptGraphValidator().validate({"decision_capture": capture})
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "concept_graph_check"
    assert call["decision"] == "failed:CONCEPT_GRAPH_MISSING_INPUT"


def test_concept_graph_emits_on_missing_file(tmp_path: Path):
    capture = _MockCapture()
    ConceptGraphValidator().validate({
        "concept_graph_path": str(tmp_path / "nope.json"),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    assert capture.calls[0]["decision"] == "failed:CONCEPT_GRAPH_NOT_FOUND"


def test_concept_graph_emits_on_valid_graph(tmp_path: Path):
    capture = _MockCapture()
    graph_path = tmp_path / "concept_graph_semantic.json"
    graph_path.write_text(
        '{"nodes": [{"id": "n1", "class": "DomainConcept"}], '
        '"edges": []}',
        encoding="utf-8",
    )
    ConceptGraphValidator().validate({
        "concept_graph_path": str(graph_path),
        "decision_capture": capture,
    })
    assert len(capture.calls) == 1
    call = capture.calls[0]
    assert call["decision_type"] == "concept_graph_check"
    assert "node_count" in call["metrics"]
    assert call["metrics"]["node_count"] == 1


# ---------------------------------------------------------------------------
# Cross-cutting: capture-None must not raise (back-compat preservation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "validator,inputs",
    [
        (IMSCCValidator(), {"imscc_path": "/missing"}),
        (IMSCCParseValidator(), {"imscc_path": "/missing"}),
        (ChunksetManifestValidator(), {}),
        (LibV2ManifestValidator(), {}),
        (LibV2ModelValidator(), {}),
        (PacketIntegrityValidator(), {}),
        (ConceptGraphValidator(), {}),
    ],
)
def test_capture_none_does_not_raise(validator: Any, inputs: dict):
    """Without a capture key, validate() must execute its normal path
    and never raise on the helper's `if capture is None: return` guard.
    """
    result = validator.validate(inputs)
    # Result shape varies — only assert the call returned (no exception).
    assert result is not None
