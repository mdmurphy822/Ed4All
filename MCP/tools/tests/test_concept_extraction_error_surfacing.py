"""Silent-degradation tests for C6 + M2 + M3 in pipeline_tools.py.

C6: ``_extract_textbook_structure`` previously caught per-file
exceptions into an ``extraction_errors[]`` list and emitted
``success: True`` regardless. A run with N of M files failing would
produce a partial textbook structure that downstream gates couldn't
catch. Fix: phase fails when ``extraction_errors_count > 0`` and
surfaces the count in the phase output envelope.

M2: ``_run_concept_extraction``'s inline-projection fallback dropped
sections silently (malformed sidecar, non-list ``sections``, non-dict
section, missing ``section_id``). The Phase-8 ST-6 fix restored the
canonical ``id`` key but left the warnings logger-only. Fix: each
drop emits a structured ``concept_projection_drop`` decision event
and surfaces ``projection_drops_count`` on the phase output.

M3: ``_run_post_rewrite_validation``'s ``_entry_to_block`` self-
documents that it "drops a CURIE / content_type / objective_ref /
source_id silently" when malformed. Fix: each drop emits a structured
``metadata_field_drop`` decision event and surfaces
``metadata_drops_count`` on the phase output.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools as _pt  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


# ---------------------------------------------------------------------- #
# Recording capture (mirrors test_pipeline_tools_rewrite_remediation
# pattern). Replaces lib.decision_capture.DecisionCapture so the test
# can inspect every emitted decision event without writing to disk.
# ---------------------------------------------------------------------- #


class _RecordingCapture:
    """DecisionCapture-compatible spy. Records every log_decision call."""

    instances: List["_RecordingCapture"] = []

    def __init__(self, **_kw: Any) -> None:
        self.events: List[Dict[str, Any]] = []
        _RecordingCapture.instances.append(self)

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


@pytest.fixture(autouse=True)
def _reset_recording_capture():
    """Reset the capture roster before each test so events from one
    test don't leak into another."""
    _RecordingCapture.instances = []
    yield
    _RecordingCapture.instances = []


@pytest.fixture
def patched_capture(monkeypatch):
    """Patch DecisionCapture in lib.decision_capture so every helper
    that imports it sees the spy."""
    import lib.decision_capture as _dc_mod
    monkeypatch.setattr(_dc_mod, "DecisionCapture", _RecordingCapture)
    return _RecordingCapture


# =====================================================================
# C6: extraction errors surface + phase fails
# =====================================================================


def _seed_staging(tmp_path: Path, file_count: int = 3) -> Path:
    staging = tmp_path / "staging"
    staging.mkdir()
    for i in range(file_count):
        (staging / f"chapter_{i:02d}.html").write_text(
            f"<html><body><h1>Chapter {i}</h1></body></html>",
            encoding="utf-8",
        )
    return staging


def test_concept_extraction_phase_fails_on_per_file_errors(
    tmp_path, monkeypatch,
):
    """C6 — patch SemanticStructureExtractor.extract to raise on N of M
    files; assert the phase output carries
    ``extraction_errors_count > 0`` AND ``success=False``."""
    monkeypatch.setattr(_pt, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "Courseforge" / "exports").mkdir(parents=True)

    staging = _seed_staging(tmp_path, file_count=3)

    # Patch the extractor so two of three files raise.
    from lib.semantic_structure_extractor import semantic_structure_extractor as _sse_mod
    orig_extract = _sse_mod.SemanticStructureExtractor.extract
    call_count = {"n": 0}

    def boom_on_first_two(self, content, source, *, format="html"):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError(f"synthetic extractor fail #{call_count['n']}")
        return orig_extract(self, content, source, format=format)

    monkeypatch.setattr(
        _sse_mod.SemanticStructureExtractor, "extract", boom_on_first_two,
    )

    registry = _build_tool_registry()
    tool = registry["extract_textbook_structure"]
    result = asyncio.run(tool(
        course_name="C6_TEST",
        staging_dir=str(staging),
        duration_weeks=8,
        duration_weeks_explicit=True,
    ))
    payload = json.loads(result)

    assert payload["success"] is False, (
        f"C6 regression: phase should fail when extraction_errors are "
        f"present; got payload={payload!r}"
    )
    assert payload["extraction_errors_count"] == 2, payload
    assert payload["extraction_error_count"] == 2, payload  # back-compat alias
    assert payload["source_file_count"] == 3, payload
    # First-3 error summaries are surfaced.
    assert "extraction_error_summaries" in payload, payload
    assert len(payload["extraction_error_summaries"]) == 2, payload
    # Each summary carries source_file + error.
    for summary in payload["extraction_error_summaries"]:
        assert summary["source_file"]
        assert "synthetic extractor fail" in summary["error"]


def test_concept_extraction_phase_succeeds_when_no_errors(
    tmp_path, monkeypatch,
):
    """C6 negative case — when every file extracts cleanly the phase
    keeps emitting ``success=True`` so we don't false-fail clean runs."""
    monkeypatch.setattr(_pt, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "Courseforge" / "exports").mkdir(parents=True)
    staging = _seed_staging(tmp_path, file_count=2)

    registry = _build_tool_registry()
    tool = registry["extract_textbook_structure"]
    result = asyncio.run(tool(
        course_name="C6_CLEAN",
        staging_dir=str(staging),
        duration_weeks=8,
        duration_weeks_explicit=True,
    ))
    payload = json.loads(result)
    assert payload["success"] is True
    assert payload["extraction_errors_count"] == 0


# =====================================================================
# M2: concept_projection_drop decision capture + count surface
# =====================================================================


def _write_synthesized_with_bad_sections(path: Path) -> None:
    """Emit a sidecar with one good + three drop-classes of sections:
       - one valid section (drives a real chunk)
       - one section missing ``section_id``  → missing_section_id
       - one section that's not a dict (a string) → non_dict_section
    """
    doc = {
        "campus_code": "drop-fixture",
        "sections": [
            {
                "section_id": "good_one",
                "section_type": "content",
                "section_title": "Good Section",
                "data": {"paragraphs": ["A real paragraph of test text."]},
            },
            {
                # Missing section_id -> drop class missing_section_id
                "section_type": "content",
                "section_title": "No id here",
                "data": {"paragraphs": ["paragraph"]},
            },
            "this is not a dict at all",  # non_dict_section drop class
        ],
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _write_synthesized_non_list_sections(path: Path) -> None:
    """Emit a sidecar where ``sections`` is a dict (not a list)."""
    doc = {"campus_code": "weird", "sections": {"not": "a list"}}
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_projection_drops_emit_decision_capture(
    tmp_path, monkeypatch, patched_capture,
):
    """M2 — fixture with malformed sections should produce
    ``concept_projection_drop`` events AND
    ``projection_drops_count > 0`` in the phase output."""
    fake_root = tmp_path / "root"
    fake_root.mkdir()
    monkeypatch.setattr(_pt, "_PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        _pt,
        "COURSEFORGE_INPUTS",
        fake_root / "Courseforge" / "inputs" / "textbooks",
    )
    (fake_root / "Courseforge" / "inputs" / "textbooks").mkdir(parents=True)

    staging = tmp_path / "staging"
    staging.mkdir()
    _write_synthesized_with_bad_sections(staging / "good_synthesized.json")
    # Second sidecar with non-list ``sections`` to drive non_list_sections drop.
    _write_synthesized_non_list_sections(
        staging / "weird_synthesized.json",
    )
    # Third sidecar that's malformed JSON to drive malformed_sidecar drop.
    (staging / "broken_synthesized.json").write_text(
        "{not valid json}", encoding="utf-8",
    )

    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(tool(
        project_id="",
        course_name="M2_TEST",
        staging_dir=str(staging),
    ))
    payload = json.loads(result)
    assert payload["success"] is True
    # Three drops: missing_section_id + non_dict_section +
    # non_list_sections + malformed_sidecar = 4.
    assert payload["projection_drops_count"] >= 3, payload
    drop_records = payload["projection_drops"]
    reasons = {r["reason"] for r in drop_records}
    assert "missing_section_id" in reasons, reasons
    assert "non_dict_section" in reasons, reasons
    assert "non_list_sections" in reasons or "malformed_sidecar" in reasons, reasons

    # Decision capture spy must have received concept_projection_drop events.
    all_events: List[Dict[str, Any]] = []
    for inst in patched_capture.instances:
        all_events.extend(inst.events)
    drop_events = [
        e for e in all_events
        if e.get("decision_type") == "concept_projection_drop"
    ]
    assert len(drop_events) >= 3, (
        f"Expected at least 3 concept_projection_drop events; "
        f"got {len(drop_events)}: {drop_events!r}"
    )
    # Each event carries reason in ml_features.
    for ev in drop_events:
        assert "reason" in (ev.get("ml_features") or {}), ev
        # Rationale is the standard 20+ char minimum.
        assert len(ev.get("rationale", "")) >= 20, ev


# =====================================================================
# M3: metadata_field_drop decision capture + count surface
# =====================================================================


def _write_blocks_jsonl_with_malformed_fields(path: Path) -> None:
    """Emit a blocks_final.jsonl with one valid block plus malformed
    CURIE / content_type_label / objective_ids / source_ids fields so
    every M3 drop class fires."""
    entries = [
        {
            "block_id": "week_01_content_01#explanation_alpha_0",
            "block_type": "explanation",
            "page_id": "week_01_content_01",
            "sequence": 0,
            "content": "<p>" + ("Body prose. " * 10) + "</p>",
            # Mixed valid + malformed LO refs
            "objective_ids": ["TO-01", "not-a-ref", "co-99-extra-bits"],
            # Mixed valid + empty source_ids
            "source_ids": ["dart:ch1#sec1", "", None],
            # Out-of-enum content_type_label
            "content_type_label": "wrongo",
            # Mixed valid CURIE-ish + malformed CURIE in key_terms.
            # ":no_prefix_value" trips the M3 CURIE-shape gate (looks
            # like a CURIE because of the colon, but fails the
            # ^[A-Za-z][A-Za-z0-9_-]*: prefix-anchor).
            "key_terms": ["plain_term", "good:ns/value", ":no_prefix_value"],
        },
        {
            # A clean block — no drops here
            "block_id": "week_01_content_01#explanation_beta_1",
            "block_type": "explanation",
            "page_id": "week_01_content_01",
            "sequence": 1,
            "content": "<p>" + ("Clean prose. " * 10) + "</p>",
            "objective_ids": ["TO-01"],
            "source_ids": ["dart:ch1#sec2"],
            "content_type_label": "definition",
            "key_terms": ["plain_term"],
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e))
            fh.write("\n")


def test_metadata_field_drops_emit_decision_capture(
    tmp_path, monkeypatch, patched_capture,
):
    """M3 — feed a blocks_final.jsonl with malformed CURIE /
    content_type / objective_ref / source_id fields; assert each drop
    class fires a ``metadata_field_drop`` event AND
    ``metadata_drops_count > 0`` in the phase output."""
    blocks_path = tmp_path / "blocks_final.jsonl"
    _write_blocks_jsonl_with_malformed_fields(blocks_path)

    # Re-point PROJECT_ROOT so any objectives lookup resolves under tmp_path.
    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)

    result = asyncio.run(_pt._run_post_rewrite_validation(
        blocks_final_path=str(blocks_path),
        project_id="M3_TEST",
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload
    assert payload["block_count"] == 2, payload
    # At least four drops: 2 bad LO refs + 2 bad source_ids
    # + 1 content_type drop + 1 CURIE drop = 6.
    assert payload["metadata_drops_count"] >= 4, payload

    drop_records = payload["metadata_drops"]
    field_names = {r["field_name"] for r in drop_records}
    # Every M3 silent-drop class must be visible.
    assert "objective_ref" in field_names, field_names
    assert "source_id" in field_names, field_names
    assert "content_type" in field_names, field_names
    assert "curie" in field_names, field_names

    # Decision-capture spy received metadata_field_drop events.
    all_events: List[Dict[str, Any]] = []
    for inst in patched_capture.instances:
        all_events.extend(inst.events)
    drop_events = [
        e for e in all_events
        if e.get("decision_type") == "metadata_field_drop"
    ]
    assert len(drop_events) >= 4, (
        f"Expected at least 4 metadata_field_drop events; "
        f"got {len(drop_events)}: {drop_events!r}"
    )
    # Every event carries field_name + reason in ml_features.
    for ev in drop_events:
        feats = ev.get("ml_features") or {}
        assert "field_name" in feats, ev
        assert "reason" in feats, ev
        # Decision-capture rationale floor (20 chars).
        assert len(ev.get("rationale", "")) >= 20, ev


def test_metadata_drops_zero_when_blocks_clean(
    tmp_path, monkeypatch, patched_capture,
):
    """M3 negative case — clean blocks emit zero drops so we don't
    regress on benign inputs."""
    blocks_path = tmp_path / "blocks_final.jsonl"
    with blocks_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "block_id": "week_01_content_01#explanation_alpha_0",
            "block_type": "explanation",
            "page_id": "week_01_content_01",
            "sequence": 0,
            "content": "<p>" + ("Clean prose. " * 10) + "</p>",
            "objective_ids": ["TO-01"],
            "source_ids": ["dart:ch1#sec1"],
            "content_type_label": "definition",
        }) + "\n")

    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)
    result = asyncio.run(_pt._run_post_rewrite_validation(
        blocks_final_path=str(blocks_path),
        project_id="M3_CLEAN",
    ))
    payload = json.loads(result)
    assert payload["success"] is True
    assert payload["metadata_drops_count"] == 0, payload
