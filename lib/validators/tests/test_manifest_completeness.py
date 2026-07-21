"""W3 — ManifestCompletenessValidator RESOLUTION tests (T7-T13).

Synthetic manifest + chunk fixtures only — no live LLM, no NLI, no model load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from lib.validators.manifest_completeness import ManifestCompletenessValidator


class _Capture:
    """Minimal decision-capture stub recording log_decision calls."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, *, decision_type, decision, rationale, **_):
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
            }
        )


def _manifest_line(**overrides) -> Dict[str, Any]:
    line = {
        "schema_version": "1.0",
        "block_id": "week_04#concept_0",
        "block_type": "concept",
        "page_id": "week_04",
        "sequence": 0,
        "source_chunk_ids": ["demo_chunk_00012"],
        "source_refs": [{"sourceId": "semantik:demo#b-12", "role": "primary"}],
        "span_granularity": "chunk",
        "template_id": "concept.definition.v1",
        "synthesis": {
            "tiers": [
                {
                    "tier": "outline",
                    "provider": "local",
                    "model": "qwen2.5:7b",
                    "timestamp": "2026-06-13T00:00:00Z",
                    "decision_capture_id": "EVT_0000000000000001",
                    "purpose": "draft",
                }
            ]
        },
        "concept_tags": ["rdf-graph"],
    }
    line.update(overrides)
    return line


def _write_manifest(tmp_path: Path, lines: List[Dict[str, Any]]) -> Path:
    cd_dir = tmp_path / "03_content_development"
    cd_dir.mkdir(parents=True, exist_ok=True)
    # Mark content as emitted so the anti-silent-degradation guard sees blocks.
    (cd_dir / "week_04.html").write_text("<html></html>", encoding="utf-8")
    path = cd_dir / "block_synthesis_manifest.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    return path


def _run(manifest_path, **inputs) -> Any:
    base = {"manifest_path": str(manifest_path) if manifest_path else None}
    base.update(inputs)
    return ManifestCompletenessValidator().validate(base)


# --------------------------------------------------------------------------
# T7 — happy path
# --------------------------------------------------------------------------
def test_t7_happy_path(tmp_path):
    path = _write_manifest(tmp_path, [_manifest_line()])
    result = _run(path, valid_source_ids={"demo_chunk_00012"})
    assert result.passed is True
    assert result.critical_count == 0
    assert result.score == 1.0


# --------------------------------------------------------------------------
# T8 — ungrounded substantive block (empty source_chunk_ids)
# --------------------------------------------------------------------------
def test_t8_ungrounded_substantive_block(tmp_path):
    path = _write_manifest(
        tmp_path, [_manifest_line(source_chunk_ids=[])]
    )
    result = _run(path, valid_source_ids={"demo_chunk_00012"})
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "MANIFEST_UNGROUNDED" in codes
    assert result.action == "regenerate"


# --------------------------------------------------------------------------
# T9 — unresolved id
# --------------------------------------------------------------------------
def test_t9_unresolved_id(tmp_path):
    path = _write_manifest(
        tmp_path, [_manifest_line(source_chunk_ids=["demo_chunk_99999"])]
    )
    result = _run(path, valid_source_ids={"demo_chunk_00012"})
    assert result.passed is False
    unresolved = [
        i for i in result.issues if i.code == "MANIFEST_SOURCE_ID_UNRESOLVED"
    ]
    assert unresolved
    assert "demo_chunk_99999" in unresolved[0].message


# --------------------------------------------------------------------------
# T10 — anti-silent-degradation
# --------------------------------------------------------------------------
def test_t10_manifest_file_missing(tmp_path):
    # Content dir populated but manifest absent.
    cd_dir = tmp_path / "03_content_development"
    cd_dir.mkdir(parents=True)
    (cd_dir / "week_04.html").write_text("<html></html>", encoding="utf-8")
    missing = cd_dir / "block_synthesis_manifest.jsonl"
    result = _run(missing, valid_source_ids={"demo_chunk_00012"})
    assert result.passed is False
    assert any(i.code == "MANIFEST_FILE_MISSING" for i in result.issues)


def test_t10_resolution_universe_empty(tmp_path):
    path = _write_manifest(tmp_path, [_manifest_line()])
    # No valid_source_ids seeded + no dart_chunks_manifest_path → empty universe
    # while the manifest declares source_chunk_ids → vacuous-pass trap.
    result = _run(path)
    assert result.passed is False
    assert any(
        i.code == "MANIFEST_RESOLUTION_UNIVERSE_EMPTY" for i in result.issues
    )


# --------------------------------------------------------------------------
# T11 — graceful pass: no manifest + no blocks
# --------------------------------------------------------------------------
def test_t11_graceful_pass_no_manifest_no_blocks(tmp_path):
    # Neither a manifest nor a populated content dir exists.
    result = _run(None)
    assert result.passed is True
    assert result.issues == []


# --------------------------------------------------------------------------
# T12 — concept_tags warning vs critical
# --------------------------------------------------------------------------
def test_t12_concept_tags_warning_default(tmp_path):
    path = _write_manifest(tmp_path, [_manifest_line(concept_tags=[])])
    result = _run(
        path, valid_source_ids={"demo_chunk_00012"}, require_concept_tags=False
    )
    assert result.passed is True  # warning-only day-1
    warn = [i for i in result.issues if i.code == "MANIFEST_NO_CONCEPT_TAGS"]
    assert warn and warn[0].severity == "warning"


def test_t12_concept_tags_critical_when_required(tmp_path):
    path = _write_manifest(tmp_path, [_manifest_line(concept_tags=[])])
    result = _run(
        path, valid_source_ids={"demo_chunk_00012"}, require_concept_tags=True
    )
    assert result.passed is False
    crit = [
        i
        for i in result.issues
        if i.code == "MANIFEST_NO_CONCEPT_TAGS" and i.severity == "critical"
    ]
    assert crit


# --------------------------------------------------------------------------
# T13 — decision-capture fires with dynamic-signal rationale
# --------------------------------------------------------------------------
def test_t13_decision_capture_fires(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            _manifest_line(),  # complete
            _manifest_line(block_id="week_04#concept_1", source_chunk_ids=[]),
            _manifest_line(
                block_id="week_04#concept_2",
                source_chunk_ids=["demo_chunk_99999"],
            ),
        ],
    )
    capture = _Capture()
    result = _run(
        path,
        valid_source_ids={"demo_chunk_00012"},
        decision_capture=capture,
    )
    assert result.passed is False
    assert len(capture.events) == 1
    event = capture.events[0]
    assert event["decision_type"] == "manifest_completeness_check"
    assert len(event["rationale"]) >= 20
    # Dynamic signals interpolated: total_blocks + ungrounded + unresolved.
    rationale = event["rationale"]
    assert "3 manifest block" in rationale  # total_blocks
    assert "1 ungrounded" in rationale
    assert "1 unresolved" in rationale


# --------------------------------------------------------------------------
# Extra — scaffolding block exempt from non-empty assertions
# --------------------------------------------------------------------------
def test_scaffolding_block_exempt(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            _manifest_line(
                block_id="week_04#chrome_0",
                block_type="chrome",
                source_chunk_ids=[],
                concept_tags=[],
                source_refs=[],
            )
        ],
    )
    result = _run(path, valid_source_ids={"demo_chunk_00012"})
    assert result.passed is True
    assert result.critical_count == 0


# --------------------------------------------------------------------------
# Extra — resolution universe loaded from a real chunks.jsonl sidecar
# --------------------------------------------------------------------------
def test_resolution_universe_from_chunks_jsonl(tmp_path):
    path = _write_manifest(tmp_path, [_manifest_line()])
    chunks_dir = tmp_path / "semantik_chunks"
    chunks_dir.mkdir()
    (chunks_dir / "manifest.json").write_text(
        json.dumps({"chunkset_kind": "semantik"}), encoding="utf-8"
    )
    (chunks_dir / "chunks.jsonl").write_text(
        json.dumps({"id": "demo_chunk_00012", "source": {}}) + "\n",
        encoding="utf-8",
    )
    result = _run(
        path, dart_chunks_manifest_path=str(chunks_dir / "manifest.json")
    )
    assert result.passed is True
    assert result.critical_count == 0
