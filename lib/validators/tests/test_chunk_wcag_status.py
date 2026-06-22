"""IB4.2 — ChunkWcagStatusValidator unit tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.chunk_wcag_status import ChunkWcagStatusValidator  # noqa: E402


def _validate(inputs):
    return ChunkWcagStatusValidator().validate(inputs)


def _codes(result):
    return [i.code for i in result.issues]


def test_flagged_chunk_warns_and_fails_passed() -> None:
    result = _validate({"chunks": [
        {"id": "c1", "wcag_block_status": "flagged"},
        {"id": "c2", "wcag_block_status": "passed"},
    ]})
    assert "CHUNK_WCAG_FLAGGED" in _codes(result)
    assert result.passed is False  # flagged is the passed=False driver
    assert all(i.severity == "warning" for i in result.issues)


def test_figure_chunk_empty_alt_warns() -> None:
    result = _validate({"chunks": [
        {"id": "f1", "chunk_type": "figure", "figure_alt": "", "wcag_block_status": "passed"},
    ]})
    assert "CHUNK_FIGURE_NO_ALT" in _codes(result)
    # figure-alt miss is warning-only day-1 → passed stays True (no flagged).
    assert result.passed is True


def test_img_bearing_chunk_without_figure_alt_warns() -> None:
    result = _validate({"chunks": [
        {"id": "i1", "text": "<p>x</p><img src='a.png'>", "wcag_block_status": "passed"},
    ]})
    assert "CHUNK_FIGURE_NO_ALT" in _codes(result)


def test_all_passed_with_alt_is_clean() -> None:
    result = _validate({"chunks": [
        {"id": "c1", "wcag_block_status": "passed"},
        {"id": "f1", "chunk_type": "figure", "figure_alt": "A bar chart", "wcag_block_status": "passed"},
    ]})
    assert result.issues == []
    assert result.passed is True


def test_legacy_chunkset_without_fields_passes_clean() -> None:
    result = _validate({"chunks": [
        {"id": "c1", "text": "Some prose with no WCAG fields."},
        {"id": "c2", "text": "More prose."},
    ]})
    assert _codes(result) == ["WCAG_FIELDS_ABSENT"]
    assert result.passed is True


def test_no_chunkset_resolved_passes_clean() -> None:
    result = _validate({})
    assert _codes(result) == ["WCAG_FIELDS_ABSENT"]
    assert result.passed is True


def test_reads_from_jsonl_path(tmp_path) -> None:
    p = tmp_path / "chunks.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(c) for c in [
                {"id": "c1", "wcag_block_status": "flagged"},
                {"id": "c2", "wcag_block_status": "passed"},
            ]
        )
    )
    result = _validate({"dart_chunks_path": str(p)})
    assert "CHUNK_WCAG_FLAGGED" in _codes(result)
    assert result.passed is False


def test_decision_type_accepted_by_schema() -> None:
    schema_path = _REPO_ROOT / "schemas" / "events" / "decision_event.schema.json"
    schema = json.loads(schema_path.read_text())
    enum = schema["properties"]["decision_type"]["enum"]
    assert "chunk_wcag_status_check" in enum
