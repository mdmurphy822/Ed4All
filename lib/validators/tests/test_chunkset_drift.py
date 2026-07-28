"""GPT Feedback v2 Wave 3 W3.D — tests for ChunksetDriftValidator.

Coverage:

* zero-drift happy path (sidecar emitted, drift_rate_* both 0)
* IMSCC_CLAIM_NOT_IN_DART warning at drift_rate_in > 0.05
* IMSCC_CLAIM_NOT_IN_DART rationale signals "high" at drift_rate_in > 0.20
* DART_EVIDENCE_DROPPED warning at drift_rate_out > 0.50
* MISSING_CHUNKSET when either chunks.jsonl is absent (fail-closed warning)
* sidecar JSON validates against schemas/library/chunkset_drift_report.schema.json
* decision-capture rationale interpolates the six load-bearing metrics
* empty-chunks edge case (no division-by-zero, passes silently)
* legacy chunks lacking ``source.source_references[]`` count as unsourced
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.chunkset_drift import (  # noqa: E402
    ChunksetDriftValidator,
    _DRIFT_RATE_IN_HIGH_THRESHOLD,
    _DRIFT_RATE_IN_WARN_THRESHOLD,
    _DRIFT_RATE_OUT_WARN_THRESHOLD,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _make_chunk(
    chunk_id: str,
    *,
    source_refs: Optional[List[str]] = None,
    drop_source: bool = False,
) -> Dict[str, Any]:
    """Build a v4-shaped chunk fixture with a configurable source refs list.

    ``drop_source=True`` produces a legacy-shape chunk where the
    ``source`` block is entirely missing — exercises the fallback arm
    that counts those as unsourced.
    """
    chunk: Dict[str, Any] = {"id": chunk_id, "schema_version": "v4"}
    if drop_source:
        return chunk
    refs = source_refs or []
    chunk["source"] = {
        "source_references": [
            {"sourceId": sid, "role": "primary", "confidence": 0.9}
            for sid in refs
        ],
    }
    return chunk


class _FakeCapture:
    """Captures decisions for substring-assertion in tests."""

    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.decisions.append(dict(kwargs))


# --------------------------------------------------------------------------- #
# Test 1 — zero-drift happy path
# --------------------------------------------------------------------------- #


def test_zero_drift_passes_and_emits_sidecar(tmp_path: Path) -> None:
    course = tmp_path / "course"
    dart_dir = course / "dart_chunks"
    imscc_dir = course / "imscc_chunks"

    # Two DART chunks; each cites itself via source_references so the
    # union covers both id-based and ref-based identity. Two IMSCC
    # chunks each cite both DART block ids → zero drift.
    dart_chunks = [
        _make_chunk("dart:00001", source_refs=["dart:00001"]),
        _make_chunk("dart:00002", source_refs=["dart:00002"]),
    ]
    imscc_chunks = [
        _make_chunk("imscc:00001", source_refs=["dart:00001", "dart:00002"]),
        _make_chunk("imscc:00002", source_refs=["dart:00001", "dart:00002"]),
    ]
    dart_path = _write_jsonl(dart_dir / "chunks.jsonl", dart_chunks)
    imscc_path = _write_jsonl(imscc_dir / "chunks.jsonl", imscc_chunks)

    capture = _FakeCapture()
    result = ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
            "decision_capture": capture,
        }
    )

    assert result.passed is True, [i.code for i in result.issues]
    assert result.action is None
    assert result.score == 1.0
    sidecar = course / "drift_report.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["drift_rate_in"] == 0.0
    assert payload["drift_rate_out"] == 0.0
    assert payload["dart_block_ids_dropped"] == []
    assert payload["imscc_chunks_unsourced"] == []
    assert payload["schema_version"] == "1.0"


# --------------------------------------------------------------------------- #
# Test 2 — drift_rate_in > 0.05 → IMSCC_CLAIM_NOT_IN_DART warning
# --------------------------------------------------------------------------- #


def test_drift_rate_in_warn_threshold_flags_imscc_claim_not_in_dart(
    tmp_path: Path,
) -> None:
    course = tmp_path / "course"
    dart_chunks = [_make_chunk(f"dart:{i:05d}", source_refs=[f"dart:{i:05d}"])
                   for i in range(10)]
    # 10 IMSCC chunks; 1 lacks source refs → drift_rate_in = 0.10 > 0.05.
    imscc_chunks = [_make_chunk(f"imscc:{i:05d}", source_refs=["dart:00000"])
                    for i in range(9)]
    imscc_chunks.append(_make_chunk("imscc:00009", source_refs=[]))

    dart_path = _write_jsonl(course / "dart_chunks" / "chunks.jsonl", dart_chunks)
    imscc_path = _write_jsonl(
        course / "imscc_chunks" / "chunks.jsonl", imscc_chunks,
    )
    result = ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
        }
    )

    codes = {i.code for i in result.issues}
    assert "IMSCC_CLAIM_NOT_IN_DART" in codes
    assert all(i.severity == "warning" for i in result.issues)
    assert result.passed is False
    assert result.action is None  # warning gate, never blocks
    # Confirm the rationale clarifies it's NOT in the elevated band.
    msgs = [i.message for i in result.issues if i.code == "IMSCC_CLAIM_NOT_IN_DART"]
    assert any("internal severity escalated" not in m for m in msgs), msgs


# --------------------------------------------------------------------------- #
# Test 3 — drift_rate_in > 0.20 → rationale signals "high" severity
# --------------------------------------------------------------------------- #


def test_drift_rate_in_high_threshold_signals_high_severity(
    tmp_path: Path,
) -> None:
    course = tmp_path / "course"
    dart_chunks = [_make_chunk(f"dart:{i:05d}", source_refs=[f"dart:{i:05d}"])
                   for i in range(10)]
    # 10 IMSCC chunks; 5 lack source refs → drift_rate_in = 0.50 > 0.20.
    imscc_chunks = [_make_chunk(f"imscc:{i:05d}", source_refs=["dart:00000"])
                    for i in range(5)]
    for i in range(5, 10):
        imscc_chunks.append(_make_chunk(f"imscc:{i:05d}", source_refs=[]))
    dart_path = _write_jsonl(course / "dart_chunks" / "chunks.jsonl", dart_chunks)
    imscc_path = _write_jsonl(
        course / "imscc_chunks" / "chunks.jsonl", imscc_chunks,
    )

    result = ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
        }
    )

    msgs = [i.message for i in result.issues if i.code == "IMSCC_CLAIM_NOT_IN_DART"]
    assert msgs, [i.code for i in result.issues]
    assert any("internal severity escalated to high" in m for m in msgs), msgs
    # Sanity-check the threshold value rendered in the message.
    assert any(f"{_DRIFT_RATE_IN_HIGH_THRESHOLD:.2f}" in m for m in msgs)


# --------------------------------------------------------------------------- #
# Test 4 — drift_rate_out > 0.50 → DART_EVIDENCE_DROPPED warning
# --------------------------------------------------------------------------- #


def test_drift_rate_out_warn_threshold_flags_dart_evidence_dropped(
    tmp_path: Path,
) -> None:
    course = tmp_path / "course"
    # 10 DART block ids in the universe (unique self-citations).
    dart_chunks = [_make_chunk(f"dart:{i:05d}", source_refs=[f"dart:{i:05d}"])
                   for i in range(10)]
    # IMSCC cites only 4/10 → drift_rate_out = 0.6 > 0.50.
    imscc_chunks = [
        _make_chunk(f"imscc:{i:05d}", source_refs=[f"dart:{i:05d}"])
        for i in range(4)
    ]
    dart_path = _write_jsonl(course / "dart_chunks" / "chunks.jsonl", dart_chunks)
    imscc_path = _write_jsonl(
        course / "imscc_chunks" / "chunks.jsonl", imscc_chunks,
    )

    result = ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
        }
    )

    codes = {i.code for i in result.issues}
    assert "DART_EVIDENCE_DROPPED" in codes
    assert all(i.severity == "warning" for i in result.issues)
    sidecar = course / "drift_report.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["drift_rate_out"] > _DRIFT_RATE_OUT_WARN_THRESHOLD
    assert len(payload["dart_block_ids_dropped"]) == 6


# --------------------------------------------------------------------------- #
# Test 5 — MISSING_CHUNKSET when a chunks.jsonl is absent
# --------------------------------------------------------------------------- #


def test_missing_chunkset_fails_closed_with_warning(tmp_path: Path) -> None:
    course = tmp_path / "course"
    # Only the IMSCC side exists.
    imscc_chunks = [_make_chunk("imscc:00001", source_refs=["dart:00001"])]
    imscc_path = _write_jsonl(
        course / "imscc_chunks" / "chunks.jsonl", imscc_chunks,
    )
    dart_path = course / "dart_chunks" / "chunks.jsonl"  # never written

    result = ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
        }
    )

    assert result.passed is False
    assert result.action == "block"
    codes = {i.code for i in result.issues}
    assert codes == {"MISSING_CHUNKSET"}
    issue = result.issues[0]
    assert issue.severity == "warning"


def test_missing_input_path_emits_missing_chunkset(tmp_path: Path) -> None:
    """Caller passes neither path → still surfaces MISSING_CHUNKSET."""
    result = ChunksetDriftValidator().validate({})
    assert result.passed is False
    assert any(i.code == "MISSING_CHUNKSET" for i in result.issues)


# --------------------------------------------------------------------------- #
# Test 6 — sidecar validates against the schema
# --------------------------------------------------------------------------- #


def test_drift_report_sidecar_matches_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator

    course = tmp_path / "course"
    dart_chunks = [
        _make_chunk("dart:00001", source_refs=["dart:00001"]),
        _make_chunk("dart:00002", source_refs=["dart:00002"]),
        _make_chunk("dart:00003", source_refs=["dart:00003"]),
    ]
    # 3 IMSCC chunks; 1 unsourced + 2 cite only one DART id → triggers
    # both drift signals so the sidecar carries non-empty arrays.
    imscc_chunks = [
        _make_chunk("imscc:00001", source_refs=["dart:00001"]),
        _make_chunk("imscc:00002", source_refs=["dart:00001"]),
        _make_chunk("imscc:00003", source_refs=[]),
    ]
    dart_path = _write_jsonl(course / "dart_chunks" / "chunks.jsonl", dart_chunks)
    imscc_path = _write_jsonl(
        course / "imscc_chunks" / "chunks.jsonl", imscc_chunks,
    )
    ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
        }
    )

    sidecar = course / "drift_report.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    schema_path = (
        _REPO_ROOT
        / "schemas"
        / "library"
        / "chunkset_drift_report.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(payload))
    assert errors == [], [e.message for e in errors]


# --------------------------------------------------------------------------- #
# Test 7 — decision-capture rationale interpolates 6 load-bearing fields
# --------------------------------------------------------------------------- #


def test_decision_capture_rationale_carries_six_metrics(tmp_path: Path) -> None:
    course = tmp_path / "course"
    dart_chunks = [_make_chunk("dart:00001", source_refs=["dart:00001"])]
    imscc_chunks = [_make_chunk("imscc:00001", source_refs=["dart:00001"])]
    dart_path = _write_jsonl(course / "dart_chunks" / "chunks.jsonl", dart_chunks)
    imscc_path = _write_jsonl(
        course / "imscc_chunks" / "chunks.jsonl", imscc_chunks,
    )

    capture = _FakeCapture()
    ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
            "decision_capture": capture,
        }
    )

    assert len(capture.decisions) == 1
    event = capture.decisions[0]
    assert event["decision_type"] == "chunkset_drift_check"
    rationale = event["rationale"]
    for substring in (
        "drift_rate_in=",
        "drift_rate_out=",
        "dropped_count=",
        "unsourced_count=",
        "total_dart_chunks=",
        "total_imscc_chunks=",
    ):
        assert substring in rationale, rationale


# --------------------------------------------------------------------------- #
# Test 8 — empty chunks files → drift_rate=0, passes silently
# --------------------------------------------------------------------------- #


def test_empty_chunks_files_passes_with_zero_drift(tmp_path: Path) -> None:
    course = tmp_path / "course"
    dart_path = _write_jsonl(course / "dart_chunks" / "chunks.jsonl", [])
    imscc_path = _write_jsonl(course / "imscc_chunks" / "chunks.jsonl", [])

    result = ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
        }
    )

    assert result.passed is True, [i.code for i in result.issues]
    assert result.action is None
    sidecar = course / "drift_report.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["drift_rate_in"] == 0.0
    assert payload["drift_rate_out"] == 0.0
    assert payload["dart_chunk_count"] == 0
    assert payload["imscc_chunk_count"] == 0


# --------------------------------------------------------------------------- #
# Test 9 — legacy chunks (no source_references) count as unsourced
# --------------------------------------------------------------------------- #


def test_legacy_chunks_without_source_references_count_as_unsourced(
    tmp_path: Path,
) -> None:
    course = tmp_path / "course"
    dart_chunks = [
        _make_chunk(f"dart:{i:05d}", source_refs=[f"dart:{i:05d}"])
        for i in range(5)
    ]
    # 5 IMSCC chunks: 4 missing the entire ``source`` block (legacy
    # shape) + 1 properly-sourced. drift_rate_in = 4/5 = 0.8 → high
    # band IMSCC_CLAIM_NOT_IN_DART.
    imscc_chunks = [
        _make_chunk(f"imscc:legacy_{i}", drop_source=True) for i in range(4)
    ]
    imscc_chunks.append(_make_chunk("imscc:modern_1", source_refs=["dart:00000"]))

    dart_path = _write_jsonl(course / "dart_chunks" / "chunks.jsonl", dart_chunks)
    imscc_path = _write_jsonl(
        course / "imscc_chunks" / "chunks.jsonl", imscc_chunks,
    )

    result = ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
        }
    )

    codes = {i.code for i in result.issues}
    assert "IMSCC_CLAIM_NOT_IN_DART" in codes
    sidecar = course / "drift_report.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["drift_rate_in"] == pytest.approx(0.8, rel=1e-6)
    assert len(payload["imscc_chunks_unsourced"]) == 4
    # Legacy chunks still carry an `id` in our fixture, so the
    # synthesized anchor fallback shouldn't fire.
    assert all(
        not anchor.startswith("<unsourced_chunk_index_")
        for anchor in payload["imscc_chunks_unsourced"]
    )


# --------------------------------------------------------------------------- #
# Test 10 — drift_rate_out denominator is the BLOCK-ID universe, not the
# source CHUNK count.
#
# REGRESSION GUARD. Every fixture above builds each evidence chunk with
# exactly ONE self-citation (`_make_chunk("dart:00001", source_refs=
# ["dart:00001"])`), so `len(block_id_universe) == len(chunks)` by
# construction and the two candidate denominators are numerically
# identical — which is precisely why the suite could not see a wrong
# denominator. Real chunks cite MANY upstream `semantik:{slug}#{block_id}`
# refs each (~31 per chunk on a live scanned textbook), so the universe
# is ~32x the chunk count and the chunk-count denominator produced
# `drift_rate_out=32.36` — a "rate" that is not a fraction and that
# violates the sidecar schema's `maximum: 1`.
#
# The fixture below deliberately decouples the two counts.
# --------------------------------------------------------------------------- #


def _fanout_fixture(course: Path) -> tuple:
    """2 evidence chunks x 9 distinct upstream refs → 20-id universe.

    Universe = 2 own ids + 18 distinct cited sourceIds = 20, against a
    source chunk count of 2. IMSCC cites 15 of the 20 → 5 dropped.

      * correct rate (dropped / universe)    = 5 / 20 = 0.25  → PASS
      * buggy rate   (dropped / chunk count) = 5 /  2 = 2.50  → would
        fire DART_EVIDENCE_DROPPED *and* emit a schema-invalid sidecar.
    """
    dart_chunks = [
        _make_chunk(
            "dart:c1",
            source_refs=[f"semantik:doc#b{i:02d}" for i in range(0, 9)],
        ),
        _make_chunk(
            "dart:c2",
            source_refs=[f"semantik:doc#b{i:02d}" for i in range(9, 18)],
        ),
    ]
    universe_ids = ["dart:c1", "dart:c2"] + [
        f"semantik:doc#b{i:02d}" for i in range(0, 18)
    ]
    assert len(universe_ids) == 20
    # One IMSCC chunk citing 15 of the 20 ids → 5 dropped.
    imscc_chunks = [_make_chunk("imscc:00001", source_refs=universe_ids[:15])]

    dart_path = _write_jsonl(course / "dart_chunks" / "chunks.jsonl", dart_chunks)
    imscc_path = _write_jsonl(
        course / "imscc_chunks" / "chunks.jsonl", imscc_chunks,
    )
    return dart_path, imscc_path


def test_drift_rate_out_denominator_is_block_id_universe_not_chunk_count(
    tmp_path: Path,
) -> None:
    course = tmp_path / "course"
    dart_path, imscc_path = _fanout_fixture(course)

    capture = _FakeCapture()
    result = ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
            "decision_capture": capture,
        }
    )

    payload = json.loads(
        (course / "drift_report.json").read_text(encoding="utf-8")
    )

    # The two counts genuinely differ — without this the test proves nothing.
    assert payload["dart_chunk_count"] == 2
    assert len(payload["dart_block_ids_dropped"]) == 5

    # Correct denominator: 5 / 20.
    assert payload["drift_rate_out"] == pytest.approx(0.25, rel=1e-9)
    # Explicitly pin that the buggy chunk-count denominator is NOT used.
    assert payload["drift_rate_out"] != pytest.approx(
        5 / payload["dart_chunk_count"], rel=1e-9,
    )
    # A rate is a fraction: the schema bounds it at 1.
    assert 0.0 <= payload["drift_rate_out"] <= 1.0

    # 0.25 is under the 0.50 warn threshold → the gate must NOT fire.
    codes = {i.code for i in result.issues}
    assert "DART_EVIDENCE_DROPPED" not in codes, codes
    assert result.passed is True, codes

    # The denominator is carried on the capture so an operator can
    # recompute the rate without re-reading both chunksets.
    rationale = capture.decisions[0]["rationale"]
    assert "total_source_block_ids=20" in rationale, rationale
    assert "total_dart_chunks=2" in rationale, rationale


def test_high_fanout_sidecar_still_satisfies_schema_rate_bounds(
    tmp_path: Path,
) -> None:
    """The buggy denominator emitted drift_rate_out > 1 → schema-invalid.

    Pins that a high-fan-out corpus (the shape that produced the live
    32.36) round-trips the ``maximum: 1`` bound on both rates.
    """
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator

    course = tmp_path / "course"
    dart_chunks = [
        _make_chunk(
            f"dart:c{c}",
            source_refs=[f"semantik:doc#c{c}_b{i:02d}" for i in range(30)],
        )
        for c in range(5)
    ]
    # IMSCC cites nothing → every id in the universe is dropped.
    imscc_chunks = [_make_chunk("imscc:00001", source_refs=[])]
    dart_path = _write_jsonl(course / "dart_chunks" / "chunks.jsonl", dart_chunks)
    imscc_path = _write_jsonl(
        course / "imscc_chunks" / "chunks.jsonl", imscc_chunks,
    )

    ChunksetDriftValidator().validate(
        {
            "dart_chunks_path": str(dart_path),
            "imscc_chunks_path": str(imscc_path),
            "course_path": str(course),
        }
    )

    payload = json.loads(
        (course / "drift_report.json").read_text(encoding="utf-8")
    )
    # 5 chunks, 5 + 150 = 155 ids, all dropped. Buggy denominator would
    # have yielded 155 / 5 = 31.0 (the live 32.36 shape).
    assert payload["dart_chunk_count"] == 5
    assert len(payload["dart_block_ids_dropped"]) == 155
    assert payload["drift_rate_out"] == pytest.approx(1.0, rel=1e-9)

    schema_path = (
        _REPO_ROOT / "schemas" / "library" / "chunkset_drift_report.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(payload))
    assert errors == [], [e.message for e in errors]
