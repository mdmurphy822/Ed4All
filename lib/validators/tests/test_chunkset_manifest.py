"""Phase 7b Subtask 13 — tests for ChunksetManifestValidator.

Mirrors the test surface of ``test_concept_graph.py`` (the closest
sibling validator). Covers:

* happy paths for both ``chunkset_kind="dart"`` and ``chunkset_kind="imscc"``
* file / JSON / shape critical-error paths (block action)
* schema-violation paths (missing required field, missing conditional
  source-SHA field for the declared kind, additionalProperties)
* SHA-256 mismatch (load-bearing tamper-detection signal)
* sibling chunks.jsonl missing
* chunks_count cross-check (matching + mismatched + absent)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.chunkset_manifest import (  # noqa: E402
    ChunksetManifestValidator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FAKE_DART_HTML_SHA = "a" * 64
_FAKE_IMSCC_SHA = "b" * 64


def _make_chunks_jsonl(tmp_path: Path, n_chunks: int = 3) -> Path:
    """Write a minimal chunks.jsonl with N chunks; return the path."""
    p = tmp_path / "chunks.jsonl"
    lines: List[str] = []
    for i in range(n_chunks):
        lines.append(json.dumps({"chunk_id": f"c{i}", "text": f"chunk {i}"}))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _make_manifest_dict(
    *,
    chunks_sha: str,
    chunkset_kind: str = "dart",
    chunker_version: str = "0.1.0",
    chunks_count: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    drop: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a baseline-valid manifest payload, then optionally apply
    drop-field / extra-field perturbations for the negative paths."""
    payload: Dict[str, Any] = {
        "chunks_sha256": chunks_sha,
        "chunker_version": chunker_version,
        "chunkset_kind": chunkset_kind,
    }
    if chunkset_kind == "dart":
        payload["source_dart_html_sha256"] = _FAKE_DART_HTML_SHA
    elif chunkset_kind == "imscc":
        payload["source_imscc_sha256"] = _FAKE_IMSCC_SHA
    if chunks_count is not None:
        payload["chunks_count"] = chunks_count
    if extra:
        payload.update(extra)
    if drop:
        for key in drop:
            payload.pop(key, None)
    return payload


def _write_manifest(tmp_path: Path, payload: Any) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_dart_chunkset(tmp_path: Path) -> None:
    """Well-formed DART manifest + matching chunks.jsonl + matching SHA → pass."""
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=5)
    manifest_payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        chunks_count=5,
    )
    manifest_path = _write_manifest(tmp_path, manifest_payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is True, [i.code for i in result.issues]
    assert result.action is None
    assert result.critical_count == 0
    assert result.score == 1.0


def test_happy_path_imscc_chunkset(tmp_path: Path) -> None:
    """Same shape with chunkset_kind=imscc + source_imscc_sha256 → pass."""
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=2)
    manifest_payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="imscc",
        chunks_count=2,
    )
    manifest_path = _write_manifest(tmp_path, manifest_payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is True, [i.code for i in result.issues]
    assert result.action is None
    assert result.critical_count == 0


def test_missing_input_returns_block(tmp_path: Path) -> None:
    """No chunkset_manifest_path key → critical MISSING_INPUT, action=block."""
    result = ChunksetManifestValidator().validate({})

    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CHUNKSET_MANIFEST_MISSING_INPUT" and i.severity == "critical"
        for i in result.issues
    )


def test_manifest_file_not_found(tmp_path: Path) -> None:
    """Path that doesn't exist → critical NOT_FOUND, action=block."""
    nonexistent = tmp_path / "missing" / "manifest.json"
    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(nonexistent)},
    )

    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CHUNKSET_MANIFEST_NOT_FOUND" and i.severity == "critical"
        for i in result.issues
    )


def test_malformed_json(tmp_path: Path) -> None:
    """Manifest exists but isn't valid JSON → critical INVALID_JSON, block."""
    p = tmp_path / "manifest.json"
    p.write_text("{not valid json", encoding="utf-8")

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(p)},
    )

    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CHUNKSET_MANIFEST_INVALID_JSON" and i.severity == "critical"
        for i in result.issues
    )


def test_schema_violation_missing_chunks_sha256(tmp_path: Path) -> None:
    """Missing chunks_sha256 → critical SCHEMA_VIOLATION, block."""
    chunks = _make_chunks_jsonl(tmp_path)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        drop=["chunks_sha256"],
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CHUNKSET_MANIFEST_SCHEMA_VIOLATION" and i.severity == "critical"
        for i in result.issues
    )


def test_schema_violation_dart_kind_missing_source_sha(tmp_path: Path) -> None:
    """chunkset_kind=dart but no source_dart_html_sha256 → SCHEMA_VIOLATION."""
    chunks = _make_chunks_jsonl(tmp_path)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        drop=["source_dart_html_sha256"],
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is False
    assert any(
        i.code == "CHUNKSET_MANIFEST_SCHEMA_VIOLATION" and i.severity == "critical"
        for i in result.issues
    )


def test_schema_violation_imscc_kind_missing_source_sha(tmp_path: Path) -> None:
    """chunkset_kind=imscc but no source_imscc_sha256 → SCHEMA_VIOLATION."""
    chunks = _make_chunks_jsonl(tmp_path)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="imscc",
        drop=["source_imscc_sha256"],
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is False
    assert any(
        i.code == "CHUNKSET_MANIFEST_SCHEMA_VIOLATION" and i.severity == "critical"
        for i in result.issues
    )


def test_schema_violation_invalid_kind_enum(tmp_path: Path) -> None:
    """chunkset_kind not in {dart, imscc} → SCHEMA_VIOLATION."""
    chunks = _make_chunks_jsonl(tmp_path)
    payload = {
        "chunks_sha256": _sha256_of(chunks),
        "chunker_version": "0.1.0",
        "chunkset_kind": "frobnicated",  # invalid enum value
    }
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is False
    assert any(
        i.code == "CHUNKSET_MANIFEST_SCHEMA_VIOLATION" and i.severity == "critical"
        for i in result.issues
    )


def test_hash_mismatch_against_disk(tmp_path: Path) -> None:
    """Manifest claims one SHA but chunks.jsonl content has a different one."""
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=3)
    bogus_sha = "0" * 64  # 64-char hex but won't match real content
    payload = _make_manifest_dict(
        chunks_sha=bogus_sha,
        chunkset_kind="dart",
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CHUNKSET_HASH_MISMATCH" and i.severity == "critical"
        for i in result.issues
    )


def test_sibling_chunks_jsonl_missing(tmp_path: Path) -> None:
    """Manifest exists but sibling chunks.jsonl absent → CHUNKSET_CHUNKS_NOT_FOUND."""
    payload = _make_manifest_dict(
        chunks_sha="c" * 64,  # arbitrary valid-looking SHA
        chunkset_kind="dart",
    )
    manifest_path = _write_manifest(tmp_path, payload)
    # Note: NO call to _make_chunks_jsonl — chunks.jsonl is intentionally absent.

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CHUNKSET_CHUNKS_NOT_FOUND" and i.severity == "critical"
        for i in result.issues
    )


def test_chunks_count_mismatch(tmp_path: Path) -> None:
    """Manifest claims chunks_count=99 but chunks.jsonl has 3 lines → COUNT_MISMATCH."""
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=3)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        chunks_count=99,  # divergent from on-disk
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CHUNKSET_COUNT_MISMATCH" and i.severity == "critical"
        for i in result.issues
    )


def test_chunks_count_optional_passes_when_absent(tmp_path: Path) -> None:
    """When chunks_count is absent, the validator does NOT cross-check."""
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=4)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        # chunks_count intentionally not set
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is True, [i.code for i in result.issues]
    assert result.action is None


def test_additional_properties_rejected(tmp_path: Path) -> None:
    """Schema enforces additionalProperties: false → unknown fields fail."""
    chunks = _make_chunks_jsonl(tmp_path)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        extra={"unrecognized_field": "should not be here"},
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is False
    assert any(
        i.code == "CHUNKSET_MANIFEST_SCHEMA_VIOLATION" and i.severity == "critical"
        for i in result.issues
    )


def test_root_not_an_object(tmp_path: Path) -> None:
    """JSON parses but root is a list, not an object → SCHEMA_VIOLATION, block."""
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(p)},
    )

    assert result.passed is False
    assert result.action == "block"
    assert any(
        i.code == "CHUNKSET_MANIFEST_SCHEMA_VIOLATION" and i.severity == "critical"
        for i in result.issues
    )


def test_validator_metadata() -> None:
    """Sanity-check the public class metadata."""
    v = ChunksetManifestValidator()
    assert v.name == "chunkset_manifest"
    assert v.version == "0.1.0"


def _source_coverage(
    *, consumed: int, emitted: int, dropped: int = 0,
    drop_reasons: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Build a schema-valid source_coverage sub-object."""
    reasons = drop_reasons or ({} if dropped == 0 else {"boilerplate": dropped})
    cov = (emitted / consumed) if consumed else 0.0
    return {
        "consumed_count": consumed,
        "emitted_count": emitted,
        "dropped_count": dropped,
        "drop_reasons": reasons,
        "coverage_pct": cov,
    }


class _RecordingCapture:
    """Minimal DecisionCapture stand-in that records log_decision calls."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


# ---------------------------------------------------------------------------
# W1.2 import-coverage gate
# ---------------------------------------------------------------------------


def test_coverage_gate_off_is_byte_identical_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset flags → the coverage gate adds no issue even on a thin,
    low-coverage chunkset. GateResult identical to the no-source_coverage
    happy path."""
    monkeypatch.delenv("ED4ALL_CHUNK_COVERAGE_FLOOR", raising=False)
    monkeypatch.delenv("ED4ALL_MIN_CHUNKS", raising=False)
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=1)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        chunks_count=1,
        extra={"source_coverage": _source_coverage(
            consumed=100, emitted=1, dropped=99,
        )},
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is True, [i.code for i in result.issues]
    assert not any(
        i.code in {"CHUNKSET_COVERAGE_LOW", "CHUNKSET_TOO_THIN"}
        for i in result.issues
    )


def test_coverage_low_fires_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """coverage_pct below ED4ALL_CHUNK_COVERAGE_FLOOR → warning, still passes."""
    monkeypatch.setenv("ED4ALL_CHUNK_COVERAGE_FLOOR", "0.80")
    monkeypatch.delenv("ED4ALL_MIN_CHUNKS", raising=False)
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=5)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        extra={"source_coverage": _source_coverage(
            consumed=100, emitted=50, dropped=50,
        )},  # 0.5 coverage < 0.80 floor
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    warn = [i for i in result.issues if i.code == "CHUNKSET_COVERAGE_LOW"]
    assert len(warn) == 1
    assert warn[0].severity == "warning"
    # Warning-day-1: does NOT block.
    assert result.passed is True
    assert result.action is None


def test_coverage_ok_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """coverage_pct at/above the floor → no CHUNKSET_COVERAGE_LOW."""
    monkeypatch.setenv("ED4ALL_CHUNK_COVERAGE_FLOOR", "0.80")
    monkeypatch.delenv("ED4ALL_MIN_CHUNKS", raising=False)
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=5)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        extra={"source_coverage": _source_coverage(
            consumed=100, emitted=90, dropped=10,
        )},  # 0.9 >= 0.80
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert not any(i.code == "CHUNKSET_COVERAGE_LOW" for i in result.issues)
    assert result.passed is True


def test_too_thin_fires_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """emitted_count below ED4ALL_MIN_CHUNKS → CHUNKSET_TOO_THIN warning."""
    monkeypatch.setenv("ED4ALL_MIN_CHUNKS", "10")
    monkeypatch.delenv("ED4ALL_CHUNK_COVERAGE_FLOOR", raising=False)
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=3)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        extra={"source_coverage": _source_coverage(
            consumed=3, emitted=3,
        )},  # 3 < 10
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    thin = [i for i in result.issues if i.code == "CHUNKSET_TOO_THIN"]
    assert len(thin) == 1
    assert thin[0].severity == "warning"
    assert result.passed is True


def test_coverage_gate_no_op_when_source_coverage_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flags ON but pre-W3.H manifest lacks source_coverage → graceful no-op."""
    monkeypatch.setenv("ED4ALL_CHUNK_COVERAGE_FLOOR", "0.99")
    monkeypatch.setenv("ED4ALL_MIN_CHUNKS", "1000")
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=2)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
    )  # no source_coverage
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is True
    assert not any(
        i.code in {"CHUNKSET_COVERAGE_LOW", "CHUNKSET_TOO_THIN"}
        for i in result.issues
    )


@pytest.mark.parametrize("raw", ["", "garbage", "-0.5", "1.5", "0", "0.0"])
def test_coverage_floor_parse_with_fallback(
    monkeypatch: pytest.MonkeyPatch, raw: str,
) -> None:
    """Garbage / out-of-range / non-positive ED4ALL_CHUNK_COVERAGE_FLOOR → 0.0 (off)."""
    from lib.validators.chunkset_manifest import resolve_coverage_floor

    monkeypatch.setenv("ED4ALL_CHUNK_COVERAGE_FLOOR", raw)
    assert resolve_coverage_floor() == 0.0


@pytest.mark.parametrize("raw", ["", "garbage", "-3", "0", "0.5"])
def test_min_chunks_parse_with_fallback(
    monkeypatch: pytest.MonkeyPatch, raw: str,
) -> None:
    """Garbage / non-positive / non-int ED4ALL_MIN_CHUNKS → 0 (off)."""
    from lib.validators.chunkset_manifest import resolve_min_chunks

    monkeypatch.setenv("ED4ALL_MIN_CHUNKS", raw)
    assert resolve_min_chunks() == 0


def test_coverage_floor_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from lib.validators.chunkset_manifest import resolve_coverage_floor

    monkeypatch.setenv("ED4ALL_CHUNK_COVERAGE_FLOOR", "0.75")
    assert resolve_coverage_floor() == 0.75


def test_min_chunks_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from lib.validators.chunkset_manifest import resolve_min_chunks

    monkeypatch.setenv("ED4ALL_MIN_CHUNKS", "25")
    assert resolve_min_chunks() == 25


def test_coverage_gate_emits_decision_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chunkset_manifest_check decision fires and carries coverage metrics."""
    monkeypatch.setenv("ED4ALL_CHUNK_COVERAGE_FLOOR", "0.80")
    monkeypatch.setenv("ED4ALL_MIN_CHUNKS", "10")
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=2)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        extra={"source_coverage": _source_coverage(
            consumed=100, emitted=2, dropped=98,
        )},
    )
    manifest_path = _write_manifest(tmp_path, payload)
    capture = _RecordingCapture()

    result = ChunksetManifestValidator().validate(
        {
            "chunkset_manifest_path": str(manifest_path),
            "decision_capture": capture,
        },
    )

    # Both warning arms fired.
    assert {i.code for i in result.issues} >= {
        "CHUNKSET_COVERAGE_LOW", "CHUNKSET_TOO_THIN",
    }
    # A chunkset_manifest_check decision was emitted with coverage metrics.
    checks = [
        c for c in capture.calls
        if c.get("decision_type") == "chunkset_manifest_check"
    ]
    assert checks, "expected a chunkset_manifest_check decision"
    metrics = checks[-1]["metrics"]
    assert metrics["coverage_floor"] == 0.80
    assert metrics["min_chunks_floor"] == 10
    assert metrics["coverage_low"] is True
    assert metrics["too_thin"] is True
    assert metrics["coverage_pct_observed"] == 0.02
    assert metrics["emitted_count_observed"] == 2


def test_chunker_version_with_local_suffix(tmp_path: Path) -> None:
    """`0.0.0+missing` (the schema-valid sentinel emitted by
    ``_resolve_chunker_version`` when the package isn't importable)
    should pass schema validation."""
    chunks = _make_chunks_jsonl(tmp_path, n_chunks=1)
    payload = _make_manifest_dict(
        chunks_sha=_sha256_of(chunks),
        chunkset_kind="dart",
        chunker_version="0.0.0+missing",
    )
    manifest_path = _write_manifest(tmp_path, payload)

    result = ChunksetManifestValidator().validate(
        {"chunkset_manifest_path": str(manifest_path)},
    )

    assert result.passed is True, [i.code for i in result.issues]
