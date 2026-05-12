"""Wave1-I2 resume-safety regression: chunker helpers preserve existing
``chunks.jsonl`` on missing input.

Background. ``MCP/tools/pipeline_tools.py::_run_imscc_chunking`` and
``_run_dart_chunking`` used to fail-soft to an empty-bytes shell when
the input path was missing or invalid. On a workflow resume that
dropped the input path from carried-forward phase context, the prior
behaviour silently overwrote a previously-valid ``chunks.jsonl`` with
an empty file — cost ~92 chunks during the 2026-05-12 OpenStax run.

Per ``plans/dispatch-7-execution-inspection-2026-05.md`` Finding 2, the
fix is twofold:

  1. When the input path is missing AND a pre-existing ``chunks.jsonl``
     exists at the target write location, preserve the artifact
     byte-identically and surface a WARNING log + a ``preserved_existing``
     flag in the PhaseOutput envelope.
  2. When the input is missing AND there is NO pre-existing
     ``chunks.jsonl``, fail-closed (raise) instead of emitting an
     empty-bytes stub.

The three regression tests below pin both legs of the fix for the IMSCC
helper plus the DART sibling, plus the fail-closed leg shared by both
surfaces.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FIXTURE_CHUNKS = [
    {
        "id": "test_chunk_00001",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "First fixture chunk.",
        "html": "<p>First fixture chunk.</p>",
        "follows_chunk": None,
        "source": {"course_id": "TEST_RESUME", "module_id": "m1"},
        "concept_tags": [],
        "learning_outcome_refs": [],
        "difficulty": "intermediate",
        "tokens_estimate": 3,
        "word_count": 3,
    },
    {
        "id": "test_chunk_00002",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Second fixture chunk.",
        "html": "<p>Second fixture chunk.</p>",
        "follows_chunk": "test_chunk_00001",
        "source": {"course_id": "TEST_RESUME", "module_id": "m1"},
        "concept_tags": [],
        "learning_outcome_refs": [],
        "difficulty": "intermediate",
        "tokens_estimate": 3,
        "word_count": 3,
    },
    {
        "id": "test_chunk_00003",
        "schema_version": "v4",
        "chunk_type": "explanation",
        "text": "Third fixture chunk.",
        "html": "<p>Third fixture chunk.</p>",
        "follows_chunk": "test_chunk_00002",
        "source": {"course_id": "TEST_RESUME", "module_id": "m1"},
        "concept_tags": [],
        "learning_outcome_refs": [],
        "difficulty": "intermediate",
        "tokens_estimate": 3,
        "word_count": 3,
    },
]


def _write_fixture_chunks(chunks_path: Path) -> bytes:
    """Write the 3-chunk fixture to ``chunks_path`` and return the bytes
    on disk so the test can assert byte-identical preservation."""
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("wb") as fh:
        for chunk in _FIXTURE_CHUNKS:
            fh.write((json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8"))
    return chunks_path.read_bytes()


def _write_fixture_manifest(manifest_path: Path, source_sha: str) -> None:
    """Write a minimal sibling manifest.json that round-trips with the
    fixture chunks. The ``source_imscc_sha256`` / ``source_dart_html_sha256``
    field lets the preserved-envelope path surface a real upstream SHA."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "chunks_sha256": "deadbeef" * 8,
                "chunker_version": "v4",
                "chunkset_kind": "imscc",
                "source_imscc_sha256": source_sha,
                "chunks_count": len(_FIXTURE_CHUNKS),
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Test 1 — IMSCC preserve-on-missing-input
# ---------------------------------------------------------------------------


def test_imscc_chunking_preserves_existing_on_missing_input(tmp_path: Path) -> None:
    """``_run_imscc_chunking`` MUST byte-preserve an existing chunks.jsonl
    when imscc_path is None / missing.

    Setup writes a 3-chunk fixture under the threaded ``libv2_root`` at
    ``courses/test-resume/imscc_chunks/chunks.jsonl``. Calling the helper
    with ``imscc_path=`` pointing at a non-existent path MUST:

      - return ``success=True``
      - return ``preserved_existing=True``
      - leave the on-disk bytes byte-identical to the fixture
      - surface the existing file's SHA-256 in the envelope
    """
    libv2_root = tmp_path / "LibV2"
    chunks_path = (
        libv2_root / "courses" / "test-resume" / "imscc_chunks" / "chunks.jsonl"
    )
    original_bytes = _write_fixture_chunks(chunks_path)
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    _write_fixture_manifest(
        chunks_path.parent / "manifest.json",
        source_sha="a" * 64,
    )

    registry = _build_tool_registry()
    tool = registry["run_imscc_chunking"]
    result_json = asyncio.run(
        tool(
            course_name="TEST_RESUME",
            imscc_path=str(tmp_path / "does_not_exist.imscc"),
            libv2_root=str(libv2_root),
        )
    )
    payload = json.loads(result_json)

    assert payload["success"] is True
    assert payload.get("preserved_existing") is True, (
        f"Expected preserved_existing=True; got payload={payload!r}"
    )
    assert payload["imscc_chunks_sha256"] == original_sha, (
        "Preserved SHA must match the on-disk fixture bytes."
    )
    assert payload["chunks_count"] == len(_FIXTURE_CHUNKS)
    # Byte-identical preservation — the helper must NOT have rewritten
    # the file (even with identical content the write order would
    # change mtime; the bytes invariant is the load-bearing one).
    assert chunks_path.read_bytes() == original_bytes
    # The preserved envelope surfaces the manifest's source SHA when
    # it's readable.
    assert payload.get("source_imscc_sha256") == "a" * 64


# ---------------------------------------------------------------------------
# Test 2 — DART preserve-on-missing-input
# ---------------------------------------------------------------------------


def test_dart_chunking_preserves_existing_on_missing_input(tmp_path: Path) -> None:
    """``_run_dart_chunking`` MUST byte-preserve an existing chunks.jsonl
    when no staging_dir resolves OR when staging_dir has zero HTML files.

    Same invariants as the IMSCC test — chunks.jsonl bytes must survive
    a resume-call with an invalid staging_dir kwarg.
    """
    libv2_root = tmp_path / "LibV2"
    chunks_path = (
        libv2_root / "courses" / "test-resume" / "dart_chunks" / "chunks.jsonl"
    )
    original_bytes = _write_fixture_chunks(chunks_path)
    original_sha = hashlib.sha256(original_bytes).hexdigest()

    registry = _build_tool_registry()
    tool = registry["run_dart_chunking"]
    # Point staging_dir at a path that does NOT exist so the resolution
    # chain falls through to COURSEFORGE_INPUTS, which (per the helper's
    # default) may or may not exist on the test box. Either way the
    # staging-walk produces zero HTML files, which is the same
    # "missing input" condition we want to test.
    result_json = asyncio.run(
        tool(
            course_name="TEST_RESUME",
            staging_dir=str(tmp_path / "does_not_exist_staging"),
            libv2_root=str(libv2_root),
        )
    )
    payload = json.loads(result_json)

    assert payload["success"] is True
    assert payload.get("preserved_existing") is True, (
        f"Expected preserved_existing=True; got payload={payload!r}"
    )
    assert payload["dart_chunks_sha256"] == original_sha
    assert payload["chunks_count"] == len(_FIXTURE_CHUNKS)
    assert chunks_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# Test 3 — fail-closed when no existing chunks and no input
# ---------------------------------------------------------------------------


def test_imscc_chunking_fails_closed_no_existing_no_input(tmp_path: Path) -> None:
    """``_run_imscc_chunking`` MUST fail-closed (raise) when both
    imscc_path is missing AND there is no pre-existing chunks.jsonl.

    Critically, no empty-bytes chunks.jsonl must be created on disk
    after the failure — the prior fail-soft path created one
    (the bug class this fix closes).
    """
    libv2_root = tmp_path / "LibV2"
    expected_chunks_path = (
        libv2_root / "courses" / "test-resume" / "imscc_chunks" / "chunks.jsonl"
    )
    # Sanity: no pre-existing chunks file.
    assert not expected_chunks_path.exists()

    registry = _build_tool_registry()
    tool = registry["run_imscc_chunking"]

    with pytest.raises(RuntimeError, match="Wave1-I2"):
        asyncio.run(
            tool(
                course_name="TEST_RESUME",
                imscc_path=str(tmp_path / "does_not_exist.imscc"),
                libv2_root=str(libv2_root),
            )
        )

    # Most important assertion of the suite: NO chunks.jsonl was
    # created during the failed call. The prior bug-class behaviour
    # was to create an empty-bytes file before raising / on the
    # success path; that's the data-loss vector this test pins.
    assert not expected_chunks_path.exists(), (
        f"Empty chunks.jsonl created at {expected_chunks_path} after "
        "fail-closed path — the data-loss vector this fix closes."
    )
