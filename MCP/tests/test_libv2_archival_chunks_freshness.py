"""Wave 74 cleanup — LibV2 archival chunks-freshness gate.

Bug observed (2026-04-24): ``trainforge_assessment`` phase failed
(parameter mapping bug — separate fix), but ``libv2_archival`` ran
anyway and stamped a fresh archive at ``LibV2/courses/rdf-shacl-550/``.
Inside, ``chunks.jsonl`` contained 32 ``smoke_hifi_rag_chunk_*`` lines
from an April 22 prior run. Somehow a stale archive's chunks survived
into the newly-created archive at this slug.

The fix in ``MCP/tools/pipeline_tools.py`` does two things:

  1. Pre-emptively delete any pre-existing ``chunks.jsonl`` at the
     LibV2 destination *before* the copy block runs, so we never
     silently preserve prior-run chunks.
  2. After the copy block, validate that any chunks at the destination
     carry IDs from this run's ``course_code`` (pattern
     ``^{course_code_lower()}_chunk_``). If not, fail with
     ``error_code = TRAINFORGE_OUTPUT_STALE`` and refuse to write the
     manifest.

The gate must NOT break DART-only batches that intentionally skip
Trainforge — when no chunks file is present at all, archival proceeds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_tool_registry,
    _check_chunks_freshness,
    _course_chunk_id_prefix,
)


# --------------------------------------------------------------------- #
# Unit tests on _check_chunks_freshness
# --------------------------------------------------------------------- #


def test_freshness_absent_when_no_file(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    result = _check_chunks_freshness(
        chunks_path=chunks,
        course_name="RDF_SHACL_550",
        run_start_ts=time.time(),
        had_prior_chunks=False,
    )
    assert result["status"] == "absent"


def test_freshness_fresh_when_mtime_ge_run_start(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"id":"foo_chunk_00001"}\n')
    # mtime on a just-written file is >= time captured before we wrote.
    run_start = time.time() - 1.0
    result = _check_chunks_freshness(
        chunks_path=chunks,
        course_name="RDF_SHACL_550",
        run_start_ts=run_start,
        had_prior_chunks=False,
    )
    assert result["status"] == "fresh"


def test_freshness_fresh_when_id_prefix_matches(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        '{"id":"rdf_shacl_550_chunk_00001","text":"x"}\n'
        '{"id":"rdf_shacl_550_chunk_00002","text":"y"}\n'
    )
    # Push mtime to the past so the prefix check is the deciding signal.
    past = time.time() - 3600
    import os as _os
    _os.utime(chunks, (past, past))
    result = _check_chunks_freshness(
        chunks_path=chunks,
        course_name="RDF_SHACL_550",
        run_start_ts=time.time(),
        had_prior_chunks=False,
    )
    assert result["status"] == "fresh", result


def test_freshness_stale_when_prefix_mismatched_and_old(tmp_path):
    """The exact bug shape — chunks file from a prior run under the same
    slug carries unrelated chunk IDs."""
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        '{"id":"smoke_hifi_rag_chunk_00001","text":"x"}\n'
        '{"id":"smoke_hifi_rag_chunk_00002","text":"y"}\n'
    )
    # mtime in the past so we don't get a false-positive ``fresh``.
    past = time.time() - 3600
    import os as _os
    _os.utime(chunks, (past, past))
    result = _check_chunks_freshness(
        chunks_path=chunks,
        course_name="RDF_SHACL_550",
        run_start_ts=time.time(),
        had_prior_chunks=True,
    )
    assert result["status"] == "stale", result
    assert result["expected_prefix"] == "rdf_shacl_550_chunk_"
    assert "smoke_hifi_rag" in result["observed_prefixes"]


def test_freshness_absent_when_empty_file(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("")
    past = time.time() - 3600
    import os as _os
    _os.utime(chunks, (past, past))
    result = _check_chunks_freshness(
        chunks_path=chunks,
        course_name="RDF_SHACL_550",
        run_start_ts=time.time(),
        had_prior_chunks=False,
    )
    assert result["status"] == "absent"


def test_chunk_id_prefix_normalises_dashes(tmp_path):
    """When callers pass a slug-shaped name, the prefix uses underscores
    (matching what Trainforge actually writes)."""
    assert _course_chunk_id_prefix("RDF_SHACL_550") == "rdf_shacl_550_chunk_"
    assert _course_chunk_id_prefix("rdf-shacl-550") == "rdf_shacl_550_chunk_"
    assert _course_chunk_id_prefix("MAT 101") == "mat_101_chunk_"
    assert _course_chunk_id_prefix("") == ""


# --------------------------------------------------------------------- #
# End-to-end: registry _archive_to_libv2 fail-closed contract
# --------------------------------------------------------------------- #


@pytest.fixture
def isolated_archive(monkeypatch, tmp_path):
    """Redirect LibV2 root to tmp_path and return the registry archival
    coroutine. No global LibV2/courses/ writes leak out of the test."""
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    registry = _build_tool_registry()
    return registry["archive_to_libv2"], tmp_path


@pytest.mark.asyncio
async def test_libv2_archival_fails_when_chunks_stale(isolated_archive):
    """When prior-run chunks under the same slug carry the wrong
    course-code prefix and no fresh chunks come in this run, archival
    must fail with TRAINFORGE_OUTPUT_STALE — and the prior chunks must
    NOT be preserved silently in the new archive."""
    archive, root = isolated_archive
    course_name = "RDF_SHACL_550"
    slug = "rdf-shacl-550"
    course_dir = root / "LibV2" / "courses" / slug

    # Prime the LibV2 destination with prior-run chunks (the rdf-shacl-550
    # leak shape: smoke_hifi_rag_chunk_* IDs from an April 22 run).
    (course_dir / "corpus").mkdir(parents=True)
    prior_chunks = course_dir / "corpus" / "chunks.jsonl"
    prior_chunks.write_text(
        "\n".join(
            json.dumps({"id": f"smoke_hifi_rag_chunk_{i:05d}", "text": "x"})
            for i in range(32)
        )
        + "\n"
    )
    # Push mtime into the past so the freshness gate doesn't accept on mtime.
    past = time.time() - 3600
    import os as _os
    _os.utime(prior_chunks, (past, past))

    # Set up a Trainforge dir whose chunks STILL carry the stale prefix —
    # i.e. trainforge_assessment failed, so the corpus dir holds last
    # week's smoke chunks. Archival will copy them in, then the gate
    # catches the prefix mismatch and refuses to write the manifest.
    # The archival code expects ``project_workspace/trainforge/`` —
    # mirror that layout exactly.
    workspace = root / "project_workspace"
    tf = workspace / "trainforge"
    (tf / "corpus").mkdir(parents=True)
    stale_chunks_src = tf / "corpus" / "chunks.jsonl"
    stale_chunks_src.write_text(
        "\n".join(
            json.dumps({"id": f"smoke_hifi_rag_chunk_{i:05d}", "text": "x"})
            for i in range(32)
        )
        + "\n"
    )
    _os.utime(stale_chunks_src, (past, past))

    result_raw = await archive(
        course_name=course_name,
        domain="general",
        project_workspace=str(workspace),
    )
    result = json.loads(result_raw)

    assert result.get("success") is False, result
    assert result.get("error_code") == "TRAINFORGE_OUTPUT_STALE", result
    assert result.get("expected_prefix") == "rdf_shacl_550_chunk_"
    # Manifest must NOT have been written — the gate refuses before the
    # manifest dump.
    assert not (course_dir / "manifest.json").exists(), (
        "archival wrote a manifest despite stale chunks — fail-closed "
        "gate is broken"
    )


@pytest.mark.asyncio
async def test_libv2_archival_succeeds_when_chunks_fresh(isolated_archive):
    """Happy path regression guard: when Trainforge produces chunks
    matching the current course_code, archival writes the manifest."""
    archive, root = isolated_archive
    course_name = "RDF_SHACL_550"
    slug = "rdf-shacl-550"
    course_dir = root / "LibV2" / "courses" / slug

    # Set up a Trainforge dir with fresh, course-matching chunks.
    workspace = root / "project_workspace"
    tf = workspace / "trainforge"
    (tf / "corpus").mkdir(parents=True)
    fresh = tf / "corpus" / "chunks.jsonl"
    fresh.write_text(
        '{"id":"rdf_shacl_550_chunk_00001","text":"x"}\n'
        '{"id":"rdf_shacl_550_chunk_00002","text":"y"}\n'
    )

    result_raw = await archive(
        course_name=course_name,
        domain="general",
        project_workspace=str(workspace),
    )
    result = json.loads(result_raw)

    assert "error" not in result, result
    assert result.get("success") is True, result
    # Manifest landed.
    manifest_path = course_dir / "manifest.json"
    assert manifest_path.exists(), result
    # Chunks landed at the destination (Phase 7c canonical path).
    archived_chunks = course_dir / "imscc_chunks" / "chunks.jsonl"
    assert archived_chunks.exists()
    # Chunk IDs match the current course code, confirming we copied the
    # *fresh* file rather than preserving anything stale.
    head = archived_chunks.read_text().splitlines()[0]
    assert json.loads(head)["id"].startswith("rdf_shacl_550_chunk_")


@pytest.mark.asyncio
async def test_libv2_archival_proceeds_when_trainforge_intentionally_absent(
    isolated_archive,
):
    """Guardrail: the fail-closed gate must NOT break workflows where
    Trainforge is intentionally skipped (e.g. DART-only batch runs that
    skip ``trainforge_assessment`` via ``--no-assessments``). With no
    Trainforge dir resolvable, no chunks file lands at the destination —
    archival must complete and write the manifest."""
    archive, root = isolated_archive
    course_name = "DART_ONLY_999"
    slug = "dart-only-999"
    course_dir = root / "LibV2" / "courses" / slug

    # No project_workspace, no assessment_path. The heuristic fallback
    # has nothing to find under tmp_path. Archival should still succeed.
    result_raw = await archive(
        course_name=course_name,
        domain="general",
        # Explicit empty paths so we don't hit the project-root scan.
        pdf_paths="",
        html_paths="",
    )
    result = json.loads(result_raw)
    assert "error" not in result, result
    assert result.get("success") is True
    assert (course_dir / "manifest.json").exists()
    # No chunks file — feature flags should advertise false (and the
    # archive shouldn't have a chunks.jsonl present).
    assert not (course_dir / "corpus" / "chunks.jsonl").exists()


@pytest.mark.asyncio
async def test_libv2_archival_drops_prior_chunks_before_copy(isolated_archive):
    """The fix removes any pre-existing chunks.jsonl at the destination
    before the copy block runs — so a re-run never silently preserves
    last week's chunks under the same slug."""
    archive, root = isolated_archive
    course_name = "RDF_SHACL_550"
    slug = "rdf-shacl-550"
    course_dir = root / "LibV2" / "courses" / slug

    # Plant a prior-run chunks file at the destination.
    (course_dir / "corpus").mkdir(parents=True)
    prior = course_dir / "corpus" / "chunks.jsonl"
    prior.write_text('{"id":"smoke_hifi_rag_chunk_00001","text":"x"}\n')

    # Run archival without any Trainforge dir to copy from
    # (Trainforge intentionally absent → no chunks copy happens).
    result_raw = await archive(
        course_name=course_name,
        domain="general",
        pdf_paths="",
        html_paths="",
    )
    result = json.loads(result_raw)

    # Archival succeeds (intentional-absence path) AND the stale chunks
    # are gone — never preserved silently into the new archive.
    assert "error" not in result, result
    assert not prior.exists(), (
        "prior-run chunks.jsonl survived into the fresh archive — "
        "fail-closed gate is broken"
    )
