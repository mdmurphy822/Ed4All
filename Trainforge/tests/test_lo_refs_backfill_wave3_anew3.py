"""Wave3-Anew3 — DART chunk LO-ref back-fill (Finding F3).

Auditor finding F3: every chunk in a real course's
``dart_chunks/chunks.jsonl``
shipped with empty ``learning_outcome_refs[]`` because the chunker emits
chunks BEFORE ``_plan_course_structure`` mints TO-NN / CO-NN IDs. After
course planning publishes ``objective_ids`` (Wave2-I7 plumbing), a back-
fill step text-scans the on-disk chunks for canonical LO IDs and writes
matches back, filtered against the synthesized allowlist.

This regression suite covers four invariants:

1. The scan helper finds whole-word ``TO-NN`` / ``CO-NN`` matches in
   ``text`` AND ``html`` bodies (auditor surface).
2. The allowlist filter rejects IDs not in the synthesized set so
   stray uppercase tokens in textbook prose (``ALGEBRA-12``,
   ``CHAPTER-03``) never become LO refs.
3. The back-fill helper rewrites the on-disk JSONL file in place,
   updates only chunks with matches, and preserves existing refs via
   union semantics (never destroys upstream-emitted refs).
4. Missing chunks file / empty allowlist degrade gracefully — the
   helper returns a structured ``skipped`` summary instead of raising.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology.learning_objectives import scan_lo_refs
from MCP.tools.pipeline_tools import _backfill_dart_chunk_lo_refs


# ---------------------------------------------------------------------------
# scan_lo_refs — the per-chunk text-scan primitive
# ---------------------------------------------------------------------------


def test_scan_finds_to_co_ids_in_text():
    """Auditor expectation: whole-word TO-NN / CO-NN in text → refs."""
    refs = scan_lo_refs(
        text="This activity supports TO-01 and develops CO-05 mastery.",
        allowed_ids=["TO-01", "CO-05", "CO-12"],
    )
    assert refs == ["CO-05", "TO-01"]


def test_scan_finds_ids_in_html_too():
    """``data-cf-objective-id`` attribute values are also surfaced."""
    refs = scan_lo_refs(
        html='<p data-cf-objective-id="TO-02">Solve linear equations.</p>',
        allowed_ids=["TO-01", "TO-02"],
    )
    assert refs == ["TO-02"]


def test_scan_dedupes_across_text_and_html():
    """Same ID in both text + html surfaces once (set semantics)."""
    refs = scan_lo_refs(
        text="See TO-03 for context.",
        html='<span data-cf-objective-id="TO-03">x</span>',
        allowed_ids=["TO-03"],
    )
    assert refs == ["TO-03"]


def test_scan_allowlist_rejects_off_list_tokens():
    """False-positive guard: ALGEBRA-12 / CHAPTER-03 are NOT LO refs."""
    refs = scan_lo_refs(
        text="See ALGEBRA-12 in CHAPTER-03; this teaches TO-01.",
        allowed_ids=["TO-01"],
    )
    # Off-list uppercase tokens dropped; in-list ID kept.
    assert refs == ["TO-01"]


def test_scan_empty_allowlist_permissive():
    """Legacy unit-test mode: no allowlist → every regex hit returned."""
    refs = scan_lo_refs(
        text="See TO-99 and CO-42.",
        allowed_ids=None,
    )
    assert refs == ["CO-42", "TO-99"]


def test_scan_word_boundary_skips_substring_hits():
    """``XTO-01`` is not a match (word-boundary required)."""
    refs = scan_lo_refs(
        text="The variable XTO-01ABC is not a learning objective.",
        allowed_ids=["TO-01"],
    )
    assert refs == []


def test_scan_empty_inputs_return_empty_list():
    assert scan_lo_refs(text="", html="", allowed_ids=["TO-01"]) == []
    assert scan_lo_refs(allowed_ids=["TO-01"]) == []


# ---------------------------------------------------------------------------
# _backfill_dart_chunk_lo_refs — the on-disk rewriter
# ---------------------------------------------------------------------------


def _write_chunks(path: Path, chunks: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")


def _read_chunks(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_backfill_writes_matches_against_allowlist(tmp_path: Path):
    """End-to-end: scan, filter, persist — only allowed IDs land."""
    libv2_root = tmp_path / "LibV2"
    chunks_path = libv2_root / "courses" / "phys-101" / "dart_chunks" / "chunks.jsonl"
    _write_chunks(chunks_path, [
        {
            "id": "phys_101_chunk_00001",
            "schema_version": "v4",
            "chunk_type": "explanation",
            "text": "This section addresses TO-01 and supports CO-03.",
            "html": "",
            "learning_outcome_refs": [],
        },
        {
            "id": "phys_101_chunk_00002",
            "schema_version": "v4",
            "chunk_type": "explanation",
            "text": "Newton's first law — see ALGEBRA-12 for vector math.",
            "html": '<p data-cf-objective-id="TO-02">Apply F=ma.</p>',
            "learning_outcome_refs": [],
        },
        {
            "id": "phys_101_chunk_00003",
            "schema_version": "v4",
            "chunk_type": "explanation",
            "text": "Energy conservation — no LO refs in this prose.",
            "html": "",
            "learning_outcome_refs": [],
        },
    ])

    summary = _backfill_dart_chunk_lo_refs(
        course_slug="phys-101",
        objective_ids=["TO-01", "TO-02", "CO-03", "CO-04"],
        libv2_root=str(libv2_root),
    )

    assert summary["chunks_scanned"] == 3
    assert summary["chunks_updated"] == 2
    assert summary["new_refs_total"] == 3  # TO-01 + CO-03 + TO-02
    assert summary["skipped"] is False

    rewritten = _read_chunks(chunks_path)
    assert rewritten[0]["learning_outcome_refs"] == ["CO-03", "TO-01"]
    assert rewritten[1]["learning_outcome_refs"] == ["TO-02"]
    assert rewritten[2]["learning_outcome_refs"] == []


def test_backfill_preserves_existing_refs_via_union(tmp_path: Path):
    """Union semantics: never destroy pre-existing refs (legacy corpora)."""
    libv2_root = tmp_path / "LibV2"
    chunks_path = libv2_root / "courses" / "phys-101" / "dart_chunks" / "chunks.jsonl"
    _write_chunks(chunks_path, [
        {
            "id": "phys_101_chunk_00001",
            "text": "Now addresses TO-02 explicitly.",
            "html": "",
            "learning_outcome_refs": ["TO-01"],  # pre-existing upstream ref
        },
    ])

    summary = _backfill_dart_chunk_lo_refs(
        course_slug="phys-101",
        objective_ids=["TO-01", "TO-02"],
        libv2_root=str(libv2_root),
    )

    assert summary["chunks_updated"] == 1
    assert summary["new_refs_total"] == 1
    rewritten = _read_chunks(chunks_path)
    # TO-01 preserved AND TO-02 added — union, not replacement.
    assert rewritten[0]["learning_outcome_refs"] == ["TO-01", "TO-02"]


def test_backfill_missing_chunks_file_skips_gracefully(tmp_path: Path):
    """Best-effort: missing JSONL → structured skip, no exception."""
    libv2_root = tmp_path / "LibV2"
    summary = _backfill_dart_chunk_lo_refs(
        course_slug="nonexistent-course",
        objective_ids=["TO-01"],
        libv2_root=str(libv2_root),
    )
    assert summary["skipped"] is True
    assert summary["reason"] == "chunks_jsonl_missing"


def test_backfill_empty_allowlist_is_noop(tmp_path: Path):
    """Empty allowlist → no scan (preserves false-positive guard)."""
    libv2_root = tmp_path / "LibV2"
    chunks_path = libv2_root / "courses" / "phys-101" / "dart_chunks" / "chunks.jsonl"
    _write_chunks(chunks_path, [
        {"id": "c1", "text": "TO-99 mentioned.", "html": "", "learning_outcome_refs": []},
    ])
    summary = _backfill_dart_chunk_lo_refs(
        course_slug="phys-101",
        objective_ids=[],
        libv2_root=str(libv2_root),
    )
    assert summary["skipped"] is True
    assert summary["reason"] == "empty_objective_ids_allowlist"
    # File untouched.
    rewritten = _read_chunks(chunks_path)
    assert rewritten[0]["learning_outcome_refs"] == []


def test_backfill_no_matches_leaves_file_unchanged(tmp_path: Path):
    """When no chunks need updates, the JSONL file is byte-identical."""
    libv2_root = tmp_path / "LibV2"
    chunks_path = libv2_root / "courses" / "phys-101" / "dart_chunks" / "chunks.jsonl"
    _write_chunks(chunks_path, [
        {"id": "c1", "text": "No LO ids here.", "html": "", "learning_outcome_refs": []},
    ])
    before = chunks_path.read_bytes()
    summary = _backfill_dart_chunk_lo_refs(
        course_slug="phys-101",
        objective_ids=["TO-01"],
        libv2_root=str(libv2_root),
    )
    assert summary["chunks_updated"] == 0
    assert summary["new_refs_total"] == 0
    after = chunks_path.read_bytes()
    assert before == after, "no-op back-fill must not rewrite bytes"
