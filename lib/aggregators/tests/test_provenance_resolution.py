"""``ProvenanceResolutionAggregator`` tests.

Coverage:

1. All three ratios computed correctly (chunks_with_provenance, anchor,
   book-chunk) over a synthetic corpus.
2. Empty-string data-cf-source-ids attr counts as NO provenance (Wave-27
   contract) — never a resolution failure.
3. An unresolved token is listed under the right ratio's ``unresolved``.
4. Per-module_id breakdown of provenance-free chunks.
5. Missing staging dir → anchor metric skipped marker (never fabricated 0s).
6. Missing imscc chunkset → aggregator returns None (skip) without raising.
7. Schema validation of the emitted report.

All fixtures are tmp_path synthetic — NO real-course slugs / paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lib.aggregators.provenance_resolution import (
    ProvenanceResolutionAggregator,
    SCHEMA_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "aggregators" / "provenance_resolution.schema.json"
)

# Two synthetic book stems + anchors.
_STEM_A = "sample-book-ch01_accessible"
_STEM_B = "sample-book-ch02_accessible"


def _cf_block(source_ids: str) -> str:
    """A course-page block with a data-cf-source-ids attr (may be empty)."""
    return f'<div data-cf-source-ids="{source_ids}"><p>body</p></div>'


def _imscc_chunk(chunk_id: str, module_id: str, html: str) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "html": html,
        "source": {"module_id": module_id},
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _build_course(course_dir: Path) -> None:
    """Build a tiny imscc + semantik chunkset.

    Distinct tokens across the corpus:
      * semantik:{A}#a1  — anchor present in staged A, book chunk present  → both resolve
      * semantik:{A}#a2  — anchor present in staged A, NO book chunk        → book unresolved
      * semantik:{B}#b9  — anchor MISSING in staged B, book chunk present   → anchor unresolved
    """
    tok_a1 = f"semantik:{_STEM_A}#a1"
    tok_a2 = f"semantik:{_STEM_A}#a2"
    tok_b9 = f"semantik:{_STEM_B}#b9"

    imscc = [
        # multi-token attr (two tokens), module week_01
        _imscc_chunk("c1", "week_01", _cf_block(f"{tok_a1},{tok_a2}")),
        # single token, module week_01
        _imscc_chunk("c2", "week_01", _cf_block(tok_b9)),
        # empty-string attr → NO provenance (Wave-27), module week_02
        _imscc_chunk("c3", "week_02", _cf_block("")),
        # no attr at all → NO provenance, module week_02
        _imscc_chunk("c4", "week_02", "<p>plain body, no provenance</p>"),
    ]
    _write_jsonl(course_dir / "imscc_chunks" / "chunks.jsonl", imscc)

    # Book-side semantik chunks: cover tok_a1 + tok_b9, but NOT tok_a2.
    book_chunks = [
        {
            "id": "d1",
            "source": {
                "source_references": [
                    {"sourceId": tok_a1, "role": "primary"},
                ]
            },
        },
        {
            "id": "d2",
            "source": {
                "source_references": [
                    {"sourceId": tok_b9, "role": "primary"},
                ]
            },
        },
    ]
    _write_jsonl(course_dir / "semantik_chunks" / "chunks.jsonl", book_chunks)


def _build_staging(staging_dir: Path) -> None:
    """Staged accessible HTML: A carries a1 + a2; B carries only b0 (not b9)."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / f"{_STEM_A}.html").write_text(
        '<section data-semantik-block-id="a1"><h2>A1</h2></section>'
        '<section data-semantik-block-id="a2"><h2>A2</h2></section>',
        encoding="utf-8",
    )
    (staging_dir / f"{_STEM_B}.html").write_text(
        '<section data-semantik-block-id="b0"><h2>B0</h2></section>',
        encoding="utf-8",
    )


def test_all_three_ratios_and_unresolved(tmp_path: Path):
    course_dir = tmp_path / "course"
    staging_dir = tmp_path / "staging"
    _build_course(course_dir)
    _build_staging(staging_dir)

    report = ProvenanceResolutionAggregator(
        course_slug="synthetic-course",
        run_id="R1",
        libv2_course_path=course_dir,
        staging_dir=staging_dir,
    ).build()
    assert report is not None

    # 3 distinct tokens (a1, a2, b9); 2/4 chunks carry provenance.
    assert report["distinct_source_id_count"] == 3
    prov = report["chunks_with_provenance"]
    assert prov == {"count": 2, "total": 4, "ratio": 0.5}

    # Anchor resolution: a1 + a2 present in staged A; b9 absent in staged B.
    anchor = report["source_ids_anchor_resolved"]
    assert anchor["count"] == 2
    assert anchor["total"] == 3
    assert anchor["ratio"] == round(2 / 3, 4)
    assert anchor["unresolved"] == [f"semantik:{_STEM_B}#b9"]

    # Book-chunk resolution: a1 + b9 covered; a2 NOT.
    book = report["source_ids_book_chunk_resolved"]
    assert book["count"] == 2
    assert book["total"] == 3
    assert book["unresolved"] == [f"semantik:{_STEM_A}#a2"]

    assert report["staging_dir_used"] == str(staging_dir)


def test_empty_and_missing_attrs_count_as_no_provenance(tmp_path: Path):
    course_dir = tmp_path / "course"
    _build_course(course_dir)

    report = ProvenanceResolutionAggregator(
        course_slug="synthetic-course",
        run_id="R1",
        libv2_course_path=course_dir,
        staging_dir=tmp_path / "staging",  # nonexistent → anchor skipped
    ).build()
    assert report is not None
    # c3 (empty attr) + c4 (no attr) are the two provenance-free chunks,
    # both in week_02.
    assert report["chunks_with_provenance"]["count"] == 2
    assert report["provenance_free_by_module"] == {"week_02": 2}


def test_per_module_breakdown(tmp_path: Path):
    course_dir = tmp_path / "course"
    # Two provenance-free chunks in different modules.
    imscc = [
        _imscc_chunk("c1", "week_01", _cf_block(f"semantik:{_STEM_A}#a1")),
        _imscc_chunk("c2", "week_03", _cf_block("")),
        _imscc_chunk("c3", "week_05", "<p>no attr</p>"),
        _imscc_chunk("c4", "week_05", _cf_block("")),
    ]
    _write_jsonl(course_dir / "imscc_chunks" / "chunks.jsonl", imscc)

    report = ProvenanceResolutionAggregator(
        libv2_course_path=course_dir,
    ).build()
    assert report is not None
    assert report["provenance_free_by_module"] == {"week_03": 1, "week_05": 2}


def test_missing_staging_dir_skips_anchor_metric(tmp_path: Path):
    course_dir = tmp_path / "course"
    _build_course(course_dir)

    report = ProvenanceResolutionAggregator(
        libv2_course_path=course_dir,
        # no staging_dir passed AND no phase_outputs.staging → skip
    ).build()
    assert report is not None
    assert report["source_ids_anchor_resolved"] == {"skipped": "no_staging_dir"}
    assert report["staging_dir_used"] is None
    # Book-chunk resolution still measured.
    assert report["source_ids_book_chunk_resolved"]["total"] == 3


def test_missing_imscc_chunkset_returns_none(tmp_path: Path):
    course_dir = tmp_path / "course"  # nothing written
    agg = ProvenanceResolutionAggregator(libv2_course_path=course_dir)
    # build() returns None (skip) without raising.
    assert agg.build() is None
    # write() also returns None and writes no file.
    out = tmp_path / "out" / "provenance_resolution_report.json"
    assert agg.write(out) is None
    assert not out.exists()


def test_staging_dir_from_phase_outputs(tmp_path: Path):
    """Anchor root resolves from phase_outputs.staging.staging_dir."""
    course_dir = tmp_path / "course"
    staging_dir = tmp_path / "staging"
    _build_course(course_dir)
    _build_staging(staging_dir)

    report = ProvenanceResolutionAggregator(
        phase_outputs={
            "libv2_archival": {"course_dir": str(course_dir)},
            "staging": {"staging_dir": str(staging_dir)},
        },
        run_id="R2",
    ).build()
    assert report is not None
    assert report["staging_dir_used"] == str(staging_dir)
    assert report["source_ids_anchor_resolved"]["count"] == 2


def test_emitted_report_validates_against_schema(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    course_dir = tmp_path / "course"
    staging_dir = tmp_path / "staging"
    _build_course(course_dir)
    _build_staging(staging_dir)

    out = tmp_path / "quality" / "provenance_resolution_report.json"
    agg = ProvenanceResolutionAggregator(
        course_slug="synthetic-course",
        run_id="R1",
        libv2_course_path=course_dir,
        staging_dir=staging_dir,
    )
    written = agg.write(out)
    assert written is not None
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == SCHEMA_VERSION
    jsonschema.validate(report, schema)


def test_skipped_anchor_report_validates_against_schema(tmp_path: Path):
    """The skipped-anchor shape must also satisfy the schema oneOf branch."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    course_dir = tmp_path / "course"
    _build_course(course_dir)
    out = tmp_path / "quality" / "provenance_resolution_report.json"
    agg = ProvenanceResolutionAggregator(libv2_course_path=course_dir)
    agg.write(out)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["source_ids_anchor_resolved"] == {"skipped": "no_staging_dir"}
    jsonschema.validate(report, schema)
