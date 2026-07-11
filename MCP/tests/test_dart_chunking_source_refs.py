"""B1+B2 end-to-end: ``_run_dart_chunking`` populates
``source.source_references[]`` from the ``data-dart-block-id`` /
``data-dart-pages`` attributes the DART converter stamps on staged HTML.

The canonical chunker harvests the ``{block_id, pages}`` pairs and the
DART-side ``_create_chunk`` callback mints
``semantik:{slug}#{block_id}`` sourceIds (slug = staged-HTML file stem,
matching the source_refs validator + source-router key). DART HTML without
those attributes keeps ``source_references`` unset — legacy corpora are
byte-stable.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402

# Canonical sourceId pattern (mirrors source_reference.schema.json).
_SOURCE_ID_RE = re.compile(r"^semantik:[a-z0-9_-]+#[a-z0-9_-]+$")


@pytest.fixture
def dart_chunking_tool(monkeypatch, tmp_path):
    libv2_root = tmp_path / "LibV2"
    libv2_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    monkeypatch.setattr(pipeline_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline_tools, "COURSEFORGE_INPUTS", tmp_path / "cf_inputs"
    )
    registry = _build_tool_registry()
    return registry["run_dart_chunking"]


_PROSE = (
    "The mitochondrion is the powerhouse of the cell. Cells use "
    "mitochondria to produce ATP through oxidative phosphorylation, which "
    "is the primary mechanism by which cells generate energy for metabolic "
    "activity across nearly every tissue type in the organism."
)


def _write_dart_html(path: Path, *, block_id: str, pages: str, title: str) -> None:
    """Write DART-shaped HTML: a section wrapper carrying data-dart-*."""
    path.write_text(
        "<!doctype html><html><head><title>" + title + "</title></head><body>"
        '<main class="dart-document">'
        f'<section class="dart-section" role="region" '
        f'aria-labelledby="h" data-dart-source="dart_converter" '
        f'data-dart-block-id="{block_id}" data-dart-pages="{pages}">'
        f"<h2>{title}</h2><p>{_PROSE}</p></section>"
        "</main></body></html>",
        encoding="utf-8",
    )


def _load_chunks(result_str: str):
    result = json.loads(result_str)
    assert result.get("success"), f"chunking errored: {result}"
    chunks_path = Path(result["dart_chunks_path"])
    assert chunks_path.exists(), f"no chunks at {chunks_path}"
    with chunks_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_dart_chunks_carry_source_references(dart_chunking_tool, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_dart_html(
        staging / "lesson_01.html", block_id="s1", pages="1-3", title="Cells"
    )
    _write_dart_html(
        staging / "lesson_02.html", block_id="s2", pages="7", title="Energy"
    )

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="DART_REFS_TEST",
        staging_dir=str(staging),
    )))
    assert chunks, "expected at least one chunk emitted"

    by_slug = {}
    for chunk in chunks:
        refs = chunk.get("source", {}).get("source_references")
        assert refs, f"chunk {chunk.get('id')!r} missing source_references"
        for ref in refs:
            assert _SOURCE_ID_RE.match(ref["sourceId"]), ref["sourceId"]
            assert ref["role"] == "primary"
            by_slug[ref["sourceId"]] = ref

    # Slug = file stem lower-hyphenated; block_id from the attribute.
    assert "semantik:lesson_01#s1" in by_slug
    assert by_slug["semantik:lesson_01#s1"]["pages"] == [1, 2, 3]
    assert "semantik:lesson_02#s2" in by_slug
    assert by_slug["semantik:lesson_02#s2"]["pages"] == [7]


def test_dart_chunks_without_attrs_have_no_source_references(
    dart_chunking_tool, tmp_path
):
    """Legacy DART HTML carrying no data-dart-* attributes: chunks keep
    ``source.source_references`` unset (byte-stable additive contract)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    # No data-dart-* attributes anywhere.
    (staging / "legacy.html").write_text(
        "<!doctype html><html><head><title>Legacy</title></head><body>"
        '<main class="dart-document"><section class="dart-section">'
        f"<h2>Legacy</h2><p>{_PROSE}</p></section></main></body></html>",
        encoding="utf-8",
    )

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="DART_LEGACY_TEST",
        staging_dir=str(staging),
    )))
    assert chunks, "expected at least one chunk emitted"
    for chunk in chunks:
        assert "source_references" not in chunk.get("source", {}), (
            f"chunk {chunk.get('id')!r} unexpectedly carries source_references"
        )


def test_minted_source_ids_resolve_against_source_refs_validator(
    dart_chunking_tool, tmp_path
):
    """The sourceIds the chunker mints validate against the canonical shape
    the PageSourceRefValidator enforces."""
    from lib.validators.source_refs import _SOURCE_ID_RE as VALIDATOR_RE

    staging = tmp_path / "staging"
    staging.mkdir()
    _write_dart_html(
        staging / "chapter_one.html", block_id="s3_c0", pages="4-6", title="Ch1"
    )

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="DART_VALIDATE_TEST",
        staging_dir=str(staging),
    )))
    seen = False
    for chunk in chunks:
        for ref in chunk.get("source", {}).get("source_references", []) or []:
            assert VALIDATOR_RE.match(ref["sourceId"]), (
                f"validator would reject minted sourceId {ref['sourceId']!r}"
            )
            seen = True
    assert seen, "expected at least one minted source reference"


def test_synthesized_suffix_stripped_from_minted_slug(
    dart_chunking_tool, tmp_path
):
    """DART's multi-source strategy emits ``{stem}_synthesized.html``. The
    minted sourceId slug MUST strip the ``_synthesized`` suffix so it matches
    the join key the source_refs validator + source-router derive from the
    SAME filename (both apply ``dart_slug_from_filename``). Without the strip
    the chunker keys ``semantik:intro_chapter_synthesized#s3_c0`` while every
    other consumer keys ``semantik:intro_chapter#s3_c0`` -> unresolvable."""
    from lib.validators.source_refs import (
        SOURCE_ID_RE as VALIDATOR_RE,
        dart_slug_from_filename,
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    fname = "intro_chapter_synthesized.html"
    _write_dart_html(
        staging / fname, block_id="s3_c0", pages="5", title="Intro"
    )

    # The slug the validator / source-router key on, derived from the SAME
    # filename. This is the authoritative join key the chunk must mint.
    canonical_slug = dart_slug_from_filename(fname)
    assert canonical_slug == "intro_chapter", canonical_slug
    expected_id = f"semantik:{canonical_slug}#s3_c0"

    chunks = _load_chunks(asyncio.run(dart_chunking_tool(
        course_name="DART_SYNTH_TEST",
        staging_dir=str(staging),
    )))

    minted = set()
    for chunk in chunks:
        for ref in chunk.get("source", {}).get("source_references", []) or []:
            assert VALIDATOR_RE.match(ref["sourceId"]), ref["sourceId"]
            minted.add(ref["sourceId"])

    assert expected_id in minted, (
        f"expected suffix-stripped sourceId {expected_id!r}; got {minted!r}"
    )
    # The un-stripped form must NEVER be minted (the regression).
    assert "semantik:intro_chapter_synthesized#s3_c0" not in minted
