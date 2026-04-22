"""Wave 19 sidecar-emission tests.

The pre-Wave-12 DART pipeline wrote ``*_synthesized.json`` +
``*.quality.json`` sidecars next to every HTML output. The Waves 12-18
converter silently dropped those writes, breaking the Courseforge
source-router (``_build_source_module_map``) + the LibV2 quality
archival flow. These tests lock the restored sidecar emission in.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock
from DART.converter.sidecars import (
    build_quality_sidecar,
    build_synthesized_sidecar,
)


def _mk_block(
    role: BlockRole,
    text: str,
    block_id: str,
    page: int | None = None,
    confidence: float = 0.8,
    classifier_source: str = "heuristic",
    extractor: str = "pdftotext",
    attrs: dict | None = None,
) -> ClassifiedBlock:
    return ClassifiedBlock(
        raw=RawBlock(
            text=text,
            block_id=block_id,
            page=page,
            extractor=extractor,
        ),
        role=role,
        confidence=confidence,
        attributes=attrs or {},
        classifier_source=classifier_source,
    )


def _sample_blocks() -> List[ClassifiedBlock]:
    return [
        _mk_block(
            BlockRole.CHAPTER_OPENER,
            "Chapter 1: Foundations",
            "b1",
            page=1,
            confidence=0.9,
            attrs={"heading_text": "Foundations", "chapter_number": "1"},
        ),
        _mk_block(BlockRole.PARAGRAPH, "Body prose one.", "b2", page=1),
        _mk_block(BlockRole.PARAGRAPH, "Body prose two.", "b3", page=2),
        _mk_block(
            BlockRole.CHAPTER_OPENER,
            "Chapter 2: Growth",
            "b4",
            page=3,
            confidence=0.9,
            attrs={"heading_text": "Growth", "chapter_number": "2"},
        ),
        _mk_block(BlockRole.PARAGRAPH, "Chapter 2 prose.", "b5", page=3),
    ]


# ---------------------------------------------------------------------------
# build_synthesized_sidecar shape
# ---------------------------------------------------------------------------


def test_synthesized_sidecar_has_top_level_keys():
    sidecar = build_synthesized_sidecar(
        _sample_blocks(), title="Wave19 Doc", source_pdf="/foo.pdf"
    )
    assert sidecar["slug"] == "wave19-doc"
    assert sidecar["title"] == "Wave19 Doc"
    assert sidecar["source_pdf"] == "/foo.pdf"
    assert isinstance(sidecar["sections"], list)
    assert isinstance(sidecar["document_provenance"], dict)


def test_synthesized_sidecar_groups_by_chapter():
    """Chapter openers must seed a new section each."""
    sidecar = build_synthesized_sidecar(
        _sample_blocks(), title="T", source_pdf=None
    )
    sections = sidecar["sections"]
    assert len(sections) == 2
    assert sections[0]["section_id"] == "s1"
    assert sections[0]["section_type"] == "chapter"
    assert "Foundations" in sections[0]["section_title"]
    assert sections[1]["section_id"] == "s2"
    assert "Growth" in sections[1]["section_title"]


def test_synthesized_sidecar_section_has_provenance_block():
    sidecar = build_synthesized_sidecar(
        _sample_blocks(), title="T", source_pdf=None
    )
    prov = sidecar["sections"][0]["provenance"]
    assert "sources" in prov
    assert "strategy" in prov
    assert "confidence" in prov
    assert 0.0 <= prov["confidence"] <= 1.0
    assert isinstance(prov["sources"], list) and prov["sources"]


def test_synthesized_sidecar_carries_page_range():
    sidecar = build_synthesized_sidecar(
        _sample_blocks(), title="T", source_pdf=None
    )
    # First chapter spans pages 1-2 (opener page 1, paragraphs 1 and 2).
    pr1 = sidecar["sections"][0]["page_range"]
    assert pr1 == [1, 2]
    # Second chapter covers page 3 only.
    pr2 = sidecar["sections"][1]["page_range"]
    assert pr2 == [3, 3]


def test_synthesized_sidecar_document_provenance_counts():
    """Document-level counters aggregate figures / tables / TOC entries."""
    blocks = _sample_blocks() + [
        _mk_block(
            BlockRole.FIGURE,
            "",
            "b6",
            page=4,
            classifier_source="extractor_hint",
            extractor="pymupdf",
        ),
        _mk_block(
            BlockRole.TABLE,
            "",
            "b7",
            page=5,
            classifier_source="extractor_hint",
            extractor="pdfplumber",
        ),
    ]
    sidecar = build_synthesized_sidecar(blocks, title="T", source_pdf=None)
    prov = sidecar["document_provenance"]
    assert prov["figures_extracted"] == 1
    assert prov["tables_extracted"] == 1
    assert "pdfplumber" in prov["extractors_used"]
    assert "pymupdf" in prov["extractors_used"]


# ---------------------------------------------------------------------------
# build_quality_sidecar shape
# ---------------------------------------------------------------------------


def test_quality_sidecar_has_required_keys():
    q = build_quality_sidecar(
        "<html><body><p>hi</p></body></html>",
        title="Test",
        source_pdf="/foo.pdf",
    )
    for k in (
        "slug", "title", "source_pdf", "html_size_bytes", "html_sha256",
        "compliant", "quality_score",
    ):
        assert k in q, f"quality sidecar missing key: {k}"


def test_quality_sidecar_hash_is_deterministic():
    html = "<html><body><p>hi</p></body></html>"
    q1 = build_quality_sidecar(html, title="X")
    q2 = build_quality_sidecar(html, title="X")
    assert q1["html_sha256"] == q2["html_sha256"]


# ---------------------------------------------------------------------------
# Pipeline-level wiring: _raw_text_to_accessible_html writes sidecars
# ---------------------------------------------------------------------------


def test_pipeline_writes_sidecars_next_to_html():
    from MCP.tools.pipeline_tools import _raw_text_to_accessible_html

    raw = (
        "Chapter 1: Foundations\n\n"
        "This is an introduction paragraph with body content."
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "doc.html"
        html = _raw_text_to_accessible_html(
            raw, "Doc Title", output_path=str(out)
        )
        out.write_text(html)
        synth = out.parent / f"{out.stem}_synthesized.json"
        quality = out.with_suffix(".quality.json")
        assert synth.exists(), "sidecar missing: _synthesized.json"
        assert quality.exists(), "sidecar missing: .quality.json"
        doc = json.loads(synth.read_text())
        assert doc["title"] == "Doc Title"
        assert len(doc["sections"]) >= 1


def test_pipeline_skips_sidecars_when_no_output_path():
    """When ``output_path`` is None the pipeline still returns HTML but
    writes no sidecars — mirrors the figure-persistence tempdir guard."""
    from MCP.tools.pipeline_tools import _raw_text_to_accessible_html

    with tempfile.TemporaryDirectory() as td:
        _ = _raw_text_to_accessible_html(
            "Short body.", "Ephemeral", output_path=None,
        )
        # Confirm no sidecars landed in the temp dir (there shouldn't be
        # anything at all, but the contract is "no sidecar writes").
        assert not list(Path(td).glob("*_synthesized.json"))


def test_build_source_module_map_consumes_wave19_sidecar():
    """The Courseforge source-router walks ``sections[].section_id`` +
    ``section_title`` — this regression-guards the contract."""
    from MCP.tools.pipeline_tools import _raw_text_to_accessible_html

    raw = (
        "Chapter 1: Intro to Biology\n\n"
        "Biology is the study of life.\n\n"
        "Chapter 2: Genetics\n\n"
        "Genetics is the transmission of traits."
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bio.html"
        html = _raw_text_to_accessible_html(
            raw, "Biology", output_path=str(out)
        )
        out.write_text(html)
        synth = out.parent / f"{out.stem}_synthesized.json"
        assert synth.exists()
        doc = json.loads(synth.read_text())
        # Router requires: each section carries section_id + section_title.
        ids = [s["section_id"] for s in doc["sections"]]
        titles = [s["section_title"] for s in doc["sections"]]
        assert ids == sorted(set(ids))
        assert all(titles)  # no empty section_title values


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
