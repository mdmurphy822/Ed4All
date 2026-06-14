"""W3 wiring — per-block synthesis-manifest EMISSION + gate-input routing.

Covers the connective tissue between W3's manifest projection (built) and the
ManifestCompletenessValidator (built):

* ``pipeline_tools._build_manifest_chunk_resolver`` — sourceId/chunk-id → record.
* ``pipeline_tools._emit_block_synthesis_manifest`` — sidecar JSONL writer
  (flag-gated, idempotent, one complete record per block).
* ``gate_input_routing._build_manifest_completeness`` — gate-input builder.
* End-to-end: the emitted manifest PASSES ManifestCompletenessValidator and a
  tampered/missing one FAILS it.

Synthetic blocks + chunks only — no live LLM, no cross-project fixture pinning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Courseforge.scripts.blocks import Block, Touch
from MCP.tools.pipeline_tools import (
    _build_manifest_chunk_resolver,
    _derive_block_concept_tags,
    _emit_block_synthesis_manifest,
)
from MCP.hardening.gate_input_routing import _build_manifest_completeness
from lib.validators.manifest_completeness import ManifestCompletenessValidator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "knowledge" / "block_synthesis_manifest.schema.json"
)


# --------------------------------------------------------------------------
# Fixtures / builders
# --------------------------------------------------------------------------


def _outline_touch() -> Touch:
    return Touch(
        model="qwen2.5:7b-instruct-q4_K_M",
        provider="local",
        tier="outline",
        timestamp="2026-06-13T00:00:00Z",
        decision_capture_id="EVT_0000000000000001",
        purpose="draft",
    )


def _rewrite_touch() -> Touch:
    return Touch(
        model="claude-sonnet-4-6",
        provider="anthropic",
        tier="rewrite",
        timestamp="2026-06-13T01:00:00Z",
        decision_capture_id="EVT_0000000000000002",
        purpose="author",
    )


def _concept_block(idx: int = 0) -> Block:
    content = {
        "source_refs": [
            {"sourceId": "dart:demo#b-12", "role": "primary"},
            {"sourceId": "dart:demo#b-13", "role": "contributing"},
        ],
        "key_claims": [
            {
                "claim": "An RDF graph is a set of triples.",
                "source_chunk_ids": ["demo_chunk_00012"],
            }
        ],
        "concept_tags": ["rdf-graph", "triple"],
    }
    return Block(
        block_id=f"week_04#concept_rdf-graph_{idx}",
        block_type="concept",
        page_id="week_04",
        sequence=idx,
        content=content,
        template_type="concept.definition.v1",
        key_terms=("rdf-graph", "triple"),
        touched_by=(_outline_touch(), _rewrite_touch()),
        content_hash="a" * 64,
    )


def _untagged_concept_block(idx: int = 0) -> Block:
    """A block that cites chunks but pins NO concept_tags of its own — the emit
    side must derive (ground) them from the cited chunks."""
    content = {
        "source_refs": [
            {"sourceId": "dart:demo#b-12", "role": "primary"},
            {"sourceId": "dart:demo#b-13", "role": "contributing"},
        ],
        "key_claims": [
            {
                "claim": "An RDF graph is a set of triples.",
                "source_chunk_ids": ["demo_chunk_00012"],
            }
        ],
        # NOTE: no concept_tags key — this is the gap the fix closes.
    }
    return Block(
        block_id=f"week_04#concept_untagged_{idx}",
        block_type="concept",
        page_id="week_04",
        sequence=idx,
        content=content,
        template_type="concept.definition.v1",
        key_terms=("rdf-graph", "triple"),
        touched_by=(_outline_touch(), _rewrite_touch()),
        content_hash="b" * 64,
    )


def _write_chunks_jsonl(chunks_dir: Path) -> Path:
    """Write a DART chunkset chunks.jsonl + sibling manifest.json. Returns the
    chunks.jsonl path (the emit-side resolver input)."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = [
        {
            "id": "demo_chunk_00012",
            "text": "An RDF graph is a set of triples.",
            "source": {
                "char_span": [40, 1180],
                "html_xpath": "/html/body/section[2]",
                "source_references": [
                    {"sourceId": "dart:demo#b-12", "role": "primary"}
                ],
            },
        },
        {
            "id": "demo_chunk_00013",
            "text": "A triple has a subject, predicate, and object.",
            "source": {
                "char_span": [1180, 2200],
                "source_references": [
                    {"sourceId": "dart:demo#b-13", "role": "contributing"}
                ],
            },
        },
    ]
    chunks_jsonl = chunks_dir / "chunks.jsonl"
    chunks_jsonl.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    # Sibling manifest.json (the validator's dart_chunks_manifest_path target).
    (chunks_dir / "manifest.json").write_text(
        json.dumps(
            {
                "chunkset_kind": "dart",
                "chunks_sha256": "0" * 64,
                "chunker_version": "v4",
            }
        ),
        encoding="utf-8",
    )
    return chunks_jsonl


def _write_custom_chunks_jsonl(chunks_dir: Path, chunks: list) -> Path:
    """Write an arbitrary chunkset chunks.jsonl + sibling manifest for the
    concept-tag grounding tests (control which chunks carry tags / text)."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_jsonl = chunks_dir / "chunks.jsonl"
    chunks_jsonl.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    (chunks_dir / "manifest.json").write_text(
        json.dumps(
            {
                "chunkset_kind": "dart",
                "chunks_sha256": "0" * 64,
                "chunker_version": "v4",
            }
        ),
        encoding="utf-8",
    )
    return chunks_jsonl


# --------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------


def test_resolver_maps_sourceid_and_chunk_id(tmp_path):
    chunks_jsonl = _write_chunks_jsonl(tmp_path / "dart_chunks")
    resolver = _build_manifest_chunk_resolver(chunks_jsonl)

    # sourceId resolves to the chunk record with span info.
    rec = resolver("dart:demo#b-12")
    assert rec is not None
    assert rec["id"] == "demo_chunk_00012"
    assert rec["char_span"] == [40, 1180]
    assert rec["html_xpath"] == "/html/body/section[2]"

    # Chunk id itself resolves too.
    assert resolver("demo_chunk_00013")["id"] == "demo_chunk_00013"

    # Unknown sourceId resolves to None.
    assert resolver("dart:demo#b-99") is None


def test_resolver_missing_chunkset_is_empty(tmp_path):
    resolver = _build_manifest_chunk_resolver(tmp_path / "nope" / "chunks.jsonl")
    assert resolver("dart:demo#b-12") is None


# --------------------------------------------------------------------------
# Emission helper
# --------------------------------------------------------------------------


def test_emit_writes_one_complete_record_per_block(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    chunks_jsonl = _write_chunks_jsonl(tmp_path / "dart_chunks")
    content_dir = tmp_path / "03_content_development"
    blocks = [_concept_block(0), _concept_block(1)]

    out = _emit_block_synthesis_manifest(blocks, content_dir, chunks_jsonl)
    assert out is not None
    assert out == content_dir / "block_synthesis_manifest.jsonl"

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for line in lines:
        rec = json.loads(line)
        jsonschema.validate(instance=rec, schema=schema)
        # Resolved source_chunk_ids (sourceId-mapped + key_claim chunk id).
        assert "demo_chunk_00012" in rec["source_chunk_ids"]
        assert "demo_chunk_00013" in rec["source_chunk_ids"]
        assert rec["template_id"] == "concept.definition.v1"
        assert rec["concept_tags"] == ["rdf-graph", "triple"]
        # char_spans lifted from the resolved chunk's source.char_span.
        spans = {s["chunk_id"]: s for s in rec.get("char_spans", [])}
        assert spans["demo_chunk_00012"]["char_span"] == [40, 1180]


def test_emit_disabled_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("COURSEFORGE_EMIT_BLOCKS", raising=False)
    chunks_jsonl = _write_chunks_jsonl(tmp_path / "dart_chunks")
    content_dir = tmp_path / "03_content_development"
    out = _emit_block_synthesis_manifest(
        [_concept_block(0)], content_dir, chunks_jsonl
    )
    assert out is None
    assert not (content_dir / "block_synthesis_manifest.jsonl").exists()


def test_emit_idempotent_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    chunks_jsonl = _write_chunks_jsonl(tmp_path / "dart_chunks")
    content_dir = tmp_path / "03_content_development"

    out = _emit_block_synthesis_manifest(
        [_concept_block(0)], content_dir, chunks_jsonl
    )
    assert out is not None
    sentinel = out.read_text(encoding="utf-8")

    # Re-emit with a DIFFERENT block set; the existing file must be untouched.
    out2 = _emit_block_synthesis_manifest(
        [_concept_block(0), _concept_block(1)], content_dir, chunks_jsonl
    )
    assert out2 == out
    assert out.read_text(encoding="utf-8") == sentinel  # byte-identical


def test_emit_no_blocks_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    out = _emit_block_synthesis_manifest(
        [], tmp_path / "03_content_development", None
    )
    assert out is None


# --------------------------------------------------------------------------
# Concept-tag grounding (W3: block tags grounded in cited chunks)
# --------------------------------------------------------------------------


def test_derive_block_concept_tags_unions_cited_chunk_tags(tmp_path):
    # Two cited chunks each carrying pre-computed concept_tags → union.
    chunks_jsonl = _write_custom_chunks_jsonl(
        tmp_path / "dart_chunks",
        [
            {
                "id": "demo_chunk_00012",
                "text": "An RDF graph is a set of triples.",
                "concept_tags": ["rdf-graph", "triple"],
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-12", "role": "primary"}
                    ]
                },
            },
            {
                "id": "demo_chunk_00013",
                "text": "A triple has a subject, predicate, and object.",
                "concept_tags": ["triple", "predicate"],
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-13", "role": "contributing"}
                    ]
                },
            },
        ],
    )
    resolver = _build_manifest_chunk_resolver(chunks_jsonl)
    tags = _derive_block_concept_tags(
        ["demo_chunk_00012", "demo_chunk_00013"], resolver
    )
    # Grounded union, deduped, order-stable (first-seen wins).
    assert tags == ["rdf-graph", "triple", "predicate"]


def test_emit_grounds_concept_tags_from_cited_chunk_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    chunks_jsonl = _write_custom_chunks_jsonl(
        tmp_path / "dart_chunks",
        [
            {
                "id": "demo_chunk_00012",
                "text": "An RDF graph is a set of triples.",
                "concept_tags": ["rdf-graph"],
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-12", "role": "primary"}
                    ]
                },
            },
            {
                "id": "demo_chunk_00013",
                "text": "A triple has a subject, predicate, and object.",
                "concept_tags": ["triple"],
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-13", "role": "contributing"}
                    ]
                },
            },
        ],
    )
    content_dir = tmp_path / "03_content_development"
    out = _emit_block_synthesis_manifest(
        [_untagged_concept_block(0)], content_dir, chunks_jsonl
    )
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    # Tags grounded in the union of the block's cited chunks' tags.
    assert rec["concept_tags"] == ["rdf-graph", "triple"]


def test_emit_derives_concept_tags_via_extract_for_untagged_chunk(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    # Cited chunk carries NO concept_tags but DOES carry taggable text — the
    # fallback derives tags via extract_concept_tags over the chunk text.
    chunks_jsonl = _write_custom_chunks_jsonl(
        tmp_path / "dart_chunks",
        [
            {
                "id": "demo_chunk_00012",
                # "behaviorism" is a CONCEPT_PATTERNS pedagogy term → a tag.
                "text": (
                    "Behaviorism explains learning through conditioning and "
                    "reinforcement of observable responses."
                ),
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-12", "role": "primary"}
                    ]
                },
            },
            {
                "id": "demo_chunk_00013",
                "text": "A triple has a subject, predicate, and object.",
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-13", "role": "contributing"}
                    ]
                },
            },
        ],
    )
    content_dir = tmp_path / "03_content_development"
    out = _emit_block_synthesis_manifest(
        [_untagged_concept_block(0)], content_dir, chunks_jsonl
    )
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    # Non-empty, derived from the chunk's own text (grounded, not invented).
    assert "behaviorism" in rec["concept_tags"]


def test_emit_honest_empty_when_cited_chunks_yield_no_tags(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    # Cited chunks carry neither concept_tags nor taggable text → honest empty.
    chunks_jsonl = _write_custom_chunks_jsonl(
        tmp_path / "dart_chunks",
        [
            {
                "id": "demo_chunk_00012",
                "text": "the and or of to a in is",
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-12", "role": "primary"}
                    ]
                },
            },
            {
                "id": "demo_chunk_00013",
                "text": "a b c d e f g",
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-13", "role": "contributing"}
                    ]
                },
            },
        ],
    )
    content_dir = tmp_path / "03_content_development"
    out = _emit_block_synthesis_manifest(
        [_untagged_concept_block(0)], content_dir, chunks_jsonl
    )
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    # No fabricated tag when nothing taggable is cited.
    assert rec["concept_tags"] == []


def test_emit_does_not_clobber_block_pinned_concept_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    # The block already pins concept_tags; the cited chunks carry DIFFERENT
    # tags. The block's own tags must win (anti-clobber, additive only).
    chunks_jsonl = _write_custom_chunks_jsonl(
        tmp_path / "dart_chunks",
        [
            {
                "id": "demo_chunk_00012",
                "text": "An RDF graph is a set of triples.",
                "concept_tags": ["some-other-tag"],
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-12", "role": "primary"}
                    ]
                },
            },
            {
                "id": "demo_chunk_00013",
                "text": "A triple has a subject, predicate, and object.",
                "concept_tags": ["another-tag"],
                "source": {
                    "source_references": [
                        {"sourceId": "dart:demo#b-13", "role": "contributing"}
                    ]
                },
            },
        ],
    )
    content_dir = tmp_path / "03_content_development"
    # _concept_block pins concept_tags=["rdf-graph", "triple"].
    out = _emit_block_synthesis_manifest(
        [_concept_block(0)], content_dir, chunks_jsonl
    )
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["concept_tags"] == ["rdf-graph", "triple"]


# --------------------------------------------------------------------------
# Gate-input builder
# --------------------------------------------------------------------------


def test_gate_input_builder_resolves_manifest_and_universe(tmp_path):
    content_dir = tmp_path / "exports" / "PROJ" / "03_content_development"
    content_dir.mkdir(parents=True)
    (content_dir / "week_01.html").write_text("<html></html>", encoding="utf-8")
    chunks_jsonl = _write_chunks_jsonl(tmp_path / "dart_chunks")

    phase_outputs = {
        "objective_extraction": {"project_path": str(content_dir.parent)},
        "chunking": {"dart_chunks_path": str(chunks_jsonl)},
    }
    inputs, missing = _build_manifest_completeness(phase_outputs, {})
    assert missing == []
    assert inputs["manifest_path"].endswith("block_synthesis_manifest.jsonl")
    assert inputs["content_development_dir"] == str(content_dir)
    assert inputs["dart_chunks_manifest_path"].endswith("manifest.json")


def test_gate_input_builder_skips_without_content_dir(tmp_path):
    inputs, missing = _build_manifest_completeness({}, {})
    assert missing == ["content_dir"]
    assert inputs == {}


# --------------------------------------------------------------------------
# End-to-end: emitted manifest PASSES, tampered/missing FAILS
# --------------------------------------------------------------------------


def test_validator_passes_on_emitted_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    chunks_dir = tmp_path / "dart_chunks"
    chunks_jsonl = _write_chunks_jsonl(chunks_dir)
    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir(parents=True)
    (content_dir / "week_04.html").write_text("<html></html>", encoding="utf-8")

    manifest_path = _emit_block_synthesis_manifest(
        [_concept_block(0), _concept_block(1)], content_dir, chunks_jsonl
    )
    assert manifest_path is not None

    result = ManifestCompletenessValidator().validate(
        {
            "manifest_path": str(manifest_path),
            "dart_chunks_manifest_path": str(chunks_dir / "manifest.json"),
            "content_development_dir": str(content_dir),
        }
    )
    assert result.passed is True
    assert result.score == 1.0
    assert [i for i in result.issues if i.severity == "critical"] == []


def test_validator_fails_on_tampered_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    chunks_dir = tmp_path / "dart_chunks"
    chunks_jsonl = _write_chunks_jsonl(chunks_dir)
    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir(parents=True)

    manifest_path = _emit_block_synthesis_manifest(
        [_concept_block(0)], content_dir, chunks_jsonl
    )
    # Tamper: replace a real resolvable chunk id with a non-existent one and
    # drop the second (so source_chunk_ids no longer resolves).
    rec = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    rec["source_chunk_ids"] = ["demo_chunk_DOES_NOT_EXIST"]
    rec.pop("char_spans", None)
    manifest_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    result = ManifestCompletenessValidator().validate(
        {
            "manifest_path": str(manifest_path),
            "dart_chunks_manifest_path": str(chunks_dir / "manifest.json"),
            "content_development_dir": str(content_dir),
        }
    )
    assert result.passed is False
    codes = {i.code for i in result.issues}
    assert "MANIFEST_SOURCE_ID_UNRESOLVED" in codes


def test_validator_fails_when_manifest_missing_but_blocks_exist(tmp_path):
    # Populated content dir, no manifest sidecar → anti-silent-degradation guard.
    chunks_dir = tmp_path / "dart_chunks"
    _write_chunks_jsonl(chunks_dir)
    content_dir = tmp_path / "03_content_development"
    content_dir.mkdir(parents=True)
    (content_dir / "week_04.html").write_text("<html></html>", encoding="utf-8")

    result = ManifestCompletenessValidator().validate(
        {
            "manifest_path": str(
                content_dir / "block_synthesis_manifest.jsonl"
            ),
            "dart_chunks_manifest_path": str(chunks_dir / "manifest.json"),
            "content_development_dir": str(content_dir),
        }
    )
    assert result.passed is False
    assert {i.code for i in result.issues} == {"MANIFEST_FILE_MISSING"}
