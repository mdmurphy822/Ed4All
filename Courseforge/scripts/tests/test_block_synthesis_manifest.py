"""W3 — Block.to_synthesis_manifest projection + schema + span tests (T1-T6).

Synthetic blocks only — no live LLM, no cross-project fixture pinning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Courseforge.scripts.blocks import Block, Touch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "knowledge" / "block_synthesis_manifest.schema.json"
)


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate(manifest: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(instance=manifest, schema=_load_schema())


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


def _concept_block(**overrides) -> Block:
    content = {
        "source_refs": [
            {"sourceId": "semantik:demo#b-12", "role": "primary"},
            {"sourceId": "semantik:demo#b-13", "role": "contributing"},
        ],
        "key_claims": [
            {
                "claim": "An RDF graph is a set of triples.",
                "source_chunk_ids": ["demo_chunk_00012"],
            }
        ],
        "concept_tags": ["rdf-graph", "triple"],
    }
    content.update(overrides.pop("content_extra", {}))
    defaults = dict(
        block_id="week_04#concept_rdf-graph_0",
        block_type="concept",
        page_id="week_04",
        sequence=7,
        content=content,
        template_type="concept.definition.v1",
        key_terms=("rdf-graph", "triple"),
        touched_by=(_outline_touch(), _rewrite_touch()),
        content_hash="a" * 64,
    )
    defaults.update(overrides)
    return Block(**defaults)


def _resolver(mapping):
    def _r(source_id):
        return mapping.get(source_id)
    return _r


# --------------------------------------------------------------------------
# T1 — projection completeness
# --------------------------------------------------------------------------
def test_t1_projection_completeness():
    block = _concept_block()
    resolver = _resolver({
        "semantik:demo#b-12": {
            "id": "demo_chunk_00012",
            "char_span": [0, 1180],
            "html_xpath": "/html/body/section[2]",
        },
        "semantik:demo#b-13": {"id": "demo_chunk_00013"},
    })
    manifest = block.to_synthesis_manifest(resolver)

    _validate(manifest)

    assert manifest["source_chunk_ids"] == ["demo_chunk_00012", "demo_chunk_00013"]
    assert manifest["template_id"] == "concept.definition.v1"
    assert len(manifest["synthesis"]["tiers"]) == 2
    assert [t["decision_capture_id"] for t in manifest["synthesis"]["tiers"]] == [
        "EVT_0000000000000001",
        "EVT_0000000000000002",
    ]
    # key_claims copied verbatim.
    assert manifest["key_claims"] == [
        {
            "claim": "An RDF graph is a set of triples.",
            "source_chunk_ids": ["demo_chunk_00012"],
        }
    ]
    assert manifest["span_granularity"] == "chunk"
    # char_spans lifted from the resolved chunk's source.char_span.
    spans = {s["chunk_id"]: s for s in manifest["char_spans"]}
    assert spans["demo_chunk_00012"]["char_span"] == [0, 1180]
    assert manifest["concept_tags"] == ["rdf-graph", "triple"]
    assert manifest["content_hash"] == "a" * 64


# --------------------------------------------------------------------------
# T2 — scaffolding block (chrome) emits empty source/concept + no flag
# --------------------------------------------------------------------------
def test_t2_scaffolding_block_empty_allowed():
    block = Block(
        block_id="week_04#chrome_0",
        block_type="chrome",
        page_id="week_04",
        sequence=0,
        content={},
        template_type="chrome.banner.v1",
        touched_by=(_outline_touch(),),
    )
    manifest = block.to_synthesis_manifest(_resolver({}))
    _validate(manifest)
    assert manifest["source_chunk_ids"] == []
    assert manifest["concept_tags"] == []
    assert manifest["block_type"] == "chrome"


# --------------------------------------------------------------------------
# T3 — single-pass / template path (one Touch, provider=deterministic)
# --------------------------------------------------------------------------
def test_t3_single_pass_template_path():
    template_touch = Touch(
        model="template",
        provider="deterministic",
        tier="outline",
        timestamp="2026-06-13T00:00:00Z",
        decision_capture_id="EVT_0000000000000003",
        purpose="template",
    )
    block = _concept_block(touched_by=(template_touch,))
    manifest = block.to_synthesis_manifest(
        _resolver({
            "semantik:demo#b-12": "demo_chunk_00012",
            "semantik:demo#b-13": "demo_chunk_00013",
        })
    )
    _validate(manifest)
    assert len(manifest["synthesis"]["tiers"]) == 1
    assert manifest["synthesis"]["tiers"][0]["provider"] == "deterministic"


# --------------------------------------------------------------------------
# T4 — JSON-LD pointer carries synthesisManifestRef, not the manifest body
# --------------------------------------------------------------------------
def test_t4_jsonld_pointer(monkeypatch):
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "true")
    block = _concept_block(
        block_type="activity",  # routes through _minimal_block_jsonld
        block_id="week_04#activity_0",
    )
    entry = block.to_jsonld_entry()
    assert entry["synthesisManifestRef"] == "week_04#activity_0"
    # The manifest body is NOT inlined into the JSON-LD entry.
    assert "synthesis" not in entry
    assert "source_chunk_ids" not in entry
    assert "concept_tags" not in entry


# --------------------------------------------------------------------------
# T5 — span lift: char_span + html_xpath exactly from the resolved chunk
# --------------------------------------------------------------------------
def test_t5_char_spans_lift():
    block = _concept_block()
    resolver = _resolver({
        "semantik:demo#b-12": {
            "id": "demo_chunk_00012",
            "char_span": [40, 1180],
            "html_xpath": "/html/body/section[2]",
        },
        "semantik:demo#b-13": {"id": "demo_chunk_00013"},
    })
    manifest = block.to_synthesis_manifest(resolver)
    _validate(manifest)
    span = next(
        s for s in manifest["char_spans"] if s["chunk_id"] == "demo_chunk_00012"
    )
    assert span["char_span"] == [40, 1180]
    assert span["html_xpath"] == "/html/body/section[2]"
    assert manifest["span_granularity"] == "chunk"
    # The id-only resolved chunk (b-13) contributes no char_spans entry.
    assert all(s["chunk_id"] != "demo_chunk_00013" for s in manifest["char_spans"])


# --------------------------------------------------------------------------
# T6 — unresolved sourceId kept in source_refs[], absent from source_chunk_ids
# --------------------------------------------------------------------------
def test_t6_unresolved_id_preserved_in_refs_only():
    block = _concept_block()
    resolver = _resolver({
        "semantik:demo#b-12": {"id": "demo_chunk_00012"},
        # semantik:demo#b-13 returns None (unresolved)
    })
    manifest = block.to_synthesis_manifest(resolver)
    _validate(manifest)
    # The unresolved sourceId is still present verbatim in source_refs[].
    ref_ids = {r["sourceId"] for r in manifest["source_refs"]}
    assert "semantik:demo#b-13" in ref_ids
    # but it does NOT appear in source_chunk_ids (it didn't resolve).
    assert "semantik:demo#b-13" not in manifest["source_chunk_ids"]
    assert "demo_chunk_00012" in manifest["source_chunk_ids"]
