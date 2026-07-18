"""Tests for ``lib.validators.feature_cache.BlockFeatureCache``.

Covers, stub-only (no DeBERTa / SentenceTransformer load required):

* flag resolver parse-with-fallback + default OFF,
* corpus-layer memoization + parity vs the direct builder functions,
* per-block content-sha keying (a re-rolled block self-invalidates),
* the three verdict-bearing splitters kept DISTINCT by splitter id,
* resolved_passages parity vs ``_resolve_block_cited_passages``,
* embedding batching (misses coalesced into ONE ``encode_batch`` call),
* compute-once-under-N-threads (thread-safety smoke),
* router-level builder parity: cache-on inputs == cache-off inputs.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.feature_cache import (  # noqa: E402
    ENV_FEATURE_CACHE,
    SPLITTER_CLAIM_SUPPORT,
    SPLITTER_REWRITE_GROUNDING,
    BlockFeatureCache,
    resolve_feature_cache_enabled,
)


def _block(
    *,
    block_id: str = "page_01#concept_intro_0",
    block_type: str = "concept",
    content: str = "<p>placeholder.</p>",
    source_ids: Tuple[str, ...] = ("dart:slug#blk_0",),
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id="page_01",
        sequence=0,
        content=content,
        source_ids=source_ids,
        source_references=tuple({"sourceId": s} for s in source_ids),
    )


# --------------------------------------------------------------------- #
# Flag resolver
# --------------------------------------------------------------------- #


def test_flag_resolver_parse_with_fallback() -> None:
    assert resolve_feature_cache_enabled({}) is False
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        assert resolve_feature_cache_enabled({ENV_FEATURE_CACHE: truthy}) is True
    for off in ("0", "false", "no", "off", "garbage", ""):
        assert resolve_feature_cache_enabled({ENV_FEATURE_CACHE: off}) is False


# --------------------------------------------------------------------- #
# Corpus layer — blocks
# --------------------------------------------------------------------- #


def test_blocks_memoized_and_matches_hydration(tmp_path: Path) -> None:
    from MCP.hardening.gate_input_routing import _hydrate_blocks_from_path

    p = tmp_path / "blocks_final.jsonl"
    rows = [
        {
            "block_id": "page_01#concept_a_0",
            "block_type": "concept",
            "page_id": "page_01",
            "sequence": 0,
            "content": "<p>Alpha body.</p>",
            "source_ids": ["dart:slug#blk_0"],
        },
        {
            "block_id": "page_01#example_b_1",
            "block_type": "example",
            "page_id": "page_01",
            "sequence": 1,
            "content": "<p>Beta body.</p>",
            "source_ids": ["dart:slug#blk_1"],
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cache = BlockFeatureCache({}, {})
    got = cache.blocks(p)
    direct = _hydrate_blocks_from_path(p)
    assert [b.block_id for b in got] == [b.block_id for b in direct]
    # Memoized: same list object on the second call.
    assert cache.blocks(p) is got


# --------------------------------------------------------------------- #
# Corpus layer — source_chunks returns a COPY (builder-safe)
# --------------------------------------------------------------------- #


def test_source_chunks_returns_copy(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "id": "c_0",
                "text": "Body zero.",
                "source": {"source_references": [{"sourceId": "dart:slug#blk_0"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    po = {"chunking": {"semantik_chunks_path": str(chunks)}}
    cache = BlockFeatureCache(po, {})
    a = cache.source_chunks()
    assert a.get("dart:slug#blk_0") == "Body zero."
    assert a.get("c_0") == "Body zero."
    # Mutating the returned dict must not corrupt the memo.
    a["injected"] = "x"
    b = cache.source_chunks()
    assert "injected" not in b


# --------------------------------------------------------------------- #
# Per-block layer — stripped text + content-sha invalidation
# --------------------------------------------------------------------- #


def test_stripped_text_matches_strip_and_memoizes() -> None:
    from lib.utils import strip_html_to_text

    cache = BlockFeatureCache({}, {})
    blk = _block(content="<p>Hello <b>world</b>.</p>")
    got = cache.stripped_text(blk)
    assert got == strip_html_to_text(blk.content)
    assert cache.stripped_text(blk) == got


def test_content_sha_keying_reroll_self_invalidates() -> None:
    cache = BlockFeatureCache({}, {})
    a = _block(block_id="pid#b_0", content="<p>Original body.</p>")
    b = _block(block_id="pid#b_0", content="<p>Re-rolled body entirely.</p>")
    ta = cache.stripped_text(a)
    tb = cache.stripped_text(b)
    assert ta != tb
    assert "Original" in ta and "Re-rolled" in tb
    # Distinct keys for distinct content under the same block_id.
    assert cache.block_key(a) != cache.block_key(b)


# --------------------------------------------------------------------- #
# Per-block layer — splitters kept DISTINCT by id (verdict-bearing)
# --------------------------------------------------------------------- #


def test_sentences_by_splitter_id_are_distinct() -> None:
    from lib.retrieval.groundedness import _split_claims_for_scoring
    from lib.validators.rewrite_source_grounding import (
        _is_non_trivial,
        _segment_sentences,
    )

    content = (
        "<p>The mitochondrion is the powerhouse of the cell and produces ATP. "
        "Short one. It also drives oxidative phosphorylation across the inner "
        "membrane continuously.</p>"
    )
    cache = BlockFeatureCache({}, {})
    blk = _block(content=content)
    text = cache.stripped_text(blk)

    cs = cache.sentences(blk, SPLITTER_CLAIM_SUPPORT)
    rg = cache.sentences(blk, SPLITTER_REWRITE_GROUNDING)

    assert cs == _split_claims_for_scoring(text)[0]
    assert rg == [s for s in _segment_sentences(text) if _is_non_trivial(s)]
    # Keyed by splitter id → independently memoized, never unified.
    assert cache.sentences(blk, SPLITTER_CLAIM_SUPPORT) is cs

    with pytest.raises(ValueError):
        cache.sentences(blk, "not-a-real-splitter")


def test_resolved_passages_matches_direct() -> None:
    from lib.validators.block_prose_entailment import _resolve_block_cited_passages

    cache = BlockFeatureCache({}, {})
    blk = _block(source_ids=("dart:slug#blk_0",))
    source_chunks = {"dart:slug#blk_0": "The cited chunk body."}
    got = cache.resolved_passages(blk, source_chunks)
    assert got == _resolve_block_cited_passages(blk, source_chunks)
    assert got == [{"chunk_id": "dart:slug#blk_0", "text": "The cited chunk body."}]


# --------------------------------------------------------------------- #
# Embedding tier — misses coalesced into ONE encode_batch call
# --------------------------------------------------------------------- #


class _FakeEmbedder:
    def __init__(self) -> None:
        self.batch_calls: List[List[str]] = []

    def encode_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        self.batch_calls.append(list(texts))
        # Deterministic 2-d "vector" per text (length-based, distinct).
        return [[float(len(t)), float(sum(map(ord, t)) % 97)] for t in texts]


def test_embed_batches_misses_and_memoizes() -> None:
    emb = _FakeEmbedder()
    cache = BlockFeatureCache({}, {}, embedder=emb)
    out = cache.embed(["alpha", "beta", "alpha"])
    # De-duplicated: exactly two distinct texts encoded in ONE batch call.
    assert len(emb.batch_calls) == 1
    assert emb.batch_calls[0] == ["alpha", "beta"]
    assert len(out) == 2
    # Second call for an already-seen text → no new encode.
    cache.embed(["alpha"])
    assert len(emb.batch_calls) == 1
    # embed_one returns the same vector.
    v = cache.embed_one("beta")
    assert v == [4.0, float(sum(map(ord, "beta")) % 97)]


def test_embed_no_embedder_degrades_empty() -> None:
    cache = BlockFeatureCache({}, {}, embedder=None)
    # embedder=None passed explicitly + probed=True short-circuits the loader.
    cache._embedder_probed = True  # simulate "no embedder available"
    assert cache.embed(["x"]) == {}
    assert cache.embed_one("x") is None


# --------------------------------------------------------------------- #
# Thread-safety smoke — compute runs exactly once under N threads
# --------------------------------------------------------------------- #


def test_memodict_computes_once_under_threads() -> None:
    from lib.validators.feature_cache import _MemoDict

    memo = _MemoDict()
    calls = {"n": 0}
    lock = threading.Lock()
    start = threading.Event()

    def _compute() -> str:
        with lock:
            calls["n"] += 1
        # Small spin so late threads pile up on the in-flight Future.
        import time

        time.sleep(0.02)
        return "value"

    results: List[str] = []
    rlock = threading.Lock()

    def _worker() -> None:
        start.wait()
        v = memo.get_or_compute("k", _compute)
        with rlock:
            results.append(v)

    threads = [threading.Thread(target=_worker) for _ in range(24)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert calls["n"] == 1, "compute ran more than once under concurrency"
    assert results == ["value"] * 24


def test_stripped_text_thread_safe_consistent() -> None:
    cache = BlockFeatureCache({}, {})
    blk = _block(content="<p>Concurrent strip body text here.</p>")
    out: List[str] = []
    olock = threading.Lock()

    def _worker() -> None:
        v = cache.stripped_text(blk)
        with olock:
            out.append(v)

    threads = [threading.Thread(target=_worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert len(set(out)) == 1
    assert out[0].startswith("Concurrent strip")


# --------------------------------------------------------------------- #
# Router-level parity: cache-on inputs == cache-off inputs
# --------------------------------------------------------------------- #


def _write_corpus(tmp_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    blocks_p = tmp_path / "blocks_final.jsonl"
    blocks_p.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "block_id": "page_01#concept_a_0",
                    "block_type": "concept",
                    "page_id": "page_01",
                    "sequence": 0,
                    "content": "<p>Alpha grounded body.</p>",
                    "source_ids": ["dart:slug#blk_0"],
                    "source_references": [{"sourceId": "dart:slug#blk_0"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    chunks_p = tmp_path / "chunks.jsonl"
    chunks_p.write_text(
        json.dumps(
            {
                "id": "c_0",
                "text": "Alpha chunk premise body.",
                "source": {"source_references": [{"sourceId": "dart:slug#blk_0"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    phase_outputs = {
        "content_generation_rewrite": {"blocks_final_path": str(blocks_p)},
        "chunking": {"semantik_chunks_path": str(chunks_p)},
    }
    return phase_outputs, {}


def test_router_rewrite_builder_parity(tmp_path: Path) -> None:
    from MCP.hardening.gate_input_routing import _build_rewrite_block_input

    phase_outputs, wp = _write_corpus(tmp_path)

    off_inputs, off_missing = _build_rewrite_block_input(phase_outputs, wp)
    cache = BlockFeatureCache(phase_outputs, wp)
    on_inputs, on_missing = _build_rewrite_block_input(phase_outputs, wp, cache=cache)

    assert off_missing == on_missing == []
    assert [b.block_id for b in on_inputs["blocks"]] == [
        b.block_id for b in off_inputs["blocks"]
    ]
    assert on_inputs.get("source_chunks") == off_inputs.get("source_chunks")
    assert on_inputs.get("chunk_provenance_index") == off_inputs.get(
        "chunk_provenance_index"
    )
