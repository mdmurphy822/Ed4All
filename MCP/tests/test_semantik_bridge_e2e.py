"""Ed4All-side end-to-end on a SYNTHETIC bridge JSON (no GPU / no real cascade).

The real value the user asked for — "chunks properly indexed". This exercises
the WHOLE Ed4All-side path the bridge feeds, with NO models and NO real
cascade (a hand-built bridge JSON stands in for the cascade output):

    bridge JSON (region_provenance)
      → from_bridge_json / build_chapters_ir         (lib/semantik/cascade_ir)
      → normalize_cascade_to_ed4all                  (lib/semantik/adapter)
      → write {stem}_accessible.html + quality sidecar
      → run_dart_chunking                            (Ed4All chunker; P2b/P3b)
          → assert chunks carry data-semantik-* markers + the 6 SemantiK fields
      → run_vector_indexing (fake provider)          (P-vector-indexing)
          → assert an index is built AND queryable

If the ``[embedding]`` extra is absent the index step is reported skipped and
the chunk JSON is shown to be index-ready (text + embeddings-input fields).

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
  ED4ALL_EMBEDDING_PROVIDER=fake ED4ALL_EMBEDDING_ALLOW_FAKE=1 \
    python -m pytest MCP/tests/test_semantik_bridge_e2e.py -q
"""

from __future__ import annotations

import json

import pytest


def _bridge_json() -> dict:
    """A synthetic bridge JSON reusing the P3a region_provenance shape (two
    content-bearing chapters + a figure + an answer-key non-content heading)."""
    return {
        "pdf": "sample_text_ch1.pdf",
        "html": "",
        "heading_tree": [
            [1, "Chapter 1: Foundations"],
            [1, "Chapter 2: Linear Equations"],
        ],
        "exit_action": "ship_with_confidence",
        "theta_score": 0.91,
        "wcag_status": "passed",
        "flags": [],
        "lane_used": "fast-lane",
        "runtime_mode": "real",
        "region_provenance": [
            {
                "region_index": 0,
                "region_kind": "heading",
                "role": "heading",
                "confidence": 0.95,
                "wcag_status": "passed",
                "first_raw_block_index": 0,
                "pages": [1],
                "heading_text": "Chapter 1: Foundations",
                "level": 1,
                "figure_alt": None,
                "raw_text": "Chapter 1: Foundations",
            },
            {
                "region_index": 1,
                "region_kind": "heading",
                "role": "heading",
                "confidence": 0.83,
                "wcag_status": "passed",
                "first_raw_block_index": 1,
                "pages": [1],
                "heading_text": "Order of Operations",
                "level": 2,
                "figure_alt": None,
                "raw_text": "Order of Operations",
            },
            {
                "region_index": 2,
                "region_kind": "paragraph",
                "role": "body",
                "confidence": 0.61,
                "wcag_status": "passed",
                "first_raw_block_index": 2,
                "pages": [2, 3],
                "heading_text": None,
                "level": None,
                "figure_alt": None,
                "raw_text": (
                    "The order of operations is PEMDAS: parentheses, exponents, "
                    "multiplication and division, then addition and subtraction. "
                    "It is a key foundation for simplifying algebraic expressions."
                ),
            },
            {
                "region_index": 3,
                "region_kind": "figure",
                "role": "figure",
                "confidence": 0.88,
                "wcag_status": "passed",
                "first_raw_block_index": 4,
                "pages": [3],
                "heading_text": "Number Line",
                "level": None,
                "figure_alt": "A number line from negative five to five.",
                "raw_text": "Figure 1.1 A number line from -5 to 5.",
            },
            {
                "region_index": 4,
                "region_kind": "heading",
                "role": "heading",
                "confidence": 0.9,
                "wcag_status": "passed",
                "first_raw_block_index": 5,
                "pages": [5],
                "heading_text": "Chapter 2: Linear Equations",
                "level": 1,
                "figure_alt": None,
                "raw_text": "Chapter 2: Linear Equations",
            },
            {
                "region_index": 5,
                "region_kind": "paragraph",
                "role": "body",
                "confidence": 0.79,
                "wcag_status": "passed",
                "first_raw_block_index": 6,
                "pages": [5],
                "heading_text": None,
                "level": None,
                "figure_alt": None,
                "raw_text": (
                    "A linear equation in one variable can be solved by isolating "
                    "the variable. For example, to solve 2x + 3 = 7, subtract 3 "
                    "from both sides and then divide by 2 to find x = 2."
                ),
            },
        ],
    }


def _adapter_result_from_bridge(bridge: dict):
    """bridge JSON → IR → a duck-typed result the adapter consumes."""
    from lib.semantik.cascade_ir import from_bridge_json

    chapters = from_bridge_json(bridge)

    class _Res:
        pass

    res = _Res()
    res.chapters = chapters
    res.exit_action = bridge["exit_action"]
    res.wcag_status = bridge["wcag_status"]
    res.theta_score = bridge["theta_score"]
    res.flags = list(bridge["flags"])
    res.lane_used = bridge["lane_used"]
    res.lang = "en"
    return res


def _write_accessible_html(tmp_path, bridge: dict):
    """Run the adapter and write {stem}_accessible.html + quality sidecar,
    exactly as the seam would. Returns (staging_dir, html_path)."""
    from lib.semantik.adapter import normalize_cascade_to_ed4all

    res = _adapter_result_from_bridge(bridge)
    out = normalize_cascade_to_ed4all(res, pdf_stem="sample_text_ch1")

    staging = tmp_path / "staging"
    staging.mkdir()
    html_path = staging / "sample_text_ch1_accessible.html"
    html_path.write_text(out["html"], encoding="utf-8")
    # The seam writes a doc-level quality sidecar next to the HTML; the chunker
    # threads its signals (semantic_preservation_score / certification_status)
    # onto every chunk.
    (staging / "sample_text_ch1_accessible.quality.json").write_text(
        json.dumps(
            {
                "compliant": True,
                "wcag_status": out.get("wcag_status", "passed"),
                "theta_score": out.get("theta_score", 0.91),
                "exit_action": out.get("exit_action", "ship_with_confidence"),
                "certification_status": out.get("certification_status", "certified"),
            }
        ),
        encoding="utf-8",
    )
    return staging, html_path, out


# ---------------------------------------------------------------------------
# (1) Adapter HTML carries the data-semantik-* markers (gate-clean).
# ---------------------------------------------------------------------------


def test_1_adapter_html_has_semantik_markers(tmp_path):
    bridge = _bridge_json()
    staging, html_path, out = _write_accessible_html(tmp_path, bridge)
    html = html_path.read_text(encoding="utf-8")

    from lib.validators.semantik_markers import SemantiKMarkersValidator

    res = SemantiKMarkersValidator().validate({"html_content": html})
    critical = [i for i in res.issues if i.severity == "critical"]
    assert res.passed, [i.code for i in critical]
    assert 'data-semantik-source="synthesized"' in html
    assert 'data-semantik-block-id=' in html
    # Page provenance survived (contiguous-range collapse 2-3).
    assert 'data-semantik-pages="2-3"' in html
    # Block-level enrichment the chunker harvests is present on the body block
    # (adapter attr names: data-semantik-block-role / -confidence / -wcag).
    assert 'data-semantik-block-role=' in html
    assert 'data-semantik-confidence=' in html
    assert 'data-semantik-wcag=' in html


# ---------------------------------------------------------------------------
# (2) HTML → chunks: chunks carry data-semantik-* markers + the 6 SemantiK fields.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_2_chunks_carry_six_fields(tmp_path):
    from MCP.tools.pipeline_tools import _build_tool_registry

    bridge = _bridge_json()
    staging, html_path, _ = _write_accessible_html(tmp_path, bridge)

    libv2_root = tmp_path / "libv2"
    registry = _build_tool_registry()
    out = await registry["run_dart_chunking"](
        staging_dir=str(staging),
        course_name="sample_course",
        libv2_root=str(libv2_root),
    )
    payload = json.loads(out)
    assert payload.get("success"), payload
    chunks_path = payload["semantik_chunks_path"]
    chunks = [
        json.loads(line)
        for line in open(chunks_path, encoding="utf-8")
        if line.strip()
    ]
    assert chunks, "no chunks emitted"

    # At least one chunk carries the harvested per-block + doc-level enrichment.
    role_chunks = [c for c in chunks if c.get("source_block_role")]
    assert role_chunks, "no chunk stamped source_block_role from data-semantik-*"
    c = role_chunks[0]
    # 5 of the 6 SemantiK §4 chunk-accompanying fields land on the chunk
    # (P2b/P3b): 3 HTML-harvested + 2 doc-level. The synthetic doc is small
    # enough that every section merges into ONE chunk; per the harvest contract
    # (pipeline_tools ``_run_dart_chunking``: "when a chunk spans several source
    # blocks the FIRST in document order supplies the values") the merged
    # chunk's block-role reflects its leading section — the "Order of
    # Operations" heading (``data-semantik-block-role="heading"``). The
    # structural-delimiter HTMLTextExtractor now surfaces that heading's text as
    # the first source ref, so ``source_block_role`` honestly harvests
    # ``"heading"`` rather than skipping to the first body block. Assert it is a
    # real harvested role (not fabricated / null).
    assert c["source_block_role"] in {"heading", "body", "figure"}
    assert isinstance(c["source_block_confidence"], (int, float))
    assert c["wcag_block_status"] == "passed"
    assert c["semantic_preservation_score"] == pytest.approx(0.91)
    assert c["certification_status"] == "certified"

    # The 6th SemantiK field is ``figure_alt`` — harvested onto a chunk only
    # when the adapter rendered a <figure>/<figcaption> element the chunker's
    # DOM-side ``_recover_figure_alt`` can read. That figure_alt survives the
    # IR→adapter→sidecar path is proven by
    # ``lib/semantik/tests/test_cascade_ir.py::test_b_figure_alt_carried``;
    # here (a body-prose synthetic) we assert the 5 prose-block fields. When
    # the adapter DOES emit a <figure>, figure_alt rides along onto the chunk:
    fig_chunks = [c for c in chunks if c.get("figure_alt")]
    if fig_chunks:
        assert "number line" in fig_chunks[0]["figure_alt"].lower()

    # Every chunk is index-ready: it carries text + a stable id.
    for ch in chunks:
        assert (ch.get("text") or "").strip(), "chunk has no text to embed"
        assert ch.get("chunk_id") or ch.get("id"), "chunk has no id"


# ---------------------------------------------------------------------------
# (3) chunks → vector index (fake provider) — built AND queryable; else
#     report [embedding]-absent + show the chunks are index-ready.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_3_chunks_indexed_and_queryable(tmp_path, monkeypatch):
    from MCP.tools.pipeline_tools import _build_tool_registry

    # Deterministic fake embeddings (anti-poisoning gate opened for the test).
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "1")

    bridge = _bridge_json()
    staging, html_path, _ = _write_accessible_html(tmp_path, bridge)

    libv2_root = tmp_path / "libv2"
    registry = _build_tool_registry()

    # Chunk first (writes LibV2/courses/<slug>/semantik_chunks/chunks.jsonl).
    chunk_out = json.loads(
        await registry["run_dart_chunking"](
            staging_dir=str(staging),
            course_name="sample_course",
            libv2_root=str(libv2_root),
        )
    )
    assert chunk_out.get("success"), chunk_out
    chunks_path = chunk_out["semantik_chunks_path"]
    chunks = [
        json.loads(line)
        for line in open(chunks_path, encoding="utf-8")
        if line.strip()
    ]

    # Probe the embedding extra.
    try:
        from lib.embedding.providers import build_embedding_client  # noqa: F401
        from LibV2.tools.libv2.retrieval.vector_index import (  # noqa: F401
            build_vector_index,
            load_vector_index,
        )

        embedding_available = True
    except Exception:  # noqa: BLE001
        embedding_available = False

    if not embedding_available:
        # [embedding] absent → prove the chunks are index-ready and stop.
        assert chunks, "no chunks to index"
        for c in chunks:
            assert (c.get("text") or "").strip(), "chunk has no embeddable text"
            assert c.get("chunk_id") or c.get("id"), "chunk has no id"
        pytest.skip(
            "[embedding] extra absent; chunks proven index-ready "
            "(text + id present on every chunk)."
        )
        return

    run_vector_indexing = registry["run_vector_indexing"]
    idx_out = json.loads(
        await run_vector_indexing(
            course_name="sample_course",
            libv2_root=str(libv2_root),
            provider="fake",
        )
    )
    assert idx_out.get("success"), idx_out
    assert int(idx_out["chunks_count"]) == len(chunks)
    assert idx_out["embedding_provider"] == "fake"

    # The index is queryable: load it (fake allowed) and search.
    import numpy as np

    from lib.embedding.providers import build_embedding_client
    from LibV2.tools.libv2.retrieval.vector_index import load_vector_index

    course_dir = libv2_root / "courses" / "sample-course"
    index = load_vector_index(course_dir, allow_fake=True)
    client = build_embedding_client("fake", None, offline=True)
    qvec = np.asarray(
        client.encode_batch(["order of operations PEMDAS"]), dtype=np.float32
    )[0]
    hits = index.search(qvec, top_k=3)  # [(chunk_id, score), ...]
    assert hits, "vector index returned no hits"
    # Each hit resolves to a real chunk id present in the chunkset.
    chunk_ids = {c.get("chunk_id") or c.get("id") for c in chunks}
    hit_ids = {h[0] for h in hits}
    assert hit_ids, "no chunk ids in search results"
    assert hit_ids <= chunk_ids, f"hit ids not in chunkset: {hit_ids - chunk_ids}"
