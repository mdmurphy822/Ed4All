"""Deterministic builder for the shared retrieval mini-course fixture.

Run this script manually whenever the source HTML pages change; it
regenerates ``semantik_chunks/chunks.jsonl``, ``source/imscc/mini.imscc``, and
keeps the fixture self-consistent (char_spans correct by construction by
re-resolving the body xpath at build time).

    python -m tests.fixtures.retrieval.mini_course.build_mini_course

The fixture is the always-on tier for the citation-anchor contract test and
the WS1 offline-retrieval suite. It contains:

  * source/html/page_alpha.html, page_beta.html — small text-only pages.
  * source/imscc/mini.imscc                     — zip of the same two pages
                                                  + a stub imsmanifest.xml.
  * semantik_chunks/chunks.jsonl                 — 7 v4 chunks; 6 with CORRECT
                                                  char_spans and 1 with a
                                                  deliberately fabricated span
                                                  (id suffix ``_fabricated``).
  * retrieval_eval/gold_set.json                — 3-question mini gold set.

Determinism: chunk ids are positional; char_spans are computed against the
body xpath text the resolver itself reads, so the always-on contract test
pins RESOLVED_EXACT for the honest chunks and SPAN_FABRICATED for the planted
one without any hand-counted offsets.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIXTURE_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.chunker.helpers import extract_plain_text  # noqa: E402
from Trainforge.parsers.xpath_walker import resolve_xpath  # noqa: E402

HTML_DIR = FIXTURE_DIR / "source" / "html"
IMSCC_DIR = FIXTURE_DIR / "source" / "imscc"
SEMANTIK_CHUNKS_DIR = FIXTURE_DIR / "semantik_chunks"
GOLD_DIR = FIXTURE_DIR / "retrieval_eval"

COURSE_ID = "MINI_RETRIEVAL_101"
BODY_XPATH = "/html[1]/body[1]"

# Per-page chunk plans. Each entry names a verbatim substring of the page's
# extract_plain_text() projection; the builder locates it to derive a correct
# char_span against the body-xpath text the resolver reads.
PAGES = [
    {
        "file": "alpha.html",
        "module_id": "alpha",
        "module_title": "Vector Stores and Embeddings",
        "heading": "Vector Stores and Embeddings",
        "chunks": [
            {
                "anchor": "A vector store is a database that indexes high-dimensional embedding vectors so that approximate nearest-neighbour search can retrieve semantically similar passages.",
                "chunk_type": "explanation",
                "concept_tags": ["vector-store", "embeddings"],
                "bloom_level": "understand",
                "difficulty": "foundational",
            },
            {
                "anchor": "FAISS is a widely used open-source library for efficient similarity search over dense vectors.",
                "chunk_type": "explanation",
                "concept_tags": ["faiss", "similarity-search"],
                "bloom_level": "understand",
                "difficulty": "intermediate",
            },
            {
                "anchor": "Retrieval quality is commonly measured with recall at k, which counts how often a relevant passage appears in the top k results returned for a query.",
                "chunk_type": "explanation",
                "concept_tags": ["recall", "retrieval-quality"],
                "bloom_level": "understand",
                "difficulty": "intermediate",
            },
            {
                "anchor": "Reranking re-scores an initial candidate set with a more expensive cross-encoder model.",
                "chunk_type": "explanation",
                "concept_tags": ["reranking", "cross-encoder"],
                "bloom_level": "apply",
                "difficulty": "advanced",
            },
        ],
    },
    {
        "file": "beta.html",
        "module_id": "beta",
        "module_title": "Chunking Strategies",
        "heading": "Chunking Strategies",
        "chunks": [
            {
                "anchor": "Chunking splits a long source document into smaller passages before embedding so that retrieval returns focused evidence rather than whole pages.",
                "chunk_type": "explanation",
                "concept_tags": ["chunking", "passages"],
                "bloom_level": "understand",
                "difficulty": "foundational",
            },
            {
                "anchor": "Sentence overlap prepends the final sentences of the previous chunk to the next one so that a fact stated near a boundary is not split across two passages.",
                "chunk_type": "explanation",
                "concept_tags": ["overlap", "chunking"],
                "bloom_level": "understand",
                "difficulty": "intermediate",
            },
        ],
    },
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_chunk(
    *,
    chunk_id: str,
    item_path: str,
    page: dict,
    body_text: str,
    source_sha: str,
    plan: dict,
    position: int,
    fabricate_span: bool = False,
) -> dict:
    anchor = plan["anchor"]
    start = body_text.find(anchor)
    if start < 0:
        raise SystemExit(
            f"build error: anchor not found in body text for {chunk_id!r}:\n  {anchor!r}"
        )
    end = start + len(anchor)
    if fabricate_span:
        # Mimic the chunker's _locate total-miss arm: a plausible-looking
        # [start, start+len(text)] span that does NOT actually slice to the
        # chunk text. We shift the start forward so the slice lands on
        # unrelated content. The text itself stays a real page substring so
        # containment still succeeds; only the span is a lie.
        bogus_start = start + 37
        span = [bogus_start, bogus_start + len(anchor)]
    else:
        span = [start, end]
    text = anchor
    word_count = len(text.split())
    return {
        "id": chunk_id,
        "schema_version": "v4",
        "chunk_type": plan["chunk_type"],
        "text": text,
        "html": f"<h1>{page['heading']}</h1>",
        "follows_chunk": None,
        "source": {
            "course_id": COURSE_ID,
            "module_id": page["module_id"],
            "module_title": page["module_title"],
            "lesson_id": page["module_id"],
            "lesson_title": page["module_title"],
            "resource_type": "page",
            "section_heading": page["heading"],
            "position_in_module": position,
            "html_xpath": BODY_XPATH,
            "char_span": span,
            "item_path": item_path,
            "source_document_sha256": source_sha,
        },
        "concept_tags": plan["concept_tags"],
        "learning_outcome_refs": [],
        "difficulty": plan["difficulty"],
        "tokens_estimate": int(word_count * 1.3),
        "word_count": word_count,
        "bloom_level": plan["bloom_level"],
    }


def build() -> None:
    SEMANTIK_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    IMSCC_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    chunks: list[dict] = []
    # Aggregate source sha over all pages: chunks share one corpus-level sha,
    # not per-page.
    page_bytes: dict[str, bytes] = {}
    for page in PAGES:
        page_bytes[page["file"]] = (HTML_DIR / page["file"]).read_bytes()
    aggregate_sha = _sha256_bytes(b"".join(page_bytes[p["file"]] for p in PAGES))

    chunk_counter = 0
    for page in PAGES:
        html = page_bytes[page["file"]].decode("utf-8")
        body_text = resolve_xpath(html, BODY_XPATH)
        if body_text is None:
            raise SystemExit(f"build error: body xpath did not resolve for {page['file']}")
        # Sanity: the chunker's plain-text projection must contain each anchor
        # (the contract the resolver's containment metric relies on).
        plain = extract_plain_text(html)
        for position, plan in enumerate(page["chunks"]):
            chunk_counter += 1
            chunk_id = f"mini_{page['module_id']}_chunk_{chunk_counter:03d}"
            if plan["anchor"] not in plain:
                raise SystemExit(
                    f"build error: anchor not in plain text for {chunk_id!r}"
                )
            chunks.append(
                _build_chunk(
                    chunk_id=chunk_id,
                    item_path=f"{page['module_id']}.html",
                    page=page,
                    body_text=body_text,
                    source_sha=aggregate_sha,
                    plan=plan,
                    position=position,
                )
            )

    # Planted fabricated-span chunk on page_alpha (negative-path fixture).
    alpha = PAGES[0]
    alpha_html = page_bytes[alpha["file"]].decode("utf-8")
    alpha_body = resolve_xpath(alpha_html, BODY_XPATH)
    chunk_counter += 1
    fabricated = _build_chunk(
        chunk_id=f"mini_{alpha['module_id']}_chunk_{chunk_counter:03d}_fabricated",
        item_path=f"{alpha['module_id']}.html",
        page=alpha,
        body_text=alpha_body,
        source_sha=aggregate_sha,
        plan={
            "anchor": "FAISS is a widely used open-source library for efficient similarity search over dense vectors.",
            "chunk_type": "explanation",
            "concept_tags": ["faiss"],
            "bloom_level": "understand",
            "difficulty": "intermediate",
        },
        position=4,
        fabricate_span=True,
    )
    chunks.append(fabricated)

    # Write chunks.jsonl (deterministic ordering).
    out = SEMANTIK_CHUNKS_DIR / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")

    # Optional schema validation when jsonschema is present. The chunk_v4
    # schema $refs remote taxonomy URIs (bloom / content-type enums); offline
    # those can't be retrieved, so we validate only the structural shape we
    # can resolve locally and skip cleanly when remote refs are unreachable.
    try:
        import jsonschema  # type: ignore

        schema = json.loads(
            (PROJECT_ROOT / "schemas" / "knowledge" / "chunk_v4.schema.json").read_text()
        )
        validator = jsonschema.Draft7Validator(schema)
        for chunk in chunks:
            try:
                errs = sorted(validator.iter_errors(chunk), key=lambda e: list(e.path))
            except Exception:  # noqa: BLE001 — remote $ref unreachable offline
                print("  (skipped chunk_v4 schema check — remote $ref unreachable offline)")
                break
            if errs:
                raise SystemExit(
                    f"build error: chunk {chunk['id']!r} fails chunk_v4 schema: "
                    f"{errs[0].message}"
                )
    except ImportError:
        pass

    # Build mini.imscc (zip of the two pages + stub imsmanifest.xml).
    manifest_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<manifest identifier="mini-retrieval-101">\n'
        "  <resources>\n"
        '    <resource identifier="r-alpha" type="webcontent" href="alpha.html"/>\n'
        '    <resource identifier="r-beta" type="webcontent" href="beta.html"/>\n'
        "  </resources>\n"
        "</manifest>\n"
    )
    imscc_path = IMSCC_DIR / "mini.imscc"
    with zipfile.ZipFile(imscc_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Deterministic member order; the imscc member item_path is "alpha.html"
        # / "beta.html" (same as the chunk item_path) so the resolver finds
        # members by item_path directly.
        zf.writestr("imsmanifest.xml", manifest_xml)
        for page in PAGES:
            zf.writestr(f"{page['module_id']}.html", page_bytes[page["file"]].decode("utf-8"))

    chunks_sha = _sha256_bytes(out.read_bytes())

    # 3-question mini gold set authored to the gold_set v1 schema shape.
    gold = {
        "schema_version": "1.0",
        "course_slug": "mini-retrieval-101",
        "chunkset": {
            "kind": "semantik",
            "chunks_path": "semantik_chunks/chunks.jsonl",
            "chunks_sha256": chunks_sha,
        },
        "authored_at": "2026-06-09T00:00:00Z",
        "frozen": False,
        "_operator_note": (
            "Fixture seed for retrieval loader + offline tests; not a real "
            "course gold set. Scale-up + freeze is out of fixture scope."
        ),
        "questions": [
            {
                "question_id": "gq-mini-retrieval-101-0001",
                "question_text": "What does a vector store index?",
                "question_type": "factual_recall",
                "relevant_passages": [
                    {
                        "chunk_id": chunks[0]["id"],
                        "relevance": "primary",
                        "anchor": {
                            "item_path": "alpha.html",
                            "section_heading": "Vector Stores and Embeddings",
                            "text_quote": "A vector store is a database that indexes high-dimensional embedding vectors",
                        },
                    }
                ],
                "answer_note": "It indexes high-dimensional embedding vectors for nearest-neighbour search.",
                "authoring": {
                    "method": "manual",
                    "author": "@executor-a",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "seed",
                },
            },
            {
                "question_id": "gq-mini-retrieval-101-0002",
                "question_text": "How is retrieval quality commonly measured?",
                "question_type": "factual_recall",
                "relevant_passages": [
                    {
                        "chunk_id": chunks[2]["id"],
                        "relevance": "primary",
                        "anchor": {
                            "item_path": "alpha.html",
                            "section_heading": "Vector Stores and Embeddings",
                            "text_quote": "Retrieval quality is commonly measured with recall at k",
                        },
                    }
                ],
                "answer_note": "With recall at k.",
                "authoring": {
                    "method": "manual",
                    "author": "@executor-a",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "seed",
                },
            },
            {
                "question_id": "gq-mini-retrieval-101-0003",
                "question_text": "Where does the course cover chunking strategies?",
                "question_type": "where_covered",
                "relevant_passages": [
                    {
                        "chunk_id": chunks[4]["id"],
                        "relevance": "primary",
                        "anchor": {
                            "item_path": "beta.html",
                            "section_heading": "Chunking Strategies",
                            "text_quote": "Chunking splits a long source document into smaller passages before embedding",
                        },
                    },
                    {
                        "chunk_id": chunks[5]["id"],
                        "relevance": "supporting",
                        "anchor": {
                            "item_path": "beta.html",
                            "section_heading": "Chunking Strategies",
                            "text_quote": "Sentence overlap prepends the final sentences of the previous chunk to the next one",
                        },
                    },
                ],
                "answer_note": "In the Chunking Strategies section of the beta page.",
                "authoring": {
                    "method": "manual",
                    "author": "@executor-a",
                    "reviewed_by": "PENDING_REVIEW",
                    "status": "seed",
                },
            },
        ],
    }
    (GOLD_DIR / "gold_set.json").write_text(
        json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {len(chunks)} chunks -> {out}")
    print(f"  chunks_sha256 = {chunks_sha}")
    print(f"wrote imscc -> {imscc_path} ({imscc_path.stat().st_size} bytes)")
    print(f"wrote gold set -> {GOLD_DIR / 'gold_set.json'}")


if __name__ == "__main__":
    build()
