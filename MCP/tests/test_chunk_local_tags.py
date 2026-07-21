"""SemantiK chunking chunk-local concept_tags (FIX #3c).

Page-level ``concept_tags`` were a known defect: every chunk on a page
tagged from the SAME page-level ``item["key_concepts"]`` set, so two
distinct sections of one page collapsed onto a byte-identical tag list
(verified on the real corpus: chunks 00055 and 00058 — different
sections, same page — had identical concept_tags).

The ``TRAINFORGE_CHUNK_LOCAL_TAGS`` flag (default OFF) makes each chunk's
``concept_tags`` derive from ITS OWN text instead of the shared page-level
key-concept set. These tests pin:

* OFF  — two same-page chunks get IDENTICAL tags (legacy parity).
* ON   — two same-page chunks with DIFFERENT text get DIFFERENT tags,
         each reflecting only the concepts present in its own text.
* ON   — deterministic across repeated runs.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools import pipeline_tools  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


@pytest.fixture
def chunking_tool(monkeypatch, tmp_path):
    """Return the chunking registry entry rooted at tmp_path."""
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


# A SINGLE HTML page with TWO sections whose prose triggers DISJOINT
# pedagogy CONCEPT_PATTERNS. Section A is about cognitive load /
# scaffolding; section B is about formative assessment / feedback. Bold
# terms in each section make the page-level key_concepts span BOTH topics
# — the exact substrate that collapses tags under the legacy path.
#
# The chunker's ``merge_small_sections`` greedily merges adjacent
# sections up to MAX_CHUNK_SIZE (800 words), so to keep the two sections
# as two DISTINCT chunks each section must be padded past ~400 words
# (A+B > 800 forces a flush boundary). The padding sentences are
# topic-disjoint filler so the two chunks stay lexically separated.
_PAD_A = (
    " Each design choice here is justified by how it shapes the mental "
    "effort of the learner during practice and review of the material."
) * 22
_PAD_B = (
    " Each grading choice here is justified by how it shapes the "
    "evidence an instructor gathers about learner progress over time."
) * 22
_SECTION_A = (
    "Effective instructional design manages <strong>cognitive load"
    "</strong> so learners are not overwhelmed by extraneous "
    "information. <strong>Scaffolding</strong> breaks complex tasks "
    "into manageable steps, and constructivism frames learning as the "
    "active construction of knowledge by the learner. Intrinsic load "
    "is the inherent difficulty of the material itself, while "
    "extraneous load comes from how the material is presented."
    + _PAD_A
)
_SECTION_B = (
    "A <strong>rubric</strong> makes grading transparent and "
    "consistent across many submissions. Formative assessment gives "
    "learners timely feedback throughout the module so misconceptions "
    "surface early and can be corrected before they harden. Summative "
    "assessment, by contrast, measures achievement at the end of a "
    "unit against the stated outcomes."
    + _PAD_B
)
_TWO_SECTION_PAGE = (
    "<!doctype html><html><head><title>Two Topics</title></head><body>"
    "<h1>Two Topics</h1>"
    "<section><h2>Managing Mental Effort</h2><p>" + _SECTION_A + "</p></section>"
    "<section><h2>Checking Understanding</h2><p>" + _SECTION_B + "</p></section>"
    "</body></html>"
)


def _emit_chunks(tool, course_name: str, staging: Path):
    """Run the chunking tool over a one-page staging dir; return chunks."""
    result = json.loads(asyncio.run(tool(
        course_name=course_name,
        staging_dir=str(staging),
    )))
    assert result.get("success"), f"chunking errored: {result}"
    chunks_path = Path(result["semantik_chunks_path"])
    with chunks_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_page(staging: Path) -> None:
    staging.mkdir(exist_ok=True)
    (staging / "page_01.html").write_text(_TWO_SECTION_PAGE, encoding="utf-8")


def _two_distinct_chunks(chunks):
    """Pick two chunks from the same page whose text differs.

    The page has two sections about different topics; return the first
    chunk whose text mentions cognitive-load prose (section A) and the
    first whose text mentions feedback/rubric prose (section B). The
    section-B chunk deliberately does NOT contain the words "cognitive
    load" — the page-level key-concept ``cognitive load`` leaks onto it
    only under the legacy (flag-OFF) page-level tag path.
    """
    sec_a = next(
        (c for c in chunks if "cognitive load" in c["text"].lower()), None
    )
    sec_b = next(
        (
            c for c in chunks
            if "rubric" in c["text"].lower()
            and "cognitive load" not in c["text"].lower()
        ),
        None,
    )
    return sec_a, sec_b


def test_flag_off_page_level_concept_leaks(chunking_tool, tmp_path, monkeypatch):
    """OFF (legacy): a page-level key concept tags a chunk lacking it.

    ``cognitive load`` is a page-level bold term harvested into
    ``item["key_concepts"]``. Section B's chunk never mentions
    "cognitive load", yet the legacy page-level tag path copies the
    page-level key concepts onto EVERY chunk — so ``cognitive-load``
    leaks onto section B's tags. This pins the defect the flag fixes.
    """
    monkeypatch.delenv("TRAINFORGE_CHUNK_LOCAL_TAGS", raising=False)
    staging = tmp_path / "staging"
    _write_page(staging)

    chunks = _emit_chunks(chunking_tool, "OFF_LEAK", staging)
    sec_a, sec_b = _two_distinct_chunks(chunks)
    assert sec_a is not None and sec_b is not None, (
        f"expected a section-A and a section-B chunk; got "
        f"{[(c['id'], c['text'][:30]) for c in chunks]}"
    )

    # Section B's own text does NOT contain "cognitive load" ...
    assert "cognitive load" not in sec_b["text"].lower()
    # ... yet the legacy page-level path leaks the page-level key concept
    # onto it (the exact bug: identical page-level tags bleed across all
    # chunks of a page regardless of each chunk's own text).
    assert "cognitive-load" in sec_b["concept_tags"], (
        "legacy (flag OFF) path is expected to leak the page-level "
        f"'cognitive-load' key concept onto section B; got "
        f"{sec_b['concept_tags']!r}"
    )


def test_flag_on_chunk_tags_are_local(chunking_tool, tmp_path, monkeypatch):
    """ON: same-page chunks tag from their OWN text → tags diverge.

    The page-level ``cognitive load`` key concept no longer leaks onto
    section B (whose text lacks it), so the two same-page chunks get
    DIFFERENT tag sets, each reflecting only its own text.
    """
    monkeypatch.setenv("TRAINFORGE_CHUNK_LOCAL_TAGS", "true")
    staging = tmp_path / "staging"
    _write_page(staging)

    chunks = _emit_chunks(chunking_tool, "ON_LOCAL", staging)
    sec_a, sec_b = _two_distinct_chunks(chunks)
    assert sec_a is not None and sec_b is not None

    tags_a = set(sec_a["concept_tags"])
    tags_b = set(sec_b["concept_tags"])

    # The two chunks have different text, so their tag sets must differ.
    assert tags_a != tags_b, (
        "flag ON must produce chunk-local tags: same-page chunks with "
        f"different text got identical tags {sorted(tags_a)!r}"
    )

    # The page-level 'cognitive load' key concept is ABSENT from section
    # B's text, so it must NOT appear on section B's tags under the
    # chunk-local path (this is the leak the flag closes).
    assert "cognitive load" not in sec_b["text"].lower()
    assert "cognitive-load" not in tags_b, (
        "a page-level concept absent from a chunk's text must not appear "
        f"on its tags under TRAINFORGE_CHUNK_LOCAL_TAGS; sec_b tags="
        f"{sorted(tags_b)!r}"
    )

    # Section A's text DOES carry cognitive-load prose, so it keeps the
    # tag — the filter is chunk-local, not a blanket drop.
    assert "cognitive load" in sec_a["text"].lower()
    assert "cognitive-load" in tags_a, (
        f"section A's own text carries cognitive-load; tag must survive: "
        f"{sorted(tags_a)!r}"
    )


def test_flag_on_deterministic(chunking_tool, tmp_path, monkeypatch):
    """ON: same input → same tags across two independent runs."""
    monkeypatch.setenv("TRAINFORGE_CHUNK_LOCAL_TAGS", "true")

    staging1 = tmp_path / "s1"
    _write_page(staging1)
    chunks1 = _emit_chunks(chunking_tool, "DET_ONE", staging1)

    staging2 = tmp_path / "s2"
    _write_page(staging2)
    chunks2 = _emit_chunks(chunking_tool, "DET_TWO", staging2)

    tags1 = [c.get("concept_tags") for c in chunks1]
    tags2 = [c.get("concept_tags") for c in chunks2]
    assert tags1 == tags2, (
        f"chunk-local tagging must be deterministic: {tags1!r} != {tags2!r}"
    )


def test_flag_off_byte_identical_to_legacy(chunking_tool, tmp_path, monkeypatch):
    """OFF: emitted tags match the legacy unfiltered page-level extractor.

    Pins byte-stability: with the flag OFF, ``_create_chunk`` must route
    the FULL page-level ``item["key_concepts"]`` through
    ``extract_concept_tags`` exactly as before — no filtering applied.
    """
    from Trainforge.parsers.html_content_parser import HTMLContentParser
    from lib.ontology.concept_tagging import extract_concept_tags

    monkeypatch.delenv("TRAINFORGE_CHUNK_LOCAL_TAGS", raising=False)
    staging = tmp_path / "staging"
    _write_page(staging)

    chunks = _emit_chunks(chunking_tool, "OFF_BYTES", staging)

    # Reconstruct the page-level item the chunker passed to _create_chunk:
    # the parser's whole-page key_concepts feed every chunk under legacy.
    parsed = HTMLContentParser().parse(_TWO_SECTION_PAGE)
    page_item = {"key_concepts": parsed.key_concepts}

    for chunk in chunks:
        legacy = extract_concept_tags(chunk["text"], page_item)
        assert chunk["concept_tags"] == legacy, (
            f"flag OFF must be byte-identical to legacy page-level "
            f"tagging for chunk {chunk['id']!r}: {chunk['concept_tags']!r} "
            f"!= {legacy!r}"
        )


# ---------------------------------------------------------------------------
# FIX #3c (extract, not just filter) — the chunk-local path must EXTRACT
# concepts from each chunk's OWN text, not merely FILTER the page-level
# key_concepts down to those appearing in the text. The filter-only path
# starved content-rich chunks: a chunk whose page-level key_concepts didn't
# surface in its text ended up with one tag or zero, thinning the graph.
#
# The fixture below is a TWO-section page where:
#   * Section A carries a single bold page-level key concept
#     (``cognitive load``) — so the page-level key_concepts set is
#     non-empty (the surviving-filter substrate stays exercised).
#   * Section B carries NO bold terms (parser key_concepts for its prose
#     are empty), but its prose repeats clean multi-word DOMAIN phrases —
#     ``retrieval pipeline``, ``vector store``, ``candidate passages``,
#     ``nearest neighbor`` — none of which appear in the page-level
#     key_concepts. Under filter-only, section B's chunk gets ~0 domain
#     tags; under extract, those phrases are harvested from its own text.
# ---------------------------------------------------------------------------
_EXTRACT_PAD_A = (
    " Each design choice here is justified by how it shapes the mental "
    "effort of the learner during practice and review of the material."
) * 22
_EXTRACT_PAD_B = (
    " Each pipeline choice here is justified by how it shapes the evidence "
    "the retrieval pipeline gathers about each vector store query made."
) * 22
_EXTRACT_SECTION_A = (
    "Effective instructional design manages <strong>cognitive load"
    "</strong> so learners are not overwhelmed by extraneous information. "
    "Intrinsic load is the inherent difficulty of the material itself, "
    "while extraneous load comes from how the material is presented."
    + _EXTRACT_PAD_A
)
# Section B: rich domain prose, NO bold terms. Each multi-word phrase is
# repeated so the lexical deriver's noun-phrase pass surfaces it.
_EXTRACT_SECTION_B = (
    "A retrieval pipeline queries the vector store for candidate passages. "
    "The retrieval pipeline ranks candidate passages by relevance score. "
    "Each retrieval pipeline stage refines the candidate passages further. "
    "A vector store indexes dense embeddings for nearest neighbor lookup. "
    "The vector store returns candidate passages to the retrieval pipeline."
    + _EXTRACT_PAD_B
)
_EXTRACT_PAGE = (
    "<!doctype html><html><head><title>Extract</title></head><body>"
    "<h1>Extract</h1>"
    "<section><h2>Managing Mental Effort</h2><p>"
    + _EXTRACT_SECTION_A
    + "</p></section>"
    "<section><h2>Retrieval Internals</h2><p>"
    + _EXTRACT_SECTION_B
    + "</p></section>"
    "</body></html>"
)


def _write_extract_page(staging: Path) -> None:
    staging.mkdir(exist_ok=True)
    (staging / "page_01.html").write_text(_EXTRACT_PAGE, encoding="utf-8")


def _extract_section_chunks(chunks):
    """Return (section_A_chunk, section_B_chunk) from the extract fixture."""
    sec_a = next(
        (c for c in chunks if "cognitive load" in c["text"].lower()), None
    )
    sec_b = next(
        (
            c for c in chunks
            if "retrieval pipeline" in c["text"].lower()
            and "cognitive load" not in c["text"].lower()
        ),
        None,
    )
    return sec_a, sec_b


def test_flag_on_extracts_chunk_local_domain_tags(
    chunking_tool, tmp_path, monkeypatch
):
    """ON: a content chunk gets tags EXTRACTED from its own text.

    Section B carries no bold page-level key concepts, but its prose
    repeats ``retrieval pipeline`` / ``vector store`` / ``candidate
    passages`` / ``nearest neighbor``. Under the extract path these are
    harvested from the chunk's own text — the chunk is NOT empty and NOT
    limited to a stray page-level tag.
    """
    monkeypatch.setenv("TRAINFORGE_CHUNK_LOCAL_TAGS", "true")
    staging = tmp_path / "staging"
    _write_extract_page(staging)

    chunks = _emit_chunks(chunking_tool, "ON_EXTRACT", staging)
    sec_a, sec_b = _extract_section_chunks(chunks)
    assert sec_a is not None and sec_b is not None, (
        f"expected section-A and section-B chunks; got "
        f"{[(c['id'], c['text'][:30]) for c in chunks]}"
    )

    tags_b = set(sec_b["concept_tags"])
    # The page-level 'cognitive load' concept is absent from B's text, so
    # filtering alone would leave B near-empty. Extraction must surface
    # B's OWN multi-word domain phrases.
    assert "cognitive-load" not in tags_b
    extracted_expected = {
        "retrieval-pipeline",
        "vector-store",
        "candidate-passage",
        "nearest-neighbor",
    }
    assert tags_b & extracted_expected, (
        "chunk-local extraction must surface section B's own domain "
        f"phrases; got {sorted(tags_b)!r}"
    )
    # The chunk is content-rich: extraction yields a substantive tag set,
    # not the one-or-zero tags the filter-only path left it with.
    assert len(tags_b) >= 4, (
        f"content-rich chunk B should carry several extracted tags; got "
        f"{sorted(tags_b)!r}"
    )


def test_flag_on_zero_tag_chunk_now_tagged(
    chunking_tool, tmp_path, monkeypatch
):
    """ON vs filter-only: a chunk that filtering leaves tag-thin gains
    ≥1 EXTRACTED tag from its own text.

    Reconstructs the filter-only result (page-level key_concepts filtered
    to those present in B's text → routed through extract_concept_tags)
    and asserts the live extract path strictly grows it with chunk-local
    domain tags absent from the page-level set.
    """
    from Trainforge.parsers.html_content_parser import HTMLContentParser
    from lib.ontology.concept_tagging import extract_concept_tags

    monkeypatch.setenv("TRAINFORGE_CHUNK_LOCAL_TAGS", "true")
    staging = tmp_path / "staging"
    _write_extract_page(staging)

    chunks = _emit_chunks(chunking_tool, "ON_GROW", staging)
    _, sec_b = _extract_section_chunks(chunks)
    assert sec_b is not None

    parsed = HTMLContentParser().parse(_EXTRACT_PAGE)
    page_concepts = parsed.key_concepts
    text_lower = sec_b["text"].lower()
    # Mirror _filter_item_key_concepts: page concepts present in B's text.
    filtered = [
        c for c in page_concepts
        if all(t in text_lower for t in str(c).lower().split())
    ]
    filter_only = set(
        extract_concept_tags(sec_b["text"], {"key_concepts": filtered})
    )
    live = set(sec_b["concept_tags"])

    # The extract path is a strict superset of the filter-only result ...
    assert filter_only <= live, (
        f"extract path must preserve filter-only tags: missing "
        f"{filter_only - live!r}"
    )
    # ... and adds ≥1 chunk-local extracted tag the filter-only path lacks.
    added = live - filter_only
    assert added, (
        "extract path must add ≥1 chunk-local tag beyond the filter-only "
        f"result; filter_only={sorted(filter_only)!r} live={sorted(live)!r}"
    )


def test_flag_on_extracted_tags_are_clean(
    chunking_tool, tmp_path, monkeypatch
):
    """ON: extracted tags carry no sentence fragments and no scaffolding
    noise (the canonical quality gates are applied to extractions)."""
    from lib.ontology.lexical_concept_seeds import is_fragment_phrase
    from lib.ontology.concept_classifier import is_scaffolding_noise

    monkeypatch.setenv("TRAINFORGE_CHUNK_LOCAL_TAGS", "true")
    staging = tmp_path / "staging"
    _write_extract_page(staging)

    chunks = _emit_chunks(chunking_tool, "ON_CLEAN", staging)

    for chunk in chunks:
        tags = chunk["concept_tags"]
        # Per-chunk cap holds.
        assert len(tags) <= 12, (
            f"chunk {chunk['id']!r} exceeded the 12-tag cap: {tags!r}"
        )
        for tag in tags:
            # No scaffolding noise survives.
            assert not is_scaffolding_noise(tag), (
                f"scaffolding-noise tag leaked: {tag!r} in {chunk['id']!r}"
            )
            # Multi-word tags must not be sentence fragments.
            if "-" in tag:
                assert not is_fragment_phrase(tag.replace("-", " ")), (
                    f"sentence-fragment tag leaked: {tag!r} in "
                    f"{chunk['id']!r}"
                )


def test_flag_on_extract_deterministic(
    chunking_tool, tmp_path, monkeypatch
):
    """ON: the extract path is byte-deterministic across two runs."""
    monkeypatch.setenv("TRAINFORGE_CHUNK_LOCAL_TAGS", "true")

    staging1 = tmp_path / "e1"
    _write_extract_page(staging1)
    chunks1 = _emit_chunks(chunking_tool, "EXT_DET_ONE", staging1)

    staging2 = tmp_path / "e2"
    _write_extract_page(staging2)
    chunks2 = _emit_chunks(chunking_tool, "EXT_DET_TWO", staging2)

    tags1 = [c.get("concept_tags") for c in chunks1]
    tags2 = [c.get("concept_tags") for c in chunks2]
    assert tags1 == tags2, (
        f"extract path must be deterministic: {tags1!r} != {tags2!r}"
    )
