"""U3 regression: the canonical chunker threads harvested ``data-cf-curie``
tokens through ``chunk_content`` -> ``chunk_text_block`` ->
``CourseProcessor._create_chunk`` so emitted chunks carry the optional
``curies`` / ``forced_curies`` fields.

Contract under test:
  - A force-injected ``<span hidden data-cf-curie="ns:concept"
    data-cf-curie-forced="true">`` keeps the 376b64f skip invariant
    (synthetic token absent from chunk ``text``) AND its token lands in
    BOTH ``curies`` and ``forced_curies``.
  - An UNFORCED ``data-cf-curie`` span -> token in ``curies`` only;
    ``forced_curies`` is absent.
  - A real CURIE in plain prose (no span) -> token in ``curies``;
    ``forced_curies`` absent.
  - A chunk with no CURIEs at all -> NEITHER ``curies`` NOR
    ``forced_curies`` key (byte-identical legacy emit).

The chunker is driven end-to-end through ``CourseProcessor._chunk_content``
so the U3 threading (helper -> chunk_content -> chunk_text_block ->
_dispatch_create_chunk additive-kwarg fallback -> _create_chunk stamp)
is exercised as a whole. The minimal-processor construction mirrors
``Trainforge/tests/test_chunk_source_document_sha.py``.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_minimal_processor() -> Any:
    """Bare ``CourseProcessor`` bypassing __init__.

    Mirrors test_chunk_source_document_sha.py: sets only the fields the
    chunker call path reads + stubs the metadata extractors that depend
    on full __init__ state. ``_extract_concept_tags`` returns ``[]`` so
    prose-derived CURIEs do NOT sneak in via concept tags — the
    ``curies`` field must come purely from the U3 plumbing.
    """
    from Trainforge.process_course import CourseProcessor

    cp = CourseProcessor.__new__(CourseProcessor)
    cp.course_code = "TEST_101"
    cp.capture = None
    cp.stats = {
        "total_words": 0,
        "total_tokens_estimate": 0,
        "chunk_types": collections.defaultdict(int),
        "difficulty_distribution": collections.defaultdict(int),
    }
    cp._all_concept_tags = set()
    cp._boilerplate_spans = []
    cp._extract_concept_tags = lambda t, i: []  # type: ignore[method-assign]
    cp._determine_difficulty = lambda t, i: "foundational"  # type: ignore[method-assign]
    cp._extract_objective_refs = (
        lambda i, section_heading=None: []
    )  # type: ignore[method-assign]
    cp._extract_section_metadata = lambda i, h, **kw: (
        None,
        None,
        [],
        {"key_claims": [], "objective_alignment": []},
        {
            "content_type_label": "none",
            "key_terms": "none",
            "key_claims": "none",
            "objective_alignment": "none",
        },
    )  # type: ignore[method-assign]
    cp._resolve_chunk_source_references = (
        lambda *, item, section_heading, section_source_ids, merged_headings=None: []
    )  # type: ignore[method-assign]
    return cp


def _item(raw_html: str) -> Dict[str, Any]:
    """A no-sections parsed item — drives the chunk_content no-sections
    branch, the simplest deterministic single-chunk path."""
    return {
        "module_id": "week_01_intro",
        "module_title": "Week 1 -- Intro",
        "item_id": "intro",
        "title": "Intro",
        "resource_type": "webcontent",
        "item_path": "week_01/intro.html",
        "raw_html": raw_html,
        "sections": [],
        "learning_objectives": [],
    }


def _chunk_one(raw_html: str) -> Dict[str, Any]:
    """Run the chunker over one no-sections item; return the sole chunk."""
    cp = _build_minimal_processor()
    chunks: List[Dict[str, Any]] = cp._chunk_content([_item(raw_html)])
    assert len(chunks) == 1, f"expected exactly one chunk, got {len(chunks)}"
    return chunks[0]


# ---------------------------------------------------------------------------
# Test 1: force-injected (forced) data-cf-curie span
# ---------------------------------------------------------------------------


def test_force_injected_curie_in_curies_and_forced_curies() -> None:
    """A force-injected hidden span carrying a CURIE token: the synthetic
    token stays out of ``text`` (376b64f) but appears in BOTH ``curies``
    and ``forced_curies``."""
    body = " ".join(f"word{n}" for n in range(60))
    raw_html = (
        "<html><body>"
        f"<p>Property shapes constrain values. {body}</p>"
        '<span hidden data-cf-curie="ns:concept" '
        'data-cf-curie-forced="true">ns:concept</span>'
        "</body></html>"
    )
    chunk = _chunk_one(raw_html)

    # 376b64f invariant: synthetic token absent from chunk text.
    assert "ns:concept" not in chunk["text"]
    assert "Property shapes constrain values" in chunk["text"]

    # U3: harvested into curies + flagged forced.
    assert "ns:concept" in chunk["curies"]
    assert "ns:concept" in chunk["forced_curies"]
    # Subset invariant.
    assert set(chunk["forced_curies"]).issubset(set(chunk["curies"]))


# ---------------------------------------------------------------------------
# Test 2: unforced data-cf-curie span
# ---------------------------------------------------------------------------


def test_unforced_curie_span_in_curies_not_forced() -> None:
    """An UNFORCED ``data-cf-curie`` span -> token in ``curies`` only;
    ``forced_curies`` key absent (nothing forced)."""
    body = " ".join(f"word{n}" for n in range(60))
    raw_html = (
        "<html><body>"
        f"<p>A node shape targets a class. {body}</p>"
        '<span hidden data-cf-curie="ns:unforced">ns:unforced</span>'
        "</body></html>"
    )
    chunk = _chunk_one(raw_html)

    assert "ns:unforced" not in chunk["text"]
    assert "ns:unforced" in chunk["curies"]
    # No forced marker -> the key is omitted entirely (conditional stamp).
    assert "forced_curies" not in chunk


# ---------------------------------------------------------------------------
# Test 3: real CURIE in plain prose (no span)
# ---------------------------------------------------------------------------


def test_prose_curie_in_curies_no_forced() -> None:
    """A genuine CURIE in plain prose (RDF/SHACL corpora, no
    data-cf-curie span) -> ``curies`` includes it; ``forced_curies``
    absent."""
    body = " ".join(f"word{n}" for n in range(60))
    raw_html = (
        "<html><body>"
        "<p>The rdfs:subClassOf predicate relates a class to a "
        f"superclass. {body}</p>"
        "</body></html>"
    )
    chunk = _chunk_one(raw_html)

    # Prose CURIE is content — it stays IN the chunk text.
    assert "rdfs:subClassOf" in chunk["text"]
    # ... and is folded into curies via the regex extractor.
    assert "rdfs:subClassOf" in chunk["curies"]
    # Nothing force-injected.
    assert "forced_curies" not in chunk


# ---------------------------------------------------------------------------
# Test 4: no CURIEs at all -> byte-identical legacy emit
# ---------------------------------------------------------------------------


def test_no_curies_emits_neither_key() -> None:
    """A chunk whose source HTML had no data-cf-curie span AND no CURIE
    in prose emits NEITHER ``curies`` NOR ``forced_curies`` — legacy /
    non-RDF corpora stay byte-identical."""
    body = " ".join(f"word{n}" for n in range(60))
    raw_html = (
        "<html><body>"
        f"<p>This passage contains ordinary prose with no identifiers. "
        f"{body}</p>"
        "</body></html>"
    )
    chunk = _chunk_one(raw_html)

    assert "curies" not in chunk
    assert "forced_curies" not in chunk
