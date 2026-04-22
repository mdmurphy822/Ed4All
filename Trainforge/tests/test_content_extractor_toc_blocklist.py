"""Wave 26 — ContentExtractor TOC blocklist tests.

Pre-Wave-26 bug: ``ContentExtractor.extract_key_terms`` greedily matched
"X is a Y" inside chunks containing TOC fragments. On a textbook's
chapter-opener chunk this produced a "key term" like:

    "1.1 Structural changes in the economy 14 1.7 From the periphery..."

Downstream, ``AssessmentGenerator._generate_multiple_choice`` uses
``terms[0].definition`` verbatim as the MCQ correct answer — so every
question on that chunk landed with a raw TOC string as its answer.

Wave 26 fix: reject TOC-fragment candidates before they become
``KeyTerm`` objects. When a chunk yields no accepted candidates, tag it
with ``EMPTY_TERMS_TOC_CHUNK`` so downstream callers can observe the
rejection.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.content_extractor import (  # noqa: E402
    ContentExtractor,
    _is_toc_fragment,
)


def test_is_toc_fragment_detects_three_integers_inline():
    """Three standalone integers in a term (page-number run)."""
    assert _is_toc_fragment("1.1 Structural changes in the economy 14 1.7") is True


def test_is_toc_fragment_detects_dotted_plus_int():
    """Dotted-numeric heading followed by an integer (4.2 + page #)."""
    assert _is_toc_fragment("4.2 Implementation notes 87") is True


def test_is_toc_fragment_rejects_leading_integer():
    """A term starting with a bare integer + punctuation is a TOC
    list-item."""
    assert _is_toc_fragment("42. Introduction") is True


def test_is_toc_fragment_rejects_bare_integer():
    """Just a number."""
    assert _is_toc_fragment("42") is True


def test_is_toc_fragment_rejects_chapter_prefix():
    """'Chapter 3', 'Section 4', etc. — TOC title prefix + number."""
    assert _is_toc_fragment("Chapter 3 Introduction") is True
    assert _is_toc_fragment("Section 4 — Advanced") is True
    assert _is_toc_fragment("Appendix 1 Reference") is True


def test_is_toc_fragment_rejects_long_run_on_term():
    """Terms over 200 chars are prose masquerading as a term."""
    long_term = "a" * 201
    assert _is_toc_fragment(long_term) is True


def test_is_toc_fragment_accepts_real_terms():
    """Genuine terminology passes the blocklist."""
    assert _is_toc_fragment("photosynthesis") is False
    assert _is_toc_fragment("mitochondrion") is False
    assert _is_toc_fragment("aerobic respiration") is False
    assert _is_toc_fragment("DNA") is False
    # A term with one number inline is fine (e.g. a protein name).
    assert _is_toc_fragment("p53 tumor suppressor protein") is False


def test_extract_key_terms_rejects_toc_chunk_and_tags_diagnostic():
    """A chunk whose "definition" matches is 'X is a Y' where X itself
    is a TOC fragment must be rejected AND tagged with
    EMPTY_TERMS_TOC_CHUNK."""
    # Build a chunk with TOC text. The DEFINITION_PATTERNS include
    # "X is a Y" — so "1.1 Structural changes ... is the first chapter"
    # would match, with group(1) being the TOC fragment.
    toc_chunk = {
        "id": "c1",
        "text": (
            "1.1 Structural changes in the economy 14 1.7 From the "
            "periphery 22 is the opening chapter of the textbook. "
            "Chapter 2 Photosynthesis 45 is the second chapter."
        ),
    }
    extractor = ContentExtractor()
    terms = extractor.extract_key_terms([toc_chunk])

    # TOC-fragment terms must NOT appear.
    for t in terms:
        assert "Structural changes" not in t.term or not t.term.startswith("1.1"), (
            f"TOC fragment leaked as key_term: {t.term!r}"
        )
        assert not t.term.strip().startswith(("1.1", "1.7", "2 ", "Chapter 2")), (
            f"TOC fragment leaked: {t.term!r}"
        )

    # Diagnostic must be set on the chunk (all candidates rejected).
    diagnostics = toc_chunk.get("metadata_diagnostics", [])
    assert "EMPTY_TERMS_TOC_CHUNK" in diagnostics, (
        f"Expected EMPTY_TERMS_TOC_CHUNK diagnostic; got {diagnostics}"
    )


def test_extract_key_terms_preserves_legitimate_terms():
    """A chunk with real key terms (bold + definition pattern) must yield
    those terms unmodified and must NOT carry the diagnostic flag."""
    chunk = {
        "id": "c2",
        "text": (
            "Photosynthesis is the biological process by which plants "
            "convert light energy to chemical energy. Chloroplasts are "
            "the organelles where photosynthesis occurs."
        ),
    }
    extractor = ContentExtractor()
    terms = extractor.extract_key_terms([chunk])
    # Expect at least one term to be extracted.
    assert len(terms) >= 1, f"No terms extracted from clean chunk: {terms}"
    # No diagnostic on a clean chunk.
    assert "EMPTY_TERMS_TOC_CHUNK" not in chunk.get("metadata_diagnostics", [])


def test_extract_key_terms_mixed_toc_and_real_preserves_real():
    """Mixed chunk with both TOC lines and real definitions: real terms
    are preserved, TOC rejected, diagnostic NOT set (some candidates
    accepted)."""
    chunk = {
        "id": "c3",
        "text": (
            "Photosynthesis is the biological process by which plants "
            "produce glucose and oxygen from carbon dioxide and water. "
            "1.1 Structural changes in the economy 14 1.7 From the "
            "periphery 22 is an unrelated textbook chapter heading."
        ),
    }
    extractor = ContentExtractor()
    terms = extractor.extract_key_terms([chunk])
    # At least the Photosynthesis term should survive.
    term_names = [t.term.lower() for t in terms]
    assert any("photosynthesis" in n for n in term_names), (
        f"Real term dropped: extracted {term_names}"
    )
    # None of the surviving terms is a TOC fragment.
    for t in terms:
        assert not _is_toc_fragment(t.term), (
            f"TOC fragment survived: {t.term!r}"
        )
    # Because at least one candidate was accepted, no diagnostic.
    assert "EMPTY_TERMS_TOC_CHUNK" not in chunk.get("metadata_diagnostics", [])
