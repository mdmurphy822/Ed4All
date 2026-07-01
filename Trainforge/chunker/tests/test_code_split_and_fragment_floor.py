"""W1b.2 / W1b.3 — code-listing split + sub-N-word fragment floor tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from Trainforge.chunker.chunker import (
    MergedSectionResult,
    fold_fragment_results,
    looks_like_code_listing,
    merge_small_sections,
    resolve_chunk_code_split,
    resolve_merge_fragment_floor,
    split_by_sentences,
    split_code_by_lines,
)


@dataclass
class _Section:
    heading: str
    content: str
    word_count: int
    source_references: Optional[list] = None
    template_type: Optional[str] = None


# --------------------------------------------------------------------------
# W1b.2 — code-listing detection + line split
# --------------------------------------------------------------------------

_CODE = """def build_index(chunks):
    vectors = []
    for chunk in chunks:
        vec = model.encode(chunk["text"])
        vectors.append(vec)
    return numpy.stack(vectors)

class Retriever:
    def __init__(self, index):
        self.index = index
"""

_PROSE = (
    "This chapter introduces retrieval augmented generation. It explains how a "
    "vector index is built and why chunking matters. The discussion continues "
    "across several paragraphs of ordinary explanatory prose without any code."
)


def test_resolve_code_split_default_off():
    assert resolve_chunk_code_split({}) is False
    assert resolve_chunk_code_split({"ED4ALL_CHUNK_CODE_SPLIT": "true"}) is True
    assert resolve_chunk_code_split({"ED4ALL_CHUNK_CODE_SPLIT": "junk"}) is False


def test_looks_like_code_listing_discriminates():
    assert looks_like_code_listing(_CODE) is True
    assert looks_like_code_listing(_PROSE) is False
    assert looks_like_code_listing("") is False


def test_split_code_by_lines_respects_budget_and_is_contiguous():
    pieces = split_code_by_lines(_CODE, target_words=4)
    assert len(pieces) >= 2
    # Every piece is a contiguous run of original lines (anti-fabrication):
    # the concatenation of pieces reproduces the non-empty source lines.
    src_lines = [ln for ln in _CODE.splitlines()]
    got = "\n".join(pieces).splitlines()
    assert [ln for ln in got if ln] == [ln for ln in src_lines if ln]


def test_split_by_sentences_leaves_code_as_one_piece():
    # Motivation for W1b.2: code has no sentence punctuation, so the legacy
    # splitter returns a single over-window piece.
    assert len(split_by_sentences(_CODE, target_words=4)) == 1


# --------------------------------------------------------------------------
# W1b.3 — fragment floor
# --------------------------------------------------------------------------


def test_resolve_fragment_floor_default_zero():
    assert resolve_merge_fragment_floor({}) == 0
    assert resolve_merge_fragment_floor({"ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR": "5"}) == 5
    assert resolve_merge_fragment_floor({"ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR": "-2"}) == 0
    assert resolve_merge_fragment_floor({"ED4ALL_CHUNK_MERGE_FRAGMENT_FLOOR": "x"}) == 0


def _mk(text: str, heading: str = "H") -> MergedSectionResult:
    return MergedSectionResult(
        heading=heading,
        combined_text=text,
        chunk_type="explanation",
        merged_source_ids=[],
        merged_headings=[heading],
    )


def test_fold_fragment_folds_runt_into_prior():
    results = [_mk("one two three four five six", "A"), _mk("tiny frag", "B")]
    folded = fold_fragment_results(results, fragment_floor=5)
    assert len(folded) == 1
    assert "tiny frag" in folded[0].combined_text
    assert "B" in folded[0].merged_headings


def test_fold_fragment_drops_leading_fragment():
    results = [_mk("two words"), _mk("plenty of real content here now indeed")]
    folded = fold_fragment_results(results, fragment_floor=5)
    assert len(folded) == 1
    assert folded[0].combined_text.startswith("plenty")


def test_fold_floor_zero_is_identity():
    results = [_mk("two words"), _mk("also short")]
    assert fold_fragment_results(results, fragment_floor=0) is results


def test_merge_small_sections_floor_off_byte_identical():
    sections = [
        _Section("Intro", "one two three four five six seven eight", 8),
        _Section("Frag", "tiny bit", 2),
    ]
    default = merge_small_sections(sections, max_chunk_size=5)
    # Two results survive with the floor OFF (legacy behavior).
    assert len(default) == 2
    floored = merge_small_sections(sections, max_chunk_size=5, fragment_floor=5)
    assert len(floored) == 1
