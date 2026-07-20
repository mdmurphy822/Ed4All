"""Stranded next-section heading-tail relocation regression net.

Covers the deterministic, default-off ``TRAINFORGE_RELOCATE_STRANDED_HEADINGS``
post-pass:

* ``resolve_relocate_stranded_headings`` parse-with-fallback (default OFF;
  truthy ``1``/``true``/``yes``/``on``).
* ``relocate_stranded_heading_tails`` — moves a standalone ``N.M
  <SECTION-MARKER>`` glued to a chunk tail onto the following SAME-FLOW
  chunk, with the ``word_count`` / ``tokens_estimate`` recompute, the
  cross-module / no-follower / mid-sentence guards, the
  already-starts-with-marker de-dup, and idempotence.

Fixtures reproduce the shape an OCR'd textbook chunkset produces: a chunk
ending ``"...Figure. 1.1 EXERCISES"`` relocating ``"1.1 EXERCISES"`` to open
the next chunk. All fixtures are inline duck-typed chunk dicts; no models /
GPU / corpus.
"""

import copy

import pytest

from Trainforge.chunker import (
    RELOCATE_STRANDED_HEADINGS_ENV,
    relocate_stranded_heading_tails,
    resolve_relocate_stranded_headings,
)


# ---------------------------------------------------------------------------
# resolve_relocate_stranded_headings — parse-with-fallback (default OFF)
# ---------------------------------------------------------------------------


def test_resolve_default_off_when_unset():
    assert resolve_relocate_stranded_headings(env={}) is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
def test_resolve_truthy_values(raw):
    assert resolve_relocate_stranded_headings(
        env={RELOCATE_STRANDED_HEADINGS_ENV: raw}
    ) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "garbage", "2"])
def test_resolve_falsey_and_garbage_off(raw):
    assert resolve_relocate_stranded_headings(
        env={RELOCATE_STRANDED_HEADINGS_ENV: raw}
    ) is False


def test_resolve_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv(RELOCATE_STRANDED_HEADINGS_ENV, "true")
    assert resolve_relocate_stranded_headings() is True
    monkeypatch.delenv(RELOCATE_STRANDED_HEADINGS_ENV, raising=False)
    assert resolve_relocate_stranded_headings() is False


# ---------------------------------------------------------------------------
# Fixtures mirroring a real OCR'd chunkset pair
# ---------------------------------------------------------------------------


def _audit_pair():
    """A chunk tail glued with the next section's "1.1 EXERCISES" marker."""
    return [
        {
            "id": "course_chunk_00007",
            "text": "See the sieve of Eratosthenes Figure. 1.1 EXERCISES",
            "word_count": 9,
            "tokens_estimate": 11,
            "html": "<p>...unchanged html...</p>",
            "follows_chunk": "course_chunk_00006",
            "source": {"module_id": "ch01_accessible"},
        },
        {
            "id": "course_chunk_00008",
            "text": (
                "In the following exercises, find the place value of each "
                "digit in the given numbers."
            ),
            "word_count": 15,
            "tokens_estimate": 19,
            "html": "<p>...unchanged html...</p>",
            "follows_chunk": "course_chunk_00007",
            "source": {"module_id": "ch01_accessible"},
        },
    ]


# ---------------------------------------------------------------------------
# Core relocation — the stranded-marker case
# ---------------------------------------------------------------------------


def test_stranded_marker_relocated_to_next_chunk():
    chunks = _audit_pair()
    html_before = [c["html"] for c in chunks]
    relocate_stranded_heading_tails(chunks)

    # Marker stripped from the source tail (Figure. retained, no trailing WS).
    assert chunks[0]["text"].endswith("Figure.")
    assert "1.1 EXERCISES" not in chunks[0]["text"]
    # Marker opens the following chunk.
    assert chunks[1]["text"].startswith("1.1 EXERCISES\n")
    assert "find the place value" in chunks[1]["text"]
    # html deliberately untouched on both chunks.
    assert [c["html"] for c in chunks] == html_before
    # id / follows_chunk / source untouched.
    assert chunks[0]["id"] == "course_chunk_00007"
    assert chunks[1]["follows_chunk"] == "course_chunk_00007"


def test_word_count_and_tokens_recomputed_both_chunks():
    chunks = _audit_pair()
    relocate_stranded_heading_tails(chunks)
    for c in chunks:
        assert c["word_count"] == len(c["text"].split())
        assert c["tokens_estimate"] == int(c["word_count"] * 1.3)


def test_newline_preceded_marker_relocated():
    chunks = [
        {
            "id": "c1",
            "text": "Some closing prose about the topic.\n1.3 REVIEW",
            "word_count": 7,
            "tokens_estimate": 9,
            "follows_chunk": None,
            "source": {"module_id": "m1"},
        },
        {
            "id": "c2",
            "text": "Review exercises follow here.",
            "word_count": 4,
            "tokens_estimate": 5,
            "follows_chunk": "c1",
            "source": {"module_id": "m1"},
        },
    ]
    relocate_stranded_heading_tails(chunks)
    assert chunks[0]["text"] == "Some closing prose about the topic."
    assert chunks[1]["text"].startswith("1.3 REVIEW\n")


# ---------------------------------------------------------------------------
# Guards — cases that must NOT relocate
# ---------------------------------------------------------------------------


def test_marker_mid_sentence_not_moved():
    # "1.1 EXERCISES" followed by lowercase continuation -> not a tail marker.
    chunks = [
        {
            "id": "c1",
            "text": "Turn to section 1.1 EXERCISES and complete them all now.",
            "word_count": 9,
            "tokens_estimate": 11,
            "follows_chunk": None,
            "source": {"module_id": "m1"},
        },
        {
            "id": "c2",
            "text": "Next chunk body here.",
            "word_count": 4,
            "tokens_estimate": 5,
            "follows_chunk": "c1",
            "source": {"module_id": "m1"},
        },
    ]
    before = copy.deepcopy(chunks)
    relocate_stranded_heading_tails(chunks)
    assert chunks == before


def test_last_chunk_no_follower_not_moved():
    chunks = [
        {
            "id": "c1",
            "text": "First body chunk.",
            "word_count": 3,
            "tokens_estimate": 3,
            "follows_chunk": None,
            "source": {"module_id": "m1"},
        },
        {
            "id": "c2",
            "text": "End of section prose here. 2.4 SUMMARY",
            "word_count": 7,
            "tokens_estimate": 9,
            "follows_chunk": "c1",
            "source": {"module_id": "m1"},
        },
    ]
    before = copy.deepcopy(chunks)
    relocate_stranded_heading_tails(chunks)
    # Last chunk has no follower -> marker stays put.
    assert chunks == before


def test_cross_module_boundary_not_moved():
    chunks = [
        {
            "id": "c1",
            "text": "Closing prose of module one. 1.9 EXERCISES",
            "word_count": 7,
            "tokens_estimate": 9,
            "follows_chunk": None,  # no linkage to c1's successor
            "source": {"module_id": "module_A"},
        },
        {
            "id": "c2",
            "text": "Opening of a different module.",
            "word_count": 5,
            "tokens_estimate": 6,
            "follows_chunk": None,  # NOT following c1
            "source": {"module_id": "module_B"},  # different module
        },
    ]
    before = copy.deepcopy(chunks)
    relocate_stranded_heading_tails(chunks)
    assert chunks == before


def test_whole_chunk_is_just_marker_not_emptied():
    # A chunk that is ONLY the marker must never be emptied.
    chunks = [
        {
            "id": "c1",
            "text": "1.1 EXERCISES",
            "word_count": 2,
            "tokens_estimate": 3,
            "follows_chunk": None,
            "source": {"module_id": "m1"},
        },
        {
            "id": "c2",
            "text": "Body of the next chunk.",
            "word_count": 5,
            "tokens_estimate": 6,
            "follows_chunk": "c1",
            "source": {"module_id": "m1"},
        },
    ]
    before = copy.deepcopy(chunks)
    relocate_stranded_heading_tails(chunks)
    assert chunks == before


def test_two_caps_not_a_single_word_marker_not_moved():
    # "1.1 KEY CONCEPTS" is two words -> the single-trailing-word regex does
    # not match, so nothing moves (conservative).
    chunks = [
        {
            "id": "c1",
            "text": "Prose ending a section. 1.1 KEY CONCEPTS",
            "word_count": 7,
            "tokens_estimate": 9,
            "follows_chunk": None,
            "source": {"module_id": "m1"},
        },
        {
            "id": "c2",
            "text": "Following body text.",
            "word_count": 3,
            "tokens_estimate": 3,
            "follows_chunk": "c1",
            "source": {"module_id": "m1"},
        },
    ]
    before = copy.deepcopy(chunks)
    relocate_stranded_heading_tails(chunks)
    assert chunks == before


# ---------------------------------------------------------------------------
# De-dup — next chunk already opens with the marker
# ---------------------------------------------------------------------------


def test_next_chunk_already_starts_with_marker_stripped_no_duplicate():
    chunks = [
        {
            "id": "c1",
            "text": "Closing prose of the section. 1.5 EXERCISES",
            "word_count": 7,
            "tokens_estimate": 9,
            "follows_chunk": None,
            "source": {"module_id": "m1"},
        },
        {
            "id": "c2",
            "text": "1.5 EXERCISES In the following problems, solve for x.",
            "word_count": 9,
            "tokens_estimate": 11,
            "follows_chunk": "c1",
            "source": {"module_id": "m1"},
        },
    ]
    relocate_stranded_heading_tails(chunks)
    # Source tail stripped.
    assert chunks[0]["text"] == "Closing prose of the section."
    # Next chunk NOT double-prefixed.
    assert chunks[1]["text"].count("1.5 EXERCISES") == 1
    assert chunks[1]["text"].startswith("1.5 EXERCISES In the following")


# ---------------------------------------------------------------------------
# Idempotence + list-length no-op
# ---------------------------------------------------------------------------


def test_idempotent_second_application_is_noop():
    chunks = _audit_pair()
    relocate_stranded_heading_tails(chunks)
    once = copy.deepcopy(chunks)
    relocate_stranded_heading_tails(chunks)
    assert chunks == once


def test_single_chunk_is_noop():
    chunks = [
        {
            "id": "c1",
            "text": "Only chunk ending with 1.1 EXERCISES",
            "word_count": 6,
            "tokens_estimate": 7,
        }
    ]
    before = copy.deepcopy(chunks)
    out = relocate_stranded_heading_tails(chunks)
    assert out is chunks
    assert chunks == before


def test_module_id_flow_without_follows_linkage():
    # No follows_chunk linkage, but equal module_id -> same flow -> moved.
    chunks = [
        {
            "id": "c1",
            "text": "Section prose ends. 3.2 EXERCISES",
            "word_count": 5,
            "tokens_estimate": 6,
            "source": {"module_id": "m9"},
        },
        {
            "id": "c2",
            "text": "Following prose in the same module.",
            "word_count": 6,
            "tokens_estimate": 7,
            "source": {"module_id": "m9"},
        },
    ]
    relocate_stranded_heading_tails(chunks)
    assert chunks[0]["text"] == "Section prose ends."
    assert chunks[1]["text"].startswith("3.2 EXERCISES\n")
