"""Track K — verbatim chunk-overlap regression net.

Covers the deterministic, default-off ``TRAINFORGE_CHUNK_OVERLAP_WORDS``
chunk-overlap option:

* ``resolve_chunk_overlap_words`` parse-with-fallback (default 0 →
  byte-identical legacy emit; garbage / non-positive → 0).
* ``apply_chunk_overlap`` — verbatim trailing-word prepend with the
  anti-fabrication contract (prefix words are a subset of the prior
  chunk's own words, never synthesized, never compounding), the
  ``follows_chunk`` continuity guard (no cross-lesson bleed), and the
  ``word_count`` / ``tokens_estimate`` recompute.
* The chunkset-manifest schema accepts the optional ``overlap_words``
  field (and a manifest WITHOUT it — the default-off shape — still
  validates).

All fixtures are inline duck-typed chunk dicts; no models / GPU / corpus.
"""

import copy
import json
from pathlib import Path

import pytest

from Trainforge.chunker import (
    CHUNK_OVERLAP_WORDS_ENV,
    apply_chunk_overlap,
    resolve_chunk_overlap_words,
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "schemas"
    / "library"
    / "chunkset_manifest.schema.json"
)


# ---------------------------------------------------------------------------
# resolve_chunk_overlap_words — parse-with-fallback
# ---------------------------------------------------------------------------


def test_resolve_default_is_zero_when_unset():
    assert resolve_chunk_overlap_words(env={}) == 0


def test_resolve_positive_int():
    assert resolve_chunk_overlap_words(env={CHUNK_OVERLAP_WORDS_ENV: "12"}) == 12


def test_resolve_strips_whitespace():
    assert resolve_chunk_overlap_words(env={CHUNK_OVERLAP_WORDS_ENV: "  7 "}) == 7


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "", "3.5", "  "])
def test_resolve_garbage_and_nonpositive_fall_back_to_zero(raw):
    assert resolve_chunk_overlap_words(env={CHUNK_OVERLAP_WORDS_ENV: raw}) == 0


def test_resolve_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv(CHUNK_OVERLAP_WORDS_ENV, "9")
    assert resolve_chunk_overlap_words() == 9
    monkeypatch.delenv(CHUNK_OVERLAP_WORDS_ENV, raising=False)
    assert resolve_chunk_overlap_words() == 0


# ---------------------------------------------------------------------------
# apply_chunk_overlap — off / no-op contracts
# ---------------------------------------------------------------------------


def _seq_chunks():
    return [
        {"id": "c1", "text": "alpha beta gamma delta", "word_count": 4,
         "tokens_estimate": 5},
        {"id": "c2", "text": "epsilon zeta eta", "word_count": 3,
         "tokens_estimate": 3},
        {"id": "c3", "text": "theta iota kappa lambda", "word_count": 4,
         "tokens_estimate": 5},
    ]


def test_overlap_zero_is_byte_identical_noop():
    chunks = _seq_chunks()
    before = copy.deepcopy(chunks)
    out = apply_chunk_overlap(chunks, 0)
    assert out is chunks
    assert chunks == before


def test_overlap_negative_is_noop():
    chunks = _seq_chunks()
    before = copy.deepcopy(chunks)
    apply_chunk_overlap(chunks, -3)
    assert chunks == before


def test_single_chunk_is_noop():
    chunks = [{"id": "c1", "text": "alpha beta", "word_count": 2}]
    before = copy.deepcopy(chunks)
    apply_chunk_overlap(chunks, 5)
    assert chunks == before


# ---------------------------------------------------------------------------
# apply_chunk_overlap — verbatim prepend (no follows_chunk → sequential)
# ---------------------------------------------------------------------------


def _generic_chunks():
    # No ``follows_chunk`` / ``id`` keys → pure sequential overlap.
    return [
        {"text": "alpha beta gamma delta"},
        {"text": "epsilon zeta eta"},
        {"text": "theta iota kappa lambda"},
    ]


def test_sequential_overlap_prepends_verbatim_tail():
    chunks = _generic_chunks()
    apply_chunk_overlap(chunks, 2)
    # c2 gets the last 2 words of c1; c3 gets the last 2 words of c2.
    assert chunks[0]["text"] == "alpha beta gamma delta"  # first untouched
    assert chunks[1]["text"] == "gamma delta epsilon zeta eta"
    assert chunks[2]["text"] == "zeta eta theta iota kappa lambda"


def test_overlap_does_not_compound():
    # c3's prefix must be c2's ORIGINAL tail ("zeta eta"), NOT c2's
    # already-overlapped tail — the snapshot prevents compounding.
    chunks = _generic_chunks()
    apply_chunk_overlap(chunks, 3)
    assert chunks[1]["text"] == "beta gamma delta epsilon zeta eta"
    # c3 prefix = original c2 tail = last 3 of "epsilon zeta eta".
    assert chunks[2]["text"] == "epsilon zeta eta theta iota kappa lambda"


def test_overlap_words_are_subset_of_prior_chunk():
    # Anti-fabrication: every prepended word exists in the prior chunk.
    originals = [c["text"].split() for c in _generic_chunks()]
    chunks = _generic_chunks()
    apply_chunk_overlap(chunks, 2)
    for i in range(1, len(chunks)):
        new_words = chunks[i]["text"].split()
        added = new_words[: len(new_words) - len(originals[i])]
        assert all(w in originals[i - 1] for w in added)


def test_n_larger_than_prior_takes_all_words():
    chunks = [{"text": "one two"}, {"text": "three four five"}]
    apply_chunk_overlap(chunks, 10)
    assert chunks[1]["text"] == "one two three four five"


def test_word_count_and_tokens_recomputed():
    chunks = _seq_chunks()  # carries follows-less ids → see continuity test
    # Remove follows linkage so sequential applies; keep id/word_count.
    for c in chunks:
        c.pop("follows_chunk", None)
    apply_chunk_overlap(chunks, 2)
    c2 = chunks[1]
    assert c2["word_count"] == len(c2["text"].split())
    assert c2["tokens_estimate"] == int(len(c2["text"].split()) * 1.3)


# ---------------------------------------------------------------------------
# apply_chunk_overlap — follows_chunk continuity guard
# ---------------------------------------------------------------------------


def test_follows_chunk_boundary_blocks_bleed():
    # c2.follows_chunk is None (lesson boundary) → no bleed from c1.
    chunks = [
        {"id": "c1", "text": "alpha beta gamma", "follows_chunk": None},
        {"id": "c2", "text": "delta epsilon", "follows_chunk": None},
        {"id": "c3", "text": "zeta eta", "follows_chunk": "c2"},
    ]
    apply_chunk_overlap(chunks, 2)
    assert chunks[1]["text"] == "delta epsilon"  # boundary: unchanged
    assert chunks[2]["text"] == "delta epsilon zeta eta"  # c3 follows c2


def test_follows_chunk_mismatch_blocks_bleed():
    # c2 claims to follow a DROPPED chunk (not the sequential prior) →
    # no bleed, so a filtered-out chunk never leaks across the gap.
    chunks = [
        {"id": "c1", "text": "alpha beta gamma", "follows_chunk": None},
        {"id": "c2", "text": "delta epsilon", "follows_chunk": "c0-dropped"},
    ]
    apply_chunk_overlap(chunks, 2)
    assert chunks[1]["text"] == "delta epsilon"


def test_follows_chunk_match_allows_bleed():
    chunks = [
        {"id": "c1", "text": "alpha beta gamma", "follows_chunk": None},
        {"id": "c2", "text": "delta epsilon", "follows_chunk": "c1"},
    ]
    apply_chunk_overlap(chunks, 2)
    assert chunks[1]["text"] == "beta gamma delta epsilon"


# ---------------------------------------------------------------------------
# Manifest schema — optional overlap_words field
# ---------------------------------------------------------------------------


def _load_schema():
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _base_manifest():
    # The current SemantiK-emit sidecar shape (chunkset_kind "semantik" +
    # source_semantik_html_sha256).
    return {
        "chunks_sha256": "a" * 64,
        "chunker_version": "v4",
        "chunkset_kind": "semantik",
        "source_semantik_html_sha256": "b" * 64,
    }


def _base_legacy_manifest():
    # Legacy-compat: the pre-purge chunkset_kind "dart" shape still accepted on
    # read by the dual-read chunkset-manifest schema.
    return {
        "chunks_sha256": "a" * 64,
        "chunker_version": "v4",
        "chunkset_kind": "dart",
        "source_dart_html_sha256": "b" * 64,
    }


def test_schema_declares_overlap_words():
    schema = _load_schema()
    assert "overlap_words" in schema["properties"]
    assert schema["properties"]["overlap_words"]["type"] == "integer"


def test_schema_accepts_manifest_with_overlap_words():
    jsonschema = pytest.importorskip("jsonschema")
    manifest = _base_manifest()
    manifest["overlap_words"] = 16
    jsonschema.validate(manifest, _load_schema())


def test_schema_accepts_manifest_without_overlap_words():
    # The default-off shape (no overlap_words key) must still validate.
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(_base_manifest(), _load_schema())


def test_schema_accepts_legacy_manifest_without_overlap_words():
    # Legacy-compat read path: the pre-purge chunkset_kind "dart" sidecar still
    # validates (dual-read).
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(_base_legacy_manifest(), _load_schema())


def test_schema_rejects_nonpositive_overlap_words():
    jsonschema = pytest.importorskip("jsonschema")
    manifest = _base_manifest()
    manifest["overlap_words"] = 0  # off-state is OMISSION, not 0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, _load_schema())
