"""W6 — within-package exact-normalised dedup regression net.

Covers the deterministic, default-off ``ED4ALL_CHUNK_DEDUP`` pass that
``chunk_content`` runs BEFORE ``_generate_chunk_id`` mints anything:

* the exact-normalised primitives (:func:`normalize_exact` /
  :func:`exact_content_hash` / :func:`exact_token_count`) — whitespace-collapse
  and casefold ONLY, so ``sh:minCount`` and ``sh minCount`` stay distinct where
  the cross-course :func:`normalize_for_dedup` would collapse them;
* ``ED4ALL_CHUNK_DEDUP`` / ``ED4ALL_CHUNK_DEDUP_MIN_TOKENS`` parse-with-fallback;
* first-occurrence preservation (the cross-course ``drop_boilerplate_chunks``
  drops EVERY occurrence, including the first — it cannot be reused);
* the token floor protecting short-but-legitimate repeats;
* flag-off byte-identity;
* the two hard correctness invariants the pre-mint ordering exists to protect —
  chunk ids stay a dense 1-based sequence, and every surviving chunk's
  ``follows_chunk`` still resolves to an emitted id;
* the drop ledger's shape and its resolvability back into the emitted chunks.

All fixtures are inline duck-typed item / section dicts; no models, no GPU, no
corpus.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from Trainforge.chunker import (
    DEFAULT_CHUNK_DEDUP_MIN_TOKENS,
    ENV_CHUNK_DEDUP,
    ENV_CHUNK_DEDUP_MIN_TOKENS,
    ChunkerContext,
    chunk_content,
    exact_content_hash,
    exact_token_count,
    normalize_exact,
    resolve_chunk_dedup_enabled,
    resolve_chunk_dedup_min_tokens,
)
from Trainforge.chunker.cross_course_dedup import normalize_for_dedup


# ---------------------------------------------------------------------------
# Primitives — normalize_exact / exact_content_hash / exact_token_count
# ---------------------------------------------------------------------------


def test_normalize_exact_collapses_whitespace_and_casefolds():
    assert normalize_exact("  The\tQuick \n Brown  ") == "the quick brown"


def test_normalize_exact_preserves_punctuation():
    assert normalize_exact("sh:minCount = 1.") == "sh:mincount = 1."


def test_normalize_exact_handles_none_and_empty():
    assert normalize_exact("") == ""
    assert normalize_exact(None) == ""  # type: ignore[arg-type]


def test_exact_hash_separates_curie_from_space_form():
    # The whole reason normalize_for_dedup cannot be reused: it strips ALL
    # punctuation, so these two DO collide there and must NOT collide here.
    assert normalize_for_dedup("sh:minCount") == normalize_for_dedup("sh minCount")
    assert exact_content_hash("sh:minCount") != exact_content_hash("sh minCount")


def test_exact_hash_separates_formula_from_prose():
    assert exact_content_hash("f(x) = 1") != exact_content_hash("f x 1")


def test_exact_hash_collides_on_whitespace_and_case_only():
    assert exact_content_hash("Alpha  Beta") == exact_content_hash("alpha\nbeta")


def test_exact_hash_empty_is_empty_string():
    assert exact_content_hash("") == ""
    assert exact_content_hash("   \n ") == ""


def test_exact_token_count():
    assert exact_token_count("one two  three") == 3
    assert exact_token_count("   ") == 0


# ---------------------------------------------------------------------------
# Resolvers — parse-with-fallback
# ---------------------------------------------------------------------------


def test_dedup_default_off():
    assert resolve_chunk_dedup_enabled(env={}) is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", " yes ", "on"])
def test_dedup_truthy_tokens(raw):
    assert resolve_chunk_dedup_enabled(env={ENV_CHUNK_DEDUP: raw}) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "garbage"])
def test_dedup_falsey_and_garbage_tokens(raw):
    assert resolve_chunk_dedup_enabled(env={ENV_CHUNK_DEDUP: raw}) is False


def test_min_tokens_default():
    assert resolve_chunk_dedup_min_tokens(env={}) == DEFAULT_CHUNK_DEDUP_MIN_TOKENS


def test_min_tokens_explicit_zero_is_honoured():
    # 0 = "no floor" is a legitimate operator choice, not garbage.
    assert resolve_chunk_dedup_min_tokens(env={ENV_CHUNK_DEDUP_MIN_TOKENS: "0"}) == 0


def test_min_tokens_positive():
    assert resolve_chunk_dedup_min_tokens(env={ENV_CHUNK_DEDUP_MIN_TOKENS: " 25 "}) == 25


@pytest.mark.parametrize("raw", ["-1", "abc", "3.5", "", "  "])
def test_min_tokens_garbage_and_negative_fall_back_to_default(raw):
    assert (
        resolve_chunk_dedup_min_tokens(env={ENV_CHUNK_DEDUP_MIN_TOKENS: raw})
        == DEFAULT_CHUNK_DEDUP_MIN_TOKENS
    )


def test_resolvers_read_os_environ_by_default(monkeypatch):
    monkeypatch.setenv(ENV_CHUNK_DEDUP, "1")
    monkeypatch.setenv(ENV_CHUNK_DEDUP_MIN_TOKENS, "4")
    assert resolve_chunk_dedup_enabled() is True
    assert resolve_chunk_dedup_min_tokens() == 4


# ---------------------------------------------------------------------------
# Fixtures — items + a recording ChunkerContext
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    heading: str
    content: str
    word_count: int
    level: int = 3
    data_dart_unit: Optional[str] = None
    data_dart_subclass: Optional[str] = None
    data_dart_opener: Optional[str] = None
    data_dart_flows: List[str] = field(default_factory=list)


#: 12 exact-normalised tokens — comfortably above the default floor of 8.
_SHARED = (
    "This project is released under a permissive licence and ships without "
    "warranty"
)


def _page(item_id: str, title: str, body: str) -> Dict[str, Any]:
    """An unsectioned page item (``sections: []`` → the whole-item emit path)."""
    return {
        "module_id": item_id,
        "item_id": item_id,
        "item_path": f"docs/{item_id}.html",
        "title": title,
        "resource_type": "page",
        "raw_html": f"<html><body><p>{body}</p></body></html>",
        "sections": [],
        "misconceptions": [],
    }


def _sectioned(item_id: str, sections: List[_Section]) -> Dict[str, Any]:
    body = "".join(
        f"<h3>{s.heading}</h3><p>{s.content}</p>" for s in sections
    )
    return {
        "module_id": item_id,
        "item_id": item_id,
        "item_path": f"docs/{item_id}.html",
        "title": f"Page {item_id}",
        "resource_type": "page",
        "raw_html": f"<html><body>{body}</body></html>",
        "sections": sections,
        "misconceptions": [],
    }


def _ctx() -> ChunkerContext:
    def _create_chunk(*, chunk_id, text, **kw) -> Dict[str, Any]:
        return {
            "id": chunk_id,
            "text": text,
            "follows_chunk": kw.get("follows_chunk_id"),
            "position_in_module": kw.get("position_in_module"),
        }

    return ChunkerContext(create_chunk=_create_chunk)


def _run(items, monkeypatch, *, enabled, min_tokens=None, **kwargs):
    if enabled is None:
        monkeypatch.delenv(ENV_CHUNK_DEDUP, raising=False)
    else:
        monkeypatch.setenv(ENV_CHUNK_DEDUP, "1" if enabled else "0")
    if min_tokens is None:
        monkeypatch.delenv(ENV_CHUNK_DEDUP_MIN_TOKENS, raising=False)
    else:
        monkeypatch.setenv(ENV_CHUNK_DEDUP_MIN_TOKENS, str(min_tokens))
    return chunk_content(copy.deepcopy(items), "TEST_101", ctx=_ctx(), **kwargs)


# ---------------------------------------------------------------------------
# Flag off — byte-identical emit, empty ledger
# ---------------------------------------------------------------------------


def _duplicate_corpus() -> List[Dict[str, Any]]:
    return [
        _page("p1", "Licence", _SHARED),
        _page("p2", "Overview", "A wholly distinct paragraph of prose about the "
                                "subject matter at hand"),
        _page("p3", "Licence", _SHARED),
        _page("p4", "Licence", "  THIS Project is  released under a Permissive "
                               "licence\tand ships without warranty  "),
    ]


def test_flag_unset_keeps_every_duplicate(monkeypatch):
    res = _run(_duplicate_corpus(), monkeypatch, enabled=None)
    assert len(res.chunks) == 4
    assert res.dedup_drops == []


def test_flag_off_is_byte_identical_to_unset(monkeypatch):
    unset = _run(_duplicate_corpus(), monkeypatch, enabled=None)
    off = _run(_duplicate_corpus(), monkeypatch, enabled=False)
    assert off.chunks == unset.chunks
    assert off.dedup_drops == []


def test_flag_off_ignores_min_tokens(monkeypatch):
    # The floor knob must be inert while the gate is off.
    off = _run(_duplicate_corpus(), monkeypatch, enabled=False, min_tokens=0)
    unset = _run(_duplicate_corpus(), monkeypatch, enabled=None)
    assert off.chunks == unset.chunks


# ---------------------------------------------------------------------------
# Flag on — first-occurrence-preserving collapse
# ---------------------------------------------------------------------------


def test_exact_repeats_collapse_to_first_occurrence(monkeypatch):
    res = _run(_duplicate_corpus(), monkeypatch, enabled=True)
    # p3 (exact) and p4 (whitespace/case variant) both collapse into p1.
    assert len(res.chunks) == 2
    assert len(res.dedup_drops) == 2
    # The SURVIVOR is the first occurrence — p1's text, verbatim, not p4's
    # re-cased variant.
    assert res.chunks[0]["text"] == _SHARED
    assert res.chunks[0]["id"] == "test_101_chunk_00001"


def test_dropped_units_do_not_consume_chunk_ids(monkeypatch):
    res = _run(_duplicate_corpus(), monkeypatch, enabled=True)
    ids = [c["id"] for c in res.chunks]
    assert ids == [f"test_101_chunk_{i:05d}" for i in range(1, len(ids) + 1)]


def test_follows_chunk_resolves_for_every_survivor(monkeypatch):
    # Force a sentence split so at least one item emits a real follows chain,
    # then drop a later duplicate of that same item.
    long_body = (
        "Alpha beta gamma delta epsilon zeta. Eta theta iota kappa lambda mu. "
        "Nu xi omicron pi rho sigma."
    )
    items = [
        _page("p1", "Guide", long_body),
        _page("p2", "Guide", long_body),
        _page("p3", "Other", "Tau upsilon phi chi psi omega alpha beta gamma "
                             "delta epsilon"),
    ]
    res = _run(
        items, monkeypatch, enabled=True, max_chunk_size=8, target_chunk_size=6
    )
    emitted = {c["id"] for c in res.chunks}
    assert len(res.dedup_drops) == 1
    assert len(res.chunks) > 2  # p1 really did split
    for chunk in res.chunks:
        follows = chunk["follows_chunk"]
        assert follows is None or follows in emitted
    ids = [c["id"] for c in res.chunks]
    assert ids == [f"test_101_chunk_{i:05d}" for i in range(1, len(ids) + 1)]


def test_position_in_module_is_dense_after_a_drop(monkeypatch):
    # A dropped unit must not advance position_in_module either, or the
    # surviving chunks carry a hole in their in-module ordinal.
    long_body = (
        "Alpha beta gamma delta epsilon zeta. Eta theta iota kappa lambda mu. "
        "Nu xi omicron pi rho sigma."
    )
    item = _sectioned(
        "p1",
        [
            _Section("Intro", long_body, 18),
            _Section("Intro", long_body, 18),
        ],
    )
    res = _run(
        [item], monkeypatch, enabled=True, max_chunk_size=8, target_chunk_size=6
    )
    assert len(res.dedup_drops) == 1
    positions = [c["position_in_module"] for c in res.chunks]
    assert positions == list(range(len(positions)))


# ---------------------------------------------------------------------------
# Conservatism — punctuation, the token floor, and heading scoping
# ---------------------------------------------------------------------------


def test_curie_and_space_form_do_not_collapse(monkeypatch):
    items = [
        _page("p1", "Shapes", "The property shape constrains sh:minCount to "
                              "exactly one value here"),
        _page("p2", "Shapes", "The property shape constrains sh minCount to "
                              "exactly one value here"),
    ]
    res = _run(items, monkeypatch, enabled=True)
    assert len(res.chunks) == 2
    assert res.dedup_drops == []


def test_sub_floor_repeats_are_never_dropped(monkeypatch):
    items = [
        _page("p1", "Note", "See the reference"),
        _page("p2", "Note", "See the reference"),
        _page("p3", "Note", "see  THE reference"),
    ]
    res = _run(items, monkeypatch, enabled=True)
    assert len(res.chunks) == 3
    assert res.dedup_drops == []


def test_floor_zero_drops_short_repeats(monkeypatch):
    items = [
        _page("p1", "Note", "See the reference"),
        _page("p2", "Note", "See the reference"),
    ]
    res = _run(items, monkeypatch, enabled=True, min_tokens=0)
    assert len(res.chunks) == 1
    assert len(res.dedup_drops) == 1


def test_same_body_under_different_headings_is_kept(monkeypatch):
    # The unit key is heading-scoped on purpose: collapsing these would
    # silently discard one section's heading provenance.
    items = [
        _page("p1", "Installing", _SHARED),
        _page("p2", "Upgrading", _SHARED),
    ]
    res = _run(items, monkeypatch, enabled=True)
    assert len(res.chunks) == 2
    assert res.dedup_drops == []


# ---------------------------------------------------------------------------
# The sectioned emit path
# ---------------------------------------------------------------------------


def test_sectioned_path_dedups_across_items(monkeypatch):
    sections = [_Section("Prerequisites", _SHARED, exact_token_count(_SHARED))]
    items = [
        _sectioned("p1", list(sections)),
        _sectioned("p2", list(sections)),
    ]
    res = _run(items, monkeypatch, enabled=True)
    assert len(res.chunks) == 1
    assert len(res.dedup_drops) == 1
    assert res.dedup_drops[0]["source_item_path"] == "docs/p2.html"


def test_sectioned_path_flag_off_keeps_both(monkeypatch):
    sections = [_Section("Prerequisites", _SHARED, exact_token_count(_SHARED))]
    items = [
        _sectioned("p1", list(sections)),
        _sectioned("p2", list(sections)),
    ]
    res = _run(items, monkeypatch, enabled=False)
    assert len(res.chunks) == 2
    assert res.dedup_drops == []


# ---------------------------------------------------------------------------
# The drop ledger
# ---------------------------------------------------------------------------


def test_ledger_rows_carry_the_documented_shape(monkeypatch):
    res = _run(_duplicate_corpus(), monkeypatch, enabled=True)
    for row in res.dedup_drops:
        assert set(row) == {
            "dropped_index",
            "kept_chunk_index",
            "kept_chunk_id",
            "normalized_hash",
            "source_item_path",
        }
        assert isinstance(row["dropped_index"], int)
        assert isinstance(row["kept_chunk_index"], int)
        assert len(row["normalized_hash"]) == 64


def test_ledger_kept_reference_resolves_into_the_emitted_chunks(monkeypatch):
    res = _run(_duplicate_corpus(), monkeypatch, enabled=True)
    emitted = {c["id"] for c in res.chunks}
    assert res.dedup_drops
    for row in res.dedup_drops:
        assert row["kept_chunk_id"] in emitted
        assert res.chunks[row["kept_chunk_index"]]["id"] == row["kept_chunk_id"]


def test_ledger_records_the_dropped_units_own_source_path(monkeypatch):
    res = _run(_duplicate_corpus(), monkeypatch, enabled=True)
    assert [r["source_item_path"] for r in res.dedup_drops] == [
        "docs/p3.html",
        "docs/p4.html",
    ]


def test_ledger_dropped_index_is_the_unit_ordinal(monkeypatch):
    # Four units in source order (p1..p4); p3 and p4 are the drops.
    res = _run(_duplicate_corpus(), monkeypatch, enabled=True)
    assert [r["dropped_index"] for r in res.dedup_drops] == [2, 3]


def test_ledger_shares_one_hash_for_normalised_variants(monkeypatch):
    res = _run(_duplicate_corpus(), monkeypatch, enabled=True)
    hashes = {r["normalized_hash"] for r in res.dedup_drops}
    assert len(hashes) == 1


def test_result_still_tuple_unpacks_to_two_names(monkeypatch):
    # dedup_drops is a third FIELD but deliberately not a third yield, so every
    # existing ``chunks, pages = chunk_content(...)`` call site keeps working.
    chunks, pages = _run(_duplicate_corpus(), monkeypatch, enabled=True)
    assert isinstance(chunks, list)
    assert isinstance(pages, set)
