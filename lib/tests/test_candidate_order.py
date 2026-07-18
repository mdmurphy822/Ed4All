"""Tests for the frontier per-claim candidate ORDERING (no models, pure CPU).

Covers the four ordering tiers (direct-cited → pool-IDF lexical anchor + locality
→ cosine → stable index), determinism, and the never-exclude invariant (output
is always a permutation of the input indices).
"""
from __future__ import annotations

from lib.retrieval.candidate_order import (
    ORDERING_VERSION,
    anchor_score,
    order_passages_for_claim,
)


def test_ordering_version_is_v2():
    # Bumped from the flat-count v1 to the pool-IDF + bigram + locality scorer.
    assert ORDERING_VERSION == 2


def test_anchor_score_uncommon_and_numeric():
    # "velocity"/"function" are shared uncommon tokens (len>=5); "25" is a shared
    # numeric literal (worth 2x). Common short tokens ("the") never count.
    claim = "the velocity function equals 25"
    passage = "velocity function chart with the value 25 shown"
    # shared uncommon: velocity, function, equals? -> "equals" not in passage.
    # velocity(1) + function(1) = 2 uncommon; numeric "25" shared = 1 -> +2.
    assert anchor_score(claim, passage) == 4


def test_anchor_score_no_overlap_is_zero():
    assert anchor_score("alpha beta gamma", "nothing common at all") == 0


def test_order_is_permutation_never_excludes():
    claim = "shared token here"
    texts = ["a", "shared token", "b", "c", "shared"]
    cids = ["c0", "c1", "c2", "c3", "c4"]
    order = order_passages_for_claim(claim, texts, cids)
    assert sorted(order) == list(range(len(texts)))


def test_tier_a_direct_cited_first():
    claim = "velocity function derivative"
    # p0 has the highest lexical anchor but is NOT direct-cited; p2 is direct-cited
    # with a weaker anchor. Tier (a) must still float p2 to the front.
    texts = [
        "velocity function derivative motion",  # strong anchor, not cited
        "unrelated filler",
        "velocity only",  # weak anchor, cited
    ]
    cids = ["c0", "c1", "c2"]
    order = order_passages_for_claim(claim, texts, cids, direct_cited_ids={"c2"})
    assert order[0] == 2


def test_tier_b_lexical_anchor_orders_noncited():
    claim = "velocity function derivative acceleration"
    texts = [
        "single velocity match",  # 1 uncommon shared
        "velocity function derivative acceleration all",  # 4 uncommon shared
        "nothing shared",  # 0
    ]
    cids = ["c0", "c1", "c2"]
    order = order_passages_for_claim(claim, texts, cids)
    assert order == [1, 0, 2]


def test_tier_d_stable_index_tiebreak():
    # Identical anchor scores (all zero overlap) -> stable original index order.
    claim = "zzzz yyyy"
    texts = ["aaaa", "bbbb", "cccc", "dddd"]
    cids = ["c0", "c1", "c2", "c3"]
    order = order_passages_for_claim(claim, texts, cids)
    assert order == [0, 1, 2, 3]


def test_tier_c_cosine_breaks_anchor_ties():
    # Two passages with an identical lexical anchor; cosine (tier c) decides,
    # ahead of the original-index tiebreak. Unit vectors: passage 1 aligns with
    # the claim vector more than passage 0, so it must sort first.
    claim = "velocity function"
    texts = ["velocity function alpha", "velocity function beta"]
    cids = ["c0", "c1"]
    claim_vec = [1.0, 0.0]
    passage_vecs = [[0.0, 1.0], [1.0, 0.0]]  # p1 higher cosine with claim
    order = order_passages_for_claim(
        claim, texts, cids, claim_vec=claim_vec, passage_vecs=passage_vecs
    )
    assert order == [1, 0]


def test_cosine_none_falls_back_to_index():
    # No vectors -> tier (c) contributes 0.0 for all; anchor-tied passages fall
    # to the stable index tiebreak (byte-identical to lexical-only ordering).
    claim = "velocity function"
    texts = ["velocity function a", "velocity function b"]
    cids = ["c0", "c1"]
    order = order_passages_for_claim(claim, texts, cids)
    assert order == [0, 1]


def test_empty_pool():
    assert order_passages_for_claim("claim", [], []) == []


def test_determinism_repeated_calls():
    claim = "velocity function derivative motion acceleration"
    texts = [f"velocity token {i} function" for i in range(20)]
    cids = [f"c{i}" for i in range(20)]
    first = order_passages_for_claim(claim, texts, cids, direct_cited_ids={"c7"})
    for _ in range(5):
        assert order_passages_for_claim(
            claim, texts, cids, direct_cited_ids={"c7"}
        ) == first


# --------------------------------------------------------------------------- #
# v2 pool-IDF lexical scoring
# --------------------------------------------------------------------------- #


def test_idf_beats_flat_count_rare_token_wins():
    # Two candidate passages each share EXACTLY one uncommon token with the claim:
    #   p0 shares "commonword" (in 4/5 pool chunks → tiny IDF),
    #   p1 shares "rareword"   (in 1/5 pool chunks → large IDF).
    # A flat |shared| count ties them (1 == 1) → index order would put p0 first;
    # the pool-IDF score floats the rare-token passage p1 to the front.
    claim = "rareword commonword"
    texts = [
        "commonword alpha",   # p0 — common token shared
        "rareword bravo",     # p1 — rare token shared
        "commonword gamma",   # pool fill: inflates commonword df
        "commonword delta",
        "commonword epsilon",
    ]
    cids = [f"c{i}" for i in range(len(texts))]
    order = order_passages_for_claim(claim, texts, cids)
    assert order[0] == 1


def test_bigram_anchor_ranks_phrase_sharing_chunk_first():
    # p0 and p1 share the SAME unigrams with the claim ("distributive",
    # "property") — equal unigram IDF — but only p0 carries the adjacent bigram
    # "distributive property". The rare-bigram tier must break the tie for p0.
    claim = "learn the distributive property today"
    texts = [
        "the distributive property section",       # p0 — bigram present
        "distributive rules and property notes",   # p1 — same unigrams, no bigram
        "unrelated filler passage entirely",       # p2 — nothing shared
    ]
    cids = ["c0", "c1", "c2"]
    order = order_passages_for_claim(claim, texts, cids)
    assert order[0] == 0
    assert order.index(0) < order.index(1)


def test_numeric_idf_weighted_and_doubled():
    # p0 shares a rare numeric literal (df 1); p1 shares an equally-rare uncommon
    # word (df 1). Same pool IDF, but the numeric tier carries the ×2 multiplier,
    # so p0 outranks p1.
    claim = "chapter yields 4096"
    texts = [
        "result 4096 here",   # p0 — rare numeric shared (2 × IDF)
        "chapter overview",   # p1 — one equally-rare uncommon word shared
        "unrelated distractor row",
    ]
    cids = ["c0", "c1", "c2"]
    order = order_passages_for_claim(claim, texts, cids)
    assert order[0] == 0
    assert order.index(0) < order.index(1)


# --------------------------------------------------------------------------- #
# v2 sequential-locality prior
# --------------------------------------------------------------------------- #


def test_locality_bonus_orders_neighbors_after_direct_cited():
    # c2 is the sole direct-cited chunk → tier (a) floats index 2 to the front.
    # Among the zero-lexical remainder, indices 1 and 3 neighbor the cited chunk
    # (by both pool index AND chunk-id ordinal) and take the locality bonus, so
    # they precede the non-neighbor index 0 despite its lower original index.
    claim = "zzzzz"
    texts = ["aaaaa", "bbbbb", "ccccc", "ddddd"]
    cids = ["c0", "c1", "c2", "c3"]
    order = order_passages_for_claim(claim, texts, cids, direct_cited_ids={"c2"})
    assert order == [2, 1, 3, 0]


def test_locality_absent_without_direct_cited():
    # No direct-cited set → no locality prior → pure stable index order for a
    # zero-lexical pool (byte-identical to the no-bonus path).
    claim = "zzzzz"
    texts = ["aaaaa", "bbbbb", "ccccc", "ddddd"]
    cids = ["c0", "c1", "c2", "c3"]
    order = order_passages_for_claim(claim, texts, cids)
    assert order == [0, 1, 2, 3]


def test_locality_does_not_dominate_idf():
    # A strong (rare) lexical match on a NON-neighbor must still outrank a
    # zero-lexical neighbor riding only the small locality bonus.
    claim = "photosynthesis"
    texts = [
        "citedchunk bbbbb",            # p0 — direct-cited (c0)
        "unrelated aaaaa",             # p1 — neighbor of cited p0, zero lexical
        "photosynthesis ccccc",        # p2 — rare lexical match, NOT a neighbor
    ]
    cids = ["c0", "c1", "c2"]
    order = order_passages_for_claim(claim, texts, cids, direct_cited_ids={"c0"})
    # cited p0 first (tier a); then the real lexical match p2 beats the
    # locality-only neighbor p1.
    assert order[0] == 0
    assert order.index(2) < order.index(1)


def test_never_exclude_with_all_v2_signals_active():
    claim = "distributive property equals 12 across sections"
    texts = [f"section {i} distributive property value 12" for i in range(15)]
    cids = [f"sec_{i:02d}" for i in range(15)]
    order = order_passages_for_claim(
        claim,
        texts,
        cids,
        direct_cited_ids={"sec_07"},
    )
    assert sorted(order) == list(range(15))
