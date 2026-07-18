"""Tests for the frontier per-claim candidate ORDERING (no models, pure CPU).

Covers the four ordering tiers (direct-cited → lexical anchor → cosine → stable
index), determinism, and the never-exclude invariant (output is always a
permutation of the input indices).
"""
from __future__ import annotations

from lib.retrieval.candidate_order import anchor_score, order_passages_for_claim


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
