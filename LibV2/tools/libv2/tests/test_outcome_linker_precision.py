"""Outcome-link precision controls (over-linking fix).

Regression coverage for the precision-collapse pathology where broad
objectives, at a low TF-IDF floor, attach to nearly every chunk (e.g.
"CO-02 on 72/112 chunks"). The fix layers four controls onto
``link_chunks_to_outcomes``:

1. per-chunk top-K cap (``max_outcomes_per_chunk``)
2. a raised default floor (``similarity_threshold`` = 0.20)
3. a global-frequency anti-signal (an outcome over-linked to > 40% of
   chunks is pruned to the chunks where it ranks top-1/top-2)
4. a key-concept overlap boost/gate vs the chunk's ``concept_tags``

These tests assert each control fires while genuinely-strong single-outcome
links are preserved.
"""

from __future__ import annotations

from typing import Any, Dict, List

from LibV2.tools.libv2.outcome_linker import (
    BROAD_OUTCOME_FREQUENCY,
    DEFAULT_SIMILARITY_THRESHOLD,
    LearningOutcome,
    link_chunks_to_outcomes,
)


def _outcome(oid: str, statement: str, concepts: List[str] | None = None) -> LearningOutcome:
    return LearningOutcome(
        objective_id=oid,
        statement=statement,
        bloom_level="understand",
        key_concepts=concepts or [],
    )


def _chunk(text: str, tags: List[str] | None = None) -> Dict[str, Any]:
    c: Dict[str, Any] = {"text": text}
    if tags is not None:
        c["concept_tags"] = tags
    return c


def test_default_threshold_is_raised() -> None:
    """The over-linking fix raises the default floor to 0.20."""
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.20


def test_broad_outcome_pruned_to_top_matches() -> None:
    """A broad outcome that textually matches ~all chunks must not stay linked
    to all of them — the global-frequency anti-signal prunes it to the chunks
    where it is a top-ranked match (<= 40%)."""
    # One broad outcome whose vocabulary appears in every chunk, plus several
    # specific outcomes. Each chunk owns TWO specific outcomes strongly, so the
    # broad outcome lands at rank 3+ (a genuinely marginal attachment) on every
    # chunk and must be pruned.
    outcomes = [
        _outcome("CO-BROAD", "system overview data process model concept"),
        _outcome("CO-A", "photosynthesis chloroplast light reaction"),
        _outcome("CO-B", "mitochondria cellular respiration atp adenosine"),
        _outcome("CO-C", "dna replication helicase polymerase"),
        _outcome("CO-D", "membrane transport osmosis diffusion"),
    ]

    # 10 chunks. Each chunk strongly matches TWO specific outcomes but ALSO
    # contains the broad outcome's generic vocabulary (once), so naive linking
    # would attach CO-BROAD to all 10 while it actually ranks behind the two
    # specific outcomes on each.
    paired_bodies = [
        # photosynthesis + cellular respiration
        "photosynthesis chloroplast light reaction photosynthesis chloroplast "
        "mitochondria cellular respiration atp adenosine mitochondria respiration",
        # dna + membrane
        "dna replication helicase polymerase dna replication helicase "
        "membrane transport osmosis diffusion membrane transport osmosis",
    ]
    chunks: List[Dict[str, Any]] = []
    broad_vocab = "system overview data process model concept"
    for i in range(10):
        body = paired_bodies[i % len(paired_bodies)]
        chunks.append(_chunk(f"{body} {broad_vocab}"))

    # Low threshold so the broad outcome WOULD over-link without the anti-signal.
    link_chunks_to_outcomes(
        chunks,
        outcomes,
        similarity_threshold=0.10,
        use_concept_gate=False,
    )

    broad_count = sum(
        1 for c in chunks if "CO-BROAD" in c.get("learning_outcome_refs", [])
    )
    assert broad_count <= BROAD_OUTCOME_FREQUENCY * len(chunks), (
        f"broad outcome still over-linked: {broad_count}/{len(chunks)}"
    )
    assert broad_count < 10, "anti-signal did not prune the broad outcome at all"


def test_per_chunk_top_k_respected() -> None:
    """A chunk that matches many outcomes keeps at most ``max_outcomes_per_chunk``."""
    # Six outcomes, all sharing the chunk's vocabulary so all clear the floor.
    outcomes = [
        _outcome(f"CO-{i:02d}", "alpha beta gamma delta epsilon zeta")
        for i in range(6)
    ]
    chunks = [_chunk("alpha beta gamma delta epsilon zeta")]

    link_chunks_to_outcomes(
        chunks,
        outcomes,
        similarity_threshold=0.10,
        max_outcomes_per_chunk=3,
        # Disable the broad-outcome pruner so we isolate the top-K cap: with
        # identical outcomes every one would be "broad", but here we only care
        # that the per-chunk cap holds.
        broad_outcome_frequency=2.0,
        use_concept_gate=False,
    )

    refs = chunks[0].get("learning_outcome_refs", [])
    assert len(refs) == 3, f"top-K cap not respected: {refs}"


def test_strong_single_outcome_still_links() -> None:
    """A chunk with a strong, near-exclusive match to one outcome still links it."""
    outcomes = [
        _outcome("CO-STRONG", "quicksort partition pivot recursion algorithm"),
        _outcome("CO-OTHER", "watercolor brush pigment canvas texture"),
    ]
    chunks = [
        _chunk("quicksort partition pivot recursion algorithm quicksort partition"),
    ]

    link_chunks_to_outcomes(chunks, outcomes)  # all defaults (0.20 floor)

    refs = chunks[0].get("learning_outcome_refs", [])
    assert "CO-STRONG" in refs
    assert "CO-OTHER" not in refs


def test_marginal_link_dropped_at_new_default() -> None:
    """A marginal ~0.12-similarity link is dropped under the new 0.20 default."""
    # Construct a chunk whose only overlap with the outcome is a single shared
    # token diluted by lots of unrelated text -> low cosine similarity.
    outcomes = [
        _outcome("CO-MARGINAL", "thermodynamics entropy enthalpy gibbs free energy"),
    ]
    # The chunk shares only "energy"; the rest is unrelated vocabulary, so the
    # cosine similarity is well below 0.20.
    chunks = [
        _chunk(
            "the marketing department scheduled a meeting about quarterly "
            "revenue and customer engagement energy levels in the office"
        ),
    ]

    # Sanity: at a low floor the marginal link would survive.
    import copy

    low = copy.deepcopy(chunks)
    link_chunks_to_outcomes(low, outcomes, similarity_threshold=0.05, use_concept_gate=False)
    assert "CO-MARGINAL" in low[0].get("learning_outcome_refs", []), (
        "fixture not actually marginal — link absent even at low floor"
    )

    # At the new default floor it is dropped.
    link_chunks_to_outcomes(chunks, outcomes)  # 0.20 default
    assert "CO-MARGINAL" not in chunks[0].get("learning_outcome_refs", [])


def test_concept_overlap_keeps_sub_threshold_link() -> None:
    """A sub-threshold link is kept when outcome key_concepts overlap the
    chunk's concept_tags; zero overlap + marginal score is dropped."""
    outcomes = [
        _outcome(
            "CO-CONCEPT",
            "thermodynamics entropy enthalpy gibbs free energy",
            concepts=["entropy", "enthalpy"],
        ),
    ]
    # Marginal text similarity (only "energy" shared), but the chunk carries a
    # concept_tag that overlaps the outcome's key_concepts.
    overlapping = [
        _chunk(
            "the marketing department scheduled a meeting about quarterly "
            "revenue and customer engagement energy levels",
            tags=["entropy", "spontaneity"],
        ),
    ]
    link_chunks_to_outcomes(overlapping, outcomes)  # default 0.20, gate on
    assert "CO-CONCEPT" in overlapping[0].get("learning_outcome_refs", []), (
        "concept-overlap boost failed to retain a sub-threshold link"
    )

    # Same marginal text but NO concept overlap -> dropped by the gate.
    no_overlap = [
        _chunk(
            "the marketing department scheduled a meeting about quarterly "
            "revenue and customer engagement energy levels",
            tags=["marketing", "revenue"],
        ),
    ]
    link_chunks_to_outcomes(no_overlap, outcomes)
    assert "CO-CONCEPT" not in no_overlap[0].get("learning_outcome_refs", [])
