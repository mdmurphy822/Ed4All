"""Tests for the P4 additive answer-scoring extensions (answer_scoring.py).

CI-safe: pure deterministic arms + an injected fake NLI classifier (no torch).
Covers key-point coverage (shingle / token-coverage / NLI arms), part coverage
(answered + correctly_flagged detection AND its brittleness boundary), and the
per-population citation breakdown.
"""
from __future__ import annotations

from lib.retrieval.answer_scoring import (
    DISCLAIMER_PHRASES,
    score_key_point_coverage,
    score_part_coverage,
    score_population_breakdown,
)


# ===========================================================================
# Fake NLI (duck-typed score_batch -> objects with .entailment)
# ===========================================================================

class _FakeScore:
    def __init__(self, entailment):
        self.entailment = entailment


class _FakeNli:
    """Entails a hypothesis iff a configured keyword is in the premise."""

    def __init__(self, entail_if_contains, entailment=0.95):
        self._kw = entail_if_contains
        self._ent = entailment

    def score_batch(self, pairs):
        out = []
        for premise, _hypothesis in pairs:
            out.append(
                _FakeScore(self._ent if self._kw in premise.lower() else 0.05)
            )
        return out


# ===========================================================================
# Key-point coverage
# ===========================================================================

def test_key_point_none_when_no_points():
    assert score_key_point_coverage("an answer", []) is None
    assert score_key_point_coverage("an answer", None) is None


def test_key_point_shingle_arm_covers_verbatim():
    answer = (
        "A vector store is a database that indexes high-dimensional embedding "
        "vectors for nearest-neighbour search."
    )
    points = [
        "A vector store is a database that indexes high-dimensional embedding vectors",
        "totally unrelated quantum chromodynamics lattice gauge field",
    ]
    cov = score_key_point_coverage(answer, points)
    assert cov is not None
    assert cov.total == 2
    assert cov.covered == 1
    verdicts = {v.key_point[:20]: v for v in cov.verdicts}
    covered = [v for v in cov.verdicts if v.covered]
    assert len(covered) == 1
    assert covered[0].arm == "shingle"
    assert covered[0].shingle >= 0.25


def test_key_point_token_coverage_arm_catches_reorder():
    # Reordered / lightly-inflected paraphrase: shingle-4 fails, token coverage
    # (>= 0.80 of the key point's content tokens present) catches it.
    answer = "Embedding vectors high-dimensional indexes database vector store the."
    points = ["vector store database indexes high-dimensional embedding vectors"]
    cov = score_key_point_coverage(answer, points)
    assert cov.covered == 1
    assert cov.verdicts[0].arm == "token_coverage"


def test_key_point_nli_arm_only_with_floor():
    # Deterministic arms fail (no lexical overlap); NLI entails → covered via nli.
    answer = "The procedure begins by gathering the reagents."
    points = ["You must collect the chemicals before starting"]
    nli = _FakeNli(entail_if_contains="procedure begins")
    cov = score_key_point_coverage(
        answer, points, nli=nli, entailment_floor=0.5
    )
    assert cov.nli_used is True
    assert cov.covered == 1
    assert cov.verdicts[0].arm == "nli"
    assert cov.verdicts[0].entailment is not None


def test_key_point_nli_ignored_without_floor():
    # nli supplied but entailment_floor None → NLI arm disabled (deterministic
    # arms decide). This mirrors with_groundedness=False.
    answer = "Unrelated text."
    points = ["Collect the chemicals before starting"]
    nli = _FakeNli(entail_if_contains="unrelated")
    cov = score_key_point_coverage(answer, points, nli=nli, entailment_floor=None)
    assert cov.nli_used is False
    assert cov.covered == 0


def test_key_point_nli_does_not_downgrade_deterministic_cover():
    # A point covered by the shingle arm stays covered even if NLI would say no.
    answer = "A vector store is a database that indexes embedding vectors today."
    points = ["A vector store is a database that indexes embedding vectors"]
    nli = _FakeNli(entail_if_contains="__never__")  # NLI always says no
    cov = score_key_point_coverage(answer, points, nli=nli, entailment_floor=0.5)
    assert cov.covered == 1
    assert cov.verdicts[0].arm == "shingle"


# ===========================================================================
# Part coverage — covered parts answered
# ===========================================================================

def _part(pid, text, covered, **kw):
    p = {"part_id": pid, "part_text": text, "covered": covered}
    p.update(kw)
    return p


def test_part_none_when_empty():
    assert score_part_coverage("answer", []) is None


def test_part_covered_answered_via_shingle():
    answer = "The five-step problem solving strategy starts with reading the problem."
    parts = [
        _part("a", "the five-step problem solving strategy starts with reading the problem", True),
        _part("b", "an unrelated cooking recipe for souffle", True),
    ]
    cov = score_part_coverage(answer, parts)
    assert cov.n_covered_parts == 2
    assert cov.n_answered == 1
    answered = {v.part_id: v.answered for v in cov.verdicts}
    assert answered["a"] is True
    assert answered["b"] is False


def test_part_covered_answered_via_key_points():
    answer = "Newton's second law relates force mass and acceleration directly."
    parts = [_part("a", "describe the law", True)]
    kp = {"a": ["Newton's second law relates force mass and acceleration"]}
    cov = score_part_coverage(answer, parts, key_points_by_part=kp)
    assert cov.n_answered == 1


def test_part_covered_answered_via_chunk_bodies():
    """The live-inertness fix: schema-valid gold authors a TERSE PROMPT as
    part_text (which does not lexically overlap a prose answer) and anchors the
    part with relevant_passage_refs. The answer-content signal is the chunk
    BODY, threaded via chunks_by_part. Both parts addressed → both answered."""
    answer = (
        "To solve, isolate the absolute value then split into two cases for x. "
        "Then draw the V-shaped graph opening upward from the vertex."
    )
    parts = [
        _part("a", "Solve for x.", True, relevant_passage_refs=["c1"]),
        _part("b", "Graph it.", True, relevant_passage_refs=["c2"]),
    ]
    chunks = {
        "a": ["isolate the absolute value then split into two cases for x"],
        "b": ["draw the V-shaped graph opening upward from the vertex"],
    }
    # No part_text overlap at all — only the chunk bodies carry the content.
    cov = score_part_coverage(answer, parts, chunks_by_part=chunks)
    assert cov.n_covered_parts == 2
    assert cov.n_answered == 2
    assert all(v.answered for v in cov.verdicts)


def test_part_silently_dropped_detected_via_chunk_bodies():
    """Real observed failure: a two-part question (absolute value + 'shaded
    ray') where the answer silently drops the shaded-ray part. The dropped
    part must score answered=False even though its sibling is answered."""
    answer = "To solve, isolate the absolute value then split into two cases for x."
    parts = [
        _part("a", "Solve for x.", True, relevant_passage_refs=["c1"]),
        _part("b", "Graph the shaded ray.", True, relevant_passage_refs=["c2"]),
    ]
    chunks = {
        "a": ["isolate the absolute value then split into two cases for x"],
        "b": ["draw the V-shaped graph opening upward from the vertex"],
    }
    cov = score_part_coverage(answer, parts, chunks_by_part=chunks)
    assert cov.n_answered == 1  # part b silently dropped
    answered = {v.part_id: v.answered for v in cov.verdicts}
    assert answered["a"] is True
    assert answered["b"] is False


def test_part_dict_key_points_arm_still_honored():
    """Forward-compat: a part dict carrying its own key_points (or the
    expected_key_points alias) is also a support source — useful for synthetic
    fixtures that author answer-content directly on the part."""
    answer = "Newton's second law relates force mass and acceleration directly."
    parts = [_part("a", "describe the law", True,
                   expected_key_points=["Newton's second law relates force mass and acceleration"])]
    cov = score_part_coverage(answer, parts)
    assert cov.n_answered == 1


def test_uncovered_part_flagged_via_absence_note_anchor():
    """absence_note pairing: the part_text is a terse prompt that does not name
    the omitted subject; the authored absence_note does. A disclaimer phrased
    in the absence_note's vocabulary must still be credited as a correct flag."""
    answer = (
        "Split into two cases for x. The corpus does not cover shaded ray "
        "notation here."
    )
    parts = [
        _part("a", "Solve for x.", True,
              key_points=["split into two cases for x"]),
        _part("b", "Graph it.", False,
              absence_note="The corpus does not cover shaded ray notation; dry-run confirmed absent."),
    ]
    cov = score_part_coverage(answer, parts)
    assert cov.n_uncovered_parts == 1
    assert cov.n_correctly_flagged == 1
    assert cov.n_answered == 1
    flagged = {v.part_id: v.correctly_flagged for v in cov.verdicts}
    assert flagged["b"] is True


def test_uncovered_part_wrongly_answered_as_covered_not_flagged():
    """An answer that fabricates content for a covered:false part (no
    disclaimer) must NOT be credited as a correct flag — the metric punishes
    inventing the absent part."""
    answer = (
        "Split into two cases for x. The shaded ray points left from "
        "negative two on the number line."
    )
    parts = [
        _part("a", "Solve for x.", True,
              key_points=["split into two cases for x"]),
        _part("b", "Graph the shaded ray.", False,
              absence_note="The corpus does not cover shaded ray notation; dry-run confirmed absent."),
    ]
    cov = score_part_coverage(answer, parts)
    assert cov.n_uncovered_parts == 1
    assert cov.n_correctly_flagged == 0
    assert cov.verdicts[1].correctly_flagged is False


# ===========================================================================
# Part coverage — uncovered parts correctly_flagged (ws3.v2 disclaimer)
# ===========================================================================

def test_uncovered_part_correctly_flagged_when_disclaimer_near_noun():
    answer = (
        "The course explains photosynthesis in detail. It does not cover "
        "cellular respiration anywhere in the material."
    )
    parts = [
        _part("a", "explain photosynthesis", True),
        _part(
            "b", "explain cellular respiration", False,
            absence_note="The corpus deliberately omits respiration; dry-run confirmed absent.",
        ),
    ]
    cov = score_part_coverage(answer, parts)
    assert cov.n_uncovered_parts == 1
    assert cov.n_correctly_flagged == 1
    flagged = {v.part_id: v.correctly_flagged for v in cov.verdicts}
    assert flagged["b"] is True


def test_uncovered_part_not_flagged_when_no_disclaimer():
    # Answer just answers part A and stops — no acknowledgment of B's absence.
    answer = "Photosynthesis converts light into chemical energy in chloroplasts."
    parts = [
        _part("a", "explain photosynthesis", True),
        _part("b", "explain cellular respiration", False,
              absence_note="Respiration is out of scope; confirmed absent by dry-run."),
    ]
    cov = score_part_coverage(answer, parts)
    assert cov.n_correctly_flagged == 0
    assert cov.verdicts[1].correctly_flagged is False


def test_disclaimer_proximity_boundary_brittleness():
    """R7 brittleness boundary: a disclaimer about a DIFFERENT, far-away topic
    must not be credited to this part (proximity window guards it)."""
    # Disclaimer ("does not cover") is about 'kinematics', far from 'respiration'.
    far = "x " * 120
    answer = (
        "The course does not cover kinematics. " + far +
        "Separately, here is a long aside about respiration biology mechanisms."
    )
    parts = [_part("b", "explain cellular respiration", False,
                   absence_note="Respiration is out of scope; dry-run confirmed absent.")]
    cov = score_part_coverage(answer, parts)
    # 'respiration' token is > proximity window away from the 'does not cover'
    # phrase → NOT credited as a correct flag (honest about the heuristic).
    assert cov.n_correctly_flagged == 0


def test_disclaimer_phrase_variants_detected():
    for phrase in ("not covered", "outside the scope", "no information"):
        answer = f"Photosynthesis is explained. Respiration is {phrase} in this corpus."
        parts = [_part("b", "explain respiration", False,
                       absence_note="Out of scope; dry-run confirmed absent here.")]
        cov = score_part_coverage(answer, parts)
        assert cov.n_correctly_flagged == 1, f"phrase {phrase!r} not detected"


def test_disclaimer_phrases_constant_nonempty():
    assert len(DISCLAIMER_PHRASES) >= 5
    assert "does not cover" in DISCLAIMER_PHRASES


# ===========================================================================
# Per-population citation breakdown
# ===========================================================================

def _chunk(cid, *, source_refs=False, item_path="week_01/page.html"):
    src = {"item_path": item_path}
    if source_refs:
        src["source_references"] = [{"block_id": "b1"}]
    return {"id": cid, "text": "t", "source": src}


def test_population_breakdown_source_vs_course():
    chunks = {
        "s1": _chunk("s1", source_refs=True, item_path="textbook.html"),
        "c1": _chunk("c1", item_path="week_02/page.html"),
        "c2": _chunk("c2", item_path="week_03/page.html"),
    }
    pb = score_population_breakdown(
        ["s1", "c1", "c2"], chunks, expected_population="any"
    )
    assert pb.cited_source == 1
    assert pb.cited_course == 2
    assert pb.expected_satisfied is None  # 'any' → no constraint


def test_population_expected_source_satisfied():
    chunks = {"s1": _chunk("s1", source_refs=True, item_path="textbook.html")}
    pb = score_population_breakdown(["s1"], chunks, expected_population="source")
    assert pb.expected_satisfied is True


def test_population_expected_source_unsatisfied():
    chunks = {"c1": _chunk("c1", item_path="week_01/page.html")}
    pb = score_population_breakdown(["c1"], chunks, expected_population="source")
    assert pb.expected_satisfied is False


def test_population_expected_both_requires_each():
    chunks = {
        "s1": _chunk("s1", source_refs=True, item_path="textbook.html"),
        "c1": _chunk("c1", item_path="week_01/page.html"),
    }
    assert score_population_breakdown(
        ["s1", "c1"], chunks, expected_population="both"
    ).expected_satisfied is True
    assert score_population_breakdown(
        ["s1"], chunks, expected_population="both"
    ).expected_satisfied is False


def test_population_unanswered_satisfaction_none():
    chunks = {"c1": _chunk("c1")}
    pb = score_population_breakdown(
        ["c1"], chunks, expected_population="course", answered=False
    )
    assert pb.expected_satisfied is None


def test_population_unknown_id_skipped():
    chunks = {"c1": _chunk("c1")}
    pb = score_population_breakdown(["c1", "ghost"], chunks)
    assert pb.cited_course == 1
    assert pb.cited_source == 0
