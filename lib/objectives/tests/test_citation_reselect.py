"""Deterministic post-hoc CO citation re-selection invariants.

``lib/objectives/citation_reselect.py::reselect_citations`` fixes the measured
7B neighbor-citation sloppiness: for each CO it re-ranks the chunk pool the
model SAW (window ∪ chapter ∪ cited) by cosine(statement, chunk text) and
re-cites the strongest supporters. Hermetic — ``FakeEmbed`` (token-hash
unit vectors, no torch/numpy), no LLM, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.objectives.citation_reselect import (  # noqa: E402
    ENV_CITATION_RESELECT,
    ENV_RESELECT_EXERCISE_DEMOTE,
    ENV_RESELECT_KEEP_ORIGINAL,
    _is_exercise_like,
    reselect_citations,
    resolve_citation_reselect,
    resolve_reselect_exercise_demote,
    resolve_reselect_keep_original,
)
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402

import copy  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture corpus: chunk c_intro is a low-relevance neighbor; c_subst matches
# the CO statement's vocabulary (token-hash cosine ≈ high); c_photo is
# off-topic (different vocabulary).
# ---------------------------------------------------------------------------
def _chunks() -> Dict[str, Dict[str, Any]]:
    return {
        "c_intro": {
            "id": "c_intro",
            "text": "welcome chapter introduction overview reading outline",
            "chapter_id": "ch5",
        },
        "c_subst": {
            "id": "c_subst",
            "text": (
                "solve linear equations using the substitution method "
                "substitute the expression solve equations"
            ),
            "chapter_id": "ch5",
        },
        "c_photo": {
            "id": "c_photo",
            "text": "photosynthesis chloroplast light energy pigment leaf",
            "chapter_id": "ch5",
        },
        "c_extra": {
            "id": "c_extra",
            "text": (
                "solve equations substitution method practice problems "
                "substitute values solve"
            ),
            "chapter_id": "ch5",
        },
    }


def _co(statement: str, cited: List[str]) -> Dict[str, Any]:
    return {
        "statement": statement,
        "source_chunk_ids": list(cited),
        "source_refs": [{"ref": "ch5", "chunk_ids": list(cited)}],
    }


_STMT = "Solve linear equations using the substitution method."


def _run(
    cos, chunks, *, window_pools=None, chapter_pools=None, embed=None, **kw,
):
    return reselect_citations(
        cos,
        chunks,
        FakeEmbed() if embed is None else embed,
        window_chunk_ids_by_co=window_pools,
        chapter_chunks_by_co=chapter_pools,
        enabled=True,
        **kw,
    )


# ---------------------------------------------------------------------------
# (a) neighbor-miss corrected
# ---------------------------------------------------------------------------
def test_neighbor_miss_corrected():
    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])  # 7B cited the chapter-intro neighbor
    res = _run(
        [co], chunks,
        window_pools={0: ["c_intro", "c_subst", "c_photo"]},
    )
    assert res.available
    assert res.reselected_count == 1
    assert co["source_chunk_ids"][0] == "c_subst"  # best supporter re-cited
    assert "c_photo" not in co["source_chunk_ids"]  # off-topic stays out
    # source_refs mirrored, ref label preserved.
    assert co["source_refs"] == [
        {"ref": "ch5", "chunk_ids": co["source_chunk_ids"]}
    ]
    # Per-CO change record carries old→new + cosines.
    change = res.per_co_changes[0]
    assert change["old"] == ["c_intro"]
    assert change["new"][0] == "c_subst"
    assert change["new_best_cosine"] > (change["old_best_cosine"] or 0.0)


# ---------------------------------------------------------------------------
# (b) floor respected — nothing clears → original kept verbatim
# ---------------------------------------------------------------------------
def test_floor_keeps_original_when_nothing_clears():
    chunks = _chunks()
    co = _co("Discuss maritime history of ancient trade routes.", ["c_intro"])
    res = _run(
        [co], chunks,
        window_pools={0: ["c_intro", "c_photo"]},
        floor=0.9,  # unreachable for disjoint-vocab texts
    )
    assert res.available
    assert res.reselected_count == 0
    assert res.kept_original_below_floor == 1
    assert co["source_chunk_ids"] == ["c_intro"]  # untouched
    assert co["source_refs"] == [{"ref": "ch5", "chunk_ids": ["c_intro"]}]


# ---------------------------------------------------------------------------
# (c) K cap
# ---------------------------------------------------------------------------
def test_top_k_cap():
    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])
    res = _run(
        [co], chunks,
        window_pools={0: ["c_intro", "c_subst", "c_extra", "c_photo"]},
        floor=0.0,   # everything clears (cosine >= 0 for bag-of-words)
        max_chunks=1,
        # Pin the pre-guard REPLACE semantics: this test isolates the cap, and
        # floor=0.0 makes the off-topic ``c_intro`` original "above-floor" (the
        # default keep-original guard would retain it). See the dedicated
        # keep-original tests below for the guard's behavior.
        keep_original=False,
    )
    assert res.available
    assert len(co["source_chunk_ids"]) == 1
    assert co["source_chunk_ids"][0] in {"c_subst", "c_extra"}  # a supporter


# ---------------------------------------------------------------------------
# (d) pool restriction — out-of-pool higher-cosine chunk is NOT cited
# ---------------------------------------------------------------------------
def test_pool_restriction_excludes_out_of_pool_chunk():
    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])
    # c_subst / c_extra are NOT in this CO's pool (different window) — even
    # though they'd out-cosine everything, they must not be cited.
    res = _run(
        [co], chunks,
        window_pools={0: ["c_intro", "c_photo"]},
        floor=0.0,
    )
    assert res.available
    assert "c_subst" not in co["source_chunk_ids"]
    assert "c_extra" not in co["source_chunk_ids"]
    assert set(co["source_chunk_ids"]) <= {"c_intro", "c_photo"}


def test_unresolvable_pool_ids_dropped_and_counted():
    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])
    res = _run(
        [co], chunks,
        window_pools={0: ["c_intro", "c_subst", "ghost-chunk-404"]},
    )
    assert res.pool_misses == 1
    assert "ghost-chunk-404" not in co["source_chunk_ids"]  # never invented


# ---------------------------------------------------------------------------
# (e) flag off → byte-identical
# ---------------------------------------------------------------------------
def test_flag_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv(ENV_CITATION_RESELECT, raising=False)
    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])
    before = copy.deepcopy(co)
    res = reselect_citations(
        [co], chunks, FakeEmbed(),
        window_chunk_ids_by_co={0: ["c_intro", "c_subst"]},
    )
    assert res.available is False
    assert co == before  # byte-identical


def test_resolver_parse_with_fallback(monkeypatch):
    monkeypatch.delenv(ENV_CITATION_RESELECT, raising=False)
    assert resolve_citation_reselect() is False  # default OFF
    for tok in ("1", "true", "YES", "on"):
        monkeypatch.setenv(ENV_CITATION_RESELECT, tok)
        assert resolve_citation_reselect() is True
    for tok in ("0", "false", "off", "garbage", ""):
        monkeypatch.setenv(ENV_CITATION_RESELECT, tok)
        assert resolve_citation_reselect() is False
    assert resolve_citation_reselect(True) is True  # explicit arg wins


# ---------------------------------------------------------------------------
# (f) density counters
# ---------------------------------------------------------------------------
def test_density_counters():
    chunks = _chunks()
    co1 = _co(_STMT, ["c_intro"])
    co2 = _co(
        "Solve equations by substitution and check the solution.",
        ["c_intro"],
    )
    res = _run(
        [co1, co2], chunks,
        window_pools={
            0: ["c_intro", "c_subst"],
            1: ["c_intro", "c_extra"],
        },
        floor=0.0,
        max_chunks=2,
    )
    assert res.available
    assert res.citation_density_before == 1  # both cited only c_intro
    # After: c_subst + c_extra (+ possibly c_intro) — strictly denser.
    assert res.citation_density_after > res.citation_density_before


# ---------------------------------------------------------------------------
# Zero-citation / degrade paths
# ---------------------------------------------------------------------------
def test_zero_citation_cos_skipped():
    chunks = _chunks()
    co = {"statement": _STMT, "source_chunk_ids": []}
    res = _run([co], chunks, window_pools={})
    assert res.skipped_no_citation == 1
    assert co["source_chunk_ids"] == []  # never adds citations from nothing


def test_embed_absent_is_logged_noop():
    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])
    res = reselect_citations(
        [co], chunks, None,
        window_chunk_ids_by_co={0: ["c_intro", "c_subst"]},
        enabled=True,
    )
    assert res.available is False
    assert co["source_chunk_ids"] == ["c_intro"]


def test_chapter_fallback_pool_used_when_no_window():
    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])
    res = _run(
        [co], chunks,
        window_pools={},  # window unresolvable (e.g. backfill-promoted CO)
        chapter_pools={0: ["c_intro", "c_subst", "c_photo"]},
    )
    assert res.available
    assert co["source_chunk_ids"][0] == "c_subst"


def test_set_identical_reorder_not_counted_no_capture():
    """FIX 2: a cosine-REORDERED but SET-IDENTICAL keep is a no-op — it must
    not be counted as a re-selection nor emit a decision-capture event.

    ``c_subst`` out-cosines ``c_extra`` to ``_STMT``, so the re-rank flips the
    LIST order of the cited pair; the SET is unchanged. Under the old
    ``kept == old_ids`` list compare this counted as a change (+capture); the
    set-based check skips it.
    """
    chunks = _chunks()
    # Cite in the LOW-cosine-first order so the cosine re-rank reorders them.
    co = _co(_STMT, ["c_extra", "c_subst"])
    events: List[Dict[str, Any]] = []

    class _Cap:
        def log_decision(self, **kw):
            events.append(kw)

    res = _run(
        [co], chunks,
        window_pools={0: ["c_extra", "c_subst"]},
        floor=0.0,
        max_chunks=2,
        capture=_Cap(),
    )
    assert res.available
    assert res.reselected_count == 0          # set-identical → not a change
    assert events == []                       # no capture for a no-op
    # Not rewritten: original list order survives verbatim.
    assert co["source_chunk_ids"] == ["c_extra", "c_subst"]
    assert co["source_refs"] == [
        {"ref": "ch5", "chunk_ids": ["c_extra", "c_subst"]}
    ]


def test_multi_chapter_kept_set_groups_source_refs_per_chapter():
    """FIX 3: when the kept set spans >1 chapter, ``source_refs`` carries one
    ``{ref, chunk_ids}`` entry per chapter (grouped by the kept chunks'
    ``chapter_id``), preserving kept order — never smearing one label."""
    chunks = {
        "c_intro": {
            "id": "c_intro",
            "text": "welcome chapter introduction overview reading",
            "chapter_id": "ch5",
        },
        "c_subst": {
            "id": "c_subst",
            "text": (
                "solve linear equations using the substitution method "
                "substitute the expression solve equations"
            ),
            "chapter_id": "ch5",
        },
        "c_solve6": {  # a same-topic supporter in a DIFFERENT chapter
            "id": "c_solve6",
            "text": "solve equations method",
            "chapter_id": "ch6",
        },
    }
    co = _co(_STMT, ["c_intro"])
    res = _run(
        [co], chunks,
        window_pools={0: ["c_intro", "c_subst", "c_solve6"]},
        floor=0.0,
        max_chunks=2,
        # Isolate the multi-chapter source_refs grouping: floor=0.0 makes the
        # off-topic ``c_intro`` original above-floor, which the default
        # keep-original guard would retain (asserted separately below).
        keep_original=False,
    )
    assert res.available
    assert res.reselected_count == 1
    # Both supporters re-cited; c_subst (ch5) out-cosines c_solve6 (ch6).
    assert co["source_chunk_ids"] == ["c_subst", "c_solve6"]
    # One source_refs entry per chapter, in kept order.
    assert co["source_refs"] == [
        {"ref": "ch5", "chunk_ids": ["c_subst"]},
        {"ref": "ch6", "chunk_ids": ["c_solve6"]},
    ]


def test_widened_pool_reaches_chapter_supporter_out_of_window():
    """FIX 1 (module consumption): the true supporter sits in the CO's CHAPTER
    but not in its (post-prune) window. The hook now merges the chapter bucket
    into the pool passed via ``window_chunk_ids_by_co``; the supporter is
    reached. (Contrast ``chapter_chunks_by_co`` — a WINDOW-ONLY fallback — which
    is NOT consulted when the window pool is non-empty, i.e. the exact hole.)"""
    chunks = _chunks()
    # OLD shape: non-empty (wrong) window pool + chapter as fallback-only →
    # supporter unreachable (documents the hole the hook now closes).
    co_old = _co(_STMT, ["c_intro"])
    _run(
        [co_old], chunks,
        window_pools={0: ["c_intro", "c_photo"]},          # wrong window
        chapter_pools={0: ["c_intro", "c_subst", "c_photo"]},  # ignored
        floor=0.0,
    )
    assert "c_subst" not in co_old["source_chunk_ids"]
    # NEW shape: the hook passes window ∪ chapter as the pool → supporter re-cited.
    co_new = _co(_STMT, ["c_intro"])
    res = _run(
        [co_new], chunks,
        window_pools={0: ["c_intro", "c_photo", "c_subst"]},
    )
    assert res.available
    assert co_new["source_chunk_ids"][0] == "c_subst"


def test_capture_reuses_objective_chunk_prune(monkeypatch):
    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])
    events: List[Dict[str, Any]] = []

    class _Cap:
        def log_decision(self, **kw):
            events.append(kw)

    _run(
        [co], chunks,
        window_pools={0: ["c_intro", "c_subst"]},
        capture=_Cap(),
    )
    assert len(events) == 1
    assert events[0]["decision_type"] == "objective_chunk_prune"  # reused
    assert "c_subst" in events[0]["decision"]
    assert len(events[0]["rationale"]) >= 20


# ---------------------------------------------------------------------------
# numpy-vector regression: the REAL embed client returns np.ndarray vectors
# (2D batch). Truthiness (`not vec` / `vec or []`) on those raises "truth
# value of an array with more than one element is ambiguous" — the exact
# failure that killed run WF-00000000-00000000 attempt 1-3 (2026-07-02).
# ---------------------------------------------------------------------------
def test_numpy_batch_vectors_do_not_raise():
    np = __import__("pytest").importorskip("numpy")

    class NumpyEmbed:
        """FakeEmbed semantics, ndarray return shape (real-client parity)."""

        def __init__(self):
            self._inner = FakeEmbed()

        def encode_batch(self, texts):
            return np.asarray(self._inner.encode_batch(texts), dtype=np.float64)

    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])
    res = _run(
        [co], chunks,
        window_pools={0: ["c_intro", "c_subst", "c_photo"]},
        embed=NumpyEmbed(),
    )
    assert res.available
    assert res.reselected_count == 1
    assert co["source_chunk_ids"][0] == "c_subst"


# ---------------------------------------------------------------------------
# Exercise-chunk demotion (ranking-quality bug fix)
# ---------------------------------------------------------------------------
_PLACE_STMT = "Identify the place value of each digit in a given number."


def _place_chunks() -> Dict[str, Dict[str, Any]]:
    """``c_ex`` nearly QUOTES the CO statement (highest cosine) but is an
    end-of-chapter exercise/answer-list; ``c_prose`` is instructional prose
    that shares fewer tokens (lower cosine)."""
    return {
        "c_ex": {
            "id": "c_ex",
            "text": (
                "Use place value with whole numbers In the following "
                "exercises identify the place value of each digit in the "
                "given number 1. 51,493 ⓐ ones 2. 3,491 ⓑ tens 3. 812 ⓒ"
            ),
            "chapter_id": "ch1",
        },
        "c_prose": {
            "id": "c_prose",
            "text": (
                "place value describes what each digit represents based on "
                "its position within the number"
            ),
            "chapter_id": "ch1",
        },
    }


def test_exercise_chunk_demoted_below_instructional():
    """An exercise/answer-list chunk that out-cosines an instructional chunk
    is DEMOTED below it: the instructional prose is cited, not the answer list.
    The opt-out path (pure cosine) proves the exercise chunk really wins on
    cosine alone."""
    # Opt-out → pure cosine → the exercise chunk (higher cosine) wins.
    co_pure = _co(_PLACE_STMT, ["c_prose"])
    res_pure = _run(
        [co_pure], _place_chunks(),
        window_pools={0: ["c_ex", "c_prose"]},
        floor=0.0, max_chunks=1, exercise_demote=False,
        # Isolate pure-cosine: floor=0.0 makes the ``c_prose`` original
        # above-floor, which the default keep-original guard would retain.
        keep_original=False,
    )
    assert res_pure.available
    assert co_pure["source_chunk_ids"] == ["c_ex"]  # higher cosine wins
    assert res_pure.exercise_demoted_total == 0     # demotion off → no demote

    # Default (demote ON) → the exercise chunk is demoted, prose is cited.
    # The 7B sloppily cited the exercise chunk; re-selection flips it to prose.
    co = _co(_PLACE_STMT, ["c_ex"])
    res = _run(
        [co], _place_chunks(),
        window_pools={0: ["c_ex", "c_prose"]},
        floor=0.0, max_chunks=1,
    )
    assert res.available
    assert res.reselected_count == 1
    assert co["source_chunk_ids"] == ["c_prose"]  # instructional wins
    assert res.exercise_demoted_total == 1
    change = res.per_co_changes[0]
    assert change["exercise_demoted"] == 1


def test_all_exercise_pool_still_cites_no_starvation():
    """When EVERY above-floor chunk is exercise-like, demotion never starves
    the CO — the top chunk by cosine is still cited (demote != exclude)."""
    chunks = {
        "c_ex1": {
            "id": "c_ex1",
            "text": (
                "In the following exercises identify the place value of each "
                "digit 1. 51,493 2. 3,491 3. 812"
            ),
            "chapter_id": "ch1",
        },
        "c_ex2": {
            "id": "c_ex2",
            "text": (
                "In the following exercises round each number to the given "
                "place 1. 640 2. 903 3. 27"
            ),
            "chapter_id": "ch1",
        },
    }
    assert _is_exercise_like(chunks["c_ex1"]["text"])
    assert _is_exercise_like(chunks["c_ex2"]["text"])
    co = _co(_PLACE_STMT, ["c_ex2"])
    res = _run(
        [co], chunks,
        window_pools={0: ["c_ex1", "c_ex2"]},
        floor=0.0, max_chunks=1,
    )
    assert res.available
    assert len(co["source_chunk_ids"]) == 1        # still cited, not starved
    assert co["source_chunk_ids"][0] in {"c_ex1", "c_ex2"}
    assert res.exercise_demoted_total == 0         # no non-exercise to demote past


def test_exercise_demote_opt_out_via_env(monkeypatch):
    """The ``ED4ALL_OBJECTIVE_RESELECT_EXERCISE_DEMOTE`` opt-out restores the
    pure-cosine rank (the exercise chunk wins)."""
    monkeypatch.setenv(ENV_RESELECT_EXERCISE_DEMOTE, "0")
    co = _co(_PLACE_STMT, ["c_prose"])
    res = reselect_citations(
        [co], _place_chunks(), FakeEmbed(),
        window_chunk_ids_by_co={0: ["c_ex", "c_prose"]},
        floor=0.0, max_chunks=1, enabled=True,   # env drives exercise_demote
        # Isolate the demote opt-out: ``c_prose`` is a non-exercise original the
        # default keep-original guard would retain (asserted separately below).
        keep_original=False,
    )
    assert res.available
    assert co["source_chunk_ids"] == ["c_ex"]  # demotion suppressed
    assert res.exercise_demoted_total == 0


# ---------------------------------------------------------------------------
# Detector unit cases
# ---------------------------------------------------------------------------
def test_is_exercise_like_signals():
    # A — leading "In the following exercises".
    assert _is_exercise_like(
        "In the following exercises, find the place value of each digit."
    )
    # B — >=3 circled answer glyphs.
    assert _is_exercise_like("Choose the correct option ⓐ five ⓑ six ⓒ seven")
    assert not _is_exercise_like("A single circled marker ⓐ in prose")  # <3
    # C — dense >=3 numbered answer-list run.
    assert _is_exercise_like("1. 51,493 2. 3,491 3. 812 4. 7,000")
    # D — exercise-section banners.
    assert _is_exercise_like("EXERCISES Practice Makes Perfect")   # shared RE
    assert _is_exercise_like("Section Exercises: apply the concept")
    assert _is_exercise_like("Review Exercises for the chapter")
    # Legit instructional prose with ONE enumeration must NOT trip.
    assert not _is_exercise_like(
        "To find a place value, follow this step. 1. Locate the digit and "
        "read its column. Place value tells you the size of each digit."
    )
    # Decimals / section numbers do not read as answer-list markers.
    assert not _is_exercise_like(
        "The value 3.5 differs from 4.2; see section 1.1 for the rule."
    )
    # Empty / whitespace.
    assert not _is_exercise_like("")
    assert not _is_exercise_like(None)


def test_resolve_reselect_exercise_demote(monkeypatch):
    monkeypatch.delenv(ENV_RESELECT_EXERCISE_DEMOTE, raising=False)
    assert resolve_reselect_exercise_demote() is True   # default ON
    for tok in ("0", "false", "OFF", "no"):
        monkeypatch.setenv(ENV_RESELECT_EXERCISE_DEMOTE, tok)
        assert resolve_reselect_exercise_demote() is False
    for tok in ("1", "true", "on", "garbage", ""):
        monkeypatch.setenv(ENV_RESELECT_EXERCISE_DEMOTE, tok)
        assert resolve_reselect_exercise_demote() is True  # fallback ON
    assert resolve_reselect_exercise_demote(False) is False  # explicit arg wins


def test_resolve_reselect_keep_original(monkeypatch):
    monkeypatch.delenv(ENV_RESELECT_KEEP_ORIGINAL, raising=False)
    assert resolve_reselect_keep_original() is True   # default ON
    for tok in ("0", "false", "OFF", "no"):
        monkeypatch.setenv(ENV_RESELECT_KEEP_ORIGINAL, tok)
        assert resolve_reselect_keep_original() is False
    for tok in ("1", "true", "on", "garbage", ""):
        monkeypatch.setenv(ENV_RESELECT_KEEP_ORIGINAL, tok)
        assert resolve_reselect_keep_original() is True  # fallback ON
    assert resolve_reselect_keep_original(False) is False  # explicit arg wins


# ---------------------------------------------------------------------------
# Keep-original guard — entailment-regression bug fix (CO-21 shape).
#
# Reproduces the sample-full-obj-01 CO-21 collapse: a chapter-level CO cited the
# section chunk that ENTAILS its statement (states both named properties), but a
# FOREIGN chunk (lexically similar — shares "solve / linear / using / properties"
# — yet about a different topic) out-cosines it. Under the pure top-K REPLACE the
# foreign chunk evicts the true supporter → the per-LO NLI entailment gate (which
# scores the statement against the UNION of cited chunk text) sees 0%. The guard
# UNIONS the above-floor original into the kept set so it can never be stripped.
# ---------------------------------------------------------------------------
_INEQ_STMT = (
    "solve linear inequalities using the subtraction and addition "
    "properties of inequality"
)


def _ineq_chunks():
    """``c_support`` ENTAILS the statement (names both inequality properties)
    but shares FEWER surface tokens with it than the off-topic ``c_foreign*``
    chunks, which are about solving *equations* yet quote more of the statement's
    wording — so pure cosine ranks the foreign chunks above the true supporter."""
    return {
        "c_support": {
            "id": "c_support",
            "text": (
                "subtraction property of inequality and addition property of "
                "inequality inequality inequality"
            ),
            "chapter_id": "ch2",
        },
        "c_foreign1": {
            "id": "c_foreign1",
            "text": (
                "solve linear equations using the subtraction and addition "
                "properties of equality combine like terms"
            ),
            "chapter_id": "ch2",
        },
        "c_foreign2": {
            "id": "c_foreign2",
            "text": (
                "solve linear equations using the subtraction and addition "
                "properties distributive property simplify each side"
            ),
            "chapter_id": "ch2",
        },
    }


def test_keep_original_default_confirms_foreign_out_cosines_supporter():
    """Sanity: with the guard OFF (pre-fix REPLACE), the foreign equation chunks
    out-cosine the true supporter and EVICT it — the exact 0%-entailment bug."""
    co = _co(_INEQ_STMT, ["c_support"])
    res = _run(
        [co], _ineq_chunks(),
        window_pools={0: ["c_support", "c_foreign1", "c_foreign2"]},
        max_chunks=2,
        keep_original=False,  # pre-fix pure-cosine REPLACE
    )
    assert res.available
    # The model's entailing supporter is STRIPPED (bug reproduced).
    assert "c_support" not in co["source_chunk_ids"]
    assert set(co["source_chunk_ids"]) <= {"c_foreign1", "c_foreign2"}


def test_keep_original_retains_entailing_supporter():
    """With the guard ON (default), the above-floor non-exercise original is
    UNIONED into the kept set — the true supporter survives even though the
    foreign chunks out-cosine it, so the premise union still entails."""
    chunks = _ineq_chunks()
    co = _co(_INEQ_STMT, ["c_support"])
    res = _run(
        [co], chunks,
        window_pools={0: ["c_support", "c_foreign1", "c_foreign2"]},
        max_chunks=2,
        # keep_original defaults ON
    )
    assert res.available
    assert res.reselected_count == 1                 # set changed (added a pick)
    assert "c_support" in co["source_chunk_ids"]     # supporter RETAINED
    assert res.kept_original_supporters >= 1
    # Union, not pure-keep: a better-cosine foreign pick is also ADDED.
    assert len(co["source_chunk_ids"]) == 2
    assert set(co["source_chunk_ids"]) & {"c_foreign1", "c_foreign2"}
    # source_refs mirrored onto the kept set.
    assert co["source_refs"][0]["chunk_ids"] == co["source_chunk_ids"]


def test_keep_original_widens_cap_so_original_never_squeezed_out():
    """When every above-floor pick would fill the cap, a protected original is
    still retained: the cap is widened to max(cap, n_protected)."""
    chunks = _ineq_chunks()
    co = _co(_INEQ_STMT, ["c_support"])
    res = _run(
        [co], chunks,
        window_pools={0: ["c_support", "c_foreign1", "c_foreign2"]},
        max_chunks=1,   # cap 1 — pure REPLACE would keep only the top foreign
    )
    assert res.available
    assert "c_support" in co["source_chunk_ids"]  # not squeezed out by the cap


def test_keep_original_does_not_protect_below_floor_original():
    """A BELOW-floor original (off-topic neighbor) is NOT protected — the guard
    only retains originals that themselves clear the relevance floor."""
    chunks = _chunks()
    co = _co(_STMT, ["c_intro"])   # c_intro is off-topic, below the 0.30 floor
    res = _run(
        [co], chunks,
        window_pools={0: ["c_intro", "c_subst"]},   # default floor 0.30
    )
    assert res.available
    assert co["source_chunk_ids"][0] == "c_subst"   # supporter re-cited
    assert "c_intro" not in co["source_chunk_ids"]  # below-floor → droppable


def test_keep_original_does_not_protect_exercise_like_original():
    """An EXERCISE-like original is NOT protected — the guard defers to the
    exercise-demotion contract so answer-list originals stay droppable."""
    co = _co(_PLACE_STMT, ["c_ex"])   # exercise/answer-list original
    res = _run(
        [co], _place_chunks(),
        window_pools={0: ["c_ex", "c_prose"]},
        floor=0.0, max_chunks=1,
        # keep_original defaults ON; exercise-demote also defaults ON.
    )
    assert res.available
    assert co["source_chunk_ids"] == ["c_prose"]     # instructional wins
    assert "c_ex" not in co["source_chunk_ids"]      # exercise original dropped


# ---------------------------------------------------------------------------
# 2026-07-04 pin — the Pass-C entailing chunk is never stripped
# ---------------------------------------------------------------------------
def test_entailing_chunk_pin_never_stripped():
    """An original citation carrying the ``entailing_chunk_id`` stamp survives
    re-selection even when it is below the cosine floor (the keep-original
    guard alone would strip it — it only protects ABOVE-floor originals)."""
    chunks = _chunks()
    # The entailing chunk: vocabulary disjoint from the statement, so its
    # token-hash cosine is ~0 (below any floor) — the sample-scan worked-example
    # shape.
    chunks["c_entail"] = {
        "id": "c_entail",
        "text": "sqrt frac latex notation qquad cdot worked example",
        "chapter_id": "ch5",
    }
    co = _co(_STMT, ["c_entail", "c_intro"])
    co["entailing_chunk_id"] = "c_entail"
    cos = [co]
    result = _run(
        cos, chunks,
        window_pools={0: ["c_entail", "c_intro", "c_subst", "c_extra"]},
    )
    assert result.available is True
    kept = cos[0]["source_chunk_ids"]
    # The cosine top-K re-cite happened (strong supporters added)...
    assert "c_subst" in kept
    # ...but the below-floor entailing original was NOT stripped.
    assert "c_entail" in kept


def test_pin_absent_below_floor_original_still_droppable():
    """Without the stamp, a below-floor original stays droppable (legacy)."""
    chunks = _chunks()
    chunks["c_weak"] = {
        "id": "c_weak",
        "text": "sqrt frac latex notation qquad cdot worked example",
        "chapter_id": "ch5",
    }
    cos = [_co(_STMT, ["c_weak", "c_intro"])]
    result = _run(
        cos, chunks,
        window_pools={0: ["c_weak", "c_intro", "c_subst", "c_extra"]},
    )
    assert result.available is True
    kept = cos[0]["source_chunk_ids"]
    assert "c_subst" in kept
    # Below-floor, unstamped original dropped exactly as before.
    assert "c_weak" not in kept
