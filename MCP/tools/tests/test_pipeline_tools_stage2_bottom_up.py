"""WS1 — bottom-up terminal-objective derivation (KEYSTONE) invariants.

Tests the module-level seam helpers in
``MCP.tools.pipeline_tools`` directly (the cleanest seam — the surrounding
``_run_stage2_window_synthesis`` coroutine is exercised end-to-end elsewhere):

- ``_derive_terminals_bottom_up`` — clusters grounded COs and authors one TO
  per cluster, setting each CO's ``terminal_id`` directly (no orphans).
- ``_clamp_cluster_count`` — clamps to ``[3, 15]`` without dropping a CO.
- ``_fallback_terminal_from_cluster`` — per-cluster author failure degrades to
  a verbatim CO subset (never invents).
- The WS1/WS2 backlink integration: backlink is a no-op for ASSIGNMENT on the
  bottom-up path (terminal_id pre-set) but the legacy fallback path drives the
  real argmax assignment.
- The R4 embed-absent fallback to the legacy reconcile.

Hermetic — uses the ``FakeEmbed`` stub (no numpy / torch) + a fake provider.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402


class _ProviderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Authors one TO per cluster; records every call.

    ``author_terminal_for_cluster`` returns a TO whose statement is the longest
    member statement (deterministic), or ``None`` for any cluster_index in
    ``fail_on`` (per-cluster fail-soft path). ``mint_lo_id``-style ids are NOT
    set here (the caller mints).
    """

    def __init__(self, *, fail_on: set | None = None) -> None:
        self.fail_on = fail_on or set()
        self.author_calls: List[int] = []
        self.reconcile_calls = 0

    def author_terminal_for_cluster(
        self,
        cluster_cos: List[Dict[str, Any]],
        *,
        course_name: str,
        cluster_index: int,
    ) -> Dict[str, Any] | None:
        self.author_calls.append(cluster_index)
        if cluster_index in self.fail_on:
            return None
        rep = max(
            cluster_cos, key=lambda c: len(str(c.get("statement") or ""))
        )
        return {
            "statement": f"TO summarising: {rep.get('statement')}",
            "bloom_level": "understand",
            "source_refs": [],
        }

    def reconcile_terminal_objectives(
        self,
        draft_tos: List[Dict[str, Any]],
        chapter_cos: List[Dict[str, Any]],
        *,
        course_name: str,
    ) -> Dict[str, Any]:
        self.reconcile_calls += 1
        return {
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Legacy reconciled TO."}
            ]
        }


def _mint(kind: str, idx: int) -> str:
    prefix = {"terminal": "TO", "chapter": "CO"}.get(kind, "XX")
    return f"{prefix}-{idx:02d}"


def _co(statement: str) -> Dict[str, Any]:
    return {"statement": statement}


def _theme_cos() -> List[Dict[str, Any]]:
    """Three tight themes of COs (distinct vocab per theme) → ~3 clusters."""
    return [
        _co("Define a prime number and identify its factors."),
        _co("Define what a prime number is and list its factors."),
        _co("Compute the area of a triangle using base and height."),
        _co("Compute the area of a right triangle from base height."),
        _co("Photosynthesis converts light energy in the chloroplast."),
        _co("Photosynthesis uses light energy inside the chloroplast cell."),
    ]


# ---------------------------------------------------------------------------
# _derive_terminals_bottom_up
# ---------------------------------------------------------------------------


def test_bottom_up_derives_one_to_per_cluster():
    cos = _theme_cos()
    provider = _FakeProvider()
    terminals, signals = pt._derive_terminals_bottom_up(
        provider=provider,
        chapter_cos=cos,
        embed=FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
        capture=None,
    )
    # One TO per cluster; the three themes cluster distinctly at threshold 0.50.
    assert len(terminals) == len(provider.author_calls)
    assert len(terminals) >= 3
    # TO ids minted sequentially by the caller seam.
    assert [t["id"] for t in terminals] == [
        f"TO-{i:02d}" for i in range(1, len(terminals) + 1)
    ]
    assert signals["to_derivation"] == "bottom_up"
    assert signals["cluster_count"] == len(terminals)
    assert 0.0 <= signals["mean_intra_cluster_cosine"] <= 1.0
    # WS1.1 — TARGET-K Ward clustering signals replace to_cluster_threshold.
    assert "to_cluster_threshold" not in signals
    assert isinstance(signals["to_cluster_k"], int) and signals["to_cluster_k"] >= 1
    assert signals["to_cluster_linkage"] in {"ward", "single_link_fallback"}


def test_bottom_up_sets_terminal_id_directly_no_orphans():
    cos = _theme_cos()
    terminals, _signals = pt._derive_terminals_bottom_up(
        provider=_FakeProvider(),
        chapter_cos=cos,
        embed=FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
    )
    minted_ids = {t["id"] for t in terminals}
    # EVERY CO carries a terminal_id resolving to an emitted TO (no orphans).
    for co in cos:
        assert co.get("terminal_id") in minted_ids


def test_per_cluster_author_failure_uses_rep_fallback():
    cos = _theme_cos()
    # Fail the SECOND cluster → that TO falls back to the verbatim rep.
    provider = _FakeProvider(fail_on={2})
    terminals, _signals = pt._derive_terminals_bottom_up(
        provider=provider,
        chapter_cos=cos,
        embed=FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
    )
    fallback = [t for t in terminals if t.get("to_synthesis") == "cluster_rep_fallback"]
    assert len(fallback) == 1
    # The fallback statement is a VERBATIM CO statement (never invents).
    member_statements = {c["statement"] for c in cos}
    assert fallback[0]["statement"] in member_statements


def test_embed_encode_failure_returns_empty_for_legacy_fallback():
    """R4 — encode_batch raising → ([], {}) so the caller runs legacy reconcile."""

    class _BoomEmbed:
        def encode_batch(self, texts):
            raise RuntimeError("embed backend exploded")

    terminals, signals = pt._derive_terminals_bottom_up(
        provider=_FakeProvider(),
        chapter_cos=_theme_cos(),
        embed=_BoomEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
    )
    assert terminals == []
    assert signals == {}


# ---------------------------------------------------------------------------
# _clamp_cluster_count — never drops a CO
# ---------------------------------------------------------------------------


def _assert_partition(clusters: List[List[int]], n: int) -> None:
    seen = [i for c in clusters for i in c]
    assert sorted(seen) == list(range(n)), "every CO index in exactly one cluster"
    assert len(seen) == len(set(seen)), "no CO index duplicated across clusters"


def test_cluster_count_clamped_max_3_min_15():
    embed = FakeEmbed()

    # --- Over-target (n>15): 20 disjoint singletons → merged down to 15. ---
    many_texts = [f"distinctterm{i:02d} unique subject area heading" for i in range(20)]
    many_vecs = embed.encode_batch(many_texts)
    over = [[i] for i in range(20)]
    clamped_over = pt._clamp_cluster_count(over, many_vecs, target_lo=3, target_hi=15)
    assert len(clamped_over) <= 15
    _assert_partition(clamped_over, 20)

    # --- Under-target (n<3): one big cluster of distinct members → split up. ---
    split_texts = [
        "algebra linear equation solving",
        "geometry triangle area computation",
        "biology cell photosynthesis light",
        "chemistry atomic bonding electrons",
    ]
    split_vecs = embed.encode_batch(split_texts)
    under = [[0, 1, 2, 3]]  # all forced into one cluster
    clamped_under = pt._clamp_cluster_count(under, split_vecs, target_lo=3, target_hi=15)
    assert len(clamped_under) >= 3
    _assert_partition(clamped_under, 4)

    # --- In-range: unchanged + partition preserved. ---
    in_range = [[0, 1], [2], [3]]
    clamped_in = pt._clamp_cluster_count(in_range, split_vecs, target_lo=3, target_hi=15)
    _assert_partition(clamped_in, 4)


def test_clamp_fewer_cos_than_target_lo_does_not_fabricate():
    """n < target_lo with no splittable cluster → returned as-is (no empties)."""
    embed = FakeEmbed()
    vecs = embed.encode_batch(["only one statement here", "and a second one"])
    clusters = [[0], [1]]
    clamped = pt._clamp_cluster_count(clusters, vecs, target_lo=3, target_hi=15)
    # Cannot reach 3 from 2 singletons; never fabricates empty clusters.
    assert clamped == [[0], [1]]
    _assert_partition(clamped, 2)


# ---------------------------------------------------------------------------
# _fallback_terminal_from_cluster
# ---------------------------------------------------------------------------


def test_fallback_terminal_copies_longest_statement_verbatim():
    cluster = [
        _co("short."),
        _co("This is the longest chapter objective statement in the cluster."),
        _co("middle length statement."),
    ]
    to = pt._fallback_terminal_from_cluster(cluster)
    assert to["statement"] == (
        "This is the longest chapter objective statement in the cluster."
    )
    assert to["to_synthesis"] == "cluster_rep_fallback"
    assert to["source_refs"] == []


# ---------------------------------------------------------------------------
# WS1/WS2 backlink integration (assignment no-op bottom-up; real on fallback)
# ---------------------------------------------------------------------------


def test_backlink_is_assignment_noop_on_bottom_up_path():
    """After bottom-up sets terminal_id directly, backlink (which honors a
    resolvable pre-set terminal_id before cosine) leaves assignments intact."""
    from lib.ontology.lo_backlink import backlink_cos_to_tos

    cos = _theme_cos()
    terminals, _signals = pt._derive_terminals_bottom_up(
        provider=_FakeProvider(),
        chapter_cos=cos,
        embed=FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
    )
    before = [co["terminal_id"] for co in cos]
    # The legacy backlink is still invoked at the call site for the WS2 signal.
    result = backlink_cos_to_tos(terminals, cos, embed=FakeEmbed())
    after = [co["terminal_id"] for co in cos]
    # No reassignment — the pre-set resolvable terminal_id is honored.
    assert before == after
    # Pre-set (hint-honored) COs are NOT scored → weak rate stays 0.
    assert result.weak_link_rate == 0.0


# ---------------------------------------------------------------------------
# Call-site integration — _run_stage2_window_synthesis (chapter_fallback)
# ---------------------------------------------------------------------------


class _ChapterProvider(_FakeProvider):
    """Extends the fake with the chapter_fallback dispatch surface."""

    def batch_chapters(self, chapters, batch_size: int = 10):
        return [chapters[i:i + batch_size] for i in range(0, len(chapters), batch_size)]

    def synthesize_chapter_objectives(self, chapter, *, course_name, **_kw):
        cid = str(chapter.get("id") or "")
        return {
            "chapter_id": cid,
            "chapter_objectives": [
                {"statement": s, "chapter_id": cid}
                for s in chapter.get("_cos", [])
            ],
        }


def _run_stage2(provider, *, monkeypatch, embed):
    """Drive _run_stage2_window_synthesis in chapter_fallback mode, pinning the
    embed-client resolution to ``embed`` (None forces the legacy fallback)."""
    import lib.embedding.providers as _providers

    def _fake_build(*_a, **_k):
        if embed is None:
            raise _providers.EmbeddingBackendUnavailable("no extras in test")
        return embed

    # _run_stage2_window_synthesis imports build_embedding_client locally from
    # lib.embedding.providers, so patch it at the source module.
    monkeypatch.setattr(_providers, "build_embedding_client", _fake_build)
    chapters = [
        {"id": "ch1", "chapter_text": "x", "_cos": [
            "Define a prime number and identify its factors.",
            "Define what a prime number is and list its factors.",
        ]},
        {"id": "ch2", "chapter_text": "y", "_cos": [
            "Compute the area of a triangle using base and height.",
            "Photosynthesis converts light energy in the chloroplast.",
        ]},
    ]
    return asyncio.run(pt._run_stage2_window_synthesis(
        provider=provider,
        provider_error=_ProviderError,
        chapters=chapters,
        draft_tos=[{"statement": "Draft course objective."}],
        chunks_by_id={},
        all_chunks=[],
        grounding_mode="chapter_fallback",
        course_name="MATH_101",
        provider_env="local",
        chapter_synthesis_failures=[],
        mint_lo_id=_mint,
        kwargs={},
        capture=None,
    ))


def test_cluster_signals_on_grounding_signals(monkeypatch):
    """Bottom-up path stamps to_derivation=bottom_up + cluster_count onto the
    grounding_signals dict (and keeps WS2's weak_to_link_* fields valid)."""
    provider = _ChapterProvider()
    chapter_cos, terminal, _mint_method, signals = _run_stage2(
        provider, monkeypatch=monkeypatch, embed=FakeEmbed()
    )
    assert provider.reconcile_calls == 0          # bottom-up ran, not reconcile
    assert provider.author_calls                  # author called per cluster
    assert signals["to_derivation"] == "bottom_up"
    assert signals["cluster_count"] == len(terminal)
    # WS1.1 — Ward target-K signals present, legacy threshold key gone.
    assert "to_cluster_threshold" not in signals
    assert signals["to_cluster_k"] >= 1
    assert signals["to_cluster_linkage"] in {"ward", "single_link_fallback"}
    # WS2 fields survive on the bottom-up path.
    assert "weak_to_link_rate" in signals
    assert "weak_to_link_count" in signals
    # Every CO has a resolvable terminal_id (no orphans).
    minted = {t["id"] for t in terminal}
    assert all(co.get("terminal_id") in minted for co in chapter_cos)


def test_embed_absent_falls_back_to_legacy_reconcile(monkeypatch):
    """R4 — _embed None → bottom-up never entered; legacy reconcile + backlink
    run; cluster_signals is {} so to_derivation defaults to legacy_reconcile."""
    provider = _ChapterProvider()
    _chapter_cos, terminal, _mint_method, signals = _run_stage2(
        provider, monkeypatch=monkeypatch, embed=None
    )
    assert provider.reconcile_calls == 1          # the legacy path ran
    assert provider.author_calls == []            # bottom-up never entered
    assert terminal == [{"id": "TO-01", "statement": "Legacy reconciled TO."}]
    # cluster_signals merged as {} → no to_derivation key.
    assert "to_derivation" not in signals
    assert signals.get("to_derivation", "legacy_reconcile") == "legacy_reconcile"
    # WS2 fields still present (backlink ran on the fallback path).
    assert "weak_to_link_rate" in signals


# ---------------------------------------------------------------------------
# P2b — TO child back-annotation (observability: TO->CO->chunk join on disk)
# ---------------------------------------------------------------------------


def test_annotate_terminals_with_children_unions_chunks():
    """Each TO gets child_co_ids + a source_refs UNION of its child COs' chunks.

    Case-insensitive TO matching (a backlink may lower-case the pointer); chunk
    ids deduped in course order; a dangling CO->TO pointer is ignored; an orphan
    TO (no children) keeps its original (empty) source_refs.
    """
    tos = [
        {"id": "TO-01", "statement": "a", "source_refs": []},
        {"id": "TO-02", "statement": "b", "source_refs": []},
        {"id": "TO-03", "statement": "c", "source_refs": []},
    ]
    cos = [
        {"id": "CO-01", "terminal_id": "TO-01", "source_chunk_ids": ["c1", "c2"]},
        {"id": "CO-02", "terminal_id": "to-01", "source_chunk_ids": ["c2", "c3"]},
        {"id": "CO-03", "parent_to": "TO-02", "source_chunk_ids": ["c9"]},
        {"id": "CO-04", "terminal_id": "TO-99", "source_chunk_ids": ["x"]},
    ]
    counts = pt._annotate_terminals_with_children(tos, cos)
    assert counts == {"tos_annotated": 2, "cos_mapped": 3, "orphan_tos": 1}
    assert tos[0]["child_co_ids"] == ["CO-01", "CO-02"]
    # FIX A — Union, deduped, course order — never invents a chunk. The entry
    # carries NO self-referential ``ref`` (a TO citing its OWN id never resolves
    # against the chapter/section universe the objective_source_refs gate audits,
    # firing on every TO); only the grounded chunk_ids survive.
    assert tos[0]["source_refs"] == [{"chunk_ids": ["c1", "c2", "c3"]}]
    assert "ref" not in tos[0]["source_refs"][0]
    assert tos[1]["child_co_ids"] == ["CO-03"]
    assert tos[1]["source_refs"] == [{"chunk_ids": ["c9"]}]
    assert "ref" not in tos[1]["source_refs"][0]
    # Orphan TO keeps its original empty source_refs (no fabricated provenance).
    assert tos[2]["child_co_ids"] == []
    assert tos[2]["source_refs"] == []


def test_annotate_terminals_no_children_is_byte_stable():
    """No resolvable child pointers → every TO childless, source_refs untouched."""
    tos = [{"id": "TO-01", "statement": "a", "source_refs": []}]
    cos = [{"id": "CO-01", "source_chunk_ids": ["c1"]}]  # no terminal_id at all
    counts = pt._annotate_terminals_with_children(tos, cos)
    assert counts == {"tos_annotated": 0, "cos_mapped": 0, "orphan_tos": 1}
    assert tos[0]["child_co_ids"] == []
    assert tos[0]["source_refs"] == []


def test_co_parent_terminal_id_resolves_any_backpointer_key():
    assert pt._co_parent_terminal_id({"terminal_id": "TO-05"}) == "TO-05"
    assert pt._co_parent_terminal_id({"parent_to": "TO-07"}) == "TO-07"
    assert pt._co_parent_terminal_id({"parent_terminal": "TO-09"}) == "TO-09"
    assert pt._co_parent_terminal_id({"nope": "x"}) == ""


# ---------------------------------------------------------------------------
# P1 — cluster-guard wiring in _derive_terminals_bottom_up (default no-op)
# ---------------------------------------------------------------------------


def test_cluster_guards_default_off_is_noop_with_signals():
    """Default (no env, no guard flags) — derivation byte-stable, but the
    to_guard_* signal keys are surfaced (all-zero) when the guards module is
    present so an operator can see the guards ran in shadow."""
    cos = _theme_cos()
    terminals, signals = pt._derive_terminals_bottom_up(
        provider=_FakeProvider(),
        chapter_cos=cos,
        embed=FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
    )
    # Still one TO per (post-guard) cluster; every CO keeps a resolvable parent.
    minted = {t["id"] for t in terminals}
    assert all(co.get("terminal_id") in minted for co in cos)
    # When the guards helper exists, default-OFF surfaces zeroed counters.
    try:
        from lib.objectives.objective_dedup import (  # noqa: F401
            apply_cluster_guards,
        )

        have_guards = True
    except Exception:  # noqa: BLE001
        have_guards = False
    if have_guards:
        assert signals.get("to_guard_outliers_absorbed", 0) == 0
        assert signals.get("to_guard_undersize_merged", 0) == 0
        assert signals.get("to_guard_near_dup_clusters_merged", 0) == 0


def _cohort_plus_outlier_cos() -> List[Dict[str, Any]]:
    """Five cohesive prime-number COs (heavy shared vocab) + ONE semantic
    outlier (disjoint insurance vocabulary).

    Mirrors the real sample-course-a CO-71 failure: an Everyday-Math
    insurance word-problem CO sitting among an otherwise-cohesive algebra
    cohort. Under both clustering paths (Ward at a pinned K=2, or the
    no-sklearn single-link fallback at the 0.50 TO threshold) the five prime
    COs join into one cluster and the insurance CO lands in its OWN singleton.
    """
    return [
        _co("Define a prime number and identify its prime factors."),
        _co("Define what a prime number is and list its prime factors."),
        _co("Identify prime numbers and find the prime factorization."),
        _co("Find the prime factorization of a composite whole number."),
        _co("Determine whether a whole number is prime or composite."),
        # Semantic outlier — disjoint vocabulary, no token overlap with prime.
        _co("Compute insurance deductible premium reimbursement payouts."),
    ]


def test_cluster_guards_on_absorbs_outlier_no_singleton_to(monkeypatch):
    """Bucket A — with the guard flags ON, a semantic-outlier CO does NOT form
    its own singleton TO: it is absorbed into its nearest (the cohesive) cluster.

    Pins ``ED4ALL_TO_CLUSTER_K=2`` so the Ward path (sklearn present) and the
    single-link fallback (sklearn absent) agree on the same 2-cluster partition
    [cohort, outlier-singleton]. ``ED4ALL_TO_OUTLIER_MIN_SIZE=2`` makes only the
    singleton a merge candidate (the size-5 cohort is untouched);
    ``ED4ALL_TO_OUTLIER_ABSORB_FLOOR=0`` lets the outlier fold into its only
    neighbor. Near-dup merge stays OFF (default).
    """
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_K", "2")
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_GUARDS", "true")
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_MIN_SIZE", "2")
    monkeypatch.setenv("ED4ALL_TO_OUTLIER_ABSORB_FLOOR", "0")
    monkeypatch.delenv("ED4ALL_TO_MERGE_NEAR_DUP", raising=False)

    cos = _cohort_plus_outlier_cos()
    terminals, signals = pt._derive_terminals_bottom_up(
        provider=_FakeProvider(),
        chapter_cos=cos,
        embed=FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
    )

    # The singleton outlier was absorbed → exactly ONE TO, no standalone outlier.
    assert len(terminals) == 1
    assert signals.get("to_guard_outliers_absorbed", 0) == 1
    # Every CO (cohort + the absorbed outlier) shares the one surviving TO.
    minted = {t["id"] for t in terminals}
    assert all(co.get("terminal_id") in minted for co in cos)
    assert len({co["terminal_id"] for co in cos}) == 1
    # Partition invariant: no CO dropped or duplicated (6 in, 6 placed).
    assert len(cos) == 6


def test_cluster_guards_off_outlier_keeps_its_own_to(monkeypatch):
    """Bucket A control — with the guards OFF (default), the SAME outlier cohort
    still produces a standalone singleton TO for the outlier (the pathology the
    guard fixes). Confirms the guard, not the cohort, is what changes behavior.
    """
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_K", "2")
    monkeypatch.delenv("ED4ALL_TO_CLUSTER_GUARDS", raising=False)
    monkeypatch.delenv("ED4ALL_TO_OUTLIER_MIN_SIZE", raising=False)
    monkeypatch.delenv("ED4ALL_TO_OUTLIER_ABSORB_FLOOR", raising=False)
    monkeypatch.delenv("ED4ALL_TO_MERGE_NEAR_DUP", raising=False)

    cos = _cohort_plus_outlier_cos()
    terminals, signals = pt._derive_terminals_bottom_up(
        provider=_FakeProvider(),
        chapter_cos=cos,
        embed=FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
    )

    # Two clusters → two TOs; the outlier sits in its OWN singleton TO.
    assert len(terminals) == 2
    assert signals.get("to_guard_outliers_absorbed", 0) == 0
    # The five cohesive COs share one TO; the outlier owns the other.
    cohort_tos = {cos[i]["terminal_id"] for i in range(5)}
    outlier_to = cos[5]["terminal_id"]
    assert len(cohort_tos) == 1                      # cohort still clusters as one
    assert outlier_to not in cohort_tos             # outlier is standalone
    # The outlier TO has exactly one member CO (the singleton pathology).
    assert sum(1 for co in cos if co["terminal_id"] == outlier_to) == 1


# ---------------------------------------------------------------------------
# Unconditional anti-hallucinated-TO backstop (dissolve_singletons) wiring
# ---------------------------------------------------------------------------


def _neutral_cohort_plus_outlier() -> List[Dict[str, Any]]:
    """Five cohesive COs (shared linear-equation vocab) + ONE disjoint outlier.

    Topic-free of any specific leaked content: the outlier's vocabulary simply
    shares no tokens with the cohort, so under a pinned ``ED4ALL_TO_CLUSTER_K=2``
    it lands in its OWN singleton cluster — the shape the backstop must fold.
    """
    return [
        _co("Solve a linear equation in one variable."),
        _co("Solve the linear equation for its single variable."),
        _co("Solve a two-step linear equation in one variable."),
        _co("Isolate the variable to solve a linear equation."),
        _co("Rearrange and solve a linear equation for the variable."),
        # Disjoint outlier — no shared tokens with the linear-equation cohort.
        _co("Label the organelles of a plant cell under a microscope."),
    ]


def test_dissolve_singletons_default_on_folds_outlier(monkeypatch):
    """DEFAULT (opt-out UNSET) — the outlier singleton is DISSOLVED (fix ON), so
    no standalone lone-CO terminal objective survives id-minting.

    Pins ``ED4ALL_TO_CLUSTER_K=2`` (2-cluster [cohort, singleton] partition) and
    ``ED4ALL_TO_MIN_CLUSTERS=1`` so the tiny-course floor permits the 2->1 fold.
    The opt-in cluster GUARDS stay OFF — the backstop, not the guards, acts.
    """
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_K", "2")
    monkeypatch.setenv("ED4ALL_TO_MIN_CLUSTERS", "1")
    monkeypatch.delenv("ED4ALL_TO_ALLOW_SINGLETON_TO", raising=False)
    for var in ("ED4ALL_TO_CLUSTER_GUARDS", "ED4ALL_TO_MERGE_NEAR_DUP"):
        monkeypatch.delenv(var, raising=False)

    cos = _neutral_cohort_plus_outlier()
    terminals, signals = pt._derive_terminals_bottom_up(
        provider=_FakeProvider(),
        chapter_cos=cos,
        embed=FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
    )

    # The singleton was folded → exactly ONE TO; no standalone lone-CO TO.
    assert len(terminals) == 1
    assert signals.get("singletons_dissolved", 0) == 1
    assert signals.get("singletons_remaining", -1) == 0
    minted = {t["id"] for t in terminals}
    assert all(co.get("terminal_id") in minted for co in cos)
    assert len({co["terminal_id"] for co in cos}) == 1
    # Partition invariant: 6 COs in, 6 placed (none dropped / duplicated).
    assert len(cos) == 6


def test_dissolve_singletons_opt_out_keeps_singleton(monkeypatch):
    """OPT-OUT (ED4ALL_TO_ALLOW_SINGLETON_TO=1) — legacy keep-singleton: the SAME
    outlier cohort still produces a standalone singleton TO (byte-stable escape
    hatch). Confirms the backstop, not the cohort, is what changes behavior.
    """
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_K", "2")
    monkeypatch.setenv("ED4ALL_TO_MIN_CLUSTERS", "1")
    monkeypatch.setenv("ED4ALL_TO_ALLOW_SINGLETON_TO", "1")
    for var in ("ED4ALL_TO_CLUSTER_GUARDS", "ED4ALL_TO_MERGE_NEAR_DUP"):
        monkeypatch.delenv(var, raising=False)

    cos = _neutral_cohort_plus_outlier()
    terminals, signals = pt._derive_terminals_bottom_up(
        provider=_FakeProvider(),
        chapter_cos=cos,
        embed=FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
    )

    # Legacy: two clusters → two TOs; the outlier owns its own singleton TO.
    assert len(terminals) == 2
    assert signals.get("singletons_dissolved", 0) == 0
    cohort_tos = {cos[i]["terminal_id"] for i in range(5)}
    outlier_to = cos[5]["terminal_id"]
    assert len(cohort_tos) == 1
    assert outlier_to not in cohort_tos
    assert sum(1 for co in cos if co["terminal_id"] == outlier_to) == 1
